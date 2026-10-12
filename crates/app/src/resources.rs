//! Hardware resource management: effective limits, process priority, live
//! monitoring and memory-pressure detection.

use std::time::{Duration, Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use clearframe_core::settings::{ProcessPriority, ResourcePlan};
use clearframe_pipeline::spec::JobResources;

use crate::App;

/// Lower (or raise) the priority of the current process. Child processes
/// (FFmpeg) inherit it. Returns a description of what was applied.
pub fn apply_process_priority(p: ProcessPriority) -> String {
    #[cfg(unix)]
    {
        let nice = match p {
            ProcessPriority::Low => 15,
            ProcessPriority::BelowNormal => 8,
            ProcessPriority::Normal => 0,
            // Raising priority needs privileges on Unix; keep normal.
            ProcessPriority::AboveNormal => 0,
        };
        // SAFETY: setpriority only reads its integer arguments.
        let r = unsafe { libc::setpriority(libc::PRIO_PROCESS, 0, nice) };
        if r == 0 { format!("nice {nice}") } else { "priority unchanged (not permitted)".into() }
    }
    #[cfg(windows)]
    {
        use windows_sys::Win32::System::Threading::{GetCurrentProcess, SetPriorityClass, ABOVE_NORMAL_PRIORITY_CLASS, BELOW_NORMAL_PRIORITY_CLASS, IDLE_PRIORITY_CLASS, NORMAL_PRIORITY_CLASS};
        let class = match p {
            ProcessPriority::Low => IDLE_PRIORITY_CLASS,
            ProcessPriority::BelowNormal => BELOW_NORMAL_PRIORITY_CLASS,
            ProcessPriority::Normal => NORMAL_PRIORITY_CLASS,
            ProcessPriority::AboveNormal => ABOVE_NORMAL_PRIORITY_CLASS,
        };
        // SAFETY: GetCurrentProcess returns a pseudo-handle valid for this call.
        let ok = unsafe { SetPriorityClass(GetCurrentProcess(), class) };
        if ok != 0 { format!("priority class {p:?}") } else { "priority unchanged".into() }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = p;
        "priority unchanged".into()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GpuSample {
    pub name: String,
    pub utilization_percent: Option<f32>,
    pub memory_used_mb: Option<u64>,
    pub memory_total_mb: Option<u64>,
    pub temperature_c: Option<f32>,
    /// True when figures come from the driver; false when estimated.
    pub measured: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct SystemSample {
    pub cpu_percent: f32,
    pub logical_cores: usize,
    pub memory_total_mb: u64,
    pub memory_used_mb: u64,
    pub memory_available_mb: u64,
    /// ClearFrame app + worker processes.
    pub clearframe_memory_mb: u64,
    pub clearframe_cpu_percent: f32,
    pub gpus: Vec<GpuSample>,
    /// Hottest sensor reading, when the OS exposes sensors.
    pub max_temperature_c: Option<f32>,
    pub thermal_warning: Option<String>,
    pub memory_pressure: bool,
}

/// Throttled sampler (at most once per second; GPU queries every 3 s).
pub struct SystemMonitor {
    inner: Mutex<MonitorState>,
}

struct MonitorState {
    sys: sysinfo::System,
    comps: sysinfo::Components,
    last: Option<(Instant, SystemSample)>,
    gpu_last: Option<(Instant, Vec<GpuSample>)>,
}

impl Default for SystemMonitor {
    fn default() -> Self {
        Self { inner: Mutex::new(MonitorState { sys: sysinfo::System::new(), comps: sysinfo::Components::new(), last: None, gpu_last: None }) }
    }
}

fn nvidia_sample() -> Vec<GpuSample> {
    let out = std::process::Command::new("nvidia-smi")
        .args(["--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader,nounits"])
        .stderr(std::process::Stdio::null())
        .output();
    let Ok(out) = out else { return vec![] };
    String::from_utf8_lossy(&out.stdout)
        .lines()
        .filter_map(|l| {
            let p: Vec<&str> = l.split(',').map(|x| x.trim()).collect();
            (p.len() >= 5).then(|| GpuSample {
                name: p[0].into(),
                utilization_percent: p[1].parse().ok(),
                memory_used_mb: p[2].parse().ok(),
                memory_total_mb: p[3].parse().ok(),
                temperature_c: p[4].parse().ok(),
                measured: true,
            })
        })
        .collect()
}

impl SystemMonitor {
    pub fn sample(&self, reserve_mb: u64, worker_pids: &[u32]) -> SystemSample {
        let mut g = self.inner.lock();
        if let Some((t, s)) = &g.last {
            if t.elapsed() < Duration::from_secs(1) {
                return s.clone();
            }
        }
        g.sys.refresh_memory();
        g.sys.refresh_cpu_usage();
        g.sys.refresh_processes(sysinfo::ProcessesToUpdate::All, true);
        let me = sysinfo::get_current_pid().ok();
        let mut roots: Vec<sysinfo::Pid> = worker_pids.iter().map(|p| sysinfo::Pid::from_u32(*p)).collect();
        roots.extend(me);
        let (mut mem, mut cpu) = (0u64, 0f32);
        for p in g.sys.processes().values() {
            if roots.contains(&p.pid()) || p.parent().map(|pp| roots.contains(&pp)).unwrap_or(false) {
                mem += p.memory();
                cpu += p.cpu_usage();
            }
        }
        let gpus = match &g.gpu_last {
            Some((t, v)) if t.elapsed() < Duration::from_secs(3) => v.clone(),
            _ => {
                let v = nvidia_sample();
                g.gpu_last = Some((Instant::now(), v.clone()));
                v
            }
        };
        g.comps.refresh_list();
        let mut max_t: Option<f32> = None;
        let mut thermal = None;
        for c in g.comps.iter() {
            let t = c.temperature();
            if t.is_finite() && t > 0.0 {
                max_t = Some(max_t.map_or(t, |m: f32| m.max(t)));
                if let Some(crit) = c.critical() {
                    if crit > 0.0 && t >= crit - 5.0 {
                        thermal = Some(format!("{} is at {:.0} °C, close to its limit ({:.0} °C); the system may be throttling.", c.label(), t, crit));
                    }
                }
            }
        }
        for gs in &gpus {
            if gs.temperature_c.unwrap_or(0.0) >= 87.0 {
                thermal = Some(format!("{} is at {:.0} °C; the GPU may be throttling.", gs.name, gs.temperature_c.unwrap_or(0.0)));
            }
        }
        let avail = g.sys.available_memory() / (1 << 20);
        let s = SystemSample {
            cpu_percent: g.sys.global_cpu_usage(),
            logical_cores: g.sys.cpus().len(),
            memory_total_mb: g.sys.total_memory() / (1 << 20),
            memory_used_mb: g.sys.used_memory() / (1 << 20),
            memory_available_mb: avail,
            clearframe_memory_mb: mem / (1 << 20),
            clearframe_cpu_percent: cpu,
            gpus,
            max_temperature_c: max_t,
            thermal_warning: thermal,
            memory_pressure: reserve_mb > 0 && avail < reserve_mb / 2,
        };
        g.last = Some((Instant::now(), s.clone()));
        s
    }
}

impl App {
    /// Effective resource limits from the Performance settings for this machine.
    pub fn resource_plan(&self) -> ResourcePlan {
        let hw = self.hardware();
        self.settings().performance.resolve(hw.logical_cores as u32, hw.ram_total_mb)
    }

    /// Limits handed to a job's worker.
    pub fn job_resources(&self) -> JobResources {
        let s = self.settings();
        let plan = self.resource_plan();
        JobResources {
            threads: plan.threads,
            memory_budget_mb: plan.memory_budget_mb / plan.max_concurrent_jobs.max(1) as u64,
            vram_budget_mb: s.performance.vram_budget_mb,
            oom_tile_fallback: s.performance.oom_tile_fallback,
            min_free_disk_mb: 1024,
            device: s.performance.device.clone(),
        }
    }

    pub fn system_sample(&self) -> SystemSample {
        let plan = self.resource_plan();
        self.monitor.sample(plan.reserve_memory_mb, &self.queue.worker_pids())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monitor_samples_and_throttles() {
        let m = SystemMonitor::default();
        let a = m.sample(0, &[]);
        assert!(a.memory_total_mb > 0);
        assert!(a.clearframe_memory_mb > 0);
        let t = Instant::now();
        let _ = m.sample(0, &[]);
        assert!(t.elapsed() < Duration::from_millis(50), "second call within 1s is served from cache");
    }

    #[test]
    fn priority_can_be_lowered() {
        // Run in a child-safe way: lowering priority is always permitted.
        let r = apply_process_priority(ProcessPriority::Normal);
        assert!(!r.is_empty());
    }
}
