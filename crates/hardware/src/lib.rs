//! Hardware, GPU and compute-backend detection, plus capability levels.

use std::process::Command;
use std::time::Duration;

use clearframe_inference::device::{self, BackendKind, BackendStatus};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GpuInfo {
    pub vendor: String,
    pub name: String,
    pub memory_total_mb: Option<u64>,
    pub memory_free_mb: Option<u64>,
    pub driver: Option<String>,
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub os: String,
    pub os_version: String,
    pub arch: String,
    pub cpu_brand: String,
    pub logical_cores: usize,
    pub physical_cores: Option<usize>,
    pub ram_total_mb: u64,
    pub ram_available_mb: u64,
    pub gpus: Vec<GpuInfo>,
    pub backends: Vec<BackendStatus>,
    pub precisions: Vec<String>,
    pub cpu_features: Vec<String>,
}

/// Capability level shown on model cards and in the job inspector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CapabilityLevel {
    /// Tested on this OS/backend in the release test matrix.
    Supported,
    /// Expected to work; not covered by the release test matrix.
    CompatibleNotCertified,
    /// Works on CPU only; may be slow.
    CpuFallbackOnly,
    Unavailable,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Capability {
    pub level: CapabilityLevel,
    pub backend: Option<BackendKind>,
    pub reason: String,
}

/// OS/backend combinations exercised by automated tests (see docs/support-matrix.md).
const CERTIFIED: &[(&str, BackendKind)] = &[("linux", BackendKind::Cpu)];

fn run(cmd: &str, args: &[&str]) -> Option<String> {
    let mut c = Command::new(cmd);
    c.args(args);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        c.creation_flags(0x0800_0000);
    }
    let child = c.stdout(std::process::Stdio::piped()).stderr(std::process::Stdio::null()).spawn().ok()?;
    let start = std::time::Instant::now();
    let mut child = child;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if start.elapsed() > Duration::from_secs(8) => {
                let _ = child.kill();
                return None;
            }
            Ok(None) => std::thread::sleep(Duration::from_millis(20)),
            Err(_) => return None,
        }
    }
    let out = child.wait_with_output().ok()?;
    if !out.status.success() {
        return None;
    }
    Some(String::from_utf8_lossy(&out.stdout).into_owned())
}

fn nvidia_smi() -> Vec<GpuInfo> {
    let Some(out) = run("nvidia-smi", &["--query-gpu=name,memory.total,memory.free,driver_version", "--format=csv,noheader,nounits"]) else { return vec![] };
    out.lines()
        .filter_map(|l| {
            let p: Vec<&str> = l.split(',').map(|s| s.trim()).collect();
            (p.len() >= 4).then(|| GpuInfo {
                vendor: "NVIDIA".into(),
                name: p[0].into(),
                memory_total_mb: p[1].parse().ok(),
                memory_free_mb: p[2].parse().ok(),
                driver: Some(p[3].into()),
                source: "nvidia-smi".into(),
            })
        })
        .collect()
}

fn vendor_name(id: &str) -> &'static str {
    match id.trim().trim_start_matches("0x").to_ascii_lowercase().as_str() {
        "10de" => "NVIDIA",
        "1002" => "AMD",
        "8086" => "Intel",
        "106b" => "Apple",
        "1af4" => "Virtio",
        "1234" => "QEMU",
        _ => "Unknown",
    }
}

#[cfg(target_os = "linux")]
fn os_gpus() -> Vec<GpuInfo> {
    let mut v = Vec::new();
    if let Ok(rd) = std::fs::read_dir("/sys/class/drm") {
        for e in rd.flatten() {
            let name = e.file_name().to_string_lossy().to_string();
            if !name.starts_with("card") || name.contains('-') {
                continue;
            }
            let dev = e.path().join("device");
            let vendor = std::fs::read_to_string(dev.join("vendor")).unwrap_or_default();
            let device = std::fs::read_to_string(dev.join("device")).unwrap_or_default();
            let vram = std::fs::read_to_string(dev.join("mem_info_vram_total")).ok().and_then(|s| s.trim().parse::<u64>().ok()).map(|b| b / (1 << 20));
            let used = std::fs::read_to_string(dev.join("mem_info_vram_used")).ok().and_then(|s| s.trim().parse::<u64>().ok()).map(|b| b / (1 << 20));
            let driver = std::fs::read_link(dev.join("driver")).ok().and_then(|p| p.file_name().map(|s| s.to_string_lossy().to_string()));
            v.push(GpuInfo {
                vendor: vendor_name(&vendor).into(),
                name: format!("{} GPU {}", vendor_name(&vendor), device.trim()),
                memory_total_mb: vram,
                memory_free_mb: vram.zip(used).map(|(t, u)| t.saturating_sub(u)),
                driver,
                source: "sysfs".into(),
            });
        }
    }
    v
}

#[cfg(target_os = "macos")]
fn os_gpus() -> Vec<GpuInfo> {
    let Some(out) = run("system_profiler", &["SPDisplaysDataType", "-json"]) else { return vec![] };
    let Ok(v): Result<serde_json::Value, _> = serde_json::from_str(&out) else { return vec![] };
    v["SPDisplaysDataType"]
        .as_array()
        .map(|a| {
            a.iter()
                .map(|g| GpuInfo {
                    vendor: g["spdisplays_vendor"].as_str().unwrap_or("Apple").replace("sppci_vendor_", ""),
                    name: g["sppci_model"].as_str().unwrap_or("GPU").into(),
                    memory_total_mb: g["spdisplays_vram"].as_str().and_then(|s| s.split_whitespace().next()?.parse::<u64>().ok()).map(|n| n * 1024),
                    memory_free_mb: None,
                    driver: g["spdisplays_mtlgpufamilysupport"].as_str().map(String::from),
                    source: "system_profiler".into(),
                })
                .collect()
        })
        .unwrap_or_default()
}

#[cfg(target_os = "windows")]
fn os_gpus() -> Vec<GpuInfo> {
    let Some(out) =
        run("powershell", &["-NoProfile", "-Command", "Get-CimInstance Win32_VideoController | Select-Object Name,AdapterRAM,DriverVersion,AdapterCompatibility | ConvertTo-Json"])
    else {
        return vec![];
    };
    let Ok(v): Result<serde_json::Value, _> = serde_json::from_str(&out) else { return vec![] };
    let arr = if v.is_array() { v.as_array().cloned().unwrap_or_default() } else { vec![v] };
    arr.iter()
        .map(|g| GpuInfo {
            vendor: g["AdapterCompatibility"].as_str().unwrap_or("").into(),
            name: g["Name"].as_str().unwrap_or("GPU").into(),
            // Win32 reports at most 4 GB here; nvidia-smi gives exact values for NVIDIA cards.
            memory_total_mb: g["AdapterRAM"].as_u64().map(|b| b / (1 << 20)),
            memory_free_mb: None,
            driver: g["DriverVersion"].as_str().map(String::from),
            source: "Win32_VideoController".into(),
        })
        .collect()
}

#[cfg(not(any(target_os = "linux", target_os = "macos", target_os = "windows")))]
fn os_gpus() -> Vec<GpuInfo> {
    vec![]
}

fn cpu_features() -> Vec<String> {
    let mut f = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if std::is_x86_feature_detected!("avx2") {
            f.push("avx2".into());
        }
        if std::is_x86_feature_detected!("fma") {
            f.push("fma".into());
        }
        if std::is_x86_feature_detected!("avx512f") {
            f.push("avx512f".into());
        }
        if std::is_x86_feature_detected!("f16c") {
            f.push("f16c".into());
        }
    }
    #[cfg(target_arch = "aarch64")]
    f.push("neon".into());
    f
}

/// Collect hardware information (takes up to a few seconds on first call).
pub fn detect() -> HardwareInfo {
    let mut sys = sysinfo::System::new();
    sys.refresh_memory();
    sys.refresh_cpu_all();
    let cpu_brand = sys.cpus().first().map(|c| c.brand().trim().to_string()).unwrap_or_default();
    let mut gpus = nvidia_smi();
    for g in os_gpus() {
        // Avoid listing NVIDIA cards twice.
        if !(g.vendor == "NVIDIA" && gpus.iter().any(|x| x.vendor == "NVIDIA")) {
            gpus.push(g);
        }
    }
    let backends = device::probe();
    let mut precisions = vec!["fp32".to_string()];
    if backends.iter().any(|b| b.available && b.kind != BackendKind::Cpu) {
        precisions.push("fp16".into());
    }
    HardwareInfo {
        os: std::env::consts::OS.into(),
        os_version: sysinfo::System::long_os_version().unwrap_or_default(),
        arch: std::env::consts::ARCH.into(),
        cpu_brand,
        logical_cores: std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1),
        physical_cores: sys.physical_core_count(),
        ram_total_mb: sys.total_memory() / (1 << 20),
        ram_available_mb: sys.available_memory() / (1 << 20),
        gpus,
        backends,
        precisions,
        cpu_features: cpu_features(),
    }
}

/// Capability of running a model (by its supported backends/platforms and memory needs) here.
pub fn capability(hw: &HardwareInfo, model_backends: &[String], model_platforms: &[String], platform: &str, min_ram_mb: u64) -> Capability {
    if !model_platforms.iter().any(|p| p == platform) {
        return Capability { level: CapabilityLevel::Unavailable, backend: None, reason: format!("Not available for {platform}.") };
    }
    if hw.ram_total_mb > 0 && hw.ram_total_mb < min_ram_mb {
        return Capability {
            level: CapabilityLevel::Unavailable,
            backend: None,
            reason: format!("Needs about {} GB of memory; this computer has {} GB.", min_ram_mb.div_ceil(1024), hw.ram_total_mb / 1024),
        };
    }
    let gpu = hw.backends.iter().find(|b| b.available && b.kind != BackendKind::Cpu && model_backends.iter().any(|m| m == b.kind.id()));
    match gpu {
        Some(b) => {
            let certified = CERTIFIED.iter().any(|(os, k)| *os == hw.os && *k == b.kind);
            Capability {
                level: if certified { CapabilityLevel::Supported } else { CapabilityLevel::CompatibleNotCertified },
                backend: Some(b.kind),
                reason: if certified {
                    format!("Tested with {}.", b.kind.label())
                } else {
                    format!("{} is available; this combination is not in the release test matrix.", b.kind.label())
                },
            }
        }
        None => {
            let certified = CERTIFIED.iter().any(|(os, k)| *os == hw.os && *k == BackendKind::Cpu);
            let reason = if hw.backends.iter().any(|b| b.kind != BackendKind::Cpu && b.compiled && !b.available) {
                "No usable GPU backend was found; processing will run on the CPU and may be slow.".to_string()
            } else {
                "This build has no GPU backend; processing runs on the CPU and may be slow.".to_string()
            };
            Capability {
                level: CapabilityLevel::CpuFallbackOnly,
                backend: Some(BackendKind::Cpu),
                reason: if certified { format!("{reason} CPU processing is tested on this OS.") } else { reason },
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_basics() {
        let hw = detect();
        assert!(hw.logical_cores >= 1);
        assert!(hw.ram_total_mb > 0);
        assert!(hw.backends.iter().any(|b| b.kind == BackendKind::Cpu && b.available));
        let plat = format!("{}-{}", if hw.os == "macos" { "macos" } else { &hw.os }, if hw.arch == "x86_64" { "x64" } else { "arm64" });
        let c = capability(&hw, &["cpu".into()], &[plat.clone()], &plat, 0);
        assert!(matches!(c.level, CapabilityLevel::CpuFallbackOnly | CapabilityLevel::Supported | CapabilityLevel::CompatibleNotCertified));
        let c = capability(&hw, &["cpu".into()], &["nowhere".into()], &plat, 0);
        assert_eq!(c.level, CapabilityLevel::Unavailable);
        let c = capability(&hw, &["cpu".into()], &[plat.clone()], &plat, u64::MAX / 2);
        assert_eq!(c.level, CapabilityLevel::Unavailable);
    }

    #[test]
    fn vendor_ids() {
        assert_eq!(vendor_name("0x10de\n"), "NVIDIA");
        assert_eq!(vendor_name("0x1002"), "AMD");
    }
}
