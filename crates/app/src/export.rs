//! Export destinations and job creation (single and batch).

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::{anyhow, bail, Result};
use serde::{Deserialize, Serialize};

use clearframe_core::db::now;
use clearframe_core::fsutil::{self, NameVars};
use clearframe_core::jobs::{JobKind, JobRecord, JobState};
use clearframe_core::paths::default_export_dir;
use clearframe_pipeline::plan::ExecutionPlan;
use clearframe_pipeline::runner::plan_job;
use clearframe_pipeline::spec::*;

use crate::App;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NewJob {
    pub source: PathBuf,
    pub pipeline: Pipeline,
    pub output: OutputSettings,
    #[serde(default)]
    pub range: Option<TimeRange>,
    /// Folder override for this job (None = default export folder).
    #[serde(default)]
    pub folder: Option<PathBuf>,
    /// File name (without extension) override.
    #[serde(default)]
    pub file_name: Option<String>,
    /// Remember `folder` as the new default.
    #[serde(default)]
    pub remember_folder: bool,
    #[serde(default)]
    pub preset_name: Option<String>,
    #[serde(default)]
    pub batch_id: Option<String>,
    #[serde(default)]
    pub engine: EngineSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportCheck {
    pub plan: ExecutionPlan,
    pub destination: PathBuf,
    pub destination_folder_ok: Result<(), String>,
    pub free_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchResult {
    pub created: Vec<JobRecord>,
    pub failed: Vec<(PathBuf, String)>,
}

fn task_name(p: &Pipeline) -> String {
    let kinds: Vec<&str> = p.enabled().map(|s| s.op.kind()).collect();
    match kinds.as_slice() {
        [] => "copy".into(),
        [one] => one.replace('_', "-"),
        many => {
            if many.len() <= 3 {
                many.iter().map(|k| k.replace('_', "-")).collect::<Vec<_>>().join("+")
            } else {
                "pipeline".into()
            }
        }
    }
}

impl App {
    /// The export folder used when none is chosen for a job.
    pub fn default_export_folder(&self) -> PathBuf {
        let s = self.settings();
        if s.remember_last_export_folder {
            if let Some(f) = s.last_export_folder {
                return f;
            }
        }
        s.output_dir.unwrap_or_else(default_export_dir)
    }

    fn reserved_destinations(&self) -> BTreeSet<PathBuf> {
        self.db.jobs(false).unwrap_or_default().into_iter().filter(|j| !j.state.is_terminal()).map(|j| PathBuf::from(j.destination)).collect()
    }

    /// Decide the output path. Never the source, never an existing file, never
    /// a path already reserved by a pending job; collisions get `_2`, `_3`...
    pub fn plan_destination(&self, job: &NewJob, plan: &ExecutionPlan, extra_reserved: &BTreeSet<PathBuf>) -> Result<PathBuf> {
        let folder = job.folder.clone().unwrap_or_else(|| self.default_export_folder());
        // The default export folder is created when needed; a custom folder must be usable.
        fsutil::ensure_writable_dir(&folder, true).map_err(|e| anyhow!("{e}. Choose another export folder."))?;
        let stem_src = job.source.file_stem().and_then(|s| s.to_str()).unwrap_or("video").to_string();
        let model = job.pipeline.required_models().first().cloned().unwrap_or_default();
        let vars = NameVars {
            source_stem: stem_src,
            task: task_name(&job.pipeline),
            model: model.rsplit('.').next().unwrap_or("").to_string(),
            width: plan.output.width,
            height: plan.output.height,
            fps: format!("{:.3}", plan.output.fps.as_f64()).trim_end_matches('0').trim_end_matches('.').to_string(),
            preset: job.preset_name.clone().unwrap_or_default(),
            date: chrono::Local::now().format("%Y-%m-%d").to_string(),
        };
        let stem = match &job.file_name {
            Some(n) if !n.trim().is_empty() => fsutil::sanitize_component(n.trim()),
            _ => fsutil::expand_template(&self.settings().naming_template, &vars),
        };
        let ext = job.output.container.extension();
        let mut reserved = self.reserved_destinations();
        reserved.extend(extra_reserved.iter().cloned());
        let mut n = 1;
        loop {
            let name = if n == 1 { format!("{stem}.{ext}") } else { format!("{stem}_{n}.{ext}") };
            let p = folder.join(name);
            let is_source = std::fs::canonicalize(&p).ok().zip(std::fs::canonicalize(&job.source).ok()).map(|(a, b)| a == b).unwrap_or(p == job.source);
            if !p.exists() && !reserved.contains(&p) && !is_source {
                return Ok(p);
            }
            n += 1;
        }
    }

    /// Plan without queueing: issues, estimates, destination preview.
    pub fn check_export(&self, job: &NewJob) -> Result<ExportCheck> {
        let tools = self.tools()?;
        let spec = self.draft_spec(job, PathBuf::from("."), "check");
        let (_, plan) = plan_job(&tools, &spec, &self.lookup(), None)?;
        let folder = job.folder.clone().unwrap_or_else(|| self.default_export_folder());
        let destination_folder_ok = fsutil::ensure_writable_dir(&folder, folder == default_export_dir());
        let destination = self.plan_destination(job, &plan, &BTreeSet::new()).unwrap_or_else(|_| folder.join("…"));
        Ok(ExportCheck { plan, destination, destination_folder_ok, free_bytes: fsutil::free_space(&folder) })
    }

    fn draft_spec(&self, job: &NewJob, destination: PathBuf, id: &str) -> JobSpec {
        let s = self.settings();
        JobSpec {
            schema_version: SCHEMA_VERSION,
            job_id: id.to_string(),
            source: job.source.clone(),
            range: job.range,
            pipeline: job.pipeline.clone(),
            output: job.output.clone(),
            destination,
            work_dir: self.paths().job_dir(id),
            engine: job.engine.clone(),
            models: Default::default(),
            segment_seconds: s.segment_seconds as f64,
            write_sidecar: s.write_sidecar_report,
            preview: false,
            resources: self.job_resources(),
            estimated_memory_mb: 0.0,
        }
    }

    /// Validate, plan and queue a render job.
    pub fn enqueue(self: &Arc<Self>, job: NewJob) -> Result<JobRecord> {
        self.enqueue_with(job, &BTreeSet::new(), JobKind::Render, None)
    }

    pub(crate) fn enqueue_with(self: &Arc<Self>, job: NewJob, extra_reserved: &BTreeSet<PathBuf>, kind: JobKind, fixed_destination: Option<PathBuf>) -> Result<JobRecord> {
        if !job.source.is_file() {
            bail!("The source file {} was not found.", job.source.display());
        }
        let tools = self.tools()?;
        let id = format!("{}{}", if kind == JobKind::Preview { "preview-" } else { "" }, uuid::Uuid::new_v4());
        let mut spec = self.draft_spec(&job, PathBuf::new(), &id);
        spec.models = self.resolve_models(&job.pipeline)?;
        let (_, plan) = plan_job(&tools, &spec, &self.lookup(), None)?;
        if plan.has_errors() {
            bail!("{}", plan.errors().map(|e| e.message.clone()).collect::<Vec<_>>().join(" "));
        }
        // Clear incompatibility: a model needs more memory than this computer has.
        let hw = self.hardware();
        for m in spec.models.keys() {
            if let Some(man) = self.catalog().get(m) {
                let need = man.minimum_ram_bytes / (1 << 20);
                if hw.ram_total_mb > 0 && hw.ram_total_mb < need {
                    bail!("{} needs about {} GB of memory; this computer has {} GB.", man.display_name, need.div_ceil(1024), hw.ram_total_mb / 1024);
                }
            }
        }
        spec.estimated_memory_mb = plan.estimates.peak_memory_mb.0;
        if kind == JobKind::Preview {
            spec.preview = true;
            spec.write_sidecar = false;
            spec.segment_seconds = 600.0;
        }
        spec.destination = match fixed_destination {
            Some(d) => d,
            None => self.plan_destination(&job, &plan, extra_reserved)?,
        };
        if kind == JobKind::Render {
            if job.remember_folder {
                if let Some(f) = &job.folder {
                    let mut s = self.settings();
                    s.output_dir = Some(f.clone());
                    self.save_settings(s)?;
                }
            }
            let mut s = self.settings();
            if s.remember_last_export_folder {
                s.last_export_folder = spec.destination.parent().map(Path::to_path_buf);
                self.save_settings(s)?;
            }
        }
        let name = match kind {
            JobKind::Preview => job.preset_name.clone().unwrap_or_default(),
            JobKind::Render => spec.destination.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(),
        };
        let rec = JobRecord {
            id: id.clone(),
            kind,
            name,
            state: JobState::Queued,
            source: job.source.to_string_lossy().into(),
            destination: spec.destination.to_string_lossy().into(),
            spec_json: serde_json::to_string(&spec)?,
            created_at: now(),
            updated_at: now(),
            started_at: None,
            finished_at: None,
            progress: 0.0,
            progress_json: None,
            detail: None,
            error: None,
            report_json: None,
            queue_order: 0,
            attempts: 0,
            batch_id: job.batch_id.clone(),
            heartbeat: None,
            warnings: plan.issues.iter().filter(|i| i.severity == clearframe_pipeline::plan::Severity::Warning).map(|i| i.message.clone()).collect(),
        };
        self.db.insert_job(&rec).map_err(|e| anyhow!("{e}"))?;
        for w in &rec.warnings {
            let _ = self.db.add_job_warning(&id, w);
        }
        let _ = self.db.add_recent(&job.source.to_string_lossy(), "video", &job.source.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default(), None);
        let rec = self.db.job(&id).map_err(|e| anyhow!("{e}"))?;
        self.emit(crate::AppEvent::JobUpdated { job: Box::new(rec.clone()) });
        self.queue.wake();
        self.queue_state_event();
        Ok(rec)
    }

    /// Queue the same pipeline for many files. One failure does not stop the rest.
    pub fn enqueue_batch(self: &Arc<Self>, sources: Vec<PathBuf>, template: NewJob) -> BatchResult {
        let batch = uuid::Uuid::new_v4().to_string();
        let mut created = Vec::new();
        let mut failed = Vec::new();
        let mut reserved = BTreeSet::new();
        for src in sources {
            let mut j = template.clone();
            j.source = src.clone();
            j.file_name = None;
            j.batch_id = Some(batch.clone());
            j.remember_folder = false;
            match self.enqueue_with(j, &reserved, JobKind::Render, None) {
                Ok(r) => {
                    reserved.insert(PathBuf::from(&r.destination));
                    created.push(r);
                }
                Err(e) => failed.push((src, e.to_string())),
            }
        }
        BatchResult { created, failed }
    }

    /// Expand folders into supported video files (non-recursive unless `recursive`).
    pub fn collect_videos(paths: &[PathBuf], recursive: bool) -> Vec<PathBuf> {
        const EXT: &[&str] = &["mp4", "mov", "mkv", "m4v", "webm", "avi", "mts", "m2ts", "ts", "mpg", "mpeg", "wmv", "flv", "3gp", "vob", "mxf", "ogv", "dv", "y4m"];
        let mut out = Vec::new();
        fn walk(p: &Path, recursive: bool, out: &mut Vec<PathBuf>, depth: usize) {
            if let Ok(rd) = std::fs::read_dir(p) {
                let mut items: Vec<PathBuf> = rd.flatten().map(|e| e.path()).collect();
                items.sort();
                for i in items {
                    if i.is_dir() {
                        if recursive && depth < 8 {
                            walk(&i, recursive, out, depth + 1);
                        }
                    } else if i.extension().and_then(|e| e.to_str()).map(|e| EXT.contains(&e.to_ascii_lowercase().as_str())).unwrap_or(false) {
                        out.push(i);
                    }
                }
            }
        }
        for p in paths {
            if p.is_dir() {
                walk(p, recursive, &mut out, 0);
            } else if p.is_file() {
                out.push(p.clone());
            }
        }
        out.dedup();
        out
    }
}
