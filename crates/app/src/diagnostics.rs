//! Local diagnostics report. Never uploaded; the user can copy or save it.

use serde::{Deserialize, Serialize};

use crate::App;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Diagnostics {
    pub generated_at: String,
    pub app_version: String,
    pub os: String,
    pub hardware: clearframe_hardware::HardwareInfo,
    pub resource_plan: clearframe_core::settings::ResourcePlan,
    pub ffmpeg: Option<serde_json::Value>,
    pub ffmpeg_error: Option<String>,
    pub storage: crate::storage::StorageOverview,
    pub models: Vec<serde_json::Value>,
    pub recent_jobs: Vec<serde_json::Value>,
    pub log_tail: String,
    pub redacted: bool,
}

fn redact(s: &str, all_paths: bool) -> String {
    let mut out = s.to_string();
    if let Some(home) = dirs::home_dir() {
        let h = home.to_string_lossy().to_string();
        if !h.is_empty() {
            out = out.replace(&h, "~");
            out = out.replace(&h.replace('\\', "\\\\"), "~");
        }
    }
    if let Ok(user) = std::env::var("USER").or_else(|_| std::env::var("USERNAME")) {
        if user.len() >= 3 {
            out = out.replace(&user, "<user>");
        }
    }
    if all_paths {
        // Replace anything that looks like an absolute path to a file name.
        let re_sep = |c: char| c == '/' || c == '\\';
        out = out
            .split('"')
            .map(|seg| {
                if (seg.starts_with('/') || seg.starts_with("~") || seg.get(1..3) == Some(":\\")) && seg.contains(re_sep) {
                    let name = seg.rsplit(re_sep).next().unwrap_or("");
                    format!("<path>/{name}")
                } else {
                    seg.to_string()
                }
            })
            .collect::<Vec<_>>()
            .join("\"");
    }
    out
}

impl App {
    pub fn diagnostics(&self) -> Diagnostics {
        let (tools, err) = self.tools_status();
        let ffmpeg = tools.map(|t| {
            serde_json::json!({
                "path": t.ffmpeg, "source": t.source, "version": t.version, "license": t.license_hint,
                "configuration": t.configuration,
                "encoders": t.video_encoders.iter().map(|e| e.name.clone()).collect::<Vec<_>>(),
                "vidstab": t.has_filter("vidstabdetect"), "zscale": t.has_filter("zscale"),
            })
        });
        let models = self
            .models
            .status()
            .into_iter()
            .map(|s| {
                serde_json::json!({
                    "id": s.manifest.as_ref().map(|m| m.id.clone()).or(s.installed.as_ref().map(|i| i.id.clone())),
                    "state": s.state,
                    "version": s.installed.as_ref().map(|i| i.version.clone()),
                    "files": s.installed.as_ref().map(|i| i.files.clone()),
                    "error": s.installed.as_ref().and_then(|i| i.error.clone()),
                })
            })
            .collect();
        let recent_jobs = self
            .db
            .jobs(false)
            .unwrap_or_default()
            .into_iter()
            .rev()
            .take(10)
            .map(|j| {
                let report: Option<serde_json::Value> = j.report_json.as_deref().and_then(|r| serde_json::from_str(r).ok());
                serde_json::json!({
                    "id": j.id, "state": j.state, "attempts": j.attempts, "error": j.error, "warnings": j.warnings,
                    "timings": report.as_ref().map(|r| r["timings"].clone()), "backend": report.as_ref().map(|r| r["backend"].clone()),
                    "events": self.db.job_events(&j.id).unwrap_or_default().into_iter().rev().take(20).collect::<Vec<_>>(),
                })
            })
            .collect();
        Diagnostics {
            generated_at: chrono::Utc::now().to_rfc3339(),
            app_version: env!("CARGO_PKG_VERSION").into(),
            os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
            hardware: self.hardware(),
            resource_plan: self.resource_plan(),
            ffmpeg,
            ffmpeg_error: err,
            storage: self.storage_overview(),
            models,
            recent_jobs,
            log_tail: crate::logging::tail(&crate::logging::session_log_path().unwrap_or_else(|| self.paths().logs_dir.join("clipfx.log")), 64 * 1024),
            redacted: false,
        }
    }

    /// JSON text of the diagnostics, optionally with paths redacted.
    pub fn diagnostics_text(&self, redact_paths: bool) -> String {
        let mut d = self.diagnostics();
        d.redacted = redact_paths;
        let s = serde_json::to_string_pretty(&d).unwrap_or_default();
        redact(&s, redact_paths)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn redaction() {
        let s = r#"{"path":"/home/alice/Videos/clip.mp4","x":"C:\\Users\\a\\b.mp4"}"#;
        let r = super::redact(s, true);
        assert!(r.contains("<path>/clip.mp4"), "{r}");
        assert!(!r.contains("alice/Videos"));
    }
}
