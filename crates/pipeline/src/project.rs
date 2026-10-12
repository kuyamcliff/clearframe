//! Project files (`.cfproj`): portable JSON describing a source, pipeline,
//! ranges and export settings. Sources are referenced, never embedded.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::spec::*;

pub const PROJECT_FORMAT: &str = "clearframe-project";
pub const PROJECT_EXTENSION: &str = "cfproj";
const MAX_PROJECT_BYTES: u64 = 4 << 20;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct SourceRef {
    /// Absolute path when saved.
    pub path: String,
    /// Path relative to the project file, when the source is near it (portability).
    #[serde(default)]
    pub relative_path: Option<String>,
    #[serde(default)]
    pub fingerprint: String,
    #[serde(default)]
    pub size_bytes: u64,
    /// Snapshot of probe results for display when the source is missing.
    #[serde(default)]
    pub summary: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
pub struct ExportSettings {
    /// Folder for this project's exports (None = application default).
    #[serde(default)]
    pub folder: Option<String>,
    /// File name without extension (None = naming template).
    #[serde(default)]
    pub file_name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Project {
    pub format: String,
    pub schema_version: u32,
    pub name: String,
    pub source: SourceRef,
    pub pipeline: Pipeline,
    pub output: OutputSettings,
    #[serde(default)]
    pub range: Option<TimeRange>,
    #[serde(default)]
    pub preview_range: Option<TimeRange>,
    #[serde(default)]
    pub export: ExportSettings,
    #[serde(default)]
    pub engine: EngineSettings,
    /// Model id -> version the project was last rendered with.
    #[serde(default)]
    pub models: std::collections::BTreeMap<String, String>,
    #[serde(default)]
    pub job_ids: Vec<String>,
    #[serde(default)]
    pub preset_id: Option<String>,
    #[serde(default)]
    pub notes: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ProjectError {
    #[error("not a ClearFrame project: {0}")]
    Format(String),
    #[error("this project was saved by a newer version of ClearFrame (schema {0})")]
    TooNew(u32),
    #[error("invalid project: {0}")]
    Invalid(String),
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

/// Problems found when opening a project, with suggested fixes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OpenReport {
    pub project: Project,
    pub source_found: bool,
    /// Source exists but its content changed since the project was saved.
    pub source_changed: bool,
    /// A candidate replacement found next to the project (relative path).
    pub suggested_source: Option<String>,
    pub missing_models: Vec<String>,
    pub migrated_from: Option<u32>,
    pub backup_path: Option<String>,
}

impl Project {
    pub fn new(name: &str, source: &Path) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        Self {
            format: PROJECT_FORMAT.into(),
            schema_version: SCHEMA_VERSION,
            name: name.into(),
            source: SourceRef { path: source.to_string_lossy().into(), ..Default::default() },
            pipeline: Pipeline::default(),
            output: OutputSettings::default(),
            range: None,
            preview_range: None,
            export: ExportSettings::default(),
            engine: EngineSettings::default(),
            models: Default::default(),
            job_ids: vec![],
            preset_id: None,
            notes: String::new(),
            created_at: now.clone(),
            updated_at: now,
        }
    }

    pub fn validate(&self) -> Result<(), ProjectError> {
        if self.format != PROJECT_FORMAT {
            return Err(ProjectError::Format("wrong format marker".into()));
        }
        if self.name.chars().count() > 200 {
            return Err(ProjectError::Invalid("name too long".into()));
        }
        self.pipeline.validate().map_err(ProjectError::Invalid)?;
        for r in [&self.range, &self.preview_range].into_iter().flatten() {
            if !(r.start.is_finite() && r.end.is_finite()) || r.start < 0.0 || r.end <= r.start {
                return Err(ProjectError::Invalid("invalid time range".into()));
            }
        }
        Ok(())
    }

    /// Save atomically (write temp then rename). Records the source path relative to the project when possible.
    pub fn save(&mut self, path: &Path) -> Result<(), ProjectError> {
        self.validate()?;
        self.updated_at = chrono::Utc::now().to_rfc3339();
        if let (Some(dir), src) = (path.parent(), Path::new(&self.source.path)) {
            self.source.relative_path = src.strip_prefix(dir).ok().map(|r| r.to_string_lossy().replace('\\', "/"));
        }
        let data = serde_json::to_vec_pretty(self).map_err(|e| ProjectError::Invalid(e.to_string()))?;
        let tmp = path.with_extension("cfproj.tmp");
        std::fs::write(&tmp, data)?;
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Upgrade older documents in place. Returns the original version when migrated.
    fn migrate(v: &mut serde_json::Value) -> Result<Option<u32>, ProjectError> {
        let ver = v.get("schema_version").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
        if ver > SCHEMA_VERSION {
            return Err(ProjectError::TooNew(ver));
        }
        if ver == SCHEMA_VERSION {
            return Ok(None);
        }
        // Version 0 (pre-release drafts): `stages` at top level instead of `pipeline`.
        if ver == 0 {
            if let Some(stages) = v.get("stages").cloned() {
                v["pipeline"] = serde_json::json!({ "stages": stages });
                v.as_object_mut().map(|o| o.remove("stages"));
            }
            if v.get("output").is_none() {
                v["output"] = serde_json::to_value(OutputSettings::default()).unwrap_or_default();
            }
            let now = chrono::Utc::now().to_rfc3339();
            for k in ["created_at", "updated_at"] {
                if v.get(k).is_none() {
                    v[k] = now.clone().into();
                }
            }
            if v.get("format").is_none() {
                v["format"] = PROJECT_FORMAT.into();
            }
        }
        v["schema_version"] = SCHEMA_VERSION.into();
        Ok(Some(ver))
    }

    /// Open a project, migrating (with a backup) and checking the source.
    pub fn open(path: &Path, installed_models: &dyn Fn(&str) -> bool) -> Result<OpenReport, ProjectError> {
        let meta = std::fs::metadata(path)?;
        if meta.len() > MAX_PROJECT_BYTES {
            return Err(ProjectError::Format("file too large".into()));
        }
        let data = std::fs::read(path)?;
        let mut v: serde_json::Value = serde_json::from_slice(&data).map_err(|e| ProjectError::Format(e.to_string()))?;
        if v.get("format").and_then(|f| f.as_str()).is_some_and(|f| f != PROJECT_FORMAT) {
            return Err(ProjectError::Format("wrong format marker".into()));
        }
        let migrated_from = Self::migrate(&mut v)?;
        let mut backup_path = None;
        if migrated_from.is_some() {
            let b = PathBuf::from(format!("{}.v{}.bak", path.display(), migrated_from.unwrap_or(0)));
            std::fs::write(&b, &data)?;
            backup_path = Some(b.to_string_lossy().into_owned());
        }
        let project: Project = serde_json::from_value(v).map_err(|e| ProjectError::Format(e.to_string()))?;
        project.validate()?;
        let src = Path::new(&project.source.path);
        let source_found = src.is_file();
        let source_changed =
            source_found && !project.source.fingerprint.is_empty() && clearframe_core::fsutil::quick_fingerprint(src).map(|f| f != project.source.fingerprint).unwrap_or(false);
        let suggested_source = if source_found {
            None
        } else {
            project.source.relative_path.as_ref().and_then(|r| path.parent().map(|d| d.join(r))).filter(|p| p.is_file()).map(|p| p.to_string_lossy().into_owned())
        };
        let missing_models = project.pipeline.required_models().into_iter().filter(|m| !installed_models(m)).collect();
        Ok(OpenReport { project, source_found, source_changed, suggested_source, missing_models, migrated_from, backup_path })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn save_open_and_relink() {
        let d = tempfile::tempdir().unwrap();
        let src = d.path().join("media/clip.mp4");
        std::fs::create_dir_all(src.parent().unwrap()).unwrap();
        std::fs::write(&src, b"video").unwrap();
        let mut p = Project::new("Test", &src);
        p.source.fingerprint = clearframe_core::fsutil::quick_fingerprint(&src).unwrap();
        p.pipeline = Pipeline::single(Operation::Enhance(EnhanceSettings::default()));
        let pf = d.path().join("test.cfproj");
        p.save(&pf).unwrap();
        let r = Project::open(&pf, &|_| false).unwrap();
        assert!(r.source_found);
        assert!(!r.source_changed);
        assert_eq!(r.missing_models, vec!["xinntao.realesr-animevideo-v3"]);
        assert_eq!(r.project.source.relative_path.as_deref(), Some("media/clip.mp4"));
        // Modify source -> changed.
        std::fs::write(&src, b"different").unwrap();
        assert!(Project::open(&pf, &|_| true).unwrap().source_changed);
        // Move the whole folder: absolute path breaks, relative path suggests the new location.
        let moved = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(moved.path().join("media")).unwrap();
        std::fs::rename(&src, moved.path().join("media/clip.mp4")).unwrap();
        std::fs::rename(&pf, moved.path().join("test.cfproj")).unwrap();
        let r = Project::open(&moved.path().join("test.cfproj"), &|_| true).unwrap();
        assert!(!r.source_found);
        assert!(r.suggested_source.unwrap().ends_with("clip.mp4"));
    }

    #[test]
    fn migrates_v0_with_backup() {
        let d = tempfile::tempdir().unwrap();
        let pf = d.path().join("old.cfproj");
        std::fs::write(&pf, r#"{"name":"Old","source":{"path":"/nope.mp4"},"stages":[{"id":"g","type":"grain","settings":{}}]}"#).unwrap();
        let r = Project::open(&pf, &|_| true).unwrap();
        assert_eq!(r.migrated_from, Some(0));
        assert!(Path::new(r.backup_path.as_ref().unwrap()).is_file());
        assert_eq!(r.project.pipeline.stages.len(), 1);
        assert_eq!(r.project.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn rejects_future_and_garbage() {
        let d = tempfile::tempdir().unwrap();
        let pf = d.path().join("f.cfproj");
        std::fs::write(&pf, r#"{"format":"clearframe-project","schema_version":42}"#).unwrap();
        assert!(matches!(Project::open(&pf, &|_| true), Err(ProjectError::TooNew(42))));
        std::fs::write(&pf, "not json").unwrap();
        assert!(Project::open(&pf, &|_| true).is_err());
    }
}
