//! Model installation lifecycle and on-disk registry.
//!
//! Layout (one folder per model, stable ids, single install even when a model
//! belongs to several categories — it is filed under its primary category):
//!
//! ```text
//! <models>/
//!   enhancement/xinntao.realesr-animevideo-v3/
//!     weights/weights.safetensors
//!     manifest.json     catalog manifest (source, license, compatibility)
//!     metadata.json     install record (version, status, sizes, smoke test)
//!     checksums.json    SHA-256 and size of every managed file
//!   interpolation/hzwer.rife-v4.26/...
//! ```
//!
//! The filesystem is the source of truth: a model whose folder or files were
//! deleted outside ClearFrame is reported as `missing`, a model whose files
//! changed is `broken`; neither is ever used for processing.
//!
//! Install: preflight (platform, disk space) -> download to the cache's
//! staging area with resumable range requests -> SHA-256 verification -> safe
//! extraction -> conversion with reviewed code -> smoke inference -> atomic
//! move into place. Nothing from a model package is ever executed. ClearFrame
//! has no separate model runtimes (inference is built into the application),
//! so there is nothing shared to deduplicate between model folders.

use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use clearframe_core::db::{now, Db};
use clearframe_core::fsutil;

use crate::catalog::{Artifact, Catalog, ModelManifest};
use crate::convert::{checkpoint_to_safetensors, KeyRules};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
#[serde(rename_all = "snake_case")]
pub enum ModelState {
    NotInstalled,
    Downloading,
    Paused,
    Verifying,
    Preparing,
    Testing,
    Installed,
    UpdateAvailable,
    /// Files or the folder were removed outside ClearFrame.
    Missing,
    /// Files exist but do not match their recorded checksums or sizes.
    Broken,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StoredFile {
    /// Path relative to the model folder (e.g. `weights/weights.safetensors`).
    pub path: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstalledModel {
    pub id: String,
    pub version: String,
    pub state: ModelState,
    pub architecture: String,
    /// Model folder relative to the models directory (e.g. `enhancement/<id>`).
    #[serde(default)]
    pub dir: String,
    /// role -> converted file.
    pub files: BTreeMap<String, StoredFile>,
    /// role -> verified SHA-256 of the downloaded upstream artifact.
    pub source_sha256: BTreeMap<String, String>,
    pub installed_size: u64,
    pub installed_at: String,
    pub verified_at: Option<String>,
    pub smoke_test: Option<String>,
    pub error: Option<String>,
    pub custom: bool,
    /// For custom models: user-supplied information.
    pub custom_info: Option<CustomModelInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CustomModelInfo {
    pub display_name: String,
    pub license: String,
    pub source: String,
    pub scale: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InstallProgress {
    pub id: String,
    pub state: ModelState,
    pub role: Option<String>,
    pub bytes_done: u64,
    pub bytes_total: u64,
    pub message: String,
}

/// Cancellation / pause control for an install.
#[derive(Clone, Default)]
pub struct InstallControl {
    pub cancel: Arc<AtomicBool>,
    pub pause: Arc<AtomicBool>,
}

#[derive(Debug, thiserror::Error)]
pub enum InstallError {
    #[error("{0} is not in the model catalog")]
    Unknown(String),
    #[error("{0}")]
    Unsupported(String),
    #[error("Not enough free disk space for this model (requires about {need_mb} MB; {have_mb} MB available).")]
    DiskSpace { need_mb: u64, have_mb: u64 },
    #[error("Download failed: {0}")]
    Download(String),
    #[error("Model checksum did not match for {0}; the file was discarded.")]
    Checksum(String),
    #[error("The model archive is unsafe or malformed: {0}")]
    Archive(String),
    #[error("Converting the model failed: {0}")]
    Convert(String),
    #[error("The model failed its test run: {0}")]
    SmokeTest(String),
    #[error("{0} is in use by a running job; remove it after the job finishes.")]
    InUse(String),
    #[error("Download paused")]
    Paused,
    #[error("Cancelled")]
    Cancelled,
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Db(#[from] clearframe_core::db::DbError),
}

/// Smoke test: given architecture and role -> file, run a tiny inference and describe the result.
pub type SmokeTest = dyn Fn(&str, &BTreeMap<String, PathBuf>) -> Result<String, String> + Send + Sync;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelStatus {
    pub manifest: Option<ModelManifest>,
    pub installed: Option<InstalledModel>,
    pub state: ModelState,
    /// Bytes already downloaded into staging (resumable).
    pub partial_bytes: u64,
    /// Absolute model folder when installed or missing.
    pub path: Option<PathBuf>,
    /// Bytes on disk for this model's folder.
    pub disk_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifyReport {
    pub id: String,
    pub ok: bool,
    pub problems: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StorageReport {
    pub models_dir: PathBuf,
    pub models_dir_exists: bool,
    pub total_bytes: u64,
    pub per_model: BTreeMap<String, u64>,
    pub staging_bytes: u64,
    pub free_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Checksums {
    algorithm: String,
    files: BTreeMap<String, StoredFile>,
}

pub struct ModelManager {
    db: Arc<Db>,
    root: RwLock<PathBuf>,
    staging_root: RwLock<PathBuf>,
    catalog: RwLock<Catalog>,
    user_agent: String,
}

/// Folder name for a model's primary category.
pub fn category_dir(categories: &[String]) -> &'static str {
    match categories.first().map(|s| s.as_str()) {
        Some("enhancement") | Some("anime") => "enhancement",
        Some("frame_interpolation") => "interpolation",
        Some("denoise") => "denoising",
        Some("deblur") | Some("restoration") => "restoration",
        Some("face_recovery") => "face-recovery",
        Some("deinterlace") => "deinterlace",
        _ => "other",
    }
}

fn file_name_ok(n: &str) -> bool {
    !n.is_empty() && !n.contains(['/', '\\']) && n != "." && n != ".."
}

/// Current platform id as used in manifests.
pub fn platform_id() -> &'static str {
    match (std::env::consts::OS, std::env::consts::ARCH) {
        ("windows", "x86_64") => "windows-x64",
        ("windows", "aarch64") => "windows-arm64",
        ("linux", "x86_64") => "linux-x64",
        ("linux", "aarch64") => "linux-arm64",
        ("macos", "aarch64") => "macos-arm64",
        ("macos", "x86_64") => "macos-x64",
        _ => "other",
    }
}

const MAX_ARCHIVE_MEMBER: u64 = 4 << 30;

/// Whether the environment proxy applies to `url` (honours NO_PROXY and never proxies loopback).
pub fn use_proxy_for(url: &str) -> bool {
    let host = url.split("://").nth(1).unwrap_or(url).split('/').next().unwrap_or("");
    let host = host.rsplit_once('@').map(|(_, h)| h).unwrap_or(host);
    let host = if host.starts_with('[') { host.split(']').next().unwrap_or("").trim_start_matches('[') } else { host.split(':').next().unwrap_or("") };
    let host = host.to_ascii_lowercase();
    if host == "localhost" || host == "127.0.0.1" || host == "::1" || host.starts_with("127.") {
        return false;
    }
    let no_proxy = std::env::var("NO_PROXY").or_else(|_| std::env::var("no_proxy")).unwrap_or_default();
    for entry in no_proxy.split(',').map(|e| e.trim().to_ascii_lowercase()).filter(|e| !e.is_empty()) {
        if entry == "*" {
            return false;
        }
        let e = entry.trim_start_matches('.');
        if host == e || host.ends_with(&format!(".{e}")) {
            return false;
        }
    }
    true
}

impl ModelManager {
    /// `models_dir`: installed models. `staging_dir`: download staging (inside the managed cache).
    pub fn new(db: Arc<Db>, models_dir: PathBuf, staging_dir: PathBuf, catalog: Catalog) -> Self {
        Self { db, root: RwLock::new(models_dir), staging_root: RwLock::new(staging_dir), catalog: RwLock::new(catalog), user_agent: format!("ClearFrame/{}", env!("CARGO_PKG_VERSION")) }
    }

    pub fn catalog(&self) -> Catalog {
        self.catalog.read().clone()
    }

    pub fn set_catalog(&self, c: Catalog) {
        *self.catalog.write() = c;
    }

    pub fn models_dir(&self) -> PathBuf {
        self.root.read().clone()
    }

    /// Point at a new models directory (after a relocation) or staging area.
    pub fn set_locations(&self, models_dir: PathBuf, staging_dir: PathBuf) {
        *self.root.write() = models_dir;
        *self.staging_root.write() = staging_dir;
    }

    fn staging_dir(&self, id: &str) -> PathBuf {
        self.staging_root.read().join(fsutil::sanitize_component(id))
    }

    /// Folder for a model (relative to the models directory).
    pub fn model_rel_dir(&self, id: &str) -> String {
        let cat = self.catalog();
        let dir = cat.get(id).map(|m| category_dir(&m.categories)).unwrap_or(if id.starts_with("custom.") { "custom" } else { "other" });
        format!("{dir}/{}", fsutil::sanitize_component(id))
    }

    pub fn model_dir(&self, id: &str) -> PathBuf {
        self.models_dir().join(self.model_rel_dir(id))
    }

    fn record_from_db(&self, id: &str) -> Option<InstalledModel> {
        self.db.model_get(id).ok().flatten().and_then(|s| serde_json::from_str(&s).ok())
    }

    /// Read an install record from its folder and check files are present with the recorded sizes.
    fn scan_one(&self, id: &str) -> Option<InstalledModel> {
        let known = self.record_from_db(id);
        let rel = known.as_ref().map(|k| k.dir.clone()).filter(|d| !d.is_empty()).unwrap_or_else(|| self.model_rel_dir(id));
        let dir = self.models_dir().join(&rel);
        let meta: Option<InstalledModel> = std::fs::read(dir.join("metadata.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());
        let mut rec = match (meta, known) {
            (Some(m), _) => m,
            (None, Some(mut k)) => {
                // Folder or metadata removed outside ClearFrame.
                k.state = ModelState::Missing;
                k.error = Some(format!("The model folder {} is missing.", dir.display()));
                return Some(k);
            }
            (None, None) => return None,
        };
        rec.dir = rel;
        if rec.id != id {
            return None;
        }
        let sums: Option<Checksums> = std::fs::read(dir.join("checksums.json")).ok().and_then(|b| serde_json::from_slice(&b).ok());
        let mut problems = Vec::new();
        if sums.is_none() {
            problems.push("checksums.json is missing or unreadable".to_string());
        }
        let mut missing = false;
        for (role, f) in &rec.files {
            match std::fs::metadata(dir.join(&f.path)) {
                Ok(m) if m.len() == f.size => {}
                Ok(_) => problems.push(format!("{role}: size differs from the install record")),
                Err(_) => {
                    missing = true;
                    problems.push(format!("{role}: file {} is missing", f.path));
                }
            }
            if let Some(s) = &sums {
                if s.files.get(role).map(|x| x.sha256 != f.sha256).unwrap_or(true) {
                    problems.push(format!("{role}: checksum record does not match"));
                }
            }
        }
        if !problems.is_empty() {
            rec.state = if missing { ModelState::Missing } else { ModelState::Broken };
            rec.error = Some(problems.join("; "));
        }
        Some(rec)
    }

    /// Install record with on-disk state.
    pub fn installed(&self, id: &str) -> Option<InstalledModel> {
        self.scan_one(id)
    }

    /// All models found on disk or known to the registry.
    pub fn all_installed(&self) -> Vec<InstalledModel> {
        let mut ids: std::collections::BTreeSet<String> = self.db.model_all().unwrap_or_default().iter().filter_map(|s| serde_json::from_str::<InstalledModel>(s).ok()).map(|m| m.id).collect();
        // Adopt folders present on disk (e.g. after restoring a backup of the models folder).
        if let Ok(cats) = std::fs::read_dir(self.models_dir()) {
            for c in cats.flatten() {
                if let Ok(models) = std::fs::read_dir(c.path()) {
                    for m in models.flatten() {
                        if let Some(rec) = std::fs::read(m.path().join("metadata.json")).ok().and_then(|b| serde_json::from_slice::<InstalledModel>(&b).ok()) {
                            ids.insert(rec.id);
                        }
                    }
                }
            }
        }
        ids.into_iter().filter_map(|id| self.scan_one(&id)).collect()
    }

    fn save_index(&self, m: &InstalledModel) -> Result<(), InstallError> {
        let state = serde_json::to_value(m.state).ok().and_then(|v| v.as_str().map(String::from)).unwrap_or_default();
        self.db.model_put(&m.id, &m.version, &state, &serde_json::to_string(m).unwrap_or_default())?;
        Ok(())
    }

    fn partial_bytes(&self, id: &str) -> u64 {
        fsutil::dir_size(&self.staging_dir(id))
    }

    /// Status of every catalog model plus installed custom models.
    pub fn status(&self) -> Vec<ModelStatus> {
        let cat = self.catalog();
        let mut out = Vec::new();
        let installed: BTreeMap<String, InstalledModel> = self.all_installed().into_iter().map(|m| (m.id.clone(), m)).collect();
        let root = self.models_dir();
        for m in &cat.models {
            let inst = installed.get(&m.id).cloned();
            let partial = self.partial_bytes(&m.id);
            let state = match &inst {
                Some(i) if i.state == ModelState::Installed && i.version != m.version => ModelState::UpdateAvailable,
                Some(i) => i.state,
                None if partial > 0 => ModelState::Paused,
                None => ModelState::NotInstalled,
            };
            let path = inst.as_ref().map(|i| root.join(&i.dir));
            let disk_bytes = path.as_ref().map(|p| fsutil::dir_size(p)).unwrap_or(0);
            out.push(ModelStatus { manifest: Some(m.clone()), installed: inst, state, partial_bytes: partial, path, disk_bytes });
        }
        for (id, i) in installed {
            if cat.get(&id).is_none() {
                let path = root.join(&i.dir);
                out.push(ModelStatus { manifest: None, state: i.state, disk_bytes: fsutil::dir_size(&path), path: Some(path), installed: Some(i), partial_bytes: 0 });
            }
        }
        out
    }

    /// (version, role -> absolute file, role -> sha256) for a usable installed model.
    pub fn resolve(&self, id: &str) -> Option<(String, BTreeMap<String, PathBuf>, BTreeMap<String, String>)> {
        let m = self.scan_one(id)?;
        if m.state != ModelState::Installed {
            return None;
        }
        let dir = self.models_dir().join(&m.dir);
        let files = m.files.iter().map(|(r, f)| (r.clone(), dir.join(&f.path))).collect();
        let shas = m.files.iter().map(|(r, f)| (r.clone(), f.sha256.clone())).collect();
        Some((m.version, files, shas))
    }

    fn agent(&self, url: &str) -> ureq::Agent {
        ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(20))
            .timeout_read(Duration::from_secs(60))
            .user_agent(&self.user_agent)
            .try_proxy_from_env(use_proxy_for(url))
            .build()
    }

    fn check(&self, ctl: &InstallControl) -> Result<(), InstallError> {
        if ctl.cancel.load(Ordering::SeqCst) {
            return Err(InstallError::Cancelled);
        }
        if ctl.pause.load(Ordering::SeqCst) {
            return Err(InstallError::Paused);
        }
        Ok(())
    }

    /// Download one artifact into `dest` with resume support; verifies size and SHA-256.
    fn download(&self, a: &Artifact, dest: &Path, ctl: &InstallControl, progress: &mut dyn FnMut(u64)) -> Result<(), InstallError> {
        let part = dest.with_extension("part");
        let mut last_err = String::new();
        for url in std::iter::once(&a.source_url).chain(a.mirrors.iter()) {
            for attempt in 0..3 {
                self.check(ctl)?;
                let have = std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0);
                if have > a.size_bytes {
                    let _ = std::fs::remove_file(&part);
                    continue;
                }
                if have == a.size_bytes {
                    break;
                }
                let mut req = self.agent(url).get(url);
                if have > 0 {
                    req = req.set("Range", &format!("bytes={have}-"));
                }
                let resp = match req.call() {
                    Ok(r) => r,
                    Err(e) => {
                        last_err = format!("{url}: {e}");
                        std::thread::sleep(Duration::from_millis(500 * (1 << attempt)));
                        continue;
                    }
                };
                let append = resp.status() == 206 && have > 0;
                let mut file = std::fs::OpenOptions::new().create(true).write(true).append(append).truncate(!append).open(&part)?;
                let mut done = if append { have } else { 0 };
                progress(done);
                let mut reader = resp.into_reader().take(a.size_bytes + 1 - done);
                let mut buf = vec![0u8; 256 * 1024];
                let mut last = Instant::now();
                let res: Result<(), InstallError> = loop {
                    if let Err(e) = self.check(ctl) {
                        break Err(e);
                    }
                    match reader.read(&mut buf) {
                        Ok(0) => break Ok(()),
                        Ok(n) => {
                            file.write_all(&buf[..n])?;
                            done += n as u64;
                            if last.elapsed() > Duration::from_millis(150) {
                                progress(done);
                                last = Instant::now();
                            }
                        }
                        Err(e) => break Err(InstallError::Download(format!("{url}: {e}"))),
                    }
                };
                file.flush()?;
                drop(file);
                progress(done);
                match res {
                    Ok(()) if done == a.size_bytes => break,
                    Ok(()) => {
                        last_err = format!("{url}: connection closed early ({done} of {} bytes)", a.size_bytes);
                        if done > a.size_bytes {
                            let _ = std::fs::remove_file(&part);
                        }
                    }
                    Err(InstallError::Download(e)) => last_err = e,
                    Err(e) => return Err(e),
                }
            }
            if std::fs::metadata(&part).map(|m| m.len()).unwrap_or(0) == a.size_bytes {
                let sha = fsutil::sha256_file(&part)?;
                if sha != a.sha256 {
                    let _ = std::fs::remove_file(&part);
                    last_err = format!("checksum mismatch from {url}");
                    // Try the next mirror; a mismatch is never accepted.
                    continue;
                }
                std::fs::rename(&part, dest)?;
                return Ok(());
            }
        }
        if last_err.starts_with("checksum mismatch") {
            Err(InstallError::Checksum(a.filename.clone()))
        } else {
            Err(InstallError::Download(last_err))
        }
    }

    /// Extract a single named member from a zip archive with size/path/symlink defences.
    fn extract_member(&self, zip_path: &Path, member: &str, dest: &Path) -> Result<(), InstallError> {
        let f = std::fs::File::open(zip_path)?;
        let mut z = zip::ZipArchive::new(f).map_err(|e| InstallError::Archive(e.to_string()))?;
        if z.len() > 10_000 {
            return Err(InstallError::Archive("too many entries".into()));
        }
        let mut entry = z.by_name(member).map_err(|_| InstallError::Archive(format!("{member} not found in archive")))?;
        if entry.is_dir() {
            return Err(InstallError::Archive("member is a directory".into()));
        }
        if let Some(mode) = entry.unix_mode() {
            if mode & 0o170000 == 0o120000 {
                return Err(InstallError::Archive("member is a symbolic link".into()));
            }
        }
        if entry.enclosed_name().is_none() {
            return Err(InstallError::Archive("member path escapes the archive".into()));
        }
        let size = entry.size();
        if size > MAX_ARCHIVE_MEMBER {
            return Err(InstallError::Archive("member is too large".into()));
        }
        if entry.compressed_size() > 0 && size / entry.compressed_size().max(1) > 200 {
            return Err(InstallError::Archive("suspicious compression ratio".into()));
        }
        let mut out = std::fs::File::create(dest)?;
        let n = std::io::copy(&mut (&mut entry).take(size + 1), &mut out)?;
        if n != size {
            return Err(InstallError::Archive("member size does not match its header".into()));
        }
        Ok(())
    }

    /// Prepare one artifact into a safetensors file in staging.
    fn prepare(&self, a: &Artifact, raw: &Path, staging: &Path) -> Result<PathBuf, InstallError> {
        let out = staging.join(format!("{}.safetensors", fsutil::sanitize_component(&a.role)));
        let input = match a.format.as_str() {
            "zip" => {
                let member = a.member.as_deref().ok_or_else(|| InstallError::Archive("no member".into()))?;
                let extracted = staging.join(format!("{}.member", fsutil::sanitize_component(&a.role)));
                self.extract_member(raw, member, &extracted)?;
                extracted
            }
            _ => raw.to_path_buf(),
        };
        if a.format == "safetensors" {
            let bytes = std::fs::read(&input)?;
            safetensors::SafeTensors::deserialize(&bytes).map_err(|e| InstallError::Convert(e.to_string()))?;
            std::fs::copy(&input, &out)?;
        } else {
            checkpoint_to_safetensors(&input, &out, &a.convert).map_err(|e| InstallError::Convert(e.to_string()))?;
        }
        if input != raw {
            let _ = std::fs::remove_file(&input);
        }
        Ok(out)
    }

    /// Write a complete model folder in staging, then move it into place atomically.
    fn commit_folder(&self, rec: &mut InstalledModel, manifest: Option<&ModelManifest>, prepared: &BTreeMap<String, PathBuf>) -> Result<(), InstallError> {
        let root = self.models_dir();
        std::fs::create_dir_all(&root)?;
        let building = root.join(".installing").join(format!("{}-{}", fsutil::sanitize_component(&rec.id), std::process::id()));
        let _ = std::fs::remove_dir_all(&building);
        std::fs::create_dir_all(building.join("weights"))?;
        let mut files = BTreeMap::new();
        for (role, p) in prepared {
            if !file_name_ok(role) {
                return Err(InstallError::Archive(format!("invalid role name {role}")));
            }
            let rel = format!("weights/{}.safetensors", fsutil::sanitize_component(role));
            let dst = building.join(&rel);
            std::fs::rename(p, &dst).or_else(|_| std::fs::copy(p, &dst).map(|_| ()))?;
            let sha = fsutil::sha256_file(&dst)?;
            let size = std::fs::metadata(&dst)?.len();
            files.insert(role.clone(), StoredFile { path: rel, sha256: sha, size });
        }
        rec.installed_size = files.values().map(|f| f.size).sum();
        rec.files = files.clone();
        rec.dir = self.model_rel_dir(&rec.id);
        let write = |name: &str, v: &dyn erased::Json| -> std::io::Result<()> { std::fs::write(building.join(name), v.to_json()) };
        if let Some(m) = manifest {
            write("manifest.json", m)?;
        }
        write("metadata.json", rec)?;
        write("checksums.json", &Checksums { algorithm: "sha256".into(), files })?;
        // Swap into place.
        let final_dir = root.join(&rec.dir);
        std::fs::create_dir_all(final_dir.parent().unwrap_or(&root))?;
        let trash = root.join(".installing").join(format!("{}-old-{}", fsutil::sanitize_component(&rec.id), std::process::id()));
        if final_dir.exists() {
            std::fs::rename(&final_dir, &trash)?;
        }
        if let Err(e) = std::fs::rename(&building, &final_dir) {
            if trash.exists() {
                let _ = std::fs::rename(&trash, &final_dir);
            }
            return Err(e.into());
        }
        let _ = std::fs::remove_dir_all(&trash);
        let _ = std::fs::remove_dir(root.join(".installing"));
        self.save_index(rec)?;
        Ok(())
    }

    /// Install (or update/repair) a catalog model. `local_files` may supply
    /// already-downloaded artifacts by role for offline installation; they are
    /// verified exactly like downloads.
    pub fn install(
        &self,
        id: &str,
        local_files: &BTreeMap<String, PathBuf>,
        ctl: &InstallControl,
        smoke: &SmokeTest,
        progress: &mut dyn FnMut(InstallProgress),
    ) -> Result<InstalledModel, InstallError> {
        let m = self.catalog().get(id).cloned().ok_or_else(|| InstallError::Unknown(id.to_string()))?;
        let plat = platform_id();
        if !m.supported_platforms.iter().any(|p| p == plat) {
            return Err(InstallError::Unsupported(format!("{} does not support this platform ({plat})", m.display_name)));
        }
        let staging = self.staging_dir(id);
        std::fs::create_dir_all(&staging)?;
        let root = self.models_dir();
        // Disk preflight: remaining download (staging) and converted size (models folder) + 10% margin.
        let partial = self.partial_bytes(id);
        let need_staging = (m.download_size_bytes.saturating_sub(partial) + m.installed_size_bytes) as f64 * 1.1;
        let need_models = m.installed_size_bytes as f64 * 1.1;
        for (dir, need) in [(&staging, need_staging), (&root, need_models)] {
            if let Some(free) = fsutil::free_space(dir) {
                if (free as f64) < need {
                    return Err(InstallError::DiskSpace { need_mb: (need / 1e6).ceil() as u64, have_mb: free / 1_000_000 });
                }
            }
        }
        let total = m.download_size_bytes;
        let mut base = 0u64;
        let mut raws: Vec<(Artifact, PathBuf)> = Vec::new();
        for a in &m.artifacts {
            self.check(ctl)?;
            if !file_name_ok(&a.filename) {
                return Err(InstallError::Archive("artifact name contains a path".into()));
            }
            let raw = staging.join(fsutil::sanitize_component(&a.filename));
            if let Some(local) = local_files.get(&a.role) {
                progress(InstallProgress { id: id.into(), state: ModelState::Verifying, role: Some(a.role.clone()), bytes_done: base, bytes_total: total, message: format!("Verifying {}", a.filename) });
                let sha = fsutil::sha256_file(local)?;
                if sha != a.sha256 {
                    return Err(InstallError::Checksum(a.filename.clone()));
                }
                std::fs::copy(local, &raw)?;
            } else if !(raw.is_file() && fsutil::sha256_file(&raw).ok().as_deref() == Some(a.sha256.as_str())) {
                let b = base;
                let role = a.role.clone();
                let fname = a.filename.clone();
                self.download(a, &raw, ctl, &mut |done| {
                    progress(InstallProgress { id: id.into(), state: ModelState::Downloading, role: Some(role.clone()), bytes_done: b + done, bytes_total: total, message: format!("Downloading {fname}") })
                })?;
            }
            base += a.size_bytes;
            raws.push((a.clone(), raw));
        }
        progress(InstallProgress { id: id.into(), state: ModelState::Preparing, role: None, bytes_done: total, bytes_total: total, message: "Converting weights to a safe format".into() });
        let mut prepared = BTreeMap::new();
        for (a, raw) in &raws {
            self.check(ctl)?;
            prepared.insert(a.role.clone(), self.prepare(a, raw, &staging)?);
        }
        progress(InstallProgress { id: id.into(), state: ModelState::Testing, role: None, bytes_done: total, bytes_total: total, message: "Running a test inference".into() });
        let smoke_result = smoke(&m.architecture, &prepared).map_err(InstallError::SmokeTest)?;
        let mut rec = InstalledModel {
            id: id.into(),
            version: m.version.clone(),
            state: ModelState::Installed,
            architecture: m.architecture.clone(),
            dir: String::new(),
            files: BTreeMap::new(),
            source_sha256: m.artifacts.iter().map(|a| (a.role.clone(), a.sha256.clone())).collect(),
            installed_size: 0,
            installed_at: now(),
            verified_at: Some(now()),
            smoke_test: Some(smoke_result),
            error: None,
            custom: false,
            custom_info: None,
        };
        self.commit_folder(&mut rec, Some(&m), &prepared)?;
        let _ = std::fs::remove_dir_all(&staging);
        progress(InstallProgress { id: id.into(), state: ModelState::Installed, role: None, bytes_done: total, bytes_total: total, message: "Installed".into() });
        Ok(rec)
    }

    /// Discard partial downloads for a model.
    pub fn discard_partial(&self, id: &str) {
        let _ = std::fs::remove_dir_all(self.staging_dir(id));
    }

    /// Install a user-provided checkpoint for a supported architecture (power users).
    pub fn import_custom(&self, info: CustomModelInfo, file: &Path, rules: &KeyRules, smoke: &SmokeTest) -> Result<InstalledModel, InstallError> {
        let slug: String = info.display_name.to_lowercase().chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect::<String>().trim_matches('-').chars().take(40).collect();
        let id = format!("custom.{}", if slug.is_empty() { "model".into() } else { slug });
        let staging = self.staging_dir(&id);
        std::fs::create_dir_all(&staging)?;
        let out = staging.join("weights.safetensors");
        if file.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            let bytes = std::fs::read(file)?;
            safetensors::SafeTensors::deserialize(&bytes).map_err(|e| InstallError::Convert(e.to_string()))?;
            std::fs::copy(file, &out)?;
        } else {
            checkpoint_to_safetensors(file, &out, rules).map_err(|e| InstallError::Convert(e.to_string()))?;
        }
        let mut prepared = BTreeMap::new();
        prepared.insert("weights".to_string(), out);
        let arch = smoke("auto", &prepared).map_err(InstallError::SmokeTest)?;
        let mut rec = InstalledModel {
            id: id.clone(),
            version: "custom".into(),
            state: ModelState::Installed,
            architecture: arch.split(':').next().unwrap_or("unknown").to_string(),
            dir: String::new(),
            files: BTreeMap::new(),
            source_sha256: [("weights".to_string(), fsutil::sha256_file(file)?)].into_iter().collect(),
            installed_size: 0,
            installed_at: now(),
            verified_at: Some(now()),
            smoke_test: Some(arch),
            error: None,
            custom: true,
            custom_info: Some(info),
        };
        self.commit_folder(&mut rec, None, &prepared)?;
        let _ = std::fs::remove_dir_all(&staging);
        Ok(rec)
    }

    /// Persist a changed record (e.g. custom scale) to the model folder and index.
    pub fn update_record(&self, rec: &InstalledModel) -> Result<(), InstallError> {
        let dir = self.models_dir().join(&rec.dir);
        std::fs::write(dir.join("metadata.json"), rec.to_json())?;
        self.save_index(rec)
    }

    /// Re-hash installed files.
    pub fn verify(&self, id: &str) -> Result<VerifyReport, InstallError> {
        let m = self.scan_one(id).ok_or_else(|| InstallError::Unknown(id.to_string()))?;
        let dir = self.models_dir().join(&m.dir);
        let mut problems: Vec<String> = m.error.iter().cloned().collect();
        if m.state == ModelState::Installed {
            for (role, f) in &m.files {
                match fsutil::sha256_file(&dir.join(&f.path)) {
                    Ok(s) if s == f.sha256 => {}
                    Ok(_) => problems.push(format!("{role}: file content changed")),
                    Err(e) => problems.push(format!("{role}: {e}")),
                }
            }
        }
        let ok = problems.is_empty();
        if ok {
            let mut rec = m.clone();
            rec.verified_at = Some(now());
            let _ = std::fs::write(dir.join("metadata.json"), rec.to_json());
            let _ = self.save_index(&rec);
        } else if m.state == ModelState::Installed {
            // Mark as broken on disk so the model is not used until repaired.
            let mut rec = m.clone();
            rec.state = ModelState::Broken;
            rec.error = Some(problems.join("; "));
            let _ = std::fs::write(dir.join("metadata.json"), rec.to_json());
            let _ = self.save_index(&rec);
        }
        Ok(VerifyReport { id: id.into(), ok, problems })
    }

    /// Remove a model's folder. `in_use` lists model ids used by running work.
    pub fn remove(&self, id: &str, in_use: &[String]) -> Result<u64, InstallError> {
        if in_use.iter().any(|x| x == id) {
            return Err(InstallError::InUse(id.to_string()));
        }
        let rel = self.scan_one(id).map(|m| m.dir).filter(|d| !d.is_empty()).unwrap_or_else(|| self.model_rel_dir(id));
        let dir = self.models_dir().join(&rel);
        let freed = fsutil::dir_size(&dir);
        // Only the selected model's managed folder is deleted.
        if dir.starts_with(self.models_dir()) && dir != self.models_dir() && dir.is_dir() {
            std::fs::remove_dir_all(&dir)?;
        }
        if let Some(parent) = dir.parent() {
            let _ = std::fs::remove_dir(parent); // only succeeds when the category folder is empty
        }
        self.db.model_delete(id)?;
        self.discard_partial(id);
        Ok(freed)
    }

    pub fn storage(&self) -> StorageReport {
        let root = self.models_dir();
        let per_model = self.all_installed().into_iter().map(|m| { let b = fsutil::dir_size(&root.join(&m.dir)); (m.id, b) }).collect();
        StorageReport {
            models_dir_exists: root.is_dir(),
            total_bytes: fsutil::dir_size(&root),
            per_model,
            staging_bytes: fsutil::dir_size(&self.staging_root.read()),
            free_bytes: fsutil::free_space(&root),
            models_dir: root,
        }
    }
}

mod erased {
    pub trait Json {
        fn to_json(&self) -> Vec<u8>;
    }
    impl<T: serde::Serialize> Json for T {
        fn to_json(&self) -> Vec<u8> {
            serde_json::to_vec_pretty(self).unwrap_or_default()
        }
    }
}
use erased::Json as _;

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn proxy_bypass_rules() {
        assert!(!use_proxy_for("http://127.0.0.1:8080/x"));
        assert!(!use_proxy_for("http://localhost/x"));
        assert!(!use_proxy_for("http://[::1]:9/x"));
        assert!(use_proxy_for("https://github.com/x"));
    }

    /// Minimal HTTP server supporting Range, for install tests (no network access needed).
    fn serve(files: BTreeMap<String, Vec<u8>>, truncate_first: bool) -> (String, std::thread::JoinHandle<()>) {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = format!("http://{}", l.local_addr().unwrap());
        let h = std::thread::spawn(move || {
            let mut first = truncate_first;
            for stream in l.incoming().take(12) {
                let mut s = stream.unwrap();
                let mut buf = [0u8; 4096];
                let n = s.read(&mut buf).unwrap_or(0);
                let req = String::from_utf8_lossy(&buf[..n]).to_string();
                let path = req.split_whitespace().nth(1).unwrap_or("/").trim_start_matches('/').to_string();
                let Some(body) = files.get(&path) else {
                    let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n");
                    continue;
                };
                let start = req.lines().find(|l| l.to_lowercase().starts_with("range:")).and_then(|l| l.split('=').nth(1)).and_then(|r| r.trim_end_matches('-').trim().trim_end_matches('-').parse::<usize>().ok()).unwrap_or(0);
                let slice = &body[start.min(body.len())..];
                let (status, send) = if first {
                    first = false;
                    ("200 OK", &body[..body.len() / 2])
                } else if start > 0 {
                    ("206 Partial Content", slice)
                } else {
                    ("200 OK", &body[..])
                };
                let declared = if status.starts_with("200") && send.len() < body.len() { body.len() } else { send.len() };
                let _ = write!(s, "HTTP/1.1 {status}\r\nContent-Length: {declared}\r\nConnection: close\r\n\r\n");
                let _ = s.write_all(send);
            }
        });
        (addr, h)
    }

    fn tiny_checkpoint() -> Vec<u8> {
        // A minimal legacy-format-free zip checkpoint is hard to build by hand; use safetensors.
        let data: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let view = safetensors::tensor::TensorView::new(safetensors::Dtype::F32, vec![2, 2], &data).unwrap();
        safetensors::serialize(vec![("w".to_string(), view)], &None).unwrap()
    }

    fn manifest(url: &str, bytes: &[u8]) -> ModelManifest {
        let sha = {
            use sha2::Digest;
            hex::encode(sha2::Sha256::digest(bytes))
        };
        let mut m = Catalog::bundled().models[0].clone();
        m.id = "test.tiny".into();
        m.artifacts = vec![Artifact {
            role: "weights".into(),
            filename: "tiny.safetensors".into(),
            source_url: format!("{url}/tiny.safetensors"),
            mirrors: vec![],
            size_bytes: bytes.len() as u64,
            sha256: sha,
            format: "safetensors".into(),
            member: None,
            convert: KeyRules::default(),
        }];
        m.download_size_bytes = bytes.len() as u64;
        m.installed_size_bytes = bytes.len() as u64;
        m
    }

    fn manager(dir: &Path, m: ModelManifest) -> ModelManager {
        let db = Arc::new(Db::open_in_memory().unwrap());
        let cat = Catalog { schema_version: 1, catalog_version: "t".into(), generated_at: "t".into(), models: vec![m] };
        ModelManager::new(db, dir.join("models"), dir.join("cache/downloads"), cat)
    }

    fn ok_smoke() -> Box<SmokeTest> {
        Box::new(|_, files| if files.contains_key("weights") { Ok("ok".into()) } else { Err("missing".into()) })
    }

    #[test]
    fn install_resume_verify_remove() {
        let bytes = tiny_checkpoint();
        let (url, _h) = serve([("tiny.safetensors".to_string(), bytes.clone())].into_iter().collect(), true);
        let d = tempfile::tempdir().unwrap();
        let mgr = manager(d.path(), manifest(&url, &bytes));
        let mut events = Vec::new();
        let rec = mgr.install("test.tiny", &BTreeMap::new(), &InstallControl::default(), &*ok_smoke(), &mut |p| events.push(p.state)).unwrap();
        assert_eq!(rec.state, ModelState::Installed);
        assert!(events.contains(&ModelState::Downloading));
        assert!(events.contains(&ModelState::Testing));
        assert!(mgr.resolve("test.tiny").is_some());
        assert!(mgr.verify("test.tiny").unwrap().ok);
        // Layout: <models>/<category>/<id>/{weights,manifest.json,metadata.json,checksums.json}
        let dir = mgr.model_dir("test.tiny");
        assert!(dir.starts_with(d.path().join("models/enhancement")));
        for f in ["manifest.json", "metadata.json", "checksums.json", "weights/weights.safetensors"] {
            assert!(dir.join(f).is_file(), "{f}");
        }
        // Same-size corruption is only detected by full verification -> broken.
        let p = mgr.resolve("test.tiny").unwrap().1["weights"].clone();
        let mut bytes2 = std::fs::read(&p).unwrap();
        let n = bytes2.len();
        bytes2[n - 1] ^= 1;
        std::fs::write(&p, bytes2).unwrap();
        assert!(!mgr.verify("test.tiny").unwrap().ok);
        assert_eq!(mgr.installed("test.tiny").unwrap().state, ModelState::Broken);
        assert!(mgr.resolve("test.tiny").is_none());
        // In-use models cannot be removed.
        assert!(matches!(mgr.remove("test.tiny", &["test.tiny".into()]), Err(InstallError::InUse(_))));
        let freed = mgr.remove("test.tiny", &[]).unwrap();
        assert!(freed > 0);
        assert!(mgr.installed("test.tiny").is_none());
        assert!(!dir.exists());
    }

    #[test]
    fn checksum_mismatch_is_discarded() {
        let bytes = tiny_checkpoint();
        let mut tampered = bytes.clone();
        let n = tampered.len();
        tampered[n - 1] ^= 0xff;
        let (url, _h) = serve([("tiny.safetensors".to_string(), tampered)].into_iter().collect(), false);
        let d = tempfile::tempdir().unwrap();
        let mgr = manager(d.path(), manifest(&url, &bytes));
        let r = mgr.install("test.tiny", &BTreeMap::new(), &InstallControl::default(), &*ok_smoke(), &mut |_| {});
        assert!(matches!(r, Err(InstallError::Checksum(_))), "{r:?}");
        assert!(mgr.installed("test.tiny").is_none());
        // No partial file is kept after a mismatch.
        assert_eq!(fsutil::dir_size(&d.path().join("cache/downloads")), 0);
    }

    #[test]
    fn offline_install_from_local_file_and_smoke_failure() {
        let bytes = tiny_checkpoint();
        let d = tempfile::tempdir().unwrap();
        let mgr = manager(d.path(), manifest("https://invalid.example", &bytes));
        let local = d.path().join("dl.safetensors");
        std::fs::write(&local, &bytes).unwrap();
        let files: BTreeMap<String, PathBuf> = [("weights".to_string(), local)].into_iter().collect();
        let failing: Box<SmokeTest> = Box::new(|_, _| Err("bad output".into()));
        assert!(matches!(mgr.install("test.tiny", &files, &InstallControl::default(), &*failing, &mut |_| {}), Err(InstallError::SmokeTest(_))));
        assert!(mgr.installed("test.tiny").is_none(), "never registered as installed after a failed test");
        let rec = mgr.install("test.tiny", &files, &InstallControl::default(), &*ok_smoke(), &mut |_| {}).unwrap();
        assert_eq!(rec.state, ModelState::Installed);
    }

    #[test]
    fn external_deletion_is_detected() {
        let bytes = tiny_checkpoint();
        let d = tempfile::tempdir().unwrap();
        let mgr = manager(d.path(), manifest("https://invalid.example", &bytes));
        let local = d.path().join("dl.safetensors");
        std::fs::write(&local, &bytes).unwrap();
        let files: BTreeMap<String, PathBuf> = [("weights".to_string(), local)].into_iter().collect();
        mgr.install("test.tiny", &files, &InstallControl::default(), &*ok_smoke(), &mut |_| {}).unwrap();
        // Delete just the weights file (as a user might in Explorer/Finder).
        std::fs::remove_file(mgr.model_dir("test.tiny").join("weights/weights.safetensors")).unwrap();
        assert_eq!(mgr.installed("test.tiny").unwrap().state, ModelState::Missing);
        assert!(mgr.resolve("test.tiny").is_none());
        // Delete the whole models directory.
        std::fs::remove_dir_all(d.path().join("models")).unwrap();
        let st = mgr.status();
        assert_eq!(st[0].state, ModelState::Missing);
        assert!(!mgr.storage().models_dir_exists);
        // Reinstall (repair) works without network using the local file.
        let local = d.path().join("dl2.safetensors");
        std::fs::write(&local, &bytes).unwrap();
        let files: BTreeMap<String, PathBuf> = [("weights".to_string(), local)].into_iter().collect();
        mgr.install("test.tiny", &files, &InstallControl::default(), &*ok_smoke(), &mut |_| {}).unwrap();
        assert!(mgr.resolve("test.tiny").is_some());
        // Adopt a folder restored onto a fresh database (e.g. after reinstalling the OS).
        let db = Arc::new(Db::open_in_memory().unwrap());
        let mgr2 = ModelManager::new(db, d.path().join("models"), d.path().join("cache/downloads"), mgr.catalog());
        assert_eq!(mgr2.all_installed().len(), 1);
        assert!(mgr2.resolve("test.tiny").is_some());
    }

    #[test]
    fn unicode_and_spaces_in_paths() {
        let bytes = tiny_checkpoint();
        let d = tempfile::tempdir().unwrap();
        let root = d.path().join("Benutzer Ñame/モデル");
        let db = Arc::new(Db::open_in_memory().unwrap());
        let cat = Catalog { schema_version: 1, catalog_version: "t".into(), generated_at: "t".into(), models: vec![manifest("https://x", &bytes)] };
        let mgr = ModelManager::new(db, root.join("models"), root.join("cache dir/downloads"), cat);
        let local = root.join("dl file.safetensors");
        std::fs::create_dir_all(&root).unwrap();
        std::fs::write(&local, &bytes).unwrap();
        let files: BTreeMap<String, PathBuf> = [("weights".to_string(), local)].into_iter().collect();
        mgr.install("test.tiny", &files, &InstallControl::default(), &*ok_smoke(), &mut |_| {}).unwrap();
        assert!(mgr.verify("test.tiny").unwrap().ok);
    }

    #[test]
    fn zip_member_extraction_defences() {
        let d = tempfile::tempdir().unwrap();
        let zp = d.path().join("a.zip");
        {
            let f = std::fs::File::create(&zp).unwrap();
            let mut z = zip::ZipWriter::new(f);
            let o: zip::write::SimpleFileOptions = zip::write::SimpleFileOptions::default();
            z.start_file("dir/model.pkl", o).unwrap();
            z.write_all(b"data").unwrap();
            z.start_file("../evil.py", o).unwrap();
            z.write_all(b"import os").unwrap();
            z.finish().unwrap();
        }
        let mgr = manager(d.path(), manifest("https://x", b"x"));
        let out = d.path().join("m");
        mgr.extract_member(&zp, "dir/model.pkl", &out).unwrap();
        assert_eq!(std::fs::read(&out).unwrap(), b"data");
        assert!(mgr.extract_member(&zp, "../evil.py", &d.path().join("e")).is_err());
        assert!(mgr.extract_member(&zp, "missing", &d.path().join("e")).is_err());
    }

    #[test]
    fn disk_space_preflight() {
        let bytes = tiny_checkpoint();
        let d = tempfile::tempdir().unwrap();
        let mut m = manifest("https://x", &bytes);
        m.installed_size_bytes = 1 << 62;
        let mgr = manager(d.path(), m);
        assert!(matches!(mgr.install("test.tiny", &BTreeMap::new(), &InstallControl::default(), &*ok_smoke(), &mut |_| {}), Err(InstallError::DiskSpace { .. })));
    }
}
