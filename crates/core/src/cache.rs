//! Managed cache: usage reporting, safe clearing and cleanup policies.
//!
//! Only data under the managed cache directory is ever touched. Original
//! videos, exports, projects, presets and installed models never live there.
//! Job work folders of unfinished jobs (needed to resume) are protected unless
//! the caller explicitly discards them.

use std::collections::{BTreeMap, HashSet};
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

use serde::{Deserialize, Serialize};

use crate::fsutil;
use crate::paths::{AppPaths, CacheArea};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AreaUsage {
    pub area: CacheArea,
    pub label: String,
    pub bytes: u64,
    pub entries: u64,
    /// Bytes that are protected (e.g. resumable job data).
    pub protected_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheReport {
    pub cache_dir: PathBuf,
    pub total_bytes: u64,
    pub free_bytes: Option<u64>,
    pub areas: Vec<AreaUsage>,
    pub low_space: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClearResult {
    pub freed_bytes: u64,
    pub removed_entries: u64,
    pub skipped_protected: u64,
    pub errors: Vec<String>,
}

fn entry_size(p: &Path) -> u64 {
    match std::fs::symlink_metadata(p) {
        Ok(m) if m.is_dir() => fsutil::dir_size(p),
        Ok(m) => m.len(),
        Err(_) => 0,
    }
}

fn entry_mtime(p: &Path) -> SystemTime {
    std::fs::metadata(p).and_then(|m| m.modified()).unwrap_or(SystemTime::UNIX_EPOCH)
}

fn entries(dir: &Path) -> Vec<PathBuf> {
    std::fs::read_dir(dir).map(|rd| rd.flatten().map(|e| e.path()).collect()).unwrap_or_default()
}

fn remove(p: &Path) -> std::io::Result<()> {
    let m = std::fs::symlink_metadata(p)?;
    if m.is_dir() {
        std::fs::remove_dir_all(p)
    } else {
        std::fs::remove_file(p)
    }
}

/// `protected` holds entry names (job ids, model ids being downloaded) that must not be removed.
pub fn report(paths: &AppPaths, protected: &HashSet<String>, low_disk_mb: u64) -> CacheReport {
    let mut areas = Vec::new();
    let mut total = 0;
    for a in CacheArea::ALL {
        let mut bytes = 0;
        let mut prot = 0;
        let list = entries(&paths.cache(a));
        for e in &list {
            let s = entry_size(e);
            bytes += s;
            if protected.contains(&e.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default()) {
                prot += s;
            }
        }
        total += bytes;
        areas.push(AreaUsage { area: a, label: a.label().into(), bytes, entries: list.len() as u64, protected_bytes: prot });
    }
    let free = fsutil::free_space(&paths.cache_dir);
    CacheReport { cache_dir: paths.cache_dir.clone(), total_bytes: total, free_bytes: free, areas, low_space: free.map(|f| f < low_disk_mb * 1_000_000).unwrap_or(false) }
}

/// Remove everything in the given areas except protected entries.
pub fn clear(paths: &AppPaths, areas: &[CacheArea], protected: &HashSet<String>) -> ClearResult {
    let mut r = ClearResult::default();
    for a in areas {
        for e in entries(&paths.cache(*a)) {
            let name = e.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if protected.contains(&name) {
                r.skipped_protected += 1;
                continue;
            }
            let s = entry_size(&e);
            match remove(&e) {
                Ok(()) => {
                    r.freed_bytes += s;
                    r.removed_entries += 1;
                }
                Err(err) => r.errors.push(format!("{}: {err}", e.display())),
            }
        }
    }
    r
}

/// Apply age and size policies to disposable areas (previews, thumbnails, proxies, temp).
/// Oldest entries are removed first. Job folders and downloads are never auto-removed here.
pub fn enforce_policy(paths: &AppPaths, limit_mb: u64, max_age_days: u32, protected: &HashSet<String>) -> ClearResult {
    let mut r = ClearResult::default();
    let disposable = [CacheArea::Previews, CacheArea::Thumbnails, CacheArea::Proxies, CacheArea::Temp];
    let mut all: Vec<(SystemTime, u64, PathBuf)> = Vec::new();
    for a in disposable {
        for e in entries(&paths.cache(a)) {
            let name = e.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
            if protected.contains(&name) {
                continue;
            }
            all.push((entry_mtime(&e), entry_size(&e), e));
        }
    }
    all.sort_by_key(|x| x.0);
    let now = SystemTime::now();
    let max_age = Duration::from_secs(max_age_days as u64 * 86400);
    let mut total: u64 = all.iter().map(|x| x.1).sum();
    let limit = limit_mb.saturating_mul(1 << 20);
    for (mt, size, p) in all {
        let too_old = max_age_days > 0 && now.duration_since(mt).map(|d| d > max_age).unwrap_or(false);
        let over = limit_mb > 0 && total > limit;
        if !(too_old || over) {
            continue;
        }
        match remove(&p) {
            Ok(()) => {
                r.freed_bytes += size;
                r.removed_entries += 1;
                total = total.saturating_sub(size);
            }
            Err(e) => r.errors.push(format!("{}: {e}", p.display())),
        }
    }
    r
}

/// Remove job work folders whose job ids are not in `keep`.
pub fn remove_orphan_job_dirs(paths: &AppPaths, keep: &HashSet<String>) -> ClearResult {
    let mut r = ClearResult::default();
    for e in entries(&paths.cache(CacheArea::Jobs)) {
        let name = e.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        if keep.contains(&name) {
            continue;
        }
        let s = entry_size(&e);
        if remove(&e).is_ok() {
            r.freed_bytes += s;
            r.removed_entries += 1;
        }
    }
    r
}

/// Per-area byte counts (for migration planning).
pub fn area_sizes(paths: &AppPaths) -> BTreeMap<String, u64> {
    CacheArea::ALL.iter().map(|a| (a.dir_name().to_string(), fsutil::dir_size(&paths.cache(*a)))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clear_respects_protection_and_policy() {
        let d = tempfile::tempdir().unwrap();
        let p = AppPaths::resolve(Some(d.path()));
        p.ensure().unwrap();
        std::fs::create_dir_all(p.job_dir("running-job")).unwrap();
        std::fs::write(p.job_dir("running-job").join("seg.mkv"), vec![0u8; 1000]).unwrap();
        std::fs::create_dir_all(p.job_dir("old-job")).unwrap();
        std::fs::write(p.cache(CacheArea::Previews).join("a.mp4"), vec![0u8; 3 << 20]).unwrap();
        std::fs::write(p.cache(CacheArea::Previews).join("b.mp4"), vec![0u8; 3 << 20]).unwrap();
        // A non-cache file next to the cache must never be touched.
        std::fs::write(d.path().join("project.cfxproj"), b"keep").unwrap();
        let prot: HashSet<String> = ["running-job".to_string()].into_iter().collect();
        let rep = report(&p, &prot, 1);
        let jobs = rep.areas.iter().find(|a| a.area == CacheArea::Jobs).unwrap();
        assert_eq!(jobs.protected_bytes, 1000);
        // Size policy: 4 MB limit removes the oldest preview only.
        let r = enforce_policy(&p, 4, 0, &prot);
        assert_eq!(r.removed_entries, 1);
        let r = clear(&p, &CacheArea::ALL, &prot);
        assert_eq!(r.skipped_protected, 1);
        assert!(p.job_dir("running-job").join("seg.mkv").is_file());
        assert!(!p.job_dir("old-job").exists());
        assert!(d.path().join("project.cfxproj").is_file());
        let keep: HashSet<String> = HashSet::new();
        assert_eq!(remove_orphan_job_dirs(&p, &keep).removed_entries, 1);
    }
}
