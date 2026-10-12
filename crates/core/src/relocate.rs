//! Journaled relocation of large managed directories (models, cache).
//!
//! Protocol (crash-safe):
//! 1. `plan` — measure the data, validate the destination and free space.
//! 2. `copy` — write a journal, copy every file, verify sizes. The source is untouched.
//! 3. The caller switches its configuration to the destination.
//! 4. `finish` — delete the old copy and the journal.
//!
//! If the process stops at any point, `recover` decides from the journal and
//! the current configuration whether to roll back (delete the partial copy) or
//! roll forward (delete the old copy). Data is never deleted before a verified
//! copy exists and the configuration points at it.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use serde::{Deserialize, Serialize};

use crate::fsutil;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct MovePlan {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub total_bytes: u64,
    pub file_count: u64,
    pub destination_free: Option<u64>,
    /// Source and destination are on the same filesystem (informational).
    pub same_filesystem: bool,
    pub problems: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    Copying,
    Copied,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Journal {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub phase: Phase,
    pub started_at: String,
    /// Destination paths created by the copy (only these are removed on rollback).
    pub created: Vec<PathBuf>,
}

#[derive(Debug, thiserror::Error)]
pub enum MoveError {
    #[error("{0}")]
    Invalid(String),
    #[error("Not enough free space at the destination (needs {need_mb} MB, {have_mb} MB available).")]
    Space { need_mb: u64, have_mb: u64 },
    #[error("Cancelled; the original location is unchanged.")]
    Cancelled,
    #[error("Copy verification failed for {0}; the original location is unchanged.")]
    Verify(PathBuf),
    #[error("{0}")]
    Io(#[from] std::io::Error),
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    for e in std::fs::read_dir(dir)? {
        let e = e?;
        let t = e.file_type()?;
        if t.is_dir() {
            walk(&e.path(), out)?;
        } else if t.is_file() {
            out.push(e.path());
        }
        // Symlinks are not followed or copied.
    }
    Ok(())
}

/// Canonical form for comparing locations. A destination usually doesn't exist yet,
/// so canonicalise its nearest existing ancestor and re-append the rest. Otherwise a
/// new folder inside the source would not be recognised: on Windows the canonical
/// source has a `\\?\` prefix that the raw destination lacks, and on macOS `/var`
/// is really `/private/var`. Windows and macOS file systems ignore case, so compare
/// lower-cased there.
fn canon(p: &Path) -> PathBuf {
    let mut base = p.to_path_buf();
    let mut rest = Vec::new();
    while !base.exists() {
        match base.file_name() {
            Some(n) => {
                rest.push(n.to_os_string());
                base.pop();
            }
            None => break,
        }
    }
    let mut c = std::fs::canonicalize(&base).unwrap_or(base);
    for n in rest.into_iter().rev() {
        c.push(n);
    }
    if cfg!(any(windows, target_os = "macos")) {
        PathBuf::from(c.to_string_lossy().to_lowercase())
    } else {
        c
    }
}

#[cfg(unix)]
fn same_fs(a: &Path, b: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    let nearest = |p: &Path| {
        let mut p = p.to_path_buf();
        while !p.exists() && p.pop() {}
        std::fs::metadata(&p).map(|m| m.dev()).ok()
    };
    matches!((nearest(a), nearest(b)), (Some(x), Some(y)) if x == y)
}

#[cfg(not(unix))]
fn same_fs(a: &Path, b: &Path) -> bool {
    let root = |p: &Path| p.components().next().map(|c| c.as_os_str().to_ascii_lowercase());
    root(&canon(a)) == root(&canon(b))
}

/// Measure and validate a move.
pub fn plan(source: &Path, destination: &Path) -> MovePlan {
    let mut problems = Vec::new();
    let mut files = Vec::new();
    if source.is_dir() {
        if let Err(e) = walk(source, &mut files) {
            problems.push(format!("cannot read {}: {e}", source.display()));
        }
    }
    let total_bytes: u64 = files.iter().filter_map(|f| std::fs::metadata(f).ok()).map(|m| m.len()).sum();
    let (cs, cd) = (canon(source), canon(destination));
    if cd == cs {
        problems.push("The new location is the same as the current one.".into());
    } else if cd.starts_with(&cs) {
        problems.push("The new location cannot be inside the current one.".into());
    } else if cs.starts_with(&cd) {
        problems.push("The new location cannot contain the current one.".into());
    }
    if destination.exists() {
        if !destination.is_dir() {
            problems.push("The destination is not a folder.".into());
        } else if std::fs::read_dir(destination).map(|mut r| r.next().is_some()).unwrap_or(false) {
            problems.push("The destination folder is not empty. Choose an empty or new folder.".into());
        }
    }
    let parent = destination.parent().unwrap_or(destination);
    let probe_dir = if destination.exists() { destination } else { parent };
    if probe_dir.exists() {
        if let Err(e) = fsutil::ensure_writable_dir(probe_dir, false) {
            problems.push(e);
        }
    } else if !parent.as_os_str().is_empty() && fsutil::free_space(parent).is_none() {
        problems.push(format!("{} is not available.", parent.display()));
    }
    let same_filesystem = source.exists() && same_fs(source, destination);
    let destination_free = fsutil::free_space(destination);
    // A verified copy is always made first, so space is needed even on the same drive.
    {
        if let Some(free) = destination_free {
            if free < total_bytes + (64 << 20) {
                problems.push(format!("Not enough free space at the destination (needs {} MB, {} MB available).", total_bytes / 1_000_000 + 64, free / 1_000_000));
            }
        }
    }
    MovePlan { source: source.into(), destination: destination.into(), total_bytes, file_count: files.len() as u64, destination_free, same_filesystem, problems }
}

fn write_journal(path: &Path, j: &Journal) -> std::io::Result<()> {
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, serde_json::to_vec_pretty(j).unwrap_or_default())?;
    std::fs::rename(tmp, path)
}

pub fn read_journal(path: &Path) -> Option<Journal> {
    serde_json::from_slice(&std::fs::read(path).ok()?).ok()
}

/// Copy (or rename on the same filesystem) `source` to `destination`.
/// `progress(bytes_done, bytes_total)`. On error or cancel the destination copy is removed.
pub fn copy(source: &Path, destination: &Path, journal: &Path, cancel: &AtomicBool, progress: &mut dyn FnMut(u64, u64)) -> Result<(), MoveError> {
    let p = plan(source, destination);
    if let Some(e) = p.problems.first() {
        return Err(MoveError::Invalid(e.clone()));
    }
    let mut j = Journal { source: source.into(), destination: destination.into(), phase: Phase::Copying, started_at: chrono::Utc::now().to_rfc3339(), created: vec![] };
    write_journal(journal, &j)?;
    if !source.exists() {
        std::fs::create_dir_all(destination)?;
        j.created.push(destination.into());
        j.phase = Phase::Copied;
        write_journal(journal, &j)?;
        return Ok(());
    }
    let created_root = !destination.exists();
    std::fs::create_dir_all(destination)?;
    if created_root {
        j.created.push(destination.into());
    }
    let result = (|| -> Result<(), MoveError> {
        let mut files = Vec::new();
        walk(source, &mut files)?;
        let mut done = 0u64;
        let mut buf = vec![0u8; 4 << 20];
        for f in &files {
            let rel = f.strip_prefix(source).map_err(|_| MoveError::Invalid("bad path".into()))?;
            let dst = destination.join(rel);
            if let Some(par) = dst.parent() {
                std::fs::create_dir_all(par)?;
            }
            let mut r = std::fs::File::open(f)?;
            let mut w = std::fs::File::create(&dst)?;
            loop {
                if cancel.load(Ordering::SeqCst) {
                    return Err(MoveError::Cancelled);
                }
                use std::io::{Read, Write};
                let n = r.read(&mut buf)?;
                if n == 0 {
                    break;
                }
                w.write_all(&buf[..n])?;
                done += n as u64;
                progress(done, p.total_bytes);
            }
            w.sync_all()?;
            if std::fs::metadata(&dst)?.len() != std::fs::metadata(f)?.len() {
                return Err(MoveError::Verify(dst));
            }
        }
        Ok(())
    })();
    match result {
        Ok(()) => {
            j.phase = Phase::Copied;
            write_journal(journal, &j)?;
            Ok(())
        }
        Err(e) => {
            rollback(&j);
            let _ = std::fs::remove_file(journal);
            Err(e)
        }
    }
}

fn rollback(j: &Journal) {
    if j.created.iter().any(|c| c == &j.destination) {
        let _ = std::fs::remove_dir_all(&j.destination);
    } else if j.destination.is_dir() {
        // Destination existed (empty) before: remove only what we copied into it.
        if let Ok(rd) = std::fs::read_dir(&j.destination) {
            for e in rd.flatten() {
                let p = e.path();
                let _ = if p.is_dir() { std::fs::remove_dir_all(&p) } else { std::fs::remove_file(&p) };
            }
        }
    }
}

/// After the configuration points at the destination: remove the old data and the journal.
pub fn finish(journal: &Path) -> std::io::Result<()> {
    if let Some(j) = read_journal(journal) {
        if j.phase == Phase::Copied && j.source.exists() && canon(&j.source) != canon(&j.destination) {
            std::fs::remove_dir_all(&j.source)?;
        }
    }
    let _ = std::fs::remove_file(journal);
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Recovery {
    None,
    RolledBack,
    Finished,
}

/// Resolve an interrupted move. `active` is the location the configuration currently uses.
pub fn recover(journal: &Path, active: &Path) -> Recovery {
    let Some(j) = read_journal(journal) else { return Recovery::None };
    if j.phase == Phase::Copied && canon(active) == canon(&j.destination) {
        let _ = finish(journal);
        Recovery::Finished
    } else {
        rollback(&j);
        let _ = std::fs::remove_file(journal);
        Recovery::RolledBack
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tree(root: &Path) {
        std::fs::create_dir_all(root.join("a/ü b")).unwrap();
        std::fs::write(root.join("a/ü b/x.bin"), vec![7u8; 100_000]).unwrap();
        std::fs::write(root.join("top.txt"), b"hello").unwrap();
    }

    #[test]
    fn copy_finish_and_cancel() {
        let d = tempfile::tempdir().unwrap();
        let src = d.path().join("src");
        tree(&src);
        let dst = d.path().join("new place/dst");
        let j = d.path().join("move.journal");
        let p = plan(&src, &dst);
        assert!(p.problems.is_empty(), "{:?}", p.problems);
        assert_eq!(p.file_count, 2);
        let mut last = 0;
        copy(&src, &dst, &j, &AtomicBool::new(false), &mut |d, _| last = d).unwrap();
        assert_eq!(last, 100_005);
        assert!(src.exists() && dst.join("a/ü b/x.bin").is_file());
        finish(&j).unwrap();
        assert!(!src.exists() && !j.exists());
        // Cancel leaves the source untouched and removes the partial copy.
        let back = d.path().join("back");
        let r = copy(&dst, &back, &j, &AtomicBool::new(true), &mut |_, _| {});
        assert!(matches!(r, Err(MoveError::Cancelled)));
        assert!(dst.join("top.txt").is_file());
        assert!(!back.exists());
        assert!(!j.exists());
    }

    #[test]
    fn invalid_destinations() {
        let d = tempfile::tempdir().unwrap();
        let src = d.path().join("src");
        tree(&src);
        assert!(!plan(&src, &src.join("inner")).problems.is_empty());
        assert!(!plan(&src, &src.join("new").join("deeper")).problems.is_empty());
        assert!(!plan(&src, &src).problems.is_empty());
        // The same folder reached through a link must still count as "inside" (this is
        // what Windows' `\\?\` prefix and macOS' /var -> /private/var look like).
        #[cfg(unix)]
        {
            let link = d.path().join("link");
            std::os::unix::fs::symlink(&src, &link).unwrap();
            assert!(!plan(&src, &link.join("inner")).problems.is_empty());
            assert!(!plan(&link, &src.join("inner")).problems.is_empty());
        }
        let full = d.path().join("full");
        std::fs::create_dir_all(&full).unwrap();
        std::fs::write(full.join("f"), b"x").unwrap();
        assert!(!plan(&src, &full).problems.is_empty());
    }

    #[test]
    fn crash_recovery_both_directions() {
        let d = tempfile::tempdir().unwrap();
        let src = d.path().join("src");
        tree(&src);
        let dst = d.path().join("dst");
        let j = d.path().join("j");
        copy(&src, &dst, &j, &AtomicBool::new(false), &mut |_, _| {}).unwrap();
        // Crash before the config switched -> roll back.
        assert_eq!(recover(&j, &src), Recovery::RolledBack);
        assert!(src.join("top.txt").is_file() && !dst.exists());
        copy(&src, &dst, &j, &AtomicBool::new(false), &mut |_, _| {}).unwrap();
        // Crash after the config switched -> roll forward.
        assert_eq!(recover(&j, &dst), Recovery::Finished);
        assert!(!src.exists() && dst.join("top.txt").is_file());
        assert_eq!(recover(&j, &dst), Recovery::None);
    }
}
