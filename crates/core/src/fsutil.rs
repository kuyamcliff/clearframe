//! Filesystem safety helpers: output naming, collision handling, atomic
//! no-overwrite finalisation, writability and free-space checks.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Characters that are invalid in file names on at least one supported OS.
const INVALID: &[char] = &['<', '>', ':', '"', '/', '\\', '|', '?', '*'];

/// Make a string safe to use as a single path component on Windows, macOS and Linux.
pub fn sanitize_component(s: &str) -> String {
    let mut out: String = s.chars().map(|c| if INVALID.contains(&c) || c.is_control() { '_' } else { c }).collect();
    out = out.trim().trim_end_matches(['.', ' ']).to_string();
    if out.is_empty() {
        out = "output".into();
    }
    // Reserved device names on Windows.
    let upper = out.to_ascii_uppercase();
    let stem = upper.split('.').next().unwrap_or("");
    const RESERVED: &[&str] = &[
        "CON", "PRN", "AUX", "NUL", "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9", "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];
    if RESERVED.contains(&stem) {
        out = format!("_{out}");
    }
    if out.len() > 200 {
        let mut cut = 200;
        while !out.is_char_boundary(cut) {
            cut -= 1;
        }
        out.truncate(cut);
    }
    out
}

/// Values available to the output naming template.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct NameVars {
    pub source_stem: String,
    pub task: String,
    pub model: String,
    pub width: u32,
    pub height: u32,
    pub fps: String,
    pub preset: String,
    pub date: String,
}

/// Expand a naming template such as `{name}_{task}_{width}x{height}`.
/// Unknown placeholders are left verbatim so mistakes are visible.
pub fn expand_template(template: &str, v: &NameVars) -> String {
    let t = if template.trim().is_empty() { "{name}_clearframe" } else { template };
    let s = t
        .replace("{name}", &v.source_stem)
        .replace("{task}", &v.task)
        .replace("{model}", &v.model)
        .replace("{width}", &v.width.to_string())
        .replace("{height}", &v.height.to_string())
        .replace("{fps}", &v.fps)
        .replace("{preset}", &v.preset)
        .replace("{date}", &v.date);
    sanitize_component(&s)
}

/// Return `dir/stem.ext`, or `dir/stem_2.ext`, `_3`, ... if it exists or equals `avoid`.
pub fn unique_path(dir: &Path, stem: &str, ext: &str, avoid: &[&Path]) -> PathBuf {
    let make = |n: u32| {
        let name = if n <= 1 { format!("{stem}.{ext}") } else { format!("{stem}_{n}.{ext}") };
        dir.join(name)
    };
    let mut n = 1;
    loop {
        let p = make(n);
        let clash = p.exists() || avoid.iter().any(|a| same_file(a, &p));
        if !clash {
            return p;
        }
        n += 1;
    }
}

fn same_file(a: &Path, b: &Path) -> bool {
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) {
        (Ok(x), Ok(y)) => x == y,
        _ => a == b,
    }
}

/// Temporary file name in the destination directory used while rendering.
pub fn temp_output_path(final_path: &Path, job_id: &str) -> PathBuf {
    let dir = final_path.parent().unwrap_or(Path::new("."));
    let ext = final_path.extension().and_then(|e| e.to_str()).unwrap_or("tmp");
    dir.join(format!(".clearframe-{}.partial.{}", sanitize_component(job_id), ext))
}

#[derive(Debug, thiserror::Error)]
pub enum FinalizeError {
    #[error("the destination {0} already exists; it was not overwritten")]
    Exists(PathBuf),
    #[error("refusing to overwrite the source file {0}")]
    IsSource(PathBuf),
    #[error("could not move the finished file into place: {0}")]
    Io(#[from] std::io::Error),
}

/// Move `tmp` to `dest` without ever replacing an existing file.
/// On Unix this uses a hard link (atomic, fails if `dest` exists) followed by
/// removing the temporary name. Elsewhere it checks and renames.
pub fn finalize_no_overwrite(tmp: &Path, dest: &Path, source: Option<&Path>) -> Result<(), FinalizeError> {
    if let Some(src) = source {
        if same_file(src, dest) {
            return Err(FinalizeError::IsSource(dest.to_path_buf()));
        }
    }
    if dest.exists() {
        return Err(FinalizeError::Exists(dest.to_path_buf()));
    }
    #[cfg(unix)]
    {
        match std::fs::hard_link(tmp, dest) {
            Ok(()) => {
                std::fs::remove_file(tmp)?;
                return Ok(());
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => return Err(FinalizeError::Exists(dest.to_path_buf())),
            Err(_) => {} // e.g. filesystems without hard links: fall through
        }
    }
    if dest.exists() {
        return Err(FinalizeError::Exists(dest.to_path_buf()));
    }
    std::fs::rename(tmp, dest)?;
    Ok(())
}

/// Check a directory exists (creating it if asked) and is writable.
pub fn ensure_writable_dir(dir: &Path, create: bool) -> Result<(), String> {
    if !dir.exists() {
        if create {
            std::fs::create_dir_all(dir).map_err(|e| format!("could not create folder {}: {e}", dir.display()))?;
        } else {
            return Err(format!("folder {} does not exist", dir.display()));
        }
    }
    if !dir.is_dir() {
        return Err(format!("{} is not a folder", dir.display()));
    }
    let probe = dir.join(format!(".clearframe-write-test-{}", std::process::id()));
    match std::fs::write(&probe, b"ok") {
        Ok(()) => {
            let _ = std::fs::remove_file(&probe);
            Ok(())
        }
        Err(e) => Err(format!("folder {} is not writable: {e}", dir.display())),
    }
}

/// Free bytes on the filesystem containing `path` (nearest existing ancestor).
pub fn free_space(path: &Path) -> Option<u64> {
    let mut p = path.to_path_buf();
    while !p.exists() {
        if !p.pop() {
            return None;
        }
    }
    fs2::available_space(&p).ok()
}

/// Total size of all files below `dir`.
pub fn dir_size(dir: &Path) -> u64 {
    let mut total = 0;
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            match e.file_type() {
                Ok(t) if t.is_dir() => total += dir_size(&p),
                Ok(t) if t.is_file() => total += e.metadata().map(|m| m.len()).unwrap_or(0),
                _ => {}
            }
        }
    }
    total
}

/// SHA-256 of a file as lowercase hex.
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::Read;
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex::encode(h.finalize()))
}

/// Cheap source fingerprint: size + mtime + SHA-256 of the first and last MiB.
pub fn quick_fingerprint(path: &Path) -> std::io::Result<String> {
    use sha2::{Digest, Sha256};
    use std::io::{Read, Seek, SeekFrom};
    let meta = std::fs::metadata(path)?;
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    h.update(meta.len().to_le_bytes());
    let mut buf = vec![0u8; 1 << 20];
    let n = f.read(&mut buf)?;
    h.update(&buf[..n]);
    if meta.len() > (2 << 20) {
        f.seek(SeekFrom::End(-(1 << 20)))?;
        let n = f.read(&mut buf)?;
        h.update(&buf[..n]);
    }
    Ok(hex::encode(&h.finalize()[..16]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize() {
        assert_eq!(sanitize_component("a/b:c*?.mp4"), "a_b_c__.mp4");
        assert_eq!(sanitize_component("CON"), "_CON");
        assert_eq!(sanitize_component("  name. "), "name");
        assert_eq!(sanitize_component(""), "output");
    }

    #[test]
    fn template() {
        let v = NameVars { source_stem: "clip".into(), task: "enhance".into(), width: 3840, height: 2160, ..Default::default() };
        assert_eq!(expand_template("{name}_{task}_{width}x{height}", &v), "clip_enhance_3840x2160");
        assert_eq!(expand_template("", &v), "clip_clearframe");
        assert_eq!(expand_template("{name}/{bogus}", &v), "clip_{bogus}");
    }

    #[test]
    fn unique_and_finalize() {
        let d = tempfile::tempdir().unwrap();
        let p1 = unique_path(d.path(), "out", "mp4", &[]);
        assert!(p1.ends_with("out.mp4"));
        std::fs::write(&p1, b"x").unwrap();
        let p2 = unique_path(d.path(), "out", "mp4", &[]);
        assert!(p2.ends_with("out_2.mp4"));
        let tmp = temp_output_path(&p2, "job1");
        std::fs::write(&tmp, b"new").unwrap();
        // Refuses to overwrite.
        assert!(matches!(finalize_no_overwrite(&tmp, &p1, None), Err(FinalizeError::Exists(_))));
        assert_eq!(std::fs::read(&p1).unwrap(), b"x");
        finalize_no_overwrite(&tmp, &p2, None).unwrap();
        assert_eq!(std::fs::read(&p2).unwrap(), b"new");
        assert!(!tmp.exists());
    }

    #[test]
    fn refuses_source() {
        let d = tempfile::tempdir().unwrap();
        let src = d.path().join("src.mp4");
        std::fs::write(&src, b"s").unwrap();
        let tmp = d.path().join("t");
        std::fs::write(&tmp, b"t").unwrap();
        assert!(matches!(finalize_no_overwrite(&tmp, &src, Some(&src)), Err(FinalizeError::IsSource(_))));
    }

    #[test]
    fn writable() {
        let d = tempfile::tempdir().unwrap();
        let sub = d.path().join("a/b");
        assert!(ensure_writable_dir(&sub, false).is_err());
        ensure_writable_dir(&sub, true).unwrap();
        assert!(free_space(&sub).unwrap() > 0);
    }
}
