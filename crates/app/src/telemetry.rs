//! Diagnostics reports: the app's own activity log, sent to the developer so problems can be found
//! and fixed in later versions.
//!
//! - Only the software's log is sent (startup, errors, crashes, job settings and speeds, hardware,
//!   performance samples). File paths, file names and user names are removed before sending.
//! - Nothing is lost offline: the position up to which the log was sent is kept on disk, and new
//!   log data is sent in the background with retries as soon as a connection is available. That
//!   also covers sessions where the app crashed or never opened: their log lines go out on the
//!   next launch.
//! - On by default, switched off in Preferences → Privacy (`share_diagnostics`).
//! - The receiving address is set when the release is built (`CLIPFX_DIAGNOSTICS_URL`); builds
//!   without one never send anything.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::App;

/// Where reports go; fixed at build time.
pub const ENDPOINT: Option<&str> = option_env!("CLIPFX_DIAGNOSTICS_URL");

/// Largest piece of log sent in one report (the newest part is kept if there is more).
const MAX_CHUNK: u64 = 4 << 20;
/// How often to look for new log data once caught up.
const INTERVAL: Duration = Duration::from_secs(10 * 60);

#[derive(Debug, Default, Serialize, Deserialize)]
struct State {
    /// Random id for this installation (not derived from the machine or the user).
    install_id: String,
    /// Bytes of the session log already sent.
    sent_offset: u64,
    /// Size of the log when last read, to notice when it was trimmed and restarted.
    #[serde(default)]
    seen_len: u64,
}

fn state_path(app: &App) -> PathBuf {
    app.paths().data_dir.join("diagnostics-state.json")
}

fn load(path: &Path) -> State {
    let mut s: State = std::fs::read(path).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or_default();
    if s.install_id.is_empty() {
        s.install_id = uuid::Uuid::new_v4().to_string();
    }
    s
}

fn save(path: &Path, s: &State) {
    if let Ok(b) = serde_json::to_vec_pretty(s) {
        let tmp = path.with_extension("tmp");
        if std::fs::write(&tmp, b).is_ok() {
            let _ = std::fs::rename(&tmp, path);
        }
    }
}

/// Remove anything personal from log text: home folder, user name, file and folder paths, and
/// video file names.
pub fn redact(text: &str) -> String {
    struct R {
        file_path: regex::Regex,
        dir_path: regex::Regex,
        media_name: regex::Regex,
    }
    static RE: OnceLock<R> = OnceLock::new();
    let r = RE.get_or_init(|| R {
        // A path (drive, UNC, Unix home or temp) ending in a file name with an extension.
        file_path: regex::Regex::new(r#"(?i)(?:[a-z]:[\\/]|\\\\[^\\/\s"']+[\\/]|/(?:home|users|tmp|var|mnt|media|volumes|run|private|root)/|~[\\/])[^"'\n\r<>|*?]*?\.([a-z0-9]{1,5})\b"#).unwrap(),
        // Any remaining folder path.
        dir_path: regex::Regex::new(r#"(?i)(?:[a-z]:[\\/]|\\\\[^\\/\s"']+[\\/]|/(?:home|users|tmp|var|mnt|media|volumes|run|private|root)/|~[\\/])[^"'\n\r<>|,;)\]}]*"#).unwrap(),
        // Bare media or project file names ("holiday clip.mp4").
        media_name: regex::Regex::new(r#"(?i)[^\s"'/\\<>|:]+(?: [^\s"'/\\<>|:]+){0,6}\.(mp4|m4v|mov|mkv|avi|webm|wmv|flv|mpg|mpeg|ts|mts|m2ts|3gp|mxf|ogv|vob|cfxproj|cfxpreset|png|jpg|jpeg|tif|tiff|exr|dpx)\b"#).unwrap(),
    });
    let mut out = text.to_string();
    for var in ["USERPROFILE", "HOME"] {
        if let Ok(h) = std::env::var(var) {
            if h.len() > 3 {
                out = out.replace(&h, "~").replace(&h.replace('\\', "\\\\"), "~");
            }
        }
    }
    if let Ok(user) = std::env::var("USERNAME").or_else(|_| std::env::var("USER")) {
        if user.len() >= 3 {
            out = out.replace(&user, "<user>");
        }
    }
    let out = r.file_path.replace_all(&out, "<file>.$1");
    let out = r.dir_path.replace_all(&out, "<folder>");
    r.media_name.replace_all(&out, "<file>.$1").into_owned()
}

/// Read the session log from `offset` (at most `MAX_CHUNK`, newest part). Returns the text, the
/// new offset, and how many bytes were skipped.
fn read_new(path: &Path, offset: u64) -> std::io::Result<(String, u64, u64)> {
    let mut f = std::fs::File::open(path)?;
    let len = f.metadata()?.len();
    let start = offset.max(len.saturating_sub(MAX_CHUNK));
    f.seek(SeekFrom::Start(start))?;
    let mut buf = Vec::with_capacity((len - start) as usize);
    f.take(len - start).read_to_end(&mut buf)?;
    // Only whole lines: a line still being written goes in the next report.
    let end = buf.iter().rposition(|&b| b == b'\n').map(|i| i + 1).unwrap_or(0);
    buf.truncate(end);
    Ok((String::from_utf8_lossy(&buf).into_owned(), start + end as u64, start - offset.min(start)))
}

fn send(url: &str, install_id: &str, header: &serde_json::Value, body: &str) -> Result<(), String> {
    let mut gz = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
    gz.write_all(header.to_string().as_bytes()).and_then(|_| gz.write_all(b"\n")).and_then(|_| gz.write_all(body.as_bytes())).map_err(|e| e.to_string())?;
    let data = gz.finish().map_err(|e| e.to_string())?;
    let agent = ureq::AgentBuilder::new()
        .timeout(Duration::from_secs(60))
        .user_agent(&format!("ClipFX-Enhance/{}", env!("CARGO_PKG_VERSION")))
        .try_proxy_from_env(clearframe_models::manager::use_proxy_for(url))
        .build();
    agent
        .post(url)
        .set("Content-Type", "application/x-ndjson")
        .set("Content-Encoding", "gzip")
        .set("X-ClipFX-Install", install_id)
        .set("X-ClipFX-Version", env!("CARGO_PKG_VERSION"))
        .send_bytes(&data)
        .map(|_| ())
        .map_err(|e| e.to_string())
}

/// Start the background sender (no-op without a configured endpoint).
pub fn start(app: Arc<App>) {
    // The environment variable overrides the built-in address (testing, staging).
    let url: Option<&'static str> = std::env::var("CLIPFX_DIAGNOSTICS_URL").ok().map(|u| &*Box::leak(u.into_boxed_str())).or(ENDPOINT);
    let Some(url) = url.filter(|u| u.starts_with("https://") || u.starts_with("http://127.0.0.1")) else {
        log::debug!("diagnostics reports: no endpoint in this build");
        return;
    };
    let _ = std::thread::Builder::new().name("diagnostics".into()).spawn(move || {
        let sp = state_path(&app);
        let mut backoff = Duration::from_secs(30);
        // Let startup finish first.
        std::thread::sleep(Duration::from_secs(20));
        loop {
            let wait = if !app.settings().share_diagnostics {
                INTERVAL
            } else {
                match send_pending(&app, url, &sp) {
                    Ok(true) => {
                        backoff = Duration::from_secs(30);
                        Duration::from_secs(5) // more to send
                    }
                    Ok(false) => {
                        backoff = Duration::from_secs(30);
                        INTERVAL
                    }
                    Err(e) => {
                        // Offline or the server is unreachable: keep the data and try again later.
                        log::debug!("diagnostics report not sent yet: {e}");
                        let w = backoff;
                        backoff = (backoff * 2).min(INTERVAL);
                        w
                    }
                }
            };
            std::thread::sleep(wait);
        }
    });
}

/// Send the next piece of unsent log. `Ok(true)` when more remains.
fn send_pending(app: &App, url: &str, sp: &Path) -> Result<bool, String> {
    let log_path = crate::logging::session_log_path().ok_or("no session log")?;
    let mut st = load(sp);
    let len = std::fs::metadata(&log_path).map(|m| m.len()).map_err(|e| e.to_string())?;
    if len < st.sent_offset || len < st.seen_len.min(st.sent_offset) {
        // The log was trimmed at startup; start again from its beginning.
        st.sent_offset = 0;
    }
    st.seen_len = len;
    if len <= st.sent_offset {
        save(sp, &st);
        return Ok(false);
    }
    let (text, new_offset, skipped) = read_new(&log_path, st.sent_offset).map_err(|e| e.to_string())?;
    if text.is_empty() {
        return Ok(false);
    }
    let hw = app.hardware();
    let header = serde_json::json!({
        "kind": "clipfx-log",
        "install_id": st.install_id,
        "app_version": env!("CARGO_PKG_VERSION"),
        "os": std::env::consts::OS,
        "arch": std::env::consts::ARCH,
        "os_version": hw.os_version,
        "cpu": hw.cpu_brand,
        "logical_cores": hw.logical_cores,
        "ram_mb": hw.ram_total_mb,
        "sent_at": chrono::Utc::now().to_rfc3339(),
        "log_from": st.sent_offset,
        "log_to": new_offset,
        "skipped_bytes": skipped,
    });
    send(url, &st.install_id, &header, &redact(&text))?;
    st.sent_offset = new_offset;
    save(sp, &st);
    Ok(new_offset < len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn removes_paths_names_and_users() {
        let s = r#"2026 job 1 'x': C:\Users\Ann Lee\Videos\our wedding 2024.mp4 -> D:\Exports\out_enhance.mp4 cwd: Some("C:\\Program Files\\ClipFX") path /home/ann/clips/a b.mkv toast: Saved "holiday clip.mov" folder /tmp/cfx/work/seg"#;
        let r = redact(s);
        for leak in ["Ann", "wedding", "Exports", "holiday", "ann/", "clips", "Program Files"] {
            assert!(!r.contains(leak), "{leak} leaked: {r}");
        }
        assert!(r.contains("<file>.mp4") && r.contains("<file>.mkv") && r.contains("<file>.mov"), "{r}");
        assert!(r.contains("job 1"));
    }

    #[test]
    fn sends_gzip_report_with_header() {
        use std::io::BufRead;
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/r", l.local_addr().unwrap());
        let h = std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            let mut r = std::io::BufReader::new(c.try_clone().unwrap());
            let mut len = 0usize;
            let mut headers = String::new();
            loop {
                let mut line = String::new();
                r.read_line(&mut line).unwrap();
                if line == "\r\n" {
                    break;
                }
                if let Some(v) = line.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
                headers.push_str(&line);
            }
            let mut body = vec![0u8; len];
            r.read_exact(&mut body).unwrap();
            c.write_all(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\n\r\n").unwrap();
            let mut text = String::new();
            flate2::read::GzDecoder::new(&body[..]).read_to_string(&mut text).unwrap();
            (headers, text)
        });
        send(&url, "id-1", &serde_json::json!({"kind": "clipfx-log"}), "line one\n").unwrap();
        let (headers, text) = h.join().unwrap();
        assert!(headers.to_ascii_lowercase().contains("x-clipfx-install: id-1"));
        assert_eq!(text, "{\"kind\":\"clipfx-log\"}\nline one\n");
    }

    #[test]
    fn offline_send_fails_and_keeps_data() {
        // Nothing listens on this port: the report is not sent and the caller keeps its offset.
        assert!(send("http://127.0.0.1:9/r", "id", &serde_json::json!({}), "x\n").is_err());
    }

    #[test]
    fn reads_only_whole_new_lines() {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("log.txt");
        std::fs::write(&p, "a\nb\npartial").unwrap();
        let (t, off, skipped) = read_new(&p, 2).unwrap();
        assert_eq!((t.as_str(), off, skipped), ("b\n", 4, 0));
        let (t, off, _) = read_new(&p, off).unwrap();
        assert_eq!((t.as_str(), off), ("", 4));
    }
}
