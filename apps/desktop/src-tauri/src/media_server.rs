//! Local HTTP server for video playback.
//!
//! WebKitGTK plays `<video>` through GStreamer, which does not reliably load
//! media from Tauri's custom `asset://` scheme. Videos are therefore served
//! over HTTP on 127.0.0.1 with Range support. Every URL carries a random
//! per-run token, and only allow-listed files (the cache folder and files the
//! user opened) are served.

use std::collections::HashSet;
use std::hash::{BuildHasher, Hasher};
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::net::{TcpListener, TcpStream};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

struct Server {
    port: u16,
    token: String,
    dirs: Mutex<Vec<PathBuf>>,
    files: Mutex<HashSet<PathBuf>>,
}

static SERVER: OnceLock<Server> = OnceLock::new();

fn random_token() -> String {
    // RandomState is seeded from the OS random source.
    let mut parts = String::new();
    for i in 0..2u64 {
        let mut h = std::collections::hash_map::RandomState::new().build_hasher();
        h.write_u64(i ^ std::process::id() as u64);
        h.write_u128(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0));
        parts.push_str(&format!("{:016x}", h.finish()));
    }
    parts
}

/// Start the server (once). Returns the URL prefix, e.g. `http://127.0.0.1:41234/<token>/`.
pub fn start() -> std::io::Result<String> {
    if let Some(s) = SERVER.get() {
        return Ok(base(s));
    }
    let listener = TcpListener::bind(("127.0.0.1", 0))?;
    let port = listener.local_addr()?.port();
    let server = Server { port, token: random_token(), dirs: Mutex::new(Vec::new()), files: Mutex::new(HashSet::new()) };
    let s = SERVER.get_or_init(|| server);
    std::thread::Builder::new().name("media-server".into()).spawn(move || {
        for conn in listener.incoming().flatten() {
            std::thread::spawn(move || {
                let _ = handle(conn);
            });
        }
    })?;
    Ok(base(s))
}

fn base(s: &Server) -> String {
    format!("http://127.0.0.1:{}/{}/", s.port, s.token)
}

pub fn allow_dir(dir: &Path) {
    if let (Some(s), Ok(d)) = (SERVER.get(), dir.canonicalize()) {
        s.dirs.lock().unwrap().push(d);
    }
}

pub fn allow_file(file: &Path) {
    if let (Some(s), Ok(f)) = (SERVER.get(), file.canonicalize()) {
        s.files.lock().unwrap().insert(f);
    }
}

fn allowed(s: &Server, p: &Path) -> bool {
    let Ok(c) = p.canonicalize() else { return false };
    s.files.lock().unwrap().contains(&c) || s.dirs.lock().unwrap().iter().any(|d| c.starts_with(d))
}

fn percent_decode(s: &str) -> Option<String> {
    let b = s.as_bytes();
    let mut out = Vec::with_capacity(b.len());
    let mut i = 0;
    while i < b.len() {
        if b[i] == b'%' && i + 2 < b.len() {
            out.push(u8::from_str_radix(std::str::from_utf8(&b[i + 1..i + 3]).ok()?, 16).ok()?);
            i += 3;
        } else {
            out.push(b[i]);
            i += 1;
        }
    }
    String::from_utf8(out).ok()
}

fn content_type(p: &Path) -> &'static str {
    match p.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase()).as_deref() {
        Some("mp4") | Some("m4v") => "video/mp4",
        Some("webm") => "video/webm",
        Some("mov") => "video/quicktime",
        Some("mkv") => "video/x-matroska",
        _ => "application/octet-stream",
    }
}

fn respond(conn: &mut TcpStream, status: &str, extra: &str) -> std::io::Result<()> {
    write!(conn, "HTTP/1.1 {status}\r\nContent-Length: 0\r\nConnection: close\r\n{extra}\r\n")
}

fn handle(mut conn: TcpStream) -> std::io::Result<()> {
    let s = SERVER.get().expect("server started");
    let mut reader = BufReader::new(conn.try_clone()?);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    let mut parts = line.split_whitespace();
    let method = parts.next().unwrap_or("");
    let target = parts.next().unwrap_or("");
    let mut range: Option<(u64, Option<u64>)> = None;
    loop {
        let mut h = String::new();
        if reader.read_line(&mut h)? == 0 || h == "\r\n" || h == "\n" {
            break;
        }
        let lower = h.to_ascii_lowercase();
        if let Some(v) = lower.strip_prefix("range:") {
            if let Some(spec) = v.trim().strip_prefix("bytes=") {
                let (a, b) = spec.split_once('-').unwrap_or((spec, ""));
                if let Ok(a) = a.trim().parse::<u64>() {
                    range = Some((a, b.trim().parse::<u64>().ok()));
                }
            }
        }
    }
    if method != "GET" && method != "HEAD" {
        return respond(&mut conn, "405 Method Not Allowed", "");
    }
    let prefix = format!("/{}/", s.token);
    let Some(rest) = target.strip_prefix(&prefix) else { return respond(&mut conn, "403 Forbidden", "") };
    let Some(decoded) = percent_decode(rest.split('?').next().unwrap_or("")) else { return respond(&mut conn, "400 Bad Request", "") };
    // URLs carry the absolute path without its leading slash on Unix.
    let path = if cfg!(windows) { PathBuf::from(decoded) } else { PathBuf::from(format!("/{decoded}")) };
    if !path.is_absolute() || !allowed(s, &path) {
        return respond(&mut conn, "403 Forbidden", "");
    }
    let Ok(mut file) = std::fs::File::open(&path) else { return respond(&mut conn, "404 Not Found", "") };
    let len = file.metadata()?.len();
    let (start, end) = match range {
        Some((a, b)) if a < len => (a, b.unwrap_or(len - 1).min(len - 1)),
        Some(_) => return respond(&mut conn, "416 Range Not Satisfiable", &format!("Content-Range: bytes */{len}\r\n")),
        None => (0, len.saturating_sub(1)),
    };
    let count = if len == 0 { 0 } else { end - start + 1 };
    let status = if range.is_some() { "206 Partial Content" } else { "200 OK" };
    let mut head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: {}\r\nContent-Length: {count}\r\nAccept-Ranges: bytes\r\nCache-Control: no-store\r\nConnection: close\r\n",
        content_type(&path)
    );
    if range.is_some() {
        head.push_str(&format!("Content-Range: bytes {start}-{end}/{len}\r\n"));
    }
    head.push_str("\r\n");
    conn.write_all(head.as_bytes())?;
    if method == "HEAD" || count == 0 {
        return Ok(());
    }
    file.seek(SeekFrom::Start(start))?;
    let mut left = count;
    let mut buf = vec![0u8; 256 * 1024];
    while left > 0 {
        let take = left.min(buf.len() as u64) as usize;
        let n = file.read(&mut buf[..take])?;
        if n == 0 {
            break;
        }
        conn.write_all(&buf[..n])?;
        left -= n as u64;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn get(url: &str, range: Option<&str>) -> (String, Vec<u8>) {
        let rest = url.strip_prefix("http://").unwrap();
        let (host, path) = rest.split_once('/').unwrap();
        let mut c = TcpStream::connect(host).unwrap();
        write!(c, "GET /{path} HTTP/1.1\r\nHost: {host}\r\n{}\r\n", range.map(|r| format!("Range: {r}\r\n")).unwrap_or_default()).unwrap();
        let mut all = Vec::new();
        c.read_to_end(&mut all).unwrap();
        let split = all.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
        (String::from_utf8_lossy(&all[..split]).to_string(), all[split + 4..].to_vec())
    }

    fn enc(p: &Path) -> String {
        p.to_string_lossy().bytes().map(|b| if b.is_ascii_alphanumeric() || b"/._-".contains(&b) { (b as char).to_string() } else { format!("%{b:02X}") }).collect()
    }

    #[test]
    fn serves_allowed_files_with_ranges_only() {
        let base = start().unwrap();
        let d = tempfile::tempdir().unwrap();
        let ok_dir = d.path().join("cache dir ü");
        std::fs::create_dir_all(&ok_dir).unwrap();
        let f = ok_dir.join("clip.mp4");
        std::fs::write(&f, (0u8..200).collect::<Vec<_>>()).unwrap();
        let secret = d.path().join("secret.txt");
        std::fs::write(&secret, b"no").unwrap();
        allow_dir(&ok_dir);

        let url = format!("{base}{}", enc(&f.canonicalize().unwrap()).trim_start_matches('/'));
        let (h, body) = get(&url, None);
        assert!(h.starts_with("HTTP/1.1 200"), "{h}");
        assert_eq!(body.len(), 200);
        let (h, body) = get(&url, Some("bytes=10-19"));
        assert!(h.starts_with("HTTP/1.1 206") && h.contains("Content-Range: bytes 10-19/200"), "{h}");
        assert_eq!(body, (10u8..20).collect::<Vec<_>>());
        let (h, body) = get(&url, Some("bytes=190-"));
        assert!(h.starts_with("HTTP/1.1 206"), "{h}");
        assert_eq!(body.len(), 10);

        let (h, _) = get(&format!("{base}{}", enc(&secret.canonicalize().unwrap()).trim_start_matches('/')), None);
        assert!(h.starts_with("HTTP/1.1 403"), "{h}");
        let traversal = format!("{base}{}/..%2Fsecret.txt", enc(&ok_dir.canonicalize().unwrap()).trim_start_matches('/'));
        assert!(get(&traversal, None).0.starts_with("HTTP/1.1 403"));
        let token = base.trim_end_matches('/').rsplit('/').next().unwrap();
        let wrong_token = url.replacen(token, &"0".repeat(token.len()), 1);
        assert!(get(&wrong_token, None).0.starts_with("HTTP/1.1 403"));
    }
}
