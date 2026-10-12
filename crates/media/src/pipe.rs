//! Raw RGB48 frame pipes to and from FFmpeg processes.
//!
//! All frames crossing process boundaries are packed little-endian RGB48
//! (`rgb48le`) so 10/12-bit sources keep their precision.

use std::collections::VecDeque;
use std::io::{BufReader, Read, Write};
use std::path::PathBuf;
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use std::sync::Arc;
use std::thread::JoinHandle;

use crossbeam_channel::{bounded, unbounded, Receiver, Sender};
use parking_lot::Mutex;

use crate::color::{ColorTags, YuvMatrix};
use crate::encode::VideoEncodeSettings;
use crate::rational::Rational;
use crate::tools::FfmpegTools;
use crate::MediaError;

/// A packed RGB48LE frame.
#[derive(Clone, Debug, PartialEq)]
pub struct RawFrame {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

impl RawFrame {
    pub fn byte_len(width: usize, height: usize) -> usize {
        width * height * 6
    }
}

/// Keeps the last lines written to stderr by a child process.
#[derive(Clone, Default)]
pub struct StderrLog(Arc<Mutex<VecDeque<String>>>);

impl StderrLog {
    fn spawn_reader<R: Read + Send + 'static>(&self, r: R) -> JoinHandle<()> {
        let log = self.0.clone();
        std::thread::spawn(move || {
            let mut reader = BufReader::new(r);
            let mut buf = Vec::new();
            let mut byte = [0u8; 1];
            loop {
                match reader.read(&mut byte) {
                    Ok(0) | Err(_) => break,
                    Ok(_) => {
                        if byte[0] == b'\n' || byte[0] == b'\r' {
                            if !buf.is_empty() {
                                let line = String::from_utf8_lossy(&buf).into_owned();
                                let mut l = log.lock();
                                l.push_back(line);
                                while l.len() > 200 {
                                    l.pop_front();
                                }
                                buf.clear();
                            }
                        } else if buf.len() < 4096 {
                            buf.push(byte[0]);
                        }
                    }
                }
            }
        })
    }

    pub fn tail(&self, n: usize) -> String {
        let l = self.0.lock();
        l.iter().rev().take(n).cloned().collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n")
    }

    /// Lines that look like real errors (ignores informational output).
    pub fn errors(&self) -> String {
        let l = self.0.lock();
        let v: Vec<String> = l.iter().filter(|s| !s.trim().is_empty()).cloned().collect();
        v.into_iter().rev().take(12).collect::<Vec<_>>().into_iter().rev().collect::<Vec<_>>().join("\n")
    }
}

fn spawn(mut cmd: Command, stdin: bool, stdout: bool) -> Result<(Child, StderrLog, JoinHandle<()>), MediaError> {
    cmd.stdin(if stdin { Stdio::piped() } else { Stdio::null() })
        .stdout(if stdout { Stdio::piped() } else { Stdio::null() })
        .stderr(Stdio::piped());
    log::debug!("spawn {:?}", cmd);
    let mut child = cmd.spawn().map_err(|e| MediaError::Spawn(e.to_string()))?;
    let log = StderrLog::default();
    let h = log.spawn_reader(child.stderr.take().unwrap());
    Ok((child, log, h))
}

/// Parameters for decoding a source into RGB48 frames.
#[derive(Debug, Clone)]
pub struct DecodeSpec {
    pub input: PathBuf,
    pub stream_index: Option<usize>,
    pub start: Option<f64>,
    pub duration: Option<f64>,
    /// FFmpeg filters applied in the source's native format before RGB conversion.
    pub filters: Vec<String>,
    /// Force constant frame rate at this rate (VFR handling).
    pub cfr: Option<Rational>,
    pub matrix: YuvMatrix,
    pub full_range: bool,
    /// Dimensions of frames after `filters`.
    pub width: usize,
    pub height: usize,
    pub threads: Option<usize>,
    /// Extra input options placed before `-i` (e.g. `-f concat -safe 0`).
    pub input_args: Vec<String>,
    /// Working directory for the process (relative filter paths resolve here).
    pub cwd: Option<PathBuf>,
}

impl DecodeSpec {
    /// A spec for decoding `input` at its native size.
    pub fn new(input: PathBuf, width: usize, height: usize) -> Self {
        Self {
            input,
            stream_index: None,
            start: None,
            duration: None,
            filters: Vec::new(),
            cfr: None,
            matrix: YuvMatrix::Bt709,
            full_range: false,
            width,
            height,
            threads: None,
            input_args: Vec::new(),
            cwd: None,
        }
    }

    pub fn command(&self, tools: &FfmpegTools) -> Command {
        let mut c = tools.ffmpeg_cmd();
        if let Some(d) = &self.cwd {
            c.current_dir(d);
        }
        c.args(["-v", "error"]);
        if let Some(t) = self.threads {
            c.args(["-threads", &t.to_string()]);
        }
        if let Some(s) = self.start {
            c.args(["-ss", &format!("{s:.6}")]);
        }
        c.args(&self.input_args);
        c.arg("-i").arg(&self.input);
        if let Some(d) = self.duration {
            c.args(["-t", &format!("{d:.6}")]);
        }
        let map = match self.stream_index {
            Some(i) => format!("0:{i}"),
            None => "0:v:0".to_string(),
        };
        c.args(["-map", &map, "-an", "-sn", "-dn"]);
        // Constant-frame-rate conversion happens first so later filters see a regular cadence.
        let mut vf: Vec<String> = Vec::new();
        if let Some(r) = self.cfr {
            vf.push(format!("fps={}", r.to_ffmpeg()));
        }
        vf.extend(self.filters.iter().cloned());
        vf.push(format!(
            "scale=in_color_matrix={}:in_range={}:flags=accurate_rnd+full_chroma_int",
            self.matrix.ffmpeg_name(),
            if self.full_range { "pc" } else { "tv" }
        ));
        vf.push("format=rgb48le".into());
        c.args(["-vf", &vf.join(",")]);
        c.args(["-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1"]);
        c
    }
}

/// Reads frames from an FFmpeg decoder process.
pub struct FrameReader {
    child: Child,
    stdout: BufReader<ChildStdout>,
    width: usize,
    height: usize,
    log: StderrLog,
    _log_thread: JoinHandle<()>,
    pub frames_read: u64,
}

impl FrameReader {
    pub fn spawn(tools: &FfmpegTools, spec: &DecodeSpec) -> Result<Self, MediaError> {
        let (mut child, log, h) = spawn(spec.command(tools), false, true)?;
        let stdout = BufReader::with_capacity(RawFrame::byte_len(spec.width, spec.height).min(64 << 20), child.stdout.take().unwrap());
        Ok(Self { child, stdout, width: spec.width, height: spec.height, log, _log_thread: h, frames_read: 0 })
    }

    /// Read the next frame; `Ok(None)` at end of stream.
    pub fn read(&mut self) -> Result<Option<RawFrame>, MediaError> {
        let len = RawFrame::byte_len(self.width, self.height);
        let mut data = vec![0u8; len];
        let mut filled = 0;
        while filled < len {
            match self.stdout.read(&mut data[filled..]) {
                Ok(0) => break,
                Ok(n) => filled += n,
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        if filled == 0 {
            let status = self.child.wait()?;
            if !status.success() {
                std::thread::sleep(std::time::Duration::from_millis(30));
                return Err(MediaError::Process(format!("decoder failed: {}", self.log.errors())));
            }
            return Ok(None);
        }
        if filled < len {
            let _ = self.child.wait();
            return Err(MediaError::Process(format!(
                "decoder produced a truncated frame ({filled} of {len} bytes); the filter chain changed the frame size unexpectedly. {}",
                self.log.errors()
            )));
        }
        self.frames_read += 1;
        Ok(Some(RawFrame { width: self.width, height: self.height, data }))
    }

    pub fn kill(&mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for FrameReader {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            self.kill();
        }
    }
}

/// Where an encoder writes.
#[derive(Debug, Clone)]
pub enum EncodeOutput {
    /// A single file (container chosen from extension or `format`).
    File { path: PathBuf, format: Option<String> },
    /// Matroska segments of ~`seconds` each, numbered from `start_number`,
    /// with a CSV list of completed segments. `time_offset` is the timestamp
    /// of the first frame when resuming, keeping keyframes on segment boundaries.
    Segments { dir: PathBuf, seconds: f64, start_number: u32, list_file: PathBuf, time_offset: f64 },
}

/// Parameters for encoding RGB48 frames.
#[derive(Debug, Clone)]
pub struct EncodeSpec {
    pub width: usize,
    pub height: usize,
    pub fps: Rational,
    /// FFmpeg filters applied to the RGB frames before conversion to the output pixel format.
    pub filters: Vec<String>,
    pub video: VideoEncodeSettings,
    pub color: ColorTags,
    pub sar: Option<String>,
    pub output: EncodeOutput,
}

impl EncodeSpec {
    pub fn command(&self, tools: &FfmpegTools) -> Command {
        let mut c = tools.ffmpeg_cmd();
        c.args(["-v", "error", "-y"]);
        c.args(["-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", &format!("{}x{}", self.width, self.height)]);
        c.args(["-framerate", &self.fps.to_ffmpeg(), "-i", "pipe:0"]);
        let mut vf = self.filters.clone();
        vf.push(format!(
            "scale=out_color_matrix={}:out_range={}:flags=accurate_rnd+full_chroma_int",
            self.color.matrix.ffmpeg_name(),
            if self.color.full_range { "pc" } else { "tv" }
        ));
        vf.push(format!("format={}", self.video.pix_fmt));
        if let Some(sar) = &self.sar {
            vf.push(format!("setsar={}", sar.replace(':', "/")));
        }
        c.args(["-vf", &vf.join(",")]);
        c.args(self.video.args(&self.color));
        c.args(self.color.ffmpeg_args());
        match &self.output {
            EncodeOutput::File { path, format } => {
                if let Some(f) = format {
                    c.args(["-f", f]);
                }
                c.arg(path);
            }
            EncodeOutput::Segments { dir, seconds, start_number, list_file, time_offset } => {
                let k = (time_offset / seconds + 1e-6).floor() as u64;
                c.args(["-force_key_frames", &format!("expr:gte(t,(n_forced+{k})*{seconds})")]);
                c.args(["-f", "segment", "-segment_time", &format!("{seconds}"), "-segment_format", "matroska"]);
                c.args(["-reset_timestamps", "1", "-segment_start_number", &start_number.to_string()]);
                c.args(["-segment_list_type", "csv", "-segment_list"]).arg(list_file);
                c.arg(dir.join("seg_%06d.mkv"));
            }
        }
        c
    }
}

/// Writes frames into an FFmpeg encoder process.
pub struct FrameWriter {
    child: Child,
    stdin: Option<ChildStdin>,
    log: StderrLog,
    _log_thread: JoinHandle<()>,
    width: usize,
    height: usize,
    pub frames_written: u64,
}

impl FrameWriter {
    pub fn spawn(tools: &FfmpegTools, spec: &EncodeSpec) -> Result<Self, MediaError> {
        let (mut child, log, h) = spawn(spec.command(tools), true, false)?;
        let stdin = child.stdin.take();
        Ok(Self { child, stdin, log, _log_thread: h, width: spec.width, height: spec.height, frames_written: 0 })
    }

    pub fn write(&mut self, f: &RawFrame) -> Result<(), MediaError> {
        if f.width != self.width || f.height != self.height {
            return Err(MediaError::Invalid(format!(
                "frame size {}x{} does not match encoder size {}x{}",
                f.width, f.height, self.width, self.height
            )));
        }
        let stdin = self.stdin.as_mut().ok_or_else(|| MediaError::Process("encoder already closed".into()))?;
        if let Err(e) = stdin.write_all(&f.data) {
            let _ = self.child.wait();
            std::thread::sleep(std::time::Duration::from_millis(30));
            return Err(MediaError::Process(format!("encoder stopped accepting frames ({e}): {}", self.log.errors())));
        }
        self.frames_written += 1;
        Ok(())
    }

    /// Close input and wait for the encoder to finish.
    pub fn finish(mut self) -> Result<u64, MediaError> {
        drop(self.stdin.take());
        let status = self.child.wait()?;
        std::thread::sleep(std::time::Duration::from_millis(20));
        if !status.success() {
            return Err(MediaError::Process(format!("encoder failed: {}", self.log.errors())));
        }
        Ok(self.frames_written)
    }

    pub fn kill(&mut self) {
        drop(self.stdin.take());
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for FrameWriter {
    fn drop(&mut self) {
        if let Ok(None) = self.child.try_wait() {
            self.kill();
        }
    }
}

/// An FFmpeg filter graph applied to a raw frame stream (RGB48 in, RGB48 out).
pub struct FilterProcess {
    child: Child,
    tx: Option<Sender<RawFrame>>,
    rx: Receiver<Result<RawFrame, MediaError>>,
    writer: Option<JoinHandle<()>>,
    reader: Option<JoinHandle<()>>,
    log: StderrLog,
    _log_thread: JoinHandle<()>,
    pub out_width: usize,
    pub out_height: usize,
}

impl FilterProcess {
    pub fn spawn(
        tools: &FfmpegTools,
        in_w: usize,
        in_h: usize,
        fps: Rational,
        filters: &[String],
        out_w: usize,
        out_h: usize,
    ) -> Result<Self, MediaError> {
        let mut c = tools.ffmpeg_cmd();
        c.args(["-v", "error", "-f", "rawvideo", "-pix_fmt", "rgb48le", "-s", &format!("{in_w}x{in_h}")]);
        c.args(["-framerate", &fps.to_ffmpeg(), "-i", "pipe:0"]);
        let mut vf = filters.to_vec();
        vf.push("format=rgb48le".into());
        c.args(["-vf", &vf.join(","), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb48le", "pipe:1"]);
        let (mut child, log, lh) = spawn(c, true, true)?;
        let mut stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let (tx, in_rx) = bounded::<RawFrame>(2);
        let (out_tx, rx) = unbounded::<Result<RawFrame, MediaError>>();
        let writer = std::thread::spawn(move || {
            for f in in_rx {
                if stdin.write_all(&f.data).is_err() {
                    break;
                }
            }
        });
        let reader = std::thread::spawn(move || {
            let mut r = BufReader::new(stdout);
            let len = RawFrame::byte_len(out_w, out_h);
            loop {
                let mut data = vec![0u8; len];
                let mut filled = 0;
                while filled < len {
                    match r.read(&mut data[filled..]) {
                        Ok(0) => break,
                        Ok(n) => filled += n,
                        Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                        Err(e) => {
                            let _ = out_tx.send(Err(e.into()));
                            return;
                        }
                    }
                }
                if filled == 0 {
                    return;
                }
                if filled < len {
                    let _ = out_tx.send(Err(MediaError::Process("filter produced a truncated frame".into())));
                    return;
                }
                if out_tx.send(Ok(RawFrame { width: out_w, height: out_h, data })).is_err() {
                    return;
                }
            }
        });
        Ok(Self { child, tx: Some(tx), rx, writer: Some(writer), reader: Some(reader), log, _log_thread: lh, out_width: out_w, out_height: out_h })
    }

    /// Push a frame and collect any frames that are ready.
    pub fn push(&mut self, f: RawFrame, out: &mut Vec<RawFrame>) -> Result<(), MediaError> {
        let tx = self.tx.clone().ok_or_else(|| MediaError::Process("filter closed".into()))?;
        // Drain outputs while waiting for room in the input queue.
        let mut pending = Some(f);
        while let Some(frame) = pending.take() {
            match tx.try_send(frame) {
                Ok(()) => {}
                Err(crossbeam_channel::TrySendError::Full(fr)) => {
                    self.drain(out)?;
                    pending = Some(fr);
                    if let Ok(r) = self.rx.recv_timeout(std::time::Duration::from_millis(5)) {
                        out.push(r?);
                    }
                    if let Ok(Some(_)) = self.child.try_wait() {
                        return Err(MediaError::Process(format!("filter process exited: {}", self.log.errors())));
                    }
                }
                Err(crossbeam_channel::TrySendError::Disconnected(_)) => {
                    return Err(MediaError::Process(format!("filter process stopped: {}", self.log.errors())));
                }
            }
        }
        self.drain(out)
    }

    fn drain(&mut self, out: &mut Vec<RawFrame>) -> Result<(), MediaError> {
        while let Ok(r) = self.rx.try_recv() {
            out.push(r?);
        }
        Ok(())
    }

    /// Close input and collect all remaining frames.
    pub fn finish(&mut self, out: &mut Vec<RawFrame>) -> Result<(), MediaError> {
        drop(self.tx.take());
        if let Some(w) = self.writer.take() {
            let _ = w.join();
        }
        for r in self.rx.iter() {
            out.push(r?);
        }
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
        let status = self.child.wait()?;
        if !status.success() {
            return Err(MediaError::Process(format!("filter failed: {}", self.log.errors())));
        }
        Ok(())
    }
}

impl Drop for FilterProcess {
    fn drop(&mut self) {
        drop(self.tx.take());
        if let Ok(None) = self.child.try_wait() {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}
