//! Shared helpers for integration tests: synthetic, license-free fixtures generated with FFmpeg.
#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use clearframe_media::FfmpegTools;
use clearframe_pipeline::runner::{run_job, Event, JobError, JobReport, ResolvedLookup, StopFlag};
use clearframe_pipeline::spec::*;

pub fn tools() -> Option<FfmpegTools> {
    FfmpegTools::discover(None, &[]).ok()
}

pub fn ffmpeg(args: &[&str]) {
    let st = Command::new("ffmpeg").args(["-v", "error", "-y"]).args(args).status().expect("ffmpeg");
    assert!(st.success(), "ffmpeg {:?}", args);
}

/// 2 s 320x240 30 fps test pattern with a 440 Hz tone.
pub fn clip(dir: &Path, name: &str, extra_video: &str, seconds: f64) -> PathBuf {
    let p = dir.join(name);
    let vf = if extra_video.is_empty() { "testsrc2=size=320x240:rate=30".to_string() } else { format!("testsrc2=size=320x240:rate=30,{extra_video}") };
    let secs = seconds.to_string();
    ffmpeg(&[
        "-f", "lavfi", "-t", &secs, "-i", &vf, "-f", "lavfi", "-t", &secs, "-i", "sine=frequency=440:sample_rate=48000",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", p.to_str().unwrap(),
    ]);
    p
}

pub fn job(dir: &Path, source: &Path, pipeline: Pipeline, output: OutputSettings, name: &str) -> JobSpec {
    JobSpec {
        schema_version: SCHEMA_VERSION,
        job_id: format!("test-{name}"),
        source: source.to_path_buf(),
        range: None,
        pipeline,
        destination: dir.join(format!("{name}.{}", output.container.extension())),
        output,
        work_dir: dir.join(format!("work-{name}")),
        engine: EngineSettings::default(),
        models: BTreeMap::new(),
        segment_seconds: 1.0,
        write_sidecar: true,
        preview: false,
        resources: Default::default(),
        estimated_memory_mb: 0.0,
    }
}

pub fn run(t: &FfmpegTools, spec: &JobSpec) -> Result<JobReport, JobError> {
    run_with(t, spec, StopFlag::default(), |_| {})
}

pub fn run_with(t: &FfmpegTools, spec: &JobSpec, stop: StopFlag, mut f: impl FnMut(&Event)) -> Result<JobReport, JobError> {
    let scales = spec.models.iter().map(|(k, m)| (k.clone(), clearframe_pipeline::runner::model_scale(m))).collect();
    let lookup = ResolvedLookup { models: &spec.models, scales };
    run_job(t, spec, &lookup, stop, &mut |e| f(&e))
}

pub fn sha(p: &Path) -> String {
    clearframe_core::fsutil::sha256_file(p).unwrap()
}

/// Per-frame MD5s of the decoded video (framemd5 muxer).
pub fn frame_md5(p: &Path) -> Vec<String> {
    let out = Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(p).args(["-map", "0:v:0", "-f", "framemd5", "-"]).output().unwrap();
    String::from_utf8_lossy(&out.stdout).lines().filter(|l| !l.starts_with('#')).map(|l| l.rsplit(',').next().unwrap().trim().to_string()).collect()
}

/// PSNR (dB) between two videos' first video streams.
pub fn psnr(a: &Path, b: &Path) -> f64 {
    let out = Command::new("ffmpeg").args(["-v", "info", "-i"]).arg(a).arg("-i").arg(b).args(["-lavfi", "[0:v][1:v]psnr", "-f", "null", "-"]).output().unwrap();
    let s = String::from_utf8_lossy(&out.stderr);
    let line = s.lines().find(|l| l.contains("PSNR") && l.contains("average:")).unwrap_or_else(|| panic!("no psnr: {s}"));
    let v = line.split("average:").nth(1).unwrap().split_whitespace().next().unwrap();
    if v == "inf" { 999.0 } else { v.parse().unwrap() }
}

pub fn lossless() -> OutputSettings {
    OutputSettings {
        container: clearframe_media::mux::Container::Mkv,
        video: clearframe_media::encode::VideoEncodeSettings {
            encoder: "ffv1".into(),
            rate: clearframe_media::encode::RateControl::Lossless,
            preset: None,
            pix_fmt: "yuv444p10le".into(),
            profile: None,
            gop: Some(1),
        },
        ..Default::default()
    }
}
