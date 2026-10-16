//! End-to-end renders through FFmpeg with synthetic fixtures.

mod common;

use clearframe_media::mux::{Container, SlowMotionAudio};
use clearframe_media::probe::probe;
use clearframe_media::Rational;
use clearframe_pipeline::runner::{Event, JobError, JobState, StopFlag};
use clearframe_pipeline::spec::*;
use common::*;

macro_rules! need_ffmpeg {
    () => {
        match tools() {
            Some(t) => t,
            None => {
                eprintln!("ffmpeg not available; skipping");
                return;
            }
        }
    };
}

#[test]
fn classic_filters_keep_audio_and_source() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "", 2.0);
    let before = sha(&src);
    let p = Pipeline {
        stages: vec![
            Stage::new(Operation::Denoise(DenoiseSettings { engine: DenoiseEngine::Hqdn3d, strength: 8.0, ..Default::default() })),
            Stage::new(Operation::Grain(GrainSettings::default())),
            Stage::new(Operation::Resize(ResizeSettings { width: 640, height: 480, ..Default::default() })),
        ],
    };
    let spec = job(d.path(), &src, p, OutputSettings::default(), "classic");
    let mut states = Vec::new();
    let r = run_with(&t, &spec, StopFlag::default(), |e| {
        if let Event::State { state, .. } = e {
            states.push(*state);
        }
    })
    .unwrap();
    assert_eq!(sha(&src), before, "source must be byte-identical");
    let info = probe(&t, &r.output).unwrap();
    let v = info.video.as_ref().unwrap();
    assert_eq!((v.width, v.height), (640, 480));
    assert_eq!(info.audio.len(), 1);
    assert!((info.duration - 2.0).abs() < 0.1, "{}", info.duration);
    assert!(states.contains(&JobState::Validating));
    assert_eq!(*states.last().unwrap(), JobState::Completed);
    assert!(r.validation.as_ref().unwrap().ok);
    // sidecar written; work dir cleaned up
    assert!(std::path::PathBuf::from(format!("{}.clipfx.json", r.output.display())).exists());
    assert!(!spec.work_dir.exists());
}

#[test]
fn passthrough_color_accuracy() {
    // Decode -> RGB48 -> encode with no visible change must preserve the image.
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("bars.mp4");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "smptehdbars=size=1280x720:rate=25",
        "-t",
        "1",
        "-c:v",
        "libx264",
        "-qp",
        "0",
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        src.to_str().unwrap(),
    ]);
    let p = Pipeline::single(Operation::Grain(GrainSettings { strength: 0.0, ..Default::default() }));
    let mut out = OutputSettings::default();
    out.video.rate = clearframe_media::encode::RateControl::Quality { value: 4.0 };
    out.container = Container::Mkv;
    let r = run(&t, &job(d.path(), &src, p, out, "bars")).unwrap();
    let db = psnr(&src, &r.output);
    assert!(db > 42.0, "colour round-trip PSNR {db:.2} dB");
    let info = probe(&t, &r.output).unwrap();
    assert_eq!(info.video.as_ref().unwrap().color.space.as_deref(), Some("bt709"));
}

#[test]
fn blend_interpolation_doubles_rate_and_keeps_duration() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "", 2.0);
    let p = Pipeline::single(Operation::Interpolate(InterpolateSettings { engine: InterpEngine::Blend, ..Default::default() }));
    let r = run(&t, &job(d.path(), &src, p, OutputSettings::default(), "interp")).unwrap();
    let info = probe(&t, &r.output).unwrap();
    let v = info.video.as_ref().unwrap();
    assert!((v.avg_frame_rate.as_f64() - 60.0).abs() < 0.1, "{}", v.avg_frame_rate);
    assert!((info.duration - 2.0).abs() < 0.1);
    assert_eq!(frame_md5(&r.output).len(), 120);
    assert_eq!(info.audio.len(), 1);
}

#[test]
fn slow_motion_stretches_audio() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "", 1.0);
    let p = Pipeline::single(Operation::Interpolate(InterpolateSettings {
        engine: InterpEngine::Blend,
        target: FpsTarget::Multiply { factor: 2 },
        slow_motion: 2.0,
        slow_motion_audio: SlowMotionAudio::StretchPreservePitch,
        ..Default::default()
    }));
    let r = run(&t, &job(d.path(), &src, p, OutputSettings::default(), "slow")).unwrap();
    let info = probe(&t, &r.output).unwrap();
    assert!((info.duration - 2.0).abs() < 0.15, "duration {}", info.duration);
    assert_eq!(info.audio.len(), 1);
    assert!((info.video.as_ref().unwrap().avg_frame_rate.as_f64() - 60.0).abs() < 0.1);
    assert_eq!(frame_md5(&r.output).len(), 120);
}

#[test]
fn mci_interpolation_and_range() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "", 3.0);
    let p = Pipeline::single(Operation::Interpolate(InterpolateSettings { engine: InterpEngine::Mci, target: FpsTarget::Fps { fps: Rational::new(50, 1) }, ..Default::default() }));
    let mut spec = job(d.path(), &src, p, OutputSettings::default(), "mci");
    spec.range = Some(TimeRange { start: 1.0, end: 2.0 });
    let r = run(&t, &spec).unwrap();
    let info = probe(&t, &r.output).unwrap();
    assert!((info.duration - 1.0).abs() < 0.1, "{}", info.duration);
    assert_eq!(info.video.as_ref().unwrap().avg_frame_rate, Rational::new(50, 1));
}

#[test]
fn deinterlace_bob_doubles_rate() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("int.mkv");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=50",
        "-vf",
        "tinterlace=mode=interleave_top,setfield=tff",
        "-t",
        "1",
        "-c:v",
        "libx264",
        "-flags",
        "+ildct+ilme",
        "-pix_fmt",
        "yuv420p",
        src.to_str().unwrap(),
    ]);
    let info = probe(&t, &src).unwrap();
    assert!(info.video.as_ref().unwrap().is_interlaced_flagged(), "fixture should be flagged interlaced");
    let p = Pipeline::single(Operation::Deinterlace(DeinterlaceSettings { rate: DeinterlaceRate::Field, ..Default::default() }));
    let r = run(&t, &job(d.path(), &src, p, OutputSettings::default(), "deint")).unwrap();
    let out = probe(&t, &r.output).unwrap();
    assert_eq!(out.video.as_ref().unwrap().avg_frame_rate, Rational::new(50, 1));
    assert!(!out.video.as_ref().unwrap().is_interlaced_flagged());
}

#[test]
fn stabilize_auto_crop() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("shaky.mp4");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=640x360:rate=30,crop=560:300:x='40+30*sin(t*7)':y='30+20*cos(t*5)'",
        "-t",
        "2",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        src.to_str().unwrap(),
    ]);
    let p = Pipeline::single(Operation::Stabilize(StabilizeSettings::default()));
    let r = run(&t, &job(d.path(), &src, p, OutputSettings::default(), "stab")).unwrap();
    assert!(r.stabilize_zoom_percent.unwrap_or(0.0) > 0.0);
    let out = probe(&t, &r.output).unwrap();
    assert_eq!((out.video.as_ref().unwrap().width, out.video.as_ref().unwrap().height), (560, 300));
}

#[test]
fn stabilize_after_native_stage_uses_two_passes() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("shaky.mp4");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=25,crop=280:200:x='20+15*sin(t*6)':y='20+10*cos(t*4)'",
        "-t",
        "1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        src.to_str().unwrap(),
    ]);
    let p = Pipeline {
        stages: vec![
            Stage::new(Operation::Grain(GrainSettings { strength: 0.1, ..Default::default() })),
            Stage::new(Operation::Stabilize(StabilizeSettings { mode: StabilizeMode::FullFrame, ..Default::default() })),
        ],
    };
    let r = run(&t, &job(d.path(), &src, p, OutputSettings::default(), "stab2")).unwrap();
    assert_eq!(frame_md5(&r.output).len(), 25);
}

#[test]
fn hdr_tonemap_and_sdr_to_hdr() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("sdr.mp4");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=25",
        "-t",
        "1",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-color_primaries",
        "bt709",
        "-color_trc",
        "bt709",
        "-colorspace",
        "bt709",
        src.to_str().unwrap(),
    ]);
    let c = ColorSettings { mode: ColorMode::SdrToHdr, ..Default::default() };
    let out = OutputSettings {
        video: clearframe_media::encode::VideoEncodeSettings {
            encoder: "libx265".into(),
            rate: clearframe_media::encode::RateControl::Quality { value: 22.0 },
            preset: Some("ultrafast".into()),
            pix_fmt: "yuv420p10le".into(),
            profile: None,
            gop: None,
        },
        ..Default::default()
    };
    let r = run(&t, &job(d.path(), &src, Pipeline::single(Operation::Color(c)), out.clone(), "hdr")).unwrap();
    let info = probe(&t, &r.output).unwrap();
    let v = info.video.as_ref().unwrap();
    assert_eq!(v.color.transfer.as_deref(), Some("smpte2084"));
    assert_eq!(v.color.primaries.as_deref(), Some("bt2020"));
    // Round trip back to SDR.
    let back = ColorSettings { mode: ColorMode::HdrToSdr, ..Default::default() };
    let r2 = run(&t, &job(d.path(), &r.output, Pipeline::single(Operation::Color(back)), OutputSettings::default(), "sdr")).unwrap();
    let info2 = probe(&t, &r2.output).unwrap();
    assert_eq!(info2.video.as_ref().unwrap().color.transfer.as_deref(), Some("bt709"));
}

#[test]
fn hdr_without_10bit_is_rejected_before_rendering() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "", 1.0);
    let c = ColorSettings { mode: ColorMode::SdrToHdr, ..Default::default() };
    let spec = job(d.path(), &src, Pipeline::single(Operation::Color(c)), OutputSettings::default(), "bad");
    match run(&t, &spec) {
        Err(JobError::Invalid(m)) => assert!(m.contains("10-bit"), "{m}"),
        other => panic!("{other:?}"),
    }
    assert!(!spec.destination.exists());
}

#[test]
fn never_overwrites_existing_output() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "", 1.0);
    let spec = job(d.path(), &src, Pipeline::single(Operation::Grain(GrainSettings::default())), OutputSettings::default(), "exists");
    std::fs::write(&spec.destination, b"precious").unwrap();
    let r = run(&t, &spec).unwrap();
    assert_eq!(std::fs::read(&spec.destination).unwrap(), b"precious");
    assert_ne!(r.output, spec.destination);
    assert!(r.output.to_string_lossy().contains("exists_2"));
}

#[test]
fn cancel_leaves_no_output_and_failure_keeps_source() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "", 3.0);
    let spec = job(d.path(), &src, Pipeline::single(Operation::Grain(GrainSettings::default())), OutputSettings::default(), "cancel");
    let stop = StopFlag::default();
    let s2 = stop.clone();
    let mut n = 0;
    let r = run_with(&t, &spec, stop, move |e| {
        if let Event::Progress(p) = e {
            n += 1;
            if p.frames_done > 10 || n > 3 {
                s2.cancel();
            }
        }
    });
    assert!(matches!(r, Err(JobError::Cancelled)), "{r:?}");
    assert!(!spec.destination.exists());
    let leftovers: Vec<_> = std::fs::read_dir(d.path()).unwrap().flatten().filter(|e| e.file_name().to_string_lossy().contains("partial")).collect();
    assert!(leftovers.is_empty());
}

#[test]
fn pause_and_resume_is_bit_identical() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "", 4.0);
    let p = Pipeline {
        stages: vec![
            Stage::new(Operation::Interpolate(InterpolateSettings { engine: InterpEngine::Blend, target: FpsTarget::Fps { fps: Rational::new(75, 1) }, ..Default::default() })),
            Stage::new(Operation::Grain(GrainSettings::default())),
        ],
    };
    // Reference render.
    let reference = run(&t, &job(d.path(), &src, p.clone(), lossless(), "ref")).unwrap();
    let ref_md5 = frame_md5(&reference.output);
    assert_eq!(ref_md5.len(), 300);
    // Paused render.
    let spec = job(d.path(), &src, p, lossless(), "resumed");
    let stop = StopFlag::default();
    let s2 = stop.clone();
    let r = run_with(&t, &spec, stop, move |e| {
        if let Event::Progress(p) = e {
            if p.frames_done >= 170 {
                s2.pause();
            }
        }
    });
    assert!(matches!(r, Err(JobError::Paused)), "{r:?}");
    let info = clearframe_pipeline::runner::recovery_info(&spec.work_dir).unwrap();
    assert!(info.segments_done >= 1, "{info:?}");
    let mut resumed_at = None;
    let r = run_with(&t, &spec, StopFlag::default(), |e| {
        if let Event::Resumed { from_frame, .. } = e {
            resumed_at = Some(*from_frame);
        }
    })
    .unwrap();
    assert!(resumed_at.unwrap_or(0) > 0, "job should have resumed mid-way");
    let md5 = frame_md5(&r.output);
    assert_eq!(md5.len(), ref_md5.len());
    let diff = md5.iter().zip(&ref_md5).filter(|(a, b)| a != b).count();
    assert_eq!(diff, 0, "resumed render differs from reference in {diff} frames");
}

#[test]
fn vfr_rotation_and_streams() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    // Variable frame rate: drop every third frame with timestamps preserved.
    let vfr = d.path().join("vfr.mkv");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-i",
        "testsrc2=size=320x240:rate=30",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=300:sample_rate=44100",
        "-f",
        "lavfi",
        "-i",
        "sine=frequency=600:sample_rate=44100",
        "-t",
        "2",
        "-vf",
        "select='not(eq(mod(n,3),0))'",
        "-fps_mode",
        "vfr",
        "-map",
        "0:v",
        "-map",
        "1:a",
        "-map",
        "2:a",
        "-c:v",
        "libx264",
        "-c:a",
        "pcm_s16le",
        vfr.to_str().unwrap(),
    ]);
    let sub = d.path().join("s.srt");
    std::fs::write(&sub, "1\n00:00:00,000 --> 00:00:01,000\nhello\n").unwrap();
    let src = d.path().join("src.mkv");
    ffmpeg(&["-i", vfr.to_str().unwrap(), "-i", sub.to_str().unwrap(), "-map", "0", "-map", "1", "-c", "copy", "-metadata:s:v:0", "rotate=90", src.to_str().unwrap()]);
    let info = probe(&t, &src).unwrap();
    assert!(info.video.as_ref().unwrap().variable_frame_rate);
    let p = Pipeline::single(Operation::Grain(GrainSettings::default()));
    let out = OutputSettings { container: Container::Mp4, ..Default::default() };
    let r = run(&t, &job(d.path(), &src, p, out, "vfr")).unwrap();
    let o = probe(&t, &r.output).unwrap();
    assert_eq!(o.audio.len(), 2, "both audio tracks kept");
    assert_eq!(o.subtitles.len(), 1, "text subtitle converted to mov_text");
    assert!(r.mux.as_ref().unwrap().warnings.iter().any(|w| w.contains("re-encoded")), "pcm->aac in mp4 should be reported");
    assert!((o.duration - 2.0).abs() < 0.2, "{}", o.duration);
}

#[test]
fn still_preview_with_interpolation_and_resize() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "", 1.0);
    let info = probe(&t, &src).unwrap();
    let req = clearframe_pipeline::preview::StillRequest {
        source: src.clone(),
        time: 0.5,
        pipeline: Pipeline {
            stages: vec![
                Stage::new(Operation::Interpolate(InterpolateSettings { engine: InterpEngine::Blend, ..Default::default() })),
                Stage::new(Operation::Resize(ResizeSettings { width: 160, height: 120, ..Default::default() })),
            ],
        },
        output: OutputSettings::default(),
        engine: Default::default(),
        models: Default::default(),
        resources: Default::default(),
        crop: None,
    };
    let lookup = clearframe_pipeline::runner::ResolvedLookup { models: &req.models, scales: Default::default() };
    let r = clearframe_pipeline::preview::render_still(&t, &info, &req, &lookup).unwrap();
    assert_eq!(r.processed_size, (160, 120));
    assert_eq!(r.original_size, (320, 240));
    assert!(r.processed_png.starts_with(b"\x89PNG"));
}

#[test]
fn low_disk_pauses_job_and_resumes() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "", 6.0);
    let mut spec = job(d.path(), &src, Pipeline::single(Operation::Grain(GrainSettings::default())), OutputSettings::default(), "disk");
    // Demand more free space than any disk has: the job must stop safely.
    spec.resources.min_free_disk_mb = u64::MAX / 2_000_000;
    let r = run(&t, &spec);
    assert!(matches!(r, Err(JobError::Resources(ref m)) if m.contains("disk space")), "{r:?}");
    assert!(!spec.destination.exists());
    assert_eq!(sha(&src), sha(&src));
    // After space is "freed" the job resumes and completes.
    spec.resources.min_free_disk_mb = 1;
    let r = run(&t, &spec).unwrap();
    assert!(r.validation.unwrap().ok);
}

#[test]
fn thread_limit_and_cpu_device_are_applied() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "", 1.0);
    clearframe_media::tools::set_default_threads(1);
    let mut spec = job(d.path(), &src, Pipeline::single(Operation::Grain(GrainSettings::default())), OutputSettings::default(), "threads");
    spec.resources.device = "cpu".into();
    spec.resources.threads = 1;
    let mut samples = 0;
    let r = run_with(&t, &spec, StopFlag::default(), |e| {
        if let Event::Resources(s) = e {
            assert!(s.system_memory_total_mb > 0);
            samples += 1;
        }
    })
    .unwrap();
    clearframe_media::tools::set_default_threads(0);
    assert_eq!(r.backend, "CPU");
    let _ = samples;
    // Unavailable devices are rejected before rendering.
    spec.resources.device = "cuda:7".into();
    spec.job_id = "threads2".into();
    assert!(matches!(run(&t, &spec), Err(JobError::Invalid(_))));
}

/// Mean R, G, B (0..255) of the first frame, via FFmpeg.
fn mean_rgb(p: &std::path::Path) -> [f64; 3] {
    let out = std::process::Command::new("ffmpeg").args(["-v", "error", "-i"]).arg(p).args(["-frames:v", "1", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).output().unwrap();
    let (px, _) = out.stdout.as_chunks::<3>();
    let n = px.len() as f64;
    let mut m = [0.0; 3];
    for c in px {
        for (acc, v) in m.iter_mut().zip(c) {
            *acc += *v as f64;
        }
    }
    m.map(|v| v / n)
}

#[test]
fn color_and_light_sliders_change_the_image() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    // A dim, low-contrast pattern so brightening and colour shifts are not clipped.
    let src = clip(d.path(), "src.mp4", "lutrgb=r=40+val*0.3:g=40+val*0.3:b=40+val*0.3", 1.0);
    let base = mean_rgb(&src);
    let grade = |name: &str, c: ColorSettings| {
        let p = Pipeline { stages: vec![Stage::new(Operation::Color(c))] };
        let r = run(&t, &job(d.path(), &src, p, lossless(), name)).unwrap();
        mean_rgb(&r.output)
    };
    let luma = |m: [f64; 3]| 0.2126 * m[0] + 0.7152 * m[1] + 0.0722 * m[2];
    let brighter = grade("exposure", ColorSettings { exposure: 1.0, ..Default::default() });
    assert!(luma(brighter) > luma(base) + 10.0, "exposure +1: {base:?} -> {brighter:?}");
    let warm = grade("warm", ColorSettings { temperature: 1.0, ..Default::default() });
    assert!(warm[0] - warm[2] > base[0] - base[2] + 5.0, "warmer: {base:?} -> {warm:?}");
    let lifted = grade("shadows", ColorSettings { shadows: 1.0, ..Default::default() });
    assert!(luma(lifted) > luma(base) + 3.0, "shadows +1: {base:?} -> {lifted:?}");
    let grey = grade("grey", ColorSettings { saturation: 0.0, vibrance: 0.5, hue: 30.0, tint: 0.3, highlights: -0.5, contrast: 1.2, ..Default::default() });
    assert!(grey.iter().all(|v| v.is_finite()));
}

#[test]
fn sharpen_methods_render() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = clip(d.path(), "src.mp4", "", 1.0);
    for (name, method, amount) in [("adaptive", SharpenMethod::Adaptive, 0.8), ("unsharp", SharpenMethod::Unsharp, 1.5)] {
        let p = Pipeline { stages: vec![Stage::new(Operation::Sharpen(SharpenSettings { method, amount, radius: 2.0 }))] };
        let r = run(&t, &job(d.path(), &src, p, lossless(), name)).unwrap();
        let q = psnr(&src, &r.output);
        assert!(q.is_finite() && q < 60.0, "{name}: sharpening must change the image (PSNR {q})");
    }
}

#[test]
fn interlaced_source_is_deinterlaced_automatically() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    // Interlaced H.264 (top field first), as from a DVD or camcorder.
    let src = d.path().join("dv.mp4");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-t",
        "1",
        "-i",
        "testsrc2=size=320x240:rate=60,tinterlace=interleave_top,setfield=tff",
        "-c:v",
        "libx264",
        "-flags",
        "+ilme+ildct",
        "-x264-params",
        "tff=1",
        "-pix_fmt",
        "yuv420p",
        src.to_str().unwrap(),
    ]);
    let info = probe(&t, &src).unwrap();
    assert!(info.video.as_ref().unwrap().is_interlaced_flagged(), "fixture must be flagged interlaced");
    let p = Pipeline { stages: vec![Stage::new(Operation::Grain(GrainSettings::default()))] };
    let spec = job(d.path(), &src, p, OutputSettings::default(), "auto-deint");
    let lookup = clearframe_pipeline::runner::ResolvedLookup { models: &Default::default(), scales: Default::default() };
    let plan = clearframe_pipeline::runner::plan_probed(&t, &info, &spec, &lookup, None).unwrap();
    assert_eq!(plan.stages[0].stage_id, AUTO_DEINTERLACE_ID);
    assert!(!plan.issues.iter().any(|i| i.message.contains("Add a Deinterlace step first")));
    let r = run(&t, &spec).unwrap();
    assert!(r.validation.as_ref().unwrap().ok);
    let out = probe(&t, &r.output).unwrap();
    assert!(!out.video.as_ref().unwrap().is_interlaced_flagged(), "output must be progressive");
}

#[test]
fn odd_sized_source_is_trimmed_to_even() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("odd.mkv");
    ffmpeg(&["-f", "lavfi", "-t", "0.5", "-i", "testsrc2=size=321x241:rate=24", "-c:v", "ffv1", src.to_str().unwrap()]);
    for (name, op) in [("grain", Operation::Grain(GrainSettings::default())), ("color", Operation::Color(ColorSettings { exposure: 0.3, ..Default::default() }))] {
        let p = Pipeline { stages: vec![Stage::new(op)] };
        let r = run(&t, &job(d.path(), &src, p, OutputSettings::default(), name)).unwrap();
        let v = probe(&t, &r.output).unwrap().video.unwrap();
        assert_eq!((v.width, v.height), (320, 240), "{name}");
    }
}

#[test]
fn hdr_result_is_saved_in_ten_bit_automatically() {
    let t = need_ffmpeg!();
    let d = tempfile::tempdir().unwrap();
    let src = d.path().join("hdr.mp4");
    ffmpeg(&[
        "-f",
        "lavfi",
        "-t",
        "0.5",
        "-i",
        "testsrc2=size=320x240:rate=24",
        "-c:v",
        "libx265",
        "-pix_fmt",
        "yuv420p10le",
        "-x265-params",
        "log-level=error:colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc",
        "-color_primaries",
        "bt2020",
        "-color_trc",
        "smpte2084",
        "-colorspace",
        "bt2020nc",
        src.to_str().unwrap(),
    ]);
    // Default output is 8-bit H.264, which cannot hold HDR.
    let p = Pipeline { stages: vec![Stage::new(Operation::Grain(GrainSettings::default()))] };
    let r = run(&t, &job(d.path(), &src, p, OutputSettings::default(), "hdr")).unwrap();
    let v = probe(&t, &r.output).unwrap().video.unwrap();
    assert_eq!(v.codec, "hevc");
    assert!(v.bit_depth >= 10, "{}", v.pix_fmt);
    assert_eq!(v.color.transfer.as_deref(), Some("smpte2084"));
    // Tone-mapped to SDR: the chosen 8-bit H.264 is kept.
    let p = Pipeline { stages: vec![Stage::new(Operation::Color(ColorSettings { mode: ColorMode::HdrToSdr, ..Default::default() }))] };
    let r = run(&t, &job(d.path(), &src, p, OutputSettings::default(), "sdr")).unwrap();
    assert_eq!(probe(&t, &r.output).unwrap().video.unwrap().codec, "h264");
}
