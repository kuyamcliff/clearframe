//! `clipfx` command-line interface. Also the isolated processing worker
//! (`--worker <job.json>`, `--still <request.json> <out-dir>`).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};
use clap::{Parser, Subcommand};

use clearframe_app::export::NewJob;
use clearframe_app::storage::{RelocationMode, StorageKind};
use clearframe_app::{App, AppEvent, WorkerCommand};
use clearframe_core::jobs::JobState;
use clearframe_core::paths::CacheArea;
use clearframe_pipeline::spec::{OutputSettings, Pipeline, TimeRange};

#[derive(Parser)]
#[command(name = "clipfx", version, about = "ClipFX Enhance — local, open-source video enhancement")]
struct Cli {
    /// Keep all data under this folder (portable mode / testing).
    #[arg(long, global = true)]
    data_root: Option<PathBuf>,
    /// Print machine-readable JSON.
    #[arg(long, global = true)]
    json: bool,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Show media information for a video.
    Info { file: PathBuf },
    /// Hardware, compute backends and the effective resource plan.
    Hardware,
    /// Model library.
    Models {
        #[command(subcommand)]
        cmd: ModelsCmd,
    },
    /// Process a video (or several) with a preset or pipeline file.
    Run {
        /// Input files or folders.
        #[arg(required = true)]
        inputs: Vec<PathBuf>,
        #[arg(long)]
        preset: Option<String>,
        /// Pipeline JSON ({"stages":[...]}).
        #[arg(long)]
        pipeline: Option<PathBuf>,
        /// Output folder (default: the configured export folder).
        #[arg(long, short)]
        output_dir: Option<PathBuf>,
        /// Output file name without extension (single input only).
        #[arg(long)]
        name: Option<String>,
        #[arg(long)]
        start: Option<f64>,
        #[arg(long)]
        end: Option<f64>,
        /// Container: mp4, mov, mkv, webm, m4v.
        #[arg(long)]
        container: Option<String>,
        /// Recurse into folders.
        #[arg(long)]
        recursive: bool,
    },
    /// Job history and queue.
    Jobs {
        #[command(subcommand)]
        cmd: JobsCmd,
    },
    /// Presets.
    Presets {
        #[command(subcommand)]
        cmd: PresetsCmd,
    },
    /// Storage locations and cache.
    Storage {
        #[command(subcommand)]
        cmd: StorageCmd,
    },
    /// Local diagnostics report (never uploaded).
    Diagnostics {
        #[arg(long)]
        redact: bool,
        #[arg(long)]
        out: Option<PathBuf>,
    },
    /// Measure model throughput on synthetic frames.
    Benchmark {
        model: String,
        #[arg(long, default_value = "640x360")]
        size: String,
        #[arg(long, default_value_t = 5)]
        frames: usize,
    },
    /// Catalog signing tools for maintainers.
    Catalog {
        #[command(subcommand)]
        cmd: CatalogCmd,
    },
    /// Check for a newer release (online, user-initiated).
    CheckUpdate,
}

#[derive(Subcommand)]
enum ModelsCmd {
    List,
    /// Download, verify and install a model.
    Install {
        id: String,
        /// Use an already-downloaded artifact: role=path (offline install).
        #[arg(long = "file")]
        files: Vec<String>,
    },
    Remove {
        id: String,
    },
    Verify {
        id: String,
    },
    /// Re-scan the models folder (detects files removed outside ClipFX Enhance).
    Refresh,
    /// Import a custom ESRGAN-family checkpoint.
    Import {
        file: PathBuf,
        #[arg(long)]
        name: String,
        #[arg(long, default_value = "unknown")]
        license: String,
    },
}

#[derive(Subcommand)]
enum JobsCmd {
    List,
    Resume {
        id: String,
    },
    Cancel {
        id: String,
    },
    Remove {
        id: String,
    },
    /// Run the queue until it is empty.
    Process,
    ClearFinished,
}

#[derive(Subcommand)]
enum PresetsCmd {
    List,
    Export { id: String, file: PathBuf },
    Import { file: PathBuf },
}

#[derive(Subcommand)]
enum StorageCmd {
    Show,
    /// Move the models folder (copy, verify, switch, delete old).
    MoveModels {
        dest: PathBuf,
        #[arg(long)]
        switch_only: bool,
    },
    /// Move the cache folder.
    MoveCache {
        dest: PathBuf,
        /// Delete regenerable cache instead of moving it.
        #[arg(long)]
        clear: bool,
    },
    /// Clear regenerable cache (resumable job data is kept unless --all).
    ClearCache {
        #[arg(long)]
        all: bool,
    },
}

#[derive(Subcommand)]
enum CatalogCmd {
    Keygen,
    Sign {
        file: PathBuf,
        #[arg(long)]
        key_file: PathBuf,
    },
    Verify {
        file: PathBuf,
        signature: PathBuf,
    },
    Validate {
        file: PathBuf,
    },
}

fn gb(b: u64) -> String {
    format!("{:.2} GB", b as f64 / 1e9)
}

fn open_app(cli: &Cli, quiet: bool) -> Result<Arc<App>> {
    let json = cli.json;
    let sink: clearframe_app::EventSink = Arc::new(move |e: AppEvent| {
        if quiet {
            return;
        }
        if json {
            if let Ok(s) = serde_json::to_string(&e) {
                println!("{s}");
            }
            return;
        }
        match e {
            AppEvent::JobProgress { progress, .. } => {
                let eta = progress.eta.map(|(a, b)| format!(" ETA {:.0}-{:.0}s", a, b)).unwrap_or_default();
                eprint!("\r{:>5.1}%  {} {}/{}  {:.2} fps{}      ", progress.fraction * 100.0, progress.phase, progress.frames_done, progress.frames_total, progress.fps, eta);
            }
            AppEvent::JobLog { level, message, .. } => eprintln!("\n[{level}] {message}"),
            AppEvent::ModelProgress { progress } => {
                eprint!("\r{:?} {} {}/{} MB      ", progress.state, progress.message, progress.bytes_done / 1_000_000, progress.bytes_total / 1_000_000)
            }
            AppEvent::StorageProgress { bytes_done, bytes_total, .. } => eprint!("\rmoving {}/{} MB   ", bytes_done / 1_000_000, bytes_total / 1_000_000),
            AppEvent::Notice { message, .. } => eprintln!("\n{message}"),
            _ => {}
        }
    });
    let mut bundled = vec![];
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            bundled.push(d.to_path_buf());
        }
    }
    let app = App::open(cli.data_root.as_deref(), WorkerCommand::current_exe(), bundled, sink)?;
    clearframe_app::logging::init(&app.paths().logs_dir, "clipfx");
    Ok(app)
}

fn wait_for_jobs(app: &Arc<App>, ids: &[String]) -> Result<bool> {
    app.start_scheduler();
    let mut all_ok = true;
    loop {
        let jobs: Vec<_> = ids.iter().filter_map(|id| app.db.job(id).ok()).collect();
        if jobs.iter().all(|j| j.state.is_terminal() || j.state == JobState::Paused || j.state == JobState::Recoverable) {
            eprintln!();
            for j in jobs {
                match j.state {
                    JobState::Completed => println!("done: {}", j.destination),
                    s => {
                        all_ok = false;
                        println!("{}: {} {}", s.as_str(), j.name, j.error.unwrap_or_default());
                    }
                }
            }
            return Ok(all_ok);
        }
        std::thread::sleep(Duration::from_millis(250));
    }
}

fn main() {
    // Worker modes are dispatched before argument parsing (internal protocol).
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("--worker") {
        let code = clearframe_app::worker::run_worker(std::path::Path::new(args.get(2).map(String::as_str).unwrap_or("")));
        std::process::exit(code);
    }
    if args.get(1).map(String::as_str) == Some("--still") {
        let code = clearframe_app::worker::run_still(
            std::path::Path::new(args.get(2).map(String::as_str).unwrap_or("")),
            std::path::Path::new(args.get(3).map(String::as_str).unwrap_or("")),
        );
        std::process::exit(code);
    }
    let cli = Cli::parse();
    match run(cli) {
        Ok(true) => {}
        Ok(false) => std::process::exit(1),
        Err(e) => {
            eprintln!("error: {e:#}");
            std::process::exit(1);
        }
    }
}

fn run(cli: Cli) -> Result<bool> {
    match &cli.cmd {
        Cmd::Catalog { cmd } => return catalog_cmd(cmd),
        Cmd::Info { file } => {
            let app = open_app(&cli, true)?;
            let i = app.source_info(file, true)?;
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&i.info)?);
            } else {
                let v = i.info.video()?;
                println!("{}  {}  {:.2}s  {}", i.info.path, i.info.container, i.info.duration, gb(i.info.size_bytes));
                println!(
                    "video: {} {}x{} ({}x{} displayed) {} fps{} {} {}-bit",
                    v.codec,
                    v.width,
                    v.height,
                    v.display_width,
                    v.display_height,
                    v.frame_rate,
                    if v.variable_frame_rate { " (variable)" } else { "" },
                    v.pix_fmt,
                    v.bit_depth
                );
                println!("color: {:?}", v.color);
                if let Some(a) = &i.interlace {
                    println!(
                        "interlace: {} (TFF {} / BFF {} / progressive {})",
                        if a.likely_interlaced() { "likely interlaced" } else { "progressive" },
                        a.tff,
                        a.bff,
                        a.progressive
                    );
                }
                for a in &i.info.audio {
                    println!("audio #{}: {} {} ch {} Hz {}", a.index, a.codec, a.channels, a.sample_rate, a.language.clone().unwrap_or_default());
                }
                for s in &i.info.subtitles {
                    println!("subtitle #{}: {} {}", s.index, s.codec, s.language.clone().unwrap_or_default());
                }
            }
        }
        Cmd::Hardware => {
            let app = open_app(&cli, true)?;
            let hw = app.hardware();
            let plan = app.resource_plan();
            if cli.json {
                println!("{}", serde_json::to_string_pretty(&serde_json::json!({"hardware": hw, "plan": plan, "sample": app.system_sample()}))?);
            } else {
                println!("{} {} — {} ({} threads), {} MB RAM ({} MB available)", hw.os, hw.arch, hw.cpu_brand, hw.logical_cores, hw.ram_total_mb, hw.ram_available_mb);
                for g in &hw.gpus {
                    println!("GPU: {} {} {:?} MB ({})", g.vendor, g.name, g.memory_total_mb, g.source);
                }
                for b in &hw.backends {
                    println!("backend {}: compiled={} available={} ({})", b.kind.label(), b.compiled, b.available, b.detail);
                }
                for l in &plan.explanation {
                    println!("{l}");
                }
            }
        }
        Cmd::Models { cmd } => {
            let app = open_app(&cli, false)?;
            match cmd {
                ModelsCmd::List | ModelsCmd::Refresh if cli.json => println!("{}", serde_json::to_string_pretty(&app.model_cards())?),
                ModelsCmd::List | ModelsCmd::Refresh => {
                    for c in app.model_cards() {
                        let id = c.status.manifest.as_ref().map(|m| m.id.clone()).or(c.status.installed.as_ref().map(|i| i.id.clone())).unwrap_or_default();
                        let name = c.status.manifest.as_ref().map(|m| m.display_name.clone()).unwrap_or_else(|| id.clone());
                        let size = c.status.manifest.as_ref().map(|m| m.download_size_bytes).unwrap_or(0);
                        println!("{:<34} {:<16} {:>9}  {:?}  {}", id, format!("{:?}", c.status.state), gb(size), c.capability.level, name);
                        if let Some(e) = c.status.installed.as_ref().and_then(|i| i.error.clone()) {
                            println!("    problem: {e}");
                        }
                    }
                }
                ModelsCmd::Install { id, files } => {
                    let mut local = BTreeMap::new();
                    for f in files {
                        let (role, path) = f.split_once('=').ok_or_else(|| anyhow!("--file expects role=path"))?;
                        local.insert(role.to_string(), PathBuf::from(path));
                    }
                    app.install_model(id, local)?;
                    while !app.installs_running().is_empty() {
                        std::thread::sleep(Duration::from_millis(200));
                    }
                    eprintln!();
                    let st = app.model_state(id);
                    println!("{id}: {st:?}");
                    return Ok(st == clearframe_models::manager::ModelState::Installed);
                }
                ModelsCmd::Remove { id } => println!("freed {}", gb(app.remove_model(id)?)),
                ModelsCmd::Verify { id } => {
                    let r = app.verify_model(id)?;
                    println!("{}: {}", id, if r.ok { "ok".to_string() } else { r.problems.join("; ") });
                    return Ok(r.ok);
                }
                ModelsCmd::Import { file, name, license } => println!("installed {}", app.import_custom_model(name, license, file.clone(), None)?),
            }
        }
        Cmd::Run { inputs, preset, pipeline, output_dir, name, start, end, container, recursive } => {
            let app = open_app(&cli, false)?;
            let (pl, out): (Pipeline, OutputSettings) = match (preset, pipeline) {
                (Some(p), _) => {
                    let all = app.presets();
                    let pr = all.iter().find(|x| &x.id == p || x.name.eq_ignore_ascii_case(p)).ok_or_else(|| anyhow!("preset {p} not found (see `clipfx presets list`)"))?;
                    (pr.pipeline.clone(), pr.output.clone().unwrap_or_default())
                }
                (None, Some(f)) => {
                    let p: Pipeline = serde_json::from_slice(&std::fs::read(f)?).context("reading pipeline JSON")?;
                    p.validate().map_err(|e| anyhow!(e))?;
                    (p, OutputSettings::default())
                }
                _ => bail!("use --preset or --pipeline"),
            };
            let mut out = out;
            if let Some(c) = container {
                out.container = clearframe_media::mux::Container::from_extension(c).ok_or_else(|| anyhow!("unknown container {c}"))?;
            }
            let range = match (start, end) {
                (Some(s), Some(e)) => Some(TimeRange { start: *s, end: *e }),
                (None, None) => None,
                _ => bail!("use --start and --end together"),
            };
            let files = App::collect_videos(inputs, *recursive);
            if files.is_empty() {
                bail!("no video files found");
            }
            let template = NewJob {
                source: files[0].clone(),
                pipeline: pl,
                output: out,
                range,
                folder: output_dir.clone(),
                file_name: if files.len() == 1 { name.clone() } else { None },
                remember_folder: false,
                preset_name: preset.clone(),
                batch_id: None,
                engine: Default::default(),
            };
            let ids: Vec<String> = if files.len() == 1 {
                vec![app.enqueue(template)?.id]
            } else {
                let r = app.enqueue_batch(files, template);
                for (f, e) in &r.failed {
                    eprintln!("skipped {}: {e}", f.display());
                }
                r.created.into_iter().map(|j| j.id).collect()
            };
            return wait_for_jobs(&app, &ids);
        }
        Cmd::Jobs { cmd } => {
            let app = open_app(&cli, false)?;
            match cmd {
                JobsCmd::List => {
                    for j in app.db.jobs(false)? {
                        println!("{}  {:<22} {:>5.1}%  {}  {}", j.id, j.state.as_str(), j.progress * 100.0, j.name, j.error.unwrap_or_default());
                    }
                }
                JobsCmd::Resume { id } => {
                    app.resume_job(id)?;
                    return wait_for_jobs(&app, &[id.clone()]);
                }
                JobsCmd::Cancel { id } => app.cancel_job(id)?,
                JobsCmd::Remove { id } => app.remove_job(id)?,
                JobsCmd::ClearFinished => println!("removed {}", app.clear_finished_jobs()?),
                JobsCmd::Process => {
                    let ids: Vec<String> = app.db.jobs(false)?.into_iter().filter(|j| !j.state.is_terminal()).map(|j| j.id).collect();
                    for id in &ids {
                        let _ = app.resume_job(id);
                    }
                    return wait_for_jobs(&app, &ids);
                }
            }
        }
        Cmd::Presets { cmd } => {
            let app = open_app(&cli, true)?;
            match cmd {
                PresetsCmd::List if cli.json => println!("{}", serde_json::to_string_pretty(&app.presets())?),
                PresetsCmd::List => {
                    for p in app.presets() {
                        println!("{:<28} {:<14} {}{}", p.id, p.category, p.name, if p.builtin { " (built-in)" } else { "" });
                    }
                }
                PresetsCmd::Export { id, file } => app.export_preset(id, file)?,
                PresetsCmd::Import { file } => {
                    let (p, missing) = app.import_preset(file)?;
                    println!("imported {} ({})", p.name, p.id);
                    for m in missing {
                        println!("needs model {m} (install with `clipfx models install {m}`)");
                    }
                }
            }
        }
        Cmd::Storage { cmd } => {
            let app = open_app(&cli, false)?;
            match cmd {
                StorageCmd::Show => {
                    let o = app.storage_overview();
                    if cli.json {
                        println!("{}", serde_json::to_string_pretty(&o)?);
                    } else {
                        println!("data:   {}", o.paths.data_dir.display());
                        println!(
                            "models: {} ({}, {} free){}",
                            o.paths.models_dir.display(),
                            gb(o.models.total_bytes),
                            o.models.free_bytes.map(gb).unwrap_or_default(),
                            if o.paths.models_is_default { "" } else { " [custom]" }
                        );
                        println!(
                            "cache:  {} ({}, {} free){}",
                            o.paths.cache_dir.display(),
                            gb(o.cache.total_bytes),
                            o.cache.free_bytes.map(gb).unwrap_or_default(),
                            if o.paths.cache_is_default { "" } else { " [custom]" }
                        );
                        for a in &o.cache.areas {
                            println!("  {:<44} {:>10} ({} protected)", a.label, gb(a.bytes), gb(a.protected_bytes));
                        }
                        for w in &o.warnings {
                            println!("warning: {w}");
                        }
                    }
                }
                StorageCmd::MoveModels { dest, switch_only } => {
                    let p = app.plan_relocation(StorageKind::Models, dest);
                    if !switch_only {
                        if let Some(e) = p.plan.problems.first() {
                            bail!("{e}");
                        }
                        println!("moving {} in {} files", gb(p.plan.total_bytes), p.plan.file_count);
                    }
                    println!("{}", app.relocate_blocking(StorageKind::Models, dest, if *switch_only { RelocationMode::SwitchOnly } else { RelocationMode::Move })?);
                }
                StorageCmd::MoveCache { dest, clear } => {
                    println!("{}", app.relocate_blocking(StorageKind::Cache, dest, if *clear { RelocationMode::ClearDisposableAndMove } else { RelocationMode::Move })?);
                }
                StorageCmd::ClearCache { all } => {
                    let r = app.clear_cache(&CacheArea::ALL, *all);
                    println!("freed {} ({} entries, {} protected kept)", gb(r.freed_bytes), r.removed_entries, r.skipped_protected);
                }
            }
        }
        Cmd::Diagnostics { redact, out } => {
            let app = open_app(&cli, true)?;
            let t = app.diagnostics_text(*redact);
            match out {
                Some(p) => std::fs::write(p, t)?,
                None => println!("{t}"),
            }
        }
        Cmd::Benchmark { model, size, frames } => {
            let app = open_app(&cli, true)?;
            let (w, h) = size.split_once('x').and_then(|(a, b)| Some((a.parse::<usize>().ok()?, b.parse::<usize>().ok()?))).ok_or_else(|| anyhow!("size must be WxH"))?;
            let (_, files, _) = app.models.resolve(model).ok_or_else(|| anyhow!("{model} is not installed"))?;
            let r = benchmark(&files, w, h, *frames, &app)?;
            println!("{}", serde_json::to_string_pretty(&r)?);
        }
        Cmd::CheckUpdate => {
            let app = open_app(&cli, true)?;
            let u = app.check_for_update()?;
            println!("current {} latest {}{}", u.current, u.latest, if u.newer { format!(" — download: {}", u.url) } else { " (up to date)".into() });
        }
    }
    Ok(true)
}

fn benchmark(files: &BTreeMap<String, PathBuf>, w: usize, h: usize, frames: usize, app: &App) -> Result<serde_json::Value> {
    use clearframe_inference::{Device, ImageModel, ImageParams, InterpModel, Precision, RgbImage, Weights};
    let plan = app.resource_plan();
    let mut img = RgbImage::new(w, h);
    for (i, v) in img.data.iter_mut().enumerate() {
        *v = ((i * 7919) % 1000) as f32 / 1000.0;
    }
    let weights = Weights::from_safetensors(files.get("weights").ok_or_else(|| anyhow!("benchmark supports single-weight models"))?, &Device::Cpu)?;
    let t0 = Instant::now();
    let arch = clearframe_inference::Architecture::detect(&weights).ok_or_else(|| anyhow!("unknown architecture"))?;
    let mut times = Vec::new();
    if arch == clearframe_inference::Architecture::Rife {
        let m = InterpModel::load(&weights, Precision::Fp32)?;
        for _ in 0..frames {
            let t = Instant::now();
            m.interpolate(&img, &img, 0.5, 1.0)?;
            times.push(t.elapsed().as_secs_f64());
        }
    } else {
        let m = ImageModel::load(&weights, Precision::Fp32)?;
        for _ in 0..frames {
            let t = Instant::now();
            m.process(&img, &ImageParams::default())?;
            times.push(t.elapsed().as_secs_f64());
        }
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = times[times.len() / 2];
    let mp = (w * h) as f64 / 1e6;
    Ok(serde_json::json!({
        "architecture": arch.id(), "input": format!("{w}x{h}"), "frames": frames, "backend": "cpu",
        "threads": plan.threads, "median_seconds_per_frame": median, "min": times[0], "max": times[times.len()-1],
        "input_megapixels_per_second": mp / median, "total_seconds": t0.elapsed().as_secs_f64(),
    }))
}

fn catalog_cmd(cmd: &CatalogCmd) -> Result<bool> {
    use clearframe_models::signature;
    match cmd {
        CatalogCmd::Keygen => {
            let (sk, pk) = signature::keygen();
            println!("public key (add to model-catalog/trusted_keys.json): {pk}");
            println!("secret key (keep offline, e.g. as a CI secret; never commit): {sk}");
        }
        CatalogCmd::Sign { file, key_file } => {
            let data = std::fs::read(file)?;
            clearframe_models::catalog::Catalog::parse(std::str::from_utf8(&data)?).map_err(|e| anyhow!("{e}"))?;
            let sk = std::fs::read_to_string(key_file)?;
            let sig = signature::sign(&data, sk.trim()).map_err(|e| anyhow!("{e}"))?;
            let out = PathBuf::from(format!("{}.sig", file.display()));
            std::fs::write(&out, sig)?;
            println!("wrote {}", out.display());
        }
        CatalogCmd::Verify { file, signature: sigf } => {
            let key = signature::verify(&std::fs::read(file)?, &std::fs::read_to_string(sigf)?, &signature::TrustedKeys::bundled()).map_err(|e| anyhow!("{e}"))?;
            println!("valid signature from {key}");
        }
        CatalogCmd::Validate { file } => {
            let c = clearframe_models::catalog::Catalog::parse(&std::fs::read_to_string(file)?).map_err(|e| anyhow!("{e}"))?;
            println!("valid catalog {} with {} models", c.catalog_version, c.models.len());
        }
    }
    Ok(true)
}
