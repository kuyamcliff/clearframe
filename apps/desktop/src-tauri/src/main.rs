// Hide the console window on Windows release builds.
#![cfg_attr(all(not(debug_assertions), target_os = "windows"), windows_subsystem = "windows")]

mod commands;
mod media_server;
mod session_log;
mod taskbar;

use std::sync::Arc;

use tauri::{Emitter, Manager};

use clearframe_app::{App, AppEvent, WorkerCommand};

/// GPU adapters and a small GPU-vs-CPU computation check for `--self-test`.
fn gpu_report() -> serde_json::Value {
    let adapters = clearframe_inference::device::gpu_adapters();
    let check = if adapters.is_empty() {
        serde_json::json!({ "error": "no GPU adapter" })
    } else {
        match clearframe_inference::gpu::self_check(0) {
            Ok(c) => serde_json::json!(c),
            Err(e) => serde_json::json!({ "error": e }),
        }
    };
    serde_json::json!({ "adapters": adapters, "check": check })
}

fn main() {
    // WebKitGTK's DMA-BUF renderer leaves the window blank where the GPU driver or display
    // server cannot share buffers (some NVIDIA drivers, VMs, remote and virtual displays). The
    // interface is light to draw, so use the renderer that always works unless the user chose.
    // Set before any thread or GTK starts.
    #[cfg(target_os = "linux")]
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
        std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    // The installed executable doubles as the isolated processing worker.
    let args: Vec<String> = std::env::args().collect();
    // One session log in Documents, started before anything else so that even a failed
    // start is recorded. Worker processes append to the same file.
    let role = match args.get(1).map(String::as_str) {
        Some("--worker") => Some("worker"),
        Some("--still") => Some("still"),
        Some("--self-test") => Some("selftest"),
        Some("--version") => None,
        _ => Some("app"),
    };
    if let Some(role) = role {
        clearframe_app::logging::init_session(role);
        if role == "app" {
            session_log::header(&args);
        } else {
            log::info!("{role} process started: {:?}", &args[1..]);
        }
    }
    match args.get(1).map(String::as_str) {
        Some("--worker") => {
            std::process::exit(clearframe_app::worker::run_worker(std::path::Path::new(args.get(2).map(String::as_str).unwrap_or(""))));
        }
        Some("--still") => {
            std::process::exit(clearframe_app::worker::run_still(
                std::path::Path::new(args.get(2).map(String::as_str).unwrap_or("")),
                std::path::Path::new(args.get(3).map(String::as_str).unwrap_or("")),
            ));
        }
        Some("--version") => {
            println!("ClipFX Enhance {}", env!("CARGO_PKG_VERSION"));
            return;
        }
        Some("--self-test") => {
            // Used by installer tests: open state, find FFmpeg, check folders, report, exit.
            // The report is also written to the optional path argument because a
            // GUI-subsystem executable has no console on Windows.
            let report = match App::open(None, WorkerCommand::current_exe(), bundled_dirs(), Arc::new(|_| {})) {
                Ok(a) => {
                    let (t, e) = a.tools_status();
                    let p = a.paths();
                    let writable = [&p.data_dir, &p.models_dir, &p.cache_dir, &p.config_dir]
                        .iter()
                        .all(|d| std::fs::create_dir_all(d).is_ok() && clearframe_core::fsutil::ensure_writable_dir(d, false).is_ok());
                    let ffmpeg = t.as_ref().map(|t| serde_json::json!({ "version": t.version, "path": t.ffmpeg, "source": t.source, "vidstab": t.has_filter("vidstabdetect"), "zscale": t.has_filter("zscale") }));
                    serde_json::json!({ "ok": writable, "version": env!("CARGO_PKG_VERSION"), "exe": std::env::current_exe().ok(), "paths": p, "folders_writable": writable, "ffmpeg": ffmpeg, "ffmpeg_error": e, "models": a.model_cards().iter().filter(|c| c.status.installed.is_some()).count(), "gpu": gpu_report() })
                }
                Err(e) => serde_json::json!({ "ok": false, "error": e.to_string() }),
            };
            println!("{report}");
            log::info!("self-test report: {report}");
            if let Some(out) = args.get(2) {
                let _ = std::fs::write(out, report.to_string());
            }
            std::process::exit(if report["ok"] == true { 0 } else { 1 });
        }
        _ => {}
    }

    let started = std::time::Instant::now();
    let startup: Vec<std::path::PathBuf> = args.iter().skip(1).filter(|a| !a.starts_with('-')).map(std::path::PathBuf::from).filter(|p| p.is_file()).collect();

    tauri::Builder::default()
        // Opening a video while the app runs (double-click, "Open with") must not start a second
        // copy on the same database and queue: hand the files to the running window instead.
        // Registered first so the second process exits before doing anything else.
        .plugin(tauri_plugin_single_instance::init(|app, argv, cwd| {
            let files: Vec<std::path::PathBuf> = argv
                .iter()
                .skip(1)
                .filter(|a| !a.starts_with('-'))
                .map(|a| {
                    let p = std::path::PathBuf::from(a);
                    if p.is_absolute() {
                        p
                    } else {
                        std::path::Path::new(&cwd).join(p)
                    }
                })
                .filter(|p| p.is_file())
                .collect();
            log::info!("second launch forwarded {} file(s) to the running window", files.len());
            if let Some(w) = app.get_webview_window("main") {
                let _ = w.unminimize();
                let _ = w.show();
                let _ = w.set_focus();
            }
            if !files.is_empty() {
                if let Some(state) = app.try_state::<commands::StartupFiles>() {
                    state.0.lock().unwrap().extend(files);
                }
                let _ = app.emit("cf://open-files", ());
            }
        }))
        .manage(commands::StartupFiles(std::sync::Mutex::new(startup)))
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .setup(move |app| {
            log::info!("window setup started after {} ms", started.elapsed().as_millis());
            let handle = app.handle().clone();
            let events = session_log::EventLogger::default();
            let bar = taskbar::Taskbar::default();
            let sink: clearframe_app::EventSink = Arc::new(move |e: AppEvent| {
                events.log(&e);
                bar.on_event(&handle, &e);
                let _ = handle.emit("cf://event", &e);
            });
            let mut dirs = bundled_dirs();
            if let Ok(r) = app.path().resource_dir() {
                dirs.push(r.clone());
                dirs.push(r.join("ffmpeg"));
            }
            log::info!("resource dirs: {dirs:?}");
            let state = App::open_background_tools(None, WorkerCommand::current_exe(), dirs, sink).map_err(|e| {
                log::error!("could not open the app state: {e:#}");
                e.to_string()
            })?;
            session_log::state_details(&state);
            session_log::background_details(state.clone());
            session_log::start_perf_sampler(app.handle().clone(), state.clone());
            if let Err(err) = media_server::start() {
                log::warn!("media server failed to start: {err}");
            }
            commands::allow_cache_assets(app.handle(), &state);
            state.start_scheduler();
            app.manage(state);
            log::info!("setup finished after {} ms", started.elapsed().as_millis());
            Ok(())
        })
        .on_window_event(|window, event| {
            match event {
                tauri::WindowEvent::Moved(_) | tauri::WindowEvent::Resized(_) => {}
                tauri::WindowEvent::DragDrop(d) => log::info!("window {}: drag and drop {d:?}", window.label()),
                other => log::info!("window {}: {other:?}", window.label()),
            }
            if let tauri::WindowEvent::Destroyed = event {
                // Stop running jobs at a safe point so they can resume next time.
                if let Some(app) = window.try_state::<Arc<App>>() {
                    app.pause_all_and_wait(std::time::Duration::from_secs(10));
                }
            }
        })
        .invoke_handler(commands::handler())
        .run(tauri::generate_context!())
        .unwrap_or_else(|e| {
            log::error!("ClipFX Enhance could not run: {e:#}");
            log::logger().flush();
            panic!("error while running ClipFX Enhance: {e}");
        });
    log::info!("ClipFX Enhance closed after {:.0} s", started.elapsed().as_secs_f64());
}

fn bundled_dirs() -> Vec<std::path::PathBuf> {
    let mut v = Vec::new();
    if let Ok(exe) = std::env::current_exe() {
        if let Some(d) = exe.parent() {
            v.push(d.to_path_buf());
            v.push(d.join("resources"));
            v.push(d.join("resources").join("ffmpeg"));
        }
    }
    v
}
