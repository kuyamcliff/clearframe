# ClipFX Enhance

Free, open-source video enhancement that runs on your own computer: upscaling, denoising,
deblurring, face recovery, frame interpolation and slow motion, stabilization,
deinterlacing, color and HDR conversion, and film grain. You can chain any of these into a
pipeline. There is no account, no subscription and no cloud processing. Once the models you
want are installed, ClipFX Enhance works offline.

ClipFX Enhance is MIT licensed. Each AI model has its own license, shown before you install it
(see [MODEL_LICENSES.md](MODEL_LICENSES.md)).

## Features

| Area | What you get |
|---|---|
| Enhance / upscale | Real-ESRGAN family (AnimeVideo v3, General v3 with adjustable denoise, x4plus, x4plus Anime 6B, x2plus). Output can be the native scale, a scale factor, a target height (1080p/4K/8K), an exact size, or the source size. Includes a strength blend and automatic tiling sized to your memory budget. |
| Denoise | FFDNet and DRUNet (AI, with noise level σ), plus the hqdn3d and NL-means filters. Mix control. |
| Deblur | NAFNet (GoPro motion blur). |
| Face recovery | RetinaFace detection with GFPGAN 1.4 restoration, strength blending and temporal smoothing. |
| Frame interpolation | RIFE 4.26, FFmpeg motion interpolation or frame blending. Targets are ×2/×3/×4/×8 or a specific frame rate. Scene-cut and duplicate-frame handling. Slow motion up to 16×, with audio muted, pitch-preserving stretch or tape-style stretch. |
| Stabilize | vid.stab two-pass analysis, with auto-crop, black borders or full frame, and a tripod mode. |
| Deinterlace | bwdif / yadif at the same or double rate, with automatic field-order detection. |
| Color & HDR | Brightness, contrast, saturation and gamma. HDR→SDR tone mapping (Hable, Mobius, Reinhard, Clip). SDR→HDR10/HLG encoding. |
| Film grain | Luminance-aware, animated, seeded grain with presets. |
| Pipelines | Any order and any number of stages, with per-stage enable, undo/redo, presets (`.cfxpreset` import/export) and projects (`.cfxproj`). |
| Preview | Real-model still frames and short clips, compared as a draggable split, side by side, original or processed. Previews are cached. |
| Export | H.264, H.265, AV1, VP9, ProRes, FFV1 and hardware encoders when your FFmpeg has them. Quality, bitrate or lossless rate control. Keeps audio, subtitles, chapters and metadata. Converts VFR to CFR. Output is validated by a full decode. |
| Queue | Separate worker processes. Pause, resume, cancel, retry and reorder. Batch export. Crash-safe resume from the last finished segment. Never overwrites a file. |
| Models | Install, pause, resume, cancel, verify, repair, update and uninstall. Install only happens when you ask. SHA-256 verification, a safe weight loader that never executes pickled code, and a smoke test before a model is marked installed. Missing or damaged files are detected. You can import custom Real-ESRGAN-compatible weights. |
| Storage | Per-user folders. You can move the models or cache folder (copy, verify, switch, with cancel and crash recovery). Cache usage per area, safe clearing, a size limit, age-based cleanup and low-disk protection. |
| Performance | Balanced, High performance and Maximum modes, or custom threads, priority and memory budget. Memory-pressure pause, compute-device choice, GPU memory limit, tile fallback when out of memory, and a live CPU/RAM/GPU monitor. |

## Install

- **Windows 10/11 (x64):** run `ClipFX Enhance_<version>_x64-setup.exe`. It installs per machine
  into Program Files and includes FFmpeg. Your settings, models and cache live in
  `%LOCALAPPDATA%\ClipFX Enhance` and survive upgrades and uninstalls; they are removed only if
  you tick “Delete the application data” in the uninstaller. If WebView2 is missing, the
  installer downloads it.
- **Linux (x64):** install the `.deb` (Debian/Ubuntu), `.rpm` (Fedora/openSUSE) or
  run the AppImage. FFmpeg comes from your distribution (`ffmpeg` package). The binary is
  `clipfx-enhance`.
- **macOS 11+:** build from source (see below) and install FFmpeg with Homebrew
  (`brew install ffmpeg`). Signed macOS builds are not provided yet.

FFmpeg needs `libvidstab` for stabilization and `zimg` (zscale) for HDR conversion. The app
tells you if either is missing and turns those features off.

See [docs/support-matrix.md](docs/support-matrix.md) for what has actually been tested.

## Quick start

1. Drop a video on the window, or press **Browse**. Pick a preset or start from scratch.
2. Turn on adjustments in the right-hand **Adjustments** panel (Enhancement, Frame
   interpolation, Stabilization and so on). If a model isn't installed, it shows its download
   size. It installs only when you click it.
3. Press **Render frame**, or choose a clip length from its menu. Compare using the
   player-mode buttons at the top: Original, Preview, Split or Side by side.
4. Set the container, codec and quality in **Codec settings**.
5. Press **Export as…** to choose a file, or the share button next to it for a quick export.
   Progress is in **Export queue**. The result is always a new file.

**Sources** holds every video you opened. **Export all with current settings** batches them.

| Shortcut | Action |
|---|---|
| ← / → | Step one frame (hold Shift to step one second) |
| I / O | Set in / out |
| 1 2 3 4 | Original / Preview / Split / Side by side |
| Ctrl+R / Ctrl+Shift+R | Render frame / render clip |
| Ctrl+E / Ctrl+Shift+E | Export as… / quick export |
| Ctrl+Z / Ctrl+Shift+Z | Undo / redo |
| Ctrl+S | Save project |
| Ctrl+J | Export queue |
| Ctrl+, | Preferences |

## Command line

The `clipfx` CLI uses the same engine and data:

```sh
clipfx info video.mp4
clipfx models list
clipfx models install xinntao.realesr-animevideo-v3
clipfx run video.mp4 --preset builtin.anime-2x --output-dir ./exports
clipfx storage show
clipfx hardware --json
```

`--data-root <dir>` keeps everything in one folder (portable mode). Run `clipfx --help`
for all commands.

## Privacy

ClipFX Enhance contacts the network only when you install a model, or when you explicitly check
for updates (both features can be disabled). There is no telemetry. Diagnostics reports are
created locally, can hide personal paths, and are never uploaded.

## Building from source

Requirements: Rust (stable), Node.js 20+, FFmpeg, and the Tauri prerequisites for your OS.
On Linux these are `libwebkit2gtk-4.1-dev libayatana-appindicator3-dev librsvg2-dev patchelf`.

```sh
cd apps/desktop
npm ci
npx tauri dev                     # run the app
npx tauri build                   # installers for the current OS
cargo build --release -p clearframe-cli   # the CLI (from the repository root)
```

On Windows, run `pwsh packaging/windows/fetch-ffmpeg.ps1` first so the installer includes
FFmpeg. GPU builds: add `--features cuda` (NVIDIA, needs the CUDA toolkit) or
`--features metal` (Apple) to the desktop or CLI crate.

## Tests

```sh
cargo test --workspace --exclude clearframe-desktop     # engine, storage, queue, CLI integration
CLIPFX_TEST_MODELS=1 cargo test -p clearframe-pipeline --test models_e2e   # real model downloads
cd apps/desktop && npm test && npx playwright test e2e/app.spec.ts            # UI
pwsh packaging/windows/test-installer.ps1 -Installer <setup.exe>             # Windows installer (disposable machine)
```

All tests use temporary folders and never touch your real ClipFX Enhance data.

## Project layout

| Path | Contents |
|---|---|
| `crates/core` | paths, settings, database, job state machine, cache, relocation, file safety |
| `crates/models` | catalog, signatures, safe checkpoint loader, conversion, model manager |
| `crates/inference` | candle implementations of every model architecture, tiling, devices |
| `crates/media` | FFmpeg discovery, probing, raw pipes, encoding, muxing, validation |
| `crates/pipeline` | operations, planner, processors, segment runner, presets, projects |
| `crates/hardware` | hardware detection and capability levels |
| `crates/app` | application services shared by the desktop app and CLI (queue, workers, storage, resources) |
| `crates/cli` | the `clipfx` command |
| `apps/desktop` | Tauri shell and React interface |
| `model-catalog` | the signed model catalog |
| `packaging` | installer hooks and scripts |
| `docs` | architecture, storage, support matrix, limitations |

## Documentation

- [Architecture](docs/architecture.md)
- [Storage locations and cache](docs/storage.md)
- [Support matrix](docs/support-matrix.md)
- [Known limitations](docs/limitations.md)
- [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md) · [Code of conduct](CODE_OF_CONDUCT.md)

## License

MIT, see [LICENSE](LICENSE). Third-party components are listed in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md); model licenses are in
[MODEL_LICENSES.md](MODEL_LICENSES.md). ClipFX Enhance is provided as is, without warranty. If you
use it commercially, you are responsible for checking that the models you use allow it.
