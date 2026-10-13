# Known limitations

These are the limits of the current release, stated plainly.

## Performance
- **GPU support covers the image models only.** Upscaling (Real-ESRGAN), denoising
  (FFDNet, DRUNet) and deblurring (NAFNet) run on any GPU through Vulkan, DirectX 12 or
  Metal. Face recovery (GFPGAN) still runs on the CPU.
- **GPU speed has not been measured on real graphics cards yet.** Correctness is tested
  on every graphics API in CI (Mesa lavapipe for Vulkan, Microsoft WARP for DirectX 12 and
  Apple's GPU for Metal) by comparing against the CPU path, but those test devices are
  software or virtual, so they say nothing about speed. The kernels are hand-written and
  not yet tuned (no FP16, no tensor cores).
- **Without a GPU, AI processing is slow.** Measured on a 4-thread Xeon (2.8 GHz, AVX-512)
  with the Balanced mode's 3 threads: Real-ESRGAN AnimeVideo v3 (the lightest upscaler)
  takes **about 3 s per 640×360 frame** (4× to 2560×1440). Software GPU drivers (shown as
  "software, slow") are never picked automatically because they are slower than the CPU path.
- Memory budgets are targets. The operating system can still swap under pressure. The app
  pauses exports when free memory runs low, if that option is on (it is by default).
- GPU utilisation is measured only for NVIDIA cards, through `nvidia-smi`. Other GPUs show
  "—" rather than an invented number.

## Platforms
- Each release is installed, self-tested and opened automatically on Windows, macOS (Apple
  Silicon, plus Intel through Rosetta) and Linux virtual machines, but not on physical PCs with
  real GPUs. See the [support matrix](support-matrix.md).
- Installers are not code-signed or notarized. Windows SmartScreen and macOS Gatekeeper will warn.
- The Windows installer and the macOS app bundle FFmpeg. On Linux, FFmpeg comes from your
  distribution (the .deb and .rpm install it; the AppImage needs it installed).
- If WebView2 is missing on Windows, the installer downloads it; that step needs internet.
- Linux: with the theme set to "System", light or dark follows your GTK theme, not the desktop
  portal's colour-scheme setting (asking the portal could hold the window back when it does not
  answer). Choose Light or Dark in Preferences to override.

## Features
- SDR→HDR re-encodes the picture as PQ/HLG with correct metadata. It does not invent highlight
  detail (no inverse tone-mapping model).
- Face recovery (GFPGAN) can change how a face looks at high strength. Use the split view to
  compare.
- Deblur (NAFNet GoPro) is trained on camera-shake blur. It does not fix out-of-focus footage.
- Preview playback in the app uses H.264 proxy copies. Exports are not limited by this.
- Interlaced output is not produced; deinterlacing always outputs progressive video.
- Custom model import supports Real-ESRGAN-compatible SRVGGNetCompact and RRDBNet weights
  only.
- Signed online catalog updates are implemented, but no signing key has been published yet,
  so remote catalog updates are refused. The bundled catalog works offline.
- The interface is English only.

## Storage
- GPU driver shader caches and the system web view's profile folder are outside the app's
  control. See [storage.md](storage.md).
- Uninstalling with "Delete the application data" removes the default data folder. Models or
  cache moved to a custom location are left in place.
