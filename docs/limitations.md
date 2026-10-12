# Known limitations

These are the limits of the current release, stated plainly.

## Performance
- **CPU-only by default.** Release builds run models on the CPU, and AI upscaling on a CPU is
  slow. Measured on a 4-thread Xeon (2.8 GHz, AVX-512) with the Balanced mode's 3 threads:
  Real-ESRGAN AnimeVideo v3 (the lightest upscaler) takes **about 3 s per 640×360 frame**
  (4× to 2560×1440). Larger models (x4plus, DRUNet) and larger inputs scale up from there.
  GPU builds (`cuda`, `metal` features) exist in code but are untested.
- Memory budgets are targets. The operating system can still swap under pressure. The app
  pauses exports when free memory runs low, if that option is on (it is by default).
- GPU utilisation is measured only for NVIDIA cards, through `nvidia-smi`. Other GPUs show
  "—" rather than an invented number.

## Platforms
- Windows and macOS builds and installers are set up in CI but have not been run or tested by
  hand for this release. See the [support matrix](support-matrix.md).
- Installers are not code-signed. Windows SmartScreen and macOS Gatekeeper will warn.
- The Windows installer bundles FFmpeg. On Linux and macOS you need FFmpeg from your
  package manager or Homebrew.
- If WebView2 is missing on Windows, the installer downloads it; that step needs internet.

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
