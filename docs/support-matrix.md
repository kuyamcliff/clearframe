# Support matrix

This page lists only what has actually been tested. "Built in CI" means the CI workflow in
`.github/workflows/ci.yml` is set up to do it. It does not mean someone has watched it pass.

| Platform | Build | Automated tests | Installer | Manually tested |
|---|---|---|---|---|
| Linux x86-64 (Ubuntu 24.04) | ✅ local | ✅ all Rust suites, UI unit tests, Playwright UI tests | ✅ `.deb` installed and self-tested; `.rpm`/AppImage built | Development container only (CPU, no GPU, no display server except Xvfb) |
| Windows 10/11 x64 | configured in CI | configured in CI | NSIS lifecycle script (`packaging/windows/test-installer.ps1`) configured in CI | **Not yet** |
| macOS 11+ (Apple silicon / Intel) | configured in CI (Rust only) | configured in CI | `.dmg` not produced or signed yet | **Not yet** |

## Compute backends

| Backend | Status |
|---|---|
| CPU (x86-64) | Tested. All model adapters match the official PyTorch output (max error ≤ 3e-5). |
| NVIDIA CUDA (`--features cuda`) | Code path present, **not built or tested** in this release. Default builds are CPU-only. |
| Apple Metal (`--features metal`) | Code path present, **not built or tested**. |
| AMD / Intel GPUs | No GPU inference backend. Hardware *encoders* (AMF, QSV) are used when FFmpeg provides them. |

The Models window shows a capability level for each model on your machine: *Tested on this
system*, *Compatible (not certified)*, *CPU only* or *Not supported*. Only Linux + CPU is
currently certified.

## FFmpeg

The tested build is the Ubuntu 24.04 package (FFmpeg 6.1.1, with vid.stab and zimg). The app
works with any FFmpeg ≥ 5 that has `ffprobe`. Stabilization needs `libvidstab` and HDR
conversion needs `zimg`. If either is missing, the app reports it and disables that feature.
