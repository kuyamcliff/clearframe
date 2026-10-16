<div align="center">

<a href="https://www.clipfx.me/a/LFeCDUGg"><img src="docs/media/hero.jpg" alt="ClipFX Enhance: free AI video enhancer" width="100%"></a>

<h1>ClipFX Enhance</h1>

<p><b>Upscale, denoise, deblur and restore video with AI, right on your own computer.</b><br>
Free and open source. No account, no subscription, no cloud.</p>

<a href="https://github.com/kuyamcliff/clipfx-enhance/releases/latest/download/ClipFX-Enhance-windows-x64-setup.exe"><img src="docs/media/btn-windows.svg" alt="Download for Windows" height="48"></a>&nbsp;
<a href="https://github.com/kuyamcliff/clipfx-enhance/releases/latest/download/ClipFX-Enhance-macos-universal.dmg"><img src="docs/media/btn-macos.svg" alt="Download for macOS" height="48"></a>&nbsp;
<a href="https://github.com/kuyamcliff/clipfx-enhance/releases/latest/download/ClipFX-Enhance-linux-x86_64.AppImage"><img src="docs/media/btn-linux.svg" alt="Download for Linux" height="48"></a>

<a href="https://www.clipfx.me/a/LFeCDUGg"><img src="docs/media/btn-video.svg" alt="Watch the trailer" height="48"></a>

<sub>Windows 10/11 x64 · macOS 11+ (Apple Silicon and Intel) · Linux x64 (<a href="https://github.com/kuyamcliff/clipfx-enhance/releases/latest/download/clipfx-enhance-linux-amd64.deb">.deb</a> · <a href="https://github.com/kuyamcliff/clipfx-enhance/releases/latest/download/clipfx-enhance-linux-x86_64.rpm">.rpm</a> · <a href="https://github.com/kuyamcliff/clipfx-enhance/releases/latest/download/ClipFX-Enhance-linux-x86_64.AppImage">AppImage</a>) · <a href="https://github.com/kuyamcliff/clipfx-enhance/releases/latest">all downloads</a><br>Also on <b>clipfx.me</b>: <a href="https://www.clipfx.me/a/LFeCDUGg">Windows</a> · <a href="https://www.clipfx.me/a/Scn6uqeP">macOS</a> · <a href="https://www.clipfx.me/a/AZwabs2e">Linux</a></sub>

<br><br>

<img src="docs/media/app-split.jpg" alt="Split view comparing the original with the AI-enhanced preview" width="100%">

</div>

<table>
  <tr>
    <td width="50%"><img src="docs/media/app-models.jpg" alt="Models"><p align="center"><sub><b>Models.</b> Pick one, see the hardware it needs, and install it with one click.</sub></p></td>
    <td width="50%"><img src="docs/media/app-presets.jpg" alt="Presets"><p align="center"><sub><b>Presets.</b> Start from a recipe or save your own.</sub></p></td>
  </tr>
  <tr>
    <td width="50%"><img src="docs/media/app-sbs.jpg" alt="Side by side"><p align="center"><sub><b>Compare.</b> Original, preview, split or side by side.</sub></p></td>
    <td width="50%"><img src="docs/media/app-queue.jpg" alt="Export queue"><p align="center"><sub><b>Queue.</b> Batch exports, each one checked after it is written.</sub></p></td>
  </tr>
</table>

## What it does

| | |
|---|---|
| **Upscale** | Real-ESRGAN models, up to 4K and beyond |
| **Denoise** | FFDNet and DRUNet, or classic filters |
| **Deblur** | NAFNet for camera-shake blur |
| **Restore faces** | GFPGAN with RetinaFace detection |
| **Smooth motion** | RIFE frame interpolation and slow motion |
| **Stabilize** | Two-pass vid.stab |
| **Also** | Deinterlace, SDR ↔ HDR, resize, film grain |

Stack as many steps as you like, preview any moment against the original, and export to H.264, H.265, AV1, ProRes and more. Audio, subtitles and chapters are kept, and your original file is never overwritten.

## Get started

1. **Download** for your system above and install it. FFmpeg is included on Windows and macOS.
2. Drop a video in, turn on **Enhancement**, and install the model it suggests.
3. Press **Render frame** to preview, then **Export**.

The app tells you when an update is out. On Windows and Linux it installs with one click, and on Windows only the app itself (a few MB) is downloaded.

<details>
<summary><b>First launch on macOS</b></summary>

The app isn't notarized by Apple, which needs a paid developer account, so macOS asks before the first launch. Drag **ClipFX Enhance** to Applications and open it once. Then go to **System Settings → Privacy & Security**, click **Open Anyway**, and confirm.
</details>

<details>
<summary><b>Linux</b></summary>

The `.deb` (Debian, Ubuntu, Mint) and `.rpm` (Fedora, openSUSE) packages install FFmpeg from your distribution. For the AppImage, install `ffmpeg` with your package manager, then run `chmod +x ClipFX-Enhance-linux-x86_64.AppImage` and open it.
</details>

> [!NOTE]
> Every AI model (upscaling, denoising, deblurring, frame interpolation, face detection and face restoration) runs on your GPU (NVIDIA, AMD, Intel or Apple, through Vulkan, DirectX 12 or Metal). Without a GPU, AI processing is slow. See [known limitations](docs/limitations.md).

## Privacy

Your videos never leave your computer. To find and fix problems in later versions, ClipFX Enhance
sends its own activity log: startup, errors and crashes, the settings and speed of each export,
and your hardware (CPU, memory, GPU). File paths, file names and your user name are removed
first. Logs collected while offline are sent when you are back online. Turn this off in
**Preferences → General → Privacy**.

## Build from source

```sh
cd apps/desktop && npm ci
npx tauri dev      # run
npx tauri build    # make an installer for this OS
```

You'll need Rust, Node.js 20+ and FFmpeg. The [guide](docs/guide.md) covers every feature, keyboard shortcuts, the `clipfx` command line and the tests.

## Learn more

[Guide](docs/guide.md) · [Architecture](docs/architecture.md) · [Storage](docs/storage.md) · [Support matrix](docs/support-matrix.md) · [Limitations](docs/limitations.md) · [Contributing](CONTRIBUTING.md) · [Security](SECURITY.md)

## License

MIT, see [LICENSE](LICENSE). Each AI model has its own license, which the app shows before you install it ([MODEL_LICENSES.md](MODEL_LICENSES.md)). Bundled third-party software is listed in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

<sub>Screenshot footage: <i>Big Buck Bunny</i> © Blender Foundation, CC BY 3.0. All "after" images are real ClipFX Enhance output.</sub>
