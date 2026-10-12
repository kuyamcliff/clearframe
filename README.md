<div align="center">

<a href="https://www.clipfx.me/a/c8DN7xPn"><img src="docs/media/hero.jpg" alt="ClipFX Enhance: free AI video enhancer" width="100%"></a>

<h1>ClipFX Enhance</h1>

<p><b>Upscale, denoise, deblur and restore video with AI, right on your own computer.</b><br>
Free and open source. No account, no subscription, no cloud.</p>

<a href="https://www.clipfx.me/a/c8DN7xPn"><img src="docs/media/btn-download.svg" alt="Download for Windows" height="52"></a>&nbsp;&nbsp;<a href="https://www.clipfx.me/a/c8DN7xPn"><img src="docs/media/btn-video.svg" alt="Watch the trailer" height="52"></a>

<sub>Windows 10/11 (x64) · 96 MB installer · FFmpeg included · Linux: <a href="docs/guide.md#install">deb / AppImage from source</a></sub>

<br><br>

<img src="docs/media/app-split.jpg" alt="Split view comparing the original with the AI-enhanced preview" width="100%">

</div>

<table>
  <tr>
    <td width="50%"><img src="docs/media/app-models.jpg" alt="Models"><p align="center"><sub><b>Models.</b> Pick one, install it with one click, and see its license first.</sub></p></td>
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

1. **[Download](https://www.clipfx.me/a/c8DN7xPn)** and run the installer.
2. Drop a video in, turn on **Enhancement**, and install the model it suggests.
3. Press **Render frame** to preview, then **Export**.

> [!NOTE]
> Upscaling, denoising and deblurring run on your GPU (NVIDIA, AMD, Intel or Apple, through Vulkan, DirectX 12 or Metal). Frame interpolation and face recovery still use the CPU. Without a GPU, AI processing is slow. See [known limitations](docs/limitations.md).

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
