# ClipFX model training

Fine-tunes ClipFX Enhance's models on Kaggle GPUs. Data, checkpoints and results live in a private Cloudflare R2
bucket (`clipfx-training`); this folder holds the code.

| File | Runs on | Purpose |
|---|---|---|
| `r2util.py` | everywhere | R2 access (credentials from a private Kaggle dataset, never committed) |
| `prep_photos.py` | Kaggle CPU | Builds a data batch: licence-filtered high-resolution photos from Wikimedia Commons (CC0, public domain, CC BY, CC BY-SA), a clean reference per photo, and camera-pan sequences degraded with blur, downscaling, noise and real H.264/H.265/JPEG compression. A held-out test set uses fixed light/medium/heavy damage. |
| `train_sr.py` | Kaggle GPU | Fine-tunes Real-ESRGAN General v3 from its published weights (L1 + VGG perceptual + GAN, EMA). Checkpoints to R2 every 10 minutes and resumes automatically. Scores PSNR, LPIPS and temporal flicker on the test set and keeps a new best only if LPIPS improves without losing more than 0.2 dB PSNR. |

Every training sample keeps its source URL, author and licence in `meta.json`, so released models can credit
their data. Original model copyright notices (Real-ESRGAN BSD-3-Clause, etc.) ship with any fine-tuned weights.
