# Model parity check

ClearFrame runs models with its own Rust implementations (crates/inference/src/arch). This
tool proves they compute the same thing as the official PyTorch code.

Last run (Linux x86-64, CPU, fp32): every architecture matched the official implementation
with a maximum absolute difference ≤ 3e-5 on 0..1 images (SRVGGNetCompact, RRDBNet, FFDNet,
DRUNet, NAFNet, RIFE 4.26, RetinaFace, GFPGAN 1.4).

## Steps

1. Python 3.10+ with `torch` (CPU is fine) and `numpy` in a virtual environment.
2. Put the upstream model code in a folder and point `CF_REFPKG` at it:
   - `basicsr/` from https://github.com/XPixelGroup/BasicSR (SRVGGNetCompact, RRDBNet)
   - `models/` from https://github.com/cszn/KAIR (FFDNet, DRUNet)
   - NAFNet's `basicsr/models/archs/NAFNet_arch.py` from https://github.com/megvii-research/NAFNet (merged into the same `basicsr/` folder)
   - `model/` (IFNet_HDv3) from https://github.com/hzwer/Practical-RIFE (v4.26)
   - `gfpgan/` and `facexlib/` from https://github.com/TencentARC/GFPGAN and https://github.com/xinntao/facexlib
3. Download the original weights listed in `model-catalog/catalog.json` into `CF_MODELS` and check their SHA-256.
4. Generate references, convert the weights with ClearFrame, and compare:

```sh
CF_REFPKG=... CF_MODELS=... CF_REFS=refs python tools/parity/make_refs.py
CF_REFPKG=... CF_MODELS=... CF_REFS=refs python tools/parity/make_refs_face.py
# put the converted weights (each installed model's weights/*.safetensors, named after the
# original file, e.g. RealESRGAN_x4plus.safetensors) into one folder, then:
cargo run --release -p clearframe-inference --example parity -- refs <safetensors-dir>
```

The scripts only load weights with `torch.load(..., weights_only=True)`.
