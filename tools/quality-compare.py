#!/usr/bin/env python3
"""Checks that enhancement actually improves low-quality video, against a known original.

    python3 tools/quality-compare.py <clipfx-cli> <data-root> <reference-video> <out-dir> [--at 2.4] [--variants a,b] [--tiers 240,360,480,720]

The reference should be a clean, high-resolution clip (e.g. a Blender open movie at 1080p).
Each tier is made the way YouTube serves low resolutions (downscaled, H.264 at YouTube-like
bitrates). Every variant processes a few frames of each tier, and the result is compared
with the reference at the reference's size:

  PSNR / SSIM  pixel accuracy (AI upscalers usually score slightly below a plain resize here)
  LPIPS        how close it looks to the original, lower is better (needs `pip install lpips`)

A variant that does not beat "plain resize" on LPIPS is not making the video better.
Writes report.md in <out-dir>.
"""
import argparse
import json
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = json.loads((ROOT / "apps/desktop/src/lib/fixtures/operation_defaults.json").read_text())
TIERS = {240: (426, "200k"), 360: (640, "400k"), 480: (854, "750k"), 720: (1280, "1500k")}


def enhance(model=None, **settings):
    st = json.loads(json.dumps(DEFAULTS["enhance"]))
    if model:
        st["settings"]["model_id"] = model
    st["settings"].update(settings)
    st.update(id="enhance", enabled=True)
    return {"stages": [st]}


VARIANTS = {
    "default": enhance(),
    "anime-v3": enhance("xinntao.realesr-animevideo-v3", denoise=None),
    "general-v3-dn0": enhance("xinntao.realesr-general-v3", denoise=0.0),
    "general-v3-dn50": enhance("xinntao.realesr-general-v3", denoise=0.5),
    "general-v3-dn100": enhance("xinntao.realesr-general-v3", denoise=1.0),
    "x2plus": enhance("xinntao.realesrgan-x2plus", denoise=None),
}


def ffmpeg(*args):
    subprocess.run(["ffmpeg", "-v", "error", "-y", *map(str, args)], check=True)


def png(src, t, dst, w, h, flags):
    ffmpeg("-ss", t, "-i", src, "-frames:v", 1, "-vf", f"scale={w}:{h}:flags={flags},format=rgb24", dst)
    return dst


def metric(a, b, name):
    err = subprocess.run(["ffmpeg", "-i", str(a), "-i", str(b), "-lavfi", name, "-f", "null", "-"], capture_output=True, text=True).stderr
    m = re.search(r"All:([\d.]+)|average:([\d.]+)", err)
    return float(next(g for g in m.groups() if g)) if m else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cli")
    ap.add_argument("data_root")
    ap.add_argument("reference")
    ap.add_argument("out")
    ap.add_argument("--at", type=float, default=2.0, help="seconds into the reference to compare")
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--tiers", default="240,360,480")
    ap.add_argument("--device", default=os.environ.get("CLIPFX_DEVICE", "gpu:0"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ref = Path(a.reference)
    w, h = (int(x) for x in subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0", str(ref)], capture_output=True, text=True, check=True).stdout.strip().split(",")[:2])
    gt = png(ref, a.at, out / "reference.png", w, h, "lanczos")
    try:
        import lpips, torch, numpy as np
        from PIL import Image
        net = lpips.LPIPS(net="alex", verbose=False)

        def look(p):
            t = lambda q: torch.from_numpy(np.asarray(Image.open(q))).permute(2, 0, 1)[None].float() / 127.5 - 1
            with torch.no_grad():
                return float(net(t(p), t(gt)))
    except ImportError:
        look = None
        print("lpips not installed: reporting PSNR/SSIM only")

    rows = []
    env = dict(os.environ, CLIPFX_DEVICE=a.device)
    for tier in map(int, a.tiers.split(",")):
        tw, rate = TIERS[tier]
        low = out / f"{tier}p.mp4"
        if not low.exists():
            ffmpeg("-i", ref, "-vf", f"scale={tw}:-2:flags=bicubic", "-c:v", "libx264", "-preset", "slow", "-b:v", rate, "-maxrate", rate, "-bufsize", rate, "-pix_fmt", "yuv420p", "-an", low)
        cands = {"plain resize": png(low, a.at, out / f"{tier}p_resize.png", w, h, "lanczos")}
        for v in a.variants.split(","):
            pf = out / f"{tier}p_{v}.json"
            pf.write_text(json.dumps(VARIANTS[v]))
            r = subprocess.run([a.cli, "--data-root", a.data_root, "run", str(low), "-o", str(out), "--name", f"{tier}p_{v}", "--pipeline", str(pf), "--start", f"{a.at:.3f}", "--end", f"{a.at + 0.2:.3f}"], capture_output=True, text=True, env=env)
            res = sorted(out.glob(f"{tier}p_{v}*.mp4"))
            if r.returncode != 0 or not res:
                print(f"{tier}p {v}: failed: {(r.stdout + r.stderr).strip()[-300:]}")
                continue
            cands[v] = png(res[-1], 0, out / f"{tier}p_{v}.png", w, h, "area")
        for k, p in cands.items():
            row = {"tier": f"{tier}p", "method": k, "psnr": metric(p, gt, "psnr"), "ssim": metric(p, gt, "ssim"), "lpips": look(p) if look else None}
            rows.append(row)
            print(f"{row['tier']:5} {k:18} PSNR {row['psnr']:6.2f}  SSIM {row['ssim']:.4f}" + (f"  LPIPS {row['lpips']:.4f}" if look else ""), flush=True)
    lines = ["| Tier | Method | PSNR | SSIM | LPIPS (lower is better) |", "|---|---|---|---|---|"]
    for r in rows:
        lp = "" if r["lpips"] is None else f"{r['lpips']:.4f}"
        lines.append(f"| {r['tier']} | {r['method']} | {r['psnr']:.2f} | {r['ssim']:.4f} | {lp} |")
    (out / "report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
