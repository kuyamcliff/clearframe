#!/usr/bin/env python3
"""Runs ClipFX Enhance on a corpus of real videos (see tools/real-corpus.sh) and checks every result.

    python3 tools/e2e-real.py <clipfx-cli> <data-root> <corpus-dir> <out-dir> [--device cpu|gpu:0] [--quick]

Three groups:
  presets   every built-in preset on live action, animation and low-quality faces
  formats   an AI upscale and AI frame interpolation on every file in the corpus
  models    every model in the catalog (and the colour and sharpen stages) on its own

Each output must decode without errors from start to finish, last as long as the source
(times the slow-motion factor), keep audio when the source had it, keep portrait video
portrait, and have the expected size where the step defines one. Writes report.md and
report.json in <out-dir>; exits non-zero if anything failed.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULTS = json.loads((ROOT / "apps/desktop/src/lib/fixtures/operation_defaults.json").read_text())


def stage(kind, sid=None, **settings):
    d = json.loads(json.dumps(DEFAULTS[kind]))
    d["settings"].update(settings)
    d["id"] = sid or kind
    d["enabled"] = True
    return d


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)], capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        return None
    return json.loads(r.stdout)


def rotation(v):
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            return int(sd["rotation"]) % 360
    return int(v.get("tags", {}).get("rotate", 0)) % 360


def displayed(v):
    w, h = int(v["width"]), int(v["height"])
    return (h, w) if rotation(v) in (90, 270) else (w, h)


def info(path):
    p = probe(path)
    if not p:
        return None
    v = next((s for s in p["streams"] if s["codec_type"] == "video"), None)
    a = [s for s in p["streams"] if s["codec_type"] == "audio"]
    dur = float(p["format"].get("duration") or v.get("duration") or 0)
    return {"v": v, "audio": len(a), "duration": dur, "size": displayed(v) if v else None}


def decode_errors(path):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0", "-f", "null", "-"], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.returncode, r.stderr.strip()[:400]


def frames(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        return int(r.stdout.strip().split(",")[0])
    except ValueError:
        return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cli")
    ap.add_argument("data_root")
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--quick", action="store_true", help="fewer formats and models (CI)")
    ap.add_argument("--only", default="", help="comma-separated groups: presets,formats,models")
    a = ap.parse_args()
    corpus = Path(a.corpus)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in corpus.iterdir() if p.is_file() and not p.name.startswith("src-"))
    by_prefix = {p.name.split(" ")[0]: p for p in files}
    live, anim, faces = by_prefix["01"], by_prefix["10"], by_prefix["02"]

    cases = []  # (group, name, input, preset|None, pipeline|None, slow_factor, expected_size_fn)
    presets = ["anime-4k", "anime-2x", "live-1080", "restore-old", "smooth-60", "slowmo-4x", "stabilize", "denoise", "deblur", "faces", "film-grain"]
    for p in presets:
        for src in ([live, anim, faces] if not a.quick else [faces]):
            cases.append(("presets", p, src, f"builtin.{p}", None, 4.0 if p == "slowmo-4x" else 1.0))
    cases.append(("presets", "hdr-to-sdr", by_prefix["27"], "builtin.hdr-to-sdr", None, 1.0))
    fmt_files = files if not a.quick else [by_prefix[k] for k in ("03", "11", "13", "20", "21", "22", "23", "24", "27", "30", "31") if k in by_prefix]
    for f in fmt_files:
        cases.append(("formats", "AI upscale (AnimeVideo v3, 2x)", f, None, {"stages": [stage("enhance", model_id="xinntao.realesr-animevideo-v3", target={"kind": "scale", "factor": 2.0})]}, 1.0))
        if not a.quick or f.name.startswith(("22", "25", "21")):
            cases.append(("formats", "AI interpolation (RIFE, 2x fps)", f, None, {"stages": [stage("interpolate")]}, 1.0))
    models = [
        ("Real-ESRGAN General v3", stage("enhance", model_id="xinntao.realesr-general-v3")),
        ("Real-ESRGAN AnimeVideo v3", stage("enhance", model_id="xinntao.realesr-animevideo-v3")),
        ("Real-ESRGAN x4plus", stage("enhance", model_id="xinntao.realesrgan-x4plus")),
        ("Real-ESRGAN x4plus anime 6B", stage("enhance", model_id="xinntao.realesrgan-x4plus-anime-6b")),
        ("Real-ESRGAN x2plus", stage("enhance", model_id="xinntao.realesrgan-x2plus")),
        ("FFDNet denoise", stage("denoise", engine="ffdnet")),
        ("DRUNet denoise", stage("denoise", engine="drunet")),
        ("NAFNet deblur", stage("deblur")),
        ("RIFE 4.26 interpolation", stage("interpolate")),
        ("GFPGAN face recovery", stage("face_restore")),
        ("Color & light + Sharpen", None),
    ]
    for name, st in models:
        if a.quick and name in ("Real-ESRGAN x4plus", "Real-ESRGAN x4plus anime 6B"):
            continue
        pipe = {"stages": [st]} if st else {"stages": [stage("color", exposure=0.4, temperature=0.3, shadows=0.3, vibrance=0.3), stage("sharpen", amount=0.6)]}
        cases.append(("models", name, faces, None, pipe, 1.0))
    if a.only:
        groups = set(a.only.split(","))
        cases = [c for c in cases if c[0] in groups]

    env = dict(os.environ, CLIPFX_DEVICE=a.device)
    results = []
    for i, (group, name, src, preset, pipe, slow) in enumerate(cases):
        tag = f"{i:03d}"
        cmd = [a.cli, "--data-root", a.data_root, "run", str(src), "-o", str(out), "--name", tag]
        if preset:
            cmd += ["--preset", preset]
        else:
            pf = out / f"{tag}.pipeline.json"
            pf.write_text(json.dumps(pipe))
            cmd += ["--pipeline", str(pf)]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", env=env)
        secs = time.time() - t0
        problems = []
        si = info(src)
        # The app never overwrites: a taken name gets "_2", "_3"... so accept those too.
        outs = sorted(out.glob(f"{tag}.*")) + sorted(out.glob(f"{tag}_*.*"))
        outs = [o for o in outs if not o.name.endswith((".json", ".log"))]
        o = outs[0] if outs else None
        oi = info(o) if o else None
        if r.returncode != 0 or not o:
            problems.append("failed: " + (r.stdout + r.stderr).strip().splitlines()[-1][:300] if (r.stdout + r.stderr).strip() else f"exit {r.returncode}")
        elif not oi or not oi["v"]:
            problems.append("output has no readable video stream")
        else:
            rc, err = decode_errors(o)
            if rc != 0 or err:
                problems.append(f"decode errors: {err}")
            want = si["duration"] * slow
            if want > 0.3 and abs(oi["duration"] - want) > max(0.12, want * 0.04):
                problems.append(f"duration {oi['duration']:.3f}s, expected {want:.3f}s")
            if si["audio"] and not oi["audio"] and slow == 1.0:
                problems.append("audio was lost")
            sw, sh = si["size"]
            ow, oh = oi["size"]
            if (sw > sh) != (ow > oh) and sw != sh:
                problems.append(f"orientation changed: {sw}x{sh} -> {ow}x{oh}")
            if frames(o) < 2 and not src.name.startswith("31"):
                problems.append("fewer than 2 frames")
        (out / f"{tag}.log").write_text(" ".join(cmd) + "\n\n" + r.stdout + r.stderr, encoding="utf-8")
        res = {
            "group": group, "case": name, "input": src.name, "seconds": round(secs, 1), "ok": not problems, "problems": problems,
            "output": o.name if o else None, "out_size": oi["size"] if oi else None, "out_duration": round(oi["duration"], 3) if oi else None,
            "in_size": si["size"], "in_duration": round(si["duration"], 3),
        }
        results.append(res)
        print(f"{'ok  ' if res['ok'] else 'FAIL'} {group:8} {name:34} {src.name:38} {secs:6.1f}s {res['out_size']} {'; '.join(problems)}", flush=True)
    (out / "report.json").write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    lines = ["| Result | Group | Case | Input | Output | Time |", "|---|---|---|---|---|---|"]
    for x in results:
        s = f"{x['out_size'][0]}x{x['out_size'][1]}, {x['out_duration']}s" if x["out_size"] else "-"
        lines.append(f"| {'✅' if x['ok'] else '❌ ' + '; '.join(x['problems'])} | {x['group']} | {x['case']} | {x['input']} | {s} | {x['seconds']}s |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    bad = [x for x in results if not x["ok"]]
    print(f"\n{len(results) - len(bad)}/{len(results)} passed")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
