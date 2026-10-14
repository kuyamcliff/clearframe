"""Data schema v2 (Kaggle CPU): re-make every degraded frame from the existing clean references.

Fixes v1's frozen noise (the same noise field on every frame of a sequence) and adds ringing. Clean references are
not copied: each v2 meta.json points at them with 'gt_prefix'. Seeds come from SHA-256 of the sample path, so a
re-run reproduces the same data. Video clips from the last 20% of each film become a held-out real-video test set
with fixed light/medium/heavy damage; they are never trained on.

  data/<SRC>-v2/<split>/<id>/lq_<s>_<t>.png + meta.json
"""
import hashlib, json, os, subprocess, sys, time
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2util as r2
from prep_photos import degrade_spatial, codec, SCALE

SRC = os.environ.get('SRC', 'b001'); DST = SRC + '-v2'; HOURS = float(os.environ.get('HOURS', '11.5'))
LEVELS = [('light', 23), ('medium', 32), ('heavy', 40)]; T_END = time.time() + HOURS * 3600

def seed(*parts): return int(hashlib.sha256('/'.join(map(str, parts)).encode()).hexdigest()[:15], 16)

def x264(frames, crf):
    h, w = frames[0].shape[:2]
    p = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', '30', '-i', '-', '-c:v', 'libx264', '-crf', str(crf),
                        '-preset', 'medium', '-pix_fmt', 'yuv420p', '-f', 'matroska', '-'], input=b''.join(f.tobytes() for f in frames), capture_output=True, check=True)
    d = subprocess.run(['ffmpeg', '-v', 'error', '-i', '-', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], input=p.stdout, capture_output=True, check=True)
    a = np.frombuffer(d.stdout, np.uint8).reshape(-1, h, w, 3); return [a[i] for i in range(len(frames))]

def fixed_level(frames, li, sd):
    """Held-out damage: light = clean 4x area downscale; medium/heavy = full v2 degradation with a fixed seed. Then x264."""
    name, crf = LEVELS[li]; cfg = {'level': name}
    if li == 0:
        c = frames[0].shape[0]; lq = [cv2.resize(f, (c // SCALE, c // SCALE), interpolation=cv2.INTER_AREA) for f in frames]; cfg['schema'] = 2
    else:
        lq = degrade_spatial(frames, np.random.default_rng(sd), cfg)
    cfg['codec'] = f'libx264 crf{crf}'; return x264(lq, crf), cfg

def gt_frames(src_pre, meta, seq):
    if meta.get('video'): return [cv2.imdecode(np.frombuffer(r2.get(src_pre + f'gt_{t}.jpg'), np.uint8), 1) for t in range(len(seq['offsets']))]
    g = cv2.imdecode(np.frombuffer(r2.get(src_pre + 'gt.jpg'), np.uint8), 1); c = seq['crop']
    return [g[y:y + c, x:x + c] for x, y in seq['offsets']]

def video_cuts(metas):
    by = {}
    for _, m in metas: by.setdefault(m['source']['film'], []).append(m['source']['time'])
    return {f: min(ts) + 0.8 * (max(ts) - min(ts)) for f, ts in by.items()}

def one(key, meta, cuts):
    src_pre = key[:-len('meta.json')]; _, _, split, sid, _ = key.split('/')
    if meta.get('video') and meta['source']['time'] >= cuts[meta['source']['film']]: split = 'test'
    dst = f'data/{DST}/{split}/{sid}/'
    if r2.exists(dst + 'meta.json'): return 'skip'
    seqs = []
    if split == 'test':
        base = meta['sequences'][0] if meta.get('video') else None
        for li in range(3):
            seq = base or meta['sequences'][li]; fr = gt_frames(src_pre, meta, seq)
            lq, cfg = fixed_level(fr, li, seed(SRC, sid, 'test', li))
            for t, f in enumerate(lq): r2.put(dst + f'lq_{li}_{t}.png', cv2.imencode('.png', f)[1].tobytes())
            seqs.append({'crop': seq['crop'], 'offsets': seq['offsets'], 'degradation': cfg})
    else:
        for si, seq in enumerate(meta['sequences']):
            fr = gt_frames(src_pre, meta, seq); rng = np.random.default_rng(seed(SRC, sid, si)); cfg = {}
            lq = codec(degrade_spatial(fr, rng, cfg), rng, cfg)
            for t, f in enumerate(lq): r2.put(dst + f'lq_{si}_{t}.png', cv2.imencode('.png', f)[1].tobytes())
            seqs.append({'crop': seq['crop'], 'offsets': seq['offsets'], 'degradation': cfg})
    out = dict(meta); out.update(sequences=seqs, gt_prefix=src_pre, schema=2, split=split)
    r2.put(dst + 'meta.json', json.dumps(out, default=int).encode()); return 'ok'

def main():
    keys = sorted(k for k in r2.ls(f'data/{SRC}/') if k.endswith('/meta.json'))
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(16) as ex: metas = list(zip(keys, ex.map(lambda k: json.loads(r2.get(k)), keys)))
    cuts = video_cuts([x for x in metas if x[1].get('video')]) if any(m.get('video') for _, m in metas) else {}
    print(SRC, '->', DST, len(metas), 'samples; video cut times', cuts, flush=True)
    stats = {}; n = 0
    def run(x):
        try: return one(*x, cuts)
        except Exception as e: return f'err:{type(e).__name__}:{str(e)[:80]}'
    with ThreadPoolExecutor(6) as ex:
        for res in ex.map(run, metas):
            k = res.split(':')[0]; stats[k] = stats.get(k, 0) + 1; n += 1
            if k == 'err' and stats[k] <= 5: print(res, flush=True)
            if n % 200 == 0:
                print(time.strftime('%H:%M'), n, stats, flush=True); r2.put(f'data/{DST}/_progress.json', json.dumps({'done': n, 'of': len(metas), 'stats': stats}).encode())
            if time.time() > T_END: break
    r2.put(f'data/{DST}/_progress.json', json.dumps({'done': n, 'of': len(metas), 'stats': stats, 'finished': n == len(metas)}).encode())
    print('done', n, stats, flush=True)

if __name__ == '__main__':
    main()
