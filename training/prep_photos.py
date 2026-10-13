"""Batch data prep (Kaggle CPU): lawful high-res photos -> clean reference + realistically degraded pan sequences.

Each sample in R2 under data/<batch>/<id>/:
  gt.jpg      reference (original downscaled 2x with area filter, JPEG q97 4:4:4)
  lq_XX.png   degraded frames (1/4 size of the crop), produced through a real codec
  meta.json   crop offsets per frame, degradation used, source URL, author, licence
"""
import json, os, random, re, subprocess, sys, time, urllib.parse, urllib.request, io, traceback
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2util as r2

UA = 'ClipFXTraining/0.1 (support@clipfx.me)'
API = 'https://commons.wikimedia.org/w/api.php'
OK_LICENSE = re.compile(r'^(cc0|cc-zero|public domain|pd\b|cc by(-sa)? ?[1-4]\.0|cc-by(-sa)?-[1-4]\.0)', re.I)
BATCH = os.environ.get('BATCH', 'b001'); TARGET = int(os.environ.get('TARGET', '2000')); HOLDOUT = int(os.environ.get('HOLDOUT', '60'))
CROP, SCALE, FRAMES = 768, 4, 6
T_END = time.time() + float(os.environ.get('HOURS', '11')) * 3600

def http(url, tries=6):
    for i in range(tries):
        try: return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': UA}), timeout=120).read()
        except Exception as e:
            if i == tries - 1: raise
            time.sleep(min(60, 3 * 2 ** i))

def manifest(limit):
    key = f'data/{BATCH}/_manifest.json'
    if r2.exists(key): return json.loads(r2.get(key))
    out, seen = [], set()
    for cat in ['Category:Featured pictures on Wikimedia Commons', 'Category:Quality images']:
        cont = {}
        while len(out) < limit:
            q = {'action': 'query', 'format': 'json', 'generator': 'categorymembers', 'gcmtitle': cat, 'gcmtype': 'file', 'gcmlimit': '200',
                 'prop': 'imageinfo', 'iiprop': 'url|size|mime|extmetadata', 'iiextmetadatafilter': 'LicenseShortName|Artist|LicenseUrl', **cont}
            d = json.loads(http(API + '?' + urllib.parse.urlencode(q)))
            for p in (d.get('query') or {}).get('pages', {}).values():
                ii = (p.get('imageinfo') or [{}])[0]; m = ii.get('extmetadata', {})
                lic = m.get('LicenseShortName', {}).get('value', '').strip()
                if ii.get('mime') != 'image/jpeg' or ii.get('width', 0) < 3000 or ii.get('height', 0) < 2000 or not OK_LICENSE.match(lic) or ii['url'] in seen: continue
                seen.add(ii['url'])
                out.append({'url': ii['url'], 'w': ii['width'], 'h': ii['height'], 'bytes': ii.get('size'), 'license': lic,
                            'license_url': m.get('LicenseUrl', {}).get('value', ''), 'author': re.sub('<[^>]+>', '', m.get('Artist', {}).get('value', ''))[:300],
                            'page': 'https://commons.wikimedia.org/wiki/' + urllib.parse.quote(p['title'].replace(' ', '_')), 'source': 'wikimedia-commons'})
            if 'continue' not in d: break
            cont = d['continue']; time.sleep(0.5)
        print(cat, len(out), flush=True)
    random.Random(1).shuffle(out)
    r2.put(key, json.dumps(out).encode()); return out

# ---------------- degradations ----------------
def kernel(rng):
    k = rng.choice([7, 9, 11, 13, 15, 17, 21]); s1, s2 = rng.uniform(0.2, 3.0), rng.uniform(0.2, 3.0)
    if rng.random() < 0.5: s2 = s1
    th = rng.uniform(0, np.pi); ax = np.arange(k) - k // 2; xx, yy = np.meshgrid(ax, ax)
    c, s = np.cos(th), np.sin(th); x, y = c * xx + s * yy, -s * xx + c * yy
    g = np.exp(-0.5 * (x ** 2 / s1 ** 2 + y ** 2 / s2 ** 2)); return (g / g.sum()).astype(np.float32)

def noise(img, rng, scale):
    if rng.random() < 0.4: return img
    if rng.random() < 0.6:
        sig = rng.uniform(1, 25) * scale
        n = rng.normal(0, sig, img.shape[:2] + (1,)) if rng.random() < 0.4 else rng.normal(0, sig, img.shape)
        return img + n
    v = 10 ** rng.uniform(1.5, 3.2); return rng.poisson(np.clip(img, 0, 255) / 255 * v) / v * 255

def degrade_spatial(frames, rng, cfg):
    """Second-order degradation (Real-ESRGAN style), applied identically per sequence."""
    out = []
    k1 = kernel(rng); k2 = kernel(rng) if rng.random() < 0.5 else None
    interp = [cv2.INTER_AREA, cv2.INTER_LINEAR, cv2.INTER_CUBIC]
    r1 = rng.uniform(0.3, 1.2); i1, i2 = rng.choice(interp), rng.choice(interp)
    nsd = rng.integers(1 << 30)
    for f in frames:
        lr = rng  # same random draw pattern per frame for temporal consistency of noise level
        x = cv2.filter2D(f.astype(np.float32), -1, k1)
        h, w = x.shape[:2]
        if k2 is not None:
            x = cv2.resize(x, (int(w * r1), int(h * r1)), interpolation=i1)
            x = noise(x, np.random.default_rng(nsd), 1.0); x = cv2.filter2D(np.clip(x, 0, 255), -1, k2)
        x = cv2.resize(x, (w // SCALE, h // SCALE), interpolation=i2)
        x = noise(x, np.random.default_rng(nsd + 1), 0.7 if k2 is not None else 1.0)
        out.append(np.clip(x, 0, 255).round().astype(np.uint8))
    cfg.update(blur=True, second_order=k2 is not None)
    return out

def codec(frames, rng, cfg):
    h, w = frames[0].shape[:2]; c = rng.random()
    if c < 0.15:
        cfg['codec'] = 'none'; return frames
    if c < 0.35:
        q = int(rng.integers(30, 95)); cfg['codec'] = f'jpeg q{q}'
        return [cv2.imdecode(cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, q])[1], 1) for f in frames]
    enc = 'libx264' if c < 0.75 else 'libx265'; crf = int(rng.integers(18, 42)); preset = rng.choice(['veryfast', 'medium', 'slow'])
    cfg['codec'] = f'{enc} crf{crf} {preset}'
    extra = ['-x265-params', 'log-level=error'] if enc == 'libx265' else []
    p = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{w}x{h}', '-r', '30', '-i', '-',
                        '-c:v', enc, '-crf', str(crf), '-preset', preset, '-pix_fmt', 'yuv420p', *extra, '-f', 'matroska', '-'],
                       input=b''.join(f.tobytes() for f in frames), capture_output=True, check=True)
    d = subprocess.run(['ffmpeg', '-v', 'error', '-i', '-', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], input=p.stdout, capture_output=True, check=True)
    a = np.frombuffer(d.stdout, np.uint8).reshape(-1, h, w, 3); return [a[i] for i in range(len(frames))]

def pan(gt, rng):
    H, W = gt.shape[:2]; c = min(CROP, H, W) // 16 * 16
    span = int(rng.uniform(4, 20)) * (FRAMES - 1)
    dx, dy = rng.choice([-1, 0, 1]), rng.choice([-1, 0, 1])
    if dx == dy == 0: dx = 1
    x0 = int(rng.integers(span, W - c - span)); y0 = int(rng.integers(span, H - c - span))
    offs = [(x0 + dx * span * t // (FRAMES - 1), y0 + dy * span * t // (FRAMES - 1)) for t in range(FRAMES)]
    return c, offs

def process(i, item, split):
    sid = f'{i:05d}'; pre = f'data/{BATCH}/{split}/{sid}/'
    if r2.exists(pre + 'meta.json'): return 'skip'
    raw = http(item['url'])
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
    if img is None: return 'bad'
    gt = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2), interpolation=cv2.INTER_AREA)
    if min(gt.shape[:2]) < CROP + 200: return 'small'
    rng = np.random.default_rng(1000 + i)
    seqs = []
    for s in range(2 if split == 'train' else 3):
        c, offs = pan(gt, rng)
        frames = [gt[y:y + c, x:x + c] for x, y in offs]
        cfg = {}
        if split == 'test':  # fixed severities for evaluation: light, medium, heavy
            cfg['level'] = ['light', 'medium', 'heavy'][s]
            sub = np.random.default_rng([7, 8, 9][s] * 10000 + i)
            lq = degrade_spatial(frames, sub, cfg) if s else [cv2.resize(f, (c // SCALE, c // SCALE), interpolation=cv2.INTER_AREA) for f in frames]
            crf = [23, 32, 40][s]; p = subprocess.run(['ffmpeg', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{c//SCALE}x{c//SCALE}', '-r', '30', '-i', '-', '-c:v', 'libx264', '-crf', str(crf), '-preset', 'medium', '-pix_fmt', 'yuv420p', '-f', 'matroska', '-'], input=b''.join(f.tobytes() for f in lq), capture_output=True, check=True)
            d = subprocess.run(['ffmpeg', '-v', 'error', '-i', '-', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'], input=p.stdout, capture_output=True, check=True)
            a = np.frombuffer(d.stdout, np.uint8).reshape(-1, c // SCALE, c // SCALE, 3); lq = [a[k] for k in range(FRAMES)]; cfg['codec'] = f'libx264 crf{crf}'
        else:
            lq = codec(degrade_spatial(frames, rng, cfg), rng, cfg)
        for t, f in enumerate(lq): r2.put(pre + f'lq_{s}_{t}.png', cv2.imencode('.png', f)[1].tobytes())
        seqs.append({'crop': c, 'offsets': offs, 'degradation': cfg})
    r2.put(pre + 'gt.jpg', cv2.imencode('.jpg', gt, [cv2.IMWRITE_JPEG_QUALITY, 97, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])[1].tobytes())
    r2.put(pre + 'meta.json', json.dumps({'source': item, 'gt_size': gt.shape[:2][::-1], 'scale': SCALE, 'sequences': seqs}, default=int).encode())
    return 'ok'

def main():
    items = manifest(TARGET + HOLDOUT + 400)
    print('manifest', len(items), flush=True)
    from concurrent.futures import ThreadPoolExecutor, FIRST_COMPLETED, wait
    jobs = iter((i, it, 'test' if i < HOLDOUT else 'train') for i, it in enumerate(items))
    stats, ok, n = {}, 0, 0
    def run(j):
        try: return process(*j)
        except Exception as e: return 'err:' + type(e).__name__ + ':' + str(e)[:80]
    with ThreadPoolExecutor(6) as ex:
        pending = {ex.submit(run, j) for j in [next(jobs, None) for _ in range(12)] if j}
        while pending:
            fin, pending = wait(pending, return_when=FIRST_COMPLETED)
            for f in fin:
                res = f.result(); k = res.split(':')[0]; stats[k] = stats.get(k, 0) + 1; n += 1
                if k == 'err' and stats[k] <= 5: print(res, flush=True)
                if k in ('ok', 'skip'): ok += 1
                if n % 50 == 0:
                    print(time.strftime('%H:%M'), n, stats, flush=True)
                    r2.put(f'data/{BATCH}/_progress.json', json.dumps({'processed': n, 'stats': stats, 'time': time.time()}).encode())
            if ok < TARGET + HOLDOUT and time.time() < T_END:
                for _ in fin:
                    j = next(jobs, None)
                    if j: pending.add(ex.submit(run, j))
    r2.put(f'data/{BATCH}/_progress.json', json.dumps({'processed': n, 'stats': stats, 'ok': ok, 'time': time.time(), 'finished': True}).encode())
    print('done', ok, stats, flush=True)

if __name__ == '__main__':
    main()
