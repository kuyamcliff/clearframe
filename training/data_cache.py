"""Training data: discovery, split accounting, fingerprinting and a validated local cache.

A cache is reused only when its fingerprint matches: the selected batches, every object's key/size/ETag, the cache
schema and the scale. It is built in a temporary directory and renamed into place only when complete.
"""
import hashlib, json, os, shutil, time, collections
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2

CACHE_SCHEMA = 3; SCALE = 4

def list_meta(r2, prefix):
    out, tok = [], None
    while True:
        r = r2.list_page(prefix, tok) if hasattr(r2, 'list_page') else r2.s3.list_objects_v2(Bucket=r2.BUCKET, Prefix=prefix, **({'ContinuationToken': tok} if tok else {}))
        out += [(o['Key'], o['Size'], o['ETag']) for o in r.get('Contents', [])]
        if not r.get('IsTruncated'): return out
        tok = r['NextContinuationToken']

def fingerprint(r2, batches):
    h = hashlib.sha256(f'schema={CACHE_SCHEMA};scale={SCALE};batches={",".join(batches)}'.encode()); n = 0
    for b in batches:
        for k, s, e in sorted(list_meta(r2, f'data/{b}/')):
            if k.endswith(('.json', '.png', '.jpg')) and '/_' not in k: h.update(f'{k}|{s}|{e}\n'.encode()); n += 1
    return h.hexdigest()[:20], n

def discover(r2, batches, work):
    """Download every meta.json and return items: dict(dir_key, gt_prefix, split, meta, batch)."""
    keys = [k for b in batches for k in [x[0] for x in list_meta(r2, f'data/{b}/')] if k.endswith('/meta.json')]
    def load(k):
        p = f'{work}/{k}'; os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p): r2.get_file(k, p)
        m = json.load(open(p)); pre = k[:-len('meta.json')]
        return {'key': pre, 'gt_prefix': m.get('gt_prefix', pre), 'split': m.get('split', k.split('/')[2]), 'batch': k.split('/')[1], 'meta': m}
    with ThreadPoolExecutor(32) as ex: return list(ex.map(load, keys))

def summary(items):
    t = collections.Counter(); codecs = collections.Counter(); levels = collections.Counter(); pairs = 0
    for it in items:
        kind = 'video' if it['meta'].get('video') else 'photo'; t[(it['batch'], it['split'], kind)] += 1
        for s in it['meta']['sequences']:
            d = s['degradation']; codecs[(d.get('codec') or 'none').split(' ')[0]] += 1
            if 'level' in d: levels[d['level']] += 1
            if it['split'] == 'train': pairs += len(s['offsets']) - 1
    return {'samples': {f'{b}/{s}/{k}': n for (b, s, k), n in sorted(t.items())}, 'codecs': dict(codecs), 'test_levels': dict(levels), 'train_pairs': pairs}

def build(r2, items, work, fp, log=print):
    """Unpack train sequences into raw uint8 arrays (GT region the pan covers, or per-frame GT for video, plus LQ frames)."""
    final = f'{work}/cache-{fp}'
    if os.path.exists(final + '/COMPLETE'):
        log('cache reused', fp); return json.load(open(final + '/index.json')), {'reused': True}
    tmp = final + '.tmp'; shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    t0 = time.time(); train = [it for it in items if it['split'] == 'train']
    need = set()
    for it in train:
        if it['meta'].get('video'): need.update(f"{it['gt_prefix']}gt_{t}.jpg" for t in range(len(it['meta']['sequences'][0]['offsets'])))
        else: need.add(it['gt_prefix'] + 'gt.jpg')
        for si, s in enumerate(it['meta']['sequences']): need.update(f"{it['key']}lq_{si}_{t}.png" for t in range(len(s['offsets'])))
    def dl(k):
        p = f'{work}/{k}'
        if not os.path.exists(p): os.makedirs(os.path.dirname(p), exist_ok=True); r2.get_file(k, p)
        return os.path.getsize(p)
    with ThreadPoolExecutor(32) as ex: nbytes = sum(ex.map(dl, sorted(need)))
    t_dl = time.time() - t0
    index, plan, gt_off, lq_off = [], [], 0, 0
    for it in train:
        m = it['meta']; video = bool(m.get('video'))
        for si, seq in enumerate(m['sequences']):
            c = seq['crop']; n = len(seq['offsets']); l = c // SCALE
            if video: x0 = y0 = 0; w = h = c
            else:
                xs = [o[0] for o in seq['offsets']]; ys = [o[1] for o in seq['offsets']]; x0, y0 = min(xs), min(ys); w, h = max(xs) - x0 + c, max(ys) - y0 + c
            index.append({'gt': gt_off, 'w': w, 'h': h, 'lq': lq_off, 'l': l, 'n': n, 'video': video,
                          'offs': [[0, 0]] * n if video else [[x - x0, y - y0] for x, y in seq['offsets']]})
            plan.append((it, si, x0, y0)); gt_off += w * h * 3 * (n if video else 1); lq_off += n * l * l * 3
    gtm = np.lib.format.open_memmap(f'{tmp}/gt.npy', 'w+', np.uint8, (max(gt_off, 1),)); lqm = np.lib.format.open_memmap(f'{tmp}/lq.npy', 'w+', np.uint8, (max(lq_off, 1),))
    def fill(k):
        (it, si, x0, y0), e = plan[k], index[k]; sz = e['w'] * e['h'] * 3
        if e['video']:
            for t in range(e['n']): gtm[e['gt'] + t * sz: e['gt'] + (t + 1) * sz] = cv2.imread(f"{work}/{it['gt_prefix']}gt_{t}.jpg").reshape(-1)
        else:
            g = cv2.imread(f"{work}/{it['gt_prefix']}gt.jpg"); gtm[e['gt']: e['gt'] + sz] = np.ascontiguousarray(g[y0:y0 + e['h'], x0:x0 + e['w']]).reshape(-1)
        for t in range(e['n']):
            f = cv2.imread(f"{work}/{it['key']}lq_{si}_{t}.png"); o = e['lq'] + t * e['l'] ** 2 * 3; lqm[o:o + f.size] = f.reshape(-1)
    with ThreadPoolExecutor(4) as ex: list(ex.map(fill, range(len(plan))))
    gtm.flush(); lqm.flush()
    json.dump(index, open(tmp + '/index.json', 'w')); open(tmp + '/COMPLETE', 'w').write(fp)
    shutil.rmtree(final, ignore_errors=True); os.rename(tmp, final)
    stats = {'reused': False, 'objects': len(need), 'download_gb': round(nbytes / 1e9, 2), 'download_s': round(t_dl), 'build_s': round(time.time() - t0 - t_dl),
             'cache_gb': round((gt_off + lq_off) / 1e9, 2), 'sequences': len(index)}
    log('cache built', json.dumps(stats)); return index, stats
