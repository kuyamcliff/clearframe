import json, os, numpy as np, cv2, pytest
import data_cache as dc

class R2:
    """Minimal in-memory R2 with ETags."""
    BUCKET = 'b'
    def __init__(s): s.o = {}
    def put(s, k, b): s.o[k] = bytes(b)
    def get_file(s, k, p): open(p, 'wb').write(s.o[k])
    @property
    def s3(s):
        o = s.o
        class S3:
            def list_objects_v2(self, Bucket, Prefix, **kw):
                import hashlib
                return {'Contents': [{'Key': k, 'Size': len(v), 'ETag': hashlib.md5(v).hexdigest()} for k, v in sorted(o.items()) if k.startswith(Prefix)], 'IsTruncated': False}
        return S3()

def make(r2, batch, sid, video=False, gt_prefix=None, split='train', color=(10, 200, 30)):
    pre = f'data/{batch}/{split}/{sid}/'; c = 64
    seq = {'crop': c, 'offsets': [[0, 0]] * 3 if video else [[4 * t, 0] for t in range(3)], 'degradation': {'codec': 'libx264 crf30'}}
    m = {'video': video, 'sequences': [seq], 'source': {}}
    if gt_prefix: m['gt_prefix'] = gt_prefix
    r2.put(pre + 'meta.json', json.dumps(m).encode())
    img = np.zeros((c, c + 16, 3), np.uint8); img[:] = color  # BGR on disk
    if video:
        for t in range(3): r2.put((gt_prefix or pre) + f'gt_{t}.jpg', cv2.imencode('.png', img[:, :c])[1].tobytes())
    else:
        r2.put((gt_prefix or pre) + 'gt.jpg', cv2.imencode('.png', img)[1].tobytes())
    for t in range(3): r2.put(pre + f'lq_0_{t}.png', cv2.imencode('.png', img[:16, :16])[1].tobytes())

def test_fingerprint_changes_with_batches_and_content(tmp_path):
    r2 = R2(); make(r2, 'b1', '0'); make(r2, 'b2', '0')
    a, _ = dc.fingerprint(r2, ['b1']); b, _ = dc.fingerprint(r2, ['b1', 'b2']); assert a != b
    make(r2, 'b1', '0', color=(1, 2, 3)); c, _ = dc.fingerprint(r2, ['b1']); assert c != a

def test_cache_built_once_reused_and_gt_prefix_followed(tmp_path):
    r2 = R2(); make(r2, 'b1', '0'); make(r2, 'b1-v2', '0', gt_prefix='data/b1/train/0/'); make(r2, 'v1', '5', video=True)
    items = dc.discover(r2, ['b1-v2', 'v1'], str(tmp_path)); fp, _ = dc.fingerprint(r2, ['b1-v2', 'v1'])
    s = dc.summary(items); assert s['samples'] == {'b1-v2/train/photo': 1, 'v1/train/video': 1} and s['train_pairs'] == 4
    idx, st = dc.build(r2, items, str(tmp_path), fp, log=lambda *a: None); assert not st['reused'] and len(idx) == 2
    _, st2 = dc.build(r2, items, str(tmp_path), fp, log=lambda *a: None); assert st2['reused']
    gt = np.load(f'{tmp_path}/cache-{fp}/gt.npy'); assert gt[:3].tolist() == [10, 200, 30]  # BGR bytes, from the referenced b1 GT
    assert idx[1]['video'] and idx[1]['offs'] == [[0, 0]] * 3

def test_incomplete_cache_is_rebuilt(tmp_path):
    r2 = R2(); make(r2, 'b1', '0'); items = dc.discover(r2, ['b1'], str(tmp_path)); fp, _ = dc.fingerprint(r2, ['b1'])
    os.makedirs(f'{tmp_path}/cache-{fp}'); open(f'{tmp_path}/cache-{fp}/index.json', 'w').write('[]')  # no COMPLETE marker
    idx, st = dc.build(r2, items, str(tmp_path), fp, log=lambda *a: None); assert not st['reused'] and len(idx) == 1

def test_packed_roundtrip_and_corruption(tmp_path, monkeypatch):
    r2 = R2(); make(r2, 'b1', '0'); make(r2, 'b1', '1', color=(5, 6, 7))
    r2.exists = lambda k: k in r2.o; r2.get = lambda k: r2.o[k]
    monkeypatch.setattr(dc, 'PART_BYTES', 1000)  # force several parts
    items = dc.discover(r2, ['b1'], str(tmp_path / 'a')); fp, _ = dc.fingerprint(r2, ['b1'])
    idx, _ = dc.build(r2, items, str(tmp_path / 'a'), fp, log=lambda *a: None); dc.upload_packed(r2, fp, str(tmp_path / 'a'), log=lambda *a: None)
    idx2, st = dc.fetch_packed(r2, fp, str(tmp_path / 'b'), log=lambda *a: None)
    assert idx2 == idx and st['packed']
    for n in ('gt.npy', 'lq.npy'):
        assert open(tmp_path / f'a/cache-{fp}/{n}', 'rb').read() == open(tmp_path / f'b/cache-{fp}/{n}', 'rb').read()
    part = json.loads(r2.o[f'caches/{fp}/manifest.json'])['files']['gt.npy']['parts'][1]['key']; r2.o[part] = b'x' + r2.o[part][1:]
    with pytest.raises(IOError): dc.fetch_packed(r2, fp, str(tmp_path / 'c'), log=lambda *a: None)
    assert dc.fetch_packed(r2, 'nope', str(tmp_path / 'd'), log=lambda *a: None) is None
