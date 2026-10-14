import json, os, random, numpy as np, torch, pytest
import ckpt

class FakeR2:
    """In-memory stand-in for r2util with the calls ckpt uses."""
    BUCKET = 'b'
    def __init__(s): s.store = {}; fake = s
    def put(s, k, d): s.store[k] = bytes(d)
    def put_file(s, k, p): s.store[k] = open(p, 'rb').read()
    def get(s, k): return s.store[k]
    def get_file(s, k, p): open(p, 'wb').write(s.store[k])
    def exists(s, k): return k in s.store
    def ls(s, p): return sorted(k for k in s.store if k.startswith(p))
    def delete(s, k): s.store.pop(k, None)
    @property
    def s3(s):
        store = s.store
        class S3:
            def head_object(self, Bucket, Key): return {'ContentLength': len(store[Key])}
        return S3()

def test_atomic_save_and_publish_and_resume(tmp_path):
    r2 = FakeR2(); p = str(tmp_path / 'ck.pt')
    net = torch.nn.Linear(4, 4); opt = torch.optim.Adam(net.parameters()); scaler = torch.amp.GradScaler(enabled=False)
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    obj = {'net': net.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(), 'rng': ckpt.rng_state(), 'step': 7}
    expect = (random.random(), np.random.rand(), torch.rand(1).item())  # what the run would draw next
    ckpt.save_local(obj, p); assert not os.path.exists(p + '.tmp')
    ptr = ckpt.publish(r2, 'runs/x/', p, 7); assert r2.exists(ptr['key']) and json.loads(r2.get('runs/x/latest.json'))['step'] == 7
    random.seed(0); np.random.seed(0); torch.manual_seed(0)  # a fresh process
    got = ckpt.fetch_latest(r2, 'runs/x/', str(tmp_path / 'r.pt')); ck = torch.load(tmp_path / 'r.pt', weights_only=False)
    ckpt.set_rng_state(ck['rng'])
    assert (random.random(), np.random.rand(), torch.rand(1).item()) == expect and ck['step'] == 7

def test_corrupt_download_is_rejected(tmp_path):
    r2 = FakeR2(); p = str(tmp_path / 'ck.pt'); ckpt.save_local({'a': 1}, p); ptr = ckpt.publish(r2, 'r/', p, 1)
    r2.store[ptr['key']] = r2.store[ptr['key']][:-10]  # truncated object
    with pytest.raises(IOError): ckpt.fetch_latest(r2, 'r/', str(tmp_path / 'x.pt'))

def test_unloadable_file_never_replaces_good_one(tmp_path, monkeypatch):
    p = str(tmp_path / 'ck.pt'); ckpt.save_local({'good': True}, p)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('corrupt')))
    with pytest.raises(RuntimeError): ckpt.save_local({'bad': True}, p)
    monkeypatch.undo(); assert torch.load(p, weights_only=False) == {'good': True}

def test_prune_keeps_newest_and_protected():
    r2 = FakeR2()
    for s in range(1, 7): r2.put(f'r/checkpoints/step-{s:08d}-abc.pt', b'x')
    ckpt.prune(r2, 'r/', protect={'r/checkpoints/step-00000002-abc.pt'})
    assert [k.split('-')[1] for k in r2.ls('r/checkpoints/')] == ['00000002', '00000004', '00000005', '00000006']

def test_background_publisher_skips_while_busy_and_flushes_final(tmp_path):
    import threading, time as _t
    r2 = FakeR2(); gate = threading.Event()
    orig = ckpt.publish
    def slow_publish(*a, **k):
        gate.wait(5); return orig(*a, **k)
    ckpt.publish = slow_publish
    try:
        pub = ckpt.BackgroundPublisher(r2, 'r/')
        p1 = str(tmp_path / 'a.pt'); ckpt.save_local({'s': 1}, p1); assert pub.submit(p1, 1)
        p2 = str(tmp_path / 'b.pt'); ckpt.save_local({'s': 2}, p2); assert not pub.submit(p2, 2)  # busy -> skipped
        assert not os.path.exists(p2) and pub.skipped == 1
        gate.set(); p3 = str(tmp_path / 'c.pt'); ckpt.save_local({'s': 3}, p3); assert pub.submit(p3, 3, wait=True)  # final waits
        assert json.loads(r2.get('r/latest.json'))['step'] == 3 and not os.path.exists(p1) and not os.path.exists(p3)
    finally:
        ckpt.publish = orig
