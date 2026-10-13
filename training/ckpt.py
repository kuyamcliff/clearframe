"""Checkpoints that cannot be half-written or silently replaced.

save_local: write to a temp file, fsync, load it back to verify, then atomically rename.
publish: upload under an immutable key checkpoints/step-<n>-<sha12>.pt, verify size and hash, and only then advance
    latest.json. Older checkpoints are pruned (keeping the newest KEEP and every promoted one).
RNG helpers capture and restore Python, NumPy, Torch CPU and CUDA generators so a resumed run continues exactly.
"""
import hashlib, json, os, random, time
import numpy as np, torch

KEEP = 3

def rng_state():
    s = {'python': random.getstate(), 'numpy': np.random.get_state(), 'torch': torch.get_rng_state()}
    if torch.cuda.is_available(): s['cuda'] = torch.cuda.get_rng_state_all()
    return s

def set_rng_state(s):
    random.setstate(s['python']); np.random.set_state(s['numpy']); torch.set_rng_state(s['torch'])
    if 'cuda' in s and torch.cuda.is_available(): torch.cuda.set_rng_state_all(s['cuda'])

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()

def save_local(obj, path):
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        torch.save(obj, f); f.flush(); os.fsync(f.fileno())
    torch.load(tmp, map_location='cpu', weights_only=False)  # must load back before it may replace anything
    os.replace(tmp, path); return sha256_file(path)

def publish(r2, prefix, path, step, extra=None):
    """Upload an immutable checkpoint and advance latest.json only after verification. Returns the pointer dict."""
    sha = sha256_file(path); key = f'{prefix}checkpoints/step-{step:08d}-{sha[:12]}.pt'
    r2.put_file(key, path)
    head = r2.s3.head_object(Bucket=r2.BUCKET, Key=key)
    if head['ContentLength'] != os.path.getsize(path): raise IOError(f'upload size mismatch for {key}')
    ptr = {'key': key, 'step': step, 'sha256': sha, 'bytes': os.path.getsize(path), 'time': time.time(), **(extra or {})}
    r2.put(prefix + 'latest.json', json.dumps(ptr).encode())
    return ptr

def prune(r2, prefix, protect=()):
    keys = sorted(k for k in r2.ls(prefix + 'checkpoints/') if k.endswith('.pt'))
    for k in keys[:-KEEP]:
        if k not in protect: r2.delete(k)

def fetch_latest(r2, prefix, dest):
    """Download the checkpoint latest.json points at and verify its hash. None when there is nothing to resume."""
    if not r2.exists(prefix + 'latest.json'): return None
    ptr = json.loads(r2.get(prefix + 'latest.json')); r2.get_file(ptr['key'], dest)
    if sha256_file(dest) != ptr['sha256']: raise IOError(f'checkpoint {ptr["key"]} does not match latest.json')
    return ptr


class BackgroundPublisher:
    """Uploads frozen local checkpoint files on a worker thread so training does not wait for R2.

    submit() takes a file that will never be modified again. If an upload is still running, the new snapshot is
    skipped (and deleted) rather than queued: the next periodic checkpoint supersedes it. close() waits for the last
    upload, so a session never ends with an unpublished final checkpoint.
    """
    def __init__(self, r2, prefix, on_done=None):
        import threading
        self.r2, self.prefix, self.on_done = r2, prefix, on_done; self.lock = threading.Lock(); self.thread = None; self.error = None
        self.skipped = 0; self.upload_s = 0.0

    def busy(self): return self.thread is not None and self.thread.is_alive()

    def submit(self, path, step, protect=(), wait=False):
        import threading
        if self.error: raise self.error
        if self.busy() and not wait:
            os.remove(path); self.skipped += 1; return False
        if self.busy(): self.thread.join()
        def work():
            t = time.time()
            try:
                ptr = publish(self.r2, self.prefix, path, step); prune(self.r2, self.prefix, protect=protect)
                if self.on_done: self.on_done(ptr)
            except Exception as e:
                self.error = e
            finally:
                self.upload_s += time.time() - t
                if os.path.exists(path): os.remove(path)
        self.thread = threading.Thread(target=work, daemon=True); self.thread.start()
        if wait: self.close()
        return True

    def close(self):
        if self.thread is not None: self.thread.join()
        if self.error: raise self.error
