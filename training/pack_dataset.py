"""Build the training cache for a set of batches once (Kaggle CPU) and upload it packed, so GPU sessions skip setup."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2util as r2
import data_cache as dc

def main():
    batches = os.environ['BATCHES'].split(','); work = '/kaggle/tmp' if os.path.isdir('/kaggle/tmp') else '/tmp/clipfx-pack'; os.makedirs(work, exist_ok=True)
    fp, n = dc.fingerprint(r2, batches); print('fingerprint', fp, 'objects', n, flush=True)
    if r2.exists(f'caches/{fp}/manifest.json'): print('already packed', flush=True); return
    items = dc.discover(r2, batches, work); print(json.dumps(dc.summary(items)), flush=True)
    dc.build(r2, items, work, fp); dc.upload_packed(r2, fp, work)

if __name__ == '__main__':
    main()
