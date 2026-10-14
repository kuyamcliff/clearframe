"""Pre-release checks for a General v3 candidate (PRD Phase 4): Rust/Candle parity and the denoise slider.

  python app_check.py <candidate key in R2 or local .pth> [--slider]

1. Parity: the candidate's 'params' are written as realesr-general-x4v3.safetensors (the app's key names); a PyTorch
   reference output is made with sr_common.SRVGGNetCompact; the app's own example (crates/inference/examples/parity.rs)
   must match within its 1e-3 tolerance. The original weights are run the same way as a control.
2. Slider (--slider): the app blends weights as  w = d * General + (1 - d) * WDN  (processors.rs). For d in
   0, .25, .5, .75, 1 the candidate+original-WDN pair is scored on the held-out suite against the original pair.
"""
import io, json, os, subprocess, sys, tempfile
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import r2util as r2, sr_common as sc

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def params(src):
    b = r2.get(src) if not os.path.exists(src) else open(src, 'rb').read()
    s = torch.load(io.BytesIO(b), map_location='cpu', weights_only=True); return s.get('params', s)

def parity(p, label):
    from safetensors.torch import save_file
    d = tempfile.mkdtemp(prefix='parity-'); refs, st = f'{d}/refs', f'{d}/st'; os.makedirs(refs); os.makedirs(st)
    save_file({k: v.contiguous().float() for k, v in p.items()}, f'{st}/realesr-general-x4v3.safetensors')
    m = sc.SRVGGNetCompact(); m.load_state_dict(p); m.eval()
    rng = np.random.default_rng(1234); base = torch.from_numpy(rng.random((1, 3, 13, 17)).astype(np.float32))
    x = (torch.nn.functional.interpolate(base, size=(48, 64), mode='bilinear', align_corners=False) + 0.05 * torch.from_numpy(rng.standard_normal((1, 3, 48, 64)).astype(np.float32))).clamp(0, 1)
    with torch.no_grad(): y = m(x)
    x.numpy().astype('<f4').tofile(f'{refs}/general_x4v3.in0.bin'); y.numpy().astype('<f4').tofile(f'{refs}/general_x4v3.out.bin')
    json.dump({'inputs': [list(x.shape)], 'output_shape': list(y.shape), 'meta': {}}, open(f'{refs}/general_x4v3.json', 'w'))
    r = subprocess.run(['cargo', 'run', '-q', '--release', '-p', 'clearframe-inference', '--example', 'parity', '--', refs, st, 'general_x4v3'],
                       cwd=REPO, capture_output=True, text=True)
    line = next((l for l in r.stdout.splitlines() if l.startswith('general_x4v3')), r.stdout.strip() + r.stderr.strip()[-400:])
    ok = r.returncode == 0 and line.rstrip().endswith('OK'); print(f'parity [{label}]', line); return {'ok': ok, 'line': line}

def blend(a, b, d): return {k: d * a[k] + (1 - d) * b[k] for k in a}

def main():
    cand_src = sys.argv[1]; slider = '--slider' in sys.argv
    gen0 = params('originals/realesr-general-x4v3.pth'); wdn = params('originals/realesr-general-wdn-x4v3.pth'); cand = params(cand_src)
    assert set(cand) == set(gen0) and all(cand[k].shape == gen0[k].shape for k in gen0), 'candidate keys/shapes differ from General v3'
    rep = {'candidate': cand_src, 'parity': {'original': parity(gen0, 'original'), 'candidate': parity(cand, 'candidate')}}
    if slider:
        import data_cache as dc, evaluate_sr as ev, train_sr
        work = os.environ.get('WORK', tempfile.mkdtemp(prefix='appcheck-')); batches = os.environ.get('BATCHES', 'b001-v2,v001-v2').split(',')
        items = [it for it in dc.discover(r2, batches, work) if it['split'] == 'test']
        E = ev.Evaluator(r2, items, work, 'cuda' if torch.cuda.is_available() else 'cpu')
        G = sc.SRVGGNetCompact().to(E.dev); rep['slider'] = {}
        for d in (0.0, 0.25, 0.5, 0.75, 1.0):
            row = {}
            for name, gen in (('original', gen0), ('candidate', cand)):
                G.load_state_dict(blend(gen, wdn, d)); row[name] = train_sr.summ_metrics(E.run(G, os.environ.get('SUITE', 'full')))
            rep['slider'][str(d)] = row; print('slider', d, json.dumps(row))
    out = json.dumps(rep, indent=1, default=str); print(out[:2000])
    if not os.path.exists(cand_src): r2.put(cand_src.rsplit('.', 1)[0] + '.app_check.json', out.encode())

if __name__ == '__main__':
    main()
