"""Network interpolation between a fine-tuned candidate and the original General v3 (a fidelity knob without retraining).

weights(a) = a * fine_tuned + (1 - a) * original. a = 1 is the candidate, a = 0 the original. Each blend is scored on the
run's own full evaluation suite and judged by the run's predeclared gates (decide_promotions, against the original
model's baseline). Results go to runs/<run>/blend/step-<n>.json; the best passing blend (lowest balanced score) is
uploaded as runs/<run>/blend/step-<n>-a<alpha>-<sha12>.pth with the same {'params': ...} layout as a candidate.

Usage: python blend_sweep.py --run general-v3-ft2 --step 16000 [--alphas 1,0.9,0.8,0.7,0.6,0.5] [--work DIR]
Runs on a GPU when one is present, otherwise on the CPU (the network is small; the full suite takes minutes).
"""
import argparse, hashlib, io, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import r2util as r2, data_cache as dc, evaluate_sr as ev, sr_common as sc
from train_sr import decide_promotions, summ_metrics


def blend(ft, orig, a):
    return {k: a * ft[k].float() + (1 - a) * orig[k].float() for k in ft}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='general-v3-ft2'); ap.add_argument('--step', type=int, required=True)
    ap.add_argument('--alphas', default='1,0.9,0.8,0.7,0.6,0.5'); ap.add_argument('--work', default='/tmp/clipfx-blend')
    args = ap.parse_args(); P = f'runs/{args.run}/'; os.makedirs(args.work, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    if dev == 'cpu': torch.set_num_threads(os.cpu_count() or 4)
    cfg = json.loads(r2.get(P + 'run_config.json'))['config']; gates = cfg['gates']
    base_full = json.loads(r2.get(P + 'baseline.json'))['full']
    rec = json.loads(r2.get(P + f'candidates/step-{args.step:08d}.json'))
    ft = torch.load(io.BytesIO(r2.get(rec['key'])), map_location='cpu', weights_only=True)['params']
    if hashlib.sha256(r2.get(rec['key'])).hexdigest() != rec['sha256']: raise SystemExit('candidate hash mismatch')
    orig = torch.load(io.BytesIO(r2.get(cfg['init_g'])), map_location='cpu', weights_only=True)['params']
    t0 = time.time(); items = dc.discover(r2, cfg['batches'], args.work)
    E = ev.Evaluator(r2, [it for it in items if it['split'] == 'test'], args.work, dev)
    print(f'setup {time.time() - t0:.0f}s on {dev}', flush=True)
    G = sc.SRVGGNetCompact().to(dev); results = []
    for a in [float(x) for x in args.alphas.split(',')]:
        G.load_state_dict(blend(ft, orig, a)); m = E.run(G, 'full'); sm = summ_metrics(m)
        wins, why = decide_promotions(m, sm, base_full, {}, gates)
        drops = {k: round(m[k]['psnr'] - base_full[k]['psnr'], 3) for k in m if not k.startswith('_') and not k.endswith('/mean')}
        results.append({'alpha': a, 'metrics': sm, 'psnr_delta_by_level': drops, 'slots': wins, **why, 'suite': m})
        print(f"a={a:.2f} still lpips {sm['still_lpips']:.4f} psnr {sm['still_psnr']:.3f} pan {sm['pan_consistency']:.3f} | video lpips "
              f"{sm['video_lpips']:.4f} psnr {sm['video_psnr']:.3f} vT {sm['video_temporal']:.3f} | score {why['score']:.4f} "
              f"gates {why['gates']} worst psnr drop {min(drops.values()):+.3f} slots {wins}", flush=True)
    base_sm = summ_metrics(base_full)
    out = {'run': args.run, 'step': args.step, 'candidate': rec['key'], 'evaluator': ev.EVALUATOR_VERSION, 'baseline': base_sm,
           'gates': gates, 'results': results, 'at': time.time()}
    passing = [r for r in results if 'best_balanced' in r['slots']]
    if passing:
        best = min(passing, key=lambda r: r['score']); buf = io.BytesIO()
        torch.save({'params': {k: v.contiguous() for k, v in blend(ft, orig, best['alpha']).items()}}, buf)
        data = buf.getvalue(); sha = hashlib.sha256(data).hexdigest()
        key = f"{P}blend/step-{args.step:08d}-a{best['alpha']:.2f}-{sha[:12]}.pth"; r2.put(key, data)
        out['best'] = {'alpha': best['alpha'], 'key': key, 'sha256': sha, 'score': best['score'], 'metrics': best['metrics']}
        print('best passing blend', json.dumps(out['best']), flush=True)
    else:
        print('no blend passes every gate', flush=True)
    r2.put(P + f'blend/step-{args.step:08d}.json', json.dumps(out, indent=1).encode())


if __name__ == '__main__':
    main()
