"""Deterministic quick/full evaluation for x4 restoration models.

Suites (held-out only, never trained on):
  still: photo pans with fixed light/medium/heavy damage -> PSNR (Y), LPIPS-Alex v0.1, synthetic_pan_consistency_l1
  video: real-motion film clips (last 20% of each film) with fixed damage -> PSNR, LPIPS, video_temporal_residual (+coverage)
Quick: 16 still + 16 video samples at the medium level. Full: every still sample, up to 96 video samples, all levels.
All test data and the reference-frame flows are loaded once; evaluations then only run the model.
"""
import json, math, os, time
from concurrent.futures import ThreadPoolExecutor
import numpy as np, cv2, torch, torch.nn.functional as F
import metrics_temporal as mt

EVALUATOR_VERSION = 'sr-eval-v1+' + mt.EVAL_VERSION
LEVELS = ['light', 'medium', 'heavy']

def even(xs, n): return xs if len(xs) <= n else [xs[round(i * (len(xs) - 1) / (n - 1))] for i in range(n)]
def y_luma(t): return 16 + 65.481 * t[:, 0] + 128.553 * t[:, 1] + 24.966 * t[:, 2]
def psnr_y(a, b, border=8):
    mse = F.mse_loss(y_luma(a)[..., border:-border, border:-border], y_luma(b)[..., border:-border, border:-border]).item()
    return 10 * math.log10(255 ** 2 / max(mse, 1e-8))

class Evaluator:
    def __init__(self, r2, items, work, device, quick_n=16, full_video_n=96, log=print):
        self.dev = device; self.log = log; t0 = time.time()
        test = sorted([it for it in items if it['split'] == 'test'], key=lambda it: it['key'])
        still = [it for it in test if not it['meta'].get('video')]; video = [it for it in test if it['meta'].get('video')]
        self.sel = {'quick': {'still': even(still, quick_n), 'video': even(video, quick_n), 'levels': [1]},
                    'full': {'still': still, 'video': even(video, full_video_n), 'levels': [0, 1, 2]}}
        use = {it['key']: it for s in self.sel.values() for k in ('still', 'video') for it in s[k]}
        def need(it):
            m = it['meta']; ks = []
            if m.get('video'): ks += [f"{it['gt_prefix']}gt_{t}.jpg" for t in (1, 2, 3)]
            else: ks.append(it['gt_prefix'] + 'gt.jpg')
            ks += [f"{it['key']}lq_{l}_{t}.png" for l in range(3) for t in (1, 2, 3)]; return ks
        keys = sorted({k for it in use.values() for k in need(it)})
        def dl(k):
            p = f'{work}/{k}'
            if not os.path.exists(p): os.makedirs(os.path.dirname(p), exist_ok=True); r2.get_file(k, p)
        with ThreadPoolExecutor(32) as ex: list(ex.map(dl, keys))
        T = lambda p: torch.from_numpy(cv2.imread(p)[..., ::-1].copy()).permute(2, 0, 1)  # BGR file -> RGB uint8 tensor
        self.data = {}
        for key, it in use.items():
            m = it['meta']; d = {'video': bool(m.get('video')), 'lq': {}, 'seq': m['sequences']}
            if d['video']: d['gt'] = [T(f"{work}/{it['gt_prefix']}gt_{t}.jpg") for t in (1, 2, 3)]
            else: d['gt_full'] = T(f"{work}/{it['gt_prefix']}gt.jpg")
            for l in range(3): d['lq'][l] = torch.stack([T(f"{work}/{key}lq_{l}_{t}.png") for t in (1, 2, 3)])
            self.data[key] = d
        flow_net = mt.FlowNet(device) if any(d['video'] for d in self.data.values()) else None
        for key, d in self.data.items():
            if d['video']:
                g2, g3 = (d['gt'][i][None].float().div(255).to(device) for i in (1, 2))
                d['flow'] = (flow_net(g2, g3).half().cpu(), flow_net(g3, g2).half().cpu())
        self.setup_s = round(time.time() - t0)
        log('evaluator ready', json.dumps({k: {'still': len(v['still']), 'video': len(v['video'])} for k, v in self.sel.items()}), f'{self.setup_s}s')
        self.lp = None

    def gt_frames(self, d, level):
        if d['video']: return [g.float().div(255) for g in d['gt']]
        seq = d['seq'][level]; c = seq['crop']; g = d['gt_full']
        return [g[:, y:y + c, x:x + c].float().div(255) for x, y in (seq['offsets'][t] for t in (1, 2, 3))]

    @torch.no_grad()
    def run(self, G, suite='quick'):
        if self.lp is None:
            import lpips; self.lp = lpips.LPIPS(net='alex', version='0.1', verbose=False).to(self.dev).eval()
        was = G.training; G.eval(); t0 = time.time(); sel = self.sel[suite]; out = {}
        kinds = [k for k in ('still', 'video') if sel[k]]
        for kind in kinds:
            for li in sel['levels']:
                ps, ls, tc, cov = [], [], [], []
                for it in sel[kind]:
                    d = self.data[it['key']]; lq = d['lq'][li].float().div(255).to(self.dev)
                    sr = G(lq).clamp(0, 1); gt = torch.stack(self.gt_frames(d, li)).to(self.dev)
                    ps.append(psnr_y(sr[1:2], gt[1:2])); ls.append(self.lp(sr[1:2] * 2 - 1, gt[1:2] * 2 - 1).item())
                    if kind == 'still':
                        o = d['seq'][li]['offsets']; vals = []
                        for a, b in ((1, 2), (2, 3)):
                            dx, dy = o[b][0] - o[a][0], o[b][1] - o[a][1]; vals.append(mt.pan_consistency(sr[a - 1:a], sr[b - 1:b], dx, dy))
                        tc.append(float(np.mean(vals)))
                    else:
                        ff, fb = (f.float().to(self.dev) for f in d['flow'])
                        t, c = mt.video_temporal_residual(sr[1:2], sr[2:3], gt[1:2], gt[2:3], (ff, fb)); tc.append(t); cov.append(c)
                r = {'psnr': float(np.mean(ps)), 'lpips': float(np.mean(ls)), 'lpips_worst10': float(np.percentile(ls, 90)), 'n': len(ps)}
                r['pan_consistency' if kind == 'still' else 'temporal_residual'] = float(np.mean(tc))
                if cov: r['flow_coverage'] = float(np.mean(cov))
                out[f'{kind}/{LEVELS[li]}'] = r
        for kind in kinds:
            rs = [v for k, v in out.items() if k.startswith(kind + '/')]
            out[f'{kind}/mean'] = {k: float(np.mean([r[k] for r in rs])) for k in rs[0] if k != 'n'}
        G.train(was)
        out['_meta'] = {'suite': suite, 'evaluator': EVALUATOR_VERSION, 'lpips': 'alex v0.1, RGB in [-1,1]', 'seconds': round(time.time() - t0, 1),
                        'samples': {k: len(sel[k]) for k in ('still', 'video')}, 'levels': [LEVELS[l] for l in sel['levels']]}
        return out
