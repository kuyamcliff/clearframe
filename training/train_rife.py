"""Fine-tune RIFE v4.26 (frame interpolation / slow motion) on real-motion 4K clips, with 10-minute R2 checkpoints.

Training triplets come from the video batches' consecutive reference frames: (t, t+1, t+2) -> t+1 and the harder
(t, t+2, t+4) -> t+2. Clips from the last part of each film (time >= 80% of the sampled span) are held out for scoring.
A new best is kept only if the held-out middle-frame PSNR improves without LPIPS getting worse.
"""
import io, json, math, os, random, sys, time, glob, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2util as r2
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lpips'], check=False)
import numpy as np, cv2, torch, torch.nn.functional as F, lpips
import rife_model as R

RUN = os.environ.get('RUN', 'rife-v426-ft1'); BATCHES = os.environ.get('BATCHES', 'v001').split(',')
HOURS = float(os.environ.get('HOURS', '11.0')); CKPT_MIN = float(os.environ.get('CKPT_MIN', '10')); EVAL_EVERY = int(os.environ.get('EVAL_EVERY', '2000'))
BS = int(os.environ.get('BS', '8')); PATCH = int(os.environ.get('PATCH', '256')); LR = float(os.environ.get('LR', '2e-5'))
P = f'runs/{RUN}/'; DEADLINE = time.time() + HOURS * 3600; DEV = 'cuda'
WORK = '/kaggle/tmp' if os.path.isdir('/kaggle/tmp') else '/tmp/clipfx-rife'; os.makedirs(WORK, exist_ok=True)
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)

def fetch():
    seqs = []
    from concurrent.futures import ThreadPoolExecutor
    for b in BATCHES:
        keys = [k for k in r2.ls(f'data/{b}/train/') if '/gt_' in k or k.endswith('meta.json')]
        def dl(k):
            p = f'{WORK}/{k}'; os.makedirs(os.path.dirname(p), exist_ok=True)
            if not os.path.exists(p): r2.get_file(k, p)
        with ThreadPoolExecutor(32) as ex: list(ex.map(dl, keys))
        for m in sorted(glob.glob(f'{WORK}/data/{b}/train/*/meta.json')):
            meta = json.load(open(m))
            if meta.get('video'): seqs.append((os.path.dirname(m), meta['source']))
    # time-based split per film: the last part of each film is never trained on
    out = {'train': [], 'test': []}
    for film in {s['film'] for _, s in seqs}:
        fs = [x for x in seqs if x[1]['film'] == film]; tmax = max(s['time'] for _, s in fs); tmin = min(s['time'] for _, s in fs)
        cut = tmin + 0.8 * (tmax - tmin)
        for d, s in fs: out['test' if s['time'] >= cut else 'train'].append(d)
    log('sequences train', len(out['train']), 'test', len(out['test'])); return out

def frames(d): return [cv2.imread(f'{d}/gt_{t}.jpg') for t in range(6)]

class Triplets(torch.utils.data.Dataset):
    def __init__(s, dirs): s.dirs = dirs
    def __len__(s): return 10 ** 9
    def __getitem__(s, _):
        d = random.choice(s.dirs); gap = random.choice([1, 1, 2]); t = random.randrange(0, 6 - 2 * gap)
        a, m, b = (cv2.imread(f'{d}/gt_{t + k * gap}.jpg') for k in range(3))
        h, w = a.shape[:2]; y = random.randrange(0, h - PATCH + 1); x = random.randrange(0, w - PATCH + 1)
        a, m, b = (im[y:y + PATCH, x:x + PATCH] for im in (a, m, b))
        if random.random() < 0.5: a, b = b, a  # time reversal
        if random.random() < 0.5: a, m, b = (im[:, ::-1] for im in (a, m, b))
        if random.random() < 0.5: a, m, b = (im[::-1] for im in (a, m, b))
        tt = lambda im: torch.from_numpy(np.ascontiguousarray(im[..., ::-1]).transpose(2, 0, 1)).float() / 255
        return tt(a), tt(m), tt(b)

def lap_loss(x, y, levels=5):
    loss = 0
    for _ in range(levels):
        dx, dy = F.avg_pool2d(x, 2), F.avg_pool2d(y, 2)
        loss = loss + F.l1_loss(x - F.interpolate(dx, scale_factor=2, mode='bilinear', align_corners=False), y - F.interpolate(dy, scale_factor=2, mode='bilinear', align_corners=False))
        x, y = dx, dy
    return loss + F.l1_loss(x, y)

@torch.no_grad()
def evaluate(net, test, lp):
    net.eval(); ps, ls = {1: [], 2: []}, {1: [], 2: []}
    for d in test:
        fr = frames(d); T = lambda im: torch.from_numpy(im[..., ::-1].copy()).permute(2, 0, 1)[None].float().div(255).to(DEV)
        for gap in (1, 2):
            a, m, b = T(fr[0]), T(fr[gap]), T(fr[2 * gap])
            o = net(a, b)
            ps[gap].append(10 * math.log10(1 / max(F.mse_loss(o[..., 8:-8, 8:-8], m[..., 8:-8, 8:-8]).item(), 1e-10)))
            ls[gap].append(lp(o * 2 - 1, m * 2 - 1).item())
    net.train()
    r = {f'gap{g}': {'psnr': float(np.mean(ps[g])), 'lpips': float(np.mean(ls[g]))} for g in (1, 2)}
    r['mean'] = {'psnr': float(np.mean(ps[1] + ps[2])), 'lpips': float(np.mean(ls[1] + ls[2]))}; return r

def better(m, best): return m['mean']['psnr'] > best['mean']['psnr'] + 0.01 and m['mean']['lpips'] <= best['mean']['lpips'] + 1e-3

def main():
    torch.backends.cudnn.benchmark = True; split = fetch()
    net = R.IFNet(R.load_base()).to(DEV); lp = lpips.LPIPS(net='alex', verbose=False).to(DEV).eval()
    opt = torch.optim.AdamW(net.parameters(), LR, weight_decay=1e-4); step = 0; state = {'baseline': None, 'best': None}
    if r2.exists(P + 'ckpt_latest.pt'):
        r2.get_file(P + 'ckpt_latest.pt', f'{WORK}/ck.pt'); ck = torch.load(f'{WORK}/ck.pt', map_location='cpu', weights_only=False)
        net.load_state_dict(ck['net']); opt.load_state_dict(ck['opt']); step = ck['step']; state = ck['state']; log('resumed at', step)
    if state['baseline'] is None:
        state['baseline'] = state['best'] = evaluate(net, split['test'], lp); r2.put(P + 'baseline.json', json.dumps(state['baseline']).encode())
        log('baseline', json.dumps(state['baseline']))
    dl = iter(torch.utils.data.DataLoader(Triplets(split['train']), batch_size=BS, num_workers=4, persistent_workers=True, prefetch_factor=4))
    last_ck = time.time(); t_log = time.time()
    def checkpoint():
        torch.save({'net': net.state_dict(), 'opt': opt.state_dict(), 'step': step, 'state': state}, f'{WORK}/ck_new.pt')
        r2.put_file(P + 'ckpt_latest.pt', f'{WORK}/ck_new.pt')
        r2.put(P + 'status.json', json.dumps({'step': step, 'time': time.time(), 'best': state['best']['mean'], 'baseline': state['baseline']['mean']}).encode()); log('checkpoint', step)
    while time.time() < DEADLINE:
        a, m, b = (v.to(DEV, non_blocking=True) for v in next(dl))
        with torch.autocast('cuda', dtype=torch.float16): o = net(a, b)
        loss = lap_loss(o.float(), m)
        opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step(); step += 1
        if time.time() - t_log > 300: log(f'step {step} loss {loss.item():.4f}'); t_log = time.time()
        if step % EVAL_EVERY == 0:
            mt = evaluate(net, split['test'], lp); mt['step'] = step; r2.put(P + f'evals/{step:07d}.json', json.dumps(mt).encode()); log('eval', json.dumps(mt['mean']))
            if better(mt, state['best']):
                state['best'] = mt; torch.save(net.state_dict(), f'{WORK}/best.pkl'); r2.put_file(P + 'best_flownet.pkl', f'{WORK}/best.pkl'); log('NEW BEST', step)
        if step % 500 == 0 and r2.exists(P + 'STOP'): r2.delete(P + 'STOP'); log('stop requested'); break
        if time.time() - last_ck > CKPT_MIN * 60: checkpoint(); last_ck = time.time()
    checkpoint(); log('session done at', step)

if __name__ == '__main__':
    main()
