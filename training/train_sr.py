"""Fine-tune Real-ESRGAN General v3 (SRVGG compact, x4) on Kaggle with 10-minute checkpoints to R2.

Resumes from runs/<RUN>/ckpt_latest.pt when present. Every EVAL_EVERY steps it scores the EMA generator on the
held-out test set (PSNR on Y, LPIPS-Alex, temporal flicker on the known pan) and keeps the best as best_G.pth.
"""
import io, json, os, random, sys, time, glob, math, subprocess
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2util as r2
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lpips'], check=False)
import numpy as np, cv2, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import spectral_norm
import lpips

RUN = os.environ.get('RUN', 'general-v3-ft1'); BATCHES = os.environ.get('BATCHES', 'b001').split(',')
HOURS = float(os.environ.get('HOURS', '11.3')); CKPT_MIN = float(os.environ.get('CKPT_MIN', '10'))
EVAL_EVERY = int(os.environ.get('EVAL_EVERY', '2000')); BS = int(os.environ.get('BS', '24')); LQ = int(os.environ.get('LQ_PATCH', '64'))
LR = float(os.environ.get('LR', '5e-5')); W_GAN = float(os.environ.get('W_GAN', '0.1')); W_PERC = float(os.environ.get('W_PERC', '1.0')); W_TEMP = float(os.environ.get('W_TEMP', '1.0'))
BASE_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
BASE_D_URL = 'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.3/RealESRGAN_x4plus_netD.pth'
T0 = time.time(); DEADLINE = T0 + HOURS * 3600; DEV = 'cuda'; P = f'runs/{RUN}/'
WORK = '/kaggle/tmp' if os.path.isdir('/kaggle/tmp') else '/tmp/clipfx'; os.makedirs(WORK, exist_ok=True)
def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)

# ---------------- models ----------------
class SRVGGNetCompact(nn.Module):
    def __init__(s, nf=64, nc=32, up=4):
        super().__init__(); s.up = up; b = [nn.Conv2d(3, nf, 3, 1, 1), nn.PReLU(nf)]
        for _ in range(nc): b += [nn.Conv2d(nf, nf, 3, 1, 1), nn.PReLU(nf)]
        b += [nn.Conv2d(nf, 3 * up * up, 3, 1, 1)]; s.body = nn.ModuleList(b); s.ps = nn.PixelShuffle(up)
    def forward(s, x):
        o = x
        for m in s.body: o = m(o)
        return s.ps(o) + F.interpolate(x, scale_factor=s.up, mode='nearest')

class UNetD(nn.Module):  # Real-ESRGAN UNetDiscriminatorSN
    def __init__(s, nf=64):
        super().__init__(); sn = spectral_norm
        s.conv0 = nn.Conv2d(3, nf, 3, 1, 1); s.conv1 = sn(nn.Conv2d(nf, nf * 2, 4, 2, 1, bias=False)); s.conv2 = sn(nn.Conv2d(nf * 2, nf * 4, 4, 2, 1, bias=False))
        s.conv3 = sn(nn.Conv2d(nf * 4, nf * 8, 4, 2, 1, bias=False)); s.conv4 = sn(nn.Conv2d(nf * 8, nf * 4, 3, 1, 1, bias=False))
        s.conv5 = sn(nn.Conv2d(nf * 4, nf * 2, 3, 1, 1, bias=False)); s.conv6 = sn(nn.Conv2d(nf * 2, nf, 3, 1, 1, bias=False))
        s.conv7 = sn(nn.Conv2d(nf, nf, 3, 1, 1, bias=False)); s.conv8 = sn(nn.Conv2d(nf, nf, 3, 1, 1, bias=False)); s.conv9 = nn.Conv2d(nf, 1, 3, 1, 1)
    def forward(s, x):
        a = lambda t: F.leaky_relu(t, 0.2, True); up = lambda t: F.interpolate(t, scale_factor=2, mode='bilinear', align_corners=False)
        x0 = a(s.conv0(x)); x1 = a(s.conv1(x0)); x2 = a(s.conv2(x1)); x3 = a(s.conv3(x2))
        x4 = a(s.conv4(up(x3))) + x2; x5 = a(s.conv5(up(x4))) + x1; x6 = a(s.conv6(up(x5))) + x0
        return s.conv9(a(s.conv8(a(s.conv7(x6)))))

class VGGLoss(nn.Module):
    def __init__(s):
        super().__init__(); import torchvision
        f = torchvision.models.vgg19(weights=torchvision.models.VGG19_Weights.IMAGENET1K_V1).features[:36].eval()
        for p in f.parameters(): p.requires_grad = False
        s.f = f; s.ids = {2: 0.1, 7: 0.1, 16: 1.0, 25: 1.0, 34: 1.0}  # conv1_2 conv2_2 conv3_4 conv4_4 conv5_4 (pre-ReLU)
        s.register_buffer('m', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)); s.register_buffer('sd', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(s, x, y):
        x = (x - s.m) / s.sd; y = (y - s.m) / s.sd; loss = 0
        for i, l in enumerate(s.f):
            x = l(x); y = l(y)
            if i in s.ids: loss = loss + s.ids[i] * F.l1_loss(x, y.detach())
            if i >= 34: break
        return loss

# ---------------- data ----------------
def fetch_batch(b):
    d = f'{WORK}/data/{b}'; flag = d + '/.complete'
    if os.path.exists(flag): return d
    keys = [k for k in r2.ls(f'data/{b}/') if '/train/' in k or '/test/' in k]
    log('downloading', b, len(keys), 'files')
    from concurrent.futures import ThreadPoolExecutor
    def dl(k):
        p = f'{WORK}/{k}'; os.makedirs(os.path.dirname(p), exist_ok=True)
        if not os.path.exists(p): r2.get_file(k, p)
    with ThreadPoolExecutor(32) as ex: list(ex.map(dl, keys))
    open(flag, 'w').close(); return d

def load_items(split):
    items = []
    for b in BATCHES:
        for m in sorted(glob.glob(f'{fetch_batch(b)}/{split}/*/meta.json')):
            d = os.path.dirname(m); items.append((d, json.load(open(m))))
    return items

def build_cache(items):
    """Unpack each training sequence once: the GT region the pan covers and its LQ frames, as raw uint8 on disk."""
    idx_p = f'{WORK}/cache_index.json'
    if os.path.exists(idx_p): return json.load(open(idx_p))
    index, gt_off, lq_off = [], 0, 0; plan = []
    for d, m in items:
        for si, seq in enumerate(m['sequences']):
            xs = [o[0] for o in seq['offsets']]; ys = [o[1] for o in seq['offsets']]; c = seq['crop']
            x0, y0, w, h = min(xs), min(ys), max(xs) - min(xs) + c, max(ys) - min(ys) + c
            n = len(seq['offsets']); l = c // 4
            index.append({'gt': gt_off, 'w': w, 'h': h, 'lq': lq_off, 'l': l, 'n': n, 'offs': [[x - x0, y - y0] for x, y in seq['offsets']]})
            plan.append((d, si, x0, y0, w, h, n)); gt_off += w * h * 3; lq_off += n * l * l * 3
    gtm = np.lib.format.open_memmap(f'{WORK}/gt.npy', 'w+', np.uint8, (gt_off,)); lqm = np.lib.format.open_memmap(f'{WORK}/lq.npy', 'w+', np.uint8, (lq_off,))
    from concurrent.futures import ThreadPoolExecutor
    def fill(k):
        d, si, x0, y0, w, h, n = plan[k]; e = index[k]
        g = cv2.imread(d + '/gt.jpg')
        gtm[e['gt']:e['gt'] + w * h * 3] = np.ascontiguousarray(g[y0:y0 + h, x0:x0 + w]).reshape(-1)
        for t in range(n):
            f = cv2.imread(f'{d}/lq_{si}_{t}.png'); o = e['lq'] + t * e['l'] ** 2 * 3
            lqm[o:o + f.size] = f.reshape(-1)
    with ThreadPoolExecutor(4) as ex: list(ex.map(fill, range(len(plan))))
    gtm.flush(); lqm.flush(); del gtm, lqm
    json.dump(index, open(idx_p, 'w')); log('cache built', len(index), 'sequences', round(gt_off / 1e9, 1), 'GB'); return index

def _aug_shift(dx, dy, hf, vf, k):
    """How a content shift (frame2[y,x] = frame1[y+dy, x+dx]) changes under the same flips and np.rot90 k times."""
    if hf: dx = -dx
    if vf: dy = -dy
    for _ in range(k): dx, dy = dy, -dx
    return dx, dy

class PairDataset(torch.utils.data.Dataset):
    """Two consecutive frames of a pan: LQ patches at the same LQ position, their GT patches, and the GT-pixel shift."""
    def __init__(s, index): s.index = index; s.gt = s.lq = None
    def __len__(s): return 10 ** 9
    def __getitem__(s, _):
        if s.gt is None: s.gt = np.load(f'{WORK}/gt.npy', mmap_mode='r'); s.lq = np.load(f'{WORK}/lq.npy', mmap_mode='r')
        e = random.choice(s.index); t = random.randrange(e['n'] - 1); L = e['l']
        g = s.gt[e['gt']: e['gt'] + e['w'] * e['h'] * 3].reshape(e['h'], e['w'], 3)
        y = random.randrange(0, L - LQ + 1); x = random.randrange(0, L - LQ + 1); ls, hs = [], []
        for f in (t, t + 1):
            ox, oy = e['offs'][f]
            ls.append(s.lq[e['lq'] + f * L * L * 3: e['lq'] + (f + 1) * L * L * 3].reshape(L, L, 3)[y:y + LQ, x:x + LQ])
            hs.append(g[oy + y * 4: oy + y * 4 + LQ * 4, ox + x * 4: ox + x * 4 + LQ * 4])
        (x1, y1), (x2, y2) = e['offs'][t], e['offs'][t + 1]; dx, dy = x2 - x1, y2 - y1
        hf, vf, k = random.random() < 0.5, random.random() < 0.5, random.randrange(4)
        def aug(a):
            if hf: a = a[:, ::-1]
            if vf: a = a[::-1]
            return np.rot90(a, k)
        dx, dy = _aug_shift(dx, dy, hf, vf, k)
        tt = lambda a: torch.from_numpy(np.ascontiguousarray(aug(a)[..., ::-1]).transpose(2, 0, 1)).float() / 255
        return tt(ls[0]), tt(hs[0]), tt(ls[1]), tt(hs[1]), torch.tensor([dx, dy])

def temporal_loss(o1, o2, shifts):
    """Mean |o1 - o2| over the overlap after aligning by the known pan shift; the reference is perfectly static."""
    c = o1.shape[-1]; tot = 0
    for i, (dx, dy) in enumerate(shifts.tolist()):
        a = o1[i, :, max(dy, 0):c + min(dy, 0), max(dx, 0):c + min(dx, 0)]
        b = o2[i, :, max(-dy, 0):c + min(-dy, 0), max(-dx, 0):c + min(-dx, 0)]
        tot = tot + (a - b).abs().mean()
    return tot / len(shifts)

# ---------------- eval ----------------
def ycbcr_y(t):  # RGB in [0,1] -> BT.601 luma in [16,235]
    return 16 + 65.481 * t[:, 0] + 128.553 * t[:, 1] + 24.966 * t[:, 2]

@torch.no_grad()
def evaluate(G, test, lp):
    G.eval(); res = {}
    for d, m in test:
        g = cv2.imread(d + '/gt.jpg')
        for si, seq in enumerate(m['sequences']):
            lvl = seq['degradation'].get('level', str(si)); outs = []
            for t in (1, 2, 3):
                lq = torch.from_numpy(cv2.imread(f'{d}/lq_{si}_{t}.png')[..., ::-1].copy()).permute(2, 0, 1)[None].float().div(255).to(DEV)
                outs.append(G(lq).clamp(0, 1))
            c = seq['crop']; ox, oy = seq['offsets'][2]
            gt = torch.from_numpy(g[oy:oy + c, ox:ox + c, ::-1].copy()).permute(2, 0, 1)[None].float().div(255).to(DEV)
            o = outs[1]; b = 8
            mse = F.mse_loss(ycbcr_y(o)[..., b:-b, b:-b], ycbcr_y(gt)[..., b:-b, b:-b]).item()
            psnr = 10 * math.log10(255 ** 2 / max(mse, 1e-8))
            l = lp(o * 2 - 1, gt * 2 - 1).item()
            # flicker: consecutive outputs aligned by the known pan offsets (GT pixels); the reference is perfectly static
            (x1, y1), (x2, y2) = seq['offsets'][1], seq['offsets'][2]; dx, dy = x2 - x1, y2 - y1
            a = outs[0][..., max(dy, 0):c + min(dy, 0), max(dx, 0):c + min(dx, 0)]; bb = outs[1][..., max(-dy, 0):c + min(-dy, 0), max(-dx, 0):c + min(-dx, 0)]
            fl = (a - bb).abs().mean().item() * 255
            r = res.setdefault(lvl, {'psnr': [], 'lpips': [], 'flicker': []}); r['psnr'].append(psnr); r['lpips'].append(l); r['flicker'].append(fl)
    G.train()
    out = {k: {kk: float(np.mean(vv)) for kk, vv in v.items()} for k, v in res.items()}
    out['mean'] = {kk: float(np.mean([out[k][kk] for k in res])) for kk in ['psnr', 'lpips', 'flicker']}; return out

BASELINE = {}
def better(m, best):
    # Looks better (LPIPS), keeps accuracy (at most 0.2 dB PSNR lost) and does not flicker more than 5% over the original model.
    flicker_cap = BASELINE.get('flicker', float('inf')) * 1.05
    return m['mean']['lpips'] < best['mean']['lpips'] - 1e-4 and m['mean']['psnr'] > best['mean']['psnr'] - 0.2 and m['mean']['flicker'] <= flicker_cap

@torch.no_grad()
def save_samples(G, test, tag):
    G.eval(); tiles = []
    for d, m in test[:4]:
        si = 2; lq = cv2.imread(f'{d}/lq_{si}_2.png'); c = m['sequences'][si]['crop']; ox, oy = m['sequences'][si]['offsets'][2]
        o = G(torch.from_numpy(lq[..., ::-1].copy()).permute(2, 0, 1)[None].float().div(255).to(DEV)).clamp(0, 1)[0].permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255
        g = cv2.imread(d + '/gt.jpg')[oy:oy + c, ox:ox + c]
        tiles.append(np.hstack([cv2.resize(lq, (c, c), interpolation=cv2.INTER_NEAREST), o.astype(np.uint8), g]))
    G.train(); r2.put(P + f'samples/{tag}.jpg', cv2.imencode('.jpg', np.vstack(tiles), [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes())

# ---------------- main ----------------
def worker(rank, world, index, test):
    import torch.distributed as dist
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29511')
    dist.init_process_group('nccl', rank=rank, world_size=world); torch.cuda.set_device(rank); dev = f'cuda:{rank}'
    global DEV; DEV = dev; torch.backends.cudnn.benchmark = True; random.seed(1234 + rank + int(time.time()))
    cv2.setNumThreads(1); torch.set_num_threads(1)
    G = SRVGGNetCompact().to(dev); D = UNetD().to(dev); vgg = VGGLoss().to(dev).to(memory_format=torch.channels_last)
    base = torch.load(io.BytesIO(__import__('urllib.request').request.urlopen(BASE_URL).read()), map_location='cpu', weights_only=True)
    G.load_state_dict(base['params'] if 'params' in base else base)
    bd = torch.load(io.BytesIO(__import__('urllib.request').request.urlopen(BASE_D_URL).read()), map_location='cpu', weights_only=True)
    D.load_state_dict(bd['params'] if 'params' in bd else bd)
    Gema = SRVGGNetCompact().to(dev); Gema.load_state_dict(G.state_dict()); Gema.requires_grad_(False)
    oG = torch.optim.Adam(G.parameters(), LR, betas=(0.9, 0.99)); oD = torch.optim.Adam(D.parameters(), LR, betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler(); step = 0; state = {'best': None, 'baseline': None}
    if os.path.exists(f'{WORK}/ck.pt'):
        ck = torch.load(f'{WORK}/ck.pt', map_location='cpu', weights_only=False)
        G.load_state_dict(ck['G']); D.load_state_dict(ck['D']); Gema.load_state_dict(ck['Gema']); oG.load_state_dict(ck['oG']); oD.load_state_dict(ck['oD'])
        step = ck['step']; state = ck['state']
        if rank == 0: log('resumed at step', step)
    lp = lpips.LPIPS(net='alex', verbose=False).to(dev).eval() if rank == 0 else None
    if rank == 0 and state['baseline'] is None:
        state['baseline'] = evaluate(Gema, test, lp); state['best'] = state['baseline']; log('baseline', json.dumps(state['baseline']['mean']))
        r2.put(P + 'baseline.json', json.dumps(state['baseline']).encode()); save_samples(Gema, test, 'baseline')
    if rank == 0: BASELINE.update(state['baseline']['mean'])
    dist.barrier()
    Gp = nn.parallel.DistributedDataParallel(G, device_ids=[rank]); Dp = nn.parallel.DistributedDataParallel(D, device_ids=[rank], broadcast_buffers=False)
    dl = iter(torch.utils.data.DataLoader(PairDataset(index), batch_size=BS // world, num_workers=2, persistent_workers=True, prefetch_factor=6, pin_memory=True))
    last_ck = time.time(); t_log = time.time(); t_start = time.time(); s0 = step
    def checkpoint():
        torch.save({'G': G.state_dict(), 'D': D.state_dict(), 'Gema': Gema.state_dict(), 'oG': oG.state_dict(), 'oD': oD.state_dict(), 'step': step, 'state': state}, f'{WORK}/ck_new.pt')
        r2.put_file(P + 'ckpt_latest.pt', f'{WORK}/ck_new.pt'); r2.put(P + 'status.json', json.dumps({'step': step, 'time': time.time(), 'best': state['best']['mean'], 'baseline': state['baseline']['mean']}).encode())
        log('checkpoint', step)
    stop = torch.zeros(1, device=dev)
    while True:
        lq, hq, lq2, hq2, shifts = next(dl); lq, hq, lq2, hq2 = (v.to(dev, non_blocking=True) for v in (lq, hq, lq2, hq2))
        for p in D.parameters(): p.requires_grad = False
        with torch.autocast('cuda', dtype=torch.float16):
            both = Gp(torch.cat([lq, lq2])); sr, sr2 = both.chunk(2)
            l_temp = temporal_loss(sr.float(), sr2.float(), shifts) if W_TEMP > 0 else torch.zeros((), device=dev)
            l_pix = (F.l1_loss(sr, hq) + F.l1_loss(sr2, hq2)) / 2; l_perc = vgg(sr.contiguous(memory_format=torch.channels_last), hq.contiguous(memory_format=torch.channels_last))
            if W_GAN > 0:
                fake = D(sr); l_gan = F.binary_cross_entropy_with_logits(fake.float(), torch.ones_like(fake, dtype=torch.float32))
            else:
                l_gan = torch.zeros((), device=dev)
        lossG = l_pix + W_PERC * l_perc.float() + W_GAN * l_gan + W_TEMP * l_temp
        oG.zero_grad(set_to_none=True); scaler.scale(lossG).backward(); scaler.step(oG)
        if W_GAN > 0:
            for p in D.parameters(): p.requires_grad = True
            oD.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16):
                dr = Dp(hq).float(); l_real = F.binary_cross_entropy_with_logits(dr, torch.ones_like(dr))
            scaler.scale(l_real).backward()
            with torch.autocast('cuda', dtype=torch.float16):
                df = Dp(sr.detach()).float(); l_fake = F.binary_cross_entropy_with_logits(df, torch.zeros_like(df))
            scaler.scale(l_fake).backward(); scaler.step(oD)
        scaler.update(); step += 1
        with torch.no_grad():
            for pe, p in zip(Gema.parameters(), G.parameters()): pe.lerp_(p, 1 - 0.999)
        if rank == 0:
            if time.time() - t_log > 300:
                log(f'step {step} pix {l_pix.item():.4f} perc {l_perc.item():.4f} gan {l_gan.item():.3f} temp {l_temp.item() * 255:.2f} {(step - s0) / (time.time() - t_start):.2f} it/s'); t_log = time.time()
            if step % EVAL_EVERY == 0:
                m = evaluate(Gema, test, lp); m['step'] = step; log('eval', json.dumps(m['mean']))
                r2.put(P + f'evals/{step:07d}.json', json.dumps(m).encode())
                if better(m, state['best']):
                    state['best'] = m; torch.save({'params': Gema.state_dict()}, f'{WORK}/best.pth'); r2.put_file(P + 'best_G.pth', f'{WORK}/best.pth')
                    save_samples(Gema, test, f'best_{step:07d}'); log('NEW BEST', step)
            if time.time() - last_ck > CKPT_MIN * 60:
                checkpoint(); last_ck = time.time()
            if step % 500 == 0 and r2.exists(P + 'STOP'): log('stop requested'); r2.delete(P + 'STOP'); DEADLINE_HIT = True
            else: DEADLINE_HIT = time.time() > DEADLINE
            stop[0] = 1.0 if DEADLINE_HIT else 0.0
        if step % 50 == 0:
            dist.broadcast(stop, 0)
            if stop.item() > 0: break
    if rank == 0: checkpoint(); log('session done at step', step)
    dist.barrier(); dist.destroy_process_group()

def main():
    test = load_items('test'); train = load_items('train'); log('train items', len(train), 'test items', len(test))
    index = build_cache(train)
    if r2.exists(P + 'ckpt_latest.pt'): r2.get_file(P + 'ckpt_latest.pt', f'{WORK}/ck.pt')
    world = torch.cuda.device_count()
    torch.multiprocessing.spawn(worker, args=(world, index, test), nprocs=world, join=True)

if __name__ == '__main__':
    main()
