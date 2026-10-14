"""Shared, side-effect-free pieces of General v3 training: models, perceptual loss, pair dataset, temporal loss.

Importing this module installs nothing and starts no timers.
"""
import random
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.nn.utils import spectral_norm

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

def _aug_shift(dx, dy, hf, vf, k):
    """How a content shift (frame2[y,x] = frame1[y+dy, x+dx]) changes under the same flips and np.rot90 k times."""
    if hf: dx = -dx
    if vf: dy = -dy
    for _ in range(k): dx, dy = dy, -dx
    return dx, dy

class PairDataset(torch.utils.data.Dataset):
    """Two consecutive frames of a pan: LQ patches at the same LQ position, their GT patches, and the GT-pixel shift."""
    def __init__(s, index, cache, lq_patch, video_frac=None):
        s.index = index; s.cache = cache; s.LQ = lq_patch; s.gt = s.lq = None
        # video_frac: share of samples drawn from real-video sequences (None = uniform over all sequences, where
        # video is only as common as its share of the index).
        s.video_frac = video_frac; s.videos = [e for e in index if e.get('video')]; s.photos = [e for e in index if not e.get('video')]
    def __len__(s): return 10 ** 9
    def pick(s):
        if s.video_frac is None or not s.videos or not s.photos: return random.choice(s.index)
        return random.choice(s.videos if random.random() < s.video_frac else s.photos)
    def __getitem__(s, _):
        if s.gt is None: s.gt = np.load(f'{s.cache}/gt.npy', mmap_mode='r'); s.lq = np.load(f'{s.cache}/lq.npy', mmap_mode='r')
        e = s.pick(); t = random.randrange(e['n'] - 1); L = e['l']
        sz = e['w'] * e['h'] * 3; video = e.get('video', False)
        LQ = s.LQ; y = random.randrange(0, L - LQ + 1); x = random.randrange(0, L - LQ + 1); ls, hs = [], []
        for f in (t, t + 1):
            ox, oy = e['offs'][f]; base = e['gt'] + (f * sz if video else 0)
            g = s.gt[base: base + sz].reshape(e['h'], e['w'], 3)
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
        # Real camera motion is unknown, so video pairs do not take part in the temporal loss (flag 0).
        return tt(ls[0]), tt(hs[0]), tt(ls[1]), tt(hs[1]), torch.tensor([dx, dy, 0 if video else 1])

def temporal_loss(o1, o2, shifts):
    """Mean |o1 - o2| over the overlap after aligning by the known pan shift; the reference is perfectly static."""
    c = o1.shape[-1]; tot = 0; cnt = 0
    for i, row in enumerate(shifts.tolist()):
        dx, dy = row[0], row[1]
        if len(row) > 2 and not row[2]: continue
        cnt += 1
        a = o1[i, :, max(dy, 0):c + min(dy, 0), max(dx, 0):c + min(dx, 0)]
        b = o2[i, :, max(-dy, 0):c + min(-dy, 0), max(-dx, 0):c + min(-dx, 0)]
        tot = tot + (a - b).abs().mean()
    return tot / max(cnt, 1)

