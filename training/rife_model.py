"""RIFE IFNet v4.26 in PyTorch, built from the published state dict's shapes so its weights load back into ClipFX.

Mirrors Practical-RIFE (MIT, hzwer) and the app's own candle implementation (crates/inference/src/arch/rife.rs).
"""
import io, zipfile, urllib.request
import torch, torch.nn as nn, torch.nn.functional as F

URL = 'https://huggingface.co/hzwer/RIFE/resolve/01fdc7e97404120c243c3ea7b427046e5dc7643e/RIFEv4.26_0921.zip'

def load_base():
    z = zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(URL).read()))
    sd = torch.load(io.BytesIO(z.read('RIFEv4.26_0921/flownet.pkl')), map_location='cpu', weights_only=True)
    sd = {k[7:] if k.startswith('module.') else k: v for k, v in sd.items()}
    return {k: v for k, v in sd.items() if not k.startswith(('teacher.', 'caltime.'))}

def warp(img, flow):
    """Backward warp with border clamping (same as Practical-RIFE's grid_sample(border, align_corners=True))."""
    b, _, h, w = img.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=img.device, dtype=flow.dtype), torch.arange(w, device=img.device, dtype=flow.dtype), indexing='ij')
    x = (gx + flow[:, 0]) / max(w - 1, 1) * 2 - 1; y = (gy + flow[:, 1]) / max(h - 1, 1) * 2 - 1
    return F.grid_sample(img, torch.stack([x, y], -1).to(img.dtype), mode='bilinear', padding_mode='border', align_corners=True)

def lrelu(x): return F.leaky_relu(x, 0.2)

class Head(nn.Module):
    def __init__(s, sd):
        super().__init__(); g = lambda n: sd[f'encode.{n}.weight'].shape
        s.cnn0 = nn.Conv2d(g('cnn0')[1], g('cnn0')[0], 3, 2, 1); s.cnn1 = nn.Conv2d(g('cnn1')[1], g('cnn1')[0], 3, 1, 1)
        s.cnn2 = nn.Conv2d(g('cnn2')[1], g('cnn2')[0], 3, 1, 1); s.cnn3 = nn.ConvTranspose2d(g('cnn3')[0], g('cnn3')[1], 4, 2, 1)
    def forward(s, x): return s.cnn3(lrelu(s.cnn2(lrelu(s.cnn1(lrelu(s.cnn0(x)))))))

class ResConv(nn.Module):
    def __init__(s, c): super().__init__(); s.conv = nn.Conv2d(c, c, 3, 1, 1); s.beta = nn.Parameter(torch.ones(1, c, 1, 1))
    def forward(s, x): return lrelu(s.conv(x) * s.beta + x)

class IFBlock(nn.Module):
    def __init__(s, sd, p):
        super().__init__(); w0 = sd[f'{p}.conv0.0.0.weight'].shape; w1 = sd[f'{p}.conv0.1.0.weight'].shape; c = w1[0]
        s.conv0 = nn.Sequential(nn.Sequential(nn.Conv2d(w0[1], w0[0], 3, 2, 1)), nn.Sequential(nn.Conv2d(w1[1], w1[0], 3, 2, 1)))
        n = len({k.split('.')[2] for k in sd if k.startswith(f'{p}.convblock.')})
        s.convblock = nn.Sequential(*[ResConv(c) for _ in range(n)])
        lw = sd[f'{p}.lastconv.0.weight'].shape; s.lastconv = nn.Sequential(nn.ConvTranspose2d(lw[0], lw[1], 4, 2, 1))
    def forward(s, x, flow, scale):
        x = F.interpolate(x, scale_factor=1. / scale, mode='bilinear', align_corners=False)
        if flow is not None:
            x = torch.cat([x, F.interpolate(flow, scale_factor=1. / scale, mode='bilinear', align_corners=False) / scale], 1)
        f = lrelu(s.conv0[1][0](lrelu(s.conv0[0][0](x)))); f = s.convblock(f)
        t = F.interpolate(F.pixel_shuffle(s.lastconv[0](f), 2), scale_factor=scale, mode='bilinear', align_corners=False)
        return t[:, :4] * scale, t[:, 4:5], t[:, 5:]

class IFNet(nn.Module):
    def __init__(s, sd):
        super().__init__(); s.encode = Head(sd); s.block0, s.block1, s.block2, s.block3, s.block4 = (IFBlock(sd, f'block{i}') for i in range(5))
        s.load_state_dict(sd)
    def forward(s, img0, img1, timestep=0.5, scale=1.0):
        ts = torch.full_like(img0[:, :1], timestep); f0 = s.encode(img0); f1 = s.encode(img1)
        scales = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]; flow = mask = feat = None; wi0, wi1 = img0, img1
        for i, blk in enumerate([s.block0, s.block1, s.block2, s.block3, s.block4]):
            if flow is None:
                flow, mask, feat = blk(torch.cat([img0, img1, f0, f1, ts], 1), None, scales[i])
            else:
                wf0 = warp(f0, flow[:, :2]); wf1 = warp(f1, flow[:, 2:4])
                fd, mask, feat = blk(torch.cat([wi0, wi1, wf0, wf1, ts, mask, feat], 1), flow, scales[i]); flow = flow + fd
            wi0 = warp(img0, flow[:, :2]); wi1 = warp(img1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        return (wi0 * m + wi1 * (1 - m)).clamp(0, 1)
