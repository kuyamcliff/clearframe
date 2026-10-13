"""Temporal metrics.

synthetic_pan_consistency_l1: mean |SR_t - SR_t+1| after aligning by an exactly known integer pan shift (x255).
    A controlled sanity metric on still photos; the reference is perfectly static, so any difference is added flicker.

video_temporal_residual: on real footage with unknown motion, estimate flow on the clean reference (RAFT), warp
    frame t+1 into frame t, mask occlusions (forward-backward check) and out-of-frame pixels, and measure
        T = mean_valid | (SR_t - W(SR_t+1)) - (GT_t - W(GT_t+1)) |   (x255)
    i.e. only the inconsistency the model adds beyond the scene's own change. Reported with valid-pixel coverage.
"""
import torch, torch.nn.functional as F

EVAL_VERSION = 'temporal-v1'

def pan_consistency(o1, o2, dx, dy):
    """o1, o2: (B,3,H,W). Frame t+1 content at (y, x) equals frame t content at (y+dy, x+dx)."""
    c_h, c_w = o1.shape[-2:]
    a = o1[..., max(dy, 0):c_h + min(dy, 0), max(dx, 0):c_w + min(dx, 0)]
    b = o2[..., max(-dy, 0):c_h + min(-dy, 0), max(-dx, 0):c_w + min(-dx, 0)]
    return (a - b).abs().mean().item() * 255

def backwarp(img, flow):
    b, _, h, w = img.shape
    gy, gx = torch.meshgrid(torch.arange(h, device=img.device, dtype=img.dtype), torch.arange(w, device=img.device, dtype=img.dtype), indexing='ij')
    x = gx + flow[:, 0]; y = gy + flow[:, 1]
    grid = torch.stack([x / (w - 1) * 2 - 1, y / (h - 1) * 2 - 1], -1)
    inside = ((x >= 0) & (x <= w - 1) & (y >= 0) & (y <= h - 1)).float()[:, None]
    return F.grid_sample(img, grid, mode='bilinear', padding_mode='border', align_corners=True), inside

class FlowNet:
    """RAFT (torchvision, small) on clean reference frames. Inputs in [0,1] RGB, sizes divisible by 8."""
    def __init__(self, device):
        from torchvision.models.optical_flow import raft_small, Raft_Small_Weights
        self.net = raft_small(weights=Raft_Small_Weights.DEFAULT).to(device).eval(); self.device = device
    @torch.no_grad()
    def __call__(self, a, b):
        return self.net(a * 2 - 1, b * 2 - 1, num_flow_updates=12)[-1]

def occlusion_mask(f_fw, f_bw):
    """Forward-backward consistency (Sundaram et al. 2010 thresholds): 1 where the flow is trustworthy."""
    f_bw_w, inside = backwarp(f_bw, f_fw)
    lhs = ((f_fw + f_bw_w) ** 2).sum(1, keepdim=True)
    rhs = 0.01 * ((f_fw ** 2).sum(1, keepdim=True) + (f_bw_w ** 2).sum(1, keepdim=True)) + 0.5
    return (lhs < rhs).float() * inside

@torch.no_grad()
def video_temporal_residual(sr_t, sr_t1, gt_t, gt_t1, flow, border=8):
    """flow = (f_fw, f_bw) between gt_t and gt_t1 at the SR/GT resolution. Returns (T x255, valid coverage 0..1)."""
    f_fw, f_bw = flow
    m = occlusion_mask(f_fw, f_bw)
    m[..., :border, :] = 0; m[..., -border:, :] = 0; m[..., :, :border] = 0; m[..., :, -border:] = 0
    w_sr, _ = backwarp(sr_t1, f_fw); w_gt, _ = backwarp(gt_t1, f_fw)
    r = ((sr_t - w_sr) - (gt_t - w_gt)).abs().mean(1, keepdim=True)
    cov = m.mean().item()
    return ((r * m).sum() / m.sum().clamp(min=1)).item() * 255, cov


def video_temporal_loss(sr_t, sr_t1, gt_t, gt_t1, f_fw, f_bw, border=4):
    """Differentiable training form of video_temporal_residual: masked mean of the added temporal change (0..1 scale)."""
    with torch.no_grad():
        msk = occlusion_mask(f_fw, f_bw)
        msk[..., :border, :] = 0; msk[..., -border:, :] = 0; msk[..., :, :border] = 0; msk[..., :, -border:] = 0
        w_gt, _ = backwarp(gt_t1, f_fw); target = gt_t - w_gt
    w_sr, _ = backwarp(sr_t1, f_fw)
    r = ((sr_t - w_sr) - target).abs().mean(1, keepdim=True)
    return (r * msk).sum() / msk.sum().clamp(min=1)
