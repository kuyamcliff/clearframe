import numpy as np, torch, pytest
import metrics_temporal as mt

def img(seed, h=256, w=256):
    g = torch.Generator().manual_seed(seed); x = torch.rand(1, 3, h // 8, w // 8, generator=g)
    return torch.nn.functional.interpolate(x, size=(h, w), mode='bicubic', align_corners=False).clamp(0, 1)

def test_pan_consistency_zero_on_exact_shift():
    big = img(0, 320, 320); a = big[..., 10:266, 10:266]; b = big[..., 17:273, 5:261]  # frame2[y,x] = frame1[y+7, x-5]
    assert mt.pan_consistency(a, b, -5, 7) < 1e-4
    assert mt.pan_consistency(a, b, 5, -7) > 1.0  # wrong sign is not aligned

def test_backwarp_identity_and_shift():
    x = img(1); z = torch.zeros(1, 2, 256, 256)
    w, inside = mt.backwarp(x, z); assert (w - x).abs().max() < 1e-5 and inside.min() == 1
    f = torch.zeros(1, 2, 256, 256); f[:, 0] = 3.0  # sample 3 px to the right
    w, inside = mt.backwarp(x, f); assert (w[..., :, :250] - x[..., :, 3:253]).abs().max() < 1e-5
    assert inside[..., :, -3:].sum() == 0  # out-of-frame pixels are masked

def test_residual_ignores_true_motion_and_detects_added_flicker():
    big = img(2, 320, 320); gt0 = big[..., 20:276, 20:276]; gt1 = big[..., 20:276, 24:280]  # gt1[y,x] = gt0[y, x+4]
    f_fw = torch.zeros(1, 2, 256, 256); f_fw[:, 0] = 4.0; f_bw = -f_fw  # t -> t+1 flow: content moved 4 px left
    # a perfect restoration has zero added inconsistency even though the scene moved
    t, cov = mt.video_temporal_residual(gt0, gt1, gt0, gt1, (f_bw * -1, f_bw), border=8)
    assert cov > 0.8 and t < 1e-3, (t, cov)
    # independent per-frame noise in the output is reported
    g = torch.Generator().manual_seed(3)
    n0 = (gt0 + 0.02 * torch.randn(gt0.shape, generator=g)).clamp(0, 1); n1 = (gt1 + 0.02 * torch.randn(gt1.shape, generator=g)).clamp(0, 1)
    t2, _ = mt.video_temporal_residual(n0, n1, gt0, gt1, (f_bw * -1, f_bw), border=8)
    assert t2 > 3.0, t2

def test_occlusion_mask_flags_inconsistent_flow():
    f = torch.zeros(1, 2, 64, 64); f[:, 0] = 2.0
    good = mt.occlusion_mask(f, -f); bad = mt.occlusion_mask(f, f)
    assert good[..., 8:-8, 8:-8].mean() > 0.99 and bad[..., 8:-8, 8:-8].mean() < 0.01

def test_video_temporal_loss_matches_metric_and_has_gradients():
    big = img(4, 320, 320); gt0 = big[..., 20:276, 20:276]; gt1 = big[..., 20:276, 24:280]
    f_fw = torch.zeros(1, 2, 256, 256); f_fw[:, 0] = 4.0; f_bw = -f_fw
    sr0 = (gt0 + 0.02 * torch.randn_like(gt0)).requires_grad_(); sr1 = gt1 + 0.02 * torch.randn_like(gt1)
    l = mt.video_temporal_loss(sr0, sr1, gt0, gt1, f_fw, f_bw, border=8); l.backward()
    t, _ = mt.video_temporal_residual(sr0.detach(), sr1, gt0, gt1, (f_fw, f_bw), border=8)
    assert abs(l.item() * 255 - t) < 1e-3 and sr0.grad.abs().sum() > 0
