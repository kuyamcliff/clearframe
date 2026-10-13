import json, numpy as np, torch, pytest, cv2
import sr_common as sc
import data_cache as dc
from test_data_cache import R2, make
import train_sr

def test_dataset_rgb_alignment_and_shift(tmp_path):
    # photo pan with a known gradient image so channel order and alignment can be checked exactly
    r2 = R2(); pre = 'data/p/train/0/'; c = 256; H, W = 256, 256 + 4 * 5
    yy, xx = np.mgrid[0:H, 0:W]; img = np.stack([xx % 256, yy % 256, np.full_like(xx, 77)], -1).astype(np.uint8)  # B=x, G=y, R=77
    offs = [[4 * t, 0] for t in range(6)]
    r2.put(pre + 'gt.jpg', cv2.imencode('.png', img)[1].tobytes())
    for t, (ox, oy) in enumerate(offs):
        r2.put(pre + f'lq_0_{t}.png', cv2.imencode('.png', cv2.resize(img[oy:oy + c, ox:ox + c], (64, 64), interpolation=cv2.INTER_AREA))[1].tobytes())
    r2.put(pre + 'meta.json', json.dumps({'sequences': [{'crop': c, 'offsets': offs, 'degradation': {}}]}).encode())
    items = dc.discover(r2, ['p'], str(tmp_path)); fp, _ = dc.fingerprint(r2, ['p']); idx, _ = dc.build(r2, items, str(tmp_path), fp, log=lambda *a: None)
    ds = sc.PairDataset(idx, f'{tmp_path}/cache-{fp}', 16)
    for i in range(30):
        lq, hq, lq2, hq2, sh = ds[i]
        assert torch.allclose(hq[0], torch.full_like(hq[0], 77 / 255))  # red channel first: RGB, not BGR
        # the LQ patch is the 4x area-downscale of its HQ patch (alignment), up to PNG rounding
        assert (torch.nn.functional.avg_pool2d(hq[None], 4)[0] - lq).abs().max() < 2 / 255
        dx, dy, valid = sh.tolist(); assert valid == 1
        from metrics_temporal import pan_consistency
        assert pan_consistency(hq[None], hq2[None], dx, dy) < 1e-4  # the reported shift aligns the two frames exactly

def test_pareto_slots():
    base = {'still/light': {'psnr': 27.0}, 'video/light': {'psnr': 30.0},
            'still/mean': {'psnr': 27.0, 'lpips': 0.40, 'pan_consistency': 2.0}, 'video/mean': {'psnr': 30.0, 'lpips': 0.30, 'temporal_residual': 3.0}}
    g = {'psnr_drop_max_db': 0.2, 'pan_consistency_max_ratio': 1.05, 'video_temporal_max_ratio': 1.0, 'perceptual_video_temporal_max_ratio': 1.10}
    def cand(lp, vt, psnr=27.0, pan=2.0):
        m = {'still/light': {'psnr': psnr}, 'video/light': {'psnr': 30.0}, 'still/mean': {'psnr': psnr, 'lpips': lp, 'pan_consistency': pan},
             'video/mean': {'psnr': 30.0, 'lpips': lp - 0.1, 'temporal_residual': vt}}
        return m, train_sr.summ_metrics(m)
    # sharper but slightly more flicker: perceptual only
    w, _ = train_sr.decide_promotions(*cand(0.34, 3.2), base, {}, g); assert w == ['best_perceptual']
    # better everywhere: all three
    w, _ = train_sr.decide_promotions(*cand(0.36, 2.5), base, {}, g); assert set(w) == {'best_perceptual', 'best_temporal', 'best_balanced'}
    # big PSNR loss: nothing, however good it looks
    w, _ = train_sr.decide_promotions(*cand(0.30, 2.0, psnr=26.5), base, {}, g); assert w == []
    # blurrier but steadier (LPIPS worse than original): not temporal, not balanced
    w, _ = train_sr.decide_promotions(*cand(0.45, 2.0), base, {}, g); assert w == []
