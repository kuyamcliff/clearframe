"""PyTorch reference outputs for the ClearFrame parity check. See tools/parity/README.md."""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.environ.get('CF_REFPKG', os.path.join(os.path.dirname(__file__), 'refpkg')))
torch.set_grad_enabled(False)
M = os.environ.get('CF_MODELS', os.path.join(os.path.dirname(__file__), 'models'))
OUT = os.environ.get('CF_REFS', os.path.join(os.path.dirname(__file__), 'refs')); os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(1234)

def img(h, w):
    # smooth-ish random content
    base = rng.random((1, 3, h // 4 + 1, w // 4 + 1)).astype(np.float32)
    t = torch.nn.functional.interpolate(torch.from_numpy(base), size=(h, w), mode='bilinear', align_corners=False)
    t = t + 0.05 * torch.from_numpy(rng.standard_normal((1, 3, h, w)).astype(np.float32))
    return t.clamp(0, 1)

def save(name, inputs, out, meta=None):
    d = {'inputs': [], 'output_shape': list(out.shape), 'meta': meta or {}}
    for i, x in enumerate(inputs):
        x.numpy().astype('<f4').tofile(f'{OUT}/{name}.in{i}.bin'); d['inputs'].append(list(x.shape))
    out.numpy().astype('<f4').tofile(f'{OUT}/{name}.out.bin')
    json.dump(d, open(f'{OUT}/{name}.json', 'w'))
    print(name, 'out', list(out.shape), float(out.abs().mean()))

def sd(f, key=None):
    s = torch.load(f, map_location='cpu', weights_only=True)
    for k in ['params_ema', 'params', 'state_dict']:
        if isinstance(s, dict) and k in s: s = s[k]; break
    return {k.replace('module.', '', 1) if k.startswith('module.') else k: v for k, v in s.items()}

from basicsr.archs.srvgg_arch import SRVGGNetCompact
from basicsr.archs.rrdbnet_arch import RRDBNet
for name, f, nc in [('animevideov3', 'realesr-animevideov3.pth', 16), ('general_x4v3', 'realesr-general-x4v3.pth', 32)]:
    m = SRVGGNetCompact(num_feat=64, num_conv=nc, upscale=4, act_type='prelu'); m.load_state_dict(sd(f'{M}/{f}')); m.eval()
    x = img(48, 64); save(name, [x], m(x))
for name, f, nb, sc in [('x4plus', 'RealESRGAN_x4plus.pth', 23, 4), ('anime6b', 'RealESRGAN_x4plus_anime_6B.pth', 6, 4), ('x2plus', 'RealESRGAN_x2plus.pth', 23, 2)]:
    m = RRDBNet(3, 3, scale=sc, num_feat=64, num_block=nb, num_grow_ch=32); m.load_state_dict(sd(f'{M}/{f}')); m.eval()
    x = img(32, 40); save(name, [x], m(x))

from models.network_ffdnet import FFDNet
from models.network_unet import UNetRes
m = FFDNet(in_nc=3, out_nc=3, nc=96, nb=12, act_mode='R'); m.load_state_dict(sd(f'{M}/ffdnet_color.pth')); m.eval()
x = img(45, 61); sigma = 25 / 255.
save('ffdnet', [x], m(x, torch.full((1, 1, 1, 1), sigma)), {'sigma': sigma})
m = UNetRes(in_nc=4, out_nc=3, nc=[64, 128, 256, 512], nb=4, act_mode='R', downsample_mode='strideconv', upsample_mode='convtranspose', bias=False)
m.load_state_dict(sd(f'{M}/drunet_color.pth')); m.eval()
x = img(64, 72); xin = torch.cat([x, torch.full((1, 1, 64, 72), sigma)], 1)
save('drunet', [x], m(xin), {'sigma': sigma})

from basicsr.models.archs.NAFNet_arch import NAFNet
m = NAFNet(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
m.load_state_dict(sd(f'{M}/NAFNet-GoPro-width32.pth')); m.eval()
x = img(50, 70); save('nafnet', [x], m(x))

from model.IFNet_HDv3 import IFNet
m = IFNet(); m.load_state_dict({k: v for k, v in sd(f'{M}/rife/RIFEv4.26_0921/flownet.pkl').items() if 'teacher' not in k and 'caltime' not in k}, strict=False); m.eval()
a = img(128, 192); b = torch.roll(a, shifts=(3, 5), dims=(2, 3)) * 0.98
flow, mask, merged = m(torch.cat([a, b], 1), 0.5, [16, 8, 4, 2, 1])
save('rife', [a, b], merged[4].clamp(0, 1), {'t': 0.5, 'scale': 1.0})

from gfpgan.archs.gfpganv1_clean_arch import GFPGANv1Clean
m = GFPGANv1Clean(out_size=512, num_style_feat=512, channel_multiplier=2, decoder_load_path=None, fix_decoder=False, num_mlp=8, input_is_latent=True, different_w=True, narrow=1, sft_half=True)
m.load_state_dict(sd(f'{M}/GFPGANv1.4.pth'), strict=True); m.eval()
x = img(512, 512); o, _ = m((x - 0.5) / 0.5, return_rgb=False, randomize_noise=False)
save('gfpgan', [x], ((o.clamp(-1, 1) + 1) / 2))

from facexlib.detection.retinaface import RetinaFace
for name, net, f in [('retina_r50', 'resnet50', 'detection_Resnet50_Final.pth'), ('retina_mnet', 'mobile0.25', 'detection_mobilenet0.25_Final.pth')]:
    m = RetinaFace(network_name=net, half=False, device=torch.device('cpu'))
    m.load_state_dict(sd(f'{M}/{f}'), strict=True); m.eval()
    x = img(96, 128)
    bgr = x[:, [2, 1, 0]] * 255 - torch.tensor([104., 117., 123.]).view(1, 3, 1, 1)
    loc, conf, landm = m(bgr)
    save(name + '_loc', [x], loc[0]); save(name + '_conf', [x], conf[0]); save(name + '_landm', [x], landm[0])
