"""PyTorch reference outputs for the ClipFX Enhance parity check. See tools/parity/README.md."""
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

from facexlib.detection.retinaface import RetinaFace
for name, net, f in [('retina_r50', 'resnet50', 'detection_Resnet50_Final.pth'), ('retina_mnet', 'mobile0.25', 'detection_mobilenet0.25_Final.pth')]:
    m = RetinaFace(network_name=net, half=False, device=torch.device('cpu'))
    m.load_state_dict(sd(f'{M}/{f}'), strict=True); m.eval()
    x = img(96, 128)
    bgr = x[:, [2, 1, 0]] * 255 - torch.tensor([104., 117., 123.]).view(1, 3, 1, 1)
    loc, conf, landm = m(bgr)
    save(name + '_loc', [x], loc[0]); save(name + '_conf', [x], conf[0]); save(name + '_landm', [x], landm[0])
