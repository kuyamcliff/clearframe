"""Test the ClipFX Enhance app on a Kaggle GPU (NVIDIA T4): build from source, Vulkan GPU detection, GPU parity tests,
per-model GPU benchmarks and end-to-end video jobs through the CLI. Logs and a summary go to R2 under apptest/<sha>/.

Env: SRC_KEY (R2 key of the source tarball), SHA.
"""
import json, os, subprocess, sys, time, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2util as r2

SHA = os.environ.get('SHA', 'unknown'); OUT = f'apptest/{SHA}/'; W = '/kaggle/working' if os.path.isdir('/kaggle/working') else '/tmp/apptest'
os.makedirs(W, exist_ok=True); summary = {'sha': SHA, 'steps': {}}
ENV = dict(os.environ, PATH=f"{os.path.expanduser('~/.cargo/bin')}:{os.environ['PATH']}", CARGO_TERM_COLOR='never', RUST_LOG='info')


def sh(name, cmd, timeout=3600, cwd=None):
    t = time.time(); print(f'\n=== {name}: {cmd}', flush=True)
    try:
        p = subprocess.run(cmd, shell=True, cwd=cwd, env=ENV, capture_output=True, text=True, timeout=timeout)
        out, rc = p.stdout + p.stderr, p.returncode
    except subprocess.TimeoutExpired as e:
        out, rc = f'TIMEOUT after {timeout}s\n{(e.stdout or b"")[-4000:]!r}', -1
    dt = round(time.time() - t, 1); print(out[-6000:], flush=True); print(f'=== {name}: rc={rc} in {dt}s', flush=True)
    r2.put(OUT + f'logs/{name}.log', out.encode()); summary['steps'][name] = {'rc': rc, 'seconds': dt}
    r2.put(OUT + 'summary.json', json.dumps(summary, indent=1).encode())
    return rc, out


# 1. GPU, driver and Vulkan
sh('nvidia_smi', 'nvidia-smi')
sh('apt', 'apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libvulkan1 vulkan-tools ffmpeg >/dev/null; echo done', 1200)
rc, libs = sh('nvidia_libs', 'ldconfig -p | grep -i -E "nvidia|vulkan|libcuda"; ls -la /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1')
ver = subprocess.run('nvidia-smi --query-gpu=driver_version --format=csv,noheader', shell=True, capture_output=True, text=True).stdout.split()[0]
summary['driver'] = ver
if 'libGLX_nvidia' not in libs:
    # Containers started with only the compute/utility driver capabilities get CUDA but not the driver's graphics part,
    # which holds its Vulkan implementation. Add the user-space graphics libraries of the exact same driver version
    # from NVIDIA's own package (the kernel module is already loaded and is not touched).
    sh('x11_libs', 'DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libx11-6 libxext6 >/dev/null; echo done', 600)
    url = None
    for u in (f'https://us.download.nvidia.com/tesla/{ver}/NVIDIA-Linux-x86_64-{ver}.run', f'https://us.download.nvidia.com/XFree86/Linux-x86_64/{ver}/NVIDIA-Linux-x86_64-{ver}.run'):
        if sh('driver_probe', f'curl -sSfIL -o /dev/null -w "%{{http_code}}" "{u}"')[0] == 0: url = u; break
    summary['driver_package'] = url
    if url:
        sh('driver_download', f'curl -sSfL -o /tmp/nv.run "{url}" && sh /tmp/nv.run --extract-only --target /opt/nv >/dev/null && ls /opt/nv | head -80', 1200)
        L = '/usr/lib/x86_64-linux-gnu'
        sh('driver_install', f'cd /opt/nv && for f in libGLX_nvidia.so.{ver} libnvidia-*.so.{ver}; do [ -e {L}/$f ] || cp -n $f {L}/; done; '
                             f'ln -sf {L}/libGLX_nvidia.so.{ver} {L}/libGLX_nvidia.so.0; ldconfig; '
                             f'mkdir -p /etc/vulkan/icd.d && sed "s#__NV_VK_ICD__#libGLX_nvidia.so.0#" nvidia_icd.json.template > /etc/vulkan/icd.d/nvidia_icd.json 2>/dev/null '
                             f'|| cp nvidia_icd.json /etc/vulkan/icd.d/nvidia_icd.json; cat /etc/vulkan/icd.d/nvidia_icd.json; ls {L} | grep -c nvidia')
summary['icd'] = [p for p in ('/usr/share/vulkan/icd.d/nvidia_icd.json', '/etc/vulkan/icd.d/nvidia_icd.json') if os.path.exists(p)]
sh('vulkaninfo', 'vulkaninfo --summary 2>&1 | head -60')

# 2. Build the app from the exact commit
sh('rustup', 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal >/dev/null 2>&1; rustc --version', 900)
r2.get_file(os.environ['SRC_KEY'], f'{W}/src.tar.gz'); os.makedirs(f'{W}/src', exist_ok=True)
sh('extract', f'tar xzf {W}/src.tar.gz -C {W}/src && ls {W}/src')
SRC = f'{W}/src'
sh('build', 'cargo build --release -p clearframe-cli 2>&1 | tail -30', 3600, SRC)
sh('build_bench', 'cargo build --release -p clearframe-inference --features gpu --example realbench 2>&1 | tail -10', 3600, SRC)
sh('build_tests', 'cargo test --release -p clearframe-inference --features gpu --test gpu_parity --no-run 2>&1 | tail -5', 3600, SRC)
CLI = f'{SRC}/target/release/clipfx'; DATA = f'{W}/cfdata'

# 3. App: hardware detection, models
sh('hardware', f'{CLI} --data-root {DATA} hardware')
models = ['xinntao.realesr-general-v3', 'xinntao.realesr-animevideo-v3', 'xinntao.realesrgan-x4plus', 'xinntao.realesrgan-x4plus-anime-6b',
          'xinntao.realesrgan-x2plus', 'kair.ffdnet-color', 'kair.drunet-color', 'megvii.nafnet-gopro-w32', 'hzwer.rife-v4.26', 'tencentarc.gfpgan-v1.4']
for m in models:
    sh(f'install_{m}', f'{CLI} --data-root {DATA} models install {m}', 1200)
sh('models_list', f'{CLI} --data-root {DATA} models list')

# 4. GPU correctness on the real GPU, then per-model speed
sh('gpu_parity', 'cargo test --release -p clearframe-inference --features gpu --test gpu_parity -- --test-threads=1 2>&1 | tail -30', 3600, SRC)
st = sorted(glob.glob(f'{DATA}/**/*.safetensors', recursive=True))
summary['model_files'] = [os.path.relpath(p, DATA) for p in st]
rc, bench = sh('realbench', f'{SRC}/target/release/examples/realbench ' + ' '.join(f'"{p}"' for p in st), 3600)
summary['bench'] = [json.loads(l) for l in bench.splitlines() if l.startswith('{')]

# 5. End to end: a real clip (Sintel trailer, (c) Blender Foundation, CC BY 3.0) through the CLI with several presets
sh('clip', f'curl -sSfL -o {W}/sintel.mp4 https://download.blender.org/durian/trailer/sintel_trailer-480p.mp4 && '
           f'ffmpeg -v error -y -ss 20 -t 4 -i {W}/sintel.mp4 -an -c:v libx264 -crf 18 {W}/clip.mp4 && {CLI} --data-root {DATA} info {W}/clip.mp4', 600)
summary['e2e'] = {}
for preset in ['builtin.live-1080', 'builtin.anime-2x', 'builtin.denoise', 'builtin.faces', 'builtin.smooth-60', 'builtin.restore-old']:
    t = time.time()
    rc, out = sh(f'run_{preset}', f'{CLI} --data-root {DATA} run {W}/clip.mp4 --preset {preset} -o {W}/out --name {preset.split(".")[1]}', 3600)
    outs = glob.glob(f'{W}/out/{preset.split(".")[1]}.*')
    summary['e2e'][preset] = {'rc': rc, 'seconds': round(time.time() - t, 1), 'output': [os.path.basename(o) for o in outs],
                              'tail': out[-1500:]}
    for o in outs:
        sh(f'info_{preset}', f'{CLI} --data-root {DATA} info "{o}"')
sh('jobs', f'{CLI} --data-root {DATA} --json jobs list')
r2.put(OUT + 'summary.json', json.dumps(summary, indent=1).encode())
print('\nSUMMARY', json.dumps({k: v for k, v in summary.items() if k != 'e2e'}, indent=1))
print('E2E', json.dumps({k: {kk: vv for kk, vv in v.items() if kk != 'tail'} for k, v in summary['e2e'].items()}, indent=1))
