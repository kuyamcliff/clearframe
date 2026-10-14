"""Video data prep (Kaggle CPU): openly licensed 4K masters -> short real-motion sequences.

Each sample under data/<batch>/train/<id>/: gt_<t>.jpg (768 px crop of consecutive 4K frames, q97 4:4:4),
lq_0_<t>.png (degraded through blur/downscale/noise/real codec, same pipeline as photos), meta.json with
'video': true, the source film, timestamp and licence.
"""
import hashlib, json, os, random, subprocess, sys, time, zipfile, glob, urllib.request
import numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2util as r2
from prep_photos import degrade_spatial, codec, SCALE, FRAMES

BATCH = os.environ.get('BATCH', 'v001'); PER_FILM = int(os.environ.get('PER_FILM', '600')); CROP = 768
WORK = '/kaggle/tmp' if os.path.isdir('/kaggle/tmp') else '/tmp/clipfx-v'; os.makedirs(WORK, exist_ok=True)
FILMS = [  # Blender Foundation open movies, CC BY 3.0 (blender.org)
    {'name': 'Tears of Steel', 'url': 'https://download.blender.org/demo/movies/ToS/tearsofsteel_4k.mov.zip', 'kind': 'live-action',
     'license': 'CC BY 3.0', 'credit': '(CC) Blender Foundation | mango.blender.org'},
    {'name': 'Sintel', 'url': 'https://download.blender.org/durian/movies/Sintel.2010.4k.mkv.zip', 'kind': 'animation',
     'license': 'CC BY 3.0', 'credit': '(CC) Blender Foundation | durian.blender.org'},
    {'name': 'Big Buck Bunny', 'url': 'https://download.blender.org/demo/movies/BBB/bbb_sunflower_2160p_30fps_normal.mp4.zip', 'kind': 'animation',
     'license': 'CC BY 3.0', 'credit': '(c) copyright 2008, Blender Foundation | www.bigbuckbunny.org'},
]

def fetch(film):
    z = f"{WORK}/{os.path.basename(film['url'])}"; d = z + '.d'
    if not os.path.exists(z) and not os.path.isdir(d):
        subprocess.run(['curl', '-sSLf', '--retry', '5', '-o', z, film['url']], check=True)
    if not os.path.isdir(d):
        with zipfile.ZipFile(z) as zf: zf.extractall(d)
        os.remove(z)  # keep only the extracted film on disk
    vids = [p for p in glob.glob(d + '/**/*', recursive=True) if p.lower().endswith(('.mov', '.mkv', '.mp4'))]
    return max(vids, key=os.path.getsize)

def probe(path):
    o = json.loads(subprocess.run(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=width,height,r_frame_rate:format=duration',
                                   '-of', 'json', path], capture_output=True, text=True, check=True).stdout)
    s = o['streams'][0]; n, d = s['r_frame_rate'].split('/')
    return s['width'], s['height'], float(n) / float(d), float(o['format']['duration'])

def frames_at(path, t, w, h, n):
    p = subprocess.run(['ffmpeg', '-v', 'error', '-ss', f'{t:.3f}', '-i', path, '-frames:v', str(n), '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-'],
                       capture_output=True, check=True)
    a = np.frombuffer(p.stdout, np.uint8)
    return a.reshape(-1, h, w, 3) if a.size else None

def main():
    sid = 0
    for film in FILMS:
        path = fetch(film); w, h, fps, dur = probe(path); print(film['name'], w, h, round(fps, 2), round(dur), flush=True)
        rng = np.random.default_rng(int(hashlib.sha256(film['name'].encode()).hexdigest()[:15], 16)); made = 0; tries = 0  # seed stable across runs
        while made < PER_FILM and tries < PER_FILM * 4:
            tries += 1; t = float(rng.uniform(dur * 0.03, dur * 0.93))  # skip titles and credits
            pre = f'data/{BATCH}/train/{sid:05d}/'
            if r2.exists(pre + 'meta.json'): sid += 1; made += 1; continue
            fr = frames_at(path, t, w, h, FRAMES)
            if fr is None or len(fr) < FRAMES: continue
            x = int(rng.integers(0, w - CROP)); y = int(rng.integers(0, h - CROP)); crops = [np.ascontiguousarray(f[y:y + CROP, x:x + CROP]) for f in fr]
            g = cv2.cvtColor(crops[0], cv2.COLOR_BGR2GRAY)
            if g.std() < 12 or cv2.Laplacian(g, cv2.CV_64F).var() < 30: continue  # skip flat or black/out-of-focus regions
            cfg = {}; lq = codec(degrade_spatial(crops, rng, cfg), rng, cfg)
            for i in range(FRAMES):
                r2.put(pre + f'gt_{i}.jpg', cv2.imencode('.jpg', crops[i], [cv2.IMWRITE_JPEG_QUALITY, 97, cv2.IMWRITE_JPEG_SAMPLING_FACTOR, cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444])[1].tobytes())
                r2.put(pre + f'lq_0_{i}.png', cv2.imencode('.png', lq[i])[1].tobytes())
            meta = {'video': True, 'source': {'film': film['name'], 'url': film['url'], 'kind': film['kind'], 'license': film['license'], 'credit': film['credit'],
                    'time': round(t, 3), 'fps': fps, 'crop_xy': [x, y]}, 'scale': SCALE, 'sequences': [{'crop': CROP, 'offsets': [[0, 0]] * FRAMES, 'degradation': cfg}]}
            r2.put(pre + 'meta.json', json.dumps(meta, default=int).encode()); sid += 1; made += 1
            if made % 50 == 0: print(film['name'], made, flush=True)
        subprocess.run(['rm', '-rf', os.path.dirname(path)])  # free disk before the next film
    print('done', sid, flush=True)

if __name__ == '__main__':
    main()
