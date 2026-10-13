"""Preflight for Kaggle training jobs: fail fast, with a clear message, before any expensive work.

Checks GPUs, library versions, free disk and RAM, R2 connectivity, the selected data batches (and warns when
video batches exist but are not selected), the original weights' SHA-256, and a resumable checkpoint's structure.
Returns a report dict that the training job stores with its run config.
"""
import hashlib, json, os, shutil, sys, urllib.request
import r2util as r2

GENERAL_SHA = '8dc7edb9ac80ccdc'
WDN_SHA = '1641f8c4464b9f09'

class PreflightError(SystemExit):
    pass

def run(batches, need_video=False, run_prefix=None, min_disk_gb=40, min_ram_gb=12, need_gpus=1):
    rep, problems, warnings = {}, [], []
    import torch
    rep['torch'] = torch.__version__; rep['cuda'] = torch.version.cuda; rep['cudnn'] = torch.backends.cudnn.version()
    rep['gpus'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(rep['gpus']) < need_gpus: problems.append(f'need {need_gpus} GPU(s), found {len(rep["gpus"])}')
    work = '/kaggle/tmp' if os.path.isdir('/kaggle/tmp') else '/tmp'
    rep['disk_free_gb'] = round(shutil.disk_usage(work).free / 1e9, 1)
    if rep['disk_free_gb'] < min_disk_gb: problems.append(f'only {rep["disk_free_gb"]} GB free in {work}')
    try:
        mem = dict(l.split(':', 1) for l in open('/proc/meminfo')); rep['ram_free_gb'] = round(int(mem['MemAvailable'].split()[0]) / 1e6, 1)
        if rep['ram_free_gb'] < min_ram_gb: problems.append(f'only {rep["ram_free_gb"]} GB RAM available')
    except OSError:
        pass
    try:
        r2.s3.head_bucket(Bucket=r2.BUCKET); rep['r2'] = 'ok'
    except Exception as e:
        problems.append(f'R2 unreachable: {type(e).__name__}')
    present = sorted({k.split('/')[1] for k in r2.ls('data/') if k.count('/') >= 2})
    rep['batches_selected'] = batches; rep['batches_present'] = present
    for b in batches:
        if b not in present: problems.append(f'batch {b} not found in R2')
    videos_present = [b for b in present if b.startswith('v')]
    if videos_present and not any(b.startswith('v') for b in batches):
        warnings.append(f'video batches {videos_present} exist but are not selected')
    if need_video and not any(b.startswith('v') for b in batches):
        problems.append('this run expects video supervision but no video batch is selected')
    for name, want in [('realesr-general-x4v3.pth', GENERAL_SHA), ('realesr-general-wdn-x4v3.pth', WDN_SHA)]:
        try:
            h = hashlib.sha256(r2.get(f'originals/{name}')).hexdigest()
            rep[f'sha256:{name}'] = h
            if not h.startswith(want): problems.append(f'{name} hash {h[:16]} != catalog {want}')
        except Exception as e:
            problems.append(f'original weights {name} missing: {type(e).__name__}')
    if run_prefix and r2.exists(run_prefix + 'latest.json'):
        lat = json.loads(r2.get(run_prefix + 'latest.json'))
        if not r2.exists(lat['key']): problems.append(f'latest.json points at missing checkpoint {lat["key"]}')
        rep['resume_from'] = lat
    rep['warnings'] = warnings
    for w in warnings: print('PREFLIGHT WARNING:', w, flush=True)
    if problems:
        for p in problems: print('PREFLIGHT FAILED:', p, flush=True)
        raise PreflightError('preflight failed: ' + '; '.join(problems))
    print('preflight ok:', json.dumps(rep), flush=True)
    return rep

if __name__ == '__main__':
    run(sys.argv[1].split(',') if len(sys.argv) > 1 else ['b001'], need_gpus=0)
