"""Fine-tune Real-ESRGAN General v3 on Kaggle (PRD: clock, resume, cache, Pareto candidates, verified checkpoints).

Order of a session: preflight -> restore checkpoint pointer -> data discovery and fingerprinted cache -> evaluator
(original-model baseline once per run) -> training. The training clock starts at the first optimizer step; a separate
wall-clock limit (from process start) keeps the session inside Kaggle's maximum.

Config: configs/<CONFIG>.json, overridable by upper-case environment variables of the same name (e.g. W_TEMP=10).
"""
import io, json, os, random, sys, time, subprocess, urllib.request
T_PROCESS = float(os.environ.setdefault('T_PROCESS', str(time.time())))  # GPU worker processes re-import this file; keep the real start
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import r2util as r2

def load_config():
    name = os.environ.get('CONFIG', 'general_v3_ft2'); here = os.path.dirname(os.path.abspath(__file__))
    cfg = json.load(open(f'{here}/configs/{name}.json')) if os.path.exists(f'{here}/configs/{name}.json') else json.loads(os.environ['CONFIG_JSON'])
    for k, v in list(cfg.items()):
        e = os.environ.get(k.upper())
        if e is not None and not isinstance(v, dict): cfg[k] = e if isinstance(v, str) else json.loads(e)
    return cfg

def log(*a): print(time.strftime('%H:%M:%S'), *a, flush=True)

def main():
    timers = {}; mark = lambda k, t0: timers.__setitem__(k, round(time.time() - t0, 1))
    cfg = load_config(); P = f"runs/{cfg['run']}/"; t0 = time.time()
    if subprocess.run([sys.executable, '-c', 'import lpips'], capture_output=True).returncode:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'lpips==0.1.4'], check=True)
    mark('setup_packages_s', t0)
    import torch
    import preflight, ckpt, data_cache as dc, evaluate_sr as ev
    t0 = time.time(); pre = preflight.run(cfg['batches'], need_video=cfg.get('need_video', False), run_prefix=P, need_gpus=1); mark('preflight_s', t0)
    work = '/kaggle/tmp' if os.path.isdir('/kaggle/tmp') else '/tmp/clipfx'; os.makedirs(work, exist_ok=True)
    # 1) resume state first: it is small, and tells us whether this is a continuation
    t0 = time.time(); resume = ckpt.fetch_latest(r2, P, f'{work}/resume.pt'); mark('resume_fetch_s', t0)
    # 2) data
    t0 = time.time(); items = dc.discover(r2, cfg['batches'], work); fp, nobj = dc.fingerprint(r2, cfg['batches']); summ = dc.summary(items); mark('discover_s', t0)
    log('data', fp, json.dumps(summ))
    if cfg.get('need_video') and not any(k.endswith('/video') and '/train/' in k for k in summ['samples']): raise SystemExit('no training video loaded')
    index, cstats = dc.build(r2, items, work, fp, log=log); timers.update({f'cache_{k}': v for k, v in cstats.items()})
    run_config = {'config': cfg, 'git_sha': os.environ.get('GIT_SHA', 'unknown'), 'data_fingerprint': fp, 'data_objects': nobj, 'data_summary': summ,
                  'preflight': pre, 'evaluator': ev.EVALUATOR_VERSION, 'torch': torch.__version__, 'world': torch.cuda.device_count()}
    if resume:
        ck = torch.load(f'{work}/resume.pt', map_location='cpu', weights_only=False)
        old = ck['run_config']
        if old['config']['lq_patch'] != cfg['lq_patch'] or old['evaluator'] != ev.EVALUATOR_VERSION:
            raise SystemExit('checkpoint was made with an incompatible config/evaluator; start a new run name')
        if old['data_fingerprint'] != fp: log('note: data changed since the checkpoint (new batches or files); continuing')
    else:
        ck = None
    r2.put(P + 'run_config.json', json.dumps(run_config, indent=1, default=str).encode())
    world = torch.cuda.device_count()
    import torch.multiprocessing as tmp
    test_items = [it for it in items if it['split'] == 'test']
    tmp.spawn(worker, args=(world, cfg, P, work, fp, index, test_items, ck, run_config, timers), nprocs=world, join=True)

def worker(rank, world, cfg, P, work, fp, index, test_items, ck, run_config, timers):
    import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torch.distributed as dist
    import ckpt, sr_common as sc, evaluate_sr as ev
    os.environ.setdefault('MASTER_ADDR', '127.0.0.1'); os.environ.setdefault('MASTER_PORT', '29511')
    dist.init_process_group('nccl', rank=rank, world_size=world); torch.cuda.set_device(rank); dev = f'cuda:{rank}'
    torch.backends.cudnn.benchmark = True; import cv2; cv2.setNumThreads(1); torch.set_num_threads(1)
    random.seed(cfg['seed'] + rank); np.random.seed(cfg['seed'] + rank); torch.manual_seed(cfg['seed'] + rank)
    t0 = time.time()
    G = sc.SRVGGNetCompact().to(dev); D = sc.UNetD().to(dev); vgg = sc.VGGLoss().to(dev).to(memory_format=torch.channels_last)
    G.load_state_dict(torch.load(io.BytesIO(r2.get(cfg['init_g'])), map_location='cpu', weights_only=True)['params'])
    bd = torch.load(io.BytesIO(urllib.request.urlopen(cfg['init_d_url']).read()), map_location='cpu', weights_only=True); D.load_state_dict(bd.get('params', bd))
    Gema = sc.SRVGGNetCompact().to(dev); Gema.load_state_dict(G.state_dict()); Gema.requires_grad_(False)
    oG = torch.optim.Adam(G.parameters(), cfg['lr'], betas=(0.9, 0.99)); oD = torch.optim.Adam(D.parameters(), cfg['lr'], betas=(0.9, 0.99))
    scaler = torch.amp.GradScaler(); step = 0
    state = {'baseline': None, 'pareto': {}, 'train_seconds_total': 0.0, 'sessions': 0}
    if ck:
        G.load_state_dict(ck['G']); D.load_state_dict(ck['D']); Gema.load_state_dict(ck['Gema']); oG.load_state_dict(ck['oG']); oD.load_state_dict(ck['oD'])
        scaler.load_state_dict(ck['scaler']); step = ck['step']; state = ck['state']
        # rank 0 restores its exact generators; other ranks reseed deterministically from the step
        if rank == 0 and ck.get('rng'): ckpt.set_rng_state(ck['rng'][0])
        else: random.seed(cfg['seed'] + rank + 1000 * step); np.random.seed((cfg['seed'] + rank + 1000 * step) % 2 ** 32); torch.manual_seed(cfg['seed'] + rank + 1000 * step)
    state['sessions'] += 1
    timers['model_init_s'] = round(time.time() - t0, 1)
    E = None
    if rank == 0:
        t1 = time.time(); E = ev.Evaluator(r2, test_items, work, dev, log=log); timers['evaluator_setup_s'] = round(time.time() - t1, 1)
        if state['baseline'] is None:  # original model, same evaluator and data, once per run
            t1 = time.time(); state['baseline'] = {'quick': E.run(Gema, 'quick'), 'full': E.run(Gema, 'full')}; timers['baseline_eval_s'] = round(time.time() - t1, 1)
            r2.put(P + 'baseline.json', json.dumps(state['baseline']).encode()); log('baseline', json.dumps(summ_metrics(state['baseline']['full'])))
    dist.barrier()
    Gp = nn.parallel.DistributedDataParallel(G, device_ids=[rank]); Dp = nn.parallel.DistributedDataParallel(D, device_ids=[rank], broadcast_buffers=False)
    dl = iter(torch.utils.data.DataLoader(sc.PairDataset(index, f'{work}/cache-{fp}', cfg['lq_patch']), batch_size=cfg['batch_size'] // world,
                                          num_workers=cfg['num_workers'], persistent_workers=True, prefetch_factor=cfg['prefetch_factor'], pin_memory=True))
    acc = {'train_s': 0.0, 'data_wait_s': 0.0, 'eval_s': 0.0, 'ckpt_s': 0.0, 'pairs': 0}
    t_start = time.time(); last_ck = time.time(); t_log = time.time(); s0 = step
    budget = cfg['train_seconds']; wall_end = T_PROCESS + cfg['max_wall_seconds'] - cfg['final_reserve_seconds']
    stop = torch.zeros(1, device=dev); last_eval = {}

    def status(extra=None):
        st = {'status_version': 2, 'run': cfg['run'], 'step': step, 'time': time.time(), 'batches': cfg['batches'], 'data_fingerprint': fp,
              'baseline': summ_metrics(state['baseline']['full']), 'last_eval': last_eval, 'pareto': {k: {'step': v['step'], 'metrics': v['metrics']} for k, v in state['pareto'].items()},
              'session': {**timers, **{k: round(v, 1) for k, v in acc.items()}, 'session_s': round(time.time() - T_PROCESS),
                          'pairs_per_s': round(acc['pairs'] / max(acc['train_s'], 1e-9), 1), 'updates_per_s': round((step - s0) / max(acc['train_s'], 1e-9), 3)},
              'train_seconds_total': round(state['train_seconds_total'] + acc['train_s'])}
        if extra: st.update(extra)
        r2.put(P + 'status.json', json.dumps(st, default=str).encode())

    def checkpoint():
        t1 = time.time()
        obj = {'G': G.state_dict(), 'D': D.state_dict(), 'Gema': Gema.state_dict(), 'oG': oG.state_dict(), 'oD': oD.state_dict(), 'scaler': scaler.state_dict(),
               'step': step, 'state': {**state, 'train_seconds_total': state['train_seconds_total'] + acc['train_s']}, 'rng': [ckpt.rng_state()], 'run_config': run_config}
        ckpt.save_local(obj, f'{work}/ck.pt'); ptr = ckpt.publish(r2, P, f'{work}/ck.pt', step)
        ckpt.prune(r2, P, protect={v['key'] for v in state['pareto'].values() if 'key' in v})
        acc['ckpt_s'] += time.time() - t1; status(); log('checkpoint', step, ptr['key'].split('/')[-1], f'{time.time() - t1:.1f}s')

    def evaluate(suite):
        nonlocal last_eval
        t1 = time.time(); m = E.run(Gema, suite); acc['eval_s'] += time.time() - t1
        sm = summ_metrics(m); last_eval = {'suite': suite, 'step': step, 'metrics': sm, 'at': time.time()}
        r2.put(P + f'evals/{step:08d}-{suite}.json', json.dumps({'step': step, 'data_fingerprint': fp, 'suite_result': m}).encode())
        log('eval', suite, step, json.dumps(sm))
        if suite == 'full': promote(m, sm)
        status()

    def promote(m, sm):
        wins, rec_extra = decide_promotions(m, sm, state['baseline']['full'], state['pareto'], cfg['gates'])
        # every full-eval candidate is kept (immutable), promoted or not
        torch.save({'params': {k: v.detach().cpu() for k, v in Gema.state_dict().items()}}, f'{work}/cand.pth')
        sha = ckpt.sha256_file(f'{work}/cand.pth'); key = f'{P}candidates/step-{step:08d}-{sha[:12]}.pth'; r2.put_file(key, f'{work}/cand.pth')
        if r2.s3.head_object(Bucket=r2.BUCKET, Key=key)['ContentLength'] != os.path.getsize(f'{work}/cand.pth'): raise IOError('candidate upload mismatch')
        rec = {'step': step, 'key': key, 'sha256': sha, 'metrics': sm, **rec_extra,
               'data_fingerprint': fp, 'evaluator': run_config['evaluator'], 'ema': True, 'at': time.time()}
        r2.put(P + f'candidates/step-{step:08d}.json', json.dumps(rec).encode())
        for w in wins:
            state['pareto'][w] = rec; r2.put(P + f'{w}.json', json.dumps(rec).encode()); log('PROMOTED', w, step, json.dumps(sm))

    while True:
        tw = time.time(); lq, hq, lq2, hq2, shifts = next(dl); acc['data_wait_s'] += time.time() - tw
        ts = time.time()
        lq, hq, lq2, hq2 = (v.to(dev, non_blocking=True) for v in (lq, hq, lq2, hq2))
        for p in D.parameters(): p.requires_grad = False
        with torch.autocast('cuda', dtype=torch.float16):
            sr, sr2 = Gp(torch.cat([lq, lq2])).chunk(2)
            l_temp = sc.temporal_loss(sr.float(), sr2.float(), shifts) if cfg['w_temp'] > 0 else torch.zeros((), device=dev)
            l_pix = (F.l1_loss(sr, hq) + F.l1_loss(sr2, hq2)) / 2
            l_perc = vgg(sr.contiguous(memory_format=torch.channels_last), hq.contiguous(memory_format=torch.channels_last))
            fake = D(sr); l_gan = F.binary_cross_entropy_with_logits(fake.float(), torch.ones_like(fake, dtype=torch.float32)) if cfg['w_gan'] > 0 else torch.zeros((), device=dev)
        lossG = l_pix + cfg['w_perc'] * l_perc.float() + cfg['w_gan'] * l_gan + cfg['w_temp'] * l_temp
        oG.zero_grad(set_to_none=True); scaler.scale(lossG).backward(); scaler.step(oG)
        if cfg['w_gan'] > 0:
            for p in D.parameters(): p.requires_grad = True
            oD.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16): dr = Dp(hq).float(); l_real = F.binary_cross_entropy_with_logits(dr, torch.ones_like(dr))
            scaler.scale(l_real).backward()
            with torch.autocast('cuda', dtype=torch.float16): df = Dp(sr.detach()).float(); l_fake = F.binary_cross_entropy_with_logits(df, torch.zeros_like(df))
            scaler.scale(l_fake).backward(); scaler.step(oD)
        scaler.update(); step += 1; acc['pairs'] += lq.shape[0] * world
        with torch.no_grad():
            for pe, p in zip(Gema.parameters(), G.parameters()): pe.lerp_(p, 1 - cfg['ema_decay'])
        if step % 50 == 0: torch.cuda.synchronize()
        acc['train_s'] += time.time() - ts
        if rank == 0:
            if time.time() - t_log > 300:
                log(f"step {step} pix {l_pix.item():.4f} perc {l_perc.item():.3f} gan {l_gan.item():.3f} temp {l_temp.item() * 255:.2f} "
                    f"{(step - s0) / max(acc['train_s'], 1e-9):.2f} upd/s data-wait {acc['data_wait_s'] / max(time.time() - t_start, 1):.0%}"); t_log = time.time()
            if step % cfg['full_eval_every'] == 0: evaluate('full')
            elif step % cfg['quick_eval_every'] == 0: evaluate('quick')
            if time.time() - last_ck > cfg['ckpt_minutes'] * 60: checkpoint(); last_ck = time.time()
            done = acc['train_s'] >= budget or time.time() >= wall_end
            if step % 500 == 0 and r2.exists(P + 'STOP'): r2.delete(P + 'STOP'); log('stop requested'); done = True
            stop[0] = 1.0 if done else 0.0
        if step % 50 == 0:
            dist.broadcast(stop, 0)
            if stop.item() > 0: break
    if rank == 0:
        evaluate('full'); checkpoint(); log('session done', step, json.dumps({k: round(v, 1) for k, v in acc.items()}))
    dist.barrier(); dist.destroy_process_group()

def decide_promotions(m, sm, base_full, pareto, g):
    """Which Pareto slots a full evaluation wins. best_perceptual: lowest LPIPS with PSNR kept and video flicker at most
    10% worse. best_temporal: lowest real-video flicker with PSNR kept and LPIPS no worse than the original.
    best_balanced: passes every gate and has the lowest predeclared score. Returns (slots, record fields)."""
    b = summ_metrics(base_full)
    psnr_ok = all(m[k]['psnr'] >= base_full[k]['psnr'] - g['psnr_drop_max_db'] for k in m if not k.startswith('_') and not k.endswith('/mean'))
    pan_ok = sm['pan_consistency'] <= b['pan_consistency'] * g['pan_consistency_max_ratio']
    vt, vt0 = sm['video_temporal'] or 0.0, b['video_temporal'] or 1.0  # runs without held-out video only gate on stills
    vid_ok = vt <= vt0 * g['video_temporal_max_ratio']
    score = sm['lpips_mean'] / b['lpips_mean'] + 0.5 * vt / vt0
    cur = lambda slot, key, default: pareto.get(slot, {}).get('metrics', {}).get(key, default) if slot in pareto else default
    wins = []
    if psnr_ok and vt <= vt0 * g['perceptual_video_temporal_max_ratio'] and sm['lpips_mean'] < cur('best_perceptual', 'lpips_mean', b['lpips_mean']): wins.append('best_perceptual')
    if psnr_ok and sm['lpips_mean'] <= b['lpips_mean'] and vt < (cur('best_temporal', 'video_temporal', None) or vt0): wins.append('best_temporal')
    if psnr_ok and pan_ok and vid_ok and sm['lpips_mean'] < b['lpips_mean'] and score < pareto.get('best_balanced', {}).get('score', 1.5): wins.append('best_balanced')
    return wins, {'score': score, 'gates': {'psnr_ok': psnr_ok, 'pan_ok': pan_ok, 'video_ok': vid_ok}}

def summ_metrics(m):
    """The headline numbers used for gates and status."""
    still, video = m.get('still/mean', {}), m.get('video/mean', {})
    return {'still_psnr': still.get('psnr'), 'still_lpips': still.get('lpips'), 'pan_consistency': still.get('pan_consistency'),
            'video_psnr': video.get('psnr'), 'video_lpips': video.get('lpips'), 'video_temporal': video.get('temporal_residual'), 'flow_coverage': video.get('flow_coverage'),
            'lpips_mean': float(sum(x for x in [still.get('lpips'), video.get('lpips')] if x is not None) / max(1, sum(x is not None for x in [still.get('lpips'), video.get('lpips')])))}

if __name__ == '__main__':
    main()
