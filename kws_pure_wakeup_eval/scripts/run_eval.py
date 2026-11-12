#!/usr/bin/env python3
"""纯唤醒语料（小沐小沐）原始 vs GTCRN 增强的双模型召回验证。

数据：
  raw: deploy/kws_dataset_1016_5/c100-01-<id>_raw.wav
  enh: gtcrn_enhanced_1016_5/c100-01-<id>_raw.wav

判决完全复用 kws_enhancement_opt/scripts/common.py 中的部署等价接口：
  SORT 0.84（连续 3 帧 + 500ms 冷却）、CTC -0.10（关键词路径 + 复位 + 冷却），
  30 ms chunk，每条音频重新初始化流式状态。
"""
import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/Users/lei/Documents/公牛沐光KWS')
OPT = ROOT / 'kws_enhancement_opt'
sys.path.insert(0, str(OPT / 'scripts'))
import common  # noqa: E402

RAW_DIR = ROOT / 'deploy' / 'kws_dataset_1016_5'
ENH_DIR = ROOT / 'gtcrn_enhanced_1016_5'
OUT_DIR = ROOT / 'kws_pure_wakeup_eval'
RES_DIR = OUT_DIR / 'results'

MODEL_PATHS = {'sort': common.SORT_MODEL, 'ctc': common.CTC_MODEL_INT8}

FIELDS = [
    'job_key', 'sid', 'filename', 'cond', 'model', 'chunk_ms', 'triggered',
    'num_triggers', 'first_trigger_ms', 'trigger_times_ms', 'trigger_scores',
    'max_score', 'blank_logprob_mean', 'num_frames', 'inference_ms',
    'duration_sec', 'rms', 'rms_db', 'peak', 'n_samples', 'samples_raw',
]


def discover_pairs():
    """按 id 配对 raw/enh，返回 [(id, raw_path, enh_path)]，缺失则抛错。"""
    pat = re.compile(r'^c100-01-(\d+)_raw\.wav$')
    raw = {}
    for p in RAW_DIR.iterdir():
        m = pat.match(p.name)
        if m:
            raw[int(m.group(1))] = p
    enh = {}
    for p in ENH_DIR.iterdir():
        m = pat.match(p.name)
        if m:
            enh[int(m.group(1))] = p
    common_ids = sorted(set(raw) & set(enh))
    only_raw = sorted(set(raw) - set(enh))
    only_enh = sorted(set(enh) - set(raw))
    if only_raw or only_enh:
        print(f'[warn] unpaired raw={only_raw[:10]} enh={only_enh[:10]}', flush=True)
    return [(i, raw[i], enh[i]) for i in common_ids], only_raw, only_enh


def job_key(model, cond, sid):
    return f'{model}|{cond}|{sid}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default='sort,ctc')
    ap.add_argument('--conds', default='raw,enh')
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--threads', type=int, default=0,
                    help='>0 时限制 ORT intra_op 线程（多进程并发建议 1）')
    ap.add_argument('--out', default='all.csv')
    args = ap.parse_args()

    if args.threads > 0:
        from run_jobs_cli import limit_onnx_threads
        limit_onnx_threads(args.threads)

    models = [m.strip() for m in args.models.split(',') if m.strip()]
    conds = [c.strip() for c in args.conds.split(',') if c.strip()]
    pairs, only_raw, only_enh = discover_pairs()
    pairs = pairs[args.shard::args.shards]
    if args.limit:
        pairs = pairs[:args.limit]

    RES_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = RES_DIR / args.out
    npz_path = RES_DIR / (Path(args.out).stem + '_frames.npz')

    done = set()
    if out_csv.exists():
        with open(out_csv, newline='', encoding='utf-8') as f:
            done = {r['job_key'] for r in csv.DictReader(f) if r.get('job_key')}
    frames = {}
    if npz_path.exists():
        with np.load(npz_path) as z:
            frames = {k: z[k] for k in z.files}
    exists = out_csv.exists()

    ev = common.eval_module().Evaluator(common.DEPLOY_DIR, models)
    n_jobs = len(pairs) * len(models) * len(conds)
    done_cnt = 0
    t0 = time.time()

    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not exists:
            w.writeheader()
        for sid, raw_path, enh_path in pairs:
            raw = common.load_wav(raw_path)
            enh = common.load_wav(enh_path)
            if len(raw) != len(enh):
                print(f'[warn] length mismatch id={sid} raw={len(raw)} enh={len(enh)}',
                      flush=True)
            for cond, wav, path in (('raw', raw, raw_path), ('enh', enh, enh_path)):
                if cond not in conds:
                    continue
                rms = common.rms(wav)
                for model in models:
                    key = job_key(model, cond, sid)
                    if key in done:
                        continue
                    t1 = time.perf_counter()
                    res = common.run_stream(ev, model, str(MODEL_PATHS[model]), wav,
                                            common.CHUNK_SAMPLES, keep_frames=True)
                    infer_ms = (time.perf_counter() - t1) * 1000.0
                    row = {
                        'job_key': key, 'sid': sid, 'filename': path.name,
                        'cond': cond, 'model': model,
                        'chunk_ms': common.CHUNK_SAMPLES * 1000 // common.SR,
                        'triggered': res['triggered'],
                        'num_triggers': res['num_triggers'],
                        'first_trigger_ms': res['first_trigger_ms'],
                        'trigger_times_ms': '|'.join(
                            f'{t:.1f}' for t in res['trigger_times_ms']),
                        'trigger_scores': '|'.join(
                            f'{s:.6f}' for s in res['trigger_scores']),
                        'max_score': round(res['max_score'], 6),
                        'blank_logprob_mean': (
                            round(res.get('blank_logprob_mean', float('nan')), 6)
                            if model == 'ctc' else ''),
                        'num_frames': res['num_frames'],
                        'inference_ms': round(infer_ms, 1),
                        'duration_sec': round(len(wav) / common.SR, 4),
                        'rms': round(float(rms), 8),
                        'rms_db': round(float(common.to_db(rms)), 3),
                        'peak': round(float(np.abs(wav).max()), 6),
                        'n_samples': len(wav), 'samples_raw': len(raw),
                    }
                    w.writerow(row)
                    frames[key] = (res['scores'] if model == 'sort'
                                   else res['logps'])
                    done_cnt += 1
                    if done_cnt % 100 == 0:
                        f.flush()
                        np.savez_compressed(npz_path, **frames)
                        el = time.time() - t0
                        print(f'  {done_cnt}/{n_jobs} {el:.0f}s '
                              f'({done_cnt / max(el, 1e-9):.1f}/s)', flush=True)

    np.savez_compressed(npz_path, **frames)
    meta = {
        'raw_dir': str(RAW_DIR), 'enh_dir': str(ENH_DIR),
        'n_pairs': len(pairs), 'models': models, 'conds': conds,
        'chunk_ms': common.CHUNK_SAMPLES * 1000 // common.SR,
        'sort_threshold': common.SORT_THR, 'ctc_threshold': common.CTC_THR,
        'unpaired_raw': only_raw, 'unpaired_enh': only_enh,
        'n_jobs': n_jobs, 'new_rows': done_cnt,
        'elapsed_sec': round(time.time() - t0, 1),
        'model_paths': {m: str(MODEL_PATHS[m]) for m in models},
    }
    (RES_DIR / 'run_metadata.json').write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'[done] +{done_cnt} rows in {meta["elapsed_sec"]}s -> {out_csv}',
          flush=True)


if __name__ == '__main__':
    main()
