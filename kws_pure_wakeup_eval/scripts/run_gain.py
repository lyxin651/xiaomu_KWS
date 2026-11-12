#!/usr/bin/env python3
"""纯唤醒语料上测试增强输出的电平补偿（+2/+4 dB、逐条匹配 raw RMS）对 CTC 召回的影响。"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path('/Users/lei/Documents/公牛沐光KWS')
OPT = ROOT / 'kws_enhancement_opt'
sys.path.insert(0, str(OPT / 'scripts'))
import common  # noqa: E402
from run_eval import MODEL_PATHS, discover_pairs  # noqa: E402

RES = ROOT / 'kws_pure_wakeup_eval' / 'results'
CONDS = ['enh_gain_p2', 'enh_gain_p4', 'enh_rms_matched']
FIELDS = ['job_key', 'sid', 'cond', 'model', 'triggered', 'num_triggers',
          'first_trigger_ms', 'max_score', 'rms_db', 'rms_ratio_db',
          'gain_db', 'clip_samples', 'duration_sec']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--shards', type=int, default=1)
    ap.add_argument('--threads', type=int, default=1)
    ap.add_argument('--out', default='gain_ctc.csv')
    args = ap.parse_args()

    if args.threads > 0:
        from run_jobs_cli import limit_onnx_threads
        limit_onnx_threads(args.threads)

    pairs, _, _ = discover_pairs()
    pairs = pairs[args.shard::args.shards]
    out_csv = RES / args.out
    model = 'ctc'
    ev = common.eval_module().Evaluator(common.DEPLOY_DIR, [model])

    done = set()
    if out_csv.exists():
        with open(out_csv, newline='', encoding='utf-8') as f:
            done = {r['job_key'] for r in csv.DictReader(f)}
    t0 = time.time()
    n_new = 0
    with open(out_csv, 'a', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if not out_csv.exists():
            w.writeheader()
        for sid, raw_path, enh_path in pairs:
            raw = common.load_wav(raw_path)
            enh = common.load_wav(enh_path)
            for cond in CONDS:
                key = f'{model}|{cond}|{sid}'
                if key in done:
                    continue
                audio, info = common.build_candidate_audio(cond, raw, enh)
                res = common.run_stream(ev, model, str(MODEL_PATHS[model]),
                                        audio, common.CHUNK_SAMPLES)
                w.writerow({
                    'job_key': key, 'sid': sid, 'cond': cond, 'model': model,
                    'triggered': res['triggered'],
                    'num_triggers': res['num_triggers'],
                    'first_trigger_ms': res['first_trigger_ms'],
                    'max_score': round(res['max_score'], 6),
                    'rms_db': round(float(common.to_db(common.rms(audio))), 3),
                    'rms_ratio_db': round(float(common.to_db(common.rms(audio))
                                                - common.to_db(common.rms(raw))), 3),
                    'gain_db': round(info['gain_db'], 3),
                    'clip_samples': info['clip_samples'],
                    'duration_sec': round(len(audio) / common.SR, 4),
                })
                n_new += 1
                if n_new % 200 == 0:
                    f.flush()
                    print(f'  {n_new} {time.time()-t0:.0f}s', flush=True)
    print(f'[done] +{n_new} rows in {time.time()-t0:.0f}s -> {out_csv}')


if __name__ == '__main__':
    main()
