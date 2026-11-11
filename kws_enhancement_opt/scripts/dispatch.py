#!/usr/bin/env python3
"""实验调度：把任务按 shard 拆到独立解释器进程并行跑，再合并 CSV / 逐帧 npz。"""
import argparse
import csv
import os
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import experiments  # noqa: E402
import run_jobs_cli  # noqa: E402


def merge_csv(parts, out_path):
    rows = []
    for p in parts:
        if p.exists():
            with open(p, newline='', encoding='utf-8') as f:
                rows.extend(csv.DictReader(f))
    rows.sort(key=lambda r: (r['model'], r['variant'], r['cand'],
                             int(r['chunk_ms']), int(r['sid'])))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=run_jobs_cli.CSV_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, '') for k in run_jobs_cli.CSV_FIELDS})
    return len(rows)


def merge_npz(parts, out_path):
    frames = {}
    for p in parts:
        if p.exists():
            with np.load(p) as z:
                for k in z.files:
                    frames[k] = z[k]
    if frames:
        np.savez_compressed(out_path, **frames)
    return len(frames)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exp', required=True, choices=list(experiments.EXP_DIR))
    ap.add_argument('--shards', type=int, default=6)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--limit', type=int, default=None, help='仅跑前 N 个任务（调试）')
    ap.add_argument('--threads', type=int, default=1, help='每进程 ORT 线程数')
    args = ap.parse_args()

    labels = common.load_labels()
    jobs = experiments.build_jobs(args.exp, labels)
    if args.limit:
        jobs = jobs[:args.limit]
    exp_dir = common.RESULTS_DIR / experiments.EXP_DIR[args.exp]
    parts_dir = exp_dir / 'parts'
    logs_dir = exp_dir / 'logs'
    if args.force and parts_dir.exists():
        for p in parts_dir.iterdir():
            p.unlink()
    parts_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    shard_lists = [jobs[k::args.shards] for k in range(args.shards)]
    env = dict(os.environ)
    # 关键：多进程并发时 BLAS/ORT 默认线程池会互相自旋，实测慢 20x 以上
    for k in ('VECLIB_MAXIMUM_THREADS', 'OMP_NUM_THREADS',
              'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        env[k] = '1'
    procs = []
    t0 = time.time()
    for k, shard in enumerate(shard_lists):
        if not shard:
            continue
        jp = parts_dir / f'jobs_{k:02d}.json'
        with open(jp, 'w', encoding='utf-8') as f:
            json.dump(shard, f)
        log = open(logs_dir / f'shard_{k:02d}.log', 'w')
        cmd = [sys.executable, str(Path(__file__).parent / 'run_jobs_cli.py'),
               '--jobs', str(jp),
               '--csv', str(parts_dir / f'part_{k:02d}.csv'),
               '--npz', str(parts_dir / f'part_{k:02d}.npz'),
               '--threads', str(args.threads)]
        procs.append((k, subprocess.Popen(cmd, cwd=str(common.OPT_DIR), env=env,
                                          stdout=log, stderr=subprocess.STDOUT), log))
        print(f'[launch] shard {k:02d}: {len(shard)} jobs')

    failed = []
    for k, p, log in procs:
        code = p.wait()
        log.close()
        if code != 0:
            failed.append(k)
        print(f'[shard {k:02d}] exit={code}')
    if failed:
        print(f'[error] shards failed: {failed}', file=sys.stderr)
        sys.exit(1)

    parts_csv = sorted(parts_dir.glob('part_*.csv'))
    parts_npz = [p.with_suffix('.npz') for p in parts_csv]
    n = merge_csv(parts_csv, exp_dir / 'all.csv')
    m = merge_npz(parts_npz, exp_dir / 'frames.npz')
    print(f'[merged] {args.exp}: {n} rows, {m} frame series, '
          f'{time.time() - t0:.0f}s -> {exp_dir}')


if __name__ == '__main__':
    main()
