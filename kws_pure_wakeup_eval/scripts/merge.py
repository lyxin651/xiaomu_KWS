#!/usr/bin/env python3
"""合并分片结果：part_*.csv / part_*_frames.npz -> all.csv / all_frames.npz。"""
import csv
import sys
from pathlib import Path

import numpy as np

RES = Path('/Users/lei/Documents/公牛沐光KWS/kws_pure_wakeup_eval/results')


def main():
    parts = sorted(RES.glob('part_*.csv'))
    if not parts:
        sys.exit('no part_*.csv')
    rows = []
    fields = None
    for p in parts:
        with open(p, newline='', encoding='utf-8') as f:
            rd = csv.DictReader(f)
            fields = fields or rd.fieldnames
            rows.extend(rd)
    uniq = {r['job_key']: r for r in rows}
    rows = sorted(uniq.values(),
                  key=lambda r: (r['model'], r['cond'], int(r['sid'])))
    out = RES / 'all.csv'
    with open(out, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    frames = {}
    for p in sorted(RES.glob('part_*_frames.npz')):
        with np.load(p) as z:
            for k in z.files:
                frames[k] = z[k]
    np.savez_compressed(RES / 'all_frames.npz', **frames)
    print(f'merged {len(parts)} parts -> {len(rows)} rows, {len(frames)} frame arrays')


if __name__ == '__main__':
    main()
