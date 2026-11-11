#!/usr/bin/env python3
"""任务 A1：阈值扫描与 Recall–误触发诊断（只归因，不改 KWS 模型）。

输入：results/sweep/all.csv + frames.npz（由 dispatch.py --exp sweep 产生）
输出：results/sweep/sweep_<model>_<cond>.csv、sweep_summary.json、figures/*.png
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SORT_GRID = np.round(np.arange(0.30, 1.00001, 0.005), 3)
CTC_GRID = np.round(np.arange(-2.0, 1.0001, 0.05), 3)


def sort_replay(prob, thr, cool=50, k=3):
    if len(prob) < k:
        return []
    win = np.min(np.stack([prob[k - 1 - j:len(prob) - j] for j in range(k)]), axis=0)
    idx = np.flatnonzero(win >= thr) + (k - 1)
    out, last = [], -10 ** 9
    for i in idx:
        if i >= last + cool:
            out.append(int(i))
            last = int(i)
    return out


def ctc_replay(logps, thr, matcher_cls, cool=50):
    m = matcher_cls()
    cooldown, times = 0, []
    for i, lp in enumerate(logps):
        cooldown = max(0, cooldown - 1)
        m.push(lp)
        if m.score >= thr and cooldown == 0:
            times.append(int(i))
            cooldown = cool
            m.reset()
    return times


def sweep_grid(labels, per_sample, grid, replay_fn):
    rows = []
    for thr in grid:
        flags, total = [], 0
        for sid, y in labels:
            times = replay_fn(per_sample[sid], float(thr))
            flags.append(1 if times else 0)
            total += len(times)
        m = common.metrics_from_flags([y for _, y in labels], flags)
        m.update({'threshold': float(thr), 'num_triggers': total})
        rows.append(m)
    return rows


def main():
    labels = common.load_labels()
    pairs = [(r['sample_id'], r['label']) for r in labels]
    sids = [s for s, _ in pairs]
    y = dict(pairs)
    frames_path = common.RESULTS_DIR / 'sweep' / 'frames.npz'
    with np.load(frames_path) as z:
        frames = {k: z[k] for k in z.files}
    matcher_cls = common.ctc_module().KeywordPathMatcher

    # 触发标记（默认阈值）与主实验结果对照
    base = {}
    for model in ('sort', 'ctc'):
        for cond in ('raw', 'enh'):
            p = common.BASE_RESULTS / f'{model}_{"enhanced" if cond == "enh" else "raw"}.csv'
            base[f'{model}_{cond}'] = common.trigger_map(common.read_rows(p))

    summary = {
        'threshold_grid': {
            'sort': [float(SORT_GRID[0]), float(SORT_GRID[-1]), 0.005],
            'ctc': [float(CTC_GRID[0]), float(CTC_GRID[-1]), 0.05]},
        'default_thresholds': {'sort': common.SORT_THR, 'ctc': common.CTC_THR},
        'validation': {}, 'separability': {}, 'conditions': {},
    }

    for model in ('sort', 'ctc'):
        grid = SORT_GRID if model == 'sort' else CTC_GRID
        thr0 = common.SORT_THR if model == 'sort' else common.CTC_THR
        for cond in ('raw', 'enh'):
            data = {sid: frames[f'{model}||{cond}|{sid}|30'] for sid in sids}
            replay = ((lambda p, t: sort_replay(p, t)) if model == 'sort'
                      else (lambda p, t: ctc_replay(p, t, matcher_cls)))
            rows = sweep_grid(pairs, data, grid, replay)
            common.write_csv(common.RESULTS_DIR / 'sweep' / f'sweep_{model}_{cond}.csv',
                             rows, ['threshold', 'TP', 'FN', 'FP', 'TN',
                                    'n_pos', 'n_neg', 'recall', 'fpr', 'num_triggers'])

            default_flags = {sid: 1 if replay(data[sid], thr0) else 0 for sid in sids}
            ref = base[f'{model}_{cond}']
            mism = [s for s in sids if int(ref.get(s, -1)) != default_flags[s]]
            summary['validation'][f'{model}_{cond}'] = {
                'default_threshold': thr0,
                'replay_triggered': int(sum(default_flags.values())),
                'baseline_triggered': int(sum(ref.get(s, 0) for s in sids)),
                'mismatch_ids': mism, 'match': not mism,
            }

            if model == 'sort':
                maxs = {s: float(np.max(data[s])) for s in sids}
                stat = 'max_frame_prob'
            else:
                maxs = {s: common.replay_ctc_maxscore(data[s], matcher_cls) for s in sids}
                stat = 'max_path_score_no_reset'
            pos = [maxs[s] for s in sids if y[s] == 1]
            neg = [maxs[s] for s in sids if y[s] == 0]
            summary['separability'][f'{model}_{cond}'] = {
                'stat': stat, 'auc': common.auc(pos, neg),
                'pos_median': float(np.median(pos)), 'neg_median': float(np.median(neg)),
                'pos_p10': float(np.percentile(pos, 10)),
                'neg_p90': float(np.percentile(neg, 90)),
                'neg_p99': float(np.percentile(neg, 99)),
                'max_scores': {int(s): round(maxs[s], 4) for s in sids},
            }
            ops = {}
            for fp_budget in (0, 1, 3, 5, 10, 14, 20, 34, 50, 102):
                cand = [r for r in rows if r['FP'] <= fp_budget]
                if cand:
                    b = max(cand, key=lambda r: (r['recall'], -r['FP']))
                    ops[str(fp_budget)] = {'threshold': b['threshold'],
                                           'recall': b['recall'],
                                           'TP': b['TP'], 'FP': b['FP']}
            summary['conditions'][f'{model}_{cond}'] = {
                'default': next(({'recall': r['recall'], 'FP': r['FP'], 'TP': r['TP']}
                                 for r in rows if abs(r['threshold'] - thr0) < 1e-9), None),
                'max_recall': max(r['recall'] for r in rows),
                'operating_points_by_fp_budget': ops,
            }
            print(f'[sweep] {model}/{cond}: match={not mism} '
                  f'auc={summary["separability"][f"{model}_{cond}"]["auc"]:.3f}')

    common.write_json(common.RESULTS_DIR / 'sweep' / 'sweep_summary.json', summary)


if __name__ == '__main__':
    main()
