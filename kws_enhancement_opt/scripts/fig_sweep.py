#!/usr/bin/env python3
"""任务 A 图：Recall–误触发曲线、分数分布、阈值-指标曲线。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

COLORS = {'raw': '#7f8c8d', 'enh': '#c0392b'}
NAMES = {'raw': 'raw audio', 'enh': 'GTCRN enhanced'}


def fig_recall_fp():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for ax, model in zip(axes, ('sort', 'ctc')):
        for cond in ('raw', 'enh'):
            rows = common.read_rows(common.RESULTS_DIR / 'sweep' / f'sweep_{model}_{cond}.csv')
            fp = np.array([int(r['FP']) for r in rows])
            rec = np.array([float(r['recall']) for r in rows])
            order = np.argsort(fp)
            ax.plot(fp[order], rec[order] * 100, color=COLORS[cond], lw=1.8,
                    label=NAMES[cond])
            thr0 = common.SORT_THR if model == 'sort' else common.CTC_THR
            thrs = np.array([float(r['threshold']) for r in rows])
            i = int(np.argmin(np.abs(thrs - thr0)))
            ax.scatter([fp[i]], [rec[i] * 100], color=COLORS[cond], s=55, zorder=5,
                       edgecolor='k', linewidth=0.6)
            ax.annotate(f'thr={thr0:g}\n{int(fp[i])} FP, {rec[i]*100:.0f}%',
                        (fp[i], rec[i] * 100), textcoords='offset points',
                        xytext=(8, -14), fontsize=8, color=COLORS[cond])
        ax.set_xlabel('false triggers on 371 negatives')
        ax.set_ylabel('recall on 21 positives (%)')
        ax.set_title(f'{model.upper()} threshold sweep (30 ms chunk)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc='lower right')
        ax.set_ylim(-2, 92)
    fig.tight_layout()
    out = common.FIG_DIR / 'fig_sweep_recall_fp.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_threshold_metrics():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, model in zip(axes, ('sort', 'ctc')):
        for cond in ('raw', 'enh'):
            rows = common.read_rows(common.RESULTS_DIR / 'sweep' / f'sweep_{model}_{cond}.csv')
            thr = [float(r['threshold']) for r in rows]
            rec = [float(r['recall']) * 100 for r in rows]
            ax.plot(thr, rec, color=COLORS[cond], lw=1.7, label=f'{NAMES[cond]} recall')
        ax.set_xlabel('threshold')
        ax.set_ylabel('recall (%)')
        ax2 = ax.twinx()
        for cond in ('raw', 'enh'):
            rows = common.read_rows(common.RESULTS_DIR / 'sweep' / f'sweep_{model}_{cond}.csv')
            thr = [float(r['threshold']) for r in rows]
            fp = [int(r['FP']) for r in rows]
            ax2.plot(thr, fp, color=COLORS[cond], lw=1.0, ls='--', alpha=0.75,
                     label=f'{NAMES[cond]} FP')
        ax2.set_ylabel('false triggers (dashed)')
        ax.set_title(f'{model.upper()}: recall & FP vs threshold')
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=7, loc='center right')
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out = common.FIG_DIR / 'fig_sweep_threshold_metrics.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_score_dist():
    s = json.load(open(common.RESULTS_DIR / 'sweep' / 'sweep_summary.json'))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    labels = common.load_labels()
    for ax, model in zip(axes, ('sort', 'ctc')):
        for cond in ('raw', 'enh'):
            maxs = s['separability'][f'{model}_{cond}']['max_scores']
            pos = [maxs[str(r['sample_id'])] for r in labels if r['label'] == 1]
            neg = [maxs[str(r['sample_id'])] for r in labels if r['label'] == 0]
            x = np.linspace(min(neg + pos) - 0.2, max(neg + pos) + 0.2, 200)
            for vals, ls in ((pos, '-'), (neg, '--')):
                hist, edges = np.histogram(vals, bins=30, density=True)
                ax.plot(0.5 * (edges[1:] + edges[:-1]), hist, color=COLORS[cond],
                        ls=ls, lw=1.2, alpha=0.85,
                        label=f'{NAMES[cond]} {"pos" if ls == "-" else "neg"}')
        ax.set_xlabel('per-file max detection score')
        ax.set_ylabel('density')
        ax.set_title(f'{model.upper()} score separability '
                     f'(AUC raw={s["separability"][f"{model}_raw"]["auc"]:.2f}, '
                     f'enh={s["separability"][f"{model}_enh"]["auc"]:.2f})')
        ax.legend(fontsize=6.5, loc='upper left')
        ax.grid(alpha=0.25)
    fig.tight_layout()
    out = common.FIG_DIR / 'fig_sweep_score_dist.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == '__main__':
    for p in (fig_recall_fp(), fig_threshold_metrics(), fig_score_dist()):
        print('saved', p)
