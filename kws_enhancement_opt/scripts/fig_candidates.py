#!/usr/bin/env python3
"""任务 B 图：候选方案 Recall–误触发散点 + 关键样本分数热图。"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

SHORT = {
    'enh': 'enh (baseline)', 'enh_gain_p1': '+1 dB', 'enh_gain_p2': '+2 dB',
    'enh_gain_p3': '+3 dB', 'enh_gain_p4': '+4 dB', 'mix_a090': 'mix 0.90',
    'mix_a075': 'mix 0.75', 'mix_a050': 'mix 0.50',
    'raw_rms_to_enh': 'raw → enh RMS', 'enh_rms_matched': 'enh → raw RMS',
    'raw': 'raw',
}
COLOR = {'base': '#7f8c8d', 'gain': '#c0392b', 'mix': '#2e86c1',
         'control': '#27ae60', 'estim': '#8e44ad'}


def color_of(c):
    if c == 'enh':
        return COLOR['base']
    if 'gain' in c:
        return COLOR['gain']
    if c.startswith('mix'):
        return COLOR['mix']
    if c == 'enh_rms_matched':
        return COLOR['estim']
    return COLOR['control']


def fig_tradeoff():
    s = json.load(open(common.RESULTS_DIR / 'candidates' / 'summary.json'))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, model in zip(axes, ('sort', 'ctc')):
        for j, cand in enumerate(common.CANDIDATE_ORDER):
            m = s['candidates'][cand]['models'][model]
            ax.scatter(m['FP'], m['recall'] * 100, s=95, color=color_of(cand),
                       edgecolor='k', linewidth=0.5, zorder=3)
            dy = 6 if j % 2 == 0 else -12
            ax.annotate(SHORT[cand], (m['FP'], m['recall'] * 100),
                        textcoords='offset points', xytext=(6, dy), fontsize=7.5)
        ax.set_xlabel('false triggers (371 negatives)')
        ax.set_ylabel('recall (%)')
        ax.set_title(f'{model.upper()}: 30 ms chunk, fixed threshold '
                     f'({0.84 if model == "sort" else -0.10})')
        ax.grid(alpha=0.25)
        ax.set_xlim(-1.5, None)
    fig.tight_layout()
    out = common.FIG_DIR / 'fig_candidates_tradeoff.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


def fig_case_heatmap():
    rows = common.read_rows(common.RESULTS_DIR / 'candidates' / 'all.csv')
    cases = [7, 25, 122, 123, 125, 121, 119, 32]
    models = ('ctc', 'sort')
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    for ax, model in zip(axes, models):
        mat = np.full((len(cases), len(common.CANDIDATE_ORDER)), np.nan)
        for r in rows:
            if r['model'] != model or int(r['sid']) not in cases:
                continue
            i = cases.index(int(r['sid']))
            j = common.CANDIDATE_ORDER.index(r['cand'])
            mat[i, j] = float(r['max_score'])
        vmax = np.nanmax(np.abs(mat))
        im = ax.imshow(mat, aspect='auto', cmap='RdYlGn', vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(common.CANDIDATE_ORDER)))
        ax.set_xticklabels([SHORT[c] for c in common.CANDIDATE_ORDER],
                           rotation=45, ha='right', fontsize=8)
        ax.set_yticks(range(len(cases)))
        ax.set_yticklabels([f'id={c}' for c in cases])
        thr = 0.84 if model == 'sort' else -0.10
        for i in range(len(cases)):
            for j in range(len(common.CANDIDATE_ORDER)):
                v = mat[i, j]
                if np.isnan(v):
                    continue
                ax.text(j, i, f'{v:.2f}', ha='center', va='center', fontsize=6.5,
                        color='k')
        ax.set_title(f'{model.upper()} max score (threshold {thr})')
        fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    out = common.FIG_DIR / 'fig_case_candidate_scores.png'
    fig.savefig(out, dpi=160)
    plt.close(fig)
    return out


if __name__ == '__main__':
    for p in (fig_tradeoff(), fig_case_heatmap()):
        print('saved', p)
