#!/usr/bin/env python3
"""任务 A2：CTC chunk 敏感性修正验证（INT8 vs FP32, 10/30/100 ms）。

修正点：旧脚本把 blank 分数当作关键词路径分数；此处统一用
KeywordPathMatcher 的实际路径分数（mean per-token log-LR, nats/token），
并同时保留 blank 均值用于说明两者差异。
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

COND_NAME = {'enh': 'GTCRN 增强', 'raw': '原始'}


def main():
    rows = common.read_rows(common.RESULTS_DIR / 'chunk' / 'all.csv')
    labels = common.load_labels()
    sids = [r['sample_id'] for r in labels]
    y = {r['sample_id']: r['label'] for r in labels}

    data = defaultdict(dict)   # (variant, chunk, cond) -> sid -> row
    for r in rows:
        data[(r['variant'], int(r['chunk_ms']), r['cand'])][int(r['sid'])] = r

    summary = {'configs': {}, 'deltas': {}, 'blank_vs_path': {}}
    for (variant, chunk, cond), d in sorted(data.items()):
        flags = [int(d[s]['triggered']) for s in sids]
        m = common.metrics_from_flags([y[s] for s in sids], flags)
        scores = [float(d[s]['max_score']) for s in sids]
        blanks = [float(d[s]['blank_logprob_mean']) for s in sids]
        summary['configs'][f'{variant}_{chunk}ms_{cond}'] = {
            'variant': variant, 'chunk_ms': chunk, 'condition': cond,
            **m,
            'num_triggers': int(sum(int(d[s]['num_triggers']) for s in sids)),
            'median_max_path_score': float(np.median(scores)),
            'median_blank_logprob_mean': float(np.median(blanks)),
            'trigger_ids': [s for s in sids if d[s]['triggered'] == '1'],
            'fn_ids': [s for s in sids if y[s] == 1 and d[s]['triggered'] == '0'],
        }

    # 与 (fp32, 30ms) 参考的分数差 / 触发翻转
    for cond in ('enh', 'raw'):
        ref_key = ('fp32', 30, cond)
        for variant in ('int8', 'fp32'):
            for chunk in (10, 30, 100):
                key = (variant, chunk, cond)
                if key not in data:
                    continue
                ref = data[ref_key]
                d = data[key]
                diffs = [abs(float(d[s]['max_score']) - float(ref[s]['max_score']))
                         for s in sids]
                blank_diffs = [abs(float(d[s]['blank_logprob_mean'])
                                   - float(ref[s]['blank_logprob_mean'])) for s in sids]
                flips = [s for s in sids if d[s]['triggered'] != ref[s]['triggered']]
                summary['deltas'][f'{variant}_{chunk}ms_{cond}'] = {
                    'ref': 'fp32_30ms', 'condition': cond,
                    'max_abs_path_score_diff': float(max(diffs)),
                    'mean_abs_path_score_diff': float(np.mean(diffs)),
                    'p95_abs_path_score_diff': float(np.percentile(diffs, 95)),
                    'mean_abs_blank_diff': float(np.mean(blank_diffs)),
                    'trigger_flip_ids': flips,
                    'n_trigger_flips': len(flips),
                }

    # INT8 自身的 chunk 效应（控制模型精度，只看 chunk 长度）
    for cond in ('enh', 'raw'):
        ref = data[('int8', 30, cond)]
        for chunk in (10, 100):
            d = data[('int8', chunk, cond)]
            diffs = [abs(float(d[s]['max_score']) - float(ref[s]['max_score'])) for s in sids]
            summary['deltas'][f'int8_{chunk}ms_vs_30ms_{cond}'] = {
                'ref': 'int8_30ms', 'condition': cond,
                'max_abs_path_score_diff': float(max(diffs)),
                'mean_abs_path_score_diff': float(np.mean(diffs)),
                'p95_abs_path_score_diff': float(np.percentile(diffs, 95)),
                'trigger_flip_ids': [s for s in sids
                                     if d[s]['triggered'] != ref[s]['triggered']],
                'n_trigger_flips': sum(1 for s in sids
                                       if d[s]['triggered'] != ref[s]['triggered']),
            }

    # blank 分数作为"关键词分数"会得出错误结论：给出直接对照
    for cond in ('enh', 'raw'):
        d = data[('fp32', 30, cond)]
        d10 = data[('fp32', 10, cond)]
        d100 = data[('fp32', 100, cond)]
        path_delta = [abs(float(d10[s]['max_score']) - float(d100[s]['max_score'])) for s in sids]
        blank_delta = [abs(float(d10[s]['blank_logprob_mean'])
                           - float(d100[s]['blank_logprob_mean'])) for s in sids]
        misrank = [s for s in sids
                   if (float(d10[s]['max_score']) >= -0.10) != (float(d100[s]['max_score']) >= -0.10)]
        summary['blank_vs_path'][cond] = {
            'fp32_10ms_vs_100ms_max_path_score_diff': float(max(path_delta)),
            'fp32_10ms_vs_100ms_blank_diff': float(max(blank_delta)),
            'fp32_trigger_disagreement_ids_10_vs_100':
                [s for s in sids if d10[s]['triggered'] != d100[s]['triggered']],
            'note': 'FP32 下路径分数与 chunk 长度无关；blank 均值随 chunk 轻微变化，'
                    '不能用 blank 分数替代路径分数做 chunk 对比',
            'misrank_ids': misrank,
        }

    common.write_json(common.RESULTS_DIR / 'chunk' / 'summary.json', summary)

    # 图：不同 chunk 下 max path score 的绝对偏差 + 触发数量
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, cond in zip(axes, ('enh', 'raw')):
        for variant, color in (('int8', '#c0392b'), ('fp32', '#2c3e50')):
            xs, ys = [30], [0.0]
            for chunk in (10, 100):
                k = f'{variant}_{chunk}ms_{cond}'
                if k in summary['deltas']:
                    xs.append(chunk)
                    ys.append(summary['deltas'][k]['max_abs_path_score_diff'])
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], 'o-', color=color, label=variant.upper())
        ax.set_xscale('log')
        ax.set_xticks([10, 30, 100])
        ax.set_xticklabels(['10 ms', '30 ms', '100 ms'])
        ax.set_xlabel('chunk length (ref = 30 ms, same model)')
        ax.set_ylabel('max |Δ keyword path score| (nats/token)')
        ax.set_title(f'{COND_NAME[cond]}: chunk sensitivity (corrected path score)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9)
    fig.tight_layout()
    out = common.FIG_DIR / 'fig_chunk_sensitivity.png'
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print('saved', out)

    for k, v in summary['deltas'].items():
        print(k, 'maxΔ=%.3f meanΔ=%.3f flips=%d' % (
            v['max_abs_path_score_diff'], v['mean_abs_path_score_diff'],
            v['n_trigger_flips']))


if __name__ == '__main__':
    main()
