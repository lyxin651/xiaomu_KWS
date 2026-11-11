#!/usr/bin/env python3
"""任务 B 分析：10 个增强输出候选在两个 KWS 上的 Recall / 误触发与配对变化。"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402


def main():
    rows = common.read_rows(common.RESULTS_DIR / 'candidates' / 'all.csv')
    labels = common.load_labels()
    sids = [r['sample_id'] for r in labels]
    y = {r['sample_id']: r['label'] for r in labels}
    npos = sum(1 for s in sids if y[s] == 1)

    by = {}
    for r in rows:
        r['triggered'] = int(r['triggered'])
        by.setdefault((r['cand'], r['model']), {})[int(r['sid'])] = r

    base = {}
    for model in ('sort', 'ctc'):
        base[model] = common.trigger_map(common.read_rows(
            common.BASE_RESULTS / f'{model}_enhanced.csv'))

    # 与主实验同源的 'enh' 重跑必须逐条一致
    consistency = {}
    for model in ('sort', 'ctc'):
        m = {s: int(by[('enh', model)][s]['triggered']) for s in sids}
        mism = [s for s in sids if m[s] != base[model].get(s, -1)]
        consistency[model] = {'mismatch_ids': mism, 'match': not mism}

    base_fp = {m: sum(1 for s in sids if y[s] == 0 and base[m].get(s, 0))
               for m in ('sort', 'ctc')}
    base_tp = {m: sum(1 for s in sids if y[s] == 1 and base[m].get(s, 0))
               for m in ('sort', 'ctc')}

    out_rows, summary = [], {'consistency_with_previous_run': consistency,
                             'candidates': {}, 'audio': {}}
    for cand in common.CANDIDATE_ORDER:
        entry = {'desc': common.CANDIDATE_SPECS[cand]['desc'],
                 'models': {}}
        for model in ('sort', 'ctc'):
            data = by[(cand, model)]
            flags = [int(data[s]['triggered']) for s in sids]
            m = common.metrics_from_flags([y[s] for s in sids], flags)
            base_flags = {s: base[model].get(s, 0) for s in sids}
            pos_lost = [s for s in sids if y[s] == 1 and base_flags[s] and not data[s]['triggered']]
            pos_rec = [s for s in sids if y[s] == 1 and not base_flags[s] and data[s]['triggered']]
            neg_new = [s for s in sids if y[s] == 0 and not base_flags[s] and data[s]['triggered']]
            neg_rem = [s for s in sids if y[s] == 0 and base_flags[s] and not data[s]['triggered']]
            pos_scores = [float(data[s]['max_score']) for s in sids if y[s] == 1]
            neg_scores = [float(data[s]['max_score']) for s in sids if y[s] == 0]
            entry['models'][model] = {
                **m,
                'num_triggers': int(sum(int(data[s]['num_triggers']) for s in sids)),
                'pos_lost_vs_baseline': pos_lost,
                'pos_recovered_vs_baseline': pos_rec,
                'neg_new_vs_baseline': neg_new,
                'neg_removed_vs_baseline': neg_rem,
                'median_max_score_pos': float(np.median(pos_scores)),
                'median_max_score_neg': float(np.median(neg_scores)),
                'recall_delta': m['recall'] - base_tp[model] / npos,
                'fp_delta': m['FP'] - base_fp[model],
            }
            out_rows.append({
                'candidate': cand, 'desc': entry['desc'], 'model': model,
                'TP': m['TP'], 'FN': m['FN'], 'FP': m['FP'], 'TN': m['TN'],
                'recall': round(m['recall'], 4), 'fpr': round(m['fpr'], 4),
                'num_triggers': entry['models'][model]['num_triggers'],
                'pos_lost': '|'.join(map(str, pos_lost)),
                'pos_recovered': '|'.join(map(str, pos_rec)),
                'neg_new': '|'.join(map(str, neg_new)),
                'neg_removed': '|'.join(map(str, neg_rem)),
                'median_max_score_pos': round(float(np.median(pos_scores)), 4),
            })
        peaks = [float(by[(cand, 'sort')][s]['audio_peak_out']) for s in sids]
        clips = [int(by[(cand, 'sort')][s]['audio_clip_samples']) for s in sids]
        gains = [float(by[(cand, 'sort')][s]['audio_gain_db']) for s in sids]
        base_rms = [float(by[('raw', 'sort')][s]['audio_rms_raw']) for s in sids]
        out_rms = [float(by[(cand, 'sort')][s]['audio_rms_out']) for s in sids]
        entry['audio'] = {
            'gain_db_median': round(float(np.median(gains)), 4),
            'gain_db_min': round(float(min(gains)), 4),
            'gain_db_max': round(float(max(gains)), 4),
            'files_with_clipping': int(sum(1 for c in clips if c > 0)),
            'clipped_samples_total': int(sum(clips)),
            'peak_max': float(max(peaks)),
            'median_rms_ratio_out_over_raw': round(
                float(np.median(np.array(out_rms) / np.maximum(base_rms, 1e-12))), 4),
            'length_check_ok': all(int(by[(cand, 'sort')][s]['audio_len_raw'])
                                   == int(by[(cand, 'sort')][s]['audio_len_enh'])
                                   for s in sids),
        }
        summary['candidates'][cand] = entry

    # 对齐与削波说明（原始配对 = 主实验已验证：长度一致、lag∈{-1,0} 样点）
    summary['alignment'] = {
        'all_pairs_same_length': all(
            int(by[('raw', 'sort')][s]['audio_len_raw'])
            == int(by[('enh', 'sort')][s]['audio_len_enh']) for s in sids),
        'source': 'kws_enhancement_eval/data_validation.json（lag_samples ∈ {-1,0}）',
    }

    # 候选排序：CTC recall 优先，约束 SORT FP 不劣于基线 34
    rank = []
    for cand in common.CANDIDATE_ORDER:
        e = summary['candidates'][cand]['models']
        ctc, srt = e['ctc'], e['sort']
        ok = srt['FP'] <= 34 and srt['TP'] >= 7
        rank.append({
            'candidate': cand,
            'ctc_recall': ctc['recall'], 'ctc_FP': ctc['FP'],
            'sort_recall': srt['recall'], 'sort_FP': srt['FP'],
            'sort_constraint_ok': ok,
            'score': (3 if ok else 0) + ctc['recall'] - 0.01 * max(0, srt['FP'] - 34),
        })
    rank.sort(key=lambda r: (-r['score'], -r['ctc_recall']))
    summary['ranking'] = rank

    common.write_csv(common.RESULTS_DIR / 'candidates' / 'candidate_summary.csv', out_rows)
    common.write_json(common.RESULTS_DIR / 'candidates' / 'summary.json', summary)

    for row in out_rows:
        print(f"{row['candidate']:>16s} {row['model']:>4s} TP={row['TP']:>2d} "
              f"FP={row['FP']:>3d} recall={row['recall']:.3f} "
              f"lost={row['pos_lost'] or '-':<10s} rec={row['pos_recovered'] or '-':<8s} "
              f"newFP={row['neg_new'] or '-'}")
    print('ranking:', [r['candidate'] for r in rank[:3]])


if __name__ == '__main__':
    main()
