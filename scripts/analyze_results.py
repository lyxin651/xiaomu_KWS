#!/usr/bin/env python3
"""四组结果分析：混淆矩阵、配对变化、典型案例与图表、报告。

输入: labels.csv + results/{model}_{condition}.csv + results/frame_scores/*.npz
输出: results/paired_comparison.csv, results/summary.json,
      figures/*.png, figures/selected_cases/*.png, reports/evaluation_report.md
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

MODELS = ('sort', 'ctc')
CONDS = ('raw', 'enhanced')
RUN_LABEL = {'sort_raw': 'A SORT/raw', 'sort_enhanced': 'B SORT/enhanced',
             'ctc_raw': 'C CTC/raw', 'ctc_enhanced': 'D CTC/enhanced'}
TRANS_NEG = ['neg_none', 'neg_removed', 'neg_new', 'neg_both']
TRANS_POS = ['pos_both_miss', 'pos_lost', 'pos_recovered', 'pos_kept']


def wilson_ci(k, n, z=1.959963984540054):
    if n == 0:
        return (float('nan'), float('nan'))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar_p(b, c):
    """配对二分类的精确 McNemar（二项检验，双侧）。"""
    n = b + c
    if n == 0:
        return 1.0
    return float(binomtest(min(b, c), n, 0.5).pvalue)


def read_rows(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def mean(xs):
    return float(np.mean(xs)) if len(xs) else float('nan')


def metrics(rows):
    pos = [r for r in rows if int(r['label']) == 1]
    neg = [r for r in rows if int(r['label']) == 0]
    tp = sum(1 for r in pos if int(r['triggered']) == 1)
    fn = len(pos) - tp
    fp = sum(1 for r in neg if int(r['triggered']) == 1)
    tn = len(neg) - fp
    return {
        'n_positive': len(pos), 'n_negative': len(neg),
        'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn,
        'recall': tp / len(pos) if pos else float('nan'),
        'recall_ci95': wilson_ci(tp, len(pos)),
        'fpr': fp / len(neg) if neg else float('nan'),
        'fpr_ci95': wilson_ci(fp, len(neg)),
        'fp_ids': sorted(int(r['sample_id']) for r in neg if int(r['triggered']) == 1),
        'fn_ids': sorted(int(r['sample_id']) for r in pos if int(r['triggered']) == 0),
        'median_max_score_pos': float(np.median([float(r['frame_score_max']) for r in pos]))
        if pos else float('nan'),
        'median_max_score_neg_triggered': float(np.median(
            [float(r['frame_score_max']) for r in neg if int(r['triggered']) == 1]))
        if any(int(r['triggered']) == 1 for r in neg) else float('nan'),
        'median_max_score_neg_silent': float(np.median(
            [float(r['frame_score_max']) for r in neg if int(r['triggered']) == 0]))
        if any(int(r['triggered']) == 0 for r in neg) else float('nan'),
        'mean_inference_ms': mean([float(r['inference_time_ms']) for r in rows]),
        'num_triggers_total': int(sum(int(r['num_triggers']) for r in rows)),
    }


def transition(label, raw_t, enh_t):
    if label == 1:
        return {(1, 1): 'pos_kept', (1, 0): 'pos_lost',
                (0, 1): 'pos_recovered', (0, 0): 'pos_both_miss'}[(raw_t, enh_t)]
    return {(1, 0): 'neg_removed', (0, 1): 'neg_new',
            (1, 1): 'neg_both', (0, 0): 'neg_none'}[(raw_t, enh_t)]


def plot_confusion(mets, out):
    fig, axes = plt.subplots(2, 2, figsize=(10, 8.6))
    for i, model in enumerate(MODELS):
        for j, cond in enumerate(CONDS):
            m = mets[f'{model}_{cond}']
            ax = axes[i][j]
            mat = np.array([[m['TP'], m['FN']], [m['FP'], m['TN']]])
            ax.imshow(mat, cmap='Blues', vmin=0)
            ax.set_xticks([0, 1], ['triggered', 'not triggered'])
            ax.set_yticks([0, 1], ['positive', 'negative'])
            for (r, c), v in np.ndenumerate(mat):
                ax.text(c, r, str(v), ha='center', va='center', fontsize=16,
                        color='white' if v > mat.max() * 0.5 else 'black')
            ax.set_title(f'{RUN_LABEL[f"{model}_{cond}"]}\n'
                         f'Recall={m["recall"]:.3f}  FPR={m["fpr"]:.3f}', fontsize=11)
            if j == 0:
                ax.set_ylabel(f'{model.upper()}  ground truth')
            ax.set_xlabel('model decision')
    fig.suptitle('Confusion matrices (392 replayed clips: 21 wake / 371 false-wake)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_recall_fpr(mets, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    x = np.arange(len(MODELS))
    w = 0.36
    for ax, key, title in ((axes[0], 'recall', 'Wake-word recall (21 positives)'),
                           (axes[1], 'fpr', 'False-trigger rate (371 negatives)')):
        for k, cond in enumerate(CONDS):
            vals = [mets[f'{m}_{cond}'][key] for m in MODELS]
            errs = []
            for m in MODELS:
                lo, hi = mets[f'{m}_{cond}'][f'{key}_ci95']
                v = mets[f'{m}_{cond}'][key]
                errs.append([v - lo, hi - v])
            errs = np.array(errs).T
            ax.bar(x + (k - 0.5) * w, vals, w, yerr=errs, capsize=4,
                   label=cond, color=['#8c8c8c', '#2b7bba'][k])
            for xi, v in zip(x + (k - 0.5) * w, vals):
                ax.text(xi, v + (0.03 if key == 'fpr' else 0.012), f'{v:.3f}',
                        ha='center', fontsize=8)
        ax.set_xticks(x, [m.upper() for m in MODELS])
        ax.set_title(title)
        ax.set_ylim(0, 1.05)
        ax.grid(axis='y', alpha=0.3)
        ax.legend(title='audio')
    fig.suptitle('Enhanced vs raw audio (95% Wilson CI)', fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_transitions(paired, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, label, cats, title in (
            (axes[0], 0, TRANS_NEG, 'Negatives (371): trigger transition'),
            (axes[1], 1, TRANS_POS, 'Positives (21): trigger transition')):
        x = np.arange(len(cats))
        w = 0.36
        for k, model in enumerate(MODELS):
            counts = [sum(1 for p in paired
                          if p['model_name'] == model and p['label'] == label
                          and p['transition'] == c) for c in cats]
            bars = ax.bar(x + (k - 0.5) * w, counts, w, label=model.upper(),
                          color=['#4c72b0', '#dd8452'][k])
            for b, v in zip(bars, counts):
                ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v),
                        ha='center', fontsize=9)
        ax.set_xticks(x, [c.replace('neg_', '').replace('pos_', '') for c in cats])
        ax.set_title(title)
        ax.set_ylabel('samples')
        ax.grid(axis='y', alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def plot_case(case, score_dir, out_dir, name):
    """raw / enhanced 逐帧分数曲线 + 阈值 + 触发时刻。"""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    colors = {'raw': '#666666', 'enhanced': '#d62728'}
    threshold = float(case['threshold'])
    for cond in CONDS:
        npz = score_dir / f'{case["model_name"]}_{cond}.npz'
        with np.load(npz) as z:
            scores = z[str(case['sample_id'])]
        t = np.arange(len(scores)) * 10.0 / 1000.0
        ax.plot(t, scores, label=f'{cond} (max={scores.max():.3f})',
                color=colors[cond], lw=1.4)
        trig = case[f'{cond}_trigger_times']
        if trig:
            ax.axvline(trig[0] / 1000.0, color=colors[cond], ls=':', lw=1.2)
            ax.scatter([trig[0] / 1000.0], [scores.min()], marker='v',
                       color=colors[cond], zorder=5)
    ax.axhline(threshold, color='black', ls='--', lw=1, label=f'threshold={threshold}')
    ax.set_xlabel('time (s)')
    ax.set_ylabel(case['score_label'])
    ax.set_title(f'{case["model_name"].upper()}  id={case["sample_id"]}  '
                 f'label={case["label"]}  ({case["transition"]})')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / name, dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    out = Path(args.output_dir).resolve()
    res = out / 'results'
    fig_dir = out / 'figures'
    case_dir = fig_dir / 'selected_cases'
    case_dir.mkdir(parents=True, exist_ok=True)

    labels = {int(r['sample_id']): r for r in read_rows(out / 'labels.csv')}
    rows = {}
    warnings = []
    for model in MODELS:
        for cond in CONDS:
            p = res / f'{model}_{cond}.csv'
            rs = read_rows(p)
            bad = [r for r in rs if r['status'] != 'ok']
            if bad:
                warnings.append(f'{model}_{cond}: {len(bad)} failed samples')
            ids = {int(r['sample_id']) for r in rs if r['status'] == 'ok'}
            if ids != set(labels):
                warnings.append(f'{model}_{cond}: id set mismatch '
                                f'missing={sorted(set(labels)-ids)}')
            rows[f'{model}_{cond}'] = [r for r in rs if r['status'] == 'ok']

    mets = {k: metrics(v) for k, v in rows.items()}
    for k, v in mets.items():
        v['suppression_vs_raw'] = None

    # ---------- 配对对比 ----------
    paired = []
    counts = {}
    for model in MODELS:
        raw = {int(r['sample_id']): r for r in rows[f'{model}_raw']}
        enh = {int(r['sample_id']): r for r in rows[f'{model}_enhanced']}
        for sid in sorted(raw):
            rt, et = int(raw[sid]['triggered']), int(enh[sid]['triggered'])
            tr = transition(int(raw[sid]['label']), rt, et)
            counts[(model, tr)] = counts.get((model, tr), 0) + 1
            paired.append({
                'model_name': model, 'sample_id': sid, 'label': int(raw[sid]['label']),
                'raw_triggered': rt, 'enhanced_triggered': et, 'transition': tr,
                'raw_num_triggers': raw[sid]['num_triggers'],
                'enhanced_num_triggers': enh[sid]['num_triggers'],
                'raw_first_trigger_time_ms': raw[sid]['first_trigger_time_ms'],
                'enhanced_first_trigger_time_ms': enh[sid]['first_trigger_time_ms'],
                'raw_frame_score_max': raw[sid]['frame_score_max'],
                'enhanced_frame_score_max': enh[sid]['frame_score_max'],
                'raw_threshold': raw[sid]['threshold'],
                'enhanced_threshold': enh[sid]['threshold'],
            })
        fp_raw, fp_enh = mets[f'{model}_raw']['FP'], mets[f'{model}_enhanced']['FP']
        mets[f'{model}_raw']['suppression_vs_raw'] = None
        mets[f'{model}_enhanced']['suppression_vs_raw'] = (
            (fp_raw - fp_enh) / fp_raw if fp_raw else None)

    with open(res / 'paired_comparison.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(paired[0].keys()))
        w.writeheader()
        w.writerows(paired)

    # ---------- 显著性 ----------
    stats = {}
    for model in MODELS:
        raw = mets[f'{model}_raw']
        enh = mets[f'{model}_enhanced']
        neg_raw = {p['sample_id']: p['raw_triggered'] for p in paired
                   if p['model_name'] == model and p['label'] == 0}
        neg_enh = {p['sample_id']: p['enhanced_triggered'] for p in paired
                   if p['model_name'] == model and p['label'] == 0}
        pos_raw = {p['sample_id']: p['raw_triggered'] for p in paired
                   if p['model_name'] == model and p['label'] == 1}
        pos_enh = {p['sample_id']: p['enhanced_triggered'] for p in paired
                   if p['model_name'] == model and p['label'] == 1}
        b_neg = sum(1 for s in neg_raw if neg_raw[s] == 0 and neg_enh[s] == 1)
        c_neg = sum(1 for s in neg_raw if neg_raw[s] == 1 and neg_enh[s] == 0)
        b_pos = sum(1 for s in pos_raw if pos_raw[s] == 0 and pos_enh[s] == 1)
        c_pos = sum(1 for s in pos_raw if pos_raw[s] == 1 and pos_enh[s] == 0)
        stats[model] = {
            'recall_raw': raw['recall'], 'recall_enhanced': enh['recall'],
            'recall_delta_pp': (enh['recall'] - raw['recall']) * 100,
            'fpr_raw': raw['fpr'], 'fpr_enhanced': enh['fpr'],
            'fpr_delta_pp': (enh['fpr'] - raw['fpr']) * 100,
            'fpr_relative_change': ((enh['fpr'] - raw['fpr']) / raw['fpr']
                                    if raw['fpr'] else None),
            'recall_relative_change': ((enh['recall'] - raw['recall']) / raw['recall']
                                       if raw['recall'] else None),
            'suppression': ((raw['FP'] - enh['FP']) / raw['FP']) if raw['FP'] else None,
            'neg_new_triggers': b_neg, 'neg_removed_triggers': c_neg,
            'neg_mcnemar_p': mcnemar_p(b_neg, c_neg),
            'pos_recovered': b_pos, 'pos_lost': c_pos,
            'pos_mcnemar_p': mcnemar_p(b_pos, c_pos),
        }

    # ---------- 交叉模型差异 ----------
    by = {}
    for p in paired:
        by[(p['model_name'], p['sample_id'])] = p
    cross = {'sort_improved_ctc_worse': [], 'sort_worse_ctc_improved': [],
             'both_improved': [], 'both_worse': [], 'pos_sort_lost_ctc_kept': [],
             'pos_sort_kept_ctc_lost': [], 'pos_both_lost': []}
    for sid in sorted(labels):
        s = by[('sort', sid)]
        c = by[('ctc', sid)]
        if s['label'] == 0:
            s_imp = (s['raw_triggered'], s['enhanced_triggered']) == (1, 0)
            c_imp = (c['raw_triggered'], c['enhanced_triggered']) == (1, 0)
            s_wor = (s['raw_triggered'], s['enhanced_triggered']) == (0, 1)
            c_wor = (c['raw_triggered'], c['enhanced_triggered']) == (0, 1)
            if s_imp and c_wor:
                cross['sort_improved_ctc_worse'].append(sid)
            if c_imp and s_wor:
                cross['sort_worse_ctc_improved'].append(sid)
            if s_imp and c_imp:
                cross['both_improved'].append(sid)
            if s_wor and c_wor:
                cross['both_worse'].append(sid)
        else:
            s_lost = (s['raw_triggered'], s['enhanced_triggered']) == (1, 0)
            c_lost = (c['raw_triggered'], c['enhanced_triggered']) == (1, 0)
            if s_lost and not c_lost:
                cross['pos_sort_lost_ctc_kept'].append(sid)
            if c_lost and not s_lost:
                cross['pos_sort_kept_ctc_lost'].append(sid)
            if s_lost and c_lost:
                cross['pos_both_lost'].append(sid)

    # ---------- 典型案例 ----------
    def pick(model, trans, key='raw_frame_score_max', reverse=True):
        cand = [p for p in paired if p['model_name'] == model and p['transition'] == trans
                and p[f'{key.split("_")[0]}_frame_score_max'] != '']
        if not cand:
            return None
        return sorted(cand, key=lambda p: float(p[key]), reverse=reverse)[0]

    cases = []
    for model in MODELS:
        cases.append((f'{model}_neg_removed', model, 'neg_removed', 'raw_frame_score_max'))
        cases.append((f'{model}_neg_new', model, 'neg_new', 'enhanced_frame_score_max'))
        cases.append((f'{model}_pos_lost', model, 'pos_lost', 'raw_frame_score_max'))
        cases.append((f'{model}_pos_recovered', model, 'pos_recovered',
                      'enhanced_frame_score_max'))

    case_files, case_info = [], {}
    for name, model, trans, key in cases:
        p = pick(model, trans, key)
        if p is None:
            case_info[name] = None
            continue
        case = dict(p)
        case['threshold'] = p['raw_threshold']
        case['score_label'] = ('frame wake probability' if model == 'sort'
                               else 'keyword path score (nats/token)')
        case['raw_trigger_times'] = ([] if not p['raw_triggered']
                                     else [float(p['raw_first_trigger_time_ms'])])
        case['enhanced_trigger_times'] = ([] if not p['enhanced_triggered']
                                          else [float(p['enhanced_first_trigger_time_ms'])])
        fn = f'{name}_id{p["sample_id"]}.png'
        plot_case(case, res / 'frame_scores', case_dir, fn)
        case_files.append(str(Path('selected_cases') / fn))
        case_info[name] = {
            'sample_id': p['sample_id'], 'label': p['label'],
            'raw_triggered': p['raw_triggered'], 'enhanced_triggered': p['enhanced_triggered'],
            'raw_frame_score_max': float(p['raw_frame_score_max']),
            'enhanced_frame_score_max': float(p['enhanced_frame_score_max']),
            'raw_path': labels[p['sample_id']]['raw_path'],
            'enhanced_path': labels[p['sample_id']]['enhanced_path'],
            'figure': fn,
        }

    # ---------- 图表 ----------
    plot_confusion(mets, fig_dir / 'confusion_matrices.png')
    plot_recall_fpr(mets, fig_dir / 'recall_fpr_comparison.png')
    plot_transitions(paired, fig_dir / 'paired_transitions.png')

    # ---------- summary.json ----------
    prev_summary = {}
    if (res / 'summary.json').exists():
        try:
            prev_summary = json.loads((res / 'summary.json').read_text(encoding='utf-8'))
        except json.JSONDecodeError:
            prev_summary = {}
    dv_path = out / 'data_validation.json'
    data_validation = (json.loads(dv_path.read_text(encoding='utf-8'))
                       if dv_path.exists() else {})
    summary = {
        'config': {
            'sample_rate': 16000, 'chunk_ms': 30,
            'sort': {'model': 'kws_cmvn_g1_e100.onnx', 'threshold': 0.84,
                     'min_trigger_frames': 3, 'score_smooth': 3, 'cooldown_ms': 500},
            'ctc': {'model': 'kws_ctc_v2verifier_e7_int8.onnx', 'threshold': -0.10,
                    'cooldown_ms': 500},
            'dataset': {'paired_samples': len(labels),
                        'positives': sum(1 for v in labels.values() if int(v['label']) == 1),
                        'negatives': sum(1 for v in labels.values() if int(v['label']) == 0)},
            'note_fpr': 'FPR 是 371 条回采负样本上的触发比例，不是 FA/h',
        },
        'data_validation_summary': {k: v for k, v in data_validation.items()
                                    if k not in ('per_file', 'anomalies')},
        'run_inference_stats': (prev_summary.get('runs')
                                or prev_summary.get('run_inference_stats', {})),
        'metrics': mets,
        'deltas': stats,
        'paired_counts': {f'{m}|{k}': v for (m, k), v in sorted(counts.items())},
        'paired_ids': {f'{m}|{k}': sorted(p['sample_id'] for p in paired
                                          if p['model_name'] == m and p['transition'] == k)
                       for m in MODELS for k in TRANS_NEG + TRANS_POS},
        'cross_model': cross,
        'cases': case_info,
        'warnings': warnings,
        'row_counts': {k: len(v) for k, v in rows.items()},
        'paired_rows': len(paired),
    }
    (res / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    print(json.dumps({k: summary[k] for k in
                      ('row_counts', 'paired_counts', 'deltas', 'cross_model', 'warnings')},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
