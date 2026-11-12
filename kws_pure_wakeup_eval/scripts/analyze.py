#!/usr/bin/env python3
"""纯唤醒语料原始 vs 增强：召回、阈值曲线、逐条配对变化与图。"""
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path('/Users/lei/Documents/公牛沐光KWS')
OPT = ROOT / 'kws_enhancement_opt'
sys.path.insert(0, str(OPT / 'scripts'))
import common  # noqa: E402

OUT = ROOT / 'kws_pure_wakeup_eval'
RES = OUT / 'results'
FIG = OUT / 'figures'
REP = OUT / 'reports'

SORT_GRID = np.round(np.arange(0.30, 1.0001, 0.005), 4)
CTC_GRID = np.round(np.arange(-2.0, 1.0001, 0.05), 4)


def load_rows():
    with open(RES / 'all.csv', newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['sid'] = int(r['sid'])
        r['triggered'] = int(r['triggered'])
        r['num_triggers'] = int(r['num_triggers'])
        r['max_score'] = float(r['max_score'])
        r['rms_db'] = float(r['rms_db'])
        r['duration_sec'] = float(r['duration_sec'])
        r['first_trigger_ms'] = (float(r['first_trigger_ms'])
                                 if r['first_trigger_ms'] else None)
    return rows


def quantile(a, q):
    return float(np.percentile(np.asarray(a, dtype=np.float64), q))


def condition_stats(rows, model, cond):
    sub = [r for r in rows if r['model'] == model and r['cond'] == cond]
    trig = [r for r in sub if r['triggered']]
    lat = [r['first_trigger_ms'] for r in trig if r['first_trigger_ms'] is not None]
    scores = [r['max_score'] for r in sub]
    return {
        'n': len(sub), 'triggered': len(trig),
        'recall': round(len(trig) / max(len(sub), 1), 4),
        'num_triggers_total': sum(r['num_triggers'] for r in sub),
        'multi_trigger_files': sum(1 for r in sub if r['num_triggers'] > 1),
        'first_trigger_ms_median': round(float(np.median(lat)), 1) if lat else None,
        'first_trigger_ms_p90': round(quantile(lat, 90), 1) if lat else None,
        'max_score_median': round(float(np.median(scores)), 6),
        'max_score_p10': round(quantile(scores, 10), 6),
        'max_score_p90': round(quantile(scores, 90), 6),
    }


def paired(rows, model):
    raw = {r['sid']: r['triggered'] for r in rows
           if r['model'] == model and r['cond'] == 'raw'}
    enh = {r['sid']: r['triggered'] for r in rows
           if r['model'] == model and r['cond'] == 'enh'}
    ids = sorted(set(raw) & set(enh))
    lost = [i for i in ids if raw[i] and not enh[i]]
    rec = [i for i in ids if enh[i] and not raw[i]]
    both_miss = [i for i in ids if not raw[i] and not enh[i]]
    both_hit = [i for i in ids if raw[i] and enh[i]]
    return {'n_pairs': len(ids), 'both_hit': len(both_hit),
            'raw_only': len(lost), 'enh_only': len(rec),
            'both_miss': len(both_miss), 'lost_ids': lost,
            'recovered_ids': rec, 'both_miss_ids': both_miss}


def sweep(rows, frames, model):
    out = []
    grid = SORT_GRID if model == 'sort' else CTC_GRID
    matcher = common.ctc_module().KeywordPathMatcher if model == 'ctc' else None
    per_cond = {c: [(r['sid'], frames[f'{model}|{c}|{r["sid"]}'])
                    for r in rows if r['model'] == model and r['cond'] == c]
                for c in ('raw', 'enh')}
    for thr in grid:
        entry = {'threshold': float(thr)}
        for cond in ('raw', 'enh'):
            items = per_cond[cond]
            hit = 0
            for sid, arr in items:
                if model == 'sort':
                    ev = common.replay_sort(arr, float(thr))
                else:
                    ev = common.replay_ctc(arr, float(thr), matcher)
                hit += int(bool(ev))
            entry[f'recall_{cond}'] = round(hit / max(len(items), 1), 4)
            entry[f'hit_{cond}'] = hit
        out.append(entry)
    return out


def make_figures(rows, sweep_data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, model, grid in ((axes[0], 'sort', SORT_GRID),
                            (axes[1], 'ctc', CTC_GRID)):
        data = sweep_data[model]
        x = [d['threshold'] for d in data]
        for cond, style in (('raw', '-'), ('enh', '--')):
            y = [d[f'recall_{cond}'] for d in data]
            ax.plot(x, y, style, lw=2, label=f'{model.upper()} {cond}')
        default = common.SORT_THR if model == 'sort' else common.CTC_THR
        ax.axvline(default, color='gray', ls=':', lw=1)
        ax.set_title(f'{model.upper()} recall vs threshold (928 positives)')
        ax.set_xlabel('threshold')
        ax.grid(alpha=0.3)
        ax.legend(loc='lower right', fontsize=8)
    axes[0].set_ylabel('recall')
    fig.tight_layout()
    fig.savefig(FIG / 'fig_recall_vs_threshold.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, model in ((axes[0], 'sort'), (axes[1], 'ctc')):
        for cond, alpha in (('raw', 0.55), ('enh', 0.55)):
            s = [r['max_score'] for r in rows
                 if r['model'] == model and r['cond'] == cond]
            ax.hist(s, bins=60, alpha=alpha, label=cond, density=True)
        thr = common.SORT_THR if model == 'sort' else common.CTC_THR
        ax.axvline(thr, color='k', ls=':', lw=1, label='threshold')
        ax.set_title(f'{model.upper()} max score distribution')
        ax.set_xlabel('max score')
        ax.legend(fontsize=8)
    axes[0].set_ylabel('density')
    fig.tight_layout()
    fig.savefig(FIG / 'fig_score_distribution.png', dpi=160)
    plt.close(fig)


def main():
    rows = load_rows()
    frames = np.load(RES / 'all_frames.npz')
    FIG.mkdir(parents=True, exist_ok=True)
    REP.mkdir(parents=True, exist_ok=True)

    summary = {'data': {
        'n_pairs': len({r['sid'] for r in rows}),
        'n_files_evaluated': len(rows) // 4,
        'chunk_ms': 30,
        'sort_threshold': common.SORT_THR,
        'ctc_threshold': common.CTC_THR,
        'all_positive': True,
    }, 'models': {}}

    for model in ('sort', 'ctc'):
        summary['models'][model] = {
            'raw': condition_stats(rows, model, 'raw'),
            'enh': condition_stats(rows, model, 'enh'),
            'paired': paired(rows, model),
        }

    # 逐条配对明细
    with open(RES / 'per_file.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['sid', 'model', 'raw_triggered', 'enh_triggered',
                    'raw_max_score', 'enh_max_score', 'delta_max_score',
                    'raw_first_ms', 'enh_first_ms', 'raw_rms_db', 'enh_rms_db',
                    'rms_ratio_db', 'raw_num_triggers', 'enh_num_triggers',
                    'transition'])
        for model in ('sort', 'ctc'):
            by = {(r['sid'], r['cond']): r for r in rows if r['model'] == model}
            for sid in sorted({r['sid'] for r in rows}):
                a, b = by.get((sid, 'raw')), by.get((sid, 'enh'))
                if not a or not b:
                    continue
                if a['triggered'] and b['triggered']:
                    tr = 'hit->hit'
                elif a['triggered']:
                    tr = 'hit->miss'
                elif b['triggered']:
                    tr = 'miss->hit'
                else:
                    tr = 'miss->miss'
                w.writerow([sid, model, a['triggered'], b['triggered'],
                            a['max_score'], b['max_score'],
                            round(b['max_score'] - a['max_score'], 6),
                            a['first_trigger_ms'], b['first_trigger_ms'],
                            a['rms_db'], b['rms_db'],
                            round(b['rms_db'] - a['rms_db'], 3),
                            a['num_triggers'], b['num_triggers'], tr])

    sweep_data = {}
    for model in ('sort', 'ctc'):
        sweep_data[model] = sweep(rows, frames, model)
        with open(RES / f'threshold_recall_{model}.csv', 'w',
                  newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(sweep_data[model][0]))
            w.writeheader()
            w.writerows(sweep_data[model])

    # 电平与时长一致性
    raw_rms = {r['sid']: r['rms_db'] for r in rows
               if r['model'] == 'sort' and r['cond'] == 'raw'}
    enh_rms = {r['sid']: r['rms_db'] for r in rows
               if r['model'] == 'sort' and r['cond'] == 'enh'}
    rms_ratio = [enh_rms[s] - raw_rms[s]
                 for s in sorted(set(raw_rms) & set(enh_rms))]
    same_len = sum(1 for r in rows if r['n_samples'] == r['samples_raw'])
    summary['audio'] = {
        'len_pairs': len(rms_ratio),
        'rms_ratio_db_median': round(float(np.median(rms_ratio)), 2),
        'rms_ratio_db_p10': round(quantile(rms_ratio, 10), 2),
        'rms_ratio_db_p90': round(quantile(rms_ratio, 90), 2),
        'n_samples_equal': same_len,
        'n_rows': len(rows),
    }
    summary['threshold_curves'] = {
        model: {'raw': [d['recall_raw'] for d in sweep_data[model]],
                'enh': [d['recall_enh'] for d in sweep_data[model]],
                'grid': [d['threshold'] for d in sweep_data[model]]}
        for model in sweep_data
    }

    # 分数变化与电平变化的相关性、命中->漏检样本特征
    def spearman(a, b):
        from scipy.stats import spearmanr
        return spearmanr(a, b)

    for model in ('sort', 'ctc'):
        by = {(r['sid'], r['cond']): r for r in rows if r['model'] == model}
        sids = sorted({r['sid'] for r in rows})
        ds, dr, lost_rows = [], [], []
        for sid in sids:
            a, b = by.get((sid, 'raw')), by.get((sid, 'enh'))
            if not a or not b:
                continue
            ds.append(b['max_score'] - a['max_score'])
            dr.append(b['rms_db'] - a['rms_db'])
            if a['triggered'] and not b['triggered']:
                lost_rows.append({'sid': sid,
                                  'raw_score': a['max_score'],
                                  'enh_score': b['max_score'],
                                  'delta_score': round(b['max_score'] - a['max_score'], 4),
                                  'rms_ratio_db': round(b['rms_db'] - a['rms_db'], 2),
                                  'duration_sec': b['duration_sec']})
        rho, pval = spearman(ds, dr)
        summary['models'][model]['score_vs_level_spearman'] = {
            'rho': round(float(rho), 4), 'p': round(float(pval), 4)}
        summary['models'][model]['lost_detail'] = sorted(
            lost_rows, key=lambda x: x['delta_score'])[:40]
        summary['models'][model]['lost_rms_ratio_db_median'] = (
            round(float(np.median([x['rms_ratio_db'] for x in lost_rows])), 2)
            if lost_rows else None)
        summary['models'][model]['all_rms_ratio_db_median'] = round(
            float(np.median(dr)), 2)

    # 恢复到 raw 默认召回所需阈值
    for model in ('sort', 'ctc'):
        base = summary['models'][model]['raw']['recall']
        grid = sweep_data[model]
        need = None
        for d in grid:
            if d['recall_enh'] >= base:
                need = d['threshold']
        summary['models'][model]['threshold_enh_ge_raw_recall'] = need
        summary['models'][model]['recall_at_default'] = {
            'raw': summary['models'][model]['raw']['recall'],
            'enh': summary['models'][model]['enh']['recall']}

    # 综合：默认阈值下两个模型的联合结果
    for model in ('sort', 'ctc'):
        p = summary['models'][model]['paired']
        summary['models'][model]['delta_recall'] = round(
            summary['models'][model]['enh']['recall']
            - summary['models'][model]['raw']['recall'], 4)
        summary['models'][model]['delta_hits'] = (
            summary['models'][model]['enh']['triggered']
            - summary['models'][model]['raw']['triggered'])
        p['lost_ids'] = p['lost_ids']

    (RES / 'summary.json').write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')

    make_figures(rows, sweep_data)
    print(json.dumps(summary['models'], ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
