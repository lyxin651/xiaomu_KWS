#!/usr/bin/env python3
"""汇总 results/ 下的逐条结果：Recall / 误触发率 / 配对变化 / McNemar 检验。

输入（由 scripts/run_matrix.py 生成）:
    results/<dataset>/<model>__<method>.csv

输出:
    results/summary.json       机器可读汇总
    results/paired_samples.csv 原始 vs 各增强方法的逐样本配对
    results/report.md          人读报告（Markdown）

用法:
    python3 scripts/analyze_results.py [--datasets ...] [--models ...] [--methods ...]
"""
import argparse
import csv
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / 'results'
BUCKET = ['tp', 'fn', 'fp', 'tn']
METHOD_LABEL = {'raw': '原始（基线）'}
MODEL_LABEL = {'sort': 'SORT', 'ctc': 'CTC-e7-int8', 'ctc_v23': 'CTC-v23-int8'}


def mcnemar_exact(b, c):
    """精确 McNemar 双侧检验：b/c 为两种条件下判决不同的样本数。"""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return min(1.0, 2 * p)


def read_run(csv_path):
    """读一份结果，按 sample_id 去重（保留最后一条），返回 {sid: row}。"""
    if not csv_path.exists():
        return {}
    rows = {}
    order = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            sid = r.get('sample_id', '')
            if not sid.isdigit():
                continue
            sid = int(sid)
            if sid not in rows:
                order.append(sid)
            rows[sid] = r
    return {sid: rows[sid] for sid in sorted(order)}


def num(v, default=0.0):
    try:
        if v is None or v == '':
            return default
        return float(v)
    except ValueError:
        return default


def summarize(rows):
    ok = [r for r in rows.values() if r.get('status') == 'ok']
    pos = [r for r in ok if int(r['label']) == 1]
    neg = [r for r in ok if int(r['label']) == 0]
    tp = sum(1 for r in pos if int(r['triggered']) == 1)
    fp = sum(1 for r in neg if int(r['triggered']) == 1)
    out = {
        'num_rows': len(rows), 'num_ok': len(ok), 'num_error': len(rows) - len(ok),
        'n_pos': len(pos), 'n_neg': len(neg),
        'tp': tp, 'fn': len(pos) - tp,
        'fp': fp, 'tn': len(neg) - fp,
        'recall': round(tp / len(pos), 4) if pos else None,
        'fpr': round(fp / len(neg), 4) if neg else None,
        'trigger_rate_all': round((tp + fp) / len(ok), 4) if ok else None,
        'frame_score_max_pos_median': (round(statistics.median(
            [num(r['frame_score_max']) for r in pos]), 4) if pos else None),
        'frame_score_max_neg_median': (round(statistics.median(
            [num(r['frame_score_max']) for r in neg]), 4) if neg else None),
    }
    out['missed_pos_ids'] = sorted(int(r['sample_id']) for r in pos
                                   if int(r['triggered']) == 0)
    out['fp_ids'] = sorted(int(r['sample_id']) for r in neg
                           if int(r['triggered']) == 1)
    return out


def paired(base_rows, cur_rows, label):
    """配对比较：返回计数与逐样本明细（仅 label 匹配的样本）。"""
    sids = sorted(set(base_rows) & set(cur_rows))
    rows = []
    b_only = c_only = both = neither = 0
    for sid in sids:
        a, b = base_rows[sid], cur_rows[sid]
        if a.get('status') != 'ok' or b.get('status') != 'ok':
            continue
        if int(a['label']) != label or int(b['label']) != label:
            continue
        ta, tb = int(a['triggered']), int(b['triggered'])
        if ta and not tb:
            b_only += 1
        elif tb and not ta:
            c_only += 1
        elif ta and tb:
            both += 1
        else:
            neither += 1
        rows.append({
            'sample_id': sid, 'label': label,
            'raw_triggered': ta, 'method_triggered': tb,
            'changed': int(ta != tb),
            'raw_num_triggers': a.get('num_triggers', ''),
            'method_num_triggers': b.get('num_triggers', ''),
            'raw_frame_score_max': a.get('frame_score_max', ''),
            'method_frame_score_max': b.get('frame_score_max', ''),
        })
    return {'both': both, 'neither': neither,
            'raw_only': b_only, 'method_only': c_only,
            'mcnemar_p': round(mcnemar_exact(b_only, c_only), 6)}, rows


def fmt_ids(ids):
    """ID 列表过长时只列前 12 个，避免报告被几百个 ID 淹没。"""
    if not ids:
        return '无'
    if len(ids) <= 12:
        return ', '.join(map(str, ids))
    return ', '.join(map(str, ids[:12])) + f' 等 {len(ids)} 条'


def fmt_pct(v):
    return '—' if v is None else f'{v * 100:.2f}%'


def pct_delta(cur, base):
    if cur is None or base is None:
        return '—'
    d = (cur - base) * 100
    return f'{d:+.2f} pp'


def headline(summary):
    """按结果自动生成首段摘要（不写死任何数字）。"""
    n_methods = max((len([m for m in d['runs'].get('sort', {}) if m != 'raw'])
                     for d in summary.values()), default=0)
    lines = [f'本次共评估原始音频与 {n_methods} 种增强算法，两套 KWS 权重与判决逻辑保持不变，'
             '只替换输入音频。']
    for dname, d in summary.items():
        ds = d['dataset']
        lines.append('')
        lines.append(f'- **{dname}**（正样本 {ds["n_pos"]}、负样本 {ds["n_neg"]}）')
        for model in d['runs']:
            runs = d['runs'].get(model, {})
            base = runs.get('raw', {}).get('metrics')
            cands = [(m, p['metrics']) for m, p in runs.items()
                     if m != 'raw' and p.get('metrics')]
            if not base or not cands:
                continue
            good = [(m, x) for m, x in cands
                    if x['recall'] is not None and base['recall'] is not None
                    and x['recall'] >= base['recall']]
            if good and base['n_neg']:
                good.sort(key=lambda t: (t[1]['fpr'], -t[1]['recall']))
                m, x = good[0]
                lines.append(f'  - {MODEL_LABEL.get(model, model.upper())}：召回不低于原始且误触发最低的是 **{m}**'
                             f'（Recall {x["recall"] * 100:.2f}%、误触发率 {x["fpr"] * 100:.2f}%）。')
            elif good:
                good.sort(key=lambda t: (-t[1]['recall'], t[1]['fp']))
                m, x = good[0]
                lines.append(f'  - {MODEL_LABEL.get(model, model.upper())}：召回不低于原始的有 {len(good)} 种，'
                             f'最高为 **{m}**（{x["recall"] * 100:.2f}%）。')
            else:
                m, x = max(cands, key=lambda t: t[1]['recall'])
                lines.append(f'  - {MODEL_LABEL.get(model, model.upper())}：全部增强算法召回都低于原始，'
                             f'最高为 {m}（{x["recall"] * 100:.2f}%）。')
    return lines


def build_report(summary, meta):
    lines = ['# 语音增强对三类 KWS 模型的影响：原始音频与全部增强算法', '']
    lines.extend(headline(summary))
    lines.append('')
    lines.append('两套 KWS 权重与判决逻辑保持不变，只替换输入音频；')
    lines.append(f'推理口径：{meta.get("sample_rate", 16000)} Hz、'
                 f'{meta.get("chunk_ms", 30)} ms chunk、逐样本状态复位。')
    lines.append('SORT 阈值 0.84（连续 3 帧 + 500 ms 冷却）；CTC 阈值 -0.10（500 ms 冷却）。')
    lines.append('')

    for dname, d in summary.items():
        cfg = d['dataset']
        n_pos, n_neg = cfg['n_pos'], cfg['n_neg']
        lines.append(f'## {dname}（正样本 {n_pos} / 负样本 {n_neg}）')
        lines.append('')
        lines.append(cfg.get('description', ''))
        lines.append('')
        lines.append('| 模型 | 方法 | TP | FN | FP | TN | Recall | 误触发率 | ΔRecall | ΔFP | p(负) | p(正) |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for model, runs in d['runs'].items():
            base = runs.get('raw', {}).get('metrics')
            for method, pack in runs.items():
                m = pack['metrics']
                if m is None:
                    lines.append(f'| {MODEL_LABEL.get(model, model.upper())} | {METHOD_LABEL.get(method, method)} '
                                 f'| — | — | — | — | — | — | — | — | — | 缺结果 |')
                    continue
                if method == 'raw' or not base:
                    dp = dq = '—'
                    p_neg = p_pos = '基线'
                else:
                    dp = pct_delta(m['recall'], base['recall'])
                    dq = (f'{m["fp"] - base["fp"]:+d}' if base['n_neg'] else '—')
                    pn, pp = pack.get('p_neg'), pack.get('p_pos')
                    p_neg = f'{pn:.3g}' if pn is not None else '—'
                    p_pos = f'{pp:.3g}' if pp is not None else '—'
                lines.append(
                    f'| {MODEL_LABEL.get(model, model.upper())} | {METHOD_LABEL.get(method, method)} | '
                    f'{m["tp"]} | {m["fn"]} | {m["fp"]} | {m["tn"]} | '
                    f'{fmt_pct(m["recall"])} | {fmt_pct(m["fpr"])} | {dp} | {dq} | '
                    f'{p_neg} | {p_pos} |')
        lines.append('')
        lines.append('ΔRecall/ΔFP 相对原始音频；p(负)/p(正) 分别为负样本误触发、'
                     '正样本召回上的精确 McNemar 双侧检验（无该类样本时为 —）。')
        lines.append('')

        for model, runs in d['runs'].items():
            if 'raw' not in runs:
                continue
            base_m = runs['raw']['metrics']
            if not base_m:
                continue
            for method, pack in runs.items():
                if method == 'raw':
                    continue
                pm = pack.get('paired_pos') or {}
                nm = pack.get('paired_neg') or {}
                if not pm and not nm:
                    continue
                bits = []
                if pm:
                    gained = sorted(r['sample_id'] for r in pack['paired_pos_rows']
                                    if r['method_triggered'] and not r['raw_triggered'])
                    lost = sorted(r['sample_id'] for r in pack['paired_pos_rows']
                                  if r['raw_triggered'] and not r['method_triggered'])
                    bits.append(f'正样本：恢复 {fmt_ids(gained)}；'
                                f'新增漏检 {fmt_ids(lost)}')
                if nm:
                    suppressed = nm['raw_only']
                    new_fp = nm['method_only']
                    bits.append(f'负样本：抑制误触发 {suppressed} 条，新增误触发 {new_fp} 条')
                lines.append(f'- **{MODEL_LABEL.get(model, model.upper())} / '
                             f'{METHOD_LABEL.get(method, method)}**：'
                             + '；'.join(bits)
                             + f'（正样本帧分数中位 {base_m["frame_score_max_pos_median"]} → '
                               f'{pack["metrics"]["frame_score_max_pos_median"]}）')
        lines.append('')
    return '\n'.join(lines) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=None)
    ap.add_argument('--models', nargs='+', default=None)
    ap.add_argument('--methods', nargs='+', default=None)
    args = ap.parse_args()

    with open(ROOT / 'config.json', encoding='utf-8') as f:
        cfg = json.load(f)
    datasets = args.datasets or [d['name'] for d in cfg['datasets']]
    methods = args.methods or cfg['methods']
    models = args.models or cfg.get('models', ['sort', 'ctc'])
    ds_by_name = {d['name']: d for d in cfg['datasets']}

    summary = {}
    paired_rows = []
    for dname in datasets:
        ds_cfg = ds_by_name[dname]
        entry = {'dataset': {'name': dname,
                             'description': ds_cfg.get('description', ''),
                             'n_pos': 0, 'n_neg': 0},
                 'runs': {}}
        for model in models:
            runs = {}
            base_rows = None
            for method in methods:
                rows = read_run(RESULTS_DIR / dname / f'{model}__{method}.csv')
                if not rows:
                    continue
                metrics = summarize(rows)
                entry['dataset']['n_pos'] = max(entry['dataset']['n_pos'], metrics['n_pos'])
                entry['dataset']['n_neg'] = max(entry['dataset']['n_neg'], metrics['n_neg'])
                pack = {'metrics': metrics, 'file': f'{model}__{method}.csv'}
                if method == 'raw':
                    base_rows = rows
                if base_rows is not None and method != 'raw':
                    pm, pm_rows = paired(base_rows, rows, 1)
                    nm, nm_rows = paired(base_rows, rows, 0)
                    if nm['both'] + nm['neither'] + nm['raw_only'] + nm['method_only'] == 0:
                        nm = {'both': 0, 'neither': 0, 'raw_only': 0,
                              'method_only': 0, 'mcnemar_p': None}
                    pack['paired_pos'] = pm
                    pack['paired_pos_rows'] = pm_rows
                    pack['paired_neg'] = nm
                    pack['paired_neg_rows'] = nm_rows
                    pack['p_pos'] = pm['mcnemar_p']
                    pack['p_neg'] = nm['mcnemar_p'] if (
                        nm['both'] + nm['neither'] + nm['raw_only'] + nm['method_only']) else None
                    for r in pm_rows:
                        paired_rows.append({'dataset': dname, 'model': model,
                                            'method': method, **r})
                runs[method] = pack
            if runs:
                entry['runs'][model] = runs
        if entry['runs']:
            summary[dname] = entry

    RESULTS_DIR.mkdir(exist_ok=True)

    if paired_rows:
        fields = ['dataset', 'model', 'method', 'sample_id', 'label',
                  'raw_triggered', 'method_triggered', 'changed',
                  'raw_num_triggers', 'method_num_triggers',
                  'raw_frame_score_max', 'method_frame_score_max']
        with open(RESULTS_DIR / 'paired_samples.csv', 'w', newline='',
                  encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            for r in paired_rows:
                w.writerow({k: r.get(k, '') for k in fields})

    with open(RESULTS_DIR / 'report.md', 'w', encoding='utf-8') as f:
        f.write(build_report(summary, {'sample_rate': cfg['sample_rate'],
                                       'chunk_ms': cfg['chunk_ms']}))

    # 逐样本明细单独放 paired_samples.csv，summary.json 里不再重复存一份
    slim = {d: {'dataset': e['dataset'],
                'runs': {m: {k: {kk: vv for kk, vv in p.items() if not kk.endswith('_rows')}
                             for k, p in runs.items()}
                         for m, runs in e['runs'].items()}}
            for d, e in summary.items()}
    with open(RESULTS_DIR / 'summary.json', 'w', encoding='utf-8') as f:
        json.dump(slim, f, ensure_ascii=False, indent=2)
    print(f'[done] 汇总写入 {RESULTS_DIR}（summary.json / report.md / paired_samples.csv）')


if __name__ == '__main__':
    main()
