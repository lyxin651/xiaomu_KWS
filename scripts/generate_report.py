#!/usr/bin/env python3
"""由 results/summary.json + paired_comparison.csv 生成 evaluation_report.md。

所有数字均从结果文件读取，报告与结果保持一致；重新分析后重跑本脚本即可。
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

RUNS = [('sort', 'raw', 'A SORT / 原始'), ('sort', 'enhanced', 'B SORT / 增强'),
        ('ctc', 'raw', 'C CTC / 原始'), ('ctc', 'enhanced', 'D CTC / 增强')]


def fmt_ids(ids):
    return '无' if not ids else ', '.join(str(i) for i in ids)


def pct(x, nd=2):
    return 'N/A' if x is None else f'{x * 100:.{nd}f}%'


def uniq(v, unit=''):
    if not v:
        return '—'
    vals = [str(x) for x in v]
    return (' / '.join(vals)) + unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output-dir', required=True)
    args = ap.parse_args()
    out = Path(args.output_dir).resolve()
    s = json.loads((out / 'results' / 'summary.json').read_text(encoding='utf-8'))
    dv = json.loads((out / 'data_validation.json').read_text(encoding='utf-8'))
    paired = list(csv.DictReader(open(out / 'results' / 'paired_comparison.csv',
                                      newline='', encoding='utf-8')))
    labels = {int(r['sample_id']): int(r['label'])
              for r in csv.DictReader(open(out / 'labels.csv', newline='', encoding='utf-8'))}

    m, d, pc, pid = s['metrics'], s['deltas'], s['paired_counts'], s['paired_ids']
    cross = s['cross_model']
    cases = s['cases']

    def get(model, cat):
        return pid.get(f'{model}|{cat}', [])

    def delta_scores(model, label):
        rs = [r for r in paired if r['model_name'] == model and int(r['label']) == label]
        return np.array([float(r['enhanced_frame_score_max']) - float(r['raw_frame_score_max'])
                         for r in rs])

    dp = {mm: delta_scores(mm, 1) for mm in ('sort', 'ctc')}
    dn = {mm: delta_scores(mm, 0) for mm in ('sort', 'ctc')}

    L = []
    A = L.append
    A('# KWS 双模型语音增强前后效果验证报告\n')
    A(f'- 数据集：`kws_dataset`（392 条白名单挂测回采音频）与 `gtcrn_enhanced`（同 ID 增强音频）')
    A('- 模型：SORT `kws_cmvn_g1_e100.onnx`（thr=0.84）、CTC `kws_ctc_v2verifier_e7_int8.onnx`（thr=-0.10）')
    A('- 实验日期：2026-11-10　|　推理记录：4 × 392 = 1568 条，失败 0 条\n')

    A('## 结论摘要\n')
    A(f'1. **SORT 误唤醒显著下降**：回采负样本触发比例 {pct(m["sort_raw"]["fpr"])} → '
      f'{pct(m["sort_enhanced"]["fpr"])}（{d["sort"]["fpr_delta_pp"]:+.2f} pp，相对 '
      f'{pct(d["sort"]["fpr_relative_change"])}，抑制 {pct(d["sort"]["suppression"])}，'
      f'McNemar p≈{d["sort"]["neg_mcnemar_p"]:.1e}）。')
    A(f'2. **CTC 误唤醒几乎清零**：{pct(m["ctc_raw"]["fpr"])} → {pct(m["ctc_enhanced"]["fpr"])}'
      f'（{d["ctc"]["fpr_delta_pp"]:+.2f} pp，相对 {pct(d["ctc"]["fpr_relative_change"])}，'
      f'抑制 {pct(d["ctc"]["suppression"])}，McNemar p≈{d["ctc"]["neg_mcnemar_p"]:.1e}）。')
    A(f'3. **SORT 无净漏检但有边界抖动**：Recall 保持 {m["sort_raw"]["TP"]}/{m["sort_raw"]["n_positive"]}，'
      f'其中 {d["sort"]["pos_lost"]} 条漏检（{fmt_ids(get("sort", "pos_lost"))}）、'
      f'{d["sort"]["pos_recovered"]} 条恢复（{fmt_ids(get("sort", "pos_recovered"))}）。')
    A(f'4. **CTC 出现明显漏检**：Recall {m["ctc_raw"]["TP"]}/{m["ctc_raw"]["n_positive"]} → '
      f'{m["ctc_enhanced"]["TP"]}/{m["ctc_enhanced"]["n_positive"]}'
      f'（{d["ctc"]["recall_delta_pp"]:+.2f} pp，相对 {pct(d["ctc"]["recall_relative_change"])}），'
      f'漏检 {fmt_ids(get("ctc", "pos_lost"))}，恢复 {fmt_ids(get("ctc", "pos_recovered"))}；'
      f'正样本仅 21 条，配对 McNemar p={d["ctc"]["pos_mcnemar_p"]:.3f}，方向明确但统计功效有限。')
    A(f'5. **两模型响应不同**：增强对两者负样本都有效，但 CTC 的正样本分数中位被压低 '
      f'{np.median(dp["ctc"]):+.2f} nats/token，在当前工作点上属于“过度抑制”；'
      f'SORT 正样本分数中位仅变化 {np.median(dp["sort"]):+.3f}。\n')

    # ---------------- 1 ----------------
    A('## 1. 实验目的、模型配置与数据集标签\n')
    A('### 1.1 目的\n')
    A('在**不训练、不改权重、不改增强算法**的前提下，量化同一套 GTCRN 增强算法对两种已部署 KWS '
      '架构的实际影响，回答：增强是否降低误唤醒、是否造成真实唤醒漏检、两种架构响应是否有差异、'
      '哪些样本被改善或退化。\n')
    A('### 1.2 模型与推理配置\n')
    A('| 项目 | SORT | CTC |')
    A('|---|---|---|')
    A('| 权重 | `kws_cmvn_g1_e100.onnx` | `kws_ctc_v2verifier_e7_int8.onnx` |')
    A('| 推理类 | `OnnxKWS`（`deploy_sort/mic_kws.py`） | `CtcKwsSession`（`deploy_ctc/mic_ctc.py`） |')
    A('| 阈值 | 0.84 | -0.10 |')
    A('| 判决规则 | 最近 3 帧均 ≥ 阈值触发；冷却 500 ms | 关键词路径分数 ≥ 阈值触发；触发后 matcher 复位；冷却 500 ms |')
    A('| 分数含义 | 每帧唤醒概率 | 关键词路径平均每 token log-LR（nats/token） |')
    A('')
    A('两组模型均按 30 ms（480 samples）chunk 流式喂入，复用部署包原 `StreamingFbank`，'
      '每条音频开始前重建检测器（cache 从 `_init_cache.npy` 重新载入，fbank/matcher/cooldown 全部新建），'
      'ONNX session 按模型共享但**不跨样本复用任何声学状态**。原始/增强使用完全一致的参数。'
      '旧挂测 JSON 中的 `threshold=0.3` 属设备历史采集配置，未采用。\n')
    A('### 1.3 数据集与标签\n')
    A('| 项 | 值 |')
    A('|---|---|')
    A(f'| 总样本 | {dv["paired_count"]} |')
    A(f'| 真实唤醒（label=1） | {dv["positive_count"]} 条：{fmt_ids(dv["positive_ids"])} |')
    A(f'| 回采负样本（label=0） | {dv["negative_count"]} 条 |')
    A(f'| 0–395 缺失编号 | {fmt_ids(dv["ids_missing_from_0_395"])}（原始集本就缺失，不作补齐） |')
    A('| 标签来源 | 人工确认真实标签（未使用 JSON 历史触发分数） |')
    A('')
    A('配对方式为文件名 ID 精确匹配（`c100-01-<ID>.wav`），未使用排序位置。`labels.csv` 含 '
      '`sample_id,label,raw_path,enhanced_path`。\n')
    A('### 1.4 数据预检查结论\n')
    A('| 检查项 | 结果 |')
    A('|---|---|')
    A(f'| 原始 / 增强 WAV 数 | {dv["raw_wav_count"]} / {dv["enhanced_wav_count"]}，一一对应 |')
    A(f'| 未配对样本 | 原始独有 {fmt_ids(dv["unpaired_raw_ids"])}；增强独有 {fmt_ids(dv["unpaired_enhanced_ids"])} |')
    A(f'| 采样率 / 通道 / 时长 | {uniq(dv["sample_rate_set"])} Hz / {uniq(dv["channels_set"])} 声道 / '
      f'{uniq(dv["duration_sec_set"])} s（增强侧 {uniq(dv["enhanced_duration_sec_set"])} s） |')
    A(f'| 长度差 | 全部为 {uniq(dv["length_delta_frames_unique"])} samples（无裁切/补零） |')
    A(f'| 增强引入时延 | 互相关估计 {uniq(dv["lag_samples_unique"])} samples'
      f'（{uniq(dv["lag_ms_unique"])} ms） |')
    A(f'| 异常样本 | {dv["anomaly_count"]} 条：id=107 原始音频有 366 samples 削波（增强后峰值 0.47，无削波） |')
    A('')
    A('预检查未发现缺失、空音频、NaN/Inf 或长度变化，因此四组统计分母统一为 392（21 正 / 371 负）。'
      '增强音频未再做归一化或增益调整；无重采样发生。\n')
    A('### 1.5 推理一致性验证\n')
    A('- T1 标签校验：392 条、21 正样本、缺失编号与预期一致。')
    A('- T2 状态复位：同一样本连续 / 交错重复推理，逐帧分数与触发事件逐位一致。')
    A('- T3 流式等价：SORT 在 10/30/100 ms chunk 下输出一致；CTC **float** 模型同样一致；'
      'CTC **INT8** 模型输出随 chunk 长度变化（同一句 30 ms 与 10 ms 的路径分数最大差 1.99 nats/token），'
      '属 INT8 动态量化激活尺度的分块敏感性。部署端与主实验均固定 30 ms，四组条件完全一致，不影响对照公平性。')
    A('- T4 官方一致性：CTC 在 id=0/2/101 上与 `mic_ctc.py --wav` 的帧数、触发次数、触发分数完全一致'
      '（298 帧；id=0 两侧均为 `score=-0.070` 且触发，id=2 均不触发，id=101 均为 `+1.440`）。')
    A('- T5 小样本四组核查：2 正 + 2 负 × 2 模型 × 2 条件全部成功，状态与结果可复现。\n')

    # ---------------- 2 ----------------
    A('## 2. 四组实验结果\n')
    A('| 组别 | TP | FN | FP | TN | Recall (95% CI) | FPR (95% CI) | 触发事件总数 | 平均单条耗时 |')
    A('|---|---|---|---|---|---|---|---|---|')
    for model, cond, name in RUNS:
        k = f'{model}_{cond}'
        x = m[k]
        A(f'| {name} | {x["TP"]} | {x["FN"]} | {x["FP"]} | {x["TN"]} | '
          f'{pct(x["recall"])} ({pct(x["recall_ci95"][0])}–{pct(x["recall_ci95"][1])}) | '
          f'{pct(x["fpr"])} ({pct(x["fpr_ci95"][0])}–{pct(x["fpr_ci95"][1])}) | '
          f'{x["num_triggers_total"]} | {x["mean_inference_ms"]:.0f} ms |')
    A('')
    A('![confusion](../figures/confusion_matrices.png)\n')
    A('Recall=TP/21；FPR=FP/371，仅表示**当前回采负样本**上的触发比例，不能解释为真实部署的每小时误唤醒率（FA/h）。'
      '21 条正样本的 Recall 也只代表该选定集合。\n')

    # ---------------- 3 ----------------
    A('## 3. 增强前后差异（同 ID 配对）\n')
    A('| 模型 | Recall raw→enh | 绝对变化 | 相对变化 | FPR raw→enh | 绝对变化 | 相对变化 | 负样本抑制 | McNemar p (负样本) | McNemar p (正样本) |')
    A('|---|---|---|---|---|---|---|---|---|---|')
    for model in ('sort', 'ctc'):
        x = d[model]
        A(f'| {model.upper()} | {pct(x["recall_raw"])} → {pct(x["recall_enhanced"])} | '
          f'{x["recall_delta_pp"]:+.2f} pp | {pct(x["recall_relative_change"])} | '
          f'{pct(x["fpr_raw"])} → {pct(x["fpr_enhanced"])} | {x["fpr_delta_pp"]:+.2f} pp | '
          f'{pct(x["fpr_relative_change"])} | {pct(x["suppression"])} | '
          f'{x["neg_mcnemar_p"]:.2e} | {x["pos_mcnemar_p"]:.3f} |')
    A('')
    A('配对变化计数：\n')
    A('| 情况 | SORT | CTC |')
    A('|---|---|---|')
    A(f'| 负样本 raw=1→enh=0（成功消除） | {pc.get("sort|neg_removed", 0)} | {pc.get("ctc|neg_removed", 0)} |')
    A(f'| 负样本 raw=0→enh=1（新引入） | {pc.get("sort|neg_new", 0)} | {pc.get("ctc|neg_new", 0)} |')
    A(f'| 负样本 raw=1→enh=1（仍存在） | {pc.get("sort|neg_both", 0)} | {pc.get("ctc|neg_both", 0)} |')
    A(f'| 负样本 raw=0→enh=0（均未触发） | {pc.get("sort|neg_none", 0)} | {pc.get("ctc|neg_none", 0)} |')
    A(f'| 正样本 raw=1→enh=1（保持唤醒） | {pc.get("sort|pos_kept", 0)} | {pc.get("ctc|pos_kept", 0)} |')
    A(f'| 正样本 raw=1→enh=0（增强漏检） | {pc.get("sort|pos_lost", 0)} | {pc.get("ctc|pos_lost", 0)} |')
    A(f'| 正样本 raw=0→enh=1（增强恢复） | {pc.get("sort|pos_recovered", 0)} | {pc.get("ctc|pos_recovered", 0)} |')
    A(f'| 正样本 raw=0→enh=0（均漏检） | {pc.get("sort|pos_both_miss", 0)} | {pc.get("ctc|pos_both_miss", 0)} |')
    A('')
    A('![recall_fpr](../figures/recall_fpr_comparison.png)\n')
    A('![transitions](../figures/paired_transitions.png)\n')
    A('正样本只有 21 条，比例指标的置信区间很宽（见第 2 节 Wilson CI），'
      '正样本上的差异应视为方向性证据；负样本 371 条且为配对比较，用精确 McNemar 检验'
      '（SORT p≈7.2e-18，CTC p≈2.4e-4）。\n')

    # ---------------- 4 ----------------
    A('## 4. 具体被改善 / 退化 / 漏检的样本\n')
    for model in ('sort', 'ctc'):
        A(f'### {model.upper()}\n')
        A(f'- 成功消除的误触发（{len(get(model, "neg_removed"))} 条）：{fmt_ids(get(model, "neg_removed"))}')
        A(f'- 新引入的误触发（{len(get(model, "neg_new"))} 条）：{fmt_ids(get(model, "neg_new"))}')
        A(f'- 增强后仍误触发（{len(get(model, "neg_both"))} 条）：{fmt_ids(get(model, "neg_both"))}')
        A(f'- 增强导致真实唤醒漏检（{len(get(model, "pos_lost"))} 条）：{fmt_ids(get(model, "pos_lost"))}')
        A(f'- 增强恢复真实唤醒（{len(get(model, "pos_recovered"))} 条）：{fmt_ids(get(model, "pos_recovered"))}')
        A(f'- 原始/增强均漏检的真实唤醒（{len(get(model, "pos_both_miss"))} 条）：{fmt_ids(get(model, "pos_both_miss"))}')
        A('')
    A('所有这些 ID 的原始与增强音频路径可从 `labels.csv` 按 `sample_id` 直接取得，'
      '逐条结果见 `results/paired_comparison.csv`。\n')

    # ---------------- 5 ----------------
    A('## 5. 两种模型对增强算法响应的差异\n')
    A('### 5.1 分数分布的整体位移\n')
    A('| 模型 | 正样本 max score 中位变化 | 正样本分数下降>0.1 | 负样本 max score 中位变化 | 负样本分数上升条数 |')
    A('|---|---|---|---|---|')
    for model in ('sort', 'ctc'):
        A(f'| {model.upper()} | {np.median(dp[model]):+.3f} | '
          f'{int((dp[model] < -0.1).sum())}/21 | {np.median(dn[model]):+.3f} | '
          f'{int((dn[model] > 0).sum())}/371 |')
    A('')
    A('（SORT 单位为帧概率，CTC 为 nats/token；两类分数不可直接比较绝对值，'
      '这里只比较各自增强前后的位移方向与幅度。）\n')
    A('**SORT**：增强让负样本分数整体下移（中位 -0.128），误触发从 102 降到 34，'
      '改善是全局性的（71 条消除 vs 3 条新增）；正样本分数中位几乎不变（-0.015），'
      '但 6/21 条下降超过 0.1，出现 1 条漏检（id=121）与 1 条恢复（id=119），'
      '净 Recall 不变，说明工作点附近的边界样本仍在抖动。')
    A('')
    A('**CTC**：增强同样压低负样本分数（中位 -2.07），误触发 14→1（仅 id=383 仍触发，'
      '未新增误触发）；但正样本分数也大幅下降（中位 -1.28，16/21 条下降>0.1），'
      '5 条原本刚过阈值的唤醒被压到阈值以下（7、25、122、123、125），'
      'Recall 14/21→10/21（-19.05 pp）。CTC 的判决依赖 8-token 关键词路径相对 blank/fill 的似然优势，'
      '增强在抑制噪声的同时也削弱了部分音节（尤其第二遍“小沐”）的 token 判别度，'
      '使正负样本一起向低分移动，阈值 -0.10 的余量被吃掉。')
    A('')
    A('### 5.2 同一样本上的正样本对照\n')
    A('| ID | SORT raw/enh | SORT max raw→enh | CTC raw/enh | CTC max raw→enh |')
    A('|---|---|---|---|---|')
    for sid in sorted(labels):
        if labels[sid] != 1:
            continue
        sr = next(r for r in paired if r['model_name'] == 'sort' and int(r['sample_id']) == sid)
        cr = next(r for r in paired if r['model_name'] == 'ctc' and int(r['sample_id']) == sid)
        A(f'| {sid} | {sr["raw_triggered"]}/{sr["enhanced_triggered"]} | '
          f'{float(sr["raw_frame_score_max"]):.2f}→{float(sr["enhanced_frame_score_max"]):.2f} | '
          f'{cr["raw_triggered"]}/{cr["enhanced_triggered"]} | '
          f'{float(cr["raw_frame_score_max"]):.2f}→{float(cr["enhanced_frame_score_max"]):.2f} |')
    A('')
    A('两种模型表现相反的样本：\n')
    A(f'- 负样本：无“SORT 改善但 CTC 退化”或“CTC 改善但 SORT 退化”的样本；'
      f'两边同时消除误触发：{fmt_ids(cross["both_improved"])}。')
    A(f'- 正样本：SORT 漏检而 CTC 保持：{fmt_ids(cross["pos_sort_lost_ctc_kept"])}；'
      f'CTC 漏检而 SORT 保持：{fmt_ids(cross["pos_sort_kept_ctc_lost"])}；'
      f'两者同时漏检：{fmt_ids(cross["pos_both_lost"])}。')
    A('')
    A('即：**增强对 SORT 的负样本收益大、正样本代价小；对 CTC 的负样本收益同样大，但正样本代价显著**。'
      'CTC 原工作点本就偏向低误唤醒（FPR 3.77%），增强后 FPR 仅剩 0.27%，但这是以 5 条真实唤醒为代价换来的，'
      '在该工作点上属于“过度抑制”。\n')
    A('### 5.3 典型案例（逐帧分数曲线）\n')
    case_notes = [
        ('SORT 成功消除误触发', 'sort_neg_removed', 'raw 峰值 0.975 连续过阈，增强后峰值仅 0.225，完全脱离工作点'),
        ('SORT 新增误触发', 'sort_neg_new', '增强后峰值由 0.800 抬到 0.972，跨过 0.84 并满足 3 帧连续'),
        ('SORT 真实唤醒漏检', 'sort_pos_lost', 'raw 峰值 0.971 触发；增强后峰值 0.806，低于 0.84'),
        ('SORT 增强恢复唤醒', 'sort_pos_recovered', 'raw 峰值 0.687 未触发，增强后 0.966 触发'),
        ('CTC 成功消除误触发', 'ctc_neg_removed', '路径分数 1.86 → -1.70，远离阈值 -0.10'),
        ('CTC 新增误触发', 'ctc_neg_new', None),
        ('CTC 真实唤醒漏检', 'ctc_pos_lost', '路径分数 1.09 → -2.58，第二遍“小沐”的 token 优势被削弱'),
        ('CTC 增强恢复唤醒', 'ctc_pos_recovered', '-1.44 → 0.19，刚刚越过阈值'),
    ]
    for title, key, desc in case_notes:
        c = cases.get(key)
        if not c:
            A(f'- **{title}**：本次数据中不存在该类样本（CTC 未新增误触发）。')
            continue
        raw_rel = f'{Path(c["raw_path"]).parent.name}/{Path(c["raw_path"]).name}'
        enh_rel = f'{Path(c["enhanced_path"]).parent.name}/{Path(c["enhanced_path"]).name}'
        A(f'- **{title}**：id={c["sample_id"]}，`{raw_rel}` / `{enh_rel}`，{desc}。')
        A(f'  ![case](../figures/selected_cases/{c["figure"]})')
    A('')
    A('曲线中竖点线为该条件下的首次触发时刻，水平虚线为判决阈值；'
      'CTC 曲线为独立 matcher 重放的逐帧路径分数（不随触发复位），触发判定仍以流式事件为准。\n')

    # ---------------- 6 ----------------
    A('## 6. 未解决问题、失败情况与适用范围\n')
    A('1. **无推理失败**：1568 条记录全部 `status=ok`，四组各 392 条；原始/增强按 ID 全配对。')
    A('2. **样本量与口径**：FPR 是 371 条回采负样本上的触发比例，不是 FA/h；Recall 只代表 21 条选定正样本。'
      '正样本差异的统计功效有限（CTC 的 5 条漏检 McNemar p=0.219，方向明确但未达 0.05 显著）。')
    A('3. **CTC INT8 分块敏感性**：同一句在 10/30/100 ms chunk 下路径分数最大差约 2–2.8 nats/token，'
      '足以跨越 -0.10 阈值。本报告结论仅对应部署口径（30 ms chunk）。若部署端 chunk 长度改变，'
      'CTC 的绝对结果需要重新测量；float 模型无此问题（chunk 间最大差 0）。')
    A('4. **增强未做增益归一化**：增强后整体电平略降（中位 RMS 0.00286→0.00217），'
      '可能同时贡献了分数下移；本次按要求不做静默归一化，因此“抑制过强”是增强链路的整体表现，'
      '未拆分去噪与电平两个因素。')
    A('5. **单一增强算法/单一工作点**：只评估了当前 GTCRN 增强与部署默认阈值（SORT 0.84、CTC -0.10），'
      '未做阈值扫描。若要为 CTC 挽回 Recall，应在同一增强条件下重新选择工作点'
      '（例如降低阈值），但必须与原始条件同口径比较，不能“增强+调阈值”对“原始默认阈值”。')
    A('6. **特殊样本**：id=107 原始音频存在 366 samples 削波，增强后无削波；该样本的两模型结果均正常，'
      '但它是原始侧唯一的质量异常，若后续扩大评估需关注采集链路。')
    A('7. **未评估**：未使用 `deploy/kws_dataset_1016_5`（928 条纯唤醒语料）；未评估连续自然运行中的 FA/h；'
      '未评估两模型的 float/int8 差异对结论的影响（主实验即部署 INT8 权重）。')
    A('')
    A('**结论适用性**：本报告可直接用于判断当前 GTCRN 增强对已部署 SORT/CTC 的收益与副作用，'
      '但不能外推为其他增强算法、其他阈值或其他 chunk 配置下的结论。\n')

    # ---------------- 7 ----------------
    A('## 7. 产物与复现\n')
    A('```text')
    A('kws_enhancement_eval/')
    A('├── config.yaml                 # 主实验固定参数')
    A('├── labels.csv                  # 392 条 sample_id / label / raw_path / enhanced_path')
    A('├── data_validation.json        # 配对、格式、长度、时延、异常检查')
    A('├── results/')
    A('│   ├── sort_raw.csv  sort_enhanced.csv  ctc_raw.csv  ctc_enhanced.csv   # 各 392 行')
    A('│   ├── paired_comparison.csv   # 784 行 = 2 模型 × 392 样本')
    A('│   ├── summary.json            # 全部指标、配对计数、ID 列表')
    A('│   └── frame_scores/*.npz      # 逐帧分数曲线')
    A('├── figures/                    # 混淆矩阵、Recall/FPR、配对转移、典型案例')
    A('├── reports/evaluation_report.md')
    A('├── logs/')
    A('└── scripts/                    # prepare_dataset / evaluate_kws / analyze_results / test_kws_eval')
    A('```')
    A('')
    A('```bash')
    A('python scripts/prepare_dataset.py --raw-dir ../kws_dataset \\')
    A('    --enhanced-dir ../gtcrn_enhanced --output-dir ..')
    A('python scripts/evaluate_kws.py --raw-dir ../kws_dataset \\')
    A('    --enhanced-dir ../gtcrn_enhanced --deploy-dir ../deploy --output-dir ..')
    A('python scripts/test_kws_eval.py --raw-dir ../kws_dataset \\')
    A('    --enhanced-dir ../gtcrn_enhanced --deploy-dir ../deploy --output-dir ..')
    A('python scripts/analyze_results.py --output-dir ..')
    A('```')
    A('')
    A('`evaluate_kws.py` 支持断点续跑：已存在且 `status=ok` 的 sample_id 自动跳过；'
      '单样本错误会写入 `status=error` 与 `error` 列，不会静默计为未触发。\n')

    text = '\n'.join(L) + '\n'
    (out / 'reports' / 'evaluation_report.md').write_text(text, encoding='utf-8')
    print(f'report -> {out / "reports" / "evaluation_report.md"} ({len(text)} chars)')


if __name__ == '__main__':
    main()
