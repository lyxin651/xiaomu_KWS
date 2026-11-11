# xiaomu_KWS — 双模型语音增强前后效果验证

在**不训练、不改权重、不改增强算法**的前提下，评估同一套 GTCRN 语音增强对两种已部署 KWS 架构
（SORT 与 CTC）的实际影响：是否降低误唤醒、是否造成真实唤醒漏检、两种架构响应差异、具体改善/退化样本。

- 数据集：392 条挂测回采音频（21 条真实唤醒「小沐小沐」+ 371 条误唤醒），每条均有原始与增强版本
- 模型：SORT `kws_cmvn_g1_e100.onnx`（thr=0.84）、CTC `kws_ctc_v2verifier_e7_int8.onnx`（thr=-0.10）
- 推理：30 ms chunk 离线流式，复用部署包原 `StreamingFbank` 与判决类，逐样本复位状态
- 规模：4 组 × 392 条 = **1568 次推理，失败 0 条**

## 结果速览

| 组别 | TP | FN | FP | TN | Recall | FPR |
|---|---|---|---|---|---|---|
| A SORT / 原始 | 7 | 14 | 102 | 269 | 33.33% | 27.49% |
| B SORT / 增强 | 7 | 14 | 34 | 337 | 33.33% | 9.16% |
| C CTC / 原始 | 14 | 7 | 14 | 357 | 66.67% | 3.77% |
| D CTC / 增强 | 10 | 11 | 1 | 370 | 47.62% | 0.27% |

- SORT：负样本误触发抑制 66.67%（p≈7.2e-18）；Recall 净不变，漏检 121、恢复 119。
- CTC：负样本误触发抑制 92.86%（p≈2.4e-4，0 条新增）；但 Recall 14→10，漏检 7/25/122/123/125、恢复 32（p=0.219，正样本仅 21 条）。
- 结论：增强对两者误唤醒都有效；CTC 在当前工作点属"过度抑制"，正样本分数中位下降 1.28 nats/token，SORT 仅 -0.015。
- 注意：FPR 是 371 条回采负样本的触发比例，不是 FA/h；结论仅对应 30 ms chunk（CTC INT8 对 chunk 长度敏感）。

完整分析见 [`reports/evaluation_report.md`](reports/evaluation_report.md)。

## 第二阶段：GTCRN 输出优化（不改 KWS 模型）

在上述基线上继续做三组实验：阈值/判决诊断、增强输出消融（固定增益 +1~+4 dB、raw↔enh RMS 匹配、干湿混合 α=0.9/0.75/0.5）、漏检归因（新增漏检 7/25/122/123/125 与 SORT 121）。

| 方案 | SORT TP/FP | CTC TP/FP |
|---|---|---|
| 现有增强（基线） | 7 / 34 | 10 / 1 |
| GTCRN +2 dB | 8 / 36 | **13 / 0** |
| 增强逐条匹配 raw RMS（仅归因） | 9 / 36 | 15 / 1 |
| raw | 7 / 102 | 14 / 14 |

- 仅调 GTCRN 输出即可把 CTC 从 10/21 恢复到 13/21 且误触发 1→0，SORT 的误触发抑制收益保持（34→36）。
- 5 条新增漏检中，固定增益只能救回 25（+2 dB）与 7（+4 dB）；122 需阈值 ≤−1.35 或掺 raw；123/125 只能在干湿混合下恢复。121（SORT）仅 mix ≥0.9 可恢复。
- 修正了旧 chunk 测试把 blank 分数当关键词路径分数的问题：FP32 对 10/30/100 ms 完全不敏感（Δ=0），INT8 存在 0.3–0.6 nats/token 的 chunk 相关抖动。

完整报告与逐样本结果见 [`kws_enhancement_opt/reports/report.md`](kws_enhancement_opt/reports/report.md)。


## 目录结构

```text
.
├── config.yaml                 # 主实验固定参数（模型、阈值、数据、指标定义）
├── labels.csv                  # 392 行：sample_id,label,raw_path,enhanced_path（绝对路径）
├── data_validation.json        # 配对/格式/长度/时延/异常检查（含逐文件统计）
├── results/
│   ├── sort_raw.csv            # 392 行逐条推理结果（A 组）
│   ├── sort_enhanced.csv       # 392 行（B 组）
│   ├── ctc_raw.csv             # 392 行（C 组）
│   ├── ctc_enhanced.csv        # 392 行（D 组）
│   ├── paired_comparison.csv   # 784 行 = 2 模型 × 392 样本，raw/enhanced 配对
│   ├── summary.json            # 全部指标、配对计数、FP/FN ID 列表、跨模型分析
│   └── frame_scores/*.npz      # 逐帧分数曲线（key=sample_id）
├── figures/
│   ├── confusion_matrices.png
│   ├── recall_fpr_comparison.png
│   ├── paired_transitions.png
│   └── selected_cases/*.png    # 典型案例 raw/enhanced 分数曲线
├── reports/evaluation_report.md
├── logs/                       # 批量推理日志、测试日志
└── scripts/
    ├── prepare_dataset.py      # 生成 labels.csv + data_validation.json
    ├── evaluate_kws.py         # 四组批量流式推理（支持断点续跑）
    ├── test_kws_eval.py        # T1–T5 标签/复位/流式等价/官方一致性/小样本
    ├── analyze_results.py      # 混淆矩阵、配对统计、图表、summary.json
    └── generate_report.py      # 由 summary.json 生成评估报告
```

`deploy/` 下为两种模型的部署推理源码（`mic_kws.py`/`mic_ctc.py`/`streaming_fbank.py`），
**不含** ONNX 权重、初始 cache 与音频数据（需在本地部署环境提供）。

## 逐条结果字段

`results/*.csv` 每条包含：`sample_id, label, model_name, model_file, audio_condition, threshold,
triggered, num_triggers, first_trigger_time_ms, trigger_scores, audio_duration_sec,
inference_time_ms, status, num_frames, frame_score_max, frame_score_p95, score_kind,
trigger_times_ms, error`。

- SORT 分数 = 每帧唤醒概率；CTC 分数 = 关键词路径平均每 token log-LR（nats/token），两者不可直接比较绝对值。
- `status=error` 的样本不会被静默计为"未触发"，需单独统计（本次为 0）。

## 复现

依赖（Python 3.9）：见 `requirements.txt`。需要本地存在以下资源（未入库）：

```text
kws_dataset/              # 392 条原始 WAV，c100-01-<ID>.wav
gtcrn_enhanced/           # 392 条增强 WAV，同 ID
deploy/deploy_sort/       # kws_cmvn_g1_e100.onnx + _init_cache.npy + mic_kws.py + streaming_fbank.py
deploy/deploy_ctc/        # kws_ctc_v2verifier_e7_int8.onnx + _init_cache.npy + mic_ctc.py + streaming_fbank.py
```

```bash
python scripts/prepare_dataset.py --raw-dir ../kws_dataset \
    --enhanced-dir ../gtcrn_enhanced --output-dir ..

python scripts/test_kws_eval.py --raw-dir ../kws_dataset \
    --enhanced-dir ../gtcrn_enhanced --deploy-dir ../deploy --output-dir ..

python scripts/evaluate_kws.py --raw-dir ../kws_dataset \
    --enhanced-dir ../gtcrn_enhanced --deploy-dir ../deploy --output-dir ..

python scripts/analyze_results.py --output-dir ..
python scripts/generate_report.py --output-dir ..
```

`evaluate_kws.py` 支持断点续跑：已有且 `status=ok` 的 sample_id 自动跳过；`labels.csv` 中路径为
生成机器上的绝对路径，换机器复现时需重新生成。

## 验证记录

| 项 | 内容 | 结果 |
|---|---|---|
| T1 | 392 条、21 正样本、缺失编号 98/99/100/222 | PASS |
| T2 | 状态完全复位（连续/交错重复推理逐位一致） | PASS |
| T3 | 流式等价：SORT、CTC-float 在 10/30/100 ms 一致；CTC-INT8 随 chunk 变化 | 已记录 |
| T4 | 与官方 `mic_ctc.py --wav` 帧数/触发/分数一致（id=0/2/101） | PASS |
| T5 | 2 正 + 2 负 × 2 模型 × 2 条件 | PASS |

## 适用范围与限制

1. 正样本仅 21 条，比例指标置信区间宽；负样本比较使用精确 McNemar。
2. FPR 为回采负样本触发比例，不能外推为部署 FA/h。
3. CTC INT8 模型输出依赖 chunk 长度（同一句 10 vs 30 ms 路径分数最大差约 1.3–1.9 nats/token，均值 0.3–0.4），
   本结论仅对应部署口径 30 ms；FP32 模型对 chunk 长度完全不敏感（第二阶段已验证）。
4. 增强后整体电平略降（中位 RMS 0.00286→0.00217，−1.98 dB）；第二阶段已用固定增益/RMS 匹配/干湿混合拆分电平与内容失真，结论见 `kws_enhancement_opt/reports/report.md`。
5. 未评估连续自然运行 FA/h 及 `kws_dataset_1016_5` 纯唤醒语料；第二阶段阈值扫描仅用于归因，候选方案仍需独立数据验证。
