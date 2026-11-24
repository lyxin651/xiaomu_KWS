# 第四阶段：FastEnhancer_T / RNNoise 对两种 KWS 模型的影响

在**不训练、不改权重、不改判决逻辑**的前提下，把同一批音频分别送入两个已部署 KWS 模型，
比较 **原始音频** 与 **两种语音增强算法输出**：

- 增强算法：`FastEnhancer_T`、`RNNoise`
- KWS 模型：SORT（`kws_cmvn_g1_e100.onnx`，阈值 0.84）、CTC（`kws_ctc_v2verifier_e7_int8.onnx`，阈值 −0.10）
- 数据集：`kws_dataset`（392 条挂测回采：21 条真实唤醒 + 371 条误唤醒）、
  `kws_dataset_1016_5`（928 条纯唤醒，全部为「小沐小沐」）
- 规模：2 数据集 × 3 条件（raw / FastEnhancer_T / RNNoise）× 2 模型 = **7920 次推理，0 错误**

推理在 GPU 服务器上完成（Python 3.8 + onnxruntime 1.19.2 CPU），产物已回传本目录 `results/`。

## 结果

### A. `kws_dataset`（21 唤醒 / 371 误唤醒）

| 模型 | 输入 | TP | FN | FP | TN | Recall | 误触发率 |
|---|---|---|---|---|---|---|---|
| SORT | 原始 | 7 | 14 | 102 | 269 | 33.33% | 27.49% |
| SORT | FastEnhancer_T | 0 | 21 | 19 | 352 | 0.00% | 5.12% |
| SORT | RNNoise | 3 | 18 | 18 | 353 | 14.29% | 4.85% |
| CTC | 原始 | 14 | 7 | 14 | 357 | 66.67% | 3.77% |
| CTC | FastEnhancer_T | 0 | 21 | 0 | 371 | 0.00% | 0.00% |
| CTC | RNNoise | 2 | 19 | 1 | 370 | 9.52% | 0.27% |

- 负样本误触发被显著抑制（SORT 102→19/18，CTC 14→0/1；McNemar p ≤ 1.2e-4 与 0）。
- 但正样本召回同步崩塌：SORT 7→0/3，CTC 14→0/2（p 分别为 0.0156 / 0.219 与 1.2e-4 / 4.9e-4）。

### B. `kws_dataset_1016_5`（928 条全正样本）

| 模型 | 原始 | FastEnhancer_T | RNNoise |
|---|---|---|---|
| SORT | 920（99.14%） | 393（42.35%） | 906（97.63%） |
| CTC | 926（99.78%） | 231（24.89%） | 907（97.74%） |

- FastEnhancer_T 在纯唤醒集上同样灾难性掉点（SORT −56.8 pp、CTC −74.9 pp）。
- RNNoise 对干净近场语料几乎无损（−1.5 pp / −2.0 pp）。

## 为什么召回会塌：增强把唤醒当噪声压掉了

逐条对齐已排除文件错配（时长一致、互相关峰在 lag=0、语音段局部相关 0.96–1.0）。
真正的机制是**增强器对真实唤醒的衰减远大于对误唤醒的衰减**（RMS 衰减中位数）：

| 方法 | 21 条真实唤醒 | 371 条误唤醒（抽样） |
|---|---|---|
| FastEnhancer_T | **−11.3 dB**（波形相关 0.51） | −4.7 dB（0.78） |
| RNNoise | **−8.8 dB**（0.63） | −5.3 dB（0.76） |

- 挂测回采音频本身电平很低（中位 −51 dBFS，远场/设备麦），增强后进一步压低，模型看到近似静音。
- RNNoise 在干净近场 `kws_dataset_1016_5` 上只衰减 1.1 dB，所以召回几乎不掉（97.6%）；
  一旦换成低电平远场音频就衰减 8.8 dB，召回从 14/21 掉到 2/21。
- FastEnhancer_T 两种场景都塌：1016_5 衰减 7.2 dB 且相关仅 0.69，失真更重。

## 结论

1. 这两个增强器**不能直接前置到 KWS**：均表现为"误唤醒抑制有效、真实唤醒同时被杀"，无法直接上线。
2. **RNNoise 明显比 FastEnhancer_T 温和**，风险主要在低电平远场场景。
3. 与第二阶段 GTCRN 结论一致：主因是**输出衰减 + 频谱失真**；可行方向是输出增益补偿、
   干湿混合（掺回原始音频）、或先 AGC 再增强，在保留误唤醒抑制的同时救回召回。

## 目录

```text
kws_enhancer_comparison/
├── config.json               # 数据集 / 正样本 ID / 方法 / 模型定义
├── scripts/
│   ├── evaluate_kws.py       # 核心：复用部署包类，30 ms chunk，逐样本状态复位
│   ├── run_matrix.py         # 数据集 × 方法 × 模型 循环 + 断点续跑 + --dry-run
│   ├── analyze_results.py    # 指标、配对、McNemar、report.md / summary.json
│   ├── run_all.sh            # 一键跑完
│   └── upload_to_server.sh   # 本地 → 服务器 目录映射上传（macOS openrsync/ssh 适配）
└── results/
    ├── report.md             # 服务器端自动生成的汇总报告
    ├── summary.json          # 机器可读指标（含 p 值与漏检/误触发 ID）
    ├── paired_samples.csv    # 原始 vs 增强 逐样本配对
    ├── kws_dataset/*.csv     # 6 组逐条推理结果
    └── kws_dataset_1016_5/*.csv
```

## 复现

服务器目录约定（相对仓库根）：

```text
data/raw/{kws_dataset,kws_dataset_1016_5}/c100-01-<id>[_raw].wav
data/enhanced/<method>/<dataset>/<同名文件>
deploy/deploy_sort/  deploy/deploy_ctc/        # 源码 + onnx + init cache
```

```bash
python3 -m pip install numpy onnxruntime soundfile
python3 scripts/run_matrix.py --dry-run        # 校验文件与任务矩阵
python3 scripts/run_matrix.py --limit 5 --no-resume   # 冒烟
bash scripts/run_all.sh                        # 全量（7920 次推理）+ 汇总
```

## 口径与限制

- 16 kHz、30 ms（480 samples）chunk；每条音频重新创建 detector，ONNX init cache / fbank /
  matcher / cooldown 全部复位；ONNX session 进程内共享。
- SORT：0.84、连续 3 帧、500 ms 冷却；CTC：−0.10（关键词路径 mean per-token log-LR）、500 ms 冷却。
- CTC 使用 INT8 模型，对 chunk 长度敏感；本结论仅对应部署口径 30 ms。
- `kws_dataset` 正样本仅 21 条，比例指标置信区间宽，结论以配对检验与具体 ID 为准。
- 误触发率是 371 条回采负样本的触发比例，不等于部署场景 FA/h。
- 单条推理失败会写 `status=error` 单独计数，不静默当作"未触发"（本次为 0）。
