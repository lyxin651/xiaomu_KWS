# 第四阶段：14 种语音增强算法对两种 KWS 模型的影响

在不训练模型、不修改权重、不调整增强算法的前提下，把同一批音频分别送入两个已部署 KWS，
比较**原始音频**与**14 种语音增强算法输出**，量化"增强是否降低误唤醒、是否损失真实唤醒"。

- 模型：SORT（`kws_cmvn_g1_e100.onnx`，阈值 0.84，连续 3 帧 + 500 ms 冷却）、
  CTC（`kws_ctc_v2verifier_e7_int8.onnx`，阈值 −0.10，500 ms 冷却）
- 数据集：`kws_dataset`（392 条挂测回采 = 21 真实唤醒 + 371 误唤醒）、
  `kws_dataset_1016_5`（928 条纯唤醒，全部为「小沐小沐」）
- 规模：15 个条件（raw + 14 增强）× 2 模型 × (392 + 928) = **39,600 次推理，0 错误**
- 口径：16 kHz、30 ms chunk、每条音频新建 detector（init cache / fbank / matcher / cooldown 全复位）

## 增强算法与数据来源

| 结果中的方法名 | 增强算法 | 增强音频来源 |
|---|---|---|
| `FastEnhancer_T` | FastEnhancer-T（本地早期版本） | `data/enhanced/FastEnhancer_T` |
| `RNNoise` | RNNoise（本地早期版本） | `data/enhanced/RNNoise` |
| `GTCRN-DNS3` | GTCRN（DNS3 训练） | `se_outputs/gtcrn_dns3` |
| `CoFi-Lite` | CoFi-Lite | `se_outputs/cofi_lite` |
| `FastEnhancer-T` | FastEnhancer-T（DNS3） | `se_outputs/fastenhancer_t_dns3` |
| `UL-UNAS` | UL-UNAS | `se_outputs/ul_unas_dns3` |
| `RNNoise-little` | RNNoise little | `se_outputs/rnnoise_little` |
| `LiSenNet` | LiSenNet | `se_outputs/lisennet_vctk` |
| `FSPEN` | FSPEN | `se_outputs/fspen_vctk` |
| `WebRTC-NS` | WebRTC 噪声抑制（修正版） | `se_outputs/webrtc_ns_fixed` |
| `SpeexDSP-NS` | SpeexDSP 噪声抑制 | `se_outputs/speex_ns` |
| `OMLSA-IMCRA` | OMLSA + IMCRA | `se_outputs/omlsa_imcra` |
| `Wiener-Filter` | Wiener 滤波（调参版） | `se_outputs/wiener_tuned` |
| `TSDCA-BA` | TSDCA-BA（修正版） | `se_outputs/tsdca_ba_fixed` |

数据侧为软链接，未复制音频：`data/enhanced/<方法>/{kws_dataset,kws_dataset_1016_5}` 分别指向
`<来源>/kws_dataset/audio`（392 条）与 `<来源>/audio`（928 条）。

## 结果 A：kws_dataset（21 唤醒 / 371 误唤醒）

| 模型 | 方法 | TP | FN | FP | TN | Recall | 误触发率 |
|---|---|---|---|---|---|---|---|
| SORT | 原始 | 7 | 14 | 102 | 269 | 33.33% | 27.49% |
| SORT | FastEnhancer_T | 0 | 21 | 19 | 352 | 0.00% | 5.12% |
| SORT | RNNoise | 3 | 18 | 18 | 353 | 14.29% | 4.85% |
| SORT | GTCRN-DNS3 | 7 | 14 | 34 | 337 | 33.33% | 9.16% |
| SORT | CoFi-Lite | 8 | 13 | 46 | 325 | 38.10% | 12.40% |
| SORT | FastEnhancer-T | 1 | 20 | 23 | 348 | 4.76% | 6.20% |
| SORT | UL-UNAS | 6 | 15 | 60 | 311 | 28.57% | 16.17% |
| SORT | RNNoise-little | 2 | 19 | 17 | 354 | 9.52% | 4.58% |
| SORT | LiSenNet | 0 | 21 | 2 | 369 | 0.00% | 0.54% |
| SORT | FSPEN | 0 | 21 | 8 | 363 | 0.00% | 2.16% |
| SORT | WebRTC-NS | 9 | 12 | 45 | 326 | 42.86% | 12.13% |
| SORT | SpeexDSP-NS | 12 | 9 | 60 | 311 | 57.14% | 16.17% |
| SORT | OMLSA-IMCRA | 11 | 10 | 47 | 324 | 52.38% | 12.67% |
| SORT | Wiener-Filter | 7 | 14 | 67 | 304 | 33.33% | 18.06% |
| SORT | TSDCA-BA | 1 | 20 | 4 | 367 | 4.76% | 1.08% |
| CTC | 原始 | 14 | 7 | 14 | 357 | 66.67% | 3.77% |
| CTC | FastEnhancer_T | 0 | 21 | 0 | 371 | 0.00% | 0.00% |
| CTC | RNNoise | 2 | 19 | 1 | 370 | 9.52% | 0.27% |
| CTC | GTCRN-DNS3 | 10 | 11 | 1 | 370 | 47.62% | 0.27% |
| CTC | CoFi-Lite | 10 | 11 | 1 | 370 | 47.62% | 0.27% |
| CTC | FastEnhancer-T | 0 | 21 | 1 | 370 | 0.00% | 0.27% |
| CTC | UL-UNAS | 8 | 13 | 1 | 370 | 38.10% | 0.27% |
| CTC | RNNoise-little | 1 | 20 | 2 | 369 | 4.76% | 0.54% |
| CTC | LiSenNet | 0 | 21 | 0 | 371 | 0.00% | 0.00% |
| CTC | FSPEN | 0 | 21 | 0 | 371 | 0.00% | 0.00% |
| CTC | WebRTC-NS | 10 | 11 | 2 | 369 | 47.62% | 0.54% |
| CTC | SpeexDSP-NS | 15 | 6 | 2 | 369 | 71.43% | 0.54% |
| CTC | OMLSA-IMCRA | 11 | 10 | 4 | 367 | 52.38% | 1.08% |
| CTC | Wiener-Filter | 7 | 14 | 2 | 369 | 33.33% | 0.54% |
| CTC | TSDCA-BA | 1 | 20 | 1 | 370 | 4.76% | 0.27% |

ΔRecall / ΔFP、McNemar p 值与逐样本恢复/漏检 ID 见 [`results/report.md`](results/report.md)。

## 结果 B：kws_dataset_1016_5（928 条全正样本）

| 模型 | 方法 | TP | FN | Recall | 相对原始 |
|---|---|---|---|---|---|
| SORT | 原始 | 920 | 8 | 99.14% | — |
| SORT | FastEnhancer_T | 393 | 535 | 42.35% | −56.79 pp |
| SORT | RNNoise | 906 | 22 | 97.63% | −1.51 pp |
| SORT | GTCRN-DNS3 | 921 | 7 | 99.25% | +0.11 pp |
| SORT | CoFi-Lite | 918 | 10 | 98.92% | −0.22 pp |
| SORT | FastEnhancer-T | 752 | 176 | 81.03% | −18.11 pp |
| SORT | UL-UNAS | 928 | 0 | 100.00% | +0.86 pp |
| SORT | RNNoise-little | 917 | 11 | 98.81% | −0.33 pp |
| SORT | LiSenNet | 353 | 575 | 38.04% | −61.10 pp |
| SORT | FSPEN | 40 | 888 | 4.31% | −94.83 pp |
| SORT | WebRTC-NS | 915 | 13 | 98.60% | −0.54 pp |
| SORT | SpeexDSP-NS | 919 | 9 | 99.03% | −0.11 pp |
| SORT | OMLSA-IMCRA | 916 | 12 | 98.71% | −0.43 pp |
| SORT | Wiener-Filter | 904 | 24 | 97.41% | −1.73 pp |
| SORT | TSDCA-BA | 899 | 29 | 96.88% | −2.26 pp |
| CTC | 原始 | 926 | 2 | 99.78% | — |
| CTC | FastEnhancer_T | 231 | 697 | 24.89% | −74.89 pp |
| CTC | RNNoise | 907 | 21 | 97.74% | −2.04 pp |
| CTC | GTCRN-DNS3 | 890 | 38 | 95.91% | −3.87 pp |
| CTC | CoFi-Lite | 902 | 26 | 97.20% | −2.58 pp |
| CTC | FastEnhancer-T | 683 | 245 | 73.60% | −26.18 pp |
| CTC | UL-UNAS | 909 | 19 | 97.95% | −1.83 pp |
| CTC | RNNoise-little | 909 | 19 | 97.95% | −1.83 pp |
| CTC | LiSenNet | 317 | 611 | 34.16% | −65.62 pp |
| CTC | FSPEN | 26 | 902 | 2.80% | −96.98 pp |
| CTC | WebRTC-NS | 874 | 54 | 94.18% | −5.60 pp |
| CTC | SpeexDSP-NS | 895 | 33 | 96.44% | −3.34 pp |
| CTC | OMLSA-IMCRA | 900 | 28 | 96.98% | −2.80 pp |
| CTC | Wiener-Filter | 804 | 124 | 86.64% | −13.14 pp |
| CTC | TSDCA-BA | 701 | 227 | 75.54% | −24.24 pp |

纯唤醒集没有负样本，误触发率不适用；该表只衡量增强对召回的影响。

## 主要发现

1. **误唤醒与召回此消彼长**：14 种算法里没有一个能在 kws_dataset 上同时做到
   "召回不低于原始 + 误触发大幅下降 + 纯唤醒集召回几乎不掉"。
2. **SORT 最稳的是 GTCRN-DNS3**：召回 33.33%（与原始持平），误触发 27.49%→9.16%（−2/3），
   纯唤醒集 99.25%（+1 条），是唯一"零召回代价"的降误触发方案。
3. **CTC 最优是 SpeexDSP-NS**：召回 66.67%→71.43%，误触发 3.77%→0.54%，但纯唤醒集
   99.78%→96.44%（−33 条），说明提升来自"更容易触发"，存在过度触发风险。
4. **激进降噪会杀唤醒**：FSPEN（SORT 4.31% / CTC 2.80%）、LiSenNet（38.04% / 34.16%）、
   FastEnhancer_T（42.35% / 24.89%）在纯唤醒集上直接崩塌，不能作为 KWS 前置。
5. **同一算法对两种架构影响不对称**：例如 TSDCA-BA 在 kws_dataset 几乎不误触发（SORT 1.08%、CTC 0.27%），
   但纯唤醒集召回掉到 96.88%/75.54%，属于"过度抑制"；Wiener-Filter 相反，SORT 误触发反而升高到 18.06%。
6. 三个已修正算法（WebRTC-NS→`webrtc_ns_fixed`、Wiener-Filter→`wiener_tuned`、TSDCA-BA→`tsdca_ba_fixed`）
   重跑后结论与旧版本明显不同：WebRTC-NS 从"几乎不触发"变为 SORT 42.86%/12.13%、CTC 47.62%/0.54%；
   TSDCA-BA 从"全零"变为 SORT 4.76%、CTC 4.76%，纯唤醒集 96.88%/75.54%。

## 修正记录（重要）

1. 旧版本的 `WebRTC-NS` / `Wiener-Filter` / `TSDCA-BA` 结果产生于 11-24 20:18–23:50，
   而软链接切换到 fixed 版本是 11-25 19:22，因此 11-26 那次"重跑"被断点续跑逻辑按
   `status=ok` 全部跳过（日志显示 `todo=0`），报告里仍是旧音频的分数。
   本次已用 `--no-resume` 强制重算这 3 个算法 × 2 数据集 × 2 模型（7,920 次推理）。
2. 历史 CSV 存在重复追加行（同一样本多行）。已核对重复行之间 `triggered` 完全一致，
   并用 `scripts/` 外的去重步骤统一为"每样本一行"，当前每个文件正好 392 / 928 行。
3. `results/report.md` 的首段结论改为由结果自动生成，不再写死数字。

## 目录

```text
kws_enhancer_comparison/
├── config.json               # 数据集 / 正样本 ID / 15 个条件 / 模型
├── scripts/
│   ├── evaluate_kws.py       # 核心推理：复用部署包类，30 ms chunk，逐样本复位
│   ├── run_matrix.py         # 条件 × 数据集 × 模型 循环 + 断点续跑 + --dry-run
│   ├── run_limited.py        # 限制 ORT 线程后运行（多进程并发必需）
│   ├── analyze_results.py    # 指标 / 配对 / McNemar / report.md / summary.json
│   ├── run_all.sh            # 一键跑完
│   └── upload_to_server.sh   # 本地 → 服务器 上传（macOS openrsync/ssh 适配）
└── results/
    ├── report.md             # 汇总报告（含 Δ、p 值、逐样本 ID）
    ├── summary.json          # 机器可读指标
    ├── paired_samples.csv    # 原始 vs 各增强 逐样本配对
    ├── kws_dataset/*.csv     # 30 份逐条结果（15 条件 × 2 模型）
    └── kws_dataset_1016_5/*.csv
```

## 复现

服务器目录 `/home/leiyuxin/code/gongniu`（已通过稀疏检出与仓库关联）：

```bash
# 依赖
python3 -m venv .venv && source .venv/bin/activate
python3 -m pip install -r requirements.txt      # numpy / onnxruntime / soundfile

# 软链接增强音频（每个方法两个数据集各一条）
ln -s /home/guoxin/kws_SE/se_outputs/<algo>/kws_dataset/audio data/enhanced/<方法>/kws_dataset
ln -s /home/guoxin/kws_SE/se_outputs/<algo>/audio             data/enhanced/<方法>/kws_dataset_1016_5

# 校验 + 全量（39,600 次推理，可断点续跑）
.venv/bin/python scripts/run_matrix.py --dry-run
ORT_THREADS=4 .venv/bin/python scripts/run_limited.py
.venv/bin/python scripts/analyze_results.py
```

## 口径与限制

- 16 kHz、30 ms（480 samples）chunk；每条音频重新创建 detector；ONNX session 进程内共享。
- SORT 分数 = 逐帧唤醒概率；CTC 分数 = 关键词路径 mean per-token log-LR（nats/token），不可跨模型比较绝对值。
- CTC 使用 INT8 模型，对 chunk 长度敏感，本结论仅对应部署口径 30 ms。
- `kws_dataset` 正样本仅 21 条，比例指标置信区间宽，结论以配对检验与具体 ID 为准。
- 误触发率是 371 条回采负样本的触发比例，不等于部署场景 FA/h。
- 单条推理失败写 `status=error` 单独计数，不静默当作"未触发"（本次为 0）。
- 增强音频来自 `/home/guoxin/kws_SE/se_outputs/`，未入库；仓库只保留推理结果。
