#!/usr/bin/env bash
# 一键跑完整个评测矩阵（原始 + 两种增强 × 两个数据集 × 两个模型）
# 用法:
#   bash scripts/run_all.sh            # 全量，自动断点续跑
#   bash scripts/run_all.sh --limit 5  # 冒烟测试（每组只跑前 5 条）
set -euo pipefail
cd "$(dirname "$0")/.."

python3 - <<'PY'
import sys
missing = []
for mod in ('numpy', 'onnxruntime', 'soundfile'):
    try:
        __import__(mod)
    except ImportError:
        missing.append(mod)
if missing:
    sys.exit('缺少依赖: ' + ', '.join(missing) +
             '\n请先执行: python3 -m pip install -r requirements.txt')
print('[env] numpy / onnxruntime / soundfile 就绪')
PY

echo '=== 1/2 批量推理 ==='
python3 scripts/run_matrix.py "$@"

echo '=== 2/2 汇总分析 ==='
python3 scripts/analyze_results.py

echo
echo '完成。结果目录: results/'
echo '  - results/report.md          汇总报告'
echo '  - results/summary.json       机器可读指标'
echo '  - results/<数据集>/*.csv      逐条推理结果'
echo '  - results/paired_samples.csv 原始 vs 增强逐样本配对'
