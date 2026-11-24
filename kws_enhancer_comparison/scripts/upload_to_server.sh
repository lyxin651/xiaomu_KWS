#!/usr/bin/env bash
# 把全部实验文件按目录结构上传到服务器 /home/leiyuxin/code/gongniu
#
# 用法:
#   bash upload_to_server.sh              # 正式上传
#   bash upload_to_server.sh --dry-run    # 只看会传哪些文件，不落盘
#   MULTIPLEX=0 bash upload_to_server.sh  # 关闭连接复用（每次传输各输一次密码）
#
# 说明:
#   * 服务器只开放密码认证（~/.ssh/config 里 PubkeyAuthentication no）。
#     这里用 ControlMaster 复用连接，正常情况下整轮只输一次密码。
#   * 刻意不读 ~/.ssh/config（-F /dev/null）：该配置带有 RemoteForward /
#     ExitOnForwardFailure，会干扰纯文件传输的连接。
#   * 可覆盖: SSH_USER / SSH_ADDR / SSH_PORT / REMOTE
set -euo pipefail

SSH_USER="${SSH_USER:-leiyuxin}"
SSH_ADDR="${SSH_ADDR:-125.220.134.27}"
SSH_PORT="${SSH_PORT:-4090}"
REMOTE="${REMOTE:-/home/leiyuxin/code/gongniu}"
MULTIPLEX="${MULTIPLEX:-1}"

WORK="$(cd "$(dirname "$0")" && pwd)"
PKG="$WORK/server_package"
OUT="$WORK/kws_enhancement_experiment/outputs"
RAW_SURVEY="$HOME/Downloads/kws_dataset"
RAW_PURE="$HOME/Downloads/deploy/kws_dataset_1016_5"

DRY=""
if [ "${1:-}" = "--dry-run" ]; then DRY="--dry-run"; fi

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
need() { [ -e "$1" ] || { printf '缺少本地路径: %s\n' "$1" >&2; exit 1; }; }

for p in "$PKG/README.md" "$PKG/config.json" "$PKG/deploy/deploy_sort" \
         "$PKG/deploy/deploy_ctc" "$RAW_SURVEY" "$RAW_PURE" \
         "$OUT/FastEnhancer_T" "$OUT/RNNoise"; do
  need "$p"
done

# ControlPath 必须很短：macOS 的 $TMPDIR 路径很长，加上 user@host:port 会超过
# Unix socket 的 104 字节上限（报错 "too long for Unix domain socket"）。
SSH_OPTS=(-F /dev/null -p "$SSH_PORT" -o "User=$SSH_USER"
          -o ConnectTimeout=15 -o NumberOfPasswordPrompts=3
          -o ServerAliveInterval=30)
if [ "$MULTIPLEX" = "1" ]; then
  SSH_OPTS+=(-o ControlMaster=auto -o ControlPath=/tmp/gcm-%C -o ControlPersist=15m)
fi
SSH=(ssh "${SSH_OPTS[@]}")
SSH_STR="${SSH[*]}"

say "0/6 连接服务器并建立目录"
if ! out=$("${SSH[@]}" "$SSH_ADDR" \
      "mkdir -p '$REMOTE'/deploy '$REMOTE'/data/raw '$REMOTE'/data/enhanced '$REMOTE'/scripts '$REMOTE'/results && echo 'remote ok: $REMOTE'" 2>&1); then
  printf '%s\n' "$out"
  case "$out" in
    *"Permission denied"*)
      cat >&2 <<EOF

SSH 认证被拒绝：服务器不接受这个密码。脚本本身没问题，请先单独验证账号：

    ssh -p $SSH_PORT $SSH_USER@$SSH_ADDR

如果这条命令也报 Permission denied，说明密码/账号/服务器侧限制有问题
（例如密码改了、账号被锁、或需要先连 VPN），与上传脚本无关。
EOF
      ;;
    *"Connection refused"*|*"timed out"*|*"No route to host"*)
      printf '\n连不上服务器（网络或端口 %s 不通），请确认校园网/VPN。\n' "$SSH_PORT" >&2 ;;
    *"too long for Unix domain socket"*)
      printf '\n控制 socket 路径过长，请用 MULTIPLEX=0 bash upload_to_server.sh 重试。\n' >&2 ;;
    *) printf '\nSSH 失败，请检查上面的原始输出。\n' >&2 ;;
  esac
  exit 1
fi
printf '%s\n' "$out"

RS=(rsync -a --exclude '.DS_Store' --exclude '__pycache__' -e "$SSH_STR")
if [ -n "$DRY" ]; then RS+=("$DRY"); fi

# macOS 自带的 openrsync 在“目标目录尚不存在”时会在建目录那一步报
# "unexpected end of file"，所以统一先 mkdir -p 再传。
mkdir_remote() {
  "${SSH[@]}" "$SSH_ADDR" "mkdir -p $(printf "'%s' " "$@")" < /dev/null
}

say "1/6 代码 / 文档 / 模型（约 11 MB）"
mkdir_remote "$REMOTE/scripts" "$REMOTE/deploy/deploy_sort" "$REMOTE/deploy/deploy_ctc"
"${RS[@]}" "$PKG/README.md" "$PKG/requirements.txt" "$PKG/config.json" "$SSH_ADDR:$REMOTE/"
"${RS[@]}" "$PKG/scripts/" "$SSH_ADDR:$REMOTE/scripts/"
"${RS[@]}" "$PKG/deploy/" "$SSH_ADDR:$REMOTE/deploy/"

say "2/6 原始音频 kws_dataset（392 条，含 21 条唤醒）"
mkdir_remote "$REMOTE/data/raw/kws_dataset"
"${RS[@]}" "$RAW_SURVEY/" "$SSH_ADDR:$REMOTE/data/raw/kws_dataset/"

say "3/6 原始音频 kws_dataset_1016_5（928 条纯唤醒，约 94 MB）"
mkdir_remote "$REMOTE/data/raw/kws_dataset_1016_5"
"${RS[@]}" "$RAW_PURE/" "$SSH_ADDR:$REMOTE/data/raw/kws_dataset_1016_5/"

say "4/6 增强音频 FastEnhancer_T（392 + 928 条）"
for ds in kws_dataset kws_dataset_1016_5; do
  mkdir_remote "$REMOTE/data/enhanced/FastEnhancer_T/$ds"
  "${RS[@]}" "$OUT/FastEnhancer_T/$ds/" "$SSH_ADDR:$REMOTE/data/enhanced/FastEnhancer_T/$ds/"
done

say "5/6 增强音频 RNNoise（392 + 928 条）"
for ds in kws_dataset kws_dataset_1016_5; do
  mkdir_remote "$REMOTE/data/enhanced/RNNoise/$ds"
  "${RS[@]}" "$OUT/RNNoise/$ds/" "$SSH_ADDR:$REMOTE/data/enhanced/RNNoise/$ds/"
done

say "6/6 远端文件数校验（期望 392 / 928 / 392 / 928 / 392 / 928）"
"${SSH[@]}" "$SSH_ADDR" "
cd '$REMOTE' || exit 1
for d in data/raw/kws_dataset data/raw/kws_dataset_1016_5 \
         data/enhanced/FastEnhancer_T/kws_dataset data/enhanced/FastEnhancer_T/kws_dataset_1016_5 \
         data/enhanced/RNNoise/kws_dataset data/enhanced/RNNoise/kws_dataset_1016_5; do
  printf '%-58s %4s wav\n' \"\$d\" \"\$(ls \$d/*.wav 2>/dev/null | wc -l | tr -d ' ')\"
done
printf '%-58s %s\n' '总占用' \"\$(du -sh . | cut -f1)\"
printf '%-58s %s\n' 'python3' \"\$(python3 -V 2>&1)\"
printf '%-58s %s\n' 'onnxruntime' \"\$(python3 -c 'import onnxruntime;print(onnxruntime.__version__)' 2>&1 | tail -1)\"
" || true

cat <<EOF

上传结束。下一步在服务器上执行：

  ssh -p $SSH_PORT $SSH_USER@$SSH_ADDR
  cd $REMOTE
  python3 -m venv .venv && source .venv/bin/activate
  python3 -m pip install -r requirements.txt
  python3 scripts/run_matrix.py --dry-run     # 校验路径与任务矩阵
  bash scripts/run_all.sh                     # 全量评测 + 汇总报告

结果在 results/report.md。
EOF
