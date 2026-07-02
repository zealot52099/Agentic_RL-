#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yans2@xiaopeng.com/agentic_rl_pipeline"
cd "$ROOT"

INITIAL_RUN="${1:-phase19_sql_grounding_sft_20260702_220500_phase19}"
STATE_DIR="${STATE_DIR:-$ROOT/runs/auto_sql_grounding_loop}"
MAX_ROUNDS="${MAX_ROUNDS:-3}"
MIN_GAIN="${MIN_GAIN:-0.005}"
TARGET_ACCURACY="${TARGET_ACCURACY:-0.62}"
POLL_SECONDS="${POLL_SECONDS:-300}"
BASELINE_ACCURACY="${BASELINE_ACCURACY:-0.52734375}"

mkdir -p "$STATE_DIR" logs

export LD_LIBRARY_PATH=/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export NCCL_SOCKET_IFNAME=hpn0
export NCCL_IB_HCA=
export PATH=/opt/ac2/bin:${PATH}
export TOKENIZERS_PARALLELISM=false

nohup /opt/ac2/bin/python scripts/remote/auto_sql_grounding_loop.py \
  --initial-run "$INITIAL_RUN" \
  --baseline-accuracy "$BASELINE_ACCURACY" \
  --target-accuracy "$TARGET_ACCURACY" \
  --min-gain "$MIN_GAIN" \
  --max-rounds "$MAX_ROUNDS" \
  --poll-seconds "$POLL_SECONDS" \
  --state-dir "$STATE_DIR" \
  > "$STATE_DIR/auto_loop.stdout.log" 2>&1 &

echo $! > "$STATE_DIR/auto_loop.pid"
echo "started auto SQL grounding loop pid $(cat "$STATE_DIR/auto_loop.pid")"
