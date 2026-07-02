#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yans2@xiaopeng.com/agentic_rl_pipeline"
cd "$ROOT"

STAMP="${1:-$(date +%Y%m%d_%H%M%S)_phase19}"
RUN_NAME="phase19_sql_grounding_sft_${STAMP}"
RUN_ROOT="$ROOT/runs/$RUN_NAME"
EVAL_ROOT="$ROOT/evals/$RUN_NAME"
DATA_DIR="$ROOT/datasets/processed/phase19_sql_grounding_sft_${STAMP}"
BASE_MODEL="${BASE_MODEL:-$ROOT/evals/phase18_canonical_data_agent_sft_20260702_155000_phase18/merged/phase18_canonical_data_agent_sft_merged}"
MERGED="$EVAL_ROOT/merged/phase19_sql_grounding_sft_merged"
TRAIN_DIR="$RUN_ROOT/${RUN_NAME}_lora"
STEPS="${STEPS:-700}"
LR="${LR:-1.5e-7}"
WARMUP_STEPS="${WARMUP_STEPS:-50}"

mkdir -p "$RUN_ROOT" "$EVAL_ROOT/logs" "$EVAL_ROOT/merged" "$DATA_DIR" logs

export LD_LIBRARY_PATH=/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export NCCL_SOCKET_IFNAME=hpn0
export NCCL_IB_HCA=
export PATH=/opt/ac2/bin:${PATH}
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
export NPROC_PER_NODE=16
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$EVAL_ROOT/phase19_queue.log"
}

stop_keepalive() {
  pkill -f './run_gpu_16.sh' || true
  for f in run_gpu_16.phase18_after_eval.pid run_gpu_16.phase18_after.pid run_gpu_16.phase17_after.pid run_gpu_16.phase16c_after.pid run_gpu_16.phase16_post_eval.pid; do
    if [[ -s "$f" ]]; then
      pid="$(cat "$f")"
      pkill -P "$pid" || true
      kill "$pid" || true
    fi
  done
  for pid in $(pgrep -f '^python -$|^python3 -$' || true); do
    cwd="$(readlink "/proc/$pid/cwd" 2>/dev/null || true)"
    if [[ "$cwd" == "$ROOT"* || "$cwd" == "/code"* ]]; then
      kill "$pid" || true
    fi
  done
}

run_eval() {
  local name="$1"
  shift
  log "starting $name"
  "$@" > "$EVAL_ROOT/logs/${name}.log" 2>&1
  log "completed $name"
}

merge_adapter() {
  local adapter="$1"
  local log_file="$EVAL_ROOT/logs/merge_phase19.log"
  if [[ -s "$MERGED/model.safetensors.index.json" ]]; then
    log "merged model already exists: $MERGED"
    return
  fi
  log "merging adapter: $adapter"
  /opt/ac2/bin/python - <<PY > "$log_file" 2>&1
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "$BASE_MODEL"
adapter = "$adapter"
out = Path("$MERGED")
tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16, device_map="cpu")
model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
model = model.merge_and_unload()
out.mkdir(parents=True, exist_ok=True)
model.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
tok.save_pretrained(out)
(out / "merge_metadata.json").write_text(json.dumps({"base": base, "adapter": adapter}, indent=2) + "\n", encoding="utf-8")
PY
  echo "$MERGED" > "$EVAL_ROOT/final_merged_model_path.txt"
  log "merge complete: $MERGED"
}

start_keepalive_if_idle() {
  if ! ps -eo cmd | grep -E 'train_lora|torchrun|swift rlhf|evaluate_|vllm|run_gpu_16' | grep -v grep >/dev/null; then
    nohup ./run_gpu_16.sh > run_gpu_16.phase19_after.log 2>&1 < /dev/null &
    echo $! > run_gpu_16.phase19_after.pid
    log "started run_gpu_16 keepalive pid $(cat run_gpu_16.phase19_after.pid)"
  fi
}

main() {
  for path in \
    "$BASE_MODEL" \
    scripts/remote/prepare_phase19_sql_grounding_data.py \
    scripts/remote/train_lora_sft.py \
    scripts/remote/evaluate_wikisql_v2.py \
    datasets/processed/phase8_swift_wikisql_grpo_20260629/train.jsonl \
    datasets/processed/phase18_canonical_data_agent_sft_20260702_155000_phase18/train_phase18_canonical_sft.jsonl \
    datasets/processed/phase17_sql_error_sft_20260702_123712_phase17b/wikisql_eval_256_normalized.jsonl \
    datasets/processed/phase17_sql_error_sft_20260702_123712_phase17b/wikisql_eval_256_normalized.sqlite; do
    if [[ ! -e "$path" ]]; then
      log "missing required path: $path"
      exit 2
    fi
  done

  stop_keepalive
  echo "$$" > "$RUN_ROOT/queue.pid"

  cat > "$EVAL_ROOT/phase19_manifest.json" <<JSON
{
  "stamp": "$STAMP",
  "run_name": "$RUN_NAME",
  "base_model": "$BASE_MODEL",
  "method": "Phase19 schema/value-grounded SQL LoRA SFT",
  "steps": $STEPS,
  "learning_rate": $LR,
  "label": "Internal SQL grounding SFT; fixed WikiSQL v2 probe is eval-only"
}
JSON

  log "building Phase19 SQL grounding SFT data"
  /opt/ac2/bin/python scripts/remote/prepare_phase19_sql_grounding_data.py \
    --phase8-wikisql-train datasets/processed/phase8_swift_wikisql_grpo_20260629/train.jsonl \
    --phase18-train datasets/processed/phase18_canonical_data_agent_sft_20260702_155000_phase18/train_phase18_canonical_sft.jsonl \
    --output-dir "$DATA_DIR" \
    > "$EVAL_ROOT/logs/prepare_phase19_sql_grounding_data.log" 2>&1

  log "starting Phase19 SQL grounding SFT"
  mkdir -p "$TRAIN_DIR"
  torchrun --standalone --nproc-per-node=16 \
    scripts/remote/train_lora_sft.py \
    --model "$BASE_MODEL" \
    --train-data "$DATA_DIR/train_phase19_sql_grounding_sft.jsonl" \
    --output-dir "$TRAIN_DIR" \
    --steps "$STEPS" \
    --seq-len 4096 \
    --micro-batch-size 1 \
    --grad-accum-steps 2 \
    --learning-rate "$LR" \
    --warmup-steps "$WARMUP_STEPS" \
    --max-grad-norm 0.5 \
    --lora-r 32 \
    --lora-alpha 64 \
    --lora-dropout 0.05 \
    --seed 20260702 \
    --log-every 5 \
    --save-every 350 \
    --swanlab-project agentic-rl-sql-tool \
    --swanlab-run-name "$RUN_NAME" \
    --swanlab-mode local \
    --swanlab-tags phase19,sql-grounding,sft,7b \
    > "$TRAIN_DIR/train.log" 2>&1

  echo "$TRAIN_DIR/adapter" > "$EVAL_ROOT/final_adapter_path.txt"
  log "Phase19 training completed: $TRAIN_DIR/adapter"
  merge_adapter "$TRAIN_DIR/adapter"

  run_eval phase19_wikisql_v2 \
    /opt/ac2/bin/python scripts/remote/evaluate_wikisql_v2.py \
      --model "$MERGED" \
      --model-label phase19_sql_grounding_sft_wikisql_v2 \
      --dataset datasets/processed/phase17_sql_error_sft_20260702_123712_phase17b/wikisql_eval_256_normalized.jsonl \
      --database datasets/processed/phase17_sql_error_sft_20260702_123712_phase17b/wikisql_eval_256_normalized.sqlite \
      --output-dir "$EVAL_ROOT/phase19_wikisql_v2" \
      --gpu-memory-utilization 0.20 \
      --batch-size 16

  log "Phase19 SQL grounding SFT and post-eval completed"
  start_keepalive_if_idle
}

main "$@"
