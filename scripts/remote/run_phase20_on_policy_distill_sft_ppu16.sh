#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yans2@xiaopeng.com/agentic_rl_pipeline"
cd "$ROOT"

STAMP="${1:-$(date +%Y%m%d_%H%M%S)_phase20_opd}"
RUN_NAME="phase20_on_policy_distill_sft_${STAMP}"
RUN_ROOT="$ROOT/runs/$RUN_NAME"
EVAL_ROOT="$ROOT/evals/$RUN_NAME"
DATA_DIR="$ROOT/datasets/processed/$RUN_NAME"
BASE_MODEL="${BASE_MODEL:-$ROOT/evals/phase16c_sql_execution_grpo_20260702_233006_auto2_sql_grpo/merged/phase16c_sql_execution_grpo_merged}"
MERGED="$EVAL_ROOT/merged/phase20_on_policy_distill_sft_merged"
TRAIN_DIR="$RUN_ROOT/${RUN_NAME}_lora"
SQL_SOURCE="${SQL_SOURCE:-$ROOT/datasets/processed/phase16_followup_assets_20260701/phase16c_grpo_train.jsonl}"
TOOL_REPLAY="${TOOL_REPLAY:-$ROOT/datasets/processed/phase16_followup_assets_20260701/data_agent_tool_action_probe.jsonl}"
STEPS="${STEPS:-300}"
LR="${LR:-5e-8}"
WARMUP_STEPS="${WARMUP_STEPS:-30}"
ROLLOUT_LIMIT="${ROLLOUT_LIMIT:-1200}"
ROLLOUT_N="${ROLLOUT_N:-4}"

mkdir -p "$RUN_ROOT" "$EVAL_ROOT/logs" "$EVAL_ROOT/merged" "$DATA_DIR" logs

export LD_LIBRARY_PATH=/usr/local/PPU_SDK/CUDA_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/CUDA_SDK/lib64:/usr/local/PPU_SDK/targets/x86_64-linux/lib:/usr/local/PPU_SDK/lib:${LD_LIBRARY_PATH:-}
export PPU_SDK=/usr/local/PPU_SDK
export PPU_HOME=/usr/local/PPU_SDK
export NCCL_SOCKET_IFNAME=hpn0
export NCCL_IB_HCA=
export PATH=/opt/ac2/bin:${PATH}
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
export NPROC_PER_NODE="${NPROC_PER_NODE:-16}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$EVAL_ROOT/phase20_queue.log"
}

stop_keepalive() {
  pkill -f './run_gpu_16.sh' || true
  for f in run_gpu_16.*.pid; do
    [[ -s "$f" ]] || continue
    pid="$(cat "$f")"
    pkill -P "$pid" || true
    kill "$pid" || true
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
  local log_file="$EVAL_ROOT/logs/merge_phase20.log"
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
    nohup ./run_gpu_16.sh > run_gpu_16.phase20_after.log 2>&1 < /dev/null &
    echo $! > run_gpu_16.phase20_after.pid
    log "started run_gpu_16 keepalive pid $(cat run_gpu_16.phase20_after.pid)"
  fi
}

main() {
  for path in "$BASE_MODEL" "$SQL_SOURCE" "$TOOL_REPLAY" scripts/remote/prepare_phase20_on_policy_distill_data.py scripts/remote/train_lora_sft.py; do
    if [[ ! -e "$path" ]]; then
      log "missing required path: $path"
      exit 2
    fi
  done

  stop_keepalive
  echo "$$" > "$RUN_ROOT/queue.pid"

  cat > "$EVAL_ROOT/phase20_manifest.json" <<JSON
{
  "stamp": "$STAMP",
  "run_name": "$RUN_NAME",
  "base_model": "$BASE_MODEL",
  "method": "execution-verified on-policy distillation SFT",
  "sql_source": "$SQL_SOURCE",
  "tool_replay": "$TOOL_REPLAY",
  "steps": $STEPS,
  "learning_rate": $LR,
  "rollout_limit": $ROLLOUT_LIMIT,
  "rollout_n": $ROLLOUT_N,
  "teacher": "none; self-rollout + SQLite/gold-result verifier"
}
JSON

  if [[ -s "$DATA_DIR/train_phase20_on_policy_distill.jsonl" && -s "$DATA_DIR/manifest.json" ]]; then
    log "reusing existing Phase20 OPD data: $DATA_DIR"
  else
    log "building Phase20 on-policy distillation data"
    /opt/ac2/bin/python scripts/remote/prepare_phase20_on_policy_distill_data.py \
      --model "$BASE_MODEL" \
      --sql-source "$SQL_SOURCE" \
      --tool-replay "$TOOL_REPLAY" \
      --output-dir "$DATA_DIR" \
      --limit "$ROLLOUT_LIMIT" \
      --num-samples "$ROLLOUT_N" \
      --batch-size 16 \
      --gpu-memory-utilization 0.20 \
      > "$EVAL_ROOT/logs/prepare_phase20_on_policy_distill_data.log" 2>&1
  fi

  log "starting Phase20 OPD SFT"
  mkdir -p "$TRAIN_DIR"
  torchrun --standalone --nproc-per-node="$NPROC_PER_NODE" \
    scripts/remote/train_lora_sft.py \
    --model "$BASE_MODEL" \
    --train-data "$DATA_DIR/train_phase20_on_policy_distill.jsonl" \
    --output-dir "$TRAIN_DIR" \
    --steps "$STEPS" \
    --seq-len 4096 \
    --micro-batch-size 1 \
    --grad-accum-steps 2 \
    --learning-rate "$LR" \
    --warmup-steps "$WARMUP_STEPS" \
    --max-grad-norm 0.5 \
    --lora-r 16 \
    --lora-alpha 32 \
    --lora-dropout 0.05 \
    --seed 20260703 \
    --log-every 5 \
    --save-every 150 \
    --swanlab-project agentic-rl-sql-tool \
    --swanlab-run-name "$RUN_NAME" \
    --swanlab-mode local \
    --swanlab-tags phase20,on-policy-distill,sql,tool-replay \
    > "$TRAIN_DIR/train.log" 2>&1

  echo "$TRAIN_DIR/adapter" > "$EVAL_ROOT/final_adapter_path.txt"
  log "Phase20 training completed: $TRAIN_DIR/adapter"
  merge_adapter "$TRAIN_DIR/adapter"

  run_eval phase20_wikisql_v2 \
    /opt/ac2/bin/python scripts/remote/evaluate_wikisql_v2.py \
      --model "$MERGED" \
      --model-label phase20_on_policy_distill_wikisql_v2 \
      --dataset datasets/processed/phase17_sql_error_sft_20260702_123712_phase17b/wikisql_eval_256_normalized.jsonl \
      --database datasets/processed/phase17_sql_error_sft_20260702_123712_phase17b/wikisql_eval_256_normalized.sqlite \
      --output-dir "$EVAL_ROOT/wikisql_v2" \
      --gpu-memory-utilization 0.20 \
      --batch-size 16

  run_eval phase20_wikisql_phase16_probe \
    /opt/ac2/bin/python scripts/remote/evaluate_wikisql.py \
      --model "$MERGED" \
      --model-label phase20_on_policy_distill \
      --dataset datasets/processed/phase16_followup_assets_20260701/wikisql_eval_256.jsonl \
      --database datasets/processed/phase16_followup_assets_20260701/wikisql_eval_256.sqlite \
      --output-dir "$EVAL_ROOT/wikisql"

  run_eval phase20_sql_repair_execution \
    /opt/ac2/bin/python scripts/remote/evaluate_sql_repair_execution.py \
      --model "$MERGED" \
      --model-label phase20_on_policy_distill \
      --dataset datasets/processed/phase16_followup_assets_20260701/sql_repair_execution_eval/sql_repair_execution_eval_128.jsonl \
      --database datasets/processed/phase16_followup_assets_20260701/wikisql_eval_256.sqlite \
      --output-dir "$EVAL_ROOT/sql_repair_execution"

  run_eval phase20_tool_guard \
    /opt/ac2/bin/python scripts/remote/evaluate_xlam_tool_calls.py \
      --model "$MERGED" \
      --dataset "$TOOL_REPLAY" \
      --output-dir "$EVAL_ROOT/tool_action_guard" \
      --model-label phase20_on_policy_distill_tool_guard \
      --max-model-len 4096 \
      --gpu-memory-utilization 0.20 \
      --batch-size 16

  run_eval phase20_general \
    /opt/ac2/bin/python scripts/remote/evaluate_general_regression.py \
      --model "$MERGED" \
      --model-label phase20_on_policy_distill \
      --gsm8k datasets/eval_suite/huggingface/openai__gsm8k/main/test-00000-of-00001.parquet \
      --mmlu-pro datasets/eval_suite/huggingface/TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet \
      --output-dir "$EVAL_ROOT/general" \
      --samples-per-benchmark 256

  log "Phase20 OPD SFT and post-eval completed"
  start_keepalive_if_idle
}

main "$@"
