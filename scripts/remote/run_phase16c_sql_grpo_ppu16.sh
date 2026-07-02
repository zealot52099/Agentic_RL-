#!/usr/bin/env bash
set -euo pipefail

ROOT="/workspace/yans2@xiaopeng.com/agentic_rl_pipeline"
cd "$ROOT"

STAMP="${1:-$(date +%Y%m%d_%H%M%S)_phase16c}"
RUN_NAME="phase16c_sql_execution_grpo_${STAMP}"
RUN_ROOT="$ROOT/runs/$RUN_NAME"
EVAL_ROOT="$ROOT/evals/$RUN_NAME"
LOG_ROOT="$ROOT/logs"
FOLLOW="$ROOT/datasets/processed/phase16_followup_assets_20260701"
BASE_MODEL="${BASE_MODEL:-$ROOT/evals/phase16a_sql_repair_sft_20260701_1138_phase16a/merged/phase16a_sql_repair_sft_merged}"
DATA="$FOLLOW/phase16c_grpo_train.jsonl"
PLUGIN="$ROOT/scripts/remote/swift_wikisql_reward_plugin.py"
MERGED="$EVAL_ROOT/merged/phase16c_sql_execution_grpo_merged"
MAX_STEPS="${MAX_STEPS:-1200}"
LR="${LR:-1.5e-7}"
REPORT_TO="${REPORT_TO:-swanlab}"
SWANLAB_MODE="${SWANLAB_MODE:-local}"

mkdir -p "$RUN_ROOT" "$EVAL_ROOT/logs" "$EVAL_ROOT/merged" "$LOG_ROOT"

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
  echo "[$(date '+%F %T')] $*" | tee -a "$EVAL_ROOT/phase16c_queue.log"
}

stop_keepalive() {
  pkill -f './run_gpu_16.sh' || true
  if [[ -s run_gpu_16.phase16_post_eval.pid ]]; then
    local pid
    pid="$(cat run_gpu_16.phase16_post_eval.pid)"
    pkill -P "$pid" || true
    kill "$pid" || true
  fi
  ps -eo pid,cmd | awk '/python -$/ {print $1}' | xargs -r kill || true
}

merge_adapter() {
  local checkpoint="$1"
  local log_file="$EVAL_ROOT/logs/merge_phase16c.log"
  if [[ -s "$MERGED/model.safetensors.index.json" ]]; then
    log "merged model already exists: $MERGED"
    return
  fi
  log "merging adapter: $checkpoint"
  /opt/ac2/bin/python - <<PY > "$log_file" 2>&1
import json
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base = "$BASE_MODEL"
adapter = "$checkpoint"
out = Path("$MERGED")
print("loading tokenizer", flush=True)
tok = AutoTokenizer.from_pretrained(base, local_files_only=True)
print("loading base", base, flush=True)
model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16, device_map="cpu")
print("loading adapter", adapter, flush=True)
model = PeftModel.from_pretrained(model, adapter, local_files_only=True)
print("merging", flush=True)
model = model.merge_and_unload()
out.mkdir(parents=True, exist_ok=True)
print("saving", out, flush=True)
model.save_pretrained(out, safe_serialization=True, max_shard_size="2GB")
tok.save_pretrained(out)
(out / "merge_metadata.json").write_text(json.dumps({"base": base, "adapter": adapter}, indent=2) + "\\n", encoding="utf-8")
print("done", flush=True)
PY
  echo "$MERGED" > "$EVAL_ROOT/final_merged_model_path.txt"
  log "merge complete: $MERGED"
}

run_eval() {
  local name="$1"
  shift
  log "starting $name"
  "$@" > "$EVAL_ROOT/logs/${name}.log" 2>&1
  log "completed $name"
}

start_keepalive_if_idle() {
  if ! ps -eo cmd | grep -E 'train_lora|torchrun|swift rlhf|evaluate_|vllm|run_gpu_16' | grep -v grep >/dev/null; then
    nohup ./run_gpu_16.sh > run_gpu_16.phase16c_after.log 2>&1 < /dev/null &
    echo $! > run_gpu_16.phase16c_after.pid
    log "started run_gpu_16 keepalive pid $(cat run_gpu_16.phase16c_after.pid)"
  fi
}

main() {
  for path in "$BASE_MODEL" "$DATA" "$PLUGIN" "$FOLLOW/wikisql_eval_256.jsonl" "$FOLLOW/wikisql_eval_256.sqlite"; do
    if [[ ! -e "$path" ]]; then
      log "missing required path: $path"
      exit 2
    fi
  done

  stop_keepalive

  cat > "$EVAL_ROOT/phase16c_manifest.json" <<JSON
{
  "stamp": "$STAMP",
  "run_name": "$RUN_NAME",
  "base_model": "$BASE_MODEL",
  "data": "$DATA",
  "method": "ms-swift GRPO, synchronized 16 PPU, LoRA",
  "max_steps": $MAX_STEPS,
  "learning_rate": $LR,
  "reward": "wikisql_exec",
  "report_to": "$REPORT_TO",
  "swanlab_mode": "$SWANLAB_MODE"
}
JSON

  log "starting Phase16c SQL execution GRPO"
  SWANLAB_ARGS=()
  if [[ " $REPORT_TO " == *" swanlab "* ]]; then
    SWANLAB_ARGS=(
      --swanlab_project agentic-rl-sql-tool
      --swanlab_exp_name "$RUN_NAME"
      --swanlab_mode "$SWANLAB_MODE"
    )
  fi

  /opt/ac2/bin/swift rlhf \
    --rlhf_type grpo \
    --model "$BASE_MODEL" \
    --model_type qwen2 \
    --template qwen2_5 \
    --external_plugins "$PLUGIN" \
    --reward_funcs wikisql_exec \
    --use_vllm true \
    --vllm_mode colocate \
    --vllm_gpu_memory_utilization 0.22 \
    --vllm_tensor_parallel_size 1 \
    --vllm_max_model_len 4096 \
    --vllm_disable_custom_all_reduce true \
    --vllm_enforce_eager true \
    --tuner_type lora \
    --lora_rank 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --torch_dtype bfloat16 \
    --dataset "$DATA" \
    --split_dataset_ratio 0 \
    --max_length 1536 \
    --max_completion_length 128 \
    --num_train_epochs 1 \
    --max_steps "$MAX_STEPS" \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --learning_rate "$LR" \
    --warmup_ratio 0.03 \
    --save_steps 300 \
    --save_total_limit 4 \
    --logging_steps 1 \
    --dataloader_num_workers 2 \
    --dataset_num_proc 4 \
    --num_generations 4 \
    --temperature 0.9 \
    --top_p 0.95 \
    --deepspeed zero1 \
    --log_completions true \
    --num_iterations 1 \
    --report_to $REPORT_TO \
    "${SWANLAB_ARGS[@]}" \
    --beta 0.03 \
    --loss_type grpo \
    --advantage_estimator grpo \
    --output_dir "$RUN_ROOT" > "$EVAL_ROOT/logs/train_phase16c.log" 2>&1

  log "Phase16c training completed"
  local checkpoint
  checkpoint="$(find "$RUN_ROOT" -path "*/checkpoint-$MAX_STEPS" -type d | sort | tail -n 1)"
  if [[ -z "$checkpoint" ]]; then
    checkpoint="$(find "$RUN_ROOT" -path "*/checkpoint-*" -type d | sort -V | tail -n 1)"
  fi
  if [[ -z "$checkpoint" ]]; then
    log "no checkpoint found under $RUN_ROOT"
    exit 3
  fi
  echo "$checkpoint" > "$EVAL_ROOT/final_adapter_checkpoint.txt"
  merge_adapter "$checkpoint"

  run_eval wikisql_phase16c \
    /opt/ac2/bin/python scripts/remote/evaluate_wikisql.py \
      --model "$MERGED" \
      --model-label phase16c_sql_execution_grpo \
      --dataset "$FOLLOW/wikisql_eval_256.jsonl" \
      --database "$FOLLOW/wikisql_eval_256.sqlite" \
      --output-dir "$EVAL_ROOT/wikisql"

  if [[ ! -s "$FOLLOW/sql_repair_execution_eval/sql_repair_execution_eval_128.jsonl" ]]; then
    mkdir -p "$FOLLOW/sql_repair_execution_eval"
    run_eval prepare_sql_repair_execution_phase16c \
      /opt/ac2/bin/python scripts/remote/prepare_sql_repair_execution_eval.py \
        --wikisql-eval "$FOLLOW/wikisql_eval_256.jsonl" \
        --database "$FOLLOW/wikisql_eval_256.sqlite" \
        --output "$FOLLOW/sql_repair_execution_eval/sql_repair_execution_eval_128.jsonl" \
        --manifest "$FOLLOW/sql_repair_execution_eval/manifest.json" \
        --limit 128
  fi

  run_eval sql_repair_execution_phase16c \
    /opt/ac2/bin/python scripts/remote/evaluate_sql_repair_execution.py \
      --model "$MERGED" \
      --model-label phase16c_sql_execution_grpo \
      --dataset "$FOLLOW/sql_repair_execution_eval/sql_repair_execution_eval_128.jsonl" \
      --database "$FOLLOW/wikisql_eval_256.sqlite" \
      --output-dir "$EVAL_ROOT/sql_repair_execution"

  run_eval multiturn_phase16c \
    /opt/ac2/bin/python scripts/remote/evaluate_data_agent_multiturn.py \
      --model "$MERGED" \
      --model-label phase16c_sql_execution_grpo \
      --traces "$FOLLOW/data_agent_multiturn_eval_500.jsonl" \
      --database "$FOLLOW/data_agent_eval.sqlite" \
      --output-dir "$EVAL_ROOT/data_agent_multiturn" \
      --limit 300

  run_eval general_phase16c \
    /opt/ac2/bin/python scripts/remote/evaluate_general_regression.py \
      --model "$MERGED" \
      --model-label phase16c_sql_execution_grpo \
      --gsm8k datasets/eval_suite/huggingface/openai__gsm8k/main/test-00000-of-00001.parquet \
      --mmlu-pro datasets/eval_suite/huggingface/TIGER-Lab__MMLU-Pro/data/test-00000-of-00001.parquet \
      --output-dir "$EVAL_ROOT/general" \
      --samples-per-benchmark 256

  log "Phase16c training and post-eval completed"
  start_keepalive_if_idle
}

main "$@"
