#!/usr/bin/env python3
"""Conservative SQL grounding auto-loop controller.

The loop implements:

  wait for current run -> evaluate result -> analyze error categories ->
  write a next plan -> optionally launch the next training/eval run.

It is intentionally rule-based and conservative. It only launches a new round
when the last round has completed, produced a WikiSQL v2 metrics file, improved
over the previous baseline by at least ``min_gain``, and has not exceeded
``max_rounds``. If the signal stalls or regresses, it starts ``run_gpu_16.sh``
as a resource keepalive and exits.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


ROOT = Path("/workspace/yans2@xiaopeng.com/agentic_rl_pipeline")
PHASE19_PREFIX = "phase19_sql_grounding_sft_"


@dataclass
class RoundState:
    round_index: int
    run_name: str
    status: str
    model_path: str | None = None
    metrics_path: str | None = None
    execution_accuracy: float | None = None
    execution_rate: float | None = None
    extraction_rate: float | None = None
    normalized_sql_exact: float | None = None
    dominant_error: str | None = None
    next_action: str | None = None
    reason: str | None = None


def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def run(cmd: list[str], *, cwd: Path = ROOT, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


def shell(cmd: str) -> subprocess.CompletedProcess[str]:
    return run(["bash", "-lc", cmd])


def log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"[{now()}] {message}\n")


def active_project_tasks() -> list[str]:
    result = shell(
        "ps -eo pid,etime,cmd | grep -E 'train_lora_sft.py|evaluate_wikisql_v2.py|torchrun|swift rlhf|vllm' | grep -v grep || true"
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def start_keepalive(log_path: Path) -> None:
    tasks = active_project_tasks()
    if tasks:
        log(log_path, "not starting keepalive because project tasks are active")
        return
    if shell("ps -eo cmd | grep -E './run_gpu_16.sh' | grep -v grep >/dev/null").returncode == 0:
        log(log_path, "run_gpu_16 keepalive already running")
        return
    subprocess.Popen(
        ["bash", "-lc", "nohup ./run_gpu_16.sh > run_gpu_16.auto_sql_loop.log 2>&1 < /dev/null & echo $! > run_gpu_16.auto_sql_loop.pid"],
        cwd=ROOT,
    )
    log(log_path, "started run_gpu_16 keepalive")


def eval_root(run_name: str) -> Path:
    return ROOT / "evals" / run_name


def run_root(run_name: str) -> Path:
    return ROOT / "runs" / run_name


def find_wikisql_metrics(run_name: str) -> Path | None:
    root = eval_root(run_name)
    candidates = sorted(root.glob("**/*wikisql*v2*metrics.json"))
    if not candidates:
        candidates = sorted(root.glob("**/*metrics.json"))
    return candidates[-1] if candidates else None


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def final_model_path(run_name: str) -> Path | None:
    pointer = eval_root(run_name) / "final_merged_model_path.txt"
    if pointer.exists():
        value = pointer.read_text(encoding="utf-8").strip()
        if value:
            return Path(value)
    merged = eval_root(run_name) / "merged" / "phase19_sql_grounding_sft_merged"
    if (merged / "config.json").exists() or (merged / "model.safetensors.index.json").exists():
        return merged
    return None


def run_completed(run_name: str) -> bool:
    queue_logs = list(eval_root(run_name).glob("*queue.log"))
    if queue_logs:
        text = "\n".join(path.read_text(encoding="utf-8", errors="replace")[-4000:] for path in queue_logs)
        if "post-eval completed" in text or "and post-eval completed" in text:
            return True
    return bool(find_wikisql_metrics(run_name))


def state_from_metrics(round_index: int, run_name: str, metrics_path: Path) -> RoundState:
    metrics = read_json(metrics_path)
    categories = metrics.get("error_categories") or {}
    dominant = None
    if categories:
        non_correct = {k: v for k, v in categories.items() if k != "correct_execution"}
        if non_correct:
            dominant = max(non_correct.items(), key=lambda item: item[1])[0]
    model = final_model_path(run_name)
    return RoundState(
        round_index=round_index,
        run_name=run_name,
        status="completed",
        model_path=str(model) if model else None,
        metrics_path=str(metrics_path),
        execution_accuracy=metrics.get("execution_accuracy"),
        execution_rate=metrics.get("execution_rate"),
        extraction_rate=metrics.get("sql_extraction_rate"),
        normalized_sql_exact=metrics.get("normalized_sql_exact"),
        dominant_error=dominant,
    )


def launch_next_round(
    *,
    base_model: str,
    round_index: int,
    steps: int,
    lr: float,
    log_path: Path,
) -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S") + f"_auto{round_index}"
    env = os.environ.copy()
    env["BASE_MODEL"] = base_model
    env["STEPS"] = str(steps)
    env["LR"] = f"{lr:.2e}"
    cmd = ["bash", "scripts/remote/run_phase19_sql_grounding_sft_ppu16.sh", stamp]
    stdout = ROOT / "logs" / f"auto_sql_loop_launch_{stamp}.log"
    stdout.parent.mkdir(parents=True, exist_ok=True)
    with stdout.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT)
    run_name = PHASE19_PREFIX + stamp
    log(log_path, f"launched next round {run_name} pid={proc.pid} base={base_model} steps={steps} lr={lr:.2e}")
    return run_name


def write_state(state_path: Path, data: dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial-run", required=True)
    parser.add_argument("--baseline-accuracy", type=float, default=0.52734375)
    parser.add_argument("--target-accuracy", type=float, default=0.62)
    parser.add_argument("--min-gain", type=float, default=0.005)
    parser.add_argument("--max-rounds", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--state-dir", type=Path, default=ROOT / "runs" / "auto_sql_grounding_loop")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.state_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.state_dir / "auto_loop.log"
    state_path = args.state_dir / "state.json"
    plan_path = args.state_dir / "latest_plan.json"
    history: list[dict[str, Any]] = []

    current_run = args.initial_run
    previous_accuracy = args.baseline_accuracy
    log(log_path, f"auto-loop started initial_run={current_run} baseline={previous_accuracy:.4f}")

    for round_index in range(1, args.max_rounds + 1):
        while not run_completed(current_run):
            tasks = active_project_tasks()
            write_state(
                state_path,
                {
                    "status": "waiting",
                    "current_run": current_run,
                    "round_index": round_index,
                    "active_tasks": tasks[:20],
                    "history": history,
                    "updated_at": now(),
                },
            )
            log(log_path, f"waiting for {current_run}; active_tasks={len(tasks)}")
            time.sleep(args.poll_seconds)

        metrics_path = find_wikisql_metrics(current_run)
        if not metrics_path:
            state = RoundState(round_index, current_run, "failed", reason="missing WikiSQL metrics")
            state.next_action = "stop_keepalive"
            history.append(asdict(state))
            write_state(state_path, {"status": "stopped", "history": history, "updated_at": now()})
            start_keepalive(log_path)
            return

        state = state_from_metrics(round_index, current_run, metrics_path)
        acc = float(state.execution_accuracy or 0.0)
        gain = acc - previous_accuracy

        plan = {
            "round": round_index,
            "current_run": current_run,
            "metrics_path": str(metrics_path),
            "execution_accuracy": acc,
            "previous_accuracy": previous_accuracy,
            "gain": gain,
            "dominant_error": state.dominant_error,
            "decision": None,
            "next": None,
        }

        if acc >= args.target_accuracy:
            state.next_action = "stop_target_reached"
            state.reason = f"target reached: {acc:.4f} >= {args.target_accuracy:.4f}"
            plan["decision"] = state.next_action
            history.append(asdict(state))
            write_state(state_path, {"status": "target_reached", "history": history, "updated_at": now()})
            write_state(plan_path, plan)
            start_keepalive(log_path)
            return

        if gain < args.min_gain:
            state.next_action = "stop_low_gain"
            state.reason = f"gain {gain:.4f} < min_gain {args.min_gain:.4f}"
            plan["decision"] = state.next_action
            history.append(asdict(state))
            write_state(state_path, {"status": "stopped_low_gain", "history": history, "updated_at": now()})
            write_state(plan_path, plan)
            log(log_path, state.reason)
            start_keepalive(log_path)
            return

        model = state.model_path
        if not model:
            state.next_action = "stop_missing_model"
            state.reason = "completed metrics found but merged model path is missing"
            plan["decision"] = state.next_action
            history.append(asdict(state))
            write_state(state_path, {"status": "stopped_missing_model", "history": history, "updated_at": now()})
            write_state(plan_path, plan)
            start_keepalive(log_path)
            return

        # Conservative continuation schedule: decay LR and steps each round.
        next_steps = max(300, int(500 * (0.8 ** (round_index - 1))))
        next_lr = max(5e-8, 1.0e-7 * (0.8 ** (round_index - 1)))
        if state.dominant_error == "wrong_missing_aggregation":
            next_steps += 100
        elif state.dominant_error == "wrong_where_or_value_or_column":
            next_steps += 50

        plan["decision"] = "continue_sql_grounding_sft"
        plan["next"] = {
            "base_model": model,
            "steps": next_steps,
            "learning_rate": next_lr,
            "reason": "metric improved and target not reached; continue with lower LR SQL grounding SFT",
        }
        state.next_action = "continue_sql_grounding_sft"
        state.reason = str(plan["next"]["reason"])
        history.append(asdict(state))
        write_state(plan_path, plan)
        write_state(state_path, {"status": "launching_next_round", "history": history, "latest_plan": plan, "updated_at": now()})

        if args.dry_run:
            log(log_path, "dry-run enabled; not launching next round")
            return

        previous_accuracy = acc
        current_run = launch_next_round(
            base_model=model,
            round_index=round_index + 1,
            steps=next_steps,
            lr=next_lr,
            log_path=log_path,
        )
        time.sleep(30)

    write_state(state_path, {"status": "max_rounds_reached", "history": history, "updated_at": now()})
    start_keepalive(log_path)


if __name__ == "__main__":
    main()
