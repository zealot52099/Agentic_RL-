#!/usr/bin/env python3
"""Build execution-verified on-policy distillation data for SQL/Data Agent.

This script does not require a strong external teacher. It samples completions
from the current student policy, executes candidate SQL in a per-example SQLite
sandbox, keeps successful on-policy SQL, and turns failures into verified repair
examples using the gold SQL/result already present in the training data.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def extract_sql(text: str | None) -> str | None:
    if not text:
        return None
    value = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", value, re.I | re.S)
    if fenced:
        value = fenced.group(1).strip()
    match = re.search(r"\bSELECT\b.*", value, re.I | re.S)
    if not match:
        return None
    sql = match.group(0).strip()
    sql = sql.split("\n\n", 1)[0].strip()
    if ";" in sql:
        sql = sql.split(";", 1)[0].strip()
    return sql


def normalize_result(rows: list[tuple[Any, ...]]) -> list[tuple[str, ...]]:
    normalized = []
    for row in rows:
        normalized.append(tuple("" if value is None else str(value).strip().lower() for value in row))
    return sorted(normalized)


def solution(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("solution")
    if isinstance(value, str):
        return json.loads(value)
    if isinstance(value, dict):
        return value
    raise ValueError(f"missing solution in {row.get('id')}")


def make_db(sol: dict[str, Any]) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    table = sol["table_name"]
    header = sol["header"]
    columns = [f'"col{i}" TEXT' for i in range(len(header))]
    conn.execute(f'CREATE TABLE "{table}" ({", ".join(columns)})')
    placeholders = ", ".join(["?"] * len(header))
    quoted_cols = ", ".join(f'"col{i}"' for i in range(len(header)))
    conn.executemany(
        f'INSERT INTO "{table}" ({quoted_cols}) VALUES ({placeholders})',
        [[str(value) if value is not None else None for value in row] for row in sol["rows"]],
    )
    return conn


def execute_sql(sol: dict[str, Any], sql: str | None) -> tuple[bool, str | None, list[tuple[Any, ...]] | None]:
    if sql is None:
        return False, "no_sql", None
    if not re.match(r"^\s*SELECT\b", sql, re.I):
        return False, "not_select", None
    try:
        conn = make_db(sol)
        result = conn.execute(sql).fetchall()
        gold = [tuple(row) for row in sol["gold_result"]]
        return normalize_result(result) == normalize_result(gold), None, result
    except sqlite3.Error as exc:
        return False, str(exc), None
    finally:
        try:
            conn.close()
        except Exception:
            pass


def repair_prompt(query: str, bad_sql: str | None, error: str | None, result: list[tuple[Any, ...]] | None) -> str:
    feedback = error if error else f"wrong_result={json.dumps(result, ensure_ascii=False)}"
    return (
        query.rstrip()
        + "\n\nPrevious SQL from the current policy:\n"
        + (bad_sql or "<no valid SQL>")
        + "\n\nExecution feedback:\n"
        + feedback
        + "\n\nReturn one corrected SQLite SELECT query only.\nSQL:"
    )


def expected_to_completion(row: dict[str, Any]) -> str | None:
    expected = row.get("expected")
    if isinstance(expected, dict):
        return json.dumps(expected, ensure_ascii=False, sort_keys=True)
    if isinstance(row.get("completion"), str):
        return row["completion"]
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--sql-source", type=Path, required=True)
    parser.add_argument("--tool-replay", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1200)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.22)
    parser.add_argument("--seed", type=int, default=20260703)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    source = load_jsonl(args.sql_source)
    rng.shuffle(source)
    source = source[: args.limit]
    prompts = [row["query"] for row in source]

    started = time.time()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
        enforce_eager=True,
    )
    sampling = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        n=args.num_samples,
        max_tokens=128,
    )
    outputs = []
    for start in range(0, len(prompts), args.batch_size):
        outputs.extend(llm.generate(prompts[start : start + args.batch_size], sampling, use_tqdm=True))

    train_rows: list[dict[str, Any]] = []
    rollout_rows: list[dict[str, Any]] = []
    counters = Counter()
    for row, output in zip(source, outputs, strict=True):
        sol = solution(row)
        best_success = None
        first_failure = None
        seen_sql = set()
        for item in output.outputs:
            raw = item.text
            sql = extract_sql(raw)
            if sql in seen_sql:
                continue
            seen_sql.add(sql)
            ok, error, result = execute_sql(sol, sql)
            rollout_rows.append(
                {
                    "id": row.get("id"),
                    "question": row.get("question"),
                    "raw_response": raw,
                    "sql": sql,
                    "execution_ok": ok,
                    "error": error,
                    "result": result,
                    "gold_sql": sol["gold_sql"],
                    "gold_result": sol["gold_result"],
                }
            )
            if ok and best_success is None:
                best_success = sql
            if not ok and first_failure is None:
                first_failure = (sql, error, result)
        if best_success:
            counters["on_policy_success"] += 1
            train_rows.append(
                {
                    "id": f"{row.get('id')}-opd-success",
                    "prompt": row["query"],
                    "completion": " " + best_success,
                    "loss_weight": 1.0,
                    "source": "phase20_on_policy_success_sql",
                }
            )
        else:
            counters["on_policy_failed_repaired"] += 1
            bad_sql, error, result = first_failure if first_failure else (None, "no_generation", None)
            train_rows.append(
                {
                    "id": f"{row.get('id')}-opd-repair",
                    "prompt": repair_prompt(row["query"], bad_sql, error, result),
                    "completion": " " + sol["gold_sql"],
                    "loss_weight": 0.9,
                    "source": "phase20_on_policy_verified_repair",
                }
            )
            train_rows.append(
                {
                    "id": f"{row.get('id')}-opd-direct-gold",
                    "prompt": row["query"],
                    "completion": " " + sol["gold_sql"],
                    "loss_weight": 0.35,
                    "source": "phase20_gold_replay_for_failed_rollout",
                }
            )

    tool_rows = []
    for row in load_jsonl(args.tool_replay):
        completion = expected_to_completion(row)
        if not completion:
            continue
        tool_rows.append(
            {
                "id": f"{row.get('id', len(tool_rows))}-tool-replay",
                "prompt": row["prompt"],
                "completion": completion,
                "loss_weight": 0.25,
                "source": "phase20_tool_action_replay",
            }
        )
    rng.shuffle(tool_rows)
    train_rows.extend(tool_rows[: max(256, len(train_rows) // 5)])
    rng.shuffle(train_rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train_phase20_on_policy_distill.jsonl", train_rows)
    write_jsonl(args.output_dir / "rollouts_phase20_on_policy_distill.jsonl", rollout_rows)
    manifest = {
        "model": args.model,
        "sql_source": str(args.sql_source),
        "tool_replay": str(args.tool_replay),
        "samples": len(source),
        "num_samples_per_prompt": args.num_samples,
        "train_rows": len(train_rows),
        "rollout_rows": len(rollout_rows),
        "counts": dict(counters),
        "elapsed_seconds": time.time() - started,
        "note": "No unverified teacher text is used; success and repair targets are execution/gold-result verified.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
