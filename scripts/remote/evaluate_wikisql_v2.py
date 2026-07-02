#!/usr/bin/env python3
"""Evaluate WikiSQL-style text-to-SQL with stricter schema prompting.

This is an internal probe evaluator. Compared with evaluate_wikisql.py, the
prompt explicitly states that only physical `colN` identifiers may be used,
because the human-readable headers are descriptions rather than SQL columns.
"""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from collections import Counter
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


def prompt(row: dict[str, Any]) -> str:
    columns = ", ".join(
        f'`col{index}` means "{name}" ({kind})'
        for index, (name, kind) in enumerate(zip(row["header"], row["types"], strict=True))
    )
    samples = "\n".join(
        json.dumps({f"col{index}": value for index, value in enumerate(values)}, ensure_ascii=False)
        for values in row["sample_rows"]
    )
    return (
        "Write one SQLite SELECT query that answers the question.\n"
        "Important rules:\n"
        "- Use only physical SQL columns named `col0`, `col1`, ... exactly as listed.\n"
        "- Human-readable names in quotes are descriptions only; never use them as SQL identifiers.\n"
        "- Use the exact table name shown.\n"
        "- Return SQL only, without explanation or Markdown.\n\n"
        f'Table: "{row["sqlite_table"]}"\n'
        f"Columns: {columns}\n"
        f"Sample rows:\n{samples}\n\n"
        f"Question: {row['question']}\nSQL:"
    )


def extract_sql(text: str) -> str | None:
    value = text.strip()
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", value, re.I | re.S)
    if fenced:
        value = fenced.group(1).strip()
    match = re.search(r"\bSELECT\b.*", value, re.I | re.S)
    if not match:
        return None
    value = match.group(0).strip()
    value = value.split("```", 1)[0].strip()
    if ";" in value:
        value = value.split(";", 1)[0].strip() + ";"
    return value


def normalized_sql(value: str | None) -> str:
    if not value:
        return ""
    value = value.strip().rstrip(";")
    value = re.sub(r"\s+", " ", value)
    return value.casefold()


def normalize_result(rows: list[Any]) -> list[tuple[str, ...]]:
    normalized = []
    for row in rows:
        if not isinstance(row, (list, tuple)):
            row = [row]
        values = []
        for value in row:
            if value is None:
                values.append("<null>")
            elif isinstance(value, float):
                values.append(f"{value:.8g}")
            else:
                values.append(str(value))
        normalized.append(tuple(values))
    return sorted(normalized)


def safe_execute(connection: sqlite3.Connection, sql: str) -> list[tuple[Any, ...]]:
    if not re.match(r"^\s*SELECT\b", sql, re.I):
        raise sqlite3.OperationalError("Only SELECT is allowed")
    connection.set_progress_handler(
        lambda: 1 if time.monotonic() > safe_execute.deadline else 0,
        1000,
    )
    safe_execute.deadline = time.monotonic() + 2.0
    try:
        return connection.execute(sql).fetchall()
    finally:
        connection.set_progress_handler(None, 0)


safe_execute.deadline = 0.0


def error_category(row: dict[str, Any], sql: str | None, error: str | None, result: Any, execution_exact: bool) -> str:
    if execution_exact:
        return "correct_execution"
    if sql is None:
        return "no_sql_extracted"
    if error:
        if "no such column" in error and re.search(r"\b[A-Za-z][A-Za-z ]*[ -][A-Za-z][A-Za-z ]*\b", sql):
            return "exec_error_original_header_or_bad_identifier"
        if "no such column" in error:
            return "exec_error_no_such_column"
        if "syntax" in error.casefold():
            return "exec_error_syntax"
        return "exec_error_other"
    upper_sql = sql.upper()
    upper_gold = row["gold_sql"].upper()
    aggregations = ["COUNT(", "MIN(", "MAX(", "SUM(", "AVG("]
    pred_has_agg = any(item in upper_sql for item in aggregations)
    gold_has_agg = any(item in upper_gold for item in aggregations)
    if gold_has_agg and not pred_has_agg:
        return "wrong_missing_aggregation"
    if pred_has_agg and not gold_has_agg:
        return "wrong_extra_aggregation"
    if "WHERE" in upper_gold and "WHERE" not in upper_sql:
        return "wrong_missing_where"
    if "WHERE" in upper_gold and "WHERE" in upper_sql:
        return "wrong_where_or_value_or_column"
    return "wrong_other_executable"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompts = [prompt(row) for row in rows]
    started = time.time()
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        tensor_parallel_size=1,
        max_model_len=4096,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=False,
        enforce_eager=True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    outputs = []
    batch_size = max(1, args.batch_size)
    for start_index in range(0, len(prompts), batch_size):
        batch = prompts[start_index : start_index + batch_size]
        outputs.extend(
            llm.generate(
                batch,
                sampling_params,
                use_tqdm=True,
            )
        )

    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    predictions = []
    counters = Counter()
    categories = Counter()
    try:
        for row, output in zip(rows, outputs, strict=True):
            raw = output.outputs[0].text
            sql = extract_sql(raw)
            error = None
            result = None
            if sql is not None:
                counters["extracted"] += 1
                try:
                    result = safe_execute(connection, sql)
                    counters["executed"] += 1
                except sqlite3.Error as exc:
                    error = str(exc)
            execution_exact = result is not None and normalize_result(result) == normalize_result(row["gold_result"])
            sql_exact = sql is not None and normalized_sql(sql) == normalized_sql(row["gold_sql"])
            counters["execution_exact"] += int(execution_exact)
            counters["sql_exact"] += int(sql_exact)
            category = error_category(row, sql, error, result, execution_exact)
            categories[category] += 1
            predictions.append(
                {
                    "id": row["id"],
                    "question": row["question"],
                    "raw_response": raw,
                    "predicted_sql": sql,
                    "gold_sql": row["gold_sql"],
                    "execution_error": error,
                    "predicted_result": result,
                    "gold_result": row["gold_result"],
                    "execution_exact": execution_exact,
                    "sql_exact": sql_exact,
                    "error_category": category,
                }
            )
    finally:
        connection.close()

    total = len(rows)
    metrics = {
        "model": args.model,
        "model_label": args.model_label,
        "dataset": str(args.dataset),
        "database": str(args.database),
        "samples": total,
        "elapsed_seconds": time.time() - started,
        "sql_extraction_rate": counters["extracted"] / total if total else 0.0,
        "execution_rate": counters["executed"] / total if total else 0.0,
        "execution_accuracy": counters["execution_exact"] / total if total else 0.0,
        "normalized_sql_exact": counters["sql_exact"] / total if total else 0.0,
        "error_categories": dict(sorted(categories.items())),
        "scoring": "internal corrected WikiSQL probe; read-only SQLite execution; result rows compared order-insensitively",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.output_dir / args.model_label
    with Path(f"{prefix}_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for row in predictions:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    Path(f"{prefix}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
