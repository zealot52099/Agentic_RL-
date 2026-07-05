#!/usr/bin/env python3
"""Evaluate text-to-SQL generation with SQLite execution accuracy."""

from __future__ import annotations

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

from vllm import LLM, SamplingParams


def prompt(row: dict) -> str:
    columns = ", ".join(
        f'col{index} = "{name}" ({kind})'
        for index, (name, kind) in enumerate(
            zip(row["header"], row["types"], strict=True)
        )
    )
    samples = "\n".join(
        json.dumps(
            {
                f"col{index}": value
                for index, value in enumerate(values)
            },
            ensure_ascii=False,
        )
        for values in row["sample_rows"]
    )
    return (
        "Write one SQLite SELECT query that answers the question. "
        "Use only the table and columns shown. Return SQL only, without "
        "explanation or Markdown.\n\n"
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


def remap_headers(sql: str, headers: list) -> str:
    """Remap display header names to colN identifiers (quoted + unquoted)."""
    for i, h in enumerate(headers):
        col = f"col{i}"
        h_str = str(h).strip()
        if not h_str or h_str.startswith("col"):
            continue
        # Quoted: "DisplayName" -> "colN"
        sql = sql.replace(chr(34) + h_str + chr(34), chr(34) + col + chr(34))
        sql = sql.replace(chr(34) + h_str.lower() + chr(34), chr(34) + col + chr(34))
        sql = sql.replace(chr(34) + h_str.upper() + chr(34), chr(34) + col + chr(34))
        # Unquoted: bare DisplayName -> colN (word boundary)
        sql = re.sub(r'\b' + re.escape(h_str) + r'\b', col, sql, flags=re.IGNORECASE)
    return sql


def normalized_sql(value: str) -> str:
    value = value.strip().rstrip(";")
    value = re.sub(r"\s+", " ", value)
    return value.casefold()


def normalize_result(rows: list[tuple]) -> list[tuple[str, ...]]:
    normalized = []
    for row in rows:
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


def safe_execute(connection: sqlite3.Connection, sql: str) -> list[tuple]:
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
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
    outputs = llm.generate(
        prompts,
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
        use_tqdm=True,
    )

    connection = sqlite3.connect(f"file:{args.database}?mode=ro", uri=True)
    predictions = []
    counters = {
        "extracted": 0,
        "executed": 0,
        "execution_exact": 0,
        "sql_exact": 0,
    }
    for row, output in zip(rows, outputs, strict=True):
        raw = output.outputs[0].text
        sql = extract_sql(raw)
        # Apply header->colN remapping (quoted + unquoted identifiers)
        if sql is not None and row.get("header"):
            sql = remap_headers(sql, row["header"])
        error = None
        result = None
        if sql is not None:
            counters["extracted"] += 1
            try:
                result = safe_execute(connection, sql)
                counters["executed"] += 1
            except sqlite3.Error as exc:
                error = str(exc)
        execution_exact = (
            result is not None
            and normalize_result(result) == normalize_result(row["gold_result"])
        )
        sql_exact = (
            sql is not None
            and normalized_sql(sql) == normalized_sql(row["gold_sql"])
        )
        counters["execution_exact"] += int(execution_exact)
        counters["sql_exact"] += int(sql_exact)
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
            }
        )
    connection.close()

    total = len(rows)
    metrics = {
        "model": args.model,
        "model_label": args.model_label,
        "dataset": str(args.dataset),
        "samples": total,
        "elapsed_seconds": time.time() - started,
        "sql_extraction_rate": counters["extracted"] / total,
        "execution_rate": counters["executed"] / total,
        "execution_accuracy": counters["execution_exact"] / total,
        "normalized_sql_exact": counters["sql_exact"] / total,
        "scoring": "read-only SQLite execution; result rows compared order-insensitively",
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
