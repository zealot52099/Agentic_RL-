#!/usr/bin/env python3
"""Build Phase19 SQL schema/value-grounding SFT data.

Phase18 unified the Data Agent action schema, but the main WikiSQL v2 failures
were still semantic: wrong WHERE/value/column grounding and missing aggregation.
This script creates targeted SFT rows from executable WikiSQL training data:

* direct SQL rows with explicit relevant-column, value, and aggregation hints;
* repair rows where a realistic wrong SQL is followed by execution feedback;
* Phase18 replay rows to preserve the Data Agent action/tool surface.

The fixed 256-row WikiSQL v2 probe is intentionally not consumed as training
data. It is used only by the runner for post-training evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any


CANONICAL_SYSTEM = (
    "You are the main model of a Data Agent. Use the provided tools when they "
    "are needed, ask for clarification when required information is missing, "
    "refuse unsafe requests, or provide a final answer when no tool is needed. "
    "Return exactly one JSON object and no extra text.\n\n"
    "Allowed actions:\n"
    "{\"action\":\"tool_call\",\"calls\":[{\"name\":\"tool_name\",\"arguments\":{...}}]}\n"
    "{\"action\":\"clarify\",\"missing\":[\"field\"],\"message\":\"question\"}\n"
    "{\"action\":\"refuse\",\"message\":\"reason\"}\n"
    "{\"action\":\"final\",\"answer\":\"answer\"}"
)


SQL_TOOLS = [
    {
        "name": "list_tables",
        "description": "List available database tables.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "describe_table",
        "description": "Return columns and types for one table.",
        "input_schema": {
            "type": "object",
            "required": ["table"],
            "properties": {"table": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "run_sql",
        "description": "Execute one read-only SQLite SELECT query.",
        "input_schema": {
            "type": "object",
            "required": ["sql"],
            "properties": {"sql": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "final_answer",
        "description": "Return the final natural-language answer to the user.",
        "input_schema": {
            "type": "object",
            "required": ["answer"],
            "properties": {"answer": {"type": "string"}},
            "additionalProperties": False,
        },
    },
]


AGG_FUNCS = ("MIN", "MAX", "COUNT", "SUM", "AVG")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_id(*parts: Any) -> str:
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def normalize_cell(value: Any) -> Any:
    if isinstance(value, str):
        return value.casefold()
    return value


def normalize_rows(rows: list[list[Any]], limit: int = 5) -> list[list[Any]]:
    return [[normalize_cell(value) for value in row] for row in rows[:limit]]


def extract_sql_columns(sql: str) -> list[str]:
    return sorted(set(re.findall(r'"?(col\d+)"?', sql, flags=re.I)), key=lambda item: int(item[3:]))


def extract_sql_literals(sql: str) -> list[str]:
    values = re.findall(r"'([^']*)'", sql)
    values.extend(re.findall(r'=\s*"([^"]*)"', sql))
    return [value.casefold() for value in values if value != ""]


def extract_aggregations(sql: str) -> list[dict[str, str]]:
    out = []
    for func, col in re.findall(r"\b(MIN|MAX|COUNT|SUM|AVG)\s*\(\s*\"?(col\d+)\"?\s*\)", sql, flags=re.I):
        out.append({"function": func.upper(), "column": col.lower()})
    return out


def relevant_values(rows: list[list[Any]], columns: list[str], literals: list[str]) -> list[dict[str, str]]:
    hints = []
    seen = set()
    for col in columns:
        index = int(col[3:])
        for row in rows:
            if index >= len(row):
                continue
            value = normalize_cell(row[index])
            value_text = str(value)
            for literal in literals:
                if literal and (literal == value_text or literal in value_text or value_text in literal):
                    key = (col, value_text)
                    if key not in seen:
                        seen.add(key)
                        hints.append({"column": col, "value": value_text})
    return hints[:12]


def parse_phase8_solution(row: dict[str, Any]) -> dict[str, Any] | None:
    try:
        solution = json.loads(row["solution"])
    except Exception:
        return None
    required = {"table_name", "header", "rows", "gold_sql"}
    if not required.issubset(solution):
        return None
    return solution


def column_lines(header: list[str], types: list[str] | None, relevant: set[str]) -> str:
    lines = []
    for index, name in enumerate(header):
        col = f"col{index}"
        kind = types[index] if types and index < len(types) else "text"
        marker = " [RELEVANT]" if col in relevant else ""
        lines.append(f'- `{col}` means "{name}" ({kind}){marker}')
    return "\n".join(lines)


def render_grounding_context(
    *,
    table: str,
    header: list[str],
    types: list[str] | None,
    rows: list[list[Any]],
    question: str,
    gold_sql: str,
    previous_sql: str | None = None,
    feedback: str | None = None,
) -> str:
    columns = extract_sql_columns(gold_sql)
    literals = extract_sql_literals(gold_sql)
    aggregations = extract_aggregations(gold_sql)
    value_hints = relevant_values(rows, columns, literals)
    samples = "\n".join(
        json.dumps({f"col{idx}": value for idx, value in enumerate(row)}, ensure_ascii=False)
        for row in normalize_rows(rows)
    )
    grounding = {
        "relevant_columns": [
            {
                "column": col,
                "header": header[int(col[3:])] if int(col[3:]) < len(header) else "",
            }
            for col in columns
        ],
        "candidate_values": value_hints,
        "required_aggregations": aggregations,
        "rules": [
            "Use only physical colN identifiers in SQL.",
            "Quoted header names are descriptions, not SQL columns.",
            "If the question asks first/earliest/least, use MIN on the answer column when appropriate.",
            "If the question asks last/latest/greatest, use MAX on the answer column when appropriate.",
            "Use WHERE values that appear in candidate_values or sample rows when available.",
        ],
    }
    parts = [
        f'Table: "{table}"',
        "Columns:\n" + column_lines(header, types, set(columns)),
        "Sample rows:\n" + samples,
        "GROUNDING_HINTS:\n" + json.dumps(grounding, ensure_ascii=False, indent=2),
    ]
    if previous_sql:
        parts.extend(
            [
                "Previous SQL:\n" + previous_sql,
                "Execution feedback:\n" + (feedback or "The SQL executed but returned a wrong result."),
            ]
        )
    parts.append("Question:\n" + question)
    return "\n\n".join(parts)


def render_prompt(context: str) -> str:
    return (
        CANONICAL_SYSTEM
        + "\n\nAVAILABLE_TOOLS:\n"
        + canonical_json(SQL_TOOLS)
        + "\n\n"
        + context.strip()
        + "\n\nUSER:\nGenerate the next Data Agent action.\n\nASSISTANT:\n"
    )


def sql_action(sql: str) -> dict[str, Any]:
    return {"action": "tool_call", "calls": [{"name": "run_sql", "arguments": {"sql": sql.strip()}}]}


def sft_row(
    *,
    id_: str,
    prompt: str,
    sql: str,
    source: str,
    mixture_source: str,
    loss_weight: float,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    return {
        "id": id_,
        "prompt": prompt,
        "completion": canonical_json(sql_action(sql)),
        "source": source,
        "mixture_source": mixture_source,
        "loss_weight": round(loss_weight, 4),
        "render_template_version": "phase19_sql_grounding_action_v1",
        "action_schema_version": "data_agent_action_v1",
        "metadata": metadata,
    }


def corrupt_missing_aggregation(sql: str) -> str | None:
    match = re.search(r"\b(MIN|MAX|COUNT|SUM|AVG)\s*\(\s*(\"?col\d+\"?)\s*\)", sql, flags=re.I)
    if not match:
        return None
    return sql[: match.start()] + match.group(2) + sql[match.end() :]


def corrupt_drop_where(sql: str) -> str | None:
    if " WHERE " not in sql.upper():
        return None
    return re.split(r"\bWHERE\b", sql, maxsplit=1, flags=re.I)[0].strip()


def corrupt_wrong_value(sql: str) -> str | None:
    literals = extract_sql_literals(sql)
    if not literals:
        return None
    value = literals[0]
    if not value:
        return None
    return re.sub(re.escape("'" + value + "'"), "'__wrong_value__'", sql, count=1, flags=re.I)


def make_phase8_rows(source_rows: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    candidates = []
    for row in source_rows:
        solution = parse_phase8_solution(row)
        if not solution:
            continue
        table = str(solution["table_name"])
        header = list(solution["header"])
        rows = list(solution["rows"])
        gold_sql = str(solution["gold_sql"]).strip()
        question = str(row.get("question") or "").strip()
        if not question or not re.match(r"^\s*SELECT\b", gold_sql, flags=re.I):
            continue
        base_context = render_grounding_context(
            table=table,
            header=header,
            types=None,
            rows=rows,
            question=question,
            gold_sql=gold_sql,
        )
        candidates.append(
            sft_row(
                id_=f"phase19-direct-{stable_id(table, question, gold_sql)}",
                prompt=render_prompt(base_context),
                sql=gold_sql,
                source="phase19_wikisql_grounded_direct",
                mixture_source="sql_grounding_direct",
                loss_weight=1.2,
                metadata={"table": table, "upstream": "phase8_swift_wikisql_grpo"},
            )
        )
        corruptions = [
            ("missing_aggregation", corrupt_missing_aggregation(gold_sql)),
            ("dropped_where", corrupt_drop_where(gold_sql)),
            ("wrong_value", corrupt_wrong_value(gold_sql)),
        ]
        for kind, previous_sql in corruptions:
            if not previous_sql or previous_sql == gold_sql:
                continue
            feedback = {
                "missing_aggregation": "The SQL executed but returned non-aggregated rows. Re-check whether the question requires MIN/MAX/COUNT/SUM/AVG.",
                "dropped_where": "The SQL executed but ignored required filters. Re-check WHERE columns and values.",
                "wrong_value": "The SQL returned an empty or wrong result. Re-check the candidate cell values.",
            }[kind]
            context = render_grounding_context(
                table=table,
                header=header,
                types=None,
                rows=rows,
                question=question,
                gold_sql=gold_sql,
                previous_sql=previous_sql,
                feedback=feedback,
            )
            candidates.append(
                sft_row(
                    id_=f"phase19-repair-{kind}-{stable_id(table, question, previous_sql, gold_sql)}",
                    prompt=render_prompt(context),
                    sql=gold_sql,
                    source="phase19_wikisql_grounded_repair",
                    mixture_source=f"sql_grounding_repair_{kind}",
                    loss_weight=1.5 if kind != "wrong_value" else 1.35,
                    metadata={"table": table, "corruption": kind, "upstream": "phase8_swift_wikisql_grpo"},
                )
            )
    rng.shuffle(candidates)
    return candidates[:limit]


def replay_phase18(rows: list[dict[str, Any]], limit: int, rng: random.Random) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        completion = str(row.get("completion", "")).strip()
        if not completion.startswith("{"):
            continue
        source = str(row.get("source", ""))
        if source.startswith("phase18_phase5_spider_low_weight"):
            continue
        copied = dict(row)
        copied["id"] = "phase19-replay-" + str(row.get("id", stable_id(row)))
        copied["source"] = "phase19_phase18_replay"
        copied["loss_weight"] = min(float(row.get("loss_weight", 1.0)), 0.45)
        copied["metadata"] = {
            **(copied.get("metadata") if isinstance(copied.get("metadata"), dict) else {}),
            "phase19_replay_reason": "retain Data Agent tool/multi-turn action surface",
        }
        out.append(copied)
    rng.shuffle(out)
    return out[:limit]


def dedup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen = set()
    out = []
    for row in rows:
        key = hashlib.sha256((row["prompt"] + "\n" + row["completion"]).encode("utf-8")).hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase8-wikisql-train", type=Path, required=True)
    parser.add_argument("--phase18-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--max-grounded", type=int, default=12000)
    parser.add_argument("--max-replay", type=int, default=6000)
    parser.add_argument("--validation", type=int, default=512)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    rows = []
    phase8 = read_jsonl(args.phase8_wikisql_train)
    phase18 = read_jsonl(args.phase18_train)
    rows.extend(make_phase8_rows(phase8, args.max_grounded, rng))
    rows.extend(replay_phase18(phase18, args.max_replay, rng))
    rows = dedup(rows)
    rng.shuffle(rows)

    validation = rows[: args.validation]
    train = rows[args.validation :]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "train_phase19_sql_grounding_sft.jsonl", train)
    write_jsonl(args.output_dir / "validation_phase19_sql_grounding_sft.jsonl", validation)
    write_jsonl(args.output_dir / "audit_examples.jsonl", rows[:80])

    manifest = {
        "dataset_id": "phase19_sql_schema_value_grounding_sft",
        "created_at": "2026-07-02",
        "train": len(train),
        "validation": len(validation),
        "total": len(rows),
        "inputs": {
            "phase8_wikisql_train": str(args.phase8_wikisql_train),
            "phase18_train": str(args.phase18_train),
        },
        "source_counts": dict(Counter(row["source"] for row in rows)),
        "mixture_counts": dict(Counter(row["mixture_source"] for row in rows).most_common()),
        "leakage_policy": "The fixed WikiSQL v2 256-row probe is not used for training.",
        "target_failure_modes": [
            "wrong_where_or_value_or_column",
            "wrong_missing_aggregation",
            "exec_error_original_header_or_bad_identifier",
        ],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
