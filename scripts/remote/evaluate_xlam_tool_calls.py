#!/usr/bin/env python3
"""Evaluate deterministic JSON tool-call generation with vLLM."""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams


def normalize_call(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    if "function" in value and isinstance(value["function"], dict):
        value = value["function"]
    name = value.get("name")
    arguments = value.get("arguments", {})
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def parse_response(text: str) -> tuple[list[dict[str, Any]] | None, str]:
    stripped = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else stripped

    starts = [index for index in (candidate.find("["), candidate.find("{")) if index >= 0]
    if not starts:
        return None, "no_json_start"
    candidate = candidate[min(starts) :]

    try:
        value, _ = json.JSONDecoder().raw_decode(candidate)
    except json.JSONDecodeError:
        return None, "json_decode_error"

    if isinstance(value, dict) and value.get("action") in {"tool_call", "no_tool", "clarify"}:
        if value.get("action") == "tool_call":
            value = value.get("calls") or value.get("tool_calls") or []
        else:
            value = []
    elif isinstance(value, dict) and "tool_calls" in value:
        value = value["tool_calls"]
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, list):
        return None, "json_not_list"

    calls = [normalize_call(item) for item in value]
    if any(call is None for call in calls):
        return None, "invalid_call_schema"
    return calls, "ok"


def canonical(calls: list[dict[str, Any]]) -> str:
    return json.dumps(calls, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def unordered(calls: list[dict[str, Any]]) -> list[str]:
    return sorted(canonical([call]) for call in calls)


def expected_calls_from_row(row: dict[str, Any]) -> list[dict[str, Any]]:
    if "expected_calls" in row:
        return row["expected_calls"]
    expected = row.get("expected")
    if isinstance(expected, dict) and expected.get("action") in {"tool_call", "no_tool", "clarify"}:
        if expected.get("action") == "tool_call":
            calls = [normalize_call(item) for item in expected.get("calls", [])]
            if all(call is not None for call in calls):
                return calls  # type: ignore[return-value]
        else:
            return []
    completion = row.get("completion")
    if isinstance(completion, str):
        try:
            value = json.loads(completion)
        except json.JSONDecodeError:
            value = None
        if isinstance(value, dict):
            if value.get("action") == "tool_call" and isinstance(value.get("tool_call"), dict):
                call = value["tool_call"]
                name = call.get("name") or call.get("tool")
                arguments = call.get("arguments", {})
                if isinstance(name, str) and isinstance(arguments, dict):
                    return [{"name": name, "arguments": arguments}]
            if isinstance(value.get("tool_calls"), list):
                calls = [normalize_call(item) for item in value["tool_calls"]]
                if all(call is not None for call in calls):
                    return calls  # type: ignore[return-value]
    raise KeyError("row must contain expected_calls or a parseable tool-call completion")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-label", required=True)
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--max-tokens", type=int, default=192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    prompts = [row["prompt"] for row in rows]

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
        temperature=0.0,
        max_tokens=args.max_tokens,
    )
    outputs = []
    for start in range(0, len(prompts), args.batch_size):
        outputs.extend(llm.generate(prompts[start : start + args.batch_size], sampling, use_tqdm=True))
    elapsed = time.time() - started

    counters = Counter()
    predictions = []
    for row, output in zip(rows, outputs, strict=True):
        text = output.outputs[0].text
        predicted, parse_status = parse_response(text)
        expected = expected_calls_from_row(row)
        counters["total"] += 1
        counters[f"parse_{parse_status}"] += 1

        ordered_exact = predicted is not None and predicted == expected
        unordered_exact = predicted is not None and unordered(predicted) == unordered(expected)
        names_exact = predicted is not None and sorted(
            call["name"] for call in predicted
        ) == sorted(call["name"] for call in expected)
        count_exact = predicted is not None and len(predicted) == len(expected)
        strict_format = text.strip() == canonical(expected)

        counters["ordered_exact"] += int(ordered_exact)
        counters["unordered_exact"] += int(unordered_exact)
        counters["names_exact"] += int(names_exact)
        counters["count_exact"] += int(count_exact)
        counters["strict_format"] += int(strict_format)

        predictions.append(
            {
                "id": row["id"],
                "family": row.get("family") or row.get("source") or row.get("mixture_source") or "unknown",
                "expected_calls": expected,
                "raw_response": text,
                "parsed_calls": predicted,
                "parse_status": parse_status,
                "ordered_exact": ordered_exact,
                "unordered_exact": unordered_exact,
                "names_exact": names_exact,
                "count_exact": count_exact,
                "strict_format": strict_format,
                "prompt_tokens": len(output.prompt_token_ids),
                "completion_tokens": len(output.outputs[0].token_ids),
            }
        )

    total = counters["total"]
    metrics = {
        "model": args.model,
        "model_label": args.model_label,
        "dataset": str(args.dataset),
        "samples": total,
        "elapsed_seconds": elapsed,
        "samples_per_second": total / elapsed if elapsed else 0,
        "json_parse_rate": counters["parse_ok"] / total,
        "ordered_exact": counters["ordered_exact"] / total,
        "unordered_exact": counters["unordered_exact"] / total,
        "tool_names_exact": counters["names_exact"] / total,
        "call_count_exact": counters["count_exact"] / total,
        "strict_format": counters["strict_format"] / total,
        "parse_status_counts": {
            key.removeprefix("parse_"): value
            for key, value in counters.items()
            if key.startswith("parse_")
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.model_label}_predictions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n" for row in predictions
        ),
        encoding="utf-8",
    )
    (args.output_dir / f"{args.model_label}_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
