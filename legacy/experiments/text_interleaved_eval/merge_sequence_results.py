#!/usr/bin/env python3
"""Overlay clean sequence rows from one text-interleaved run into another."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("datasets/science_arena_text_interleaved_v1")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model)


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('benchmark_id')}::{row.get('sequence_id')}::{row.get('turn_id')}"


def is_bad_finish_reason(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text in {"stop", "completed", "complete", "objective_key"}:
        return False
    return any(
        fragment in text
        for fragment in (
            "content_filter",
            "finish_content_filter",
            "length",
            "max_token",
            "max_output",
            "truncat",
        )
    )


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def recompute_summary(rows: list[dict[str, Any]], dataset_rows: list[dict[str, Any]], model: str) -> list[dict[str, Any]]:
    report_scales = {
        str(row.get("benchmark_id")): float(row.get("report_scale") or 0)
        for row in dataset_rows
    }
    by_bench: dict[str, dict[str, Any]] = {}
    for result in rows:
        bench = str(result.get("benchmark_id"))
        summary = by_bench.setdefault(
            bench,
            {
                "benchmark_id": bench,
                "model": model,
                "score": 0.0,
                "max_score": 0.0,
                "reported_score": 0.0,
                "report_scale": report_scales.get(bench) or 0.0,
                "turn_count": 0,
                "generation_error_count": 0,
                    "judge_error_count": 0,
                    "bad_finish_reason_count": 0,
                    "cost_cny": 0.0,
                },
        )
        summary["score"] += float(result.get("score") or 0)
        summary["max_score"] += float(result.get("max_score") or 0)
        summary["turn_count"] += 1
        summary["cost_cny"] += float(result.get("total_cost_cny") or 0)
        if result.get("generation_error"):
            summary["generation_error_count"] += 1
        judge_result = result.get("judge_result") or {}
        if result.get("judge_error") or judge_result.get("parse_error"):
            summary["judge_error_count"] += 1
        if is_bad_finish_reason(result.get("generation_finish_reason")) or is_bad_finish_reason(result.get("judge_finish_reason")):
            summary["bad_finish_reason_count"] += 1
    for summary in by_bench.values():
        if summary["max_score"] and summary["report_scale"]:
            summary["reported_score"] = summary["score"] / summary["max_score"] * summary["report_scale"]
    return sorted(by_bench.values(), key=lambda row: row["benchmark_id"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--target_run_dir", type=Path, required=True)
    parser.add_argument("--source_run_dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--sequence_id", action="append", default=[])
    parser.add_argument("--allow_dirty_source", action="store_true")
    parser.add_argument(
        "--create_target",
        action="store_true",
        help="Create the target model result file when it does not yet exist.",
    )
    args = parser.parse_args()

    model_dir_name = safe_model_name(args.model)
    target_dir = args.target_run_dir / model_dir_name
    source_dir = args.source_run_dir / model_dir_name
    target_path = target_dir / "turn_results.jsonl"
    source_path = source_dir / "turn_results.jsonl"
    if not target_path.exists() and not args.create_target:
        raise SystemExit(f"Missing target turn_results: {target_path}")
    if not source_path.exists():
        raise SystemExit(f"Missing source turn_results: {source_path}")

    dataset_rows = iter_jsonl(args.dataset_dir / "benchmark_items.jsonl")
    order = {row_key(row): index for index, row in enumerate(dataset_rows)}
    target_rows = iter_jsonl(target_path) if target_path.exists() else []
    source_rows = iter_jsonl(source_path)
    requested_sequences = set(args.sequence_id)
    if requested_sequences:
        source_rows = [row for row in source_rows if str(row.get("sequence_id")) in requested_sequences]
    if not source_rows:
        raise SystemExit("No source rows selected.")
    if not args.allow_dirty_source:
        dirty = [
            row_key(row)
            for row in source_rows
            if row.get("generation_error")
            or row.get("judge_error")
            or (row.get("judge_result") or {}).get("parse_error")
            or is_bad_finish_reason(row.get("generation_finish_reason"))
            or is_bad_finish_reason(row.get("judge_finish_reason"))
        ]
        if dirty:
            raise SystemExit(f"Refusing to merge dirty source rows: {dirty[:10]}")

    by_key = {row_key(row): row for row in target_rows}
    for row in source_rows:
        key = row_key(row)
        if key not in order:
            raise SystemExit(f"Source row is not in dataset order: {key}")
        by_key[key] = row
    merged = sorted(by_key.values(), key=lambda row: order.get(row_key(row), 10**12))
    write_jsonl(target_path, merged)
    save_json(target_dir / "benchmark_summary.json", recompute_summary(merged, dataset_rows, args.model))
    print(
        json.dumps(
            {
                "target": str(target_path),
                "source": str(source_path),
                "merged_rows": len(source_rows),
                "target_rows": len(merged),
                "sequences": sorted({str(row.get("sequence_id")) for row in source_rows}),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
