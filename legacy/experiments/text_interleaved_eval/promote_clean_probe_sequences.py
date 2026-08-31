#!/usr/bin/env python3
"""Promote clean completed sequences from a probe run into canonical runs."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("datasets/science_arena_text_interleaved_v1")
DEFAULT_OUTPUT_ROOT = Path("artifacts/runs/text_interleaved_eval")


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('benchmark_id')}::{row.get('sequence_id')}::{row.get('turn_id')}"


def image_payload_count(row: dict[str, Any]) -> int:
    meta = row.get("generation_client_meta") or {}
    summary = meta.get("final_payload_image_summary") or {}
    return sum(
        int(summary.get(key) or 0)
        for key in [
            "chat_image_url_count",
            "data_url_image_count",
            "empty_image_url_count",
            "remote_url_image_count",
            "responses_input_image_count",
        ]
    )


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


def is_clean(row: dict[str, Any]) -> bool:
    judge_result = row.get("judge_result") or {}
    return not (
        row.get("generation_error")
        or row.get("judge_error")
        or judge_result.get("parse_error")
        or image_payload_count(row)
        or is_bad_finish_reason(row.get("generation_finish_reason"))
        or is_bad_finish_reason(row.get("judge_finish_reason"))
    )


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
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--source_run_dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_suffix", default="20260524_v2")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    model_dir = safe_model_name(args.model)
    source_path = args.source_run_dir / model_dir / "turn_results.jsonl"
    source_rows = iter_jsonl(source_path)
    if not source_rows:
        raise SystemExit(f"No source rows found: {source_path}")

    dataset_rows = iter_jsonl(args.dataset_dir / "benchmark_items.jsonl")
    dataset_by_key = {row_key(row): row for row in dataset_rows}
    order = {row_key(row): index for index, row in enumerate(dataset_rows)}
    expected_by_sequence: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in dataset_rows:
        expected_by_sequence[(str(row.get("benchmark_id")), str(row.get("sequence_id")))].add(row_key(row))

    source_by_sequence: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        source_by_sequence[(str(row.get("benchmark_id")), str(row.get("sequence_id")))].append(row)

    promoted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for (benchmark_id, sequence_id), rows in sorted(source_by_sequence.items()):
        keys = {row_key(row) for row in rows}
        expected_keys = expected_by_sequence.get((benchmark_id, sequence_id), set())
        dirty_keys = [row_key(row) for row in rows if not is_clean(row)]
        missing_keys = sorted(expected_keys - keys)
        extra_keys = sorted(keys - set(dataset_by_key))
        if dirty_keys or missing_keys or extra_keys:
            skipped.append(
                {
                    "benchmark_id": benchmark_id,
                    "sequence_id": sequence_id,
                    "dirty_count": len(dirty_keys),
                    "missing_count": len(missing_keys),
                    "extra_count": len(extra_keys),
                }
            )
            continue

        run_name = f"text_staticized_{safe_model_name(args.model)}_{benchmark_id}_{args.run_suffix}"
        target_dir = args.output_root / run_name / model_dir
        target_path = target_dir / "turn_results.jsonl"
        target_rows = iter_jsonl(target_path)
        by_key = {row_key(row): row for row in target_rows}
        for row in rows:
            by_key[row_key(row)] = row
        merged = sorted(by_key.values(), key=lambda row: order.get(row_key(row), 10**12))
        if not args.dry_run:
            write_jsonl(target_path, merged)
            save_json(target_dir / "benchmark_summary.json", recompute_summary(merged, dataset_rows, args.model))
        promoted.append(
            {
                "benchmark_id": benchmark_id,
                "sequence_id": sequence_id,
                "rows": len(rows),
                "target_rows": len(merged),
                "target_run_dir": str(target_dir.parent),
            }
        )

    print(json.dumps({"source": str(source_path), "promoted": promoted, "skipped": skipped}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
