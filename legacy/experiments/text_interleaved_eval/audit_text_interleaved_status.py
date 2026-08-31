#!/usr/bin/env python3
"""Audit canonical text-interleaved run completeness.

This is intentionally read-only. It checks expected turn counts, generation
and judge errors, dirty finish reasons, and accidental image payloads.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("datasets/science_arena_text_interleaved_v1")
DEFAULT_RUN_ROOT = Path("artifacts/runs/text_interleaved_eval")
DEFAULT_MODEL_CONFIG = Path("configs/text_interleaved_models_20260524.json")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model)


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


def image_payload_count(row: dict[str, Any]) -> int:
    summary = ((row.get("generation_client_meta") or {}).get("final_payload_image_summary") or {})
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--model_config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--benchmark_id", action="append", default=[])
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    dataset_rows = iter_jsonl(args.dataset_dir / "benchmark_items.jsonl")
    expected_by_bench_seq: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in dataset_rows:
        expected_by_bench_seq.setdefault((str(row.get("benchmark_id")), str(row.get("sequence_id"))), []).append(row)
    benchmarks = sorted({bench for bench, _ in expected_by_bench_seq})
    if args.benchmark_id:
        benchmarks = [bench for bench in benchmarks if bench in set(args.benchmark_id)]

    if args.model:
        models = args.model
    else:
        models = json.loads(args.model_config.read_text(encoding="utf-8"))["models"]

    issues: list[dict[str, Any]] = []
    clean_counts: dict[str, dict[str, int]] = {}
    for model in models:
        model_dir = safe_model_name(model)
        clean_counts[model] = {"clean": 0, "issue": 0}
        for bench in benchmarks:
            run_dir = args.run_root / f"text_staticized_{model_dir}_{bench}_20260524_v2" / model_dir
            path = run_dir / "turn_results.jsonl"
            if not path.exists():
                issues.append({"model": model, "benchmark_id": bench, "reason": "missing_cell"})
                clean_counts[model]["issue"] += 1
                continue
            rows = iter_jsonl(path)
            rows_by_sequence: dict[str, list[dict[str, Any]]] = {}
            for row in rows:
                rows_by_sequence.setdefault(str(row.get("sequence_id")), []).append(row)
            missing: list[str] = []
            errors = 0
            bad_finish = 0
            image_rows = 0
            for (bench_id, sequence_id), expected in expected_by_bench_seq.items():
                if bench_id != bench:
                    continue
                got = rows_by_sequence.get(sequence_id, [])
                if len(got) != len(expected):
                    missing.append(f"{sequence_id}:{len(got)}/{len(expected)}")
                for row in got:
                    if row.get("generation_error") or row.get("judge_error") or (row.get("judge_result") or {}).get("parse_error"):
                        errors += 1
                    if is_bad_finish_reason(row.get("generation_finish_reason")) or is_bad_finish_reason(row.get("judge_finish_reason")):
                        bad_finish += 1
                    if image_payload_count(row):
                        image_rows += 1
            if missing or errors or bad_finish or image_rows:
                issues.append(
                    {
                        "model": model,
                        "benchmark_id": bench,
                        "missing_sequence_count": len(missing),
                        "error_count": errors,
                        "bad_finish_reason_count": bad_finish,
                        "image_payload_row_count": image_rows,
                        "missing_sequence_sample": missing[:8],
                        "row_count": len(rows),
                    }
                )
                clean_counts[model]["issue"] += 1
            else:
                clean_counts[model]["clean"] += 1

    payload = {"issue_count": len(issues), "clean_counts": clean_counts, "issues": issues}
    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    print(f"issue_count {len(issues)}")
    for model, counts in clean_counts.items():
        print(f"{model}: clean={counts['clean']} issue={counts['issue']}")
    print()
    for issue in issues:
        if issue.get("reason") == "missing_cell":
            print(f"{issue['model']} | {issue['benchmark_id']} | missing_cell")
        else:
            print(
                f"{issue['model']} | {issue['benchmark_id']} | "
                f"missingseq={issue['missing_sequence_count']} err={issue['error_count']} "
                f"bad={issue['bad_finish_reason_count']} img={issue['image_payload_row_count']} | "
                f"{','.join(issue['missing_sequence_sample'])}"
            )


if __name__ == "__main__":
    main()
