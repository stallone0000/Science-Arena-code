#!/usr/bin/env python3
"""Validate the text-only interleaved Science Arena dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/science_arena_text_interleaved_v1"


REQUIRED_FIELDS = [
    "benchmark_id",
    "discipline",
    "sequence_id",
    "turn_id",
    "turn_index",
    "question_text_staticized",
    "official_answer_staticized",
    "official_detailed_solution_staticized",
    "official_rubric_staticized",
    "max_score",
    "grading_type",
    "evaluation_protocol",
]

TEXT_FIELDS_SENT_TO_MODELS = [
    "question_text_staticized",
    "official_answer_staticized",
    "official_detailed_solution_staticized",
    "official_rubric_staticized",
]

FORBIDDEN_TEXT_PATTERNS = [
    "http://",
    "https://",
    "data:image",
    "image_url",
    "myqcloud.com",
    "cos.ap-",
    "<img",
    "<!-- media",
    "figuretext",
]


EXPECTED_MAX_SCORE_SUMS = {
    "ipho2025": 30.0,
    "icho2025": 385.0,
    "ccho2025": 267.0,
    "cpho2025": 320.0,
    "ibo2023": 453.0,
    "eupho2025": 30.0,
    "apho2025": 30.0,
    "nbpho2025": 72.0,
    "incho2026": 106.0,
    "usapho2025": 150.0,
    "usnco2025_part2": 100.0,
}


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def is_blank(value: Any) -> bool:
    return value is None or str(value).strip() == ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = iter_jsonl(args.dataset_dir / "benchmark_items.jsonl")
    errors: list[dict[str, Any]] = []
    warnings: list[dict[str, Any]] = []
    by_benchmark: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_turns: set[str] = set()

    for row in rows:
        benchmark_id = str(row.get("benchmark_id") or "")
        by_benchmark[benchmark_id].append(row)
        key = f"{benchmark_id}::{row.get('sequence_id')}::{row.get('turn_id')}"
        if key in seen_turns:
            errors.append({"kind": "duplicate_turn", "key": key})
        seen_turns.add(key)
        for field in REQUIRED_FIELDS:
            if is_blank(row.get(field)):
                errors.append({"kind": "missing_required_field", "key": key, "field": field})
        try:
            score = float(row.get("max_score"))
            if score <= 0:
                errors.append({"kind": "non_positive_max_score", "key": key, "max_score": row.get("max_score")})
        except Exception:
            errors.append({"kind": "invalid_max_score", "key": key, "max_score": row.get("max_score")})
        if row.get("evaluation_protocol") != "text_only_interleaved_staticized_v1":
            errors.append({"kind": "wrong_evaluation_protocol", "key": key, "value": row.get("evaluation_protocol")})
        for field in TEXT_FIELDS_SENT_TO_MODELS:
            text = str(row.get(field) or "")
            lowered = text.lower()
            for pattern in FORBIDDEN_TEXT_PATTERNS:
                if pattern in lowered:
                    errors.append({"kind": "forbidden_image_or_url_reference_in_model_text", "key": key, "field": field, "pattern": pattern})
        if row.get("image_refs") and not row.get("question_figures_staticized"):
            errors.append({"kind": "image_refs_without_staticized_figures", "key": key})
        if row.get("official_answer_image_refs") and not row.get("solution_figures_staticized"):
            errors.append({"kind": "answer_image_refs_without_staticized_figures", "key": key})
        comp = row.get("staticization_completeness") or {}
        for flag in ["question_complete", "figures_complete", "answer_complete", "rubric_complete", "max_score_reconciled"]:
            if flag not in comp:
                errors.append({"kind": "missing_staticization_flag", "key": key, "flag": flag})
            elif comp.get(flag) is not True:
                errors.append(
                    {
                        "kind": "staticization_flag_not_true",
                        "key": key,
                        "flag": flag,
                        "value": comp.get(flag),
                        "issues": comp.get("issues"),
                    }
                )
        if benchmark_id == "usnco2025_part2":
            if row.get("grading_type") != "derived_rubric_llm_judge":
                errors.append({"kind": "usnco_wrong_grading_type", "key": key, "value": row.get("grading_type")})
            rubric_status = str(row.get("official_rubric_status") or "").lower()
            rubric_text = str(row.get("official_rubric_staticized") or "").lower()
            if "derived" not in rubric_status or "derived" not in rubric_text:
                errors.append({"kind": "usnco_missing_derived_rubric_disclosure", "key": key})

    benchmark_summary: list[dict[str, Any]] = []
    for benchmark_id, bench_rows in sorted(by_benchmark.items()):
        score_sum = sum(float(row.get("max_score") or 0) for row in bench_rows)
        expected = EXPECTED_MAX_SCORE_SUMS.get(benchmark_id)
        if expected is not None and abs(score_sum - expected) > 1e-6:
            errors.append(
                {
                    "kind": "benchmark_score_sum_mismatch",
                    "benchmark_id": benchmark_id,
                    "expected": expected,
                    "observed": score_sum,
                }
            )
        sequence_turns: dict[str, list[int]] = defaultdict(list)
        for row in bench_rows:
            sequence_turns[str(row.get("sequence_id"))].append(int(row.get("turn_index") or 0))
        for sequence_id, turn_indexes in sequence_turns.items():
            expected_turns = list(range(1, len(turn_indexes) + 1))
            if sorted(turn_indexes) != expected_turns:
                errors.append(
                    {
                        "kind": "non_contiguous_turn_indexes",
                        "benchmark_id": benchmark_id,
                        "sequence_id": sequence_id,
                        "turn_indexes": sorted(turn_indexes),
                    }
                )
        benchmark_summary.append(
            {
                "benchmark_id": benchmark_id,
                "item_count": len(bench_rows),
                "sequence_count": len(sequence_turns),
                "max_score_sum": score_sum,
                "expected_max_score_sum": expected,
                "rows_with_images": sum(1 for row in bench_rows if row.get("image_refs")),
                "rows_with_answer_images": sum(1 for row in bench_rows if row.get("official_answer_image_refs")),
                "staticization_errors": sum(1 for row in bench_rows if row.get("staticization_error")),
            }
        )

    payload = {
        "dataset_dir": str(args.dataset_dir),
        "row_count": len(rows),
        "benchmark_count": len(by_benchmark),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "benchmark_summary": benchmark_summary,
        "errors": errors,
        "warnings": warnings,
    }
    output = args.output or (args.dataset_dir / "validation_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
