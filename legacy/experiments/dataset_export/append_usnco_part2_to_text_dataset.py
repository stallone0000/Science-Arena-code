#!/usr/bin/env python3
"""Append the staticized USNCO 2025 Part II rows to the text dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/science_arena_text_interleaved_v1"
DEFAULT_USNCO = ROOT / "data/science_arena/usnco2025/fix_USNCO-2025-PartII-staticized.json"


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('benchmark_id')}::{row.get('sequence_id')}::{row.get('turn_id')}"


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def convert_usnco(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        qid = str(row.get("question_id") or f"usnco2025_part2_q{index}")
        max_score = float(row.get("max_score") or 0)
        completeness = dict(row.get("completeness") or {})
        completeness.setdefault("max_score_reconciled", True)
        converted.append(
            {
                "benchmark_id": "usnco2025_part2",
                "discipline": "chemistry",
                "sequence_id": "usnco2025_part2_full",
                "turn_id": qid,
                "turn_index": index,
                "interleaved_policy": "include_prior_turns_in_same_sequence",
                "grading_type": "derived_rubric_llm_judge",
                "max_score": max_score,
                "report_scale": 100,
                "release_date": "2025-06",
                "source_quality": "official_pdf_text_staticized_derived_rubric",
                "prompt": as_text(row.get("question_text_staticized")),
                "question_text_staticized": as_text(row.get("question_text_staticized")),
                "official_answer": as_text(row.get("official_answer_staticized")),
                "official_answer_staticized": as_text(row.get("official_answer_staticized")),
                "official_detailed_solution_staticized": as_text(row.get("official_detailed_solution_staticized")),
                "official_rubric": as_text(row.get("official_rubric_staticized")),
                "official_rubric_staticized": as_text(row.get("official_rubric_staticized")),
                "official_rubric_status": "derived_from_official_solution_and_official_question_weight",
                "image_refs": [],
                "images": [],
                "official_answer_image_refs": [],
                "official_answer_images": [],
                "question_figures_staticized": [],
                "solution_figures_staticized": [],
                "staticization_completeness": completeness,
                "staticization_reviewer_notes": as_text(row.get("visual_staticization_notes")),
                "staticization_model": row.get("staticization_model", ""),
                "staticization_finish_reason": "pre_staticized_usnco_builder",
                "staticization_error": "",
                "staticization_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_cny": 0},
                "evaluation_protocol": "text_only_interleaved_staticized_v1",
                "source_refs": row.get("source_files") or [],
                "source_urls": row.get("source_urls") or [],
                "status": "staticized_text_complete_derived_rubric",
                "caveats": [
                    "USNCO public Part II PDF provides official question weights and official solutions, but not a fine-grained official point rubric. The rubric in this row is derived from the official solution and official question weight.",
                    "Question figures were staticized into text from the official question and solution text; original PDF download is blocked locally by ACS WAF, so this row keeps official URL references for audit.",
                ],
            }
        )
    return converted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--usnco_json", type=Path, default=DEFAULT_USNCO)
    args = parser.parse_args()

    current = iter_jsonl(args.dataset_dir / "benchmark_items.jsonl")
    current = [row for row in current if row.get("benchmark_id") != "usnco2025_part2"]
    usnco_rows = json.loads(args.usnco_json.read_text(encoding="utf-8"))
    converted = convert_usnco(usnco_rows)
    merged = current + converted
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in merged:
        key = row_key(row)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        raise SystemExit(f"Duplicate rows after merge: {duplicates[:5]}")
    write_jsonl(args.dataset_dir / "benchmark_items.jsonl", merged)
    print(json.dumps({"dataset_dir": str(args.dataset_dir), "added_usnco_rows": len(converted), "total_rows": len(merged)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
