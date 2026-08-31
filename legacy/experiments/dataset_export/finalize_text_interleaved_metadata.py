#!/usr/bin/env python3
"""Finalize non-prompt metadata for the text-only interleaved dataset."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_DATASET = Path("datasets/science_arena_text_interleaved_v1")

RUBRIC_STATUS_BY_BENCHMARK = {
    "apho2025": "official_rubric_staticized_needs_human_review",
    "ccho2025": "official_rubric_staticized_source_caveat",
    "cpho2025": "official_rubric_staticized_some_coarse_or_repaired_rows",
    "eupho2025": "official_rubric_staticized",
    "ibo2023": "objective_key_official",
    "icho2025": "official_rubric_staticized_postprocessed_caveat",
    "incho2026": "official_rubric_page_range_staticized_draft",
    "ipho2025": "official_rubric_staticized_postprocessed_caveat",
    "nbpho2025": "official_rubric_staticized_experimental_caveat",
    "usapho2025": "derived_from_official_solution_no_public_point_rubric",
    "usnco2025_part2": "derived_from_official_solution_and_official_question_weight",
}

RELEASE_DATE_BY_BENCHMARK = {
    "ccho2025": "2025-10",
    "cpho2025": "2025-10",
    "nbpho2025": "2025-04",
}

ADDITIONAL_SOURCE_FILES_BY_BENCHMARK = {
    "apho2025": [
        "data/science_arena/apho2025/fix_APHO-2025-latex.json",
        "data/science_arena/candidate_benchmarks/apho2025/theory_questions_apho_2025.pdf",
        "data/science_arena/candidate_benchmarks/apho2025/theory_solution_apho_2025.pdf",
    ],
    "ccho2025": [
        "data/science_arena/ccho2025/fix_CCHO-2025-latex.json",
        "data/science_arena/candidate_benchmarks/sources/ccho2025/ccho2025_final_round1_questions_answers_scoring_draft.pdf",
        "data/science_arena/candidate_benchmarks/sources/ccho2025/ccho2025_final_theory_weighting_official.pdf",
    ],
    "cpho2025": [
        "data/science_arena/cpho2025/fix_CPHO-2025-latex.json",
    ],
    "eupho2025": [
        "data/science_arena/eupho2025/fix_EuPhO-2025-latex.json",
        "data/science_arena/candidate_benchmarks/eupho2025/EuPhO_2025_Theory_ENG.pdf",
        "data/science_arena/candidate_benchmarks/eupho2025/EuPhO25_Theory_Solutions_20250616_1120.pdf",
    ],
    "ibo2023": [
        "data/science_arena/ibo2023/items_data.json",
        "data/science_arena/ibo2023/scoring_key.json",
    ],
    "icho2025": [
        "data/science_arena/icho2025/fix_ICHO-2025-latex.json",
        "data/science_arena/candidate_benchmarks/icho2025/ICHO-2025-UAE-Theory-Exam-Official-English.pdf",
        "data/science_arena/candidate_benchmarks/icho2025/ICHO-2025-UAE-Theory-Solutions-Official-English.pdf",
    ],
    "incho2026": [
        "data/science_arena/candidate_benchmarks/hbcse2026/INChO2026-Question-en.pdf",
        "data/science_arena/candidate_benchmarks/hbcse2026/INChO2026-Solution-20260220.pdf",
    ],
    "ipho2025": [
        "data/science_arena/ipho2025/IPhO_2025_S1_solution.json",
        "data/science_arena/ipho2025/IPhO_2025_S2_solution.json",
        "data/science_arena/ipho2025/IPhO_2025_S3_solution.json",
        "data/science_arena/candidate_benchmarks/ipho2025/exam-theory-Q1-english_2025-07-20_1904-UTC.pdf",
        "data/science_arena/candidate_benchmarks/ipho2025/exam-theory-Q2-english_2025-07-20_1917-UTC.pdf",
        "data/science_arena/candidate_benchmarks/ipho2025/exam-theory-Q3-english_2025-07-20_1917-UTC.pdf",
        "data/science_arena/candidate_benchmarks/ipho2025/solution-theory-Q1-english.pdf",
        "data/science_arena/candidate_benchmarks/ipho2025/solution-theory-Q2-english.pdf",
        "data/science_arena/candidate_benchmarks/ipho2025/solution-theory-Q3-english.pdf",
    ],
    "nbpho2025": [
        "data/science_arena/nbpho2025/fix_NBPhO-2025-latex.json",
        "data/science_arena/candidate_benchmarks/nbpho2025/NBPhO_2025_problems_V1.0.pdf",
        "data/science_arena/candidate_benchmarks/nbpho2025/00_NBPhO_2025_solutions_v1.0.pdf",
    ],
    "usapho2025": [
        "data/science_arena/candidate_benchmarks/usapho2025/USAPhO_Exam_25.pdf",
        "data/science_arena/candidate_benchmarks/usapho2025/2025-USAPhO-Exam_solutions.pdf",
    ],
    "usnco2025_part2": [
        "data/science_arena/usnco2025/fix_USNCO-2025-PartII-staticized.json",
        "data/science_arena/candidate_benchmarks/usnco2025_jina/2025-usnco-national-exam-part-ii.pdf.md",
        "data/science_arena/candidate_benchmarks/usnco2025_official_download_retry/2025-usnco-national-exam-part-ii.pdf",
    ],
}

ADDITIONAL_SOURCE_URLS_BY_BENCHMARK = {
    "usnco2025_part2": [
        "https://www.acs.org/content/dam/acsorg/education/students/highschool/olympiad/pastexams/2025-usnco-national-exam-part-ii.pdf",
    ],
}


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def existing_local_sources(row: dict[str, Any]) -> list[str]:
    sources: set[str] = set(str(item) for item in row.get("source_files") or [] if str(item).strip())
    for ref in row.get("source_refs") or []:
        ref_text = str(ref)
        if ref_text.startswith(("http://", "https://")):
            continue
        if ref_text and Path(ref_text).exists():
            sources.add(ref_text)
    for path in ADDITIONAL_SOURCE_FILES_BY_BENCHMARK.get(str(row.get("benchmark_id")), []):
        if Path(path).exists():
            sources.add(path)
    return sorted(sources)


def source_urls(row: dict[str, Any]) -> list[str]:
    urls: set[str] = set()
    for item in row.get("source_urls") or []:
        item_text = str(item).strip()
        if not item_text:
            continue
        if any(marker in item_text for marker in ["myqcloud.com", "cos.ap-", "science-data-360"]):
            continue
        urls.add(item_text)
    urls.update(ADDITIONAL_SOURCE_URLS_BY_BENCHMARK.get(str(row.get("benchmark_id")), []))
    return sorted(urls)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    items_path = args.dataset_dir / "benchmark_items.jsonl"
    rows = iter_jsonl(items_path)
    if not rows:
        raise SystemExit(f"No rows found in {items_path}")

    now = datetime.now().isoformat(timespec="seconds")
    for row in rows:
        benchmark_id = str(row.get("benchmark_id"))
        row["source_urls"] = source_urls(row)
        row["source_files"] = existing_local_sources(row)
        row.setdefault("staticization_audit_notes", [])
        row.setdefault("staticization_client_meta", {"metadata_finalized_without_api_call": True})
        row.setdefault("original_prompt", row.get("question_text_staticized") or row.get("question_text") or "")
        row["release_date"] = RELEASE_DATE_BY_BENCHMARK.get(benchmark_id, row.get("release_date"))
        row["official_rubric_status"] = row.get("official_rubric_status") or RUBRIC_STATUS_BY_BENCHMARK.get(
            benchmark_id,
            "unspecified",
        )

    write_jsonl(items_path, rows)

    summaries: dict[str, dict[str, Any]] = {}
    for row in rows:
        benchmark_id = str(row.get("benchmark_id"))
        summary = summaries.setdefault(
            benchmark_id,
            {
                "benchmark_id": benchmark_id,
                "discipline": row.get("discipline"),
                "release_date": row.get("release_date"),
                "source_quality": row.get("source_quality"),
                "official_rubric_status_values": set(),
                "item_count": 0,
                "sequence_ids": set(),
                "max_score_sum": 0.0,
                "report_scale": float(row.get("report_scale") or 0),
                "staticization_errors": 0,
                "incomplete_rows_with_notes": 0,
                "source_urls": set(),
                "source_files": set(),
            },
        )
        summary["item_count"] += 1
        summary["sequence_ids"].add(str(row.get("sequence_id")))
        summary["max_score_sum"] += float(row.get("max_score") or 0)
        summary["official_rubric_status_values"].add(str(row.get("official_rubric_status")))
        summary["staticization_errors"] += 1 if row.get("staticization_error") else 0
        issues_value = (row.get("staticization_completeness") or {}).get("issues")
        if isinstance(issues_value, list):
            issues = " ".join(str(item).strip() for item in issues_value if str(item).strip())
        else:
            issues = str(issues_value or "").strip()
        summary["incomplete_rows_with_notes"] += 1 if issues else 0
        for url in row.get("source_urls") or []:
            summary["source_urls"].add(str(url))
        for source_file in row.get("source_files") or []:
            summary["source_files"].add(str(source_file))

    summary_rows: list[dict[str, Any]] = []
    total_staticization_cost = 0.0
    old_summary_path = args.dataset_dir / "benchmark_summary.jsonl"
    old_costs: dict[str, float] = {}
    if old_summary_path.exists():
        for row in iter_jsonl(old_summary_path):
            old_costs[str(row.get("benchmark_id"))] = float(row.get("staticization_cost_cny") or 0)
    for benchmark_id in sorted(summaries):
        summary = summaries[benchmark_id]
        cost = old_costs.get(benchmark_id, 0.0)
        total_staticization_cost += cost
        summary_rows.append(
            {
                "benchmark_id": benchmark_id,
                "discipline": summary["discipline"],
                "release_date": summary["release_date"],
                "source_quality": summary["source_quality"],
                "official_rubric_status_values": sorted(summary["official_rubric_status_values"]),
                "item_count": summary["item_count"],
                "sequence_count": len(summary["sequence_ids"]),
                "max_score_sum": summary["max_score_sum"],
                "report_scale": summary["report_scale"],
                "staticization_cost_cny": cost,
                "staticization_errors": summary["staticization_errors"],
                "incomplete_rows_with_notes": summary["incomplete_rows_with_notes"],
                "source_urls": sorted(summary["source_urls"]),
                "source_files": sorted(summary["source_files"]),
            }
        )
    write_jsonl(args.dataset_dir / "benchmark_summary.jsonl", summary_rows)

    info_path = args.dataset_dir / "dataset_info.json"
    info = {}
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
    info.update(
        {
            "metadata_finalized_at": now,
            "row_count": len(rows),
            "benchmark_count": len(summaries),
            "benchmark_ids": sorted(summaries),
            "total_staticization_cost_cny": total_staticization_cost,
            "known_quality_boundary": (
                "Validator-clean means schema/URL-image-payload invariants pass. "
                "It is not a semantic guarantee that every source/rubric is perfect."
            ),
        }
    )
    info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "benchmarks": len(summaries), "dataset_dir": str(args.dataset_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
