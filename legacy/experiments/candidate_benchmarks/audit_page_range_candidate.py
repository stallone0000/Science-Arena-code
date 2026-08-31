#!/usr/bin/env python3
"""Audit page-range candidate benchmark completeness.

This is a local deterministic audit. It does not call an LLM; it checks that a
page-range candidate config resolves to text pages, renders/images, and point
totals visible in question/solution text. The output is meant to guide manual
or LLM-assisted per-problem verification before final benchmark normalization.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def selected_pages(pages_json: dict[str, Any], page_numbers: list[int]) -> list[dict[str, Any]]:
    wanted = {int(page) for page in page_numbers}
    return [page for page in pages_json.get("pages", []) if int(page.get("page", -1)) in wanted]


def text_blob(pages: list[dict[str, Any]]) -> str:
    return "\n".join(str(page.get("text") or "") for page in pages)


def count_page_assets(pages: list[dict[str, Any]]) -> dict[str, int]:
    return {
        "embedded_images": sum(int(page.get("image_count") or 0) for page in pages),
        "page_renders": sum(1 for page in pages if page.get("page_render")),
    }


def points_from_text(text: str) -> list[float]:
    values: list[float] = []
    patterns = [
        r"\((\d+(?:\.\d+)?)\s*points?\)",
        r"\[(\d+(?:\.\d+)?)\s*marks?\]",
        r"TOTAL MARKS:\s*(\d+(?:\.\d+)?)",
        r"Maximum Marks\s*[:\-]?\s*(\d+(?:\.\d+)?)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text, flags=re.I):
            try:
                values.append(float(match.group(1)))
            except ValueError:
                continue
    return values


def problem_total_points_from_text(text: str) -> list[dict[str, Any]]:
    normalized = re.sub(r"[ \t]+", " ", text)
    normalized = re.sub(r"\n+", "\n", normalized)
    results: list[dict[str, Any]] = []
    seen: set[tuple[str, float]] = set()
    patterns = [
        r"(?ms)^\s*([1-9]\d*)\.\s+(.{0,220}?)\((\d+(?:\.\d+)?)\s*points?\)",
        r"(?ms)^\s*([1-9]\d*)\.\s+(.{0,220}?)\[(\d+(?:\.\d+)?)\s*marks?\]",
        r"(?ms)^Problem\s+(\d+)\b(.{0,220}?)\(TOTAL MARKS:\s*(\d+(?:\.\d+)?)\)",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, normalized, flags=re.I):
            problem_no = match.group(1)
            points = float(match.group(3))
            key = (problem_no, points)
            if key in seen:
                continue
            seen.add(key)
            title = " ".join(match.group(2).split())
            results.append({"problem_no": problem_no, "title_fragment": title[:140], "points": points})
    return results


def numbered_problem_headers(text: str) -> list[str]:
    headers: list[str] = []
    for line in text.splitlines():
        stripped = " ".join(line.split())
        if re.match(r"^\d+\.\s+\S", stripped) or re.match(r"^Problem\s+\d+", stripped, flags=re.I):
            headers.append(stripped)
    return headers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(""))
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config = load_json(config_path)
    question_pages_json_path = ROOT / config["source_files"]["question_pages_json"]
    solution_pages_json_path = ROOT / config["source_files"]["solution_pages_json"]
    question_pages = selected_pages(load_json(question_pages_json_path), config.get("question_pages", []))
    solution_pages = selected_pages(load_json(solution_pages_json_path), config.get("solution_pages", []))
    question_text = text_blob(question_pages)
    solution_text = text_blob(solution_pages)
    q_points = points_from_text(question_text)
    s_points = points_from_text(solution_text)
    problem_totals = problem_total_points_from_text(question_text)
    declared_total = float(config.get("total_points") or 0)
    observed_problem_total_sum = sum(row["points"] for row in problem_totals)

    result = {
        "config": str(config_path.relative_to(ROOT) if config_path.is_relative_to(ROOT) else config_path),
        "benchmark_id": config.get("benchmark_id"),
        "problem_id": config.get("problem_id"),
        "declared_total_points": declared_total,
        "question_pages_selected": [page.get("page") for page in question_pages],
        "solution_pages_selected": [page.get("page") for page in solution_pages],
        "question_page_count": len(question_pages),
        "solution_page_count": len(solution_pages),
        "question_text_chars": len(question_text),
        "solution_text_chars": len(solution_text),
        "question_assets": count_page_assets(question_pages),
        "solution_assets": count_page_assets(solution_pages),
        "problem_total_points": problem_totals,
        "problem_total_point_sum": observed_problem_total_sum,
        "question_point_values": q_points,
        "question_point_value_sum_including_subparts": sum(q_points),
        "solution_point_values_sample": s_points[:80],
        "solution_point_value_count": len(s_points),
        "numbered_problem_headers": numbered_problem_headers(question_text),
        "checks": {
            "has_question_text": bool(question_text.strip()),
            "has_solution_text": bool(solution_text.strip()),
            "has_question_visual_payload": count_page_assets(question_pages)["embedded_images"] > 0
            or count_page_assets(question_pages)["page_renders"] > 0,
            "declared_total_matches_problem_totals": abs(observed_problem_total_sum - declared_total) < 1e-6,
            "has_solution_point_mentions": bool(s_points),
        },
        "manual_review_required": [
            "Confirm every figure in page renders is legible and assigned to the correct problem/subquestion.",
            "Confirm solution point mentions are official rubric criteria, not only incidental point references.",
            "Normalize the page range into item-level JSON before treating this as a final benchmark.",
        ],
    }
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
