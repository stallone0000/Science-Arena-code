#!/usr/bin/env python3
"""Convert the EuPhO 2025 structured draft into Science Arena item JSON."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "data/science_arena/candidate_benchmarks/structured/eupho2025/eupho2025_theory_draft.json"
OUTPUT = ROOT / "data/science_arena/eupho2025/fix_EuPhO-2025-latex.json"
QUESTION_PAGE_RENDER = (
    "data/science_arena/candidate_benchmarks/extracted/rendered_pages_v1/"
    "EuPhO_2025_Theory_ENG/page_renders/page_001.png"
)


def rubric_text(subquestion_rubric: dict[str, Any]) -> str:
    lines = [f"Total: {subquestion_rubric.get('total_points')} pt"]
    for criterion in subquestion_rubric.get("criteria", []):
        line = f"{criterion.get('code')}: {criterion.get('points')} pt - {criterion.get('text')}"
        partials = criterion.get("partials") or []
        deductions = criterion.get("deductions") or []
        if partials:
            line += " Partials: " + "; ".join(str(item) for item in partials)
        if deductions:
            line += " Deductions: " + "; ".join(str(item) for item in deductions)
        lines.append(line)
    return "\n".join(lines)


def build_problem(problem: dict[str, Any]) -> dict[str, Any]:
    rubrics = {
        str(item.get("id")): item
        for item in problem.get("marking_scheme", {}).get("subquestions", [])
    }
    assets = [asset.get("path") for asset in problem.get("assets", []) if asset.get("path")]
    # The one-page EuPhO theory PDF contains vector figures for T2/T3 that are
    # only available through a full-page render. Include it for every problem
    # until final crops are checked.
    if QUESTION_PAGE_RENDER not in assets:
        assets.append(QUESTION_PAGE_RENDER)
    detailed = []
    for subquestion in problem.get("subquestions", []):
        sid = str(subquestion.get("id"))
        answer_summary = problem.get("answer_key_summary", {}).get(sid, "")
        detailed.append(
            {
                "sub_question_id": [sid],
                "sub_question_instructions": problem.get("question_context", []),
                "sub_question_content": [subquestion.get("prompt", "")],
                "sub_question_instructions_fig": assets,
                "sub_question_solution": [answer_summary],
                "sub_question_solution_fig": [],
                "score": [str(subquestion.get("points"))],
                "scoring_rubric": [rubric_text(rubrics.get(sid, {"total_points": subquestion.get("points"), "criteria": []}))],
            }
        )
    return {
        "question_id": [problem.get("id")],
        "overall_instructions": [
            f"{problem.get('title')} ({problem.get('total_points')} points)",
            *problem.get("question_context", []),
        ],
        "overall_instructions_fig": assets,
        "detailed_question": detailed,
        "metadata": {
            "source": "European Physics Olympiad 2025 Theory structured draft",
            "status": "candidate_formal_json_needs_final_figure_crop_review",
            "known_limitations": problem.get("todo", []),
        },
    }


def main() -> None:
    draft = json.loads(SOURCE.read_text(encoding="utf-8"))
    problems = [build_problem(problem) for problem in draft.get("problems", [])]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(problems, ensure_ascii=False, indent=2), encoding="utf-8")
    total = sum(float(sq["score"][0]) for problem in problems for sq in problem["detailed_question"])
    print(json.dumps({"output": str(OUTPUT.relative_to(ROOT)), "problems": len(problems), "total_points": total}, indent=2))


if __name__ == "__main__":
    main()
