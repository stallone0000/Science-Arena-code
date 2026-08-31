#!/usr/bin/env python3
"""Summarize latest-model batch output into model-by-exam scores and costs."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path, help="latest_eval_summary.json from run_latest_model_batch.py")
    parser.add_argument("--out", type=Path, default=Path(""))
    args = parser.parse_args()

    data = load_json(args.summary)
    grouped: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "model": "",
            "exam": "",
            "sections": {},
            "score": 0.0,
            "complete": True,
            "generation_cost_cny": 0.0,
            "judge_cost_cny": 0.0,
            "generation_prompt_tokens": 0.0,
            "generation_completion_tokens": 0.0,
            "judge_prompt_tokens": 0.0,
            "judge_completion_tokens": 0.0,
            "generation_errors": 0,
            "judge_errors": 0,
            "items": 0,
            "attempted_items": 0,
            "statuses": [],
        }
    )

    for result in data.get("results", []):
        model = result.get("model", "")
        exam = result.get("exam", "")
        section = result.get("section", "") or "all"
        key = (model, exam)
        row = grouped[key]
        row["model"] = model
        row["exam"] = exam
        generation = result.get("generation", {}) or {}
        judge = result.get("judge", {}) or {}
        score = float(judge.get("score", 0) or 0)
        row["sections"][section] = {
            "score": score,
            "generation_status": result.get("generation_status", ""),
            "judge_status": result.get("judge_status", ""),
            "generation_cost_cny": float(generation.get("cost_cny", 0) or 0),
            "judge_cost_cny": float(judge.get("cost_cny", 0) or 0),
            "items": int(judge.get("items", 0) or 0),
            "attempted_items": int(judge.get("attempted_items", 0) or 0),
            "errors": int(judge.get("errors", 0) or 0),
        }
        row["score"] += score
        row["generation_cost_cny"] += float(generation.get("cost_cny", 0) or 0)
        row["judge_cost_cny"] += float(judge.get("cost_cny", 0) or 0)
        row["generation_prompt_tokens"] += float(generation.get("prompt_tokens", 0) or 0)
        row["generation_completion_tokens"] += float(generation.get("completion_tokens", 0) or 0)
        row["judge_prompt_tokens"] += float(judge.get("prompt_tokens", 0) or 0)
        row["judge_completion_tokens"] += float(judge.get("completion_tokens", 0) or 0)
        row["generation_errors"] += int(generation.get("errors", 0) or 0)
        row["judge_errors"] += int(judge.get("errors", 0) or 0)
        row["items"] += int(judge.get("items", 0) or 0)
        row["attempted_items"] += int(judge.get("attempted_items", 0) or 0)
        row["statuses"].append(
            {
                "section": section,
                "generation": result.get("generation_status", ""),
                "judge": result.get("judge_status", ""),
            }
        )

    rows = []
    for (_, exam), row in sorted(grouped.items()):
        expected_sections = {"S1", "S2", "S3"} if exam == "ipho" else {"all"}
        present_sections = set(row["sections"])
        row["complete"] = (
            expected_sections.issubset(present_sections)
            and all(item.get("generation") in {"ok", "skipped_existing"} for item in row["statuses"])
            and all(item.get("judge") in {"ok", "skipped_existing"} for item in row["statuses"])
            and row["judge_errors"] == 0
        )
        row["total_cost_cny"] = row["generation_cost_cny"] + row["judge_cost_cny"]
        rows.append(row)

    report = {
        "source_summary": str(args.summary),
        "skipped": data.get("skipped", []),
        "rows": rows,
    }
    out = args.out or args.summary.with_name("latest_score_table.json")
    save_json(out, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
