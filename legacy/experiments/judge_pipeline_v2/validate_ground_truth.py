#!/usr/bin/env python3
"""Validate expert ground-truth files against official item max scores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import science_arena_judge


ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def official_icho_max() -> dict[str, float]:
    items = load(ROOT / "data/science_arena/icho2025/fix_ICHO-2025-latex.json")
    out = {}
    for question in items:
        qid = science_arena_judge.safe_join(question.get("question_id", [])).strip()
        for sub in question.get("detailed_question", []):
            sid = science_arena_judge.safe_join(sub.get("sub_question_id", [])).strip()
            score = float(science_arena_judge.safe_join(sub.get("score", ["0"])).replace("pt", "").strip() or 0)
            out[f"ICHO-{qid}-{sid}"] = score
    return out


def validate_icho() -> list[dict[str, Any]]:
    gt = load(ROOT / "experiments/judge_pipeline_v2/ground_truth/icho2025_expert_ground_truth.json")["models"]
    max_by_case = official_icho_max()
    issues = []
    for model, data in gt.items():
        for item in data.get("items", []):
            case_id = f"ICHO-{item.get('question_id')}-{item.get('sub_question_id')}"
            score = item.get("human_score_raw")
            max_score = max_by_case.get(case_id)
            if score is None or max_score is None:
                continue
            score = float(score)
            if score < -1e-9 or score - max_score > 1e-9:
                issues.append(
                    {
                        "model": model,
                        "case_id": case_id,
                        "human_score_raw": score,
                        "official_max_score": max_score,
                        "source_file": item.get("source_file"),
                        "source_index": item.get("source_index"),
                        "human_comments": item.get("human_comments", ""),
                    }
                )
    return issues


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam", choices=["icho"], default="icho")
    args = parser.parse_args()
    if args.exam == "icho":
        issues = validate_icho()
    else:
        issues = []
    out = {"exam": args.exam, "issue_count": len(issues), "issues": issues}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

