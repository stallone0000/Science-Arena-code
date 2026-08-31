#!/usr/bin/env python3
"""Summarize largest item-level judge errors against expert ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
ICHO_PROPORTION = [0, 6, 6.5, 7.5, 6, 6.5, 6, 7, 7.5, 7]
ICHO_MAX_POINT = [0, 34, 67, 111, 40, 16, 22, 22, 35, 38]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def trim(text: str, limit: int = 500) -> str:
    text = " ".join(str(text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def build_gt(exam: str) -> dict[tuple[str, str], dict[str, Any]]:
    name = "icho2025_expert_ground_truth.json" if exam == "icho" else "ipho2025_expert_ground_truth.json"
    data = load(ROOT / "experiments/judge_pipeline_v2/ground_truth" / name)["models"]
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for model, rec in data.items():
        if exam == "icho":
            for item in rec.get("items", []):
                case_id = f"ICHO-{item.get('question_id')}-{item.get('sub_question_id')}"
                out[(model, case_id)] = item
        else:
            for section, section_data in rec.get("sections", {}).items():
                for index, item in enumerate(section_data.get("scores", []), start=1):
                    case_id = f"IPHO-{section}-{index}"
                    out[(model, case_id)] = item
    return out


def expert_score(item: dict[str, Any], exam: str) -> float | None:
    key = "human_score_raw" if exam == "icho" else "human_score"
    value = item.get(key)
    return None if value is None else float(value)


def expert_comment(item: dict[str, Any], exam: str) -> str:
    if exam == "icho":
        return str(item.get("human_comments", ""))
    return str(item.get("expert_comment") or item.get("human_comment") or item.get("comments") or "")


def weighted_error(row: dict[str, Any], signed_raw: float, exam: str) -> float:
    if exam != "icho":
        return signed_raw
    qid = int(float(row.get("question_id")))
    if qid >= len(ICHO_MAX_POINT) or not ICHO_MAX_POINT[qid]:
        return 0.0
    return signed_raw / ICHO_MAX_POINT[qid] * ICHO_PROPORTION[qid]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--exam", choices=["icho", "ipho"], required=True)
    parser.add_argument("--top", type=int, default=30)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    rows = load(run_dir / "judge_results.json")
    gt = build_gt(args.exam)
    diffs = []
    for row in rows:
        key = (row.get("model_under_test"), row.get("case_id"))
        item = gt.get(key)
        if not item:
            continue
        exp = expert_score(item, args.exam)
        if exp is None:
            continue
        judge = float(row.get("score", 0.0))
        signed = judge - exp
        parsed = row.get("parsed") or {}
        diffs.append(
            {
                "model": key[0],
                "case_id": key[1],
                "expert_raw": exp,
                "judge_raw": judge,
                "signed_raw": signed,
                "abs_raw": abs(signed),
                "weighted_signed": weighted_error(row, signed, args.exam),
                "weighted_abs": abs(weighted_error(row, signed, args.exam)),
                "expert_comment": trim(expert_comment(item, args.exam), 700),
                "judge_summary": trim(parsed.get("credit_summary") or row.get("raw_output", ""), 700),
                "deductions": parsed.get("deductions", []),
            }
        )

    diffs.sort(key=lambda x: (x["weighted_abs"], x["abs_raw"]), reverse=True)
    out = {
        "run_dir": str(run_dir),
        "exam": args.exam,
        "top": args.top,
        "largest_errors": diffs[: args.top],
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

