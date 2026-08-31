#!/usr/bin/env python3
"""Compare a judge run aggregate against canonical expert ground truth."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ITEMS = {"icho": 65, "ipho": 40}


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def build_item_ground_truth(gt: dict, exam: str) -> dict[tuple[str, str], float]:
    by_case: dict[tuple[str, str], float] = {}
    for model, data in gt.items():
        if exam == "icho":
            for item in data.get("items", []):
                qid = str(item.get("question_id", "")).strip()
                sid = str(item.get("sub_question_id", "")).strip()
                if qid and sid and item.get("human_score_raw") is not None:
                    by_case[(model, f"ICHO-{qid}-{sid}")] = float(item["human_score_raw"])
        else:
            for section, section_data in data.get("sections", {}).items():
                for idx, item in enumerate(section_data.get("scores", []), start=1):
                    if item.get("human_score") is not None:
                        by_case[(model, f"IPHO-{section}-{idx}")] = float(item["human_score"])
    return by_case


def item_level_summary(run_dir: Path, gt: dict, exam: str) -> dict:
    result_file = run_dir / "judge_results.json"
    if not result_file.exists():
        return {"available": False}
    item_gt = build_item_ground_truth(gt, exam)
    rows = load(result_file)
    by_model: dict[str, list[dict]] = {}
    skipped_errors = 0
    for row in rows:
        if row.get("error"):
            skipped_errors += 1
            continue
        key = (row.get("model_under_test"), row.get("case_id"))
        if key not in item_gt:
            continue
        err = float(row.get("score", 0.0)) - item_gt[key]
        by_model.setdefault(row["model_under_test"], []).append(
            {
                "case_id": row.get("case_id"),
                "expert_raw": item_gt[key],
                "judge_raw": float(row.get("score", 0.0)),
                "signed_error": err,
                "absolute_error": abs(err),
            }
        )
    model_rows = []
    all_abs = []
    all_signed = []
    for model, items in sorted(by_model.items()):
        abs_errors = [row["absolute_error"] for row in items]
        signed_errors = [row["signed_error"] for row in items]
        all_abs.extend(abs_errors)
        all_signed.extend(signed_errors)
        model_rows.append(
            {
                "model": model,
                "comparable_items": len(items),
                "raw_mae": sum(abs_errors) / len(abs_errors) if abs_errors else None,
                "raw_mean_signed_error": sum(signed_errors) / len(signed_errors) if signed_errors else None,
                "raw_exact_matches": sum(1 for row in items if row["absolute_error"] == 0),
                "examples": items[:5],
            }
        )
    return {
        "available": True,
        "rows": model_rows,
        "overall_raw_mae": sum(all_abs) / len(all_abs) if all_abs else None,
        "overall_raw_mean_signed_error": sum(all_signed) / len(all_signed) if all_signed else None,
        "comparable_items": len(all_abs),
        "skipped_error_items": skipped_errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir")
    parser.add_argument("--exam", choices=["icho", "ipho"], required=True)
    args = parser.parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    agg = load(run_dir / "aggregate.json")
    gt_file = ROOT / "experiments/judge_pipeline_v2/ground_truth" / (
        "icho2025_expert_ground_truth.json" if args.exam == "icho" else "ipho2025_expert_ground_truth.json"
    )
    gt = load(gt_file)["models"]
    rows = []
    for model, data in agg.items():
        if model not in gt:
            continue
        expert = gt[model].get("weighted_total_60", gt[model].get("total"))
        judge = data["score"]
        complete = data.get("items", 0) == EXPECTED_ITEMS[args.exam] and data.get("errors", 0) == 0
        signed_error = judge - expert if complete else None
        absolute_error = abs(judge - expert) if complete else None
        rows.append(
            {
                "model": model,
                "expert": expert,
                "judge": judge,
                "is_complete_run": complete,
                "signed_error": signed_error,
                "absolute_error": absolute_error,
                "items": data.get("items", 0),
                "cost_cny": data.get("cost_cny", 0),
                "prompt_tokens": data.get("prompt_tokens", 0),
                "completion_tokens": data.get("completion_tokens", 0),
                "errors": data.get("errors", 0),
            }
        )
    complete_rows = [row for row in rows if row["absolute_error"] is not None]
    item_summary = item_level_summary(run_dir, gt, args.exam)
    summary = {
        "run_dir": str(run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir),
        "exam": args.exam,
        "rows": rows,
        "mean_absolute_error": sum(r["absolute_error"] for r in complete_rows) / len(complete_rows) if complete_rows else None,
        "mean_signed_error": sum(r["signed_error"] for r in complete_rows) / len(complete_rows) if complete_rows else None,
        "item_level": item_summary,
        "total_cost_cny": sum(r["cost_cny"] for r in rows),
        "total_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
        "total_completion_tokens": sum(r["completion_tokens"] for r in rows),
    }
    out = run_dir / "comparison_to_expert.json"
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
