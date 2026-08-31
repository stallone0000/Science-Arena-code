#!/usr/bin/env python3
"""Collect Gemini judge calibration against human expert scores."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNS = [
    ("google/gemini-3.5-flash", "ipho2025", ROOT / "experiments/judge_pipeline_v2/runs/merged_ipho_gemini35_v6/comparison_to_expert.json"),
    ("google/gemini-3.5-flash", "icho2025", ROOT / "experiments/judge_pipeline_v2/runs/merged_icho_gemini35_v8_reparsed/comparison_to_expert.json"),
    ("google/google/gemini-3.1-pro-preview", "ipho2025", ROOT / "experiments/judge_pipeline_v2/runs/merged_ipho_gemini31pro_strictnotes_v2_reparsed/comparison_to_expert.json"),
    ("google/google/gemini-3.1-pro-preview", "icho2025", ROOT / "experiments/judge_pipeline_v2/runs/fullcalib_icho_gemini31pro_strictnotes_v4_reparsed/comparison_to_expert.json"),
]


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    records: list[dict[str, Any]] = []
    for judge_model, benchmark_id, path in RUNS:
        data = json.loads(path.read_text(encoding="utf-8"))
        for row in data["rows"]:
            records.append(
                {
                    "judge_model": judge_model,
                    "benchmark_id": benchmark_id,
                    "answer_model": row["model"],
                    "human_expert_score": row["expert"],
                    "judge_score": row["judge"],
                    "signed_error": row["signed_error"],
                    "absolute_error": row["absolute_error"],
                    "items": row["items"],
                    "errors": row["errors"],
                    "run_dir": data["run_dir"],
                }
            )
        records.append(
            {
                "judge_model": judge_model,
                "benchmark_id": benchmark_id,
                "answer_model": "__MEAN__",
                "human_expert_score": None,
                "judge_score": None,
                "signed_error": data["mean_signed_error"],
                "absolute_error": data["mean_absolute_error"],
                "items": None,
                "errors": 0,
                "run_dir": data["run_dir"],
            }
        )

    out_dir = ROOT / "artifacts/runs/judge_calibration_gemini_20260524"
    save_json(out_dir / "records.json", records)
    lines = [
        "| Judge model | Benchmark | Answer model | Human | Judge | Signed err | Abs err |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for record in records:
        human = "" if record["human_expert_score"] is None else f"{record['human_expert_score']:.4f}"
        judge = "" if record["judge_score"] is None else f"{record['judge_score']:.4f}"
        lines.append(
            "| {judge_model} | {benchmark_id} | {answer_model} | {human} | {judge} | {signed_error:.4f} | {absolute_error:.4f} |".format(
                human=human,
                judge=judge,
                **record,
            )
        )
    (out_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(out_dir), "records": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
