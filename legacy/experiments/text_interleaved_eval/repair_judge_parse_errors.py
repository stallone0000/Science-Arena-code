#!/usr/bin/env python3
"""Reparse stored raw judge text for text-interleaved runs.

This does not call any API and does not alter candidate answers.  It only fixes
rows where the judge response was present but the local JSON parser was too
strict, usually because the evidence field contained LaTeX backslashes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.text_interleaved_eval.run_text_interleaved_eval import extract_json_object, iter_jsonl, write_jsonl  # noqa: E402


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def repair_file(path: Path) -> dict[str, Any]:
    rows = iter_jsonl(path)
    changed = 0
    repaired = 0
    for row in rows:
        judge_result = row.get("judge_result") or {}
        raw_text = judge_result.get("raw_text")
        if not judge_result.get("parse_error") or not raw_text:
            continue
        parsed = extract_json_object(str(raw_text))
        if parsed.get("parse_error"):
            continue
        row["judge_result"] = parsed
        row["score"] = parsed.get("total_score")
        row["max_score"] = parsed.get("max_score") or row.get("max_score")
        row["judge_error"] = ""
        changed += 1
        repaired += 1
    if changed:
        write_jsonl(path, rows)

    model_dir = path.parent
    summary_path = model_dir / "benchmark_summary.json"
    if changed and summary_path.exists():
        summaries = load_json(summary_path, [])
        by_benchmark: dict[str, dict[str, Any]] = {}
        for row in rows:
            bench = str(row.get("benchmark_id"))
            summary = by_benchmark.setdefault(
                bench,
                {
                    "benchmark_id": bench,
                    "model": row.get("model"),
                    "score": 0.0,
                    "max_score": 0.0,
                    "reported_score": 0.0,
                    "report_scale": 0.0,
                    "turn_count": 0,
                    "generation_error_count": 0,
                    "judge_error_count": 0,
                    "cost_cny": 0.0,
                },
            )
            summary["score"] += float(row.get("score") or 0)
            summary["max_score"] += float(row.get("max_score") or 0)
            summary["turn_count"] += 1
            summary["cost_cny"] += float(row.get("total_cost_cny") or 0)
            summary["generation_error_count"] += 1 if row.get("generation_error") else 0
            summary["judge_error_count"] += 1 if row.get("judge_error") or (row.get("judge_result") or {}).get("parse_error") else 0
        old_report_scales = {str(item.get("benchmark_id")): float(item.get("report_scale") or 0) for item in summaries}
        for bench, summary in by_benchmark.items():
            summary["report_scale"] = old_report_scales.get(bench, summary["max_score"])
            if summary["max_score"] and summary["report_scale"]:
                summary["reported_score"] = summary["score"] / summary["max_score"] * summary["report_scale"]
        save_json(summary_path, sorted(by_benchmark.values(), key=lambda item: item["benchmark_id"]))
    return {"path": str(path), "rows": len(rows), "repaired": repaired}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
    records = []
    for path in args.paths:
        if path.is_dir():
            for child in path.glob("*/turn_results.jsonl"):
                records.append(repair_file(child))
        else:
            records.append(repair_file(path))
    print(json.dumps(records, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
