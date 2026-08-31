#!/usr/bin/env python3
"""Collect candidate benchmark smoke summaries into one table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_files", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in args.summary_files:
        if not path.exists():
            rows.append({"summary_path": str(path), "missing": True})
            continue
        data = load_json(path)
        parsed = data.get("parsed_judge") or {}
        rows.append(
            {
                "summary_path": str(path.relative_to(ROOT) if path.resolve().is_relative_to(ROOT) else path),
                "benchmark_id": data.get("benchmark_id"),
                "problem_id": data.get("problem_id"),
                "model": data.get("model"),
                "judge_model": data.get("judge_model"),
                "score": parsed.get("total_score"),
                "max_score": parsed.get("max_score"),
                "pre_multimodal_mode": data.get("pre_multimodal_mode"),
                "available_image_count": data.get("available_image_count"),
                "sent_image_count": data.get("sent_image_count"),
                "generation_finish_reason": data.get("generation_finish_reason"),
                "generation_error": data.get("generation_error"),
                "judge_finish_reason": data.get("judge_finish_reason"),
                "generation_prompt_tokens": (data.get("generation_usage") or {}).get("prompt_tokens"),
                "generation_completion_tokens": (data.get("generation_usage") or {}).get("completion_tokens"),
                "judge_prompt_tokens": (data.get("judge_usage") or {}).get("prompt_tokens"),
                "judge_completion_tokens": (data.get("judge_usage") or {}).get("completion_tokens"),
                "cost_cny": data.get("total_cost_cny"),
            }
        )
    payload = {
        "rows": rows,
        "total_cost_cny": sum(float(row.get("cost_cny") or 0) for row in rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
