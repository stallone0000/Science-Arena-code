#!/usr/bin/env python3
"""Reparse saved text-interleaved turn results with the latest JSON parser."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from run_text_interleaved_eval import extract_json_object, write_jsonl


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("turn_results", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    rows = iter_jsonl(args.turn_results)
    reparsed = 0
    for row in rows:
        result = row.get("judge_result") or {}
        raw = result.get("raw_text")
        if result.get("parse_error") and raw:
            parsed = extract_json_object(raw)
            if not parsed.get("parse_error"):
                row["judge_result"] = parsed
                row["score"] = parsed.get("total_score")
                row["max_score"] = parsed.get("max_score") or row.get("max_score")
                reparsed += 1
    output = args.output or args.turn_results.with_name(args.turn_results.stem + ".reparsed.jsonl")
    write_jsonl(output, rows)
    print(json.dumps({"input": str(args.turn_results), "output": str(output), "reparsed": reparsed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
