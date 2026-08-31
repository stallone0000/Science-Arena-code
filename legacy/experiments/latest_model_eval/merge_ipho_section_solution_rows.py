#!/usr/bin/env python3
"""Merge replacement rows into an IPhO section solutions.json.

This is intended for small fallback probes where one or more item-level
generations failed in an otherwise complete section. It is file-only and does
not call any API.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_rows(path: Path) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise TypeError(f"{path} must contain a JSON list")
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(data):
        if not isinstance(row, dict):
            raise TypeError(f"{path} row {idx} is not an object")
        if "item_index" not in row:
            raise KeyError(f"{path} row {idx} has no item_index")
        rows.append(row)
    return rows


def has_error(row: dict[str, Any]) -> bool:
    if row.get("error"):
        return True
    answer = row.get("answer")
    output = row.get("llm_output")
    return not bool((answer or output or "").strip())


def usage_totals(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    prompt_tokens = 0.0
    completion_tokens = 0.0
    total_tokens = 0.0
    errors = 0
    for row in rows:
        if has_error(row):
            errors += 1
        usage = row.get("usage") or {}
        if not isinstance(usage, dict):
            usage = {}
        prompt = float(usage.get("prompt_tokens") or 0)
        completion = float(usage.get("completion_tokens") or 0)
        total = float(usage.get("total_tokens") or (prompt + completion))
        prompt_tokens += prompt
        completion_tokens += completion
        total_tokens += total
    return {
        "units": len(rows),
        "errors": errors,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, type=Path, help="Base section solutions.json.")
    parser.add_argument("--patch", required=True, type=Path, action="append", help="Patch/probe section solutions.json.")
    parser.add_argument("--output_dir", required=True, type=Path, help="Directory to write merged solutions.json.")
    parser.add_argument(
        "--replace_error_only",
        action="store_true",
        help="Only replace a base row if the base row has an error or empty answer.",
    )
    parser.add_argument(
        "--allow_error_patch",
        action="store_true",
        help="Allow patch rows that still contain errors. By default they are ignored.",
    )
    args = parser.parse_args()

    base_rows = load_rows(args.base)
    patch_rows: list[dict[str, Any]] = []
    for patch_path in args.patch:
        patch_rows.extend(load_rows(patch_path))
    merged_by_item = {int(row["item_index"]): dict(row) for row in base_rows}
    replaced: list[int] = []
    skipped: list[dict[str, Any]] = []

    for row in patch_rows:
        item_index = int(row["item_index"])
        if has_error(row) and not args.allow_error_patch:
            skipped.append({"item_index": item_index, "reason": "patch row has error or empty answer"})
            continue
        if args.replace_error_only and item_index in merged_by_item and not has_error(merged_by_item[item_index]):
            skipped.append({"item_index": item_index, "reason": "base row is already non-error"})
            continue
        merged_by_item[item_index] = dict(row)
        replaced.append(item_index)

    merged_rows = [merged_by_item[key] for key in sorted(merged_by_item)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    solutions_path = args.output_dir / "solutions.json"
    summary_path = args.output_dir / "summary.json"
    manifest_path = args.output_dir / "merge_manifest.json"
    solutions_path.write_text(json.dumps(merged_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "generation": {
                    "method": "interleaved",
                    "exam": "ipho",
                    "unit_count": len(merged_rows),
                    "solutions_path": str(solutions_path),
                    **usage_totals(merged_rows),
                    "merged_at": datetime.now().isoformat(timespec="seconds"),
                }
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_path.write_text(
        json.dumps(
            {
                "base": str(args.base),
                "patch": [str(path) for path in args.patch],
                "output_dir": str(args.output_dir),
                "replaced_item_indices": replaced,
                "skipped": skipped,
                "base_rows": len(base_rows),
                "patch_rows": len(patch_rows),
                "merged_rows": len(merged_rows),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[MERGE] wrote {solutions_path}")
    print(f"[MERGE] replaced items: {replaced}")
    if skipped:
        print(f"[MERGE] skipped: {skipped}")


if __name__ == "__main__":
    main()
