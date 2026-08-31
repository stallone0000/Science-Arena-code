#!/usr/bin/env python3
"""Merge selected IBO model rows from a supplemental run into a base run."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from run_ibo2023_latest_models import (
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PRICES_PATH,
    DEFAULT_SCORING_KEY_PATH,
    DEFAULT_ITEMS_PATH,
    load_json,
    model_tag,
    quiz_index,
    rescore_existing_row,
    save_json,
    scoring_map_from_key,
    summarize_rows,
)


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\n,]+", raw or "") if item.strip()]


def is_better_patch(base: dict[str, Any] | None, patch: dict[str, Any], *, allow_replace_scored: bool) -> bool:
    if patch.get("error"):
        return False
    if patch.get("score_status") in {"api_error", "api_exception", "missing_tags"}:
        return False
    if base is None:
        return True
    if allow_replace_scored:
        return True
    if base.get("error") or base.get("score_status") in {"api_error", "api_exception", "missing_tags"}:
        return True
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_run", required=True)
    parser.add_argument("--patch_run", required=True)
    parser.add_argument("--models", required=True)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--items_path", type=Path, default=DEFAULT_ITEMS_PATH)
    parser.add_argument("--scoring_key_path", type=Path, default=DEFAULT_SCORING_KEY_PATH)
    parser.add_argument("--prices_path", type=Path, default=DEFAULT_PRICES_PATH)
    parser.add_argument("--allow_replace_scored", action="store_true")
    args = parser.parse_args()

    items = load_json(args.items_path)
    scoring_map = scoring_map_from_key(load_json(args.scoring_key_path))
    prices = load_json(args.prices_path)
    items_by_qid = {quiz_index(item): item for item in items}

    report: dict[str, Any] = {"base_run": args.base_run, "patch_run": args.patch_run, "models": []}
    for model in parse_csv(args.models):
        tag = model_tag(model)
        base_dir = args.output_root / args.base_run / tag
        patch_dir = args.output_root / args.patch_run / tag
        base_path = base_dir / "solutions.json"
        patch_path = patch_dir / "solutions.json"
        if not base_path.exists():
            raise FileNotFoundError(base_path)
        if not patch_path.exists():
            raise FileNotFoundError(patch_path)

        base_rows = {str(row["quiz_index"]): row for row in load_json(base_path) if isinstance(row, dict) and row.get("quiz_index")}
        patch_rows = {str(row["quiz_index"]): row for row in load_json(patch_path) if isinstance(row, dict) and row.get("quiz_index")}
        merged = dict(base_rows)
        replaced: list[str] = []
        skipped: list[str] = []

        for qid, patch_row in patch_rows.items():
            item = items_by_qid.get(qid)
            if item is None or qid not in scoring_map:
                skipped.append(qid)
                continue
            patch_row = rescore_existing_row(
                row=patch_row,
                item=item,
                sub_scores=scoring_map[qid],
                prices=prices,
                model=model,
                allow_tagless_parse=True,
            )
            if is_better_patch(base_rows.get(qid), patch_row, allow_replace_scored=args.allow_replace_scored):
                merged[qid] = patch_row
                replaced.append(qid)
            else:
                skipped.append(qid)

        ordered = [merged[quiz_index(item)] for item in items if quiz_index(item) in merged]
        save_json(base_path, ordered)
        save_json(base_dir / "scores.json", ordered)
        summary = {
            "model": model,
            "model_tag": tag,
            "run_dir": str(base_dir),
            "solutions_path": str(base_path),
            "items": len(ordered),
            **summarize_rows(ordered),
            "merge_source_run": args.patch_run,
            "merged_replaced_items": replaced,
            "merge_skipped_items": skipped,
        }
        save_json(base_dir / "summary.json", summary)
        report["models"].append(summary)

    save_json(args.output_root / args.base_run / "merge_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
