#!/usr/bin/env python3
"""Merge judge reruns into a base run, replacing failed or stale rows."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import science_arena_judge


ROOT = Path(__file__).resolve().parents[2]


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("model_under_test")), str(row.get("case_id"))


def load_result_rows(run: Path) -> list[dict[str, Any]]:
    for name in ("judge_results.json", "judge_results_partial.json"):
        path = run / name
        if path.exists():
            rows = load(path)
            if not isinstance(rows, list):
                raise TypeError(f"{path} must contain a JSON list")
            return rows
    raise FileNotFoundError(f"{run} has neither judge_results.json nor judge_results_partial.json")


def should_replace(old: dict[str, Any] | None, new: dict[str, Any], replace_all: bool) -> bool:
    if old is None:
        return True
    if new.get("error") or new.get("score") is None:
        return False
    if replace_all:
        return True
    return bool(old.get("error") or old.get("score") is None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam", choices=["icho", "ipho", "ccho", "cpho"], required=True)
    parser.add_argument("--base_run", required=True)
    parser.add_argument("--patch_runs", nargs="+", required=True)
    parser.add_argument("--out_run", required=True)
    parser.add_argument("--replace_all", action="store_true")
    args = parser.parse_args()

    base = Path(args.base_run).resolve()
    out = ROOT / "experiments/judge_pipeline_v2/runs" / args.out_run
    rows_by_key = {row_key(row): row for row in load_result_rows(base)}
    replacements = []
    for patch_name in args.patch_runs:
        patch = Path(patch_name).resolve()
        for row in load_result_rows(patch):
            key = row_key(row)
            old = rows_by_key.get(key)
            if should_replace(old, row, args.replace_all):
                rows_by_key[key] = row
                replacements.append({"model": key[0], "case_id": key[1], "from": str(patch), "old_error": old.get("error") if old else None})

    rows = sorted(rows_by_key.values(), key=lambda r: (r.get("model_under_test", ""), r.get("case_id", "")))
    out.mkdir(parents=True, exist_ok=True)
    if (base / "metadata.json").exists():
        meta = load(base / "metadata.json")
    else:
        meta = {}
    meta.update(
        {
            "merged_from_base": str(base),
            "patch_runs": args.patch_runs,
            "replace_all": args.replace_all,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "replacement_count": len(replacements),
        }
    )
    if (base / "cases_manifest.json").exists():
        shutil.copy2(base / "cases_manifest.json", out / "cases_manifest.json")
    save(out / "metadata.json", meta)
    save(out / "merge_replacements.json", replacements)
    save(out / "judge_results.json", rows)
    save(out / "aggregate.json", science_arena_judge.aggregate(rows, args.exam))
    print(f"Wrote {out}")
    print(f"Replacements: {len(replacements)}")


if __name__ == "__main__":
    main()
