#!/usr/bin/env python3
"""Re-parse saved judge raw outputs after parser/policy changes."""

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


def reparse_row(row: dict[str, Any]) -> dict[str, Any]:
    if row.get("error") or not row.get("raw_output"):
        return row
    new_row = dict(row)
    parsed = science_arena_judge.parse_json_object(str(row.get("raw_output", "")))
    max_score = float(row.get("max_score") or parsed.get("max_score") or 0.0)
    score = float(parsed.get("score", 0.0))
    if max_score:
        score = max(0.0, min(score, max_score))
    score, overrides = science_arena_judge.postprocess_score(new_row, parsed, score)
    new_row["parsed"] = parsed
    new_row["score"] = score
    new_row["validation_overrides"] = overrides
    return new_row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam", choices=["icho", "ipho"], required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--out_run", required=True)
    args = parser.parse_args()

    src = Path(args.run_dir).resolve()
    out = ROOT / "experiments/judge_pipeline_v2/runs" / args.out_run
    rows = [reparse_row(row) for row in load(src / "judge_results.json")]
    out.mkdir(parents=True, exist_ok=True)
    for name in ("metadata.json", "cases_manifest.json", "merge_replacements.json"):
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    meta_path = out / "metadata.json"
    meta = load(meta_path) if meta_path.exists() else {}
    meta.update({"reparsed_from": str(src), "reparsed_at": datetime.now().isoformat(timespec="seconds")})
    save(meta_path, meta)
    save(out / "judge_results.json", sorted(rows, key=lambda r: (r.get("model_under_test", ""), r.get("case_id", ""))))
    save(out / "aggregate.json", science_arena_judge.aggregate(rows, args.exam))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

