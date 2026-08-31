#!/usr/bin/env python3
"""Normalize nested JSON types in the text-only interleaved dataset.

This does not change solver-visible or judge-visible text.  It only makes
metadata fields stable enough for JSON/Hugging Face schema inference.
"""

from __future__ import annotations

import argparse
import ast
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/science_arena_text_interleaved_v1"


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
            except Exception:
                return [value]
            if isinstance(parsed, list):
                return parsed
        return [value]
    return [value]


def normalize_figure_list(figures: Any) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for fig in figures or []:
        if not isinstance(fig, dict):
            normalized.append({"description": str(fig), "asset_index": "", "extracted_values": []})
            continue
        out = dict(fig)
        asset_index = out.get("asset_index")
        out["asset_index"] = "" if asset_index is None else str(asset_index)
        out["extracted_values"] = as_list(out.get("extracted_values"))
        normalized.append(out)
    return normalized


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    out = dict(row)
    comp = dict(out.get("staticization_completeness") or {})
    comp["issues"] = [str(item) for item in as_list(comp.get("issues")) if str(item).strip()]
    out["staticization_completeness"] = comp
    out["question_figures_staticized"] = normalize_figure_list(out.get("question_figures_staticized"))
    out["solution_figures_staticized"] = normalize_figure_list(out.get("solution_figures_staticized"))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    path = args.dataset_dir / "benchmark_items.jsonl"
    rows = [normalize_row(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    else:
        path.write_text(payload, encoding="utf-8")
        info_path = args.dataset_dir / "dataset_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["hf_schema_normalized_at"] = datetime.now().replace(microsecond=0).isoformat()
            info["hf_schema_normalization"] = {
                "staticization_completeness.issues": "list[str]",
                "question_figures_staticized.asset_index": "str",
                "question_figures_staticized.extracted_values": "list",
                "solution_figures_staticized.asset_index": "str",
                "solution_figures_staticized.extracted_values": "list",
            }
            info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output or path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
