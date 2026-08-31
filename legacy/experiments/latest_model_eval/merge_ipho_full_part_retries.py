#!/usr/bin/env python3
"""Merge IPhO full-method part-level retries for one section."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def row_key(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    part_id = str(row.get("part_id") or "").strip()
    return part_id or None


def is_success(row: Any) -> bool:
    if not isinstance(row, dict) or row.get("error"):
        return False
    finish = str(row.get("finish_reason") or "").lower()
    if "content_filter" in finish:
        return False
    answer = str(row.get("answer") or row.get("llm_output") or "").strip()
    if not answer:
        return False
    lowered = answer.lower()
    refusal_markers = (
        "i'm sorry, but i cannot assist",
        "i cannot assist with that request",
        "i can't assist with that request",
        "unable to assist with that request",
    )
    return not any(marker in lowered for marker in refusal_markers)


def collect_json_files(roots: list[Path], filename: str, section: str, method: str) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        candidates = [root] if root.is_file() and root.name == filename else sorted(root.rglob(filename))
        for path in candidates:
            parts = path.parts
            section_segments = [
                parts[index + 2]
                for index, part in enumerate(parts[:-2])
                if part == method and index + 2 < len(parts) and parts[index + 1] == "ipho"
            ]
            if not section_segments or section.upper() in [segment.upper() for segment in section_segments]:
                paths.append(path)
    return paths


def filter_path_contains(paths: list[Path], needles: list[str]) -> list[Path]:
    if not needles:
        return paths
    return [path for path in paths if any(needle in str(path) for needle in needles)]


def add_rows(target: dict[str, Any], rows: list[Any]) -> None:
    for row in rows:
        key = row_key(row)
        if key is None:
            continue
        existing = target.get(key)
        if existing is None or not is_success(existing):
            target[key] = row


def usage_totals(rows: list[dict[str, Any]]) -> dict[str, float]:
    prompt = completion = total = 0.0
    for row in rows:
        usage = row.get("usage", {}) if isinstance(row, dict) else {}
        p = float(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        c = float(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        t = float(usage.get("total_tokens") or p + c)
        prompt += p
        completion += c
        total += t
    return {"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": total}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--section", required=True, help="IPhO section, e.g. S2.")
    parser.add_argument("--output_run_dir", type=Path, required=True)
    parser.add_argument("--base_solution", type=Path, action="append", default=[])
    parser.add_argument("--base_prompt", type=Path, action="append", default=[])
    parser.add_argument("--retry_root", type=Path, action="append", default=[])
    parser.add_argument("--retry_path_contains", action="append", default=[])
    parser.add_argument("--run_name", default="merged")
    parser.add_argument("--method", default="full")
    args = parser.parse_args()

    section = args.section.upper()
    if not section.startswith("S"):
        section = f"S{section}"

    solution_map: dict[str, Any] = {}
    prompt_map: dict[str, Any] = {}
    base_order: list[str] = []
    for path in args.base_solution:
        rows = load_json(path, [])
        for row in rows:
            key = row_key(row)
            if key and key not in base_order:
                base_order.append(key)
        add_rows(solution_map, rows)
    for path in args.base_prompt:
        add_rows(prompt_map, load_json(path, []))

    retry_solution_paths = collect_json_files(args.retry_root, "solutions.json", section, args.method)
    retry_solution_paths += collect_json_files(args.retry_root, "solutions.partial.json", section, args.method)
    retry_prompt_paths = collect_json_files(args.retry_root, "prompts.json", section, args.method)
    retry_prompt_paths += collect_json_files(args.retry_root, "prompts.partial.json", section, args.method)
    retry_solution_paths = filter_path_contains(retry_solution_paths, args.retry_path_contains)
    retry_prompt_paths = filter_path_contains(retry_prompt_paths, args.retry_path_contains)

    for path in retry_solution_paths:
        add_rows(solution_map, load_json(path, []))
    for path in retry_prompt_paths:
        add_rows(prompt_map, load_json(path, []))

    merged_order = base_order + sorted(key for key in solution_map if key not in base_order)
    merged_solutions = [solution_map[key] for key in merged_order]
    merged_prompts = [prompt_map.get(key) or {"part_id": key, "unit_type": "part"} for key in merged_order]
    unresolved = [
        {"part_id": key, "error": str(solution_map[key].get("error") or solution_map[key].get("finish_reason") or "empty_answer")}
        for key in merged_order
        if not is_success(solution_map[key])
    ]

    save_json(args.output_run_dir / "solutions.json", merged_solutions)
    save_json(args.output_run_dir / "prompts.json", merged_prompts)
    save_json(
        args.output_run_dir / "summary.json",
        {
            "generation": {
                "run_name": args.run_name,
                "method": args.method,
                "exam": "ipho",
                "section": section,
                "unit_count": len(merged_solutions),
                "prompt_count": len(merged_prompts),
                "unresolved_count": len(unresolved),
                "unresolved": unresolved,
                "merged_at": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "base_solution_paths": [str(path) for path in args.base_solution],
                "retry_roots": [str(path) for path in args.retry_root],
                "retry_solution_paths": [str(path) for path in retry_solution_paths],
                **usage_totals(merged_solutions),
            }
        },
    )
    print(f"[MERGED] {args.output_run_dir}")
    print(f"[MERGED] solutions={len(merged_solutions)} unresolved={len(unresolved)}")


if __name__ == "__main__":
    main()
