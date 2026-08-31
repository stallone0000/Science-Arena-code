#!/usr/bin/env python3
"""Merge full-method standard-exam question-level retries.

Full/one-shot generation stores one row per whole question. This complements
merge_standard_retries.py, which is keyed by sub_question_id for interleaved
runs.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_FILES = {
    "ccho": ROOT / "data/science_arena/ccho2025/fix_CCHO-2025-latex.json",
    "cpho": ROOT / "data/science_arena/cpho2025/fix_CPHO-2025-latex.json",
    "icho": ROOT / "data/science_arena/icho2025/fix_ICHO-2025-latex.json",
}


def load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def scalar(value: Any) -> str:
    if isinstance(value, list):
        value = value[0] if value else ""
    return str(value or "").strip()


def row_key(row: Any) -> str | None:
    if not isinstance(row, dict):
        return None
    qid = scalar(row.get("question_id"))
    return qid or None


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


def official_questions(exam: str) -> list[str]:
    data = load_json(DATA_FILES[exam], [])
    qids: list[str] = []
    for question in data:
        qid = scalar(question.get("question_id"))
        if qid:
            qids.append(qid)
    return qids


def collect_json_files(roots: list[Path], filename: str) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        if root.is_file() and root.name == filename:
            paths.append(root)
            continue
        paths.extend(sorted(root.rglob(filename)))
    return paths


def filter_exam_paths(paths: list[Path], exam: str, method: str) -> list[Path]:
    filtered: list[Path] = []
    for path in paths:
        parts = path.parts
        exam_segments = [
            parts[index + 1]
            for index, part in enumerate(parts[:-1])
            if part == method and index + 1 < len(parts)
        ]
        if not exam_segments or exam in exam_segments:
            filtered.append(path)
    return filtered


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
    parser.add_argument("--exam", required=True, choices=sorted(DATA_FILES))
    parser.add_argument("--output_run_dir", type=Path, required=True)
    parser.add_argument("--base_solution", type=Path, action="append", default=[])
    parser.add_argument("--base_prompt", type=Path, action="append", default=[])
    parser.add_argument("--retry_root", type=Path, action="append", default=[])
    parser.add_argument("--retry_path_contains", action="append", default=[])
    parser.add_argument("--run_name", default="merged")
    parser.add_argument("--method", default="full")
    args = parser.parse_args()

    solution_map: dict[str, Any] = {}
    prompt_map: dict[str, Any] = {}
    for path in args.base_solution:
        add_rows(solution_map, load_json(path, []))
    for path in args.base_prompt:
        add_rows(prompt_map, load_json(path, []))

    retry_solution_paths = filter_exam_paths(collect_json_files(args.retry_root, "solutions.json"), args.exam, args.method)
    retry_solution_paths += filter_exam_paths(collect_json_files(args.retry_root, "solutions.partial.json"), args.exam, args.method)
    retry_prompt_paths = filter_exam_paths(collect_json_files(args.retry_root, "prompts.json"), args.exam, args.method)
    retry_prompt_paths += filter_exam_paths(collect_json_files(args.retry_root, "prompts.partial.json"), args.exam, args.method)
    retry_solution_paths = filter_path_contains(retry_solution_paths, args.retry_path_contains)
    retry_prompt_paths = filter_path_contains(retry_prompt_paths, args.retry_path_contains)

    for path in retry_solution_paths:
        add_rows(solution_map, load_json(path, []))
    for path in retry_prompt_paths:
        add_rows(prompt_map, load_json(path, []))

    merged_solutions: list[dict[str, Any]] = []
    merged_prompts: list[dict[str, Any]] = []
    unresolved: list[dict[str, str]] = []
    for qid in official_questions(args.exam):
        row = solution_map.get(qid)
        if row is None:
            row = {"question_id": qid, "answer": "", "llm_output": "", "usage": {}, "error": "missing_after_targeted_retries"}
        if not is_success(row):
            unresolved.append({"question_id": qid, "error": str(row.get("error") or row.get("finish_reason") or "empty_answer")})
        merged_solutions.append(row)
        merged_prompts.append(prompt_map.get(qid) or {"question_id": qid, "unit_type": "question"})

    save_json(args.output_run_dir / "solutions.json", merged_solutions)
    save_json(args.output_run_dir / "prompts.json", merged_prompts)
    save_json(
        args.output_run_dir / "summary.json",
        {
            "generation": {
                "run_name": args.run_name,
                "method": args.method,
                "exam": args.exam,
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
