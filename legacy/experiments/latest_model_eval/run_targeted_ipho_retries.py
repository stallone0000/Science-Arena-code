#!/usr/bin/env python3
"""Run IPhO item-level retries with one subprocess per item."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GENERATOR = ROOT / "new-things/interleaved/run_arena_generate.py"


def parse_cases(text: str) -> list[tuple[str, int]]:
    cases: list[tuple[str, int]] = []
    for raw in text.replace(",", "\n").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        if ":" not in item:
            raise ValueError(f"Case must be section:item_index, got: {item}")
        section, index = item.split(":", 1)
        section = section.strip().upper()
        if section.startswith("S"):
            section = section[1:]
        cases.append((section, int(index.strip())))
    return cases


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if isinstance(data, list) else []


def run_case(args: argparse.Namespace, section: str, item_index: int) -> dict[str, Any]:
    run_name = f"{args.run_prefix}__S{section}__item{item_index}"
    cmd = [
        sys.executable,
        str(GENERATOR),
        "--exam",
        "ipho",
        "--method",
        args.method,
        "--question_num_index",
        section,
        "--run_name",
        run_name,
        "--data_root",
        str(args.data_root),
        "--output_root",
        str(args.output_root),
        "--api_backend",
        "requests",
        "--pre_model",
        args.model,
        "--pre_multimodal_mode",
        args.pre_multimodal_mode,
        "--thinking_mode",
        args.thinking_mode,
        "--thinking_effort",
        args.thinking_effort,
        "--max_tokens",
        str(args.max_tokens),
        "--max_try",
        str(args.max_try),
        "--retry_delay",
        str(int(args.retry_delay)),
        "--timeout",
        str(float(args.api_timeout)),
        "--hard_timeout",
        str(float(args.hard_timeout)),
        "--item_indices",
        str(item_index),
    ]
    timeout_cmd = ["timeout", "--kill-after=10s", f"{int(args.wall_timeout)}s", *cmd]
    log_dir = args.output_root / "_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{run_name}.log"
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.run(timeout_cmd, cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT, check=False)

    run_dir = args.output_root / run_name / args.method / "ipho" / f"S{section}"
    solution_path = run_dir / "solutions.json"
    if not solution_path.exists():
        solution_path = run_dir / "solutions.partial.json"
    rows = load_rows(solution_path)
    errors = 0
    for row in rows:
        if row.get("error") or not str(row.get("answer") or row.get("llm_output") or "").strip():
            errors += 1
    return {
        "section": f"S{section}",
        "item_index": item_index,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "solution_path": str(solution_path) if solution_path.exists() else "",
        "returncode": proc.returncode,
        "rows": len(rows),
        "errors": errors,
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", help="Comma/newline separated section:item_index cases, e.g. S2:7,S3:11.")
    parser.add_argument("--cases_file", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--run_prefix", required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, default=ROOT / "data/science_arena/ipho2025")
    parser.add_argument("--method", default="interleaved")
    parser.add_argument("--pre_multimodal_mode", default="auto")
    parser.add_argument("--thinking_mode", default="auto")
    parser.add_argument("--thinking_effort", default="auto")
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--max_try", type=int, default=2)
    parser.add_argument("--retry_delay", type=float, default=20)
    parser.add_argument("--api_timeout", type=int, default=1800)
    parser.add_argument("--hard_timeout", type=int, default=1800)
    parser.add_argument("--wall_timeout", type=int, default=1860)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cases: list[tuple[str, int]] = []
    if args.cases:
        cases.extend(parse_cases(args.cases))
    if args.cases_file:
        cases.extend(parse_cases(args.cases_file.read_text(encoding="utf-8")))
    if not cases:
        raise SystemExit("No cases provided.")

    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {executor.submit(run_case, args, section, item_index): (section, item_index) for section, item_index in cases}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[DONE] {result['section']} item={result['item_index']} "
                f"rc={result['returncode']} rows={result['rows']} errors={result['errors']}",
                flush=True,
            )

    summary_path = args.output_root / f"{args.run_prefix}_targeted_ipho_retry_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "created_at": datetime.now().strftime("%Y%m%d-%H%M%S"),
                "model": args.model,
                "run_prefix": args.run_prefix,
                "output_root": str(args.output_root),
                "case_count": len(cases),
                "results": sorted(results, key=lambda row: (row["section"], row["item_index"])),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()
