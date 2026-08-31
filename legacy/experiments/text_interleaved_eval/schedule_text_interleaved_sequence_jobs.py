#!/usr/bin/env python3
"""Repair dirty text-interleaved runs one sequence subprocess at a time."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/science_arena_text_interleaved_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/runs/text_interleaved_eval"


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('benchmark_id')}::{row.get('sequence_id')}::{row.get('turn_id')}"


def row_has_image_payload(row: dict[str, Any]) -> bool:
    meta = row.get("generation_client_meta") or {}
    image_summary = meta.get("final_payload_image_summary") or {}
    for key in [
        "chat_image_url_count",
        "data_url_image_count",
        "empty_image_url_count",
        "remote_url_image_count",
        "responses_input_image_count",
    ]:
        if int(image_summary.get(key) or 0):
            return True
    return False


def is_bad_finish_reason(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text in {"stop", "completed", "complete", "objective_key"}:
        return False
    return any(fragment in text for fragment in ("content_filter", "finish_content_filter", "length", "max_token", "max_output", "truncat"))


def row_is_clean(row: dict[str, Any]) -> bool:
    judge_result = row.get("judge_result") or {}
    return not (
        row.get("generation_error")
        or row.get("judge_error")
        or judge_result.get("parse_error")
        or row_has_image_payload(row)
        or is_bad_finish_reason(row.get("generation_finish_reason"))
        or is_bad_finish_reason(row.get("judge_finish_reason"))
    )


def sequence_is_clean(sequence_rows: list[dict[str, Any]], existing_by_key: dict[str, dict[str, Any]]) -> bool:
    for row in sequence_rows:
        existing = existing_by_key.get(row_key(row))
        if not existing or not row_is_clean(existing):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log_root", type=Path, required=True)
    parser.add_argument("--run_suffix", default="20260524_v2")
    parser.add_argument("--model", action="append", required=True)
    parser.add_argument("--request_model", default="")
    parser.add_argument("--judge_model", default="google/gemini-3.5-flash")
    parser.add_argument("--benchmark_id", action="append", required=True)
    parser.add_argument("--sequence_id", action="append", default=[])
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--judge_max_tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=30)
    parser.add_argument("--sequence_timeout", type=float, default=7200)
    parser.add_argument("--sleep_between", type=float, default=0)
    parser.add_argument("--thinking_mode", default="auto")
    parser.add_argument("--thinking_effort", default="auto")
    parser.add_argument("--judge_thinking_mode", default="on")
    parser.add_argument("--judge_thinking_effort", default="high")
    parser.add_argument(
        "--force_rerun_selected_sequences",
        action="store_true",
        help="Run selected sequences even if existing rows are clean.",
    )
    args = parser.parse_args()

    rows = [row for row in iter_jsonl(args.dataset_dir / "benchmark_items.jsonl") if row.get("benchmark_id") in set(args.benchmark_id)]
    rows_by_bench_seq: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_bench_seq.setdefault((str(row.get("benchmark_id")), str(row.get("sequence_id"))), []).append(row)
    requested_sequences = set(args.sequence_id)

    jobs: list[dict[str, Any]] = []
    for model in args.model:
        for (benchmark_id, sequence_id), sequence_rows in sorted(rows_by_bench_seq.items()):
            if requested_sequences and sequence_id not in requested_sequences:
                continue
            run_name = f"text_staticized_{safe_name(model)}_{benchmark_id}_{args.run_suffix}"
            model_dir = args.output_root / run_name / safe_name(model)
            existing_by_key = {row_key(row): row for row in iter_jsonl(model_dir / "turn_results.jsonl")}
            if sequence_is_clean(sequence_rows, existing_by_key) and not args.force_rerun_selected_sequences:
                continue
            cmd = [
                sys.executable,
                "experiments/text_interleaved_eval/run_text_interleaved_eval.py",
                "--dataset_dir",
                str(args.dataset_dir),
                "--output_root",
                str(args.output_root),
                "--run_name",
                run_name,
                "--model",
                model,
                "--judge_model",
                args.judge_model,
                "--benchmark_id",
                benchmark_id,
                "--sequence_id",
                sequence_id,
                "--max_tokens",
                str(args.max_tokens),
                "--judge_max_tokens",
                str(args.judge_max_tokens),
                "--timeout",
                str(args.timeout),
                "--max_retries",
                str(args.max_retries),
                "--retry_delay",
                str(args.retry_delay),
                "--workers",
                "1",
                "--thinking_mode",
                args.thinking_mode,
                "--thinking_effort",
                args.thinking_effort,
                "--judge_thinking_mode",
                args.judge_thinking_mode,
                "--judge_thinking_effort",
                args.judge_thinking_effort,
                "--resume",
            ]
            if args.force_rerun_selected_sequences:
                cmd.append("--force_rerun_selected_sequences")
            if args.request_model:
                cmd[cmd.index("--judge_model"):cmd.index("--judge_model")] = ["--request_model", args.request_model]
            jobs.append(
                {
                    "model": model,
                    "benchmark_id": benchmark_id,
                    "sequence_id": sequence_id,
                    "run_name": run_name,
                    "cmd": cmd,
                    "expected_rows": len(sequence_rows),
                }
            )

    args.log_root.mkdir(parents=True, exist_ok=True)
    (args.log_root / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"jobs_to_run": len(jobs), "models": args.model, "benchmarks": args.benchmark_id}, ensure_ascii=False), flush=True)

    completed: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        model = str(job["model"])
        log_path = args.log_root / f"{safe_name(model)}__{job['benchmark_id']}__{safe_name(job['sequence_id'])}.log"
        started = time.time()
        with log_path.open("ab", buffering=0) as log_file:
            log_file.write((f"\n=== START {time.strftime('%F %T')} {model} {job['benchmark_id']} {job['sequence_id']} ===\n").encode())
            process = subprocess.Popen(job["cmd"], cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
            timed_out = False
            try:
                returncode = process.wait(timeout=args.sequence_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                process.kill()
                returncode = process.wait(timeout=30)
            elapsed = time.time() - started
            log_file.write((f"\n=== END {time.strftime('%F %T')} rc={returncode} elapsed={elapsed:.1f}s timed_out={timed_out} ===\n").encode())
        record = {
            "model": model,
            "request_model": args.request_model or model,
            "benchmark_id": job["benchmark_id"],
            "sequence_id": job["sequence_id"],
            "returncode": returncode,
            "elapsed_seconds": round(elapsed, 1),
            "timed_out": timed_out,
            "log_path": str(log_path),
        }
        completed.append(record)
        (args.log_root / "completed.json").write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"event": "finished", "index": index, "total": len(jobs), **record}, ensure_ascii=False), flush=True)
        if args.sleep_between:
            time.sleep(args.sleep_between)


if __name__ == "__main__":
    main()
