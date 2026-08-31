#!/usr/bin/env python3
"""Run text-only interleaved model x benchmark jobs with bounded concurrency."""

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
DEFAULT_MODEL_CONFIG = ROOT / "configs/text_interleaved_models_20260524.json"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/runs/text_interleaved_eval"
DEFAULT_LOG_ROOT = DEFAULT_OUTPUT_ROOT / "scheduled_text_staticized_jobs_20260524"


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", value)


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('benchmark_id')}::{row.get('sequence_id')}::{row.get('turn_id')}"


def is_bad_finish_reason(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text in {"stop", "completed", "complete", "objective_key"}:
        return False
    return any(fragment in text for fragment in ("content_filter", "finish_content_filter", "length", "max_token", "max_output", "truncat"))


def expected_by_benchmark(dataset_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in iter_jsonl(dataset_dir / "benchmark_items.jsonl"):
        bench = str(row.get("benchmark_id"))
        counts[bench] = counts.get(bench, 0) + 1
    return counts


def model_dir_name(model: str) -> str:
    return safe_name(model)


def job_is_clean(run_dir: Path, model: str, benchmark_id: str, expected_rows: int) -> bool:
    model_dir = run_dir / model_dir_name(model)
    summary_path = model_dir / "benchmark_summary.json"
    results_path = model_dir / "turn_results.jsonl"
    if not summary_path.exists() or not results_path.exists():
        return False
    summaries = load_json(summary_path, [])
    summary = next((item for item in summaries if item.get("benchmark_id") == benchmark_id), None)
    if not summary:
        return False
    if int(summary.get("turn_count") or 0) != expected_rows:
        return False
    if int(summary.get("generation_error_count") or 0) or int(summary.get("judge_error_count") or 0):
        return False
    rows = [row for row in iter_jsonl(results_path) if row.get("benchmark_id") == benchmark_id]
    if len(rows) != expected_rows:
        return False
    image_keys = [
        "chat_image_url_count",
        "data_url_image_count",
        "empty_image_url_count",
        "remote_url_image_count",
        "responses_input_image_count",
    ]
    for row in rows:
        if is_bad_finish_reason(row.get("generation_finish_reason")) or is_bad_finish_reason(row.get("judge_finish_reason")):
            return False
        meta = row.get("generation_client_meta") or {}
        image_summary = meta.get("final_payload_image_summary") or {}
        if any(int(image_summary.get(key) or 0) for key in image_keys):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--model_config", type=Path, default=DEFAULT_MODEL_CONFIG)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--log_root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--run_suffix", default="20260524_v2")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--request_model",
        default="",
        help="Optional API model name to request while recording under the logical --model. Requires exactly one --model.",
    )
    parser.add_argument("--benchmark_id", action="append", default=[])
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument("--workers_per_job", type=int, default=4)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--judge_max_tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_delay", type=float, default=5.0)
    parser.add_argument("--judge_model", default="google/gemini-3.5-flash")
    parser.add_argument("--job_timeout", type=float, default=14400.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_config = load_json(args.model_config, {})
    models = args.model or list(model_config.get("models", []))
    if not models:
        raise SystemExit("No models requested.")
    if args.request_model and len(models) != 1:
        raise SystemExit("--request_model can only be used with exactly one logical --model.")
    expected_counts = expected_by_benchmark(args.dataset_dir)
    benchmarks = args.benchmark_id or sorted(expected_counts)
    args.log_root.mkdir(parents=True, exist_ok=True)

    jobs: list[dict[str, Any]] = []
    for model in models:
        for benchmark_id in benchmarks:
            if benchmark_id not in expected_counts:
                raise SystemExit(f"Unknown benchmark_id: {benchmark_id}")
            run_name = f"text_staticized_{safe_name(model)}_{benchmark_id}_{args.run_suffix}"
            run_dir = args.output_root / run_name
            clean = job_is_clean(run_dir, model, benchmark_id, expected_counts[benchmark_id])
            if args.resume and clean:
                continue
            log_path = args.log_root / f"{safe_name(model)}__{benchmark_id}.log"
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
            ]
            if args.request_model:
                cmd.extend(["--request_model", args.request_model])
            cmd.extend(
                [
                "--judge_model",
                args.judge_model,
                "--benchmark_id",
                benchmark_id,
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
                str(args.workers_per_job),
                "--thinking_mode",
                "auto",
                "--thinking_effort",
                "auto",
                "--judge_thinking_mode",
                "on",
                "--judge_thinking_effort",
                "high",
                "--resume",
                ]
            )
            jobs.append(
                {
                    "model": model,
                    "benchmark_id": benchmark_id,
                    "run_name": run_name,
                    "run_dir": str(run_dir),
                    "log_path": str(log_path),
                    "cmd": cmd,
                    "expected_rows": expected_counts[benchmark_id],
                }
            )

    (args.log_root / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"jobs_to_run": len(jobs), "benchmarks": benchmarks, "models": models}, ensure_ascii=False), flush=True)

    running: dict[subprocess.Popen[bytes], dict[str, Any]] = {}
    completed: list[dict[str, Any]] = []
    index = 0
    while index < len(jobs) or running:
        while index < len(jobs) and len(running) < max(1, args.concurrency):
            job = jobs[index]
            index += 1
            log_file = open(job["log_path"], "ab", buffering=0)
            log_file.write((f"\n=== START {time.strftime('%F %T')} {job['model']} {job['benchmark_id']} ===\n").encode())
            process = subprocess.Popen(job["cmd"], cwd=ROOT, stdout=log_file, stderr=subprocess.STDOUT)
            job["pid"] = process.pid
            job["started_at"] = time.time()
            job["_log_file"] = log_file
            running[process] = job
            print(json.dumps({"event": "started", "pid": process.pid, "model": job["model"], "benchmark_id": job["benchmark_id"], "running": len(running), "remaining": len(jobs) - index}, ensure_ascii=False), flush=True)

        time.sleep(10)
        for process, job in list(running.items()):
            elapsed = time.time() - float(job["started_at"])
            returncode = process.poll()
            if returncode is None and elapsed > args.job_timeout:
                process.kill()
                returncode = process.wait(timeout=30)
                job["timed_out"] = True
            if returncode is None:
                continue
            log_file = job.pop("_log_file")
            log_file.write((f"\n=== END {time.strftime('%F %T')} rc={returncode} elapsed={elapsed:.1f}s ===\n").encode())
            log_file.close()
            run_dir = Path(job["run_dir"])
            clean = job_is_clean(run_dir, job["model"], job["benchmark_id"], int(job["expected_rows"]))
            record = {
                "model": job["model"],
                "benchmark_id": job["benchmark_id"],
                "run_dir": job["run_dir"],
                "log_path": job["log_path"],
                "returncode": returncode,
                "elapsed_seconds": round(elapsed, 1),
                "clean": clean,
                "timed_out": bool(job.get("timed_out")),
            }
            completed.append(record)
            (args.log_root / "completed.json").write_text(json.dumps(completed, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"event": "finished", **record, "running": len(running) - 1, "completed": len(completed), "total": len(jobs)}, ensure_ascii=False), flush=True)
            running.pop(process, None)

    failures = [record for record in completed if not record["clean"]]
    print(json.dumps({"completed": len(completed), "failures": len(failures), "failure_records": failures[:20]}, ensure_ascii=False, indent=2), flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
