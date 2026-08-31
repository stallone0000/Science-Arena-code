#!/usr/bin/env python3
"""Run latest-model Science Arena generation plus calibrated judge scoring.

This orchestrator intentionally reuses the existing generation runner and the
calibrated judge runner instead of duplicating prompt logic.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GEN_SCRIPT = ROOT / "new-things/interleaved/run_arena_generate.py"
JUDGE_SCRIPT = ROOT / "experiments/judge_pipeline_v2/science_arena_judge.py"
PRICES_PATH = ROOT / "experiments/judge_pipeline_v2/model_prices_360.json"
SMOKE_RESULTS_PATH = ROOT / "experiments/judge_pipeline_v2/api_smoke/target_models_20260522/smoke_results.json"
DEFAULT_ENV_FILE = Path("/home/jovyan/zhaoguangxiang-data-zzzc/zhaoguangxiang/.config/science_arena/360.env")

DEFAULT_MODELS = [
    "cloudsway/gemini-3-flash-preview",
    "google/gemini-3.5-flash",
    "qwen/qwen3.7-max",
    "qwen/qwen3.6-plus",
    "bytedance/doubao-seed-2-0-pro",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "moonshotai/kimi-k2.6",
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.7",
    "minimax/MiniMax-M2.7",
    "z-ai/glm-5.1",
    "xiaomi/mimo-v2.5-pro",
]

DATA_ROOTS = {
    "ccho": ROOT / "data/science_arena/ccho2025",
    "cpho": ROOT / "data/science_arena/cpho2025",
    "icho": ROOT / "data/science_arena/icho2025",
    "ipho": ROOT / "data/science_arena/ipho2025",
}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\s,]+", raw or "") if item.strip()]


def model_tag(model: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "__", model.strip())
    return tag.strip("_") or "model"


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().lstrip("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def smoke_status() -> dict[str, bool]:
    data = load_json(SMOKE_RESULTS_PATH, {})
    result: dict[str, bool] = {}
    for row in data.get("models", []) if isinstance(data, dict) else []:
        if isinstance(row, dict) and row.get("model"):
            result[str(row["model"])] = bool(row.get("ok"))
    return result


def usage_cost(usage: dict[str, Any], model: str, prices: dict[str, Any]) -> dict[str, float]:
    row = prices.get(model, {})
    prompt_tokens = float(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = float(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    input_price = float(row.get("input_price_cny_per_1m", 0))
    output_price = float(row.get("output_price_cny_per_1m", 0))
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": float(usage.get("total_tokens") or prompt_tokens + completion_tokens),
        "cost_cny": prompt_tokens / 1_000_000 * input_price + completion_tokens / 1_000_000 * output_price,
    }


def summarize_generation(solution_path: Path, model: str, prices: dict[str, Any]) -> dict[str, Any]:
    rows = load_json(solution_path, [])
    prompt_tokens = 0.0
    completion_tokens = 0.0
    total_tokens = 0.0
    cost_cny = 0.0
    errors = 0
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and row.get("error"):
            errors += 1
        usage = row.get("usage", {}) if isinstance(row, dict) else {}
        cost = usage_cost(usage, model, prices)
        prompt_tokens += cost["prompt_tokens"]
        completion_tokens += cost["completion_tokens"]
        total_tokens += cost["total_tokens"]
        cost_cny += cost["cost_cny"]
    return {
        "units": len(rows) if isinstance(rows, list) else 0,
        "errors": errors,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_cny": cost_cny,
    }


def run_dir_for(output_root: Path, run_name: str, method: str, exam: str, section: str | None) -> Path:
    path = output_root / run_name / method / exam
    if exam == "ipho":
        path = path / (section or "S1")
    return path


def redact_command(cmd: list[str]) -> str:
    return " ".join("<redacted>" if "api_key" in part.lower() else part for part in cmd)


def run_command(cmd: list[str], log_path: Path, env: dict[str, str]) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {redact_command(cmd)}\n")
        log.flush()
        completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"[exit_code] {completed.returncode}\n")
        return int(completed.returncode)


def task_key(model: str, exam: str, section: str | None) -> str:
    return f"{model_tag(model)}::{exam}{'::' + section if section else ''}"


def run_task(args: argparse.Namespace, model: str, exam: str, section: str | None, env: dict[str, str]) -> dict[str, Any]:
    prices = load_json(PRICES_PATH, {})
    tag = model_tag(model)
    run_name = tag
    section_label = section or ""
    run_dir = run_dir_for(args.output_root, run_name, args.method, exam, section)
    solution_path = run_dir / "solutions.json"
    log_dir = args.output_root / "_logs" / tag
    log_suffix = f"{exam}_{section_label}" if section_label else exam

    record: dict[str, Any] = {
        "model": model,
        "model_tag": tag,
        "exam": exam,
        "section": section_label,
        "method": args.method,
        "run_dir": str(run_dir),
        "solution_path": str(solution_path),
    }

    gen_returncode = 0
    if solution_path.exists() and not args.force:
        record["generation_status"] = "skipped_existing"
    else:
        cmd = [
            sys.executable,
            str(GEN_SCRIPT),
            "--exam",
            exam,
            "--method",
            args.method,
            "--run_name",
            run_name,
            "--data_root",
            str(DATA_ROOTS[exam]),
            "--output_root",
            str(args.output_root),
            "--api_backend",
            "requests",
            "--pre_model",
            model,
            "--pre_multimodal_mode",
            args.pre_multimodal_mode,
            "--thinking_mode",
            args.generation_thinking_mode,
            "--thinking_effort",
            args.generation_thinking_effort,
            "--max_tokens",
            str(args.max_tokens),
            "--max_try",
            str(args.max_try),
            "--retry_delay",
            str(int(args.retry_delay)),
            "--timeout",
            str(args.timeout),
        ]
        if args.hard_timeout:
            cmd.extend(["--hard_timeout", str(args.hard_timeout)])
        if args.temperature is not None:
            cmd.extend(["--temperature", str(args.temperature)])
        if args.top_p is not None:
            cmd.extend(["--top_p", str(args.top_p)])
        if args.top_k is not None:
            cmd.extend(["--top_k", str(args.top_k)])
        if exam == "ipho" and section:
            cmd.extend(["--question_num_index", section.removeprefix("S")])
        if args.question_ids:
            cmd.extend(["--question_ids", args.question_ids])
        if args.sub_question_ids:
            cmd.extend(["--sub_question_ids", args.sub_question_ids])
        if args.item_indices:
            cmd.extend(["--item_indices", args.item_indices])
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.interleaved_current_image_persist != "inherit":
            cmd.extend(["--interleaved_current_image_persist", args.interleaved_current_image_persist])
        if args.interleaved_single_user_image_turns != "inherit":
            cmd.extend(["--interleaved_single_user_image_turns", args.interleaved_single_user_image_turns])
        if args.max_images_per_turn:
            cmd.extend(["--max_images_per_turn", str(args.max_images_per_turn)])
        if args.system_prompt:
            cmd.extend(["--system_prompt", args.system_prompt])
        if args.public_benchmark_context:
            cmd.extend(["--public_benchmark_context", "true"])
        gen_returncode = run_command(cmd, log_dir / f"{log_suffix}_generation.log", env)
        record["generation_status"] = "ok" if gen_returncode == 0 else "failed"
        record["generation_returncode"] = gen_returncode

    if solution_path.exists():
        record["generation"] = summarize_generation(solution_path, model, prices)
    else:
        record["generation"] = {"units": 0, "errors": 1, "prompt_tokens": 0, "completion_tokens": 0, "cost_cny": 0}

    judge_run_name = f"{args.run_prefix}_{tag}_{exam}{'_' + section if section else ''}_judge"
    judge_dir = ROOT / "experiments/judge_pipeline_v2/runs" / judge_run_name
    aggregate_path = judge_dir / "aggregate.json"
    if gen_returncode != 0 or not solution_path.exists():
        record["judge_status"] = "skipped_generation_failed"
    elif aggregate_path.exists() and not args.force:
        record["judge_status"] = "skipped_existing"
    else:
        cmd = [
            sys.executable,
            str(JUDGE_SCRIPT),
            "--exam",
            exam,
            "--judge_model",
            args.judge_model,
            "--input_solution_path",
            str(solution_path),
            "--method",
            args.method,
            "--model_under_test",
            model,
            "--run_name",
            judge_run_name,
            "--max_workers",
            str(args.judge_workers),
            "--max_tokens",
            str(args.max_tokens),
            "--max_try",
            str(args.max_try),
            "--retry_delay",
            str(args.retry_delay),
            "--timeout",
            str(args.timeout),
            "--multimodal",
            "--enable_thinking",
        ]
        if exam == "ipho" and section:
            cmd.extend(["--sections", section])
        if args.case_ids:
            cmd.extend(["--case_ids", args.case_ids])
        if args.item_indices and exam == "ipho":
            wanted_case_ids = []
            for item in parse_csv(args.item_indices):
                wanted_case_ids.append(f"IPHO-{section}-{int(item) + 1}")
            cmd.extend(["--case_ids", ",".join(wanted_case_ids)])
        if args.limit and not args.item_indices:
            cmd.extend(["--limit", str(args.limit)])
        judge_returncode = run_command(cmd, log_dir / f"{log_suffix}_judge.log", env)
        record["judge_status"] = "ok" if judge_returncode == 0 else "failed"
        record["judge_returncode"] = judge_returncode

    record["judge_run_dir"] = str(judge_dir)
    aggregate = load_json(aggregate_path, {})
    record["judge"] = aggregate.get(model, {}) if isinstance(aggregate, dict) else {}
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--exams", default="icho,ipho", help="Comma-separated: icho,ipho,ccho,cpho.")
    parser.add_argument("--run_prefix", default="latest", help="Prefix for judge run directories.")
    parser.add_argument("--sections", default="S1,S2,S3")
    parser.add_argument("--question_ids", default="", help="Forwarded to generation for standard-exam smoke runs.")
    parser.add_argument("--sub_question_ids", default="", help="Forwarded to generation for targeted standard retries.")
    parser.add_argument("--item_indices", default="", help="Forwarded to generation; for IPhO also scopes judge case IDs.")
    parser.add_argument("--case_ids", default="", help="Optional explicit judge case IDs.")
    parser.add_argument("--limit", type=int, default=0, help="Forwarded to generation; use carefully for judge scoping.")
    parser.add_argument("--method", choices=["iterative", "interleaved", "full"], default="interleaved")
    parser.add_argument("--output_root", type=Path, default=ROOT / f"artifacts/runs/latest_360_{datetime.now():%Y%m%d_%H%M%S}")
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--judge_model", default="google/gemini-3.5-flash")
    parser.add_argument("--pre_multimodal_mode", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--max_try", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=5)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--hard_timeout",
        type=float,
        default=0.0,
        help="Forwarded to generation as a wall-clock alarm around one request; 0 disables it.",
    )
    parser.add_argument("--workers", type=int, default=2, help="Parallel model/exam/section tasks.")
    parser.add_argument("--judge_workers", type=int, default=100, help="Per-task judge API concurrency.")
    parser.add_argument("--generation_thinking_mode", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--system_prompt", default="", help="Forwarded to generation as system/developer context.")
    parser.add_argument(
        "--public_benchmark_context",
        action="store_true",
        help="Forwarded to generation to replace live-test/candidate wording with public benchmark context.",
    )
    parser.add_argument(
        "--generation_thinking_effort",
        default="auto",
        choices=["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument(
        "--interleaved_current_image_persist",
        default="inherit",
        choices=["inherit", "on", "off"],
        help="Forwarded to generation; use off for routes that reject image_url blocks in historical messages.",
    )
    parser.add_argument(
        "--interleaved_single_user_image_turns",
        default="inherit",
        choices=["inherit", "on", "off"],
        help="Forwarded to generation; use on for routes that reject image_url blocks outside a single user message.",
    )
    parser.add_argument(
        "--max_images_per_turn",
        type=int,
        default=0,
        help="Forwarded to generation; compatibility limit for routes that fail on multi-image turns.",
    )
    parser.add_argument("--include_failed_smoke", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    env = os.environ.copy()
    env.update(load_env_file(args.env_file))
    if not any(env.get(name) for name in ("QIHOO_API_KEY", "QIHOO360_API_KEY", "360_API_KEY", "API_KEY")):
        raise RuntimeError(f"Missing API key. Source env or provide --env_file. Not found: {args.env_file}")

    requested_models = parse_csv(args.models)
    requested_exams = parse_csv(args.exams)
    requested_sections = parse_csv(args.sections)
    smoke = smoke_status()
    tasks: list[tuple[str, str, str | None]] = []
    skipped: list[dict[str, Any]] = []
    for model in requested_models:
        if not args.include_failed_smoke and smoke.get(model) is False:
            skipped.append({"model": model, "status": "skipped_unavailable_in_smoke"})
            continue
        for exam in requested_exams:
            if exam == "ipho":
                for section in requested_sections:
                    tasks.append((model, exam, section.upper()))
            else:
                tasks.append((model, exam, None))

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_root / "latest_eval_summary.json"
    summary: dict[str, Any] = {
        "created_at": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "output_root": str(args.output_root),
        "method": args.method,
        "judge_model": args.judge_model,
        "task_count": len(tasks),
        "skipped": skipped,
        "results": [],
    }
    save_json(summary_path, summary)

    lock = threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_task, args, model, exam, section, env): task_key(model, exam, section) for model, exam, section in tasks}
        for future in concurrent.futures.as_completed(futures):
            key = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"task": key, "status": "exception", "error": f"{type(exc).__name__}: {exc}"}
            with lock:
                summary["results"].append(result)
                save_json(summary_path, summary)
            print(f"[DONE] {key}: gen={result.get('generation_status')} judge={result.get('judge_status')}")

    totals = {"generation_cost_cny": 0.0, "judge_cost_cny": 0.0, "generation_prompt_tokens": 0.0, "generation_completion_tokens": 0.0, "judge_prompt_tokens": 0.0, "judge_completion_tokens": 0.0}
    for result in summary["results"]:
        generation = result.get("generation", {}) if isinstance(result, dict) else {}
        judge = result.get("judge", {}) if isinstance(result, dict) else {}
        totals["generation_cost_cny"] += float(generation.get("cost_cny", 0) or 0)
        totals["generation_prompt_tokens"] += float(generation.get("prompt_tokens", 0) or 0)
        totals["generation_completion_tokens"] += float(generation.get("completion_tokens", 0) or 0)
        totals["judge_cost_cny"] += float(judge.get("cost_cny", 0) or 0)
        totals["judge_prompt_tokens"] += float(judge.get("prompt_tokens", 0) or 0)
        totals["judge_completion_tokens"] += float(judge.get("completion_tokens", 0) or 0)
    totals["total_cost_cny"] = totals["generation_cost_cny"] + totals["judge_cost_cny"]
    summary["totals"] = totals
    save_json(summary_path, summary)
    print(f"[SUMMARY] {summary_path}")
    print(json.dumps(totals, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
