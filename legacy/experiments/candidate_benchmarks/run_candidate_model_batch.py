#!/usr/bin/env python3
"""Run a candidate page-range smoke config across multiple 360 models."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RUNNER = ROOT / "experiments/candidate_benchmarks/run_candidate_problem_smoke.py"


DEFAULT_MODELS = [
    "google/gemini-3.5-flash",
    "google/google/gemini-3.1-pro-preview",
    "cloudsway/gemini-3-flash-preview",
    "qiniu/openai/gpt-5.4",
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.7",
    "bytedance/doubao-seed-2-0-pro",
    "qwen/qwen3.6-plus",
    "qwen/qwen3.7-max",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "minimax/MiniMax-M2.7",
    "z-ai/glm-5.1",
    "xiaomi/mimo-v2.5-pro",
]

TEXT_ONLY_MODELS = {
    "qwen/qwen3.7-max",
    "minimax/MiniMax-M2.7",
    "xiaomi/mimo-v2.5-pro",
}


def safe_name(model: str) -> str:
    return model.replace("/", "__").replace(":", "_")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def run_one(args: argparse.Namespace, model: str) -> dict[str, Any]:
    mode = "off" if model in TEXT_ONLY_MODELS and args.allow_text_only_fallback else args.pre_multimodal_mode
    run_name = f"{args.run_prefix}_{safe_name(model)}_{'textonly' if mode == 'off' else 'mm'}"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--config",
        str(args.config),
        "--model",
        model,
        "--judge_model",
        args.judge_model,
        "--run_name",
        run_name,
        "--output_root",
        str(args.output_root),
        "--max_tokens",
        str(args.max_tokens),
        "--timeout",
        str(args.timeout),
        "--max_try",
        str(args.max_try),
        "--retry_delay",
        str(args.retry_delay),
        "--pre_multimodal_mode",
        mode,
        "--thinking_mode",
        args.thinking_mode,
        "--thinking_effort",
        args.thinking_effort,
        "--image_asset_mode",
        args.image_asset_mode,
    ]
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True, timeout=args.subprocess_timeout)
    run_dir = args.output_root / run_name
    summary_path = run_dir / "summary.json"
    summary = load_json(summary_path) if summary_path.exists() else {}
    return {
        "model": model,
        "pre_multimodal_mode": mode,
        "returncode": proc.returncode,
        "run_dir": str(run_dir.relative_to(ROOT) if run_dir.is_relative_to(ROOT) else run_dir),
        "summary_path": str(summary_path.relative_to(ROOT) if summary_path.is_relative_to(ROOT) else summary_path),
        "stdout_tail": proc.stdout[-2000:],
        "stderr_tail": proc.stderr[-2000:],
        "score": (summary.get("parsed_judge") or {}).get("total_score"),
        "max_score": (summary.get("parsed_judge") or {}).get("max_score"),
        "generation_error": summary.get("generation_error"),
        "generation_finish_reason": summary.get("generation_finish_reason"),
        "judge_finish_reason": summary.get("judge_finish_reason"),
        "available_image_count": summary.get("available_image_count"),
        "sent_image_count": summary.get("sent_image_count"),
        "cost_cny": summary.get("total_cost_cny"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--models", nargs="*", default=DEFAULT_MODELS)
    parser.add_argument("--judge_model", default="google/gemini-3.5-flash")
    parser.add_argument("--output_root", type=Path, default=ROOT / "artifacts/runs/candidate_benchmarks")
    parser.add_argument("--run_prefix", default="")
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--subprocess_timeout", type=float, default=2400.0)
    parser.add_argument("--max_try", type=int, default=2)
    parser.add_argument("--retry_delay", type=float, default=5.0)
    parser.add_argument("--pre_multimodal_mode", choices=["auto", "on", "off"], default="on")
    parser.add_argument("--allow_text_only_fallback", action="store_true")
    parser.add_argument("--thinking_mode", default="auto")
    parser.add_argument("--thinking_effort", default="auto")
    parser.add_argument("--image_asset_mode", choices=["all", "page_renders", "embedded"], default="all")
    parser.add_argument("--summary", type=Path, default=Path(""))
    args = parser.parse_args()

    if not args.run_prefix:
        problem_id = load_json(args.config).get("problem_id", "candidate")
        args.run_prefix = f"{problem_id}_batch_{datetime.now():%Y%m%d_%H%M%S}"
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_model = {executor.submit(run_one, args, model): model for model in args.models}
        for future in concurrent.futures.as_completed(future_to_model):
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                row = {"model": future_to_model[future], "error": str(exc)}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "judge_model": args.judge_model,
        "rows": sorted(rows, key=lambda row: row.get("model", "")),
        "total_cost_cny": sum(float(row.get("cost_cny") or 0) for row in rows),
    }
    summary_path = args.summary or args.output_root / f"{args.run_prefix}_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "total_cost_cny": payload["total_cost_cny"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
