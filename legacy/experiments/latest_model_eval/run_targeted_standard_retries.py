#!/usr/bin/env python3
"""Run standard-exam generation retries one sub-question at a time.

This wrapper is for unstable API routes where a single request can hang beyond
the client timeout. It scopes each subprocess to one (question_id,
sub_question_id) and wraps it in the system `timeout` command, so one stuck
sub-question cannot block the rest of a recovery run.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
GEN_SCRIPT = ROOT / "new-things/interleaved/run_arena_generate.py"
DEFAULT_ENV_FILE = Path("/home/jovyan/zhaoguangxiang-data-zzzc/zhaoguangxiang/.config/science_arena/360.env")
DATA_ROOTS = {
    "ccho": ROOT / "data/science_arena/ccho2025",
    "cpho": ROOT / "data/science_arena/cpho2025",
    "icho": ROOT / "data/science_arena/icho2025",
}


def parse_cases(raw: str) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []
    for item in re.split(r"[\s,]+", raw.strip()):
        if not item:
            continue
        if ":" not in item:
            raise ValueError(f"Case must be qid:sid, got: {item}")
        qid, sid = item.split(":", 1)
        cases.append((qid.strip(), sid.strip()))
    return cases


def parse_cases_file(path: Path) -> list[tuple[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Cases file not found: {path}")
    return parse_cases(path.read_text(encoding="utf-8"))


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_")


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def run_case(args: argparse.Namespace, qid: str, sid: str, env: dict[str, str]) -> dict[str, Any]:
    run_name = f"{args.run_prefix}__q{slug(qid)}__s{slug(sid)}"
    run_dir = args.output_root / run_name / args.method / args.exam
    solution_path = run_dir / "solutions.json"
    partial_solution_path = run_dir / "solutions.partial.json"
    log_path = args.output_root / "_logs" / f"{run_name}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "timeout",
        "--kill-after=10s",
        f"{int(args.wall_timeout)}s",
        sys.executable,
        str(GEN_SCRIPT),
        "--exam",
        args.exam,
        "--method",
        args.method,
        "--run_name",
        run_name,
        "--data_root",
        str(DATA_ROOTS[args.exam]),
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
        str(args.api_timeout),
        "--question_ids",
        qid,
        "--sub_question_ids",
        sid,
    ]
    if args.hard_timeout:
        cmd.extend(["--hard_timeout", str(args.hard_timeout)])
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

    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        completed = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        log.write(f"[exit_code] {completed.returncode}\n")

    rows = []
    if solution_path.exists():
        rows = json.loads(solution_path.read_text(encoding="utf-8"))
    elif partial_solution_path.exists():
        rows = json.loads(partial_solution_path.read_text(encoding="utf-8"))
    errors = [row for row in rows if isinstance(row, dict) and row.get("error")]
    return {
        "question_id": qid,
        "sub_question_id": sid,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "solution_path": str(solution_path if solution_path.exists() else partial_solution_path),
        "returncode": completed.returncode,
        "rows": len(rows),
        "errors": len(errors),
        "log_path": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam", required=True, choices=sorted(DATA_ROOTS))
    parser.add_argument("--cases", default="", help="Comma/space-separated qid:sid cases.")
    parser.add_argument("--cases_file", type=Path, default=None, help="Optional text file containing qid:sid cases.")
    parser.add_argument("--model", default="deepseek/deepseek-v4-pro")
    parser.add_argument("--method", default="interleaved", choices=["iterative", "interleaved"])
    parser.add_argument("--run_prefix", required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--pre_multimodal_mode", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--thinking_mode", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--thinking_effort", default="auto")
    parser.add_argument("--system_prompt", default="")
    parser.add_argument("--public_benchmark_context", action="store_true")
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--max_try", type=int, default=1)
    parser.add_argument("--retry_delay", type=float, default=5)
    parser.add_argument("--api_timeout", type=float, default=240)
    parser.add_argument("--hard_timeout", type=float, default=0)
    parser.add_argument("--wall_timeout", type=float, default=330)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--interleaved_current_image_persist", default="inherit", choices=["inherit", "on", "off"])
    parser.add_argument("--interleaved_single_user_image_turns", default="inherit", choices=["inherit", "on", "off"])
    parser.add_argument("--max_images_per_turn", type=int, default=0)
    args = parser.parse_args()

    env = os.environ.copy()
    env.update(load_env_file(args.env_file))
    if not any(env.get(name) for name in ("QIHOO_API_KEY", "QIHOO360_API_KEY", "360_API_KEY", "API_KEY")):
        raise RuntimeError(f"Missing API key. Source env or provide --env_file. Not found: {args.env_file}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    cases: list[tuple[str, str]] = []
    if args.cases_file:
        cases.extend(parse_cases_file(args.cases_file))
    if args.cases:
        cases.extend(parse_cases(args.cases))
    if not cases:
        raise ValueError("No targeted retry cases provided. Use --cases or --cases_file.")
    summary: dict[str, Any] = {
        "created_at": datetime.now().strftime("%Y%m%d-%H%M%S"),
        "exam": args.exam,
        "model": args.model,
        "run_prefix": args.run_prefix,
        "output_root": str(args.output_root),
        "case_count": len(cases),
        "results": [],
    }
    summary_path = args.output_root / "targeted_retry_summary.json"
    save_json(summary_path, summary)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_case, args, qid, sid, env) for qid, sid in cases]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            summary["results"].append(result)
            save_json(summary_path, summary)
            print(
                f"[DONE] {args.exam} {result['question_id']}:{result['sub_question_id']} "
                f"rc={result['returncode']} rows={result['rows']} errors={result['errors']}"
            )

    save_json(summary_path, summary)
    print(f"[SUMMARY] {summary_path}")


if __name__ == "__main__":
    main()
