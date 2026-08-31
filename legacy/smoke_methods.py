# -*- coding: utf-8 -*-
"""Closed-loop smoke test for iterative/interleaved/full on a tiny subset."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from api_client_interleaved import DEFAULT_API_URL, DEFAULT_HOST
from interleaved_common import (
    build_invocation_metadata,
    build_run_dir,
    default_smoke_root,
    load_json,
    namespace_to_dict,
    redact_argv,
    redact_sensitive_fields,
    save_json,
    timestamp_tag,
)


REPO_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--exam", type=str, default="cpho", choices=["ccho", "cpho", "icho", "ipho"])
    parser.add_argument("--question_num_index", type=str, default="1")
    parser.add_argument("--pre_model", type=str, default="deepseek-v3.2")
    parser.add_argument("--judge_model", type=str, default="deepseek-v3.2")
    parser.add_argument("--api_backend", type=str, default="requests", choices=["openai", "requests"])
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--api_host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--pre_multimodal_mode", type=str, default="off", choices=["auto", "on", "off"])
    parser.add_argument("--judge_multimodal_mode", type=str, default="off", choices=["auto", "on", "off"])
    parser.add_argument("--pre_thinking_mode", type=str, default="on", choices=["auto", "on", "off"])
    parser.add_argument(
        "--pre_thinking_effort",
        type=str,
        default="auto",
        choices=["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--pre_thinking_budget", type=int, default=None)
    parser.add_argument("--judge_thinking_mode", type=str, default="on", choices=["auto", "on", "off"])
    parser.add_argument(
        "--judge_thinking_effort",
        type=str,
        default="auto",
        choices=["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--judge_thinking_budget", type=int, default=None)
    parser.add_argument("--reasoning_effort", type=str, default="")
    parser.add_argument("--question_ids", type=str, default="3")
    parser.add_argument("--item_indices", type=str, default="0")
    parser.add_argument("--part_ids", type=str, default="A")
    parser.add_argument(
        "--generation_image_preset",
        type=str,
        default="unified",
        choices=["unified", "legacy"],
    )
    parser.add_argument(
        "--judge_image_preset",
        type=str,
        default="reference_only",
        choices=["reference_only", "question_plus_reference"],
    )
    parser.add_argument(
        "--iterative_image_mode",
        type=str,
        default="inherit",
        choices=["inherit", "current_only", "global_current", "cumulative"],
    )
    parser.add_argument(
        "--interleaved_global_image_replay",
        type=str,
        default="inherit",
        choices=["inherit", "on", "off"],
    )
    parser.add_argument(
        "--interleaved_replay_global_persist",
        type=str,
        default="inherit",
        choices=["inherit", "on", "off"],
    )
    parser.add_argument(
        "--interleaved_answer_style",
        type=str,
        default="self_contained",
        choices=["self_contained", "contextual"],
    )
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--max_try", type=int, default=3)
    parser.add_argument("--retry_delay", type=int, default=10)
    parser.add_argument("--output_root", type=str, default="", help="Defaults to artifacts/smoke/methods/<timestamp>/runs")
    parser.add_argument("--report_path", type=str, default="")
    parser.add_argument("--timeout_sec", type=int, default=1800)
    return parser.parse_args()


def run_cmd(cmd: List[str], *, timeout_sec: int) -> Dict[str, Any]:
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO_DIR),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
    )
    output_chunks: List[str] = []

    def _reader() -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            output_chunks.append(line)
            print(line, end="", flush=True)

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()
    try:
        returncode = proc.wait(timeout=timeout_sec)
    except subprocess.TimeoutExpired:
        proc.kill()
        reader_thread.join(timeout=2)
        output = "".join(output_chunks)
        return {
            "ok": False,
            "returncode": -1,
            "stdout": output,
            "stderr": f"Timed out after {timeout_sec} seconds",
        }

    reader_thread.join(timeout=2)
    output = "".join(output_chunks)
    return {
        "ok": returncode == 0,
        "returncode": returncode,
        "stdout": output,
        "stderr": "",
    }


def snapshot_run_artifacts(run_dir: Path, *, snapshot_dir: Path) -> List[str]:
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    copied: List[str] = []
    for name in (
        "config.json",
        "prompts.json",
        "solutions.json",
        "judge_prompts.json",
        "judge_results.json",
        "scores.json",
        "summary.json",
    ):
        src = run_dir / name
        if not src.exists():
            continue
        dst = snapshot_dir / name
        shutil.copy2(src, dst)
        copied.append(str(dst))
    return copied


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    invocation = build_invocation_metadata(argv=sys.argv, entrypoint=__file__)
    smoke_root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else default_smoke_root() / "methods" / timestamp_tag() / "runs"
    )
    report_path = (
        Path(args.report_path).expanduser().resolve()
        if args.report_path
        else smoke_root.parent / "report.json"
    )
    snapshot_root = report_path.parent / f"{report_path.stem}_snapshots"
    if args.output_root and smoke_root.exists():
        print(
            f"[WARN] output_root already exists: {smoke_root}. Live run artifacts may be overwritten; report snapshots will be preserved under {snapshot_root}",
            flush=True,
        )

    methods = ["iterative", "interleaved", "full"]
    report_rows: List[Dict[str, Any]] = []
    for method in methods:
        run_name = f"smoke_{args.exam}_{method}"
        common_args = [
            f"--exam={args.exam}",
            f"--method={method}",
            f"--run_name={run_name}",
            f"--output_root={smoke_root}",
            f"--api_backend={args.api_backend}",
            f"--api_url={args.api_url}",
            f"--api_host={args.api_host}",
            f"--api_key={args.api_key}",
            f"--max_try={args.max_try}",
            f"--retry_delay={args.retry_delay}",
            f"--max_tokens={args.max_tokens}",
            f"--limit={args.limit}",
        ]
        if args.exam == "ipho":
            common_args.append(f"--question_num_index={args.question_num_index}")
            if method == "full":
                common_args.append(f"--part_ids={args.part_ids}")
            else:
                common_args.append(f"--item_indices={args.item_indices}")
        else:
            common_args.append(f"--question_ids={args.question_ids}")

        gen_cmd = [
            sys.executable,
            "run_arena_generate.py",
            *common_args,
            f"--pre_model={args.pre_model}",
            f"--pre_multimodal_mode={args.pre_multimodal_mode}",
            f"--generation_image_preset={args.generation_image_preset}",
            f"--iterative_image_mode={args.iterative_image_mode}",
            f"--interleaved_global_image_replay={args.interleaved_global_image_replay}",
            f"--interleaved_replay_global_persist={args.interleaved_replay_global_persist}",
            f"--thinking_mode={args.pre_thinking_mode}",
            f"--thinking_effort={args.pre_thinking_effort}",
            f"--reasoning_effort={args.reasoning_effort}",
        ]
        if method == "interleaved":
            gen_cmd.append(f"--interleaved_answer_style={args.interleaved_answer_style}")
        if args.pre_thinking_budget is not None:
            gen_cmd.append(f"--thinking_budget={args.pre_thinking_budget}")

        eval_cmd = [
            sys.executable,
            "run_arena_eval.py",
            *common_args,
            f"--judge_model={args.judge_model}",
            f"--judge_multimodal_mode={args.judge_multimodal_mode}",
            f"--judge_image_preset={args.judge_image_preset}",
            f"--thinking_mode={args.judge_thinking_mode}",
            f"--thinking_effort={args.judge_thinking_effort}",
            f"--reasoning_effort={args.reasoning_effort}",
            "--allow_flexible_score_parsing=true",
            "--eval_context_mode=answer_only",
        ]
        if args.judge_thinking_budget is not None:
            eval_cmd.append(f"--thinking_budget={args.judge_thinking_budget}")

        print(f"[SMOKE] method={method} generation", flush=True)
        gen_result = run_cmd(gen_cmd, timeout_sec=args.timeout_sec)
        print(f"  generation ok={gen_result['ok']} returncode={gen_result['returncode']}", flush=True)

        print(f"[SMOKE] method={method} evaluation", flush=True)
        eval_result = run_cmd(eval_cmd, timeout_sec=args.timeout_sec)
        print(f"  evaluation ok={eval_result['ok']} returncode={eval_result['returncode']}", flush=True)

        run_dir = build_run_dir(
            run_name=run_name,
            method=method,
            exam=args.exam,
            output_root=smoke_root,
            question_num_index=args.question_num_index,
        )
        scores_path = run_dir / "scores.json"
        scores = load_json(scores_path) if scores_path.exists() else {}
        snapshot_dir = snapshot_root / method
        snapshot_files = snapshot_run_artifacts(run_dir, snapshot_dir=snapshot_dir)

        report_rows.append(
            {
                "method": method,
                "run_name": run_name,
                "run_dir": str(run_dir),
                "snapshot_dir": str(snapshot_dir),
                "snapshot_files": snapshot_files,
                "generation": gen_result,
                "evaluation": eval_result,
                "generation_cmd": redact_argv(gen_cmd),
                "evaluation_cmd": redact_argv(eval_cmd),
                "scores_path": str(scores_path),
                "scores": scores,
            }
        )

    report = {
        **invocation,
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": redact_sensitive_fields(namespace_to_dict(args)),
        "exam": args.exam,
        "question_num_index": args.question_num_index,
        "api_backend": args.api_backend,
        "pre_model": args.pre_model,
        "judge_model": args.judge_model,
        "smoke_root": str(smoke_root),
        "snapshot_root": str(snapshot_root),
        "rows": report_rows,
    }
    save_json(report_path, report)
    print(f"[DONE] report written to: {report_path}", flush=True)

    if not all(row["generation"]["ok"] and row["evaluation"]["ok"] for row in report_rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
