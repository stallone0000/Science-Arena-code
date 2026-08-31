#!/usr/bin/env python3
"""Continue a text-only interleaved sequence from a clean partial prefix."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INTERLEAVED_DIR = ROOT / "new-things/interleaved"
if str(INTERLEAVED_DIR) not in sys.path:
    sys.path.insert(0, str(INTERLEAVED_DIR))

from api_client_interleaved import UnifiedChatClient  # noqa: E402
from run_text_interleaved_eval import (  # noqa: E402
    DEFAULT_ENV_FILE,
    DEFAULT_PRICES,
    build_judge_prompt,
    build_solver_prompt,
    extract_json_object,
    is_bad_finish_reason,
    iter_jsonl,
    load_env_file,
    load_json,
    row_key,
    safe_model_name,
    score_objective,
    usage_cost,
    write_jsonl,
)


def is_clean_result(row: dict[str, Any]) -> bool:
    return not (
        row.get("generation_error")
        or row.get("judge_error")
        or (row.get("judge_result") or {}).get("parse_error")
        or is_bad_finish_reason(row.get("generation_finish_reason"))
        or is_bad_finish_reason(row.get("judge_finish_reason"))
        or not str(row.get("candidate_answer") or "").strip()
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=ROOT / "datasets/science_arena_text_interleaved_v1")
    parser.add_argument("--partial_jsonl", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, default=ROOT / "artifacts/runs/text_interleaved_eval")
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--request_model", default="")
    parser.add_argument("--judge_model", default="google/gemini-3.5-flash")
    parser.add_argument("--sequence_id", required=True)
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--judge_max_tokens", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--solver_prompt_variant",
        choices=["standard", "answer_first", "final_only"],
        default="standard",
        help="Prompt style for newly generated turns. Non-standard variants are diagnostic and should not be merged silently.",
    )
    parser.add_argument(
        "--prefix_solver_prompt_variant",
        choices=["standard", "answer_first", "final_only"],
        default="standard",
        help="Prompt style used when replaying the clean prefix from an existing partial run.",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--max_retries", type=int, default=1)
    parser.add_argument("--retry_delay", type=float, default=60.0)
    parser.add_argument("--thinking_mode", default="auto")
    parser.add_argument("--thinking_effort", default="auto")
    parser.add_argument("--judge_thinking_mode", default="on")
    parser.add_argument("--judge_thinking_effort", default="high")
    args = parser.parse_args()

    os.environ.update(load_env_file(args.env_file))
    api_key = (
        os.environ.get("QIHOO_API_KEY")
        or os.environ.get("QIHOO360_API_KEY")
        or os.environ.get("360_API_KEY")
        or os.environ.get("API_KEY")
    )
    if not api_key:
        raise SystemExit("Missing 360 API key.")

    dataset_rows = [
        row
        for row in iter_jsonl(args.dataset_dir / "benchmark_items.jsonl")
        if str(row.get("sequence_id")) == args.sequence_id
    ]
    if not dataset_rows:
        raise SystemExit(f"Unknown sequence_id: {args.sequence_id}")
    dataset_rows = sorted(dataset_rows, key=lambda item: int(item.get("turn_index") or 0))
    by_key = {row_key(row): row for row in dataset_rows}

    partial_rows = [row for row in iter_jsonl(args.partial_jsonl) if str(row.get("sequence_id")) == args.sequence_id]
    partial_rows = sorted(partial_rows, key=lambda item: int(item.get("turn_index") or 0))
    prefix: list[dict[str, Any]] = []
    for result in partial_rows:
        if not is_clean_result(result):
            break
        prefix.append(result)

    messages: list[dict[str, str]] = []
    for result in prefix:
        source_row = by_key.get(row_key(result))
        if not source_row:
            raise SystemExit(f"Partial row not in dataset: {row_key(result)}")
        messages.append({"role": "user", "content": build_solver_prompt(source_row, args.prefix_solver_prompt_variant)})
        messages.append({"role": "assistant", "content": str(result.get("candidate_answer") or "")})

    prices = load_json(DEFAULT_PRICES, {})
    client = UnifiedChatClient(
        api_key=api_key,
        backend="requests",
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
    )
    request_model = args.request_model or args.model
    outputs = list(prefix)
    evaluation_protocol = (
        "text_only_interleaved_staticized_v1_prefix_continuation"
        if args.solver_prompt_variant == "standard"
        else f"text_only_interleaved_staticized_v1_prefix_continuation_{args.solver_prompt_variant}_diagnostic"
    )
    start_index = len(prefix)
    for source_row in dataset_rows[start_index:]:
        prompt = build_solver_prompt(source_row, args.solver_prompt_variant)
        user_message = {"role": "user", "content": prompt}
        started = time.time()
        generation_response = client.chat(
            model=request_model,
            messages=messages + [user_message],
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            thinking_mode=args.thinking_mode,
            thinking_effort=args.thinking_effort,
        )
        elapsed = time.time() - started
        answer = generation_response.get("content") or generation_response.get("llm_output") or ""
        generation_usage = usage_cost(generation_response.get("usage", {}) or {}, request_model, prices)
        if not generation_usage["cost_cny"] and request_model != args.model:
            generation_usage = usage_cost(generation_response.get("usage", {}) or {}, args.model, prices)
        generation_finish_reason = generation_response.get("finish_reason")
        generation_error = str(generation_response.get("error") or "")
        if is_bad_finish_reason(generation_finish_reason) and not generation_error:
            generation_error = f"bad_finish_reason:{generation_finish_reason}"

        if generation_error or not answer.strip():
            judge_response = {"finish_reason": "generation_failed", "usage": {}, "client_meta": {}}
            judge_result = {
                "parse_error": "generation_failed",
                "generation_error": generation_error or "empty_generation",
                "total_score": 0.0,
                "max_score": float(source_row.get("max_score") or 0),
            }
            judge_usage = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0, "cost_cny": 0.0}
        else:
            objective = score_objective(source_row, answer)
            if objective is not None:
                judge_response = {"finish_reason": "objective_key", "usage": {}, "client_meta": {}}
                judge_result = objective
                judge_usage = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0, "cost_cny": 0.0}
            else:
                judge_response = client.chat(
                    model=args.judge_model,
                    messages=[{"role": "user", "content": build_judge_prompt(source_row, answer)}],
                    max_tokens=args.judge_max_tokens,
                    thinking_mode=args.judge_thinking_mode,
                    thinking_effort=args.judge_thinking_effort,
                )
                judge_text = judge_response.get("content") or judge_response.get("llm_output") or ""
                judge_result = extract_json_object(judge_text)
                judge_usage = usage_cost(judge_response.get("usage", {}) or {}, args.judge_model, prices)
            if is_bad_finish_reason(judge_response.get("finish_reason")) and not judge_response.get("error"):
                judge_response["error"] = f"bad_finish_reason:{judge_response.get('finish_reason')}"
                judge_result = {
                    "parse_error": "judge_bad_finish_reason",
                    "judge_error": judge_response["error"],
                    "total_score": 0.0,
                    "max_score": float(source_row.get("max_score") or 0),
                }

        result = {
            "benchmark_id": source_row.get("benchmark_id"),
            "sequence_id": source_row.get("sequence_id"),
            "turn_id": source_row.get("turn_id"),
            "turn_index": source_row.get("turn_index"),
            "model": args.model,
            "request_model": request_model,
            "judge_model": args.judge_model,
            "evaluation_protocol": evaluation_protocol,
            "solver_prompt_variant": args.solver_prompt_variant,
            "prefix_solver_prompt_variant": args.prefix_solver_prompt_variant,
            "generation_finish_reason": generation_finish_reason,
            "generation_error": generation_error,
            "generation_elapsed_seconds": elapsed,
            "generation_usage": generation_usage,
            "generation_client_meta": generation_response.get("client_meta", {}),
            "generation_sampling": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
            },
            "candidate_answer": answer,
            "judge_finish_reason": judge_response.get("finish_reason"),
            "judge_error": judge_response.get("error", ""),
            "judge_usage": judge_usage,
            "judge_result": judge_result,
            "score": judge_result.get("total_score"),
            "max_score": judge_result.get("max_score") or source_row.get("max_score"),
            "total_cost_cny": generation_usage["cost_cny"] + judge_usage["cost_cny"],
            "timeout_label": str(args.timeout),
        }
        outputs.append(result)
        out_dir = args.output_root / args.run_name / safe_model_name(args.model)
        write_jsonl(out_dir / "partial_sequences" / f"{args.sequence_id}.jsonl", outputs)
        if not is_clean_result(result):
            break
        messages.append(user_message)
        messages.append({"role": "assistant", "content": answer})

    out_dir = args.output_root / args.run_name / safe_model_name(args.model)
    write_jsonl(out_dir / "turn_results.jsonl", outputs)
    print(json.dumps({"run_dir": str(out_dir.parent), "model": args.model, "sequence_id": args.sequence_id, "rows": len(outputs)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
