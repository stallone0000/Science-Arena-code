#!/usr/bin/env python3
"""Rejudge a candidate full-exam run when the first judge focused on one problem."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
INTERLEAVED_DIR = ROOT / "new-things/interleaved"
if str(INTERLEAVED_DIR) not in sys.path:
    sys.path.insert(0, str(INTERLEAVED_DIR))

from api_client_interleaved import UnifiedChatClient  # noqa: E402
from run_candidate_problem_smoke import (  # noqa: E402
    DEFAULT_ENV_FILE,
    DEFAULT_PRICES,
    api_key_from_env,
    extract_json_object,
    load_env_file,
    load_json,
    page_text_blob,
    save_json,
    selected_pages,
    usage_cost,
)


def build_strict_prompt(config: dict[str, Any], question_text: str, solution_text: str, candidate_answer: str) -> str:
    total = config.get("total_points")
    problem_points = config.get("problem_points") or [10, 7, 6, 12, 6, 4, 6, 9, 12]
    return (
        "You are a strict olympiad examiner. The previous judge may have focused on only one problem; "
        "do not do that. Grade the entire submitted answer across every problem in the full exam.\n\n"
        f"Required full-exam max_score: {total}.\n"
        f"Required top-level problem point totals, in order: {problem_points}.\n"
        "If the candidate omits a problem or subpart, award 0 for that part. "
        "Do not invent experimental measurements. Use only the official solution/rubric text. "
        "Return exactly one valid JSON object with keys: total_score, max_score, sub_scores, evidence, deductions. "
        f"The JSON max_score must be exactly {total}, and sub_scores must contain top-level keys for all problems.\n\n"
        + json.dumps(
            {
                "benchmark_id": config.get("benchmark_id"),
                "problem_id": config.get("problem_id"),
                "title": config.get("title"),
                "total_points": total,
                "problem_points": problem_points,
                "official_problem_text": question_text,
                "official_solution_and_marking_scheme": solution_text,
                "candidate_answer": candidate_answer,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", type=Path, required=True)
    parser.add_argument("--judge_model", default="google/gemini-3.5-flash")
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=2400.0)
    parser.add_argument("--max_try", type=int, default=2)
    parser.add_argument("--retry_delay", type=float, default=5.0)
    parser.add_argument("--suffix", default="rejudged_full_exam")
    args = parser.parse_args()

    os.environ.update(load_env_file(args.env_file))
    run_dir = args.run_dir.resolve()
    summary = load_json(run_dir / "summary.json", {})
    config = load_json(run_dir / "config.json", {})
    generation_response = load_json(run_dir / "generation_response.json", {})
    candidate_answer = generation_response.get("content") or generation_response.get("llm_output") or ""
    if not candidate_answer.strip():
        raise SystemExit("No candidate answer found in generation_response.json.")

    question_pages_json_path = ROOT / config["source_files"]["question_pages_json"]
    solution_pages_json_path = ROOT / config["source_files"]["solution_pages_json"]
    question_pages = selected_pages(load_json(question_pages_json_path, {}), config.get("question_pages", []))
    solution_pages = selected_pages(load_json(solution_pages_json_path, {}), config.get("solution_pages", []))
    question_text = page_text_blob(question_pages, "question")
    solution_text = page_text_blob(solution_pages, "solution")

    client = UnifiedChatClient(
        api_key=api_key_from_env(),
        backend="requests",
        max_retries=args.max_try,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
    )
    prices = load_json(DEFAULT_PRICES, {})
    prompt = build_strict_prompt(config, question_text, solution_text, candidate_answer)
    response = client.chat(
        model=args.judge_model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        thinking_mode="on",
        thinking_effort="high",
    )
    judge_text = response.get("content") or response.get("llm_output") or ""
    parsed = extract_json_object(judge_text)
    judge_usage = usage_cost(response.get("usage", {}) or {}, args.judge_model, prices)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_summary": str((run_dir / "summary.json").relative_to(ROOT)),
        "model": summary.get("model"),
        "judge_model": args.judge_model,
        "expected_max_score": config.get("total_points"),
        "judge_finish_reason": response.get("finish_reason"),
        "parsed_judge": parsed,
        "judge_usage": judge_usage,
        "total_cost_cny": judge_usage["cost_cny"],
    }
    save_json(run_dir / f"summary.{args.suffix}.json", payload)
    save_json(run_dir / f"judge_response.{args.suffix}.json", response)
    save_json(run_dir / f"judge_prompt.{args.suffix}.json", {"judge": prompt})
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
