#!/usr/bin/env python3
"""Resume one standard-exam interleaved sub-question from saved prior answers."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INTERLEAVED_DIR = ROOT / "new-things/interleaved"
sys.path.insert(0, str(INTERLEAVED_DIR))

from api_client_interleaved import (
    DEFAULT_API_URL,
    DEFAULT_HOST,
    UnifiedChatClient,
    build_user_message,
    clean_image_urls,
    resolve_multimodal_mode,
)  # noqa: E402
from image_policy import (  # noqa: E402
    extract_message_image_urls,
    resolve_generation_image_policy,
    resolve_standard_interleaved_background_images,
    resolve_standard_interleaved_turn_images,
    standard_question_image_categories,
)
from interleaved_common import (  # noqa: E402
    apply_interleaved_answer_style,
    build_standard_background_text,
    build_standard_subquestion_text,
    flatten_usage,
    get_standard_spec,
    load_json,
    save_json,
    timestamp_tag,
    usage_totals,
)
from run_arena_generate import _chat, _spec_for_generation, _with_system_prompt  # noqa: E402


DATA_ROOTS = {
    "ccho": ROOT / "data/science_arena/ccho2025",
    "cpho": ROOT / "data/science_arena/cpho2025",
    "icho": ROOT / "data/science_arena/icho2025",
}
DATA_FILES = {
    "ccho": "fix_CCHO-2025-latex.json",
    "cpho": "fix_CPHO-2025-latex.json",
    "icho": "fix_ICHO-2025-latex.json",
}


def load_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        data = load_json(path)
        if isinstance(data, list):
            rows.extend([row for row in data if isinstance(row, dict)])
    return rows


def row_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("question_id") or "").strip(), str(row.get("sub_question_id") or "").strip()


def answer_map(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    result: dict[tuple[str, str], str] = {}
    for row in rows:
        key = row_key(row)
        if not all(key):
            continue
        answer = str(row.get("answer") or row.get("llm_output") or "").strip()
        if answer and not row.get("error"):
            result[key] = answer
    return result


def response_to_row(response: dict[str, Any], answer: str) -> dict[str, Any]:
    return {
        "answer": answer,
        "llm_output": response.get("llm_output", "") or answer,
        "usage": flatten_usage(response.get("usage")),
        "error": response.get("error", ""),
        "finish_reason": response.get("finish_reason", ""),
        "client_meta": response.get("client_meta", {}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam", required=True, choices=sorted(DATA_ROOTS))
    parser.add_argument("--question_id", required=True)
    parser.add_argument("--sub_question_id", required=True)
    parser.add_argument("--prior_solution", type=Path, action="append", required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--max_try", type=int, default=2)
    parser.add_argument("--retry_delay", type=float, default=20)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--hard_timeout", type=float, default=1800)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--reasoning_effort", default="")
    parser.add_argument("--thinking_mode", default="auto")
    parser.add_argument("--thinking_effort", default="auto")
    parser.add_argument("--thinking_budget", type=int, default=None)
    parser.add_argument("--pre_multimodal_mode", default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--print_curl", action="store_true")
    parser.add_argument("--system_prompt", default="")
    parser.add_argument("--public_benchmark_context", action="store_true")
    parser.add_argument("--interleaved_answer_style", default="self_contained", choices=["self_contained", "contextual"])
    args = parser.parse_args()
    use_multimodal = resolve_multimodal_mode(model=args.model, mode=args.pre_multimodal_mode)

    data = load_json(DATA_ROOTS[args.exam] / DATA_FILES[args.exam])
    question = next(
        (row for row in data if str(row.get("question_id", [""])[0]).strip() == args.question_id),
        None,
    )
    if question is None:
        raise SystemExit(f"Question not found: {args.question_id}")

    sub_questions = list(question.get("detailed_question", []))
    target_index = None
    for index, sub_question in enumerate(sub_questions):
        sid = str(sub_question.get("sub_question_id", [""])[0]).strip()
        if sid == args.sub_question_id:
            target_index = index
            break
    if target_index is None:
        raise SystemExit(f"Sub-question not found: {args.sub_question_id}")

    prior_answers = answer_map(load_rows(args.prior_solution))
    spec = _spec_for_generation(args, get_standard_spec(args.exam))
    generation_policy = resolve_generation_image_policy(
        preset="unified",
        method="interleaved",
        exam=args.exam,
        iterative_image_mode="global_current",
        interleaved_global_image_replay="on",
        interleaved_replay_global_persist="off",
        interleaved_current_image_persist="on",
    )
    question_images = standard_question_image_categories(question)
    background_history_images = resolve_standard_interleaved_background_images(
        generation_policy,
        global_question_images=question_images["global_question_images"],
    )
    history: list[dict[str, Any]] = [
        build_user_message(
            text=build_standard_background_text(spec, question),
            image_urls=background_history_images if use_multimodal else [],
            use_multimodal=use_multimodal,
        )
    ]

    for sub_question in sub_questions[:target_index]:
        sid = str(sub_question.get("sub_question_id", [""])[0]).strip()
        current_images = standard_question_image_categories(question, sub_question)
        turn_plan = resolve_standard_interleaved_turn_images(
            generation_policy,
            global_question_images=question_images["global_question_images"],
            current_local_images=current_images["local_question_images"],
        )
        current_text = apply_interleaved_answer_style(
            build_standard_subquestion_text(sub_question),
            answer_style=args.interleaved_answer_style,
        )
        history.append(
            build_user_message(
                text=current_text,
                image_urls=turn_plan["persisted_image_urls"] if use_multimodal else [],
                use_multimodal=use_multimodal,
            )
        )
        answer = prior_answers.get((args.question_id, sid), "")
        if not answer:
            raise SystemExit(f"Missing prior answer for {args.question_id}:{sid}")
        history.append({"role": "assistant", "content": answer})

    target_sub = sub_questions[target_index]
    current_images = standard_question_image_categories(question, target_sub)
    turn_plan = resolve_standard_interleaved_turn_images(
        generation_policy,
        global_question_images=question_images["global_question_images"],
        current_local_images=current_images["local_question_images"],
    )
    current_text = apply_interleaved_answer_style(
        build_standard_subquestion_text(target_sub),
        answer_style=args.interleaved_answer_style,
    )
    call_user_message = build_user_message(
        text=current_text,
        image_urls=turn_plan["call_image_urls"] if use_multimodal else [],
        use_multimodal=use_multimodal,
    )
    history_image_urls = extract_message_image_urls(history)
    call_messages = copy.deepcopy(history)
    call_messages.append(call_user_message)
    call_messages = _with_system_prompt(args, call_messages)

    client = UnifiedChatClient(
        api_key=os.environ.get("QIHOO_API_KEY") or os.environ.get("QIHOO360_API_KEY") or os.environ.get("360_API_KEY") or os.environ.get("API_KEY") or "",
        backend="requests",
        api_url=args.api_url,
        host=args.host,
        max_retries=args.max_try,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
        model_tag=args.model,
    )
    response = _chat(client, args, model=args.model, messages=call_messages)
    answer = response.get("content", "") or response.get("llm_output", "")

    prompt_row = {
        "question_id": args.question_id,
        "sub_question_id": args.sub_question_id,
        "unit_type": "sub_question",
        "messages": call_messages,
        "call_image_urls": extract_message_image_urls(call_messages),
        "history_image_urls": history_image_urls,
        "background_history_image_urls": background_history_images,
        "current_turn_image_urls": turn_plan["call_image_urls"],
        "persisted_turn_image_urls": turn_plan["persisted_image_urls"],
        "replay_global_image_urls": turn_plan["replay_global_image_urls"],
        "global_question_image_urls": question_images["global_question_images"],
        "current_local_image_urls": current_images["local_question_images"],
        "image_policy": generation_policy.to_dict(),
        "resume_prior_solution_paths": [str(path) for path in args.prior_solution],
    }
    solution_row = {
        "question_id": args.question_id,
        "sub_question_id": args.sub_question_id,
        **response_to_row(response, answer),
    }

    run_dir = args.output_root / args.run_name / "interleaved" / args.exam
    save_json(run_dir / "prompts.json", [prompt_row])
    save_json(run_dir / "solutions.json", [solution_row])
    save_json(
        run_dir / "summary.json",
        {
            "generation": {
                "run_name": args.run_name,
                "method": "interleaved",
                "exam": args.exam,
                "unit_count": 1,
                "prompt_count": 1,
                "generated_at": timestamp_tag(),
                "resumed_from_prior_answers": True,
                "target": f"{args.question_id}:{args.sub_question_id}",
                **usage_totals([solution_row]),
            }
        },
    )
    print(f"[DONE] wrote {run_dir}")
    print(f"[DONE] call_images={len(prompt_row['call_image_urls'])} history_images={len(history_image_urls)}")


if __name__ == "__main__":
    main()
