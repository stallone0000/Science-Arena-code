# -*- coding: utf-8 -*-
"""Unified evaluation runner for ScienceArena iterative/interleaved/full methods."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from api_client_interleaved import (
    DEFAULT_API_URL,
    DEFAULT_HOST,
    UnifiedChatClient,
    build_user_message,
    parse_bool,
    resolve_multimodal_mode,
)
from interleaved_common import (
    EVAL_PROMPT6_STANDARD,
    EVAL_PROMPT7_STANDARD,
    EVAL_PROMPT_SUB_SOLUTION,
    IPHO_EVAL_PROMPT5,
    IPHO_EVAL_PROMPT6,
    build_ipho_full_reference,
    build_invocation_metadata,
    build_run_dir,
    build_standard_full_reference,
    build_standard_question_map,
    build_standard_reference_map,
    coerce_answer_from_solution_row,
    compute_standard_weighted_score,
    extract_boxed_score,
    filter_ipho_solution_rows,
    filter_standard_solution_rows,
    flatten_usage,
    get_standard_spec,
    group_ipho_parts,
    load_json,
    maybe_load_json,
    model_tag_basename,
    namespace_to_dict,
    parse_item_indices,
    parse_ipho_item,
    parse_part_ids,
    parse_question_ids,
    resolve_data_root,
    redact_sensitive_fields,
    save_json,
    timestamp_tag,
    update_named_json,
)
from image_policy import (
    extract_message_image_urls,
    extract_user_only_messages,
    ipho_part_reference_images,
    resolve_judge_image_policy,
    standard_full_reference_images,
)


def _build_parser(default_method: Optional[str], default_exam: Optional[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exam", type=str, default=default_exam or "ccho", choices=["ccho", "cpho", "icho", "ipho"])
    parser.add_argument(
        "--method",
        type=str,
        default=default_method or "interleaved",
        choices=["iterative", "interleaved", "full"],
    )
    parser.add_argument("--question_num_index", type=str, default="1", help="Only used for IPHO.")
    parser.add_argument("--run_name", type=str, default="", help="Primary run identifier used for output layout.")
    parser.add_argument("--data_root", type=str, default="", help="Optional data root; defaults to interleaved/data.")
    parser.add_argument("--output_root", type=str, default="", help="Optional output root; defaults to artifacts/runs.")

    parser.add_argument("--pre_model_save_tag", type=str, default="")
    parser.add_argument("--judge_model", type=str, default="google/gemini-3-pro-preview")

    parser.add_argument("--api_backend", type=str, default="openai", choices=["openai", "requests"])
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--api_host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--api_key", type=str, default="")

    parser.add_argument("--max_try", type=int, default=5)
    parser.add_argument("--retry_delay", type=int, default=60)
    parser.add_argument("--max_tokens", type=int, default=81920)

    parser.add_argument(
        "--judge_multimodal_mode",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Judge-model multimodal policy: auto by model capability, or force on/off.",
    )
    parser.add_argument(
        "--is_judge_multi",
        type=parse_bool,
        default=None,
        help="Legacy override for judge multimodal; if set, overrides --judge_multimodal_mode.",
    )
    parser.add_argument(
        "--eval_context_mode",
        type=str,
        default="answer_only",
        choices=["answer_only", "interleaved"],
        help="Judge context mode. interleaved only applies to interleaved generation history.",
    )

    parser.add_argument("--thinking_mode", type=str, default="on", choices=["auto", "on", "off"])
    parser.add_argument(
        "--thinking_effort",
        type=str,
        default="auto",
        choices=["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--thinking_budget", type=int, default=None)
    parser.add_argument("--reasoning_effort", type=str, default="")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)

    parser.add_argument("--question_ids", type=str, default="", help="Comma-separated standard-exam question IDs.")
    parser.add_argument("--item_indices", type=str, default="", help="Comma-separated IPHO item indices.")
    parser.add_argument("--part_ids", type=str, default="", help="Comma-separated IPHO part IDs for full mode.")
    parser.add_argument("--limit", type=int, default=0, help="Optional hard limit after explicit slicing.")
    parser.add_argument("--input_solution_path", type=str, default="")
    parser.add_argument(
        "--judge_image_preset",
        type=str,
        default="reference_only",
        choices=["reference_only", "question_plus_reference"],
        help="Judge-side image policy preset.",
    )
    parser.add_argument(
        "--allow_flexible_score_parsing",
        type=parse_bool,
        default=False,
        help="Default is strict boxed-only parsing. Enable only for smoke/debug route checks.",
    )

    parser.add_argument("--print_curl", type=parse_bool, default=False)
    parser.add_argument("--verbose", type=parse_bool, default=False)
    return parser


def _chat(
    client: UnifiedChatClient,
    args: argparse.Namespace,
    *,
    model: str,
    messages: List[Dict[str, Any]],
) -> Dict[str, Any]:
    return client.chat(
        model=model,
        messages=messages,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        thinking_mode=args.thinking_mode,
        thinking_effort=args.thinking_effort,
        thinking_budget=args.thinking_budget,
        reasoning_effort=(args.reasoning_effort or None),
        print_curl=args.print_curl,
    )


def _load_solution_rows(args: argparse.Namespace, run_dir: Path) -> List[Any]:
    if args.input_solution_path:
        return load_json(Path(args.input_solution_path).expanduser().resolve())
    return load_json(run_dir / "solutions.json")


def _load_interleaved_history(
    args: argparse.Namespace,
    run_dir: Path,
) -> Dict[Any, List[Dict[str, Any]]]:
    if args.eval_context_mode != "interleaved" or args.method != "interleaved":
        return {}
    prompt_rows = maybe_load_json(run_dir / "prompts.json", [])
    history_map: Dict[Any, List[Dict[str, Any]]] = {}
    for item in prompt_rows:
        if not isinstance(item, dict) or "messages" not in item:
            continue
        if args.exam == "ipho":
            if "item_index" in item:
                history_map[int(item["item_index"])] = item["messages"]
        else:
            key = (str(item.get("question_id", "")).strip(), str(item.get("sub_question_id", "")).strip())
            history_map[key] = item["messages"]
    return history_map


def _load_prompt_context_map(
    run_dir: Path,
) -> Dict[Any, Dict[str, Any]]:
    prompt_rows = maybe_load_json(run_dir / "prompts.json", [])
    context_map: Dict[Any, Dict[str, Any]] = {}
    for item in prompt_rows:
        if not isinstance(item, dict):
            continue
        if "item_index" in item:
            context_map[int(item["item_index"])] = item
        elif "part_id" in item:
            context_map[str(item.get("part_id", "")).upper()] = item
        elif item.get("sub_question_id") is not None:
            key = (str(item.get("question_id", "")).strip(), str(item.get("sub_question_id", "")).strip())
            context_map[key] = item
        elif item.get("question_id") is not None:
            context_map[str(item.get("question_id", "")).strip()] = item
    return context_map


def _build_question_context_messages(
    prompt_row: Dict[str, Any] | None,
    *,
    judge_use_multimodal: bool,
) -> List[Dict[str, Any]]:
    if not prompt_row:
        return []
    if isinstance(prompt_row.get("messages"), list):
        return extract_user_only_messages(prompt_row["messages"])
    prompt_text = str(prompt_row.get("prompt_text", "")).strip()
    image_urls = prompt_row.get("call_image_urls") or prompt_row.get("image_urls") or []
    if not prompt_text:
        return []
    return [
        build_user_message(
            text=prompt_text,
            image_urls=image_urls,
            use_multimodal=judge_use_multimodal,
        )
    ]


def _response_to_judge_row(response: Dict[str, Any], judge_text: str) -> Dict[str, Any]:
    return {
        "judge": judge_text,
        "llm_output": response.get("llm_output", "") or judge_text,
        "usage": flatten_usage(response.get("usage")),
        "error": response.get("error", ""),
    }


def _evaluate_standard_subquestions(
    args: argparse.Namespace,
    client: UnifiedChatClient,
    *,
    data: List[Dict[str, Any]],
    solution_rows: List[Any],
    run_dir: Path,
    judge_use_multimodal: bool,
    judge_image_policy: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    spec = get_standard_spec(args.exam)
    ref_map = build_standard_reference_map(data)
    prompt_history_map = _load_interleaved_history(args, run_dir)
    prompt_context_map = _load_prompt_context_map(run_dir)

    judge_prompts: List[Dict[str, Any]] = []
    judge_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    total_weighted_score = 0.0

    filtered_rows = filter_standard_solution_rows(
        solution_rows,
        parse_question_ids(args.question_ids),
        limit_by_question=True,
        limit=args.limit,
    )
    for row in filtered_rows:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("question_id", "")).strip()
        sid = str(row.get("sub_question_id", "")).strip()
        answer = coerce_answer_from_solution_row(row)
        ref = ref_map.get((qid, sid))
        if ref is None:
            print(f"[WARN] missing reference for qid={qid} sid={sid}")
            continue

        if ref["solution"]:
            prompt_head = (
                f"{EVAL_PROMPT7_STANDARD}{ref['score']}"
                f"{EVAL_PROMPT_SUB_SOLUTION}{ref['solution']}"
                f"{spec.eval_prompt5}{ref['rubric']}"
            )
        else:
            prompt_head = f"{EVAL_PROMPT7_STANDARD}{ref['score']}{spec.eval_prompt5}{ref['rubric']}"
        prompt_text = f"{prompt_head}\n{EVAL_PROMPT6_STANDARD}\n{answer}"
        message = build_user_message(
            text=prompt_text,
            image_urls=ref["solution_fig"],
            use_multimodal=judge_use_multimodal,
        )
        question_context_messages = []
        if judge_image_policy.preset == "question_plus_reference":
            question_context_messages = _build_question_context_messages(
                prompt_context_map.get((qid, sid)),
                judge_use_multimodal=judge_use_multimodal,
            )

        key = (qid, sid)
        if args.eval_context_mode == "interleaved" and key in prompt_history_map:
            call_messages = copy.deepcopy(prompt_history_map[key])
            call_messages.append({"role": "assistant", "content": answer})
            call_messages.append(message)
        else:
            call_messages = question_context_messages + [message]
        call_image_urls = extract_message_image_urls(call_messages)
        question_context_image_urls = extract_message_image_urls(question_context_messages)

        response = _chat(client, args, model=args.judge_model, messages=call_messages)
        judge_text = response.get("content", "") or response.get("llm_output", "")
        parsed_score = extract_boxed_score(judge_text, allow_flexible=args.allow_flexible_score_parsing)
        weighted_score = compute_standard_weighted_score(spec, qid, parsed_score)
        total_weighted_score += weighted_score

        judge_prompts.append(
            {
                "question_id": qid,
                "sub_question_id": sid,
                "prompt_text": prompt_text,
                "image_urls": ref["solution_fig"],
                "reference_image_urls": ref["solution_fig"],
                "question_context_image_urls": question_context_image_urls,
                "call_image_urls": call_image_urls,
                "judge_image_policy": judge_image_policy.to_dict(),
                "eval_context_mode": args.eval_context_mode,
                "call_messages": call_messages,
            }
        )
        judge_rows.append(
            {
                "question_id": qid,
                "sub_question_id": sid,
                **_response_to_judge_row(response, judge_text),
            }
        )
        score_rows.append(
            {
                "question_id": qid,
                "sub_question_id": sid,
                "parsed_score": parsed_score,
                "weighted_score": weighted_score,
            }
        )
        print(f"[EVAL] {args.method} {args.exam.upper()} question={qid} sub={sid}")

    return judge_prompts, judge_rows, {"rows": score_rows, "total_weighted_score": total_weighted_score}


def _evaluate_standard_full(
    args: argparse.Namespace,
    client: UnifiedChatClient,
    *,
    data: List[Dict[str, Any]],
    solution_rows: List[Any],
    run_dir: Path,
    judge_use_multimodal: bool,
    judge_image_policy: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    spec = get_standard_spec(args.exam)
    question_map = build_standard_question_map(data)
    prompt_context_map = _load_prompt_context_map(run_dir)
    judge_prompts: List[Dict[str, Any]] = []
    judge_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    total_weighted_score = 0.0

    filtered_rows = filter_standard_solution_rows(
        solution_rows,
        parse_question_ids(args.question_ids),
        limit_by_question=True,
        limit=args.limit,
    )
    for row in filtered_rows:
        if not isinstance(row, dict):
            continue
        qid = str(row.get("question_id", "")).strip()
        answer = coerce_answer_from_solution_row(row)
        question = question_map.get(qid)
        if question is None:
            print(f"[WARN] missing question reference for qid={qid}")
            continue
        reference = build_standard_full_reference(question)
        prompt_text = (
            f"{EVAL_PROMPT7_STANDARD} {reference['total_score']}\n"
            f"{spec.eval_prompt5}{reference['reference_text']}\n"
            f"{EVAL_PROMPT6_STANDARD}\n{answer}"
        )
        message = build_user_message(
            text=prompt_text,
            image_urls=reference["image_urls"],
            use_multimodal=judge_use_multimodal,
        )
        question_context_messages = []
        if judge_image_policy.preset == "question_plus_reference":
            question_context_messages = _build_question_context_messages(
                prompt_context_map.get(qid),
                judge_use_multimodal=judge_use_multimodal,
            )
        call_messages = question_context_messages + [message]
        call_image_urls = extract_message_image_urls(call_messages)
        question_context_image_urls = extract_message_image_urls(question_context_messages)
        response = _chat(client, args, model=args.judge_model, messages=call_messages)
        judge_text = response.get("content", "") or response.get("llm_output", "")
        parsed_score = extract_boxed_score(judge_text, allow_flexible=args.allow_flexible_score_parsing)
        weighted_score = compute_standard_weighted_score(spec, qid, parsed_score)
        total_weighted_score += weighted_score

        judge_prompts.append(
            {
                "question_id": qid,
                "prompt_text": prompt_text,
                "image_urls": reference["image_urls"],
                "reference_image_urls": standard_full_reference_images(question),
                "question_context_image_urls": question_context_image_urls,
                "call_image_urls": call_image_urls,
                "judge_image_policy": judge_image_policy.to_dict(),
                "call_messages": call_messages,
            }
        )
        judge_rows.append(
            {
                "question_id": qid,
                **_response_to_judge_row(response, judge_text),
            }
        )
        score_rows.append(
            {
                "question_id": qid,
                "parsed_score": parsed_score,
                "weighted_score": weighted_score,
            }
        )
        print(f"[EVAL] full {args.exam.upper()} question={qid}")

    return judge_prompts, judge_rows, {"rows": score_rows, "total_weighted_score": total_weighted_score}


def _evaluate_ipho_items(
    args: argparse.Namespace,
    client: UnifiedChatClient,
    *,
    data: List[Any],
    solution_rows: List[Any],
    run_dir: Path,
    judge_use_multimodal: bool,
    judge_image_policy: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    prompt_history_map = _load_interleaved_history(args, run_dir)
    prompt_context_map = _load_prompt_context_map(run_dir)
    judge_prompts: List[Dict[str, Any]] = []
    judge_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    total_score = 0.0

    filtered_rows = filter_ipho_solution_rows(
        solution_rows,
        item_indices=parse_item_indices(args.item_indices),
        limit=args.limit,
    )
    for row in filtered_rows:
        if not isinstance(row, dict):
            continue
        item_index = int(row.get("item_index", -1))
        if item_index < 0 or item_index >= len(data):
            print(f"[WARN] invalid IPHO item index: {item_index}")
            continue
        parsed_item = parse_ipho_item(data[item_index])
        answer = coerce_answer_from_solution_row(row)
        prompt_text = f"{IPHO_EVAL_PROMPT5}{parsed_item['judge_text']}\n{IPHO_EVAL_PROMPT6}\n{answer}"
        message = build_user_message(
            text=prompt_text,
            image_urls=parsed_item["judge_images"],
            use_multimodal=judge_use_multimodal,
        )
        question_context_messages = []
        if judge_image_policy.preset == "question_plus_reference":
            question_context_messages = _build_question_context_messages(
                prompt_context_map.get(item_index),
                judge_use_multimodal=judge_use_multimodal,
            )
        if args.eval_context_mode == "interleaved" and item_index in prompt_history_map:
            call_messages = copy.deepcopy(prompt_history_map[item_index])
            call_messages.append({"role": "assistant", "content": answer})
            call_messages.append(message)
        else:
            call_messages = question_context_messages + [message]
        call_image_urls = extract_message_image_urls(call_messages)
        question_context_image_urls = extract_message_image_urls(question_context_messages)

        response = _chat(client, args, model=args.judge_model, messages=call_messages)
        judge_text = response.get("content", "") or response.get("llm_output", "")
        parsed_score = extract_boxed_score(judge_text, allow_flexible=args.allow_flexible_score_parsing)
        total_score += parsed_score

        judge_prompts.append(
            {
                "item_index": item_index,
                "prompt_text": prompt_text,
                "image_urls": parsed_item["judge_images"],
                "reference_image_urls": parsed_item["judge_images"],
                "question_context_image_urls": question_context_image_urls,
                "call_image_urls": call_image_urls,
                "judge_image_policy": judge_image_policy.to_dict(),
                "eval_context_mode": args.eval_context_mode,
                "call_messages": call_messages,
            }
        )
        judge_rows.append(
            {
                "item_index": item_index,
                **_response_to_judge_row(response, judge_text),
            }
        )
        score_rows.append({"item_index": item_index, "parsed_score": parsed_score})
        print(f"[EVAL] {args.method} IPHO S{args.question_num_index} item={item_index}")

    return judge_prompts, judge_rows, {"rows": score_rows, "total_score": total_score}


def _evaluate_ipho_full(
    args: argparse.Namespace,
    client: UnifiedChatClient,
    *,
    data: List[Any],
    solution_rows: List[Any],
    run_dir: Path,
    judge_use_multimodal: bool,
    judge_image_policy: Any,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    part_groups = group_ipho_parts(data)
    group_map = {str(group["part_id"]).upper(): group for group in part_groups}
    prompt_context_map = _load_prompt_context_map(run_dir)
    judge_prompts: List[Dict[str, Any]] = []
    judge_rows: List[Dict[str, Any]] = []
    score_rows: List[Dict[str, Any]] = []
    total_score = 0.0

    filtered_rows = filter_ipho_solution_rows(
        solution_rows,
        part_ids=parse_part_ids(args.part_ids),
        limit=args.limit,
    )
    for row in filtered_rows:
        if not isinstance(row, dict):
            continue
        part_id = str(row.get("part_id", "")).upper()
        if not part_id:
            continue
        group = group_map.get(part_id)
        if group is None:
            print(f"[WARN] missing IPHO part reference for part={part_id}")
            continue
        answer = coerce_answer_from_solution_row(row)
        reference = build_ipho_full_reference(group)
        prompt_text = f"{IPHO_EVAL_PROMPT5}{reference['reference_text']}\n{IPHO_EVAL_PROMPT6}\n{answer}"
        message = build_user_message(
            text=prompt_text,
            image_urls=reference["image_urls"],
            use_multimodal=judge_use_multimodal,
        )
        question_context_messages = []
        if judge_image_policy.preset == "question_plus_reference":
            question_context_messages = _build_question_context_messages(
                prompt_context_map.get(part_id),
                judge_use_multimodal=judge_use_multimodal,
            )
        call_messages = question_context_messages + [message]
        call_image_urls = extract_message_image_urls(call_messages)
        question_context_image_urls = extract_message_image_urls(question_context_messages)
        response = _chat(client, args, model=args.judge_model, messages=call_messages)
        judge_text = response.get("content", "") or response.get("llm_output", "")
        parsed_score = extract_boxed_score(judge_text, allow_flexible=args.allow_flexible_score_parsing)
        total_score += parsed_score

        judge_prompts.append(
            {
                "part_id": part_id,
                "prompt_text": prompt_text,
                "image_urls": reference["image_urls"],
                "reference_image_urls": ipho_part_reference_images(group),
                "question_context_image_urls": question_context_image_urls,
                "call_image_urls": call_image_urls,
                "judge_image_policy": judge_image_policy.to_dict(),
                "call_messages": call_messages,
            }
        )
        judge_rows.append(
            {
                "part_id": part_id,
                **_response_to_judge_row(response, judge_text),
            }
        )
        score_rows.append({"part_id": part_id, "parsed_score": parsed_score})
        print(f"[EVAL] full IPHO S{args.question_num_index} part={part_id}")

    return judge_prompts, judge_rows, {"rows": score_rows, "total_score": total_score}


def main(default_method: Optional[str] = None, default_exam: Optional[str] = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = _build_parser(default_method, default_exam)
    args = parser.parse_args()
    invocation = build_invocation_metadata(argv=sys.argv, entrypoint=__file__)

    run_name = args.run_name or model_tag_basename(args.pre_model_save_tag)
    if not run_name:
        raise ValueError("Please provide --run_name or --pre_model_save_tag for evaluation.")

    data_root = resolve_data_root(args.data_root)
    run_dir = build_run_dir(
        run_name=run_name,
        method=args.method,
        exam=args.exam,
        output_root=args.output_root,
        question_num_index=args.question_num_index,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    judge_use_multimodal = resolve_multimodal_mode(
        model=args.judge_model,
        mode=args.judge_multimodal_mode,
        explicit_override=args.is_judge_multi,
    )
    judge_image_policy = resolve_judge_image_policy(
        preset=args.judge_image_preset,
        method=args.method,
        exam=args.exam,
    )
    print(
        f"[CFG] run_name={run_name} method={args.method} exam={args.exam} data_root={data_root} run_dir={run_dir}"
    )
    print(
        f"[CFG] judge_model={args.judge_model} judge_multimodal={judge_use_multimodal} "
        f"(mode={args.judge_multimodal_mode}, override={args.is_judge_multi})"
    )
    print(f"[CFG] eval_context_mode={args.eval_context_mode}")
    print(f"[CFG] judge_image_policy={judge_image_policy.to_dict()}")

    client = UnifiedChatClient(
        api_key=args.api_key,
        backend=args.api_backend,
        api_url=args.api_url,
        host=args.api_host,
        max_retries=args.max_try,
        retry_delay=args.retry_delay,
        timeout=1800,
        model_tag=args.judge_model,
        verbose=args.verbose,
    )

    solution_rows = _load_solution_rows(args, run_dir)

    if args.exam == "ipho":
        data_path = data_root / f"IPhO_2025_S{args.question_num_index}_solution.json"
        data = load_json(data_path)
        if args.method == "full":
            judge_prompts, judge_rows, scores = _evaluate_ipho_full(
                args,
                client,
                data=data,
                solution_rows=solution_rows,
                run_dir=run_dir,
                judge_use_multimodal=judge_use_multimodal,
                judge_image_policy=judge_image_policy,
            )
        else:
            judge_prompts, judge_rows, scores = _evaluate_ipho_items(
                args,
                client,
                data=data,
                solution_rows=solution_rows,
                run_dir=run_dir,
                judge_use_multimodal=judge_use_multimodal,
                judge_image_policy=judge_image_policy,
            )
    else:
        spec = get_standard_spec(args.exam)
        data_path = data_root / spec.data_file
        data = load_json(data_path)
        if args.method == "full":
            judge_prompts, judge_rows, scores = _evaluate_standard_full(
                args,
                client,
                data=data,
                solution_rows=solution_rows,
                run_dir=run_dir,
                judge_use_multimodal=judge_use_multimodal,
                judge_image_policy=judge_image_policy,
            )
        else:
            judge_prompts, judge_rows, scores = _evaluate_standard_subquestions(
                args,
                client,
                data=data,
                solution_rows=solution_rows,
                run_dir=run_dir,
                judge_use_multimodal=judge_use_multimodal,
                judge_image_policy=judge_image_policy,
            )

    judge_prompts_path = run_dir / "judge_prompts.json"
    judge_results_path = run_dir / "judge_results.json"
    scores_path = run_dir / "scores.json"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    save_json(judge_prompts_path, judge_prompts)
    save_json(judge_results_path, judge_rows)
    save_json(scores_path, scores)

    update_named_json(
        config_path,
        "evaluation",
        {
            **redact_sensitive_fields(namespace_to_dict(args)),
            **invocation,
            "run_name": run_name,
            "resolved_data_root": str(data_root),
            "resolved_run_dir": str(run_dir),
            "resolved_judge_multimodal": judge_use_multimodal,
            "evaluated_at": timestamp_tag(),
        },
    )
    summary_payload: Dict[str, Any] = {
        "run_name": run_name,
        "method": args.method,
        "exam": args.exam,
        "question_num_index": args.question_num_index,
        "judge_prompts_path": str(judge_prompts_path),
        "judge_results_path": str(judge_results_path),
        "scores_path": str(scores_path),
        "judged_units": len(judge_rows),
    }
    summary_payload.update({key: value for key, value in scores.items() if key.startswith("total_")})
    update_named_json(summary_path, "evaluation", summary_payload)

    print(f"[DONE] wrote judge prompts to: {judge_prompts_path}")
    print(f"[DONE] wrote judge results to: {judge_results_path}")
    print(f"[DONE] wrote scores to: {scores_path}")
    if "total_weighted_score" in scores:
        print(f"[DONE] total weighted score: {scores['total_weighted_score']}")
    if "total_score" in scores:
        print(f"[DONE] total score: {scores['total_score']}")


if __name__ == "__main__":
    main()
