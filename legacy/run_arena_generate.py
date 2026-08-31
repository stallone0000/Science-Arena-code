# -*- coding: utf-8 -*-
"""Unified generation runner for ScienceArena iterative/interleaved/full methods."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from api_client_interleaved import (
    DEFAULT_API_URL,
    DEFAULT_HOST,
    UnifiedChatClient,
    build_user_message,
    parse_bool,
    resolve_multimodal_mode,
)
from interleaved_common import (
    IPHO_PREVIOUS_ANSWER_PROMPT,
    STANDARD_PREVIOUS_ANSWER_PROMPT,
    apply_interleaved_answer_style,
    build_ipho_full_prompt,
    build_ipho_question_text,
    build_ipho_iterative_append_text,
    build_invocation_metadata,
    build_run_dir,
    build_standard_background_text,
    build_standard_full_prompt,
    build_standard_subquestion_text,
    build_standard_iterative_append_text,
    clean_image_urls,
    default_data_root,
    flatten_usage,
    get_standard_spec,
    group_ipho_parts,
    load_json,
    model_tag_basename,
    namespace_to_dict,
    parse_item_indices,
    parse_ipho_item,
    parse_part_ids,
    parse_question_ids,
    resolve_data_root,
    redact_sensitive_fields,
    save_json,
    select_ipho_items,
    select_standard_questions,
    timestamp_tag,
    update_named_json,
)
from image_policy import (
    extract_message_image_urls,
    ipho_part_question_images,
    resolve_generation_image_policy,
    resolve_ipho_interleaved_turn_images,
    resolve_ipho_iterative_question_images,
    resolve_standard_interleaved_background_images,
    resolve_standard_interleaved_turn_images,
    resolve_standard_iterative_question_images,
    standard_full_question_images,
    standard_question_image_categories,
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

    parser.add_argument("--pre_model", type=str, default="google/gemini-3-pro-preview")
    parser.add_argument("--pre_model_save_tag", type=str, default="")

    parser.add_argument("--api_backend", type=str, default="openai", choices=["openai", "requests"])
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--api_host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--api_key", type=str, default="")

    parser.add_argument("--max_try", type=int, default=5)
    parser.add_argument("--retry_delay", type=int, default=60)
    parser.add_argument("--max_tokens", type=int, default=81920)

    parser.add_argument(
        "--pre_multimodal_mode",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help="Pre-model multimodal policy: auto by model capability, or force on/off.",
    )
    parser.add_argument(
        "--is_pre_multi",
        type=parse_bool,
        default=None,
        help="Legacy override for multimodal; if set, overrides --pre_multimodal_mode.",
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
    parser.add_argument(
        "--generation_image_preset",
        type=str,
        default="unified",
        choices=["unified", "legacy"],
        help="Generation-side image policy preset.",
    )
    parser.add_argument(
        "--iterative_image_mode",
        type=str,
        default="inherit",
        choices=["inherit", "current_only", "global_current", "cumulative"],
        help="Override iterative image accumulation policy; applies to standard exams.",
    )
    parser.add_argument(
        "--interleaved_global_image_replay",
        type=str,
        default="inherit",
        choices=["inherit", "on", "off"],
        help="For standard interleaved generation, whether to replay global images on every turn.",
    )
    parser.add_argument(
        "--interleaved_replay_global_persist",
        type=str,
        default="inherit",
        choices=["inherit", "on", "off"],
        help="Whether replayed global images should persist in interleaved history turns.",
    )
    parser.add_argument(
        "--interleaved_answer_style",
        type=str,
        default="self_contained",
        choices=["self_contained", "contextual"],
        help="For interleaved generation, require each current answer to be self-contained or allow fully contextual replies.",
    )

    parser.add_argument("--print_curl", type=parse_bool, default=False)
    parser.add_argument("--verbose", type=parse_bool, default=False)
    return parser


def _join_blocks(*blocks: str) -> str:
    cleaned = [block.strip() for block in blocks if block and block.strip()]
    return "\n".join(cleaned).strip()


def _response_to_row(response: Dict[str, Any], *, answer: str) -> Dict[str, Any]:
    return {
        "answer": answer,
        "llm_output": response.get("llm_output", "") or answer,
        "usage": flatten_usage(response.get("usage")),
        "error": response.get("error", ""),
    }


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


def _resolve_generation_policy(args: argparse.Namespace):
    return resolve_generation_image_policy(
        preset=args.generation_image_preset,
        method=args.method,
        exam=args.exam,
        iterative_image_mode=args.iterative_image_mode,
        interleaved_global_image_replay=args.interleaved_global_image_replay,
        interleaved_replay_global_persist=args.interleaved_replay_global_persist,
    )


def _generate_standard_iterative(
    args: argparse.Namespace,
    client: UnifiedChatClient,
    *,
    data: List[Dict[str, Any]],
    use_multimodal: bool,
    generation_policy: Any,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    spec = get_standard_spec(args.exam)
    prompt_rows: List[Dict[str, Any]] = []
    solution_rows: List[Dict[str, Any]] = []

    for question in select_standard_questions(data, parse_question_ids(args.question_ids), limit=args.limit):
        qid = str(question.get("question_id", [""])[0]).strip()
        prompt_text = build_standard_background_text(spec, question)
        question_images = standard_question_image_categories(question)
        history_local_images: List[str] = []
        for sub_question in question.get("detailed_question", []):
            sid = str(sub_question.get("sub_question_id", [""])[0]).strip()
            prompt_text = _join_blocks(prompt_text, build_standard_iterative_append_text(sub_question))
            current_images = standard_question_image_categories(question, sub_question)
            resolved_image_urls = resolve_standard_iterative_question_images(
                generation_policy,
                global_question_images=question_images["global_question_images"],
                history_local_images=history_local_images,
                current_local_images=current_images["local_question_images"],
            )
            call_message = build_user_message(
                text=prompt_text,
                image_urls=resolved_image_urls,
                use_multimodal=use_multimodal,
            )
            response = _chat(client, args, model=args.pre_model, messages=[call_message])
            answer = response.get("content", "") or response.get("llm_output", "")
            call_image_urls = extract_message_image_urls([call_message])

            prompt_rows.append(
                {
                    "question_id": qid,
                    "sub_question_id": sid,
                    "unit_type": "sub_question",
                    "prompt_text": prompt_text,
                    "image_urls": resolved_image_urls,
                    "call_image_urls": call_image_urls,
                    "global_question_image_urls": question_images["global_question_images"],
                    "history_local_image_urls": list(history_local_images),
                    "current_local_image_urls": current_images["local_question_images"],
                    "image_policy": generation_policy.to_dict(),
                }
            )
            solution_rows.append(
                {
                    "question_id": qid,
                    "sub_question_id": sid,
                    **_response_to_row(response, answer=answer),
                }
            )

            prompt_text = _join_blocks(prompt_text, STANDARD_PREVIOUS_ANSWER_PROMPT, answer)
            history_local_images.extend(current_images["local_question_images"])
            print(f"[GEN] iterative {args.exam.upper()} question={qid} sub={sid}")

    return prompt_rows, solution_rows


def _generate_standard_interleaved(
    args: argparse.Namespace,
    client: UnifiedChatClient,
    *,
    data: List[Dict[str, Any]],
    use_multimodal: bool,
    generation_policy: Any,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    spec = get_standard_spec(args.exam)
    prompt_rows: List[Dict[str, Any]] = []
    solution_rows: List[Dict[str, Any]] = []

    for question in select_standard_questions(data, parse_question_ids(args.question_ids), limit=args.limit):
        qid = str(question.get("question_id", [""])[0]).strip()
        question_images = standard_question_image_categories(question)
        background_history_images = resolve_standard_interleaved_background_images(
            generation_policy,
            global_question_images=question_images["global_question_images"],
        )
        history: List[Dict[str, Any]] = [
            build_user_message(
                text=build_standard_background_text(spec, question),
                image_urls=background_history_images,
                use_multimodal=use_multimodal,
            )
        ]

        for sub_question in question.get("detailed_question", []):
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
            call_user_message = build_user_message(
                text=current_text,
                image_urls=turn_plan["call_image_urls"],
                use_multimodal=use_multimodal,
            )
            persisted_user_message = build_user_message(
                text=current_text,
                image_urls=turn_plan["persisted_image_urls"],
                use_multimodal=use_multimodal,
            )
            history_image_urls = extract_message_image_urls(history)
            call_messages = copy.deepcopy(history)
            call_messages.append(call_user_message)
            response = _chat(client, args, model=args.pre_model, messages=call_messages)
            answer = response.get("content", "") or response.get("llm_output", "")
            call_image_urls = extract_message_image_urls(call_messages)

            prompt_rows.append(
                {
                    "question_id": qid,
                    "sub_question_id": sid,
                    "unit_type": "sub_question",
                    "messages": call_messages,
                    "call_image_urls": call_image_urls,
                    "history_image_urls": history_image_urls,
                    "background_history_image_urls": background_history_images,
                    "current_turn_image_urls": turn_plan["call_image_urls"],
                    "persisted_turn_image_urls": turn_plan["persisted_image_urls"],
                    "replay_global_image_urls": turn_plan["replay_global_image_urls"],
                    "global_question_image_urls": question_images["global_question_images"],
                    "current_local_image_urls": current_images["local_question_images"],
                    "image_policy": generation_policy.to_dict(),
                }
            )
            solution_rows.append(
                {
                    "question_id": qid,
                    "sub_question_id": sid,
                    **_response_to_row(response, answer=answer),
                }
            )

            history.append(persisted_user_message)
            history.append({"role": "assistant", "content": answer})
            print(f"[GEN] interleaved {args.exam.upper()} question={qid} sub={sid}")

    return prompt_rows, solution_rows


def _generate_standard_full(
    args: argparse.Namespace,
    client: UnifiedChatClient,
    *,
    data: List[Dict[str, Any]],
    use_multimodal: bool,
    generation_policy: Any,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    spec = get_standard_spec(args.exam)
    prompt_rows: List[Dict[str, Any]] = []
    solution_rows: List[Dict[str, Any]] = []

    for question in select_standard_questions(data, parse_question_ids(args.question_ids), limit=args.limit):
        qid = str(question.get("question_id", [""])[0]).strip()
        prompt_text, image_urls = build_standard_full_prompt(spec, question)
        call_message = build_user_message(text=prompt_text, image_urls=image_urls, use_multimodal=use_multimodal)
        response = _chat(client, args, model=args.pre_model, messages=[call_message])
        answer = response.get("content", "") or response.get("llm_output", "")
        call_image_urls = extract_message_image_urls([call_message])

        prompt_rows.append(
            {
                "question_id": qid,
                "unit_type": "question",
                "prompt_text": prompt_text,
                "image_urls": image_urls,
                "call_image_urls": call_image_urls,
                "global_question_image_urls": clean_image_urls(question.get("overall_instructions_fig", [])),
                "all_question_image_urls": standard_full_question_images(question),
                "image_policy": generation_policy.to_dict(),
            }
        )
        solution_rows.append(
            {
                "question_id": qid,
                **_response_to_row(response, answer=answer),
            }
        )
        print(f"[GEN] full {args.exam.upper()} question={qid}")

    return prompt_rows, solution_rows


def _generate_ipho_iterative(
    args: argparse.Namespace,
    client: UnifiedChatClient,
    *,
    data: List[Any],
    use_multimodal: bool,
    generation_policy: Any,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt_rows: List[Dict[str, Any]] = []
    solution_rows: List[Dict[str, Any]] = []
    prompt_text = ""

    for logical_idx, (item_index, raw_item) in enumerate(
        select_ipho_items(data, parse_item_indices(args.item_indices), limit=args.limit)
    ):
        parsed_item = parse_ipho_item(raw_item)
        prompt_text = _join_blocks(
            prompt_text,
            build_ipho_iterative_append_text(parsed_item, item_index, is_first=(logical_idx == 0)),
        )
        image_urls = resolve_ipho_iterative_question_images(parsed_item)

        call_message = build_user_message(text=prompt_text, image_urls=image_urls, use_multimodal=use_multimodal)
        response = _chat(client, args, model=args.pre_model, messages=[call_message])
        answer = response.get("content", "") or response.get("llm_output", "")
        call_image_urls = extract_message_image_urls([call_message])

        prompt_rows.append(
            {
                "item_index": item_index,
                "unit_type": "item",
                "prompt_text": prompt_text,
                "image_urls": image_urls,
                "call_image_urls": call_image_urls,
                "background_image_urls": clean_image_urls(parsed_item.get("background_images", [])),
                "question_image_urls": clean_image_urls(parsed_item.get("question_images", [])),
                "image_policy": generation_policy.to_dict(),
            }
        )
        solution_rows.append(
            {
                "item_index": item_index,
                **_response_to_row(response, answer=answer),
            }
        )

        prompt_text = _join_blocks(prompt_text, IPHO_PREVIOUS_ANSWER_PROMPT, answer)
        print(f"[GEN] iterative IPHO S{args.question_num_index} item={item_index}")

    return prompt_rows, solution_rows


def _generate_ipho_interleaved(
    args: argparse.Namespace,
    client: UnifiedChatClient,
    *,
    data: List[Any],
    use_multimodal: bool,
    generation_policy: Any,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt_rows: List[Dict[str, Any]] = []
    solution_rows: List[Dict[str, Any]] = []
    history: List[Dict[str, Any]] = []

    for item_index, raw_item in select_ipho_items(data, parse_item_indices(args.item_indices), limit=args.limit):
        parsed_item = parse_ipho_item(raw_item)
        turn_plan = resolve_ipho_interleaved_turn_images(parsed_item)
        current_text = apply_interleaved_answer_style(
            build_ipho_question_text(parsed_item, item_index),
            answer_style=args.interleaved_answer_style,
        )
        history_image_urls = extract_message_image_urls(history)
        call_user_message = build_user_message(
            text=current_text,
            image_urls=turn_plan["call_image_urls"],
            use_multimodal=use_multimodal,
        )
        persisted_user_message = build_user_message(
            text=current_text,
            image_urls=turn_plan["persisted_image_urls"],
            use_multimodal=use_multimodal,
        )
        call_messages = copy.deepcopy(history)
        call_messages.append(call_user_message)
        response = _chat(client, args, model=args.pre_model, messages=call_messages)
        answer = response.get("content", "") or response.get("llm_output", "")
        call_image_urls = extract_message_image_urls(call_messages)

        prompt_rows.append(
            {
                "item_index": item_index,
                "unit_type": "item",
                "messages": call_messages,
                "call_image_urls": call_image_urls,
                "history_image_urls": history_image_urls,
                "current_turn_image_urls": turn_plan["call_image_urls"],
                "persisted_turn_image_urls": turn_plan["persisted_image_urls"],
                "replay_global_image_urls": turn_plan["replay_global_image_urls"],
                "background_image_urls": clean_image_urls(parsed_item.get("background_images", [])),
                "question_image_urls": clean_image_urls(parsed_item.get("question_images", [])),
                "image_policy": generation_policy.to_dict(),
            }
        )
        solution_rows.append(
            {
                "item_index": item_index,
                **_response_to_row(response, answer=answer),
            }
        )

        history.append(persisted_user_message)
        history.append({"role": "assistant", "content": answer})
        print(f"[GEN] interleaved IPHO S{args.question_num_index} item={item_index}")

    return prompt_rows, solution_rows


def _generate_ipho_full(
    args: argparse.Namespace,
    client: UnifiedChatClient,
    *,
    data: List[Any],
    use_multimodal: bool,
    generation_policy: Any,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    prompt_rows: List[Dict[str, Any]] = []
    solution_rows: List[Dict[str, Any]] = []
    part_groups = group_ipho_parts(
        data,
        part_ids=parse_part_ids(args.part_ids),
        item_indices=parse_item_indices(args.item_indices),
    )
    if args.limit:
        part_groups = part_groups[: args.limit]

    for part_group in part_groups:
        part_id = str(part_group.get("part_id", "Misc"))
        prompt_text, image_urls = build_ipho_full_prompt(part_group)
        call_message = build_user_message(text=prompt_text, image_urls=image_urls, use_multimodal=use_multimodal)
        response = _chat(client, args, model=args.pre_model, messages=[call_message])
        answer = response.get("content", "") or response.get("llm_output", "")
        call_image_urls = extract_message_image_urls([call_message])

        prompt_rows.append(
            {
                "part_id": part_id,
                "item_indices": [item["item_index"] for item in part_group.get("items", [])],
                "unit_type": "part",
                "prompt_text": prompt_text,
                "image_urls": image_urls,
                "call_image_urls": call_image_urls,
                "all_question_image_urls": ipho_part_question_images(part_group),
                "image_policy": generation_policy.to_dict(),
            }
        )
        solution_rows.append(
            {
                "part_id": part_id,
                **_response_to_row(response, answer=answer),
            }
        )
        print(f"[GEN] full IPHO S{args.question_num_index} part={part_id}")

    return prompt_rows, solution_rows


def main(default_method: Optional[str] = None, default_exam: Optional[str] = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    parser = _build_parser(default_method, default_exam)
    args = parser.parse_args()
    invocation = build_invocation_metadata(argv=sys.argv, entrypoint=__file__)

    run_name = args.run_name or model_tag_basename(args.pre_model_save_tag or args.pre_model)
    data_root = resolve_data_root(args.data_root)
    run_dir = build_run_dir(
        run_name=run_name,
        method=args.method,
        exam=args.exam,
        output_root=args.output_root,
        question_num_index=args.question_num_index,
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    use_multimodal = resolve_multimodal_mode(
        model=args.pre_model,
        mode=args.pre_multimodal_mode,
        explicit_override=args.is_pre_multi,
    )
    generation_policy = _resolve_generation_policy(args)
    print(
        f"[CFG] run_name={run_name} method={args.method} exam={args.exam} data_root={data_root} run_dir={run_dir}"
    )
    print(
        f"[CFG] pre_model={args.pre_model} pre_multimodal={use_multimodal} "
        f"(mode={args.pre_multimodal_mode}, override={args.is_pre_multi})"
    )
    print(f"[CFG] generation_image_policy={generation_policy.to_dict()}")
    if args.method == "interleaved":
        print(f"[CFG] interleaved_answer_style={args.interleaved_answer_style}")

    client = UnifiedChatClient(
        api_key=args.api_key,
        backend=args.api_backend,
        api_url=args.api_url,
        host=args.api_host,
        max_retries=args.max_try,
        retry_delay=args.retry_delay,
        timeout=1800,
        model_tag=(args.pre_model_save_tag or args.pre_model),
        verbose=args.verbose,
    )

    if args.exam == "ipho":
        data_path = data_root / f"IPhO_2025_S{args.question_num_index}_solution.json"
        data = load_json(data_path)
        if args.method == "iterative":
            prompt_rows, solution_rows = _generate_ipho_iterative(
                args,
                client,
                data=data,
                use_multimodal=use_multimodal,
                generation_policy=generation_policy,
            )
        elif args.method == "interleaved":
            prompt_rows, solution_rows = _generate_ipho_interleaved(
                args,
                client,
                data=data,
                use_multimodal=use_multimodal,
                generation_policy=generation_policy,
            )
        else:
            prompt_rows, solution_rows = _generate_ipho_full(
                args,
                client,
                data=data,
                use_multimodal=use_multimodal,
                generation_policy=generation_policy,
            )
    else:
        spec = get_standard_spec(args.exam)
        data_path = data_root / spec.data_file
        data = load_json(data_path)
        if args.method == "iterative":
            prompt_rows, solution_rows = _generate_standard_iterative(
                args,
                client,
                data=data,
                use_multimodal=use_multimodal,
                generation_policy=generation_policy,
            )
        elif args.method == "interleaved":
            prompt_rows, solution_rows = _generate_standard_interleaved(
                args,
                client,
                data=data,
                use_multimodal=use_multimodal,
                generation_policy=generation_policy,
            )
        else:
            prompt_rows, solution_rows = _generate_standard_full(
                args,
                client,
                data=data,
                use_multimodal=use_multimodal,
                generation_policy=generation_policy,
            )

    prompts_path = run_dir / "prompts.json"
    solutions_path = run_dir / "solutions.json"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    save_json(prompts_path, prompt_rows)
    save_json(solutions_path, solution_rows)
    update_named_json(
        config_path,
        "generation",
        {
            **redact_sensitive_fields(namespace_to_dict(args)),
            **invocation,
            "run_name": run_name,
            "resolved_data_root": str(data_root),
            "resolved_run_dir": str(run_dir),
            "resolved_pre_multimodal": use_multimodal,
            "generated_at": timestamp_tag(),
        },
    )
    update_named_json(
        summary_path,
        "generation",
        {
            "run_name": run_name,
            "method": args.method,
            "exam": args.exam,
            "question_num_index": args.question_num_index,
            "unit_count": len(solution_rows),
            "prompt_count": len(prompt_rows),
            "prompts_path": str(prompts_path),
            "solutions_path": str(solutions_path),
        },
    )

    print(f"[DONE] wrote prompts to: {prompts_path}")
    print(f"[DONE] wrote solutions to: {solutions_path}")
    print(f"[DONE] generated units: {len(solution_rows)}")


if __name__ == "__main__":
    main()
