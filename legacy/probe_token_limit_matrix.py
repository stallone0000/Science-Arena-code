# -*- coding: utf-8 -*-
"""Token-limit audit matrix for ScienceArena models.

This script is intentionally separate from:
- smoke_api_matrix.py: generic API connectivity + multimodal availability
- probe_thinking_matrix.py: thinking semantics / controllability

It answers:
1. Which token-limit parameter name is accepted on this route?
2. Does a small cap actually truncate output in a visible way?
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from api_client_interleaved import (
    DEFAULT_API_URL,
    DEFAULT_HOST,
    UnifiedChatClient,
    build_user_message,
)
from interleaved_common import (
    build_invocation_metadata,
    default_smoke_root,
    namespace_to_dict,
    redact_sensitive_fields,
    save_json,
    timestamp_tag,
    toacl_root,
)
from model_registry import resolve_model_profile


DEFAULT_EXTRA_MODELS = [
    "qiniu/openai/gpt-5.4",
    "openai/gpt-5.4",
    "openai/gpt-5.5",
    "anthropic/claude-sonnet-4-6",
]

TOKEN_PARAM_VARIANTS = ("preferred", "max_tokens", "max_completion_tokens")
TASK_MODES = ("smoke", "real")

TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "long_steps_smoke": {
        "task_id": "long_steps_smoke",
        "task_mode": "smoke",
        "source_note": "A forced long-output task for checking whether a small token cap visibly truncates the answer.",
        "messages": [
            build_user_message(
                text=(
                    "Write exactly 40 numbered lines.\n"
                    "Each line must start with `Step XX:` where XX is two digits from 01 to 40.\n"
                    "Each line must contain one full sentence of at least 12 words.\n"
                    "After the 40 lines, write one final line exactly equal to `END_OF_RESPONSE`.\n"
                    "Do not use bullet points, markdown tables, or code fences."
                ),
                image_urls=[],
                use_multimodal=False,
            )
        ],
    },
    "long_json_smoke": {
        "task_id": "long_json_smoke",
        "task_mode": "smoke",
        "source_note": "A structured long JSON output task for visible truncation checks.",
        "messages": [
            build_user_message(
                text=(
                    "Return a JSON object with keys item_01 through item_30.\n"
                    "Each value must be a sentence of at least 20 words explaining a fictional scientific observation.\n"
                    "After item_30 also include a key `final_summary` with a 40-word paragraph.\n"
                    "Return only JSON."
                ),
                image_urls=[],
                use_multimodal=False,
            )
        ],
    },
    "aime_hard_long_derivation": {
        "task_id": "aime_hard_long_derivation",
        "task_mode": "real",
        "source_note": "AIME-style long derivation task for testing whether a small completion cap truncates multi-step reasoning.",
        "messages": [
            build_user_message(
                text=(
                    "Solve this problem with a full derivation.\n"
                    "Write at least 12 numbered reasoning steps before the final answer line.\n"
                    "Do not skip algebra.\n\n"
                    "Patrick started walking at a constant rate along a straight road from school to the park. "
                    "One hour after Patrick left, Tanya started running along the same road from school to the park. "
                    "One hour after Tanya left, Jose started bicycling along the same road from school to the park. "
                    "Tanya ran at a constant rate of 2 miles per hour faster than Patrick walked, "
                    "Jose bicycled at a constant rate of 7 miles per hour faster than Tanya ran, "
                    "and all three arrived at the park at the same time. "
                    "The distance from the school to the park is m/n miles, where m and n are relatively prime positive integers. "
                    "Find m + n.\n\n"
                    "End with a final line exactly in the form `Final: <integer>`."
                ),
                image_urls=[],
                use_multimodal=False,
            )
        ],
    },
    "structured_reasoning_long": {
        "task_id": "structured_reasoning_long",
        "task_mode": "real",
        "source_note": "A longer structured optimization task with exact JSON output and explicit derivation requirements.",
        "messages": [
            build_user_message(
                text=(
                    "Find the minimum-cost valid assignment and show your full reasoning in at least 10 numbered steps before the answer.\n"
                    "Assign five workers A, B, C, D, E to five jobs J1, J2, J3, J4, J5, "
                    "with each worker taking exactly one job and each job assigned exactly once. "
                    "The costs are:\n"
                    "A: J1=13, J2=7, J3=impossible, J4=6, J5=15\n"
                    "B: J1=9, J2=impossible, J3=8, J4=14, J5=7\n"
                    "C: J1=12, J2=10, J3=6, J4=impossible, J5=11\n"
                    "D: J1=impossible, J2=5, J3=12, J4=9, J5=8\n"
                    "E: J1=7, J2=13, J3=10, J4=8, J5=impossible\n"
                    "Additional constraints:\n"
                    "1. D cannot take J1 or J3.\n"
                    "2. If C takes J5, then A cannot take J4.\n"
                    "3. E must take either J1 or J4.\n"
                    "4. If B takes J5, then A must take J2.\n"
                    "5. If A takes J4, then D must take J2.\n\n"
                    "After the reasoning, output exactly one final JSON object with no extra text in this shape: "
                    "{\"A\":\"J?\",\"B\":\"J?\",\"C\":\"J?\",\"D\":\"J?\",\"E\":\"J?\",\"total\":<int>}."
                ),
                image_urls=[],
                use_multimodal=False,
            )
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--api_host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--only_models", nargs="*", default=[])
    parser.add_argument("--only_backend", type=str, default="", choices=["", "openai", "requests"])
    parser.add_argument("--token_caps", nargs="*", type=int, default=[100, 400])
    parser.add_argument("--task_ids", nargs="*", default=[])
    parser.add_argument("--task_mode", type=str, default="", choices=["", *TASK_MODES])
    parser.add_argument("--repeat_count", type=int, default=1)
    parser.add_argument("--max_try", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--sleep_between_calls", type=float, default=0.5)
    parser.add_argument("--print_curl", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out_file", type=str, default="")
    return parser.parse_args()


def unique_items(values: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)
    return result


def _is_probable_model_name(candidate: str) -> bool:
    import re

    x = (candidate or "").strip()
    if not x:
        return False
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", x):
        return False
    return "/" in x or any(k in x.lower() for k in ("gpt", "claude", "gemini", "qwen", "glm", "doubao", "deepseek", "kimi", "minimax"))


def resolve_models(args: argparse.Namespace) -> List[str]:
    if args.only_models:
        return unique_items(args.only_models)
    import re

    scripts_root = toacl_root() / "scripts"
    pattern = re.compile(r"--(?:pre_model|judge_model)\s+[\"']?([^\"'\s\\]+)[\"']?")
    models = set()
    if scripts_root.exists():
        for script in scripts_root.glob("*.sh"):
            text = script.read_text(encoding="utf-8", errors="ignore")
            for candidate in pattern.findall(text):
                if _is_probable_model_name(candidate):
                    models.add(candidate)
    for extra in DEFAULT_EXTRA_MODELS:
        models.add(extra)
    return sorted(models)


def resolve_backends(args: argparse.Namespace) -> List[str]:
    return [args.only_backend] if args.only_backend else ["openai"]


def resolve_task_ids(args: argparse.Namespace) -> List[str]:
    if not args.task_ids:
        task_ids = list(TASK_REGISTRY.keys())
    else:
        task_ids = unique_items([str(x).strip() for x in args.task_ids if str(x).strip()])
    if args.task_mode:
        task_ids = [task_id for task_id in task_ids if TASK_REGISTRY[task_id]["task_mode"] == args.task_mode]
    invalid = [task_id for task_id in task_ids if task_id not in TASK_REGISTRY]
    if invalid:
        raise ValueError(f"Unsupported task ids: {invalid}")
    return task_ids


def resolve_output_paths(args: argparse.Namespace) -> tuple[Path, Path, Path, Path]:
    if args.out_file:
        combined = Path(args.out_file).expanduser().resolve()
    else:
        combined = default_smoke_root() / "token_limit_matrix" / f"token_limit_matrix_{timestamp_tag()}.json"
    rows_path = combined.with_name(f"{combined.stem}_rows.json")
    summary_path = combined.with_name(f"{combined.stem}_summary.json")
    markdown_path = combined.with_name(f"{combined.stem}_summary.md")
    return combined, rows_path, summary_path, markdown_path


def short_text(text: str, max_len: int = 240) -> str:
    value = (text or "").replace("\n", " ").strip()
    if len(value) <= max_len:
        return value
    return value[:max_len] + "..."


def visible_chars(text: str) -> int:
    return len((text or "").strip())


def truncated_by_length(*, finish_reason: str, content: str, completion_tokens: int, requested_token_cap: int) -> bool:
    if finish_reason == "length":
        return True
    if completion_tokens and requested_token_cap and completion_tokens >= max(requested_token_cap - 5, 1):
        return True
    tail = (content or "").rstrip()
    if tail and tail[-1] not in ".!]}\"')":
        return True
    return False


def extract_reasoning_tokens(usage: Dict[str, Any]) -> int:
    if not isinstance(usage, dict):
        return 0
    if isinstance(usage.get("reasoning_tokens"), int):
        return int(usage["reasoning_tokens"])
    for key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
            return int(details["reasoning_tokens"])
    return 0


def extract_completion_tokens(usage: Dict[str, Any]) -> int:
    if not isinstance(usage, dict):
        return 0
    for key in ("completion_tokens", "output_tokens"):
        if isinstance(usage.get(key), int):
            return int(usage[key])
    return 0


def simplify_usage(usage: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(usage, dict):
        return {}
    result: Dict[str, Any] = {}
    for key in ("prompt_tokens", "completion_tokens", "total_tokens", "input_tokens", "output_tokens"):
        if key in usage:
            result[key] = usage[key]
    for key in ("completion_tokens_details", "prompt_tokens_details", "input_tokens_details", "output_tokens_details"):
        if key in usage and usage[key]:
            result[key] = usage[key]
    return result or usage


def build_markdown_summary(summary: Dict[str, Any]) -> str:
    lines = [
        "# Token Limit Audit Summary",
        "",
        f"- Time: {summary['time']}",
        f"- API URL: `{summary['api_url']}`",
        f"- Models tested: `{summary['models_tested']}`",
        f"- Total cases: `{summary['total_cases']}`",
        "",
        "## Model Labels",
    ]
    for model, item in summary["model_summaries"].items():
        lines.append(
            f"- `{model}`: preferred=`{item['preferred_param_label']}` | alternate=`{item['alternate_param_label']}` | "
            f"observable=`{item['token_limit_label']}`"
        )
    return "\n".join(lines) + "\n"


def run_single_case(
    *,
    client: UnifiedChatClient,
    model: str,
    backend: str,
    task: Dict[str, Any],
    token_param_variant: str,
    requested_token_cap: int,
    print_curl: bool,
) -> Dict[str, Any]:
    started_at = time.time()
    try:
        response = client.chat(
            model=model,
            messages=task["messages"],
            max_tokens=requested_token_cap,
            token_param_override="auto" if token_param_variant == "preferred" else token_param_variant,
            print_curl=print_curl,
        )
    except Exception as exc:
        latency_sec = round(time.time() - started_at, 3)
        return {
            "model": model,
            "backend": backend,
            "task_id": task["task_id"],
            "task_mode": task["task_mode"],
            "token_param_variant": token_param_variant,
            "requested_token_cap": requested_token_cap,
            "latency_sec": latency_sec,
            "finish_reason": "unknown",
            "completion_tokens": 0,
            "reasoning_tokens": 0,
            "visible_chars": 0,
            "truncated_by_length": False,
            "content_preview": "",
            "error": str(exc),
            "client_meta": {},
        }

    latency_sec = round(time.time() - started_at, 3)
    usage = simplify_usage(response.get("usage", {}))
    content = str(response.get("content") or response.get("llm_output") or "")
    finish_reason = str(response.get("finish_reason") or "unknown")
    completion = extract_completion_tokens(usage)
    reasoning = extract_reasoning_tokens(usage)
    visible = visible_chars(content)
    truncated = truncated_by_length(
        finish_reason=finish_reason,
        content=content,
        completion_tokens=completion,
        requested_token_cap=requested_token_cap,
    )
    return {
        "model": model,
        "backend": backend,
        "task_id": task["task_id"],
        "task_mode": task["task_mode"],
        "token_param_variant": token_param_variant,
        "requested_token_cap": requested_token_cap,
        "latency_sec": latency_sec,
        "finish_reason": finish_reason,
        "completion_tokens": completion,
        "reasoning_tokens": reasoning,
        "visible_chars": visible,
        "truncated_by_length": truncated,
        "content_preview": short_text(content),
        "error": response.get("error", ""),
        "client_meta": response.get("client_meta", {}),
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    invocation = build_invocation_metadata(argv=sys.argv, entrypoint=__file__)
    models = resolve_models(args)
    backends = resolve_backends(args)
    task_ids = resolve_task_ids(args)
    tasks = [TASK_REGISTRY[task_id] for task_id in task_ids]

    rows: List[Dict[str, Any]] = []
    for model in models:
        for backend in backends:
            client = UnifiedChatClient(
                api_key=args.api_key,
                backend=backend,
                api_url=args.api_url,
                host=args.api_host,
                max_retries=args.max_try,
                retry_delay=args.retry_delay,
                timeout=args.timeout,
                model_tag=model,
                verbose=args.verbose,
            )
            for task in tasks:
                for token_param_variant in TOKEN_PARAM_VARIANTS:
                    for token_cap in args.token_caps:
                        for trial in range(1, args.repeat_count + 1):
                            print(
                                f"[TOKEN] model={model} backend={backend} task={task['task_id']} "
                                f"variant={token_param_variant} cap={token_cap} trial={trial}",
                                flush=True,
                            )
                            row = run_single_case(
                                client=client,
                                model=model,
                                backend=backend,
                                task=task,
                                token_param_variant=token_param_variant,
                                requested_token_cap=int(token_cap),
                                print_curl=args.print_curl,
                            )
                            row["trial"] = trial
                            rows.append(row)
                            print(
                                f"  finish={row['finish_reason']} completion={row['completion_tokens']} "
                                f"truncated={row['truncated_by_length']} effective="
                                f"{row.get('client_meta', {}).get('effective_token_param_name')} "
                                f"preview={row['content_preview'] or row['error']}",
                                flush=True,
                            )
                            if args.sleep_between_calls:
                                time.sleep(args.sleep_between_calls)

    grouped_models: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_models[row["model"]].append(row)

    model_summaries: Dict[str, Any] = {}
    for model, model_rows in grouped_models.items():
        preferred_rows = [r for r in model_rows if r["token_param_variant"] == "preferred"]
        alternate_rows = [r for r in model_rows if r["token_param_variant"] != "preferred"]

        preferred_fallback = any(r.get("client_meta", {}).get("token_param_fallback_used") for r in preferred_rows)
        alternate_fallback = any(r.get("client_meta", {}).get("token_param_fallback_used") for r in alternate_rows)
        preferred_errors = [r for r in preferred_rows if r.get("error")]
        alternate_errors = [r for r in alternate_rows if r.get("error")]

        if preferred_errors and len(preferred_errors) == len(preferred_rows):
            preferred_label = "route_blocked"
        elif preferred_fallback:
            preferred_label = "preferred_param_requires_fallback"
        else:
            preferred_label = "preferred_param_working"

        if alternate_errors and len(alternate_errors) == len(alternate_rows):
            alternate_label = "alternate_param_rejected"
        elif alternate_fallback:
            alternate_label = "alternate_param_requires_fallback"
        else:
            alternate_label = "alternate_param_also_works"

        observable = any(r["truncated_by_length"] for r in model_rows if r["requested_token_cap"] == min(args.token_caps))
        token_limit_label = "token_limit_observable" if observable else "token_limit_unverified"

        model_summaries[model] = {
            "preferred_param_label": preferred_label,
            "alternate_param_label": alternate_label,
            "token_limit_label": token_limit_label,
            "resolved_profile": resolve_model_profile(model).to_summary_dict(),
            "rows": model_rows,
        }

    label_counts = Counter(item["token_limit_label"] for item in model_summaries.values())
    summary = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "api_url": args.api_url,
        "models_tested": len(models),
        "backends_tested": backends,
        "task_ids": task_ids,
        "token_caps": list(args.token_caps),
        "repeat_count": args.repeat_count,
        "total_cases": len(rows),
        "label_counts": dict(label_counts),
        "model_summaries": model_summaries,
    }
    report = {
        **invocation,
        "config": redact_sensitive_fields(namespace_to_dict(args)),
        "summary": summary,
        "rows": rows,
    }

    combined_path, rows_path, summary_path, markdown_path = resolve_output_paths(args)
    save_json(combined_path, report)
    save_json(rows_path, {"rows": rows})
    save_json(summary_path, summary)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(build_markdown_summary(summary), encoding="utf-8")

    print(f"[DONE] combined report written to: {combined_path}", flush=True)
    print(f"[DONE] rows written to: {rows_path}", flush=True)
    print(f"[DONE] summary written to: {summary_path}", flush=True)
    print(f"[DONE] markdown summary written to: {markdown_path}", flush=True)


if __name__ == "__main__":
    main()
