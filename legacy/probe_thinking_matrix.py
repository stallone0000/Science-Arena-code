# -*- coding: utf-8 -*-
"""Thinking parameter audit and AIME26 probing matrix.

This script is intentionally separate from smoke_api_matrix.py:
- smoke_api_matrix.py answers "can generic API calls work?"
- probe_thinking_matrix.py answers "did the final successful request really keep
  the intended thinking knobs, and can we observe any effect?"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from api_client_interleaved import (
    DEFAULT_API_URL,
    DEFAULT_HOST,
    ThinkingSemanticsError,
    UnifiedChatClient,
    build_supported_thinking_controls,
    build_user_message,
    resolve_thinking_request_semantics,
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
from model_registry import resolve_model_profile, resolve_thinking_capability


AIME26_PROMPT = (
    "Patrick started walking at a constant rate along a straight road from school to the park. "
    "One hour after Patrick left, Tanya started running along the same road from school to the park. "
    "One hour after Tanya left, Jose started bicycling along the same road from school to the park. "
    "Tanya ran at a constant rate of 2 miles per hour faster than Patrick walked, "
    "Jose bicycled at a constant rate of 7 miles per hour faster than Tanya ran, "
    "and all three arrived at the park at the same time. "
    "The distance from the school to the park is \\frac{m}{n} miles, where m and n are relatively prime positive integers. "
    "Find m + n."
)
DEFAULT_EXTRA_MODELS = [
    "openai/gpt-5.4",
    "openai/gpt-5.5",
    "qiniu/openai/gpt-5.4",
    "moonshot/kimi-k2.5",
    "volcengine/doubao-seed-2-0-pro",
    "minimax/MiniMax-M2.5",
]
PROMPT_STYLES = ("answer_only", "concise_derivation")


TASK_REGISTRY: Dict[str, Dict[str, Any]] = {
    "aime26": {
        "task_id": "aime26",
        "task_type": "math_reasoning",
        "source_note": "AIME-style algebra / rate problem used as a compact strongest-vs-weaker reasoning probe.",
        "base_prompt": AIME26_PROMPT,
        "expected_answer": "277",
        "grading_mode": "int_exact",
        "prompt_styles": ("answer_only", "concise_derivation"),
    },
    "logic_boxes": {
        "task_id": "logic_boxes",
        "task_type": "logic_constraint",
        "source_note": "Synthetic multi-constraint puzzle with a single exact answer.",
        "base_prompt": (
            "There are four numbered boxes: 1, 2, 3, 4. Exactly one box contains a gold coin. "
            "The following statements are written on the boxes:\n"
            "Box 1: The coin is not in box 1.\n"
            "Box 2: The coin is in box 4.\n"
            "Box 3: The coin is not in box 2.\n"
            "Box 4: The coin is in box 2.\n"
            "Exactly two of the four statements are true. Which box contains the coin? "
            "Answer with only the box number."
        ),
        "expected_answer": "2",
        "grading_mode": "int_exact",
        "prompt_styles": ("answer_only",),
    },
    "code_trace": {
        "task_id": "code_trace",
        "task_type": "code_output_reasoning",
        "source_note": "Short code-tracing task with an exact numeric output.",
        "base_prompt": (
            "Consider this Python code:\n"
            "def f(n):\n"
            "    total = 0\n"
            "    step = 1\n"
            "    for i in range(1, n + 1):\n"
            "        total += i * step\n"
            "        if i % 2 == 0:\n"
            "            step += 2\n"
            "    return total\n"
            "print(f(7))\n"
            "What exact integer is printed? Answer with only the integer."
        ),
        "expected_answer": "128",
        "grading_mode": "int_exact",
        "prompt_styles": ("answer_only",),
    },
    "ledger_json": {
        "task_id": "ledger_json",
        "task_type": "long_context_synthesis",
        "source_note": "Long-context structured aggregation task that requires exact JSON output.",
        "base_prompt": (
            "You are given an inventory ledger for three items: A, B, C. "
            "Start-of-day inventory is A=10, B=6, C=4.\n"
            "Transactions, in order:\n"
            "1. Receive A +4, B +1.\n"
            "2. Ship A -3, C -1.\n"
            "3. Receive C +5.\n"
            "4. Ship B -2, C -3.\n"
            "5. Audit correction: A +1, B -1.\n"
            "6. Receive B +6.\n"
            "7. Ship A -5, B -4, C -2.\n"
            "8. Receive A +2, C +1.\n"
            "Return exactly this JSON object with no extra text: "
            "{\"A\": <final A>, \"B\": <final B>, \"C\": <final C>, \"total\": <sum>}."
        ),
        "expected_answer": {"A": 9, "B": 6, "C": 4, "total": 19},
        "grading_mode": "json_exact",
        "prompt_styles": ("answer_only",),
    },
    "assignment_opt_json": {
        "task_id": "assignment_opt_json",
        "task_type": "constraint_optimization",
        "source_note": "Synthetic one-to-one assignment optimization task designed to amplify strongest-vs-weaker search differences.",
        "base_prompt": (
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
            "5. If A takes J4, then D must take J2.\n"
            "Find the minimum-total-cost valid assignment. "
            "Return exactly this JSON object with no extra text: "
            "{\"A\":\"J?\",\"B\":\"J?\",\"C\":\"J?\",\"D\":\"J?\",\"E\":\"J?\",\"total\":<int>}."
        ),
        "expected_answer": {"A": "J2", "B": "J5", "C": "J3", "D": "J4", "E": "J1", "total": 36},
        "grading_mode": "json_exact",
        "prompt_styles": ("answer_only",),
    },
    "state_machine_trace_hard": {
        "task_id": "state_machine_trace_hard",
        "task_type": "code_output_reasoning",
        "source_note": "A longer state-machine trace intended to magnify strongest-vs-weaker reasoning differences in exact numeric mode.",
        "base_prompt": (
            "Consider this Python-like pseudocode:\n"
            "acc = 11\n"
            "x = 4\n"
            "y = 9\n"
            "for t in [3, 1, 4, 1, 5, 9, 2, 6, 5, 3]:\n"
            "    acc = (acc * x + y + t) % 101\n"
            "    if acc % 2 == 0:\n"
            "        x = (x + t + y) % 29\n"
            "    else:\n"
            "        y = (y * 2 + x - t) % 31\n"
            "    if (x + y + acc) % 5 == 0:\n"
            "        acc = (acc + 2 * x - y) % 101\n"
            "    if t % 3 == 0:\n"
            "        y = (y + acc + t) % 31\n"
            "After the loop, compute value = 10000*y + 100*x + acc. "
            "What exact integer is value? Answer with only the integer."
        ),
        "expected_answer": "191582",
        "grading_mode": "int_exact",
        "prompt_styles": ("answer_only",),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--api_host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--max_tokens", type=int, default=4096)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max_try", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=10.0)
    parser.add_argument("--repeat_count", type=int, default=3)
    parser.add_argument("--only_models", nargs="*", default=[])
    parser.add_argument("--only_backend", type=str, default="", choices=["", "openai", "requests"])
    parser.add_argument("--task_ids", nargs="*", default=[])
    parser.add_argument("--prompt_styles", nargs="*", default=list(PROMPT_STYLES))
    parser.add_argument("--out_file", type=str, default="")
    parser.add_argument("--sleep_between_calls", type=float, default=0.5)
    parser.add_argument("--print_curl", action="store_true")
    parser.add_argument("--verbose", action="store_true")
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


def resolve_models(args: argparse.Namespace) -> List[str]:
    if args.only_models:
        return unique_items(args.only_models)
    scripts_root = toacl_root() / "scripts"
    pattern = re.compile(r"--(?:pre_model|judge_model)\s+[\"']?([^\"'\s\\]+)[\"']?")
    models = set()
    if scripts_root.exists():
        for script in scripts_root.glob("*.sh"):
            text = script.read_text(encoding="utf-8", errors="ignore")
            for candidate in pattern.findall(text):
                x = (candidate or "").strip()
                if not x:
                    continue
                if re.fullmatch(r"[A-Za-z0-9._/-]+", x):
                    models.add(x)
    for extra in DEFAULT_EXTRA_MODELS:
        models.add(extra)
    return sorted(models)


def resolve_backends(args: argparse.Namespace) -> List[str]:
    if args.only_backend:
        return [args.only_backend]
    return ["openai", "requests"]


def resolve_prompt_styles(args: argparse.Namespace) -> List[str]:
    requested = unique_items([str(x).strip().lower() for x in args.prompt_styles if str(x).strip()])
    if not requested:
        return list(PROMPT_STYLES)
    invalid = [style for style in requested if style not in PROMPT_STYLES]
    if invalid:
        raise ValueError(f"Unsupported prompt styles: {invalid}")
    return requested


def resolve_task_ids(args: argparse.Namespace) -> List[str]:
    if not args.task_ids:
        return list(TASK_REGISTRY.keys())
    requested = unique_items([str(x).strip().lower() for x in args.task_ids if str(x).strip()])
    invalid = [task_id for task_id in requested if task_id not in TASK_REGISTRY]
    if invalid:
        raise ValueError(f"Unsupported task ids: {invalid}")
    return requested


def build_control_requests(model: str) -> List[Dict[str, Any]]:
    capability = resolve_thinking_capability(model)
    controls: List[Dict[str, Any]] = [
        {
            "control_label": "strongest",
            "thinking_mode": capability.benchmark_mode,
            "thinking_effort": capability.benchmark_effort,
        }
    ]
    if capability.supported_efforts:
        weakest_effort = capability.supported_efforts[0]
        if weakest_effort != capability.benchmark_effort:
            controls.append(
                {
                    "control_label": "weakest",
                    "thinking_mode": "on",
                    "thinking_effort": weakest_effort,
                }
            )
    if capability.off_support == "strict_supported":
        controls.append(
            {
                "control_label": "off",
                "thinking_mode": "off",
                "thinking_effort": "auto",
            }
        )
    return controls


def build_case_prompt(task: Dict[str, Any], prompt_style: str) -> str:
    base_prompt = str(task["base_prompt"])
    if prompt_style == "answer_only":
        if task["grading_mode"] == "json_exact":
            return base_prompt
        return base_prompt
    if prompt_style == "concise_derivation":
        if task["task_id"] == "aime26":
            return (
                f"{base_prompt}\n"
                "Give a concise derivation in at most 8 lines, then end with a line exactly of the form Final: <integer>."
            )
        return base_prompt
    raise ValueError(f"Unsupported prompt style: {prompt_style}")


def build_case_messages(task: Dict[str, Any], prompt_style: str) -> List[Dict[str, Any]]:
    return [
        build_user_message(
            text=build_case_prompt(task, prompt_style),
            image_urls=[],
            use_multimodal=False,
        )
    ]


def short_text(text: str, max_len: int = 240) -> str:
    value = (text or "").replace("\n", " ").strip()
    if len(value) <= max_len:
        return value
    return value[:max_len] + "..."


def extract_visible_reasoning(response: Dict[str, Any]) -> Tuple[str, str]:
    reasoning_content = (response.get("reasoning_content") or "").strip()
    if reasoning_content:
        return reasoning_content, "reasoning_content"

    for source_name in ("content", "llm_output"):
        text = str(response.get(source_name) or "")
        match = re.search(r"<think>\s*(.*?)\s*</think>", text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip(), source_name

    return "", "none"


def extract_answer_int(text: str, *, prompt_style: str) -> str:
    text = text or ""
    if prompt_style == "concise_derivation":
        match = re.search(r"Final:\s*(-?\d+)", text, flags=re.IGNORECASE)
        if match:
            return match.group(1)
        candidates = re.findall(r"-?\d+", text)
        return candidates[-1] if candidates else ""

    match = re.search(r"-?\d+", text)
    return match.group(0) if match else ""


def canonicalize_json_answer(text: str) -> Dict[str, Any] | None:
    try:
        start = text.index("{")
        end = text.rindex("}") + 1
    except ValueError:
        return None
    try:
        data = json.loads(text[start:end])
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    return data


def grade_task_output(
    task: Dict[str, Any],
    *,
    prompt_style: str,
    content_text: str,
) -> Tuple[str, bool]:
    if task["grading_mode"] == "int_exact":
        answer = extract_answer_int(content_text, prompt_style=prompt_style)
        return answer, answer == str(task["expected_answer"])
    if task["grading_mode"] == "json_exact":
        parsed = canonicalize_json_answer(content_text)
        if parsed is None:
            return "", False
        expected = task["expected_answer"]
        return json.dumps(parsed, sort_keys=True, ensure_ascii=False), parsed == expected
    raise ValueError(f"Unsupported grading mode: {task['grading_mode']}")


def extract_reasoning_tokens(usage: Dict[str, Any]) -> int:
    if not isinstance(usage, dict):
        return 0
    if "reasoning_tokens" in usage and isinstance(usage["reasoning_tokens"], int):
        return int(usage["reasoning_tokens"])
    for key in ("completion_tokens_details", "output_tokens_details"):
        details = usage.get(key)
        if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
            return int(details["reasoning_tokens"])
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


def classify_model_rows(rows: Sequence[Dict[str, Any]], capability: Dict[str, Any]) -> str:
    if any(row.get("semantic_failure") for row in rows):
        return "semantic_failure"
    if any(row.get("client_meta", {}).get("thinking_fields_preserved_in_final_success") is False for row in rows):
        return "final_request_dropped_thinking"
    if capability.get("current_client_mapping") == "likely_wrong":
        return "mapping_bug"
    if capability.get("current_client_mapping") == "no_public_knob":
        return "no_public_knob"
    if capability.get("off_support") == "toggle_not_supported":
        return "no_public_knob"
    return "controls_working_but_hidden"


def build_control_observability_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    by_label: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("unsupported_request") or row.get("semantic_failure"):
            continue
        by_label[str(row.get("control_label", ""))].append(row)

    def avg(label: str, field: str) -> float:
        xs = [float(row.get(field) or 0.0) for row in by_label.get(label, [])]
        return round(sum(xs) / len(xs), 3) if xs else 0.0

    def rate(label: str, pred) -> float:
        xs = by_label.get(label, [])
        if not xs:
            return 0.0
        return round(sum(1 for row in xs if pred(row)) / len(xs), 4)

    strongest_label = "strongest"
    contrast_label = "weakest" if by_label.get("weakest") else ("off" if by_label.get("off") else "")
    summary = {
        "available_controls": sorted([label for label, values in by_label.items() if values]),
        "strongest_avg_reasoning_tokens": avg(strongest_label, "reasoning_tokens"),
        "contrast_label": contrast_label,
        "contrast_avg_reasoning_tokens": avg(contrast_label, "reasoning_tokens") if contrast_label else 0.0,
        "strongest_avg_latency_sec": avg(strongest_label, "latency_sec"),
        "contrast_avg_latency_sec": avg(contrast_label, "latency_sec") if contrast_label else 0.0,
        "strongest_visible_reasoning_rate": rate(strongest_label, lambda row: (row.get("visible_reasoning_chars") or 0) > 0),
        "contrast_visible_reasoning_rate": (
            rate(contrast_label, lambda row: (row.get("visible_reasoning_chars") or 0) > 0) if contrast_label else 0.0
        ),
        "strongest_correct_rate": rate(strongest_label, lambda row: bool(row.get("correct"))),
        "contrast_correct_rate": rate(contrast_label, lambda row: bool(row.get("correct"))) if contrast_label else 0.0,
    }
    observable_effect = False
    if contrast_label:
        if abs(summary["strongest_avg_reasoning_tokens"] - summary["contrast_avg_reasoning_tokens"]) >= 50:
            observable_effect = True
        if abs(summary["strongest_visible_reasoning_rate"] - summary["contrast_visible_reasoning_rate"]) >= 0.25:
            observable_effect = True
        if abs(summary["strongest_avg_latency_sec"] - summary["contrast_avg_latency_sec"]) >= 2.0:
            observable_effect = True
        if abs(summary["strongest_correct_rate"] - summary["contrast_correct_rate"]) >= 0.25:
            observable_effect = True
    summary["observable_effect_detected"] = observable_effect
    return summary


def case_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    count = len(rows)
    correct = sum(1 for row in rows if row.get("correct"))
    visible_reasoning = sum(1 for row in rows if (row.get("visible_reasoning_chars") or 0) > 0)
    unsupported = sum(1 for row in rows if row.get("unsupported_request"))
    semantic_failures = sum(1 for row in rows if row.get("semantic_failure"))
    preserved_values = [
        row.get("client_meta", {}).get("thinking_fields_preserved_in_final_success")
        for row in rows
        if row.get("client_meta", {}).get("thinking_fields_preserved_in_final_success") is not None
    ]
    return {
        "trials": count,
        "correct_count": correct,
        "correct_rate": round(correct / count, 4) if count else 0.0,
        "unsupported_count": unsupported,
        "semantic_failure_count": semantic_failures,
        "answers": [row.get("answer_int") or "" for row in rows],
        "avg_latency_sec": round(sum(float(row.get("latency_sec") or 0.0) for row in rows) / count, 3) if count else 0.0,
        "avg_visible_reasoning_chars": round(
            sum(int(row.get("visible_reasoning_chars") or 0) for row in rows) / count, 1
        ) if count else 0.0,
        "visible_reasoning_rate": round(visible_reasoning / count, 4) if count else 0.0,
        "avg_reasoning_tokens": round(sum(int(row.get("reasoning_tokens") or 0) for row in rows) / count, 1)
        if count
        else 0.0,
        "thinking_fields_preserved_rate": (
            round(sum(1 for value in preserved_values if value) / len(preserved_values), 4)
            if preserved_values
            else None
        ),
    }


def resolve_output_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path]:
    if args.out_file:
        combined = Path(args.out_file).expanduser().resolve()
    else:
        combined = default_smoke_root() / "thinking_matrix" / f"thinking_matrix_{timestamp_tag()}.json"
    rows_path = combined.with_name(f"{combined.stem}_rows.json")
    summary_path = combined.with_name(f"{combined.stem}_summary.json")
    markdown_path = combined.with_name(f"{combined.stem}_summary.md")
    return combined, rows_path, summary_path, markdown_path


def build_markdown_summary(summary: Dict[str, Any]) -> str:
    lines = [
        "# Thinking Audit Summary",
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
            f"- `{model}`: `{item['model_label']}` | family=`{item['thinking_capability']['family']}` | "
            f"off_support=`{item['thinking_capability']['off_support']}`"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- `thinking_mode=on` + `thinking_effort=auto` is interpreted as the benchmark-strongest supported thinking request for that model.")
    lines.append("- `unsupported_request` means the model/provider combination does not support the requested on/off/effort semantics.")
    lines.append("- `semantic_failure` means the request shape was theoretically supported, but the final successful request could not preserve the intended thinking semantics.")
    lines.append("- `final_request_dropped_thinking` means the final successful request lost at least one derived thinking field during compatibility retries.")
    lines.append("- `off_not_supported` means provider docs do not promise a strict thinking-off mode.")
    lines.append("- `toggle_not_supported` means the official route does not expose a trustworthy public on/off knob for this model family.")
    return "\n".join(lines) + "\n"


def run_single_case(
    *,
    client: UnifiedChatClient,
    model: str,
    backend: str,
    task: Dict[str, Any],
    control: Dict[str, Any],
    prompt_style: str,
    max_tokens: int,
    print_curl: bool,
) -> Dict[str, Any]:
    messages = build_case_messages(task, prompt_style)
    resolved_profile = resolve_model_profile(model)
    try:
        semantics = resolve_thinking_request_semantics(
            model=model,
            max_tokens=max_tokens,
            thinking_mode=str(control["thinking_mode"]),
            thinking_effort=str(control["thinking_effort"]),
            resolved_profile=resolved_profile,
        )
    except ThinkingSemanticsError as exc:
        return {
            "model": model,
            "backend": backend,
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "source_note": task["source_note"],
            "grading_mode": task["grading_mode"],
            "control_label": control["control_label"],
            "prompt_style": prompt_style,
            "thinking_mode": control["thinking_mode"],
            "thinking_effort": control["thinking_effort"],
            "latency_sec": 0.0,
            "answer_int": "",
            "correct": False,
            "content_preview": "",
            "visible_reasoning_chars": 0,
            "visible_reasoning_source": "none",
            "reasoning_tokens": 0,
            "usage": {},
            "error": str(exc),
            "unsupported_request": True,
            "semantic_failure": False,
            "client_meta": {
                "resolved_profile": resolved_profile.to_summary_dict(),
                "resolved_thinking_family": resolved_profile.profile.thinking_family,
                "request_semantics_valid": False,
                "semantic_failure_reason": str(exc),
                "supported_thinking_controls": exc.supported_thinking_controls,
                "recommended_benchmark_thinking_request": exc.recommended_benchmark_thinking_request,
                "normalized_provider_thinking_request": exc.normalized_provider_thinking_request,
                "derived_thinking_options": {},
                "final_payload_thinking_fields": {},
                "dropped_thinking_fields": [],
                "thinking_drop_events": [],
                "thinking_fields_preserved_in_final_success": None,
                "control_observability_summary": {
                    "visible_reasoning_chars": 0,
                    "visible_reasoning_source": "none",
                    "reasoning_tokens": 0,
                },
            },
        }
    started_at = time.time()
    try:
        response = client.chat(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            thinking_mode=str(control["thinking_mode"]),
            thinking_effort=str(control["thinking_effort"]),
            print_curl=print_curl,
        )
        semantic_failure = False
    except Exception as exc:
        latency_sec = round(time.time() - started_at, 3)
        return {
            "model": model,
            "backend": backend,
            "task_id": task["task_id"],
            "task_type": task["task_type"],
            "source_note": task["source_note"],
            "grading_mode": task["grading_mode"],
            "control_label": control["control_label"],
            "prompt_style": prompt_style,
            "thinking_mode": control["thinking_mode"],
            "thinking_effort": control["thinking_effort"],
            "latency_sec": latency_sec,
            "answer_int": "",
            "correct": False,
            "content_preview": "",
            "visible_reasoning_chars": 0,
            "visible_reasoning_source": "none",
            "reasoning_tokens": 0,
            "usage": {},
            "error": str(exc),
            "unsupported_request": False,
            "semantic_failure": True,
            "client_meta": {
                "resolved_profile": resolved_profile.to_summary_dict(),
                "resolved_thinking_family": resolved_profile.profile.thinking_family,
                "request_semantics_valid": True,
                "semantic_failure_reason": str(exc),
                "supported_thinking_controls": build_supported_thinking_controls(resolved_profile),
                "recommended_benchmark_thinking_request": semantics["recommended_benchmark_thinking_request"],
                "normalized_provider_thinking_request": semantics["normalized_provider_thinking_request"],
                "derived_thinking_options": semantics["normalized_provider_thinking_request"],
                "final_payload_thinking_fields": {},
                "dropped_thinking_fields": [],
                "thinking_drop_events": [],
                "thinking_fields_preserved_in_final_success": None,
                "control_observability_summary": {
                    "visible_reasoning_chars": 0,
                    "visible_reasoning_source": "none",
                    "reasoning_tokens": 0,
                },
            },
        }
    latency_sec = round(time.time() - started_at, 3)
    content = str(response.get("content") or "").strip()
    visible_reasoning, reasoning_source = extract_visible_reasoning(response)
    usage = simplify_usage(response.get("usage", {}))
    content_for_grading = content or str(response.get("llm_output") or "")
    normalized_answer, correct = grade_task_output(
        task,
        prompt_style=prompt_style,
        content_text=content_for_grading,
    )
    client_meta = response.get("client_meta", {})
    client_meta = dict(client_meta)
    client_meta["control_observability_summary"] = {
        "visible_reasoning_chars": len(visible_reasoning),
        "visible_reasoning_source": reasoning_source,
        "reasoning_tokens": extract_reasoning_tokens(usage),
    }
    return {
        "model": model,
        "backend": backend,
        "task_id": task["task_id"],
        "task_type": task["task_type"],
        "source_note": task["source_note"],
        "grading_mode": task["grading_mode"],
        "control_label": control["control_label"],
        "prompt_style": prompt_style,
        "thinking_mode": control["thinking_mode"],
        "thinking_effort": control["thinking_effort"],
        "latency_sec": latency_sec,
        "answer_int": normalized_answer,
        "correct": correct,
        "content_preview": short_text(content_for_grading),
        "visible_reasoning_chars": len(visible_reasoning),
        "visible_reasoning_source": reasoning_source,
        "reasoning_tokens": extract_reasoning_tokens(usage),
        "usage": usage,
        "error": response.get("error", ""),
        "unsupported_request": False,
        "semantic_failure": semantic_failure,
        "client_meta": client_meta,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    invocation = build_invocation_metadata(argv=sys.argv, entrypoint=__file__)
    models = resolve_models(args)
    backends = resolve_backends(args)
    prompt_styles = resolve_prompt_styles(args)
    task_ids = resolve_task_ids(args)
    tasks = [TASK_REGISTRY[task_id] for task_id in task_ids]

    rows: List[Dict[str, Any]] = []
    static_audit: Dict[str, Any] = {}
    for model in models:
        resolved_profile = resolve_model_profile(model)
        static_audit[model] = {
            "resolved_profile": resolved_profile.to_summary_dict(),
            "thinking_capability": resolve_thinking_capability(resolved_profile).to_summary_dict(),
        }

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
            controls = build_control_requests(model)
            for task in tasks:
                task_prompt_styles = [style for style in prompt_styles if style in task["prompt_styles"]]
                if not task_prompt_styles:
                    continue
                for prompt_style in task_prompt_styles:
                    for control in controls:
                        for trial in range(1, args.repeat_count + 1):
                            print(
                                f"[PROBE] model={model} backend={backend} task={task['task_id']} style={prompt_style} "
                                f"control={control['control_label']} mode={control['thinking_mode']} "
                                f"effort={control['thinking_effort']} trial={trial}",
                                flush=True,
                            )
                            row = run_single_case(
                                client=client,
                                model=model,
                                backend=backend,
                                task=task,
                                control=control,
                                prompt_style=prompt_style,
                                max_tokens=args.max_tokens,
                                print_curl=args.print_curl,
                            )
                            row["trial"] = trial
                            rows.append(row)
                            preview = row["content_preview"] or row["error"]
                            print(
                                f"  correct={row['correct']} answer={row['answer_int'] or '<none>'} "
                                f"latency={row['latency_sec']} preserved="
                                f"{row['client_meta'].get('thinking_fields_preserved_in_final_success')} "
                                f"preview={preview}",
                                flush=True,
                            )
                            if args.sleep_between_calls:
                                time.sleep(args.sleep_between_calls)

    grouped_cases: Dict[Tuple[str, str, str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    grouped_models: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped_cases[(row["model"], row["backend"], row["task_id"], row["prompt_style"], row["control_label"])].append(row)
        grouped_models[row["model"]].append(row)

    per_case_summary = {}
    for key, case_rows in grouped_cases.items():
        model, backend, task_id, prompt_style, control_label = key
        per_case_summary[f"{model}::{backend}::{task_id}::{prompt_style}::{control_label}"] = case_summary(case_rows)

    model_summaries: Dict[str, Any] = {}
    label_counts: Counter[str] = Counter()
    for model, model_rows in grouped_models.items():
        capability = static_audit[model]["thinking_capability"]
        label = classify_model_rows(model_rows, capability)
        observability = build_control_observability_summary(model_rows)
        if label == "controls_working_but_hidden":
            if observability["observable_effect_detected"]:
                if (
                    observability["strongest_avg_reasoning_tokens"] != observability["contrast_avg_reasoning_tokens"]
                    or observability["strongest_visible_reasoning_rate"] != observability["contrast_visible_reasoning_rate"]
                ):
                    label = "controls_working_and_observable"
                else:
                    label = "controls_working_but_hidden"
            elif observability["available_controls"] and len(observability["available_controls"]) > 1:
                label = "controls_unverified_or_ignored"
        label_counts[label] += 1
        model_summaries[model] = {
            "model_label": label,
            "thinking_capability": capability,
            "resolved_profile": static_audit[model]["resolved_profile"],
            "control_observability_summary": observability,
            "case_summaries": {
                key: value
                for key, value in per_case_summary.items()
                if key.startswith(f"{model}::")
            },
        }

    summary = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "api_url": args.api_url,
        "models_tested": len(models),
        "backends_tested": backends,
        "task_ids": task_ids,
        "prompt_styles": prompt_styles,
        "repeat_count": args.repeat_count,
        "total_cases": len(rows),
        "label_counts": dict(label_counts),
        "model_summaries": model_summaries,
    }
    report = {
        **invocation,
        "config": redact_sensitive_fields(namespace_to_dict(args)),
        "static_audit": static_audit,
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
