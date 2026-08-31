# -*- coding: utf-8 -*-
"""Model API smoke test for ScienceArena.

This script first validates whether generic API calls work for each model/backend.
It intentionally uses a tiny fixed prompt rather than benchmark data so we can
separate "API route is broken" from "evaluation pipeline is broken".
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from api_client_interleaved import (
    DEFAULT_API_URL,
    DEFAULT_HOST,
    UnifiedChatClient,
    build_user_message,
    describe_model_modality,
    resolve_multimodal_mode,
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


MODEL_FAMILY_KEYWORDS = (
    "gemini",
    "claude",
    "gpt",
    "o1",
    "o3",
    "deepseek",
    "qwen",
    "glm",
    "doubao",
    "minimax",
    "kimi",
)

DEFAULT_IMAGE_URL = "https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.5/demo/CI_Demo/mathv-1327.jpg"
MULTIMODAL_QUESTION = (
    "The centres of the four illustrated circles are in the corners of the square. "
    "The two big circles touch each other and also the two little circles. "
    "With which factor do you have to multiply the radii of the little circles to obtain the radius of the big circles?\n"
    "Choices:\n"
    "(A) $\\frac{2}{9}$\n"
    "(B) $\\sqrt{5}$\n"
    "(C) $0.8 \\cdot \\pi$\n"
    "(D) 2.5\n"
    "(E) $1+\\sqrt{2}$\n"
    "Answer with only the option letter."
)
TEXT_QUESTION = (
    "Please solve a tiny reasoning check and answer with only one token.\n"
    "Question: what is 7 + 5?\n"
    "Answer with only: 12"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_key", type=str, default="")
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--api_host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--max_try", type=int, default=4)
    parser.add_argument("--retry_delay", type=int, default=15)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--thinking_mode", type=str, default="on", choices=["auto", "on", "off"])
    parser.add_argument(
        "--thinking_effort",
        type=str,
        default="auto",
        choices=["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--thinking_budget", type=int, default=None)
    parser.add_argument("--reasoning_effort", type=str, default="")
    parser.add_argument("--only_backend", type=str, default="", choices=["", "openai", "requests"])
    parser.add_argument("--only_models", nargs="*", default=[])
    parser.add_argument("--extra_models", nargs="*", default=[])
    parser.add_argument("--sleep_between_calls", type=float, default=0.5)
    parser.add_argument("--print_curl", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--out_file", type=str, default="")
    return parser.parse_args()


def extract_models_from_scripts() -> List[str]:
    scripts_root = toacl_root() / "scripts"
    if not scripts_root.exists():
        return []

    pattern = re.compile(r"--(?:pre_model|judge_model)\s+[\"']?([^\"'\s\\]+)[\"']?")
    models = set()
    for script in scripts_root.glob("*.sh"):
        text = script.read_text(encoding="utf-8", errors="ignore")
        for candidate in pattern.findall(text):
            if _is_probable_model_name(candidate):
                models.add(candidate)
    return sorted(models)


def _is_probable_model_name(candidate: str) -> bool:
    x = (candidate or "").strip()
    if not x:
        return False
    if not re.fullmatch(r"[A-Za-z0-9._/-]+", x):
        return False
    xl = x.lower()
    if xl in {"true", "false", "none"}:
        return False
    if any(keyword in xl for keyword in MODEL_FAMILY_KEYWORDS):
        return True
    return "/" in x


def short_text(text: str, max_len: int = 200) -> str:
    value = (text or "").replace("\n", " ").strip()
    if len(value) <= max_len:
        return value
    return value[:max_len] + "..."


def resolve_case_messages(model: str) -> tuple[List[Dict[str, Any]], bool, str]:
    use_multimodal = resolve_multimodal_mode(model=model, mode="auto", explicit_override=None)
    modality_note = describe_model_modality(model)
    if use_multimodal:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": DEFAULT_IMAGE_URL,
                        },
                    },
                    {
                        "type": "text",
                        "text": MULTIMODAL_QUESTION,
                    },
                ],
            }
        ]
    else:
        messages = [
            build_user_message(
                text=TEXT_QUESTION,
                image_urls=[],
                use_multimodal=False,
            )
        ]
    return messages, use_multimodal, modality_note


def run_single_case(
    *,
    backend: str,
    model: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
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
    messages, use_multimodal, modality_note = resolve_case_messages(model)
    started_at = time.time()
    response = client.chat(
        model=model,
        messages=messages,
        max_tokens=args.max_tokens,
        thinking_mode=args.thinking_mode,
        thinking_effort=args.thinking_effort,
        thinking_budget=args.thinking_budget,
        reasoning_effort=(args.reasoning_effort or None),
        print_curl=args.print_curl,
    )
    elapsed = round(time.time() - started_at, 3)
    answer = response.get("content", "") or response.get("llm_output", "")
    error = response.get("error", "")
    client_meta = response.get("client_meta", {})
    return {
        "backend": backend,
        "model": model,
        "ok": bool(answer.strip()) and not error,
        "resolved_use_multimodal": use_multimodal,
        "resolved_input_mode": "multimodal" if use_multimodal else "text",
        "model_modality_note": modality_note,
        "thinking_mode": args.thinking_mode,
        "thinking_effort": args.thinking_effort,
        "latency_sec": elapsed,
        "content_preview": short_text(answer),
        "usage": response.get("usage", {}),
        "error": error,
        "client_meta": client_meta,
    }


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    invocation = build_invocation_metadata(argv=sys.argv, entrypoint=__file__)
    models = extract_models_from_scripts()
    for extra in args.extra_models:
        if extra not in models:
            models.append(extra)
    if args.only_models:
        for model in args.only_models:
            if model not in models:
                models.append(model)
        wanted = set(args.only_models)
        models = [model for model in models if model in wanted]
    models = sorted(models)

    backends = [args.only_backend] if args.only_backend else ["openai", "requests"]
    report_rows: List[Dict[str, Any]] = []
    for backend in backends:
        for model in models:
            print(f"[SMOKE] backend={backend} model={model}", flush=True)
            row = run_single_case(backend=backend, model=model, args=args)
            report_rows.append(row)
            preview = row["content_preview"] or row["error"]
            print(
                f"  ok={row['ok']} input={row['resolved_input_mode']} "
                f"thinking={row['thinking_mode']} latency={row['latency_sec']} preview={preview}",
                flush=True,
            )
            if args.sleep_between_calls:
                time.sleep(args.sleep_between_calls)

    summary = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "models_tested": len(models),
        "backends_tested": backends,
        "total_cases": len(report_rows),
        "ok_cases": sum(1 for row in report_rows if row["ok"]),
        "failed_cases": sum(1 for row in report_rows if not row["ok"]),
        "api_url": args.api_url,
        "thinking_mode": args.thinking_mode,
        "thinking_effort": args.thinking_effort,
    }
    report = {
        **invocation,
        "config": redact_sensitive_fields(namespace_to_dict(args)),
        "summary": summary,
        "rows": report_rows,
    }
    out_path = (
        Path(args.out_file).expanduser().resolve()
        if args.out_file
        else default_smoke_root() / "api_matrix" / f"api_matrix_{timestamp_tag()}.json"
    )
    save_json(out_path, report)
    print(f"[DONE] report written to: {out_path}", flush=True)


if __name__ == "__main__":
    main()
