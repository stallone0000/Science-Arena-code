# -*- coding: utf-8 -*-
"""Small API connectivity test for interleaved pipelines.

This script is intentionally lightweight to control token cost.
"""

from __future__ import annotations

import argparse
from typing import List

from api_client_interleaved import (
    DEFAULT_API_URL,
    DEFAULT_HOST,
    UnifiedChatClient,
    build_user_message,
    parse_bool,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--api_backend", type=str, default="openai", choices=["openai", "requests"])
    parser.add_argument("--api_url", type=str, default=DEFAULT_API_URL)
    parser.add_argument("--api_host", type=str, default=DEFAULT_HOST)
    parser.add_argument("--api_key", type=str, default="")

    parser.add_argument("--models", nargs="+", default=["volcengine/doubao-seed-2-0-pro"])
    parser.add_argument("--max_tokens", type=int, default=65536)

    parser.add_argument("--image_url", type=str, default="")

    parser.add_argument("--max_try", type=int, default=3)
    parser.add_argument("--retry_delay", type=int, default=20)

    parser.add_argument("--thinking_mode", type=str, default="on", choices=["auto", "on", "off"])
    parser.add_argument(
        "--thinking_effort",
        type=str,
        default="auto",
        choices=["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
    )
    parser.add_argument("--thinking_budget", type=int, default=None)

    parser.add_argument("--reasoning_effort", type=str, default="")
    parser.add_argument("--print_curl", type=parse_bool, default=False)
    parser.add_argument("--verbose", type=parse_bool, default=False)
    return parser


def _short_text(text: str, max_len: int = 180) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _run_once(
    *,
    client: UnifiedChatClient,
    model: str,
    messages: List[dict],
    max_tokens: int,
    reasoning_effort: str,
    thinking_mode: str,
    thinking_effort: str,
    thinking_budget: int | None,
    print_curl: bool,
    case_name: str,
) -> None:
    response = client.chat(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        reasoning_effort=(reasoning_effort or None),
        thinking_mode=thinking_mode,
        thinking_effort=thinking_effort,
        thinking_budget=thinking_budget,
        print_curl=print_curl,
    )
    answer = response.get("content", "") or response.get("llm_output", "")
    error = response.get("error", "")

    print(f"[{case_name}] model={model}")
    if error:
        print(f"  status=ERROR error={error}")
    else:
        print(f"  status=OK answer={_short_text(answer)}")
    print(f"  usage={response.get('usage', {})}")


def main() -> None:
    args = _parser().parse_args()

    client = UnifiedChatClient(
        api_key=args.api_key,
        backend=args.api_backend,
        api_url=args.api_url,
        host=args.api_host,
        max_retries=args.max_try,
        retry_delay=args.retry_delay,
        timeout=300,
        verbose=args.verbose,
    )

    for model in args.models:
        text_messages = [
            build_user_message(
                text="Please reply with exactly one token: API_OK",
                image_urls=[],
                use_multimodal=False,
            )
        ]
        _run_once(
            client=client,
            model=model,
            messages=text_messages,
            max_tokens=args.max_tokens,
            reasoning_effort=args.reasoning_effort,
            thinking_mode=args.thinking_mode,
            thinking_effort=args.thinking_effort,
            thinking_budget=args.thinking_budget,
            print_curl=args.print_curl,
            case_name="TEXT",
        )

        if args.image_url.strip():
            mm_messages = [
                build_user_message(
                    text="Describe this image in one short sentence.",
                    image_urls=[args.image_url.strip()],
                    use_multimodal=True,
                )
            ]
            _run_once(
                client=client,
                model=model,
                messages=mm_messages,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                thinking_mode=args.thinking_mode,
                thinking_effort=args.thinking_effort,
                thinking_budget=args.thinking_budget,
                print_curl=args.print_curl,
                case_name="MULTIMODAL",
            )


if __name__ == "__main__":
    main()
