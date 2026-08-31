#!/usr/bin/env python3
"""Live two-turn Responses probe with a direct image on turn two."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pipeline


FIRST_TOKEN = "SCIENCE-ARENA-RESPONSES-TURN1-OK"
PIXEL_PROMPT = (
    "Inspect only the attached image pixels. Return exactly one line: "
    "token=<token>; triangles=<number of blue triangles>; "
    "arrow=<direction the red arrow points>."
)


def pixel_passed(text: str) -> bool:
    normalized = re.sub(r"[\s,;:_`$\\(){}\[\].-]+", "", text.casefold())
    return (
        "token=q7mx4829" in normalized
        and "triangles=4" in normalized
        and any(
            token in normalized
            for token in (
                "arrow=northwest",
                "arrow=nw",
                "arrow=upleft",
                "arrow=upperleft",
                "arrow=topleft",
            )
        )
    )


def call(
    *,
    model: dict[str, Any],
    messages: list[dict[str, Any]],
    attempt_dir: Path,
    api_key: str,
    validator: Any,
) -> tuple[dict[str, Any] | None, list[dict[str, Any]], str, dict[str, Any]]:
    payload = pipeline.base_payload(model, messages, judge=False)
    response, attempts, error = pipeline.call_with_retries(
        payload_factory=lambda _feedback: payload,
        api_key=api_key,
        attempt_dir=attempt_dir,
        timeout_seconds=int(model["timeout_seconds"]),
        max_attempts=int(model["max_attempts"]),
        validator=validator,
    )
    return response, attempts, error, payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--route", default="openai/gpt-5.5")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    models = {str(model["route"]): model for model in pipeline.load_model_manifest(args.models.resolve())}
    if args.route not in models:
        raise RuntimeError(f"route absent from manifest: {args.route}")
    model = models[args.route]
    if model.get("api_mode") != "responses":
        raise RuntimeError(f"route is not configured for Responses: {args.route}")
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    api_key = pipeline.load_api_key()

    first_messages = [
        {"role": "system", "content": pipeline.SOLVER_SYSTEM},
        {"role": "user", "content": f"Return exactly {FIRST_TOKEN}"},
    ]

    def validate_first(response: dict[str, Any]) -> tuple[bool, str]:
        content, _, _ = pipeline.response_text(response)
        passed = FIRST_TOKEN in content
        return passed, "turn-one token passed" if passed else "turn-one token failed"

    first, first_attempts, first_error, first_payload = call(
        model=model,
        messages=first_messages,
        attempt_dir=output_dir / "attempts" / "turn1",
        api_key=api_key,
        validator=validate_first,
    )
    first_content, first_reasoning, first_finish = pipeline.response_text(first or {})
    if first is None:
        result = {
            "schema_version": "science_arena_responses_multiturn_probe_v1",
            "created_at": pipeline.utc_now(),
            "route": args.route,
            "passed": False,
            "turn1": {
                "passed": False,
                "attempt_count": len(first_attempts),
                "error": first_error,
                "request": pipeline.sanitize_for_sidecar(first_payload),
            },
            "turn2": None,
        }
        pipeline.atomic_write_json(output_dir / "result.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 2

    second_messages = first_messages + [
        {"role": "assistant", "content": first_content},
        pipeline.make_image_message(PIXEL_PROMPT, [image_path]),
    ]

    def validate_second(response: dict[str, Any]) -> tuple[bool, str]:
        content, _, _ = pipeline.response_text(response)
        passed = pixel_passed(content)
        return passed, "turn-two direct pixel check passed" if passed else "turn-two direct pixel check failed"

    second, second_attempts, second_error, second_payload = call(
        model=model,
        messages=second_messages,
        attempt_dir=output_dir / "attempts" / "turn2",
        api_key=api_key,
        validator=validate_second,
    )
    second_content, second_reasoning, second_finish = pipeline.response_text(second or {})
    result = {
        "schema_version": "science_arena_responses_multiturn_probe_v1",
        "created_at": pipeline.utc_now(),
        "route": args.route,
        "passed": second is not None,
        "turn1": {
            "passed": True,
            "attempt_count": len(first_attempts),
            "visible_content": first_content,
            "reasoning_content": first_reasoning,
            "finish_reason": first_finish,
            "request_sha256": pipeline.sha256_bytes(pipeline.canonical_json_bytes(first_payload)),
            "request": pipeline.sanitize_for_sidecar(first_payload),
            "raw_response": first,
        },
        "turn2": {
            "passed": second is not None,
            "attempt_count": len(second_attempts),
            "error": second_error,
            "visible_content": second_content,
            "reasoning_content": second_reasoning,
            "finish_reason": second_finish,
            "request_sha256": pipeline.sha256_bytes(pipeline.canonical_json_bytes(second_payload)),
            "request": pipeline.sanitize_for_sidecar(second_payload),
            "raw_response": second,
        },
        "image": pipeline.image_descriptor(image_path),
    }
    pipeline.atomic_write_json(output_dir / "result.json", result)
    print(json.dumps({
        "route": result["route"],
        "passed": result["passed"],
        "turn1_attempts": len(first_attempts),
        "turn2_attempts": len(second_attempts),
        "turn2_visible_content": second_content,
        "turn2_error": second_error,
    }, ensure_ascii=False, indent=2))
    return 0 if result["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
