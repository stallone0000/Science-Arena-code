#!/usr/bin/env python3
"""Exercise every frozen Table 6 route with its exact formal request options.

This is a transport/options preflight, not a benchmark score. Vision routes are
required to read pixels from the synthetic nonce image through a native image
block. Text-staticized routes receive text only. The calibrated judge route is
also checked with its judge-specific options and a native image block.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
from pathlib import Path
from typing import Any

import pipeline


TEXT_TOKEN = "SCIENCE-ARENA-FROZEN-PREFLIGHT-OK"
VISION_PROMPT = (
    "Inspect only the attached image pixels. Return exactly one line in this "
    "format: token=<token>; triangles=<number of blue triangles>; "
    "arrow=<direction the red arrow points>. Do not guess or add prose."
)


def compact(value: str) -> str:
    return re.sub(r"[\s,;:_`$\\(){}\[\].]+", "", value.casefold())


def vision_content_passed(value: str) -> bool:
    normalized = compact(value)
    arrow = any(
        token in normalized
        for token in (
            "arrow=northwest",
            "arrow=nw",
            "arrow=up-left",
            "arrow=upleft",
            "arrow=upper-left",
            "arrow=upperleft",
            "arrow=top-left",
            "arrow=topleft",
        )
    )
    return (
        "token=q7mx-4829" in normalized
        and "triangles=4" in normalized
        and arrow
    )


def message_for(model: dict[str, Any], image_path: Path) -> dict[str, Any]:
    if model["input_mode"] == "vision":
        return pipeline.make_image_message(VISION_PROMPT, [image_path])
    return {
        "role": "user",
        "content": f"Return exactly this token and nothing else: {TEXT_TOKEN}",
    }


def validator_for(input_mode: str):
    def validate(response: dict[str, Any]) -> tuple[bool, str]:
        content, _, _ = pipeline.response_text(response)
        if input_mode == "vision":
            passed = vision_content_passed(content)
            return passed, "pixel nonce/shape check passed" if passed else "pixel nonce/shape check failed"
        passed = TEXT_TOKEN in content
        return passed, "text instruction check passed" if passed else "text instruction check failed"

    return validate


def run_one(
    *,
    model: dict[str, Any],
    image_path: Path,
    output_dir: Path,
    api_key: str,
    judge: bool = False,
) -> dict[str, Any]:
    route = str(model["route"])
    input_mode = "vision" if judge else str(model["input_mode"])
    message = (
        pipeline.make_image_message(VISION_PROMPT, [image_path])
        if judge
        else message_for(model, image_path)
    )
    payload = pipeline.base_payload(model, [message], judge=judge)
    kind = "judge" if judge else "solver"
    attempt_dir = output_dir / "attempts" / pipeline.safe_name(route) / kind
    response, attempts, final_error = pipeline.call_with_retries(
        payload_factory=lambda _feedback: payload,
        api_key=api_key,
        attempt_dir=attempt_dir,
        timeout_seconds=int(model["timeout_seconds"]),
        max_attempts=int(model["max_attempts"]),
        validator=validator_for(input_mode),
    )
    content, reasoning, finish = pipeline.response_text(response or {})
    result = {
        "schema_version": "science_arena_frozen_route_preflight_v1",
        "created_at": pipeline.utc_now(),
        "route": route,
        "kind": kind,
        "declared_input_mode": model["input_mode"],
        "preflight_input_mode": input_mode,
        "passed": response is not None,
        "attempt_count": len(attempts),
        "timeout_seconds": model["timeout_seconds"],
        "max_attempts": model["max_attempts"],
        "request_sha256": pipeline.sha256_bytes(pipeline.canonical_json_bytes(payload)),
        "request": pipeline.sanitize_for_sidecar(payload),
        "visible_content": content,
        "reasoning_content": reasoning,
        "finish_reason": finish,
        "final_error": final_error,
        "raw_response": response,
    }
    suffix = "__judge.json" if judge else ".json"
    pipeline.atomic_write_json(output_dir / f"{pipeline.safe_name(route)}{suffix}", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--judge-route", default="google/gemini-3.5-flash")
    parser.add_argument("--routes", default="")
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--base-summary", type=Path)
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    models_path = args.models.resolve()
    image_path = args.image.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    all_models = pipeline.load_model_manifest(models_path)
    routes = {str(model["route"]): model for model in all_models}
    wanted = {value.strip() for value in args.routes.split(",") if value.strip()}
    missing = wanted - set(routes)
    if missing:
        raise RuntimeError(f"selected routes are absent from manifest: {sorted(missing)}")
    models = all_models if not wanted else [model for model in all_models if str(model["route"]) in wanted]
    if not args.skip_judge and args.judge_route not in routes:
        raise RuntimeError(f"judge route is absent from manifest: {args.judge_route}")
    api_key = pipeline.load_api_key()
    jobs = [(model, False) for model in models]
    if not args.skip_judge:
        jobs.append((routes[args.judge_route], True))
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(jobs))) as executor:
        futures = [
            executor.submit(
                run_one,
                model=model,
                image_path=image_path,
                output_dir=output_dir,
                api_key=api_key,
                judge=judge,
            )
            for model, judge in jobs
        ]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                json.dumps(
                    {
                        "route": result["route"],
                        "kind": result["kind"],
                        "passed": result["passed"],
                        "attempt_count": result["attempt_count"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    summary_rows = [
        {
            "route": result["route"],
            "kind": result["kind"],
            "declared_input_mode": result["declared_input_mode"],
            "passed": result["passed"],
            "attempt_count": result["attempt_count"],
            "final_error": result["final_error"],
        }
        for result in results
    ]
    if args.base_summary:
        base = pipeline.read_json(args.base_summary.resolve())
        merged = {
            (str(row["route"]), str(row["kind"])): dict(row)
            for row in (base.get("results") or [])
        }
        for row in summary_rows:
            merged[(str(row["route"]), str(row["kind"]))] = row
        summary_rows = list(merged.values())
    summary_rows.sort(key=lambda value: (value["route"], value["kind"]))
    summary = {
        "schema_version": "science_arena_frozen_preflight_summary_v1",
        "created_at": pipeline.utc_now(),
        "models_path": str(models_path),
        "models_sha256": pipeline.sha256_file(models_path),
        "image": pipeline.image_descriptor(image_path),
        "base_summary": str(args.base_summary.resolve()) if args.base_summary else None,
        "job_count": len(summary_rows),
        "passed_count": sum(bool(result["passed"]) for result in summary_rows),
        "all_passed": all(bool(result["passed"]) for result in summary_rows),
        "results": summary_rows,
    }
    pipeline.atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
