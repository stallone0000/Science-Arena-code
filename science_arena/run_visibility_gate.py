#!/usr/bin/env python3
"""Run the frozen multi-image visibility gate for candidate vision routes.

The gate is separate from formal solving.  It sends image pixels directly as
``data:image`` blocks, saves sanitized requests plus raw responses, and checks
both a synthetic nonce image and full-page official science images.  A no-image
negative control is issued with the same synthetic prompt to catch guessing.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import copy
import json
import re
from pathlib import Path
from typing import Any

import pipeline


DEFAULT_CANDIDATE_ROUTES = (
    "anthropic/claude-opus-4.7",
    "bytedance/doubao-seed-2-0-pro",
    "cloudsway/gemini-3-flash-preview",
    "google/gemini-3.5-flash",
    "google/google/gemini-3.1-pro-preview",
    "openai/gpt-5.5",
    "qiniu/openai/gpt-5.4",
    "qwen/qwen3.6-plus",
)


PROBES: tuple[dict[str, Any], ...] = (
    {
        "probe_id": "synthetic_nonce_shapes_v1",
        "kind": "synthetic_runtime_visibility",
        "image": "experiments/2026_reval/assets/probes/synthetic_visibility_q7mx_4829.png",
        "prompt": (
            "Inspect only the attached image pixels. If no image is attached or visible, return exactly "
            "IMAGE_UNAVAILABLE. Otherwise return exactly one line in this format, using values read from "
            "the image: token=<token>; triangles=<number of blue triangles>; arrow=<direction the red arrow "
            "points>. Do not guess and do not add prose."
        ),
        "validator": "synthetic",
        "negative_control": True,
    },
    {
        "probe_id": "icho2026_chemistry_structure_v1",
        "kind": "direct_official_full_page_chemistry",
        "image": (
            "experiments/2026_reval/datasets_draft/icho2026/question_images/"
            "exam-theory-Q3-english_4/page-3.png"
        ),
        "prompt": (
            "Inspect only the attached official IChO page. If no image is attached or visible, return "
            "exactly IMAGE_UNAVAILABLE. Otherwise, in the row of structures, read the central atom label "
            "inside structure A4 and the central atom placeholder inside structure B1. Return exactly "
            "A4=<label>|B1=<label>. Do not solve the chemistry and do not infer missing pixels."
        ),
        "validator": "chemistry",
        "negative_control": False,
    },
    {
        "probe_id": "icho2026_numeric_graph_v1",
        "kind": "direct_official_full_page_numeric_graph",
        "image": (
            "experiments/2026_reval/datasets_draft/icho2026/question_images/"
            "exam-theory-Q8-english_9/page-3.png"
        ),
        "prompt": (
            "Inspect only the attached official IChO page. If no image is attached or visible, return "
            "exactly IMAGE_UNAVAILABLE. Otherwise read the numbers printed above the red graph points at "
            "x=0.1, x=1, and x=2.9, in that order. Return exactly three integers separated by vertical "
            "bars. Do not calculate or infer."
        ),
        "validator": "graph",
        "negative_control": False,
    },
)


def compact(value: str) -> str:
    return re.sub(r"[\s,;:_`$\\(){}\[\].]+", "", value.casefold())


def validate_visible(kind: str, content: str) -> bool:
    value = compact(content)
    if kind == "synthetic":
        return (
            "token=q7mx-4829" in value
            and "triangles=4" in value
            and ("arrow=northwest" in value or "arrow=nw" in value)
        )
    if kind == "chemistry":
        return "a4=si|b1=m" in value
    if kind == "graph":
        return "62|29|11" in value
    raise ValueError(f"Unknown validator: {kind}")


def cap_probe_output(payload: dict[str, Any], maximum: int = 4096) -> None:
    for key in ("max_output_tokens", "max_completion_tokens", "max_tokens"):
        if key in payload:
            payload[key] = min(int(payload[key]), maximum)
            return


def call_once(
    *,
    model: dict[str, Any],
    prompt: str,
    image_path: Path | None,
    api_key: str,
    timeout_cap_seconds: int = 900,
) -> dict[str, Any]:
    message = (
        pipeline.make_image_message(prompt, [image_path])
        if image_path is not None
        else {"role": "user", "content": prompt}
    )
    payload = pipeline.base_payload(model, [message], judge=False)
    cap_probe_output(payload)
    http = pipeline.post_chat(
        payload,
        api_key,
        min(int(model.get("timeout_seconds") or 600), timeout_cap_seconds),
    )
    content, reasoning, finish = pipeline.response_text(http.response or {})
    return {
        "http_ok": http.ok,
        "http_status": http.status,
        "latency_seconds": http.latency_seconds,
        "visible_content": content,
        "reasoning_content": reasoning,
        "finish_reason": finish,
        "error": http.error,
        "response_body_on_error": http.body[:20000] if not http.ok else "",
        "request_sha256": pipeline.sha256_bytes(pipeline.canonical_json_bytes(payload)),
        "request": pipeline.sanitize_for_sidecar(payload),
        "raw_response": http.response,
    }


def run_route(
    *,
    model: dict[str, Any],
    workspace: Path,
    output_dir: Path,
    p1_dir: Path,
    api_key: str,
) -> dict[str, Any]:
    route = str(model["route"])
    p1_path = p1_dir / f"{pipeline.safe_name(route)}.json"
    if not p1_path.is_file():
        raise RuntimeError(f"Missing P1 proof for {route}: {p1_path}")
    p1 = pipeline.read_json(p1_path)
    p1_visible = str((p1.get("vision_probe") or {}).get("visible_content") or "")
    p1_passed = pipeline_compact_ipho_pass(p1_visible) and (
        (p1.get("vision_probe") or {}).get("http_status") == 200
    )

    probe_results: list[dict[str, Any]] = []
    for spec in PROBES:
        image_path = (workspace / str(spec["image"])).resolve()
        if not image_path.is_file():
            raise RuntimeError(f"Missing probe image: {image_path}")
        positive = call_once(
            model=model,
            prompt=str(spec["prompt"]),
            image_path=image_path,
            api_key=api_key,
        )
        positive["content_passed"] = bool(
            positive["http_ok"]
            and validate_visible(str(spec["validator"]), str(positive["visible_content"]))
        )
        negative = None
        if spec.get("negative_control"):
            negative = call_once(
                model=model,
                prompt=str(spec["prompt"]),
                image_path=None,
                api_key=api_key,
                timeout_cap_seconds=300,
            )
            negative["control_passed"] = bool(
                negative["http_ok"]
                and str(negative["visible_content"]).strip().upper() == "IMAGE_UNAVAILABLE"
            )
        probe_results.append(
            {
                "probe_id": spec["probe_id"],
                "kind": spec["kind"],
                "image": pipeline.image_descriptor(image_path),
                "positive": positive,
                "negative_control": negative,
            }
        )

    positives_passed = p1_passed and all(row["positive"]["content_passed"] for row in probe_results)
    controls_passed = all(
        row["negative_control"] is None or row["negative_control"]["control_passed"]
        for row in probe_results
    )
    result = {
        "schema_version": "science_arena_visibility_gate_v1",
        "created_at": pipeline.utc_now(),
        "route": route,
        "p1_official_physics": {
            "source": str(p1_path),
            "visible_content": p1_visible,
            "content_passed": p1_passed,
            "image": p1.get("probe_image"),
        },
        "probes": probe_results,
        "positive_probe_count": 1 + len(probe_results),
        "positive_pass_count": int(p1_passed)
        + sum(bool(row["positive"]["content_passed"]) for row in probe_results),
        "negative_controls_passed": controls_passed,
        "visibility_gate_passed": positives_passed and controls_passed,
        "frozen_input_mode": "vision" if positives_passed and controls_passed else "text_staticized",
    }
    pipeline.atomic_write_json(output_dir / f"{pipeline.safe_name(route)}.json", result)
    return result


def pipeline_compact_ipho_pass(text: str) -> bool:
    value = text.casefold().replace("½", "1/2")
    value = re.sub(r"\\(?:t|d)?frac\s*\{\s*1\s*\}\s*\{\s*2\s*\}", "1/2", value)
    value = re.sub(r"[\s,;|:_`$\\(){}\[\].]+", "", value)
    return "mno" in value and ("1/2a" in value or "a/2" in value)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--p1-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--routes", default=",".join(DEFAULT_CANDIDATE_ROUTES))
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    wanted = {value.strip() for value in args.routes.split(",") if value.strip()}
    all_models = pipeline.load_model_manifest(args.models.resolve())
    models = [copy.deepcopy(model) for model in all_models if str(model.get("route")) in wanted]
    missing = wanted - {str(model.get("route")) for model in models}
    if missing:
        raise RuntimeError(f"Routes absent from manifest: {sorted(missing)}")

    api_key = pipeline.load_api_key()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers, len(models))) as executor:
        futures = [
            executor.submit(
                run_route,
                model=model,
                workspace=workspace,
                output_dir=output_dir,
                p1_dir=args.p1_dir.resolve(),
                api_key=api_key,
            )
            for model in models
        ]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    summary = {
        "schema_version": "science_arena_visibility_gate_summary_v1",
        "created_at": pipeline.utc_now(),
        "model_count": len(results),
        "gate_pass_count": sum(bool(row["visibility_gate_passed"]) for row in results),
        "results": sorted(
            [
                {
                    "route": row["route"],
                    "positive_pass_count": row["positive_pass_count"],
                    "positive_probe_count": row["positive_probe_count"],
                    "negative_controls_passed": row["negative_controls_passed"],
                    "visibility_gate_passed": row["visibility_gate_passed"],
                    "frozen_input_mode": row["frozen_input_mode"],
                }
                for row in results
            ],
            key=lambda value: value["route"],
        ),
    }
    pipeline.atomic_write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
