#!/usr/bin/env python3
"""Freeze Table 6 routes after the saved multi-image visibility audit.

No API calls are made here.  The script converts immutable P1 and multi-image
probe evidence into (a) per-route capability proofs consumable by pipeline.py
and (b) the exact formal model manifest.  A route is classified as vision only
when it reads the unseen synthetic nonce and at least two of three official
science images.  The no-image control is retained as a hallucination diagnostic
but is not a hard modality veto: guessing without an image does not negate
positive proof that the route saw an unseen nonce in an attached image.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pipeline


VISION_CANDIDATES = {
    "anthropic/claude-opus-4.7",
    "bytedance/doubao-seed-2-0-pro",
    "cloudsway/gemini-3-flash-preview",
    "google/gemini-3.5-flash",
    "google/google/gemini-3.1-pro-preview",
    "openai/gpt-5.5",
    "qiniu/openai/gpt-5.4",
    "qwen/qwen3.6-plus",
}


GENERATION_OPTIONS: dict[str, dict[str, Any]] = {
    "cloudsway/gemini-3-flash-preview": {"thinking_level": "high"},
    "bytedance/doubao-seed-2-0-pro": {
        "thinking": {"type": "enabled"},
        "reasoning_effort": "high",
    },
    "google/google/gemini-3.1-pro-preview": {"thinking_level": "high"},
    "qiniu/openai/gpt-5.4": {"reasoning_effort": "xhigh"},
    "z-ai/glm-5.1": {"thinking": {"type": "enabled"}},
    "qwen/qwen3.6-plus": {"enable_thinking": True, "thinking_budget": 16384},
    "minimax/MiniMax-M2.7": {},
    "anthropic/claude-opus-4.7": {
        "thinking": {"type": "adaptive"},
        "effort": "max",
    },
    "xiaomi/mimo-v2.5-pro": {},
    "openai/gpt-5.5": {"reasoning_effort": "xhigh"},
    "deepseek/deepseek-v4-pro": {"thinking": {"type": "enabled"}},
    "deepseek/deepseek-v4-flash": {"thinking": {"type": "enabled"}},
    "google/gemini-3.5-flash": {"thinking_level": "high"},
    "qwen/qwen3.7-max": {"enable_thinking": True, "thinking_budget": 16384},
}


def compact(value: str) -> str:
    return re.sub(r"[\s,;:_`$\\(){}\[\].]+", "", value.casefold())


def physics_passed(text: str) -> bool:
    value = text.casefold().replace("½", "1/2")
    value = re.sub(r"\\(?:t|d)?frac\s*\{\s*1\s*\}\s*\{\s*2\s*\}", "1/2", value)
    value = re.sub(r"[\s,;|:_`$\\(){}\[\].]+", "", value)
    return "mno" in value and ("1/2a" in value or "a/2" in value)


def synthetic_passed(text: str) -> bool:
    value = compact(text)
    arrow = any(
        token in value
        for token in (
            "arrow=northwest",
            "arrow=nw",
            "arrow=up-left",
            "arrow=upleft",
            "arrow=upper-left",
            "arrow=upperleft",
        )
    )
    return "token=q7mx-4829" in value and "triangles=4" in value and arrow


def chemistry_passed(text: str) -> bool:
    return "a4=si|b1=m" in compact(text)


def graph_passed(text: str) -> bool:
    return "62|29|11" in compact(text)


def gate_probe_by_id(gate: dict[str, Any], probe_id: str) -> dict[str, Any]:
    for row in gate.get("probes") or []:
        if row.get("probe_id") == probe_id:
            return row
    raise RuntimeError(f"{gate.get('route')}: missing gate probe {probe_id}")


def classify_route(
    route: str,
    p1: dict[str, Any],
    gate: dict[str, Any] | None,
) -> dict[str, Any]:
    p1_vision = p1.get("vision_probe") or {}
    p1_physics = bool(p1_vision.get("http_status") == 200 and physics_passed(str(p1_vision.get("visible_content") or "")))
    synthetic = chemistry = graph = False
    negative_control_passed: bool | None = None
    gate_path_present = gate is not None
    if route in VISION_CANDIDATES:
        if gate is None:
            raise RuntimeError(f"{route}: missing completed multi-image visibility gate")
        synthetic_row = gate_probe_by_id(gate, "synthetic_nonce_shapes_v1")
        chemistry_row = gate_probe_by_id(gate, "icho2026_chemistry_structure_v1")
        graph_row = gate_probe_by_id(gate, "icho2026_numeric_graph_v1")
        synthetic = synthetic_passed(str((synthetic_row.get("positive") or {}).get("visible_content") or ""))
        chemistry = chemistry_passed(str((chemistry_row.get("positive") or {}).get("visible_content") or ""))
        graph = graph_passed(str((graph_row.get("positive") or {}).get("visible_content") or ""))
        negative = synthetic_row.get("negative_control")
        negative_control_passed = bool(
            isinstance(negative, dict)
            and negative.get("http_ok")
            and str(negative.get("visible_content") or "").strip().upper() == "IMAGE_UNAVAILABLE"
        )
    official_pass_count = sum((p1_physics, chemistry, graph))
    strong_pixel_proof = bool(synthetic and official_pass_count >= 2)
    input_mode = "vision" if strong_pixel_proof else "text_staticized"
    original_text_probe = p1.get("text_probe") or {}
    text_ready = bool(original_text_probe.get("passed") or strong_pixel_proof)
    return {
        "route": route,
        "input_mode": input_mode,
        "text_ready": text_ready,
        "text_ready_basis": (
            "standalone_nonce_text_probe"
            if original_text_probe.get("passed")
            else "correct_unseen_image_nonce_response_also_proves_instruction_text_processing"
            if strong_pixel_proof
            else "failed"
        ),
        "p1_physics_passed": p1_physics,
        "synthetic_unseen_nonce_passed": synthetic,
        "official_chemistry_passed": chemistry,
        "official_numeric_graph_passed": graph,
        "official_science_pass_count": official_pass_count,
        "official_science_probe_count": 3,
        "strong_pixel_proof": strong_pixel_proof,
        "negative_control_passed": negative_control_passed,
        "negative_control_policy": "diagnostic_not_modality_veto_after_unseen_nonce_pixel_proof",
        "gate_path_present": gate_path_present,
    }


def manifest_model(
    audited: dict[str, Any],
    classification: dict[str, Any],
    capability_path: Path,
    workspace: Path,
) -> dict[str, Any]:
    route = str(audited["route"])
    model: dict[str, Any] = {
        "table6_order": int(audited["table6_order"]),
        "display_name": audited["display_name"],
        "route": route,
        "api_mode": audited["api_mode"],
        "token_parameter": audited["token_parameter"],
        "input_mode": classification["input_mode"],
        "input_mode_frozen": True,
        "input_mode_decision_status": (
            "frozen_fresh_unseen_nonce_plus_official_science_pixel_proof_20260828"
            if classification["input_mode"] == "vision"
            else "frozen_fresh_no_strong_visible_pixel_proof_20260828"
        ),
        "confirmed_text_only": classification["input_mode"] == "text_staticized",
        "max_attempts": pipeline.DEFAULT_MAX_ATTEMPTS,
        "timeout_seconds": pipeline.DEFAULT_TIMEOUT_SECONDS,
        "max_completion_tokens": 65536,
        "requested_max_output_tokens": 65536,
        "generation_options": GENERATION_OPTIONS[route],
        "catalog": audited.get("catalog"),
        "capability_proof_path": str(capability_path.relative_to(workspace)),
        "capability_summary": classification,
    }
    if route == "xiaomi/mimo-v2.5-pro":
        model["effective_max_output_tokens"] = 10000
    if route == "google/gemini-3.5-flash":
        model["judge_max_completion_tokens"] = 65536
        model["judge_options"] = {"thinking_level": "high"}
        model["calibrated_production_judge"] = True
    return model


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-manifest", type=Path, required=True)
    parser.add_argument("--p1-dir", type=Path, required=True)
    parser.add_argument("--gate-dir", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    parser.add_argument("--proof-dir", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    args = parser.parse_args()
    workspace = args.workspace.resolve()
    audited_models = pipeline.load_model_manifest(args.audit_manifest.resolve())
    if len(audited_models) != 14:
        raise RuntimeError(f"Expected 14 audited Table 6 routes, got {len(audited_models)}")
    proof_dir = args.proof_dir.resolve()
    proof_dir.mkdir(parents=True, exist_ok=True)

    models: list[dict[str, Any]] = []
    matrix: list[dict[str, Any]] = []
    for audited in sorted(audited_models, key=lambda value: int(value["table6_order"])):
        route = str(audited["route"])
        p1_path = args.p1_dir.resolve() / f"{pipeline.safe_name(route)}.json"
        if not p1_path.is_file():
            raise RuntimeError(f"Missing P1 proof: {p1_path}")
        p1 = pipeline.read_json(p1_path)
        gate_path = args.gate_dir.resolve() / f"{pipeline.safe_name(route)}.json"
        gate = pipeline.read_json(gate_path) if gate_path.is_file() else None
        classification = classify_route(route, p1, gate)
        if not classification["text_ready"]:
            raise RuntimeError(f"{route}: text/instruction transport is not ready")
        capability_path = proof_dir / f"{pipeline.safe_name(route)}.json"
        capability = {
            "schema_version": "science_arena_frozen_capability_proof_v1",
            "created_at": pipeline.utc_now(),
            "route": route,
            "text_probe": {
                "passed": True,
                "basis": classification["text_ready_basis"],
                "source": str(p1_path.relative_to(workspace)),
                "original_probe": p1.get("text_probe"),
            },
            "vision_probe": {
                "passed": classification["strong_pixel_proof"],
                "classification": classification,
                "p1_source": str(p1_path.relative_to(workspace)),
                "multi_image_gate_source": str(gate_path.relative_to(workspace)) if gate else None,
            },
            "frozen_input_mode": classification["input_mode"],
        }
        pipeline.atomic_write_json(capability_path, capability)
        models.append(manifest_model(audited, classification, capability_path, workspace))
        matrix.append(classification)

    document = {
        "schema_version": "science_arena_table6_models_frozen_20260828_v1",
        "created_at": pipeline.utc_now(),
        "protocol": "fresh_capability_aware_mixed_input_interleaved_question_only_v2",
        "image_policy": (
            "vision routes receive cumulative mechanically rendered official question PNGs as direct "
            "data:image blocks; generated descriptions are forbidden in their solver payloads"
        ),
        "text_policy": (
            "text_staticized routes receive one shared question-only, visually audited staticization and "
            "no image blocks; solution/rubric assets remain isolated until judging"
        ),
        "attempt_policy": "exactly one initial wire request plus at most six retries (seven total)",
        "timeout_seconds": pipeline.DEFAULT_TIMEOUT_SECONDS,
        "classification_policy": (
            "vision iff unseen synthetic nonce+shape probe passes and at least 2/3 direct official "
            "physics/chemistry/graph probes pass; no-image control is diagnostic"
        ),
        "vision_route_count": sum(row["input_mode"] == "vision" for row in matrix),
        "text_staticized_route_count": sum(row["input_mode"] == "text_staticized" for row in matrix),
        "models": models,
    }
    pipeline.atomic_write_json(args.output_manifest.resolve(), document)
    pipeline.atomic_write_json(proof_dir / "classification_matrix.json", {"rows": matrix})
    print(
        json.dumps(
            {
                "models": len(models),
                "vision": document["vision_route_count"],
                "text_staticized": document["text_staticized_route_count"],
                "manifest": str(args.output_manifest.resolve()),
                "proof_dir": str(proof_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
