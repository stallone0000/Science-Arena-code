#!/usr/bin/env python3
"""Aggregate run-to-run variability for the three fresh IPhO/IChO repeats."""

from __future__ import annotations

import argparse
import csv
import base64
import hashlib
import importlib.util
import json
import math
import mimetypes
import re
import stat
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


WORKSPACE = Path(__file__).resolve().parents[2]
REPLICATES = ("repA", "repB", "repC")
CONTESTS = {
    "ipho2026": {
        "benchmark": "ipho2026",
        "items": WORKSPACE / "experiments/2026_reval/staticizations/qonly_gemini31_reviewed_frozen_final_v2_20260829/datasets/ipho2026/items.jsonl",
        "raw_max": 30.0,
    },
    "icho2026": {
        "benchmark": "icho2026_theory",
        "items": WORKSPACE / "experiments/2026_reval/staticizations/qonly_gemini31_reviewed_frozen_final_v2_20260829/datasets/icho2026_theory/items.jsonl",
        "raw_max": 60.0,
    },
}
MODELS = WORKSPACE / "experiments/2026_reval/manifests/table6_models_frozen_20260828.json"
PIPELINE_SNAPSHOT = WORKSPACE / "experiments/2026_reval/snapshots/pipeline_table6_ipho_icho_formal_6a6289be6d2fa036.py"
PIPELINE_SNAPSHOT_SHA256 = "6a6289be6d2fa036de0958a079303ce5096e3be3e02dc311275b11ba6a55f89b"
EXTERNAL_BASELINE = WORKSPACE / "experiments/2026_reval/analysis/table6_four_competition_20260829/table6_scores.json"
OUTPUT = WORKSPACE / "experiments/2026_reval/analysis/ipho_icho_three_fresh_repeats_20260830"
JUDGE_ROUTE = "google/gemini-3.5-flash"
QUOTE_RECOVERY_ROOT = WORKSPACE / "experiments/2026_reval/runs/ipho2026_repeat_repC_gemini35_judge_20260830_recovery_quote_replay_v1"
QUOTE_RECOVERY_ALGORITHM = "unique_longest_exact_subquote_edge_trim_v1"
QUOTE_RECOVERY_CASES = {
    ("z-ai/glm-5.1", "ipho2026_t1_b2"),
    ("google/gemini-3.5-flash", "ipho2026_t3_b1"),
}
UNSUPPORTED_QUOTE = re.compile(
    r"^criterion (\d+) awards credit without a supported exact candidate quote$"
)
HEX64 = re.compile(r"^[0-9a-f]{64}$")
NO_ANSWER_LABEL = "不会做这道题"


def run_root(contest: str, replicate: str) -> Path:
    return WORKSPACE / f"experiments/2026_reval/runs/{contest}_repeat_{replicate}_gemini35_judge_20260830"


def effective_aggregate_path(contest: str, replicate: str) -> Path:
    if contest == "ipho2026" and replicate == "repC":
        return QUOTE_RECOVERY_ROOT / "aggregate.json"
    return run_root(contest, replicate) / "aggregate.json"


def ensure_fileprovider_materialized(path: Path) -> None:
    """Materialize one exact FileProvider file before strict reads, if needed."""
    info = path.stat()
    offline_flag = getattr(stat, "UF_OFFLINE", 0)
    needs_download = bool(info.st_flags & offline_flag) or (info.st_size > 0 and info.st_blocks == 0)
    if not needs_download:
        return
    completed = subprocess.run(
        ["brctl", "download", str(path.resolve())],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"exact-file FileProvider download failed for {path}: "
            f"{(completed.stderr or completed.stdout).strip()}"
        )
    deadline = time.monotonic() + 180.0
    while time.monotonic() < deadline:
        info = path.stat()
        is_offline = bool(info.st_flags & offline_flag)
        has_blocks = info.st_size == 0 or info.st_blocks > 0
        if not is_offline and has_blocks:
            with path.open("rb") as handle:
                first = handle.read(1)
                if info.st_size == 0 or first:
                    return
        time.sleep(0.25)
    raise RuntimeError(f"FileProvider file did not materialize within 180s: {path}")


def strict_read_bytes(path: Path) -> bytes:
    ensure_fileprovider_materialized(path)
    expected_size = path.stat().st_size
    payload = path.read_bytes()
    if len(payload) != expected_size:
        raise RuntimeError(f"short read after FileProvider materialization: {path}")
    return payload


def load_json(path: Path) -> Any:
    return json.loads(strict_read_bytes(path).decode("utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in strict_read_bytes(path).decode("utf-8").splitlines()
        if line.strip()
    ]


def safe_route(route: str) -> str:
    return route.replace("/", "__")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit and summarize three fresh IPhO/IChO repeat runs."
    )
    parser.add_argument(
        "--model-route",
        help=(
            "Restrict the formal stability analysis to one exact frozen model route. "
            "Unselected route files may remain in the run roots but are excluded from "
            "terminal/attempt closure checks."
        ),
    )
    return parser.parse_args()


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    ensure_fileprovider_materialized(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_frozen_pipeline() -> Any:
    actual = sha256_file(PIPELINE_SNAPSHOT)
    if actual != PIPELINE_SNAPSHOT_SHA256:
        raise RuntimeError(
            f"frozen pipeline snapshot hash mismatch: {actual} != {PIPELINE_SNAPSHOT_SHA256}"
        )
    spec = importlib.util.spec_from_file_location("science_arena_frozen_repeat_pipeline", PIPELINE_SNAPSHOT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load frozen pipeline snapshot")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def resolve_item_paths(values: list[str]) -> list[Path]:
    result = []
    for value in values:
        path = Path(value)
        result.append((path if path.is_absolute() else WORKSPACE / path).resolve())
    return result


def ordered_unique(paths: list[Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        if path not in seen:
            result.append(path)
            seen.add(path)
    return result


def frozen_solver_prompt(item: dict[str, Any], input_mode: str) -> str:
    question = item["question_text_native"] if input_mode == "vision" else item["question_text_staticized"]
    score_line = (
        f"Maximum raw score: {item['max_score']}\n"
        if item.get("solver_prompt_disclose_max_score", True) is not False
        else ""
    )
    return (
        f"Competition: {item['benchmark_id']}\n"
        f"Problem sequence: {item['sequence_id']}\n"
        f"Current item: {item['item_id']}\n"
        f"{score_line}\n"
        f"{question}\n\n"
        "Give a concrete answer for the current item. End with a line beginning `FINAL ANSWER:`."
    )


def stats(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "sample_sd": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
        "range": max(values) - min(values),
    }


def average_ranks(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    result = [0.0] * len(values)
    position = 0
    while position < len(order):
        end = position + 1
        while end < len(order) and values[order[end]] == values[order[position]]:
            end += 1
        rank = (position + 1 + end) / 2.0
        for index in order[position:end]:
            result[index] = rank
        position = end
    return result


def pearson(left: list[float], right: list[float]) -> float:
    mean_left = statistics.fmean(left)
    mean_right = statistics.fmean(right)
    numerator = sum((a - mean_left) * (b - mean_right) for a, b in zip(left, right))
    denominator = math.sqrt(
        sum((a - mean_left) ** 2 for a in left) * sum((b - mean_right) ** 2 for b in right)
    )
    return numerator / denominator if denominator else float("nan")


def spearman(left: list[float], right: list[float]) -> float:
    return pearson(average_ranks(left), average_ranks(right))


def terminal_path(root: Path, phase: str, route: str, item: dict[str, Any]) -> Path:
    return (
        root
        / phase
        / safe_route(route)
        / safe_route(str(item["benchmark_id"]))
        / safe_route(str(item["sequence_id"]))
        / f"{safe_route(str(item['item_id']))}.json"
    )


def recovery_exact_subquote(original: str, candidate: str) -> str | None:
    options: list[tuple[int, int, int, str]] = []
    for start in range(len(original)):
        for end in range(start + 1, len(original) + 1):
            value = original[start:end].strip()
            if len(value) >= 16 and value in candidate and candidate.count(value) == 1:
                options.append((len(value), -start, end, value))
    return max(options)[-1] if options else None


def semantic_without_evidence(parsed: dict[str, Any]) -> dict[str, Any]:
    value = json.loads(json.dumps(parsed, ensure_ascii=False))
    for criterion in value.get("criteria", []):
        if isinstance(criterion, dict):
            criterion.pop("candidate_evidence_quote", None)
    return value


def response_with_parsed_content(
    frozen_pipeline: Any,
    response: dict[str, Any],
    parsed: dict[str, Any],
) -> dict[str, Any]:
    value = json.loads(json.dumps(response, ensure_ascii=False))
    replacement = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    replaced = False
    for choice in value.get("choices") or []:
        message = choice.get("message") if isinstance(choice, dict) else None
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            message["content"] = replacement
            replaced = True
            break
        if isinstance(choice, dict) and isinstance(choice.get("text"), str):
            choice["text"] = replacement
            replaced = True
            break
    if not replaced or not frozen_pipeline.response_text(value)[0]:
        raise RuntimeError("quote recovery response content is not replayable")
    return value


def audit_quote_recovery_overlay(
    *,
    source_root: Path,
    source_judge_path: Path,
    source_judge: dict[str, Any],
    solver_path: Path,
    solver: dict[str, Any],
    item: dict[str, Any],
    route: str,
    frozen_pipeline: Any,
) -> tuple[dict[str, Any], int]:
    key = (route, str(item["item_id"]))
    if key not in QUOTE_RECOVERY_CASES:
        raise RuntimeError(f"{route}::{item['item_id']}: unregistered recovery overlay")
    if (
        source_judge.get("terminal") is not False
        or source_judge.get("status") != "judge_failed_validation_after_retries"
        or source_judge.get("attempt_count") != 7
        or source_judge.get("judge_route") != JUDGE_ROUTE
    ):
        raise RuntimeError(f"{route}::{item['item_id']}: recovery source is not the pinned exhausted failure")
    overlay_path = terminal_path(QUOTE_RECOVERY_ROOT, "judge", route, item)
    if not overlay_path.is_file() or overlay_path.is_symlink() or overlay_path.stat().st_nlink != 1:
        raise RuntimeError(f"{route}::{item['item_id']}: recovery overlay missing/linked")
    overlay = load_json(overlay_path)
    candidate = str(solver["candidate_answer"])
    source_aggregate_path = source_root / "aggregate.json"
    source_attempt_paths = [
        source_root / str(relative) for relative in (source_judge.get("attempt_files") or [])
    ]
    if (
        overlay.get("schema_version") != "science_arena_judge_recovery_overlay_v1"
        or overlay.get("terminal") is not True
        or overlay.get("algorithm") != QUOTE_RECOVERY_ALGORITHM
        or overlay.get("minimum_quote_chars") != 16
        or overlay.get("api_calls_made") != 0
        or overlay.get("snapshot_sha256") != PIPELINE_SNAPSHOT_SHA256
        or overlay.get("source_incomplete_aggregate_sha256") != sha256_file(source_aggregate_path)
        or overlay.get("source_solver_terminal_sha256") != sha256_file(solver_path)
        or overlay.get("source_judge_terminal_sha256") != sha256_file(source_judge_path)
        or overlay.get("candidate_answer_sha256") != sha_text(candidate)
        or overlay.get("judge_input_fingerprint") != source_judge.get("judge_input_fingerprint")
        or overlay.get("all_seven_saved_responses_strict_valid_after_quote_only_replay") is not True
        or overlay.get("selected_source_attempt") != 1
    ):
        raise RuntimeError(f"{route}::{item['item_id']}: recovery provenance mismatch")
    replay_rows = overlay.get("saved_attempt_replays") or []
    if len(source_attempt_paths) != 7 or len(replay_rows) != 7:
        raise RuntimeError(f"{route}::{item['item_id']}: recovery attempt ledger is not exactly seven")
    expected_criteria = frozen_pipeline.normalized_rubric_criteria(
        frozen_pipeline.grading_rubric_for(item)
    )
    validated_outputs: list[dict[str, Any]] = []
    for index, (attempt_path, replay_row) in enumerate(zip(source_attempt_paths, replay_rows), 1):
        if not attempt_path.is_file() or attempt_path.is_symlink() or attempt_path.stat().st_nlink != 1:
            raise RuntimeError(f"{route}::{item['item_id']}: recovery source attempt missing/linked")
        attempt = load_json(attempt_path)
        response = attempt.get("raw_response")
        if (
            attempt.get("attempt") != index
            or attempt.get("http_status") != 200
            or attempt.get("ok") is not True
            or attempt.get("validation_ok") is not False
            or attempt.get("state") != "completed"
            or not isinstance(response, dict)
            or replay_row.get("attempt") != index
            or replay_row.get("source_attempt_sha256") != sha256_file(attempt_path)
            or replay_row.get("source_request_sha256") != attempt.get("request_sha256")
            or replay_row.get("source_raw_response_sha256") != canonical_json_sha256(response)
            or replay_row.get("source_validation_reason") != attempt.get("validation_reason")
            or replay_row.get("strict_valid") is not True
        ):
            raise RuntimeError(f"{route}::{item['item_id']}: recovery attempt {index} provenance mismatch")
        original_valid, original_reason, _ = frozen_pipeline.validate_judge_response(
            response,
            candidate,
            float(item["max_score"]),
            expected_criteria=expected_criteria,
        )
        if original_valid or original_reason != attempt.get("validation_reason"):
            raise RuntimeError(f"{route}::{item['item_id']}: source validator failure does not replay")
        content, _, _ = frozen_pipeline.response_text(response)
        original = frozen_pipeline.extract_json_object(content)
        modified = json.loads(json.dumps(original, ensure_ascii=False))
        changes = replay_row.get("quote_changes") or []
        if len(changes) != replay_row.get("quote_change_count") or not changes:
            raise RuntimeError(f"{route}::{item['item_id']}: empty/inconsistent recovery edits")
        for change in changes:
            current_response = response_with_parsed_content(frozen_pipeline, response, modified)
            valid, reason, _ = frozen_pipeline.validate_judge_response(
                current_response,
                candidate,
                float(item["max_score"]),
                expected_criteria=expected_criteria,
            )
            match = UNSUPPORTED_QUOTE.fullmatch(reason)
            index_from_validator = int(match.group(1)) if match is not None else -1
            criterion_index = int(change.get("criterion_index", -2))
            criterion = modified["criteria"][criterion_index]
            old_quote = criterion["candidate_evidence_quote"]
            new_quote = recovery_exact_subquote(old_quote, candidate)
            if (
                valid
                or criterion_index != index_from_validator
                or change.get("criterion_id") != criterion.get("criterion_id")
                or change.get("old_quote") != old_quote
                or change.get("new_quote") != new_quote
                or change.get("old_quote_sha256") != sha_text(old_quote)
                or change.get("new_quote_sha256") != sha_text(str(new_quote))
                or change.get("new_quote_candidate_occurrences") != 1
            ):
                raise RuntimeError(f"{route}::{item['item_id']}: recovery quote edit is not deterministic")
            criterion["candidate_evidence_quote"] = new_quote
        if semantic_without_evidence(modified) != semantic_without_evidence(original):
            raise RuntimeError(f"{route}::{item['item_id']}: recovery changed judge semantics")
        final_response = response_with_parsed_content(frozen_pipeline, response, modified)
        valid, reason, validated = frozen_pipeline.validate_judge_response(
            final_response,
            candidate,
            float(item["max_score"]),
            expected_criteria=expected_criteria,
        )
        if (
            not valid
            or reason
            or validated is None
            or float(validated["score"]) != float(replay_row["score"])
            or canonical_json_sha256(validated) != replay_row.get("validated_parsed_sha256")
        ):
            raise RuntimeError(f"{route}::{item['item_id']}: recovered response is not strict-valid")
        validated_outputs.append(validated)
    if len({float(value["score"]) for value in validated_outputs}) != 1:
        raise RuntimeError(f"{route}::{item['item_id']}: seven replayed scores disagree")
    effective = overlay.get("effective_judge_terminal")
    if (
        not isinstance(effective, dict)
        or effective.get("terminal") is not True
        or effective.get("status") != "judged_deterministic_evidence_quote_replay_v1"
        or effective.get("recovery_algorithm") != QUOTE_RECOVERY_ALGORITHM
        or effective.get("recovery_source_attempt") != 1
        or effective.get("parsed") != validated_outputs[0]
        or float(effective.get("score")) != float(validated_outputs[0]["score"])
        or canonical_json_sha256(effective.get("raw_response"))
        != canonical_json_sha256(load_json(source_attempt_paths[0])["raw_response"])
    ):
        raise RuntimeError(f"{route}::{item['item_id']}: effective recovery terminal mismatch")
    return effective, len(source_attempt_paths)


def compare_image_descriptors(
    actual: Any,
    expected: list[Path],
    *,
    image_cache: dict[Path, dict[str, Any]],
    label: str,
) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise RuntimeError(f"{label}: image count mismatch {len(actual) if isinstance(actual, list) else 'non-list'} != {len(expected)}")
    for index, (descriptor, path) in enumerate(zip(actual, expected), start=1):
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"{label}: image {index} descriptor is not an object")
        proof = image_cache[path]
        actual_path = Path(str(descriptor.get("path") or "")).resolve()
        if actual_path != path:
            raise RuntimeError(f"{label}: image {index} path mismatch")
        if (
            descriptor.get("sha256") != proof["sha256"]
            or descriptor.get("bytes") != proof["bytes"]
            or descriptor.get("mime_type") != proof["mime_type"]
        ):
            raise RuntimeError(f"{label}: image {index} byte/hash mismatch")


def compare_request_image_proofs(
    actual: Any,
    expected: list[Path],
    *,
    image_cache: dict[Path, dict[str, Any]],
    label: str,
) -> None:
    if not isinstance(actual, list) or len(actual) != len(expected):
        raise RuntimeError(f"{label}: request image count mismatch")
    for index, (descriptor, path) in enumerate(zip(actual, expected), start=1):
        if not isinstance(descriptor, dict):
            raise RuntimeError(f"{label}: request image {index} is not an object")
        proof = image_cache[path]
        if descriptor.get("order") != index:
            raise RuntimeError(f"{label}: request image order mismatch")
        if Path(str(descriptor.get("path") or "")).resolve() != path:
            raise RuntimeError(f"{label}: request image path mismatch")
        for key in ("sha256", "bytes", "mime_type", "data_url_sha256", "data_url_utf8_bytes"):
            if descriptor.get(key) != proof[key]:
                raise RuntimeError(f"{label}: request image {index} {key} mismatch")


def image_transport_node_count(value: Any) -> int:
    """Count explicit chat/responses image blocks in the persisted sanitized payload."""
    if isinstance(value, list):
        return sum(image_transport_node_count(item) for item in value)
    if not isinstance(value, dict):
        return 0
    own = int(value.get("type") in {"image_url", "input_image"})
    return own + sum(image_transport_node_count(item) for item in value.values())


def verify_attempts(
    root: Path,
    record: dict[str, Any],
    expected_images: list[Path],
    *,
    frozen_pipeline: Any,
    image_cache: dict[Path, dict[str, Any]],
    label: str,
) -> tuple[int, int, int]:
    count = int(record.get("attempt_count") or 0)
    recovered_inflight_terminal_count = 0
    abandoned_inflight_count = 0
    paths = record.get("attempt_files") or []
    if count != len(paths) or count < 1 or count > 7:
        raise RuntimeError(f"{label}: invalid attempt closure count={count}, files={len(paths)}")
    resolved_paths = [root / str(relative) for relative in paths]
    if len({path.parent for path in resolved_paths}) != 1:
        raise RuntimeError(f"{label}: attempt sidecars do not share one item ledger directory")
    if sorted(resolved_paths[0].parent.glob("attempt-*.json")) != resolved_paths:
        raise RuntimeError(f"{label}: listed attempt ledger does not exactly close its directory")
    for index, relative in enumerate(paths, start=1):
        path = root / str(relative)
        if not path.is_file() or path.is_symlink() or path.stat().st_nlink != 1:
            raise RuntimeError(f"{label}: missing/linked attempt sidecar {path}")
        attempt = load_json(path)
        if attempt.get("attempt") != index:
            raise RuntimeError(f"{label}: attempt {index} is not completed in order")
        state = attempt.get("state")
        if state == "inflight":
            empty_inflight = (
                attempt.get("raw_response") is None
                and attempt.get("http_status") is None
                and attempt.get("ok") is None
                and attempt.get("validation_ok") is None
            )
            terminal_response = record.get("raw_response")
            terminal_storage = record.get("raw_response_storage")
            visible, _, finish = frozen_pipeline.response_text(
                terminal_response if isinstance(terminal_response, dict) else {}
            )
            recoverable_solver_terminal = (
                index == count
                and record.get("schema_version") == "science_arena_solver_result_v1"
                and record.get("terminal") is True
                and record.get("status") == "answered"
                and isinstance(terminal_response, dict)
                and isinstance(terminal_storage, dict)
                and terminal_storage.get("request_sha256") == attempt.get("request_sha256")
                and empty_inflight
                and visible == record.get("visible_content") == record.get("candidate_answer")
                and finish == record.get("finish_reason")
            )
            abandoned_before_later_attempt = index < count and empty_inflight
            if recoverable_solver_terminal:
                recovered_inflight_terminal_count += 1
            elif abandoned_before_later_attempt:
                abandoned_inflight_count += 1
            else:
                raise RuntimeError(f"{label}: attempt {index} is not completed in order")
        elif state != "completed":
            raise RuntimeError(f"{label}: attempt {index} is not completed in order")
        if state == "completed" and attempt.get("ok") is True:
            if attempt.get("http_status") != 200:
                raise RuntimeError(f"{label}: successful attempt {index} lacks HTTP 200")
        elif state == "completed" and not str(attempt.get("error") or attempt.get("response_body_on_error") or ""):
            raise RuntimeError(f"{label}: failed attempt {index} lacks an auditable error")
        if not HEX64.fullmatch(str(attempt.get("request_sha256") or "")):
            raise RuntimeError(f"{label}: attempt {index} lacks a request hash")
        if not isinstance(attempt.get("request"), dict):
            raise RuntimeError(f"{label}: attempt {index} lacks the sanitized raw request")
        if image_transport_node_count(attempt["request"]) != len(expected_images):
            raise RuntimeError(f"{label}: attempt {index} image transport block count mismatch")
        if attempt.get("ok") is True and not isinstance(attempt.get("raw_response"), dict):
            raise RuntimeError(f"{label}: successful attempt {index} lacks the preserved raw response")
        compare_request_image_proofs(
            attempt.get("request_images"),
            expected_images,
            image_cache=image_cache,
            label=f"{label}/attempt-{index:02d}",
        )
    return count, recovered_inflight_terminal_count, abandoned_inflight_count


def validate_staticization_item(item: dict[str, Any], items_path: Path) -> None:
    freeze = item.get("question_only_staticization_freeze")
    if not isinstance(freeze, dict) or freeze.get("schema_version") != "science_arena_2026_question_staticization_freeze_v1":
        raise RuntimeError(f"{item['item_id']}: missing question-only freeze")
    for field, value in (
        ("question_text_native_sha256", item["question_text_native"]),
        ("question_text_staticized_sha256", item["question_text_staticized"]),
        ("judge_question_text_staticized_sha256", item["judge_question_text_staticized"]),
    ):
        if freeze.get(field) != sha_text(str(value)):
            raise RuntimeError(f"{item['item_id']}: {field} mismatch")
    provenance = (items_path.parent / str(freeze.get("provenance_file") or "")).resolve()
    if not provenance.is_file():
        raise RuntimeError(f"{item['item_id']}: frozen staticization provenance is missing")
    staticization = item.get("staticization") or {}
    if staticization.get("solution_leakage_check") != "passed" or staticization.get("human_visual_audited") is not True:
        raise RuntimeError(f"{item['item_id']}: staticization did not pass human/leakage review")


def independently_recompute_score(
    contest: str,
    contest_items: list[dict[str, Any]],
    item_scores: dict[str, float],
) -> tuple[float, float, dict[str, Any]]:
    if contest == "ipho2026":
        return sum(item_scores[item["item_id"]] for item in contest_items), 30.0, {}
    by_sequence: dict[str, list[dict[str, Any]]] = {}
    for item in contest_items:
        by_sequence.setdefault(str(item["sequence_id"]), []).append(item)
    total = 0.0
    breakdown: dict[str, Any] = {}
    for sequence_id, sequence_items in sorted(by_sequence.items()):
        raw_max_values = {float(item["problem_raw_max"]) for item in sequence_items}
        weight_values = {float(item["problem_weight"]) for item in sequence_items}
        report_values = {float(item["report_scale"]) for item in sequence_items}
        if len(raw_max_values) != 1 or len(weight_values) != 1 or report_values != {60.0}:
            raise RuntimeError(f"IChO weighting metadata inconsistent for {sequence_id}")
        raw_max = next(iter(raw_max_values))
        weight = next(iter(weight_values))
        if abs(sum(float(item["max_score"]) for item in sequence_items) - raw_max) > 1e-9:
            raise RuntimeError(f"IChO item maxima do not close for {sequence_id}")
        raw_score = sum(item_scores[item["item_id"]] for item in sequence_items)
        weighted = raw_score / raw_max * weight
        total += weighted
        breakdown[sequence_id] = {
            "raw_score": raw_score,
            "raw_max": raw_max,
            "official_problem_weight": weight,
            "weighted_score": weighted,
        }
    if abs(sum(row["official_problem_weight"] for row in breakdown.values()) - 60.0) > 1e-9:
        raise RuntimeError("IChO official problem weights do not sum to 60")
    return total, 60.0, breakdown


def audit_all_runs(
    models: list[dict[str, Any]],
    items: dict[str, list[dict[str, Any]]],
    aggregates: dict[tuple[str, str], dict[str, Any]],
    frozen_pipeline: Any,
) -> tuple[dict[str, Any], dict[tuple[str, str, str, str], dict[str, Any]]]:
    image_cache: dict[Path, dict[str, Any]] = {}
    expected_solver_images: dict[tuple[str, str], list[Path]] = {}
    current_question_images: dict[tuple[str, str], list[Path]] = {}
    current_solution_images: dict[tuple[str, str], list[Path]] = {}
    staticization_item_count = 0
    contest_asset_paths: dict[str, dict[str, set[Path]]] = {
        contest: {"question": set(), "solution": set(), "provenance": set()}
        for contest in items
    }
    for contest, contest_items in items.items():
        items_path = CONTESTS[contest]["items"]
        by_sequence: dict[str, list[dict[str, Any]]] = {}
        for item in contest_items:
            validate_staticization_item(item, items_path)
            staticization_item_count += 1
            provenance = (
                items_path.parent
                / str(item["question_only_staticization_freeze"]["provenance_file"])
            ).resolve()
            contest_asset_paths[contest]["provenance"].add(provenance)
            by_sequence.setdefault(str(item["sequence_id"]), []).append(item)
            question = resolve_item_paths(list(item.get("question_image_paths") or []))
            solution = resolve_item_paths(list(item.get("official_solution_image_paths") or []))
            current_question_images[(contest, item["item_id"])] = question
            current_solution_images[(contest, item["item_id"])] = solution
            contest_asset_paths[contest]["question"].update(question)
            contest_asset_paths[contest]["solution"].update(solution)
            assets = {
                Path(str(asset["path"])).as_posix(): asset
                for asset in (item.get("question_image_assets") or [])
            }
            expected_asset_paths = {
                path.relative_to(WORKSPACE).as_posix() for path in ordered_unique(question)
            }
            if set(assets) != expected_asset_paths:
                raise RuntimeError(f"{item['item_id']}: frozen question-image asset manifest is not closed")
            for path in ordered_unique(question + solution):
                if path not in image_cache:
                    data = strict_read_bytes(path)
                    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
                    data_url = f"data:{mime};base64,".encode("ascii") + base64.b64encode(data)
                    image_cache[path] = {
                        "sha256": hashlib.sha256(data).hexdigest(),
                        "bytes": len(data),
                        "mime_type": mime,
                        "data_url_sha256": hashlib.sha256(data_url).hexdigest(),
                        "data_url_utf8_bytes": len(data_url),
                        "png_magic_verified": data.startswith(b"\x89PNG\r\n\x1a\n"),
                    }
                if path.suffix.lower() != ".png" or image_cache[path]["png_magic_verified"] is not True:
                    raise RuntimeError(f"{item['item_id']}: referenced image is not an original PNG: {path}")
                workspace_relative = path.relative_to(WORKSPACE).as_posix()
                if workspace_relative in assets:
                    asset = assets[workspace_relative]
                    proof = image_cache[path]
                    if (
                        asset.get("sha256") != proof["sha256"]
                        or ("bytes" in asset and asset.get("bytes") != proof["bytes"])
                    ):
                        raise RuntimeError(f"{item['item_id']}: frozen question asset hash mismatch")
        for sequence_items in by_sequence.values():
            cumulative: list[Path] = []
            for item in sorted(sequence_items, key=lambda value: int(value["turn_index"])):
                cumulative = ordered_unique(cumulative + current_question_images[(contest, item["item_id"])])
                expected_solver_images[(contest, item["item_id"])] = list(cumulative)

    observations: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    counts = {
        "solver_terminals": 0,
        "judge_terminals": 0,
        "solver_api_attempt_records": 0,
        "solver_terminals_recovered_from_inflight_attempt_sidecar": 0,
        "abandoned_inflight_attempt_sidecars_counted_as_wire_ordinals": 0,
        "judge_api_attempt_records": 0,
        "vision_solver_terminals_direct_png_verified": 0,
        "text_solver_terminals_zero_image_verified": 0,
        "vision_judge_terminals_direct_question_and_solution_png_verified": 0,
        "deterministic_zero_judge_terminals": 0,
        "deterministic_quote_recovery_judge_terminals": 0,
        "deterministic_quote_recovery_saved_responses_replayed": 0,
        "staticization_items_hash_and_leakage_verified": staticization_item_count,
        "icho_aggregate_route_replicates_weighted_to_60_verified": 0,
        "ipho_aggregate_route_replicates_summed_to_30_verified": 0,
    }
    run_summaries: list[dict[str, Any]] = []
    allowed_solver_status = {"answered", "model_completed_no_answer", "model_no_answer_after_retries"}
    judge_model = next((model for model in models if model.get("route") == JUDGE_ROUTE), None)
    if not isinstance(judge_model, dict) or judge_model.get("input_mode") != "vision":
        raise RuntimeError("fixed judge route is absent or is not frozen as vision-capable")
    selected_route_dirs = {safe_route(str(model["route"])) for model in models}
    for replicate in REPLICATES:
        for contest, contest_items in items.items():
            root = run_root(contest, replicate)
            if any(path.is_symlink() for phase in ("solver", "judge") for path in (root / phase).rglob("*")):
                raise RuntimeError(f"{root}: symlink found in fresh terminal tree")
            expected_solver_paths: set[Path] = set()
            expected_judge_paths: set[Path] = set()
            per_route_scores: dict[str, dict[str, float]] = {}
            for model in models:
                route = str(model["route"])
                item_scores: dict[str, float] = {}
                for item in contest_items:
                    identity = f"{contest}/{replicate}/{route}/{item['item_id']}"
                    solver_path = terminal_path(root, "solver", route, item)
                    judge_path = terminal_path(root, "judge", route, item)
                    expected_solver_paths.add(solver_path)
                    expected_judge_paths.add(judge_path)
                    if not solver_path.is_file() or solver_path.is_symlink() or solver_path.stat().st_nlink != 1:
                        raise RuntimeError(f"{identity}: solver terminal missing/linked")
                    solver = load_json(solver_path)
                    if solver.get("schema_version") != "science_arena_solver_result_v1" or solver.get("terminal") is not True:
                        raise RuntimeError(f"{identity}: invalid solver terminal schema/state")
                    if (
                        solver.get("route") != route
                        or solver.get("input_mode") != model["input_mode"]
                        or solver.get("item_id") != item["item_id"]
                        or solver.get("benchmark_id") != item["benchmark_id"]
                        or solver.get("sequence_id") != item["sequence_id"]
                        or int(solver.get("turn_index")) != int(item["turn_index"])
                        or solver.get("status") not in allowed_solver_status
                        or not str(solver.get("candidate_answer") or "")
                        or not HEX64.fullmatch(str(solver.get("solver_input_fingerprint") or ""))
                    ):
                        raise RuntimeError(f"{identity}: solver identity/status/fingerprint mismatch")
                    expected_images = (
                        expected_solver_images[(contest, item["item_id"])]
                        if model["input_mode"] == "vision"
                        else []
                    )
                    expected_current = (
                        current_question_images[(contest, item["item_id"])]
                        if model["input_mode"] == "vision"
                        else []
                    )
                    compare_image_descriptors(solver.get("question_images"), expected_images, image_cache=image_cache, label=identity)
                    compare_image_descriptors(solver.get("current_question_images"), expected_current, image_cache=image_cache, label=identity)
                    expected_prompt_hash = sha_text(frozen_solver_prompt(item, str(model["input_mode"])))
                    if solver.get("question_text_sha256") != expected_prompt_hash:
                        raise RuntimeError(f"{identity}: solver prompt/staticization hash mismatch")
                    expected_solver_fingerprint = frozen_pipeline.solver_input_fingerprint(
                        model,
                        item,
                        frozen_solver_prompt(item, str(model["input_mode"])),
                        expected_images,
                    )
                    if solver.get("solver_input_fingerprint") != expected_solver_fingerprint:
                        raise RuntimeError(f"{identity}: solver input fingerprint does not recompute")
                    solver_attempt_count, recovered_inflight_count, abandoned_inflight_count = verify_attempts(
                        root,
                        solver,
                        expected_images,
                        frozen_pipeline=frozen_pipeline,
                        image_cache=image_cache,
                        label=f"{identity}/solver",
                    )
                    counts["solver_api_attempt_records"] += solver_attempt_count
                    counts["solver_terminals_recovered_from_inflight_attempt_sidecar"] += recovered_inflight_count
                    counts["abandoned_inflight_attempt_sidecars_counted_as_wire_ordinals"] += abandoned_inflight_count
                    if model["input_mode"] == "vision":
                        counts["vision_solver_terminals_direct_png_verified"] += 1
                    else:
                        counts["text_solver_terminals_zero_image_verified"] += 1
                    counts["solver_terminals"] += 1

                    if not judge_path.is_file() or judge_path.is_symlink() or judge_path.stat().st_nlink != 1:
                        raise RuntimeError(f"{identity}: judge terminal missing/linked")
                    source_judge = load_json(judge_path)
                    recovered_attempt_count = 0
                    if source_judge.get("terminal") is True:
                        judge = source_judge
                    else:
                        judge, recovered_attempt_count = audit_quote_recovery_overlay(
                            source_root=root,
                            source_judge_path=judge_path,
                            source_judge=source_judge,
                            solver_path=solver_path,
                            solver=solver,
                            item=item,
                            route=route,
                            frozen_pipeline=frozen_pipeline,
                        )
                    score = judge.get("score")
                    if (
                        judge.get("schema_version") != "science_arena_judge_result_v1"
                        or judge.get("terminal") is not True
                        or judge.get("item_id") != item["item_id"]
                        or judge.get("benchmark_id") != item["benchmark_id"]
                        or judge.get("sequence_id") != item["sequence_id"]
                        or judge.get("solver_route") != route
                        or judge.get("judge_route") != JUDGE_ROUTE
                        or not isinstance(score, (int, float))
                        or not math.isfinite(float(score))
                        or float(score) < -1e-12
                        or float(score) > float(item["max_score"]) + 1e-12
                        or abs(float(judge.get("max_score")) - float(item["max_score"])) > 1e-12
                        or not HEX64.fullmatch(str(judge.get("judge_input_fingerprint") or ""))
                    ):
                        raise RuntimeError(f"{identity}: judge identity/score/fingerprint mismatch")
                    judge_images = ordered_unique(
                        current_question_images[(contest, item["item_id"])]
                        + current_solution_images[(contest, item["item_id"])]
                    )
                    expected_judge_fingerprint = frozen_pipeline.judge_input_fingerprint(
                        judge_model,
                        item,
                        str(solver["candidate_answer"]),
                        current_question_images[(contest, item["item_id"])],
                        current_solution_images[(contest, item["item_id"])],
                    )
                    if judge.get("judge_input_fingerprint") != expected_judge_fingerprint:
                        raise RuntimeError(f"{identity}: judge input fingerprint does not recompute")
                    if judge.get("status") == "deterministic_zero_model_no_answer":
                        if float(score) != 0.0 or judge.get("attempt_count") != 0 or solver["candidate_answer"] != NO_ANSWER_LABEL:
                            raise RuntimeError(f"{identity}: invalid deterministic-zero closure")
                        counts["deterministic_zero_judge_terminals"] += 1
                    elif judge.get("status") == "judged":
                        compare_image_descriptors(judge.get("question_images"), current_question_images[(contest, item["item_id"])], image_cache=image_cache, label=identity)
                        compare_image_descriptors(judge.get("official_solution_images"), current_solution_images[(contest, item["item_id"])], image_cache=image_cache, label=identity)
                        compare_image_descriptors(judge.get("reference_images"), judge_images, image_cache=image_cache, label=identity)
                        judge_attempt_count, recovered_inflight_count, abandoned_inflight_count = verify_attempts(
                            root,
                            judge,
                            judge_images,
                            frozen_pipeline=frozen_pipeline,
                            image_cache=image_cache,
                            label=f"{identity}/judge",
                        )
                        if recovered_inflight_count:
                            raise RuntimeError(f"{identity}: judge attempt cannot use solver inflight-terminal recovery")
                        counts["judge_api_attempt_records"] += judge_attempt_count
                        counts["abandoned_inflight_attempt_sidecars_counted_as_wire_ordinals"] += abandoned_inflight_count
                        counts["vision_judge_terminals_direct_question_and_solution_png_verified"] += 1
                    elif judge.get("status") == "judged_deterministic_evidence_quote_replay_v1":
                        compare_image_descriptors(judge.get("question_images"), current_question_images[(contest, item["item_id"])], image_cache=image_cache, label=identity)
                        compare_image_descriptors(judge.get("official_solution_images"), current_solution_images[(contest, item["item_id"])], image_cache=image_cache, label=identity)
                        compare_image_descriptors(judge.get("reference_images"), judge_images, image_cache=image_cache, label=identity)
                        judge_attempt_count, recovered_inflight_count, abandoned_inflight_count = verify_attempts(
                            root,
                            judge,
                            judge_images,
                            frozen_pipeline=frozen_pipeline,
                            image_cache=image_cache,
                            label=f"{identity}/judge-recovery-source",
                        )
                        if recovered_inflight_count:
                            raise RuntimeError(f"{identity}: recovered judge attempt cannot use solver inflight-terminal recovery")
                        counts["judge_api_attempt_records"] += judge_attempt_count
                        counts["abandoned_inflight_attempt_sidecars_counted_as_wire_ordinals"] += abandoned_inflight_count
                        if recovered_attempt_count != int(judge.get("attempt_count") or 0):
                            raise RuntimeError(f"{identity}: recovered attempt count mismatch")
                        counts["deterministic_quote_recovery_judge_terminals"] += 1
                        counts["deterministic_quote_recovery_saved_responses_replayed"] += recovered_attempt_count
                        counts["vision_judge_terminals_direct_question_and_solution_png_verified"] += 1
                    else:
                        raise RuntimeError(f"{identity}: unsupported judge terminal status")
                    counts["judge_terminals"] += 1
                    item_scores[item["item_id"]] = float(score)
                    observations[(replicate, contest, route, item["item_id"])] = {
                        "candidate_answer_sha256": sha_text(str(solver["candidate_answer"])),
                        "score": float(score),
                    }
                per_route_scores[route] = item_scores
                recomputed, report_scale, breakdown = independently_recompute_score(contest, contest_items, item_scores)
                aggregate_row = aggregates[(replicate, contest)]["aggregates"][route][CONTESTS[contest]["benchmark"]]
                if (
                    aggregate_row.get("complete") is not True
                    or abs(float(aggregate_row["score"]) - recomputed) > 1e-8
                    or abs(float(aggregate_row["report_scale"]) - report_scale) > 1e-12
                ):
                    raise RuntimeError(f"{contest}/{replicate}/{route}: aggregate does not close")
                if contest == "icho2026":
                    counts["icho_aggregate_route_replicates_weighted_to_60_verified"] += 1
                else:
                    counts["ipho_aggregate_route_replicates_summed_to_30_verified"] += 1
            actual_solver_paths = {
                path
                for path in (root / "solver").glob("*/*/*/*.json")
                if "-attempts" not in path.parts
                and path.relative_to(root / "solver").parts[0] in selected_route_dirs
            }
            actual_judge_paths = {
                path
                for path in (root / "judge").glob("*/*/*/*.json")
                if "-attempts" not in path.parts
                and path.relative_to(root / "judge").parts[0] in selected_route_dirs
            }
            if actual_solver_paths != expected_solver_paths or actual_judge_paths != expected_judge_paths:
                raise RuntimeError(f"{contest}/{replicate}: missing or extra terminal paths")
            for phase, expected_paths in (
                ("solver", expected_solver_paths),
                ("judge", expected_judge_paths),
            ):
                for attempt_path in (root / phase).glob("*/*/*/*-attempts/attempt-*.json"):
                    if attempt_path.relative_to(root / phase).parts[0] not in selected_route_dirs:
                        continue
                    terminal = attempt_path.parent.parent / f"{attempt_path.parent.name[:-len('-attempts')]}.json"
                    if terminal not in expected_paths:
                        raise RuntimeError(
                            f"{contest}/{replicate}: orphan or extra {phase} attempt sidecar {attempt_path}"
                        )
            run_summaries.append(
                {
                    "contest": contest,
                    "replicate": replicate,
                    "solver_terminal_count": len(expected_solver_paths),
                    "judge_terminal_count": len(expected_judge_paths),
                    "aggregate_path": effective_aggregate_path(contest, replicate).relative_to(WORKSPACE).as_posix(),
                    "aggregate_sha256": sha256_file(effective_aggregate_path(contest, replicate)),
                    "source_aggregate_sha256": sha256_file(root / "aggregate.json"),
                }
            )
    frozen_input_summary: dict[str, Any] = {}
    for contest, paths_by_kind in contest_asset_paths.items():
        image_rows = []
        for kind in ("question", "solution"):
            for path in sorted(paths_by_kind[kind]):
                proof = image_cache[path]
                image_rows.append(
                    {
                        "kind": kind,
                        "workspace_relative_path": path.relative_to(WORKSPACE).as_posix(),
                        "sha256": proof["sha256"],
                        "bytes": proof["bytes"],
                        "mime_type": proof["mime_type"],
                        "png_magic_verified": proof["png_magic_verified"],
                    }
                )
        provenance_rows = [
            {
                "workspace_relative_path": path.relative_to(WORKSPACE).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in sorted(paths_by_kind["provenance"])
        ]
        frozen_input_summary[contest] = {
            "items_jsonl_sha256": sha256_file(CONTESTS[contest]["items"]),
            "question_png_unique_count": len(paths_by_kind["question"]),
            "solution_png_unique_count": len(paths_by_kind["solution"]),
            "png_manifest_sha256": canonical_json_sha256(image_rows),
            "all_png_magic_verified": all(row["png_magic_verified"] for row in image_rows),
            "staticization_provenance": provenance_rows,
            "staticization_provenance_manifest_sha256": canonical_json_sha256(provenance_rows),
        }
    audit = {
        "status": "PASS_all_three_fresh_replicates_complete_and_wire_verified",
        "counts": counts,
        "image_asset_unique_file_count": len(image_cache),
        "runs": run_summaries,
        "frozen_inputs": frozen_input_summary,
        "score_semantics": {
            "ipho2026": "sum of 23 rubric-item scores; three problems each close to 10, report scale 30",
            "icho2026": "within each of nine problems, raw rubric score / official raw maximum * official problem weight; official weights sum to report scale 60",
        },
    }
    return audit, observations


def main() -> int:
    args = parse_args()
    frozen_pipeline = load_frozen_pipeline()
    all_models = load_json(MODELS)["models"]
    models = (
        [model for model in all_models if model.get("route") == args.model_route]
        if args.model_route
        else all_models
    )
    if not models:
        raise RuntimeError(f"model route is not present in frozen manifest: {args.model_route!r}")
    if args.model_route and len(models) != 1:
        raise RuntimeError(f"model route is not unique in frozen manifest: {args.model_route!r}")
    items = {contest: load_jsonl(config["items"]) for contest, config in CONTESTS.items()}
    baseline_rows = {row["route"]: row for row in load_json(EXTERNAL_BASELINE)["rows"]}
    aggregates: dict[tuple[str, str], dict[str, Any]] = {}
    for replicate in REPLICATES:
        for contest in CONTESTS:
            path = effective_aggregate_path(contest, replicate)
            if not path.is_file():
                raise RuntimeError(f"repeat is not complete: missing {path}")
            value = load_json(path)
            expected_rows = len(models) * len(items[contest])
            allowed_row_counts = {expected_rows}
            if args.model_route:
                # IPhO remains a sealed all-route aggregate, while the new IChO
                # stability rerun writes a selected-route 68-row aggregate.
                allowed_row_counts.add(len(all_models) * len(items[contest]))
            if not value.get("complete") or value.get("missing_count") != 0:
                raise RuntimeError(f"incomplete aggregate: {path}")
            if value.get("row_count") not in allowed_row_counts:
                raise RuntimeError(f"unexpected row_count in {path}")
            missing_routes = {
                str(model["route"])
                for model in models
                if str(model["route"]) not in (value.get("aggregates") or {})
            }
            if missing_routes:
                raise RuntimeError(f"selected model routes are absent from aggregate {path}: {sorted(missing_routes)}")
            if contest == "ipho2026" and replicate == "repC":
                source_path = run_root(contest, replicate) / "aggregate.json"
                source = load_json(source_path)
                expected_missing = {f"{route}::{item_id}" for route, item_id in QUOTE_RECOVERY_CASES}
                provenance = value.get("recovery_provenance") or {}
                manifest_path = QUOTE_RECOVERY_ROOT / "recovery_manifest.json"
                hashes_path = QUOTE_RECOVERY_ROOT / "FILES_SHA256.json"
                manifest = load_json(manifest_path)
                hashes = load_json(hashes_path)
                if (
                    value.get("schema_version") != "science_arena_aggregate_recovery_overlay_v1"
                    or source.get("complete") is not False
                    or source.get("row_count") != 320
                    or source.get("missing_count") != 2
                    or set(source.get("missing") or []) != expected_missing
                    or provenance.get("source_incomplete_aggregate_sha256") != sha256_file(source_path)
                    or provenance.get("recovery_manifest_sha256") != sha256_file(manifest_path)
                    or provenance.get("recovered_cell_count") != 2
                    or provenance.get("api_calls_made") != 0
                    or manifest.get("algorithm") != QUOTE_RECOVERY_ALGORITHM
                    or manifest.get("recovered_cell_count") != 2
                    or manifest.get("api_calls_made") != 0
                    or manifest.get("all_fourteen_saved_responses_strict_valid_after_quote_only_replay") is not True
                    or hashes.get("schema_version") != "science_arena_recovery_file_hashes_v1"
                ):
                    raise RuntimeError("IPhO repC deterministic recovery aggregate provenance mismatch")
                for file_row in hashes.get("files") or []:
                    recovered_file = WORKSPACE / str(file_row["path"])
                    if sha256_file(recovered_file) != file_row.get("sha256"):
                        raise RuntimeError(f"recovery file hash mismatch: {recovered_file}")
            aggregates[(replicate, contest)] = value

    protocol_audit, observations = audit_all_runs(models, items, aggregates, frozen_pipeline)

    rows: list[dict[str, Any]] = []
    detailed: list[dict[str, Any]] = []
    macro_by_rep: dict[str, dict[str, float]] = {replicate: {} for replicate in REPLICATES}
    for model in models:
        route = model["route"]
        contest_detail: dict[str, Any] = {}
        normalized_by_contest: dict[str, list[float]] = {}
        exact_same_answers = 0
        answer_cells = 0
        changed_score_cells = 0
        total_score_cells = 0
        for contest, config in CONTESTS.items():
            benchmark = config["benchmark"]
            raw_max = config["raw_max"]
            raw_scores = [
                float(aggregates[(replicate, contest)]["aggregates"][route][benchmark]["score"])
                for replicate in REPLICATES
            ]
            normalized = [score / raw_max * 100.0 for score in raw_scores]
            normalized_by_contest[contest] = normalized
            for item in items[contest]:
                solver_hashes: list[str] = []
                judge_scores: list[float] = []
                for replicate in REPLICATES:
                    observation = observations[(replicate, contest, route, item["item_id"])]
                    solver_hashes.append(observation["candidate_answer_sha256"])
                    judge_scores.append(observation["score"])
                answer_cells += 1
                exact_same_answers += int(len(set(solver_hashes)) == 1)
                total_score_cells += 1
                if max(judge_scores) - min(judge_scores) > 1e-12:
                    changed_score_cells += 1
            external = baseline_rows[route]["competitions"][contest]
            contest_detail[contest] = {
                "raw_scores": dict(zip(REPLICATES, raw_scores)),
                "normalized_percent": dict(zip(REPLICATES, normalized)),
                "normalized_percent_stats": stats(normalized),
                "external_published_table6": {
                    "score": float(external["score"]),
                    "raw_max": float(external["raw_max"]),
                    "normalized_percent": float(external["normalized_percent"]),
                    "delta_fresh_mean_minus_external_pp": statistics.fmean(normalized)
                    - float(external["normalized_percent"]),
                },
            }
        macro = [
            statistics.fmean([normalized_by_contest[contest][index] for contest in CONTESTS])
            for index in range(len(REPLICATES))
        ]
        for replicate, value in zip(REPLICATES, macro):
            macro_by_rep[replicate][route] = value
        macro_stats = stats(macro)
        ipho_raw = [contest_detail["ipho2026"]["raw_scores"][replicate] for replicate in REPLICATES]
        icho_raw = [contest_detail["icho2026"]["raw_scores"][replicate] for replicate in REPLICATES]
        ipho_normalized = [
            contest_detail["ipho2026"]["normalized_percent"][replicate]
            for replicate in REPLICATES
        ]
        icho_normalized = [
            contest_detail["icho2026"]["normalized_percent"][replicate]
            for replicate in REPLICATES
        ]
        row = {
            "route": route,
            "display_name": model.get("display_name", route),
            "input_mode": model["input_mode"],
            "ipho_repA_score": ipho_raw[0],
            "ipho_repB_score": ipho_raw[1],
            "ipho_repC_score": ipho_raw[2],
            "ipho_mean_score": statistics.fmean(ipho_raw),
            "ipho_sample_sd_points": statistics.stdev(ipho_raw),
            "ipho_range_points": max(ipho_raw) - min(ipho_raw),
            "ipho_mean_percent": statistics.fmean(ipho_normalized),
            "ipho_sample_sd_pp": statistics.stdev(ipho_normalized),
            "ipho_range_pp": max(ipho_normalized) - min(ipho_normalized),
            "icho_repA_score": icho_raw[0],
            "icho_repB_score": icho_raw[1],
            "icho_repC_score": icho_raw[2],
            "icho_mean_score": statistics.fmean(icho_raw),
            "icho_sample_sd_points": statistics.stdev(icho_raw),
            "icho_range_points": max(icho_raw) - min(icho_raw),
            "icho_mean_percent": statistics.fmean(icho_normalized),
            "icho_sample_sd_pp": statistics.stdev(icho_normalized),
            "icho_range_pp": max(icho_normalized) - min(icho_normalized),
            "macro_repA_percent": macro[0],
            "macro_repB_percent": macro[1],
            "macro_repC_percent": macro[2],
            "macro_mean_percent": macro_stats["mean"],
            "macro_sample_sd_pp": macro_stats["sample_sd"],
            "macro_range_pp": macro_stats["range"],
            "exact_same_answer_fraction": exact_same_answers / answer_cells,
            "end_to_end_item_score_changed_fraction": changed_score_cells / total_score_cells,
        }
        rows.append(row)
        detailed.append(
            {
                **row,
                "competitions": contest_detail,
                "macro_percent_stats": macro_stats,
                "answer_cell_count": answer_cells,
                "exact_same_answer_count": exact_same_answers,
                "end_to_end_item_score_cell_count": total_score_cells,
                "end_to_end_item_score_changed_count": changed_score_cells,
            }
        )

    pairwise_rank = []
    if len(models) > 1:
        for left_index, left in enumerate(REPLICATES):
            for right in REPLICATES[left_index + 1 :]:
                routes = [model["route"] for model in models]
                pairwise_rank.append(
                    {
                        "left": left,
                        "right": right,
                        "spearman_macro": spearman(
                            [macro_by_rep[left][route] for route in routes],
                            [macro_by_rep[right][route] for route in routes],
                        ),
                    }
                )
    row_by_route = {row["route"]: row for row in rows}
    ipho_item_by_id = {item["item_id"]: item for item in items["ipho2026"]}
    recovery_sensitivity = []
    for route, item_id in sorted(QUOTE_RECOVERY_CASES):
        if route not in row_by_route:
            continue
        recovered_score = float(observations[("repC", "ipho2026", route, item_id)]["score"])
        including_repC = float(
            aggregates[("repC", "ipho2026")]["aggregates"][route]["ipho2026"]["score"]
        )
        recovery_sensitivity.append(
            {
                "route": route,
                "item_id": item_id,
                "recovered_score": recovered_score,
                "item_max_score": float(ipho_item_by_id[item_id]["max_score"]),
                "repC_ipho_including_recovery_raw_over_30": including_repC,
                "repC_ipho_counterfactual_recovered_cell_zero_raw_over_30": including_repC
                - recovered_score,
                "repC_ipho_impact_points": recovered_score,
                "repC_ipho_impact_pp": recovered_score / 30.0 * 100.0,
                "three_repeat_ipho_mean_including_recovery_raw_over_30": row_by_route[route][
                    "ipho_mean_score"
                ],
                "three_repeat_ipho_mean_counterfactual_recovered_cell_zero_raw_over_30": row_by_route[
                    route
                ]["ipho_mean_score"]
                - recovered_score / 3.0,
                "three_repeat_ipho_mean_impact_pp": recovered_score / 30.0 * 100.0 / 3.0,
                "three_repeat_two_contest_macro_mean_impact_pp": recovered_score
                / 30.0
                * 100.0
                / 3.0
                / 2.0,
                "counterfactual_policy": "Conservative sensitivity only: retain every other judged cell and set this recovered cell to zero; the failed source terminal itself remains missing and unmodified.",
            }
        )
    summary = {
        "schema_version": "science_arena_ipho_icho_three_fresh_repeat_stability_v1",
        "scope": {
            "replicates": list(REPLICATES),
            "contests": list(CONTESTS),
            "route_count": len(models),
            "model_route_filter": args.model_route,
            "item_count_per_route": sum(len(value) for value in items.values()),
            "judge_route": JUDGE_ROUTE,
            "pipeline_snapshot_sha256": PIPELINE_SNAPSHOT_SHA256,
            "model_manifest_sha256": sha256_file(MODELS),
            "randomness": "No explicit seed or temperature; identical frozen payload/options were independently called. Variation therefore measures provider/backend stochasticity plus solver-judge interaction.",
            "external_table6_policy": "Published Table-6 values are external comparisons only and are excluded from repeat variance.",
            "variance_separation": "This experiment changes solver answers and re-runs the fixed judge, so score variation is end-to-end. Judge-only variance is estimated separately by cross-judge fixed-answer repeats.",
        },
        "protocol_audit": protocol_audit,
        "repC_quote_recovery_sensitivity": recovery_sensitivity,
        "pairwise_rank_stability": pairwise_rank,
        "overall": {
            "mean_model_macro_sample_sd_pp": statistics.fmean(row["macro_sample_sd_pp"] for row in rows),
            "max_model_macro_sample_sd_pp": max(row["macro_sample_sd_pp"] for row in rows),
            "mean_model_macro_range_pp": statistics.fmean(row["macro_range_pp"] for row in rows),
            "max_model_macro_range_pp": max(row["macro_range_pp"] for row in rows),
            "mean_exact_same_answer_fraction": statistics.fmean(row["exact_same_answer_fraction"] for row in rows),
            "mean_end_to_end_item_score_changed_fraction": statistics.fmean(row["end_to_end_item_score_changed_fraction"] for row in rows),
        },
        "models": detailed,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT / "repeat_stability.json"
    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    csv_path = OUTPUT / "repeat_stability.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    md = [
        "# IPhO/IChO 2026 three-fresh-repeat stability",
        "",
        "Three all-new independent calls per route; the published Table-6 run is an external comparison only.",
        "",
        "| Model | Input | IPhO A/B/C (/30) | IPhO mean±SD | IChO A/B/C (/60) | IChO mean±SD | Macro A/B/C (%) | Macro mean±SD (pp) | Range (pp) | Exact-answer same | End-to-end item score changed |",
        "|---|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda value: value["macro_mean_percent"], reverse=True):
        md.append(
            "| {display_name} | {input_mode} | {ipho_repA_score:.2f}/{ipho_repB_score:.2f}/{ipho_repC_score:.2f} | "
            "{ipho_mean_score:.2f}±{ipho_sample_sd_points:.2f} | "
            "{icho_repA_score:.2f}/{icho_repB_score:.2f}/{icho_repC_score:.2f} | "
            "{icho_mean_score:.2f}±{icho_sample_sd_points:.2f} | "
            "{macro_repA_percent:.2f}/{macro_repB_percent:.2f}/{macro_repC_percent:.2f} | "
            "{macro_mean_percent:.2f}±{macro_sample_sd_pp:.2f} | "
            "{macro_range_pp:.2f} | {exact_same_answer_fraction:.1%} | "
            "{end_to_end_item_score_changed_fraction:.1%} |".format(**row)
        )
    recovery_note = (
        f"IPhO repC required {len(recovery_sensitivity)} deterministic evidence-quote overlay(s) "
        f"({len(recovery_sensitivity)}/{len(models) * len(items['ipho2026'])} selected cells): all "
        f"{len(recovery_sensitivity) * 7} saved HTTP-200 responses replayed strict-valid after "
        "quote-lexeme-only repair, with scores and all rubric decisions unchanged and no new API call. "
        "A conservative sensitivity that instead assigns each recovered cell zero is included in the JSON output."
        if recovery_sensitivity
        else "No deterministic evidence-quote recovery was required for the selected IPhO route(s)."
    )
    md.extend(
        [
            "",
            "SD is the sample standard deviation across the three fresh run-level macro scores. "
            "No explicit seed or temperature was sent. These repeats estimate end-to-end provider/backend "
            "stochasticity under an otherwise identical frozen protocol; they do not isolate solver variance from judge variance. "
            "Judge-only variance is reported by the separate fixed-answer cross-judge experiment.",
            "",
            recovery_note,
            "",
            "IChO scores are independently recomputed problem by problem as raw rubric points divided by the "
            "official raw maximum and multiplied by the official problem weight; the nine weights sum to 60. "
            "The machine-readable protocol audit must be PASS before this table is emitted.",
        ]
    )
    (OUTPUT / "repeat_stability.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    hashes = []
    for path in (json_path, csv_path, OUTPUT / "repeat_stability.md"):
        hashes.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (OUTPUT / "SHA256SUMS").write_text("\n".join(hashes) + "\n", encoding="utf-8")
    print(json.dumps(summary["overall"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
