#!/usr/bin/env python3
"""Fail-closed composite analysis for the six-judge rebuttal experiment.

Effective score panel:
* five routes from the original 1,638-cell run;
* one Gemini-3.1 cell from its immutable same-config recovery ledger;
* all 273 DeepSeek cells from the uniform 65k thinking-enabled v2 route.

The old DeepSeek 16k route is retained and costed as failure diagnostics, but
none of its scores enter reliability, agreement, totals, or baseline metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import analyze_rebuttal_cross_judges_20260830 as base
import pipeline
import run_rebuttal_cross_judges_20260830 as cross


HERE = Path(__file__).resolve().parent
ORIGINAL_ROOT = HERE / "runs/ipho_icho2026_gemini35solver_crossjudge_6x3_20260830"
G31_ROOT = HERE / "runs/ipho_icho2026_gemini31_postexhaustion_recovery_20260830"
DEEP_ROOT = HERE / "runs/deepseek_longthinking_v2_full_materialized_20260830"
DEEP_SMOKE_ROOT = HERE / "runs/deepseek_longthinking_v2_smoke_fast_v2_20260830"
PANEL = HERE / "manifests/rebuttal_judges_6x3_20260830.json"
DEEP_PANEL = HERE / "manifests/rebuttal_deepseek_longthinking_v2_20260830.json"
DEEP_RUNNER = HERE / "run_rebuttal_deepseek_longthinking_v2_20260830.py"
RECOVERY_RUNNER = HERE / "recover_rebuttal_gemini31_postexhaustion_20260830.py"
EXPECTED_RECOVERY_RUNNER_SHA256 = "2cc8ec42d2f03f0364abefad5c404e30e7d32225fb6ba8a2206918c309aadfd9"
PIPELINE = HERE / "pipeline.py"
EXPECTED_PIPELINE_SHA256 = "bb52de6b27696551e523a6cab1359e62736496aa5807624551098716fbcf87fd"
CASES = base.CASES
DEEP_ROUTE = "deepseek/deepseek-v4-pro"
G31_ROUTE = "google/google/gemini-3.1-pro-preview"
G31_KEY = (G31_ROUTE, 3, "icho2026", "icho2026_t4_4.3")
RECOVERY_SCHEMA = "science_arena_rebuttal_cross_judge_post_exhaustion_recovery_v1"
DEEP_RUN_SCHEMA = "science_arena_rebuttal_deepseek_longthinking_run_v2"
EXPECTED_ORIGINAL_FAILURES = {
    G31_KEY,
    *{
        (DEEP_ROUTE, repeat, "icho2026", item_id)
        for repeat in range(1, 4)
        for item_id in ("icho2026_t6_6.2", "icho2026_t9_9.8")
    },
}


# FileProvider residency is not artifact content. On macOS, an immutable cloud
# file can retain its logical size while returning premature EOF until an exact
# download is requested. Every source read below therefore requests only the
# exact affected file and immediately verifies its pinned logical identity and
# complete bytes. No source file is written or replaced.
_ORIGINAL_PIPELINE_READ = pipeline.read_regular_file_bytes
_FILEPROVIDER_MATERIALIZATIONS: dict[str, dict[str, Any]] = {}


def _strict_read_with_timeout(source: Path, seconds: float = 20.0) -> bytes:
    """Bound a potentially racy FileProvider pread in the analyzer process."""
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, 0)

    def timeout_handler(_signum: int, _frame: Any) -> None:
        raise TimeoutError(f"timed out reading materialized source: {source}")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        return _ORIGINAL_PIPELINE_READ(source)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer != (0.0, 0.0):
            signal.setitimer(signal.ITIMER_REAL, *previous_timer)


def materialized_regular_file_bytes(path: Path) -> bytes:
    source = Path(path).resolve()
    last_error: Exception | None = None
    for ordinal in range(1, 9):
        before = source.stat()
        logical_identity = (before.st_size, before.st_mtime_ns)
        requested = bool(
            sys.platform == "darwin"
            and before.st_size > 0
            and before.st_blocks == 0
        )
        if requested:
            completed = subprocess.run(
                ["brctl", "download", os.fspath(source)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                timeout=15,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"brctl exact-file download failed for {source}: "
                    f"{completed.stderr.decode('utf-8', errors='replace')}"
                )
            # Give FileProvider a short window to expose allocated blocks. A
            # subsequent strict read remains authoritative even if the flag is
            # slow to update or the service later evicts the local blocks.
            # st_blocks is advisory for FileProvider and can remain zero even
            # when a complete read is immediately available. Give the request
            # only a brief head start; the bounded strict read below decides.
            deadline = time.monotonic() + 0.05
            while time.monotonic() < deadline:
                current = source.stat()
                if current.st_blocks > 0:
                    break
                time.sleep(0.01)
        try:
            data = _strict_read_with_timeout(source)
            after = source.stat()
            if (after.st_size, after.st_mtime_ns) != logical_identity:
                raise RuntimeError(f"logical source identity changed during materialization: {source}")
            if len(data) != before.st_size:
                raise RuntimeError(
                    f"materialized byte count mismatch for {source}: {len(data)}!={before.st_size}"
                )
            if requested or ordinal > 1:
                key = str(source)
                old = _FILEPROVIDER_MATERIALIZATIONS.get(key, {})
                _FILEPROVIDER_MATERIALIZATIONS[key] = {
                    "path": key,
                    "logical_size_bytes": before.st_size,
                    "logical_mtime_ns": before.st_mtime_ns,
                    "sha256_after_materialization": hashlib.sha256(data).hexdigest(),
                    "exact_file_download_requests": int(old.get("exact_file_download_requests", 0)) + int(requested),
                    "read_attempts_until_complete": ordinal,
                    "residency_only_not_source_mutation": True,
                }
            return data
        except (RuntimeError, TimeoutError) as exc:
            last_error = exc
            retryable = (
                isinstance(exc, TimeoutError)
                or "premature EOF before its pinned size" in str(exc)
                or "verified fixed-length pread failed at offset" in str(exc)
            )
            if not retryable or ordinal == 8:
                raise
            if sys.platform == "darwin":
                subprocess.run(
                    ["brctl", "download", os.fspath(source)],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=15,
                    check=False,
                )
            time.sleep(min(0.25 * ordinal, 2.0))
    raise RuntimeError(f"could not materialize {source}: {last_error}")


def materialized_json(path: Path) -> dict[str, Any]:
    value = json.loads(materialized_regular_file_bytes(Path(path)).decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def materialized_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    text = materialized_regular_file_bytes(Path(path)).decode("utf-8")
    for number, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"non-object JSONL row {number}: {path}")
        rows.append(value)
    return rows


def materialized_sha256(path: Path) -> str:
    return hashlib.sha256(materialized_regular_file_bytes(Path(path))).hexdigest()


# Imported helpers look these names up dynamically; patching only this analysis
# process keeps the production artifacts and runners untouched.
pipeline.read_regular_file_bytes = materialized_regular_file_bytes
pipeline.sha256_file = materialized_sha256
base.read_json = materialized_json
base.read_jsonl = materialized_jsonl
base.sha256 = materialized_sha256


def retrospective_pipeline_provenance(
    original: Path,
    g31_root: Path,
    deep_root: Path,
) -> tuple[dict[str, Any], list[str]]:
    """Pin this post-hoc replay dependency without backdating it to old runs.

    The sealed run provenance files did not contemporaneously hash pipeline.py.
    The current bytes are therefore evidence about *this* strict replay only.
    The previously expected hash is retained as a historical lead, not asserted
    to be the source that generated any run.
    """
    errors: list[str] = []
    if not PIPELINE.is_file():
        return {}, [f"imported pipeline missing: {PIPELINE}"]
    stat = PIPELINE.stat()
    pipeline_sha = base.sha256(PIPELINE)
    run_files = {
        "original_cross_judge": original / "run_provenance.json",
        "gemini31_recovery": g31_root / "provenance.json",
        "deepseek_65k_smoke": DEEP_SMOKE_ROOT / "run_provenance.json",
        "deepseek_65k_full": deep_root / "run_provenance.json",
    }
    runs: list[dict[str, Any]] = []
    pipeline_mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
    for role, path in run_files.items():
        if not path.is_file():
            errors.append(f"provenance missing for retrospective pin: {path}")
            continue
        data = base.read_json(path)
        created_at = data.get("created_at")
        try:
            created = datetime.fromisoformat(str(created_at))
            if created.tzinfo is None:
                raise ValueError("timezone missing")
            predates = pipeline_mtime < created
        except Exception as exc:
            errors.append(f"invalid created_at in {path}: {created_at!r}: {exc}")
            predates = False
        runs.append({
            "role": role,
            "provenance_path": str(path),
            "provenance_sha256": base.sha256(path),
            "created_at": created_at,
            "current_replay_pipeline_mtime_predates_run": predates,
        })
    return {
        "schema_version": "science_arena_posthoc_replay_pipeline_dependency_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "pipeline_path": str(PIPELINE),
        "posthoc_replay_pipeline_sha256": pipeline_sha,
        "pipeline_size_bytes": stat.st_size,
        "pipeline_mtime_utc": pipeline_mtime.isoformat(),
        "historical_expected_pipeline_sha256_not_contemporaneously_sealed": EXPECTED_PIPELINE_SHA256,
        "current_matches_historical_expected_sha256": pipeline_sha == EXPECTED_PIPELINE_SHA256,
        "runs": runs,
        "evidentiary_scope": (
            "The current pipeline.py hash identifies only the code imported by this "
            "post-hoc strict replay. The original sealed run provenance did not "
            "contemporaneously pin pipeline.py; neither the current file nor the "
            "historical expected hash is represented as the source that generated "
            "the old runs. Filesystem mtime is reported descriptively and is not "
            "used to backdate or authenticate run-generation code."
        ),
    }, errors


def audit_gemini31_recovery_payloads(
    g31_root: Path,
    catalog: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    """Rebuild both recovery payloads, including the retry-feedback delta."""
    import copy

    errors: list[str] = []
    result_path = g31_root / "result.json"
    if not result_path.is_file():
        return {}, ["Gemini-3.1 recovery result missing for payload binding"]
    result = base.read_json(result_path)
    item = catalog[("icho2026", "icho2026_t4_4.3")]
    source = Path(str(result.get("source_solver_path") or ""))
    if not source.is_absolute():
        source = base.REPO / source
    try:
        source_raw = pipeline.read_regular_file_bytes(source)
        solver = json.loads(source_raw.decode("utf-8"))
        candidate = str(solver.get("candidate_answer") or "")
    except Exception as exc:
        return {}, [f"cannot reload Gemini-3.1 recovery candidate: {exc}"]
    if not candidate:
        errors.append("Gemini-3.1 recovery candidate is empty")
    if pipeline.sha256_bytes(source_raw) != result.get("source_solver_sha256"):
        errors.append("Gemini-3.1 recovery source solver SHA mismatch")
    if pipeline.sha256_bytes(candidate.encode("utf-8")) != result.get("candidate_sha256"):
        errors.append("Gemini-3.1 recovery candidate SHA mismatch")

    loaded_panel = cross.load_panel(PANEL)
    judge = next(value for value in loaded_panel["judges"] if value["route"] == G31_ROUTE)
    if (
        judge.get("route") != G31_ROUTE
        or judge.get("judge_max_completion_tokens") != 16384
        or judge.get("judge_options") != {"thinking_level": "high"}
    ):
        errors.append("Gemini-3.1 recovery frozen route/options changed")
    q = pipeline.resolve_paths(pipeline.question_image_values(item), cross.REPO)
    s = pipeline.resolve_paths(pipeline.reference_image_values(item), cross.REPO)
    images = pipeline.ordered_unique_paths(q, s)
    prompt = cross.modality_prompt(item, candidate, judge, len(q), len(s))
    base_message = pipeline.make_image_message(prompt, images)
    expected_criteria = pipeline.normalized_rubric_criteria(
        pipeline.grading_rubric_for(item)
    )
    criterion_ids = {str(row["criterion_id"]) for row in expected_criteria}

    def expected_payload(feedback: str) -> dict[str, Any]:
        message = copy.deepcopy(base_message)
        if feedback:
            addition = pipeline.judge_retry_instruction(
                feedback,
                candidate=candidate,
                expected_criterion_ids=criterion_ids,
            )
            if isinstance(message["content"], str):
                message["content"] += addition
            else:
                message["content"][0]["text"] += addition
        return pipeline.base_payload(
            judge,
            [{"role": "system", "content": pipeline.JUDGE_SYSTEM}, message],
            judge=True,
        )

    paths = sorted((g31_root / "online-attempts").glob("attempt-*.json"))
    if len(paths) != 2:
        errors.append(f"Gemini-3.1 recovery payload ledger has {len(paths)} attempts, not 2")
    feedback = ""
    rows: list[dict[str, Any]] = []
    validation_pattern: list[bool] = []
    for ordinal, path in enumerate(paths, 1):
        sidecar = base.read_json(path)
        expected = expected_payload(feedback)
        observed_images = sidecar.get("request_images")
        try:
            if not isinstance(observed_images, list):
                raise RuntimeError("request_images is not a list")
            request = sidecar.get("request")
            if not isinstance(request, dict):
                raise RuntimeError("sanitized request is not an object")
            reconstructed = pipeline.reconstruct_request_from_sidecar(
                request,
                str(sidecar.get("request_sha256")),
                observed_images,
            )
            payload_match = (
                pipeline.canonical_json_bytes(reconstructed)
                == pipeline.canonical_json_bytes(expected)
            )
        except Exception as exc:
            payload_match = False
            errors.append(f"Gemini-3.1 recovery attempt {ordinal} reconstruction failed: {exc}")
            reconstructed = {}
        expected_sha = pipeline.sha256_bytes(pipeline.canonical_json_bytes(expected))
        config_match = (
            reconstructed.get("model") == G31_ROUTE
            and reconstructed.get("max_tokens") == 16384
            and reconstructed.get("thinking_level") == "high"
            and reconstructed.get("stream") is False
        )
        if sidecar.get("attempt") != ordinal:
            errors.append(f"Gemini-3.1 recovery attempt ordinal mismatch: {path}")
        if sidecar.get("request_sha256") != expected_sha or not payload_match:
            errors.append(f"Gemini-3.1 recovery expected payload mismatch: {path}")
        if not config_match:
            errors.append(f"Gemini-3.1 recovery route/options mismatch: {path}")
        validation_ok = sidecar.get("validation_ok") is True
        validation_pattern.append(validation_ok)
        rows.append({
            "attempt": ordinal,
            "request_sha256": sidecar.get("request_sha256"),
            "expected_request_sha256": expected_sha,
            "exact_expected_payload_match": payload_match,
            "route_max_tokens_thinking_stream_match": config_match,
            "retry_feedback_applied": bool(feedback),
            "validation_ok": validation_ok,
            "direct_official_png_count": len(observed_images) if isinstance(observed_images, list) else None,
        })
        feedback = str(
            sidecar.get("validation_reason")
            or sidecar.get("error")
            or sidecar.get("response_body_on_error")
            or "previous attempt failed validation"
        )
    if validation_pattern != [False, True]:
        errors.append(f"Gemini-3.1 recovery validation pattern changed: {validation_pattern}")
    return {
        "recovery_runner_path": str(RECOVERY_RUNNER),
        "recovery_runner_sha256": base.sha256(RECOVERY_RUNNER),
        "judge_route": G31_ROUTE,
        "max_tokens": 16384,
        "thinking_level": "high",
        "stream": False,
        "base_prompt_sha256": pipeline.sha256_bytes(prompt.encode("utf-8")),
        "question_image_count": len(q),
        "solution_image_count": len(s),
        "wire_image_count": len(images),
        "payloads": rows,
        "payload_delta_policy": (
            "Attempt 1 uses the frozen base prompt; attempt 2 preserves the same "
            "candidate, rubric, images, route and options and appends only the "
            "deterministic strict-validator retry instruction derived from attempt 1."
        ),
    }, errors


def final_jsons(root: Path) -> list[Path]:
    return [
        path for path in root.glob("judgments/**/*.json")
        if not any(part.endswith("-attempts") for part in path.parts)
    ]


def identity(data: dict[str, Any]) -> tuple[str, int, str, str]:
    return (
        str(data.get("judge_route")),
        int(data.get("repeat", -1)),
        str(data.get("contest")),
        str(data.get("item_id")),
    )


def validate_sources(
    original: Path,
    g31_root: Path,
    deep_root: Path,
    panel: dict[str, Any],
    catalog: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    evidence: dict[str, Any] = {}

    # Original run is intentionally a complete diagnostic accounting with
    # seven nonterminal cells, not a successful six-route terminal panel.
    provenance_path = original / "run_provenance.json"
    summary_path = original / "run_summary.json"
    if not provenance_path.is_file() or not summary_path.is_file():
        errors.append("original run provenance or summary missing")
    else:
        provenance, summary = base.read_json(provenance_path), base.read_json(summary_path)
        if provenance.get("schema_version") != base.RUN_SCHEMA:
            errors.append("original provenance schema mismatch")
        if provenance.get("panel_sha256") != base.sha256(PANEL) or provenance.get("cases_sha256") != base.sha256(CASES):
            errors.append("original provenance panel/cases hash mismatch")
        if summary.get("schema_version") != base.RUN_SCHEMA or summary.get("command") != "run" or summary.get("logical_tasks") != 1638:
            errors.append("original run summary identity mismatch")
        if summary.get("status_counts") != {"judge_failed_validation_after_retries": 7, "judged": 1631}:
            errors.append("original run is not the exact 1631+7 diagnostic accounting")
        evidence["original_run"] = {
            "root": str(original),
            "provenance_sha256": base.sha256(provenance_path),
            "summary_sha256": base.sha256(summary_path),
            "status_counts": summary.get("status_counts"),
        }

    observed_original = final_jsons(original)
    failure_keys: set[tuple[str, int, str, str]] = set()
    for path in observed_original:
        try:
            data = base.read_json(path)
        except Exception as exc:
            errors.append(f"unreadable original final {path}: {exc}")
            continue
        if data.get("terminal") is not True:
            failure_keys.add(identity(data))
    if len(observed_original) != 1638:
        errors.append(f"original final-file count is {len(observed_original)}, not 1638")
    if failure_keys != EXPECTED_ORIGINAL_FAILURES:
        errors.append(f"original failure identity set changed: {sorted(failure_keys)}")

    # One same-configuration Gemini-3.1 recovery, with every source artifact
    # hash checked against the still-present original failure ledger.
    g31_provenance_path, g31_result_path = g31_root / "provenance.json", g31_root / "result.json"
    if not g31_provenance_path.is_file() or not g31_result_path.is_file():
        errors.append("Gemini-3.1 recovery provenance or result missing")
    else:
        gp, gr = base.read_json(g31_provenance_path), base.read_json(g31_result_path)
        if gp.get("schema_version") != RECOVERY_SCHEMA or gr.get("schema_version") != RECOVERY_SCHEMA:
            errors.append("Gemini-3.1 recovery schema mismatch")
        recovery_runner_sha = base.sha256(RECOVERY_RUNNER)
        if (
            recovery_runner_sha != EXPECTED_RECOVERY_RUNNER_SHA256
            or gp.get("recovery_runner_sha256") != recovery_runner_sha
        ):
            errors.append("Gemini-3.1 recovery runner hash mismatch")
        if gp.get("source_run_provenance_sha256") != base.sha256(provenance_path) or gp.get("source_run_summary_sha256") != base.sha256(summary_path):
            errors.append("Gemini-3.1 recovery source run hashes mismatch")
        failure_path = base.REPO / str(gp.get("source_failure_final"))
        if not failure_path.is_file() or gp.get("source_failure_final_sha256") != base.sha256(failure_path):
            errors.append("Gemini-3.1 recovery source failure hash mismatch")
        source_attempts = gp.get("source_attempts")
        if not isinstance(source_attempts, list) or len(source_attempts) != 7:
            errors.append("Gemini-3.1 recovery does not pin seven original attempts")
        else:
            for entry in source_attempts:
                path = base.REPO / str(entry.get("path"))
                if not path.is_file() or entry.get("sha256") != base.sha256(path):
                    errors.append(f"Gemini-3.1 pinned attempt changed: {path}")
        if (
            gr.get("terminal") is not True
            or identity(gr) != G31_KEY
            or gr.get("solver_route") != base.SOLVER
            or gr.get("judge_input_mode") != "vision"
            or gr.get("status") not in {
                "judged_same_payload_post_exhaustion_network_recovery",
                "judged_same_payload_network_recovery_then_V8_canonicalization",
                "judged_post_exhaustion_unique_evidence_canonicalization",
            }
            or gr.get("source_artifacts_overwritten") is not False
            or not base.finite_number(gr.get("score"))
            or not (0 <= float(gr["score"]) <= float(gr.get("max_score", -1)))
        ):
            errors.append("Gemini-3.1 recovery result identity/terminal/score mismatch")
        item = catalog[("icho2026", "icho2026_t4_4.3")]
        q, s, combined = base.item_image_paths(item)
        if (gr.get("question_image_count"), gr.get("solution_image_count"), gr.get("wire_image_count")) != (len(q), len(s), len(combined)):
            errors.append("Gemini-3.1 recovery wire image counts mismatch")
        recovery_attempts = sorted((g31_root / "online-attempts").glob("attempt-*.json"))
        if len(recovery_attempts) != int(gr.get("recovery_attempt_count", -1)):
            errors.append("Gemini-3.1 recovery attempt accounting mismatch")
        elif not recovery_attempts or base.read_json(recovery_attempts[-1]).get("validation_ok") is not True:
            errors.append("Gemini-3.1 recovery has no strict-valid final online attempt")
        evidence["gemini31_recovery"] = {
            "root": str(g31_root),
            "provenance_sha256": base.sha256(g31_provenance_path),
            "result_sha256": base.sha256(g31_result_path),
            "original_attempts": gr.get("original_attempt_count"),
            "recovery_attempts": gr.get("recovery_attempt_count"),
            "score": gr.get("score"),
            "recovery_runner_sha256": recovery_runner_sha,
        }

    recovery_payload_evidence, recovery_payload_errors = audit_gemini31_recovery_payloads(
        g31_root, catalog
    )
    evidence["gemini31_recovery_payload_binding"] = recovery_payload_evidence
    errors.extend(recovery_payload_errors)

    # Uniform DeepSeek v2 route: thinking remains enabled and the only model
    # option change is the frozen route's supported 65,536-token output cap.
    deep_panel = base.read_json(DEEP_PANEL)
    if (
        deep_panel.get("route") != DEEP_ROUTE
        or deep_panel.get("input_mode") != "text_staticized"
        or deep_panel.get("judge_options") != {"thinking": {"type": "enabled"}}
        or deep_panel.get("judge_max_completion_tokens") != 65536
        or deep_panel.get("repeat_count") != 3
    ):
        errors.append("DeepSeek v2 panel is not the uniform enabled-thinking 65k protocol")
    smoke_provenance = DEEP_SMOKE_ROOT / "run_provenance.json"
    smoke_summary = DEEP_SMOKE_ROOT / "run_summary.json"
    smoke_finals = [
        path for path in DEEP_SMOKE_ROOT.glob("preflight/**/*.json")
        if not any(part.endswith("-attempts") for part in path.parts)
    ]
    if not smoke_provenance.is_file() or not smoke_summary.is_file() or len(smoke_finals) != 2:
        errors.append("DeepSeek v2 two-item smoke artifacts are incomplete")
    else:
        smoke_p, smoke_s = base.read_json(smoke_provenance), base.read_json(smoke_summary)
        if (
            smoke_p.get("schema_version") != DEEP_RUN_SCHEMA
            or smoke_p.get("command") != "smoke"
            or smoke_p.get("judge_options") != {"thinking": {"type": "enabled"}}
            or smoke_p.get("judge_max_completion_tokens") != 65536
            or smoke_s.get("logical_tasks") != 2
            or smoke_s.get("status_counts") != {"judged": 2}
        ):
            errors.append("DeepSeek v2 smoke provenance/summary mismatch")
        smoke_ids: set[str] = set()
        for smoke_final in smoke_finals:
            result = base.read_json(smoke_final)
            attempt_path = smoke_final.parent / (smoke_final.stem + "-attempts") / "attempt-01.json"
            attempt = base.read_json(attempt_path) if attempt_path.is_file() else {}
            smoke_ids.add(str(result.get("item_id")))
            if (
                result.get("terminal") is not True
                or result.get("status") != "judged"
                or result.get("attempt_count") != 1
                or result.get("wire_image_count") != 0
                or attempt.get("validation_ok") is not True
                or attempt.get("request", {}).get("max_tokens") != 65536
                or attempt.get("request", {}).get("thinking") != {"type": "enabled"}
                or len(attempt.get("request_images") or []) != 0
            ):
                errors.append(f"DeepSeek v2 smoke cell failed strict/config/wire validation: {smoke_final}")
        if smoke_ids != {"icho2026_t6_6.2", "icho2026_t9_9.8"}:
            errors.append("DeepSeek v2 smoke item identities changed")
        evidence["deepseek_longthinking_smoke"] = {
            "root": str(DEEP_SMOKE_ROOT),
            "provenance_sha256": base.sha256(smoke_provenance),
            "summary_sha256": base.sha256(smoke_summary),
            "status_counts": smoke_s.get("status_counts"),
            "item_ids": sorted(smoke_ids),
        }
    dp_path, ds_path = deep_root / "run_provenance.json", deep_root / "run_summary.json"
    if not dp_path.is_file() or not ds_path.is_file():
        errors.append("DeepSeek v2 full-run provenance or summary missing")
    else:
        dp, ds = base.read_json(dp_path), base.read_json(ds_path)
        if (
            dp.get("schema_version") != DEEP_RUN_SCHEMA
            or dp.get("command") != "run"
            or dp.get("route") != DEEP_ROUTE
            or dp.get("judge_options") != {"thinking": {"type": "enabled"}}
            or dp.get("judge_max_completion_tokens") != 65536
            or dp.get("panel_sha256") != base.sha256(DEEP_PANEL)
            or dp.get("runner_sha256") != base.sha256(DEEP_RUNNER)
            or dp.get("source_run_provenance_sha256") != base.sha256(provenance_path)
            or dp.get("source_run_summary_sha256") != base.sha256(summary_path)
        ):
            errors.append("DeepSeek v2 provenance mismatch")
        counts = ds.get("status_counts")
        if (
            ds.get("schema_version") != DEEP_RUN_SCHEMA
            or ds.get("command") != "run"
            or ds.get("logical_tasks") != 273
            or not isinstance(counts, dict)
            or any(key not in {"judged", "resumed", "deterministic_zero_model_no_answer"} for key in counts)
            or sum(counts.values()) != 273
        ):
            errors.append("DeepSeek v2 run summary is not 273 successful judgments")
        evidence["deepseek_longthinking_v2"] = {
            "root": str(deep_root),
            "provenance_sha256": base.sha256(dp_path),
            "summary_sha256": base.sha256(ds_path),
            "status_counts": counts,
        }
    if len(final_jsons(deep_root)) != 273:
        errors.append(f"DeepSeek v2 final-file count is {len(final_jsons(deep_root))}, not 273")
    return evidence, errors


def effective_result_path(
    original: Path,
    g31_root: Path,
    deep_root: Path,
    route: str,
    repeat: int,
    contest: str,
    item: dict[str, Any],
) -> tuple[Path, str]:
    key = (route, repeat, contest, str(item["item_id"]))
    if key == G31_KEY:
        return g31_root / "result.json", "gemini31_same_config_recovery"
    if route == DEEP_ROUTE:
        return base.expected_path(deep_root, route, repeat, contest, item), "deepseek_longthinking_v2"
    return base.expected_path(original, route, repeat, contest, item), "original_run"


def load_effective_records(
    original: Path,
    g31_root: Path,
    deep_root: Path,
    panel: dict[str, Any],
    catalog: dict[tuple[str, str], dict[str, Any]],
) -> tuple[dict[tuple[str, int, str, str], dict[str, Any]], list[dict[str, Any]], list[str]]:
    records: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    coverage: list[dict[str, Any]] = []
    errors: list[str] = []
    judges = {str(judge["route"]): judge for judge in panel["judges"]}
    for route, judge in judges.items():
        for repeat in range(1, 4):
            for (contest, item_id), item in catalog.items():
                key = (route, repeat, contest, item_id)
                path, source_kind = effective_result_path(original, g31_root, deep_root, route, repeat, contest, item)
                row = {
                    "judge_route": route,
                    "judge_input_mode": judge["input_mode"],
                    "repeat": repeat,
                    "contest": contest,
                    "item_id": item_id,
                    "source_kind": source_kind,
                    "path": str(path),
                    "state": "missing",
                }
                if not path.is_file():
                    errors.append(f"missing effective judgment: {path}")
                    coverage.append(row)
                    continue
                try:
                    stored = base.read_json(path)
                except Exception as exc:
                    errors.append(f"unreadable effective judgment {path}: {exc}")
                    row["state"] = "invalid_json"
                    coverage.append(row)
                    continue
                data = dict(stored)
                problems: list[str] = []
                if source_kind == "gemini31_same_config_recovery":
                    source_failure = base.expected_path(original, route, repeat, contest, item)
                    failure = base.read_json(source_failure)
                    original_count = int(stored.get("original_attempt_count", -1))
                    recovery_count = int(stored.get("recovery_attempt_count", -1))
                    original_sidecars = list((source_failure.parent / (source_failure.stem + "-attempts")).glob("attempt-*.json"))
                    recovery_sidecars = list((g31_root / "online-attempts").glob("attempt-*.json"))
                    if original_count != len(original_sidecars) or recovery_count != len(recovery_sidecars):
                        problems.append("recovery_attempt_count")
                    if failure.get("terminal") is not False or identity(failure) != key:
                        problems.append("source_failure_identity")
                    data.update({
                        "schema_version": base.RESULT_SCHEMA,
                        "terminal": True,
                        "status": "judged_protocol_recovery",
                        "attempt_count": original_count + recovery_count,
                        "input_fingerprint": failure.get("input_fingerprint"),
                        "protocol_recovery": {
                            "source_failure_path": str(source_failure),
                            "source_failure_sha256": base.sha256(source_failure),
                            "original_attempt_count": original_count,
                            "recovery_attempt_count": recovery_count,
                            "recovery_status": stored.get("status"),
                        },
                    })
                else:
                    sidecars = list((path.parent / (path.stem + "-attempts")).glob("attempt-*.json"))
                    if data.get("attempt_count") != len(sidecars):
                        problems.append("attempt_count")
                wanted = (route, repeat, contest, base.SOLVER, item_id)
                observed_identity = (
                    data.get("judge_route"),
                    data.get("repeat"),
                    data.get("contest"),
                    data.get("solver_route"),
                    data.get("item_id"),
                )
                if data.get("schema_version") != base.RESULT_SCHEMA:
                    problems.append("schema")
                if data.get("terminal") is not True:
                    problems.append("nonterminal")
                if data.get("status") not in {"judged", "deterministic_zero_model_no_answer", "judged_protocol_recovery"}:
                    problems.append("terminal_status")
                if observed_identity != wanted:
                    problems.append("identity")
                if data.get("judge_input_mode") != judge["input_mode"]:
                    problems.append("mode")
                score, maximum = data.get("score"), data.get("max_score")
                if not base.finite_number(score) or not base.finite_number(maximum):
                    problems.append("non_numeric_score")
                elif float(maximum) != float(item["max_score"]) or not 0 <= float(score) <= float(maximum):
                    problems.append("score_range")
                if not isinstance(data.get("input_fingerprint"), str) or len(data["input_fingerprint"]) != 64:
                    problems.append("fingerprint")
                for field in ("source_solver_sha256", "candidate_sha256"):
                    if not isinstance(data.get(field), str) or len(data[field]) != 64:
                        problems.append(field)
                q, s, combined = base.item_image_paths(item)
                expected_wire = len(combined) if judge["input_mode"] == "vision" else 0
                if data.get("question_image_count") != len(q):
                    problems.append("question_image_count")
                if data.get("solution_image_count") != len(s):
                    problems.append("solution_image_count")
                if data.get("wire_image_count") != expected_wire:
                    problems.append("wire_image_count")
                if not isinstance(data.get("attempt_count"), int) or data["attempt_count"] < 0:
                    problems.append("attempt_count_type")
                if problems:
                    errors.append(f"invalid effective judgment {path}: {','.join(problems)}")
                    row["state"] = "invalid:" + ",".join(problems)
                    coverage.append(row)
                    continue
                if key in records:
                    errors.append(f"duplicate effective result key: {key}")
                data["effective_source_kind"] = source_kind
                data["effective_source_path"] = str(path)
                records[key] = data
                row.update({"state": "terminal", "score": score, "max_score": maximum})
                coverage.append(row)
    if len(records) != 1638:
        errors.append(f"effective terminal panel contains {len(records)}, not 1638 records")
    return records, coverage, errors


def audit_final_attempt_bindings(
    records: dict[tuple[str, int, str, str], dict[str, Any]],
    catalog: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Recompute every final fingerprint and replay its strict-valid response."""
    errors: list[str] = []
    rows: list[dict[str, Any]] = []
    loaded_panel = cross.load_panel(PANEL)
    judges = {str(value["route"]): value for value in loaded_panel["judges"]}
    # The replacement route intentionally used the standalone 65k manifest as
    # its exact judge object; the old six-route manifest contains the excluded
    # 16k diagnostic configuration.
    judges[DEEP_ROUTE] = base.read_json(DEEP_PANEL)
    candidate_cache: dict[str, tuple[str, str, str]] = {}

    def candidate_for(data: dict[str, Any]) -> tuple[str, str, str]:
        source_value = str(data.get("source_solver_path") or "")
        source = Path(source_value)
        if not source.is_absolute():
            source = base.REPO / source
        key = str(source.resolve())
        if key not in candidate_cache:
            raw = pipeline.read_regular_file_bytes(source)
            solver = json.loads(raw.decode("utf-8"))
            candidate = str(solver.get("candidate_answer") or "")
            if not candidate:
                raise RuntimeError(f"empty frozen candidate in {source}")
            candidate_cache[key] = (
                candidate,
                pipeline.sha256_bytes(raw),
                pipeline.sha256_bytes(candidate.encode("utf-8")),
            )
        return candidate_cache[key]

    for key, data in sorted(records.items()):
        route, repeat, contest, item_id = key
        item = catalog[(contest, item_id)]
        problems: list[str] = []
        try:
            candidate, source_sha, candidate_sha = candidate_for(data)
        except Exception as exc:
            errors.append(f"cannot reload frozen candidate for {key}: {exc}")
            continue
        if source_sha != data.get("source_solver_sha256"):
            problems.append("source_solver_sha256")
        if candidate_sha != data.get("candidate_sha256"):
            problems.append("candidate_sha256")
        judge = judges[route]
        q = pipeline.resolve_paths(pipeline.question_image_values(item), cross.REPO)
        s = pipeline.resolve_paths(pipeline.reference_image_values(item), cross.REPO)
        images = pipeline.ordered_unique_paths(q, s) if judge["input_mode"] == "vision" else []
        prompt = cross.modality_prompt(item, candidate, judge, len(q), len(s))
        fingerprint_material = {
            "schema": cross.RUNNER_SCHEMA,
            "judge": judge,
            "repeat": repeat,
            "contest": contest,
            "solver_route": base.SOLVER,
            "item_id": item_id,
            "candidate_sha256": candidate_sha,
            "source_sha256": source_sha,
            "prompt": prompt,
            "image_sha256": [pipeline.sha256_file(path) for path in images],
        }
        expected_fingerprint = pipeline.sha256_bytes(
            pipeline.canonical_json_bytes(fingerprint_material)
        )
        fingerprint_match = data.get("input_fingerprint") == expected_fingerprint
        if not fingerprint_match:
            problems.append("input_fingerprint")

        source_kind = str(data.get("effective_source_kind"))
        if source_kind == "gemini31_same_config_recovery":
            attempt_paths = sorted((G31_ROOT / "online-attempts").glob("attempt-*.json"))
        else:
            final_path = Path(str(data["effective_source_path"]))
            attempt_paths = sorted(
                (final_path.parent / (final_path.stem + "-attempts")).glob("attempt-*.json")
            )
        strict_replay_match = False
        selected_attempt: int | None = None
        if data.get("status") == "deterministic_zero_model_no_answer":
            if attempt_paths or data.get("attempt_count") != 0:
                problems.append("deterministic_zero_attempts")
            expected_parsed = {
                "score": 0.0,
                "max_score": float(item["max_score"]),
                "criteria": [],
            }
            strict_replay_match = data.get("parsed") == expected_parsed and float(data["score"]) == 0.0
            if not strict_replay_match:
                problems.append("deterministic_zero_parsed")
        else:
            attempt_objects = [base.read_json(path) for path in attempt_paths]
            ordinals = [value.get("attempt") for value in attempt_objects]
            if ordinals != list(range(1, len(attempt_objects) + 1)):
                problems.append("attempt_ordinals")
            if not attempt_objects:
                problems.append("missing_strict_valid_attempt")
            else:
                last = attempt_objects[-1]
                selected_attempt = int(last.get("attempt", -1))
                stored_response = last.get("raw_response")
                expected_criteria = pipeline.normalized_rubric_criteria(
                    pipeline.grading_rubric_for(item)
                )
                if not isinstance(stored_response, dict):
                    problems.append("last_raw_response")
                else:
                    try:
                        sanitized_request = last.get("request")
                        request_images = last.get("request_images")
                        if not isinstance(sanitized_request, dict) or not isinstance(request_images, list):
                            raise RuntimeError("last attempt request metadata malformed")
                        authenticated_request = pipeline.reconstruct_request_from_sidecar(
                            sanitized_request,
                            str(last.get("request_sha256")),
                            request_images,
                        )
                        response = pipeline.restore_provider_response(
                            stored_response,
                            authenticated_request,
                            last.get("raw_response_storage"),
                        )
                        ok, reason, parsed = pipeline.validate_judge_response(
                            response,
                            candidate,
                            float(item["max_score"]),
                            expected_criteria=expected_criteria,
                        )
                        if last.get("validation_ok") is not True or not ok or parsed is None:
                            problems.append(f"last_attempt_not_strict_valid:{reason}")
                        else:
                            strict_replay_match = (
                                pipeline.canonical_json_bytes(parsed)
                                == pipeline.canonical_json_bytes(data.get("parsed"))
                                and float(parsed["score"]) == float(data["score"])
                                and float(parsed["max_score"]) == float(data["max_score"])
                            )
                            if not strict_replay_match:
                                problems.append("strict_replay_final_parsed_score")
                    except Exception as exc:
                        problems.append(f"last_attempt_authenticated_replay:{exc}")
        row = {
            "judge_route": route,
            "judge_input_mode": judge["input_mode"],
            "repeat": repeat,
            "contest": contest,
            "item_id": item_id,
            "effective_source_kind": source_kind,
            "effective_source_path": data.get("effective_source_path"),
            "source_solver_sha256_match": source_sha == data.get("source_solver_sha256"),
            "candidate_sha256_match": candidate_sha == data.get("candidate_sha256"),
            "input_fingerprint_match": fingerprint_match,
            "selected_strict_valid_attempt_ordinal": selected_attempt,
            "strict_replay_matches_final_parsed_and_score": strict_replay_match,
            "status": "pass" if not problems else "fail:" + ",".join(problems),
        }
        rows.append(row)
        if problems:
            errors.append(f"final/attempt binding failed for {key}: {','.join(problems)}")
    if len(rows) != 1638:
        errors.append(f"final/attempt binding audited {len(rows)} records, not 1638")
    return rows, errors


def scan_attempt_ledgers(
    original: Path,
    g31_root: Path,
    deep_root: Path,
    panel: dict[str, Any],
    catalog: dict[tuple[str, str], dict[str, Any]],
    records: dict[tuple[str, int, str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    prices = {str(judge["route"]): judge for judge in panel["judges"]}
    primary: list[dict[str, Any]] = []
    diagnostic: list[dict[str, Any]] = []
    errors: list[str] = []
    file_hashes: dict[str, tuple[str, int]] = {}
    wire: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    initial_requests: dict[tuple[str, str, str], list[str]] = defaultdict(list)

    def consume(
        *,
        route: str,
        path: Path,
        contest: str,
        item_id: str,
        ledger_role: str,
        include_primary: bool,
        initial_eligible: bool,
    ) -> None:
        judge = prices[route]
        item = catalog.get((contest, item_id))
        if item is None:
            errors.append(f"attempt outside frozen catalog: {path}")
            return
        try:
            raw = base.read_json(path)
        except Exception as exc:
            errors.append(f"unreadable attempt {path}: {exc}")
            return
        usage = raw.get("usage")
        if usage is None and isinstance(raw.get("raw_response"), dict):
            usage = raw["raw_response"].get("usage")
        if usage is None and isinstance(raw.get("response"), dict):
            usage = raw["response"].get("usage")
        inp, out = base.usage_tokens(usage)
        ordinal, request_digest = raw.get("attempt"), raw.get("request_sha256")
        entry = {
            "judge_route": route,
            "judge_input_mode": judge["input_mode"],
            "contest": contest,
            "item_id": item_id,
            "attempt_ordinal": ordinal,
            "attempt_path": str(path),
            "ledger_role": ledger_role,
            "included_in_primary_scores_and_cost": include_primary,
            "request_sha256": request_digest,
            "input_tokens": inp,
            "output_tokens": out,
            "usage_available": inp is not None and out is not None,
            "input_cny": None if inp is None else inp * float(judge["input_cny_per_million"]) / 1_000_000,
            "output_cny": None if out is None else out * float(judge["output_cny_per_million"]) / 1_000_000,
        }
        (primary if include_primary else diagnostic).append(entry)
        if initial_eligible and ordinal == 1:
            if not isinstance(request_digest, str) or len(request_digest) != 64:
                errors.append(f"invalid initial request SHA: {path}")
            else:
                initial_requests[(route, contest, item_id)].append(request_digest)
        q, s, combined = base.item_image_paths(item)
        expected = combined if judge["input_mode"] == "vision" else []
        observed = raw.get("request_images")
        if not isinstance(observed, list):
            errors.append(f"attempt request_images is not a list: {path}")
            return
        observed_paths = [str(meta.get("path")) for meta in observed if isinstance(meta, dict)]
        if observed_paths != expected:
            errors.append(f"attempt official image path/order mismatch: {path}")
            return
        try:
            request = raw.get("request")
            if not isinstance(request, dict):
                raise RuntimeError("sanitized request is not an object")
            reconstructed = pipeline.reconstruct_request_from_sidecar(
                request,
                str(request_digest),
                observed,
            )
            reproved = pipeline.request_image_proofs(
                reconstructed,
                [Path(value) for value in expected],
            )
            if (
                pipeline.canonical_json_bytes(reproved)
                != pipeline.canonical_json_bytes(observed)
            ):
                raise RuntimeError("reconstructed global image proof/order differs from sidecar")
        except Exception as exc:
            errors.append(f"attempt authenticated request reconstruction failed: {path}: {exc}")
            return
        if include_primary:
            wire[route]["attempt_files"] += 1
            wire[route]["expected_image_instances"] += len(expected)
            wire[route]["observed_image_instances"] += len(observed)
            wire[route]["authenticated_request_reconstructions"] += 1
            wire[route]["global_image_order_matches"] += 1
            wire[route]["data_url_proof_matches"] += len(reproved)
            if not expected:
                wire[route]["zero_image_attempts"] += 1
                wire[route]["zero_image_request_sha_matches"] += 1
            if ledger_role == "gemini31_post_exhaustion_recovery":
                wire[route]["protocol_recovery_attempts"] += 1
        for expected_path, meta in zip(expected, observed):
            if not isinstance(meta, dict):
                errors.append(f"attempt image metadata is not an object: {path}")
                continue
            source = Path(expected_path)
            if expected_path not in file_hashes:
                if not source.is_file():
                    errors.append(f"missing official PNG source: {source}")
                    continue
                file_hashes[expected_path] = (base.sha256(source), source.stat().st_size)
            digest, byte_count = file_hashes[expected_path]
            if (
                source.suffix.lower() != ".png"
                or meta.get("mime_type") != "image/png"
                or meta.get("sha256") != digest
                or meta.get("bytes") != byte_count
                or not isinstance(meta.get("request_json_path"), list)
            ):
                errors.append(f"attempt official PNG byte/SHA proof mismatch: {path}::{source}")
            elif include_primary:
                wire[route]["source_png_hash_matches"] += 1

    def scan_standard_tree(root: Path, route: str, role: str, include_primary: bool) -> None:
        folder = root / "judgments" / base.safe(route)
        for path in sorted(folder.glob("**/*-attempts/attempt-*.json")) if folder.exists() else []:
            parts = path.relative_to(folder).parts
            if len(parts) < 6 or not parts[-2].endswith("-attempts"):
                errors.append(f"unparseable attempt path: {path}")
                continue
            contest = parts[1]
            item_id = parts[-2][:-len("-attempts")]
            consume(
                route=route,
                path=path,
                contest=contest,
                item_id=item_id,
                ledger_role=role,
                include_primary=include_primary,
                initial_eligible=include_primary,
            )

    for route in prices:
        if route == DEEP_ROUTE:
            scan_standard_tree(deep_root, route, "deepseek_longthinking_v2", True)
        else:
            scan_standard_tree(original, route, "original_run", True)
    # The two same-config recovery calls are additional primary costs and wire
    # evidence, but they are not a fourth independent repeat/initial request.
    for path in sorted((g31_root / "online-attempts").glob("attempt-*.json")):
        consume(
            route=G31_ROUTE,
            path=path,
            contest="icho2026",
            item_id="icho2026_t4_4.3",
            ledger_role="gemini31_post_exhaustion_recovery",
            include_primary=True,
            initial_eligible=False,
        )
    # Old DeepSeek 16k attempts are a separate failure-diagnostic cost ledger.
    scan_standard_tree(original, DEEP_ROUTE, "deepseek_16k_failure_diagnostic_excluded", False)

    wire_rows: list[dict[str, Any]] = []
    for route, judge in prices.items():
        groups = [digests for (judge_route, _, _), digests in initial_requests.items() if judge_route == route]
        identical = sum(len(digests) == 3 and len(set(digests)) == 1 for digests in groups)
        if len(groups) != 91 or identical != 91:
            errors.append(f"initial request is not identical across three repeats for all cases: {route}")
        counts = wire[route]
        expected_attempts = sum(data["attempt_count"] for (judge_route, _, _, _), data in records.items() if judge_route == route)
        observed_attempts = sum(entry["judge_route"] == route for entry in primary)
        if observed_attempts != expected_attempts:
            errors.append(f"effective attempt accounting mismatch for {route}: {observed_attempts}!={expected_attempts}")
        wire_rows.append({
            "judge_route": route,
            "judge_input_mode": judge["input_mode"],
            "attempt_files": counts["attempt_files"],
            "expected_image_instances": counts["expected_image_instances"],
            "observed_image_instances": counts["observed_image_instances"],
            "source_png_hash_matches": counts["source_png_hash_matches"],
            "data_url_proof_matches": counts["data_url_proof_matches"],
            "authenticated_request_reconstructions": counts["authenticated_request_reconstructions"],
            "global_image_order_matches": counts["global_image_order_matches"],
            "zero_image_attempts": counts["zero_image_attempts"],
            "zero_image_request_sha_matches": counts["zero_image_request_sha_matches"],
            "protocol_recovery_attempts": counts["protocol_recovery_attempts"],
            "initial_request_case_groups": len(groups),
            "initial_request_groups_identical_across_three_repeats": identical,
            "wire_contract": "direct_original_official_png" if judge["input_mode"] == "vision" else "zero_image_text_staticized",
        })
    return primary, diagnostic, wire_rows, errors


def summarize_cost(entries: list[dict[str, Any]], panel: dict[str, Any], records: dict[tuple[str, int, str, str], dict[str, Any]] | None) -> list[dict[str, Any]]:
    routes = [str(judge["route"]) for judge in panel["judges"]]
    modes = {str(judge["route"]): str(judge["input_mode"]) for judge in panel["judges"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        grouped[str(entry["judge_route"])].append(entry)
    rows: list[dict[str, Any]] = []
    for route in routes:
        values = grouped[route]
        if records is not None:
            route_records = [data for (judge_route, _, _, _), data in records.items() if judge_route == route]
            attempt_counts = [int(data["attempt_count"]) for data in route_records]
            logical = len(route_records)
        else:
            attempt_counts = []
            logical = None
        known = [entry for entry in values if entry["usage_available"]]
        rows.append({
            "judge_route": route,
            "judge_input_mode": modes[route],
            "logical_judgments": logical,
            "successful_terminal_judgments": logical,
            "failed_terminal_judgments": 0 if logical is not None else None,
            "attempt_files": len(values),
            "extra_retry_attempts": None if logical is None else len(values) - logical,
            "judgments_requiring_retry": None if records is None else sum(value > 1 for value in attempt_counts),
            "maximum_attempts_for_one_judgment": None if not attempt_counts else max(attempt_counts),
            "attempts_with_complete_usage": len(known),
            "input_tokens": sum(int(entry["input_tokens"]) for entry in known),
            "output_tokens": sum(int(entry["output_tokens"]) for entry in known),
            "actual_cost_cny_for_usage_known_attempts": sum(float(entry["input_cny"]) + float(entry["output_cny"]) for entry in known),
            "cost_complete": len(values) == len(known),
        })
    return rows


def compute_analysis(
    panel: dict[str, Any],
    catalog: dict[tuple[str, str], dict[str, Any]],
    records: dict[tuple[str, int, str, str], dict[str, Any]],
    primary_attempts: list[dict[str, Any]],
    diagnostic_attempts: list[dict[str, Any]],
    wire_rows: list[dict[str, Any]],
    source_evidence: dict[str, Any],
    coverage: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    routes = [str(judge["route"]) for judge in panel["judges"]]
    modes = {str(judge["route"]): str(judge["input_mode"]) for judge in panel["judges"]}
    normalized = {
        (route, repeat, contest, item_id): float(data["score"]) / float(data["max_score"])
        for (route, repeat, contest, item_id), data in records.items()
    }
    means: dict[tuple[str, str], float] = {}
    case_rows: list[dict[str, Any]] = []
    repeat_rows: list[dict[str, Any]] = []
    baseline, baseline_errors = base.baseline_scores(catalog)
    if baseline_errors:
        raise RuntimeError("; ".join(baseline_errors))
    for route in routes:
        for (contest, item_id), item in sorted(catalog.items()):
            values = [normalized[(route, repeat, contest, item_id)] for repeat in range(1, 4)]
            score_mean = base.mean(values)
            means[(route, item_id)] = score_mean
            pair_mae = base.mean(abs(values[left] - values[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
            case_rows.append({
                "judge_route": route,
                "judge_input_mode": modes[route],
                "contest": contest,
                "item_id": item_id,
                "sequence_id": item["sequence_id"],
                "max_score": item["max_score"],
                "repeat_1_normalized": values[0],
                "repeat_2_normalized": values[1],
                "repeat_3_normalized": values[2],
                "repeat_mean_normalized": score_mean,
                "repeat_population_sd": statistics.pstdev(values),
                "repeat_pairwise_mae": pair_mae,
                "baseline_gemini35_normalized": baseline[(contest, item_id)],
                "delta_vs_baseline": score_mean - baseline[(contest, item_id)],
            })
        for subset in ("overall", "ipho2026", "icho2026"):
            selected = [
                (contest, item_id) for contest, item_id in catalog
                if subset == "overall" or contest == subset
            ]
            triplets = [
                [normalized[(route, repeat, contest, item_id)] for repeat in range(1, 4)]
                for contest, item_id in selected
            ]
            repeat_rows.append({
                "judge_route": route,
                "judge_input_mode": modes[route],
                "subset": subset,
                "cases": len(triplets),
                "mean_pairwise_repeat_mae_normalized": base.mean(
                    base.mean(abs(values[left] - values[right]) for left, right in ((0, 1), (0, 2), (1, 2)))
                    for values in triplets
                ),
                "mean_case_repeat_population_sd_normalized": base.mean(statistics.pstdev(values) for values in triplets),
                "exact_three_repeat_agreement_rate": sum(values[0] == values[1] == values[2] for values in triplets) / len(triplets),
                "all_repeat_score_population_sd_normalized": statistics.pstdev(value for values in triplets for value in values),
            })

    pairwise_rows: list[dict[str, Any]] = []
    for index, left in enumerate(routes):
        for right in routes[index + 1:]:
            for subset in ("overall", "ipho2026", "icho2026"):
                keys = [(contest, item_id) for contest, item_id in catalog if subset == "overall" or contest == subset]
                x = [means[(left, item_id)] for _, item_id in keys]
                y = [means[(right, item_id)] for _, item_id in keys]
                pairwise_rows.append({
                    "judge_a": left,
                    "judge_b": right,
                    "judge_a_mode": modes[left],
                    "judge_b_mode": modes[right],
                    "subset": subset,
                    "cases": len(keys),
                    "mean_absolute_error_normalized": base.mean(abs(a - b) for a, b in zip(x, y)),
                    **base.correlation_metrics(x, y),
                    "interpretation_scope": "vision_judge_agreement" if modes[left] == modes[right] == "vision" else "modality_sensitivity_only",
                })
    baseline_rows: list[dict[str, Any]] = []
    for route in routes:
        for subset in ("overall", "ipho2026", "icho2026"):
            keys = [(contest, item_id) for contest, item_id in catalog if subset == "overall" or contest == subset]
            x = [means[(route, item_id)] for _, item_id in keys]
            b = [baseline[(contest, item_id)] for contest, item_id in keys]
            baseline_rows.append({
                "judge_route": route,
                "judge_input_mode": modes[route],
                "subset": subset,
                "cases": len(keys),
                "signed_delta_normalized": base.mean(value - reference for value, reference in zip(x, b)),
                "mean_absolute_delta_normalized": base.mean(abs(value - reference) for value, reference in zip(x, b)),
                **base.correlation_metrics(x, b),
                "baseline": "frozen production Gemini-3.5 vision judge",
            })
    aggregate_rows = base.score_aggregates(routes, means, catalog)
    repeat_total_rows, repeat_total_summary = base.repeat_contest_totals(routes, normalized, catalog)
    primary_cost = summarize_cost(primary_attempts, panel, records)
    diagnostic_cost = summarize_cost(diagnostic_attempts, panel, None)
    diagnostic_cost = [row for row in diagnostic_cost if row["judge_route"] == DEEP_ROUTE]
    primary_cost_total = sum(float(row["actual_cost_cny_for_usage_known_attempts"]) for row in primary_cost)
    diagnostic_cost_total = sum(float(row["actual_cost_cny_for_usage_known_attempts"]) for row in diagnostic_cost)
    analysis = {
        "schema_version": "science_arena_rebuttal_cross_judge_composite_analysis_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "scope": {
            "solver_route": base.SOLVER,
            "frozen_answers": 91,
            "judges": 6,
            "repeats": 3,
            "effective_terminal_judgments": 1638,
            "limitation": "One frozen solver route only: no 14-model ranking correlation and no cross-candidate robustness inference.",
        },
        "source_evidence": source_evidence,
        "protocol_recovery": {
            "gemini31": "One cell uses a separate same-model, same-payload/options post-exhaustion strict recovery ledger (two additional calls); the original failure and seven attempts remain immutable.",
            "deepseek": "The complete 273-cell DeepSeek route uses thinking enabled with max_tokens=65536. The complete old 16384-token route is excluded from all scores because six long-item cells exhausted reasoning with empty visible content.",
            "mixed_cell_policy": "No old DeepSeek score is mixed into the effective route.",
        },
        "wire_modality_audit": wire_rows,
        "repeat_reliability": repeat_rows,
        "vision_pairwise_and_modality_sensitivity": pairwise_rows,
        "baseline_comparison": baseline_rows,
        "cost_and_tokens": primary_cost,
        "diagnostic_cost_not_used_for_scores": diagnostic_cost,
        "known_usage_cost_totals_cny": {
            "effective_panel_and_recovery": primary_cost_total,
            "excluded_deepseek_16k_diagnostic": diagnostic_cost_total,
            "all_known_usage": primary_cost_total + diagnostic_cost_total,
        },
        "contest_aggregates_from_item_repeat_means": aggregate_rows,
        "contest_repeat_totals": repeat_total_rows,
        "contest_total_mean_sample_sd": repeat_total_summary,
        "interpretation": {
            "vision_primary": "Only vision-vs-vision rows are primary full-fidelity judge agreement results.",
            "text_only": "Rows involving Qwen or DeepSeek are text-staticized modality-sensitivity results, not evidence against the full-fidelity vision pipeline.",
            "validity": "Repeat and pairwise agreement measure reliability/concordance, not validity. Validity requires the separate human-calibration evidence.",
            "candidate_and_images": "Candidate answers are text. Multimodal requirements come from official question and solution/marking PNGs, passed directly to the four vision judges; text-staticized judges receive zero image blocks.",
            "contest_uncertainty": "Each repeat is aggregated independently; mean and sample SD are computed over the three contest totals (IPhO /30, IChO /60).",
        },
    }
    tables = {
        "coverage": coverage,
        "case_scores": case_rows,
        "repeat_reliability": repeat_rows,
        "pairwise_metrics": pairwise_rows,
        "baseline_comparison": baseline_rows,
        "cost_and_tokens": primary_cost,
        "diagnostic_cost_not_used_for_scores": diagnostic_cost,
        "contest_aggregates": aggregate_rows,
        "contest_repeat_totals": repeat_total_rows,
        "contest_total_mean_sample_sd": repeat_total_summary,
        "wire_modality_audit": wire_rows,
        "attempt_usage": primary_attempts,
        "diagnostic_attempt_usage": diagnostic_attempts,
    }
    return analysis, tables


def write_outputs(out: Path, analysis: dict[str, Any], tables: dict[str, list[dict[str, Any]]], candidate_rows: list[dict[str, Any]]) -> None:
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing analysis directory: {out}")
    out.mkdir(parents=False)

    def dump(path: Path, value: Any) -> None:
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
        fields = sorted({key for row in rows for key in row})
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    dump(out / "summary.json", analysis)
    dump(
        out / "retrospective_pipeline_provenance_addendum.json",
        analysis["source_evidence"]["retrospective_pipeline_provenance_addendum"],
    )
    for name, rows in tables.items():
        if name in {"attempt_usage", "diagnostic_attempt_usage"}:
            dump(out / f"{name}.json", rows)
        else:
            csv_write(out / f"{name}.csv", rows)
    csv_write(out / "frozen_candidate_audit.csv", candidate_rows)
    overall_repeat = [row for row in analysis["repeat_reliability"] if row["subset"] == "overall"]
    vision = [
        row for row in analysis["vision_pairwise_and_modality_sensitivity"]
        if row["subset"] == "overall" and row["interpretation_scope"] == "vision_judge_agreement"
    ]
    totals = analysis["contest_total_mean_sample_sd"]
    lines = [
        "# Six-judge frozen-answer composite replay",
        "",
        "Status: complete (1,638 effective terminal judgments; source and recovery ledgers passed fail-closed validation).",
        "",
        "This covers all 91 items for one frozen Gemini-3.5 Flash solver answer corpus. It is not a 14-model ranking experiment and cannot establish cross-candidate robustness. Agreement measures reliability, not validity.",
        "",
        "Four vision judges receive the original official question and solution/marking PNGs directly. Qwen and DeepSeek receive zero image blocks and are reported only as modality sensitivity.",
        "",
        "## Repeat reliability",
        "",
    ]
    lines += [
        f"- {row['judge_route']}: normalized repeat MAE {row['mean_pairwise_repeat_mae_normalized']:.4f}; mean case SD {row['mean_case_repeat_population_sd_normalized']:.4f}; exact 3-way {row['exact_three_repeat_agreement_rate']:.1%}."
        for row in overall_repeat
    ]
    lines += ["", "## Vision-judge agreement (91 item repeat means)", ""]
    lines += [
        f"- {row['judge_a']} vs {row['judge_b']}: MAE {row['mean_absolute_error_normalized']:.4f}; Pearson {row['pearson']:.4f}; Spearman {row['spearman']:.4f}; Kendall tau-b {row['kendall_tau_b']:.4f}."
        for row in vision
    ]
    lines += ["", "## Contest totals (mean ± sample SD across three repeats)", ""]
    lines += [
        f"- {row['judge_route']} — {row['contest']}: {row['mean_total']:.4f} ± {row['sample_sd_total']:.4f} /{row['reported_scale']:.0f} (repeats {row['repeat_1_total']:.4f}, {row['repeat_2_total']:.4f}, {row['repeat_3_total']:.4f})."
        for row in totals
    ]
    lines += [
        "",
        "## Protocol recovery",
        "",
        "One Gemini-3.1 cell required a same-configuration post-exhaustion recovery. DeepSeek is represented only by a complete uniform thinking-enabled 65k-token rerun; all old 16k DeepSeek scores are excluded and its spend is reported separately as failure-diagnostic cost.",
        "",
    ]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    artifact_hashes = {
        path.name: base.sha256(path)
        for path in sorted(out.iterdir())
        if path.is_file()
    }
    dump(out / "artifact_hashes.json", artifact_hashes)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("check", "analyze"))
    parser.add_argument("--original-root", type=Path, default=ORIGINAL_ROOT)
    parser.add_argument("--g31-root", type=Path, default=G31_ROOT)
    parser.add_argument("--deep-root", type=Path, default=DEEP_ROOT)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    original, g31_root, deep_root = args.original_root.resolve(), args.g31_root.resolve(), args.deep_root.resolve()
    try:
        panel, cases = base.read_json(PANEL), base.read_json(CASES)
        catalog, _ = base.item_catalog()
        source_evidence, errors = validate_sources(original, g31_root, deep_root, panel, catalog)
        retrospective, retrospective_errors = retrospective_pipeline_provenance(
            original, g31_root, deep_root
        )
        source_evidence["retrospective_pipeline_provenance_addendum"] = retrospective
        errors.extend(retrospective_errors)
        records, coverage, record_errors = load_effective_records(original, g31_root, deep_root, panel, catalog)
        binding_rows, binding_errors = audit_final_attempt_bindings(records, catalog)
        candidate_rows, candidate_errors = base.frozen_candidate_audit(records, panel, catalog)
        primary_attempts, diagnostic_attempts, wire_rows, attempt_errors = scan_attempt_ledgers(
            original, g31_root, deep_root, panel, catalog, records
        )
        errors.extend(record_errors)
        errors.extend(binding_errors)
        errors.extend(candidate_errors)
        errors.extend(attempt_errors)
    except Exception as exc:
        print(json.dumps({"status": "incomplete_or_invalid", "fatal": [str(exc)]}, indent=2), file=__import__("sys").stderr)
        return 2
    check = {
        "schema_version": "science_arena_rebuttal_cross_judge_composite_check_v2",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "expected_effective_terminal_judgments": 1638,
        "validated_effective_terminal_judgments": len(records),
        "validated_final_fingerprint_and_strict_replay_bindings": sum(
            row["status"] == "pass" for row in binding_rows
        ),
        "primary_attempt_files": len(primary_attempts),
        "excluded_diagnostic_attempt_files": len(diagnostic_attempts),
        "fileprovider_exact_materializations": len(_FILEPROVIDER_MATERIALIZATIONS),
        "fileprovider_materialization_scope": (
            "Residency-only exact-file downloads; every complete byte stream is "
            "hash-checked and its logical size/mtime must remain unchanged."
        ),
        "status": "complete" if not errors and len(records) == 1638 else "incomplete_or_invalid",
        "errors": errors,
    }
    if args.command == "check" or check["status"] != "complete":
        print(json.dumps(check, indent=2, ensure_ascii=False))
        return 0 if check["status"] == "complete" else 2
    try:
        analysis, tables = compute_analysis(panel, catalog, records, primary_attempts, diagnostic_attempts, wire_rows, source_evidence, coverage)
        analysis["coverage"] = check
        analysis["final_fingerprint_and_strict_replay_binding"] = {
            "records": len(binding_rows),
            "passed": sum(row["status"] == "pass" for row in binding_rows),
            "contract": (
                "Each final input fingerprint is recomputed from frozen candidate, exact "
                "judge configuration, prompt and ordered image SHA list; the final strict-valid "
                "raw response is replayed through the frozen validator and must exactly match "
                "the stored parsed criteria and score."
            ),
        }
        tables["final_fingerprint_and_strict_replay_binding"] = binding_rows
        materialization_rows = [
            _FILEPROVIDER_MATERIALIZATIONS[key]
            for key in sorted(_FILEPROVIDER_MATERIALIZATIONS)
        ]
        analysis["fileprovider_materialization_audit"] = {
            "files_materialized_on_demand": len(materialization_rows),
            "exact_file_download_requests": sum(
                int(row["exact_file_download_requests"])
                for row in materialization_rows
            ),
            "interpretation": (
                "FileProvider materialization changes local residency only. The analyzer "
                "does not write source artifacts, and verifies complete bytes, SHA-256, "
                "logical size and logical mtime before using them."
            ),
        }
        tables["fileprovider_materialization_audit"] = materialization_rows
        output = (args.output_dir or (original / "analysis_composite_longthinking_v2")).resolve()
        write_outputs(output, analysis, tables, candidate_rows)
    except Exception as exc:
        print(json.dumps({**check, "status": "analysis_failed", "fatal": [str(exc)]}, indent=2), file=__import__("sys").stderr)
        return 2
    print(json.dumps({"status": "complete", "analysis_dir": str(output), "terminal_judgments": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
