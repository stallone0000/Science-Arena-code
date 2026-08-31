#!/usr/bin/env python3
"""Fail-closed analysis for the frozen Gemini-3.5 answer / six-judge replay.

``check`` is read-only and is safe while the runner is active.  ``analyze``
refuses to write an analysis unless every expected terminal judgment is
present and internally consistent.  It has no third-party dependencies.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_ROOT = HERE / "runs/ipho_icho2026_gemini35solver_crossjudge_6x3_20260830"
PANEL = HERE / "manifests/rebuttal_judges_6x3_20260830.json"
CASES = HERE / "manifests/rebuttal_crossjudge_gemini35_all91_20260830.json"
ITEMS = {
    "ipho2026": HERE / "staticizations/qonly_gemini31_reviewed_frozen_final_v2_20260829/datasets/ipho2026/items.jsonl",
    "icho2026": HERE / "staticizations/qonly_gemini31_reviewed_frozen_final_v2_20260829/datasets/icho2026_theory/items.jsonl",
}
BASELINE_ROOT = {
    "ipho2026": HERE / "runs/ipho2026_table6_formal_t1c1_corrected_20260828",
    "icho2026": HERE / "runs/icho2026_table6_formal_20260828",
}
RESULT_SCHEMA = "science_arena_rebuttal_cross_judge_result_v1"
RUN_SCHEMA = "science_arena_rebuttal_cross_judge_run_v1"
SOLVER = "google/gemini-3.5-flash"


def safe(s: str) -> str:
    return s.replace("/", "__")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON object required: {path}")
    return data


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError(f"non-object JSONL row {n}: {path}")
                rows.append(row)
    return rows


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def mean(xs: Iterable[float]) -> float:
    ys = list(xs)
    return sum(ys) / len(ys) if ys else float("nan")


def pearson(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    mx, my = mean(x), mean(y)
    dx = sum((a - mx) ** 2 for a in x)
    dy = sum((b - my) ** 2 for b in y)
    if dx <= 0 or dy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(x, y)) / math.sqrt(dx * dy)


def average_ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    result = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and xs[order[j]] == xs[order[i]]:
            j += 1
        rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            result[order[k]] = rank
        i = j
    return result


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(average_ranks(x), average_ranks(y))


def kendall_tau_b(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 2:
        return None
    concordant = discordant = tx = ty = 0
    for i in range(len(x)):
        for j in range(i + 1, len(x)):
            sx = (x[i] > x[j]) - (x[i] < x[j])
            sy = (y[i] > y[j]) - (y[i] < y[j])
            if sx == 0 and sy == 0:
                continue
            if sx == 0:
                tx += 1
            elif sy == 0:
                ty += 1
            elif sx == sy:
                concordant += 1
            else:
                discordant += 1
    denom = math.sqrt((concordant + discordant + tx) * (concordant + discordant + ty))
    return None if denom == 0 else (concordant - discordant) / denom


def correlation_metrics(x: list[float], y: list[float]) -> dict[str, float | None]:
    return {"pearson": pearson(x, y), "spearman": spearman(x, y), "kendall_tau_b": kendall_tau_b(x, y)}


def usage_tokens(usage: Any) -> tuple[int | None, int | None]:
    if not isinstance(usage, dict):
        return None, None
    inp = usage.get("prompt_tokens", usage.get("input_tokens"))
    out = usage.get("completion_tokens", usage.get("output_tokens"))
    return (int(inp) if finite_number(inp) else None, int(out) if finite_number(out) else None)


def item_image_paths(item: dict[str, Any]) -> tuple[list[str], list[str], list[str]]:
    """Return frozen question, official-reference, and ordered-unique paths."""
    q_values = item.get("question_image_paths")
    if q_values is None and isinstance(item.get("visual_dependency"), dict):
        q_values = item["visual_dependency"].get("question_image_paths")
    s_values = item.get("official_solution_image_paths")

    def absolute(values: Any) -> list[str]:
        result: list[str] = []
        for value in values or []:
            path = Path(str(value))
            if not path.is_absolute():
                path = REPO / path
            result.append(os.path.abspath(os.fspath(path)))
        return result

    questions, solutions = absolute(q_values), absolute(s_values)
    combined: list[str] = []
    seen: set[str] = set()
    for path in questions + solutions:
        if path not in seen:
            combined.append(path)
            seen.add(path)
    return questions, solutions, combined


def item_catalog() -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    by_contest: dict[str, list[dict[str, Any]]] = {}
    for contest, path in ITEMS.items():
        rows = read_jsonl(path)
        by_contest[contest] = rows
        for item in rows:
            key = (contest, str(item.get("item_id")))
            if key in by_key or not finite_number(item.get("max_score")):
                raise ValueError(f"invalid or duplicate frozen item: {key}")
            by_key[key] = item
    if len(by_contest.get("ipho2026", [])) != 23 or len(by_contest.get("icho2026", [])) != 68:
        raise ValueError("frozen item counts are not 23 IPhO + 68 IChO")
    return by_key, by_contest


def expected_path(root: Path, judge: str, repeat: int, contest: str, item: dict[str, Any]) -> Path:
    return root / "judgments" / safe(judge) / f"repeat-{repeat:02d}" / contest / safe(SOLVER) / safe(str(item["sequence_id"])) / f"{safe(str(item['item_id']))}.json"


def validate_provenance(root: Path, panel: dict[str, Any], cases: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    path = root / "run_provenance.json"
    if not path.is_file():
        return [f"missing run provenance: {path}"]
    try:
        p = read_json(path)
    except Exception as e:
        return [f"unreadable run provenance: {e}"]
    if p.get("schema_version") != RUN_SCHEMA:
        errors.append("run provenance schema mismatch")
    expected = {
        "panel_sha256": sha256(PANEL), "cases_sha256": sha256(CASES),
        "repeat_count": 3,
    }
    for key, value in expected.items():
        if p.get(key) != value:
            errors.append(f"run provenance {key} mismatch")
    if panel.get("repeat_count") != 3 or len(panel.get("judges") or []) != 6:
        errors.append("judge panel is not exactly six judges x three repeats")
    if cases.get("solver_route") != SOLVER or cases.get("total_item_answers") != 91:
        errors.append("case manifest no longer specifies exact 91-answer Gemini-3.5 corpus")
    for contest, spec in {str(x.get("competition_key")): x for x in cases.get("competitions") or []}.items():
        if contest not in ITEMS or spec.get("items_sha256") != sha256(ITEMS[contest]):
            errors.append(f"case manifest items hash mismatch: {contest}")
    return errors


def validate_run_summary(root: Path, expected: int) -> tuple[dict[str, Any] | None, list[str]]:
    path = root / "run_summary.json"
    if not path.is_file():
        return None, [f"missing completed run summary: {path}"]
    try:
        summary = read_json(path)
    except Exception as e:
        return None, [f"unreadable run summary: {e}"]
    errors: list[str] = []
    counts = summary.get("status_counts")
    if summary.get("schema_version") != RUN_SCHEMA:
        errors.append("run summary schema mismatch")
    if summary.get("command") != "run" or summary.get("logical_tasks") != expected:
        errors.append("run summary command/logical task mismatch")
    if (
        not isinstance(counts, dict)
        or any(key not in {"judged", "resumed", "deterministic_zero_model_no_answer"} for key in counts)
        or any(not isinstance(value, int) or isinstance(value, bool) or value < 0 for value in counts.values())
        or sum(counts.values()) != expected
    ):
        errors.append("run summary status accounting is not complete and successful")
    return summary, errors


def terminal_records(root: Path, panel: dict[str, Any], catalog: dict[tuple[str, str], dict[str, Any]]) -> tuple[dict[tuple[str, int, str, str], dict[str, Any]], list[str], list[dict[str, Any]]]:
    records: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    errors: list[str] = []
    coverage: list[dict[str, Any]] = []
    judges = {str(j["route"]): j for j in panel["judges"]}
    for route, judge in judges.items():
        for repeat in range(1, 4):
            for (contest, item_id), item in catalog.items():
                path = expected_path(root, route, repeat, contest, item)
                row = {"judge_route": route, "judge_input_mode": judge["input_mode"], "repeat": repeat, "contest": contest, "item_id": item_id, "path": str(path.relative_to(root)), "state": "missing"}
                if not path.is_file():
                    errors.append(f"missing judgment: {row['path']}")
                    coverage.append(row); continue
                try:
                    data = read_json(path)
                except Exception as e:
                    errors.append(f"unreadable judgment {row['path']}: {e}")
                    row["state"] = "invalid_json"; coverage.append(row); continue
                identity = (data.get("judge_route"), data.get("repeat"), data.get("contest"), data.get("solver_route"), data.get("item_id"))
                wanted = (route, repeat, contest, SOLVER, item_id)
                problems: list[str] = []
                if data.get("schema_version") != RESULT_SCHEMA: problems.append("schema")
                if data.get("terminal") is not True: problems.append("nonterminal")
                if data.get("status") not in {"judged", "deterministic_zero_model_no_answer"}: problems.append("terminal_status")
                if identity != wanted: problems.append("identity")
                if data.get("judge_input_mode") != judge["input_mode"]: problems.append("mode")
                score, maximum = data.get("score"), data.get("max_score")
                if not finite_number(score) or not finite_number(maximum): problems.append("non_numeric_score")
                elif float(maximum) != float(item["max_score"]) or not (0 <= float(score) <= float(maximum)): problems.append("score_range")
                if not isinstance(data.get("input_fingerprint"), str) or len(data["input_fingerprint"]) != 64: problems.append("fingerprint")
                if not isinstance(data.get("source_solver_sha256"), str) or len(data["source_solver_sha256"]) != 64: problems.append("source_hash")
                if not isinstance(data.get("candidate_sha256"), str) or len(data["candidate_sha256"]) != 64: problems.append("candidate_hash")
                question_images, solution_images, combined_images = item_image_paths(item)
                expected_wire = len(combined_images) if judge["input_mode"] == "vision" else 0
                if data.get("question_image_count") != len(question_images): problems.append("question_image_count")
                if data.get("solution_image_count") != len(solution_images): problems.append("solution_image_count")
                if data.get("wire_image_count") != expected_wire: problems.append("wire_image_count")
                count = data.get("attempt_count")
                sidecars = list((path.parent / (path.stem + "-attempts")).glob("attempt-*.json"))
                if not isinstance(count, int) or isinstance(count, bool) or count < 0 or count != len(sidecars): problems.append("attempt_count")
                if problems:
                    errors.append(f"invalid judgment {row['path']}: {','.join(problems)}")
                    row["state"] = "invalid:" + ",".join(problems); coverage.append(row); continue
                key = (route, repeat, contest, item_id)
                if key in records:
                    errors.append(f"duplicate result key: {key}")
                records[key] = data
                row["state"] = "terminal"; row["score"] = data["score"]; row["max_score"] = data["max_score"]
                coverage.append(row)
    # A differently named result is not silently accepted.
    observed = list((root / "judgments").glob("**/*.json")) if (root / "judgments").exists() else []
    # Attempt directories are named ``<item>-attempts``.  Testing for a
    # literal ``"-attempts"`` path component would therefore misclassify
    # every sidecar as a terminal result.
    final_observed = [
        p for p in observed
        if not any(part.endswith("-attempts") for part in p.parts)
    ]
    if len(final_observed) != len(records):
        errors.append(f"unexpected/unvalidated final JSON files: observed={len(final_observed)}, valid={len(records)}")
    return records, errors, coverage


def frozen_candidate_audit(
    records: dict[tuple[str, int, str, str], dict[str, Any]],
    panel: dict[str, Any],
    catalog: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    routes = [str(judge["route"]) for judge in panel["judges"]]
    for contest, item_id in sorted(catalog):
        case_records = [records.get((route, repeat, contest, item_id)) for route in routes for repeat in range(1, 4)]
        present = [record for record in case_records if record is not None]
        candidate_hashes = {str(record.get("candidate_sha256")) for record in present}
        source_hashes = {str(record.get("source_solver_sha256")) for record in present}
        source_paths = {str(record.get("source_solver_path")) for record in present}
        ok = len(present) == 18 and len(candidate_hashes) == len(source_hashes) == len(source_paths) == 1
        if not ok:
            errors.append(f"frozen candidate/source differs across judges or repeats: {contest}/{item_id}")
        rows.append({
            "contest": contest,
            "item_id": item_id,
            "judgment_records": len(present),
            "candidate_sha256": next(iter(candidate_hashes)) if len(candidate_hashes) == 1 else None,
            "source_solver_sha256": next(iter(source_hashes)) if len(source_hashes) == 1 else None,
            "source_solver_path": next(iter(source_paths)) if len(source_paths) == 1 else None,
            "identical_across_six_judges_three_repeats": ok,
        })
    return rows, errors


def attempt_summary(
    root: Path,
    panel: dict[str, Any],
    catalog: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    entries: list[dict[str, Any]] = []
    errors: list[str] = []
    wire_by_route: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    file_hashes: dict[str, tuple[str, int]] = {}
    initial_requests: dict[tuple[str, str, str], list[str]] = defaultdict(list)
    prices = {str(j["route"]): j for j in panel["judges"]}
    for route, judge in prices.items():
        base = root / "judgments" / safe(route)
        for path in sorted(base.glob("**/*-attempts/attempt-*.json")) if base.exists() else []:
            try:
                raw = read_json(path)
            except Exception as e:
                errors.append(f"unreadable attempt {path.relative_to(root)}: {e}"); continue
            usage = raw.get("usage")
            if usage is None and isinstance(raw.get("raw_response"), dict): usage = raw["raw_response"].get("usage")
            if usage is None and isinstance(raw.get("response"), dict): usage = raw["response"].get("usage")
            inp, out = usage_tokens(usage)
            rel_parts = path.relative_to(base).parts
            if len(rel_parts) < 6 or not rel_parts[-2].endswith("-attempts"):
                errors.append(f"unparseable attempt path: {path.relative_to(root)}")
                continue
            contest = rel_parts[1]
            item_id = rel_parts[-2][:-len("-attempts")]
            item = catalog.get((contest, item_id))
            if item is None:
                errors.append(f"attempt item is outside frozen catalog: {path.relative_to(root)}")
                continue
            attempt_ordinal = raw.get("attempt")
            request_digest = raw.get("request_sha256")
            entries.append({"judge_route": route, "contest": contest, "item_id": item_id, "attempt_ordinal": attempt_ordinal, "attempt_path": str(path.relative_to(root)), "request_sha256": request_digest, "input_tokens": inp, "output_tokens": out, "usage_available": inp is not None and out is not None, "input_cny": None if inp is None else inp * float(judge["input_cny_per_million"]) / 1_000_000, "output_cny": None if out is None else out * float(judge["output_cny_per_million"]) / 1_000_000})
            if attempt_ordinal == 1:
                if not isinstance(request_digest, str) or len(request_digest) != 64:
                    errors.append(f"invalid initial request hash: {path.relative_to(root)}")
                else:
                    initial_requests[(route, contest, item_id)].append(request_digest)
            _, _, combined = item_image_paths(item)
            expected = combined if judge["input_mode"] == "vision" else []
            observed = raw.get("request_images")
            wire_by_route[route]["attempt_files"] += 1
            wire_by_route[route]["expected_image_instances"] += len(expected)
            if not isinstance(observed, list):
                errors.append(f"attempt request_images is not a list: {path.relative_to(root)}")
                continue
            wire_by_route[route]["observed_image_instances"] += len(observed)
            observed_paths = [str(meta.get("path")) for meta in observed if isinstance(meta, dict)]
            if observed_paths != expected:
                errors.append(f"attempt direct-image path/order mismatch: {path.relative_to(root)}")
                continue
            if not expected:
                wire_by_route[route]["zero_image_attempts"] += 1
            for expected_path, meta in zip(expected, observed):
                if not isinstance(meta, dict):
                    errors.append(f"attempt image metadata is not an object: {path.relative_to(root)}")
                    continue
                source = Path(expected_path)
                if expected_path not in file_hashes:
                    if not source.is_file():
                        errors.append(f"missing official PNG source: {expected_path}")
                        continue
                    file_hashes[expected_path] = (sha256(source), source.stat().st_size)
                digest, byte_count = file_hashes[expected_path]
                if (
                    source.suffix.lower() != ".png"
                    or meta.get("mime_type") != "image/png"
                    or meta.get("sha256") != digest
                    or meta.get("bytes") != byte_count
                    or not isinstance(meta.get("request_json_path"), list)
                ):
                    errors.append(f"attempt official PNG byte/hash proof mismatch: {path.relative_to(root)}::{expected_path}")
                else:
                    wire_by_route[route]["source_png_hash_matches"] += 1
    wire_rows = []
    for route, judge in prices.items():
        counts = wire_by_route[route]
        initial_groups = [digests for (judge_route, _, _), digests in initial_requests.items() if judge_route == route]
        identical_groups = sum(len(digests) == 3 and len(set(digests)) == 1 for digests in initial_groups)
        if len(initial_groups) != 91 or identical_groups != 91:
            errors.append(f"initial requests are not identical across all three repeats: {route}")
        wire_rows.append({
            "judge_route": route,
            "judge_input_mode": judge["input_mode"],
            "attempt_files": counts["attempt_files"],
            "expected_image_instances": counts["expected_image_instances"],
            "observed_image_instances": counts["observed_image_instances"],
            "source_png_hash_matches": counts["source_png_hash_matches"],
            "zero_image_attempts": counts["zero_image_attempts"],
            "initial_request_case_groups": len(initial_groups),
            "initial_request_groups_identical_across_three_repeats": identical_groups,
            "wire_contract": "direct_original_official_png" if judge["input_mode"] == "vision" else "zero_image_text_staticized",
        })
    return entries, errors, wire_rows


def baseline_scores(catalog: dict[tuple[str, str], dict[str, Any]]) -> tuple[dict[tuple[str, str], float], list[str]]:
    scores: dict[tuple[str, str], float] = {}
    errors: list[str] = []
    for (contest, item_id), item in catalog.items():
        path = BASELINE_ROOT[contest] / "judge" / safe(SOLVER) / safe(str(item["benchmark_id"])) / safe(str(item["sequence_id"])) / f"{safe(item_id)}.json"
        if not path.is_file():
            errors.append(f"missing frozen production baseline: {path}"); continue
        try:
            d = read_json(path)
            if d.get("terminal") is not True or d.get("judge_route") != SOLVER or d.get("solver_route") != SOLVER or d.get("item_id") != item_id or not finite_number(d.get("score")) or float(d.get("max_score")) != float(item["max_score"]) or not 0 <= float(d["score"]) <= float(d["max_score"]):
                raise ValueError("identity, terminal, or score range mismatch")
            scores[(contest, item_id)] = float(d["score"]) / float(item["max_score"])
        except Exception as e:
            errors.append(f"invalid frozen production baseline {path}: {e}")
    return scores, errors


def score_aggregates(routes: list[str], means: dict[tuple[str, str], float], catalog: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for route in routes:
        ipho_raw = sum(means[(route, item_id)] * float(item["max_score"]) for (contest, item_id), item in catalog.items() if contest == "ipho2026")
        if abs(sum(float(x["max_score"]) for (c, _), x in catalog.items() if c == "ipho2026") - 30) > 1e-9:
            raise ValueError("IPhO frozen raw maximum is not 30")
        icho_total = 0.0
        by_seq: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
        for (contest, item_id), item in catalog.items():
            if contest == "icho2026": by_seq[str(item["sequence_id"])].append((item, means[(route, item_id)]))
        for seq, values in sorted(by_seq.items()):
            raw = sum(float(item["max_score"]) * score for item, score in values)
            maxes = {float(item["problem_raw_max"]) for item, _ in values}; weights = {float(item["problem_weight"]) for item, _ in values}
            if len(maxes) != 1 or len(weights) != 1 or abs(sum(float(item["max_score"]) for item, _ in values) - next(iter(maxes))) > 1e-9:
                raise ValueError(f"IChO sequence aggregation mismatch: {seq}")
            contrib = raw / next(iter(maxes)) * next(iter(weights)); icho_total += contrib
            rows.append({"judge_route": route, "contest": "icho2026", "sequence_id": seq, "raw_score": raw, "raw_max": next(iter(maxes)), "weight": next(iter(weights)), "weighted_contribution": contrib, "reported_scale": 60})
        if abs(sum(r["weight"] for r in rows if r["judge_route"] == route and r["contest"] == "icho2026") - 60) > 1e-9:
            raise ValueError("IChO frozen weights do not sum to 60")
        rows.append({"judge_route": route, "contest": "ipho2026", "sequence_id": "ALL", "raw_score": ipho_raw, "raw_max": 30.0, "weight": None, "weighted_contribution": ipho_raw, "reported_scale": 30})
        rows.append({"judge_route": route, "contest": "icho2026", "sequence_id": "ALL", "raw_score": None, "raw_max": None, "weight": 60.0, "weighted_contribution": icho_total, "reported_scale": 60})
    return rows


def repeat_contest_totals(
    routes: list[str],
    normalized: dict[tuple[str, int, str, str], float],
    catalog: dict[tuple[str, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Compute contest totals independently for each of the three repeats.

    This deliberately aggregates each repeat first.  The reported SD is the
    sample SD across the three independent contest totals (n-1 denominator),
    not an SD of item-level repeat means.
    """
    ipho_max = sum(
        float(item["max_score"])
        for (contest, _), item in catalog.items()
        if contest == "ipho2026"
    )
    if abs(ipho_max - 30.0) > 1e-9:
        raise ValueError("IPhO frozen raw maximum is not 30")

    icho_by_seq: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for (contest, item_id), item in catalog.items():
        if contest == "icho2026":
            icho_by_seq[str(item["sequence_id"])].append((item_id, item))
    icho_specs: dict[str, tuple[float, float]] = {}
    for seq, values in sorted(icho_by_seq.items()):
        raw_maxes = {float(item["problem_raw_max"]) for _, item in values}
        weights = {float(item["problem_weight"]) for _, item in values}
        if (
            len(raw_maxes) != 1
            or len(weights) != 1
            or abs(sum(float(item["max_score"]) for _, item in values) - next(iter(raw_maxes))) > 1e-9
        ):
            raise ValueError(f"IChO sequence aggregation mismatch: {seq}")
        icho_specs[seq] = (next(iter(raw_maxes)), next(iter(weights)))
    if abs(sum(weight for _, weight in icho_specs.values()) - 60.0) > 1e-9:
        raise ValueError("IChO frozen weights do not sum to 60")

    detail: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for route in routes:
        totals: dict[str, list[float]] = {"ipho2026": [], "icho2026": []}
        for repeat in range(1, 4):
            ipho_total = sum(
                normalized[(route, repeat, contest, item_id)] * float(item["max_score"])
                for (contest, item_id), item in catalog.items()
                if contest == "ipho2026"
            )
            icho_total = 0.0
            for seq, values in sorted(icho_by_seq.items()):
                raw = sum(
                    normalized[(route, repeat, "icho2026", item_id)] * float(item["max_score"])
                    for item_id, item in values
                )
                raw_max, weight = icho_specs[seq]
                icho_total += raw / raw_max * weight
            for contest, total, scale in (
                ("ipho2026", ipho_total, 30.0),
                ("icho2026", icho_total, 60.0),
            ):
                totals[contest].append(total)
                detail.append({
                    "judge_route": route,
                    "contest": contest,
                    "repeat": repeat,
                    "total_score": total,
                    "reported_scale": scale,
                })
        for contest, values in totals.items():
            scale = 30.0 if contest == "ipho2026" else 60.0
            summaries.append({
                "judge_route": route,
                "contest": contest,
                "repeat_1_total": values[0],
                "repeat_2_total": values[1],
                "repeat_3_total": values[2],
                "mean_total": statistics.mean(values),
                "sample_sd_total": statistics.stdev(values),
                "reported_scale": scale,
            })
    return detail, summaries


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=("check", "analyze"), help="check is read-only; analyze writes only after a complete pass")
    ap.add_argument("--run-root", type=Path, default=DEFAULT_ROOT)
    args = ap.parse_args()
    root = args.run_root.resolve()
    try:
        panel, cases = read_json(PANEL), read_json(CASES)
        catalog, _ = item_catalog()
        expected = 6 * 3 * len(catalog)
        errors = validate_provenance(root, panel, cases)
        run_summary, run_summary_errors = validate_run_summary(root, expected)
        records, record_errors, coverage = terminal_records(root, panel, catalog)
        candidate_audit, candidate_errors = frozen_candidate_audit(records, panel, catalog)
        attempts, attempt_errors, wire_audit = attempt_summary(root, panel, catalog)
        errors.extend(run_summary_errors); errors.extend(record_errors); errors.extend(candidate_errors); errors.extend(attempt_errors)
    except Exception as e:
        print(json.dumps({"status": "incomplete_or_invalid", "fatal": [str(e)]}, indent=2), file=sys.stderr)
        return 2
    check = {"schema_version": "science_arena_rebuttal_cross_judge_analysis_check_v1", "created_at": datetime.now(timezone.utc).isoformat(), "run_root": str(root), "expected_terminal_judgments": expected, "validated_terminal_judgments": len(records), "attempt_files_seen": len(attempts), "status": "complete" if not errors and len(records) == expected else "incomplete_or_invalid", "errors": errors}
    if args.command == "check" or check["status"] != "complete":
        print(json.dumps(check, indent=2, ensure_ascii=False))
        return 0 if check["status"] == "complete" else 2

    baseline, baseline_errors = baseline_scores(catalog)
    if baseline_errors:
        print(json.dumps({**check, "status": "incomplete_or_invalid", "errors": baseline_errors}, indent=2, ensure_ascii=False), file=sys.stderr)
        return 2
    routes = [str(j["route"]) for j in panel["judges"]]
    modes = {str(j["route"]): str(j["input_mode"]) for j in panel["judges"]}
    normalized: dict[tuple[str, int, str, str], float] = {(route, repeat, contest, item): float(d["score"]) / float(d["max_score"]) for (route, repeat, contest, item), d in records.items()}
    means: dict[tuple[str, str], float] = {}
    case_rows: list[dict[str, Any]] = []
    repeat_metrics: list[dict[str, Any]] = []
    for route in routes:
        values_all: list[float] = []; mae_values: list[float] = []; sd_values: list[float] = []; exact = 0
        for (contest, item_id), item in sorted(catalog.items()):
            vals = [normalized[(route, r, contest, item_id)] for r in range(1, 4)]
            m = mean(vals); sd = statistics.pstdev(vals); means[(route, item_id)] = m
            pair_mae = mean(abs(vals[a] - vals[b]) for a, b in ((0, 1), (0, 2), (1, 2)))
            mae_values.append(pair_mae); sd_values.append(sd); exact += int(vals[0] == vals[1] == vals[2]); values_all.extend(vals)
            case_rows.append({"judge_route": route, "judge_input_mode": modes[route], "contest": contest, "item_id": item_id, "sequence_id": item["sequence_id"], "max_score": item["max_score"], "repeat_1_normalized": vals[0], "repeat_2_normalized": vals[1], "repeat_3_normalized": vals[2], "repeat_mean_normalized": m, "repeat_population_sd": sd, "repeat_pairwise_mae": pair_mae, "baseline_gemini35_normalized": baseline[(contest, item_id)], "delta_vs_baseline": m - baseline[(contest, item_id)]})
        repeat_metrics.append({"judge_route": route, "judge_input_mode": modes[route], "cases": len(catalog), "mean_pairwise_repeat_mae_normalized": mean(mae_values), "mean_case_repeat_population_sd_normalized": mean(sd_values), "exact_three_repeat_agreement_rate": exact / len(catalog), "all_repeat_score_population_sd_normalized": statistics.pstdev(values_all)})
    pairwise: list[dict[str, Any]] = []
    for i, left in enumerate(routes):
        for right in routes[i + 1:]:
            for label, predicate in (("overall", lambda c: True), ("ipho2026", lambda c: c == "ipho2026"), ("icho2026", lambda c: c == "icho2026")):
                keys = [(c, item) for c, item in catalog if predicate(c)]
                x, y = [means[(left, item)] for _, item in keys], [means[(right, item)] for _, item in keys]
                row = {"judge_a": left, "judge_b": right, "judge_a_mode": modes[left], "judge_b_mode": modes[right], "subset": label, "cases": len(keys), "mean_absolute_error_normalized": mean(abs(a - b) for a, b in zip(x, y)), **correlation_metrics(x, y), "interpretation_scope": "vision_judge_agreement" if modes[left] == modes[right] == "vision" else "modality_sensitivity_only"}
                pairwise.append(row)
    baseline_metrics: list[dict[str, Any]] = []
    for route in routes:
        for label, predicate in (("overall", lambda c: True), ("ipho2026", lambda c: c == "ipho2026"), ("icho2026", lambda c: c == "icho2026")):
            keys = [(c, item) for c, item in catalog if predicate(c)]
            x, b = [means[(route, item)] for _, item in keys], [baseline[(c, item)] for c, item in keys]
            baseline_metrics.append({"judge_route": route, "judge_input_mode": modes[route], "subset": label, "cases": len(keys), "signed_delta_normalized": mean(a - z for a, z in zip(x, b)), "mean_absolute_delta_normalized": mean(abs(a-z) for a, z in zip(x, b)), **correlation_metrics(x, b), "baseline": "frozen production Gemini-3.5 vision judge"})
    cost_rows: list[dict[str, Any]] = []
    by_route: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in attempts: by_route[a["judge_route"]].append(a)
    for route in routes:
        vals = by_route[route]; known = [x for x in vals if x["usage_available"]]
        route_records = [data for (judge_route, _, _, _), data in records.items() if judge_route == route]
        attempt_counts = [int(data["attempt_count"]) for data in route_records]
        cost_rows.append({"judge_route": route, "judge_input_mode": modes[route], "logical_judgments": len(route_records), "successful_terminal_judgments": len(route_records), "failed_terminal_judgments": 0, "attempt_files": len(vals), "extra_retry_attempts": len(vals) - len(route_records), "judgments_requiring_retry": sum(value > 1 for value in attempt_counts), "maximum_attempts_for_one_judgment": max(attempt_counts), "attempts_with_complete_usage": len(known), "input_tokens": sum(int(x["input_tokens"]) for x in known), "output_tokens": sum(int(x["output_tokens"]) for x in known), "actual_cost_cny_for_usage_known_attempts": sum(float(x["input_cny"]) + float(x["output_cny"]) for x in known), "cost_complete": len(vals) == len(known)})
    aggregate_rows = score_aggregates(routes, means, catalog)
    repeat_total_rows, repeat_total_summary = repeat_contest_totals(routes, normalized, catalog)
    analysis = {"schema_version": "science_arena_rebuttal_cross_judge_analysis_v1", "created_at": datetime.now(timezone.utc).isoformat(), "scope": {"solver_route": SOLVER, "frozen_answers": 91, "judges": 6, "repeats": 3, "logical_judgments": expected, "limitation": "One frozen solver route only: no 14-model ranking correlation and no cross-candidate robustness inference."}, "coverage": check, "run_summary": run_summary, "frozen_candidate_audit": candidate_audit, "wire_modality_audit": wire_audit, "repeat_reliability": repeat_metrics, "vision_pairwise_and_modality_sensitivity": pairwise, "baseline_comparison": baseline_metrics, "cost_and_tokens": cost_rows, "contest_aggregates_from_item_repeat_means": aggregate_rows, "contest_repeat_totals": repeat_total_rows, "contest_total_mean_sample_sd": repeat_total_summary, "interpretation": {"vision_primary": "Only vision-vs-vision rows are primary full-fidelity judge agreement results.", "text_only": "Rows involving either text-staticized judge are modality-sensitivity results, not evidence against the full-fidelity vision pipeline.", "validity": "Repeat/pairwise agreement measures reliability or concordance, not validity. Validity needs the separate human-calibration evidence.", "aggregation": "Cross-judge item comparisons first average the three normalized item judgments. Contest uncertainty instead aggregates each repeat independently: IPhO is summed on /30; IChO sums sequence raw scores divided by official problem raw maxima and weighted to /60. Mean and sample SD are then computed across the three independent contest totals.", "multimodal_scope": "The frozen candidate answers are text. Full-fidelity multimodal judging is required because the official question and solution/marking references include PNG pixels; vision judges receive those original official PNGs directly, while text-staticized judges receive zero image blocks."}}
    out = root / "analysis"
    if out.exists():
        raise RuntimeError(f"refusing to overwrite existing analysis directory: {out}")
    out.mkdir(parents=False)
    def dump(name: str, value: Any) -> None:
        (out / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    def write_csv(name: str, rows: list[dict[str, Any]]) -> None:
        fields = sorted({k for row in rows for k in row})
        with (out / name).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    dump("summary.json", analysis); dump("coverage.json", check); dump("attempt_usage.json", attempts)
    write_csv("case_scores.csv", case_rows); write_csv("frozen_candidate_audit.csv", candidate_audit); write_csv("wire_modality_audit.csv", wire_audit); write_csv("repeat_reliability.csv", repeat_metrics); write_csv("pairwise_metrics.csv", pairwise); write_csv("baseline_comparison.csv", baseline_metrics); write_csv("cost_and_tokens.csv", cost_rows); write_csv("contest_aggregates.csv", aggregate_rows); write_csv("contest_repeat_totals.csv", repeat_total_rows); write_csv("contest_total_mean_sample_sd.csv", repeat_total_summary); write_csv("coverage.csv", coverage)
    vision = [x for x in pairwise if x["interpretation_scope"] == "vision_judge_agreement" and x["subset"] == "overall"]
    lines = ["# Six-judge frozen-answer replay", "", "Status: complete (all 1,638 terminal judgment artifacts passed structural validation).", "", "## Scope and limits", "", "This is full 91-item coverage for one frozen Gemini-3.5 Flash solver answer corpus, not a 14-model experiment. It cannot estimate ranking correlation across candidate models or cross-candidate robustness. Three-repeat and cross-judge agreement are reliability/concordance measures, not validity; validity requires the separate human-calibration evidence.", "", "Vision-vs-vision comparisons are the primary full-fidelity result. Any comparison involving a text-staticized judge is labelled modality sensitivity only.", "", "## Repeat reliability", ""]
    lines += [f"- {x['judge_route']}: repeat MAE {x['mean_pairwise_repeat_mae_normalized']:.4f}; mean case SD {x['mean_case_repeat_population_sd_normalized']:.4f}; exact 3-way {x['exact_three_repeat_agreement_rate']:.1%}." for x in repeat_metrics]
    lines += ["", "## Vision judge agreement (three-repeat means; normalized item scores)", ""]
    lines += [f"- {x['judge_a']} vs {x['judge_b']}: MAE {x['mean_absolute_error_normalized']:.4f}; Pearson {x['pearson']}; Spearman {x['spearman']}; Kendall tau-b {x['kendall_tau_b']}." for x in vision]
    lines += ["", "## Contest totals across three independent repeats", ""]
    lines += [
        f"- {x['judge_route']} — {x['contest']}: {x['mean_total']:.4f} ± {x['sample_sd_total']:.4f} /{x['reported_scale']:.0f} (sample SD; repeat totals {x['repeat_1_total']:.4f}, {x['repeat_2_total']:.4f}, {x['repeat_3_total']:.4f})."
        for x in repeat_total_summary
    ]
    lines += ["", "## Aggregate scales", "", "IPhO is reported on /30. IChO aggregates each repeat independently as raw/problem raw max × problem weight and sums to /60; the displayed uncertainty is the sample SD across the three repeat totals. The frozen candidate answers themselves are text. Vision is required by official question and solution/marking PNG references, which the four vision judges receive directly; the two text-staticized judges receive zero image blocks. Detailed machine-readable outputs are the adjacent JSON and CSV files.", ""]
    (out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "analysis_dir": str(out), "terminal_judgments": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
