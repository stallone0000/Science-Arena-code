#!/usr/bin/env python3
"""Apply fingerprint-locked human score adjudications to a separate aggregate."""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any

import pipeline


def dec(value: Any) -> Decimal:
    return Decimal(str(value))


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def json_number(value: Decimal) -> int | float:
    integral = value.to_integral_value()
    return int(integral) if value == integral else float(value)


def rubric_criteria(item: dict[str, Any]) -> list[dict[str, Any]]:
    return pipeline.normalized_rubric_criteria(pipeline.grading_rubric_for(item))


def judge_path(run_root: Path, row: dict[str, Any], item: dict[str, Any]) -> Path:
    return (
        run_root
        / "judge"
        / pipeline.safe_name(str(row["solver_route"]))
        / pipeline.safe_name(str(item["benchmark_id"]))
        / pipeline.safe_name(str(item["sequence_id"]))
        / f"{pipeline.safe_name(str(item['item_id']))}.json"
    )


def validate_override(
    row: dict[str, Any],
    *,
    run_root: Path,
    item: dict[str, Any],
    routes: set[str],
) -> tuple[Decimal, Decimal]:
    route = str(row.get("solver_route") or "")
    item_id = str(row.get("item_id") or "")
    if route not in routes:
        raise ValueError(f"override route is absent from model manifest: {route}")
    if str(row.get("sequence_id") or "") != str(item["sequence_id"]):
        raise ValueError(f"override sequence mismatch: {route}::{item_id}")

    path = judge_path(run_root, row, item)
    judge = load_json(path)
    if not judge.get("terminal"):
        raise ValueError(f"judge record is not terminal: {path}")
    if str(judge.get("solver_route") or "") != route or str(judge.get("item_id") or "") != item_id:
        raise ValueError(f"judge identity mismatch: {path}")
    if str(judge.get("judge_input_fingerprint") or "") != str(row.get("judge_input_fingerprint") or ""):
        raise ValueError(f"judge fingerprint mismatch: {route}::{item_id}")

    original = dec(row["original_score"])
    override = dec(row["override_score"])
    if dec(judge.get("score")) != original:
        raise ValueError(f"original score mismatch: {route}::{item_id}")
    maximum = dec(item["max_score"])
    if override < 0 or override > maximum:
        raise ValueError(f"override score out of range: {route}::{item_id}")

    expected = rubric_criteria(item)
    supplied = row.get("criteria")
    if not isinstance(supplied, list) or len(supplied) != len(expected):
        raise ValueError(f"override criteria count mismatch: {route}::{item_id}")
    awarded_sum = Decimal("0")
    for actual, wanted in zip(supplied, expected):
        if str(actual.get("criterion_id") or "") != str(wanted["criterion_id"]):
            raise ValueError(f"override criterion id/order mismatch: {route}::{item_id}")
        max_score = dec(wanted["max_score"])
        if dec(actual.get("max_score")) != max_score:
            raise ValueError(f"override criterion maximum mismatch: {route}::{item_id}")
        awarded = dec(actual.get("awarded"))
        if awarded < 0 or awarded > max_score:
            raise ValueError(f"override criterion award out of range: {route}::{item_id}")
        awarded_sum += awarded
    if awarded_sum != override:
        raise ValueError(f"override criterion sum mismatch: {route}::{item_id}")
    return original, override


def apply_overrides(
    *,
    run_root: Path,
    items_path: Path,
    models_path: Path,
    overrides_path: Path,
) -> dict[str, Any]:
    base_path = run_root / "aggregate.json"
    base = load_json(base_path)
    if not base.get("complete"):
        raise ValueError("refusing to adjudicate an incomplete aggregate")
    overrides = load_json(overrides_path)
    if overrides.get("schema_version") != "science_arena_manual_score_overrides_v1":
        raise ValueError("unsupported override schema")
    if pipeline.sha256_file(items_path) != str(overrides.get("run_items_sha256") or ""):
        raise ValueError("items SHA-256 does not match override lock")
    declared_root = (Path.cwd() / str(overrides.get("run_root") or "")).resolve()
    if declared_root != run_root.resolve():
        raise ValueError("run root does not match override lock")
    report_path = (Path.cwd() / str(overrides.get("adjudication_report") or "")).resolve()
    if pipeline.sha256_file(report_path) != str(overrides.get("adjudication_report_sha256") or ""):
        raise ValueError("adjudication report SHA-256 mismatch")

    items = {str(item["item_id"]): item for item in pipeline.read_jsonl(items_path)}
    routes = {str(model["route"]) for model in pipeline.load_model_manifest(models_path)}
    result = copy.deepcopy(base)
    result["schema_version"] = "science_arena_aggregate_adjudicated_v1"
    result["created_at"] = datetime.now(timezone.utc).isoformat()
    result["machine_aggregate_path"] = str(base_path)
    result["machine_aggregate_sha256"] = pipeline.sha256_file(base_path)
    result["manual_overrides_path"] = str(overrides_path)
    result["manual_overrides_sha256"] = pipeline.sha256_file(overrides_path)
    result["adjudication_report"] = str(report_path)
    result["adjudication_report_sha256"] = pipeline.sha256_file(report_path)
    for route_aggregates in result["aggregates"].values():
        for benchmark in route_aggregates.values():
            benchmark["machine_score"] = benchmark.get("score")
            benchmark["score_basis"] = "machine_judge"

    seen: set[tuple[str, str]] = set()
    applied: list[dict[str, Any]] = []
    for row in overrides.get("overrides") or []:
        route = str(row.get("solver_route") or "")
        item_id = str(row.get("item_id") or "")
        key = (route, item_id)
        if key in seen:
            raise ValueError(f"duplicate override: {route}::{item_id}")
        seen.add(key)
        if item_id not in items:
            raise ValueError(f"override item is absent from dataset: {item_id}")
        item = items[item_id]
        if str(item["benchmark_id"]) != str(overrides.get("benchmark_id") or ""):
            raise ValueError(f"override benchmark mismatch: {route}::{item_id}")
        original, override = validate_override(
            row,
            run_root=run_root,
            item=item,
            routes=routes,
        )
        delta_raw = override - original
        benchmark = result["aggregates"][route][str(item["benchmark_id"])]
        current_score = dec(benchmark.get("score"))
        breakdown = benchmark.get("sequence_breakdown") or {}
        if breakdown:
            sequence = breakdown[str(item["sequence_id"])]
            raw_max = dec(sequence["raw_max"])
            weight = dec(sequence["weight"])
            weighted_delta = delta_raw / raw_max * weight
            sequence["raw_score"] = json_number(dec(sequence["raw_score"]) + delta_raw)
            sequence["weighted_score"] = json_number(dec(sequence["weighted_score"]) + weighted_delta)
            benchmark["score"] = json_number(current_score + weighted_delta)
        else:
            weighted_delta = delta_raw
            benchmark["score"] = json_number(current_score + delta_raw)
        benchmark["score_basis"] = "manual_adjudicated"
        applied.append(
            {
                **copy.deepcopy(row),
                "raw_score_delta": json_number(delta_raw),
                "reported_score_delta": json_number(weighted_delta),
            }
        )
    result["manual_adjudication"] = {
        "benchmark_id": overrides["benchmark_id"],
        "override_count": len(applied),
        "overrides": applied,
    }
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--overrides", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = apply_overrides(
        run_root=args.run_root.resolve(),
        items_path=args.items.resolve(),
        models_path=args.models.resolve(),
        overrides_path=args.overrides.resolve(),
    )
    pipeline.atomic_write_json(args.output.resolve(), result)
    print(json.dumps({"complete": result["complete"], "override_count": result["manual_adjudication"]["override_count"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
