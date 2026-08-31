#!/usr/bin/env python3
"""Run a few formal judge cases in-place before launching the full judge fanout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pipeline


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--capability-dir", type=Path, required=True)
    parser.add_argument("--solver-route", required=True)
    parser.add_argument("--item-ids", required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    run_root = args.run_root.resolve()
    items = pipeline.read_jsonl(args.items.resolve())
    models = pipeline.load_model_manifest(args.models.resolve())
    validation = pipeline.validate_inputs(items, models, workspace)
    if not validation["valid"]:
        raise RuntimeError("full canonical input validation failed")
    calibrated = [model for model in models if model.get("calibrated_production_judge") is True]
    if len(calibrated) != 1 or calibrated[0].get("input_mode") != "vision":
        raise RuntimeError("manifest must contain exactly one calibrated vision judge")
    judge_model = calibrated[0]
    pipeline.require_capability_proofs([judge_model], args.capability_dir.resolve())
    wanted = [value.strip() for value in args.item_ids.split(",") if value.strip()]
    items_by_id = pipeline.item_map(items)
    missing_items = [item_id for item_id in wanted if item_id not in items_by_id]
    if missing_items:
        raise RuntimeError(f"unknown item IDs: {missing_items}")
    solver_paths: dict[str, Path] = {}
    route_root = run_root / "solver" / pipeline.safe_name(args.solver_route)
    for item_id in wanted:
        item = items_by_id[item_id]
        path = (
            route_root
            / pipeline.safe_name(str(item["benchmark_id"]))
            / pipeline.safe_name(str(item["sequence_id"]))
            / f"{pipeline.safe_name(item_id)}.json"
        )
        if path.is_file():
            row = pipeline.read_json(path)
            if row.get("route") != args.solver_route or row.get("item_id") != item_id:
                raise RuntimeError(f"solver record identity mismatch: {path}")
            solver_paths[item_id] = path
    missing_solver = [item_id for item_id in wanted if item_id not in solver_paths]
    if missing_solver:
        raise RuntimeError(f"missing requested solver records: {missing_solver}")

    api_key = pipeline.load_api_key()
    results = []
    for item_id in wanted:
        result = pipeline.run_judge_case(
            solver_path=solver_paths[item_id],
            item=items_by_id[item_id],
            judge_model=judge_model,
            workspace=workspace,
            run_root=run_root,
            api_key=api_key,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    summary = {
        "schema_version": "science_arena_formal_judge_smoke_v1",
        "created_at": pipeline.utc_now(),
        "pipeline_sha256": pipeline.sha256_file(Path(pipeline.__file__).resolve()),
        "items_sha256": pipeline.sha256_file(args.items.resolve()),
        "models_sha256": pipeline.sha256_file(args.models.resolve()),
        "run_root": str(run_root),
        "solver_route": args.solver_route,
        "judge_route": judge_model["route"],
        "results": results,
        "all_judged": all(result["status"] in {"judged", "resumed"} for result in results),
    }
    output = run_root / f"judge_smoke__{pipeline.safe_name(args.solver_route)}.json"
    pipeline.atomic_write_json(output, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["all_judged"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
