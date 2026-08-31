#!/usr/bin/env python3
"""Collect text-only interleaved evaluation outputs into benchmark tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/science_arena_text_interleaved_v1"
DEFAULT_RUN_ROOT = ROOT / "artifacts/runs/text_interleaved_eval"
DEFAULT_OUTPUT = ROOT / "artifacts/runs/text_interleaved_eval/collected_text_interleaved_table"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def dataset_summary(dataset_dir: Path) -> dict[str, dict[str, Any]]:
    rows = iter_jsonl(dataset_dir / "benchmark_items.jsonl")
    by_bench: dict[str, dict[str, Any]] = {}
    for row in rows:
        bench = str(row.get("benchmark_id"))
        summary = by_bench.setdefault(
            bench,
            {
                "benchmark_id": bench,
                "turn_count": 0,
                "sequence_count": set(),
                "max_score": 0.0,
                "report_scale": float(row.get("report_scale") or 0),
                "release_date": row.get("release_date", ""),
                "source_quality": row.get("source_quality", ""),
            },
        )
        summary["turn_count"] += 1
        summary["sequence_count"].add(str(row.get("sequence_id")))
        summary["max_score"] += float(row.get("max_score") or 0)
        if row.get("report_scale"):
            summary["report_scale"] = float(row.get("report_scale") or 0)
    for summary in by_bench.values():
        summary["sequence_count"] = len(summary["sequence_count"])
    return by_bench


def image_payload_counts(turn_rows: list[dict[str, Any]]) -> dict[str, int]:
    keys = [
        "chat_image_url_count",
        "data_url_image_count",
        "empty_image_url_count",
        "remote_url_image_count",
        "responses_input_image_count",
    ]
    totals = {key: 0 for key in keys}
    for row in turn_rows:
        meta = row.get("generation_client_meta") or {}
        image_summary = meta.get("final_payload_image_summary") or {}
        for key in keys:
            totals[key] += int(image_summary.get(key) or 0)
    totals["any_image_payload_rows"] = sum(
        1
        for row in turn_rows
        if any(int(((row.get("generation_client_meta") or {}).get("final_payload_image_summary") or {}).get(key) or 0) for key in keys)
    )
    return totals


def is_bad_finish_reason(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    if text in {"stop", "completed", "complete", "objective_key"}:
        return False
    return any(
        fragment in text
        for fragment in (
            "content_filter",
            "finish_content_filter",
            "length",
            "max_token",
            "max_output",
            "truncat",
        )
    )


def bad_finish_reason_rows(turn_rows: list[dict[str, Any]]) -> list[str]:
    bad: list[str] = []
    for row in turn_rows:
        if is_bad_finish_reason(row.get("generation_finish_reason")) or is_bad_finish_reason(row.get("judge_finish_reason")):
            bad.append(str(row.get("turn_id") or row.get("row_id") or row.get("item_id") or ""))
    return bad


def collect_run_dirs(run_dirs: list[Path], run_root: Path, glob_pattern: str) -> list[Path]:
    dirs: list[Path] = []
    for run_dir in run_dirs:
        dirs.append(run_dir)
    if glob_pattern:
        dirs.extend(path for path in run_root.glob(glob_pattern) if path.is_dir())
    return sorted(set(path.resolve() for path in dirs))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--run_root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run_dir", type=Path, action="append", default=[])
    parser.add_argument("--glob", default="")
    parser.add_argument("--model_config", type=Path, default=None)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    expected = dataset_summary(args.dataset_dir)
    run_dirs = collect_run_dirs(args.run_dir, args.run_root, args.glob)
    allowed_models: set[str] | None = None
    if args.model_config:
        config = load_json(args.model_config, {})
        allowed_models = set(config.get("models") or [])
    cells: dict[tuple[str, str], dict[str, Any]] = {}
    provenance: defaultdict[tuple[str, str], list[str]] = defaultdict(list)

    for run_dir in run_dirs:
        for summary_path in sorted(run_dir.glob("*/benchmark_summary.json")):
            model_dir = summary_path.parent
            summaries = load_json(summary_path, [])
            turn_rows = iter_jsonl(model_dir / "turn_results.jsonl")
            turn_rows_by_bench: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in turn_rows:
                turn_rows_by_bench[str(row.get("benchmark_id"))].append(row)
            for summary in summaries:
                bench = str(summary.get("benchmark_id"))
                model = str(summary.get("model"))
                if allowed_models is not None and model not in allowed_models:
                    continue
                key = (bench, model)
                exp = expected.get(bench, {})
                payload_counts = image_payload_counts(turn_rows_by_bench.get(bench, []))
                turn_count = int(summary.get("turn_count") or 0)
                expected_turn_count = int(exp.get("turn_count") or 0)
                gen_errors = int(summary.get("generation_error_count") or 0)
                judge_errors = int(summary.get("judge_error_count") or 0)
                image_payload_rows = int(payload_counts.get("any_image_payload_rows") or 0)
                bad_finish_rows = bad_finish_reason_rows(turn_rows_by_bench.get(bench, []))
                status = "ok"
                reasons: list[str] = []
                if expected_turn_count and turn_count != expected_turn_count:
                    status = "invalid"
                    reasons.append(f"turn_count {turn_count}!={expected_turn_count}")
                if gen_errors:
                    status = "invalid"
                    reasons.append(f"generation_errors={gen_errors}")
                if judge_errors:
                    status = "invalid"
                    reasons.append(f"judge_errors={judge_errors}")
                if image_payload_rows:
                    status = "invalid"
                    reasons.append(f"nonzero_image_payload_rows={image_payload_rows}")
                if bad_finish_rows:
                    status = "invalid"
                    reasons.append(f"bad_finish_reasons={len(bad_finish_rows)}")
                cell = {
                    "benchmark_id": bench,
                    "model": model,
                    "run_dir": str(run_dir),
                    "status": status,
                    "status_reasons": reasons,
                    "score": float(summary.get("score") or 0),
                    "max_score": float(summary.get("max_score") or 0),
                    "reported_score": float(summary.get("reported_score") or 0),
                    "report_scale": float(summary.get("report_scale") or exp.get("report_scale") or 0),
                    "turn_count": turn_count,
                    "expected_turn_count": expected_turn_count,
                    "generation_error_count": gen_errors,
                    "judge_error_count": judge_errors,
                    "bad_finish_reason_count": len(bad_finish_rows),
                    "bad_finish_reason_turn_ids": bad_finish_rows,
                    "cost_cny": float(summary.get("cost_cny") or 0),
                    "image_payload_counts": payload_counts,
                }
                # Prefer valid cells; otherwise keep the latest discovered cell.
                if key not in cells or (cells[key]["status"] != "ok" and cell["status"] == "ok"):
                    cells[key] = cell
                provenance[key].append(str(run_dir))

    models = sorted({model for _, model in cells})
    benches = sorted(expected)
    records = [cells[key] | {"all_candidate_run_dirs": provenance[key]} for key in sorted(cells)]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    save_json(args.output_dir / "records.json", records)
    save_json(args.output_dir / "dataset_summary.json", expected)

    with (args.output_dir / "records.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "benchmark_id",
                "model",
                "status",
                "reported_score",
                "report_scale",
                "score",
                "max_score",
                "turn_count",
                "expected_turn_count",
                "generation_error_count",
                "judge_error_count",
                "bad_finish_reason_count",
                "cost_cny",
                "run_dir",
                "status_reasons",
            ],
        )
        writer.writeheader()
        for record in records:
            out = {key: record.get(key) for key in writer.fieldnames}
            out["status_reasons"] = "; ".join(record.get("status_reasons") or [])
            writer.writerow(out)

    lines: list[str] = []
    lines.append("| Benchmark | Release | Max | " + " | ".join(models) + " |")
    lines.append("|---|---:|---:|" + "|".join(["---:"] * len(models)) + "|")
    for bench in benches:
        exp = expected[bench]
        row = [bench, str(exp.get("release_date") or ""), str(exp.get("report_scale") or exp.get("max_score") or "")]
        for model in models:
            cell = cells.get((bench, model))
            if not cell:
                row.append("missing")
            elif cell["status"] != "ok":
                row.append("invalid")
            else:
                row.append(f"{cell['reported_score']:.4g}")
        lines.append("| " + " | ".join(row) + " |")
    (args.output_dir / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "run_dirs": [str(path) for path in run_dirs], "records": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
