#!/usr/bin/env python3
"""Offline replay of the archived two-Gemini-judge calibration.

The replay reads four frozen final judge-result arrays (two judges by two
exams), the two frozen expert ground-truth files, and archived aggregate /
comparison artifacts.  It performs no inference and no API call.  Generated
CSV/JSON/Markdown/SHA content is emitted to stdout; this script does not write
the checked-in artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[2]
SCHEMA_VERSION = "science_arena_archived_two_gemini_calibration_replay_v1"
FIXED_DATE = "2026-08-29"
BUNDLE_REL = "audits/cr_rebuttal_local_supplements_20260829"
REL_SCRIPT = f"{BUNDLE_REL}/replay_archived_two_gemini_calibration_20260829.py"
REL_CSV = f"{BUNDLE_REL}/archived_two_gemini_calibration_rows_20260829.csv"
REL_JSON = f"{BUNDLE_REL}/archived_two_gemini_calibration_replay_20260829.json"
REL_MD = f"{BUNDLE_REL}/archived_two_gemini_calibration_replay_20260829.md"
REL_SIDECAR = f"{BUNDLE_REL}/archived_two_gemini_calibration_replay_20260829.sha256"
TOL = 1e-12

GT_INPUTS = {
    "ipho2025": {
        "path": "science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/ground_truth/ipho2025_expert_ground_truth.json",
        "sha256": "0719eb5cf59843fb44d3e8125f63a95b4711425590e1b0e77c40dce1ba2406db",
        "expected_rows": 200,
        "exam_scale": 30.0,
    },
    "icho2025": {
        "path": "science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/ground_truth/icho2025_expert_ground_truth.json",
        "sha256": "5ed62c788ec71b20f5f7b63d70d6cede6e12db3da6bb938e847c96500cb40d66",
        "expected_rows": 325,
        "exam_scale": 60.0,
    },
}

SUPPORT_INPUTS = {
    "science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/science_arena_judge.py": "6548f5ef0e78386f233e129015d998ac87d1b8a3b6d631470e8c326133f7206e",
    "science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/compare_to_ground_truth.py": "4b053fcd3c20cf601182942fe77aa586a6f270616be8996998f331decc5769fd",
    "science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/collect_gemini_judge_calibration_table.py": "8b951371462f3e4fa35a2bc375550451ebdf26aa4b7572c942404fd0ddcb2d38",
}

RUNS = [
    {
        "run_id": "gemini_3_1_pro_preview__ipho2025",
        "judge_id": "gemini_3_1_pro_preview",
        "judge_model": "google/google/gemini-3.1-pro-preview",
        "exam": "ipho2025",
        "directory": "science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/runs/merged_ipho_gemini31pro_strictnotes_v2_reparsed",
        "expected_rows": 200,
        "hashes": {
            "judge_results.json": "388b83e342f91fab30f32e0b7f4e7f7d7f436a38c49713fe9385de27dd214357",
        },
    },
    {
        "run_id": "gemini_3_1_pro_preview__icho2025",
        "judge_id": "gemini_3_1_pro_preview",
        "judge_model": "google/google/gemini-3.1-pro-preview",
        "exam": "icho2025",
        "directory": "science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/runs/fullcalib_icho_gemini31pro_strictnotes_v4_reparsed",
        "expected_rows": 325,
        "hashes": {
            "judge_results.json": "1530eb733e14178ed6ae71793107374344181046648652a97c1457cfc972ea47",
        },
    },
    {
        "run_id": "gemini_3_5_flash__ipho2025",
        "judge_id": "gemini_3_5_flash",
        "judge_model": "google/gemini-3.5-flash",
        "exam": "ipho2025",
        "directory": "science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/runs/merged_ipho_gemini35_v6",
        "expected_rows": 200,
        "hashes": {
            "judge_results.json": "af660907e73684dea6b5e81c585964a0ae58dbc4353b37f7a0edfc6f76def198",
        },
    },
    {
        "run_id": "gemini_3_5_flash__icho2025",
        "judge_id": "gemini_3_5_flash",
        "judge_model": "google/gemini-3.5-flash",
        "exam": "icho2025",
        "directory": "science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/runs/merged_icho_gemini35_v8_reparsed",
        "expected_rows": 325,
        "hashes": {
            "judge_results.json": "3adb825fc941046b5d4a2848438c502a70e0fcd7c7bc59f732abe58423101664",
        },
    },
]

CSV_FIELDS = [
    "record_id", "judge_id", "judge_model", "exam", "run_id", "run_directory",
    "source_row_index", "model_under_test", "case_id", "question_id",
    "sub_question_id", "section", "item_index", "max_score", "expert_score",
    "judge_score", "signed_error_judge_minus_expert", "absolute_error",
    "expert_fraction_of_max", "judge_fraction_of_max", "normalized_pair_status",
    "raw_judgment_sha256", "judge_results_sha256", "ground_truth_path",
    "ground_truth_sha256", "ground_truth_locator",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_sha(value: Any) -> str:
    return sha256_bytes(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def clean(value: float | None) -> float | None:
    if value is None or not math.isfinite(value):
        return None
    rounded = round(float(value), 12)
    return 0.0 if abs(rounded) < 0.5e-12 else rounded


def ratio(count: int, denominator: int) -> float | None:
    return clean(count / denominator) if denominator else None


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    assert mx is not None and my is not None
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= TOL or vy <= TOL:
        return None
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / math.sqrt(vx * vy)


def average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[indexed[k][0]] = avg_rank
        i = j
    return ranks


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    return pearson(average_ranks(xs), average_ranks(ys))


def lin_ccc(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    assert mx is not None and my is not None
    n = len(xs)
    vx = sum((x - mx) ** 2 for x in xs) / n
    vy = sum((y - my) ** 2 for y in ys) / n
    covariance = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / n
    denominator = vx + vy + (mx - my) ** 2
    return 2 * covariance / denominator if denominator > TOL else None


def linear_calibration(xs: Sequence[float], ys: Sequence[float]) -> dict[str, Any]:
    if len(xs) != len(ys) or len(xs) < 2:
        return {"denominator": len(xs), "intercept": None, "slope": None, "r_squared": None}
    mx, my = mean(xs), mean(ys)
    assert mx is not None and my is not None
    denominator = sum((x - mx) ** 2 for x in xs)
    if denominator <= TOL:
        return {"denominator": len(xs), "intercept": None, "slope": None, "r_squared": None}
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denominator
    intercept = my - slope * mx
    r = pearson(xs, ys)
    return {
        "denominator": len(xs),
        "intercept": clean(intercept),
        "slope": clean(slope),
        "r_squared": clean(r * r) if r is not None else None,
        "contract": "ordinary least squares: comparator = intercept + slope * reference",
    }


def alignment_bins(xs: Sequence[float], ys: Sequence[float], bins: int = 10) -> dict[str, Any]:
    bucket_values: list[list[tuple[float, float]]] = [[] for _ in range(bins)]
    for x, y in zip(xs, ys):
        index = min(bins - 1, max(0, int(y * bins)))
        bucket_values[index].append((x, y))
    rows = []
    weighted_gap = 0.0
    max_gap = 0.0
    for index, values in enumerate(bucket_values):
        lower = index / bins
        upper = (index + 1) / bins
        if values:
            mean_reference = mean([pair[0] for pair in values])
            mean_comparator = mean([pair[1] for pair in values])
            assert mean_reference is not None and mean_comparator is not None
            gap = mean_comparator - mean_reference
            weighted_gap += len(values) * abs(gap)
            max_gap = max(max_gap, abs(gap))
        else:
            mean_reference = mean_comparator = gap = None
        rows.append(
            {
                "bin": index,
                "comparator_fraction_interval": f"[{lower:.1f},{upper:.1f}{']' if index == bins - 1 else ')'}",
                "denominator": len(values),
                "mean_reference_fraction": clean(mean_reference),
                "mean_comparator_fraction": clean(mean_comparator),
                "signed_gap": clean(gap),
            }
        )
    denominator = len(xs)
    return {
        "denominator": denominator,
        "binning_variable": "comparator_fraction_of_row_max",
        "bins": rows,
        "weighted_mean_absolute_bin_gap": clean(weighted_gap / denominator) if denominator else None,
        "maximum_absolute_nonempty_bin_gap": clean(max_gap) if denominator else None,
        "interpretation": "descriptive score-alignment curve; not probabilistic expected calibration error",
    }


def pair_metrics(
    pairs: Sequence[tuple[float, float]],
    thresholds: Sequence[float],
    include_alignment_bins: bool = False,
) -> dict[str, Any]:
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    errors = [y - x for x, y in pairs]
    abs_errors = [abs(value) for value in errors]
    n = len(pairs)
    exact = sum(value <= TOL for value in abs_errors)
    over = sum(value > TOL for value in errors)
    under = sum(value < -TOL for value in errors)
    result: dict[str, Any] = {
        "denominator": n,
        "mean_reference": clean(mean(xs)),
        "mean_comparator": clean(mean(ys)),
        "mean_signed_error_comparator_minus_reference": clean(mean(errors)),
        "mean_absolute_error": clean(mean(abs_errors)),
        "root_mean_squared_error": clean(math.sqrt(sum(value * value for value in errors) / n)) if n else None,
        "maximum_absolute_error": clean(max(abs_errors)) if abs_errors else None,
        "exact_agreement": {"count": exact, "rate": ratio(exact, n), "tolerance": TOL},
        "comparator_over_reference": {"count": over, "rate": ratio(over, n), "tolerance": TOL},
        "comparator_under_reference": {"count": under, "rate": ratio(under, n), "tolerance": TOL},
        "within_threshold": {
            f"{threshold:g}": {
                "count": sum(value <= threshold + TOL for value in abs_errors),
                "rate": ratio(sum(value <= threshold + TOL for value in abs_errors), n),
                "inclusive": True,
            }
            for threshold in thresholds
        },
        "pearson_r": clean(pearson(xs, ys)),
        "spearman_rho_average_ties": clean(spearman(xs, ys)),
        "lin_concordance_correlation_coefficient": clean(lin_ccc(xs, ys)),
        "linear_calibration": linear_calibration(xs, ys),
    }
    if include_alignment_bins:
        result["ten_bin_score_alignment"] = alignment_bins(xs, ys)
    return result


def pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def load_expert_rows() -> tuple[dict[str, dict[tuple[str, str], dict[str, Any]]], dict[str, Any]]:
    maps: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    metadata: dict[str, Any] = {}
    for exam, spec in GT_INPUTS.items():
        path = ROOT / spec["path"]
        actual_hash = sha256_file(path)
        if actual_hash != spec["sha256"]:
            raise ValueError(f"ground-truth hash mismatch: {spec['path']}")
        data = load_json(path)
        rows: dict[tuple[str, str], dict[str, Any]] = {}
        expert_totals: dict[str, float] = {}
        if exam == "ipho2025":
            for model, model_data in data["models"].items():
                total = 0.0
                for section in ("S1", "S2", "S3"):
                    for zero_index, item in enumerate(model_data["sections"][section]["scores"]):
                        case_id = f"IPHO-{section}-{zero_index + 1}"
                        value = float(item["human_score"])
                        total += value
                        rows[(model, case_id)] = {
                            "expert_score": value,
                            "expert_max_score": float(item["max_score"]),
                            "question_id": None,
                            "sub_question_id": None,
                            "section": section,
                            "item_index": zero_index + 1,
                            "locator": f"/models/{pointer_escape(model)}/sections/{section}/scores/{zero_index}",
                        }
                expert_totals[model] = total
        else:
            proportions = [float(value) for value in data["proportion"]]
            maxima = [float(value) for value in data["max_point"]]
            for model, model_data in data["models"].items():
                by_problem: dict[int, float] = defaultdict(float)
                for zero_index, item in enumerate(model_data["items"]):
                    qid = str(item["question_id"]).strip()
                    sid = str(item["sub_question_id"]).strip()
                    case_id = f"ICHO-{qid}-{sid}"
                    value = float(item["human_score_raw"])
                    by_problem[int(float(qid))] += value
                    rows[(model, case_id)] = {
                        "expert_score": value,
                        "expert_max_score": None,
                        "question_id": qid,
                        "sub_question_id": sid,
                        "section": None,
                        "item_index": zero_index + 1,
                        "locator": f"/models/{pointer_escape(model)}/items/{zero_index}",
                    }
                expert_totals[model] = sum(
                    by_problem[index] / maxima[index] * proportions[index]
                    for index in by_problem
                    if index < len(maxima) and maxima[index]
                )
        if len(rows) != spec["expected_rows"]:
            raise ValueError(f"{exam} expert row count {len(rows)} != {spec['expected_rows']}")
        maps[exam] = rows
        metadata[exam] = {
            "expert_totals": expert_totals,
            "proportion": data.get("proportion"),
            "max_point": data.get("max_point"),
        }
    return maps, metadata


def verify_frozen_file(path: str, expected_hash: str) -> None:
    target = ROOT / path
    if not target.is_file():
        raise ValueError(f"missing frozen input: {path}")
    actual = sha256_file(target)
    if actual != expected_hash:
        raise ValueError(f"frozen input hash mismatch: {path}: {actual} != {expected_hash}")


def normalized_status(expert: float, judge: float, maximum: float) -> str:
    if maximum <= 0:
        return "excluded_nonpositive_max_score"
    if expert < -TOL or expert > maximum + TOL:
        return "excluded_expert_score_outside_archived_judge_row_max"
    if judge < -TOL or judge > maximum + TOL:
        return "excluded_judge_score_outside_archived_judge_row_max"
    return "usable"


def reconstruct_total(exam: str, rows: Sequence[dict[str, Any]], expert_meta: dict[str, Any]) -> float:
    if exam == "ipho2025":
        return sum(float(row["judge_score"]) for row in rows)
    by_problem: dict[int, float] = defaultdict(float)
    for row in rows:
        by_problem[int(float(row["question_id"]))] += float(row["judge_score"])
    proportions = [float(value) for value in expert_meta[exam]["proportion"]]
    maxima = [float(value) for value in expert_meta[exam]["max_point"]]
    return sum(
        by_problem[index] / maxima[index] * proportions[index]
        for index in by_problem
        if index < len(maxima) and maxima[index]
    )


def replay() -> dict[str, Any]:
    for path, digest in SUPPORT_INPUTS.items():
        verify_frozen_file(path, digest)
    expert_maps, expert_meta = load_expert_rows()
    output_rows: list[dict[str, Any]] = []
    run_accounting = []
    run_archives: dict[str, dict[str, Any]] = {}
    candidate_totals: list[dict[str, Any]] = []

    for run in RUNS:
        directory = run["directory"]
        for filename, digest in run["hashes"].items():
            verify_frozen_file(f"{directory}/{filename}", digest)
        result_path = ROOT / directory / "judge_results.json"
        raw_rows = load_json(result_path)
        expected = expert_maps[run["exam"]]
        seen = Counter((str(row.get("model_under_test")), str(row.get("case_id"))) for row in raw_rows)
        duplicate_keys = sorted([list(key) for key, count in seen.items() if count > 1])
        missing_keys = sorted([list(key) for key in set(expected) - set(seen)])
        unexpected_keys = sorted([list(key) for key in set(seen) - set(expected)])
        stored_error_rows = sum(bool(row.get("error")) for row in raw_rows)
        missing_numeric_score_rows = sum(not finite_number(row.get("score")) for row in raw_rows)
        missing_numeric_max_rows = sum(not finite_number(row.get("max_score")) for row in raw_rows)
        normalized_statuses = Counter()
        run_output_rows: list[dict[str, Any]] = []

        for source_index, raw in enumerate(raw_rows):
            key = (str(raw.get("model_under_test")), str(raw.get("case_id")))
            if key not in expected or bool(raw.get("error")):
                continue
            if not finite_number(raw.get("score")) or not finite_number(raw.get("max_score")):
                continue
            expert = expected[key]
            expert_score = float(expert["expert_score"])
            judge_score = float(raw["score"])
            maximum = float(raw["max_score"])
            status = normalized_status(expert_score, judge_score, maximum)
            normalized_statuses[status] += 1
            if status == "usable":
                expert_fraction: float | None = expert_score / maximum
                judge_fraction: float | None = judge_score / maximum
            else:
                expert_fraction = judge_fraction = None
            record = {
                "record_id": f"calibration::{run['judge_id']}::{run['exam']}::{key[0]}::{key[1]}",
                "judge_id": run["judge_id"],
                "judge_model": run["judge_model"],
                "exam": run["exam"],
                "run_id": run["run_id"],
                "run_directory": directory,
                "source_row_index": source_index,
                "model_under_test": key[0],
                "case_id": key[1],
                "question_id": expert["question_id"],
                "sub_question_id": expert["sub_question_id"],
                "section": expert["section"],
                "item_index": expert["item_index"],
                "max_score": maximum,
                "expert_score": expert_score,
                "judge_score": judge_score,
                "signed_error_judge_minus_expert": judge_score - expert_score,
                "absolute_error": abs(judge_score - expert_score),
                "expert_fraction_of_max": expert_fraction,
                "judge_fraction_of_max": judge_fraction,
                "normalized_pair_status": status,
                "raw_judgment_sha256": canonical_json_sha(raw),
                "judge_results_sha256": run["hashes"]["judge_results.json"],
                "ground_truth_path": GT_INPUTS[run["exam"]]["path"],
                "ground_truth_sha256": GT_INPUTS[run["exam"]]["sha256"],
                "ground_truth_locator": expert["locator"],
            }
            output_rows.append(record)
            run_output_rows.append(record)

        run_output_rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in run_output_rows:
            run_output_rows_by_model[row["model_under_test"]].append(row)
        for model in sorted(expert_meta[run["exam"]]["expert_totals"]):
            model_rows = run_output_rows_by_model[model]
            judge_total = reconstruct_total(run["exam"], model_rows, expert_meta)
            expert_total = float(expert_meta[run["exam"]]["expert_totals"][model])
            candidate_totals.append(
                {
                    "run_id": run["run_id"],
                    "judge_id": run["judge_id"],
                    "judge_model": run["judge_model"],
                    "exam": run["exam"],
                    "model_under_test": model,
                    "exam_scale": GT_INPUTS[run["exam"]]["exam_scale"],
                    "expert_total": clean(expert_total),
                    "replayed_judge_total": clean(judge_total),
                    "signed_error_judge_minus_expert": clean(judge_total - expert_total),
                    "absolute_error": clean(abs(judge_total - expert_total)),
                    "row_denominator": len(model_rows),
                }
            )
        accounting = {
            "run_id": run["run_id"],
            "judge_id": run["judge_id"],
            "judge_model": run["judge_model"],
            "exam": run["exam"],
            "expected_judgment_rows": run["expected_rows"],
            "stored_judgment_rows": len(raw_rows),
            "unique_judgment_keys": len(seen),
            "duplicate_key_count": len(duplicate_keys),
            "duplicate_keys": duplicate_keys,
            "missing_expected_key_count": len(missing_keys),
            "missing_expected_keys": missing_keys,
            "unexpected_key_count": len(unexpected_keys),
            "unexpected_keys": unexpected_keys,
            "stored_error_rows": stored_error_rows,
            "missing_numeric_score_rows": missing_numeric_score_rows,
            "missing_numeric_max_score_rows": missing_numeric_max_rows,
            "usable_raw_pairs": len(run_output_rows),
            "normalized_pair_status_counts": dict(sorted(normalized_statuses.items())),
        }
        run_accounting.append(accounting)
        run_archives[run["run_id"]] = {
            "spec": run,
            "rows": run_output_rows,
        }

    if len(output_rows) != 1050 or len({row["record_id"] for row in output_rows}) != 1050:
        raise ValueError("replay did not produce exactly 1,050 unique row records")
    output_rows.sort(key=lambda row: (row["judge_id"], row["exam"], row["source_row_index"]))
    return {
        "rows": output_rows,
        "run_accounting": run_accounting,
        "run_archives": run_archives,
        "candidate_totals": candidate_totals,
        "expert_meta": expert_meta,
    }


def csv_scalar(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return repr(value)
    return value


def csv_bytes(rows: Sequence[dict[str, Any]], include_header: bool = True) -> bytes:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n", extrasaction="raise")
    if include_header:
        writer.writeheader()
    for row in rows:
        writer.writerow({key: csv_scalar(row.get(key)) for key in CSV_FIELDS})
    return handle.getvalue().encode("utf-8")


def metric_bundle(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    raw_pairs = [(float(row["expert_score"]), float(row["judge_score"])) for row in rows]
    normalized_rows = [row for row in rows if row["normalized_pair_status"] == "usable"]
    normalized_pairs = [
        (float(row["expert_fraction_of_max"]), float(row["judge_fraction_of_max"]))
        for row in normalized_rows
    ]
    exclusions = Counter(
        row["normalized_pair_status"] for row in rows if row["normalized_pair_status"] != "usable"
    )
    return {
        "raw_points": pair_metrics(raw_pairs, thresholds=(0.1, 0.5, 1.0)),
        "fraction_of_archived_row_max": pair_metrics(
            normalized_pairs, thresholds=(0.01, 0.05, 0.10), include_alignment_bins=True
        ),
        "normalized_pair_accounting": {
            "candidate_rows": len(rows),
            "usable": len(normalized_rows),
            "excluded": len(rows) - len(normalized_rows),
            "exclusion_reasons": dict(sorted(exclusions.items())),
        },
    }


def total_metric_bundle(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    raw_pairs = [(float(row["expert_total"]), float(row["replayed_judge_total"])) for row in rows]
    normalized_pairs = [
        (
            float(row["expert_total"]) / float(row["exam_scale"]),
            float(row["replayed_judge_total"]) / float(row["exam_scale"]),
        )
        for row in rows
    ]
    return {
        "exam_points": pair_metrics(raw_pairs, thresholds=(0.5, 1.0, 2.0)),
        "fraction_of_exam_scale": pair_metrics(
            normalized_pairs, thresholds=(0.01, 0.02, 0.05), include_alignment_bins=True
        ),
    }


def interjudge_item_metrics(rows: Sequence[dict[str, Any]], exam: str | None = None) -> dict[str, Any]:
    selected = [row for row in rows if exam is None or row["exam"] == exam]
    by_judge: dict[str, dict[tuple[str, str, str], dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        key = (row["exam"], row["model_under_test"], row["case_id"])
        by_judge[row["judge_id"]][key] = row
    ids = sorted(by_judge)
    if ids != ["gemini_3_1_pro_preview", "gemini_3_5_flash"]:
        raise ValueError(f"unexpected judge IDs for interjudge replay: {ids}")
    keys_a, keys_b = set(by_judge[ids[0]]), set(by_judge[ids[1]])
    shared = sorted(keys_a & keys_b)
    raw_pairs = [
        (float(by_judge[ids[0]][key]["judge_score"]), float(by_judge[ids[1]][key]["judge_score"]))
        for key in shared
    ]
    normalized_pairs = []
    max_mismatches = []
    direction_counts = Counter()
    for key in shared:
        a, b = by_judge[ids[0]][key], by_judge[ids[1]][key]
        if abs(float(a["max_score"]) - float(b["max_score"])) > TOL:
            max_mismatches.append({"key": list(key), "judge_a_max": a["max_score"], "judge_b_max": b["max_score"]})
            continue
        maximum = float(a["max_score"])
        if maximum > 0 and 0 <= float(a["judge_score"]) <= maximum and 0 <= float(b["judge_score"]) <= maximum:
            normalized_pairs.append((float(a["judge_score"]) / maximum, float(b["judge_score"]) / maximum))
        ea = float(a["judge_score"]) - float(a["expert_score"])
        eb = float(b["judge_score"]) - float(b["expert_score"])
        sign_a = 0 if abs(ea) <= TOL else 1 if ea > 0 else -1
        sign_b = 0 if abs(eb) <= TOL else 1 if eb > 0 else -1
        if sign_a == 0 and sign_b == 0:
            direction_counts["both_exact"] += 1
        elif sign_a == sign_b:
            direction_counts["same_nonzero_direction"] += 1
        elif sign_a == 0 or sign_b == 0:
            direction_counts["one_exact_one_nonzero"] += 1
        else:
            direction_counts["opposite_nonzero_directions"] += 1
    return {
        "judge_a_reference": ids[0],
        "judge_b_comparator": ids[1],
        "expected_keys": len(keys_a | keys_b),
        "shared_pair_denominator": len(shared),
        "missing_from_judge_a": len(keys_b - keys_a),
        "missing_from_judge_b": len(keys_a - keys_b),
        "max_score_mismatch_count": len(max_mismatches),
        "max_score_mismatches": max_mismatches,
        "raw_points": pair_metrics(raw_pairs, thresholds=(0.1, 0.5, 1.0)),
        "fraction_of_archived_row_max": pair_metrics(
            normalized_pairs, thresholds=(0.01, 0.05, 0.10), include_alignment_bins=False
        ),
        "relative_to_expert_error_direction": {
            "denominator": len(shared),
            **dict(sorted(direction_counts.items())),
        },
    }


def interjudge_total_metrics(candidate_totals: Sequence[dict[str, Any]], exam: str | None = None) -> dict[str, Any]:
    selected = [row for row in candidate_totals if exam is None or row["exam"] == exam]
    by_judge: dict[str, dict[tuple[str, str], dict[str, Any]]] = defaultdict(dict)
    for row in selected:
        by_judge[row["judge_id"]][(row["exam"], row["model_under_test"])] = row
    ids = sorted(by_judge)
    keys = sorted(set(by_judge[ids[0]]) & set(by_judge[ids[1]]))
    raw_pairs = [
        (
            float(by_judge[ids[0]][key]["replayed_judge_total"]),
            float(by_judge[ids[1]][key]["replayed_judge_total"]),
        )
        for key in keys
    ]
    normalized_pairs = [
        (
            float(by_judge[ids[0]][key]["replayed_judge_total"]) / float(by_judge[ids[0]][key]["exam_scale"]),
            float(by_judge[ids[1]][key]["replayed_judge_total"]) / float(by_judge[ids[1]][key]["exam_scale"]),
        )
        for key in keys
    ]
    return {
        "judge_a_reference": ids[0],
        "judge_b_comparator": ids[1],
        "shared_pair_denominator": len(keys),
        "exam_points": pair_metrics(raw_pairs, thresholds=(0.5, 1.0, 2.0)),
        "fraction_of_exam_scale": pair_metrics(normalized_pairs, thresholds=(0.01, 0.02, 0.05)),
    }


def source_inventory() -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for exam, spec in GT_INPUTS.items():
        entries.append({"role": "expert_ground_truth", "exam": exam, "path": spec["path"], "sha256": spec["sha256"]})
    for path, digest in SUPPORT_INPUTS.items():
        entries.append({"role": "archived_replay_support", "path": path, "sha256": digest})
    for run in RUNS:
        for filename, digest in run["hashes"].items():
            entries.append(
                {
                    "role": "archived_run_artifact",
                    "run_id": run["run_id"],
                    "path": f"{run['directory']}/{filename}",
                    "sha256": digest,
                }
            )
    return entries


def report_object(replayed: dict[str, Any]) -> dict[str, Any]:
    rows = replayed["rows"]
    totals = replayed["candidate_totals"]
    by_run_metrics = {}
    by_judge_metrics = {}
    by_run_total_metrics = {}
    by_judge_total_metrics = {}
    for run in RUNS:
        run_rows = [row for row in rows if row["run_id"] == run["run_id"]]
        run_totals = [row for row in totals if row["run_id"] == run["run_id"]]
        by_run_metrics[run["run_id"]] = metric_bundle(run_rows)
        by_run_total_metrics[run["run_id"]] = total_metric_bundle(run_totals)
    for judge_id in sorted({row["judge_id"] for row in rows}):
        judge_rows = [row for row in rows if row["judge_id"] == judge_id]
        judge_totals = [row for row in totals if row["judge_id"] == judge_id]
        by_judge_metrics[judge_id] = metric_bundle(judge_rows)
        by_judge_total_metrics[judge_id] = total_metric_bundle(judge_totals)

    normalized_exclusions = [
        {
            "record_id": row["record_id"],
            "exam": row["exam"],
            "judge_id": row["judge_id"],
            "model_under_test": row["model_under_test"],
            "case_id": row["case_id"],
            "expert_score": row["expert_score"],
            "judge_score": row["judge_score"],
            "archived_judge_row_max_score": row["max_score"],
            "reason": row["normalized_pair_status"],
        }
        for row in rows
        if row["normalized_pair_status"] != "usable"
    ]
    csv_payload = csv_bytes(rows)
    report = {
        "schema_version": SCHEMA_VERSION,
        "fixed_audit_date": FIXED_DATE,
        "status": "PASS",
        "execution": {"offline": True, "api_calls": 0, "key_reads": 0},
        "scope": "Two archived Gemini-family judges on five saved model answers each for IPhO 2025 and IChO 2025.",
        "row_accounting": {
            "distinct_expert_rows": 525,
            "expected_stored_judgments": 1050,
            "observed_stored_judgments": sum(item["stored_judgment_rows"] for item in replayed["run_accounting"]),
            "usable_raw_expert_judge_pairs": len(rows),
            "usable_normalized_expert_judge_pairs": sum(row["normalized_pair_status"] == "usable" for row in rows),
            "excluded_normalized_pairs": len(normalized_exclusions),
            "candidate_total_pairs": len(totals),
            "per_run": replayed["run_accounting"],
        },
        "normalized_pair_exclusions": normalized_exclusions,
        "item_level_expert_agreement": {
            "by_run": by_run_metrics,
            "by_judge_across_exams": by_judge_metrics,
        },
        "candidate_total_expert_agreement": {
            "rows": totals,
            "by_run": by_run_total_metrics,
            "by_judge_across_exams": by_judge_total_metrics,
        },
        "interjudge_agreement": {
            "item_level": {
                "ipho2025": interjudge_item_metrics(rows, "ipho2025"),
                "icho2025": interjudge_item_metrics(rows, "icho2025"),
                "combined": interjudge_item_metrics(rows, None),
            },
            "candidate_total_level": {
                "ipho2025": interjudge_total_metrics(totals, "ipho2025"),
                "icho2025": interjudge_total_metrics(totals, "icho2025"),
                "combined": interjudge_total_metrics(totals, None),
            },
        },
        "archive_consistency": {
            "contract": "all metrics and totals are recomputed directly from the four hash-frozen final judge_results.json arrays and two hash-frozen expert ground truths",
            "aggregation_implementation_reference": {
                "path": "science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/science_arena_judge.py",
                "sha256": SUPPORT_INPUTS["science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/science_arena_judge.py"],
            },
        },
        "metric_contract": {
            "raw_item_denominator": "all matched finite raw expert/judge score pairs, including source-scale inconsistencies",
            "normalized_item_denominator": "matched pairs with positive archived judge-row max and both scores in [0,max]",
            "signed_error": "comparator minus reference; judge minus expert for expert agreement and Gemini 3.5 Flash minus Gemini 3.1 Pro Preview for interjudge agreement",
            "exact_tolerance": TOL,
            "thresholds_are_inclusive": True,
            "rmse": "sqrt(mean((comparator-reference)^2))",
            "pearson": "Pearson product-moment correlation on the stated denominator",
            "spearman": "Pearson correlation of average ranks; ties receive their average rank",
            "lin_ccc": "2*population_covariance/(population_var_x+population_var_y+(mean_x-mean_y)^2)",
            "linear_calibration": "ordinary least squares comparator = intercept + slope*reference",
            "ten_bin_score_alignment": "equal-width bins by comparator fraction; descriptive weighted absolute mean-bin gap, not probabilistic ECE",
            "icho_candidate_total": "sum raw rows within each problem, divide by frozen problem maximum, multiply by frozen official problem proportion, then sum (/60)",
            "ipho_candidate_total": "sum 40 raw item scores (/30)",
        },
        "source_inventory": source_inventory(),
        "artifacts": {
            "replay_script": {"path": REL_SCRIPT, "sha256": sha256_file(ROOT / REL_SCRIPT)},
            "row_csv": {"path": REL_CSV, "sha256": sha256_bytes(csv_payload), "data_rows": len(rows), "bytes": len(csv_payload)},
        },
        "limitations": [
            "Both archived judges are Gemini-family models; this is not a cross-family judge robustness study.",
            "The calibration covers only five saved candidate answers on IPhO 2025 and five on IChO 2025; candidate-total denominators are five per judge/exam.",
            "The replay evaluates archived final merged/reparsed result arrays. It does not resend requests and cannot measure current endpoint behavior or run-to-run stochasticity.",
            "IChO row ICHO-2-2.3c for gpt-5 has expert_score=10.0 but archived judge-row max_score=6.0. Both judge copies are retained in raw-point metrics and explicitly excluded from normalized expert/judge metrics; no score or maximum is repaired.",
            "Candidate-total agreement can look strong even when individual rubric rows disagree; item-level and total-level denominators and metrics are therefore reported separately.",
            "Correlation and regression on five candidate totals per judge/exam are descriptive and statistically fragile; no confidence interval or significance claim is made.",
        ],
    }
    all_pass = all(
        item["stored_judgment_rows"] == item["expected_judgment_rows"]
        and item["unique_judgment_keys"] == item["expected_judgment_rows"]
        and item["duplicate_key_count"] == 0
        and item["missing_expected_key_count"] == 0
        and item["unexpected_key_count"] == 0
        and item["stored_error_rows"] == 0
        and item["missing_numeric_score_rows"] == 0
        and item["missing_numeric_max_score_rows"] == 0
        for item in replayed["run_accounting"]
    )
    if not all_pass:
        report["status"] = "FAIL"
    return report


def report_json_bytes(replayed: dict[str, Any]) -> bytes:
    return (json.dumps(report_object(replayed), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def report_markdown(replayed: dict[str, Any]) -> str:
    report = report_object(replayed)
    lines = [
        "# Archived two-Gemini-judge calibration replay",
        "",
        f"Fixed audit date: {FIXED_DATE}  ",
        f"Status: **{report['status']}**  ",
        "Execution: fully local; 0 API calls; 0 key reads.",
        "",
        "## Row accounting",
        "",
        "The replay finds exactly **1,050 archived judgment rows** over **525 distinct expert rows**. "
        "All 1,050 rows have a unique expected key, a finite score, a finite archived row maximum, "
        "and no stored error. Raw expert/judge metrics therefore use 1,050/1,050 pairs. "
        "Normalized expert/judge metrics use 1,048/1,050 pairs because one underlying IChO source-scale "
        "mismatch appears once for each judge and is not repaired.",
        "",
        "| Judge | Exam | Stored / expected | Raw pairs | Normalized pairs | Missing | Errors |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for item in report["row_accounting"]["per_run"]:
        usable_norm = item["normalized_pair_status_counts"].get("usable", 0)
        lines.append(
            f"| {item['judge_id']} | {item['exam']} | {item['stored_judgment_rows']} / {item['expected_judgment_rows']} "
            f"| {item['usable_raw_pairs']} | {usable_norm} | {item['missing_expected_key_count']} | {item['stored_error_rows']} |"
        )
    lines += [
        "",
        "## Agreement with experts",
        "",
        "Item-level raw-point metrics retain every archived score. Fraction-of-row-maximum metrics "
        "exclude only the explicitly listed invalid source-scale pair(s). Candidate-total metrics "
        "use five saved answer models per judge/exam.",
        "",
        "| Judge | Exam | Item n | Raw MAE | Raw RMSE | Exact | Normalized n | Normalized MAE | Pearson (norm.) | Lin CCC (norm.) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        metrics = report["item_level_expert_agreement"]["by_run"][run["run_id"]]
        raw = metrics["raw_points"]
        norm = metrics["fraction_of_archived_row_max"]
        lines.append(
            f"| {run['judge_id']} | {run['exam']} | {raw['denominator']} | {raw['mean_absolute_error']:.6f} "
            f"| {raw['root_mean_squared_error']:.6f} | {raw['exact_agreement']['count']} ({raw['exact_agreement']['rate']:.3f}) "
            f"| {norm['denominator']} | {norm['mean_absolute_error']:.6f} | {norm['pearson_r']:.6f} "
            f"| {norm['lin_concordance_correlation_coefficient']:.6f} |"
        )
    lines += [
        "",
        "| Judge | Exam | Total n | Total MAE | Mean signed error | Maximum absolute error |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for run in RUNS:
        metrics = report["candidate_total_expert_agreement"]["by_run"][run["run_id"]]["exam_points"]
        lines.append(
            f"| {run['judge_id']} | {run['exam']} | {metrics['denominator']} | {metrics['mean_absolute_error']:.6f} "
            f"| {metrics['mean_signed_error_comparator_minus_reference']:.6f} | {metrics['maximum_absolute_error']:.6f} |"
        )
    lines += [
        "",
        "## Interjudge agreement",
        "",
        "Here Gemini 3.1 Pro Preview is the reference and Gemini 3.5 Flash is the comparator; this "
        "orientation affects only the sign of signed differences.",
        "",
        "| Exam | Shared item n | Raw MAE | Exact | Normalized MAE | Pearson (norm.) | Shared total n | Total MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for exam in ("ipho2025", "icho2025", "combined"):
        item = report["interjudge_agreement"]["item_level"][exam]
        total = report["interjudge_agreement"]["candidate_total_level"][exam]
        raw = item["raw_points"]
        norm = item["fraction_of_archived_row_max"]
        lines.append(
            f"| {exam} | {item['shared_pair_denominator']} | {raw['mean_absolute_error']:.6f} "
            f"| {raw['exact_agreement']['count']} ({raw['exact_agreement']['rate']:.3f}) "
            f"| {norm['mean_absolute_error']:.6f} | {norm['pearson_r']:.6f} "
            f"| {total['shared_pair_denominator']} | {total['exam_points']['mean_absolute_error']:.6f} |"
        )
    lines += [
        "",
        "## Explicit source-scale exclusion",
        "",
        "The frozen expert row `gpt-5 / ICHO-2-2.3c` has raw expert score 10.0, while each archived "
        "judge row declares `max_score=6.0`. The two raw score comparisons remain in the 1,050-row "
        "metrics; both are excluded from normalized expert/judge metrics. The replay does not guess "
        "which field should change.",
        "",
        "## Reproduction",
        "",
        "```bash",
        f"python3 {REL_SCRIPT} --check {REL_CSV}",
        "```",
        "",
        f"Row CSV SHA-256: `{report['artifacts']['row_csv']['sha256']}`. The JSON report contains "
        "complete metric definitions, denominators, per-bin calibration summaries, candidate totals, "
        "source hashes, missingness, and exclusion records.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def sidecar_text() -> str:
    artifact_paths = [REL_SCRIPT, REL_CSV, REL_JSON, REL_MD]
    input_paths = [item["path"] for item in source_inventory()]
    paths = artifact_paths + input_paths
    return "\n".join(f"{sha256_file(ROOT / path)}  {path}" for path in paths) + "\n"


def check_csv(path: Path) -> dict[str, Any]:
    replayed = replay()
    expected_csv = csv_bytes(replayed["rows"])
    actual_csv = path.read_bytes()
    if actual_csv != expected_csv:
        raise ValueError(
            f"CSV differs from deterministic replay: actual={sha256_bytes(actual_csv)} "
            f"expected={sha256_bytes(expected_csv)}"
        )
    with path.open("r", encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))
    if len(parsed) != 1050 or set(parsed[0]) != set(CSV_FIELDS):
        raise ValueError("CSV strict header/row-count validation failed")
    report = report_object(replayed)
    if report["status"] != "PASS":
        raise ValueError("replay report status is not PASS")
    if (ROOT / REL_JSON).is_file() and (ROOT / REL_JSON).read_bytes() != report_json_bytes(replayed):
        raise ValueError("checked-in JSON report differs from deterministic replay")
    if (ROOT / REL_MD).is_file() and (ROOT / REL_MD).read_bytes() != report_markdown(replayed).encode("utf-8"):
        raise ValueError("checked-in Markdown report differs from deterministic replay")
    if (ROOT / REL_SIDECAR).is_file() and (ROOT / REL_SIDECAR).read_bytes() != sidecar_text().encode("utf-8"):
        raise ValueError("checked-in SHA sidecar differs from deterministic replay")
    return {
        "status": "PASS",
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": sha256_bytes(actual_csv),
        "judgment_rows": len(parsed),
        "distinct_expert_rows": 525,
        "usable_raw_pairs": report["row_accounting"]["usable_raw_expert_judge_pairs"],
        "usable_normalized_pairs": report["row_accounting"]["usable_normalized_expert_judge_pairs"],
        "api_calls": 0,
        "byte_identical_to_frozen_archive_replay": True,
    }


def emit_bytes(data: bytes) -> None:
    sys.stdout.buffer.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-csv", action="store_true")
    group.add_argument("--emit-json", action="store_true")
    group.add_argument("--emit-md", action="store_true")
    group.add_argument("--emit-sidecar", action="store_true")
    group.add_argument("--check", metavar="CSV_PATH")
    parser.add_argument("--start", type=int, default=0, help="first CSV data row to emit (inclusive)")
    parser.add_argument("--stop", type=int, default=None, help="last CSV data row to emit (exclusive)")
    args = parser.parse_args()

    if args.emit_sidecar:
        sys.stdout.write(sidecar_text())
        return
    if args.check:
        result = check_csv(Path(args.check).expanduser().resolve())
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return

    replayed = replay()
    if args.emit_csv:
        start = args.start
        stop = len(replayed["rows"]) if args.stop is None else args.stop
        if start < 0 or stop < start or stop > len(replayed["rows"]):
            parser.error(f"invalid CSV slice [{start}:{stop}] for {len(replayed['rows'])} rows")
        emit_bytes(csv_bytes(replayed["rows"][start:stop], include_header=start == 0))
    elif args.emit_json:
        emit_bytes(report_json_bytes(replayed))
    elif args.emit_md:
        sys.stdout.write(report_markdown(replayed))


if __name__ == "__main__":
    main()
