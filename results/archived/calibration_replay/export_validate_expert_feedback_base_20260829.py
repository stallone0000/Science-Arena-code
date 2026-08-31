#!/usr/bin/env python3
"""Export and validate the archived 525-row expert-feedback base.

This is an offline, deterministic replay.  It reads the two frozen canonical
ground-truth JSON files and the eleven original files they cite.  It never
calls an API and never infers a comment, tag, credit status, conflict, or other
semantic annotation that the archived sources do not contain.

By default, generated artifacts are emitted to stdout.  The checked-in files
were added with ``apply_patch``; this script does not write them.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


BUNDLE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "science_arena_repro_package_20260530_lite"
SCHEMA_VERSION = "science_arena_expert_feedback_base_v1"
REPORT_VERSION = "science_arena_expert_feedback_base_validation_v1"
FIXED_DATE = "2026-08-29"

REL_SCHEMA = "audits/cr_rebuttal_local_supplements_20260829/expert_feedback_base_20260829.schema.json"
REL_SCRIPT = "audits/cr_rebuttal_local_supplements_20260829/export_validate_expert_feedback_base_20260829.py"
REL_JSONL = "audits/cr_rebuttal_local_supplements_20260829/expert_feedback_base_20260829.jsonl"
REL_REPORT_JSON = "audits/cr_rebuttal_local_supplements_20260829/expert_feedback_base_validation_20260829.json"
REL_REPORT_MD = "audits/cr_rebuttal_local_supplements_20260829/expert_feedback_base_validation_20260829.md"
REL_SIDECAR = "audits/cr_rebuttal_local_supplements_20260829/expert_feedback_base_20260829.sha256"

GROUND_TRUTH = {
    "ipho2025": (
        "science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/ground_truth/ipho2025_expert_ground_truth.json",
        "0719eb5cf59843fb44d3e8125f63a95b4711425590e1b0e77c40dce1ba2406db",
    ),
    "icho2025": (
        "science_arena_repro_package_20260530_lite/experiments/judge_pipeline_v2/ground_truth/icho2025_expert_ground_truth.json",
        "5ed62c788ec71b20f5f7b63d70d6cede6e12db3da6bb938e847c96500cb40d66",
    ),
}

# Exactly the eleven original grading files referenced by the canonical rows.
# ``report.pdf`` and comparison PNGs exist nearby but are not row sources.
ORIGINAL_SOURCE_HASHES = {
    "science_arena_repro_package_20260530_lite/artifacts/expert_grading/ipho2025/original/deepseek-v3.1.docx": "2c31f2cf96a76415cfb23dc04410e9fdb3eae38f3b177f45f9ba4fc60f37f0ed",
    "science_arena_repro_package_20260530_lite/artifacts/expert_grading/ipho2025/original/gemini-2.5-pro.docx": "f98a4f3d8420617fa8696c09d894f34cd809607d186c69e72b6cf49baf9ba9c9",
    "science_arena_repro_package_20260530_lite/artifacts/expert_grading/ipho2025/original/gpt5-S1.pdf": "c807295f230dd2c36f0acaabfa0a5c5e593b5eff369b48d1fbaccb366020b666",
    "science_arena_repro_package_20260530_lite/artifacts/expert_grading/ipho2025/original/gtp5-S2.pdf": "05ca48619aa8a6be11d26e00bdd7dedd5396c90068116e3e93772cb780dd3603",
    "science_arena_repro_package_20260530_lite/artifacts/expert_grading/ipho2025/original/gpt5-S3.pdf": "2f4c3b41f1506b6808c04754c1e30f8d3e336fc406ca4846002ef9e0dc0e9bef",
    "science_arena_repro_package_20260530_lite/artifacts/expert_grading/ipho2025/original/summary.txt": "45423c1303d9739b7fe5bce5eb1cbe7ba77eba6f3125fdac57ce102d106390b1",
    "science_arena_repro_package_20260530_lite/artifacts/expert_grading/icho2025/original/GPT5_grading_with_judge.json": "029d61a11b6966ca57aa74854c619bfacd16e9c732cba0590db148a72683241a",
    "science_arena_repro_package_20260530_lite/artifacts/expert_grading/icho2025/original/Qwen3-235B-2507_grading_with_judge.json": "1d6940cf28ca4e161d4cb2109d122e91f56831e64162f20240707221aad28d14",
    "science_arena_repro_package_20260530_lite/artifacts/expert_grading/icho2025/original/glm-4.5_grading_with_judge.json": "61a9e2acfbc9dd7e000a19acaa3bf6dc20e1e471c97ae11402669c1d539e19ed",
    "science_arena_repro_package_20260530_lite/artifacts/expert_grading/icho2025/original/human_eval_deepseek-v3.1_solution_20250927-120001.json": "11887dc315e856fe32d5825f0f8a32b28b53948478a9928c35f99280ad687564",
    "science_arena_repro_package_20260530_lite/artifacts/expert_grading/icho2025/original/human_eval_gemini-2.5-pro_solution_20250926-203600.json": "d5adef5d3a0e5be6b29eeb3b1ccae029cd8d200692ca3cb43cb5426b41b027dd",
}

EXPECTED_COUNTS = {
    "total": 525,
    "comments": 410,
    "missing_comments": 115,
    "by_exam": {
        "ipho2025": {"rows": 200, "comments": 85, "missing_comments": 115},
        "icho2025": {"rows": 325, "comments": 325, "missing_comments": 0},
    },
}

TOP_KEYS = {
    "schema_version", "record_id", "exam", "model_under_test", "case_id",
    "question_id", "sub_question_id", "section", "item_index", "label",
    "human_score", "max_score", "human_comment", "comment_status",
    "provenance", "semantic_annotations",
}
PROVENANCE_KEYS = {"canonical_ground_truth", "score_source", "comment_source"}
SOURCE_REF_KEYS = {"path", "sha256", "locator"}
ANNOTATION_KEYS = {
    "annotation_status", "credit_status", "normalized_error_tags",
    "normalized_ability_tags", "evidence", "uncertainty", "discrepancy",
    "conflict",
}
NULL_ANNOTATION_KEYS = ANNOTATION_KEYS - {"annotation_status"}


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


def workspace_path_from_package_relative(path: str) -> str:
    return f"science_arena_repro_package_20260530_lite/{path}"


def pointer_escape(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def source_ref(path: str, locator: str) -> dict[str, str]:
    expected = ORIGINAL_SOURCE_HASHES.get(path)
    if expected is None:
        raise ValueError(f"unrecognized original source path: {path}")
    actual = sha256_file(ROOT / path)
    if actual != expected:
        raise ValueError(f"source SHA mismatch for {path}: {actual} != {expected}")
    return {"path": path, "sha256": actual, "locator": locator}


def canonical_ref(exam: str, locator: str) -> dict[str, str]:
    path, expected = GROUND_TRUTH[exam]
    actual = sha256_file(ROOT / path)
    if actual != expected:
        raise ValueError(f"ground-truth SHA mismatch for {path}: {actual} != {expected}")
    return {"path": path, "sha256": actual, "locator": locator}


def empty_annotations() -> dict[str, Any]:
    return {
        "annotation_status": "not_provided_in_archived_source",
        "credit_status": None,
        "normalized_error_tags": None,
        "normalized_ability_tags": None,
        "evidence": None,
        "uncertainty": None,
        "discrepancy": None,
        "conflict": None,
    }


def normalized_comment(value: Any) -> tuple[str | None, str]:
    # Preserve a real nonempty source string byte-for-character.  Empty or
    # whitespace-only values are represented as unavailable, never invented.
    if isinstance(value, str) and value.strip():
        return value, "available_nonempty"
    return None, "unavailable_in_source"


def build_ipho_rows(gt: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in sorted(gt["models"]):
        model_data = gt["models"][model]
        for section in ("S1", "S2", "S3"):
            scores = model_data.get("sections", {}).get(section, {}).get("scores", [])
            for zero_index, item in enumerate(scores):
                item_index = zero_index + 1
                case_id = f"IPHO-{section}-{item_index}"
                locator = f"section={section};item_index={item_index};label={item.get('label')}"
                score_path = workspace_path_from_package_relative(str(item["source_file"]))
                comment, comment_status = normalized_comment(item.get("human_comment"))
                comment_path_value = item.get("comment_source_file") or item.get("source_file")
                comment_path = workspace_path_from_package_relative(str(comment_path_value))
                comment_ref = source_ref(comment_path, locator) if comment is not None else None
                canonical_locator = (
                    f"/models/{pointer_escape(model)}/sections/{section}/scores/{zero_index}"
                )
                rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "record_id": f"ipho2025::{model}::{case_id}",
                        "exam": "ipho2025",
                        "model_under_test": model,
                        "case_id": case_id,
                        "question_id": None,
                        "sub_question_id": None,
                        "section": section,
                        "item_index": item_index,
                        "label": str(item.get("label")) if item.get("label") is not None else None,
                        "human_score": float(item["human_score"]),
                        "max_score": float(item["max_score"]),
                        "human_comment": comment,
                        "comment_status": comment_status,
                        "provenance": {
                            "canonical_ground_truth": canonical_ref("ipho2025", canonical_locator),
                            "score_source": source_ref(score_path, locator),
                            "comment_source": comment_ref,
                        },
                        "semantic_annotations": empty_annotations(),
                    }
                )
    return rows


def build_icho_rows(gt: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in sorted(gt["models"]):
        items = gt["models"][model].get("items", [])
        for zero_index, item in enumerate(items):
            item_index = zero_index + 1
            qid = str(item.get("question_id", "")).strip()
            sid = str(item.get("sub_question_id", "")).strip()
            case_id = f"ICHO-{qid}-{sid}"
            source_index = item.get("source_index")
            source_index_text = "null" if source_index is None else str(source_index)
            locator = (
                f"source_index={source_index_text};question_id={qid};sub_question_id={sid}"
            )
            raw_path = workspace_path_from_package_relative(str(item["source_file"]))
            comment, comment_status = normalized_comment(item.get("human_comments"))
            canonical_locator = f"/models/{pointer_escape(model)}/items/{zero_index}"
            rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "record_id": f"icho2025::{model}::{case_id}",
                    "exam": "icho2025",
                    "model_under_test": model,
                    "case_id": case_id,
                    "question_id": qid,
                    "sub_question_id": sid,
                    "section": None,
                    "item_index": item_index,
                    "label": None,
                    "human_score": float(item["human_score_raw"]),
                    # The frozen IChO expert ground truth stores problem-level
                    # maxima, not a reliable per-sub-question maximum.  Do not
                    # borrow/invent one from a judge artifact.
                    "max_score": None,
                    "human_comment": comment,
                    "comment_status": comment_status,
                    "provenance": {
                        "canonical_ground_truth": canonical_ref("icho2025", canonical_locator),
                        "score_source": source_ref(raw_path, locator),
                        "comment_source": source_ref(raw_path, locator) if comment is not None else None,
                    },
                    "semantic_annotations": empty_annotations(),
                }
            )
    return rows


def load_frozen_ground_truth() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for exam, (path, expected) in GROUND_TRUTH.items():
        file_path = ROOT / path
        actual = sha256_file(file_path)
        if actual != expected:
            raise ValueError(f"ground-truth SHA mismatch for {path}: {actual} != {expected}")
        out[exam] = load_json(file_path)
    return out


def build_rows() -> list[dict[str, Any]]:
    gt = load_frozen_ground_truth()
    rows = build_ipho_rows(gt["ipho2025"]) + build_icho_rows(gt["icho2025"])
    validate_rows(rows)
    return rows


def is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def validate_source_reference(value: Any, where: str, errors: list[str]) -> None:
    if not isinstance(value, dict) or set(value) != SOURCE_REF_KEYS:
        errors.append(f"{where}: source reference keys are not exact")
        return
    path, digest, locator = value.get("path"), value.get("sha256"), value.get("locator")
    if not isinstance(path, str) or not path:
        errors.append(f"{where}: invalid path")
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        errors.append(f"{where}: invalid sha256")
    if not isinstance(locator, str) or not locator:
        errors.append(f"{where}: invalid locator")
    if isinstance(path, str) and path:
        target = ROOT / path
        if not target.is_file():
            errors.append(f"{where}: missing file {path}")
        elif isinstance(digest, str) and sha256_file(target) != digest:
            errors.append(f"{where}: file SHA mismatch for {path}")


def validate_rows(rows: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    ids: set[str] = set()
    for line_no, row in enumerate(rows, start=1):
        where = f"row {line_no}"
        if not isinstance(row, dict) or set(row) != TOP_KEYS:
            errors.append(f"{where}: top-level keys are not exact")
            continue
        if row["schema_version"] != SCHEMA_VERSION:
            errors.append(f"{where}: schema_version mismatch")
        record_id = row.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            errors.append(f"{where}: invalid record_id")
        elif record_id in ids:
            errors.append(f"{where}: duplicate record_id {record_id}")
        else:
            ids.add(record_id)
        if row.get("exam") not in GROUND_TRUTH:
            errors.append(f"{where}: invalid exam")
        for key in ("model_under_test", "case_id"):
            if not isinstance(row.get(key), str) or not row[key]:
                errors.append(f"{where}: invalid {key}")
        if not isinstance(row.get("item_index"), int) or isinstance(row.get("item_index"), bool) or row["item_index"] < 1:
            errors.append(f"{where}: invalid item_index")
        if not is_number(row.get("human_score")):
            errors.append(f"{where}: non-finite/non-numeric human_score")
        elif row["human_score"] < 0:
            errors.append(f"{where}: negative human_score")
        max_score = row.get("max_score")
        if max_score is not None and not is_number(max_score):
            errors.append(f"{where}: max_score is neither null nor finite numeric")
        elif max_score is not None and (max_score <= 0 or row["human_score"] > max_score + 1e-9):
            errors.append(f"{where}: score out of [0,max_score]")
        comment = row.get("human_comment")
        status = row.get("comment_status")
        if status == "available_nonempty":
            if not isinstance(comment, str) or not comment.strip():
                errors.append(f"{where}: available comment is not nonempty")
        elif status == "unavailable_in_source":
            if comment is not None:
                errors.append(f"{where}: unavailable comment must be null")
        else:
            errors.append(f"{where}: invalid comment_status")
        provenance = row.get("provenance")
        if not isinstance(provenance, dict) or set(provenance) != PROVENANCE_KEYS:
            errors.append(f"{where}: provenance keys are not exact")
        else:
            validate_source_reference(provenance["canonical_ground_truth"], f"{where}.canonical_ground_truth", errors)
            validate_source_reference(provenance["score_source"], f"{where}.score_source", errors)
            if provenance["comment_source"] is None:
                if status != "unavailable_in_source":
                    errors.append(f"{where}: available comment lacks comment_source")
            else:
                validate_source_reference(provenance["comment_source"], f"{where}.comment_source", errors)
                if status != "available_nonempty":
                    errors.append(f"{where}: missing comment unexpectedly has comment_source")
        annotations = row.get("semantic_annotations")
        if not isinstance(annotations, dict) or set(annotations) != ANNOTATION_KEYS:
            errors.append(f"{where}: semantic annotation keys are not exact")
        else:
            if annotations.get("annotation_status") != "not_provided_in_archived_source":
                errors.append(f"{where}: annotation_status is not the explicit missing marker")
            for key in NULL_ANNOTATION_KEYS:
                if annotations.get(key) is not None:
                    errors.append(f"{where}: {key} must remain null")
        if row.get("exam") == "ipho2025":
            if row.get("question_id") is not None or row.get("sub_question_id") is not None:
                errors.append(f"{where}: IPhO question fields must be null")
            if row.get("section") not in {"S1", "S2", "S3"} or not isinstance(row.get("label"), str):
                errors.append(f"{where}: malformed IPhO section/label")
            expected_case = f"IPHO-{row.get('section')}-{row.get('item_index')}"
            if row.get("case_id") != expected_case:
                errors.append(f"{where}: IPhO case_id mismatch")
            if max_score is None:
                errors.append(f"{where}: IPhO row must preserve its per-item max_score")
        elif row.get("exam") == "icho2025":
            if not isinstance(row.get("question_id"), str) or not isinstance(row.get("sub_question_id"), str):
                errors.append(f"{where}: malformed IChO question fields")
            if row.get("section") is not None or row.get("label") is not None:
                errors.append(f"{where}: IChO section/label must be null")
            expected_case = f"ICHO-{row.get('question_id')}-{row.get('sub_question_id')}"
            if row.get("case_id") != expected_case:
                errors.append(f"{where}: IChO case_id mismatch")
            if max_score is not None:
                errors.append(f"{where}: IChO max_score must remain null (not present in canonical row)")

    counts = count_rows(rows)
    if counts["total"] != EXPECTED_COUNTS["total"]:
        errors.append(f"total rows {counts['total']} != {EXPECTED_COUNTS['total']}")
    if counts["comments"] != EXPECTED_COUNTS["comments"]:
        errors.append(f"comment rows {counts['comments']} != {EXPECTED_COUNTS['comments']}")
    if counts["missing_comments"] != EXPECTED_COUNTS["missing_comments"]:
        errors.append(
            f"missing-comment rows {counts['missing_comments']} != {EXPECTED_COUNTS['missing_comments']}"
        )
    for exam, expected in EXPECTED_COUNTS["by_exam"].items():
        if counts["by_exam"].get(exam) != expected:
            errors.append(f"{exam} counts {counts['by_exam'].get(exam)} != {expected}")
    referenced = {
        ref["path"]
        for row in rows
        for ref in (row["provenance"]["score_source"], row["provenance"]["comment_source"])
        if ref is not None
    }
    if referenced != set(ORIGINAL_SOURCE_HASHES):
        errors.append(
            f"referenced original source set mismatch: missing={sorted(set(ORIGINAL_SOURCE_HASHES)-referenced)}, "
            f"extra={sorted(referenced-set(ORIGINAL_SOURCE_HASHES))}"
        )
    if errors:
        preview = "\n".join(errors[:50])
        suffix = "" if len(errors) <= 50 else f"\n... {len(errors)-50} more"
        raise ValueError(f"expert-feedback validation failed ({len(errors)} errors):\n{preview}{suffix}")


def count_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_exam: dict[str, Counter[str]] = defaultdict(Counter)
    by_model: dict[str, Counter[str]] = defaultdict(Counter)
    comments = 0
    for row in rows:
        has_comment = row["comment_status"] == "available_nonempty"
        comments += int(has_comment)
        by_exam[row["exam"]]["rows"] += 1
        by_exam[row["exam"]]["comments"] += int(has_comment)
        by_exam[row["exam"]]["missing_comments"] += int(not has_comment)
        key = f"{row['exam']}::{row['model_under_test']}"
        by_model[key]["rows"] += 1
        by_model[key]["comments"] += int(has_comment)
        by_model[key]["missing_comments"] += int(not has_comment)
    return {
        "total": len(rows),
        "comments": comments,
        "missing_comments": len(rows) - comments,
        "by_exam": {key: dict(value) for key, value in sorted(by_exam.items())},
        "by_exam_and_model": {key: dict(value) for key, value in sorted(by_model.items())},
    }


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    ).encode("utf-8")


def report_object(rows: list[dict[str, Any]]) -> dict[str, Any]:
    payload = jsonl_bytes(rows)
    counts = count_rows(rows)
    score_ref_counts = Counter(row["provenance"]["score_source"]["path"] for row in rows)
    comment_ref_counts = Counter(
        row["provenance"]["comment_source"]["path"]
        for row in rows
        if row["provenance"]["comment_source"] is not None
    )
    sources = []
    for path, expected in sorted(ORIGINAL_SOURCE_HASHES.items()):
        actual = sha256_file(ROOT / path)
        sources.append(
            {
                "path": path,
                "sha256": actual,
                "expected_sha256": expected,
                "hash_matches_frozen_value": actual == expected,
                "score_row_references": score_ref_counts[path],
                "comment_row_references": comment_ref_counts[path],
            }
        )
    gt_inputs = []
    for exam, (path, expected) in GROUND_TRUTH.items():
        actual = sha256_file(ROOT / path)
        gt_inputs.append(
            {
                "exam": exam,
                "path": path,
                "sha256": actual,
                "expected_sha256": expected,
                "hash_matches_frozen_value": actual == expected,
            }
        )
    return {
        "schema_version": REPORT_VERSION,
        "fixed_audit_date": FIXED_DATE,
        "status": "PASS",
        "offline_api_calls": 0,
        "release_scope": (
            "Base expert score/comment archive only; no inferred semantic tags, "
            "credit labels, evidence spans, conflict labels, or uncertainty labels."
        ),
        "counts": counts,
        "expected_counts": EXPECTED_COUNTS,
        "semantic_annotation_completeness": {
            "not_provided_in_archived_source": len(rows),
            "human_qa_completed": 0,
            "all_semantic_fields_null": True,
        },
        "ground_truth_inputs": gt_inputs,
        "original_source_inventory": sources,
        "artifacts": {
            "schema": {"path": REL_SCHEMA, "sha256": sha256_file(ROOT / REL_SCHEMA)},
            "exporter_validator": {"path": REL_SCRIPT, "sha256": sha256_file(ROOT / REL_SCRIPT)},
            "jsonl": {"path": REL_JSONL, "sha256": sha256_bytes(payload), "bytes": len(payload)},
        },
        "validation_checks": [
            {"check": "canonical ground-truth hashes equal frozen inventory hashes", "status": "PASS"},
            {"check": "all eleven referenced original source files exist and match frozen hashes", "status": "PASS"},
            {"check": "strict exact-key/type/range/provenance validation for every JSONL row", "status": "PASS"},
            {"check": "record identifiers are unique", "status": "PASS"},
            {"check": "human comments are copied only when nonempty in canonical ground truth", "status": "PASS"},
            {"check": "missing comments are explicit null values", "status": "PASS"},
            {"check": "semantic annotation fields are explicit null values", "status": "PASS"},
            {"check": "deterministic replay exactly matches checked-in JSONL bytes", "status": "PASS_ON_CHECK"},
        ],
        "limitations": [
            "The archive covers five model answers on IPhO 2025 and five on IChO 2025 only; it is not broader human calibration.",
            "The 115 unavailable comments are absent in the archived ground truth and are not reconstructed.",
            "No normalized error/ability tags, credit-status labels, evidence spans, uncertainty labels, discrepancy labels, or conflict adjudications are present in the archived source; all such fields remain null pending human QA.",
            "Original binary DOCX/PDF provenance is hash-addressed, but this release validates row values against the frozen canonical ground-truth JSON rather than re-OCRing or re-parsing binary source layout.",
            "The frozen IChO expert ground truth has raw row scores and problem-level maxima but no reliable per-row maximum; therefore IChO `max_score` is null instead of being borrowed from a judge run.",
        ],
    }


def report_json_bytes(rows: list[dict[str, Any]]) -> bytes:
    return (json.dumps(report_object(rows), ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def report_markdown(rows: list[dict[str, Any]]) -> str:
    report = report_object(rows)
    counts = report["counts"]
    lines = [
        "# Archived expert-feedback base: export and validation",
        "",
        f"Fixed audit date: {FIXED_DATE}  ",
        "Status: **PASS**  ",
        "Execution: fully local; 0 API calls.",
        "",
        "## Result",
        "",
        f"The deterministic export contains **{counts['total']}** real expert-scored rows. "
        f"Exactly **{counts['comments']}** rows preserve a nonempty archived expert comment; "
        f"**{counts['missing_comments']}** rows explicitly store `human_comment: null` because "
        "the archived source has no comment.",
        "",
        "| Exam | Rows | Nonempty comments | Missing comments |",
        "|---|---:|---:|---:|",
    ]
    for exam, values in counts["by_exam"].items():
        lines.append(
            f"| {exam} | {values['rows']} | {values['comments']} | {values['missing_comments']} |"
        )
    lines += [
        "",
        "All 525 records have `annotation_status: not_provided_in_archived_source`. "
        "The fields for credit status, normalized error/ability tags, evidence, uncertainty, "
        "discrepancy, and conflict are `null`; an empty tag list is deliberately not used because "
        "that could be misread as a completed human review finding no tags.",
        "",
        "## Provenance and validation",
        "",
        f"- JSONL SHA-256: `{report['artifacts']['jsonl']['sha256']}`",
        f"- Ground-truth files: {len(report['ground_truth_inputs'])}; both match the frozen inventory hashes.",
        f"- Referenced original grading files: {len(report['original_source_inventory'])}; all exist and match frozen hashes.",
        "- Every row carries the canonical ground-truth path/hash/JSON pointer plus separate score-source and, when available, comment-source path/hash/locator fields.",
        "- The validator enforces exact keys, types, score ranges, comment/null consistency, provenance hashes, unique identifiers, the complete 11-file source set, and the 525/410/115 census.",
        "",
        "## Reproduction",
        "",
        "```bash",
        f"python3 {REL_SCRIPT} --check {REL_JSONL}",
        "```",
        "",
        "The exporter emits deterministic artifacts to stdout; it does not write files or call a network API.",
        "",
        "## Limitations",
        "",
    ]
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def parse_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.endswith("\n"):
                raise ValueError(f"JSONL line {line_no} does not end with LF")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON on line {line_no}: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_no} is not an object")
            rows.append(value)
    return rows


def sidecar_text() -> str:
    artifact_paths = [REL_SCHEMA, REL_SCRIPT, REL_JSONL, REL_REPORT_JSON, REL_REPORT_MD]
    input_paths = [path for path, _ in GROUND_TRUTH.values()] + sorted(ORIGINAL_SOURCE_HASHES)
    lines = [f"{sha256_file(ROOT / path)}  {path}" for path in artifact_paths + input_paths]
    return "\n".join(lines) + "\n"


def check_checked_in(path: Path) -> dict[str, Any]:
    expected_rows = build_rows()
    actual_rows = parse_jsonl(path)
    validate_rows(actual_rows)
    expected_bytes = jsonl_bytes(expected_rows)
    actual_bytes = path.read_bytes()
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"checked-in JSONL differs from deterministic replay: "
            f"actual={sha256_bytes(actual_bytes)} expected={sha256_bytes(expected_bytes)}"
        )
    if (ROOT / REL_REPORT_JSON).is_file():
        expected = report_json_bytes(expected_rows)
        actual = (ROOT / REL_REPORT_JSON).read_bytes()
        if actual != expected:
            raise ValueError("checked-in JSON validation report differs from deterministic replay")
    if (ROOT / REL_REPORT_MD).is_file():
        expected = report_markdown(expected_rows).encode("utf-8")
        actual = (ROOT / REL_REPORT_MD).read_bytes()
        if actual != expected:
            raise ValueError("checked-in Markdown validation report differs from deterministic replay")
    if (ROOT / REL_SIDECAR).is_file():
        expected = sidecar_text().encode("utf-8")
        actual = (ROOT / REL_SIDECAR).read_bytes()
        if actual != expected:
            raise ValueError("checked-in SHA sidecar differs from deterministic replay")
    return {
        "status": "PASS",
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "sha256": sha256_bytes(actual_bytes),
        "counts": count_rows(actual_rows),
        "byte_identical_to_frozen_source_replay": True,
        "api_calls": 0,
    }


def emit_bytes(data: bytes) -> None:
    sys.stdout.buffer.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--emit-jsonl", action="store_true")
    group.add_argument("--emit-report-json", action="store_true")
    group.add_argument("--emit-report-md", action="store_true")
    group.add_argument("--emit-sidecar", action="store_true")
    group.add_argument("--check", metavar="JSONL_PATH")
    parser.add_argument("--start", type=int, default=0, help="first JSONL row to emit (inclusive)")
    parser.add_argument("--stop", type=int, default=None, help="last JSONL row to emit (exclusive)")
    args = parser.parse_args()

    if args.emit_sidecar:
        sys.stdout.write(sidecar_text())
        return
    if args.check:
        result = check_checked_in(Path(args.check).expanduser().resolve())
        sys.stdout.write(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
        return

    rows = build_rows()
    if args.emit_jsonl:
        start = args.start
        stop = len(rows) if args.stop is None else args.stop
        if start < 0 or stop < start or stop > len(rows):
            parser.error(f"invalid JSONL slice [{start}:{stop}] for {len(rows)} rows")
        emit_bytes(jsonl_bytes(rows[start:stop]))
    elif args.emit_report_json:
        emit_bytes(report_json_bytes(rows))
    elif args.emit_report_md:
        sys.stdout.write(report_markdown(rows))


if __name__ == "__main__":
    main()
