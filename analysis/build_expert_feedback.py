#!/usr/bin/env python3
"""Build and validate the provenance-preserving 525-row feedback export.

This is an offline join of two already-audited local lineages:

* the frozen IPhO/IChO 2025 expert ground truth and its eleven cited source
  files; and
* the four archived judge-result arrays (two Gemini judges by two contests).

No comment, label, maximum, or semantic tag is inferred.  In particular, the
IChO expert source has no per-row maximum, so ``expert_max_score`` remains
null.  The archived judge-row maximum is preserved separately, including the
one known expert-score/max discrepancy.  The two judge scores are never
collapsed, so disagreements remain visible in every record.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent
BASE_SCRIPT = ROOT / "audits/cr_rebuttal_local_supplements_20260829/export_validate_expert_feedback_base_20260829.py"
CAL_SCRIPT = ROOT / "audits/cr_rebuttal_local_supplements_20260829/replay_archived_two_gemini_calibration_20260829.py"
BASE_JSONL = ROOT / "audits/cr_rebuttal_local_supplements_20260829/expert_feedback_base_20260829.jsonl"
CAL_CSV = ROOT / "audits/cr_rebuttal_local_supplements_20260829/archived_two_gemini_calibration_rows_20260829.csv"
FROZEN_AUDIT_HASHES = {
    BASE_SCRIPT: "e99d7c1c8b218b6fc5bc3fb6d285b4043fa6ae8e89e3d67ba3b76e2570243cbc",
    CAL_SCRIPT: "d490b15f58a7973ca35a6142b2fbd5e061123a2e61536caf07166b63a9411bd5",
    BASE_JSONL: "1fc337441cd882beff8b925ec34044906d32170dd054ebbf680ae59d9be20f56",
    CAL_CSV: "36aa01ae8dcd85b1a8680b2f7034f833907a67b7e377a2e22d8cf27efd257d98",
}
SCHEMA_VERSION = "science_arena_expert_feedback_with_archived_judges_v1"
MANIFEST_VERSION = "science_arena_expert_feedback_with_archived_judges_manifest_v1"
FIXED_DATE = "2026-08-30"
FILES = {
    "jsonl": "expert_feedback_with_archived_judges_20260830.jsonl",
    "schema": "expert_feedback_with_archived_judges_20260830.schema.json",
    "manifest": "expert_feedback_with_archived_judges_20260830.manifest.json",
    "checksums": "SHA256SUMS",
}
EXPECTED = {
    "rows": 525,
    "comments": 410,
    "missing_comments": 115,
    "judge_rows": 1050,
    "judges_per_row": 2,
    "judge_disagreements": 186,
    "judge_normalized_status": {
        "usable": 1048,
        "excluded_expert_score_outside_archived_judge_row_max": 2,
    },
}
TOL = 1e-12


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@lru_cache(maxsize=None)
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def canonical_json_sha(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


@lru_cache(maxsize=None)
def load_json_file(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def source_ref(path: str, digest: str, locator: str) -> dict[str, str]:
    target = ROOT / path
    if not target.is_file():
        raise ValueError(f"missing provenance file: {path}")
    actual = sha256_file(target)
    if actual != digest:
        raise ValueError(f"provenance SHA mismatch for {path}: {actual} != {digest}")
    return {"path": path, "sha256": digest, "locator": locator}


def schema_object() -> dict[str, Any]:
    source_reference = {
        "type": "object",
        "additionalProperties": False,
        "required": ["path", "sha256", "locator"],
        "properties": {
            "path": {"type": "string", "minLength": 1},
            "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
            "locator": {"type": "string", "minLength": 1},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "https://science-arena.local/schemas/expert_feedback_with_archived_judges_20260830.schema.json",
        "title": "Science Arena provenance-preserving expert feedback with archived judge deltas",
        "description": (
            "One frozen expert-score row plus two independently preserved archived Gemini judge rows. "
            "Null rich annotations mean unavailable, not negative."
        ),
        "type": "object",
        "additionalProperties": False,
        "required": [
            "schema_version", "record_id", "contest", "model_under_test",
            "item_identity", "expert_evaluation", "archived_judge_evaluations",
            "judge_disagreement", "provenance", "rich_annotations",
        ],
        "properties": {
            "schema_version": {"const": SCHEMA_VERSION},
            "record_id": {"type": "string", "minLength": 1},
            "contest": {"enum": ["ipho2025", "icho2025"]},
            "model_under_test": {"type": "string", "minLength": 1},
            "item_identity": {"type": "object"},
            "expert_evaluation": {"type": "object"},
            "archived_judge_evaluations": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "items": {"type": "object"},
            },
            "judge_disagreement": {"type": "object"},
            "provenance": {"type": "object"},
            "rich_annotations": {"type": "object"},
        },
        "$defs": {"sourceReference": source_reference},
        "x-science-arena-contract": {
            "expected_rows": 525,
            "expected_nonempty_verbatim_comments": 410,
            "expected_explicit_null_comments": 115,
            "expected_archived_judge_rows": 1050,
            "semantic_null_policy": "unavailable_not_negative",
        },
    }


def build_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    # Pin the already-audited deterministic exports and their builders.  This
    # avoids re-parsing the same large binary source documents hundreds of
    # times while every row-level source reference is still re-hashed below.
    for path, expected in FROZEN_AUDIT_HASHES.items():
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"frozen audited input mismatch: {rel(path)}: {actual} != {expected}")
    base_rows = [json.loads(line) for line in BASE_JSONL.read_text(encoding="utf-8").splitlines()]
    with CAL_CSV.open("r", encoding="utf-8", newline="") as handle:
        judge_rows = list(csv.DictReader(handle))
    if len(base_rows) != 525 or len(judge_rows) != 1050:
        raise ValueError(f"audited input census mismatch: base={len(base_rows)} judge={len(judge_rows)}")

    result_arrays: dict[str, list[dict[str, Any]]] = {}
    result_hashes: dict[str, str] = {}
    for row in judge_rows:
        result_path = f"{row['run_directory']}/judge_results.json"
        if result_path not in result_arrays:
            expected = row["judge_results_sha256"]
            actual = sha256_file(ROOT / result_path)
            if actual != expected:
                raise ValueError(f"frozen judge array mismatch: {result_path}: {actual} != {expected}")
            result_arrays[result_path] = load_json_file(ROOT / result_path)
            result_hashes[result_path] = actual

    by_key: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in judge_rows:
        by_key[(row["exam"], row["model_under_test"], row["case_id"])].append(row)

    final_rows: list[dict[str, Any]] = []
    for expert in base_rows:
        key = (expert["exam"], expert["model_under_test"], expert["case_id"])
        joined = sorted(by_key.get(key, []), key=lambda value: value["judge_id"])
        if len(joined) != 2 or len({row["judge_id"] for row in joined}) != 2:
            raise ValueError(f"expected exactly two distinct judges for {key}, found {len(joined)}")
        if any(abs(float(row["expert_score"]) - float(expert["human_score"])) > TOL for row in joined):
            raise ValueError(f"expert-score join mismatch for {key}")
        judge_maxima = {float(row["max_score"]) for row in joined}
        if len(judge_maxima) != 1:
            raise ValueError(f"archived judges disagree on row maximum for {key}: {sorted(judge_maxima)}")
        archived_max = next(iter(judge_maxima))
        expert_max = expert["max_score"]
        if expert_max is not None and abs(float(expert_max) - archived_max) > TOL:
            raise ValueError(f"expert and archived judge maxima disagree for sourced IPhO maximum {key}")

        judge_exports: list[dict[str, Any]] = []
        for row in joined:
            result_path = f"{row['run_directory']}/judge_results.json"
            raw_results = result_arrays[result_path]
            source_index = int(row["source_row_index"])
            raw = raw_results[source_index]
            if canonical_json_sha(raw) != row["raw_judgment_sha256"]:
                raise ValueError(f"raw judgment canonical hash mismatch for {row['record_id']}")
            judge_exports.append(
                {
                    "judge_id": row["judge_id"],
                    "judge_model": row["judge_model"],
                    "score": float(row["judge_score"]),
                    "max_score": float(row["max_score"]),
                    "signed_delta_judge_minus_expert": float(row["signed_error_judge_minus_expert"]),
                    "absolute_delta": float(row["absolute_error"]),
                    "normalized_pair_status": row["normalized_pair_status"],
                    "source": source_ref(
                        result_path,
                        row["judge_results_sha256"],
                        f"/{source_index}",
                    ),
                    "raw_judgment_canonical_sha256": row["raw_judgment_sha256"],
                }
            )

        scores = [row["score"] for row in judge_exports]
        disagreement = abs(scores[1] - scores[0])
        max_vs_expert = (
            None if expert_max is None else float(expert["human_score"]) <= float(expert_max) + TOL
        )
        archived_max_vs_expert = float(expert["human_score"]) <= archived_max + TOL
        final_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "record_id": expert["record_id"],
                "contest": expert["exam"],
                "model_under_test": expert["model_under_test"],
                "item_identity": {
                    "case_id": expert["case_id"],
                    "question_id": expert["question_id"],
                    "sub_question_id": expert["sub_question_id"],
                    "section": expert["section"],
                    "item_index": expert["item_index"],
                    "label": expert["label"],
                },
                "expert_evaluation": {
                    "score": float(expert["human_score"]),
                    "max_score": None if expert_max is None else float(expert_max),
                    "max_score_status": (
                        "present_in_expert_ground_truth"
                        if expert_max is not None
                        else "not_available_per_row_in_expert_ground_truth"
                    ),
                    "archived_judge_row_max_score": archived_max,
                    "archived_judge_max_status": "two_judge_artifacts_exact_consensus_not_expert_authored",
                    "comment": expert["human_comment"],
                    "comment_status": expert["comment_status"],
                },
                "archived_judge_evaluations": judge_exports,
                "judge_disagreement": {
                    "status": "different_scores_preserved" if disagreement > TOL else "identical_scores",
                    "signed_gemini_3_5_flash_minus_gemini_3_1_pro_preview": scores[1] - scores[0],
                    "absolute_score_difference": disagreement,
                },
                "provenance": {
                    "expert_canonical_ground_truth": expert["provenance"]["canonical_ground_truth"],
                    "expert_score_source": expert["provenance"]["score_source"],
                    "expert_comment_source": expert["provenance"]["comment_source"],
                },
                "rich_annotations": {
                    "annotation_status": "not_provided_in_archived_source",
                    "credit_status": None,
                    "error_tags": None,
                    "ability_tags": None,
                    "evidence_spans": None,
                    "uncertainty_label": None,
                    "source_discrepancy": (
                        None
                        if archived_max_vs_expert
                        else {
                            "status": "expert_score_exceeds_archived_judge_row_max_preserved",
                            "expert_score": float(expert["human_score"]),
                            "archived_judge_row_max_score": archived_max,
                            "resolution": None,
                        }
                    ),
                    "expert_max_range_check": max_vs_expert,
                },
            }
        )

    final_rows.sort(key=lambda row: row["record_id"])
    validate_rows(final_rows)
    run_counts: dict[tuple[str, str, str], int] = Counter(
        (row["run_id"], row["judge_id"], row["exam"]) for row in judge_rows
    )
    replay = {
        "run_accounting": [
            {
                "run_id": run_id,
                "judge_id": judge_id,
                "exam": exam,
                "expected_rows": count,
                "stored_judgment_rows": count,
                "joined_numeric_rows": count,
            }
            for (run_id, judge_id, exam), count in sorted(run_counts.items())
        ]
    }
    return final_rows, replay


def validate_source(value: Any, where: str, errors: list[str]) -> None:
    if not isinstance(value, dict) or set(value) != {"path", "sha256", "locator"}:
        errors.append(f"{where}: malformed source reference")
        return
    path, digest, locator = value["path"], value["sha256"], value["locator"]
    if not isinstance(locator, str) or not locator:
        errors.append(f"{where}: empty locator")
    target = ROOT / path if isinstance(path, str) else None
    if target is None or not target.is_file():
        errors.append(f"{where}: missing source {path}")
    elif not isinstance(digest, str) or sha256_file(target) != digest:
        errors.append(f"{where}: source hash mismatch")


def validate_rows(rows: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    ids: set[str] = set()
    comments = 0
    judge_statuses: Counter[str] = Counter()
    judge_rows = 0
    disagreements = 0
    discrepancies = 0
    expected_top = {
        "schema_version", "record_id", "contest", "model_under_test", "item_identity",
        "expert_evaluation", "archived_judge_evaluations", "judge_disagreement",
        "provenance", "rich_annotations",
    }
    null_annotation_keys = {
        "credit_status", "error_tags", "ability_tags", "evidence_spans", "uncertainty_label"
    }
    for index, row in enumerate(rows, 1):
        where = f"row {index}"
        if not isinstance(row, dict) or set(row) != expected_top:
            errors.append(f"{where}: top-level keys not exact")
            continue
        if row["schema_version"] != SCHEMA_VERSION:
            errors.append(f"{where}: schema mismatch")
        if row["record_id"] in ids:
            errors.append(f"{where}: duplicate record_id {row['record_id']}")
        ids.add(row["record_id"])
        expert = row["expert_evaluation"]
        comment = expert.get("comment")
        if expert.get("comment_status") == "available_nonempty":
            if not isinstance(comment, str) or not comment.strip():
                errors.append(f"{where}: nonempty comment status without nonempty string")
            else:
                comments += 1
        elif expert.get("comment_status") == "unavailable_in_source":
            if comment is not None:
                errors.append(f"{where}: unavailable comment is not null")
        else:
            errors.append(f"{where}: invalid comment status")
        if not isinstance(expert.get("score"), (int, float)) or not math.isfinite(expert["score"]):
            errors.append(f"{where}: invalid expert score")
        if row["contest"] == "ipho2025":
            if expert.get("max_score_status") != "present_in_expert_ground_truth":
                errors.append(f"{where}: IPhO expert max status mismatch")
            if expert.get("max_score") is None or expert["score"] > expert["max_score"] + TOL:
                errors.append(f"{where}: invalid IPhO expert score/max")
        elif row["contest"] == "icho2025":
            if expert.get("max_score") is not None or expert.get("max_score_status") != "not_available_per_row_in_expert_ground_truth":
                errors.append(f"{where}: IChO expert max must be explicit unavailable/null")
        else:
            errors.append(f"{where}: invalid contest")
        judges = row["archived_judge_evaluations"]
        if not isinstance(judges, list) or len(judges) != 2:
            errors.append(f"{where}: expected two judge rows")
            continue
        if [value.get("judge_id") for value in judges] != ["gemini_3_1_pro_preview", "gemini_3_5_flash"]:
            errors.append(f"{where}: judge order/identity mismatch")
        if len({value.get("max_score") for value in judges}) != 1:
            errors.append(f"{where}: judge max disagreement")
        for j, value in enumerate(judges):
            judge_rows += 1
            judge_statuses[value.get("normalized_pair_status")] += 1
            expected_delta = float(value["score"]) - float(expert["score"])
            if abs(expected_delta - float(value["signed_delta_judge_minus_expert"])) > TOL:
                errors.append(f"{where}.judge[{j}]: signed delta mismatch")
            if abs(abs(expected_delta) - float(value["absolute_delta"])) > TOL:
                errors.append(f"{where}.judge[{j}]: absolute delta mismatch")
            validate_source(value.get("source"), f"{where}.judge[{j}].source", errors)
            raw = load_json_file(ROOT / value["source"]["path"])[int(value["source"]["locator"].lstrip("/"))]
            if canonical_json_sha(raw) != value.get("raw_judgment_canonical_sha256"):
                errors.append(f"{where}.judge[{j}]: raw row hash mismatch")
        score_gap = abs(float(judges[1]["score"]) - float(judges[0]["score"]))
        expected_status = "different_scores_preserved" if score_gap > TOL else "identical_scores"
        if row["judge_disagreement"].get("status") != expected_status:
            errors.append(f"{where}: disagreement status mismatch")
        disagreements += int(score_gap > TOL)
        annotations = row["rich_annotations"]
        if annotations.get("annotation_status") != "not_provided_in_archived_source":
            errors.append(f"{where}: invalid annotation status")
        for key in null_annotation_keys:
            if annotations.get(key) is not None:
                errors.append(f"{where}: {key} must remain null")
        if annotations.get("source_discrepancy") is not None:
            discrepancies += 1
        provenance = row["provenance"]
        for key in ("expert_canonical_ground_truth", "expert_score_source"):
            validate_source(provenance.get(key), f"{where}.{key}", errors)
        comment_source = provenance.get("expert_comment_source")
        if comment is None:
            if comment_source is not None:
                errors.append(f"{where}: null comment unexpectedly has source")
        else:
            validate_source(comment_source, f"{where}.expert_comment_source", errors)

    actual = {
        "rows": len(rows),
        "comments": comments,
        "missing_comments": len(rows) - comments,
        "judge_rows": judge_rows,
        "judges_per_row": 2,
        "judge_disagreements": disagreements,
        "judge_normalized_status": dict(sorted(judge_statuses.items())),
    }
    for key in ("rows", "comments", "missing_comments", "judge_rows", "judge_disagreements"):
        if actual[key] != EXPECTED[key]:
            errors.append(f"census {key}: {actual[key]} != {EXPECTED[key]}")
    if actual["judge_normalized_status"] != EXPECTED["judge_normalized_status"]:
        errors.append(f"judge status census mismatch: {actual['judge_normalized_status']}")
    if discrepancies != 1:
        errors.append(f"source discrepancy census {discrepancies} != 1")
    if errors:
        raise ValueError("validation failed:\n" + "\n".join(errors[:100]))


def json_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def jsonl_bytes(rows: list[dict[str, Any]]) -> bytes:
    return ("".join(canonical_json(row) + "\n" for row in rows)).encode("utf-8")


def manifest_object(rows: list[dict[str, Any]], replay: dict[str, Any], schema_bytes: bytes, rows_bytes: bytes) -> dict[str, Any]:
    by_contest = Counter(row["contest"] for row in rows)
    comment_counts = Counter(
        "nonempty_verbatim" if row["expert_evaluation"]["comment"] is not None else "explicit_null"
        for row in rows
    )
    judge_statuses = Counter(
        judge["normalized_pair_status"]
        for row in rows
        for judge in row["archived_judge_evaluations"]
    )
    disagree = sum(row["judge_disagreement"]["status"] == "different_scores_preserved" for row in rows)
    discrepancy_rows = [
        row["record_id"] for row in rows if row["rich_annotations"]["source_discrepancy"] is not None
    ]
    return {
        "schema_version": MANIFEST_VERSION,
        "fixed_audit_date": FIXED_DATE,
        "status": "PASS",
        "execution": "offline_deterministic_no_api_calls",
        "scope": "Five archived model answers each on IPhO 2025 and IChO 2025; two archived Gemini judges.",
        "counts": {
            "unique_rows": len(rows),
            "by_contest": dict(sorted(by_contest.items())),
            "comments": dict(sorted(comment_counts.items())),
            "archived_judge_rows": sum(len(row["archived_judge_evaluations"]) for row in rows),
            "judge_score_disagreements_preserved": disagree,
            "judge_normalized_status": dict(sorted(judge_statuses.items())),
            "source_discrepancy_rows": len(discrepancy_rows),
        },
        "source_discrepancy_record_ids": discrepancy_rows,
        "semantic_annotation_policy": {
            "status": "not_provided_in_archived_source",
            "credit_status": None,
            "error_tags": None,
            "ability_tags": None,
            "evidence_spans": None,
            "uncertainty_label": None,
            "model_generated_annotations": False,
        },
        "max_score_policy": {
            "ipho2025": "Expert per-row maximum is preserved from expert ground truth.",
            "icho2025": (
                "Expert per-row maximum is unavailable and remains null. The exact-consensus maximum "
                "from the two archived judge artifacts is stored separately and is not labeled expert-authored."
            ),
            "known_conflict": (
                "gpt-5 / ICHO-2-2.3c has expert score 10 but archived judge-row max 6 in both judge runs; "
                "the discrepancy is preserved without adjudication."
            ),
        },
        "artifacts": {
            "jsonl": {"path": rel(OUT / FILES["jsonl"]), "sha256": sha256_bytes(rows_bytes), "bytes": len(rows_bytes)},
            "schema": {"path": rel(OUT / FILES["schema"]), "sha256": sha256_bytes(schema_bytes), "bytes": len(schema_bytes)},
            "builder_validator": {"path": rel(Path(__file__)), "sha256": sha256_file(Path(__file__))},
        },
        "audited_lineage": {
            "expert_exporter": {"path": rel(BASE_SCRIPT), "sha256": sha256_file(BASE_SCRIPT)},
            "judge_replay": {"path": rel(CAL_SCRIPT), "sha256": sha256_file(CAL_SCRIPT)},
            "judge_runs": [
                {
                    "run_id": entry["run_id"],
                    "judge_id": entry["judge_id"],
                    "exam": entry["exam"],
                    "expected_rows": entry["expected_rows"],
                    "stored_rows": entry["stored_judgment_rows"],
                    "joined_numeric_rows": entry["joined_numeric_rows"],
                }
                for entry in replay["run_accounting"]
            ],
        },
        "validation_contract": [
            "525 unique expert rows",
            "410 byte-preserved nonempty comments and 115 explicit null comments",
            "exactly two distinct archived judges and 1,050 judge rows",
            "judge score/max/delta recomputation and raw-row canonical SHA verification",
            "all referenced source paths exist and hashes match",
            "186 nonidentical two-judge scores preserved rather than collapsed",
            "all unavailable rich error/ability/credit/evidence/uncertainty fields remain null",
        ],
        "limitations": [
            "The export does not add human-reviewed rich error, ability, credit, evidence, or uncertainty labels.",
            "IChO per-row expert maxima are absent; archived judge maxima are kept in a separately labeled field.",
            "The two archived judges are from the Gemini family; this artifact does not establish cross-family judge validity.",
            "One archived IChO source discrepancy remains unresolved and explicitly represented.",
        ],
    }


def expected_artifacts() -> dict[str, bytes]:
    rows, replay = build_rows()
    schema_bytes = json_bytes(schema_object())
    rows_bytes = jsonl_bytes(rows)
    manifest_bytes = json_bytes(manifest_object(rows, replay, schema_bytes, rows_bytes))
    checksum_order = ["schema", "jsonl", "manifest"]
    artifact_bytes = {"schema": schema_bytes, "jsonl": rows_bytes, "manifest": manifest_bytes}
    checksums = "".join(
        f"{sha256_bytes(artifact_bytes[name])}  {FILES[name]}\n" for name in checksum_order
    ).encode("utf-8")
    artifact_bytes["checksums"] = checksums
    return artifact_bytes


def write_artifacts() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, payload in expected_artifacts().items():
        (OUT / FILES[name]).write_bytes(payload)


def check_artifacts() -> None:
    expected = expected_artifacts()
    for name, payload in expected.items():
        target = OUT / FILES[name]
        if not target.is_file():
            raise ValueError(f"missing generated artifact: {target}")
        actual = target.read_bytes()
        if actual != payload:
            raise ValueError(
                f"artifact differs from deterministic replay: {target}: "
                f"actual={sha256_bytes(actual)} expected={sha256_bytes(payload)}"
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--write", action="store_true", help="write deterministic outputs")
    action.add_argument("--check", action="store_true", help="validate checked-in outputs byte-for-byte")
    args = parser.parse_args()
    if args.write:
        write_artifacts()
    else:
        check_artifacts()
    manifest = json.loads((OUT / FILES["manifest"]).read_text(encoding="utf-8"))
    print(json.dumps({
        "status": "PASS",
        "rows": manifest["counts"]["unique_rows"],
        "comments": manifest["counts"]["comments"],
        "judge_rows": manifest["counts"]["archived_judge_rows"],
        "judge_disagreements_preserved": manifest["counts"]["judge_score_disagreements_preserved"],
        "jsonl_sha256": manifest["artifacts"]["jsonl"]["sha256"],
    }, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
