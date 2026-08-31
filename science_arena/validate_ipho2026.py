#!/usr/bin/env python3
"""Offline integrity and runner-contract validation for canonical IPhO 2026."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import jsonschema  # type: ignore
except ModuleNotFoundError:  # The repository validator remains dependency-free.
    jsonschema = None


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
DEFAULT_DATASET = HERE / "datasets" / "ipho2026"
PIPELINE_PATH = HERE / "pipeline.py"
QUESTION_PDF = "new-test/IPhO 2026/theoretical-exam.pdf"
QUESTION_PDF_SHA256 = "25f555e5e2ff07b9720cb47347981f3d8951d8f9ce44e1e1e3bc04fd5842f729"
EXPECTED_SEQUENCE_COUNTS = {"ipho2026_t1": 5, "ipho2026_t2": 8, "ipho2026_t3": 10}
EXPECTED_PARENT_SCORES = {"T1": 10.0, "T2": 10.0, "T3": 10.0}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def workspace_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (WORKSPACE / path).resolve()


def close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=1e-9)


def schema_type_matches(value: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(value, dict)
    if expected == "array":
        return isinstance(value, list)
    if expected == "string":
        return isinstance(value, str)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "null":
        return value is None
    raise ValueError(f"unsupported schema type {expected!r}")


def validate_schema_offline(
    value: Any, schema: dict[str, Any], *, location: str = "$"
) -> list[str]:
    """Validate every assertion keyword used by this repository's item schema."""
    import re

    failures: list[str] = []
    expected_type = schema.get("type")
    if expected_type is not None:
        allowed = expected_type if isinstance(expected_type, list) else [expected_type]
        if not any(schema_type_matches(value, entry) for entry in allowed):
            return [f"{location}: expected type {allowed}, got {type(value).__name__}"]
    if "const" in schema and value != schema["const"]:
        failures.append(f"{location}: value does not equal const {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        failures.append(f"{location}: value is not in enum {schema['enum']!r}")
    if isinstance(value, dict):
        for field in schema.get("required", []):
            if field not in value:
                failures.append(f"{location}: missing required property {field!r}")
        properties = schema.get("properties", {})
        for field, field_schema in properties.items():
            if field in value:
                failures.extend(
                    validate_schema_offline(
                        value[field], field_schema, location=f"{location}.{field}"
                    )
                )
        if schema.get("additionalProperties") is False:
            for field in value:
                if field not in properties:
                    failures.append(f"{location}: unexpected property {field!r}")
    if isinstance(value, list):
        if len(value) < int(schema.get("minItems", 0)):
            failures.append(f"{location}: fewer than minItems")
        if "maxItems" in schema and len(value) > int(schema["maxItems"]):
            failures.append(f"{location}: more than maxItems")
        item_schema = schema.get("items")
        if isinstance(item_schema, dict):
            for index, item in enumerate(value):
                failures.extend(
                    validate_schema_offline(item, item_schema, location=f"{location}[{index}]")
                )
    if isinstance(value, str):
        if len(value) < int(schema.get("minLength", 0)):
            failures.append(f"{location}: shorter than minLength")
        if "pattern" in schema and re.search(str(schema["pattern"]), value) is None:
            failures.append(f"{location}: does not match pattern {schema['pattern']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            failures.append(f"{location}: below minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            failures.append(f"{location}: above maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            failures.append(f"{location}: not above exclusiveMinimum {schema['exclusiveMinimum']}")
    return failures


def load_pipeline() -> Any:
    spec = importlib.util.spec_from_file_location("science_arena_pipeline_for_validation", PIPELINE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pipeline from {PIPELINE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def runner_dry_validation(items_path: Path, items: list[dict[str, Any]]) -> dict[str, Any]:
    models = {
        "schema_version": "science_arena_validation_only_models_v1",
        "models": [
            {
                "route": "validation-only/native-vision",
                "input_mode": "vision",
                "max_attempts": 7,
                "timeout_seconds": 1800,
            },
            {
                "route": "validation-only/text-staticized",
                "input_mode": "text_staticized",
                "max_attempts": 7,
                "timeout_seconds": 1800,
            },
        ],
    }
    with tempfile.TemporaryDirectory(prefix="ipho2026-runner-dry-") as directory:
        temporary = Path(directory)
        models_path = temporary / "models.json"
        models_path.write_text(json.dumps(models, ensure_ascii=False), encoding="utf-8")
        run_root = temporary / "run"
        command = [
            sys.executable,
            str(PIPELINE_PATH),
            "validate",
            "--items",
            str(items_path),
            "--models",
            str(models_path),
            "--workspace",
            str(WORKSPACE),
            "--run-root",
            str(run_root),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        validation_path = run_root / "input_validation.json"
        cli_validation = read_json(validation_path) if validation_path.exists() else None

    pipeline = load_pipeline()
    direct_validation = pipeline.validate_inputs(items, models["models"], WORKSPACE)
    blocked_item = copy.deepcopy(next(item for item in items if item["visual_dependency"]))
    blocked_item["question_image_paths"] = []
    blocked_item["visual_dependency_metadata"]["question_image_paths"] = []
    blocked_item["visual_dependency_metadata"]["assets"] = []
    blocked_validation = pipeline.validate_inputs(
        [blocked_item], [models["models"][0]], WORKSPACE
    )
    missing_image_blocked = (
        not blocked_validation["valid"]
        and any(
            "must block" in error or "refusing" in error or "question_image_paths" in error
            for error in blocked_validation["errors"]
        )
    )
    prompt_probe = {
        "benchmark_id": "probe",
        "sequence_id": "probe",
        "item_id": "probe",
        "max_score": 1,
        "question_text_native": "NATIVE_TRACK_MARKER",
        "question_text_staticized": "STATICIZED_TRACK_MARKER",
    }
    vision_prompt = pipeline.solver_prompt(prompt_probe, "vision")
    text_prompt = pipeline.solver_prompt(prompt_probe, "text_staticized")
    track_routing_correct = (
        "NATIVE_TRACK_MARKER" in vision_prompt
        and "STATICIZED_TRACK_MARKER" not in vision_prompt
        and "STATICIZED_TRACK_MARKER" in text_prompt
        and "NATIVE_TRACK_MARKER" not in text_prompt
    )
    return {
        "cli_command": "pipeline.py validate (temporary validation-only model manifest)",
        "pipeline_path": str(PIPELINE_PATH.relative_to(WORKSPACE)),
        "pipeline_sha256": sha256_file(PIPELINE_PATH),
        "cli_return_code": completed.returncode,
        "cli_stdout": completed.stdout.strip(),
        "cli_stderr": completed.stderr.strip(),
        "cli_input_validation": cli_validation,
        "direct_validate_inputs": direct_validation,
        "native_and_staticized_prompt_routing_correct": track_routing_correct,
        "visual_item_without_direct_image_is_blocked": missing_image_blocked,
        "api_or_model_calls_made": False,
        "passed": bool(
            completed.returncode == 0
            and cli_validation
            and cli_validation.get("valid")
            and direct_validation["valid"]
            and track_routing_correct
            and missing_image_blocked
        ),
    }


def validate(dataset: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    dataset = dataset.resolve()
    manifest_path = dataset / "manifest.json"
    items_path = dataset / "items.jsonl"
    schema_path = dataset / "item_schema.json"
    contexts_path = dataset / "contexts.json"
    images_manifest_path = dataset / "question_images_manifest.json"

    manifest = read_json(manifest_path)
    items = read_jsonl(items_path)
    schema = read_json(schema_path)
    image_manifest = read_json(images_manifest_path)

    schema_errors: list[str] = []
    if jsonschema is not None:
        schema_validator_backend = "jsonschema.Draft202012Validator"
        validator = jsonschema.Draft202012Validator(schema)
        for index, item in enumerate(items):
            for failure in sorted(validator.iter_errors(item), key=lambda value: list(value.path)):
                schema_errors.append(f"item[{index}] {failure.json_path}: {failure.message}")
    else:
        schema_validator_backend = (
            "offline_builtin_validator_all_assertion_keywords_used_by_item_schema"
        )
        for index, item in enumerate(items):
            schema_errors.extend(
                f"item[{index}] {failure}"
                for failure in validate_schema_offline(item, schema)
            )
    errors.extend(schema_errors)

    ids = [str(item.get("item_id")) for item in items]
    if len(items) != 23:
        errors.append(f"item count {len(items)} != 23")
    if len(set(ids)) != len(ids):
        errors.append("item ids are not unique")
    if ids != manifest.get("item_order"):
        errors.append("items.jsonl order differs from manifest.item_order")
    if [item.get("global_order") for item in items] != list(range(1, 24)):
        errors.append("global_order is not exactly 1..23")

    sequence_counts = Counter(str(item.get("sequence_id")) for item in items)
    if dict(sequence_counts) != EXPECTED_SEQUENCE_COUNTS:
        errors.append(f"sequence counts {dict(sequence_counts)} != {EXPECTED_SEQUENCE_COUNTS}")
    turns: dict[str, list[int]] = defaultdict(list)
    for item in items:
        turns[str(item["sequence_id"])].append(int(item["turn_index"]))
    for sequence_id, values in turns.items():
        if values != list(range(1, len(values) + 1)):
            errors.append(f"{sequence_id}: noncontiguous/order-invalid turn indexes {values}")

    parent_scores: dict[str, float] = defaultdict(float)
    rubric_total = 0.0
    criterion_mismatches: list[str] = []
    for item in items:
        item_id = str(item["item_id"])
        score = float(item["max_score"])
        parent_scores[str(item["parent_question"])] += score
        rubric = item["grading_rubric"]
        criterion_sum = sum(float(criterion["points"]) for criterion in rubric["criteria"])
        rubric_total += criterion_sum
        if not close(float(rubric["max_score"]), score) or not close(criterion_sum, score):
            criterion_mismatches.append(
                f"{item_id}: item={score}, rubric={rubric['max_score']}, criteria={criterion_sum}"
            )
    if dict(parent_scores) != EXPECTED_PARENT_SCORES:
        errors.append(f"parent score totals {dict(parent_scores)} != {EXPECTED_PARENT_SCORES}")
    question_score_total = sum(parent_scores.values())
    if not close(question_score_total, 30.0):
        errors.append(f"question score total {question_score_total} != 30")
    if not close(rubric_total, 30.0):
        errors.append(f"rubric criterion total {rubric_total} != 30")
    errors.extend(criterion_mismatches)

    provenance_failures: list[str] = []
    for item in items:
        if item.get("official_answer") is not None:
            provenance_failures.append(f"{item['item_id']}: official_answer is not null")
        if item.get("official_solution") is not None:
            provenance_failures.append(f"{item['item_id']}: official_solution is not null")
        if item.get("official_rubric") is not None:
            provenance_failures.append(f"{item['item_id']}: official_rubric is not null")
        if item.get("rubric_type") != "derived_from_solution_not_official":
            provenance_failures.append(f"{item['item_id']}: rubric_type is not explicitly derived")
        if item.get("grading_rubric", {}).get("status") != "derived_from_solution_not_official":
            provenance_failures.append(f"{item['item_id']}: grading_rubric status is not derived")
        if not str(item.get("gold_solution") or "").strip():
            provenance_failures.append(f"{item['item_id']}: gold_solution is empty")
        if item.get("solution_source", {}).get("provenance_status") != (
            "provided_solution_unconfirmed_official"
        ):
            provenance_failures.append(f"{item['item_id']}: solution provenance was weakened")
    errors.extend(provenance_failures)

    document_by_path = {document["path"]: document for document in manifest["documents"]}
    document_hash_failures: list[str] = []
    for source_path, document in document_by_path.items():
        path = workspace_path(source_path)
        if not path.is_file():
            document_hash_failures.append(f"missing source document {source_path}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != document["sha256"]:
            document_hash_failures.append(
                f"source document hash mismatch {source_path}: {actual_hash}"
            )
        if path.stat().st_size != int(document["bytes"]):
            document_hash_failures.append(f"source document byte-size mismatch {source_path}")
    if document_by_path.get(QUESTION_PDF, {}).get("sha256") != QUESTION_PDF_SHA256:
        document_hash_failures.append("official question PDF audited hash is not preserved")
    for item in items:
        for source_field in ("question_source", "solution_source"):
            source = item[source_field]
            document = document_by_path.get(source["source_pdf"])
            if document is None:
                document_hash_failures.append(
                    f"{item['item_id']}: {source_field} document is absent from manifest"
                )
            elif source.get("source_pdf_sha256") != document.get("sha256"):
                document_hash_failures.append(
                    f"{item['item_id']}: {source_field} hash differs from manifest document"
                )
    errors.extend(document_hash_failures)

    image_asset_by_path = {asset["path"]: asset for asset in image_manifest["assets"]}
    image_hash_failures: list[str] = []
    solution_image_leakage: list[str] = []
    declared_image_files = {workspace_path(path) for path in image_asset_by_path}
    actual_image_files = {
        path.resolve() for path in (dataset / "question_images").glob("**/*") if path.is_file()
    }
    for unexpected in sorted(actual_image_files - declared_image_files):
        solution_image_leakage.append(
            f"undeclared file in canonical question_images: {unexpected.relative_to(WORKSPACE)}"
        )
    for missing in sorted(declared_image_files - actual_image_files):
        image_hash_failures.append(
            f"declared canonical image is absent: {missing.relative_to(WORKSPACE)}"
        )
    for asset_path, asset in image_asset_by_path.items():
        path = workspace_path(asset_path)
        if not path.is_file():
            image_hash_failures.append(f"missing canonical question image {asset_path}")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != asset["sha256"]:
            image_hash_failures.append(f"canonical question image hash mismatch {asset_path}")
        if path.stat().st_size != int(asset["bytes"]):
            image_hash_failures.append(f"canonical question image byte-size mismatch {asset_path}")
        source_asset_path = workspace_path(asset["source_asset_path"])
        if not source_asset_path.is_file() or sha256_file(source_asset_path) != asset["sha256"]:
            image_hash_failures.append(f"draft-source image/hash mismatch {asset['source_asset_path']}")
        if asset.get("source_pdf") != QUESTION_PDF:
            solution_image_leakage.append(f"{asset_path}: source is not official question PDF")
        if asset.get("source_pdf_sha256") != QUESTION_PDF_SHA256:
            solution_image_leakage.append(f"{asset_path}: source PDF hash is not official question hash")
        if asset.get("contains_solution") is not False:
            solution_image_leakage.append(f"{asset_path}: contains_solution is not false")
    errors.extend(image_hash_failures)

    asset_refs_verified = 0
    visual_required_count = 0
    for item in items:
        if item["visual_dependency"]:
            visual_required_count += 1
        if item.get("official_solution_image_paths"):
            solution_image_leakage.append(f"{item['item_id']}: solution image path is attached")
        if item.get("staticization", {}).get("solution_leakage_check") != "passed":
            solution_image_leakage.append(f"{item['item_id']}: staticization leakage audit not passed")
        paths = item.get("question_image_paths") or []
        if not paths:
            solution_image_leakage.append(f"{item['item_id']}: no direct official question page image")
        if paths != item["visual_dependency_metadata"]["question_image_paths"]:
            image_hash_failures.append(f"{item['item_id']}: flat/nested image paths differ")
        item_asset_by_path = {
            asset["path"]: asset for asset in item["visual_dependency_metadata"]["assets"]
        }
        if set(paths) != set(item_asset_by_path):
            image_hash_failures.append(f"{item['item_id']}: image paths/assets differ")
        for asset_path in paths:
            asset_refs_verified += 1
            asset = item_asset_by_path.get(asset_path)
            global_asset = image_asset_by_path.get(asset_path)
            if asset is None or global_asset is None:
                image_hash_failures.append(f"{item['item_id']}: undeclared image {asset_path}")
                continue
            for field in ("sha256", "source_page", "source_pdf", "source_pdf_sha256"):
                if asset.get(field) != global_asset.get(field):
                    image_hash_failures.append(
                        f"{item['item_id']}: {asset_path} disagrees on {field}"
                    )
            if asset.get("contains_solution") is not False:
                solution_image_leakage.append(f"{item['item_id']}: image marked as containing solution")
        if item["question_source"]["source_pdf"] != QUESTION_PDF:
            solution_image_leakage.append(f"{item['item_id']}: question source is not official PDF")
        if item["question_source"]["source_pdf_sha256"] != QUESTION_PDF_SHA256:
            solution_image_leakage.append(f"{item['item_id']}: question source hash changed")
        declared_pages = {item_asset_by_path[path]["source_page"] for path in paths if path in item_asset_by_path}
        if not set(item["question_source"]["pdf_pages"]).issubset(declared_pages):
            image_hash_failures.append(
                f"{item['item_id']}: official question source page lacks a direct page asset"
            )
    errors.extend(image_hash_failures)
    errors.extend(solution_image_leakage)

    artifact_hash_failures: list[str] = []
    for artifact in manifest["artifacts"].values():
        path = workspace_path(artifact["path"])
        if not path.is_file() or sha256_file(path) != artifact["sha256"]:
            artifact_hash_failures.append(f"canonical artifact hash mismatch {artifact['path']}")
    for source_artifact in manifest["draft_provenance"]["source_artifacts"]:
        path = workspace_path(source_artifact["path"])
        if not path.is_file() or sha256_file(path) != source_artifact["sha256"]:
            artifact_hash_failures.append(f"draft provenance hash mismatch {source_artifact['path']}")
    for tool in manifest["conversion"]["tooling"].values():
        path = workspace_path(tool["path"])
        if not path.is_file() or sha256_file(path) != tool["sha256"]:
            artifact_hash_failures.append(f"conversion tool hash mismatch {tool['path']}")
    errors.extend(artifact_hash_failures)

    draft_items_path = workspace_path(
        next(
            source["path"]
            for source in manifest["draft_provenance"]["source_artifacts"]
            if source["path"].endswith("/items.jsonl")
        )
    )
    draft_items = read_jsonl(draft_items_path)
    draft_by_id = {item["id"]: item for item in draft_items}
    preservation_failures: list[str] = []
    if ids != [draft_item["id"] for draft_item in draft_items]:
        preservation_failures.append("canonical item order differs from audited draft order")
    for item in items:
        draft_item = draft_by_id.get(item["item_id"])
        if draft_item is None:
            preservation_failures.append(f"{item['item_id']}: absent from audited draft")
            continue
        comparisons = {
            "question_text_native": draft_item["question_text"],
            "question_text_staticized": draft_item["question_text_staticized"],
            "reference_answer": draft_item["official_answer_normalized"],
            "gold_solution": draft_item["official_solution_summary"],
            "grading_rubric": draft_item["derived_rubric"],
            "max_score": draft_item["max_score"],
            "question_id": draft_item["question_id"],
            "sequence_id": draft_item["sequence_id"],
        }
        for field, expected in comparisons.items():
            if item.get(field) != expected:
                preservation_failures.append(f"{item['item_id']}: {field} differs from draft")
        if item["question_source"]["pdf_pages"] != draft_item["question_source"]["pdf_pages"]:
            preservation_failures.append(f"{item['item_id']}: question source pages changed")
        if item["solution_source"]["pdf_pages"] != draft_item["solution_source"]["pdf_pages"]:
            preservation_failures.append(f"{item['item_id']}: solution source pages changed")
    errors.extend(preservation_failures)

    runner = runner_dry_validation(items_path, items)
    if not runner["passed"]:
        errors.append("pipeline runner dry validation failed")

    report = {
        "schema_version": "science_arena_dataset_validation_v1",
        "dataset_id": "ipho2026",
        "validated_at": "2026-08-28",
        "validation": "passed" if not errors else "failed",
        "manifest_sha256": sha256_file(manifest_path),
        "records": len(items),
        "unique_ids": len(set(ids)),
        "json_schema": str(schema_path.relative_to(WORKSPACE)),
        "json_schema_validator_backend": schema_validator_backend,
        "json_schema_validation": "passed_for_all_23_items" if not schema_errors else "failed",
        "sequence_counts": dict(sequence_counts),
        "sequence_turns_contiguous_and_ordered": not any("turn indexes" in error for error in errors),
        "score_by_parent": dict(parent_scores),
        "question_score_total": question_score_total,
        "derived_rubric_score_total": rubric_total,
        "per_item_criterion_sums_match_max_score": not criterion_mismatches,
        "official_answers_present": sum(item.get("official_answer") is not None for item in items),
        "official_solutions_present": sum(item.get("official_solution") is not None for item in items),
        "official_rubrics_present": sum(item.get("official_rubric") is not None for item in items),
        "derived_nonofficial_rubrics_present": sum(
            item.get("rubric_type") == "derived_from_solution_not_official" for item in items
        ),
        "gold_solutions_present": sum(bool(str(item.get("gold_solution") or "").strip()) for item in items),
        "human_visual_audited_items": sum(
            item.get("staticization", {}).get("human_visual_audited") is True for item in items
        ),
        "staticized_solution_leakage_checks_passed": sum(
            item.get("staticization", {}).get("solution_leakage_check") == "passed" for item in items
        ),
        "visual_required_items": visual_required_count,
        "stable_question_assets": len(image_asset_by_path),
        "asset_refs_verified": asset_refs_verified,
        "official_question_pdf_hash_verified": not document_hash_failures,
        "all_declared_sha256_match": not (
            document_hash_failures or image_hash_failures or artifact_hash_failures
        ),
        "solution_images_in_question_assets": len(solution_image_leakage),
        "source_draft_fields_preserved": not preservation_failures,
        "runner_dry_validation": runner,
        "errors": errors,
        "constraints": {
            "model_calls_made": False,
            "evaluation_api_calls_made": False,
            "external_api_calls_made": False,
            "keys_or_secrets_accessed": False,
        },
    }
    return report, errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()
    report, errors = validate(args.dataset)
    if args.write_report:
        write_json(args.dataset / "validation.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
