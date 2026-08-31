#!/usr/bin/env python3
"""Build the canonical IPhO 2026 dataset from the audited draft.

This converter is deliberately offline.  It never imports API clients, reads
credentials, or mutates ``datasets_draft``.  Official-question page renders are
copied into the canonical dataset so a frozen run does not depend on draft
paths.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
DEFAULT_DRAFT = HERE / "datasets_draft" / "ipho2026"
DEFAULT_OUTPUT = HERE / "datasets" / "ipho2026"
QUESTION_PDF_SHA256 = "25f555e5e2ff07b9720cb47347981f3d8951d8f9ce44e1e1e3bc04fd5842f729"


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
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def rel_workspace(path: Path) -> str:
    return path.resolve().relative_to(WORKSPACE.resolve()).as_posix()


def canonical_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "science-arena/ipho2026/canonical_item_schema.json",
        "title": "Science Arena canonical IPhO 2026 item",
        "type": "object",
        "required": [
            "schema_version",
            "item_id",
            "benchmark_id",
            "sequence_id",
            "turn_index",
            "global_order",
            "question_id",
            "parent_question",
            "max_score",
            "question_text_native",
            "question_text_staticized",
            "reference_answer",
            "gold_solution",
            "grading_rubric",
            "rubric_type",
            "official_answer",
            "official_solution",
            "official_rubric",
            "visual_dependency",
            "question_image_paths",
            "question_image_assets",
            "official_solution_image_paths",
            "question_source",
            "solution_source",
            "visual_dependency_metadata",
            "input_tracks",
        ],
        "properties": {
            "schema_version": {"const": "science_arena_item_v1"},
            "item_id": {"type": "string", "pattern": "^ipho2026_t[123]_[abc][1-5]$"},
            "benchmark_id": {"const": "ipho2026"},
            "sequence_id": {"enum": ["ipho2026_t1", "ipho2026_t2", "ipho2026_t3"]},
            "turn_index": {"type": "integer", "minimum": 1},
            "global_order": {"type": "integer", "minimum": 1, "maximum": 23},
            "question_id": {"type": "string", "pattern": "^T[123]-[ABC][1-5]$"},
            "parent_question": {"enum": ["T1", "T2", "T3"]},
            "max_score": {"type": "number", "exclusiveMinimum": 0},
            "question_text_native": {"type": "string", "minLength": 1},
            "question_text_staticized": {"type": "string", "minLength": 1},
            "reference_answer": {"type": ["string", "object"]},
            "gold_solution": {"type": "string", "minLength": 1},
            "gold_solution_status": {
                "const": "human_audited_summary_from_provided_solution_unconfirmed_official"
            },
            "official_answer": {"type": "null"},
            "official_solution": {"type": "null"},
            "grading_rubric": {
                "type": "object",
                "required": [
                    "status",
                    "max_score",
                    "criteria",
                    "accepted_equivalents",
                    "numeric_tolerance",
                    "error_carry_forward",
                    "source_error_handling",
                ],
                "properties": {
                    "status": {"const": "derived_from_solution_not_official"},
                    "max_score": {"type": "number", "exclusiveMinimum": 0},
                    "criteria": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": ["points", "description"],
                            "properties": {
                                "points": {"type": "number", "exclusiveMinimum": 0},
                                "description": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
            },
            "rubric_type": {"const": "derived_from_solution_not_official"},
            "official_rubric": {"type": "null"},
            "official_rubric_status": {"type": "string", "pattern": "^missing"},
            "visual_dependency": {"type": "boolean"},
            "question_image_paths": {
                "type": "array",
                "minItems": 1,
                "items": {"type": "string"},
            },
            "question_image_assets": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "required": [
                        "path",
                        "source_asset_path",
                        "source_pdf",
                        "source_pdf_sha256",
                        "source_page",
                        "sha256",
                        "contains_solution",
                    ],
                    "properties": {
                        "source_pdf_sha256": {"const": QUESTION_PDF_SHA256},
                        "contains_solution": {"const": False},
                    },
                },
            },
            "official_solution_image_paths": {"type": "array", "maxItems": 0},
            "question_source": {
                "type": "object",
                "required": [
                    "source_pdf",
                    "source_pdf_sha256",
                    "pdf_pages",
                    "printed_pages",
                    "locator",
                ],
            },
            "solution_source": {
                "type": "object",
                "required": [
                    "source_pdf",
                    "source_pdf_sha256",
                    "pdf_pages",
                    "printed_pages",
                    "locator",
                    "provenance_status",
                ],
            },
            "visual_dependency_metadata": {
                "type": "object",
                "required": [
                    "required",
                    "reason",
                    "multimodal_policy",
                    "text_only_policy",
                    "question_image_paths",
                    "assets",
                    "missing_asset_policy",
                    "validation",
                ],
                "properties": {
                    "missing_asset_policy": {"const": "block_no_fallback"},
                    "assets": {
                        "type": "array",
                        "minItems": 1,
                        "items": {
                            "type": "object",
                            "required": [
                                "path",
                                "source_asset_path",
                                "source_pdf",
                                "source_pdf_sha256",
                                "source_page",
                                "sha256",
                                "contains_solution",
                            ],
                            "properties": {
                                "source_pdf_sha256": {"const": QUESTION_PDF_SHA256},
                                "contains_solution": {"const": False},
                            },
                        },
                    },
                },
            },
            "input_tracks": {
                "type": "object",
                "required": ["native_vision", "text_staticized"],
                "properties": {
                    "native_vision": {
                        "type": "object",
                        "required": [
                            "question_field",
                            "direct_official_question_images",
                            "missing_asset_policy",
                        ],
                        "properties": {
                            "question_field": {"const": "question_text_native"},
                            "direct_official_question_images": {"const": True},
                            "missing_asset_policy": {"const": "block_no_fallback"},
                        },
                    },
                    "text_staticized": {
                        "type": "object",
                        "required": [
                            "question_field",
                            "question_only",
                            "human_visual_audited",
                            "solution_leakage_check",
                        ],
                        "properties": {
                            "question_field": {"const": "question_text_staticized"},
                            "question_only": {"const": True},
                            "human_visual_audited": {"const": True},
                            "solution_leakage_check": {"const": "passed"},
                        },
                    },
                },
            },
        },
        "additionalProperties": True,
    }


def source_hashes(documents: list[dict[str, Any]]) -> dict[str, str]:
    return {str(document["path"]): str(document["sha256"]) for document in documents}


def build(draft: Path, output: Path) -> None:
    draft = draft.resolve()
    output = output.resolve()
    if draft == output or draft in output.parents:
        raise ValueError("canonical output must not equal or be nested inside the draft directory")
    if output in draft.parents:
        raise ValueError("draft directory must not be nested inside canonical output")

    draft_manifest = read_json(draft / "manifest.json")
    draft_items = read_jsonl(draft / "items.jsonl")
    contexts = read_json(draft / "contexts.json")
    draft_image_manifest = read_json(draft / "question_images_manifest.json")
    document_hash_by_path = source_hashes(draft_manifest["documents"])

    if len(draft_items) != 23:
        raise ValueError(f"refusing to build from {len(draft_items)} draft items; expected 23")
    if document_hash_by_path.get("new-test/IPhO 2026/theoretical-exam.pdf") != QUESTION_PDF_SHA256:
        raise ValueError("draft official-question PDF hash is not the audited hash")
    for document in draft_manifest["documents"]:
        document_path = (WORKSPACE / document["path"]).resolve()
        if not document_path.is_file():
            raise FileNotFoundError(f"missing audited source document: {document['path']}")
        if sha256_file(document_path) != document["sha256"]:
            raise ValueError(f"audited source document hash mismatch: {document['path']}")
        if document_path.stat().st_size != int(document["bytes"]):
            raise ValueError(f"audited source document byte-size mismatch: {document['path']}")

    output.mkdir(parents=True, exist_ok=True)
    output_images = output / "question_images"
    output_images.mkdir(parents=True, exist_ok=True)

    canonical_asset_by_draft_path: dict[str, dict[str, Any]] = {}
    canonical_assets: list[dict[str, Any]] = []
    for source_asset in draft_image_manifest["assets"]:
        draft_asset_path = str(source_asset["path"])
        source_path = (WORKSPACE / draft_asset_path).resolve()
        destination = output_images / source_path.name
        if sha256_file(source_path) != source_asset["sha256"]:
            raise ValueError(f"draft image hash mismatch: {draft_asset_path}")
        shutil.copy2(source_path, destination)
        canonical_path = rel_workspace(destination)
        asset = copy.deepcopy(source_asset)
        asset.update(
            {
                "path": canonical_path,
                "source_asset_path": draft_asset_path,
                "kind": "official_question_full_page_render",
                "source_pdf": draft_image_manifest["source_pdf"],
                "source_pdf_sha256": draft_image_manifest["source_pdf_sha256"],
                "render_dpi": 144,
                "contains_solution": False,
            }
        )
        canonical_asset_by_draft_path[draft_asset_path] = asset
        canonical_assets.append(asset)

    canonical_items: list[dict[str, Any]] = []
    turn_by_sequence: dict[str, int] = {}
    for global_order, draft_item in enumerate(draft_items, 1):
        sequence_id = str(draft_item["sequence_id"])
        turn_by_sequence[sequence_id] = turn_by_sequence.get(sequence_id, 0) + 1
        turn_index = turn_by_sequence[sequence_id]

        draft_visual = copy.deepcopy(draft_item["visual_dependency"])
        item_assets: list[dict[str, Any]] = []
        for draft_asset in draft_visual["assets"]:
            source_asset_path = str(draft_asset["path"])
            if source_asset_path not in canonical_asset_by_draft_path:
                raise ValueError(f"{draft_item['id']}: undeclared question image {source_asset_path}")
            canonical_asset = copy.deepcopy(canonical_asset_by_draft_path[source_asset_path])
            for field in ("source_page", "sha256", "source_pdf_sha256", "contains_solution"):
                if draft_asset[field] != canonical_asset[field]:
                    raise ValueError(f"{draft_item['id']}: asset metadata mismatch for {field}")
            canonical_asset.update(
                {
                    "crop": draft_asset["crop"],
                    "render_dpi": draft_asset["render_dpi"],
                }
            )
            item_assets.append(canonical_asset)

        question_image_paths = [asset["path"] for asset in item_assets]
        draft_visual["question_image_paths"] = question_image_paths
        draft_visual["assets"] = item_assets
        draft_visual["draft_missing_asset_policy"] = draft_visual["missing_asset_policy"]
        draft_visual["missing_asset_policy"] = "block_no_fallback"
        draft_visual["validation"] = {
            "all_paths_exist": True,
            "hashes_verified": True,
            "solution_pages_included": False,
        }

        question_source = copy.deepcopy(draft_item["question_source"])
        question_source["source_pdf_sha256"] = document_hash_by_path[question_source["source_pdf"]]
        solution_source = copy.deepcopy(draft_item["solution_source"])
        solution_source["source_pdf_sha256"] = document_hash_by_path[solution_source["source_pdf"]]

        reference_answer = copy.deepcopy(draft_item["official_answer_normalized"])
        gold_solution = str(draft_item["official_solution_summary"])
        grading_rubric = copy.deepcopy(draft_item["derived_rubric"])
        canonical_item = {
            "schema_version": "science_arena_item_v1",
            "item_id": draft_item["id"],
            "benchmark_id": "ipho2026",
            "sequence_id": sequence_id,
            "turn_index": turn_index,
            "global_order": global_order,
            "question_id": draft_item["question_id"],
            "parent_question": draft_item["parent_question"],
            "section": draft_item["section"],
            "title": draft_item["title"],
            "max_score": draft_item["max_score"],
            "question_text_native": draft_item["question_text"],
            "question_text_staticized": draft_item["question_text_staticized"],
            "reference_answer": reference_answer,
            "gold_solution": gold_solution,
            "gold_solution_status": (
                "human_audited_summary_from_provided_solution_unconfirmed_official"
            ),
            "official_answer": None,
            "official_solution": None,
            "official_named_fields_note": (
                "official_answer, official_solution, and official_rubric are intentionally null; "
                "the provided solutions are not confirmed official/final"
            ),
            "grading_rubric": grading_rubric,
            "rubric_type": "derived_from_solution_not_official",
            "official_rubric": None,
            "official_rubric_status": draft_item["official_rubric_status"],
            "grading_type": "derived_rubric_llm_judge",
            "question_source": question_source,
            "solution_source": solution_source,
            "official_answer_as_printed": draft_item["official_answer_as_printed"],
            "source_defects": copy.deepcopy(draft_item["source_defects"]),
            "context_ids": copy.deepcopy(draft_item["context_ids"]),
            "staticization": copy.deepcopy(draft_item["staticization"]),
            "visual_dependency": bool(draft_visual["required"]),
            "question_image_paths": question_image_paths,
            "question_image_assets": item_assets,
            "official_solution_image_paths": [],
            "visual_dependency_metadata": draft_visual,
            "input_tracks": {
                "native_vision": {
                    "question_field": "question_text_native",
                    "direct_official_question_images": True,
                    "question_image_paths": question_image_paths,
                    "missing_asset_policy": "block_no_fallback",
                    "staticized_description_as_image_fallback": False,
                },
                "text_staticized": {
                    "question_field": "question_text_staticized",
                    "question_only": True,
                    "human_visual_audited": True,
                    "solution_leakage_check": "passed",
                    "question_image_paths_sent": [],
                },
            },
            "evaluation_readiness": copy.deepcopy(draft_item["evaluation_readiness"]),
            "draft_provenance": {
                "draft_item_schema_version": draft_item["schema_version"],
                "draft_item_id": draft_item["id"],
                "draft_items_path": rel_workspace(draft / "items.jsonl"),
                "draft_items_sha256": sha256_file(draft / "items.jsonl"),
            },
        }
        canonical_items.append(canonical_item)

    schema_path = output / "item_schema.json"
    items_path = output / "items.jsonl"
    contexts_path = output / "contexts.json"
    images_manifest_path = output / "question_images_manifest.json"
    write_json(schema_path, canonical_schema())
    write_jsonl(items_path, canonical_items)
    write_json(contexts_path, contexts)

    image_manifest = {
        "schema_version": "science_arena_question_image_assets_v1",
        "dataset_id": "ipho2026",
        "policy": {
            "source": (
                "Full-page PNG renders made directly from the official question PDF; "
                "no solution PDF content is included."
            ),
            "render_command": draft_image_manifest["policy"]["render_command"],
            "dimensions_px": draft_image_manifest["policy"]["dimensions_px"],
            "native_vision_use": "Pass item.question_image_paths directly to the model API.",
            "description_fallback_for_native_vision": False,
            "missing_required_asset_action": "block_no_fallback",
        },
        "source_pdf": draft_image_manifest["source_pdf"],
        "source_pdf_sha256": draft_image_manifest["source_pdf_sha256"],
        "assets": canonical_assets,
    }
    write_json(images_manifest_path, image_manifest)

    source_artifact_names = [
        "manifest.json",
        "contexts.json",
        "items.jsonl",
        "item_schema.json",
        "item_enrichments.json",
        "question_images_manifest.json",
        "validation.json",
    ]
    manifest = {
        "schema_version": "science_arena_dataset_manifest_v1",
        "dataset_id": "ipho2026",
        "benchmark_id": "ipho2026",
        "competition": draft_manifest["competition"],
        "year": draft_manifest["year"],
        "discipline": draft_manifest["discipline"],
        "exam_component": draft_manifest["exam_component"],
        "language": draft_manifest["language"],
        "exam_date": draft_manifest["exam_date"],
        "canonicalized_at": "2026-08-28",
        "status": "ready_for_derived_rubric_eval",
        "status_reason": draft_manifest["status_reason"],
        "evaluation_readiness": copy.deepcopy(draft_manifest["evaluation_readiness"]),
        "rubric_policy": {
            "grading_field": "grading_rubric",
            "rubric_type": "derived_from_solution_not_official",
            "official_rubrics_present": 0,
            "official_rubric_field_is_null_for_every_item": True,
            "official_answer_and_solution_fields_are_null_for_every_item": True,
            "claim_boundary": "Scores are derived-rubric scores, not reproduced official IPhO marks.",
        },
        "input_tracks": {
            "native_vision": {
                "question_field": "question_text_native",
                "image_field": "question_image_paths",
                "image_kind": "direct full-page official-question PNG",
                "missing_asset_policy": "block_no_fallback",
                "description_fallback": False,
            },
            "text_staticized": {
                "question_field": "question_text_staticized",
                "question_only": True,
                "human_visual_audited": True,
                "solution_leakage_check": "passed_for_all_23_items",
                "image_payload": False,
            },
        },
        "documents": copy.deepcopy(draft_manifest["documents"]),
        "content_counts": copy.deepcopy(draft_manifest["content_counts"]),
        "score_reconciliation": copy.deepcopy(draft_manifest["score_reconciliation"]),
        "item_order": [item["item_id"] for item in canonical_items],
        "sequence_order": ["ipho2026_t1", "ipho2026_t2", "ipho2026_t3"],
        "conversion": {
            "source_item_schema": "ipho2026_item_draft_v1",
            "target_item_schema": "science_arena_item_v1",
            "field_mappings": {
                "id": "item_id",
                "question_text": "question_text_native",
                "official_answer_normalized": "reference_answer",
                "official_solution_summary": "gold_solution",
                "derived_rubric": "grading_rubric",
                "visual_dependency.required": "visual_dependency",
                "visual_dependency.assets": "question_image_assets and visual_dependency_metadata.assets",
            },
            "schema_conflicts_and_minimal_resolutions": [
                {
                    "conflict": (
                        "The draft correctly stores official_rubric=null, while the legacy runner "
                        "required a nonempty field named official_rubric."
                    ),
                    "resolution": (
                        "Use grading_rubric as the preferred runner field, keep official_rubric null, "
                        "and require rubric_type=derived_from_solution_not_official."
                    ),
                },
                {
                    "conflict": (
                        "Legacy runner fields called official_answer and official_solution could "
                        "overstate the provenance of the provided, unconfirmed solution PDFs."
                    ),
                    "resolution": (
                        "Use reference_answer and gold_solution as preferred fields and fix all "
                        "three official-named fields to null in the canonical schema."
                    ),
                },
                {
                    "conflict": (
                        "The draft stores visual_dependency as an audit object, while the runner "
                        "uses a top-level boolean."
                    ),
                    "resolution": (
                        "Flatten required to visual_dependency, expose question_image_paths and "
                        "question_image_assets for the runner, and preserve the full object as "
                        "visual_dependency_metadata."
                    ),
                },
                {
                    "conflict": (
                        "Two nonvisual draft items use the equivalent policy label "
                        "block_multimodal_if_declared_path_missing."
                    ),
                    "resolution": (
                        "Normalize every canonical native-vision item to block_no_fallback and "
                        "preserve the draft label as draft_missing_asset_policy."
                    ),
                },
            ],
            "draft_overwritten": False,
            "validation_report": rel_workspace(output / "validation.json"),
            "tooling": {
                "builder": {
                    "path": rel_workspace(Path(__file__)),
                    "sha256": sha256_file(Path(__file__)),
                },
                "validator": {
                    "path": rel_workspace(HERE / "validate_ipho2026_canonical.py"),
                    "sha256": sha256_file(HERE / "validate_ipho2026_canonical.py"),
                },
            },
        },
        "artifacts": {
            "items": {"path": rel_workspace(items_path), "sha256": sha256_file(items_path)},
            "item_schema": {"path": rel_workspace(schema_path), "sha256": sha256_file(schema_path)},
            "contexts": {"path": rel_workspace(contexts_path), "sha256": sha256_file(contexts_path)},
            "question_images_manifest": {
                "path": rel_workspace(images_manifest_path),
                "sha256": sha256_file(images_manifest_path),
            },
        },
        "draft_provenance": {
            "source_directory": rel_workspace(draft),
            "source_artifacts": [
                {
                    "path": rel_workspace(draft / name),
                    "sha256": sha256_file(draft / name),
                }
                for name in source_artifact_names
            ],
            "draft_directory_mutated": False,
        },
        "constraints": {
            "model_calls_made": False,
            "evaluation_api_calls_made": False,
            "external_api_calls_made": False,
            "keys_or_secrets_accessed": False,
        },
    }
    write_json(output / "manifest.json", manifest)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.draft, args.output)
    print(f"Built canonical IPhO 2026 dataset at {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
