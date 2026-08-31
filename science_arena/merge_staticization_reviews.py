#!/usr/bin/env python3
"""Merge independently reviewed staticization decisions into one locked template.

The combined review template is the authority for all read-only fields and for
the aggregate candidate/canonical fingerprints.  Per-competition review files
may contribute only the seven human-decision fields.  Any duplicate, missing,
extra, or changed read-only item fails the entire merge.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import freeze_question_only_staticizations as freezer


READ_ONLY_FIELDS = (
    "benchmark_id",
    "sequence_id",
    "turn_index",
    "item_id",
    "question_text_native",
    "question_text_native_sha256",
    "question_image_paths",
    "candidate_description",
    "candidate_description_sha256",
    "input_fingerprint",
)

DECISION_FIELDS = (
    "decision",
    "replacement_description",
    "replacement_description_sha256",
    "reason",
    "reviewed_by",
    "reviewed_at",
    "reviewer_declaration",
)


def read_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"cannot read JSON object {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def require_exact_fields(
    value: Mapping[str, Any], expected: set[str], label: str
) -> None:
    missing = sorted(expected - set(value))
    extra = sorted(set(value) - expected)
    if missing or extra:
        raise ValueError(f"{label}: missing={missing}, unexpected={extra}")


def identity(item: Mapping[str, Any], label: str) -> tuple[str, str]:
    benchmark_id = item.get("benchmark_id")
    item_id = item.get("item_id")
    if not isinstance(benchmark_id, str) or not benchmark_id:
        raise ValueError(f"{label}: benchmark_id is required")
    if not isinstance(item_id, str) or not item_id:
        raise ValueError(f"{label}: item_id is required")
    return benchmark_id, item_id


def validate_review_document(document: Mapping[str, Any], label: str) -> list[dict[str, Any]]:
    require_exact_fields(document, freezer.REVIEW_TOP_LEVEL_FIELDS, label)
    if document.get("schema_version") != freezer.REVIEW_SCHEMA_VERSION:
        raise ValueError(f"{label}: expected finalized review-decisions schema")
    if document.get("required_reviewer_declaration") != freezer.REVIEWER_DECLARATION:
        raise ValueError(f"{label}: reviewer declaration contract changed")
    items = document.get("items")
    if not isinstance(items, list):
        raise ValueError(f"{label}: items must be a list")
    if document.get("expected_item_count") != len(items):
        raise ValueError(f"{label}: expected_item_count does not match items")
    for index, item in enumerate(items, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"{label} item {index}: expected an object")
        require_exact_fields(
            item,
            freezer.REVIEW_ITEM_FIELDS,
            f"{label} item {index}",
        )
    return items


def merge_reviews(
    template: Mapping[str, Any], reviews: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    require_exact_fields(template, freezer.REVIEW_TOP_LEVEL_FIELDS, "template")
    # The freezer deliberately emits a fillable decisions document so that the
    # exact same schema is validated both before and after human decisions.
    if template.get("schema_version") != freezer.REVIEW_SCHEMA_VERSION:
        raise ValueError("template schema_version mismatch")
    if template.get("required_reviewer_declaration") != freezer.REVIEWER_DECLARATION:
        raise ValueError("template reviewer declaration contract changed")
    template_items = template.get("items")
    if not isinstance(template_items, list):
        raise ValueError("template items must be a list")
    if template.get("expected_item_count") != len(template_items):
        raise ValueError("template expected_item_count does not match items")

    source_items: dict[tuple[str, str], dict[str, Any]] = {}
    for review_index, document in enumerate(reviews, start=1):
        for item_index, item in enumerate(
            validate_review_document(document, f"review {review_index}"),
            start=1,
        ):
            key = identity(item, f"review {review_index} item {item_index}")
            if key in source_items:
                raise ValueError(f"duplicate reviewed identity: {key}")
            source_items[key] = item

    merged_items: list[dict[str, Any]] = []
    template_identities: set[tuple[str, str]] = set()
    for index, template_item in enumerate(template_items, start=1):
        if not isinstance(template_item, dict):
            raise ValueError(f"template item {index}: expected an object")
        require_exact_fields(
            template_item,
            freezer.REVIEW_ITEM_FIELDS,
            f"template item {index}",
        )
        key = identity(template_item, f"template item {index}")
        if key in template_identities:
            raise ValueError(f"duplicate template identity: {key}")
        template_identities.add(key)
        source = source_items.get(key)
        if source is None:
            raise ValueError(f"missing reviewed identity: {key}")
        for field in READ_ONLY_FIELDS:
            if source.get(field) != template_item.get(field):
                raise ValueError(f"review {key}: read-only field changed: {field}")
        merged = copy.deepcopy(template_item)
        for field in DECISION_FIELDS:
            merged[field] = copy.deepcopy(source[field])
        merged_items.append(merged)

    extras = sorted(set(source_items) - template_identities)
    if extras:
        raise ValueError(f"review files contain identities outside template: {extras}")
    if len(merged_items) != int(template["expected_item_count"]):
        raise ValueError("merged item count mismatch")

    merged_document = copy.deepcopy(dict(template))
    merged_document["schema_version"] = freezer.REVIEW_SCHEMA_VERSION
    merged_document["generated_at"] = datetime.now(timezone.utc).isoformat()
    merged_document["items"] = merged_items
    return merged_document


def atomic_write_json(path: Path, document: Mapping[str, Any]) -> None:
    path = path.resolve()
    if path.exists():
        raise ValueError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(
            document,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.tmp.", dir=str(path.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Strictly merge per-competition reviews into a combined template."
    )
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    template = read_object(args.template)
    reviews = [read_object(path) for path in args.reviews]
    merged = merge_reviews(template, reviews)
    atomic_write_json(args.output, merged)
    print(
        f"merged {len(merged['items'])} reviewed items: {args.output.resolve()}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
