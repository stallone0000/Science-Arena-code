#!/usr/bin/env python3
"""Freeze reviewed question-only staticization candidates without API access.

This is an offline, fail-closed merge step.  It reads canonical items and the
candidate JSONL emitted by ``staticize_question_only.py``, verifies the exact
staticizer request/fingerprint/image/candidate hashes, and requires one
hash-locked human review decision for every item before it can create output.

The canonical JSONL files are never edited.  A successful freeze is published
as one atomically-renamed, independent tree containing derived item JSONLs and
a question-only provenance sidecar.  The sidecar is deliberately built from a
small allowlist and never copies answer, solution, rubric, or other gold fields
from the canonical rows.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
import threading
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import staticize_question_only as staticizer


FREEZE_SCHEMA_VERSION = "science_arena_2026_question_staticization_freeze_v1"
PROVENANCE_SCHEMA_VERSION = (
    "science_arena_2026_question_staticization_freeze_provenance_v1"
)
REVIEW_SCHEMA_VERSION = "science_arena_2026_staticization_review_decisions_v1"
REVIEW_TEMPLATE_SCHEMA_VERSION = (
    "science_arena_2026_staticization_review_template_v1"
)

NO_ADDITIONAL_IMAGE_INFORMATION = "[NO ADDITIONAL IMAGE INFORMATION]"
VISUAL_ADDENDUM_PREFIX = (
    "\n\n<<<FROZEN GEMINI QUESTION-ONLY VISUAL ADDENDUM>>>\n"
)
VISUAL_ADDENDUM_SUFFIX = "\n<<<END FROZEN GEMINI QUESTION-ONLY VISUAL ADDENDUM>>>"

REVIEWER_DECLARATION = (
    "I reviewed only the question text and official question image(s). I attest "
    "that this decision corrects visual transcription only and adds no answer, "
    "solution, answer-directed reasoning, scoring guidance, or rubric content."
)

EXPECTED_2026_GROUP_COUNTS = {
    "icho2026_theory": 68,
    "ipho2026": 23,
    "nbpho2026": 28,
    "usapho2026": 34,
}
EXPECTED_2026_TOTAL = sum(EXPECTED_2026_GROUP_COUNTS.values())

WIRE_GENERATION_STATICIZER_SHA256 = (
    "d8d929e71c030982b0c39470bf70356c350436a84c7ee8ec4d081282c77b0768"
)
WIRE_GENERATION_STATICIZER_ARCHIVE = (
    Path(__file__).resolve().parent
    / "staticizer_source_archive"
    / "staticize_question_only.d8d929e71c030982.py"
)
WIRE_GENERATION_STATICIZER_ARCHIVE_README = (
    WIRE_GENERATION_STATICIZER_ARCHIVE.parent / "README.md"
)

HEX_SHA256 = re.compile(r"[0-9a-f]{64}")
SAFE_GROUP = re.compile(r"[A-Za-z0-9_.-]+")
SENSITIVE_CANONICAL_KEY = re.compile(
    r"(?:gold|answer|solution|rubric|criteri|grading|adjudicat)", re.I
)
DIRECT_LEAKAGE_PATTERNS = (
    re.compile(r"<\s*/?\s*think\b", re.I),
    re.compile(r"\bfinal\s+answer\b", re.I),
    re.compile(
        r"\b(?:the\s+)?(?:correct\s+)?answer\s*(?::|=|(?:is|equals?)\b)",
        re.I,
    ),
    re.compile(
        r"\b(?:the\s+)?solution\s*(?::|=|(?:is|equals?)\s+"
        r"(?=(?:[-+]?(?:\d|\.\d)|[A-Za-z]\s*=|[A-H](?:[.)]|\b)|"
        r"\\?boxed\b|true\b|false\b)))",
        re.I,
    ),
    re.compile(r"\b(?:the\s+)?correct\s+answer\s*(?:is|:)", re.I),
    re.compile(r"\b(?:the\s+)?answer\s+(?:is|equals?)\b", re.I),
    re.compile(r"\b(?:official|worked)\s+solution\b", re.I),
    re.compile(r"\bsolution\s*:\s*", re.I),
    re.compile(r"\bgrading\s+rubric\b", re.I),
    re.compile(r"\baward(?:ed)?\s+\d+(?:\.\d+)?\s+(?:marks?|points?)\b", re.I),
    re.compile(r"\b(?:i|we)\s+(?:calculated?|derived?|deduced?|solved?)\b", re.I),
)

# Short rubric fragments such as "3 points" or "repeat unit." routinely occur
# in legitimate question-figure transcriptions.  Substring matching is reserved
# for longer, answer-specific material; short sensitive text is still blocked
# when it is the entire candidate/replacement and by the explicit leakage
# patterns above.
MIN_SENSITIVE_SUBSTRING_LENGTH = 16

CANDIDATE_ALLOWED_FIELDS = {
    "schema_version",
    "terminal",
    "created_at",
    "status",
    "route",
    "benchmark_id",
    "sequence_id",
    "turn_index",
    "item_id",
    "question_text_native_sha256",
    "question_source_fingerprint",
    "candidate_description",
    "candidate_description_sha256",
    "candidate_requires_human_review",
    "candidate_only_not_applied",
    "canonical_writeback_performed",
    "input_fingerprint",
    "request_sha256",
    "wire_attempt_limit",
    "wire_timeout_seconds",
    "wire_attempt_count",
    "wire_attempts",
    "current_question_images",
    "cumulative_question_images",
    "leakage_audit",
    "raw_response_artifact",
    "response_message_artifact",
    "reasoning_artifacts",
    "usage_artifact",
    "reasoning_tokens",
    "audit_record_path",
}

REVIEW_TOP_LEVEL_FIELDS = {
    "schema_version",
    "generated_at",
    "candidate_jsonl_sha256",
    "canonical_input_fingerprint",
    "expected_item_count",
    "required_reviewer_declaration",
    "instructions",
    "items",
}

REVIEW_ITEM_FIELDS = {
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
    "decision",
    "replacement_description",
    "replacement_description_sha256",
    "reason",
    "reviewed_by",
    "reviewed_at",
    "reviewer_declaration",
}


@dataclass(frozen=True)
class CanonicalRecord:
    row: dict[str, Any]
    source_path: Path
    line_number: int

    @property
    def identity(self) -> tuple[str, str]:
        return (str(self.row["benchmark_id"]), str(self.row["item_id"]))


@dataclass(frozen=True)
class CandidateRecord:
    row: dict[str, Any]
    source_path: Path
    line_number: int

    @property
    def identity(self) -> tuple[str, str]:
        return (str(self.row["benchmark_id"]), str(self.row["item_id"]))


@dataclass(frozen=True)
class PreparedInputs:
    canonical: tuple[CanonicalRecord, ...]
    candidates: Mapping[tuple[str, str], CandidateRecord]
    canonical_paths: tuple[Path, ...]
    candidate_path: Path
    canonical_input_fingerprint: str
    candidate_jsonl_sha256: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    return sha256_bytes(value.encode("utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def pretty_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def jsonl_bytes(rows: Iterable[dict[str, Any]]) -> bytes:
    return b"".join(canonical_json_bytes(row) + b"\n" for row in rows)


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    try:
        with temporary.open("xb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"invalid JSON file {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    try:
        handle = path.open(encoding="utf-8")
    except OSError as exc:
        raise ValueError(f"cannot read JSONL {path}: {exc}") from exc
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except Exception as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append((line_number, value))
    return rows


def require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or HEX_SHA256.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 hex digest")
    return value


def require_exact_fields(value: Mapping[str, Any], allowed: set[str], label: str) -> None:
    missing = sorted(allowed - set(value))
    unexpected = sorted(set(value) - allowed)
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")
    if unexpected:
        raise ValueError(f"{label} has unexpected fields: {unexpected}")


def discover_canonical_paths(values: Sequence[Path]) -> tuple[Path, ...]:
    discovered: list[Path] = []
    for raw in values:
        path = raw.resolve()
        if path.is_file():
            discovered.append(path)
            continue
        if not path.is_dir():
            raise ValueError(f"canonical input does not exist: {raw}")
        direct = path / "items.jsonl"
        matches = [direct] if direct.is_file() else sorted(path.glob("*/items.jsonl"))
        if not matches:
            raise ValueError(f"no items.jsonl found directly or one level below {path}")
        discovered.extend(match.resolve() for match in matches)
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in discovered:
        if path in seen:
            raise ValueError(f"duplicate canonical input path: {path}")
        seen.add(path)
        unique.append(path)
    if not unique:
        raise ValueError("at least one canonical items JSONL is required")
    return tuple(unique)


def load_canonical_records(paths: Sequence[Path]) -> tuple[CanonicalRecord, ...]:
    records: list[CanonicalRecord] = []
    seen: dict[tuple[str, str], CanonicalRecord] = {}
    seen_turns: dict[tuple[str, str, int], tuple[str, str]] = {}
    for path in paths:
        for line_number, row in read_jsonl(path):
            label = f"{path}:{line_number}"
            for field in ("benchmark_id", "sequence_id", "item_id"):
                if not isinstance(row.get(field), str) or not row[field]:
                    raise ValueError(f"{label}: {field} must be non-empty text")
            if not isinstance(row.get("turn_index"), int) or isinstance(
                row.get("turn_index"), bool
            ):
                raise ValueError(f"{label}: turn_index must be an integer")
            turn_index = row["turn_index"]
            if turn_index <= 0:
                raise ValueError(f"{label}: turn_index must be a positive integer")
            native = row.get("question_text_native")
            previous = row.get("question_text_staticized")
            image_paths = row.get("question_image_paths")
            if not isinstance(native, str) or not native.strip():
                raise ValueError(f"{label}: question_text_native must be non-empty text")
            if not isinstance(previous, str) or not previous.strip():
                raise ValueError(
                    f"{label}: existing question_text_staticized must be non-empty text"
                )
            if (
                not isinstance(image_paths, list)
                or not image_paths
                or any(not isinstance(item, str) or not item for item in image_paths)
            ):
                raise ValueError(
                    f"{label}: question_image_paths must be a non-empty list of paths"
                )
            record = CanonicalRecord(row=row, source_path=path, line_number=line_number)
            if record.identity in seen:
                first = seen[record.identity]
                raise ValueError(
                    f"duplicate canonical item {record.identity}: "
                    f"{first.source_path}:{first.line_number} and {label}"
                )
            seen[record.identity] = record
            turn_key = (str(row["benchmark_id"]), str(row["sequence_id"]), turn_index)
            if turn_key in seen_turns:
                raise ValueError(
                    f"duplicate sequence turn {turn_key}: items "
                    f"{seen_turns[turn_key]} and {record.identity}"
                )
            seen_turns[turn_key] = record.identity
            records.append(record)
    if not records:
        raise ValueError("canonical inputs contain no items")

    grouped: dict[tuple[str, str], list[int]] = {}
    for record in records:
        row = record.row
        grouped.setdefault(
            (str(row["benchmark_id"]), str(row["sequence_id"])), []
        ).append(int(row["turn_index"]))
    for identity, turns in grouped.items():
        ordered = sorted(turns)
        if ordered != list(range(1, len(ordered) + 1)):
            raise ValueError(
                f"sequence {identity} must have contiguous turn_index values from 1; "
                f"got {ordered}"
            )
    return tuple(records)


def validate_expected_inventory(
    records: Sequence[CanonicalRecord],
    *,
    expected_item_count: int | None,
    require_complete_153: bool,
) -> None:
    if expected_item_count is not None and len(records) != expected_item_count:
        raise ValueError(
            f"expected {expected_item_count} canonical items, found {len(records)}"
        )
    if require_complete_153:
        counts = Counter(record.identity[0] for record in records)
        if len(records) != EXPECTED_2026_TOTAL or dict(counts) != EXPECTED_2026_GROUP_COUNTS:
            raise ValueError(
                "complete-153 inventory mismatch: expected "
                f"{EXPECTED_2026_GROUP_COUNTS} (total {EXPECTED_2026_TOTAL}), "
                f"found {dict(sorted(counts.items()))} (total {len(records)})"
            )


def canonical_input_fingerprint(
    records: Sequence[CanonicalRecord], paths: Sequence[Path]
) -> str:
    material = {
        "source_files": [
            {"path": str(path), "sha256": sha256_file(path)} for path in paths
        ],
        "items": [
            {
                "benchmark_id": record.identity[0],
                "item_id": record.identity[1],
                "sequence_id": record.row["sequence_id"],
                "turn_index": int(record.row["turn_index"]),
                "question_text_native_sha256": sha256_text(
                    record.row["question_text_native"]
                ),
                "question_text_staticized_sha256": sha256_text(
                    record.row["question_text_staticized"]
                ),
                "question_image_paths": record.row["question_image_paths"],
                "source_path": str(record.source_path),
                "source_line": record.line_number,
            }
            for record in records
        ],
    }
    return sha256_bytes(canonical_json_bytes(material))


def load_candidate_records(
    path: Path, canonical_by_id: Mapping[tuple[str, str], CanonicalRecord]
) -> dict[tuple[str, str], CandidateRecord]:
    records: dict[tuple[str, str], CandidateRecord] = {}
    for line_number, row in read_jsonl(path):
        label = f"{path}:{line_number}"
        require_exact_fields(row, CANDIDATE_ALLOWED_FIELDS, label)
        for field in ("benchmark_id", "sequence_id", "item_id"):
            if not isinstance(row.get(field), str) or not row[field]:
                raise ValueError(f"{label}: {field} must be non-empty text")
        record = CandidateRecord(row=row, source_path=path, line_number=line_number)
        if record.identity in records:
            raise ValueError(f"duplicate candidate item {record.identity}")
        if record.identity not in canonical_by_id:
            raise ValueError(f"candidate has no canonical item: {record.identity}")
        records[record.identity] = record
    missing = sorted(set(canonical_by_id) - set(records))
    if missing:
        raise ValueError(f"missing candidate rows for canonical items: {missing}")
    extras = sorted(set(records) - set(canonical_by_id))
    if extras:
        raise ValueError(f"unexpected candidate rows: {extras}")
    return records


def normalize_for_leak_check(value: str) -> str:
    return " ".join(value.casefold().split())


def sensitive_canonical_strings(row: Mapping[str, Any]) -> set[str]:
    values: set[str] = set()

    def visit(value: Any, sensitive: bool = False) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, sensitive or bool(SENSITIVE_CANONICAL_KEY.search(str(key))))
        elif isinstance(value, list):
            for child in value:
                visit(child, sensitive)
        elif sensitive and isinstance(value, str):
            normalized = normalize_for_leak_check(value)
            if normalized:
                values.add(normalized)

    visit(row)
    # A word or formula can legitimately occur on both sides of the benchmark
    # (for example a topology label printed in the question and repeated in a
    # rubric).  Such text is not answer-exclusive and must remain available to
    # a question-image transcription.  The old human-audited staticization is
    # used only as an offline allowlist here; it is never added to the fixed
    # Gemini request.  Gold strings absent from both question-side texts remain
    # protected by the exact-copy guard below.
    authorized_question_text: list[str] = []

    def collect_question_side(value: Any) -> None:
        if isinstance(value, dict):
            for child in value.values():
                collect_question_side(child)
        elif isinstance(value, list):
            for child in value:
                collect_question_side(child)
        elif isinstance(value, str):
            normalized = normalize_for_leak_check(value)
            if normalized:
                authorized_question_text.append(normalized)

    for key in (
        "question_text_native",
        "question_text_staticized",
        "question_image_paths",
        "question_image_refs",
        "question_image_assets",
    ):
        collect_question_side(row.get(key))
    return {
        value
        for value in values
        if not any(value in question for question in authorized_question_text)
    }


def assert_question_only_text(text: str, canonical: CanonicalRecord, label: str) -> None:
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{label} must be non-empty text")
    # Manual LaTeX transcriptions can be silently corrupted when strings such
    # as ``\frac`` or ``\alpha`` are constructed with unescaped backslashes:
    # ``\f``/``\a`` then become C0 controls in the parsed JSON.  Newlines are
    # the only control character needed in a visual description; fail closed
    # on tabs, carriage returns, DEL, and every other C0 character.
    illegal_controls = sorted(
        {
            ord(character)
            for character in text
            if (ord(character) < 32 and character != "\n")
            or ord(character) == 127
        }
    )
    if illegal_controls:
        rendered = ", ".join(f"U+{codepoint:04X}" for codepoint in illegal_controls)
        raise ValueError(f"{label} contains forbidden control characters: {rendered}")
    if "data:image/" in text.casefold():
        raise ValueError(f"{label} contains an inline image payload")
    for pattern in DIRECT_LEAKAGE_PATTERNS:
        if pattern.search(text):
            raise ValueError(
                f"{label} violates the answer/solution/reasoning/rubric guard: "
                f"{pattern.pattern}"
            )
    normalized = normalize_for_leak_check(text)
    for sensitive in sensitive_canonical_strings(canonical.row):
        if (
            len(sensitive) >= MIN_SENSITIVE_SUBSTRING_LENGTH
            and sensitive in normalized
        ) or (
            len(sensitive) >= 3 and sensitive == normalized
        ):
            raise ValueError(
                f"{label} contains text copied from a canonical answer/solution/rubric field"
            )


def offline_image_input(source_path: str, resolved_path: Path) -> staticizer.ImageInput:
    raw = resolved_path.read_bytes()
    mime_type = staticizer.detect_image_mime(raw, resolved_path)
    data_url = f"data:{mime_type};base64,{base64.b64encode(raw).decode('ascii')}"
    digest = sha256_bytes(raw)
    return staticizer.ImageInput(
        source_path=source_path,
        resolved_path=resolved_path,
        mime_type=mime_type,
        raw=raw,
        data_url=data_url,
        sha256=digest,
        data_url_sha256=sha256_text(data_url),
        blob_path=f"blobs/data-uri/{digest}.bin",
    )


def validate_candidate_core(
    candidate: CandidateRecord, canonical: CanonicalRecord
) -> None:
    row = candidate.row
    label = f"candidate {candidate.identity}"
    if row.get("schema_version") != staticizer.CANDIDATE_SCHEMA_VERSION:
        raise ValueError(f"{label}: unsupported candidate schema_version")
    if row.get("terminal") is not True or row.get("status") != "candidate_ready":
        raise ValueError(f"{label}: only terminal candidate_ready rows may be frozen")
    if row.get("route") != staticizer.FIXED_ROUTE:
        raise ValueError(f"{label}: staticizer route mismatch")
    if row.get("candidate_requires_human_review") is not True:
        raise ValueError(f"{label}: candidate_requires_human_review must be true")
    if row.get("candidate_only_not_applied") is not True:
        raise ValueError(f"{label}: candidate_only_not_applied must be true")
    if row.get("canonical_writeback_performed") is not False:
        raise ValueError(f"{label}: candidate claims canonical writeback")
    if str(row.get("sequence_id")) != str(canonical.row["sequence_id"]):
        raise ValueError(f"{label}: sequence_id does not match canonical item")
    if not isinstance(row.get("turn_index"), int) or isinstance(
        row.get("turn_index"), bool
    ):
        raise ValueError(f"{label}: invalid turn_index")
    turn_index = row["turn_index"]
    if turn_index != int(canonical.row["turn_index"]):
        raise ValueError(f"{label}: turn_index does not match canonical item")
    native_hash = sha256_text(canonical.row["question_text_native"])
    if row.get("question_text_native_sha256") != native_hash:
        raise ValueError(f"{label}: question_text_native hash mismatch")
    candidate_text = row.get("candidate_description")
    if not isinstance(candidate_text, str):
        raise ValueError(f"{label}: candidate_description must be text")
    candidate_hash = require_sha256(
        row.get("candidate_description_sha256"),
        f"{label} candidate_description_sha256",
    )
    if sha256_text(candidate_text) != candidate_hash:
        raise ValueError(f"{label}: candidate description hash mismatch")
    valid, reason = staticizer.validate_candidate_description(candidate_text, "stop")
    if not valid:
        raise ValueError(f"{label}: invalid candidate description: {reason}")
    assert_question_only_text(candidate_text, canonical, f"{label} description")
    for field in ("question_source_fingerprint", "input_fingerprint", "request_sha256"):
        require_sha256(row.get(field), f"{label} {field}")
    if row.get("wire_attempt_limit") != staticizer.WIRE_ATTEMPTS:
        raise ValueError(f"{label}: wire attempt limit mismatch")
    if row.get("wire_timeout_seconds") != staticizer.WIRE_TIMEOUT_SECONDS:
        raise ValueError(f"{label}: wire timeout mismatch")
    if not isinstance(row.get("wire_attempt_count"), int) or isinstance(
        row.get("wire_attempt_count"), bool
    ):
        raise ValueError(f"{label}: wire_attempt_count must be an integer")
    attempt_count = row["wire_attempt_count"]
    attempts = row.get("wire_attempts")
    if (
        attempt_count <= 0
        or attempt_count > staticizer.WIRE_ATTEMPTS
        or not isinstance(attempts, list)
        or len(attempts) != attempt_count
    ):
        raise ValueError(f"{label}: invalid wire-attempt summary")
    # This also applies the strict allowlists for attempt summaries and all
    # persisted artifact pointers before a review template can be emitted.
    candidate_provenance(row)
    for ordinal, attempt in enumerate(attempts, start=1):
        if attempt.get("attempt") != ordinal:
            raise ValueError(f"{label}: non-contiguous wire-attempt ordinals")
    passing_attempts = [
        ordinal
        for ordinal, attempt in enumerate(attempts, start=1)
        if attempt.get("validation_ok") is True
    ]
    if passing_attempts != [attempt_count]:
        raise ValueError(
            f"{label}: candidate_ready must end at exactly one passing wire attempt"
        )
    audit = row.get("leakage_audit")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise ValueError(f"{label}: candidate lacks a passing leakage audit")
    if audit.get("forbidden_source_fields_used") not in (None, []):
        raise ValueError(f"{label}: leakage audit used forbidden source fields")
    if audit.get("gold_field_count") not in (None, 0):
        raise ValueError(f"{label}: leakage audit reports gold fields")
    if audit.get("solution_image_count") not in (None, 0):
        raise ValueError(f"{label}: leakage audit reports solution images")
    if audit.get("official_solution_image_overlap_count") not in (None, 0):
        raise ValueError(f"{label}: solution/question image overlap was reported")
    for field in ("current_question_images", "cumulative_question_images"):
        images = row.get(field)
        if not isinstance(images, list) or not images:
            raise ValueError(f"{label}: {field} must be a non-empty list")


def verify_candidate_requests(
    canonical: Sequence[CanonicalRecord],
    candidates: Mapping[tuple[str, str], CandidateRecord],
    *,
    workspace: Path,
) -> None:
    canonical_by_id = {record.identity: record for record in canonical}
    for identity, candidate in candidates.items():
        validate_candidate_core(candidate, canonical_by_id[identity])

    turns = staticizer.project_question_turns(
        [record.row for record in canonical], workspace=workspace
    )
    turns_by_sequence: dict[tuple[str, str], list[staticizer.QuestionTurn]] = {}
    for turn in turns:
        turns_by_sequence.setdefault((turn.benchmark_id, turn.sequence_id), []).append(turn)

    for sequence_key, sequence in sorted(turns_by_sequence.items()):
        history: list[staticizer.HistoryEntry] = []
        cumulative: list[staticizer.ImageInput] = []
        cache: dict[Path, staticizer.ImageInput] = {}
        seen_paths: set[Path] = set()
        for turn in sorted(sequence, key=lambda item: item.turn_index):
            identity = (turn.benchmark_id, turn.item_id)
            candidate = candidates[identity]
            current: list[staticizer.ImageInput] = []
            for source_path, resolved_path in zip(
                turn.question_image_values, turn.question_image_paths
            ):
                image = cache.get(resolved_path)
                if image is None:
                    image = offline_image_input(source_path, resolved_path)
                    cache[resolved_path] = image
                current.append(image)
                if resolved_path not in seen_paths:
                    cumulative.append(image)
                    seen_paths.add(resolved_path)

            expected_current = [image.descriptor() for image in current]
            expected_cumulative = [image.descriptor() for image in cumulative]
            if candidate.row["current_question_images"] != expected_current:
                raise ValueError(f"candidate {identity}: current question-image provenance mismatch")
            if candidate.row["cumulative_question_images"] != expected_cumulative:
                raise ValueError(
                    f"candidate {identity}: cumulative question-image provenance mismatch"
                )
            source_fingerprint = sha256_bytes(
                canonical_json_bytes(
                    {
                        "question_text_native": turn.question_text_native,
                        "question_image_sha256": [image.sha256 for image in current],
                    }
                )
            )
            if candidate.row["question_source_fingerprint"] != source_fingerprint:
                raise ValueError(f"candidate {identity}: question source fingerprint mismatch")
            fingerprint = staticizer.input_fingerprint(
                history=history,
                current_turn=turn,
                cumulative_images=cumulative,
            )
            if candidate.row["input_fingerprint"] != fingerprint:
                raise ValueError(f"candidate {identity}: input fingerprint mismatch")
            payload = staticizer.build_payload(
                history=history,
                current_turn=turn,
                cumulative_images=cumulative,
            )
            expected_audit = staticizer.audit_payload(
                payload=payload,
                history=history,
                current_turn=turn,
                cumulative_images=cumulative,
            )
            if candidate.row["leakage_audit"] != expected_audit:
                raise ValueError(f"candidate {identity}: leakage audit does not match request")
            request_hash = sha256_bytes(canonical_json_bytes(payload))
            if candidate.row["request_sha256"] != request_hash:
                raise ValueError(f"candidate {identity}: exact request hash mismatch")
            if any(
                attempt.get("request_sha256") != request_hash
                for attempt in candidate.row["wire_attempts"]
                if isinstance(attempt, dict)
            ):
                raise ValueError(f"candidate {identity}: wire-attempt request hash mismatch")
            history.append(
                staticizer.HistoryEntry(
                    turn.question_text_native,
                    candidate.row["candidate_description"],
                )
            )


def prepare_inputs(
    *,
    canonical_inputs: Sequence[Path],
    candidate_path: Path,
    workspace: Path,
    expected_item_count: int | None = None,
    require_complete_153: bool = False,
) -> PreparedInputs:
    paths = discover_canonical_paths(canonical_inputs)
    candidate_path = candidate_path.resolve()
    if not candidate_path.is_file():
        raise ValueError(f"candidate JSONL does not exist: {candidate_path}")
    canonical = load_canonical_records(paths)
    validate_expected_inventory(
        canonical,
        expected_item_count=expected_item_count,
        require_complete_153=require_complete_153,
    )
    canonical_by_id = {record.identity: record for record in canonical}
    candidates = load_candidate_records(candidate_path, canonical_by_id)
    verify_candidate_requests(
        canonical,
        candidates,
        workspace=workspace.resolve(),
    )
    return PreparedInputs(
        canonical=canonical,
        candidates=candidates,
        canonical_paths=paths,
        candidate_path=candidate_path,
        canonical_input_fingerprint=canonical_input_fingerprint(canonical, paths),
        candidate_jsonl_sha256=sha256_file(candidate_path),
    )


def review_item_template(
    canonical: CanonicalRecord, candidate: CandidateRecord
) -> dict[str, Any]:
    return {
        "benchmark_id": canonical.row["benchmark_id"],
        "sequence_id": canonical.row["sequence_id"],
        "turn_index": int(canonical.row["turn_index"]),
        "item_id": canonical.row["item_id"],
        "question_text_native": canonical.row["question_text_native"],
        "question_text_native_sha256": sha256_text(
            canonical.row["question_text_native"]
        ),
        "question_image_paths": list(canonical.row["question_image_paths"]),
        "candidate_description": candidate.row["candidate_description"],
        "candidate_description_sha256": candidate.row[
            "candidate_description_sha256"
        ],
        "input_fingerprint": candidate.row["input_fingerprint"],
        "decision": "",
        "replacement_description": "",
        "replacement_description_sha256": "",
        "reason": "",
        "reviewed_by": "",
        "reviewed_at": "",
        "reviewer_declaration": "",
    }


def build_review_template(prepared: PreparedInputs) -> dict[str, Any]:
    return {
        # The generated document already has the decisions schema.  Its blank
        # decision/audit fields intentionally make it invalid until a reviewer
        # completes every item.
        "schema_version": REVIEW_SCHEMA_VERSION,
        "generated_at": utc_now(),
        "candidate_jsonl_sha256": prepared.candidate_jsonl_sha256,
        "canonical_input_fingerprint": prepared.canonical_input_fingerprint,
        "expected_item_count": len(prepared.canonical),
        "required_reviewer_declaration": REVIEWER_DECLARATION,
        "instructions": (
            "For every item set decision to pass, replace, or reject. Pass keeps the "
            "hash-locked candidate verbatim. Replace may correct visible transcription "
            "only and requires replacement text, its SHA-256, a non-empty reason, the "
            "reviewer identity/time, and the exact declaration. Reject requires a reason "
            "and blocks the entire freeze. Do not consult or copy answers, solutions, "
            "scoring guidance, or rubrics."
        ),
        "items": [
            review_item_template(record, prepared.candidates[record.identity])
            for record in prepared.canonical
        ],
    }


def write_review_template(path: Path, prepared: PreparedInputs) -> None:
    output = path.resolve()
    inputs = set(prepared.canonical_paths) | {prepared.candidate_path}
    if output in inputs:
        raise ValueError("review template output must not overwrite an input file")
    if output.exists():
        raise ValueError(f"review template output already exists: {output}")
    atomic_write_bytes(output, pretty_json_bytes(build_review_template(prepared)))


def parse_review_time(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} reviewed_at must be a non-empty RFC 3339 timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception as exc:
        raise ValueError(f"{label} reviewed_at is not a valid RFC 3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(f"{label} reviewed_at must include a timezone")
    return value


def validate_review_decisions(
    path: Path, prepared: PreparedInputs
) -> dict[tuple[str, str], dict[str, Any]]:
    document = read_json(path)
    if not isinstance(document, dict):
        raise ValueError("review decisions must be one JSON object")
    require_exact_fields(document, REVIEW_TOP_LEVEL_FIELDS, "review decisions")
    if document.get("schema_version") != REVIEW_SCHEMA_VERSION:
        raise ValueError("review decisions schema_version mismatch")
    if document.get("candidate_jsonl_sha256") != prepared.candidate_jsonl_sha256:
        raise ValueError("review decisions are locked to a different candidate JSONL hash")
    if (
        document.get("canonical_input_fingerprint")
        != prepared.canonical_input_fingerprint
    ):
        raise ValueError("review decisions are locked to different canonical inputs")
    if document.get("expected_item_count") != len(prepared.canonical):
        raise ValueError("review decisions expected_item_count mismatch")
    if document.get("required_reviewer_declaration") != REVIEWER_DECLARATION:
        raise ValueError("review decisions changed the required reviewer declaration")
    items = document.get("items")
    if not isinstance(items, list):
        raise ValueError("review decisions items must be a list")

    canonical_by_id = {record.identity: record for record in prepared.canonical}
    decisions: dict[tuple[str, str], dict[str, Any]] = {}
    for index, review in enumerate(items, start=1):
        label = f"review item {index}"
        if not isinstance(review, dict):
            raise ValueError(f"{label} must be an object")
        require_exact_fields(review, REVIEW_ITEM_FIELDS, label)
        identity = (str(review.get("benchmark_id") or ""), str(review.get("item_id") or ""))
        if identity in decisions:
            raise ValueError(f"duplicate review decision for {identity}")
        canonical = canonical_by_id.get(identity)
        candidate = prepared.candidates.get(identity)
        if canonical is None or candidate is None:
            raise ValueError(f"review decision has no matching input item: {identity}")
        expected_read_only = review_item_template(canonical, candidate)
        for field in (
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
        ):
            if review.get(field) != expected_read_only[field]:
                raise ValueError(f"review decision {identity}: read-only field {field} changed")
        decision = review.get("decision")
        if decision not in {"pass", "replace", "reject"}:
            raise ValueError(
                f"review decision {identity}: decision must be pass, replace, or reject"
            )
        reviewed_by = review.get("reviewed_by")
        if not isinstance(reviewed_by, str) or not reviewed_by.strip():
            raise ValueError(f"review decision {identity}: reviewed_by is required")
        parse_review_time(review.get("reviewed_at"), f"review decision {identity}")
        if review.get("reviewer_declaration") != REVIEWER_DECLARATION:
            raise ValueError(
                f"review decision {identity}: exact reviewer declaration is required"
            )
        reason = review.get("reason")
        if not isinstance(reason, str):
            raise ValueError(f"review decision {identity}: reason must be text")
        if reason.strip():
            # Reasons are audit-only and never enter the solver prompt, but the
            # provenance sidecar must not become a second channel for answer
            # text, inline images, or corrupted control characters.
            assert_question_only_text(
                reason,
                canonical,
                f"review decision {identity} reason",
            )
        replacement = review.get("replacement_description")
        replacement_hash = review.get("replacement_description_sha256")
        if decision == "replace":
            if not reason.strip():
                raise ValueError(f"review decision {identity}: replace requires a reason")
            if not isinstance(replacement, str) or not replacement.strip():
                raise ValueError(
                    f"review decision {identity}: replace requires replacement_description"
                )
            if replacement == candidate.row["candidate_description"]:
                raise ValueError(
                    f"review decision {identity}: replacement must change the candidate"
                )
            require_sha256(
                replacement_hash,
                f"review decision {identity} replacement_description_sha256",
            )
            if sha256_text(replacement) != replacement_hash:
                raise ValueError(
                    f"review decision {identity}: replacement description hash mismatch"
                )
            assert_question_only_text(
                replacement,
                canonical,
                f"review decision {identity} replacement",
            )
        else:
            if replacement not in (None, "") or replacement_hash not in (None, ""):
                raise ValueError(
                    f"review decision {identity}: only replace may carry replacement text"
                )
            if decision == "reject" and not reason.strip():
                raise ValueError(f"review decision {identity}: reject requires a reason")
        decisions[identity] = copy.deepcopy(review)

    missing = sorted(set(canonical_by_id) - set(decisions))
    extras = sorted(set(decisions) - set(canonical_by_id))
    if missing:
        raise ValueError(f"missing review decisions (fail closed): {missing}")
    if extras:
        raise ValueError(f"unexpected review decisions: {extras}")
    return decisions


def solver_staticized_text(native: str, reviewed_description: str) -> str:
    if reviewed_description == NO_ADDITIONAL_IMAGE_INFORMATION:
        return native
    return (
        native
        + VISUAL_ADDENDUM_PREFIX
        + reviewed_description
        + VISUAL_ADDENDUM_SUFFIX
    )


def artifact_pointer(value: Any) -> Any:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("candidate artifact pointer must be an object or null")
    allowed = {
        "manifest_path",
        "body_sha256",
        "body_bytes",
        "externalized_data_uri_count",
        "inline_data_uri_marker_count",
    }
    require_exact_fields(value, allowed, "candidate artifact pointer")
    return {key: copy.deepcopy(value[key]) for key in sorted(value)}


def candidate_provenance(row: Mapping[str, Any]) -> dict[str, Any]:
    attempts: list[dict[str, Any]] = []
    for attempt in row["wire_attempts"]:
        if not isinstance(attempt, dict):
            raise ValueError("candidate wire-attempt summary must contain objects")
        allowed = {
            "attempt",
            "state",
            "attempt_record_path",
            "request_sha256",
            "http_status",
            "response_body_sha256",
            "validation_ok",
            "validation_reason",
        }
        require_exact_fields(attempt, allowed, "candidate wire-attempt summary")
        attempts.append({key: copy.deepcopy(attempt.get(key)) for key in sorted(allowed)})
    reasoning_artifacts = row.get("reasoning_artifacts") or []
    if not isinstance(reasoning_artifacts, list):
        raise ValueError("candidate reasoning_artifacts must be a list")
    safe_reasoning: list[dict[str, Any]] = []
    for entry in reasoning_artifacts:
        if not isinstance(entry, dict) or "json_pointer" not in entry:
            raise ValueError("candidate reasoning artifact is malformed")
        pointer = artifact_pointer({key: value for key, value in entry.items() if key != "json_pointer"})
        safe_reasoning.append({"json_pointer": entry["json_pointer"], **(pointer or {})})
    return {
        "candidate_schema_version": row["schema_version"],
        "candidate_created_at": row.get("created_at"),
        "candidate_status": row["status"],
        "candidate_description": row["candidate_description"],
        "candidate_description_sha256": row["candidate_description_sha256"],
        "question_source_fingerprint": row["question_source_fingerprint"],
        "input_fingerprint": row["input_fingerprint"],
        "request": {
            "sha256": row["request_sha256"],
            "route": row["route"],
            "wire_attempt_limit": row["wire_attempt_limit"],
            "wire_timeout_seconds": row["wire_timeout_seconds"],
            "wire_attempt_count": row["wire_attempt_count"],
            "wire_attempts": attempts,
        },
        "images": {
            "current_question_images": copy.deepcopy(row["current_question_images"]),
            "cumulative_question_images": copy.deepcopy(
                row["cumulative_question_images"]
            ),
        },
        "leakage_audit": copy.deepcopy(row["leakage_audit"]),
        "audit_record_path": row.get("audit_record_path"),
        "raw_response_artifact": artifact_pointer(row.get("raw_response_artifact")),
        "response_message_artifact": artifact_pointer(
            row.get("response_message_artifact")
        ),
        "reasoning_artifacts": safe_reasoning,
        "usage_artifact": artifact_pointer(row.get("usage_artifact")),
        "reasoning_tokens": row.get("reasoning_tokens"),
    }


def derive_item(
    canonical: CanonicalRecord,
    candidate: CandidateRecord,
    review: Mapping[str, Any],
) -> tuple[dict[str, Any], str]:
    old_staticized = canonical.row["question_text_staticized"]
    reviewed = (
        review["replacement_description"]
        if review["decision"] == "replace"
        else candidate.row["candidate_description"]
    )
    solver_text = solver_staticized_text(
        canonical.row["question_text_native"], reviewed
    )
    derived = copy.deepcopy(canonical.row)
    derived["question_text_staticized_human_reference"] = old_staticized
    derived["judge_question_text_staticized"] = old_staticized
    derived["question_text_staticized"] = solver_text
    derived["question_only_staticization_freeze"] = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "route": staticizer.FIXED_ROUTE,
        "review_decision": review["decision"],
        "candidate_description_sha256": candidate.row[
            "candidate_description_sha256"
        ],
        "reviewed_description_sha256": sha256_text(reviewed),
        "question_text_native_sha256": sha256_text(
            canonical.row["question_text_native"]
        ),
        "question_text_staticized_sha256": sha256_text(solver_text),
        "judge_question_text_staticized_sha256": sha256_text(old_staticized),
        "provenance_file": "../../provenance.json",
    }
    if derived["judge_question_text_staticized"] != old_staticized:
        raise RuntimeError(f"judge staticized text changed for {canonical.identity}")
    if derived["question_text_staticized_human_reference"] != old_staticized:
        raise RuntimeError(f"human reference changed for {canonical.identity}")
    if reviewed == NO_ADDITIONAL_IMAGE_INFORMATION:
        if solver_text != canonical.row["question_text_native"]:
            raise RuntimeError(f"no-information candidate changed solver text for {canonical.identity}")
    elif solver_text != (
        canonical.row["question_text_native"]
        + VISUAL_ADDENDUM_PREFIX
        + reviewed
        + VISUAL_ADDENDUM_SUFFIX
    ):
        raise RuntimeError(f"solver text composition changed for {canonical.identity}")
    return derived, reviewed


def assert_no_sensitive_copy(
    provenance: Mapping[str, Any], canonical: Sequence[CanonicalRecord]
) -> None:
    serialized = normalize_for_leak_check(
        json.dumps(provenance, ensure_ascii=False, sort_keys=True)
    )
    for record in canonical:
        for sensitive in sensitive_canonical_strings(record.row):
            if (
                len(sensitive) >= MIN_SENSITIVE_SUBSTRING_LENGTH
                and sensitive in serialized
            ):
                raise RuntimeError(
                    "provenance sidecar would copy canonical answer/solution/rubric content"
                )


def guard_freeze_output(
    output_root: Path,
    *,
    prepared: PreparedInputs,
    review_path: Path,
) -> Path:
    root = output_root.resolve()
    input_paths = set(prepared.canonical_paths) | {
        prepared.candidate_path,
        review_path.resolve(),
    }
    if root.exists():
        raise ValueError(f"freeze output tree already exists; refusing overwrite: {root}")
    planned = {root / "provenance.json"}
    for benchmark_id in {record.identity[0] for record in prepared.canonical}:
        if SAFE_GROUP.fullmatch(benchmark_id) is None:
            raise ValueError(f"unsafe benchmark_id for output grouping: {benchmark_id!r}")
        planned.add(root / "datasets" / benchmark_id / "items.jsonl")
    collisions = sorted(str(path) for path in planned & input_paths)
    if collisions:
        raise ValueError(f"freeze output would overwrite input files: {collisions}")
    return root


def build_provenance(
    *,
    prepared: PreparedInputs,
    reviews: Mapping[tuple[str, str], Mapping[str, Any]],
    review_path: Path,
    derived_by_group: Mapping[str, Sequence[dict[str, Any]]],
    derived_bytes_by_group: Mapping[str, bytes],
    reviewed_descriptions: Mapping[tuple[str, str], str],
) -> dict[str, Any]:
    source_files = [
        {
            "path": str(path),
            "sha256": sha256_file(path),
            "row_count": sum(1 for record in prepared.canonical if record.source_path == path),
        }
        for path in prepared.canonical_paths
    ]
    derived_files = [
        {
            "benchmark_id": benchmark_id,
            "path": f"datasets/{benchmark_id}/items.jsonl",
            "row_count": len(derived_by_group[benchmark_id]),
            "sha256": sha256_bytes(derived_bytes_by_group[benchmark_id]),
        }
        for benchmark_id in sorted(derived_by_group)
    ]
    item_entries: list[dict[str, Any]] = []
    for canonical in prepared.canonical:
        identity = canonical.identity
        candidate = prepared.candidates[identity]
        review = reviews[identity]
        reviewed = reviewed_descriptions[identity]
        solver_text = solver_staticized_text(
            canonical.row["question_text_native"], reviewed
        )
        item_entries.append(
            {
                "benchmark_id": identity[0],
                "item_id": identity[1],
                "sequence_id": canonical.row["sequence_id"],
                "turn_index": int(canonical.row["turn_index"]),
                "canonical_source": {
                    "path": str(canonical.source_path),
                    "line": canonical.line_number,
                    "question_text_native_sha256": sha256_text(
                        canonical.row["question_text_native"]
                    ),
                    "previous_question_text_staticized_sha256": sha256_text(
                        canonical.row["question_text_staticized"]
                    ),
                },
                "candidate_source": {
                    "path": str(candidate.source_path),
                    "line": candidate.line_number,
                },
                "staticizer": candidate_provenance(candidate.row),
                "human_review": {
                    "decision": review["decision"],
                    "reason": review["reason"],
                    "reviewed_by": review["reviewed_by"],
                    "reviewed_at": review["reviewed_at"],
                    "reviewer_declaration": review["reviewer_declaration"],
                    "replacement_description": (
                        review["replacement_description"]
                        if review["decision"] == "replace"
                        else None
                    ),
                    "replacement_description_sha256": (
                        review["replacement_description_sha256"]
                        if review["decision"] == "replace"
                        else None
                    ),
                },
                "freeze": {
                    "reviewed_description_sha256": sha256_text(reviewed),
                    "no_additional_image_information": (
                        reviewed == NO_ADDITIONAL_IMAGE_INFORMATION
                    ),
                    "solver_question_text_sha256": sha256_text(solver_text),
                    "judge_question_text_sha256": sha256_text(
                        canonical.row["question_text_staticized"]
                    ),
                    "judge_text_exactly_preserved": True,
                },
            }
        )
    wire_generation_archive = WIRE_GENERATION_STATICIZER_ARCHIVE.resolve()
    wire_generation_readme = WIRE_GENERATION_STATICIZER_ARCHIVE_README.resolve()
    if not wire_generation_archive.is_file():
        raise RuntimeError(
            f"wire-generation staticizer archive is missing: {wire_generation_archive}"
        )
    wire_generation_sha256 = sha256_file(wire_generation_archive)
    if wire_generation_sha256 != WIRE_GENERATION_STATICIZER_SHA256:
        raise RuntimeError(
            "wire-generation staticizer archive hash mismatch: "
            f"{wire_generation_sha256}"
        )
    if not wire_generation_readme.is_file():
        raise RuntimeError(
            f"wire-generation archive README is missing: {wire_generation_readme}"
        )
    validation_staticizer_path = Path(staticizer.__file__).resolve()
    validation_staticizer_sha256 = sha256_file(validation_staticizer_path)
    freeze_implementation_path = Path(__file__).resolve()
    freeze_implementation_sha256 = sha256_file(freeze_implementation_path)

    fingerprint_material = {
        "schema_version": FREEZE_SCHEMA_VERSION,
        "canonical_input_fingerprint": prepared.canonical_input_fingerprint,
        "candidate_jsonl_sha256": prepared.candidate_jsonl_sha256,
        "review_decisions_sha256": sha256_file(review_path),
        "wire_generation_staticizer_sha256": wire_generation_sha256,
        "freeze_time_validation_staticizer_sha256": validation_staticizer_sha256,
        "freeze_implementation_sha256": freeze_implementation_sha256,
        "derived_files": derived_files,
        "review_item_hashes": [
            {
                "benchmark_id": entry["benchmark_id"],
                "item_id": entry["item_id"],
                "candidate_description_sha256": entry["staticizer"][
                    "candidate_description_sha256"
                ],
                "reviewed_description_sha256": entry["freeze"][
                    "reviewed_description_sha256"
                ],
                "solver_question_text_sha256": entry["freeze"][
                    "solver_question_text_sha256"
                ],
            }
            for entry in item_entries
        ],
    }
    provenance = {
        "schema_version": PROVENANCE_SCHEMA_VERSION,
        "created_at": utc_now(),
        "freeze_fingerprint": sha256_bytes(canonical_json_bytes(fingerprint_material)),
        "offline_only": True,
        "api_calls_performed": 0,
        "canonical_writeback_performed": False,
        "item_count": len(prepared.canonical),
        "group_counts": dict(
            sorted(Counter(record.identity[0] for record in prepared.canonical).items())
        ),
        "canonical_input_fingerprint": prepared.canonical_input_fingerprint,
        "source_canonical_files": source_files,
        "candidate_input": {
            "path": str(prepared.candidate_path),
            "sha256": prepared.candidate_jsonl_sha256,
            "row_count": len(prepared.candidates),
        },
        "review_input": {
            "path": str(review_path.resolve()),
            "sha256": sha256_file(review_path),
            "schema_version": REVIEW_SCHEMA_VERSION,
            "all_items_decided": True,
            "reject_count": 0,
        },
        "staticizer_contract": {
            "schema_version": staticizer.SCHEMA_VERSION,
            "candidate_schema_version": staticizer.CANDIDATE_SCHEMA_VERSION,
            "implementation_path": str(validation_staticizer_path),
            "implementation_sha256": validation_staticizer_sha256,
            "implementation_role": "freeze_time_replay_and_output_validation",
            "wire_generation_implementation_path": str(wire_generation_archive),
            "wire_generation_implementation_sha256": wire_generation_sha256,
            "wire_generation_archive_readme_path": str(wire_generation_readme),
            "wire_generation_archive_readme_sha256": sha256_file(
                wire_generation_readme
            ),
            "post_generation_changes": (
                "candidate-output validation guards only; request construction, "
                "system prompt, route, model options, image bytes, timeouts, and "
                "seven-attempt wire policy are unchanged"
            ),
            "api_url": staticizer.API_URL,
            "route": staticizer.FIXED_ROUTE,
            "system_prompt": staticizer.STATICIZER_SYSTEM,
            "system_prompt_sha256": sha256_text(staticizer.STATICIZER_SYSTEM),
            "max_tokens": staticizer.MAX_TOKENS,
            "thinking_level": staticizer.THINKING_LEVEL,
            "wire_attempt_limit": staticizer.WIRE_ATTEMPTS,
            "wire_timeout_seconds": staticizer.WIRE_TIMEOUT_SECONDS,
            "source_content_fields": list(staticizer.SOURCE_CONTENT_FIELDS),
            "sequence_policy": (
                "prior exact native user text and raw candidate assistant output; "
                "current turn replays unique cumulative official question images"
            ),
        },
        "merge_contract": {
            "implementation_path": str(freeze_implementation_path),
            "implementation_sha256": freeze_implementation_sha256,
            "solver_text_rule": (
                "exact question_text_native plus the fixed reviewed Gemini visual "
                "addendum; the exact no-information sentinel appends nothing"
            ),
            "visual_addendum_prefix": VISUAL_ADDENDUM_PREFIX,
            "visual_addendum_suffix": VISUAL_ADDENDUM_SUFFIX,
            "no_information_sentinel": NO_ADDITIONAL_IMAGE_INFORMATION,
            "judge_text_rule": (
                "judge_question_text_staticized is byte-for-byte the prior canonical "
                "question_text_staticized"
            ),
            "human_reference_rule": (
                "question_text_staticized_human_reference is byte-for-byte the prior "
                "canonical question_text_staticized"
            ),
            "unreviewed_policy": "fail_closed",
            "reject_policy": "block_entire_freeze",
        },
        "derived_files": derived_files,
        "items": item_entries,
        "canonical_sensitive_content_copied_to_provenance": False,
    }
    assert_no_sensitive_copy(provenance, prepared.canonical)
    return provenance


def freeze_staticizations(
    *,
    prepared: PreparedInputs,
    review_path: Path,
    output_root: Path,
) -> Path:
    review_path = review_path.resolve()
    if not review_path.is_file():
        raise ValueError(f"review decisions JSON does not exist: {review_path}")
    reviews = validate_review_decisions(review_path, prepared)
    rejected = sorted(identity for identity, review in reviews.items() if review["decision"] == "reject")
    if rejected:
        raise ValueError(f"freeze blocked by rejected review decisions: {rejected}")
    root = guard_freeze_output(
        output_root,
        prepared=prepared,
        review_path=review_path,
    )

    derived_by_group: dict[str, list[dict[str, Any]]] = {}
    reviewed_descriptions: dict[tuple[str, str], str] = {}
    for canonical in prepared.canonical:
        candidate = prepared.candidates[canonical.identity]
        derived, reviewed = derive_item(
            canonical,
            candidate,
            reviews[canonical.identity],
        )
        derived_by_group.setdefault(canonical.identity[0], []).append(derived)
        reviewed_descriptions[canonical.identity] = reviewed
    derived_bytes_by_group = {
        benchmark_id: jsonl_bytes(rows)
        for benchmark_id, rows in derived_by_group.items()
    }
    provenance = build_provenance(
        prepared=prepared,
        reviews=reviews,
        review_path=review_path,
        derived_by_group=derived_by_group,
        derived_bytes_by_group=derived_bytes_by_group,
        reviewed_descriptions=reviewed_descriptions,
    )

    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging.", dir=str(root.parent))
    )
    try:
        for benchmark_id, payload in derived_bytes_by_group.items():
            atomic_write_bytes(
                staging / "datasets" / benchmark_id / "items.jsonl",
                payload,
            )
        atomic_write_bytes(staging / "provenance.json", pretty_json_bytes(provenance))
        if root.exists():
            raise ValueError(f"freeze output appeared during publication: {root}")
        os.rename(staging, root)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return root


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline, fail-closed freeze of reviewed question-only staticization candidates."
        )
    )
    parser.add_argument(
        "--items",
        type=Path,
        nargs="+",
        required=True,
        help=(
            "canonical items JSONL file(s), or a directory containing items.jsonl "
            "one level below"
        ),
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--review-decisions", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--write-review-template", type=Path)
    parser.add_argument("--expected-item-count", type=int)
    parser.add_argument(
        "--require-complete-153",
        action="store_true",
        help=(
            "require the frozen 2026 grouping: icho2026_theory=68, ipho2026=23, "
            "nbpho2026=28, usapho2026=34"
        ),
    )
    args = parser.parse_args(argv)
    if args.expected_item_count is not None and args.expected_item_count <= 0:
        parser.error("--expected-item-count must be positive")
    template_mode = args.write_review_template is not None
    freeze_mode = args.output_root is not None or args.review_decisions is not None
    if template_mode and freeze_mode:
        parser.error(
            "--write-review-template cannot be combined with --review-decisions/--output-root"
        )
    if not template_mode and not (
        args.review_decisions is not None and args.output_root is not None
    ):
        parser.error(
            "freeze mode requires both --review-decisions and --output-root; "
            "unreviewed candidates are never frozen"
        )
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    prepared = prepare_inputs(
        canonical_inputs=args.items,
        candidate_path=args.candidates,
        workspace=args.workspace,
        expected_item_count=args.expected_item_count,
        require_complete_153=args.require_complete_153,
    )
    if args.write_review_template is not None:
        write_review_template(args.write_review_template, prepared)
        print(
            f"review template written for {len(prepared.canonical)} items: "
            f"{args.write_review_template.resolve()}",
            flush=True,
        )
        return 0
    root = freeze_staticizations(
        prepared=prepared,
        review_path=args.review_decisions,
        output_root=args.output_root,
    )
    print(
        f"offline staticization freeze complete: {len(prepared.canonical)} items; {root}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
