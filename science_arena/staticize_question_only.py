#!/usr/bin/env python3
"""Auditable, resumable question-only image staticization for 2026 datasets.

This runner is deliberately separate from the canonical builders and evaluation
pipeline.  It writes review candidates only; it never edits an ``items.jsonl``.

The provider receives exactly two item-derived inputs: ``question_text_native``
and the exact bytes named by ``question_image_paths``.  Gold, answer, solution,
rubric, staticized-text, score, and adjudication fields are not accepted by the
payload builder.  Sequences follow ``pipeline.py`` semantics: prior user text and
assistant output stay in history, while the current user turn replays all unique
question images seen so far in the sequence.

Large request/response data URIs are stored losslessly as a small gzip template
plus content-addressed decoded blobs.  The recorded body SHA-256 is verified on
reconstruction, so provider-side base64 echo remains auditable without copying
the same image into every JSON sidecar.
"""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import gzip
import hashlib
import json
import os
import random
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence


API_URL = "https://api.360.cn/v1/chat/completions"
FIXED_ROUTE = "google/google/gemini-3.1-pro-preview"
WIRE_ATTEMPTS = 7
WIRE_TIMEOUT_SECONDS = 1800
MAX_TOKENS = 65536
THINKING_LEVEL = "high"
DEFAULT_WORKERS = 6
NO_DESCRIPTION_LABEL = "[NO STATICIZATION CANDIDATE AFTER 7 WIRE ATTEMPTS]"

SUPPORTED_BENCHMARKS = {
    "nbpho2026",
    "usapho2026",
    "ipho2026",
    "icho2026",
    "icho2026_theory",
}

STATICIZER_SYSTEM = (
    "You are a question-only visual staticizer for released 2026 Science "
    "Olympiad questions. Your sole task is to describe information visibly "
    "present in the supplied official QUESTION images that is necessary to "
    "make the latest question text self-contained for a text-only reader. "
    "Describe only necessary figure labels, quantities, units, geometry, "
    "arrows, legends, axes, scales, coordinates, tables, plots, apparatus, "
    "and chemical structures or stereochemistry. Transcribe visible symbols "
    "and relationships faithfully; distinguish solid, dashed, wedged, colored, "
    "or directed elements when that distinction is visible and relevant. Do "
    "not solve the question, calculate, derive, reason toward a result, infer "
    "or complete anything not visibly supplied, use outside knowledge, provide "
    "an answer, discuss a solution or rubric, or predict what should appear. "
    "Do not expose chain-of-thought. Do not repeat ordinary question text unless "
    "needed to connect it to a visual label. Return only the candidate visual "
    "description, with no preface or conclusion. If the images add no necessary "
    "information, return exactly [NO ADDITIONAL IMAGE INFORMATION]."
)

SCHEMA_VERSION = "science_arena_2026_question_only_staticizer_v1"
CANDIDATE_SCHEMA_VERSION = "science_arena_2026_staticization_candidate_v1"
ATTEMPT_SCHEMA_VERSION = "science_arena_2026_staticization_wire_attempt_v1"

SOURCE_CONTENT_FIELDS = ("question_text_native", "question_image_paths")
SOLUTION_IMAGE_FIELDS = (
    "official_solution_image_paths",
    "reference_image_paths",
    "answer_image_paths",
    "gold_image_paths",
)

WIRE_PAYLOAD_KEYS = {
    "model",
    "messages",
    "stream",
    "max_tokens",
    "thinking_level",
    "role",
    "content",
    "type",
    "text",
    "image_url",
    "url",
}
FORBIDDEN_PAYLOAD_KEY = re.compile(
    r"(?:gold|solution|rubric|official|reference|answer|staticized|"
    r"adjudicat|score|criterion)",
    flags=re.I,
)
FORBIDDEN_IMAGE_COMPONENT = re.compile(
    r"(?:^|[_-])(?:solutions?|answers?|gold|rubrics?)(?:[_-]|$)",
    flags=re.I,
)
DISALLOWED_CANDIDATE_OUTPUT = (
    re.compile(r"data:image/", flags=re.I),
    re.compile(r"<\s*/?\s*think\b", flags=re.I),
    re.compile(r"\bfinal\s+answer\b", flags=re.I),
    re.compile(
        r"\b(?:the\s+)?(?:correct\s+)?answer\s*(?::|=|(?:is|equals?)\b)",
        flags=re.I,
    ),
    re.compile(
        r"\b(?:the\s+)?solution\s*(?::|=|(?:is|equals?)\s+"
        r"(?=(?:[-+]?(?:\d|\.\d)|[A-Za-z]\s*=|[A-H](?:[.)]|\b)|"
        r"\\?boxed\b|true\b|false\b)))",
        flags=re.I,
    ),
    re.compile(r"\b(?:i|we)\s+(?:calculated?|derived?|deduced?|solved?)\b", flags=re.I),
)

# Only canonical base64 strings are externalized.  Non-canonical or escaped
# strings remain verbatim in the gzip template and are still losslessly kept.
DATA_URI_PATTERN = re.compile(
    rb"data:image/(?:png|jpeg|gif|webp);base64,[A-Za-z0-9+/]+={0,2}"
)


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


def atomic_write_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.parent / (
        f".{path.name}.tmp.{os.getpid()}.{threading.get_ident()}"
    )
    with temporary.open("wb") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_write_json(path: Path, value: Any) -> None:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode("utf-8")
    atomic_write_bytes(path, encoded)


def atomic_write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    encoded = b"".join(canonical_json_bytes(row) + b"\n" for row in rows)
    atomic_write_bytes(path, encoded)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            rows.append(value)
    return rows


def safe_component(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "__", str(value)).strip("._") or "unnamed"
    return f"{slug[:80]}--{sha256_text(str(value))[:12]}"


def relative_path(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def ensure_content_blob(root: Path, namespace: str, raw: bytes) -> Path:
    digest = sha256_bytes(raw)
    path = root / "blobs" / namespace / f"{digest}.bin"
    if path.exists():
        if sha256_file(path) != digest:
            raise RuntimeError(f"content-addressed blob hash mismatch: {path}")
    else:
        atomic_write_bytes(path, raw)
    return path


def store_lossless_body(
    *,
    body: bytes,
    base_path: Path,
    audit_root: Path,
    kind: str,
) -> dict[str, Any]:
    """Store bytes exactly while deduplicating canonical image data URIs."""

    chunks: list[bytes] = []
    replacements: list[dict[str, Any]] = []
    cursor = 0
    externalized = 0
    for match in DATA_URI_PATTERN.finditer(body):
        segment = match.group(0)
        prefix, encoded = segment.split(b",", 1)
        try:
            decoded = base64.b64decode(encoded, validate=True)
        except Exception:
            continue
        if base64.b64encode(decoded) != encoded:
            continue
        token = (
            f"@@SCIENCE_ARENA_DATA_URI_{externalized:06d}_"
            f"{sha256_bytes(segment)[:16]}@@"
        ).encode("ascii")
        if token in body:
            raise RuntimeError("lossless-body placeholder collision")
        chunks.extend((body[cursor : match.start()], token))
        cursor = match.end()
        blob_path = ensure_content_blob(audit_root, "data-uri", decoded)
        replacements.append(
            {
                "token": token.decode("ascii"),
                "data_uri_prefix": prefix.decode("ascii"),
                "data_uri_sha256": sha256_bytes(segment),
                "data_uri_bytes": len(segment),
                "base64_characters": len(encoded),
                "decoded_sha256": sha256_bytes(decoded),
                "decoded_bytes": len(decoded),
                "decoded_blob_path": relative_path(blob_path, audit_root),
            }
        )
        externalized += 1
    chunks.append(body[cursor:])
    template = b"".join(chunks)

    template_path = base_path.with_name(base_path.name + ".template.gz")
    manifest_path = base_path.with_name(base_path.name + ".manifest.json")
    compressed = gzip.compress(template, compresslevel=9, mtime=0)
    atomic_write_bytes(template_path, compressed)
    manifest = {
        "schema_version": "science_arena_lossless_body_v1",
        "kind": kind,
        "encoding": "gzip_template_plus_content_addressed_data_uri_blobs",
        "body_sha256": sha256_bytes(body),
        "body_bytes": len(body),
        "template_sha256": sha256_bytes(template),
        "template_bytes": len(template),
        "template_gzip_sha256": sha256_bytes(compressed),
        "template_gzip_bytes": len(compressed),
        "template_path": relative_path(template_path, audit_root),
        "externalized_data_uri_count": len(replacements),
        "inline_data_uri_marker_count": max(
            body.count(b"data:image/") - len(replacements), 0
        ),
        "replacements": replacements,
    }
    atomic_write_json(manifest_path, manifest)
    restored = restore_lossless_body(manifest_path, audit_root)
    if restored != body:
        raise RuntimeError(f"lossless body self-check failed: {manifest_path}")
    return {
        "manifest_path": relative_path(manifest_path, audit_root),
        "body_sha256": manifest["body_sha256"],
        "body_bytes": manifest["body_bytes"],
        "externalized_data_uri_count": manifest["externalized_data_uri_count"],
        "inline_data_uri_marker_count": manifest["inline_data_uri_marker_count"],
    }


def restore_lossless_body(manifest_path: Path, audit_root: Path) -> bytes:
    manifest = read_json(manifest_path)
    template_path = audit_root / str(manifest["template_path"])
    compressed = template_path.read_bytes()
    if sha256_bytes(compressed) != manifest["template_gzip_sha256"]:
        raise RuntimeError(f"gzip template hash mismatch: {template_path}")
    template = gzip.decompress(compressed)
    if sha256_bytes(template) != manifest["template_sha256"]:
        raise RuntimeError(f"template hash mismatch: {template_path}")
    restored = template
    for replacement in manifest["replacements"]:
        token = str(replacement["token"]).encode("ascii")
        if restored.count(token) != 1:
            raise RuntimeError(f"placeholder count mismatch in {template_path}")
        blob_path = audit_root / str(replacement["decoded_blob_path"])
        decoded = blob_path.read_bytes()
        if sha256_bytes(decoded) != replacement["decoded_sha256"]:
            raise RuntimeError(f"decoded data-URI blob hash mismatch: {blob_path}")
        segment = (
            str(replacement["data_uri_prefix"]).encode("ascii")
            + b","
            + base64.b64encode(decoded)
        )
        if sha256_bytes(segment) != replacement["data_uri_sha256"]:
            raise RuntimeError("reconstructed data URI hash mismatch")
        restored = restored.replace(token, segment, 1)
    if len(restored) != int(manifest["body_bytes"]):
        raise RuntimeError("reconstructed body byte count mismatch")
    if sha256_bytes(restored) != manifest["body_sha256"]:
        raise RuntimeError("reconstructed body hash mismatch")
    return restored


def store_json_value(
    value: Any,
    *,
    base_path: Path,
    audit_root: Path,
    kind: str,
) -> dict[str, Any]:
    return store_lossless_body(
        body=canonical_json_bytes(value),
        base_path=base_path,
        audit_root=audit_root,
        kind=kind,
    )


def load_api_key() -> str:
    for name in ("QIHOO_API_KEY", "API_360_KEY"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    raise RuntimeError("No 360 API key found. Set QIHOO_API_KEY or API_360_KEY.")


def fixed_route(value: str) -> str:
    if value != FIXED_ROUTE:
        raise argparse.ArgumentTypeError(
            f"production route is fixed; expected {FIXED_ROUTE!r}"
        )
    return value


def assert_within(path: Path, root: Path, label: str) -> None:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ValueError(f"{label} must stay inside workspace: {path}") from exc


def guard_output_paths(
    *, workspace: Path, items_path: Path, output_path: Path, audit_root: Path
) -> None:
    if output_path.resolve() == items_path.resolve():
        raise ValueError("candidate output must not overwrite its input items JSONL")
    forbidden_roots = (
        workspace / "experiments" / "2026_reval" / "datasets",
        workspace / "experiments" / "2026_reval" / "datasets_draft",
        workspace / "experiments" / "2026_reval" / "runs",
        workspace / "experiments" / "2026_reval" / "validation",
        workspace / "experiments" / "2026_reval" / "capabilities",
        workspace / "experiments" / "2026_reval" / "adjudications",
        workspace / "experiments" / "2026_reval" / "snapshots",
        workspace / "experiments" / "2026_reval" / "audits",
    )
    for candidate, label in ((output_path, "output"), (audit_root, "audit root")):
        for forbidden in forbidden_roots:
            try:
                candidate.resolve().relative_to(forbidden.resolve())
            except ValueError:
                continue
            raise ValueError(f"{label} must not be inside canonical or formal-result trees")


def resolve_question_path(value: str, workspace: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve()
    assert_within(path, workspace, "question image")
    if any(FORBIDDEN_IMAGE_COMPONENT.search(part) for part in path.parts):
        raise ValueError(f"question_image_paths contains a solution/gold-like path: {value}")
    if not path.is_file():
        raise ValueError(f"missing question image: {path}")
    return path


def resolve_for_overlap(value: str, workspace: Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = workspace / path
    return path.resolve()


@dataclass(frozen=True)
class QuestionTurn:
    item_id: str
    benchmark_id: str
    sequence_id: str
    turn_index: int
    question_text_native: str
    question_image_values: tuple[str, ...]
    question_image_paths: tuple[Path, ...]
    official_solution_image_overlap_count: int


@dataclass(frozen=True)
class ImageInput:
    source_path: str
    resolved_path: Path
    mime_type: str
    raw: bytes
    data_url: str
    sha256: str
    data_url_sha256: str
    blob_path: str

    def descriptor(self) -> dict[str, Any]:
        return {
            "source_path": self.source_path,
            "mime_type": self.mime_type,
            "bytes": len(self.raw),
            "sha256": self.sha256,
            "data_url_sha256": self.data_url_sha256,
            "data_url_bytes": len(self.data_url.encode("ascii")),
            "audit_blob_path": self.blob_path,
            "transformation": "none_exact_original_file_bytes",
        }


@dataclass(frozen=True)
class HistoryEntry:
    question_text_native: str
    candidate_description: str


@dataclass(frozen=True)
class WireResult:
    ok: bool
    status: int | None
    body: bytes
    response: dict[str, Any] | None
    error: str
    retry_after: float | None
    latency_seconds: float


def project_question_turns(
    rows: Sequence[dict[str, Any]],
    *,
    workspace: Path,
    benchmark_filter: set[str] | None = None,
) -> list[QuestionTurn]:
    turns: list[QuestionTurn] = []
    seen_ids: set[tuple[str, str]] = set()
    safe_paths: set[Path] = set()
    for row_number, row in enumerate(rows, start=1):
        benchmark_id = str(row.get("benchmark_id") or "")
        if benchmark_filter and benchmark_id not in benchmark_filter:
            continue
        if benchmark_id not in SUPPORTED_BENCHMARKS:
            raise ValueError(f"row {row_number}: unsupported 2026 benchmark {benchmark_id!r}")
        item_id = str(row.get("item_id") or "")
        sequence_id = str(row.get("sequence_id") or "")
        if not item_id or not sequence_id:
            raise ValueError(f"row {row_number}: item_id and sequence_id are required")
        identity = (benchmark_id, item_id)
        if identity in seen_ids:
            raise ValueError(f"duplicate item identity: {identity}")
        seen_ids.add(identity)
        try:
            turn_index = int(row["turn_index"])
        except Exception as exc:
            raise ValueError(f"{identity}: turn_index must be an integer") from exc
        question_text = row.get("question_text_native")
        if not isinstance(question_text, str) or not question_text.strip():
            raise ValueError(f"{identity}: question_text_native must be non-empty text")
        image_values = row.get("question_image_paths")
        if not isinstance(image_values, list) or not image_values:
            raise ValueError(f"{identity}: question_image_paths must be a non-empty list")
        if any(not isinstance(value, str) or not value for value in image_values):
            raise ValueError(f"{identity}: question_image_paths must contain path strings")
        question_paths = tuple(resolve_question_path(value, workspace) for value in image_values)
        solution_paths: set[Path] = set()
        for field in SOLUTION_IMAGE_FIELDS:
            values = row.get(field)
            if values is None:
                continue
            if not isinstance(values, list):
                raise ValueError(f"{identity}: {field} must be a list when present")
            solution_paths.update(resolve_for_overlap(str(value), workspace) for value in values)
        overlap = len(set(question_paths) & solution_paths)
        if overlap:
            raise ValueError(f"{identity}: question and official solution images overlap")
        turn = QuestionTurn(
            item_id=item_id,
            benchmark_id=benchmark_id,
            sequence_id=sequence_id,
            turn_index=turn_index,
            question_text_native=question_text,
            question_image_values=tuple(image_values),
            question_image_paths=question_paths,
            official_solution_image_overlap_count=overlap,
        )
        item_record_path = Path(
            safe_component(benchmark_id),
            safe_component(sequence_id),
            f"{turn_index:04d}--{safe_component(item_id)}.json",
        )
        if item_record_path in safe_paths:
            raise ValueError(f"audit path collision for item {identity}")
        safe_paths.add(item_record_path)
        turns.append(turn)
    if not turns:
        raise ValueError("no selected items")

    grouped: dict[tuple[str, str], list[QuestionTurn]] = {}
    for turn in turns:
        grouped.setdefault((turn.benchmark_id, turn.sequence_id), []).append(turn)
    for key, sequence in grouped.items():
        ordered = sorted(turn.turn_index for turn in sequence)
        if ordered != list(range(1, len(sequence) + 1)):
            raise ValueError(f"{key}: turn_index values must be contiguous from 1; got {ordered}")
    return sorted(
        turns,
        key=lambda turn: (
            turn.benchmark_id,
            turn.sequence_id,
            turn.turn_index,
            turn.item_id,
        ),
    )


def detect_image_mime(raw: bytes, path: Path) -> str:
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
        suffixes = {".png"}
    elif raw.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
        suffixes = {".jpg", ".jpeg"}
    elif raw.startswith((b"GIF87a", b"GIF89a")):
        mime = "image/gif"
        suffixes = {".gif"}
    elif len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        mime = "image/webp"
        suffixes = {".webp"}
    else:
        raise ValueError(f"unsupported or invalid raster image: {path}")
    if path.suffix.lower() not in suffixes:
        raise ValueError(f"image extension/content MIME mismatch: {path}")
    return mime


def load_image_input(
    source_path: str, resolved_path: Path, *, audit_root: Path
) -> ImageInput:
    raw = resolved_path.read_bytes()
    mime = detect_image_mime(raw, resolved_path)
    data_url = f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"
    # Use the same content-addressed namespace as request/response URI
    # externalization so an original question image and any provider echo share
    # one physical blob.
    blob = ensure_content_blob(audit_root, "data-uri", raw)
    return ImageInput(
        source_path=source_path,
        resolved_path=resolved_path,
        mime_type=mime,
        raw=raw,
        data_url=data_url,
        sha256=sha256_bytes(raw),
        data_url_sha256=sha256_text(data_url),
        blob_path=relative_path(blob, audit_root),
    )


def build_payload(
    *,
    history: Sequence[HistoryEntry],
    current_turn: QuestionTurn,
    cumulative_images: Sequence[ImageInput],
) -> dict[str, Any]:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": STATICIZER_SYSTEM}
    ]
    for entry in history:
        messages.append({"role": "user", "content": entry.question_text_native})
        messages.append({"role": "assistant", "content": entry.candidate_description})
    current_content: list[dict[str, Any]] = [
        {"type": "text", "text": current_turn.question_text_native}
    ]
    current_content.extend(
        {
            "type": "image_url",
            "image_url": {"url": image.data_url},
        }
        for image in cumulative_images
    )
    messages.append({"role": "user", "content": current_content})
    return {
        "model": FIXED_ROUTE,
        "messages": messages,
        "stream": False,
        "max_tokens": MAX_TOKENS,
        "thinking_level": THINKING_LEVEL,
    }


def all_mapping_keys(value: Any) -> list[str]:
    result: list[str] = []
    if isinstance(value, dict):
        for key, item in value.items():
            result.append(str(key))
            result.extend(all_mapping_keys(item))
    elif isinstance(value, list):
        for item in value:
            result.extend(all_mapping_keys(item))
    return result


def decode_data_url(value: str) -> tuple[str, bytes]:
    match = re.fullmatch(
        r"data:(image/(?:png|jpeg|gif|webp));base64,([A-Za-z0-9+/]+={0,2})",
        value,
    )
    if not match:
        raise ValueError("wire image is not a canonical raster data URI")
    return match.group(1), base64.b64decode(match.group(2), validate=True)


def audit_payload(
    *,
    payload: dict[str, Any],
    history: Sequence[HistoryEntry],
    current_turn: QuestionTurn,
    cumulative_images: Sequence[ImageInput],
) -> dict[str, Any]:
    errors: list[str] = []
    keys = all_mapping_keys(payload)
    unexpected_keys = sorted(set(keys) - WIRE_PAYLOAD_KEYS)
    forbidden_keys = sorted({key for key in keys if FORBIDDEN_PAYLOAD_KEY.search(key)})
    if unexpected_keys:
        errors.append(f"unexpected payload keys: {unexpected_keys}")
    if forbidden_keys:
        errors.append(f"forbidden payload keys: {forbidden_keys}")
    if payload.get("model") != FIXED_ROUTE:
        errors.append("model route is not the fixed production route")
    if set(payload) != {"model", "messages", "stream", "max_tokens", "thinking_level"}:
        errors.append("top-level request shape changed")

    messages = payload.get("messages")
    expected_count = 2 + 2 * len(history)
    if not isinstance(messages, list) or len(messages) != expected_count:
        errors.append("message count does not match cumulative sequence history")
        messages = []
    if messages:
        if messages[0] != {"role": "system", "content": STATICIZER_SYSTEM}:
            errors.append("system prompt changed")
        for index, entry in enumerate(history):
            user = messages[1 + 2 * index]
            assistant = messages[2 + 2 * index]
            if user != {"role": "user", "content": entry.question_text_native}:
                errors.append(f"history user turn {index + 1} is not exact native question text")
            if assistant != {"role": "assistant", "content": entry.candidate_description}:
                errors.append(f"history assistant turn {index + 1} changed")
        current = messages[-1]
        content = current.get("content") if isinstance(current, dict) else None
        if not isinstance(content, list) or not content:
            errors.append("current user content is not a text-plus-images list")
            content = []
        if content and content[0] != {
            "type": "text",
            "text": current_turn.question_text_native,
        }:
            errors.append("current user text is not exact question_text_native")
        image_blocks = content[1:] if content else []
        if len(image_blocks) != len(cumulative_images):
            errors.append("question image count changed")
        for index, (block, expected) in enumerate(
            zip(image_blocks, cumulative_images), start=1
        ):
            try:
                if set(block) != {"type", "image_url"} or block["type"] != "image_url":
                    raise ValueError("invalid image block shape")
                image_url = block["image_url"]
                if not isinstance(image_url, dict) or set(image_url) != {"url"}:
                    raise ValueError("invalid image_url shape")
                mime, raw = decode_data_url(str(image_url["url"]))
                if mime != expected.mime_type or sha256_bytes(raw) != expected.sha256:
                    raise ValueError("image bytes differ from question image")
            except Exception as exc:
                errors.append(f"question image block {index}: {exc}")

    audit = {
        "passed": not errors,
        "errors": errors,
        "source_content_fields_used": list(SOURCE_CONTENT_FIELDS),
        "forbidden_source_fields_used": [],
        "payload_key_allowlist_verified": not unexpected_keys,
        "forbidden_payload_keys": forbidden_keys,
        "gold_field_count": len(forbidden_keys),
        "solution_image_count": current_turn.official_solution_image_overlap_count,
        "official_solution_image_overlap_count": current_turn.official_solution_image_overlap_count,
        "question_image_count": len(cumulative_images),
        "question_image_sha256": [image.sha256 for image in cumulative_images],
        "history_user_text_exact_native": not any(
            "history user" in error for error in errors
        ),
        "current_user_text_exact_native": not any(
            "current user text" in error for error in errors
        ),
        "sequence_semantics": (
            "prior native user text plus prior assistant candidate; current user turn "
            "replays unique cumulative official question-image data URIs"
        ),
    }
    if not audit["passed"]:
        raise RuntimeError("payload leakage audit failed: " + "; ".join(errors))
    return audit


def input_fingerprint(
    *,
    history: Sequence[HistoryEntry],
    current_turn: QuestionTurn,
    cumulative_images: Sequence[ImageInput],
) -> str:
    material = {
        "schema_version": SCHEMA_VERSION,
        "route": FIXED_ROUTE,
        "system_prompt_sha256": sha256_text(STATICIZER_SYSTEM),
        "wire_attempts": WIRE_ATTEMPTS,
        "wire_timeout_seconds": WIRE_TIMEOUT_SECONDS,
        "max_tokens": MAX_TOKENS,
        "thinking_level": THINKING_LEVEL,
        "benchmark_id": current_turn.benchmark_id,
        "sequence_id": current_turn.sequence_id,
        "turn_index": current_turn.turn_index,
        "item_id": current_turn.item_id,
        "question_text_native_sha256": sha256_text(current_turn.question_text_native),
        "history": [
            {
                "question_text_native_sha256": sha256_text(entry.question_text_native),
                "candidate_description_sha256": sha256_text(entry.candidate_description),
            }
            for entry in history
        ],
        "cumulative_question_images": [image.sha256 for image in cumulative_images],
    }
    return sha256_bytes(canonical_json_bytes(material))


def parse_retry_after(headers: Any) -> float | None:
    try:
        value = headers.get("Retry-After")
        return float(value) if value else None
    except Exception:
        return None


def post_chat(request_body: bytes, api_key: str, timeout_seconds: int) -> WireResult:
    request = urllib.request.Request(
        API_URL,
        data=request_body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Host": "api.360.cn",
        },
    )
    started = time.monotonic()
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as handle:
            body = handle.read()
            try:
                parsed = json.loads(body.decode("utf-8"))
                if not isinstance(parsed, dict):
                    raise ValueError("top-level response is not an object")
            except Exception as exc:
                return WireResult(
                    False,
                    handle.status,
                    body,
                    None,
                    f"InvalidJSONResponse: {type(exc).__name__}: {exc}",
                    parse_retry_after(handle.headers),
                    time.monotonic() - started,
                )
            return WireResult(
                True,
                handle.status,
                body,
                parsed,
                "",
                parse_retry_after(handle.headers),
                time.monotonic() - started,
            )
    except urllib.error.HTTPError as exc:
        body = exc.read()
        return WireResult(
            False,
            exc.code,
            body,
            None,
            f"HTTPError: {exc}",
            parse_retry_after(exc.headers),
            time.monotonic() - started,
        )
    except Exception as exc:
        return WireResult(
            False,
            None,
            b"",
            None,
            f"{type(exc).__name__}: {exc}",
            None,
            time.monotonic() - started,
        )


def response_text(response: dict[str, Any]) -> tuple[str, str]:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "", ""
    choice = choices[0]
    message = choice.get("message")
    if not isinstance(message, dict):
        return "", str(choice.get("finish_reason") or "")
    content = message.get("content")
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                value = block.get("text") or block.get("content")
                if value is not None:
                    parts.append(str(value))
            elif block is not None:
                parts.append(str(block))
        content = "\n".join(parts)
    return str(content or "").strip(), str(choice.get("finish_reason") or "")


def validate_candidate_description(text: str, finish_reason: str) -> tuple[bool, str]:
    if not text.strip():
        return False, "empty visible candidate description"
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
        return False, f"candidate contains forbidden control characters: {rendered}"
    finish = finish_reason.strip().lower()
    if any(fragment in finish for fragment in ("length", "token", "filter", "error", "fail")):
        return False, f"nonterminal finish_reason={finish_reason!r}"
    for pattern in DISALLOWED_CANDIDATE_OUTPUT:
        if pattern.search(text):
            return False, f"candidate violates no-solution/no-reasoning guard: {pattern.pattern}"
    return True, ""


def response_audit_views(
    response: dict[str, Any], *, attempt_dir: Path, audit_root: Path
) -> dict[str, Any]:
    result: dict[str, Any] = {"reasoning_artifacts": []}
    choices = response.get("choices")
    message: Any = None
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
    if isinstance(message, dict):
        result["response_message_artifact"] = store_json_value(
            message,
            base_path=attempt_dir / "response-message",
            audit_root=audit_root,
            kind="full_choices_0_message",
        )
        for field in ("reasoning_content", "reasoning", "thinking"):
            if field in message and message[field] not in (None, "", [], {}):
                artifact = store_json_value(
                    message[field],
                    base_path=attempt_dir / f"reasoning-{field}",
                    audit_root=audit_root,
                    kind=f"choices_0_message_{field}",
                )
                result["reasoning_artifacts"].append(
                    {"json_pointer": f"/choices/0/message/{field}", **artifact}
                )
    if "usage" in response:
        result["usage_artifact"] = store_json_value(
            response["usage"],
            base_path=attempt_dir / "usage",
            audit_root=audit_root,
            kind="response_usage",
        )
        try:
            result["reasoning_tokens"] = response["usage"][
                "completion_tokens_details"
            ]["reasoning_tokens"]
        except Exception:
            result["reasoning_tokens"] = None
    return result


def redact_secret(value: str, api_key: str) -> str:
    return value.replace(api_key, "[REDACTED]") if api_key else value


def retry_delay(attempt: int, retry_after: float | None) -> float:
    if retry_after is not None:
        return min(max(float(retry_after), 0.0), 60.0)
    return min((2 ** max(attempt - 1, 0)) + random.random(), 60.0)


def item_record_path(audit_root: Path, turn: QuestionTurn) -> Path:
    return (
        audit_root
        / "items"
        / safe_component(turn.benchmark_id)
        / safe_component(turn.sequence_id)
        / f"{turn.turn_index:04d}--{safe_component(turn.item_id)}.json"
    )


def item_attempt_dir(audit_root: Path, turn: QuestionTurn) -> Path:
    return (
        audit_root
        / "attempts"
        / safe_component(turn.benchmark_id)
        / safe_component(turn.sequence_id)
        / f"{turn.turn_index:04d}--{safe_component(turn.item_id)}"
    )


def attempt_paths(attempt_dir: Path) -> list[Path]:
    paths = (
        sorted(
            path
            for path in attempt_dir.glob("attempt-*.json")
            if re.fullmatch(r"attempt-\d{2}\.json", path.name)
        )
        if attempt_dir.exists()
        else []
    )
    expected = [attempt_dir / f"attempt-{index:02d}.json" for index in range(1, len(paths) + 1)]
    if paths != expected or len(paths) > WIRE_ATTEMPTS:
        raise RuntimeError(f"corrupt or non-contiguous attempt ledger: {attempt_dir}")
    return paths


def validate_stored_attempt(
    record: dict[str, Any],
    *,
    ordinal: int,
    request_sha256: str,
    fingerprint: str,
    audit_root: Path,
) -> None:
    if int(record.get("attempt", -1)) != ordinal:
        raise RuntimeError("attempt ordinal mismatch")
    if record.get("request_sha256") != request_sha256:
        raise RuntimeError("refusing stale attempt: request SHA-256 changed")
    if record.get("input_fingerprint") != fingerprint:
        raise RuntimeError("refusing stale attempt: input fingerprint changed")
    artifact = record.get("request_artifact")
    if not isinstance(artifact, dict) or not artifact.get("manifest_path"):
        raise RuntimeError("stored attempt is missing request artifact")
    restored = restore_lossless_body(
        audit_root / str(artifact["manifest_path"]), audit_root
    )
    if sha256_bytes(restored) != request_sha256:
        raise RuntimeError("stored request artifact does not match ledger")


def summarize_attempts(paths: Sequence[Path], audit_root: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for path in paths:
        record = read_json(path)
        summary = {
            "attempt": record.get("attempt"),
            "state": record.get("state"),
            "attempt_record_path": relative_path(path, audit_root),
            "request_sha256": record.get("request_sha256"),
            "http_status": record.get("http_status"),
            "response_body_sha256": record.get("response_body_sha256"),
            "validation_ok": record.get("validation_ok", False),
            "validation_reason": record.get("validation_reason") or record.get("error") or "",
        }
        summaries.append(summary)
    return summaries


def build_candidate_record(
    *,
    turn: QuestionTurn,
    current_images: Sequence[ImageInput],
    cumulative_images: Sequence[ImageInput],
    fingerprint: str,
    request_sha256: str,
    leakage_audit: dict[str, Any],
    attempt_dir: Path,
    audit_root: Path,
    success_attempt: dict[str, Any] | None,
) -> dict[str, Any]:
    paths = attempt_paths(attempt_dir)
    success = success_attempt is not None
    candidate = (
        str(success_attempt["candidate_description"])
        if success_attempt is not None
        else NO_DESCRIPTION_LABEL
    )
    final_attempt = success_attempt or (read_json(paths[-1]) if paths else {})
    return {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "terminal": True,
        "created_at": utc_now(),
        "status": "candidate_ready" if success else "failed_after_7_wire_attempts",
        "route": FIXED_ROUTE,
        "benchmark_id": turn.benchmark_id,
        "sequence_id": turn.sequence_id,
        "turn_index": turn.turn_index,
        "item_id": turn.item_id,
        "question_text_native_sha256": sha256_text(turn.question_text_native),
        "question_source_fingerprint": sha256_bytes(
            canonical_json_bytes(
                {
                    "question_text_native": turn.question_text_native,
                    "question_image_sha256": [image.sha256 for image in current_images],
                }
            )
        ),
        "candidate_description": candidate,
        "candidate_description_sha256": sha256_text(candidate),
        "candidate_requires_human_review": True,
        "candidate_only_not_applied": True,
        "canonical_writeback_performed": False,
        "input_fingerprint": fingerprint,
        "request_sha256": request_sha256,
        "wire_attempt_limit": WIRE_ATTEMPTS,
        "wire_timeout_seconds": WIRE_TIMEOUT_SECONDS,
        "wire_attempt_count": len(paths),
        "wire_attempts": summarize_attempts(paths, audit_root),
        "current_question_images": [image.descriptor() for image in current_images],
        "cumulative_question_images": [image.descriptor() for image in cumulative_images],
        "leakage_audit": leakage_audit,
        "raw_response_artifact": final_attempt.get("raw_response_artifact"),
        "response_message_artifact": final_attempt.get("response_message_artifact"),
        "reasoning_artifacts": final_attempt.get("reasoning_artifacts", []),
        "usage_artifact": final_attempt.get("usage_artifact"),
        "reasoning_tokens": final_attempt.get("reasoning_tokens"),
    }


def existing_candidate(
    path: Path, *, fingerprint: str, request_sha256: str
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    record = read_json(path)
    if record.get("schema_version") != CANDIDATE_SCHEMA_VERSION or record.get("terminal") is not True:
        raise RuntimeError(f"invalid candidate record: {path}")
    if record.get("input_fingerprint") != fingerprint:
        raise RuntimeError(f"refusing stale candidate: input fingerprint changed: {path}")
    if record.get("request_sha256") != request_sha256:
        raise RuntimeError(f"refusing stale candidate: request SHA-256 changed: {path}")
    candidate = str(record.get("candidate_description") or "")
    if sha256_text(candidate) != record.get("candidate_description_sha256"):
        raise RuntimeError(f"candidate description hash mismatch: {path}")
    audit = record.get("leakage_audit")
    if not isinstance(audit, dict) or audit.get("passed") is not True:
        raise RuntimeError(f"stored candidate lacks a passing leakage audit: {path}")
    return record


def run_turn(
    *,
    turn: QuestionTurn,
    history: Sequence[HistoryEntry],
    current_images: Sequence[ImageInput],
    cumulative_images: Sequence[ImageInput],
    audit_root: Path,
    api_key: str,
    wire_call: Callable[[bytes, str, int], WireResult] = post_chat,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    payload = build_payload(
        history=history,
        current_turn=turn,
        cumulative_images=cumulative_images,
    )
    leakage_audit = audit_payload(
        payload=payload,
        history=history,
        current_turn=turn,
        cumulative_images=cumulative_images,
    )
    request_body = canonical_json_bytes(payload)
    request_sha = sha256_bytes(request_body)
    fingerprint = input_fingerprint(
        history=history,
        current_turn=turn,
        cumulative_images=cumulative_images,
    )
    final_path = item_record_path(audit_root, turn)
    existing = existing_candidate(
        final_path, fingerprint=fingerprint, request_sha256=request_sha
    )
    if existing is not None:
        return existing

    ledger_dir = item_attempt_dir(audit_root, turn)
    ledger_dir.mkdir(parents=True, exist_ok=True)
    paths = attempt_paths(ledger_dir)
    success_attempt: dict[str, Any] | None = None
    for ordinal, path in enumerate(paths, start=1):
        record = read_json(path)
        validate_stored_attempt(
            record,
            ordinal=ordinal,
            request_sha256=request_sha,
            fingerprint=fingerprint,
            audit_root=audit_root,
        )
        if record.get("state") == "completed" and record.get("validation_ok") is True:
            text = str(record.get("candidate_description") or "")
            valid, reason = validate_candidate_description(
                text, str(record.get("finish_reason") or "")
            )
            if not valid:
                raise RuntimeError(f"stored valid attempt no longer validates: {reason}")
            success_attempt = record
            break

    for ordinal in range(len(paths) + 1, WIRE_ATTEMPTS + 1):
        if success_attempt is not None:
            break
        attempt_path = ledger_dir / f"attempt-{ordinal:02d}.json"
        request_artifact = store_lossless_body(
            body=request_body,
            base_path=ledger_dir / f"attempt-{ordinal:02d}-request-body",
            audit_root=audit_root,
            kind="exact_http_request_body",
        )
        attempt: dict[str, Any] = {
            "schema_version": ATTEMPT_SCHEMA_VERSION,
            "attempt": ordinal,
            "state": "wire_started",
            "started_at": utc_now(),
            "route": FIXED_ROUTE,
            "input_fingerprint": fingerprint,
            "request_sha256": request_sha,
            "request_bytes": len(request_body),
            "request_artifact": request_artifact,
            "leakage_audit": leakage_audit,
            "wire_attempt_limit": WIRE_ATTEMPTS,
            "timeout_seconds": WIRE_TIMEOUT_SECONDS,
        }
        # This write occurs before I/O.  If interrupted, resume conservatively
        # counts the uncertain wire attempt and continues with the next ordinal.
        atomic_write_json(attempt_path, attempt)
        wire = wire_call(request_body, api_key, WIRE_TIMEOUT_SECONDS)
        if api_key and api_key.encode("utf-8") in wire.body:
            raise RuntimeError(
                "provider response unexpectedly echoed the credential; raw body was not persisted"
            )
        response_artifact = store_lossless_body(
            body=wire.body,
            base_path=ledger_dir / f"attempt-{ordinal:02d}-response-body",
            audit_root=audit_root,
            kind="exact_raw_http_response_body",
        )
        body_response: dict[str, Any] | None = None
        response_consistency_error = ""
        if wire.ok:
            try:
                parsed_body = json.loads(wire.body.decode("utf-8"))
                if not isinstance(parsed_body, dict):
                    raise ValueError("top-level raw response is not an object")
                if wire.response is not None and parsed_body != wire.response:
                    raise ValueError("parsed response differs from exact raw response body")
                body_response = parsed_body
            except Exception as exc:
                response_consistency_error = (
                    f"RawResponseConsistencyError: {type(exc).__name__}: {exc}"
                )
        attempt.update(
            {
                "state": "completed",
                "completed_at": utc_now(),
                "http_status": wire.status,
                "latency_seconds": wire.latency_seconds,
                "wire_ok": wire.ok,
                "error": redact_secret(wire.error, api_key),
                "response_body_sha256": sha256_bytes(wire.body),
                "response_body_bytes": len(wire.body),
                "raw_response_artifact": response_artifact,
                "raw_response_semantically_verified": body_response is not None,
            }
        )
        if wire.ok and body_response is not None:
            text, finish_reason = response_text(body_response)
            valid, reason = validate_candidate_description(text, finish_reason)
            attempt.update(
                {
                    "candidate_description": text,
                    "candidate_description_sha256": sha256_text(text),
                    "finish_reason": finish_reason,
                    "validation_ok": valid,
                    "validation_reason": reason,
                }
            )
            attempt.update(
                response_audit_views(
                    body_response,
                    attempt_dir=ledger_dir / f"attempt-{ordinal:02d}-views",
                    audit_root=audit_root,
                )
            )
        else:
            attempt.update(
                {
                    "candidate_description": "",
                    "candidate_description_sha256": sha256_text(""),
                    "finish_reason": "",
                    "validation_ok": False,
                    "validation_reason": response_consistency_error
                    or redact_secret(wire.error, api_key)
                    or f"HTTP {wire.status}",
                }
            )
        atomic_write_json(attempt_path, attempt)
        if attempt["validation_ok"] is True:
            success_attempt = attempt
            break
        if ordinal < WIRE_ATTEMPTS:
            sleep_fn(retry_delay(ordinal, wire.retry_after))

    record = build_candidate_record(
        turn=turn,
        current_images=current_images,
        cumulative_images=cumulative_images,
        fingerprint=fingerprint,
        request_sha256=request_sha,
        leakage_audit=leakage_audit,
        attempt_dir=ledger_dir,
        audit_root=audit_root,
        success_attempt=success_attempt,
    )
    if success_attempt is None and record["wire_attempt_count"] != WIRE_ATTEMPTS:
        raise RuntimeError("non-success record did not consume exactly seven wire-attempt ordinals")
    record["audit_record_path"] = relative_path(final_path, audit_root)
    atomic_write_json(final_path, record)
    return record


def grouped_sequences(turns: Sequence[QuestionTurn]) -> list[list[QuestionTurn]]:
    grouped: dict[tuple[str, str], list[QuestionTurn]] = {}
    for turn in turns:
        grouped.setdefault((turn.benchmark_id, turn.sequence_id), []).append(turn)
    return [
        sorted(sequence, key=lambda turn: turn.turn_index)
        for _, sequence in sorted(grouped.items())
    ]


def run_sequence(
    *,
    turns: Sequence[QuestionTurn],
    audit_root: Path,
    api_key: str,
    on_candidate: Callable[[dict[str, Any]], None],
    wire_call: Callable[[bytes, str, int], WireResult] = post_chat,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    history: list[HistoryEntry] = []
    cumulative: list[ImageInput] = []
    seen_paths: set[Path] = set()
    cache: dict[Path, ImageInput] = {}
    ready = 0
    for turn in sorted(turns, key=lambda value: value.turn_index):
        current: list[ImageInput] = []
        for source, path in zip(turn.question_image_values, turn.question_image_paths):
            image = cache.get(path)
            if image is None:
                image = load_image_input(source, path, audit_root=audit_root)
                cache[path] = image
            current.append(image)
            if path not in seen_paths:
                cumulative.append(image)
                seen_paths.add(path)
        record = run_turn(
            turn=turn,
            history=history,
            current_images=current,
            cumulative_images=cumulative,
            audit_root=audit_root,
            api_key=api_key,
            wire_call=wire_call,
            sleep_fn=sleep_fn,
        )
        on_candidate(record)
        history.append(
            HistoryEntry(turn.question_text_native, record["candidate_description"])
        )
        ready += record["status"] == "candidate_ready"
    return {
        "benchmark_id": turns[0].benchmark_id,
        "sequence_id": turns[0].sequence_id,
        "item_count": len(turns),
        "candidate_ready_count": ready,
    }


def candidate_rows(audit_root: Path) -> list[dict[str, Any]]:
    paths = sorted((audit_root / "items").glob("**/*.json")) if (audit_root / "items").exists() else []
    rows = [read_json(path) for path in paths]
    for path, row in zip(paths, rows):
        if row.get("schema_version") != CANDIDATE_SCHEMA_VERSION or row.get("terminal") is not True:
            raise RuntimeError(f"invalid terminal candidate record: {path}")
    return sorted(
        rows,
        key=lambda row: (
            str(row["benchmark_id"]),
            str(row["sequence_id"]),
            int(row["turn_index"]),
            str(row["item_id"]),
        ),
    )


def publish_candidates(audit_root: Path, output_path: Path) -> None:
    atomic_write_jsonl(output_path, candidate_rows(audit_root))


def ensure_run_manifest(
    *,
    audit_root: Path,
    workspace: Path,
    items_path: Path,
    output_path: Path,
    selected_benchmarks: Sequence[str],
) -> dict[str, Any]:
    contract = {
        "schema_version": SCHEMA_VERSION,
        "api_url": API_URL,
        "route": FIXED_ROUTE,
        "workspace": str(workspace.resolve()),
        "items_path": str(items_path.resolve()),
        "items_sha256": sha256_file(items_path),
        "output_path": str(output_path.resolve()),
        "selected_benchmarks": sorted(selected_benchmarks),
        "wire_attempts": WIRE_ATTEMPTS,
        "wire_timeout_seconds": WIRE_TIMEOUT_SECONDS,
        "max_tokens": MAX_TOKENS,
        "thinking_level": THINKING_LEVEL,
        "system_prompt": STATICIZER_SYSTEM,
        "system_prompt_sha256": sha256_text(STATICIZER_SYSTEM),
        "source_content_fields": list(SOURCE_CONTENT_FIELDS),
        "sequence_policy": "pipeline_cumulative_history_and_question_images",
        "candidate_only_not_applied": True,
        "canonical_writeback_performed": False,
    }
    fingerprint = sha256_bytes(canonical_json_bytes(contract))
    path = audit_root / "run_manifest.json"
    if path.exists():
        existing = read_json(path)
        if existing.get("run_contract_fingerprint") != fingerprint:
            raise RuntimeError(f"refusing incompatible resume in {audit_root}")
        return existing
    if output_path.exists():
        raise RuntimeError("candidate output exists without its audit run manifest")
    manifest = {**contract, "created_at": utc_now(), "run_contract_fingerprint": fingerprint}
    atomic_write_json(path, manifest)
    return manifest


def run_dataset(
    *,
    turns: Sequence[QuestionTurn],
    audit_root: Path,
    output_path: Path,
    api_key: str,
    workers: int,
    wire_call: Callable[[bytes, str, int], WireResult] = post_chat,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    sequences = grouped_sequences(turns)
    if workers <= 0:
        raise ValueError("workers must be positive")
    publish_lock = threading.Lock()

    def on_candidate(_: dict[str, Any]) -> None:
        with publish_lock:
            publish_candidates(audit_root, output_path)

    # Reconstruct the independent JSONL from any already-terminal item records.
    on_candidate({})
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(workers, len(sequences))
    ) as executor:
        futures = [
            executor.submit(
                run_sequence,
                turns=sequence,
                audit_root=audit_root,
                api_key=api_key,
                on_candidate=on_candidate,
                wire_call=wire_call,
                sleep_fn=sleep_fn,
            )
            for sequence in sequences
        ]
        for future in concurrent.futures.as_completed(futures):
            summary = future.result()
            print(
                f"staticized sequence {summary['benchmark_id']}::{summary['sequence_id']} "
                f"({summary['candidate_ready_count']}/{summary['item_count']} candidates ready)",
                flush=True,
            )
    return candidate_rows(audit_root)


def parse_benchmark_filter(value: str) -> set[str] | None:
    selected = {part.strip() for part in value.split(",") if part.strip()}
    unknown = selected - SUPPORTED_BENCHMARKS
    if unknown:
        raise ValueError(f"unsupported benchmark filter: {sorted(unknown)}")
    return selected or None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate isolated question-only staticization candidate JSONL."
    )
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path)
    parser.add_argument("--benchmarks", default="")
    parser.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    parser.add_argument("--route", type=fixed_route, default=FIXED_ROUTE)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate and write the run contract without reading a key or calling the API",
    )
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    items_path = args.items.resolve()
    output_path = args.output.resolve()
    audit_root = (
        args.audit_dir.resolve()
        if args.audit_dir
        else output_path.with_name(output_path.name + ".audit")
    )
    if args.workers <= 0:
        parser.error("--workers must be positive")
    if not items_path.is_file():
        parser.error(f"items file does not exist: {items_path}")
    guard_output_paths(
        workspace=workspace,
        items_path=items_path,
        output_path=output_path,
        audit_root=audit_root,
    )
    selected = parse_benchmark_filter(args.benchmarks)
    rows = read_jsonl(items_path)
    turns = project_question_turns(
        rows, workspace=workspace, benchmark_filter=selected
    )
    selected_benchmarks = sorted({turn.benchmark_id for turn in turns})
    ensure_run_manifest(
        audit_root=audit_root,
        workspace=workspace,
        items_path=items_path,
        output_path=output_path,
        selected_benchmarks=selected_benchmarks,
    )
    preflight = {
        "schema_version": "science_arena_2026_question_only_staticizer_preflight_v1",
        "created_at": utc_now(),
        "valid": True,
        "item_count": len(turns),
        "sequence_count": len(grouped_sequences(turns)),
        "benchmarks": selected_benchmarks,
        "route": FIXED_ROUTE,
        "wire_attempts": WIRE_ATTEMPTS,
        "wire_timeout_seconds": WIRE_TIMEOUT_SECONDS,
        "source_content_fields_used": list(SOURCE_CONTENT_FIELDS),
        "forbidden_source_fields_used": [],
        "gold_field_count_in_wire_payload_design": 0,
        "solution_image_count_in_wire_payload_design": 0,
        "candidate_only_not_applied": True,
        "canonical_writeback_performed": False,
    }
    atomic_write_json(audit_root / "preflight.json", preflight)
    if args.preflight_only:
        print(json.dumps(preflight, ensure_ascii=False, indent=2))
        return 0

    api_key = load_api_key()
    candidates = run_dataset(
        turns=turns,
        audit_root=audit_root,
        output_path=output_path,
        api_key=api_key,
        workers=args.workers,
    )
    ready = sum(row["status"] == "candidate_ready" for row in candidates)
    print(
        f"candidate JSONL complete: {ready}/{len(candidates)} ready; "
        f"review-only output={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
