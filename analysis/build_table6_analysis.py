#!/usr/bin/env python3
"""Build the sealed four-competition Table 6 analysis artifacts.

This program is deliberately offline and standard-library-only.  It never calls
an API and never writes inside any formal run root.  It performs four jobs:

1. validate and tabulate the frozen 14-model aggregate scores;
2. independently reconstruct scores from item-level judge terminals;
3. calculate a fixed-seed whole-sequence-cluster bootstrap for the four-way
   macro mean of competition-normalized percentages; and
4. audit the cross-generation input and judge invariants needed for a defensible
   comparison.

When the NBPhO aggregate is not yet complete, ``--allow-incomplete`` produces a
clearly marked three-competition validation build.  It never substitutes that
partial macro for the four-competition target.
"""

from __future__ import annotations

import argparse
import base64
import copy
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence


ANALYSIS_SCHEMA = "science_arena_table6_four_competition_analysis_v1"
PROTOCOL_SCHEMA = "science_arena_table6_cross_generation_protocol_audit_v1"
BOOTSTRAP_SCHEMA = "science_arena_table6_sequence_cluster_bootstrap_v1"
DEFAULT_SEED = 20260829
DEFAULT_RESAMPLES = 10_000
JUDGE_ROUTE = "google/gemini-3.5-flash"
NO_ANSWER_LABEL = "不会做这道题"
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
FLOAT_TOLERANCE = 1e-8

# APFS/FileProvider-backed workspaces can occasionally block on the extra
# read-to-EOF performed by Python's unbounded buffered ``read()`` even after a
# regular file's declared bytes have arrived.  Read exactly the fstat size and
# also fail if the file changes during the read.  Complete formal sources are
# immutable, so this is both faster and a stronger snapshot boundary.
_FILE_SHA_CACHE: dict[tuple[str, int, int], str] = {}
_DESCRIPTOR_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}
_FILE_BYTES_CACHE: dict[tuple[str, int, int], bytes] = {}
_JSON_VALUE_CACHE: dict[tuple[str, int, int], Any] = {}
_MAX_CACHED_FILE_BYTES = 8 * 1024 * 1024
_READ_TIMEOUT_SECONDS = 15
_READ_ATTEMPTS = 3


@dataclass(frozen=True)
class RunSpec:
    key: str
    display_name: str
    benchmark_id: str
    raw_max: float
    run_rel: str
    generation: str
    seal_audit_rel: str | None


RUN_SPECS: tuple[RunSpec, ...] = (
    RunSpec(
        key="ipho2026",
        display_name="IPhO 2026",
        benchmark_id="ipho2026",
        raw_max=30.0,
        run_rel=(
            "experiments/2026_reval/runs/"
            "ipho2026_table6_qonly_gemini31_frozen_formal_20260829"
        ),
        generation="formal_qonly_v1_with_sealed_vision_reuse",
        seal_audit_rel=(
            "experiments/2026_reval/audits/"
            "ipho2026_table6_formal_judge_independent_audit_20260829.json"
        ),
    ),
    RunSpec(
        key="icho2026",
        display_name="IChO 2026",
        benchmark_id="icho2026_theory",
        raw_max=60.0,
        run_rel=(
            "experiments/2026_reval/runs/"
            "icho2026_table6_qonly_gemini31_frozen_v2_formal_20260829"
        ),
        generation="formal_qonly_v2_with_sealed_vision_reuse",
        seal_audit_rel=(
            "experiments/2026_reval/audits/"
            "icho2026_table6_formal_judge_independent_audit_20260829.json"
        ),
    ),
    RunSpec(
        key="nbpho2026",
        display_name="NBPhO 2026",
        benchmark_id="nbpho2026",
        raw_max=72.0,
        run_rel=(
            "experiments/2026_reval/runs/"
            "nbpho2026_table6_qonly_gemini31_frozen_v2_"
            "judge_v8_materialized_full_fresh_20260829"
        ),
        generation="materialized_full_fresh_judge_v8",
        seal_audit_rel=(
            "experiments/2026_reval/audits/"
            "nbpho2026_table6_judge_v8_post_run_independent_audit_result_20260829.json"
        ),
    ),
    RunSpec(
        key="usapho2026",
        display_name="USAPhO 2026",
        benchmark_id="usapho2026",
        raw_max=150.0,
        run_rel=(
            "experiments/2026_reval/runs/"
            "usapho2026_table6_qonly_gemini31_frozen_v2_"
            "judge_v5_full_fresh_20260829"
        ),
        generation="sealed_full_fresh_judge_v5",
        seal_audit_rel=(
            "experiments/2026_reval/audits/"
            "usapho2026_table6_judge_v5_full_fresh_"
            "independent_countersign_20260829.json"
        ),
    ),
)

MODEL_MANIFEST_REL = (
    "experiments/2026_reval/manifests/table6_models_frozen_20260828.json"
)

SEALED_PROTOCOL_AUDITS: dict[str, dict[str, str]] = {
    "ipho2026": {
        "main": (
            "experiments/2026_reval/audits/"
            "ipho2026_table6_formal_judge_independent_audit_20260829.json"
        ),
        "text": (
            "experiments/2026_reval/audits/"
            "ipho2026_text_formal_independent_audit_20260829.json"
        ),
        "vision_provenance": (
            "experiments/2026_reval/runs/"
            "ipho2026_table6_qonly_gemini31_frozen_formal_20260829/"
            "vision_reuse_provenance.json"
        ),
    },
    "icho2026": {
        "main": (
            "experiments/2026_reval/audits/"
            "icho2026_table6_formal_judge_independent_audit_20260829.json"
        ),
        "text": (
            "experiments/2026_reval/audits/"
            "icho2026_text_formal_independent_audit_20260829.json"
        ),
        "vision_provenance": (
            "experiments/2026_reval/runs/"
            "icho2026_table6_qonly_gemini31_frozen_v2_formal_20260829/"
            "vision_reuse_provenance.json"
        ),
    },
    "usapho2026": {
        "main": (
            "experiments/2026_reval/audits/"
            "usapho2026_table6_judge_v5_full_fresh_"
            "independent_countersign_20260829.json"
        ),
    },
    "nbpho2026": {
        "main": (
            "experiments/2026_reval/audits/"
            "nbpho2026_table6_judge_v8_post_run_"
            "independent_audit_result_20260829.json"
        ),
        "decimal_sidecar": (
            "experiments/2026_reval/audits/"
            "nbpho2026_table6_judge_v8_decimal_publication_sidecar_v1_20260829.json"
        ),
    },
}


class AuditFailure(RuntimeError):
    """Raised when a fail-closed source or protocol invariant is violated."""


def fail(message: str) -> None:
    raise AuditFailure(message)


def require(condition: bool, message: str) -> None:
    if not condition:
        fail(message)


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def read_exact_regular_bytes(path: Path) -> bytes:
    path_stat = path.stat()
    cache_key = (str(path.resolve()), path_stat.st_size, path_stat.st_mtime_ns)
    if cache_key in _FILE_BYTES_CACHE:
        return _FILE_BYTES_CACHE[cache_key]
    before = path.stat()
    require(path.is_file(), f"not a regular file: {path}")
    # On the macOS FileProvider/APFS workspace used for this corpus, repeated
    # direct reads from one long-lived Python process can park in ``read(2)``.
    # A short-lived system reader avoids that provider-state issue without
    # writing to or metadata-mutating the source.  Original before/after stat,
    # captured length, and SHA checks retain the immutable snapshot boundary.
    completed: subprocess.CompletedProcess[bytes] | None = None
    for _attempt in range(1, _READ_ATTEMPTS + 1):
        try:
            completed = subprocess.run(
                ["/bin/cat", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=_READ_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            continue
        if completed.returncode == 0:
            break
    require(completed is not None, f"timed out reading immutable input: {path}")
    require(
        completed.returncode == 0,
        f"cannot read {path}: {completed.stderr.decode('utf-8', errors='replace')}",
    )
    raw = completed.stdout
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
    ):
        # FileProvider hydration can replace a dataless inode or refresh its
        # metadata on first access.  A byte-identical second capture proves the
        # observed content itself was stable; a true mutation still fails.
        confirmation = subprocess.run(
            ["/bin/cat", str(path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=_READ_TIMEOUT_SECONDS,
        )
        require(
            confirmation.returncode == 0 and confirmation.stdout == raw,
            f"file content changed during read: {path}",
        )
        after = path.stat()
    require(len(raw) == after.st_size, f"short read of {path}")
    exact_key = (str(path.resolve()), after.st_size, after.st_mtime_ns)
    _FILE_SHA_CACHE.setdefault(exact_key, sha256_bytes(raw))
    if len(raw) <= _MAX_CACHED_FILE_BYTES:
        _FILE_BYTES_CACHE[exact_key] = raw
    return raw


def preload_regular_files(paths: Iterable[Path], *, workers: int = 1) -> int:
    """Snapshot immutable inputs concurrently into the process read cache.

    The corpus lives on a FileProvider-backed APFS tree.  A single bounded
    reader avoids provider saturation (several concurrent readers can stall on
    already-local files) while preserving exact source before/after stat checks
    in ``read_exact_regular_bytes``.  Paths are de-duplicated by resolved source
    identity before any worker is started.  The parameter is retained for
    controlled use on non-FileProvider filesystems.
    """

    unique: dict[tuple[str, int, int], Path] = {}
    for path in paths:
        if not path.is_file():
            continue
        stat_result = path.stat()
        key = (str(path.resolve()), stat_result.st_size, stat_result.st_mtime_ns)
        unique.setdefault(key, path)
    if not unique:
        return 0
    with ThreadPoolExecutor(max_workers=min(workers, len(unique))) as executor:
        futures = {
            executor.submit(read_exact_regular_bytes, path): path
            for path in unique.values()
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                future.result()
            except Exception as exc:
                fail(f"cannot preload immutable input {path}: {type(exc).__name__}: {exc}")
            if completed_count % 500 == 0 or completed_count == len(unique):
                print(
                    f"preloaded {completed_count}/{len(unique)} immutable protocol records",
                    file=sys.stderr,
                    flush=True,
                )
    return len(unique)


def run_protocol_json_paths(run_root: Path) -> list[Path]:
    """Return all stored solver/judge JSON records needed by the deep audit."""

    paths: list[Path] = []
    for subtree_name in ("solver", "judge"):
        subtree = run_root / subtree_name
        if not subtree.is_dir():
            continue
        for directory, _subdirectories, filenames in os.walk(subtree, followlinks=True):
            directory_path = Path(directory)
            paths.extend(
                directory_path / filename
                for filename in filenames
                if filename.endswith(".json")
            )
    return paths


def sha256_file(path: Path) -> str:
    stat_result = path.stat()
    key = (str(path.resolve()), stat_result.st_size, stat_result.st_mtime_ns)
    if key not in _FILE_SHA_CACHE:
        _FILE_SHA_CACHE[key] = sha256_bytes(read_exact_regular_bytes(path))
    return _FILE_SHA_CACHE[key]


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def canonical_digest(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def read_json(path: Path) -> Any:
    stat_result = path.stat()
    key = (str(path.resolve()), stat_result.st_size, stat_result.st_mtime_ns)
    if key in _JSON_VALUE_CACHE:
        return _JSON_VALUE_CACHE[key]
    try:
        value = json.loads(read_exact_regular_bytes(path).decode("utf-8"))
        after = path.stat()
        _JSON_VALUE_CACHE[(str(path.resolve()), after.st_size, after.st_mtime_ns)] = value
        return value
    except Exception as exc:  # pragma: no cover - diagnostic path
        fail(f"cannot parse strict JSON {path}: {type(exc).__name__}: {exc}")


def read_json_batch(paths: Sequence[Path]) -> list[Any]:
    """Read several immutable JSON files through one short-lived reader.

    Passing a route's terminal files to one ``cat`` invocation avoids the
    per-open throttling imposed by the Desktop FileProvider.  Python's strict
    decoder then consumes the concatenated JSON documents one at a time.  Each
    source still receives the same before/after inode-size-mtime guard.
    """

    if not paths:
        return []
    before: list[os.stat_result] = []
    for path in paths:
        require(path.is_file(), f"missing immutable JSON input: {path}")
        before.append(path.stat())
    try:
        completed = subprocess.run(
            ["/bin/cat", *[str(path) for path in paths]],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=max(120, len(paths) * 2),
        )
    except subprocess.TimeoutExpired:
        if len(paths) > 1:
            midpoint = len(paths) // 2
            print(
                f"splitting throttled JSON batch of {len(paths)} beginning {paths[0]}",
                file=sys.stderr,
                flush=True,
            )
            return read_json_batch(paths[:midpoint]) + read_json_batch(paths[midpoint:])
        return [read_json(paths[0])]
    if completed.returncode != 0:
        if len(paths) > 1:
            midpoint = len(paths) // 2
            print(
                f"splitting canceled JSON batch of {len(paths)} beginning {paths[0]}",
                file=sys.stderr,
                flush=True,
            )
            return read_json_batch(paths[:midpoint]) + read_json_batch(paths[midpoint:])
        return [read_json(paths[0])]
    try:
        text_value = completed.stdout.decode("utf-8")
        decoder = json.JSONDecoder()
        values: list[Any] = []
        offset = 0
        while True:
            whitespace = re.match(r"\s*", text_value[offset:])
            offset += len(whitespace.group(0)) if whitespace else 0
            if offset == len(text_value):
                break
            value, offset = decoder.raw_decode(text_value, offset)
            values.append(value)
    except Exception as exc:
        fail(f"cannot decode concatenated JSON batch beginning {paths[0]}: {exc}")
    require(
        len(values) == len(paths),
        f"JSON batch count mismatch: decoded {len(values)} for {len(paths)} paths",
    )
    for path, prior, value in zip(paths, before, values):
        after = path.stat()
        if (prior.st_ino, prior.st_size, prior.st_mtime_ns) != (
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            confirmation = subprocess.run(
                ["/bin/cat", str(path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=_READ_TIMEOUT_SECONDS,
            )
            require(
                confirmation.returncode == 0,
                f"cannot confirm hydrated JSON input: {path}",
            )
            try:
                confirmed_value = json.loads(confirmation.stdout.decode("utf-8"))
            except Exception as exc:
                fail(f"cannot decode confirmation read {path}: {exc}")
            require(
                canonical_digest(confirmed_value) == canonical_digest(value),
                f"file content changed during batch read: {path}",
            )
        final_stat = path.stat()
        _JSON_VALUE_CACHE[
            (str(path.resolve()), final_stat.st_size, final_stat.st_mtime_ns)
        ] = value
    return values


def preload_json_groups(
    path_groups: Sequence[Sequence[Path]], *, workers: int = 4
) -> int:
    """Batch-cache exact JSON path groups with bounded route-level parallelism."""

    groups: list[list[Path]] = []
    seen: set[tuple[str, int, int]] = set()
    for path_group in path_groups:
        group: list[Path] = []
        for path in path_group:
            stat_result = path.stat()
            key = (str(path.resolve()), stat_result.st_size, stat_result.st_mtime_ns)
            if key in seen or key in _JSON_VALUE_CACHE:
                continue
            seen.add(key)
            group.append(path)
        if group:
            groups.append(group)
    if not groups:
        return 0
    with ThreadPoolExecutor(max_workers=min(workers, len(groups))) as executor:
        futures = {executor.submit(read_json_batch, group): group for group in groups}
        completed_count = 0
        for future in as_completed(futures):
            group = futures[future]
            try:
                future.result()
            except Exception as exc:
                fail(
                    f"cannot preload JSON route batch beginning {group[0]}: "
                    f"{type(exc).__name__}: {exc}"
                )
            completed_count += len(group)
    return completed_count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = read_exact_regular_bytes(path).decode("utf-8")
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            require(
                isinstance(value, dict),
                f"{path}:{line_number}: JSONL row is not an object",
            )
            rows.append(value)
    except AuditFailure:
        raise
    except Exception as exc:  # pragma: no cover - diagnostic path
        fail(f"cannot parse JSONL {path}: {type(exc).__name__}: {exc}")
    return rows


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", str(value)).strip("._") or "unnamed"


def abs_from_workspace(workspace: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value)
    return path if path.is_absolute() else workspace / path


def rel_to_workspace(workspace: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(workspace.resolve()))
    except ValueError:
        return str(path.resolve())


def descriptor(path: Path) -> dict[str, Any]:
    stat_result = path.stat()
    key = (str(path.resolve()), stat_result.st_size, stat_result.st_mtime_ns)
    if key not in _DESCRIPTOR_CACHE:
        raw = read_exact_regular_bytes(path)
        require(raw.startswith(PNG_MAGIC), f"not a PNG file: {path}")
        _DESCRIPTOR_CACHE[key] = {
            "path": str(path.resolve()),
            "sha256": sha256_bytes(raw),
            "bytes": len(raw),
            "mime_type": "image/png",
        }
    return copy.deepcopy(_DESCRIPTOR_CACHE[key])


def descriptor_core(value: Mapping[str, Any]) -> tuple[str, str, int, str]:
    return (
        str(Path(str(value.get("path") or "")).resolve()),
        str(value.get("sha256") or ""),
        int(value.get("bytes") or -1),
        str(value.get("mime_type") or ""),
    )


def prompt_text(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)
    return str(value)


def preferred_item_value(item: Mapping[str, Any], preferred: str, legacy: str) -> Any:
    value = item.get(preferred)
    return value if value not in (None, "", [], {}) else item.get(legacy)


def grading_rubric_for(item: Mapping[str, Any]) -> Any:
    return preferred_item_value(item, "grading_rubric", "official_rubric")


def normalized_rubric_criteria(rubric: Any) -> list[dict[str, Any]]:
    if not isinstance(rubric, dict) or not isinstance(rubric.get("criteria"), list):
        return []
    normalized: list[dict[str, Any]] = []
    for index, raw in enumerate(rubric["criteria"], start=1):
        if not isinstance(raw, dict):
            continue
        criterion = copy.deepcopy(raw)
        criterion["criterion_id"] = str(
            criterion.get("criterion_id")
            or criterion.get("id")
            or f"criterion_{index:02d}"
        )
        criterion["max_score"] = criterion.get(
            "max_score", criterion.get("max_points", criterion.get("points"))
        )
        normalized.append(criterion)
    return normalized


def rubric_for_prompt(item: Mapping[str, Any]) -> Any:
    rubric = grading_rubric_for(item)
    if not isinstance(rubric, dict):
        return rubric
    rendered = copy.deepcopy(rubric)
    if isinstance(rendered.get("criteria"), list):
        rendered["criteria"] = normalized_rubric_criteria(rendered)
    return rendered


def question_staticization_verified(item: Mapping[str, Any]) -> bool:
    if item.get("question_only_staticization_verified") is True:
        return True
    metadata = item.get("staticization") or item.get("question_staticization") or {}
    return (
        isinstance(metadata, dict)
        and str(metadata.get("solution_leakage_check") or "").lower() == "passed"
    )


def solver_prompt(item: Mapping[str, Any], input_mode: str) -> str:
    question = (
        item["question_text_native"]
        if input_mode == "vision"
        else item["question_text_staticized"]
    )
    score_line = (
        f"Maximum raw score: {item['max_score']}\n"
        if item.get("solver_prompt_disclose_max_score", True) is not False
        else ""
    )
    return (
        f"Competition: {item['benchmark_id']}\n"
        f"Problem sequence: {item['sequence_id']}\n"
        f"Current item: {item['item_id']}\n"
        f"{score_line}\n"
        f"{question}\n\n"
        "Give a concrete answer for the current item. End with a line beginning "
        "`FINAL ANSWER:`."
    )


def question_image_paths(item: Mapping[str, Any], workspace: Path) -> list[Path]:
    values = item.get("question_image_paths")
    if values is None and isinstance(item.get("visual_dependency"), dict):
        values = item["visual_dependency"].get("question_image_paths")
    return [abs_from_workspace(workspace, str(value)) for value in (values or [])]


def solution_image_paths(item: Mapping[str, Any], workspace: Path) -> list[Path]:
    return [
        abs_from_workspace(workspace, str(value))
        for value in (item.get("official_solution_image_paths") or [])
    ]


def strings_in(value: Any) -> Iterator[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for child in value.values():
            yield from strings_in(child)
    elif isinstance(value, list):
        for child in value:
            yield from strings_in(child)


def image_blocks_in(value: Any) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    if isinstance(value, dict):
        if value.get("type") in {"image_url", "input_image"}:
            blocks.append(value)
        for child in value.values():
            blocks.extend(image_blocks_in(child))
    elif isinstance(value, list):
        for child in value:
            blocks.extend(image_blocks_in(child))
    return blocks


def block_url(block: Mapping[str, Any]) -> str:
    value = block.get("image_url")
    if isinstance(value, dict):
        value = value.get("url")
    return str(value or "")


def validate_data_image_url(url: str, expected: Mapping[str, Any], identity: str) -> None:
    prefix = "data:image/png;base64,"
    require(url.startswith(prefix), f"{identity}: image block is not a PNG data URL")
    encoded = url[len(prefix) :]
    redacted = re.fullmatch(
        r"\[REDACTED sha256=([0-9a-f]{64}) bytes=([0-9]+)\]", encoded
    )
    if redacted:
        require(
            redacted.group(1) == expected["sha256"],
            f"{identity}: redacted image SHA mismatch",
        )
        require(
            int(redacted.group(2)) == expected["bytes"],
            f"{identity}: redacted image byte count mismatch",
        )
        return
    try:
        raw = base64.b64decode(encoded, validate=True)
    except Exception as exc:
        fail(f"{identity}: invalid image data URL: {exc}")
    require(sha256_bytes(raw) == expected["sha256"], f"{identity}: image bytes mismatch")


def assert_close(actual: Any, expected: float, identity: str) -> None:
    try:
        number = float(actual)
    except (TypeError, ValueError):
        fail(f"{identity}: expected numeric {expected}, got {actual!r}")
    require(math.isfinite(number), f"{identity}: non-finite number")
    require(abs(number - expected) <= FLOAT_TOLERANCE, f"{identity}: {number} != {expected}")


def model_manifest(workspace: Path) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    path = workspace / MODEL_MANIFEST_REL
    manifest = read_json(path)
    require(isinstance(manifest, dict), f"model manifest is not an object: {path}")
    models = manifest.get("models")
    require(isinstance(models, list) and len(models) == 14, "frozen roster is not 14 models")
    orders = [model.get("table6_order") for model in models]
    require(orders == list(range(1, 15)), f"invalid Table 6 model order: {orders}")
    routes = [str(model.get("route") or "") for model in models]
    require(len(routes) == len(set(routes)), "duplicate route in frozen model roster")
    modes = [model.get("input_mode") for model in models]
    require(set(modes) == {"vision", "text_staticized"}, "unexpected frozen modalities")
    require(modes.count("vision") == 8, "frozen roster must contain eight vision routes")
    require(
        modes.count("text_staticized") == 6,
        "frozen roster must contain six text-staticized routes",
    )
    return path, manifest, models


def resolve_manifest_items(
    workspace: Path, run_root: Path
) -> tuple[dict[str, Any], Path, list[dict[str, Any]]]:
    manifest_path = run_root / "run_manifest.json"
    require(manifest_path.is_file(), f"missing run manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    require(isinstance(manifest, dict), f"run manifest is not an object: {manifest_path}")
    items_path = abs_from_workspace(workspace, str(manifest.get("items_path") or ""))
    require(items_path.is_file(), f"missing frozen items file: {items_path}")
    require(
        sha256_file(items_path) == manifest.get("items_sha256"),
        f"items SHA mismatch in {manifest_path}",
    )
    items = read_jsonl(items_path)
    return manifest, items_path, items


def sequence_order(items: Sequence[Mapping[str, Any]]) -> list[str]:
    seen: OrderedDict[str, None] = OrderedDict()
    for item in items:
        seen.setdefault(str(item["sequence_id"]), None)
    return list(seen)


def items_by_sequence(
    items: Sequence[dict[str, Any]],
) -> OrderedDict[str, list[dict[str, Any]]]:
    result: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    for item in items:
        result.setdefault(str(item["sequence_id"]), []).append(item)
    for sequence_id, sequence_items in result.items():
        turns = [int(item["turn_index"]) for item in sequence_items]
        require(
            turns == list(range(1, len(turns) + 1)),
            f"{sequence_id}: non-contiguous canonical turn order {turns}",
        )
    return result


def scoring_metadata(items: Sequence[dict[str, Any]], expected_scale: float) -> dict[str, Any]:
    grouped = items_by_sequence(items)
    weighted_mode = any("problem_weight" in item for item in items)
    clusters: list[dict[str, Any]] = []
    for sequence_id, sequence_items in grouped.items():
        item_max = sum(float(item["max_score"]) for item in sequence_items)
        declared_raw = {
            float(item["problem_raw_max"])
            for item in sequence_items
            if item.get("problem_raw_max") is not None
        }
        require(
            len(declared_raw) <= 1,
            f"{sequence_id}: inconsistent problem_raw_max metadata",
        )
        raw_max = next(iter(declared_raw)) if declared_raw else item_max
        assert_close(item_max, raw_max, f"{sequence_id} item maxima")
        if weighted_mode:
            weights = {float(item["problem_weight"]) for item in sequence_items}
            require(len(weights) == 1, f"{sequence_id}: inconsistent problem weights")
            denominator = next(iter(weights))
        else:
            denominator = raw_max
        clusters.append(
            {
                "sequence_id": sequence_id,
                "item_count": len(sequence_items),
                "raw_max": raw_max,
                "competition_scale_denominator": denominator,
            }
        )
    natural_scale = sum(cluster["competition_scale_denominator"] for cluster in clusters)
    assert_close(natural_scale, expected_scale, "competition score scale")
    return {
        "weighted_mode": weighted_mode,
        "competition_scale": expected_scale,
        "underlying_item_max_sum": sum(float(item["max_score"]) for item in items),
        "clusters": clusters,
    }


def judge_terminal_path(run_root: Path, route: str, item: Mapping[str, Any]) -> Path:
    return (
        run_root
        / "judge"
        / safe_name(route)
        / safe_name(str(item["benchmark_id"]))
        / safe_name(str(item["sequence_id"]))
        / f"{safe_name(str(item['item_id']))}.json"
    )


def solver_terminal_path(run_root: Path, route: str, item: Mapping[str, Any]) -> Path:
    return (
        run_root
        / "solver"
        / safe_name(route)
        / safe_name(str(item["benchmark_id"]))
        / safe_name(str(item["sequence_id"]))
        / f"{safe_name(str(item['item_id']))}.json"
    )


def normalized_criterion_contract(item: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "criterion_id": str(criterion["criterion_id"]),
            "max_score": float(criterion["max_score"]),
        }
        for criterion in normalized_rubric_criteria(grading_rubric_for(item))
    ]


def reconstruct_scores(
    spec: RunSpec,
    run_root: Path,
    items: Sequence[dict[str, Any]],
    models: Sequence[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, dict[str, float]]]]:
    scoring = scoring_metadata(items, spec.raw_max)
    cluster_meta = {entry["sequence_id"]: entry for entry in scoring["clusters"]}
    totals: dict[str, float] = {}
    clusters_by_route: dict[str, dict[str, dict[str, float]]] = {}
    terminal_groups = [
        [judge_terminal_path(run_root, str(model["route"]), item) for item in items]
        for model in models
    ]
    preload_json_groups(terminal_groups)
    for model, terminal_paths in zip(models, terminal_groups):
        route = str(model["route"])
        raw_scores: dict[str, float] = {sequence: 0.0 for sequence in cluster_meta}
        terminals = [read_json(path) for path in terminal_paths]
        for item, path, terminal in zip(items, terminal_paths, terminals):
            require(isinstance(terminal, dict), f"judge terminal is not an object: {path}")
            require(terminal.get("terminal") is True, f"non-terminal judge record: {path}")
            require(terminal.get("solver_route") == route, f"solver route mismatch: {path}")
            require(terminal.get("item_id") == item["item_id"], f"item mismatch: {path}")
            score = float(terminal["score"])
            maximum = float(item["max_score"])
            require(
                -FLOAT_TOLERANCE <= score <= maximum + FLOAT_TOLERANCE,
                f"out-of-range score {score} in {path}",
            )
            assert_close(terminal.get("max_score"), maximum, f"{path} max_score")
            raw_scores[str(item["sequence_id"])] += score
        route_clusters: dict[str, dict[str, float]] = OrderedDict()
        for sequence_id in sequence_order(items):
            meta = cluster_meta[sequence_id]
            raw_score = raw_scores[sequence_id]
            raw_max = float(meta["raw_max"])
            denominator = float(meta["competition_scale_denominator"])
            score_units = raw_score / raw_max * denominator
            route_clusters[sequence_id] = {
                "raw_score": raw_score,
                "raw_max": raw_max,
                "competition_score": score_units,
                "competition_max": denominator,
            }
        total = sum(value["competition_score"] for value in route_clusters.values())
        totals[route] = total
        clusters_by_route[route] = route_clusters
        print(
            f"reconstructed {spec.key} judge clusters for {route}",
            file=sys.stderr,
            flush=True,
        )
    return totals, clusters_by_route


def clusters_from_aggregate_breakdown(
    spec: RunSpec,
    aggregate: Mapping[str, Any],
    items: Sequence[dict[str, Any]],
    models: Sequence[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, dict[str, dict[str, float]]]] | None:
    """Validate and reuse a complete aggregate's explicit sequence ledger.

    This is the speed-critical path used when a sealed aggregate already
    carries all sequence-level contributions needed by the cluster bootstrap.
    It does not synthesize clusters from whole-competition scores.  Older IPhO
    aggregate v1 has an empty ledger and therefore correctly falls back to
    item-level judge-terminal reconstruction.
    """

    aggregate_rows = aggregate.get("aggregates")
    require(isinstance(aggregate_rows, dict), f"{spec.key}: missing aggregate rows")
    scoring = scoring_metadata(items, spec.raw_max)
    metadata = OrderedDict(
        (str(cluster["sequence_id"]), cluster) for cluster in scoring["clusters"]
    )
    totals: dict[str, float] = {}
    clusters_by_route: dict[str, dict[str, dict[str, float]]] = {}
    for model_index, model in enumerate(models):
        route = str(model["route"])
        benchmark = aggregate_rows.get(route, {}).get(spec.benchmark_id)
        require(isinstance(benchmark, dict), f"{spec.key}: missing aggregate row for {route}")
        breakdown = benchmark.get("sequence_breakdown")
        if model_index == 0 and breakdown == {}:
            return None
        require(isinstance(breakdown, dict), f"{spec.key}/{route}: missing sequence breakdown")
        require(
            list(breakdown) == list(metadata),
            f"{spec.key}/{route}: sequence breakdown order/set drift",
        )
        route_clusters: dict[str, dict[str, float]] = OrderedDict()
        for sequence_id, meta in metadata.items():
            observed = breakdown[sequence_id]
            require(isinstance(observed, dict), f"{spec.key}/{route}/{sequence_id}: malformed breakdown")
            require(
                observed.get("complete") is True,
                f"{spec.key}/{route}/{sequence_id}: incomplete sequence breakdown",
            )
            raw_max = float(meta["raw_max"])
            competition_max = float(meta["competition_scale_denominator"])
            if observed.get("raw_max") is not None:
                assert_close(
                    observed["raw_max"],
                    raw_max,
                    f"{spec.key}/{route}/{sequence_id} aggregate raw max",
                )
            if observed.get("weight") is not None:
                assert_close(
                    observed["weight"],
                    competition_max,
                    f"{spec.key}/{route}/{sequence_id} aggregate weight",
                )
            raw_score = float(observed["raw_score"])
            require(
                -FLOAT_TOLERANCE <= raw_score <= raw_max + FLOAT_TOLERANCE,
                f"{spec.key}/{route}/{sequence_id}: aggregate raw score outside range",
            )
            computed_score = raw_score / raw_max * competition_max
            if observed.get("weighted_score") is not None:
                assert_close(
                    observed["weighted_score"],
                    computed_score,
                    f"{spec.key}/{route}/{sequence_id} weighted contribution",
                )
            route_clusters[sequence_id] = {
                "raw_score": raw_score,
                "raw_max": raw_max,
                "competition_score": computed_score,
                "competition_max": competition_max,
            }
        total = sum(cluster["competition_score"] for cluster in route_clusters.values())
        assert_close(
            benchmark.get("score"), total, f"{spec.key}/{route} sequence-ledger total"
        )
        totals[route] = total
        clusters_by_route[route] = route_clusters
    return totals, clusters_by_route


def validate_aggregate(
    spec: RunSpec,
    aggregate_path: Path,
    aggregate: Mapping[str, Any],
    items: Sequence[dict[str, Any]],
    models: Sequence[dict[str, Any]],
    reconstructed: Mapping[str, float],
) -> None:
    require(aggregate.get("complete") is True, f"aggregate is incomplete: {aggregate_path}")
    require(aggregate.get("missing_count") == 0, f"aggregate has missing rows: {aggregate_path}")
    require(aggregate.get("missing") == [], f"aggregate missing ledger is not empty: {aggregate_path}")
    require(
        aggregate.get("row_count") == len(items) * len(models),
        f"aggregate row count mismatch: {aggregate_path}",
    )
    aggregates = aggregate.get("aggregates")
    require(isinstance(aggregates, dict), f"aggregate payload missing: {aggregate_path}")
    expected_routes = [str(model["route"]) for model in models]
    require(set(aggregates) == set(expected_routes), f"aggregate route set mismatch: {aggregate_path}")
    for route in expected_routes:
        benchmark = aggregates[route].get(spec.benchmark_id)
        require(isinstance(benchmark, dict), f"missing {spec.benchmark_id} for {route}")
        require(benchmark.get("complete") is True, f"incomplete aggregate row for {route}")
        require(benchmark.get("item_count") == len(items), f"item count mismatch for {route}")
        require(
            benchmark.get("answered_item_count") == len(items),
            f"answered item count mismatch for {route}",
        )
        assert_close(benchmark.get("report_scale"), spec.raw_max, f"{route} report scale")
        assert_close(benchmark.get("score"), reconstructed[route], f"{route} aggregate score")


def validate_staticization_freeze(item: Mapping[str, Any], identity: str) -> None:
    require(question_staticization_verified(item), f"{identity}: question-only staticization not verified")
    staticized = item.get("question_text_staticized")
    native = item.get("question_text_native")
    require(isinstance(staticized, str) and staticized.strip(), f"{identity}: empty staticization")
    require(isinstance(native, str) and native.strip(), f"{identity}: empty native question text")
    freeze = item.get("question_only_staticization_freeze")
    require(isinstance(freeze, dict), f"{identity}: missing staticization freeze record")
    require(
        freeze.get("question_text_staticized_sha256")
        == sha256_bytes(staticized.encode("utf-8")),
        f"{identity}: staticized text freeze SHA mismatch",
    )
    require(
        freeze.get("question_text_native_sha256") == sha256_bytes(native.encode("utf-8")),
        f"{identity}: native question text freeze SHA mismatch",
    )


def image_asset_map(item: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    assets = item.get("question_image_assets")
    if assets is None and isinstance(item.get("visual_dependency"), dict):
        assets = item["visual_dependency"].get("assets")
    result: dict[str, dict[str, Any]] = {}
    for asset in assets or []:
        if isinstance(asset, dict) and isinstance(asset.get("path"), str):
            result[str(asset["path"])] = asset
    return result


def validate_official_question_assets(
    workspace: Path, items: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    checked_png: dict[str, dict[str, Any]] = {}
    checked_pdf: dict[str, str] = {}
    for item in items:
        item_id = str(item["item_id"])
        assets = image_asset_map(item)
        for path in question_image_paths(item, workspace):
            require(path.is_file() and not path.is_symlink(), f"{item_id}: missing/linked question PNG {path}")
            desc = descriptor(path)
            asset = assets.get(rel_to_workspace(workspace, path))
            if asset is None:
                asset = next(
                    (
                        value
                        for key, value in assets.items()
                        if abs_from_workspace(workspace, key).resolve() == path.resolve()
                    ),
                    None,
                )
            require(asset is not None, f"{item_id}: question PNG lacks source metadata: {path}")
            declared_sha = asset.get("sha256") or asset.get("image_sha256")
            require(declared_sha == desc["sha256"], f"{item_id}: PNG metadata SHA mismatch: {path}")
            require(asset.get("mime_type") in (None, "image/png"), f"{item_id}: PNG MIME mismatch")
            require(asset.get("contains_solution") is False, f"{item_id}: question PNG not solution-free")
            if "contains_solution_or_rubric" in asset:
                require(
                    asset.get("contains_solution_or_rubric") is False,
                    f"{item_id}: question PNG may contain solution/rubric",
                )
            source_value = asset.get("source_pdf") or asset.get("source_pdf_path")
            source_sha = asset.get("source_pdf_sha256")
            source_page = asset.get("source_pdf_page", asset.get("source_page"))
            require(isinstance(source_value, str), f"{item_id}: question PNG lacks source PDF")
            source_path = abs_from_workspace(workspace, source_value)
            require(source_path.is_file() and not source_path.is_symlink(), f"{item_id}: invalid source PDF")
            require(source_path.suffix.lower() == ".pdf", f"{item_id}: source is not PDF")
            source_key = str(source_path.resolve())
            if source_key not in checked_pdf:
                checked_pdf[source_key] = sha256_file(source_path)
            actual_pdf_sha = checked_pdf[source_key]
            require(actual_pdf_sha == source_sha, f"{item_id}: source PDF SHA mismatch")
            require(type(source_page) is int and source_page > 0, f"{item_id}: invalid source page")
            checked_png.setdefault(desc["sha256"], desc)
    return {
        "unique_question_png_count": len(checked_png),
        "unique_question_png_sha256s": sorted(checked_png),
        "unique_source_pdf_count": len(checked_pdf),
        "source_pdf_set_sha256": canonical_digest(checked_pdf),
        "all_question_pngs_are_direct_nonlinked_source_bound_solution_free": True,
    }


def validate_attempt_images(
    attempt: Mapping[str, Any],
    expected: Sequence[Mapping[str, Any]],
    identity: str,
) -> tuple[int, list[str], bool]:
    request_images = attempt.get("request_images")
    descriptor_ledger_present = request_images is not None
    if descriptor_ledger_present:
        require(isinstance(request_images, list), f"{identity}: request_images is not a list")
        require(len(request_images) == len(expected), f"{identity}: request image count mismatch")
        for index, (observed, wanted) in enumerate(zip(request_images, expected), start=1):
            require(isinstance(observed, dict), f"{identity}: malformed request image descriptor")
            require(
                descriptor_core(observed) == descriptor_core(wanted),
                f"{identity}: request image descriptor {index} does not bind original PNG",
            )
    request = attempt.get("request")
    require(isinstance(request, dict), f"{identity}: request is not an object")
    blocks = image_blocks_in(request)
    require(len(blocks) == len(expected), f"{identity}: request image-block count mismatch")
    evidence: list[str] = []
    for index, (block, wanted) in enumerate(zip(blocks, expected), start=1):
        validate_data_image_url(block_url(block), wanted, f"{identity} image {index}")
        evidence.append(str(wanted["sha256"]))
    return len(blocks), evidence, descriptor_ledger_present


def audit_solver_payloads(
    workspace: Path,
    spec: RunSpec,
    run_root: Path,
    items: Sequence[dict[str, Any]],
    models: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    grouped = items_by_sequence(items)
    solver_terminal_groups = [
        [
            solver_terminal_path(run_root, str(model["route"]), item)
            for item in items
        ]
        for model in models
    ]
    preload_json_groups(solver_terminal_groups)
    solver_attempt_groups: list[list[Path]] = []
    for terminal_group in solver_terminal_groups:
        attempts: list[Path] = []
        for terminal_path in terminal_group:
            terminal = read_json(terminal_path)
            attempt_files = terminal.get("attempt_files") if isinstance(terminal, dict) else None
            if isinstance(attempt_files, list):
                attempts.extend(run_root / str(value) for value in attempt_files)
        solver_attempt_groups.append(attempts)
    preload_json_groups(solver_attempt_groups)
    route_rows: list[dict[str, Any]] = []
    for model in models:
        route = str(model["route"])
        mode = str(model["input_mode"])
        terminal_count = 0
        attempt_count = 0
        image_block_count = 0
        request_image_descriptor_ledger_count = 0
        evidence_lines: list[str] = []
        prompt_binding_lines: list[str] = []
        solver_route_root = run_root / "solver" / safe_name(route)
        solver_root_is_symlink = solver_route_root.is_symlink()
        symlink_target = (
            str(solver_route_root.resolve()) if solver_root_is_symlink else None
        )
        for sequence_id, sequence_items in grouped.items():
            cumulative: list[Path] = []
            seen: set[Path] = set()
            for item in sequence_items:
                current = question_image_paths(item, workspace) if mode == "vision" else []
                for image_path in current:
                    resolved = image_path.resolve()
                    if resolved not in seen:
                        cumulative.append(image_path)
                        seen.add(resolved)
                expected_images = [descriptor(path) for path in cumulative]
                expected_current = [descriptor(path) for path in current]
                path = solver_terminal_path(run_root, route, item)
                require(path.is_file(), f"missing solver terminal: {path}")
                terminal = read_json(path)
                identity = f"{spec.key}::{route}::{item['item_id']}"
                for key, expected_value in {
                    "terminal": True,
                    "route": route,
                    "input_mode": mode,
                    "benchmark_id": spec.benchmark_id,
                    "sequence_id": sequence_id,
                    "item_id": item["item_id"],
                    "turn_index": item["turn_index"],
                }.items():
                    require(terminal.get(key) == expected_value, f"{identity}: solver {key} mismatch")
                expected_prompt = solver_prompt(item, mode)
                expected_prompt_sha = sha256_bytes(expected_prompt.encode("utf-8"))
                require(
                    terminal.get("question_text_sha256") == expected_prompt_sha,
                    f"{identity}: solver prompt is not bound to frozen {mode} field",
                )
                require(
                    [descriptor_core(value) for value in terminal.get("question_images", [])]
                    == [descriptor_core(value) for value in expected_images],
                    f"{identity}: cumulative original-image descriptor mismatch",
                )
                require(
                    [descriptor_core(value) for value in terminal.get("current_question_images", [])]
                    == [descriptor_core(value) for value in expected_current],
                    f"{identity}: current original-image descriptor mismatch",
                )
                declared_attempt_count = terminal.get("attempt_count")
                attempt_files = terminal.get("attempt_files")
                require(
                    type(declared_attempt_count) is int
                    and 1 <= declared_attempt_count <= 7
                    and isinstance(attempt_files, list)
                    and len(attempt_files) == declared_attempt_count,
                    f"{identity}: invalid solver attempt ledger",
                )
                for ordinal, relative_attempt in enumerate(attempt_files, start=1):
                    attempt_path = run_root / str(relative_attempt)
                    require(attempt_path.is_file(), f"{identity}: missing solver attempt {attempt_path}")
                    attempt = read_json(attempt_path)
                    request = attempt.get("request")
                    require(isinstance(request, dict), f"{identity}: malformed solver request")
                    require(request.get("model") == route, f"{identity}: solver request route mismatch")
                    request_strings = list(strings_in(request))
                    require(
                        any(value.startswith(expected_prompt) for value in request_strings),
                        f"{identity}: frozen solver prompt not found in wire request",
                    )
                    block_count, hashes, descriptor_ledger_present = validate_attempt_images(
                        attempt,
                        expected_images,
                        f"{identity} solver attempt {ordinal}",
                    )
                    request_image_descriptor_ledger_count += int(
                        descriptor_ledger_present
                    )
                    if mode == "text_staticized":
                        require(block_count == 0, f"{identity}: text route contains an image block")
                    else:
                        require(
                            block_count == len(expected_images) and block_count > 0,
                            f"{identity}: vision route lacks direct original PNG blocks",
                        )
                    attempt_count += 1
                    image_block_count += block_count
                    for image_ordinal, image_hash in enumerate(hashes, start=1):
                        evidence_lines.append(
                            f"{relative_attempt}\t{image_ordinal}\t{image_hash}"
                        )
                terminal_count += 1
                prompt_binding_lines.append(f"{item['item_id']}\t{expected_prompt_sha}")
        route_rows.append(
            {
                "route": route,
                "display_name": model.get("display_name") or route,
                "input_mode": mode,
                "solver_route_root_is_symlink": solver_root_is_symlink,
                "solver_route_root_symlink_target": symlink_target,
                "terminal_count": terminal_count,
                "attempt_count": attempt_count,
                "request_image_descriptor_ledger_attempt_count": (
                    request_image_descriptor_ledger_count
                ),
                "legacy_wire_only_image_evidence_attempt_count": (
                    attempt_count - request_image_descriptor_ledger_count
                ),
                "wire_image_block_count": image_block_count,
                "wire_image_evidence_sha256": sha256_bytes(
                    ("\n".join(evidence_lines) + ("\n" if evidence_lines else "")).encode("utf-8")
                ),
                "frozen_prompt_binding_sha256": sha256_bytes(
                    ("\n".join(prompt_binding_lines) + "\n").encode("utf-8")
                ),
                "direct_original_png_payload_pass": (
                    mode == "vision" and image_block_count > 0
                ),
                "zero_image_blocks_and_staticized_prompt_pass": (
                    mode == "text_staticized" and image_block_count == 0
                ),
                "pass": True,
            }
        )
    return {
        "route_count": len(route_rows),
        "vision_route_count": sum(row["input_mode"] == "vision" for row in route_rows),
        "text_staticized_route_count": sum(
            row["input_mode"] == "text_staticized" for row in route_rows
        ),
        "vision_all_direct_original_png_payloads": all(
            row["direct_original_png_payload_pass"]
            for row in route_rows
            if row["input_mode"] == "vision"
        ),
        "text_all_zero_image_blocks_and_frozen_staticized_prompts": all(
            row["zero_image_blocks_and_staticized_prompt_pass"]
            for row in route_rows
            if row["input_mode"] == "text_staticized"
        ),
        "routes": route_rows,
    }


def audit_judge_payloads(
    spec: RunSpec,
    run_root: Path,
    items: Sequence[dict[str, Any]],
    models: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    rubric_map = OrderedDict(
        (str(item["item_id"]), rubric_for_prompt(item)) for item in items
    )
    rubric_set_sha = canonical_digest(rubric_map)
    terminal_count = 0
    attempt_count = 0
    deterministic_zero_count = 0
    terminal_schemas: set[str] = set()
    fingerprint_contracts: set[str] = set()
    retry_contracts: set[str] = set()
    evidence_lines: list[str] = []
    judge_terminal_groups = [
        [judge_terminal_path(run_root, str(model["route"]), item) for item in items]
        for model in models
    ]
    preload_json_groups(judge_terminal_groups)
    judge_attempt_groups: list[list[Path]] = []
    for terminal_group in judge_terminal_groups:
        attempts: list[Path] = []
        for terminal_path in terminal_group:
            terminal = read_json(terminal_path)
            attempt_files = terminal.get("attempt_files") if isinstance(terminal, dict) else None
            if isinstance(attempt_files, list):
                attempts.extend(run_root / str(value) for value in attempt_files)
        judge_attempt_groups.append(attempts)
    preload_json_groups(judge_attempt_groups)
    for model in models:
        route = str(model["route"])
        for item in items:
            path = judge_terminal_path(run_root, route, item)
            require(path.is_file(), f"missing judge terminal: {path}")
            terminal = read_json(path)
            identity = f"{spec.key}::{route}::{item['item_id']}"
            require(terminal.get("terminal") is True, f"{identity}: non-terminal judge record")
            require(terminal.get("solver_route") == route, f"{identity}: judge solver route mismatch")
            require(terminal.get("judge_route") == JUDGE_ROUTE, f"{identity}: judge route drift")
            require(terminal.get("benchmark_id") == spec.benchmark_id, f"{identity}: benchmark drift")
            require(terminal.get("sequence_id") == item["sequence_id"], f"{identity}: sequence drift")
            require(terminal.get("item_id") == item["item_id"], f"{identity}: item drift")
            assert_close(terminal.get("max_score"), float(item["max_score"]), f"{identity} max")
            score = float(terminal.get("score"))
            require(
                -FLOAT_TOLERANCE <= score <= float(item["max_score"]) + FLOAT_TOLERANCE,
                f"{identity}: out-of-range judge score",
            )
            terminal_schemas.add(str(terminal.get("schema_version") or "missing"))
            fingerprint_contracts.add(
                str(terminal.get("judge_input_fingerprint_contract") or "legacy_implicit")
            )
            expected_contract = normalized_criterion_contract(item)
            parsed = terminal.get("parsed")
            require(isinstance(parsed, dict), f"{identity}: missing parsed judge object")
            assert_close(parsed.get("score"), score, f"{identity} parsed score")
            assert_close(parsed.get("max_score"), float(item["max_score"]), f"{identity} parsed max")
            declared_attempt_count = terminal.get("attempt_count")
            # Judge v1 deterministic-zero terminals predate the explicit
            # empty attempt-files ledger.  Absence is equivalent to [] only
            # when the declared count is exactly zero; nonzero records remain
            # fail-closed.
            attempt_files = terminal.get("attempt_files", [])
            require(
                type(declared_attempt_count) is int
                and 0 <= declared_attempt_count <= 7
                and isinstance(attempt_files, list)
                and len(attempt_files) == declared_attempt_count,
                f"{identity}: invalid judge attempt ledger",
            )
            if declared_attempt_count == 0:
                deterministic_zero_count += 1
                require(score == 0.0, f"{identity}: attempt-free judge result is not zero")
                require(parsed.get("criteria") == [], f"{identity}: deterministic zero has criteria")
            else:
                parsed_criteria = parsed.get("criteria")
                require(isinstance(parsed_criteria, list), f"{identity}: judge criteria missing")
                observed_contract = [
                    {
                        "criterion_id": str(criterion.get("criterion_id") or ""),
                        "max_score": float(criterion.get("max_score")),
                    }
                    for criterion in parsed_criteria
                ]
                require(
                    observed_contract == expected_contract,
                    f"{identity}: parsed judge criteria drift from frozen rubric",
                )
            rendered_rubric = prompt_text(rubric_for_prompt(item))
            rendered_contract = prompt_text(
                [
                    {
                        "criterion_id": criterion["criterion_id"],
                        # Preserve the frozen JSON number representation used
                        # by the production payload builder (for example 1.0
                        # remains 1.0) while the parsed-result comparison above
                        # deliberately normalizes numbers to float.
                        "max_score": criterion["max_score"],
                    }
                    for criterion in normalized_rubric_criteria(
                        grading_rubric_for(item)
                    )
                ]
            )
            for ordinal, relative_attempt in enumerate(attempt_files, start=1):
                attempt_path = run_root / str(relative_attempt)
                require(attempt_path.is_file(), f"{identity}: missing judge attempt {attempt_path}")
                attempt = read_json(attempt_path)
                request = attempt.get("request")
                require(isinstance(request, dict), f"{identity}: malformed judge request")
                require(request.get("model") == JUDGE_ROUTE, f"{identity}: judge wire route drift")
                text = "\n".join(strings_in(request))
                require(rendered_rubric in text, f"{identity}: frozen rubric absent from judge wire")
                require(rendered_contract in text, f"{identity}: criterion contract absent from judge wire")
                retry_contracts.add(str(attempt.get("retry_contract") or "legacy_implicit"))
                evidence_lines.append(
                    f"{relative_attempt}\t{canonical_digest(rubric_for_prompt(item))}\t{JUDGE_ROUTE}"
                )
                attempt_count += 1
            terminal_count += 1
    return {
        "fixed_judge_route": JUDGE_ROUTE,
        "terminal_count": terminal_count,
        "attempt_count": attempt_count,
        "deterministic_zero_count": deterministic_zero_count,
        "judge_terminal_schemas": sorted(terminal_schemas),
        "judge_input_fingerprint_contracts": sorted(fingerprint_contracts),
        "judge_retry_contracts": sorted(retry_contracts),
        "frozen_rubric_item_count": len(rubric_map),
        "frozen_rubric_set_sha256": rubric_set_sha,
        "judge_wire_rubric_route_evidence_sha256": sha256_bytes(
            ("\n".join(evidence_lines) + ("\n" if evidence_lines else "")).encode("utf-8")
        ),
        "all_judge_terminals_and_wire_requests_use_fixed_route": True,
        "all_non_deterministic_judgments_bind_exact_frozen_rubrics": True,
        "pass": True,
    }


def source_seal_record(workspace: Path, spec: RunSpec, run_root: Path) -> dict[str, Any]:
    candidates: list[Path] = []
    if spec.seal_audit_rel:
        candidates.append(workspace / spec.seal_audit_rel)
    else:
        audits_root = workspace / "experiments/2026_reval/audits"
        needle = spec.run_rel
        # The NB seal name is assigned post-run.  Restrict discovery to its
        # exact judge-v8 audit family instead of traversing unrelated audits.
        for path in sorted(
            audits_root.glob("nbpho2026_table6_judge_v8*20260829.json")
        ):
            try:
                if needle in read_exact_regular_bytes(path).decode("utf-8"):
                    candidates.append(path)
            except (OSError, UnicodeDecodeError):
                continue
    records: list[dict[str, Any]] = []
    for path in candidates:
        if not path.is_file():
            continue
        payload = read_json(path)
        verdict = payload.get("verdict") if isinstance(payload, dict) else None
        records.append(
            {
                "path": rel_to_workspace(workspace, path),
                "sha256": sha256_file(path),
                "schema_version": payload.get("schema_version") if isinstance(payload, dict) else None,
                "verdict": verdict,
            }
        )
    endorsed = any(record.get("verdict") == "PASS" for record in records)
    return {
        "independent_audit_records": records,
        "independent_pass_endorsement_present": endorsed,
        "run_root": rel_to_workspace(workspace, run_root),
    }


def audit_one_complete_run(
    workspace: Path,
    spec: RunSpec,
    run_root: Path,
    manifest: Mapping[str, Any],
    items_path: Path,
    items: Sequence[dict[str, Any]],
    models_path: Path,
    models: Sequence[dict[str, Any]],
    aggregate_path: Path,
) -> dict[str, Any]:
    for item in items:
        validate_staticization_freeze(item, f"{spec.key}::{item['item_id']}")
    assets = validate_official_question_assets(workspace, items)
    solver = audit_solver_payloads(workspace, spec, run_root, items, models)
    judge = audit_judge_payloads(spec, run_root, items, models)
    seal = source_seal_record(workspace, spec, run_root)
    # NBPhO may receive its independent countersign immediately after this
    # analysis.  The other three roots must already be independently endorsed.
    if spec.key != "nbpho2026":
        require(
            seal["independent_pass_endorsement_present"],
            f"{spec.key}: no independent PASS seal/countersign",
        )
    pipeline_record: dict[str, Any] | None = None
    provenance_path = run_root / "run_provenance.json"
    if provenance_path.is_file():
        provenance = read_json(provenance_path)
        pipeline_record = provenance.get("pipeline")
    result = {
        "competition": spec.display_name,
        "competition_key": spec.key,
        "benchmark_id": spec.benchmark_id,
        "run_root": rel_to_workspace(workspace, run_root),
        "generation": spec.generation,
        "run_manifest_schema": manifest.get("schema_version"),
        "run_manifest_sha256": sha256_file(run_root / "run_manifest.json"),
        "items_path": rel_to_workspace(workspace, items_path),
        "items_sha256": sha256_file(items_path),
        "model_manifest_path": rel_to_workspace(workspace, models_path),
        "model_manifest_sha256": sha256_file(models_path),
        "aggregate_path": rel_to_workspace(workspace, aggregate_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "aggregate_complete": True,
        "item_count": len(items),
        "sequence_cluster_count": len(sequence_order(items)),
        "table_score_raw_max": spec.raw_max,
        "pipeline_provenance": pipeline_record,
        "official_question_assets": assets,
        "solver_payload_audit": solver,
        "judge_payload_audit": judge,
        "seal_evidence": seal,
        "requested_protocol_invariants_pass": (
            solver["vision_all_direct_original_png_payloads"]
            and solver["text_all_zero_image_blocks_and_frozen_staticized_prompts"]
            and judge["all_judge_terminals_and_wire_requests_use_fixed_route"]
            and judge["all_non_deterministic_judgments_bind_exact_frozen_rubrics"]
        ),
    }
    require(result["requested_protocol_invariants_pass"], f"{spec.key}: protocol audit failed")
    return result


def audit_one_from_sealed_countersigns(
    workspace: Path,
    spec: RunSpec,
    run_root: Path,
    manifest: Mapping[str, Any],
    items_path: Path,
    items: Sequence[dict[str, Any]],
    models_path: Path,
    models: Sequence[dict[str, Any]],
    aggregate_path: Path,
) -> dict[str, Any]:
    """Compose already-exhaustive independent PASS audits, fail closed.

    This mode avoids rehydrating thousands of sealed FileProvider records.  It
    does not weaken the evidence scope: the referenced independent auditors
    already replayed every stored wire/terminal and sealed their results.  This
    function verifies their scope, route/modality sets, source hashes, required
    assertions, and current aggregate binding before countersigning them into
    the four-run compatibility report.
    """

    expected_routes = [str(model["route"]) for model in models]
    expected_vision = [
        str(model["route"]) for model in models if model["input_mode"] == "vision"
    ]
    expected_text = [
        str(model["route"])
        for model in models
        if model["input_mode"] == "text_staticized"
    ]
    source_config = SEALED_PROTOCOL_AUDITS[spec.key]
    source_payloads: dict[str, tuple[Path, dict[str, Any]]] = {}
    source_records: list[dict[str, Any]] = []
    for role, relative in source_config.items():
        path = workspace / relative
        require(path.is_file(), f"{spec.key}: missing sealed {role} evidence {path}")
        payload = read_json(path)
        require(isinstance(payload, dict), f"{spec.key}: malformed {role} evidence")
        source_payloads[role] = (path, payload)
        source_records.append(
            {
                "role": role,
                "path": relative,
                "sha256": sha256_file(path),
                "schema_version": payload.get("schema_version"),
                "verdict": payload.get("verdict"),
            }
        )
    main_path, main = source_payloads["main"]
    require(main.get("verdict") == "PASS", f"{spec.key}: main independent audit is not PASS")
    scope = main.get("scope")
    require(isinstance(scope, dict), f"{spec.key}: main audit lacks scope")
    scoped_root = str(scope.get("run_root") or "")
    require(scoped_root.endswith(spec.run_rel), f"{spec.key}: independent audit run scope drift")
    main_checks = main.get("checks")
    require(isinstance(main_checks, dict), f"{spec.key}: independent audit lacks checks")
    require(not main.get("blockers"), f"{spec.key}: independent audit has blockers")

    hashes = main.get("hashes") if isinstance(main.get("hashes"), dict) else {}
    declared_models_sha = hashes.get("models_manifest_sha256")
    if spec.key == "nbpho2026":
        declared_models_sha = (
            main.get("evidence", {})
            .get("source_and_rubric", {})
            .get("source_hashes", {})
            .get("models")
        )
    if declared_models_sha is not None:
        require(
            declared_models_sha == sha256_file(models_path),
            f"{spec.key}: sealed audit model hash drift",
        )
    declared_items_sha = hashes.get("frozen_items_sha256")
    if spec.key == "nbpho2026":
        declared_items_sha = (
            main.get("evidence", {})
            .get("source_and_rubric", {})
            .get("source_hashes", {})
            .get("items")
        )
    if declared_items_sha is not None:
        require(
            declared_items_sha == sha256_file(items_path),
            f"{spec.key}: sealed audit items hash drift",
        )

    if spec.key in {"ipho2026", "icho2026"}:
        require(scope.get("routes") == expected_routes, f"{spec.key}: audited route order drift")
        require(scope.get("vision_routes") == expected_vision, f"{spec.key}: audited vision routes drift")
        require(scope.get("text_routes") == expected_text, f"{spec.key}: audited text routes drift")
        require(
            scope.get("calibrated_judge_route") == JUDGE_ROUTE,
            f"{spec.key}: audited judge route drift",
        )
        text_path, text_audit = source_payloads["text"]
        require(text_audit.get("verdict") == "PASS", f"{spec.key}: text-wire audit not PASS")
        require(
            text_audit.get("scope", {}).get("routes") == expected_text,
            f"{spec.key}: text-wire route scope drift",
        )
        text_checks = text_audit.get("checks", {})
        if spec.key == "ipho2026":
            required_main = [
                "all_request_shas_and_image_proofs_match",
                "real_wires_reconstructed",
                "vision_reuse_8x23",
                "all_raw_judge_json_revalidated_with_decimal_and_evidence_contract",
            ]
            required_text = [
                "all_requests_exactly_reconstructed_from_frozen_staticized_prompts",
                "all_requests_have_zero_image_markers",
                "coverage_6x23",
            ]
        else:
            required_main = [
                "all_question_and_solution_png_bytes_bound",
                "all_real_wires_reconstructed",
                "vision_reuse_8x68",
                "raw_parsed_decimal_evidence_contract_replayed",
            ]
            required_text = [
                "all_multiturn_requests_reconstructed_and_fingerprinted",
                "frozen_staticized_prompts_and_full_prompt_sha_verified",
                "zero_request_image_url_or_data_image_and_empty_image_lists",
                "coverage_6x68",
            ]
        require(all(main_checks.get(key) is True for key in required_main), f"{spec.key}: main audit invariant missing")
        require(all(text_checks.get(key) is True for key in required_text), f"{spec.key}: text audit invariant missing")
        declared_text_sha = hashes.get("prior_text_audit_json_sha256")
        require(
            declared_text_sha == sha256_file(text_path),
            f"{spec.key}: main audit does not bind text audit",
        )
        vision_path, vision_provenance = source_payloads["vision_provenance"]
        require(
            hashes.get("vision_reuse_provenance_sha256") == sha256_file(vision_path),
            f"{spec.key}: main audit does not bind vision provenance",
        )
        require(
            [row.get("route") for row in vision_provenance.get("routes", [])]
            == expected_vision,
            f"{spec.key}: vision provenance route drift",
        )
        production_aggregate_sha = main.get("aggregate", {}).get(
            "production_aggregate_sha256"
        )
        require(
            production_aggregate_sha == sha256_file(aggregate_path),
            f"{spec.key}: independent audit aggregate hash drift",
        )
    elif spec.key == "usapho2026":
        required = [
            "exact_14_routes_x_34_items",
            "aggregate_476_exact",
            "vision_direct_official_png_bytes",
            "text_zero_image_blocks",
            "score_recomputation",
            "provenance_sha_bindings",
        ]
        require(all(main_checks.get(key) is True for key in required), "USAPhO sealed invariant missing")
        visual = main.get("solver_visual_wire_proof", {})
        text_proof = main.get("solver_text_wire_proof", {})
        require(list(visual.get("per_vision_route", {})) == sorted(expected_vision), "USAPhO vision route scope drift")
        require(list(text_proof.get("per_text_route", {})) == sorted(expected_text), "USAPhO text route scope drift")
        require(visual.get("wire_payload_rebuilt_from_official_bytes") is True, "USAPhO vision bytes not bound")
        require(text_proof.get("all_text_wires_zero_image") is True, "USAPhO text images present")
        require(main.get("judge_validation", {}).get("judge_route") == JUDGE_ROUTE, "USAPhO judge route drift")
        require(
            main.get("aggregate_recomputation", {}).get("aggregate_sha256")
            == sha256_file(aggregate_path),
            "USAPhO aggregate seal drift",
        )
    else:
        required = [
            "exact_14_routes_x_28_items",
            "aggregate_complete_392_rows_exact_decimal_over_72",
            "judge_exact_392_terminal_coverage",
            "solver_text_every_wire_zero_images",
            "solver_vision_every_wire_direct_official_png",
            "judge_attempt_ledgers_contiguous_max7_no_attempt8",
            "solver_and_judge_write_once_seals_exact",
            "versioned_decimal_publication_sidecar_392_hashes_pass",
        ]
        require(all(main_checks.get(key) is True for key in required), "NBPhO sealed invariant missing")
        per_route = main.get("evidence", {}).get("solver", {}).get("per_route", {})
        require(list(per_route) == sorted(expected_routes), "NBPhO audited route set drift")
        for model in models:
            require(
                per_route[str(model["route"])].get("input_mode") == model["input_mode"],
                f"NBPhO modality drift for {model['route']}",
            )
        provenance = read_json(run_root / "run_provenance.json")
        require(provenance.get("judge", {}).get("route") == JUDGE_ROUTE, "NBPhO judge route drift")
        sidecar_path, sidecar = source_payloads["decimal_sidecar"]
        require(
            main.get("hashes", {}).get("decimal_publication_sidecar_sha256")
            == sha256_file(sidecar_path),
            "NBPhO audit does not bind Decimal sidecar",
        )
        require(
            sidecar.get("formal_run", {}).get("aggregate_sha256")
            == sha256_file(aggregate_path),
            "NBPhO Decimal sidecar aggregate drift",
        )

    main_sha = sha256_file(main_path)
    route_rows: list[dict[str, Any]] = []
    for model in models:
        route = str(model["route"])
        mode = str(model["input_mode"])
        route_rows.append(
            {
                "route": route,
                "display_name": model.get("display_name") or route,
                "input_mode": mode,
                "solver_route_root_is_symlink": (run_root / "solver" / safe_name(route)).is_symlink(),
                "solver_route_root_symlink_target": (
                    str((run_root / "solver" / safe_name(route)).resolve())
                    if (run_root / "solver" / safe_name(route)).is_symlink()
                    else None
                ),
                "terminal_count": len(items),
                "attempt_count": None,
                "wire_image_block_count": None,
                "wire_image_evidence_sha256": main_sha if mode == "vision" else None,
                "frozen_prompt_binding_sha256": main_sha if mode == "text_staticized" else None,
                "direct_original_png_payload_pass": mode == "vision",
                "zero_image_blocks_and_staticized_prompt_pass": mode == "text_staticized",
                "evidence_method": "sealed_independent_exhaustive_audit",
                "pass": True,
            }
        )
    solver = {
        "route_count": 14,
        "vision_route_count": 8,
        "text_staticized_route_count": 6,
        "vision_all_direct_original_png_payloads": True,
        "text_all_zero_image_blocks_and_frozen_staticized_prompts": True,
        "evidence_method": "sealed_independent_exhaustive_audit_countersign",
        "routes": route_rows,
    }
    judge = {
        "fixed_judge_route": JUDGE_ROUTE,
        "terminal_count": len(items) * len(models),
        "attempt_count": None,
        "frozen_rubric_set_sha256": canonical_digest(
            OrderedDict((str(item["item_id"]), rubric_for_prompt(item)) for item in items)
        ),
        "all_judge_terminals_and_wire_requests_use_fixed_route": True,
        "all_non_deterministic_judgments_bind_exact_frozen_rubrics": True,
        "evidence_method": "sealed_independent_exhaustive_audit_countersign",
        "pass": True,
    }
    seal = source_seal_record(workspace, spec, run_root)
    require(seal["independent_pass_endorsement_present"], f"{spec.key}: independent seal absent")
    return {
        "competition": spec.display_name,
        "competition_key": spec.key,
        "benchmark_id": spec.benchmark_id,
        "run_root": spec.run_rel,
        "generation": spec.generation,
        "run_manifest_schema": manifest.get("schema_version"),
        "run_manifest_sha256": sha256_file(run_root / "run_manifest.json"),
        "items_path": rel_to_workspace(workspace, items_path),
        "items_sha256": sha256_file(items_path),
        "model_manifest_path": rel_to_workspace(workspace, models_path),
        "model_manifest_sha256": sha256_file(models_path),
        "aggregate_path": rel_to_workspace(workspace, aggregate_path),
        "aggregate_sha256": sha256_file(aggregate_path),
        "aggregate_complete": True,
        "item_count": len(items),
        "sequence_cluster_count": len(sequence_order(items)),
        "table_score_raw_max": spec.raw_max,
        "protocol_evidence_mode": "sealed_independent_exhaustive_audit_countersign",
        "independent_protocol_evidence": source_records,
        "solver_payload_audit": solver,
        "judge_payload_audit": judge,
        "seal_evidence": seal,
        "requested_protocol_invariants_pass": True,
    }


class SplitMix64:
    """Small specified PRNG so bootstrap draws do not depend on Python's RNG API."""

    MASK = (1 << 64) - 1

    def __init__(self, seed: int):
        self.state = seed & self.MASK

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & self.MASK
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & self.MASK
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & self.MASK
        return (value ^ (value >> 31)) & self.MASK

    def randbelow(self, bound: int) -> int:
        require(type(bound) is int and bound > 0, f"invalid PRNG bound {bound!r}")
        limit = (1 << 64) - ((1 << 64) % bound)
        while True:
            value = self.next_u64()
            if value < limit:
                return value % bound


def quantile_type7(sorted_values: Sequence[float], probability: float) -> float:
    require(sorted_values, "cannot calculate a quantile of an empty sample")
    require(0.0 <= probability <= 1.0, "quantile probability outside [0,1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    fraction = position - lower
    return float(sorted_values[lower]) * (1.0 - fraction) + float(
        sorted_values[upper]
    ) * fraction


def bootstrap_macro_ci(
    complete_specs: Sequence[RunSpec],
    models: Sequence[dict[str, Any]],
    clusters: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    *,
    seed: int,
    resamples: int,
    target_is_complete: bool,
) -> dict[str, Any]:
    require(resamples == 10_000, "publication bootstrap requires exactly 10,000 resamples")
    routes = [str(model["route"]) for model in models]
    rng = SplitMix64(seed)
    samples: dict[str, list[float]] = {route: [] for route in routes}
    cluster_orders: dict[str, list[str]] = {}
    for spec in complete_specs:
        one_route = clusters[spec.key][routes[0]]
        cluster_orders[spec.key] = list(one_route)
    for _ in range(resamples):
        draws: dict[str, list[int]] = {}
        for spec in complete_specs:
            count = len(cluster_orders[spec.key])
            draws[spec.key] = [rng.randbelow(count) for _ in range(count)]
        for route in routes:
            percentages: list[float] = []
            for spec in complete_specs:
                order = cluster_orders[spec.key]
                route_clusters = clusters[spec.key][route]
                numerator = 0.0
                denominator = 0.0
                for index in draws[spec.key]:
                    cluster = route_clusters[order[index]]
                    numerator += float(cluster["competition_score"])
                    denominator += float(cluster["competition_max"])
                percentages.append(numerator / denominator * 100.0)
            samples[route].append(sum(percentages) / len(percentages))
    rows: list[dict[str, Any]] = []
    for model in models:
        route = str(model["route"])
        values = sorted(samples[route])
        point_percentages = []
        for spec in complete_specs:
            route_clusters = clusters[spec.key][route]
            point_percentages.append(
                sum(value["competition_score"] for value in route_clusters.values())
                / sum(value["competition_max"] for value in route_clusters.values())
                * 100.0
            )
        rows.append(
            {
                "table6_order": model["table6_order"],
                "display_name": model.get("display_name") or route,
                "route": route,
                "input_mode": model["input_mode"],
                "competition_count": len(complete_specs),
                "point_macro_mean_percent": sum(point_percentages) / len(point_percentages),
                "bootstrap_mean_percent": sum(values) / len(values),
                "ci95_lower_percent": quantile_type7(values, 0.025),
                "ci95_upper_percent": quantile_type7(values, 0.975),
            }
        )
    return {
        "schema_version": BOOTSTRAP_SCHEMA,
        "status": "complete_four_competition_target" if target_is_complete else "validation_only_partial_scope",
        "publication_eligible": target_is_complete,
        "competition_keys": [spec.key for spec in complete_specs],
        "competition_cluster_counts": {
            spec.key: len(cluster_orders[spec.key]) for spec in complete_specs
        },
        "resampling_unit": (
            "whole question/problem sequence cluster within each competition; "
            "all item scores in a selected sequence move together"
        ),
        "within_competition_estimator": (
            "sum of resampled cluster competition-score contributions divided by "
            "sum of their resampled competition-scale denominators"
        ),
        "across_competition_estimator": (
            "unweighted arithmetic macro mean of competition-normalized percentages"
        ),
        "paired_across_models": True,
        "resamples": resamples,
        "seed": seed,
        "prng": "SplitMix64 with rejection-sampled randbelow",
        "interval": "two-sided 95% empirical percentile interval; Hyndman-Fan type 7",
        "uncertainty_scope": (
            "item/question-sequence sampling only; it does not estimate generation "
            "variance or judge variance because each model-item generation and judgment "
            "is a single frozen realization"
        ),
        "rows": rows,
    }


def build_score_rows(
    specs: Sequence[RunSpec],
    models: Sequence[dict[str, Any]],
    aggregates: Mapping[str, Mapping[str, Any] | None],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        route = str(model["route"])
        competitions: OrderedDict[str, dict[str, Any]] = OrderedDict()
        percentages: list[float] = []
        missing: list[str] = []
        for spec in specs:
            aggregate = aggregates.get(spec.key)
            benchmark: Mapping[str, Any] | None = None
            if aggregate is not None:
                benchmark = aggregate["aggregates"][route][spec.benchmark_id]
            complete = bool(benchmark and benchmark.get("complete") and benchmark.get("score") is not None)
            score = float(benchmark["score"]) if complete and benchmark is not None else None
            percent = score / spec.raw_max * 100.0 if score is not None else None
            competitions[spec.key] = {
                "display_name": spec.display_name,
                "score": score,
                "raw_max": spec.raw_max,
                "normalized_percent": percent,
                "complete": complete,
                "missing": not complete,
            }
            if percent is None:
                missing.append(spec.key)
            else:
                percentages.append(percent)
        rows.append(
            {
                "table6_order": model["table6_order"],
                "display_name": model.get("display_name") or route,
                "route": route,
                "input_mode": model["input_mode"],
                "competitions": competitions,
                "complete_competition_count": len(percentages),
                "missing_competitions": missing,
                "macro_mean_percent": (
                    sum(percentages) / len(specs) if len(percentages) == len(specs) else None
                ),
            }
        )
    return rows


def protocol_compatibility_summary(run_audits: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    complete = [row for row in run_audits if row.get("status") == "complete"]
    model_hashes = {row.get("model_manifest_sha256") for row in complete}
    judge_routes = {
        row["judge_payload_audit"]["fixed_judge_route"]
        for row in complete
        if isinstance(row.get("judge_payload_audit"), dict)
    }
    requested_pass = all(row.get("requested_protocol_invariants_pass") for row in complete)
    seals_pass = all(
        isinstance(row.get("seal_evidence"), dict)
        and row["seal_evidence"].get("independent_pass_endorsement_present") is True
        for row in complete
    )
    if len(complete) == len(RUN_SPECS) and requested_pass and seals_pass:
        overall_status = "PASS_all_four_complete_and_independently_sealed"
    elif len(complete) == len(RUN_SPECS) and requested_pass:
        overall_status = "HOLD_complete_protocol_pass_waiting_for_independent_seal"
    else:
        overall_status = "PARTIAL_waiting_for_nbpho"
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "overall_status": overall_status,
        "table_comparability_on_requested_invariants": requested_pass,
        "all_complete_runs_independently_sealed": seals_pass,
        "byte_identical_protocol_across_generations": False,
        "same_frozen_14_model_roster_and_modality_assignments": len(model_hashes) == 1,
        "fixed_judge_route_across_complete_runs": judge_routes == {JUDGE_ROUTE},
        "direct_original_png_payloads_for_every_vision_route": all(
            row["solver_payload_audit"]["vision_all_direct_original_png_payloads"]
            for row in complete
        ),
        "zero_image_blocks_and_question_only_staticization_for_every_text_route": all(
            row["solver_payload_audit"][
                "text_all_zero_image_blocks_and_frozen_staticized_prompts"
            ]
            for row in complete
        ),
        "fixed_item_rubrics_bound_in_every_non_deterministic_judge_request": all(
            row["judge_payload_audit"][
                "all_non_deterministic_judgments_bind_exact_frozen_rubrics"
            ]
            for row in complete
        ),
        "generation_differences": [
            "IPhO/IChO are run-schema v1 roots with independently audited vision-tree reuse and fresh text-staticized routes.",
            "USAPhO is the sealed full-fresh judge-v5 generation; the earlier two-missing-row aggregate is excluded.",
            "NBPhO is the materialized full-fresh judge-v8 generation with stronger judge evidence/fingerprint validation.",
            "Pipeline, run-schema, fingerprint, retry, and evidence-validation contracts therefore differ by generation; compatibility is asserted only for the explicitly audited scientific input/scoring invariants, not byte identity.",
        ],
        "runs": list(run_audits),
    }


def json_ready_number(value: Any) -> Any:
    if isinstance(value, float):
        require(math.isfinite(value), "attempt to serialize a non-finite float")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_ready_number,
        )
        + "\n",
        encoding="utf-8",
    )


def score_csv_rows(score_rows: Sequence[Mapping[str, Any]]) -> tuple[list[str], list[dict[str, Any]]]:
    fields = ["table6_order", "display_name", "route", "input_mode"]
    for spec in RUN_SPECS:
        fields.extend(
            [
                f"{spec.key}_score",
                f"{spec.key}_raw_max",
                f"{spec.key}_normalized_percent",
                f"{spec.key}_status",
            ]
        )
    fields.extend(
        ["macro_mean_percent", "complete_competition_count", "missing_competitions"]
    )
    flat: list[dict[str, Any]] = []
    for row in score_rows:
        output: dict[str, Any] = {
            "table6_order": row["table6_order"],
            "display_name": row["display_name"],
            "route": row["route"],
            "input_mode": row["input_mode"],
        }
        for spec in RUN_SPECS:
            competition = row["competitions"][spec.key]
            output[f"{spec.key}_score"] = competition["score"]
            output[f"{spec.key}_raw_max"] = competition["raw_max"]
            output[f"{spec.key}_normalized_percent"] = competition["normalized_percent"]
            output[f"{spec.key}_status"] = "complete" if competition["complete"] else "missing"
        output["macro_mean_percent"] = row["macro_mean_percent"]
        output["complete_competition_count"] = row["complete_competition_count"]
        output["missing_competitions"] = ";".join(row["missing_competitions"])
        flat.append(output)
    return fields, flat


def write_csv(path: Path, fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: (
                        ""
                        if row.get(field) is None
                        else f"{row[field]:.12g}"
                        if isinstance(row.get(field), float)
                        else row.get(field)
                    )
                    for field in fields
                }
            )


def format_score(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2f}"


def format_percent(value: Any) -> str:
    return "—" if value is None else f"{float(value):.2f}%"


def score_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Fresh 14-model four-competition score table",
        "",
        f"Status: **{payload['status']}**.",
        "",
        "Scores are shown in competition score units with explicit denominators: "
        "IPhO /30, IChO theory /60, NBPhO /72, and USAPhO /150. The macro is "
        "the unweighted mean of the four competition-normalized percentages and is "
        "left missing unless all four scores are complete.",
        "",
        "| # | Model | Modality | IPhO /30 | IChO /60 | NBPhO /72 | USAPhO /150 | Macro | Missing |",
        "|---:|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in payload["rows"]:
        competitions = row["competitions"]
        missing = ", ".join(row["missing_competitions"]) or "none"
        lines.append(
            "| {order} | {name} | {mode} | {ipho} | {icho} | {nbpho} | {usapho} | {macro} | {missing} |".format(
                order=row["table6_order"],
                name=row["display_name"],
                mode=row["input_mode"],
                ipho=format_score(competitions["ipho2026"]["score"]),
                icho=format_score(competitions["icho2026"]["score"]),
                nbpho=format_score(competitions["nbpho2026"]["score"]),
                usapho=format_score(competitions["usapho2026"]["score"]),
                macro=format_percent(row["macro_mean_percent"]),
                missing=missing,
            )
        )
    lines.extend(
        [
            "",
            "IChO's /60 score is the official problem-weighted theory scale; the underlying "
            "frozen item-rubric maxima sum to 437 before the nine problem weights are applied.",
            "",
        ]
    )
    return "\n".join(lines)


def bootstrap_markdown(payload: Mapping[str, Any]) -> str:
    lines = [
        "# Sequence-cluster bootstrap for the normalized macro mean",
        "",
        f"Status: **{payload['status']}**; publication eligible: **{str(payload['publication_eligible']).lower()}**.",
        "",
        f"The calculation uses {payload['resamples']:,} paired resamples, seed "
        f"`{payload['seed']}`, and whole question/problem sequences as clusters within "
        "each competition. A sampled cluster carries all of its item scores together. "
        "The interval is a two-sided 95% percentile interval (type-7 interpolation).",
        "",
        "This estimates **item/question-sequence sampling uncertainty only**. It does not "
        "estimate model-generation variance or judge variance; every model-item answer and "
        "judgment is one frozen realization.",
        "",
        "| # | Model | Modality | Competitions | Macro | Bootstrap mean | 95% CI |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in payload["rows"]:
        lines.append(
            f"| {row['table6_order']} | {row['display_name']} | {row['input_mode']} | "
            f"{row['competition_count']} | {row['point_macro_mean_percent']:.2f}% | "
            f"{row['bootstrap_mean_percent']:.2f}% | "
            f"[{row['ci95_lower_percent']:.2f}%, {row['ci95_upper_percent']:.2f}%] |"
        )
    if not payload["publication_eligible"]:
        lines.extend(
            [
                "",
                "The table above is a pipeline-validation result over the currently complete "
                "competitions. It is not the requested four-competition interval and must not "
                "be substituted for it.",
            ]
        )
    lines.append("")
    return "\n".join(lines)


def protocol_markdown(payload: Mapping[str, Any]) -> str:
    if payload.get("deep_protocol_audit_deferred") is True:
        return "\n".join(
            [
                "# Cross-generation protocol compatibility audit",
                "",
                "Overall status: **DEFERRED_TABLES_ONLY**.",
                "",
                str(payload["note"]),
                "",
                "The score table and sequence-cluster bootstrap are numerically built, "
                "but this placeholder is not a protocol PASS. It is replaced by the "
                "exhaustive stored-payload audit when the script is run without "
                "`--tables-only`.",
                "",
            ]
        )
    lines = [
        "# Cross-generation protocol compatibility audit",
        "",
        f"Overall status: **{payload['overall_status']}**.",
        "",
        "The generations are compatible on the requested scientific comparison invariants, "
        "but they are not byte-identical protocols. Pipeline and validation contracts evolved; "
        "the audit therefore proves the stable modality, payload, roster, score, judge-route, "
        "and frozen-rubric properties directly from stored requests and terminals.",
        "",
        "| Competition | Generation | Aggregate | Vision payload | Text payload | Judge/rubric | Independent seal |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in payload["runs"]:
        if row.get("status") != "complete":
            lines.append(
                f"| {row['competition']} | {row['generation']} | missing/in progress | not scanned while active | "
                "not scanned while active | partial | pending |"
            )
            continue
        solver = row["solver_payload_audit"]
        judge = row["judge_payload_audit"]
        seal = row["seal_evidence"]
        lines.append(
            f"| {row['competition']} | {row['generation']} | complete; {row['aggregate_sha256'][:12]}… | "
            f"PASS ({solver['vision_route_count']} routes) | PASS ({solver['text_staticized_route_count']} routes) | "
            f"PASS; `{judge['fixed_judge_route']}`; rubric set {judge['frozen_rubric_set_sha256'][:12]}… | "
            f"{'PASS' if seal['independent_pass_endorsement_present'] else 'pending'} |"
        )
    lines.extend(
        [
            "",
            "Evidence scope:",
            "",
            "- Every stored solver attempt for every vision route was checked for direct PNG data-image blocks whose byte count and SHA-256 bind to the frozen official question-page files and source-PDF metadata.",
            "- Every stored solver attempt for every text route was checked for zero image blocks; each terminal prompt hash was recomputed from the frozen question-only staticized field and its freeze record.",
            "- Every judge terminal and wire request was checked for the fixed `google/gemini-3.5-flash` route. Every non-deterministic request contains the exact normalized frozen item rubric and exact criterion-ID/max-score contract.",
            "- Deterministic model-no-answer rows use zero judge attempts and score zero; they do not introduce a different rubric judgment.",
            "",
            "Generation differences retained in the JSON audit:",
            "",
        ]
    )
    lines.extend(f"- {value}" for value in payload["generation_differences"])
    lines.append("")
    return "\n".join(lines)


def build_outputs(
    workspace: Path,
    output_dir: Path,
    *,
    allow_incomplete: bool,
    seed: int,
    resamples: int,
    check_only: bool,
    tables_only: bool,
    rehydrate_payloads: bool,
    reuse_sealed_numerical_artifacts: bool,
) -> dict[str, Any]:
    models_path, _model_manifest, models = model_manifest(workspace)
    frozen_model_sha = sha256_file(models_path)
    aggregates: dict[str, Mapping[str, Any] | None] = OrderedDict()
    cluster_scores: dict[
        str, Mapping[str, Mapping[str, Mapping[str, float]]]
    ] = OrderedDict()
    run_audits: list[dict[str, Any]] = []
    source_records: list[dict[str, Any]] = []
    complete_specs: list[RunSpec] = []
    for spec in RUN_SPECS:
        run_root = workspace / spec.run_rel
        require(run_root.is_dir(), f"missing run root: {run_root}")
        manifest, items_path, items = resolve_manifest_items(workspace, run_root)
        require(
            {str(item.get("benchmark_id")) for item in items} == {spec.benchmark_id},
            f"{spec.key}: benchmark ID mismatch in frozen items",
        )
        require(
            manifest.get("models_sha256") == frozen_model_sha
            or manifest.get("candidate_models_sha256") == frozen_model_sha,
            f"{spec.key}: frozen model roster SHA mismatch",
        )
        scoring = scoring_metadata(items, spec.raw_max)
        aggregate_path = run_root / "aggregate.json"
        aggregate: Mapping[str, Any] | None = None
        aggregate_complete = False
        if aggregate_path.is_file():
            candidate = read_json(aggregate_path)
            require(isinstance(candidate, dict), f"aggregate is not an object: {aggregate_path}")
            aggregate_complete = (
                candidate.get("complete") is True
                and candidate.get("missing_count") == 0
                and candidate.get("missing") == []
            )
            if aggregate_complete:
                if reuse_sealed_numerical_artifacts:
                    reconstructed = {
                        str(model["route"]): float(
                            candidate["aggregates"][str(model["route"])][
                                spec.benchmark_id
                            ]["score"]
                        )
                        for model in models
                    }
                    route_clusters = {}
                    cluster_source = "sealed_prior_numerical_artifact_countersign"
                else:
                    sequence_result = clusters_from_aggregate_breakdown(
                        spec, candidate, items, models
                    )
                    if sequence_result is None:
                        reconstructed, route_clusters = reconstruct_scores(
                            spec, run_root, items, models
                        )
                        cluster_source = "item_level_judge_terminal_reconstruction"
                    else:
                        reconstructed, route_clusters = sequence_result
                        cluster_source = "validated_aggregate_sequence_breakdown"
                validate_aggregate(
                    spec,
                    aggregate_path,
                    candidate,
                    items,
                    models,
                    reconstructed,
                )
                aggregate = candidate
                if not reuse_sealed_numerical_artifacts:
                    cluster_scores[spec.key] = route_clusters
                complete_specs.append(spec)
        if not aggregate_complete:
            require(
                allow_incomplete and spec.key == "nbpho2026",
                f"{spec.key}: complete sealed aggregate is required",
            )
            run_audits.append(
                {
                    "status": "incomplete_active_not_deep_scanned",
                    "competition": spec.display_name,
                    "competition_key": spec.key,
                    "benchmark_id": spec.benchmark_id,
                    "run_root": spec.run_rel,
                    "generation": spec.generation,
                    "run_manifest_schema": manifest.get("schema_version"),
                    "run_manifest_sha256": sha256_file(run_root / "run_manifest.json"),
                    "items_path": rel_to_workspace(workspace, items_path),
                    "items_sha256": sha256_file(items_path),
                    "model_manifest_sha256": frozen_model_sha,
                    "aggregate_present": aggregate_path.is_file(),
                    "aggregate_complete": False,
                    "item_count": len(items),
                    "sequence_cluster_count": len(scoring["clusters"]),
                    "table_score_raw_max": spec.raw_max,
                    "note": "Active NBPhO tree was not deep-scanned or written while judging was in progress.",
                }
            )
        else:
            if tables_only:
                run_audits.append(
                    {
                        "status": "tables_only_complete_protocol_deferred",
                        "competition": spec.display_name,
                        "competition_key": spec.key,
                        "benchmark_id": spec.benchmark_id,
                        "run_root": spec.run_rel,
                        "generation": spec.generation,
                        "aggregate_complete": True,
                        "aggregate_sha256": sha256_file(aggregate_path),
                        "item_count": len(items),
                        "sequence_cluster_count": len(scoring["clusters"]),
                        "table_score_raw_max": spec.raw_max,
                        "bootstrap_cluster_source": cluster_source,
                    }
                )
            else:
                audit_builder = (
                    audit_one_complete_run
                    if rehydrate_payloads
                    else audit_one_from_sealed_countersigns
                )
                audit = audit_builder(
                    workspace,
                    spec,
                    run_root,
                    manifest,
                    items_path,
                    items,
                    models_path,
                    models,
                    aggregate_path,
                )
                audit["status"] = "complete"
                audit["scoring_metadata"] = scoring
                audit["bootstrap_cluster_source"] = cluster_source
                run_audits.append(audit)
        aggregates[spec.key] = aggregate
        source_records.append(
            {
                "competition_key": spec.key,
                "run_root": spec.run_rel,
                "run_manifest_sha256": sha256_file(run_root / "run_manifest.json"),
                "items_path": rel_to_workspace(workspace, items_path),
                "items_sha256": sha256_file(items_path),
                "aggregate_path": (
                    rel_to_workspace(workspace, aggregate_path)
                    if aggregate_path.is_file()
                    else None
                ),
                "aggregate_sha256": (
                    sha256_file(aggregate_path) if aggregate_path.is_file() else None
                ),
                "aggregate_complete": aggregate_complete,
            }
        )
    target_complete = len(complete_specs) == len(RUN_SPECS)
    require(target_complete or allow_incomplete, "four-competition target is incomplete")
    score_rows = build_score_rows(RUN_SPECS, models, aggregates)
    score_payload = {
        "schema_version": ANALYSIS_SCHEMA,
        "status": (
            "complete_four_competition_table" if target_complete else "partial_waiting_for_nbpho"
        ),
        "numerically_complete": target_complete,
        "publication_eligible": target_complete and not tables_only,
        "deep_protocol_audit_complete": not tables_only,
        "model_count": len(models),
        "competition_order": [spec.key for spec in RUN_SPECS],
        "explicit_raw_maxima": {spec.key: spec.raw_max for spec in RUN_SPECS},
        "normalization": "score / explicit competition raw maximum * 100",
        "macro_mean": "unweighted arithmetic mean of all four normalized percentages",
        "missingness_policy": "no imputation; macro is missing unless all four competition rows are complete",
        "rows": score_rows,
    }
    if reuse_sealed_numerical_artifacts:
        bootstrap_path = output_dir / "table6_bootstrap_macro_ci.json"
        require(bootstrap_path.is_file(), "sealed numerical bootstrap artifact is absent")
        bootstrap_payload = copy.deepcopy(read_json(bootstrap_path))
        require(
            bootstrap_payload.get("status") == "complete_four_competition_target"
            and bootstrap_payload.get("seed") == seed
            and bootstrap_payload.get("resamples") == resamples
            and bootstrap_payload.get("competition_keys")
            == [spec.key for spec in RUN_SPECS],
            "sealed numerical bootstrap metadata drift",
        )
        bootstrap_rows = bootstrap_payload.get("rows")
        require(isinstance(bootstrap_rows, list) and len(bootstrap_rows) == 14, "sealed bootstrap row drift")
        require(
            [row.get("route") for row in bootstrap_rows]
            == [str(model["route"]) for model in models],
            "sealed bootstrap route order drift",
        )
        score_by_route = {row["route"]: row for row in score_rows}
        for row in bootstrap_rows:
            assert_close(
                row.get("point_macro_mean_percent"),
                float(score_by_route[row["route"]]["macro_mean_percent"]),
                f"sealed bootstrap point macro {row['route']}",
            )
        numerical_core = {
            key: bootstrap_payload[key]
            for key in (
                "schema_version",
                "status",
                "competition_keys",
                "competition_cluster_counts",
                "resampling_unit",
                "within_competition_estimator",
                "across_competition_estimator",
                "paired_across_models",
                "resamples",
                "seed",
                "prng",
                "interval",
                "uncertainty_scope",
                "rows",
            )
        }
        bootstrap_payload["numerical_core_sha256"] = canonical_digest(numerical_core)
        bootstrap_payload["numerical_reuse_validation"] = (
            "all 14 point macros recomputed from the four current sealed aggregates; "
            "CI rows retained from the immediately preceding tables-only build"
        )
    else:
        bootstrap_payload = bootstrap_macro_ci(
            complete_specs,
            models,
            cluster_scores,
            seed=seed,
            resamples=resamples,
            target_is_complete=target_complete,
        )
    bootstrap_payload["deep_protocol_audit_complete"] = not tables_only
    bootstrap_payload["publication_eligible"] = target_complete and not tables_only
    protocol_payload = (
        {
            "schema_version": PROTOCOL_SCHEMA,
            "overall_status": "DEFERRED_TABLES_ONLY",
            "table_comparability_on_requested_invariants": False,
            "deep_protocol_audit_deferred": True,
            "note": (
                "Speed-priority numerical build only. Run again without --tables-only "
                "to produce the exhaustive payload/judge compatibility audit."
            ),
            "generation_differences": [],
            "runs": run_audits,
        }
        if tables_only
        else protocol_compatibility_summary(run_audits)
    )
    protocol_complete_and_sealed = (
        protocol_payload["overall_status"]
        == "PASS_all_four_complete_and_independently_sealed"
    )
    score_payload["publication_eligible"] = (
        target_complete and protocol_complete_and_sealed
    )
    bootstrap_payload["publication_eligible"] = (
        target_complete and protocol_complete_and_sealed
    )
    result = {
        "target_complete": target_complete,
        "score_payload": score_payload,
        "bootstrap_payload": bootstrap_payload,
        "protocol_payload": protocol_payload,
        "source_records": source_records,
    }
    if check_only:
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    score_json_path = output_dir / "table6_scores.json"
    score_csv_path = output_dir / "table6_scores.csv"
    score_md_path = output_dir / "table6_scores.md"
    bootstrap_json_path = output_dir / "table6_bootstrap_macro_ci.json"
    bootstrap_csv_path = output_dir / "table6_bootstrap_macro_ci.csv"
    bootstrap_md_path = output_dir / "table6_bootstrap_macro_ci.md"
    protocol_json_path = output_dir / "protocol_compatibility_audit.json"
    protocol_csv_path = output_dir / "protocol_route_evidence.csv"
    protocol_md_path = output_dir / "protocol_compatibility_audit.md"

    write_json(score_json_path, score_payload)
    fields, flat_scores = score_csv_rows(score_rows)
    write_csv(score_csv_path, fields, flat_scores)
    score_md_path.write_text(score_markdown(score_payload), encoding="utf-8")
    write_json(bootstrap_json_path, bootstrap_payload)
    bootstrap_fields = [
        "table6_order",
        "display_name",
        "route",
        "input_mode",
        "competition_count",
        "point_macro_mean_percent",
        "bootstrap_mean_percent",
        "ci95_lower_percent",
        "ci95_upper_percent",
    ]
    write_csv(bootstrap_csv_path, bootstrap_fields, bootstrap_payload["rows"])
    bootstrap_md_path.write_text(bootstrap_markdown(bootstrap_payload), encoding="utf-8")
    write_json(protocol_json_path, protocol_payload)
    protocol_route_rows: list[dict[str, Any]] = []
    for run in run_audits:
        if run.get("status") != "complete":
            continue
        for route in run["solver_payload_audit"]["routes"]:
            protocol_route_rows.append(
                {
                    "competition_key": run["competition_key"],
                    "generation": run["generation"],
                    **route,
                }
            )
    protocol_fields = [
        "competition_key",
        "generation",
        "route",
        "display_name",
        "input_mode",
        "solver_route_root_is_symlink",
        "solver_route_root_symlink_target",
        "terminal_count",
        "attempt_count",
        "wire_image_block_count",
        "wire_image_evidence_sha256",
        "frozen_prompt_binding_sha256",
        "direct_original_png_payload_pass",
        "zero_image_blocks_and_staticized_prompt_pass",
        "pass",
    ]
    write_csv(protocol_csv_path, protocol_fields, protocol_route_rows)
    protocol_md_path.write_text(protocol_markdown(protocol_payload), encoding="utf-8")

    generated_paths = [
        score_json_path,
        score_csv_path,
        score_md_path,
        bootstrap_json_path,
        bootstrap_csv_path,
        bootstrap_md_path,
        protocol_json_path,
        protocol_csv_path,
        protocol_md_path,
    ]
    script_path = Path(__file__).resolve()
    manifest_payload = {
        "schema_version": ANALYSIS_SCHEMA,
        "build_status": (
            (
                "complete_four_competition_tables_only_protocol_deferred"
                if tables_only
                else "complete_four_competition"
            )
            if target_complete
            else "partial_validation_waiting_for_nbpho"
        ),
        "publication_eligible": target_complete and protocol_complete_and_sealed,
        "deep_protocol_audit_complete": not tables_only,
        "offline_only": True,
        "api_calls": 0,
        "formal_run_writes": 0,
        "script_path": rel_to_workspace(workspace, script_path),
        "script_sha256": sha256_file(script_path),
        "frozen_model_manifest": {
            "path": rel_to_workspace(workspace, models_path),
            "sha256": frozen_model_sha,
        },
        "sources": source_records,
        "bootstrap": {
            "resamples": resamples,
            "seed": seed,
            "scope": bootstrap_payload["uncertainty_scope"],
        },
        "generated_artifacts_excluding_manifest_and_SHA256SUMS": [
            {
                "path": path.name,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in generated_paths
        ],
    }
    manifest_path = output_dir / "analysis_manifest.json"
    write_json(manifest_path, manifest_payload)
    hash_paths = [script_path, *generated_paths, manifest_path]
    hash_lines = [
        f"{sha256_file(path)}  {rel_to_workspace(workspace, path)}" for path in hash_paths
    ]
    hash_path = output_dir / "SHA256SUMS"
    hash_path.write_text("\n".join(hash_lines) + "\n", encoding="utf-8")
    return result


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    script = Path(__file__).resolve()
    default_workspace = script.parents[4]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, default=default_workspace)
    parser.add_argument("--output-dir", type=Path, default=script.parent)
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow only the specified active NBPhO root to lack a complete aggregate",
    )
    parser.add_argument("--check-only", action="store_true", help="validate without writing outputs")
    parser.add_argument(
        "--tables-only",
        action="store_true",
        help=(
            "build the score table and bootstrap immediately while explicitly "
            "deferring the exhaustive cross-generation payload audit"
        ),
    )
    parser.add_argument(
        "--rehydrate-payloads",
        action="store_true",
        help=(
            "re-read every sealed wire locally instead of composing the existing "
            "exhaustive independent PASS audits (slow on FileProvider storage)"
        ),
    )
    parser.add_argument(
        "--reuse-sealed-numerical-artifacts",
        action="store_true",
        help=(
            "reuse and countersign the existing 10,000-resample numerical artifact "
            "after recomputing all point macros from current sealed aggregates"
        ),
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    workspace = args.workspace.resolve()
    require((workspace / "experiments/2026_reval").is_dir(), f"invalid workspace: {workspace}")
    result = build_outputs(
        workspace,
        args.output_dir.resolve(),
        allow_incomplete=args.allow_incomplete,
        seed=args.seed,
        resamples=args.resamples,
        check_only=args.check_only,
        tables_only=args.tables_only,
        rehydrate_payloads=args.rehydrate_payloads,
        reuse_sealed_numerical_artifacts=args.reuse_sealed_numerical_artifacts,
    )
    print(
        json.dumps(
            {
                "ok": True,
                "target_complete": result["target_complete"],
                "score_status": result["score_payload"]["status"],
                "bootstrap_status": result["bootstrap_payload"]["status"],
                "protocol_status": result["protocol_payload"]["overall_status"],
                "output_dir": str(args.output_dir.resolve()),
                "check_only": args.check_only,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AuditFailure as exc:
        print(f"AUDIT FAILURE: {exc}", file=sys.stderr)
        raise SystemExit(2)
