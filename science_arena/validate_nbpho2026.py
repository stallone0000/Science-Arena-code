#!/usr/bin/env python3
"""Fail-closed offline validator for the canonical NBPhO 2026 dataset."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
OUT = HERE / "datasets/nbpho2026"
QUESTION_PDF = WORKSPACE / "new-test/NBPhO 2026/NBPhO_2026_problems.pdf"
SOLUTION_PDF = WORKSPACE / "new-test/NBPhO 2026/NBPhO_2026_solutions.pdf"
EXPECTED_Q_HASH = "d796b6610aec45c7b0d1b2823b9d19561d94fce424867673a1da481d1b02145e"
EXPECTED_S_HASH = "6df4a2f9630e971034e2d6f4141b1e5bfa0816ee7b61c7b0e658996e1c26f457"
EXPECTED = {
    "P1": (2, 6.0), "P2": (1, 6.0), "P3": (3, 8.0), "P4": (4, 8.0),
    "P5": (3, 8.0), "P6": (3, 5.0), "P7": (3, 5.0), "P8": (3, 8.0),
    "P9": (3, 8.0), "P10": (3, 10.0),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def rerender_hashes(pdf: Path, expected_pages: int) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="nbpho2026-validate-render-") as tmp_name:
        prefix = Path(tmp_name) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", "180", str(pdf), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        pages = sorted(Path(tmp_name).glob("page-*.png"))
        if len(pages) != expected_pages:
            raise RuntimeError(f"re-rendered {len(pages)} pages from {pdf}, expected {expected_pages}")
        return [sha256_file(path) for path in pages]


def validate(output: Path = OUT, *, rerender: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    digest_cache: dict[Path, str] = {}

    def cached_sha256(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in digest_cache:
            digest_cache[resolved] = sha256_file(resolved)
        return digest_cache[resolved]

    rows = load_jsonl(output / "items.jsonl")
    q_manifest = load_jsonl(output / "question_image_manifest.jsonl")
    s_manifest = load_jsonl(output / "official_solution_image_manifest.jsonl")
    manifest = load_json(output / "benchmark_manifest.json")
    page_audit = load_json(output / "page_audit.json")

    check(cached_sha256(QUESTION_PDF) == EXPECTED_Q_HASH, "problem PDF SHA-256 mismatch", errors)
    check(cached_sha256(SOLUTION_PDF) == EXPECTED_S_HASH, "solution PDF SHA-256 mismatch", errors)
    check(len(rows) == 28, f"item_count={len(rows)} expected 28", errors)
    check(len({row.get("item_id") for row in rows}) == 28, "item IDs are not unique", errors)
    check(abs(sum(float(row["max_score"]) for row in rows) - 72.0) < 1e-9, "total score is not 72", errors)
    check([row.get("global_order") for row in rows] == list(range(1, 29)), "global_order is not contiguous 1..28", errors)
    check(len(q_manifest) == 2, f"question image count={len(q_manifest)} expected 2", errors)
    check(len(s_manifest) == 14, f"solution image count={len(s_manifest)} expected 14", errors)

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        item_id = str(row.get("item_id"))
        by_problem[str(row.get("problem_id"))].append(row)
        check(row.get("benchmark_id") == "nbpho2026", f"{item_id}: benchmark mismatch", errors)
        check(row.get("schema_version") == "science_arena_item_v1", f"{item_id}: schema mismatch", errors)
        check(bool(row.get("question_text_native")), f"{item_id}: native question empty", errors)
        staticized = str(row.get("question_text_staticized") or "")
        check("QUESTION-ONLY" in staticized, f"{item_id}: missing question-only marker", errors)
        check("HUMAN-VISUAL-AUDITED" in staticized, f"{item_id}: missing visual audit marker", errors)
        check(row.get("question_only_staticization_verified") is True, f"{item_id}: staticization not verified", errors)
        check(row.get("human_visual_audited") is True, f"{item_id}: human visual audit false", errors)
        for forbidden in (
            SOLUTION_REL := "new-test/NBPhO 2026/NBPhO_2026_solutions.pdf",
            EXPECTED_S_HASH,
            "official_solution_images/",
            "OFFICIAL ANSWER",
            "OFFICIAL SCORING",
        ):
            check(forbidden not in staticized, f"{item_id}: solver text leaks forbidden marker {forbidden!r}", errors)

        criteria = (row.get("grading_rubric") or {}).get("criteria") or []
        check(bool(criteria), f"{item_id}: rubric criteria empty", errors)
        check(len({c.get("criterion_id") for c in criteria}) == len(criteria), f"{item_id}: duplicate criterion IDs", errors)
        try:
            criterion_sum = sum(float(c["max_score"]) for c in criteria)
        except Exception as exc:
            criterion_sum = -1
            errors.append(f"{item_id}: invalid rubric maximum: {exc}")
        check(abs(criterion_sum - float(row["max_score"])) < 1e-9, f"{item_id}: criterion sum {criterion_sum} != item max", errors)
        check(row.get("rubric_type") == "official_explicit_grading_from_provided_solution_pdf", f"{item_id}: rubric provenance mismatch", errors)
        check("present" in str(row.get("official_rubric_status")), f"{item_id}: official rubric status missing", errors)

        q_assets = row.get("question_image_assets") or []
        s_assets = row.get("official_solution_image_assets") or []
        check(bool(q_assets), f"{item_id}: no question assets", errors)
        check(bool(s_assets), f"{item_id}: no solution assets", errors)
        check([a.get("path") for a in q_assets] == row.get("question_image_paths"), f"{item_id}: question asset/path mismatch", errors)
        check([a.get("path") for a in s_assets] == row.get("official_solution_image_paths"), f"{item_id}: solution asset/path mismatch", errors)
        check(not (set(row["question_image_paths"]) & set(row["official_solution_image_paths"])), f"{item_id}: question/solution path overlap", errors)
        for asset in q_assets:
            path = WORKSPACE / str(asset.get("path"))
            check(path.is_file(), f"{item_id}: missing question image {path}", errors)
            if path.is_file():
                check(cached_sha256(path) == asset.get("sha256"), f"{item_id}: question image hash mismatch", errors)
            check(asset.get("source_pdf") == "new-test/NBPhO 2026/NBPhO_2026_problems.pdf", f"{item_id}: question image wrong PDF", errors)
            check(asset.get("source_pdf_sha256") == EXPECTED_Q_HASH, f"{item_id}: question PDF hash mismatch", errors)
            check(asset.get("contains_solution") is False, f"{item_id}: question asset marked as solution", errors)
        for asset in s_assets:
            path = WORKSPACE / str(asset.get("path"))
            check(path.is_file(), f"{item_id}: missing solution image {path}", errors)
            if path.is_file():
                check(cached_sha256(path) == asset.get("sha256"), f"{item_id}: solution image hash mismatch", errors)
            check(asset.get("source_pdf") == SOLUTION_REL, f"{item_id}: solution image wrong PDF", errors)
            check(asset.get("source_pdf_sha256") == EXPECTED_S_HASH, f"{item_id}: solution PDF hash mismatch", errors)
            check(asset.get("gold_only") is True, f"{item_id}: solution asset is not gold-only", errors)
            check(asset.get("must_not_be_sent_to_solver") is True, f"{item_id}: solution asset solver exclusion absent", errors)

        native_track = (row.get("input_tracks") or {}).get("native_vision") or {}
        text_track = (row.get("input_tracks") or {}).get("text_staticized") or {}
        check(native_track.get("direct_official_question_images") is True, f"{item_id}: native vision not direct-image", errors)
        check(native_track.get("description_is_not_sent") is True, f"{item_id}: vision route may receive description", errors)
        check(text_track.get("question_only") is True and text_track.get("no_image_payload") is True, f"{item_id}: text-only contract invalid", errors)
        check((row.get("solver_gold_isolation") or {}).get("leakage_check") == "passed", f"{item_id}: gold isolation not passed", errors)

    for problem, (count, score) in EXPECTED.items():
        group = by_problem[problem]
        check(len(group) == count, f"{problem}: count={len(group)} expected {count}", errors)
        check(abs(sum(float(row["max_score"]) for row in group) - score) < 1e-9, f"{problem}: score mismatch", errors)
        check(sorted(int(row["turn_index"]) for row in group) == list(range(1, count + 1)), f"{problem}: noncontiguous turn indexes", errors)

    q_by_path = {row["path"]: row for row in q_manifest}
    s_by_path = {row["path"]: row for row in s_manifest}
    all_q_refs = {path for row in rows for path in row["question_image_paths"]}
    all_s_refs = {path for row in rows for path in row["official_solution_image_paths"]}
    check(all_q_refs == set(q_by_path), "not all and only question-manifest pages are item-referenced", errors)
    check(all_s_refs == {path for path, ref in s_by_path.items() if int(ref["source_page"]) >= 3}, "solution item references should be exactly pages 3-14", errors)
    check(not any(int(ref["source_page"]) in {1, 2} for row in rows for ref in row["official_solution_image_assets"]), "duplicate problem pages 1-2 referenced as gold", errors)

    for path_text, ref in {**q_by_path, **s_by_path}.items():
        path = WORKSPACE / path_text
        check(path.is_file(), f"manifest asset missing {path_text}", errors)
        if path.is_file():
            check(cached_sha256(path) == ref["image_sha256"], f"manifest image hash mismatch {path_text}", errors)

    if rerender:
        q_rendered = rerender_hashes(QUESTION_PDF, 2)
        s_rendered = rerender_hashes(SOLUTION_PDF, 14)
        check(q_rendered == [q_by_path[path]["image_sha256"] for path in sorted(q_by_path)], "question PNGs are not exact fresh 180-DPI renders", errors)
        check(s_rendered == [s_by_path[path]["image_sha256"] for path in sorted(s_by_path)], "solution PNGs are not exact fresh 180-DPI renders", errors)

    check(page_audit.get("all_pages_text_and_visual_audited") is True, "page dual-audit flag not true", errors)
    check(len(page_audit.get("problem_pdf_pages") or []) == 2, "problem page audit count mismatch", errors)
    check(len(page_audit.get("solution_pdf_pages") or []) == 14, "solution page audit count mismatch", errors)
    for section in ("problem_pdf_pages", "solution_pdf_pages"):
        for page in page_audit.get(section) or []:
            check(all(page.get(flag) is True for flag in ("text_extracted", "rendered", "visually_reviewed")), f"page audit incomplete: {section} {page.get('page')}", errors)

    defects = {value.get("id") for value in manifest.get("source_defects") or []}
    expected_defects = {
        "nbpho2026_p3_ii_arc_wording", "nbpho2026_p5_missing_session_values",
        "nbpho2026_p9_duplicate_iv", "nbpho2026_p10_iii_mixed_unit",
    }
    check(defects == expected_defects, f"source defect set mismatch: {defects}", errors)
    check(manifest.get("items_sha256") == sha256_file(output / "items.jsonl"), "manifest items hash mismatch", errors)
    check(manifest.get("raw_score_total") == 72.0, "manifest total mismatch", errors)

    report = {
        "schema_version": "nbpho2026_canonical_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "item_count": len(rows),
            "problem_count": len(by_problem),
            "raw_score_total": sum(float(row["max_score"]) for row in rows),
            "visual_dependency_item_count": sum(bool(row["visual_dependency"]) for row in rows),
            "staticized_verified_count": sum(row.get("question_only_staticization_verified") is True for row in rows),
            "question_image_count": len(q_manifest),
            "solution_image_count": len(s_manifest),
            "question_pdf_sha256": cached_sha256(QUESTION_PDF),
            "solution_pdf_sha256": cached_sha256(SOLUTION_PDF),
            "items_sha256": cached_sha256(output / "items.jsonl"),
            "question_manifest_sha256": cached_sha256(output / "question_image_manifest.jsonl"),
            "solution_manifest_sha256": cached_sha256(output / "official_solution_image_manifest.jsonl"),
            "fresh_rerender_compared": rerender,
        },
    }
    return report


def main() -> None:
    report = validate()
    output_path = OUT / "validation_report.json"
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (OUT / "validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
