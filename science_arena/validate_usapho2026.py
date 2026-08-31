#!/usr/bin/env python3
"""Fail-closed offline validator for the canonical 2026 USAPhO dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
OUT = HERE / "datasets/usapho2026"
QUESTION_REL = "new-test/USAPhO 2026/2026-USAPhO-Problems.pdf"
SOLUTION_REL = "new-test/USAPhO 2026/2026-USAPhO-Solutions.pdf"
QUESTION_PDF = WORKSPACE / QUESTION_REL
SOLUTION_PDF = WORKSPACE / SOLUTION_REL
ADJUDICATION_REL = "experiments/2026_reval/adjudications/usapho2026_b2c_scattering_20260828.json"
ADJUDICATION_PATH = WORKSPACE / ADJUDICATION_REL
ADJUDICATION_ID = "usapho2026_b2c_scattering_20260828"
EXPECTED_Q_HASH = "95c2f0d278aa206d7386d868dca2a5af4f4135c12bae23a3dce6a55812cad982"
EXPECTED_S_HASH = "da401e05d009e1dbd14c231484142de03a7cbec9b1a04e7e8b131cc3d3854d64"
EXPECTED = {
    "A1": (6, 25.0), "A2": (4, 25.0), "A3": (6, 25.0),
    "B1": (9, 25.0), "B2": (4, 25.0), "B3": (5, 25.0),
}
EXPECTED_DEFECTS = {
    "usapho2026_no_official_points_or_rubric",
    "usapho2026_a2a_component_wording",
    "usapho2026_a3_vortex_force_balance_sign",
    "usapho2026_b1_duplicate_as",
    "usapho2026_b2c_cross_reference",
    "usapho2026_b2c_scattering_solution",
    "usapho2026_b3c_cross_reference",
}
ADJUDICATED_RUBRIC_STATUS = "scientifically_adjudicated_independent_correction_with_derived_criteria_not_official"
PRINTED_PAGE_BY_PDF_PAGE = {1: None, 2: 2, 3: 3, 4: 4, 5: 5, 6: None, 7: 6, 8: 7, 9: 8, 10: 9}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def rerender_hashes(pdf: Path, expected_pages: int) -> list[str]:
    with tempfile.TemporaryDirectory(prefix="usapho2026-validate-render-") as tmp_name:
        prefix = Path(tmp_name) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", "180", str(pdf), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        pages = sorted(Path(tmp_name).glob("page-*.png"))
        if len(pages) != expected_pages:
            raise RuntimeError(f"re-rendered {len(pages)} pages from {pdf}; expected {expected_pages}")
        return [sha256_file(path) for path in pages]


def validate(output: Path = OUT, *, rerender: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    digest_cache: dict[Path, str] = {}

    def cached_sha(path: Path) -> str:
        resolved = path.resolve()
        if resolved not in digest_cache:
            digest_cache[resolved] = sha256_file(resolved)
        return digest_cache[resolved]

    rows = load_jsonl(output / "items.jsonl")
    q_manifest = load_jsonl(output / "question_image_manifest.jsonl")
    s_manifest = load_jsonl(output / "official_solution_image_manifest.jsonl")
    manifest = load_json(output / "benchmark_manifest.json")
    page_audit = load_json(output / "page_audit.json")
    staticization_matrix = load_json(output / "staticization_completeness_matrix.json")
    schema = load_json(output / "item_schema.json")
    adjudication = load_json(ADJUDICATION_PATH)

    check(cached_sha(QUESTION_PDF) == EXPECTED_Q_HASH, "problem PDF SHA-256 mismatch", errors)
    check(cached_sha(SOLUTION_PDF) == EXPECTED_S_HASH, "solution PDF SHA-256 mismatch", errors)
    check(len(rows) == 34, f"item_count={len(rows)} expected 34", errors)
    check(len({row.get("item_id") for row in rows}) == 34, "item IDs are not unique", errors)
    check(abs(sum(float(row["max_score"]) for row in rows) - 150.0) < 1e-9, "total score is not 150", errors)
    check([row.get("global_order") for row in rows] == list(range(1, 35)), "global_order is not contiguous 1..34", errors)
    check(len(q_manifest) == 10, f"question image count={len(q_manifest)} expected 10", errors)
    check(len(s_manifest) == 22, f"solution image count={len(s_manifest)} expected 22", errors)
    check(schema.get("properties", {}).get("rubric_type", {}).get("const") == "derived_from_solution_not_official", "schema rubric provenance incorrect", errors)

    by_problem: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        item_id = str(row.get("item_id"))
        problem = str(row.get("problem_id"))
        by_problem[problem].append(row)
        check(re.fullmatch(r"usapho2026_(a|b)[1-3]_[a-z0-9_]+", item_id) is not None, f"{item_id}: invalid ID", errors)
        check(row.get("benchmark_id") == "usapho2026", f"{item_id}: benchmark mismatch", errors)
        check(row.get("schema_version") == "science_arena_item_v1", f"{item_id}: schema mismatch", errors)
        check(row.get("rubric_type") == "derived_from_solution_not_official", f"{item_id}: rubric provenance mismatch", errors)
        check(row.get("official_rubric_status") == "absent_in_provided_pdfs_derived_rubric_used", f"{item_id}: rubric absence not explicit", errors)
        check(row.get("official_answer") == row.get("provided_solution_answer_normalized"), f"{item_id}: provided source answer was not retained consistently", errors)
        check(row.get("adjudicated_reference_answer") == row.get("reference_answer"), f"{item_id}: adjudicated/reference answer mismatch", errors)
        if row.get("adjudication_id") is None:
            check(row.get("official_answer") == row.get("reference_answer"), f"{item_id}: answer changed without an adjudication", errors)
        check(float(row.get("problem_raw_max")) == 25.0 and float(row.get("problem_weight")) == 25.0, f"{item_id}: problem normalization mismatch", errors)
        check(bool(row.get("question_text_native")), f"{item_id}: native question empty", errors)
        native = str(row.get("question_text_native") or "")
        staticized = str(row.get("question_text_staticized") or "")
        check("QUESTION-ONLY" in staticized, f"{item_id}: missing question-only marker", errors)
        check("HUMAN-VISUAL-AUDITED" in staticized, f"{item_id}: missing visual-audit marker", errors)
        check(row.get("question_only_staticization_verified") is True, f"{item_id}: staticization not verified", errors)
        check(row.get("human_visual_audited") is True, f"{item_id}: human visual audit false", errors)
        for forbidden in (SOLUTION_REL, EXPECTED_S_HASH, "official_solution_images/", "provided worked solution", "B=(r0/8)", "benchmark-derived maximum"):
            check(forbidden not in staticized, f"{item_id}: staticized solver text leaks forbidden marker {forbidden!r}", errors)
            check(forbidden not in native, f"{item_id}: native solver text leaks forbidden marker {forbidden!r}", errors)
        check(row.get("solver_prompt_disclose_max_score") is False, f"{item_id}: curator-derived score cue was not disabled", errors)
        staticization = row.get("staticization") or {}
        check(staticization.get("completeness_matrix_status") == "PASS", f"{item_id}: completeness matrix status absent", errors)
        check(bool(staticization.get("requirements")), f"{item_id}: staticization requirements empty", errors)

        rubric = row.get("grading_rubric") or {}
        criteria = rubric.get("criteria") or []
        expected_rubric_status = ADJUDICATED_RUBRIC_STATUS if row.get("adjudication_id") else "derived_from_provided_solution_not_official"
        check(rubric.get("status") == expected_rubric_status, f"{item_id}: derived/adjudicated status mismatch", errors)
        check(bool(criteria), f"{item_id}: rubric criteria empty", errors)
        check(len({criterion.get("criterion_id") for criterion in criteria}) == len(criteria), f"{item_id}: duplicate criterion IDs", errors)
        try:
            criterion_sum = sum(float(criterion["max_score"]) for criterion in criteria)
        except Exception as exc:
            criterion_sum = -1.0
            errors.append(f"{item_id}: invalid rubric maximum: {exc}")
        check(abs(criterion_sum - float(row["max_score"])) < 1e-9, f"{item_id}: criterion sum {criterion_sum} != max", errors)

        q_assets = row.get("question_image_assets") or []
        s_assets = row.get("official_solution_image_assets") or []
        check(bool(q_assets), f"{item_id}: no question assets", errors)
        check(bool(s_assets), f"{item_id}: no solution assets", errors)
        check([asset.get("path") for asset in q_assets] == row.get("question_image_paths"), f"{item_id}: question asset/path mismatch", errors)
        check([asset.get("path") for asset in s_assets] == row.get("official_solution_image_paths"), f"{item_id}: solution asset/path mismatch", errors)
        check(not (set(row["question_image_paths"]) & set(row["official_solution_image_paths"])), f"{item_id}: question/solution path overlap", errors)
        for asset in q_assets:
            path = WORKSPACE / str(asset.get("path"))
            check(path.is_file(), f"{item_id}: missing question image {path}", errors)
            if path.is_file():
                check(cached_sha(path) == asset.get("sha256"), f"{item_id}: question image hash mismatch", errors)
            check(asset.get("source_pdf") == QUESTION_REL, f"{item_id}: question image wrong source PDF", errors)
            check(asset.get("source_pdf_sha256") == EXPECTED_Q_HASH, f"{item_id}: question source hash mismatch", errors)
            check(asset.get("contains_solution") is False and asset.get("contains_solution_or_rubric") is False, f"{item_id}: question asset not certified solution-free", errors)
        for asset in s_assets:
            path = WORKSPACE / str(asset.get("path"))
            check(path.is_file(), f"{item_id}: missing solution image {path}", errors)
            if path.is_file():
                check(cached_sha(path) == asset.get("sha256"), f"{item_id}: solution image hash mismatch", errors)
            check(asset.get("source_pdf") == SOLUTION_REL, f"{item_id}: solution image wrong source PDF", errors)
            check(asset.get("source_pdf_sha256") == EXPECTED_S_HASH, f"{item_id}: solution source hash mismatch", errors)
            check(asset.get("gold_only") is True and asset.get("must_not_be_sent_to_solver") is True, f"{item_id}: gold-only solver exclusion absent", errors)

        check(row.get("visual_dependency") is True, f"{item_id}: direct-page-pixel enforcement is not enabled", errors)
        native_track = (row.get("input_tracks") or {}).get("native_vision") or {}
        text_track = (row.get("input_tracks") or {}).get("text_staticized") or {}
        check(native_track.get("direct_official_question_images") is True, f"{item_id}: native vision not direct-image", errors)
        check(native_track.get("description_is_not_sent") is True, f"{item_id}: vision route may receive description", errors)
        check(native_track.get("missing_asset_policy") == "block_no_fallback", f"{item_id}: missing-asset route does not fail closed", errors)
        check(text_track.get("question_only") is True and text_track.get("no_image_payload") is True, f"{item_id}: text-only contract invalid", errors)
        check(native_track.get("include_curator_derived_score_cue") is False, f"{item_id}: native track score cue not disabled", errors)
        check(text_track.get("include_curator_derived_score_cue") is False, f"{item_id}: text track score cue not disabled", errors)
        check((row.get("solver_gold_isolation") or {}).get("leakage_check") == "passed", f"{item_id}: gold isolation not passed", errors)

        pdf_pages = [int(value) for value in (row.get("question_source") or {}).get("pdf_pages") or []]
        expected_printed_pages = [PRINTED_PAGE_BY_PDF_PAGE[value] for value in pdf_pages if PRINTED_PAGE_BY_PDF_PAGE[value] is not None]
        check((row.get("question_source") or {}).get("printed_pages") == expected_printed_pages, f"{item_id}: printed/PDF page labels are conflated", errors)

    for problem, (expected_count, expected_score) in EXPECTED.items():
        group = by_problem[problem]
        check(len(group) == expected_count, f"{problem}: count={len(group)} expected {expected_count}", errors)
        check(abs(sum(float(row["max_score"]) for row in group) - expected_score) < 1e-9, f"{problem}: score mismatch", errors)
        check(sorted(int(row["turn_index"]) for row in group) == list(range(1, expected_count + 1)), f"{problem}: noncontiguous turns", errors)

    q_by_path = {row["path"]: row for row in q_manifest}
    s_by_path = {row["path"]: row for row in s_manifest}
    all_q_refs = {path for row in rows for path in row["question_image_paths"]}
    all_s_refs = {path for row in rows for path in row["official_solution_image_paths"]}
    check(all_q_refs == set(q_by_path), "not all and only question pages are item-referenced", errors)
    check(all_s_refs == {path for path, ref in s_by_path.items() if int(ref["source_page"]) >= 2}, "solution references should be exactly pages 2-22", errors)
    check(not any(int(asset["source_page"]) == 1 for row in rows for asset in row["official_solution_image_assets"]), "solution reference page was sent as gold for an item", errors)
    for path_text, ref in {**q_by_path, **s_by_path}.items():
        path = WORKSPACE / path_text
        check(path.is_file(), f"manifest asset missing {path_text}", errors)
        if path.is_file():
            check(cached_sha(path) == ref["image_sha256"], f"manifest image hash mismatch {path_text}", errors)

    if rerender:
        q_rendered = rerender_hashes(QUESTION_PDF, 10)
        s_rendered = rerender_hashes(SOLUTION_PDF, 22)
        q_ordered = sorted(q_manifest, key=lambda value: int(value["source_page"]))
        s_ordered = sorted(s_manifest, key=lambda value: int(value["source_page"]))
        check(q_rendered == [value["image_sha256"] for value in q_ordered], "question PNGs are not exact fresh 180-DPI renders", errors)
        check(s_rendered == [value["image_sha256"] for value in s_ordered], "solution PNGs are not exact fresh 180-DPI renders", errors)

    check(page_audit.get("all_pages_text_and_visual_audited") is True, "page dual-audit flag not true", errors)
    check(len(page_audit.get("problem_pdf_pages") or []) == 10, "problem page audit count mismatch", errors)
    check(len(page_audit.get("solution_pdf_pages") or []) == 22, "solution page audit count mismatch", errors)
    for section in ("problem_pdf_pages", "solution_pdf_pages"):
        for page in page_audit.get(section) or []:
            check(all(page.get(flag) is True for flag in ("text_extracted", "rendered", "visually_reviewed")), f"page audit incomplete: {section} {page.get('page')}", errors)

    matrix_items = staticization_matrix.get("items") or []
    check(staticization_matrix.get("status") == "PASS", "staticization completeness matrix did not pass", errors)
    check(staticization_matrix.get("solution_content_used") is False, "staticization matrix permits solution content", errors)
    check(len(matrix_items) == 34, f"staticization matrix item_count={len(matrix_items)} expected 34", errors)
    check({value.get("item_id") for value in matrix_items} == {row.get("item_id") for row in rows}, "staticization matrix item IDs differ from canonical items", errors)
    for value in matrix_items:
        matrix_item_id = str(value.get("item_id"))
        check(value.get("status") == "PASS", f"{matrix_item_id}: matrix row not PASS", errors)
        check(bool(value.get("requirements")), f"{matrix_item_id}: matrix requirements empty", errors)
        check(value.get("question_only") is True and value.get("solution_or_rubric_content_included") is False, f"{matrix_item_id}: matrix question-only isolation failed", errors)
        check(value.get("curator_derived_score_cue_in_question_fields") is False, f"{matrix_item_id}: matrix score cue policy failed", errors)

    defects = {value.get("id") for value in manifest.get("source_defects") or []}
    check(defects == EXPECTED_DEFECTS, f"source defect set mismatch: {defects}", errors)
    check(manifest.get("items_sha256") == cached_sha(output / "items.jsonl"), "manifest items hash mismatch", errors)
    check(manifest.get("raw_score_total") == 150.0, "manifest total mismatch", errors)
    check(manifest.get("direct_question_page_pixel_required_item_count") == 34, "manifest direct-image count mismatch", errors)
    check(manifest.get("intrinsic_visual_dependency_item_count") == 0, "manifest intrinsic visual count mismatch", errors)
    check(manifest.get("official_question_figure_item_count") == 6, "manifest figure item count mismatch", errors)
    check(manifest.get("staticization_completeness_status") == "PASS", "manifest staticization completeness status mismatch", errors)
    check(manifest.get("staticization_completeness_item_count") == 34, "manifest staticization completeness count mismatch", errors)
    check(manifest.get("staticization_completeness_matrix_sha256") == cached_sha(output / "staticization_completeness_matrix.json"), "manifest staticization matrix hash mismatch", errors)
    check("2026 USAPhO" in (output / "source_text/problems_layout.txt").read_text(encoding="utf-8"), "year/competition identity absent from extracted problem text", errors)
    solution_text = (output / "source_text/solutions_layout.txt").read_text(encoding="utf-8").lower()
    check("grading" not in solution_text and "rubric" not in solution_text, "unexpected official grading/rubric text found", errors)

    b2ci = next((row for row in rows if row.get("item_id") == "usapho2026_b2_c_i"), {})
    b2cii = next((row for row in rows if row.get("item_id") == "usapho2026_b2_c_ii"), {})
    check("cot(alpha_s/2)" in str(b2ci.get("reference_answer")), "B2(c)(i) corrected Rutherford answer absent", errors)
    check("r0/8" in str(b2cii.get("reference_answer")), "B2(c)(ii) corrected factor 1/8 absent", errors)
    for row in (b2ci, b2cii):
        item_id = str(row.get("item_id"))
        check(row.get("official_answer") == "B(alpha)=r0 cot(alpha/2).", f"{item_id}: original provided answer not preserved", errors)
        check(row.get("provided_solution_answer_normalized") == "B(alpha)=r0 cot(alpha/2).", f"{item_id}: normalized provided answer missing", errors)
        check(row.get("reference_answer") != row.get("official_answer"), f"{item_id}: source and corrected answers were silently conflated", errors)
        check(row.get("reference_answer_status") == "scientifically_adjudicated_correction_of_provided_solution", f"{item_id}: correction status absent", errors)
        check(row.get("adjudication_id") == ADJUDICATION_ID, f"{item_id}: adjudication ID mismatch", errors)
        check((row.get("grading_rubric") or {}).get("status") == ADJUDICATED_RUBRIC_STATUS, f"{item_id}: rubric status does not disclose independent adjudication", errors)
        linked = row.get("adjudication") or {}
        check(linked.get("path") == ADJUDICATION_REL, f"{item_id}: adjudication path mismatch", errors)
        check(linked.get("sha256") == cached_sha(ADJUDICATION_PATH), f"{item_id}: adjudication hash mismatch", errors)
        dataset_copy = WORKSPACE / str(linked.get("dataset_copy"))
        check(dataset_copy.is_file(), f"{item_id}: dataset adjudication copy missing", errors)
        if dataset_copy.is_file():
            check(cached_sha(dataset_copy) == cached_sha(ADJUDICATION_PATH), f"{item_id}: dataset adjudication copy differs", errors)
    b2_defects = {value.get("id") for value in b2cii.get("source_defects") or []}
    check("usapho2026_b2c_scattering_solution" in b2_defects, "B2(c) source error is not attached to item", errors)
    check(adjudication.get("adjudication_id") == ADJUDICATION_ID, "standalone adjudication ID mismatch", errors)
    check(adjudication.get("affected_item_ids") == ["usapho2026_b2_c_i", "usapho2026_b2_c_ii"], "standalone adjudication affected items mismatch", errors)
    check((adjudication.get("provided_solution_record") or {}).get("final_answer_normalized") == "B(alpha)=r0 cot(alpha/2).", "standalone adjudication lost provided answer", errors)
    check("a_C cot(alpha_s/2)" in str((adjudication.get("adjudicated_reference") or {}).get("part_c_i")), "standalone adjudication corrected part c(i) absent", errors)
    check("r0/8" in str((adjudication.get("adjudicated_reference") or {}).get("part_c_ii")), "standalone adjudication corrected part c(ii) absent", errors)
    manifest_adjudications = manifest.get("adjudications") or []
    check(len(manifest_adjudications) == 1, "manifest adjudication count mismatch", errors)
    if manifest_adjudications:
        manifest_adjudication = manifest_adjudications[0]
        check(manifest_adjudication.get("adjudication_id") == ADJUDICATION_ID, "manifest adjudication ID mismatch", errors)
        check(manifest_adjudication.get("sha256") == cached_sha(ADJUDICATION_PATH), "manifest adjudication hash mismatch", errors)

    report = {
        "schema_version": "usapho2026_canonical_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "warnings": warnings,
        "checks": {
            "item_count": len(rows),
            "problem_count": len(by_problem),
            "raw_score_total": sum(float(row["max_score"]) for row in rows),
            "direct_question_page_pixel_required_item_count": sum(row.get("visual_dependency") is True for row in rows),
            "intrinsic_visual_dependency_item_count": sum(row.get("intrinsic_visual_dependency") is True for row in rows),
            "official_question_figure_item_count": sum(row.get("has_official_question_figure") is True for row in rows),
            "staticized_verified_count": sum(row.get("question_only_staticization_verified") is True for row in rows),
            "staticization_matrix_complete_count": sum(value.get("status") == "PASS" for value in matrix_items),
            "question_image_count": len(q_manifest),
            "solution_image_count": len(s_manifest),
            "dual_audited_page_count": len(page_audit.get("problem_pdf_pages") or []) + len(page_audit.get("solution_pdf_pages") or []),
            "question_pdf_sha256": cached_sha(QUESTION_PDF),
            "solution_pdf_sha256": cached_sha(SOLUTION_PDF),
            "items_sha256": cached_sha(output / "items.jsonl"),
            "question_manifest_sha256": cached_sha(output / "question_image_manifest.jsonl"),
            "solution_manifest_sha256": cached_sha(output / "official_solution_image_manifest.jsonl"),
            "staticization_completeness_matrix_sha256": cached_sha(output / "staticization_completeness_matrix.json"),
            "adjudication_count": len(manifest.get("adjudications") or []),
            "adjudication_sha256": cached_sha(ADJUDICATION_PATH),
            "fresh_rerender_compared": rerender,
        },
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--no-rerender", action="store_true")
    args = parser.parse_args()
    report = validate(args.output, rerender=not args.no_rerender)
    for name in ("validation_report.json", "validation.json"):
        (args.output / name).write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
