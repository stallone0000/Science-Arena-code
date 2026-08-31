#!/usr/bin/env python3
"""Fail-closed validator for the canonical IChO 2026 dataset."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[4]
OUT = ROOT / "experiments/2026_reval/datasets/icho2026"

EXPECTED_PROBLEMS = {
    "T1": (6, 25, 7), "T2": (7, 35, 6), "T3": (7, 63, 7),
    "T4": (9, 22, 6), "T5": (6, 18, 6), "T6": (7, 83, 7),
    "T7": (7, 54, 7), "T8": (10, 84, 7), "T9": (9, 53, 7),
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate() -> dict:
    rows = load_jsonl(OUT / "items.jsonl")
    errors: list[str] = []
    source_pdf_hash_cache: dict[Path, str] = {}

    def validate_source_pdf(ref: dict, item_id: str, kind: str) -> None:
        source_value = ref.get("source_pdf_path") or ref.get("source_pdf")
        expected = ref.get("source_pdf_sha256")
        check(bool(source_value), f"{item_id} {kind} lacks source PDF path", errors)
        check(bool(expected), f"{item_id} {kind} lacks source PDF hash", errors)
        if not source_value:
            return
        source = ROOT / str(source_value)
        check(source.is_file(), f"{item_id} {kind} source PDF missing: {source}", errors)
        if source.is_file() and expected:
            if source not in source_pdf_hash_cache:
                source_pdf_hash_cache[source] = sha256_file(source)
            check(
                source_pdf_hash_cache[source] == expected,
                f"{item_id} {kind} source PDF hash mismatch: {source}",
                errors,
            )
    check(len(rows) == 68, f"item_count={len(rows)} expected 68", errors)
    check(len({r["item_id"] for r in rows}) == 68, "item_id values are not unique", errors)
    check(sum(r["max_raw_points"] for r in rows) == 437, "raw point total is not 437", errors)
    check(sum(meta[2] for meta in EXPECTED_PROBLEMS.values()) == 60, "weight total is not 60", errors)
    check(sum(bool(r["visual_dependency"]) for r in rows) == 50, "visual dependency total is not 50", errors)
    check(sum(bool(r["question_text_staticized_complete"]) for r in rows) == 68, "not all staticized prompts are complete", errors)
    check(sum(bool(r["human_visual_audited"]) for r in rows) == 68, "not all items are human visual audited", errors)

    by_problem: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_problem[row["problem_id"]].append(row)
        required = [
            "question_text_native", "question_text_staticized", "official_answer",
            "official_solution", "official_rubric", "official_rubric_criteria",
        ]
        for field in required:
            check(bool(row.get(field)), f"{row['item_id']} missing {field}", errors)
        check(
            "<<<HUMAN-VISUAL-AUDITED NON-TEXT TRANSCRIPTION" in row["question_text_staticized"],
            f"{row['item_id']} lacks visual transcription appendix",
            errors,
        )
        check(row["solver_gold_isolation"]["leakage_check"] == "passed", f"{row['item_id']} leakage status not passed", errors)
        check(row.get("question_only_staticization_verified") is True, f"{row['item_id']} lacks production staticization verification flag", errors)
        check(bool(row["question_image_paths"]), f"{row['item_id']} has no question images", errors)
        assets = row.get("question_image_assets") or []
        check(len(assets) == len(row["question_image_paths"]), f"{row['item_id']} question_image_assets length mismatch", errors)
        check({a.get("path") for a in assets} == set(row["question_image_paths"]), f"{row['item_id']} asset paths do not match question_image_paths", errors)
        for asset in assets:
            check(bool(asset.get("sha256")), f"{row['item_id']} asset lacks sha256", errors)
            check(bool(asset.get("source_pdf_path")), f"{row['item_id']} asset lacks source_pdf_path", errors)
            check(bool(asset.get("source_pdf_sha256")), f"{row['item_id']} asset lacks source_pdf_sha256", errors)
            check(asset.get("source_page") is not None, f"{row['item_id']} asset lacks source_page", errors)
            check(asset.get("contains_solution") is False, f"{row['item_id']} asset is not certified solver-safe", errors)
            validate_source_pdf(asset, row["item_id"], "question asset")
            check(
                "solutions" not in str(asset.get("source_pdf_path") or asset.get("source_pdf")),
                f"{row['item_id']} question asset points to solution material",
                errors,
            )
        if row["visual_dependency"]:
            check(
                row["multimodal_missing_asset_policy"] == "BLOCK_ITEM_DO_NOT_FALL_BACK_TO_STATICIZED_TEXT",
                f"{row['item_id']} visual missing-asset policy is not fail-closed",
                errors,
            )
        solver_text = row["question_text_native"] + "\n" + row["question_text_staticized"]
        forbidden_solver_markers = [
            "theoretical-exam-solutions/", "OFFICIAL SOURCE-COLORED ANSWER",
            "OFFICIAL SCORING-RULE TRANSCRIPTION", "official_solution_image_paths",
        ]
        for marker in forbidden_solver_markers:
            check(marker not in solver_text, f"{row['item_id']} solver text contains gold marker {marker!r}", errors)
        for image_ref in row["question_image_refs"]:
            check("solutions" not in image_ref["source_pdf"], f"{row['item_id']} solver image points to solution PDF", errors)
            check(not image_ref.get("contains_solution_or_rubric", True), f"{row['item_id']} solver image marked as solution/rubric", errors)
        criteria = row["official_rubric_criteria"]
        check(len(criteria) >= 1, f"{row['item_id']} has no rubric criterion", errors)
        if criteria:
            check(
                abs(sum(float(c["max_score"]) for c in criteria) - float(row["max_raw_points"])) < 1e-9,
                f"{row['item_id']} official criterion maxima do not sum to item cap",
                errors,
            )
            check(
                len({c.get("criterion_id") for c in criteria}) == len(criteria),
                f"{row['item_id']} official criterion ids are not unique",
                errors,
            )
        production_rubric = row.get("grading_rubric") or {}
        production_criteria = production_rubric.get("criteria") or []
        check(bool(production_criteria), f"{row['item_id']} grading_rubric.criteria empty", errors)
        ids = [c.get("criterion_id") for c in production_criteria]
        check(len(ids) == len(set(ids)), f"{row['item_id']} grading criterion ids not unique", errors)
        try:
            criterion_sum = sum(float(c["max_score"]) for c in production_criteria)
        except Exception:
            criterion_sum = -1
        check(abs(criterion_sum - float(row["max_score"])) < 1e-9, f"{row['item_id']} grading criterion maxima sum {criterion_sum}", errors)
        declared_rule_ids = {
            rule.get("rule_id") for rule in row.get("official_rubric_rules", [])
        }
        for criterion in production_criteria:
            for rule_id in criterion.get("official_rule_ids", []):
                check(
                    rule_id in declared_rule_ids,
                    f"{row['item_id']} criterion references unknown official rule {rule_id}",
                    errors,
                )
            if criterion.get("official_rule_ids"):
                check(
                    bool(criterion.get("source_provenance")),
                    f"{row['item_id']} curated criterion lacks source provenance",
                    errors,
                )

    for pid, (count, raw, weight) in EXPECTED_PROBLEMS.items():
        group = by_problem[pid]
        check(len(group) == count, f"{pid} count={len(group)} expected {count}", errors)
        check(sum(r["max_raw_points"] for r in group) == raw, f"{pid} raw total mismatch", errors)
        check(all(r["problem_weight"] == weight for r in group), f"{pid} weight mismatch", errors)
        check(all(r["problem_raw_max"] == raw for r in group), f"{pid} problem_raw_max mismatch", errors)
        check(sorted(r["turn_index"] for r in group) == list(range(1, count + 1)), f"{pid} turn indices mismatch", errors)

    referenced_question_refs = {r["path"]: r for item in rows for r in item["question_image_refs"]}
    referenced_solution_refs = {r["path"]: r for item in rows for r in item["official_solution_image_refs"]}
    question_refs = {r["path"]: r for r in load_jsonl(OUT / "question_image_manifest.jsonl")}
    solution_refs = {r["path"]: r for r in load_jsonl(OUT / "official_solution_image_manifest.jsonl")}
    check(len(referenced_question_refs) == 91, f"item-referenced question image count={len(referenced_question_refs)} expected 91", errors)
    check(len(referenced_solution_refs) == 83, f"item-referenced solution image count={len(referenced_solution_refs)} expected 83", errors)
    check(len(question_refs) == 93, f"full question image manifest count={len(question_refs)} expected 93", errors)
    check(len(solution_refs) == 85, f"full solution image manifest count={len(solution_refs)} expected 85", errors)
    check(set(referenced_question_refs) <= set(question_refs), "an item-referenced question image is absent from full manifest", errors)
    check(set(referenced_solution_refs) <= set(solution_refs), "an item-referenced solution image is absent from full manifest", errors)
    for kind, refs in (("question", question_refs), ("solution", solution_refs)):
        for rel, ref in refs.items():
            path = ROOT / rel
            check(path.is_file(), f"missing {kind} image {rel}", errors)
            if path.is_file():
                check(sha256_file(path) == ref["image_sha256"], f"hash mismatch {rel}", errors)
            validate_source_pdf(ref, rel, f"{kind} image")
            if kind == "question":
                check(
                    not ref.get("contains_solution_or_rubric", True),
                    f"question manifest asset is not certified solver-safe: {rel}",
                    errors,
                )
                check(
                    "solutions" not in str(ref.get("source_pdf", "")),
                    f"question manifest asset points to solution material: {rel}",
                    errors,
                )
            else:
                check(ref.get("gold_only") is True, f"solution image is not gold-only: {rel}", errors)
                check(ref.get("must_not_be_sent_to_solver") is True, f"solution image lacks solver exclusion: {rel}", errors)

    t43 = next((r for r in rows if r["subquestion_id"] == "4.3"), None)
    check(t43 is not None, "T4.3 missing", errors)
    if t43:
        criteria = t43["grading_rubric"]["criteria"]
        check(
            [c["criterion_id"] for c in criteria]
            == [
                "icho2026_4.3_mass_numbers_or_sum",
                "icho2026_4.3_elements",
                "icho2026_4.3_fission_equation",
            ],
            "T4.3 normalized scoring components are missing or reordered",
            errors,
        )
        check(
            [c["max_score"] for c in criteria] == [1.2, 1.2, 0.6],
            "T4.3 normalized scoring maxima are incorrect",
            errors,
        )
        check(
            "Reaction (a)" not in t43["official_solution"],
            "T4.3 gold transcription still contains T4.2 material",
            errors,
        )
        check(
            all(ref["source_pdf_page"] == 3 for ref in t43["official_solution_image_refs"]),
            "T4.3 item-level gold images are not restricted to solution page 3",
            errors,
        )

    t49 = next((r for r in rows if r["subquestion_id"] == "4.9"), None)
    check(t49 is not None, "T4.9 missing", errors)
    if t49:
        policy = t49.get("adjudication_policy") or {}
        check(policy.get("policy_id") == "icho2026_t4_9_fallback_and_printed_mass_typo_v2", "T4.9 adjudication policy missing", errors)
        fallback = policy.get("official_fallback") or {}
        check("200 MeV" in fallback.get("printed_rule", ""), "T4.9 official fallback rule is missing", errors)
        check("3.92e24" in json.dumps(fallback), "T4.9 fallback fission result is missing", errors)
        check("5.15 kg" in json.dumps(fallback), "T4.9 fallback mass result is missing", errors)
        check(
            all("200 MeV" in c["description"] for c in t49["grading_rubric"]["criteria"]),
            "T4.9 normalized criteria do not both preserve the full-credit fallback",
            errors,
        )
        check("5.62" in policy.get("incorrect_value_policy", ""), "T4.9 5.62 handling is not explicit", errors)
        check("Do not award full credit" in policy.get("incorrect_value_policy", ""), "T4.9 typo policy does not prevent hidden-solution leakage into adjudication", errors)
        check("1.67" in t49["official_solution"], "T4.9 raw official transcription does not preserve 1.67", errors)
        check("5.56" in t49["official_solution"], "T4.9 raw official transcription does not preserve 5.56", errors)

    by_sid = {row["subquestion_id"]: row for row in rows}
    targeted = {
        "6.2": ("radicals missing or incor-", "rect."),
        "6.5": ("Error carry forward", "chemistry is sensible"),
        "6.6": ("two t-Bu", "rest of the parts", "at most once"),
        "8.1": ("fully correct reaction equation",),
        "8.2": ("shown above",),
        "8.5": ("calculating", "2.3"),
        "8.8": ("not be negative",),
    }
    for sid, fragments in targeted.items():
        rubric_text = by_sid[sid]["official_rubric"]
        for fragment in fragments:
            check(fragment in rubric_text, f"T{sid} rubric lost official fragment {fragment!r}", errors)
    check(
        bool(by_sid["8.1"]["official_color_channel_transcription"]),
        "T8.1 official colored transcription is empty",
        errors,
    )
    t84_criteria = by_sid["8.4"]["grading_rubric"]["criteria"]
    check(len(t84_criteria) == 7, "T8.4 must have seven species criteria", errors)
    check(
        abs(sum(float(c["max_score"]) for c in t84_criteria) - 29) < 1e-9,
        "T8.4 criterion maxima do not sum to 29",
        errors,
    )
    t84_logic = "\n".join(c["description"] for c in t84_criteria) + "\n" + by_sid["8.4"]["grading_rubric"]["judge_instruction"]
    for fragment in ("16 answer-sheet fields", "OS=+3", "CN=6", "CN=5", "OS=+2", "VE=16", "double punishment"):
        check(fragment in t84_logic, f"T8.4 normalized rubric lacks {fragment!r}", errors)
    for sid in ["4.9", *[f"6.{i}" for i in range(1, 8)], *[f"8.{i}" for i in range(1, 11)]]:
        check(
            by_sid[sid]["grading_rubric"].get("criteria_normalization")
            == "curated_point_units_with_official_source_provenance",
            f"T{sid} is not marked as curated rubric normalization",
            errors,
        )

    report = {
        "schema_version": "icho2026_canonical_validation_v1",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": {
            "item_count": len(rows),
            "raw_point_total": sum(r["max_raw_points"] for r in rows),
            "problem_weight_total": sum(meta[2] for meta in EXPECTED_PROBLEMS.values()),
            "visual_dependency_item_count": sum(bool(r["visual_dependency"]) for r in rows),
            "staticized_complete_count": sum(bool(r["question_text_staticized_complete"]) for r in rows),
            "human_visual_audited_count": sum(bool(r["human_visual_audited"]) for r in rows),
            "unique_question_image_count": len(question_refs),
            "unique_solution_image_count": len(solution_refs),
            "item_referenced_question_image_count": len(referenced_question_refs),
            "item_referenced_solution_image_count": len(referenced_solution_refs),
            "solver_solution_path_or_gold_marker_hits": sum(
                1
                for row in rows
                for marker in (
                    "theoretical-exam-solutions/", "OFFICIAL SOURCE-COLORED ANSWER",
                    "OFFICIAL SCORING-RULE TRANSCRIPTION", "official_solution_image_paths",
                )
                if marker in row["question_text_native"] + row["question_text_staticized"]
            ),
            "all_declared_image_hashes_match": not any("hash mismatch" in e for e in errors),
            "all_declared_source_pdf_hashes_match": not any("source PDF hash mismatch" in e for e in errors),
            "unique_source_pdf_hashes_verified": len(source_pdf_hash_cache),
            "t4_9_explicit_adjudication_policy": bool(t49 and t49.get("adjudication_policy")),
            "production_question_image_assets_complete_count": sum(
                bool(r.get("question_image_assets")) and
                len(r["question_image_assets"]) == len(r["question_image_paths"])
                for r in rows
            ),
            "production_grading_rubric_exact_sum_count": sum(
                bool((r.get("grading_rubric") or {}).get("criteria")) and
                abs(sum(float(c["max_score"]) for c in r["grading_rubric"]["criteria"]) - float(r["max_score"])) < 1e-9
                for r in rows
            ),
            "grading_criterion_count": sum(
                len((r.get("grading_rubric") or {}).get("criteria") or [])
                for r in rows
            ),
            "curated_rubric_item_count": sum(
                (r.get("grading_rubric") or {}).get("criteria_normalization")
                == "curated_point_units_with_official_source_provenance"
                for r in rows
            ),
            "curated_criterion_source_provenance_complete": not any(
                "criterion references unknown official rule" in e
                or "curated criterion lacks source provenance" in e
                for e in errors
            ),
            "no_model_or_api_calls": True,
        },
    }
    (OUT / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)
    return report


if __name__ == "__main__":
    validate()
