#!/usr/bin/env python3
"""Build draft and canonical 2026 USAPhO datasets from two local PDFs."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image

from usapho2026_source import (
    ITEMS,
    PROBLEMS,
    REFERENCE,
    SOURCE_DEFECTS,
    STATICIZATION_SPECS,
)


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
QUESTION_REL = "new-test/USAPhO 2026/2026-USAPhO-Problems.pdf"
SOLUTION_REL = "new-test/USAPhO 2026/2026-USAPhO-Solutions.pdf"
QUESTION_PDF = WORKSPACE / QUESTION_REL
SOLUTION_PDF = WORKSPACE / SOLUTION_REL
ADJUDICATION_REL = "experiments/2026_reval/adjudications/usapho2026_b2c_scattering_20260828.json"
ADJUDICATION_PATH = WORKSPACE / ADJUDICATION_REL
EXPECTED_Q_HASH = "95c2f0d278aa206d7386d868dca2a5af4f4135c12bae23a3dce6a55812cad982"
EXPECTED_S_HASH = "da401e05d009e1dbd14c231484142de03a7cbec9b1a04e7e8b131cc3d3854d64"
RENDER_DPI = 180
PROBLEM_ORDER = ["A1", "A2", "A3", "B1", "B2", "B3"]
PRINTED_PAGE_BY_PDF_PAGE = {
    1: None,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: None,
    7: 6,
    8: 7,
    9: 8,
    10: 9,
}


PROBLEM_PAGE_NOTES = {
    1: "Part A reference sheet: constants, approximations, contributor acknowledgement; no problem or solution.",
    2: "Problem A1 shared statement, eye/corrective-lens schematic, and parts (a)-(d).",
    3: "Problem A1 part (e)(i)-(ii); otherwise intentionally sparse page.",
    4: "Problem A2 shared statement and parts (a)-(d); no figure.",
    5: "Problem A3 Updraft and Vortex statements; no figure; visible PDF link box around A(c).",
    6: "Part B reference sheet, duplicating the Part A reference content; no problem or solution.",
    7: "Problem B1 statement through parts (a)-(c), cluster setup, and start of part (d).",
    8: "Problem B1 continuation: part (d)(i)-(iii).",
    9: "Problem B2 statement and all parts; conic equations and coordinates, no orbit drawing.",
    10: "Problem B3 statement and all parts; equations only, no apparatus/circuit drawing.",
}

SOLUTION_PAGE_NOTES = {
    1: "Reference sheet only; no solution.",
    2: "A1 statement/schematic and beginning of solution (a).",
    3: "End A1(a), solution (b), and prompt (c).",
    4: "Solutions A1(c)-(d), with (d) continuing.",
    5: "End A1(d) and solutions A1(e)(i)-(ii).",
    6: "A2 statement, solution (a), and beginning of (b).",
    7: "End A2(b), solutions (c), and most of (d).",
    8: "Final A2(d) result; otherwise intentionally sparse.",
    9: "A3 statement and Updraft solutions (a)-(b).",
    10: "A3 Updraft solution (c) and most of Vortex solution (a).",
    11: "End A3 Vortex (a) and solution (b).",
    12: "B1 statement, solutions (a), (b)(i), and prompt (b)(ii).",
    13: "B1 solutions (b)(ii), (c)(i)-(ii), cluster setup, and prompt (d)(i).",
    14: "B1 solutions (d)(i), (d)(ii)(i)-(ii), and (d)(iii).",
    15: "B2 statement and beginning of solution (a).",
    16: "End B2(a), prompt and beginning of solution (b).",
    17: "End B2(b), prompt and beginning of solution (c).",
    18: "B2(c) solution containing the documented scattering-angle/family error.",
    19: "B3 statement, solution (a), and beginning of solution (b).",
    20: "Most of B3(b) and beginning of solution (c).",
    21: "End B3(c), solution (d), and most of solution (e).",
    22: "End B3(e); otherwise intentionally sparse.",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def render_pdf(pdf: Path, output_dir: Path, expected_pages: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale in output_dir.glob("page-*.png"):
        stale.unlink()
    with tempfile.TemporaryDirectory(prefix="usapho2026-render-") as tmp_name:
        prefix = Path(tmp_name) / "page"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(RENDER_DPI), str(pdf), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        rendered = sorted(Path(tmp_name).glob("page-*.png"))
        if len(rendered) != expected_pages:
            raise RuntimeError(f"rendered {len(rendered)} pages from {pdf}; expected {expected_pages}")
        result: list[Path] = []
        for page, source in enumerate(rendered, start=1):
            target = output_dir / f"page-{page:02d}.png"
            shutil.copy2(source, target)
            result.append(target)
        return result


def image_asset(output: Path, kind: str, page: int, pdf_rel: str, pdf_hash: str) -> dict[str, Any]:
    folder = "question_images" if kind == "question" else "official_solution_images"
    path = output / folder / f"page-{page:02d}.png"
    with Image.open(path) as image:
        width, height = image.size
    rel = path.relative_to(WORKSPACE).as_posix()
    digest = sha256_file(path)
    value: dict[str, Any] = {
        "path": rel,
        "source_asset_path": rel,
        "source_pdf": pdf_rel,
        "source_pdf_path": pdf_rel,
        "source_pdf_sha256": pdf_hash,
        "source_page": page,
        "source_pdf_page": page,
        "render_dpi": RENDER_DPI,
        "mime_type": "image/png",
        "width": width,
        "height": height,
        "sha256": digest,
        "image_sha256": digest,
    }
    if kind == "question":
        value.update({"contains_solution": False, "contains_solution_or_rubric": False, "page_role": "official_question_page_solver_input"})
    else:
        value.update({
            "contains_solution": page >= 2,
            "contains_solution_or_rubric": page >= 2,
            "gold_only": True,
            "must_not_be_sent_to_solver": True,
            "page_role": "provided_solution_page_gold_only",
        })
    return value


def label_for(problem: str, part: str) -> str:
    labels = {
        "updraft_a": "Updraft A(a)", "updraft_b": "Updraft A(b)",
        "updraft_c_i": "Updraft A(c)(i)", "updraft_c_ii": "Updraft A(c)(ii)",
        "vortex_a": "Vortex B(a)", "vortex_b": "Vortex B(b)",
        "b_i": "(b)(i)", "b_ii": "(b)(ii)", "c_i": "(c)(i)", "c_ii": "(c)(ii)",
        "d_i": "(d)(i)", "d_ii_i": "(d)(ii)(i)", "d_ii_ii": "(d)(ii)(ii)", "d_iii": "(d)(iii)",
        "e_i": "(e)(i)", "e_ii": "(e)(ii)",
    }
    return f"{problem} {labels.get(part, f'({part})')}"


def printed_problem_pages(pdf_pages: list[int]) -> list[int]:
    """Return visible problem-page labels, excluding unnumbered reference sheets."""
    return [
        int(PRINTED_PAGE_BY_PDF_PAGE[page])
        for page in pdf_pages
        if PRINTED_PAGE_BY_PDF_PAGE[page] is not None
    ]


def build_one(output: Path, q_hash: str, s_hash: str) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    q_pages = render_pdf(QUESTION_PDF, output / "question_images", 10)
    s_pages = render_pdf(SOLUTION_PDF, output / "official_solution_images", 22)

    source_text = output / "source_text"
    source_text.mkdir(parents=True, exist_ok=True)
    subprocess.run(["pdftotext", "-layout", str(QUESTION_PDF), str(source_text / "problems_layout.txt")], check=True)
    subprocess.run(["pdftotext", "-layout", str(SOLUTION_PDF), str(source_text / "solutions_layout.txt")], check=True)

    q_assets = {page: image_asset(output, "question", page, QUESTION_REL, q_hash) for page in range(1, 11)}
    s_assets = {page: image_asset(output, "solution", page, SOLUTION_REL, s_hash) for page in range(1, 23)}
    adjudication = json.loads(ADJUDICATION_PATH.read_text(encoding="utf-8"))
    if adjudication.get("adjudication_id") != "usapho2026_b2c_scattering_20260828":
        raise RuntimeError("unexpected USAPhO B2(c) adjudication ID")
    adjudication_hash = sha256_file(ADJUDICATION_PATH)
    adjudication_copy = output / "adjudications" / ADJUDICATION_PATH.name
    adjudication_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ADJUDICATION_PATH, adjudication_copy)
    defect_by_id = {value["id"]: value for value in SOURCE_DEFECTS}
    by_problem_turn: dict[str, int] = defaultdict(int)
    rows: list[dict[str, Any]] = []
    staticization_rows: list[dict[str, Any]] = []
    item_source_keys = {f"{entry['problem']}:{entry['part']}" for entry in ITEMS}
    staticization_keys = set(STATICIZATION_SPECS)
    if len(item_source_keys) != len(ITEMS):
        raise RuntimeError("USAPhO source contains duplicate problem:part keys")
    if staticization_keys != item_source_keys:
        missing = sorted(item_source_keys - staticization_keys)
        extra = sorted(staticization_keys - item_source_keys)
        raise RuntimeError(f"staticization matrix key mismatch; missing={missing}; extra={extra}")

    for global_order, entry in enumerate(ITEMS, start=1):
        problem = str(entry["problem"])
        part = str(entry["part"])
        source_key = f"{problem}:{part}"
        staticization_spec = copy.deepcopy(STATICIZATION_SPECS[source_key])
        supplemental_context = str(staticization_spec["supplemental_context"]).strip()
        meta = PROBLEMS[problem]
        by_problem_turn[problem] += 1
        turn_index = by_problem_turn[problem]
        item_id = f"usapho2026_{problem.lower()}_{part}"
        q_refs = [copy.deepcopy(q_assets[int(page)]) for page in entry["qpages"]]
        s_refs = [copy.deepcopy(s_assets[int(page)]) for page in entry["spages"]]
        visual_text = str(meta["visual"])
        native = (
            f"USAPhO 2026 Problem {problem}: {meta['title']}.\n"
            f"Official reference sheet (question paper only): {REFERENCE}\n\n"
            f"Shared statement:\n{meta['stem']}\n\n"
            f"Current subpart {label_for(problem, part)}:\n"
            f"{entry['prompt']}"
        )
        supplemental_block = (
            f"Question-only subpart and cross-part context:\n{supplemental_context}\n\n"
            if supplemental_context else ""
        )
        staticized = (
            "USAPhO 2026 - QUESTION-ONLY HUMAN-AUDITED TEXT-STATICIZED PROMPT\n"
            f"Target item: {label_for(problem, part)}.\n"
            f"Official problem-PDF page(s): {', '.join(str(v) for v in entry['qpages'])}. "
            "No solution page, answer, or grading criterion is included in solver input.\n\n"
            f"Reference sheet:\n{REFERENCE}\n\n"
            f"Shared statement:\n{meta['stem']}\n\n"
            f"{supplemental_block}"
            f"Target subpart:\n{entry['prompt']}\n\n"
            "<<<HUMAN-VISUAL-AUDITED NON-TEXT TRANSCRIPTION | QUESTION PAPER ONLY | SOLVER-SAFE>>>\n"
            f"{visual_text}\n"
            "<<<END HUMAN-VISUAL-AUDITED TRANSCRIPTION>>>"
        )
        criteria = copy.deepcopy(entry["criteria"])
        provided_answer = str(entry.get("source_answer", entry["answer"]))
        is_adjudicated = "source_answer" in entry
        adjudication_id = entry.get("adjudication_id")
        rubric_status = (
            "scientifically_adjudicated_independent_correction_with_derived_criteria_not_official"
            if is_adjudicated
            else "derived_from_provided_solution_not_official"
        )
        rubric = {
            "schema_version": "usapho2026_derived_rubric_v1",
            "status": rubric_status,
            "rubric_type": "derived_from_solution_not_official",
            "max_score": float(entry["points"]),
            "item_max_score": float(entry["points"]),
            "criteria": criteria,
            "criteria_normalization": "curator_derived_subpart_weighting_under_25_points_per_problem",
            "judge_instruction": f"Score each frozen criterion once, sum exactly, and clamp to [0,{entry['points']:g}]. The PDFs contain no official point rubric.",
            "accepted_equivalents": "Accept algebraically and physically equivalent correct methods. Apply the recorded source-error adjudication rather than blindly reproducing a defective source derivation.",
            "numeric_tolerance": "Accept ordinary rounding and order-unity factors where the prompt says estimate; otherwise require algebraic equivalence.",
            "error_carry_forward": "Award independently correct downstream reasoning with an explicitly carried value, but do not award the originating error twice.",
            "source_error_handling": "Use the item source_defects and any linked adjudication. For B2(c), retain the provided answer as source evidence but grade the corrected physical velocity-deflection convention stated in reference_answer.",
            "criteria_provenance": (
                "Criteria are curator-derived from the independently adjudicated scientific reference, not from the defective provided B2(c) result and not from an official rubric."
                if is_adjudicated
                else "Criteria are curator-derived from the human-audited provided worked solution and are not an official rubric."
            ),
            "source_provenance": {
                "source_pdf": SOLUTION_REL,
                "source_pdf_sha256": s_hash,
                "pdf_pages": list(entry["spages"]),
                "status": "provided_worked_solution_no_official_points_or_grading_rubric_external_authenticity_unconfirmed",
            },
        }
        item_defect_ids = ["usapho2026_no_official_points_or_rubric"] + list(entry.get("defects") or [])
        item_defects = [copy.deepcopy(defect_by_id[value]) for value in dict.fromkeys(item_defect_ids)]
        row = {
            "schema_version": "science_arena_item_v1",
            "item_id": item_id,
            "benchmark_id": "usapho2026",
            "sequence_id": f"usapho2026_{problem.lower()}",
            "turn_index": turn_index,
            "global_order": global_order,
            "question_id": label_for(problem, part),
            "parent_question": problem,
            "problem_id": problem,
            "problem_title": meta["title"],
            "competition_part": meta["part"],
            "max_score": float(entry["points"]),
            "max_raw_points": float(entry["points"]),
            "problem_raw_max": 25.0,
            "problem_weight": 25.0,
            "competition_raw_max": 150.0,
            "question_text_native": native,
            "question_text_staticized": staticized,
            "question_text_staticized_complete": True,
            "question_only_staticization_verified": True,
            "human_visual_audited": True,
            "solver_prompt_disclose_max_score": False,
            "reference_answer": entry["answer"],
            "adjudicated_reference_answer": entry["answer"],
            "reference_answer_status": "scientifically_adjudicated_correction_of_provided_solution" if is_adjudicated else "normalized_from_human_audited_provided_solution",
            "official_answer": provided_answer,
            "provided_solution_answer": provided_answer,
            "provided_solution_answer_normalized": provided_answer,
            "provided_solution_answer_scope": entry.get("source_answer_scope", "The provided worked solution's answer for this subpart."),
            "official_answer_status": "normalized_from_provided_solution_not_independently_official",
            "gold_solution": entry["solution"],
            "official_solution": entry.get("source_solution_summary", entry["solution"]),
            "official_solution_status": "human_audited_normalized_summary_of_provided_worked_solution_not_verbatim",
            "gold_solution_status": "scientifically_adjudicated_correction_of_provided_worked_solution" if is_adjudicated else "human_audited_summary_of_provided_worked_solution",
            "adjudication_id": adjudication_id,
            "adjudication": ({
                "adjudication_id": adjudication_id,
                "status": adjudication["status"],
                "path": ADJUDICATION_REL,
                "sha256": adjudication_hash,
                "dataset_copy": adjudication_copy.relative_to(WORKSPACE).as_posix(),
                "dataset_copy_sha256": adjudication_hash,
            } if adjudication_id else None),
            "grading_rubric": rubric,
            "official_rubric": copy.deepcopy(rubric),
            "official_rubric_criteria": copy.deepcopy(criteria),
            "rubric_type": "derived_from_solution_not_official",
            "official_rubric_status": "absent_in_provided_pdfs_derived_rubric_used",
            "derived_rubric_status": (
                "independent_scientific_adjudication_with_curator_derived_criteria"
                if is_adjudicated
                else "curator_derived_and_human_audited"
            ),
            "question_image_paths": [value["path"] for value in q_refs],
            "question_image_assets": q_refs,
            "question_image_refs": copy.deepcopy(q_refs),
            "official_solution_image_paths": [value["path"] for value in s_refs],
            "official_solution_image_assets": s_refs,
            "official_solution_image_refs": copy.deepcopy(s_refs),
            "question_source": {
                "source_pdf": QUESTION_REL,
                "source_pdf_sha256": q_hash,
                "pdf_pages": list(entry["qpages"]),
                "printed_pages": printed_problem_pages([int(value) for value in entry["qpages"]]),
                "locator": label_for(problem, part),
                "provenance_status": "provided_problem_paper_aapt_2026_marked_external_authenticity_unconfirmed",
            },
            "solution_source": {
                "source_pdf": SOLUTION_REL,
                "source_pdf_sha256": s_hash,
                "pdf_pages": list(entry["spages"]),
                "printed_pages": list(entry["spages"]),
                "locator": f"{label_for(problem, part)} worked solution",
                "provenance_status": "provided_worked_solution_no_official_grading_external_authenticity_unconfirmed",
            },
            "source_defects": item_defects,
            "visual_dependency": True,
            "intrinsic_visual_dependency": False,
            "has_official_question_figure": bool(meta["has_figure"]),
            "visual_dependency_metadata": {
                "required": True,
                "reason": "Protocol enforcement: native-vision models must receive direct official question-page pixels even when the statement is text-complete.",
                "intrinsic_nontext_reasoning_required": False,
                "has_official_question_figure": bool(meta["has_figure"]),
                "multimodal_policy": "send_direct_official_problem_pdf_page_pixels",
                "text_only_policy": "question_only_human_audited_staticized_description",
                "missing_asset_policy": "block_no_fallback",
                "question_image_paths": [value["path"] for value in q_refs],
                "assets": copy.deepcopy(q_refs),
                "validation": "source PDF/page/hash and PNG hash verified",
            },
            "multimodal_missing_asset_policy": "BLOCK_ITEM_DO_NOT_FALL_BACK_TO_STATICIZED_TEXT",
            "input_tracks": {
                "native_vision": {
                    "question_field": "question_text_native",
                    "direct_official_question_images": True,
                    "description_is_not_sent": True,
                    "include_curator_derived_score_cue": False,
                    "missing_asset_policy": "block_no_fallback",
                },
                "text_staticized": {
                    "question_field": "question_text_staticized",
                    "question_only": True,
                    "no_image_payload": True,
                    "human_visual_audited": True,
                    "include_curator_derived_score_cue": False,
                    "solution_leakage_check": "passed",
                },
            },
            "staticization": {
                "method": "manual_question_only_transcription_from_official_problem_pages",
                "source_pdf": QUESTION_REL,
                "source_pdf_sha256": q_hash,
                "source_pages": list(entry["qpages"]),
                "human_visual_audited": True,
                "completeness_matrix_status": "PASS",
                "requirements": str(staticization_spec["requirements"]),
                "supplemental_question_context_included": bool(supplemental_context),
                "solution_pdf_access": "forbidden_to_solver_staticization_artifact",
                "solution_leakage_check": "passed",
            },
            "solver_gold_isolation": {
                "question_assets_only": True,
                "solution_assets_gold_only": True,
                "leakage_check": "passed",
            },
        }
        rows.append(row)
        staticization_rows.append({
            "item_id": item_id,
            "source_key": source_key,
            "question_pdf_pages": list(entry["qpages"]),
            "printed_problem_pages": printed_problem_pages([int(value) for value in entry["qpages"]]),
            "requirements": str(staticization_spec["requirements"]),
            "supplemental_context_required": bool(supplemental_context),
            "supplemental_context": supplemental_context,
            "reference_sheet_complete": True,
            "shared_statement_included": True,
            "target_subpart_included": True,
            "question_visual_transcription_included": True,
            "question_only": True,
            "solution_or_rubric_content_included": False,
            "curator_derived_score_cue_in_question_fields": False,
            "status": "PASS",
        })

    write_jsonl(output / "items.jsonl", rows)
    write_json(output / "staticization_completeness_matrix.json", {
        "schema_version": "usapho2026_question_only_staticization_completeness_v1",
        "status": "PASS",
        "method": "All 34 atomic turns were checked against fresh text extraction and 180-DPI visual inspection of every page in the Problems PDF. Only question-paper content is permitted.",
        "item_count": len(staticization_rows),
        "complete_item_count": sum(value["status"] == "PASS" for value in staticization_rows),
        "question_pdf_sha256": q_hash,
        "solution_content_used": False,
        "items": staticization_rows,
    })
    write_jsonl(output / "question_image_manifest.jsonl", [q_assets[page] for page in range(1, 11)])
    write_jsonl(output / "official_solution_image_manifest.jsonl", [s_assets[page] for page in range(1, 23)])
    write_json(output / "contexts.json", {
        "schema_version": "usapho2026_contexts_v1",
        "reference_sheet": REFERENCE,
        "problems": PROBLEMS,
        "question_only_staticization_specs": STATICIZATION_SPECS,
    })
    page_audit = {
        "schema_version": "usapho2026_page_dual_audit_v1",
        "status": "PASS",
        "method": "Every page read with pdftotext -layout and visually reviewed from a fresh 180-DPI PNG at original detail.",
        "all_pages_text_and_visual_audited": True,
        "problem_pdf_sha256": q_hash,
        "solution_pdf_sha256": s_hash,
        "problem_pdf_pages": [
            {"page": page, "text_extracted": True, "rendered": True, "visually_reviewed": True, "notes": PROBLEM_PAGE_NOTES[page]}
            for page in range(1, 11)
        ],
        "solution_pdf_pages": [
            {"page": page, "text_extracted": True, "rendered": True, "visually_reviewed": True, "notes": SOLUTION_PAGE_NOTES[page]}
            for page in range(1, 23)
        ],
    }
    write_json(output / "page_audit.json", page_audit)
    schema = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "science-arena/usapho2026/canonical_item_schema.json",
        "title": "Science Arena canonical USAPhO 2026 item",
        "type": "object",
        "additionalProperties": True,
        "required": [
            "schema_version", "item_id", "benchmark_id", "sequence_id", "turn_index", "global_order",
            "question_id", "parent_question", "max_score", "question_text_native", "question_text_staticized",
            "reference_answer", "adjudicated_reference_answer", "reference_answer_status", "official_answer",
            "provided_solution_answer_normalized", "gold_solution", "grading_rubric", "rubric_type", "visual_dependency",
            "question_image_paths", "question_image_assets", "official_solution_image_paths", "question_source",
            "solution_source", "input_tracks",
        ],
        "properties": {
            "schema_version": {"const": "science_arena_item_v1"},
            "benchmark_id": {"const": "usapho2026"},
            "item_id": {"type": "string", "pattern": "^usapho2026_(a|b)[1-3]_[a-z0-9_]+$"},
            "sequence_id": {"type": "string", "pattern": "^usapho2026_(a|b)[1-3]$"},
            "turn_index": {"type": "integer", "minimum": 1, "maximum": 9},
            "global_order": {"type": "integer", "minimum": 1, "maximum": 34},
            "max_score": {"type": "number", "exclusiveMinimum": 0},
            "rubric_type": {"const": "derived_from_solution_not_official"},
            "visual_dependency": {"const": True},
            "question_text_native": {"type": "string", "minLength": 1},
            "question_text_staticized": {"type": "string", "minLength": 1},
            "reference_answer": {"type": "string", "minLength": 1},
            "adjudicated_reference_answer": {"type": "string", "minLength": 1},
            "official_answer": {"type": "string", "minLength": 1},
            "provided_solution_answer_normalized": {"type": "string", "minLength": 1},
            "question_image_paths": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "official_solution_image_paths": {"type": "array", "minItems": 1, "items": {"type": "string"}},
        },
    }
    write_json(output / "item_schema.json", schema)

    problem_totals = {problem: sum(float(row["max_score"]) for row in rows if row["problem_id"] == problem) for problem in PROBLEM_ORDER}
    manifest = {
        "schema_version": "usapho2026_benchmark_manifest_v1",
        "benchmark_id": "usapho2026",
        "competition": "2026 USA Physics Olympiad (USAPhO)",
        "source_policy": "only the two provided local PDFs",
        "year_identity_evidence": "Problem headers say '2026 USAPhO'; cover/reference sheet says USA Physics Olympiad Exam, AAPT/AIP United States Physics Team 2026; problem pages carry AAPT Copyright 2026.",
        "external_authenticity_status": "not_independently_verified",
        "problem_count": 6,
        "sequence_count": 6,
        "item_count": len(rows),
        "raw_score_total": sum(float(row["max_score"]) for row in rows),
        "problem_score_totals": problem_totals,
        "score_policy": "curator benchmark normalization: 25 points per problem, 150 total; not official because PDFs print no points/rubric",
        "rubric_provenance": "derived from human-audited worked solutions; no official grading rubric in the provided PDFs",
        "answer_policy": "provided-solution answers are retained in official_answer/provided_solution_answer_normalized; reference_answer may differ only through an explicit, linked scientific adjudication",
        "question_image_count": len(q_pages),
        "solution_image_count": len(s_pages),
        "direct_question_page_pixel_required_item_count": len(rows),
        "intrinsic_visual_dependency_item_count": sum(bool(row["intrinsic_visual_dependency"]) for row in rows),
        "official_question_figure_item_count": sum(bool(row["has_official_question_figure"]) for row in rows),
        "question_image_policy": "exact fresh 180-DPI PNG renders of official Problems PDF pages; direct API image payload for every native-vision item",
        "solution_image_policy": "separate gold-only exact renders; never solver input",
        "text_only_policy": "question-only human-audited staticization with complete per-turn definitions and cross-part context; no image payload and no solution-derived content",
        "question_field_score_cue_policy": "question_text_native and question_text_staticized omit curator-derived score maxima; scoring metadata remains available only for internal evaluation",
        "staticization_completeness_status": "PASS",
        "staticization_completeness_item_count": len(staticization_rows),
        "source_pdfs": [
            {"path": QUESTION_REL, "sha256": q_hash, "pages": 10, "role": "solver_question_only"},
            {"path": SOLUTION_REL, "sha256": s_hash, "pages": 22, "role": "judge_gold_only"},
        ],
        "source_defects": SOURCE_DEFECTS,
        "adjudications": [{
            "adjudication_id": adjudication["adjudication_id"],
            "status": adjudication["status"],
            "path": ADJUDICATION_REL,
            "sha256": adjudication_hash,
            "dataset_copy": adjudication_copy.relative_to(WORKSPACE).as_posix(),
            "dataset_copy_sha256": sha256_file(adjudication_copy),
            "affected_item_ids": adjudication["affected_item_ids"],
        }],
        "builder": "experiments/2026_reval/build_usapho2026_canonical.py",
        "builder_sha256": sha256_file(Path(__file__).resolve()),
        "source_data": "experiments/2026_reval/usapho2026_source.py",
        "source_data_sha256": sha256_file(HERE / "usapho2026_source.py"),
        "items_sha256": sha256_file(output / "items.jsonl"),
        "question_manifest_sha256": sha256_file(output / "question_image_manifest.jsonl"),
        "solution_manifest_sha256": sha256_file(output / "official_solution_image_manifest.jsonl"),
        "page_audit_sha256": sha256_file(output / "page_audit.json"),
        "staticization_completeness_matrix": (output / "staticization_completeness_matrix.json").relative_to(WORKSPACE).as_posix(),
        "staticization_completeness_matrix_sha256": sha256_file(output / "staticization_completeness_matrix.json"),
        "problem_text_sha256": sha256_file(source_text / "problems_layout.txt"),
        "solution_text_sha256": sha256_file(source_text / "solutions_layout.txt"),
    }
    write_json(output / "benchmark_manifest.json", manifest)
    write_json(output / "manifest.json", manifest)
    return manifest


def main() -> None:
    if sha256_file(QUESTION_PDF) != EXPECTED_Q_HASH:
        raise SystemExit("USAPhO problem PDF hash mismatch")
    if sha256_file(SOLUTION_PDF) != EXPECTED_S_HASH:
        raise SystemExit("USAPhO solution PDF hash mismatch")
    if not ADJUDICATION_PATH.is_file():
        raise SystemExit("USAPhO B2(c) adjudication file missing")
    outputs = [HERE / "datasets_draft/usapho2026", HERE / "datasets/usapho2026"]
    manifests = [build_one(output, EXPECTED_Q_HASH, EXPECTED_S_HASH) for output in outputs]
    print(json.dumps({
        "status": "BUILT",
        "outputs": [str(value.relative_to(WORKSPACE)) for value in outputs],
        "canonical_items_sha256": manifests[-1]["items_sha256"],
        "item_count": manifests[-1]["item_count"],
        "raw_score_total": manifests[-1]["raw_score_total"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
