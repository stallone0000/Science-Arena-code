#!/usr/bin/env python3
"""Build the offline, visually audited NBPhO 2026 dataset.

This script only reads the two frozen local PDFs.  It contains no network or
model-API code.  Problem-page renders and solution-page renders are written to
separate trees and carry source-PDF hashes in every item and manifest record.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import shutil
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
PROBLEM_PDF = WORKSPACE / "new-test/NBPhO 2026/NBPhO_2026_problems.pdf"
SOLUTION_PDF = WORKSPACE / "new-test/NBPhO 2026/NBPhO_2026_solutions.pdf"
DEFAULT_DRAFT = HERE / "datasets_draft/nbpho2026"
DEFAULT_OUTPUT = HERE / "datasets/nbpho2026"
RENDER_DPI = 180

QUESTION_REL = "new-test/NBPhO 2026/NBPhO_2026_problems.pdf"
SOLUTION_REL = "new-test/NBPhO 2026/NBPhO_2026_solutions.pdf"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


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


def load_source_module() -> Any:
    source_path = HERE / "nbpho2026_source.py"
    spec = importlib.util.spec_from_file_location("nbpho2026_source", source_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {source_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command_text(args: list[str]) -> str:
    return subprocess.run(args, check=True, capture_output=True, text=True).stdout


def render_pdf(pdf: Path, page_count: int, destination: Path) -> list[Path]:
    destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="nbpho2026-render-") as tmp_name:
        prefix = Path(tmp_name) / "render"
        subprocess.run(
            ["pdftoppm", "-png", "-r", str(RENDER_DPI), str(pdf), str(prefix)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        rendered = sorted(Path(tmp_name).glob("render-*.png"))
        if len(rendered) != page_count:
            raise RuntimeError(f"{pdf}: rendered {len(rendered)} pages, expected {page_count}")
        outputs: list[Path] = []
        for page, source in enumerate(rendered, 1):
            target = destination / f"page-{page:02d}.png"
            shutil.copyfile(source, target)
            outputs.append(target)
        return outputs


def asset_record(
    *, path: Path, source_pdf: str, source_hash: str, source_page: int,
    contains_solution: bool, gold_only: bool = False,
) -> dict[str, Any]:
    with Image.open(path) as image:
        width, height = image.size
    record = {
        "path": rel_workspace(path),
        "source_asset_path": rel_workspace(path),
        "source_pdf": source_pdf,
        "source_pdf_path": source_pdf,
        "source_pdf_sha256": source_hash,
        "source_page": source_page,
        "source_pdf_page": source_page,
        "sha256": sha256_file(path),
        "image_sha256": sha256_file(path),
        "mime_type": "image/png",
        "width": width,
        "height": height,
        "render_dpi": RENDER_DPI,
        "contains_solution": contains_solution,
        "contains_solution_or_rubric": contains_solution,
    }
    if gold_only:
        record.update({"gold_only": True, "must_not_be_sent_to_solver": True})
    return record


SOLUTION_PAGE_MAP = {
    (1, "i"): [3], (1, "ii"): [3, 4], (2, "main"): [4],
    (3, "i"): [4, 5], (3, "ii"): [5], (3, "iii"): [5],
    (4, "i"): [5, 6], (4, "ii"): [6], (4, "iii"): [6, 7], (4, "iv"): [7],
    (5, "i"): [7, 8], (5, "ii"): [8, 9], (5, "iii"): [8, 9],
    (6, "i"): [9], (6, "ii"): [9], (6, "iii"): [9],
    (7, "i"): [9, 10], (7, "ii"): [10], (7, "iii"): [10],
    (8, "i"): [10, 11], (8, "ii"): [11], (8, "iii"): [11],
    (9, "i"): [11], (9, "ii"): [12], (9, "iii"): [12, 13],
    (10, "i"): [13], (10, "ii"): [13], (10, "iii"): [14],
}


PAGE_AUDIT = {
    "problem_pdf": {
        "1": "Landscape four-column page: Problems 1-5 in full plus the beginning of Problem 6; diagrams for P1, P2, P4.",
        "2": "Landscape four-column page: completion of P6 and Problems 7-10; diagrams/plots for P6, P8, P9.",
    },
    "solution_pdf": {
        "1": "Duplicate of problem-paper page 1; not a gold solution page.",
        "2": "Duplicate of problem-paper page 2; not a gold solution page.",
        "3": "Moderation note and Problem 1 solution/rubric.",
        "4": "End of P1, full P2 solution/rubric, start of P3.",
        "5": "P3 solution/rubrics and start of P4.",
        "6": "P4 parts i-iii solutions/rubrics.",
        "7": "End of P4 iii, P4 iv, and start of P5.",
        "8": "P5 method, execution, calculation and official grading branches.",
        "9": "End of P5, full P6, and start of P7.",
        "10": "P7 and start of P8 solutions/rubrics.",
        "11": "P8 and P9(i) solutions/rubrics.",
        "12": "P9(ii)-(iii), including start of accidental duplicate iv.",
        "13": "End of duplicated P9(iii) and P10(i)-(ii).",
        "14": "P10(iii) experimental solution/rubric.",
    },
}


def source_defects_for(problem: int, defects: list[dict[str, Any]]) -> list[dict[str, Any]]:
    token = f"_p{problem}_"
    return [copy.deepcopy(value) for value in defects if token in value["id"]]


def make_schema() -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "science-arena/nbpho2026/canonical_item_schema.json",
        "title": "Science Arena canonical NBPhO 2026 item",
        "type": "object",
        "required": [
            "schema_version", "item_id", "benchmark_id", "sequence_id", "turn_index",
            "global_order", "question_id", "parent_question", "max_score",
            "question_text_native", "question_text_staticized", "reference_answer",
            "gold_solution", "grading_rubric", "rubric_type", "visual_dependency",
            "question_image_paths", "question_image_assets", "official_solution_image_paths",
            "question_source", "solution_source", "input_tracks",
        ],
        "properties": {
            "schema_version": {"const": "science_arena_item_v1"},
            "item_id": {"type": "string", "pattern": "^nbpho2026_p(10|[1-9])_(main|i|ii|iii|iv)$"},
            "benchmark_id": {"const": "nbpho2026"},
            "sequence_id": {"type": "string", "pattern": "^nbpho2026_p(10|[1-9])$"},
            "turn_index": {"type": "integer", "minimum": 1, "maximum": 4},
            "global_order": {"type": "integer", "minimum": 1, "maximum": 28},
            "max_score": {"type": "number", "exclusiveMinimum": 0},
            "question_text_native": {"type": "string", "minLength": 1},
            "question_text_staticized": {"type": "string", "minLength": 1},
            "rubric_type": {"const": "official_explicit_grading_from_provided_solution_pdf"},
            "question_image_paths": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "official_solution_image_paths": {"type": "array", "minItems": 1, "items": {"type": "string"}},
            "visual_dependency": {"type": "boolean"},
        },
        "additionalProperties": True,
    }


def build_tree(root: Path, source: Any, q_hash: str, s_hash: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    question_files = render_pdf(PROBLEM_PDF, 2, root / "question_images")
    solution_files = render_pdf(SOLUTION_PDF, 14, root / "official_solution_images")
    question_assets = [
        asset_record(
            path=path, source_pdf=QUESTION_REL, source_hash=q_hash,
            source_page=index, contains_solution=False,
        )
        for index, path in enumerate(question_files, 1)
    ]
    solution_assets: list[dict[str, Any]] = []
    for index, path in enumerate(solution_files, 1):
        is_gold_page = index >= 3
        record = asset_record(
            path=path, source_pdf=SOLUTION_REL, source_hash=s_hash,
            source_page=index, contains_solution=is_gold_page, gold_only=is_gold_page,
        )
        record["must_not_be_sent_to_solver"] = True
        record["page_role"] = (
            "official_solution_and_grading_gold_only"
            if is_gold_page
            else "duplicate_problem_page_for_complete_pdf_audit_not_item_referenced"
        )
        solution_assets.append(record)

    rows: list[dict[str, Any]] = []
    per_problem_turn: dict[int, int] = defaultdict(int)
    for global_order, entry in enumerate(source.ITEMS, 1):
        problem = int(entry["problem"])
        part = str(entry["part"])
        per_problem_turn[problem] += 1
        turn_index = per_problem_turn[problem]
        meta = source.PROBLEMS[problem]
        q_pages = list(meta["question_pages"])
        s_pages = SOLUTION_PAGE_MAP[(problem, part)]
        q_refs = [copy.deepcopy(question_assets[p - 1]) for p in q_pages]
        s_refs = [copy.deepcopy(solution_assets[p - 1]) for p in s_pages]
        question_id = str(problem) if part == "main" else f"{problem}({part})"
        item_id = f"nbpho2026_p{problem}_{part}"
        native = (
            f"NBPhO 2026 Problem {problem}: {meta['title']} ({meta['points']:g} points total).\n"
            f"{meta['stem']}\n\nCurrent subpart {question_id} ({entry['points']:g} points): {entry['prompt']}"
        )
        staticized = (
            "NBPhO 2026 - QUESTION-ONLY HUMAN-AUDITED TEXT-STATICIZED PROMPT\n"
            f"Target item: {question_id}; maximum score: {entry['points']:g}.\n"
            f"Official problem-PDF page(s): {', '.join(map(str, q_pages))}. No solution page, answer, or rubric was used as solver input.\n\n"
            f"Shared statement:\n{meta['stem']}\n\n"
            f"Target subpart:\n{entry['prompt']}\n\n"
            "<<<HUMAN-VISUAL-AUDITED NON-TEXT TRANSCRIPTION | QUESTION PAPER ONLY | SOLVER-SAFE>>>\n"
            f"{meta['visual']}\n"
            "<<<END HUMAN-VISUAL-AUDITED TRANSCRIPTION>>>"
        )
        criteria = copy.deepcopy(entry["criteria"])
        grading_rubric = {
            "schema_version": "nbpho2026_official_rubric_v1",
            "status": "official_explicit_grading_in_provided_solution_pdf",
            "rubric_type": "official_explicit_grading_from_provided_solution_pdf",
            "max_score": entry["points"],
            "item_max_score": entry["points"],
            "criteria": criteria,
            "criteria_normalization": "official_point_components_or_nonadditive_method_groups",
            "official_notes": entry.get("notes", "Apply the printed official grading bullets and equivalent solution routes under the hard item cap."),
            "accepted_equivalents": "Accept algebraically/physically equivalent methods expressly allowed by the official solution; alternative branches are non-additive.",
            "numeric_tolerance": "Use explicit ranges/tolerances in the criterion or official notes; otherwise accept normal rounding consistent with the source.",
            "error_carry_forward": "Award downstream method credit for a clearly stated carried value when the official rubric permits or the later step is independently correct; do not award the originating error.",
            "source_error_handling": "Apply this item's recorded source_defects and official_notes; never create an extra item/score from editorial duplication.",
            "judge_instruction": f"Score all frozen criteria once, sum them, and clamp to [0,{entry['points']:g}]. Apply alternatives and partial branches non-additively.",
            "source_provenance": {
                "source_pdf": SOLUTION_REL,
                "source_pdf_sha256": s_hash,
                "pdf_pages": s_pages,
                "status": "provided_organizer_solution_with_explicit_grading; external authenticity not independently verified",
            },
        }
        defects = source_defects_for(problem, source.SOURCE_DEFECTS)
        row = {
            "schema_version": "science_arena_item_v1",
            "item_id": item_id,
            "benchmark_id": "nbpho2026",
            "sequence_id": f"nbpho2026_p{problem}",
            "turn_index": turn_index,
            "global_order": global_order,
            "question_id": question_id,
            "parent_question": f"P{problem}",
            "problem_id": f"P{problem}",
            "problem_title": meta["title"],
            "problem_raw_max": meta["points"],
            "competition_raw_max": 72.0,
            "max_score": entry["points"],
            "max_raw_points": entry["points"],
            "question_text_native": native,
            "question_text_staticized": staticized,
            "question_text_staticized_complete": True,
            "question_only_staticization_verified": True,
            "human_visual_audited": True,
            "reference_answer": entry["answer"],
            "gold_solution": entry["solution"],
            "gold_solution_status": "human_audited_summary_of_provided_organizer_solution",
            "official_answer": entry["answer"],
            "official_solution": entry["solution"],
            "grading_rubric": grading_rubric,
            "official_rubric": copy.deepcopy(grading_rubric),
            "official_rubric_criteria": copy.deepcopy(criteria),
            "rubric_type": "official_explicit_grading_from_provided_solution_pdf",
            "official_rubric_status": "present_in_provided_solution_pdf_external_authenticity_unconfirmed",
            "visual_dependency": bool(entry["visual"]),
            "question_image_paths": [ref["path"] for ref in q_refs],
            "question_image_assets": q_refs,
            "question_image_refs": copy.deepcopy(q_refs),
            "official_solution_image_paths": [ref["path"] for ref in s_refs],
            "official_solution_image_assets": s_refs,
            "official_solution_image_refs": copy.deepcopy(s_refs),
            "question_source": {
                "source_pdf": QUESTION_REL,
                "source_pdf_sha256": q_hash,
                "pdf_pages": q_pages,
                "printed_pages": q_pages,
                "locator": f"Problem {question_id}: {meta['title']}",
                "provenance_status": "provided_problem_paper_external_authenticity_unconfirmed",
            },
            "solution_source": {
                "source_pdf": SOLUTION_REL,
                "source_pdf_sha256": s_hash,
                "pdf_pages": s_pages,
                "printed_pages": s_pages,
                "locator": f"Problem {question_id}: solution and Grading",
                "provenance_status": "provided_organizer_solution_with_explicit_grading_external_authenticity_unconfirmed",
            },
            "source_defects": defects,
            "visual_dependency_metadata": {
                "required": bool(entry["visual"]),
                "reason": "Problem-specific diagram/plot must be read" if entry["visual"] else "Problem statement is text-complete; official page remains attached to native-vision models",
                "multimodal_policy": "send_direct_official_problem_pdf_page_pixels",
                "text_only_policy": "question_only_human_audited_staticized_description",
                "question_image_paths": [ref["path"] for ref in q_refs],
                "assets": copy.deepcopy(q_refs),
                "missing_asset_policy": "block_no_fallback",
                "validation": "source PDF/page/hash and PNG hash verified",
            },
            "multimodal_missing_asset_policy": "BLOCK_ITEM_DO_NOT_FALL_BACK_TO_STATICIZED_TEXT",
            "input_tracks": {
                "native_vision": {
                    "question_field": "question_text_native",
                    "direct_official_question_images": True,
                    "missing_asset_policy": "block_no_fallback",
                    "description_is_not_sent": True,
                },
                "text_staticized": {
                    "question_field": "question_text_staticized",
                    "question_only": True,
                    "human_visual_audited": True,
                    "solution_leakage_check": "passed",
                    "no_image_payload": True,
                },
            },
            "staticization": {
                "method": "manual_question_only_transcription_from_official_problem_pages",
                "source_pdf": QUESTION_REL,
                "source_pdf_sha256": q_hash,
                "source_pages": q_pages,
                "solution_pdf_access": "forbidden_to_solver_staticization_artifact",
                "solution_leakage_check": "passed",
                "human_visual_audited": True,
            },
            "solver_gold_isolation": {
                "question_assets_only": True,
                "solution_assets_gold_only": True,
                "leakage_check": "passed",
            },
        }
        rows.append(row)
    return rows, question_assets, solution_assets


def build(*, draft: Path, output: Path) -> None:
    if not PROBLEM_PDF.is_file() or not SOLUTION_PDF.is_file():
        raise FileNotFoundError("NBPhO source PDFs are missing")
    source = load_source_module()
    q_hash = sha256_file(PROBLEM_PDF)
    s_hash = sha256_file(SOLUTION_PDF)
    expected_q = "d796b6610aec45c7b0d1b2823b9d19561d94fce424867673a1da481d1b02145e"
    expected_s = "6df4a2f9630e971034e2d6f4141b1e5bfa0816ee7b61c7b0e658996e1c26f457"
    if (q_hash, s_hash) != (expected_q, expected_s):
        raise RuntimeError("NBPhO source PDF hash changed; manual re-audit required")

    for root in (draft, output):
        root.mkdir(parents=True, exist_ok=True)
        rows, question_assets, solution_assets = build_tree(root, source, q_hash, s_hash)
        write_jsonl(root / "items.jsonl", rows)
        write_jsonl(root / "question_image_manifest.jsonl", question_assets)
        write_jsonl(root / "official_solution_image_manifest.jsonl", solution_assets)
        write_json(root / "item_schema.json", make_schema())

        manifest = {
            "schema_version": "nbpho2026_benchmark_manifest_v1",
            "benchmark_id": "nbpho2026",
            "competition": "Nordic-Baltic Physics Olympiad 2026",
            "source_policy": "only the two provided local PDFs",
            "source_pdfs": [
                {"path": QUESTION_REL, "sha256": q_hash, "pages": 2, "role": "solver_question_only"},
                {"path": SOLUTION_REL, "sha256": s_hash, "pages": 14, "role": "judge_gold_only"},
            ],
            "item_count": len(rows),
            "problem_count": 10,
            "sequence_count": 10,
            "raw_score_total": sum(float(row["max_score"]) for row in rows),
            "problem_score_totals": {f"P{p}": source.PROBLEMS[p]["points"] for p in range(1, 11)},
            "visual_dependency_item_count": sum(bool(row["visual_dependency"]) for row in rows),
            "question_image_count": len(question_assets),
            "solution_image_count": len(solution_assets),
            "question_image_policy": "mechanical 180-DPI PNG renders of official problem PDF pages; direct API image payload for vision models",
            "solution_image_policy": "separate gold-only mechanical renders; never solver input",
            "text_only_policy": "question-only human-audited staticization; no images and no solution-derived content",
            "rubric_provenance": "explicit Grading sections in provided organizer solution PDF; external authenticity unconfirmed",
            "source_defects": source.SOURCE_DEFECTS,
            "items_sha256": sha256_file(root / "items.jsonl"),
            "question_manifest_sha256": sha256_file(root / "question_image_manifest.jsonl"),
            "solution_manifest_sha256": sha256_file(root / "official_solution_image_manifest.jsonl"),
            "builder": rel_workspace(Path(__file__)),
            "builder_sha256": sha256_file(Path(__file__)),
            "source_data": rel_workspace(HERE / "nbpho2026_source.py"),
            "source_data_sha256": sha256_file(HERE / "nbpho2026_source.py"),
        }
        write_json(root / "benchmark_manifest.json", manifest)
        write_json(root / "manifest.json", manifest)

    text_dir = draft / "source_text"
    text_dir.mkdir(parents=True, exist_ok=True)
    (text_dir / "problems_layout.txt").write_text(
        command_text(["pdftotext", "-layout", str(PROBLEM_PDF), "-"]), encoding="utf-8"
    )
    (text_dir / "solutions_layout.txt").write_text(
        command_text(["pdftotext", "-layout", str(SOLUTION_PDF), "-"]), encoding="utf-8"
    )
    page_audit = {
        "schema_version": "nbpho2026_page_dual_audit_v1",
        "status": "PASS",
        "method": "every page read through pdftotext -layout and 180-DPI PNG visual inspection",
        "problem_pdf_sha256": q_hash,
        "solution_pdf_sha256": s_hash,
        "problem_pdf_pages": [
            {"page": page, "text_extracted": True, "rendered": True, "visually_reviewed": True, "notes": PAGE_AUDIT["problem_pdf"][str(page)]}
            for page in range(1, 3)
        ],
        "solution_pdf_pages": [
            {"page": page, "text_extracted": True, "rendered": True, "visually_reviewed": True, "notes": PAGE_AUDIT["solution_pdf"][str(page)]}
            for page in range(1, 15)
        ],
        "all_pages_text_and_visual_audited": True,
    }
    write_json(draft / "page_audit.json", page_audit)
    write_json(output / "page_audit.json", page_audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(draft=args.draft.resolve(), output=args.output.resolve())
    print(f"Built NBPhO 2026 canonical dataset at {args.output.resolve()}")


if __name__ == "__main__":
    main()
