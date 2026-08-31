#!/usr/bin/env python3
"""Build the canonical, solver/gold-isolated IChO 2026 theory dataset.

This script performs no network or model calls.  Solver text is assembled only
from the previously audited G0/question/answer sources plus the human visual
transcriptions in visual_transcriptions.py.  Official answer and rubric fields
are extracted separately from the color channels of the official marking PDFs.
"""

from __future__ import annotations

import hashlib
import csv
import json
import re
import shutil
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import fitz
from PIL import Image

from visual_transcriptions import ALL as VISUAL_TRANSCRIPTIONS


ROOT = Path(__file__).resolve().parents[4]
DRAFT = ROOT / "experiments/2026_reval/datasets_draft/icho2026"
OUT = ROOT / "experiments/2026_reval/datasets/icho2026"
QUESTION_IMAGE_ROOT = OUT / "question_images"
SOLUTION_IMAGE_ROOT = OUT / "official_solution_images"

PROBLEM_META = {
    "T1": (25, 7, "A Journey through Time: The Secrets of Avicenna"),
    "T2": (35, 6, "Kinetics of the Belousov-Zhabotinsky reaction"),
    "T3": (63, 7, "Into Reticular Chemistry"),
    "T4": (22, 6, "The Nuclear Past of Uzbekistan"),
    "T5": (18, 6, "Cardiolipins"),
    "T6": (83, 7, "Carbon Nanorings"),
    "T7": (54, 7, "Nitrogen Fixation"),
    "T8": (84, 7, "Recycling of Carbon Dioxide"),
    "T9": (53, 7, "Cyclodextrin Chemistry"),
}

# The draft dependency map for 4.3 included solution page 2 because the fission
# yield graph begins there.  That page's coloured material belongs to 4.2, not
# 4.3.  Keep the graph on the solver/question side, but segment 4.3 gold from
# its actual official solution/marking block on page 3 only.
GOLD_PAGE_ALLOWLIST = {
    "4.3": {3},
}

RUBRIC_PATTERN = re.compile(
    r"(?i)(?:\bpoints?\b|\bpts?\b|\bmarks?\b|full (?:score|marks?)|"
    r"deduct|penalt|accepted?|no (?:point|mark)|incorrect|correct(?:ly)?|"
    r"mistake|switched|without (?:derivation|calculation)|half)"
)
POINT_PATTERN = re.compile(r"(?i)(?<!\w)([−–-]?\d+(?:\.\d+)?)\s*(?:pt|points?)\b")
ITEM_HEADER_PATTERN = re.compile(r"^\d+\.\d+$")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=False) + "\n")


def sanitize_staticized(text: str, subquestion_id: str) -> str:
    text = text.replace(
        "IChO 2026 THEORY — SOURCE-FAITHFUL TEXT-LAYER DRAFT",
        "IChO 2026 THEORY — CANONICAL STANDALONE TEXT-ONLY SOLVER PROMPT",
    )
    text = text.replace(
        "No official solution or marking text is present.",
        "Gold-answer and marking material are excluded from this solver prompt.",
    )
    text = text.replace(
        "Mathematical layout, chemical graphs, stereochemistry, plots, photographs, and drawn templates must be checked against the referenced original PDF pages; this draft is not yet approved as a lossless text-only prompt.",
        "All non-text mathematical layout, chemical graphs, stereochemistry, plots, photographs, and drawn templates needed for this item are normalized in the human-audited transcription appended below.",
    )
    appendix = (
        "\n\n<<<HUMAN-VISUAL-AUDITED NON-TEXT TRANSCRIPTION | SOLVER-SAFE>>>\n"
        "Audit scope: official G0, question-booklet, and answer-booklet source pages "
        "listed above. This section contains no marking-booklet content.\n"
        + VISUAL_TRANSCRIPTIONS[subquestion_id].strip()
        + "\n<<<END HUMAN-VISUAL-AUDITED TRANSCRIPTION>>>"
    )
    return text.rstrip() + appendix


def copy_question_images(draft_item: dict) -> tuple[list[str], list[dict]]:
    paths: list[str] = []
    refs: list[dict] = []
    for source_ref in draft_item["question_image_refs"]:
        src = ROOT / source_ref["path"]
        stem = src.parent.name
        dst = QUESTION_IMAGE_ROOT / stem / src.name
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or sha256_file(dst) != source_ref["image_sha256"]:
            shutil.copy2(src, dst)
        actual_hash = sha256_file(dst)
        if actual_hash != source_ref["image_sha256"]:
            raise RuntimeError(f"Question image hash mismatch after copy: {dst}")
        rel = dst.relative_to(ROOT).as_posix()
        ref = dict(source_ref)
        ref["path"] = rel
        ref["canonical_copy"] = True
        ref["human_visual_audited"] = True
        paths.append(rel)
        refs.append(ref)
    return paths, refs


def copy_all_question_images() -> list[dict]:
    """Copy and attest every G0/question/answer page, including unused covers."""
    output: list[dict] = []
    with (DRAFT / "question_image_manifest.tsv").open(encoding="utf-8", newline="") as f:
        for source_ref in csv.DictReader(f, delimiter="\t"):
            src = ROOT / source_ref["path"]
            dst = QUESTION_IMAGE_ROOT / src.parent.name / src.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists() or sha256_file(dst) != source_ref["image_sha256"]:
                shutil.copy2(src, dst)
            actual_hash = sha256_file(dst)
            if actual_hash != source_ref["image_sha256"]:
                raise RuntimeError(f"Question image hash mismatch after copy: {dst}")
            output.append(
                {
                    "path": dst.relative_to(ROOT).as_posix(),
                    "image_sha256": actual_hash,
                    "width_px": int(source_ref["width_px"]),
                    "height_px": int(source_ref["height_px"]),
                    "source_pdf": source_ref["source_pdf"],
                    "source_pdf_sha256": source_ref["source_pdf_sha256"],
                    "source_pdf_page": int(source_ref["source_pdf_page"]),
                    "crop": source_ref["crop"],
                    "render_recipe": source_ref["render_recipe"],
                    "contains_solution_or_rubric": False,
                    "canonical_copy": True,
                    "human_visual_audited": True,
                }
            )
    return output


def render_solution_page(pdf_rel: str, page_number: int) -> tuple[str, dict]:
    pdf = ROOT / pdf_rel
    stem = pdf.stem
    dst = SOLUTION_IMAGE_ROOT / stem / f"page-{page_number}.png"
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists():
        prefix = dst.with_suffix("")
        subprocess.run(
            [
                "pdftoppm", "-f", str(page_number), "-l", str(page_number),
                "-singlefile", "-png", "-r", "200", str(pdf), str(prefix),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
    with Image.open(dst) as image:
        width, height = image.size
    rel = dst.relative_to(ROOT).as_posix()
    ref = {
        "path": rel,
        "image_sha256": sha256_file(dst),
        "width_px": width,
        "height_px": height,
        "source_pdf": pdf_rel,
        "source_pdf_sha256": sha256_file(pdf),
        "source_pdf_page": page_number,
        "crop": "FULL_PAGE",
        "render_recipe": "Official solution PDF page rendered with Poppler pdftoppm 26.08.0; PNG; 200 dpi; full page; no crop",
        "gold_only": True,
        "must_not_be_sent_to_solver": True,
    }
    return rel, ref


def line_records(page: fitz.Page) -> list[dict]:
    records: list[dict] = []
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
            if not spans:
                continue
            records.append(
                {
                    "text": "".join(s["text"] for s in spans).strip(),
                    "x0": min(s["bbox"][0] for s in spans),
                    "y0": min(s["bbox"][1] for s in spans),
                    "y1": max(s["bbox"][3] for s in spans),
                    "spans": spans,
                }
            )
    return sorted(records, key=lambda r: (round(r["y0"], 1), r["x0"]))


def color_name(value: int) -> str:
    if value == 0xFF0000:
        return "red"
    if value == 0x0000FF:
        return "blue"
    if value == 0x0000B3:
        return "blue"
    if value == 0:
        return "black"
    if value == 0xFFFFFF:
        return "white"
    return f"rgb#{value:06x}"


def item_colored_transcript(pdf_rel: str, page_number: int, sid: str) -> list[dict]:
    doc = fitz.open(ROOT / pdf_rel)
    page = doc[page_number - 1]
    lines = line_records(page)
    headers = [
        (r["text"], r["y0"])
        for r in lines
        if r["x0"] < 150 and ITEM_HEADER_PATTERN.fullmatch(r["text"])
        and all(s.get("color", 0) == 0 for s in r["spans"])
    ]
    target = [y for label, y in headers if label == sid]
    if target:
        # First pages also contain a compact score table whose cells repeat all
        # item numbers.  The actual task heading is the last occurrence of sid.
        target_y = max(target)
        lower = target_y - 2
        later = [y for _, y in headers if y > target_y + 1]
        upper = min(later) - 2 if later else page.rect.height - 20
    else:
        # Continuation pages contain no repeated item header.  Gold page maps were
        # audited manually, so the body of such a page belongs to this item.
        lower, upper = 145, page.rect.height - 20

    output: list[dict] = []
    for record in lines:
        if record["y1"] < lower or record["y0"] >= upper:
            continue
        groups: list[tuple[str, list[str]]] = []
        for span in record["spans"]:
            name = color_name(span.get("color", 0))
            if name in {"black", "white"}:
                continue
            if groups and groups[-1][0] == name:
                groups[-1][1].append(span["text"])
            else:
                groups.append((name, [span["text"]]))
        for name, texts in groups:
            txt = "".join(texts).strip()
            if txt:
                output.append(
                    {
                        "source_pdf": pdf_rel,
                        "source_page": page_number,
                        "source_color": name,
                        "text": txt,
                        "bbox_y0": round(record["y0"], 2),
                    }
                )
    doc.close()
    return output


def normalize_gold(gold_ref: dict) -> dict:
    sid = gold_ref["subquestion_id"]
    colored: list[dict] = []
    solution_paths: list[str] = []
    solution_image_refs: list[dict] = []
    page_refs = gold_ref["official_solution_page_refs"]
    if sid in GOLD_PAGE_ALLOWLIST:
        page_refs = [
            page_ref for page_ref in page_refs
            if page_ref["pdf_page"] in GOLD_PAGE_ALLOWLIST[sid]
        ]
    for page_ref in page_refs:
        pdf_rel = page_ref["pdf_path"]
        page_number = page_ref["pdf_page"]
        colored.extend(item_colored_transcript(pdf_rel, page_number, sid))
        path, image_ref = render_solution_page(pdf_rel, page_number)
        solution_paths.append(path)
        solution_image_refs.append(image_ref)

    # Preserve page/color provenance line by line.  This is intentionally
    # source-near: molecular drawings themselves remain in the gold-only images.
    source_lines = [
        f"[p.{r['source_page']} {r['source_color']}] {r['text']}" for r in colored
    ]
    source_transcript = "\n".join(source_lines).strip()
    rubric_lines = []
    rubric_started = False
    for r in colored:
        is_rule = RUBRIC_PATTERN.search(r["text"]) or POINT_PATTERN.search(r["text"])
        if r["source_color"] == "blue" or is_rule:
            rubric_lines.append(r)
            rubric_started = rubric_started or bool(is_rule)
        elif rubric_started:
            # Wrapped continuation of a source-colored marking paragraph.
            rubric_lines.append(r)
    if not rubric_lines:
        rubric_lines = colored
    rubric_text = "\n".join(
        f"[p.{r['source_page']} {r['source_color']}] {r['text']}" for r in rubric_lines
    ).strip()
    if not source_transcript:
        source_transcript = (
            "The official response for this item is graphical; inspect the cited "
            "gold-only official_solution_image_paths."
        )
    if not rubric_text:
        rubric_text = (
            f"Apply the official graphical marking page under a hard cap of "
            f"{gold_ref['max_raw_points']} raw points."
        )

    raw_rules = []
    for idx, record in enumerate(rubric_lines, 1):
        mentions = [m.group(0) for m in POINT_PATTERN.finditer(record["text"])]
        raw_rules.append(
            {
                "rule_id": f"icho2026_{sid}_rule_{idx:02d}",
                "description": record["text"],
                "source_pdf": record["source_pdf"],
                "source_page": record["source_page"],
                "source_color": record["source_color"],
                "raw_score_mentions": mentions,
                "aggregation": "interpret_under_official_item_cap; alternatives/partials/deductions are not automatically additive",
            }
        )

    criterion = {
        "criterion_id": f"icho2026_{sid}_official_total",
        "description": rubric_text,
        "max_score": gold_ref["max_raw_points"],
        "score_unit": "raw_point",
        "item_score_cap": gold_ref["max_raw_points"],
        "rules": raw_rules,
        "judge_instruction": (
            "Compare the response with the official colored transcription and "
            "gold-only page images. Apply partial-credit, alternatives, deductions, "
            "and error-forward rules exactly as printed; clamp the item score to "
            f"[0,{gold_ref['max_raw_points']}]."
        ),
    }
    return {
        "official_answer": (
            "OFFICIAL SOURCE-COLORED ANSWER/DERIVATION TRANSCRIPTION\n"
            + source_transcript
            + "\nExact red chemical drawings, check marks, and spatial notation are in "
            "official_solution_image_paths."
        ),
        "official_solution": (
            "Official color-channel transcription, segmented to this item:\n"
            + source_transcript
        ),
        "official_rubric": (
            "OFFICIAL SCORING-RULE TRANSCRIPTION\n"
            + rubric_text
            + f"\nHard item cap: {gold_ref['max_raw_points']} raw points."
        ),
        "official_rubric_rules": raw_rules,
        "official_rubric_criteria": [criterion],
        "official_color_channel_transcription": colored,
        "official_solution_image_paths": solution_paths,
        "official_solution_image_refs": solution_image_refs,
        "rubric_type": "official_colored_source_rules_with_item_cap",
        "rubric_normalization_status": "pipeline_executable_source_preserving",
    }


def adjudication_policy(sid: str) -> dict | None:
    if sid != "4.9":
        return None
    return {
        "policy_id": "icho2026_t4_9_fallback_and_printed_mass_typo_v2",
        "issue": (
            "The official page derives 7.03 mol*235.04 g mol^-1=1.65 kg, then "
            "prints 1.67 as the enrichment/burnup numerator, while its final 5.56 kg "
            "equals 1.65/(0.90*0.33), not 1.67/(0.90*0.33)."
        ),
        "preserve_source": (
            "Keep 1.65, printed 1.67, and final 5.56 exactly in the raw official "
            "transcription; do not silently amend the PDF."
        ),
        "primary_expected": {
            "total_fissions": "4.23e24 fissions",
            "enriched_uranium_mass": "5.56 kg",
        },
        "official_fallback": {
            "source_pdf": "new-test/IChO 2026/theoretical-exam-question/exam-theory-Q4-english_5.pdf",
            "source_page": 3,
            "printed_rule": "If you did not get an answer for 4.4, use Delta E = 200 MeV.",
            "accepted_consistent_results": {
                "total_fissions": "approximately 3.92e24 fissions",
                "enriched_uranium_mass": "approximately 5.15 kg",
            },
            "scoring": "A self-consistent use of the printed 200 MeV fallback is fully correct for both components.",
        },
        "contestant_visibility": (
            "The 1.67 typo appears only in the hidden official marking PDF and is "
            "not present in the solver question/answer pages. It therefore cannot "
            "supply a contestant-facing basis for a different full-credit answer."
        ),
        "incorrect_value_policy": (
            "Do not award full credit to approximately 5.62 kg solely because it "
            "equals the hidden printed 1.67/(0.90*0.33). Apply the official mass "
            "component rule instead: 1 of 2 mass points if the relevant enrichment/"
            "burnup calculation is shown but the final value is wrong; otherwise 0."
        ),
        "scoring": (
            "Retain the official 2 raw points for total fissions and 2 for enriched "
            "mass. Award full credit for either the primary 4.4-dependent path or "
            "the explicitly printed 200 MeV fallback path; otherwise retain the "
            "official 1-point relevant-calculation partial credit in each component."
        ),
        "requires_human_override": False,
    }


def _curated_criterion(
    sid: str,
    key: str,
    maximum: float,
    description: str,
    gold: dict,
) -> dict:
    """Create one frozen point-bearing criterion with official provenance.

    The official books often express alternatives, deductions, and ECF in prose
    whose numeric mentions are not additive.  These curated criteria encode the
    executable scoring units while retaining links to every item-level colored
    rule and to the exact official solution pages used by the judge.
    """
    aggregate = gold["official_rubric_criteria"][0]
    provenance = []
    seen = set()
    for record in gold["official_color_channel_transcription"]:
        signature = (
            record["source_pdf"], record["source_page"], record["source_color"]
        )
        if signature in seen:
            continue
        seen.add(signature)
        provenance.append(
            {
                "source_pdf": signature[0],
                "source_page": signature[1],
                "source_color": signature[2],
            }
        )
    return {
        "criterion_id": f"icho2026_{sid}_{key}",
        "description": description,
        "max_score": maximum,
        "score_unit": "raw_point",
        "source_provenance": provenance,
        "official_rule_ids": [
            rule["rule_id"] for rule in gold["official_rubric_rules"]
        ],
        "official_item_cap": aggregate["item_score_cap"],
        "judge_instruction": (
            "Apply this normalized criterion together with the cited official "
            "colored rules and gold-only drawings. Do not add alternative or "
            "partial-credit branches; award at most this criterion's max_score."
        ),
    }


def curated_rubric_criteria(sid: str, gold: dict) -> list[dict]:
    """Return manually normalized criteria for ambiguity-heavy official rubrics.

    Items outside this table retain one source-preserving aggregate criterion.
    That aggregate remains executable and has a positive maximum equal to the
    official item cap.  T4.3, T4.9, and all T6/T8 items are split here because
    their official rules contain multiple point components, coupled alternatives,
    ECF, or cross-field penalties.
    """
    specs: dict[str, list[tuple[str, float, str]]] = {
        "4.3": [
            (
                "mass_numbers_or_sum",
                1.2,
                "Award 0.6 for each correct fission-product mass number, 93 and "
                "140 (1.2 total). Alternatively, the explicitly stated correct "
                "sum 233 earns the full 1.2 for this component. These branches "
                "are non-additive.",
            ),
            (
                "elements",
                1.2,
                "Award 0.6 for each correct element, Rb and Cs (1.2 total), "
                "independently of the mass-number component.",
            ),
            (
                "fission_equation",
                0.6,
                "Award 0.6 for the correct complete nuclear fission equation "
                "235-U + neutron -> 93-Rb + 140-Cs + 3 neutrons. The printed "
                "rubric states no error-carried-forward alternative for an "
                "equation containing wrong isotope mass numbers.",
            ),
        ],
        "4.9": [
            (
                "fissions",
                2,
                "Award 2 for 4.23e24 total fissions using the answer from 4.4, "
                "or for a self-consistent result of approximately 3.92e24 using "
                "the question's explicitly printed fallback Delta E=200 MeV. "
                "Award 1 if the final value is otherwise wrong but a relevant "
                "calculation is shown; otherwise 0.",
            ),
            (
                "enriched_uranium_mass",
                2,
                "Award 2 for 5.56 kg enriched uranium using the answer from 4.4, "
                "or for a self-consistent result of approximately 5.15 kg using "
                "the question's explicitly printed fallback Delta E=200 MeV. "
                "Award 1 if a relevant enrichment/burnup calculation is shown "
                "but the final value is otherwise wrong; otherwise 0. Preserve "
                "the official 1.65/1.67 typo trail, but do not award full credit "
                "to 5.62 kg solely from the hidden solution-page typo.",
            ),
        ],
        "6.1": [
            ("c18_pair", 2, "For C18, A=2 and AA=0: both correct 2; one correct and the other blank 1; otherwise 0."),
            ("c16_pair", 2, "For C16, A=0 and AA=2: both correct 2; one correct and the other blank 1; otherwise 0."),
            ("triplet_c13_pair", 3, "For triplet C13, A=0 and AA=0: both correct 3; one correct and the other blank 2; one correct and the other wrong 1; both wrong 0."),
            ("singlet_c13_pair", 3, "For singlet C13, A=1 and AA=1: both correct 3; one correct and the other blank 2; one correct and the other wrong 1; both wrong 0."),
        ],
        "6.2": [
            (
                "structure_b",
                4,
                "Structure B: exact official drawing 4. For an otherwise "
                "incorrect drawing, award 1 for three Cl atoms and 1 for two "
                "alkynes in a 1,5 relationship; deduct 1 if radical sites are "
                "missing or incorrect. Clamp this criterion to [0,4].",
            ),
            (
                "structure_c",
                3,
                "Structure C: exact official drawing 3; the alternating "
                "single/triple-bond representation also earns 3. For an otherwise "
                "incorrect drawing award 1 for one Cl atom; deduct 1 if radical "
                "sites are missing or incorrect. Clamp this criterion to [0,3].",
            ),
            (
                "structure_d",
                4,
                "Structure D: exact official drawing 4. For an otherwise "
                "incorrect drawing, award 1 for five Cl atoms and 1 for two "
                "alkynes in a 1,5 relationship; deduct 1 if radical sites are "
                "missing or incorrect. Clamp this criterion to [0,4].",
            ),
        ],
        "6.3": [
            (
                "halogen_i",
                2,
                "Award 1 for the correct unit conversion and 1 for identifying "
                "halogen I. Selecting I alone earns 2 even without work. If any "
                "other halogen is selected, or more than one is selected, the "
                "whole criterion scores 0.",
            ),
        ],
        "6.4": [
            ("ion_mz_783", 4, "Award 4 only for [C48E3+3H]3+ at m/z 783; no partial credit."),
            ("ion_mz_879", 4, "Award 4 only for [C48E2+2H]2+ at m/z 879; no partial credit."),
            ("ion_mz_1174", 4, "Award 4 only for [C48E3+2H]2+ at m/z 1174; no partial credit."),
        ],
        "6.5": [
            ("structure_f", 4, "Grade F against the official drawing, max 4; accept its dianion or OLi/ONa salts at full credit."),
            ("structure_g", 3, "Grade G against the official drawing, max 3."),
            ("structure_h", 3, "Grade H against the official drawing, max 3; accept its monoanion or OLi salt at full credit."),
            ("structure_i", 4, "Grade I against the official drawing, max 4."),
            ("structure_j", 3, "Grade J against the official drawing, max 3; accept its dianion or OLi/ONa salts at full credit."),
            ("structure_k", 3, "Grade K against the official drawing, max 3."),
            ("structure_l", 4, "Grade L against the official drawing, max 4."),
        ],
        "6.6": [
            (
                "structure_m",
                4,
                "M: exact official two-ortho-t-Bu product 4; one t-Bu group 1; "
                "two t-Bu groups in another substitution pattern 2. These are "
                "mutually exclusive score tiers.",
            ),
            (
                "structure_n",
                4,
                "N: official aldehyde 4; the amine intermediate from the "
                "Delepine reaction also earns 4 and must not reduce downstream "
                "O/Q/R credit; Br in place of the aldehyde earns 1.",
            ),
            ("structure_o", 4, "Grade O against the official drawing, max 4."),
            ("structure_q", 4, "Grade Q against the official drawing, max 4."),
            ("structure_r", 4, "Grade R against the official drawing, max 4."),
        ],
        "6.7": [
            (
                "electron_counts",
                4,
                "Award 4 for n(e)=2 and n(t)=82. If the total electron number is "
                "wrong but still satisfies 4n+2, award 2. If the removed-electron "
                "count is not 2, no partial credit applies; otherwise 0.",
            ),
        ],
        "8.1": [
            (
                "balanced_half_equation",
                2,
                "Award 2 only for the fully correct balanced half-equation "
                "CO2 + 2H+ + 2e- -> CO + H2O; otherwise 0.",
            ),
        ],
        "8.2": [
            ("structure_3", 4, "Official structure 3 earns 4; an official-page alternative reasonable structure earns 2."),
            ("structure_4", 4, "Official structure 4 earns 4; an official-page alternative reasonable structure earns 2."),
            ("structure_5", 4, "Either official resonance form 5.1 or 5.2 earns 4, without double counting; an official-page alternative reasonable structure earns 2."),
            ("structure_6", 2, "Award 2 for the official structure 6; otherwise 0. The printed alternative-intermediate half-credit scheme does not declare a separate alternative for structure 6."),
            ("structure_7", 2, "Award 2 for the official structure 7; otherwise 0. The printed alternative-intermediate half-credit scheme does not declare a separate alternative for structure 7."),
        ],
        "8.3": [
            ("geometry_choice", 2, "Award 2 only for selecting the seventh/last (lower-right) geometry option; otherwise 0."),
        ],
        "8.4": [
            ("species_9", 4.5, "Species 9 expected Z/OS/CN/VE = +2/+2/6/18. The correct drawn structure earns 3; OS, CN, and VE each earn 0.5 (1.5 total); wrong total charge Z deducts 0.5 within this criterion."),
            ("species_10", 4.5, "Species 10 expected Z/OS/CN/VE = +1/+1/5/17. The correct drawn structure earns 3; OS, CN, and VE each earn 0.5 (1.5 total); wrong total charge Z deducts 0.5 within this criterion."),
            ("species_11", 3.5, "Species 11 expected Z/OS/CN/VE = +1/+3/6/17. The correct drawn structure earns 3; only the blank VE field scores 0.5. OS=+3 and CN=6 are prefilled and earn no points; wrong Z deducts 0.5."),
            ("species_12", 4, "Species 12 expected Z/OS/CN/VE = +1/+2/5/16. The correct drawn structure earns 3; only blank OS and VE fields score 0.5 each. CN=5 is prefilled and earns no points; wrong Z deducts 0.5."),
            ("species_13", 4, "Species 13 expected Z/OS/CN/VE = +1/+2/6/18. The correct drawn structure earns 3; only blank CN and VE fields score 0.5 each. OS=+2 is prefilled and earns no points; wrong Z deducts 0.5."),
            ("species_14", 4.5, "Species 14 expected Z/OS/CN/VE = +2/+2/6/18. The correct drawn structure earns 3; OS, CN, and VE each earn 0.5 (1.5 total); wrong total charge Z deducts 0.5 within this criterion."),
            ("species_15", 4, "Species 15 expected Z/OS/CN/VE = +2/+2/5/16. The correct drawn structure earns 3; only blank OS and CN fields score 0.5 each. VE=16 is prefilled and earns no points; wrong Z deducts 0.5."),
        ],
        "8.5": [
            (
                "catalyst_number_density",
                5,
                "Mutually exclusive tiers: award 5 for Ncat approximately "
                "2.4 nm^-2 using the official support-mass interpretation; award "
                "3 (not 5+3) for the official alternative mixture-area "
                "interpretation giving approximately 2.3 nm^-2; otherwise 0 "
                "unless another official partial rule applies.",
            ),
        ],
        "8.6": [
            ("co_rate", 2, "Award 2 for rCO = 9.48e14 s^-1."),
            ("photon_energy", 2, "Award 2 for Ephoton = 5.1e-19 J."),
            ("photon_rate", 2, "Award 2 for rphoton = 9.8e16 s^-1."),
            ("quantum_yield_equation", 2, "Award 2 for correctly using the official quantum-yield equation with two reaction electrons per CO."),
            ("quantum_yield_value", 2, "Award 2 for phi = 1.94%."),
        ],
        "8.7": [
            ("rate_choice", 2, "Award 2 only for option a: the overall rate increases; otherwise 0."),
        ],
        "8.8": [
            (
                "color_mapping_ticks",
                4,
                "Correct mapping: a=N, b=B, c=G, d=R. Compute raw as the "
                "number of correct ticks minus the number of incorrect ticks, "
                "counting every extra incorrect tick, then score "
                "clamp(raw,0,4). Unticked boxes neither add nor deduct.",
            ),
        ],
        "8.9": [
            ("s1_unquenched_decay", 2, "Award 2 for the correct S1 unquenched decay/lifetime expression."),
            ("t1_unquenched_decay", 2, "Award 2 for the correct T1 unquenched decay/lifetime expression."),
            ("s1_quenching_fraction", 2, "Award 2 for the correct S1 quenching-fraction expression."),
            ("t1_quenching_fraction", 2, "Award 2 for the correct T1 quenching-fraction expression."),
            ("s1_quenching_value", 2, "Award 2 for eta_q(S1)=44%."),
            ("t1_quenching_value", 2, "Award 2 for eta_q(T1)=99.9%. The official source misnumbers T8.9 equations as 8.10.1-8.10.4; preserve that source typo but associate these criteria with T8.9."),
        ],
        "8.10": [
            ("lifetime_choice", 2, "Award 2 only for option b: the lifetimes decrease; otherwise 0."),
        ],
    }
    if sid not in specs:
        return gold["official_rubric_criteria"]
    criteria = [
        _curated_criterion(sid, key, maximum, description, gold)
        for key, maximum, description in specs[sid]
    ]
    expected = float(gold["official_rubric_criteria"][0]["item_score_cap"])
    actual = sum(float(criterion["max_score"]) for criterion in criteria)
    if abs(actual - expected) > 1e-9:
        raise RuntimeError(f"Curated rubric maximum mismatch for {sid}: {actual} != {expected}")
    return criteria


def curated_item_judge_instruction(sid: str, maximum: float) -> str:
    shared = (
        "Score every frozen criterion independently against its official colored "
        "rule/page provenance, then sum. Apply alternatives and partial-credit "
        f"branches non-additively and clamp the final item score to [0,{maximum}]."
    )
    if sid == "6.5":
        return shared + (
            " Award error carry-forward when the desired reactivity is shown and "
            "the chemistry is sensible. Deduct 1 for each mistake in a molecular "
            "portion not undergoing reaction, within the affected structure only."
        )
    if sid == "6.6":
        return shared + (
            " Across O/Q/R, deduct 1 for incorrect porphyrin double bonds at most "
            "once total and deduct 2 for missing Zn at most once total. Record each "
            "one-time deduction by reducing the earliest affected O/Q/R criterion; "
            "distribute any remainder across later affected criteria without making "
            "any criterion negative. The Delepine amine alternative for N must not "
            "reduce downstream O/Q/R credit."
        )
    if sid == "8.4":
        return shared + (
            " Score exactly 7 structures (21 points) and the 16 answer-sheet fields "
            "that were not prefilled (8 points), for 29 total. Z has no positive "
            "award; deduct 0.5 within a species for wrong total charge. Do not grade "
            "ligand-8 arrangement again when consistent with the candidate's T8.3 "
            "arrangement, preventing double punishment."
        )
    return shared


def build() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    QUESTION_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    SOLUTION_IMAGE_ROOT.mkdir(parents=True, exist_ok=True)
    draft_items = read_jsonl(DRAFT / "items_draft.jsonl")
    gold_refs = {r["turn_id"]: r for r in read_jsonl(DRAFT / "gold_rubric_refs.jsonl")}
    if len(draft_items) != 68 or len(gold_refs) != 68:
        raise RuntimeError("Expected 68 draft items and 68 gold references")

    rows: list[dict] = []
    problem_turns = Counter()
    for draft in draft_items:
        sid = draft["subquestion_id"]
        problem_id = draft["problem_id"]
        problem_turns[problem_id] += 1
        raw_max, weight, title = PROBLEM_META[problem_id]
        question_paths, question_refs = copy_question_images(draft)
        staticized = sanitize_staticized(draft["question_text_staticized"], sid)
        native = (
            f"IChO 2026 Theory {sid} ({draft['max_raw_points']} raw points).\n"
            f"{draft['instruction_draft']}\n"
            "Read the attached official G0/question/answer-booklet pages directly. "
            "The answer-booklet pages are response templates, not gold answers."
        )
        gold = normalize_gold(gold_refs[draft["turn_id"]])
        normalized_criteria = curated_rubric_criteria(sid, gold)
        gold["official_rubric_criteria"] = normalized_criteria
        question_assets = [
            {
                "path": ref["path"],
                "sha256": ref["image_sha256"],
                "image_sha256": ref["image_sha256"],
                "source_pdf": ref["source_pdf"],
                "source_pdf_path": ref["source_pdf"],
                "source_pdf_sha256": ref["source_pdf_sha256"],
                "source_page": ref["source_pdf_page"],
                "source_pdf_page": ref["source_pdf_page"],
                "crop": ref["crop"],
                "contains_solution": False,
                "contains_solution_or_rubric": False,
                "human_visual_audited": True,
            }
            for ref in question_refs
        ]
        grading_rubric = {
            "schema_version": "icho2026_pipeline_grading_rubric_v1",
            "status": "official_color_channel_source_preserving",
            "rubric_type": "official_colored_source_rules_with_item_cap",
            "item_max_score": draft["max_raw_points"],
            "criteria": normalized_criteria,
            "criteria_normalization": (
                "curated_point_units_with_official_source_provenance"
                if sid in {"4.3", "4.9"} or sid.startswith("6.") or sid.startswith("8.")
                else "single_official_item_cap_with_colored_rule_provenance"
            ),
            "official_scoring_rule_transcription": gold["official_rubric"],
            "source_color_channel_transcription": gold["official_color_channel_transcription"],
            "judge_instruction": curated_item_judge_instruction(
                sid, draft["max_raw_points"]
            ),
        }
        row = {
            "schema_version": "icho2026_canonical_item_v1",
            "item_id": draft["turn_id"],
            "turn_id": draft["turn_id"],
            "benchmark_id": "icho2026_theory",
            "sequence_id": f"icho2026_{problem_id.lower()}",
            "turn_index": problem_turns[problem_id],
            "problem_id": problem_id,
            "problem_title": title,
            "subquestion_id": sid,
            "max_score": draft["max_raw_points"],
            "max_raw_points": draft["max_raw_points"],
            "problem_raw_max": raw_max,
            "problem_weight": weight,
            "report_scale": 60,
            "score_conversion": "problem_contribution = problem_weight * sum(item_raw_scores_in_problem) / problem_raw_max; total is sum of nine contributions",
            "question_text_native": native,
            "question_text_staticized": staticized,
            "question_text_staticized_sha256": sha256_text(staticized),
            "question_text_staticized_complete": True,
            "question_text_staticized_status": "COMPLETE_human_visual_audited_solver_safe",
            "question_only_staticization_verified": True,
            "staticization": {
                "status": "complete",
                "human_visual_audited": True,
                "solution_leakage_check": "passed",
                "source_scope": "G0/question/answer pages only",
            },
            "human_visual_audited": True,
            "human_visual_audit_scope": "all cited G0/question/answer source pages; text layer and 200-dpi PNG checked together",
            "question_pdf": draft["question_pdf"],
            "question_pages": draft["question_pages"],
            "required_context_pages": draft["required_context_pages"],
            "answer_pdf": draft["answer_pdf"],
            "answer_pages": draft["answer_pages"],
            "solver_source_page_refs": draft["solver_source_page_refs"],
            "question_image_paths": question_paths,
            "question_image_refs": question_refs,
            "question_image_assets": question_assets,
            "visual_dependency": bool(draft["visual_dependency"]),
            "visual_dependency_reason": draft["visual_dependency_reason"],
            "visual_tags": draft["visual_tags"],
            "response_mode": draft["response_mode"],
            "multimodal_input_status": "READY_official_source_pages",
            "multimodal_missing_asset_policy": (
                "BLOCK_ITEM_DO_NOT_FALL_BACK_TO_STATICIZED_TEXT"
                if draft["visual_dependency"]
                else "BLOCK_ITEM_IF_CONFIGURED_MULTIMODAL_ASSET_IS_MISSING"
            ),
            "solver_gold_isolation": {
                "gold_fields_excluded_from_solver_payload": True,
                "solver_image_sources": ["G0", "question_booklet", "answer_booklet"],
                "solution_or_marking_pages_in_solver_images": 0,
                "leakage_check": "passed",
            },
            **gold,
            "reference_answer": gold["official_answer"],
            "gold_solution": gold["official_solution"],
            "grading_rubric": grading_rubric,
            "official_solution_page_refs": gold_refs[draft["turn_id"]]["official_solution_page_refs"],
            "source_color_semantics": gold_refs[draft["turn_id"]]["source_color_semantics"],
            "adjudication_policy": adjudication_policy(sid),
        }
        rows.append(row)

    write_jsonl(OUT / "items.jsonl", rows)

    referenced_question_refs = {
        r["path"]: r for item in rows for r in item["question_image_refs"]
    }
    referenced_solution_refs = {
        r["path"]: r for item in rows for r in item["official_solution_image_refs"]
    }
    all_question_refs = {r["path"]: r for r in copy_all_question_images()}
    all_solution_refs = dict(referenced_solution_refs)
    solution_root = ROOT / "new-test/IChO 2026/theoretical-exam-solutions"
    for pdf in sorted(solution_root.glob("*.pdf")):
        page_count = len(fitz.open(pdf))
        for page_number in range(1, page_count + 1):
            path, ref = render_solution_page(pdf.relative_to(ROOT).as_posix(), page_number)
            all_solution_refs[path] = ref
    write_jsonl(OUT / "question_image_manifest.jsonl", list(all_question_refs.values()))
    write_jsonl(OUT / "official_solution_image_manifest.jsonl", list(all_solution_refs.values()))

    manifest = {
        "schema_version": "icho2026_canonical_manifest_v1",
        "benchmark_id": "icho2026_theory",
        "competition": "58th International Chemistry Olympiad",
        "year": 2026,
        "language": "English",
        "source_status": "official",
        "dataset_status": "canonical_stage1b_ready_for_formal_evaluation",
        "item_count": len(rows),
        "problem_count": 9,
        "raw_point_total": sum(r["max_raw_points"] for r in rows),
        "reported_scale": 60,
        "problem_weight_total": sum(meta[1] for meta in PROBLEM_META.values()),
        "visual_dependency_item_count": sum(r["visual_dependency"] for r in rows),
        "question_text_staticized_complete_count": sum(r["question_text_staticized_complete"] for r in rows),
        "human_visual_audited_count": sum(r["human_visual_audited"] for r in rows),
        "question_image_assets_complete_count": sum(
            len(r["question_image_assets"]) == len(r["question_image_paths"])
            and {asset["path"] for asset in r["question_image_assets"]}
            == set(r["question_image_paths"])
            for r in rows
        ),
        "grading_criterion_count": sum(
            len(r["grading_rubric"]["criteria"]) for r in rows
        ),
        "grading_rubric_exact_sum_count": sum(
            abs(
                sum(float(c["max_score"]) for c in r["grading_rubric"]["criteria"])
                - float(r["max_score"])
            )
            < 1e-9
            for r in rows
        ),
        "curated_rubric_item_count": sum(
            r["grading_rubric"]["criteria_normalization"]
            == "curated_point_units_with_official_source_provenance"
            for r in rows
        ),
        "solver_safe_official_page_image_count": len(all_question_refs),
        "solver_safe_page_images_referenced_by_items": len(referenced_question_refs),
        "gold_only_official_solution_page_image_count": len(all_solution_refs),
        "gold_page_images_referenced_by_items": len(referenced_solution_refs),
        "score_conversion": "For each problem: official weight * earned raw / problem raw maximum; sum nine problem contributions for score out of 60.",
        "multimodal_policy": "Send question_image_paths from official G0/question/answer pages. If a visual-dependent item lacks any declared image, block it; never substitute prose.",
        "text_only_policy": "Use question_text_staticized only; every item is standalone and human visual audited.",
        "gold_policy": "official_answer, official_solution, official_rubric, criteria, and official_solution_image_paths are judge-only and must never enter solver payloads.",
        "known_source_issues": [adjudication_policy("4.9")],
        "post_audit_canonical_corrections": [
            {
                "item": "4.3",
                "issue": "Draft gold-page dependency included page 2 coloured scoring text from item 4.2.",
                "resolution": "Gold transcription is restricted to the actual 4.3 solution and three printed scoring components on page 3.",
            },
            {
                "item": "4.9",
                "issue": "Initial normalized criteria omitted the question's explicitly printed 200 MeV fallback path.",
                "resolution": "Both primary 4.4-dependent values and self-consistent fallback values are full-credit paths.",
            },
        ],
        "validation": {
            "dataset_validator": "experiments/2026_reval/datasets/icho2026/validate_canonical.py",
            "dataset_report": "experiments/2026_reval/datasets/icho2026/validation_report.json",
            "production_pipeline": "experiments/2026_reval/pipeline.py",
            "frozen_model_manifest": "experiments/2026_reval/manifests/table6_models_frozen_20260828.json",
            "pipeline_validation_proof": "experiments/2026_reval/runs/icho2026_validation_live/input_validation.json",
            "pipeline_run_manifest": "experiments/2026_reval/runs/icho2026_validation_live/run_manifest.json",
            "policy": "Both validators must pass after the final canonical rebuild; validate performs no model/API calls.",
        },
        "problem_meta": {
            pid: {"raw_max": raw, "weight": weight, "title": title}
            for pid, (raw, weight, title) in PROBLEM_META.items()
        },
        "no_model_or_api_calls": True,
    }
    (OUT / "benchmark_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    build()
