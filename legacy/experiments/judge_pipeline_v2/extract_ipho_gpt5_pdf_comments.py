#!/usr/bin/env python3
"""Extract GPT-5 IPhO expert comments from the three annotated PDF reports.

This helper requires pypdf. It writes a small cache consumed by
build_expert_ground_truth.py so the main ground-truth builder can run even in
minimal environments without PDF tooling.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "judge_pipeline_v2" / "ground_truth" / "ipho_gpt5_pdf_comments.json"
PDFS = {
    "S1": ROOT / "artifacts/expert_grading/ipho2025/original/gpt5-S1.pdf",
    "S2": ROOT / "artifacts/expert_grading/ipho2025/original/gtp5-S2.pdf",
    "S3": ROOT / "artifacts/expert_grading/ipho2025/original/gpt5-S3.pdf",
}
LABELS = {
    "S1": ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4"],
    "S2": ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "C4", "C5"],
    "S3": ["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "C1", "C2", "C3"],
}


def clean_comment(lines: list[str]) -> str:
    text = "\n".join(line.strip() for line in lines).strip()
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def extract_text(path: Path) -> str:
    reader = PdfReader(str(path))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def parse_section(section: str, text: str) -> dict:
    lines = [line.strip() for line in text.splitlines()]
    rows = []
    buffer: list[str] = []
    in_intro = True
    i = 0
    while i < len(lines):
        line = lines[i]
        if in_intro:
            if line in {"Perfect.", "Perfect. Perfect analysis (instead of giving the result directly)."}:
                in_intro = False
            else:
                i += 1
                continue
        if re.match(r"^Total(?: Score)?:", line):
            break
        if line.startswith("Score:"):
            score_text = line.removeprefix("Score:").strip()
            j = i + 1
            while "/" not in score_text and j < len(lines):
                score_text = f"{score_text} {lines[j].strip()}".strip()
                j += 1
            score_match = re.search(r"(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)", score_text)
            if not score_match:
                raise ValueError(f"Could not parse score in {section}: {score_text!r}")
            index = len(rows)
            rows.append(
                {
                    "label": LABELS[section][index],
                    "human_comment": clean_comment(buffer),
                    "pdf_score": float(score_match.group(1)),
                    "pdf_max_score": float(score_match.group(2)),
                    "raw_score_line": score_text,
                }
            )
            buffer = []
            i = j
            continue
        if line:
            buffer.append(line)
        i += 1

    expected = len(LABELS[section])
    if len(rows) != expected:
        raise ValueError(f"{section}: expected {expected} items, parsed {len(rows)}")
    return {
        "source_file": str(PDFS[section].relative_to(ROOT)),
        "items": rows,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "model": "gpt-5",
        "sections": {
            section: parse_section(section, extract_text(path))
            for section, path in PDFS.items()
        },
    }
    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
