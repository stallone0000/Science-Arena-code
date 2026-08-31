#!/usr/bin/env python3
"""Build canonical expert ground-truth score files for Science Arena.

The script preserves human comments verbatim where available. It does not call
any model API.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from html import unescape
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "experiments" / "judge_pipeline_v2" / "ground_truth"
IPHO_GPT5_PDF_COMMENTS = OUT / "ipho_gpt5_pdf_comments.json"
ICHO_PROPORTION = [0, 6, 6.5, 7.5, 6, 6.5, 6, 7, 7.5, 7]
ICHO_MAX_POINT = [0, 34, 67, 111, 40, 16, 22, 22, 35, 38]


ICHO_FILES = {
    "gemini-2.5-pro": ROOT / "artifacts/expert_grading/icho2025/original/human_eval_gemini-2.5-pro_solution_20250926-203600.json",
    "gpt-5": ROOT / "artifacts/expert_grading/icho2025/original/GPT5_grading_with_judge.json",
    "glm-4.5": ROOT / "artifacts/expert_grading/icho2025/original/glm-4.5_grading_with_judge.json",
    "deepseek-v3.1": ROOT / "artifacts/expert_grading/icho2025/original/human_eval_deepseek-v3.1_solution_20250927-120001.json",
    "qwen3-235b-a22b": ROOT / "artifacts/expert_grading/icho2025/original/Qwen3-235B-2507_grading_with_judge.json",
}


IPHO_DOCX = {
    "gemini-2.5-pro": ROOT / "artifacts/expert_grading/ipho2025/original/gemini-2.5-pro.docx",
    "deepseek-v3.1": ROOT / "artifacts/expert_grading/ipho2025/original/deepseek-v3.1.docx",
}


IPHO_SUMMARY_MODEL_NAMES = {
    "gpt5": "gpt-5",
    "qwen": "qwen3-235b-a22b",
    "glm4.5": "glm-4.5",
}

IPHO_SECTION_LABELS = {
    "S1": ["A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "C1", "C2", "C3", "C4", "C5", "D1", "D2", "D3", "D4"],
    "S2": ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3", "C4", "C5"],
    "S3": ["A1", "A2", "A3", "A4", "A5", "A6", "B1", "B2", "B3", "C1", "C2", "C3"],
}
IPHO_SECTION_MAX_SCORES = {
    "S1": [0.2, 0.5, 0.5, 0.5, 0.5, 0.2, 0.5, 1.8, 0.5, 0.6, 0.7, 0.6, 0.6, 0.4, 0.2, 0.8, 0.9],
    "S2": [0.2, 0.8, 0.3, 0.3, 1.4, 0.3, 1.0, 1.0, 2.0, 1.0, 1.7],
    "S3": [0.2, 0.5, 1.2, 0.8, 0.5, 1.1, 1.1, 1.1, 1.1, 0.4, 0.7, 1.3],
}


def read_docx_text(path: Path) -> str:
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", xml)
    return unescape(re.sub(r"\s+", " ", text)).strip()


def read_docx_paragraphs(path: Path) -> list[str]:
    with ZipFile(path) as zf:
        xml = zf.read("word/document.xml").decode("utf-8", errors="ignore")
    paragraphs = []
    for para in re.findall(r"<w:p[\s\S]*?</w:p>", xml):
        text = "".join(re.findall(r"<w:t[^>]*>(.*?)</w:t>", para))
        text = unescape(re.sub(r"\s+", " ", text)).strip()
        if text:
            paragraphs.append(text)
    return paragraphs


def make_ipho_score_item(section: str, index: int, score: float, source_file: str, comment: str = "", label: str | None = None, raw_line: str | None = None) -> dict:
    item = {
        "label": label or IPHO_SECTION_LABELS[section][index],
        "human_score": score,
        "max_score": IPHO_SECTION_MAX_SCORES[section][index],
        "human_comment": comment,
        "source_file": source_file,
    }
    if raw_line is not None:
        item["raw_score_line"] = raw_line
    return item


def parse_docx_score_line(line: str) -> tuple[float, str] | None:
    if re.fullmatch(r"[A-Z]\.\d+缺失", line):
        return 0.0, line
    match = re.fullmatch(r"(\d+(?:\.\d+)?)(?:\((.*)\))?", line)
    if not match:
        return None
    return float(match.group(1)), match.group(2) or ""


def parse_ipho_docx_line_items(path: Path) -> dict:
    """Parse IPhO expert docx files that contain explicit per-subquestion rows.

    Gemini 2.5 Pro is structured as repeated triples: subquestion label, human
    score, human comment. DeepSeek V3.1 stores section totals plus score lines
    in official subquestion order, sometimes with short parenthetical comments.
    """

    paragraphs = read_docx_paragraphs(path)
    sections: dict[str, dict] = {}
    current_section = None
    source_file = str(path.relative_to(ROOT))
    i = 0
    while i < len(paragraphs):
        line = paragraphs[i].strip()
        section_match = re.fullmatch(r"S([123])\s+(?:total:?\s*)?(\d+(?:\.\d+)?)", line, flags=re.I)
        if section_match:
            current_section = f"S{section_match.group(1)}"
            sections.setdefault(current_section, {"section_total": float(section_match.group(2)), "section_max": 10.0, "scores": []})
            sections[current_section]["section_total"] = float(section_match.group(2))
            i += 1
            continue
        section_only_match = re.fullmatch(r"S([123])", line, flags=re.I)
        if section_only_match:
            current_section = f"S{section_only_match.group(1)}"
            sections.setdefault(current_section, {"section_max": 10.0, "scores": []})
            i += 1
            continue
        total_match = re.fullmatch(r"total:?\s*(\d+(?:\.\d+)?)", line, flags=re.I)
        if current_section and total_match:
            sections[current_section]["section_total"] = float(total_match.group(1))
            i += 1
            continue

        if current_section and re.fullmatch(r"[A-Z]\.\d+", line):
            score_line = paragraphs[i + 1].strip() if i + 1 < len(paragraphs) else ""
            if re.fullmatch(r"\d+(?:\.\d+)?", score_line):
                comment = paragraphs[i + 2].strip() if i + 2 < len(paragraphs) else ""
                score_index = len(sections[current_section].setdefault("scores", []))
                sections[current_section]["scores"].append(
                    make_ipho_score_item(
                        current_section,
                        score_index,
                        float(score_line),
                        source_file,
                        comment=comment,
                        label=line.replace(".", ""),
                        raw_line=score_line,
                    )
                )
                i += 3
                continue
        if current_section:
            parsed_score = parse_docx_score_line(line)
            score_index = len(sections[current_section].setdefault("scores", []))
            if parsed_score is not None and score_index < len(IPHO_SECTION_LABELS[current_section]):
                score, comment = parsed_score
                sections[current_section]["scores"].append(
                    make_ipho_score_item(
                        current_section,
                        score_index,
                        score,
                        source_file,
                        comment=comment,
                        raw_line=line,
                    )
                )
                i += 1
                continue
        i += 1

    for section_data in sections.values():
        section_data["line_item_sum"] = sum(item["human_score"] for item in section_data.get("scores", []))
        section_data.setdefault("section_total", section_data["line_item_sum"])
    return sections


def build_icho() -> dict:
    result = {
        "exam": "IChO 2025",
        "score_scale": "weighted to 60 via official problem proportions",
        "proportion": ICHO_PROPORTION,
        "max_point": ICHO_MAX_POINT,
        "models": {},
    }
    for model, path in ICHO_FILES.items():
        rows = json.loads(path.read_text(encoding="utf-8"))
        normalized_rows = []
        by_problem = defaultdict(float)
        raw_total = 0.0
        for row in rows:
            if not isinstance(row, dict):
                continue
            score = row.get("score")
            if score is None:
                score = row.get("human_score")
            if score is None:
                continue
            score = float(score)
            qid = str(row.get("question_id", "")).strip()
            sid = str(row.get("sub_question_id", "")).strip()
            q_index = int(float(qid))
            by_problem[q_index] += score
            raw_total += score
            normalized_rows.append(
                {
                    "source_index": row.get("index"),
                    "question_id": qid,
                    "sub_question_id": sid,
                    "human_score_raw": score,
                    "human_comments": row.get("comments", row.get("human_comments", "")),
                    "answer": row.get("answer", ""),
                    "judge_question_id": row.get("judge_question_id"),
                    "judge_score": row.get("judge_score"),
                    "judge_text": row.get("judge_text"),
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
        weighted_total = sum(
            by_problem[i] / ICHO_MAX_POINT[i] * ICHO_PROPORTION[i]
            for i in sorted(by_problem)
            if i < len(ICHO_MAX_POINT) and ICHO_MAX_POINT[i]
        )
        result["models"][model] = {
            "raw_total": raw_total,
            "weighted_total_60": weighted_total,
            "by_problem_raw": {str(i): by_problem[i] for i in sorted(by_problem)},
            "items": normalized_rows,
        }
    return result


def parse_ipho_summary() -> dict:
    path = ROOT / "artifacts/expert_grading/ipho2025/original/summary.txt"
    if not path.exists():
        return {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    result = {}
    current = None
    current_section = None
    for line in lines:
        key = line.lower()
        if key in IPHO_SUMMARY_MODEL_NAMES:
            current = IPHO_SUMMARY_MODEL_NAMES[key]
            result.setdefault(current, {"sections": {}, "source_file": str(path.relative_to(ROOT))})
            current_section = None
            continue
        if not current:
            continue
        if key in {"s1", "s2", "s3"}:
            current_section = key.upper()
            result[current]["sections"].setdefault(current_section, {"scores": []})
            continue
        m = re.match(r"^total:(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$", key)
        if m and float(m.group(2)) <= 10.0 and current_section:
            result[current]["sections"][current_section]["section_total"] = float(m.group(1))
            result[current]["sections"][current_section]["section_max"] = float(m.group(2))
            continue
        m = re.match(r"^total:(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$", key)
        if m:
            result[current]["total"] = float(m.group(1))
            result[current]["max_total"] = float(m.group(2))
            current_section = None
            continue
        m = re.match(r"^(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)$", key)
        if m and current_section:
            score_index = len(result[current]["sections"][current_section]["scores"])
            result[current]["sections"][current_section]["scores"].append(
                {
                    "label": IPHO_SECTION_LABELS[current_section][score_index],
                    "human_score": float(m.group(1)),
                    "max_score": float(m.group(2)),
                    "human_comment": "",
                    "source_file": str(path.relative_to(ROOT)),
                }
            )
    for model, data in result.items():
        for section_data in data["sections"].values():
            section_data["line_item_sum"] = sum(item["human_score"] for item in section_data.get("scores", []))
        if model == "gpt-5" and "S3" in data["sections"]:
            s3 = data["sections"]["S3"]
            if abs(float(s3.get("line_item_sum", 0.0)) - float(s3.get("section_total", 0.0))) > 1e-9:
                s3["printed_section_total"] = s3.get("section_total")
                s3["section_total"] = s3["line_item_sum"]
                s3["correction_note"] = "Use line-item sum; original summary/PDF printed 8.8 but the 12 item scores sum to 8.6."
        if "total" not in data:
            data["total"] = sum(sec.get("section_total", 0.0) for sec in data["sections"].values())
            data["max_total"] = sum(sec.get("section_max", 0.0) for sec in data["sections"].values())
        elif model == "gpt-5":
            corrected_total = sum(sec.get("section_total", 0.0) for sec in data["sections"].values())
            if abs(corrected_total - float(data["total"])) > 1e-9:
                data["printed_total"] = data["total"]
                data["total"] = corrected_total
                data["correction_note"] = "Use section totals after correcting S3 to its line-item sum."
    if IPHO_GPT5_PDF_COMMENTS.exists() and "gpt-5" in result:
        comment_cache = json.loads(IPHO_GPT5_PDF_COMMENTS.read_text(encoding="utf-8"))
        for section, cached_section in comment_cache.get("sections", {}).items():
            section_scores = result["gpt-5"]["sections"].get(section, {}).get("scores", [])
            for item, cached_item in zip(section_scores, cached_section.get("items", [])):
                item["human_comment"] = cached_item.get("human_comment", "")
                item["comment_source_file"] = cached_section.get("source_file")
                item["pdf_score"] = cached_item.get("pdf_score")
                item["pdf_max_score"] = cached_item.get("pdf_max_score")
                item["pdf_raw_score_line"] = cached_item.get("raw_score_line")
    return result


def build_ipho() -> dict:
    result = {
        "exam": "IPhO 2025",
        "score_scale": "30-point theory score; three 10-point sections",
        "models": {},
    }
    result["models"].update(parse_ipho_summary())
    for model, path in IPHO_DOCX.items():
        text = read_docx_text(path)
        sections = parse_ipho_docx_line_items(path)
        item_count = sum(len(section.get("scores", [])) for section in sections.values())
        if item_count != 40:
            sections = {}
            for sid, total in re.findall(r"\b(S[123])\s+total:?\s*(\d+(?:\.\d+)?)", text, flags=re.I):
                sections[sid.upper()] = {"section_total": float(total), "section_max": 10.0}
            if not sections:
                for sid, total in re.findall(r"\b(S[123])\s+(\d+(?:\.\d+)?)\b", text, flags=re.I):
                    sections.setdefault(sid.upper(), {"section_total": float(total), "section_max": 10.0})
        result["models"][model] = {
            "sections": sections,
            "total": sum(section["section_total"] for section in sections.values()),
            "max_total": 30.0,
            "human_comments_raw": text,
            "line_item_count": sum(len(section.get("scores", [])) for section in sections.values()),
            "source_file": str(path.relative_to(ROOT)),
        }
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "icho2025_expert_ground_truth.json").write_text(
        json.dumps(build_icho(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUT / "ipho2025_expert_ground_truth.json").write_text(
        json.dumps(build_ipho(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
