#!/usr/bin/env python3
"""Build a text-only interleaved USNCO 2025 Part II benchmark draft.

ACS' public PDF is accessible through the browser indexer here even when direct
scripted PDF download is blocked by ACS' WAF.  The source text contains the
official questions and solutions, but not a fine-grained official point rubric.
This builder therefore preserves official question weights and solutions, and
uses the staticizer model to derive subpart rubrics from those official sources.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INTERLEAVED_DIR = ROOT / "new-things/interleaved"
if str(INTERLEAVED_DIR) not in sys.path:
    sys.path.insert(0, str(INTERLEAVED_DIR))

from api_client_interleaved import UnifiedChatClient  # noqa: E402


DEFAULT_ENV_FILE = Path("/home/jovyan/zhaoguangxiang-data-zzzc/zhaoguangxiang/.config/science_arena/360.env")
DEFAULT_SOURCE = ROOT / "data/science_arena/candidate_benchmarks/usnco2025_jina/2025-usnco-national-exam-part-ii.pdf.md"
DEFAULT_OUT = ROOT / "data/science_arena/usnco2025/fix_USNCO-2025-PartII-staticized.json"
DEFAULT_ARTIFACT = ROOT / "artifacts/runs/candidate_benchmarks/usnco2025_part2_staticize_20260524"


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip().removeprefix("export").strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def api_key_from_env() -> str:
    key = (
        os.environ.get("QIHOO_API_KEY")
        or os.environ.get("QIHOO360_API_KEY")
        or os.environ.get("360_API_KEY")
        or os.environ.get("API_KEY")
    )
    if not key:
        raise RuntimeError("Missing 360 API key in QIHOO_API_KEY/QIHOO360_API_KEY/360_API_KEY/API_KEY.")
    return key


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def extract_json_array(text: str) -> list[dict[str, Any]]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, list):
            return [row for row in parsed if isinstance(row, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("problems"), list):
            return [row for row in parsed["problems"] if isinstance(row, dict)]
    except json.JSONDecodeError:
        pass
    match = re.search(r"(\[\s*\{.*\}\s*\])", cleaned, flags=re.S)
    if not match:
        raise ValueError("No JSON array found in staticizer response.")
    parsed = json.loads(match.group(1))
    if not isinstance(parsed, list):
        raise ValueError("Parsed JSON is not an array.")
    return [row for row in parsed if isinstance(row, dict)]


def build_prompt(source_md: str) -> str:
    return (
        "You are building a Science Arena benchmark from the public 2025 U.S. National Chemistry Olympiad "
        "National Exam Part II official PDF text. Convert the source into exactly 8 interleaved text-only "
        "problem rows, one row per written question. The source includes official question weights in percent "
        "and official solutions, but no official fine-grained subpart point rubric. Preserve that fact.\n\n"
        "For each row return these keys:\n"
        "- question_id: usnco2025_part2_q1 ... usnco2025_part2_q8\n"
        "- title\n"
        "- max_score: numeric official percent weight, so the total is 100\n"
        "- question_text_staticized: complete standalone text-only question. Replace every graph, spectrum, "
        "unit-cell diagram, organic structure, and reaction drawing with all information recoverable from the "
        "official question+solution text. If a visual cannot be fully reconstructed, say so explicitly.\n"
        "- official_answer_staticized\n"
        "- official_detailed_solution_staticized\n"
        "- official_rubric_staticized: a detailed DERIVED partial-credit rubric adding up to max_score, based only "
        "on the official solution and subparts. Mark it as derived, not official, in the text.\n"
        "- visual_staticization_notes\n"
        "- completeness: {question_complete, figures_complete, answer_complete, rubric_complete, issues}\n"
        "- source_refs: include ACS official PDF URL and local Jina markdown path.\n\n"
        "Important known visuals to staticize from the surrounding text and official solution: Q1 visible spectrum "
        "has maximum absorption at 763 nm; Q3 calorimetry plot has line y = 1.141x + 19.03 for <12 g DMEA and "
        "plateau/average y = 31.84 for >=12 g DMEA; Q4 standard cell graph has E = 0.7887 - 0.000515 T, and the "
        "0.100 M Pb2+ condition keeps the intercept but makes the slope more negative by 9.92e-5 V/K; Q7 diagrams "
        "are perovskite/YBa2Cu3O7 unit cells with sphere assignments described in the official solution; Q5/Q8 "
        "organic structure drawings must be rendered as unambiguous IUPAC/condensed structural text.\n\n"
        "Return only valid JSON, no markdown.\n\n"
        "SOURCE MARKDOWN:\n"
        + source_md
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--artifact_dir", type=Path, default=DEFAULT_ARTIFACT)
    parser.add_argument("--model", default="google/google/gemini-3.1-pro-preview")
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--max_retries", type=int, default=2)
    args = parser.parse_args()

    os.environ.update(load_env_file(args.env_file))
    source_md = args.source.read_text(encoding="utf-8")
    prompt = build_prompt(source_md)
    client = UnifiedChatClient(
        api_key=api_key_from_env(),
        backend="requests",
        max_retries=args.max_retries,
        retry_delay=5,
        timeout=args.timeout,
    )
    response = client.chat(
        model=args.model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=args.max_tokens,
        thinking_mode="auto",
        thinking_effort="auto",
    )
    text = response.get("content") or response.get("llm_output") or ""
    rows = extract_json_array(text)
    for index, row in enumerate(rows, start=1):
        row.setdefault("question_id", f"usnco2025_part2_q{index}")
        row["benchmark_id"] = "usnco2025_part2"
        row["discipline"] = "chemistry"
        row["release_date"] = "2025-06"
        row["source_quality"] = "official_pdf_text_staticized_derived_rubric"
        row["evaluation_protocol"] = "text_only_interleaved_staticized_v1"
        row["grading_type"] = "derived_rubric_llm_judge"
        row["staticization_model"] = args.model
        row["official_rubric_status"] = "derived_from_official_solution_and_official_question_weight"
        row["source_urls"] = [
            "https://www.acs.org/content/dam/acsorg/education/students/highschool/olympiad/pastexams/2025-usnco-national-exam-part-ii.pdf"
        ]
        row["source_files"] = [str(args.source.relative_to(ROOT))]
    save_json(args.output, rows)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        args.artifact_dir / "build_config.json",
        jsonable(vars(args) | {"created_at": datetime.now().isoformat(timespec="seconds")}),
    )
    save_json(args.artifact_dir / "response.json", response)
    save_json(args.artifact_dir / "parsed_rows.json", rows)
    print(json.dumps({"output": str(args.output), "artifact_dir": str(args.artifact_dir), "rows": len(rows)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
