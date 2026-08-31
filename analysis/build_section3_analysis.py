#!/usr/bin/env python3
"""Build EMNLP Section 3 analysis artifacts for ScienceArena.

The script keeps the paper-facing artifacts separate from the canonical
benchmark files: it only reads dataset/results JSONL files and writes derived
metadata, classifications, summaries, and LaTeX table snippets.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import textwrap
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


ROOT = Path(__file__).resolve().parents[2]

DATASET = ROOT / "datasets/science_arena_text_interleaved_v1/benchmark_items.jsonl"
SUMMARY = ROOT / "datasets/science_arena_text_interleaved_v1/benchmark_summary.jsonl"
RECORDS = ROOT / "artifacts/runs/text_interleaved_eval/emnlp_section3_table_20260526/records.json"
OUT_DIR = ROOT / "emnlp-version/analysis"
RAW_DIR = ROOT / "artifacts/runs/emnlp_section3_subfield_classification_20260526"


MODEL_METADATA: list[dict[str, Any]] = [
    {
        "model_id": "cloudsway/gemini-3-flash-preview",
        "display": "Gemini 3 Flash",
        "provider": "Google/Cloudsway",
        "release_date": "2025-12-17",
        "release_source": "https://blog.google/products-and-platforms/products/gemini/gemini-3-flash/",
        "release_note": "Google Gemini 3 Flash preview/public rollout; third-party 360 route.",
        "confidence": "official_family_release",
    },
    {
        "model_id": "google/gemini-3.5-flash",
        "display": "Gemini 3.5 Flash",
        "provider": "Google",
        "release_date": "2026-05-19",
        "release_source": "https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-5/",
        "release_note": "Google official general availability date.",
        "confidence": "official",
    },
    {
        "model_id": "google/google/gemini-3.1-pro-preview",
        "display": "Gemini 3.1 Pro",
        "provider": "Google",
        "release_date": "2026-02-19",
        "release_source": "https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-3-1-pro/",
        "release_note": "Google official preview release date.",
        "confidence": "official",
    },
    {
        "model_id": "qiniu/openai/gpt-5.4",
        "display": "GPT-5.4",
        "provider": "OpenAI/Qiniu",
        "release_date": "2026-03-05",
        "release_source": "https://openai.com/index/introducing-gpt-5-4/",
        "release_note": "OpenAI official API release; third-party 360 route.",
        "confidence": "official_family_release",
    },
    {
        "model_id": "openai/gpt-5.5",
        "display": "GPT-5.5",
        "provider": "OpenAI",
        "release_date": "2026-04-23",
        "release_source": "https://openai.com/index/introducing-gpt-5-5/",
        "release_note": "OpenAI official announcement page; API availability update followed on 2026-04-24. The 360 route is accessed through Responses.",
        "confidence": "official_announcement_page",
    },
    {
        "model_id": "anthropic/claude-opus-4.7",
        "display": "Claude Opus 4.7",
        "provider": "Anthropic",
        "release_date": "2026-04-16",
        "release_source": "https://www.anthropic.com/news/claude-opus-4-7",
        "release_note": "Anthropic official general availability date.",
        "confidence": "official",
    },
    {
        "model_id": "bytedance/doubao-seed-2-0-pro",
        "display": "Doubao Seed 2.0 Pro",
        "provider": "ByteDance",
        "release_date": "2026-02-14",
        "release_source": "https://technode.com/2026/02/14/bytedance-releases-doubao-seed-2-0-positions-pro-model-against-gpt-5-2-and-gemini-3-pro/",
        "release_note": "Public report of ByteDance release; official public source not found in the local scrape.",
        "confidence": "secondary_public_report",
    },
    {
        "model_id": "qwen/qwen3.7-max",
        "display": "Qwen3.7-Max",
        "provider": "Alibaba Qwen",
        "release_date": "2026-05-20",
        "release_source": "https://qwen.ai/research",
        "release_note": "Qwen research index lists Qwen3.7-Max release on 2026-05-20.",
        "confidence": "official_index",
    },
    {
        "model_id": "qwen/qwen3.6-plus",
        "display": "Qwen3.6-Plus",
        "provider": "Alibaba Qwen",
        "release_date": "2026-04-02",
        "release_source": "https://qwen.ai/blog?id=qwen3.6-max-preview",
        "release_note": "Qwen official page references Qwen3.6-Plus as prior release; exact date cross-checked by public reports.",
        "confidence": "official_indirect_plus_secondary_date",
    },
    {
        "model_id": "deepseek/deepseek-v4-pro",
        "display": "DeepSeek V4 Pro",
        "provider": "DeepSeek",
        "release_date": "2026-04-24",
        "release_source": "https://techcrunch.com/2026/04/24/deepseek-previews-new-ai-model-that-closes-the-gap-with-frontier-models/",
        "release_note": "Public report of DeepSeek V4 Pro/Flash preview release.",
        "confidence": "secondary_public_report",
    },
    {
        "model_id": "deepseek/deepseek-v4-flash",
        "display": "DeepSeek V4 Flash",
        "provider": "DeepSeek",
        "release_date": "2026-04-24",
        "release_source": "https://techcrunch.com/2026/04/24/deepseek-previews-new-ai-model-that-closes-the-gap-with-frontier-models/",
        "release_note": "Public report of DeepSeek V4 Pro/Flash preview release.",
        "confidence": "secondary_public_report",
    },
    {
        "model_id": "minimax/MiniMax-M2.7",
        "display": "MiniMax M2.7",
        "provider": "MiniMax",
        "release_date": "2026-04-11",
        "release_source": "https://developer.nvidia.com/blog/minimax-m2-7-advances-scalable-agentic-workflows-on-nvidia-platforms-for-complex-ai-applications/",
        "release_note": "NVIDIA technical blog documents the M2.7 release; exact initial vendor launch may precede this.",
        "confidence": "partner_public_report",
    },
    {
        "model_id": "z-ai/glm-5.1",
        "display": "GLM-5.1",
        "provider": "Z.ai",
        "release_date": "2026-04-01",
        "release_source": "https://www.scbx.com/wp-content/uploads/2026/04/RD_News_ZHIPU_GLM51_MOE_MODEL_010426.pdf",
        "release_note": "Public news note on GLM-5.1 launch; exact official release page not found in scrape.",
        "confidence": "secondary_public_report",
    },
    {
        "model_id": "xiaomi/mimo-v2.5-pro",
        "display": "MiMo v2.5 Pro",
        "provider": "Xiaomi",
        "release_date": "2026-04-22",
        "release_source": "https://www1.hkexnews.hk/listedco/listconews/sehk/2026/0428/2026042800526.pdf",
        "release_note": "Xiaomi public filings and reports describe MiMo V2 family releases; exact v2.5 Pro date cross-checked by public reports.",
        "confidence": "filing_plus_secondary_date",
    },
]


MEDAL_BASELINES = {
    "ipho2025": {"winner": 29.2, "gold": 23.4, "silver": 17.0, "bronze": 11.5},
    "icho2025": {"winner": 57.1, "gold": 36.6, "silver": 28.7, "bronze": 26.3},
    "ibo2023": {"winner": 395.4, "gold": 354.7, "silver": 293.1, "bronze": 249.7},
}

IBO_OBJECTIVE_SCORES = {
    "cloudsway/gemini-3-flash-preview": 428.0,
    "google/gemini-3.5-flash": 430.0,
    "google/google/gemini-3.1-pro-preview": 431.0,
    "qiniu/openai/gpt-5.4": 358.0,
    "openai/gpt-5.5": 415.0,
    "anthropic/claude-opus-4.7": 407.0,
    "bytedance/doubao-seed-2-0-pro": 415.0,
    "qwen/qwen3.7-max": 349.0,
    "qwen/qwen3.6-plus": 413.0,
    "deepseek/deepseek-v4-pro": 327.0,
    "deepseek/deepseek-v4-flash": 324.0,
    "minimax/MiniMax-M2.7": 337.0,
    "z-ai/glm-5.1": 349.0,
    "xiaomi/mimo-v2.5-pro": 317.0,
}

IBO_OBJECTIVE_SCORE_PATHS = {
    "cloudsway/gemini-3-flash-preview": "experiments/latest_model_eval/ibo2023_runs/latest_360_20260523_v1/cloudsway__gemini-3-flash-preview/scores.json",
    "google/gemini-3.5-flash": "experiments/latest_model_eval/ibo2023_runs/latest_360_20260523_v1/google__gemini-3.5-flash/scores.json",
    "google/google/gemini-3.1-pro-preview": "experiments/latest_model_eval/ibo2023_runs/requested_20260523_v1/google__google__gemini-3.1-pro-preview/scores.json",
    "qiniu/openai/gpt-5.4": "experiments/latest_model_eval/ibo2023_runs/requested_20260523_v1/qiniu__openai__gpt-5.4/scores.json",
    "openai/gpt-5.5": "experiments/latest_model_eval/ibo2023_runs/requested_gpt55_dataurl_cleanmerge_v2_20260524/openai__gpt-5.5/scores.json",
    "anthropic/claude-opus-4.7": "experiments/latest_model_eval/ibo2023_runs/latest_360_20260524_claude47_ibo_cf_strictmerge_v3/anthropic__claude-opus-4.7/scores.json",
    "bytedance/doubao-seed-2-0-pro": "experiments/latest_model_eval/ibo2023_runs/requested_20260523_v1/bytedance__doubao-seed-2-0-pro/scores.json",
    "qwen/qwen3.7-max": "experiments/latest_model_eval/ibo2023_runs/requested_20260523_v1/qwen__qwen3.7-max/scores.json",
    "qwen/qwen3.6-plus": "experiments/latest_model_eval/ibo2023_runs/latest_360_20260523_v1/qwen__qwen3.6-plus/scores.json",
    "deepseek/deepseek-v4-pro": "experiments/latest_model_eval/ibo2023_runs/requested_20260523_ibo_merged_retries_v2/deepseek__deepseek-v4-pro/scores.json",
    "deepseek/deepseek-v4-flash": "experiments/latest_model_eval/ibo2023_runs/latest_360_20260523_v1/deepseek__deepseek-v4-flash/scores.json",
    "minimax/MiniMax-M2.7": "experiments/latest_model_eval/ibo2023_runs/latest_360_20260524_minimax_ibo_cleanmerge_v2/minimax__MiniMax-M2.7/scores.json",
    "z-ai/glm-5.1": "experiments/latest_model_eval/ibo2023_runs/latest_360_20260523_v1/z-ai__glm-5.1/scores.json",
    "xiaomi/mimo-v2.5-pro": "experiments/latest_model_eval/ibo2023_runs/latest_360_20260523_mimo_v1/xiaomi__mimo-v2.5-pro/scores.json",
}

BENCHMARK_ORDER = [
    "apho2025",
    "eupho2025",
    "nbpho2025",
    "usapho2025",
    "cpho2025",
    "ipho2025",
    "usnco2025_part2",
    "incho2026",
    "ccho2025",
    "icho2025",
    "ibo2023",
]

MODEL_ORDER = [m["model_id"] for m in MODEL_METADATA]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def safe_date(value: str | None) -> tuple[int, int, int] | None:
    if not value:
        return None
    parts = value.split("-")
    try:
        year = int(parts[0])
        month = int(parts[1]) if len(parts) > 1 else 12
        day = int(parts[2]) if len(parts) > 2 else 28
        return (year, month, day)
    except Exception:
        return None


def compare_release(model_date: str | None, exam_date: str | None) -> str:
    md = safe_date(model_date)
    ed = safe_date(exam_date)
    if not md or not ed:
        return "unknown"
    return "possible_public_contamination" if md > ed else "pre_release_holdout"


def has_staticized_figure(row: dict[str, Any]) -> bool:
    if row.get("images") or row.get("image_refs"):
        return True
    for key in ("question_figures_staticized", "solution_figures_staticized"):
        val = row.get(key)
        if isinstance(val, list) and val:
            return True
        if isinstance(val, str) and val.strip():
            return True
    text = "\n".join(
        str(row.get(k, ""))
        for k in ("question_text_staticized", "official_answer_staticized", "official_rubric_staticized")
    ).lower()
    return any(x in text for x in ["figure", "diagram", "graph", "shown below", "structure below", "image"])


def heuristic_classify(row: dict[str, Any]) -> dict[str, Any]:
    text = "\n".join(
        str(row.get(k, ""))
        for k in (
            "question_text_staticized",
            "official_answer_staticized",
            "official_detailed_solution_staticized",
            "official_rubric_staticized",
        )
    ).lower()
    discipline = row["discipline"]
    subfield = "other"
    secondary: list[str] = []
    tags: set[str] = set()

    def mark_if(cond: bool, tag: str) -> None:
        if cond:
            tags.add(tag)

    if discipline == "physics":
        checks = [
            ("astrophysics", ["galaxy", "stellar", "star", "orbit", "hydrogen line", "rotation curve", "doppler"]),
            ("modern_quantum_atomic", ["bohr", "quantum", "photon", "electron", "wavefunction", "energy level"]),
            ("mechanics", ["force", "mass", "velocity", "acceleration", "trajectory", "momentum", "angular momentum"]),
            ("electromagnetism", ["electric", "magnetic", "field", "charge", "capacitor", "current", "voltage"]),
            ("thermodynamics", ["temperature", "heat", "entropy", "pressure", "gas", "thermal"]),
            ("optics_waves", ["wave", "interference", "diffraction", "lens", "optical", "sound"]),
            ("experimental_data", ["uncertainty", "measurement", "fit", "graph", "plot", "table", "data"]),
        ]
    elif discipline == "chemistry":
        checks = [
            ("organic_structure", ["structure", "stereochem", "stereoisomer", "draw", "nmr", "cycl", "benz", "alkene", "compound"]),
            ("analytical_spectroscopy", ["spectrum", "nmr", "ir ", "mass spect", "absorbance", "chromatography", "assay"]),
            ("physical_chemistry", ["equilibrium", "thermodynamic", "kinetic", "rate", "gibbs", "enthalpy", "entropy"]),
            ("inorganic_coordination", ["complex", "ligand", "coordination", "metal", "oxidation state"]),
            ("electrochemistry", ["electrode", "cell potential", "redox", "voltage", "nernst"]),
            ("biochemistry", ["enzyme", "protein", "amino acid", "biosynthesis", "metabol"]),
            ("laboratory_experimental", ["titration", "reagent", "procedure", "yield", "precipitate", "experiment"]),
        ]
    else:
        checks = [
            ("molecular_genetics", ["dna", "rna", "gene", "mutation", "pcr", "genome", "transcription", "translation"]),
            ("biochemistry", ["enzyme", "protein", "metabol", "atp", "pathway", "substrate"]),
            ("cell_biology", ["cell", "membrane", "organelle", "mitosis", "meiosis"]),
            ("physiology", ["blood", "hormone", "neuron", "muscle", "kidney", "respiration"]),
            ("ecology_evolution", ["population", "species", "evolution", "selection", "ecology", "phylogen"]),
            ("plant_biology", ["plant", "leaf", "chloroplast", "photosynthesis", "root"]),
            ("experimental_data", ["figure", "graph", "table", "experiment", "control", "statistic"]),
        ]

    scored = []
    for name, keys in checks:
        score = sum(1 for k in keys if k in text)
        if score:
            scored.append((score, name))
    if scored:
        scored.sort(reverse=True)
        subfield = scored[0][1]
        secondary = [name for _, name in scored[1:3]]

    mark_if(has_staticized_figure(row), "staticized_visual_evidence")
    mark_if(any(k in text for k in ["figure", "graph", "plot", "shown", "diagram", "table"]), "visual_or_table_reasoning")
    mark_if(any(k in text for k in ["structure", "stereochem", "draw", "skeletal", "compound"]), "chemical_structure_reasoning")
    mark_if(any(k in text for k in ["calculate", "derive", "numerical", "estimate", "compute", "equation"]), "quantitative_calculation")
    mark_if(len(text) > 7000, "long_context")

    return {
        "benchmark_id": row["benchmark_id"],
        "sequence_id": row["sequence_id"],
        "turn_id": row["turn_id"],
        "discipline": discipline,
        "primary_subfield": subfield,
        "secondary_subfields": secondary,
        "skill_tags": sorted(tags),
        "requires_visual_reasoning": "visual_or_table_reasoning" in tags or "staticized_visual_evidence" in tags,
        "requires_structural_diagram": "chemical_structure_reasoning" in tags,
        "requires_quantitative_calculation": "quantitative_calculation" in tags,
        "requires_long_context": "long_context" in tags,
        "classification_source": "heuristic_fallback",
        "confidence": 0.55,
        "rationale_short": "Keyword- and metadata-based fallback classification.",
    }


def make_sequence_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["benchmark_id"], row["sequence_id"])].append(row)
    summaries = []
    for (benchmark, seq), seq_rows in sorted(grouped.items()):
        seq_rows = sorted(seq_rows, key=lambda r: r.get("turn_index", 0))
        sample_text = "\n\n".join(
            f"[{r['turn_id']}]\nQUESTION: {str(r.get('question_text_staticized',''))[:900]}\n"
            f"ANSWER/RUBRIC: {str(r.get('official_answer_staticized',''))[:450]} "
            f"{str(r.get('official_rubric_staticized',''))[:450]}"
            for r in seq_rows[:6]
        )
        summaries.append(
            {
                "benchmark_id": benchmark,
                "sequence_id": seq,
                "discipline": seq_rows[0]["discipline"],
                "turn_ids": [r["turn_id"] for r in seq_rows],
                "turn_count": len(seq_rows),
                "has_staticized_visual": any(has_staticized_figure(r) for r in seq_rows),
                "text_excerpt": sample_text[:7000],
            }
        )
    return summaries


def load_api_key() -> str | None:
    for name in ("QIHOO_API_KEY", "QIHOO360_API_KEY", "360_API_KEY"):
        if os.environ.get(name):
            return os.environ[name]
    return None


def extract_json_array(text: str) -> list[dict[str, Any]]:
    match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, flags=re.S)
    if match:
        text = match.group(1)
    else:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            text = text[start : end + 1]
    return json.loads(text)


def classify_with_api(
    summaries: list[dict[str, Any]],
    *,
    model: str,
    batch_size: int,
    raw_dir: Path,
    sleep: float = 0.0,
) -> dict[str, dict[str, Any]]:
    if requests is None:
        raise RuntimeError("requests is not installed")
    key = load_api_key()
    if not key:
        raise RuntimeError("Missing QIHOO_API_KEY/QIHOO360_API_KEY/360_API_KEY")

    raw_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, dict[str, Any]] = {}
    taxonomy = """
Physics taxonomy: mechanics, electromagnetism, thermodynamics, optics_waves,
modern_quantum_atomic, astrophysics, experimental_data, mathematical_methods, other.
Chemistry taxonomy: organic_structure, stereochemistry, analytical_spectroscopy,
physical_chemistry, inorganic_coordination, electrochemistry, biochemistry,
laboratory_experimental, materials_polymer, other.
Biology taxonomy: molecular_genetics, biochemistry, cell_biology, physiology,
ecology_evolution, plant_biology, animal_behavior_neurobiology, bioinformatics_statistics,
experimental_data, other.
Skill tags: quantitative_calculation, visual_or_table_reasoning, chemical_structure_reasoning,
long_context, symbolic_derivation, experimental_design, factual_recall.
"""
    for batch_idx in range(0, len(summaries), batch_size):
        batch = summaries[batch_idx : batch_idx + batch_size]
        prompt = f"""You are classifying Science Olympiad benchmark sequences for paper analysis.
Use the taxonomy below. Return ONLY a JSON array, one object per input sequence.

{taxonomy}

For every object include:
sequence_id, primary_subfield, secondary_subfields (array), skill_tags (array),
requires_visual_reasoning (boolean), requires_structural_diagram (boolean),
requires_quantitative_calculation (boolean), requires_long_context (boolean),
confidence (0-1), rationale_short (<=18 words).

Do not solve or grade the problems. Classify from the text/rubric only.

INPUT SEQUENCES:
{json.dumps(batch, ensure_ascii=False)}
"""
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "max_completion_tokens": 65536,
            "temperature": 0,
        }
        started = time.time()
        resp = requests.post(
            "https://api.360.cn/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {key}",
                "Content-Type": "application/json",
                "Host": "api.360.cn",
            },
            json=payload,
            timeout=240,
        )
        elapsed = time.time() - started
        raw_payload = {
            "batch_start": batch_idx,
            "batch_size": len(batch),
            "request_model": model,
            "elapsed_seconds": elapsed,
            "status_code": resp.status_code,
            "usage": None,
            "raw_text": "",
            "error": "",
        }
        try:
            data = resp.json()
        except Exception as exc:
            raw_payload["error"] = f"non_json_response: {exc}"
            raw_payload["raw_text"] = resp.text[:5000]
            write_json(raw_dir / f"batch_{batch_idx:03d}.json", raw_payload)
            raise RuntimeError(raw_payload["error"])
        raw_payload["usage"] = data.get("usage")
        if resp.status_code >= 400:
            raw_payload["error"] = json.dumps(data, ensure_ascii=False)[:2000]
            write_json(raw_dir / f"batch_{batch_idx:03d}.json", raw_payload)
            raise RuntimeError(raw_payload["error"])
        content = data["choices"][0]["message"].get("content", "")
        raw_payload["raw_text"] = content
        write_json(raw_dir / f"batch_{batch_idx:03d}.json", raw_payload)
        parsed = extract_json_array(content)
        ids = {s["sequence_id"] for s in batch}
        for item in parsed:
            seq = item.get("sequence_id")
            if seq in ids:
                out[seq] = {
                    "primary_subfield": item.get("primary_subfield", "other"),
                    "secondary_subfields": item.get("secondary_subfields") or [],
                    "skill_tags": item.get("skill_tags") or [],
                    "requires_visual_reasoning": bool(item.get("requires_visual_reasoning")),
                    "requires_structural_diagram": bool(item.get("requires_structural_diagram")),
                    "requires_quantitative_calculation": bool(item.get("requires_quantitative_calculation")),
                    "requires_long_context": bool(item.get("requires_long_context")),
                    "classification_source": f"api:{model}",
                    "confidence": float(item.get("confidence", 0.7)),
                    "rationale_short": str(item.get("rationale_short", ""))[:160],
                }
        if sleep:
            time.sleep(sleep)
    return out


def build_classifications(
    rows: list[dict[str, Any]],
    *,
    use_api: bool,
    model: str,
    batch_size: int,
) -> list[dict[str, Any]]:
    seq_class: dict[str, dict[str, Any]] = {}
    if use_api:
        summaries = make_sequence_summaries(rows)
        seq_class = classify_with_api(summaries, model=model, batch_size=batch_size, raw_dir=RAW_DIR)

    classifications = []
    for row in rows:
        base = heuristic_classify(row)
        if row["sequence_id"] in seq_class:
            api_cls = seq_class[row["sequence_id"]]
            base.update(api_cls)
            base["classification_source"] = api_cls["classification_source"]
        classifications.append(base)
    return classifications


def load_turn_results(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        if rec.get("status") != "ok":
            continue
        if rec.get("benchmark_id") == "ibo2023":
            continue
        run_dir = Path(rec["run_dir"])
        candidates = list(run_dir.glob("*/turn_results.jsonl"))
        if not candidates:
            continue
        with candidates[0].open() as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                if item.get("generation_error") or item.get("judge_error"):
                    continue
                max_score = float(item.get("max_score") or 0.0)
                score = float(item.get("score") or 0.0)
                rows.append(
                    {
                        "benchmark_id": item["benchmark_id"],
                        "model": rec["model"],
                        "turn_id": item["turn_id"],
                        "sequence_id": item.get("sequence_id"),
                        "score": score,
                        "max_score": max_score,
                        "normalized_score": score / max_score if max_score > 0 else None,
                    }
                )
    return rows


def load_ibo_objective_turn_results() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model, rel_path in IBO_OBJECTIVE_SCORE_PATHS.items():
        path = ROOT / rel_path
        if not path.exists():
            continue
        items = json.loads(path.read_text())
        for idx, item in enumerate(items, start=1):
            max_score = float(item.get("max_score") or 0.0)
            score = float(item.get("score") or 0.0)
            turn_id = f"ibo2023_item_{idx:03d}"
            rows.append(
                {
                    "benchmark_id": "ibo2023",
                    "model": model,
                    "turn_id": turn_id,
                    "sequence_id": turn_id,
                    "score": score,
                    "max_score": max_score,
                    "normalized_score": score / max_score if max_score > 0 else None,
                }
            )
    return rows


def aggregate(items: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for item in items:
        key = tuple(item.get(k) for k in keys)
        g = groups.setdefault(
            key,
            {
                **{k: item.get(k) for k in keys},
                "score_sum": 0.0,
                "max_score_sum": 0.0,
                "row_count": 0,
                "model_count_set": set(),
                "benchmark_count_set": set(),
            },
        )
        g["score_sum"] += float(item.get("score") or 0.0)
        g["max_score_sum"] += float(item.get("max_score") or 0.0)
        g["row_count"] += 1
        if item.get("model"):
            g["model_count_set"].add(item["model"])
        if item.get("benchmark_id"):
            g["benchmark_count_set"].add(item["benchmark_id"])
    out = []
    for g in groups.values():
        max_sum = g.pop("max_score_sum")
        score_sum = g.pop("score_sum")
        g["score_sum"] = round(score_sum, 4)
        g["max_score_sum"] = round(max_sum, 4)
        g["score_rate"] = round(score_sum / max_sum, 4) if max_sum > 0 else None
        g["model_count"] = len(g.pop("model_count_set"))
        g["benchmark_count"] = len(g.pop("benchmark_count_set"))
        out.append(g)
    out.sort(key=lambda x: (str(x.get(keys[0])), -(x.get("max_score_sum") or 0)))
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def score_to_medal(benchmark: str, score: float | None, text: str) -> str:
    if score is None or benchmark not in MEDAL_BASELINES:
        return text
    base = MEDAL_BASELINES[benchmark]
    if score >= base["winner"]:
        return f"\\Winner{{{text}}}"
    if score >= base["gold"]:
        return f"\\Gold{{{text}}}"
    if score >= base["silver"]:
        return f"\\Silver{{{text}}}"
    if score >= base["bronze"]:
        return f"\\Bronze{{{text}}}"
    return text


def latex_escape(s: str) -> str:
    return (
        s.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
        .replace("#", "\\#")
    )


def fmt_score(score: float | None, benchmark: str, contam: str, rank_macro: str | None = None) -> str:
    if score is None:
        return "--"
    text = f"{score:.2f}"
    text = score_to_medal(benchmark, score, text)
    if rank_macro:
        text = f"{rank_macro}{{{text}}}"
    if contam == "possible_public_contamination":
        text += "$^{\\dagger}$"
    elif contam == "unknown":
        text += "$^{?}$"
    return text


def make_main_table(
    records: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    contamination: dict[str, dict[str, str]],
) -> str:
    by_bench = {r["benchmark_id"]: r for r in summaries}
    by_cell: dict[tuple[str, str], dict[str, Any]] = {
        (r["model"], r["benchmark_id"]): r for r in records
    }
    headers = []
    for b in BENCHMARK_ORDER:
        s = by_bench[b]
        short = {
            "apho2025": "APhO",
            "eupho2025": "EuPhO",
            "nbpho2025": "NBPhO",
            "usapho2025": "USAPhO",
            "cpho2025": "CPhO",
            "ipho2025": "IPhO",
            "usnco2025_part2": "USNCO",
            "incho2026": "INChO",
            "ccho2025": "CChO",
            "icho2025": "IChO",
            "ibo2023": "IBO",
        }[b]
        headers.append(f"\\makecell{{{short}\\\\{s['release_date']}/{int(s['report_scale'])}}}")
    non_world = [b for b in BENCHMARK_ORDER if b not in MEDAL_BASELINES]
    rank_macros: dict[tuple[str, str], str] = {}
    for b in non_world:
        vals = []
        for model in MODEL_ORDER:
            rec = by_cell.get((model, b))
            if rec and rec.get("status") == "ok":
                vals.append((float(rec["reported_score"]), model))
        vals.sort(reverse=True)
        for idx, (_, model) in enumerate(vals[:3], start=1):
            rank_macros[(model, b)] = {1: "\\RankOne", 2: "\\RankTwo", 3: "\\RankThree"}[idx]
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.1pt}",
        "\\renewcommand{\\arraystretch}{1.08}",
        "\\resizebox{\\textwidth}{!}{%",
        "\\begin{tabular}{ll" + "r" * len(BENCHMARK_ORDER) + "}",
        "\\toprule",
        "\\textbf{Model} & \\textbf{Rel.} & " + " & ".join(headers) + " \\\\",
        "\\midrule",
    ]
    meta = {m["model_id"]: m for m in MODEL_METADATA}
    for model in MODEL_ORDER:
        m = meta[model]
        row = [latex_escape(m["display"]), m["release_date"][:7]]
        for b in BENCHMARK_ORDER:
            rec = by_cell.get((model, b))
            score = None
            if b == "ibo2023" and model in IBO_OBJECTIVE_SCORES:
                score = IBO_OBJECTIVE_SCORES[model]
            elif rec and rec.get("status") == "ok":
                score = float(rec["reported_score"])
            row.append(fmt_score(score, b, contamination[model][b], rank_macros.get((model, b))))
        lines.append(" & ".join(row) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}%",
        "}",
        "\\vspace{-0.4em}",
        "\\caption{Main evaluation on eleven olympiad-style benchmarks. Open-ended columns use text-only interleaved solving and rubric-faithful judging; IBO uses the dedicated objective-key IBO runner. Column headers show competition, public release month, and full score. The release column reports public/model-registry availability month. $^{\\dagger}$ marks cells where the model became publicly available after the exam materials, so public-data contamination cannot be ruled out. Medal icons are shown only for international competitions (IPhO, IChO, IBO). For other competitions, \\RankOne{red}, \\RankTwo{orange}, and \\RankThree{green} mark first, second, and third place within the column. ``--'' denotes an API/generation/judge route failure and is not a capability score.}",
        "\\label{tab:latest-all-benchmarks}",
        "\\end{table*}",
    ]
    return "\n".join(lines)


def make_subfield_table(subfield_rows: list[dict[str, Any]]) -> str:
    # Compact paper table: best/worst substantial buckets only.
    substantial = [
        r for r in subfield_rows if r.get("max_score_sum", 0) >= 50 and r.get("row_count", 0) >= 20
    ]
    substantial.sort(key=lambda r: r["score_rate"])
    selected = substantial[:6] + substantial[-6:]
    seen = set()
    unique = []
    for r in selected:
        key = (r["discipline"], r["primary_subfield"])
        if key not in seen:
            seen.add(key)
            unique.append(r)
    unique.sort(key=lambda r: (r["discipline"], r["score_rate"]))
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\begin{tabular}{llrr}",
        "\\toprule",
        "\\textbf{Domain} & \\textbf{Subfield} & \\textbf{Score} & \\textbf{N} \\\\",
        "\\midrule",
    ]
    for r in unique:
        domain = {"physics": "Phys.", "chemistry": "Chem.", "biology": "Bio."}.get(r["discipline"], r["discipline"])
        sf = latex_escape(str(r["primary_subfield"]).replace("_", " "))
        lines.append(f"{domain} & {sf} & {100*r['score_rate']:.1f}\\% & {r['row_count']} \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "\\caption{Subfield diagnostics aggregated over clean model--benchmark cells. Score is total awarded points divided by total available points within the bucket; $N$ counts model--turn grading instances.}",
        "\\label{tab:subfield-diagnostics}",
        "\\end{table}",
    ]
    return "\n".join(lines)


def make_equal_benchmark_domain_summary(
    records: list[dict[str, Any]], summaries: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    bench_by_domain: dict[str, list[str]] = defaultdict(list)
    for summary in summaries:
        bench_by_domain[summary["discipline"]].append(summary["benchmark_id"])
    cells: dict[str, dict[str, float]] = defaultdict(dict)
    for record in records:
        if record.get("status") != "ok":
            continue
        cells[record["model"]][record["benchmark_id"]] = float(record["reported_score"]) / float(
            record["report_scale"]
        )
    out = []
    for domain, benchmarks in sorted(bench_by_domain.items()):
        for model in MODEL_ORDER:
            vals = [cells[model][b] for b in benchmarks if b in cells[model]]
            out.append(
                {
                    "domain": domain,
                    "model": model,
                    "benchmark_count": len(vals),
                    "required_benchmark_count": len(benchmarks),
                    "complete": len(vals) == len(benchmarks),
                    "equal_benchmark_score_rate": round(sum(vals) / len(vals), 4) if vals else None,
                }
            )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--use_api", action="store_true")
    ap.add_argument("--classification_model", default="google/gemini-3.5-flash")
    ap.add_argument("--batch_size", type=int, default=12)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(DATASET)
    summaries = read_jsonl(SUMMARY)
    records = json.loads(RECORDS.read_text())

    write_json(OUT_DIR / "model_release_metadata.json", MODEL_METADATA)
    write_json(OUT_DIR / "benchmark_release_metadata.json", summaries)

    contamination = {
        m["model_id"]: {
            b["benchmark_id"]: compare_release(m.get("release_date"), b.get("release_date"))
            for b in summaries
        }
        for m in MODEL_METADATA
    }
    write_json(OUT_DIR / "contamination_matrix.json", contamination)

    classifications = build_classifications(
        rows,
        use_api=args.use_api,
        model=args.classification_model,
        batch_size=args.batch_size,
    )
    with (OUT_DIR / "item_subfield_classification.jsonl").open("w") as f:
        for item in classifications:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    class_by_turn = {c["turn_id"]: c for c in classifications}
    turn_results = load_turn_results(records)
    turn_results.extend(load_ibo_objective_turn_results())
    joined = []
    for tr in turn_results:
        cls = class_by_turn.get(tr["turn_id"], {})
        joined.append({**tr, **cls})

    by_subfield = aggregate(joined, ["discipline", "primary_subfield"])
    by_image = aggregate(joined, ["discipline", "requires_visual_reasoning"])
    by_structure = aggregate(joined, ["discipline", "requires_structural_diagram"])
    by_quant = aggregate(joined, ["discipline", "requires_quantitative_calculation"])
    by_model_domain = aggregate(joined, ["model", "discipline"])
    by_benchmark = aggregate(joined, ["benchmark_id"])
    domain_complete = make_equal_benchmark_domain_summary(records, summaries)

    analysis = {
        "created_at": dt.datetime.now().isoformat(timespec="seconds"),
        "dataset_rows": len(rows),
        "clean_turn_result_rows": len(joined),
        "classification_source_counts": Counter(c["classification_source"] for c in classifications),
        "by_subfield": by_subfield,
        "by_image": by_image,
        "by_structure": by_structure,
        "by_quantitative": by_quant,
        "by_model_domain": by_model_domain,
        "by_model_domain_equal_benchmark": domain_complete,
        "by_benchmark": by_benchmark,
        "invalid_cells": [
            {
                "benchmark_id": r["benchmark_id"],
                "model": r["model"],
                "status_reasons": r.get("status_reasons", []),
            }
            for r in records
            if r.get("status") != "ok"
        ],
    }
    write_json(OUT_DIR / "section3_analysis_summary.json", analysis)
    write_csv(OUT_DIR / "subfield_performance_summary.csv", by_subfield)
    write_csv(OUT_DIR / "image_performance_summary.csv", by_image)
    write_csv(OUT_DIR / "model_domain_summary.csv", by_model_domain)
    write_csv(OUT_DIR / "domain_complete_model_summary.csv", domain_complete)
    write_json(OUT_DIR / "domain_complete_model_summary.json", domain_complete)

    main_table = make_main_table(records, summaries, contamination)
    subfield_table = make_subfield_table(by_subfield)
    (OUT_DIR / "section3_main_table.tex").write_text(main_table + "\n")
    (OUT_DIR / "section3_subfield_table.tex").write_text(subfield_table + "\n")

    # Short machine-readable facts for paper drafting.
    ok_records = [r for r in records if r.get("status") == "ok"]
    model_domain_best = {}
    for domain in ("physics", "chemistry", "biology"):
        candidates = [r for r in by_model_domain if r.get("discipline") == domain]
        if candidates:
            best = max(candidates, key=lambda r: r["score_rate"])
            model_domain_best[domain] = best
    facts = {
        "model_count": len(MODEL_METADATA),
        "benchmark_count": len(summaries),
        "ok_cells": len(ok_records),
        "total_cells": len(records),
        "invalid_cell_count": len(records) - len(ok_records),
        "best_by_domain": model_domain_best,
        "best_by_domain_equal_benchmark_complete_only": {
            domain: max(
                [r for r in domain_complete if r["domain"] == domain and r["complete"]],
                key=lambda r: r["equal_benchmark_score_rate"] or 0.0,
            )
            for domain in sorted({r["domain"] for r in domain_complete})
            if any(r["domain"] == domain and r["complete"] for r in domain_complete)
        },
        "main_table_tex": str(OUT_DIR / "section3_main_table.tex"),
        "subfield_table_tex": str(OUT_DIR / "section3_subfield_table.tex"),
    }
    write_json(OUT_DIR / "section3_key_facts.json", facts)
    print(json.dumps(facts, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
