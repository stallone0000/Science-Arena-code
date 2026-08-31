#!/usr/bin/env python3
"""Build the current Science Arena full benchmark result table.

This is a reporting helper, not an evaluator. It consolidates the latest
clean/caveated artifacts into JSON, CSV, and Markdown tables so stale collector
outputs are not mixed into the final report.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_ROOT = ROOT / "artifacts/runs/full_benchmark_table_20260524"

MODELS = [
    "google/gemini-3.5-flash",
    "google/google/gemini-3.1-pro-preview",
    "cloudsway/gemini-3-flash-preview",
    "qiniu/openai/gpt-5.4",
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.7",
    "bytedance/doubao-seed-2-0-pro",
    "qwen/qwen3.7-max",
    "qwen/qwen3.6-plus",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "minimax/MiniMax-M2.7",
    "z-ai/glm-5.1",
    "xiaomi/mimo-v2.5-pro",
]

BENCHMARKS = [
    ("ipho2025", "IPhO 2025", 30),
    ("icho2025", "IChO 2025", 60),
    ("ccho2025", "CChO 2025", 100),
    ("cpho2025", "CPhO 2025", 320),
    ("ibo2023", "IBO 2023", 453),
    ("eupho2025", "EuPhO 2025 Theory", 30),
    ("apho2025", "APhO 2025 Theory", 30),
    ("incho2026", "INChO 2026", 106),
    ("usapho2025", "USAPhO 2025", 150),
    ("nbpho2025", "NBPhO 2025", 72),
]

BENCHMARK_STATUS = {
    "ipho2025": "paper_original; interleaved; HF rows fixed to /30",
    "icho2025": "paper_original; interleaved; image-only answer refs preserved",
    "ccho2025": "paper_original; CChO Q7 missing question-side structure image",
    "cpho2025": "paper_original; interleaved",
    "ibo2023": "paper_original; objective key scored, not LLM-judge",
    "eupho2025": "new official PDF; image-enabled static benchmark",
    "apho2025": "new official PDF; draft/human-review page-range + item-level data",
    "incho2026": "new official PDF; official rubric page-range candidate",
    "usapho2025": "new official PDF; derived-rubric auxiliary benchmark",
    "nbpho2025": "new official PDF; item-level draft, experimental/staticization caveats",
}

ORIGINAL5: dict[str, dict[str, float | str]] = {
    "google/gemini-3.5-flash": {"ipho2025": 25.67, "icho2025": 48.43, "ccho2025": 57.12, "cpho2025": 275.0, "ibo2023": 430},
    "google/google/gemini-3.1-pro-preview": {"ipho2025": 27.30, "icho2025": 46.45, "ccho2025": 57.44, "cpho2025": 302.1, "ibo2023": 431},
    "cloudsway/gemini-3-flash-preview": {"ipho2025": 24.30, "icho2025": 44.40, "ccho2025": 52.53, "cpho2025": 270.0, "ibo2023": 428},
    "qiniu/openai/gpt-5.4": {"ipho2025": 19.45, "icho2025": 30.79, "ccho2025": 41.16, "cpho2025": 211.6, "ibo2023": 358},
    "openai/gpt-5.5": {"ipho2025": 25.63, "icho2025": 33.98, "ccho2025": 46.59, "cpho2025": 289.5, "ibo2023": 415},
    "anthropic/claude-opus-4.7": {"ipho2025": 22.73, "icho2025": 41.68, "ccho2025": 49.61, "cpho2025": 273.0, "ibo2023": 407},
    "bytedance/doubao-seed-2-0-pro": {"ipho2025": 18.02, "icho2025": 38.32, "ccho2025": 45.21, "cpho2025": 227.8, "ibo2023": 415},
    "qwen/qwen3.7-max": {"ipho2025": "23.89*", "icho2025": "28.86*", "ccho2025": "45.36*", "cpho2025": "286.6*", "ibo2023": "349*"},
    "qwen/qwen3.6-plus": {"ipho2025": 22.45, "icho2025": 41.79, "ccho2025": 46.90, "cpho2025": 271.0, "ibo2023": 413},
    "deepseek/deepseek-v4-pro": {"ipho2025": "21.60*", "icho2025": "33.46*", "ccho2025": "46.59*", "cpho2025": "250.7*", "ibo2023": "327*"},
    "deepseek/deepseek-v4-flash": {"ipho2025": 23.20, "icho2025": 29.65, "ccho2025": 39.51, "cpho2025": 224.0, "ibo2023": 324},
    "minimax/MiniMax-M2.7": {"ipho2025": "17.70*", "icho2025": "19.94*", "ccho2025": "36.39*", "cpho2025": "211.2*", "ibo2023": "337*"},
    "z-ai/glm-5.1": {"ipho2025": 22.26, "icho2025": 31.23, "ccho2025": 43.81, "cpho2025": 262.4, "ibo2023": 349},
    "xiaomi/mimo-v2.5-pro": {"ipho2025": "22.35*", "icho2025": "29.10*", "ccho2025": "35.21*", "cpho2025": "219.0*", "ibo2023": "317*"},
}

MANUAL_CANDIDATE: dict[tuple[str, str], dict[str, Any]] = {
    ("incho2026", "google/gemini-3.5-flash"): {"score": 91.6, "max_score": 106, "status": "valid", "note": "standalone smoke"},
    ("incho2026", "openai/gpt-5.5"): {"score": 74.875, "max_score": 106, "status": "caveated", "note": "problem-level recovery; P2 content-filter/truncated"},
    ("usapho2025", "google/gemini-3.5-flash"): {"score": 150, "max_score": 150, "status": "valid", "note": "standalone/all-model smoke"},
    ("usapho2025", "google/google/gemini-3.1-pro-preview"): {"score": 150, "max_score": 150, "status": "valid", "note": "six problem-level repaired sum; full-row judge focus ignored"},
}


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def rows_from_batch(path: str, benchmark_id: str, expected_max: float) -> dict[str, dict[str, Any]]:
    data = load_json(ROOT / path, {})
    out: dict[str, dict[str, Any]] = {}
    for row in data.get("rows", []):
        model = row.get("model")
        if model not in MODELS:
            continue
        score = row.get("score")
        max_score = row.get("max_score")
        error = row.get("generation_error") or row.get("error") or ""
        finish = row.get("generation_finish_reason")
        status = "valid"
        note_parts = []
        if row.get("sent_image_count") == 0:
            note_parts.append("text-only")
            status = "caveated"
        if error:
            status = "invalid"
            note_parts.append(str(error)[:120])
        if score is None or max_score is None:
            status = "invalid"
        elif abs(float(max_score) - expected_max) > 1e-6:
            status = "invalid"
            note_parts.append(f"judge max_score {max_score} != {expected_max}")
        if isinstance(finish, str) and "content_filter" in finish:
            status = "invalid"
            note_parts.append("content_filter")
        elif finish == "length":
            status = "caveated"
            note_parts.append("generation length stop")
        out[model] = {
            "score": score,
            "max_score": max_score,
            "status": status,
            "note": "; ".join(note_parts),
            "source": path,
            "images_sent": row.get("sent_image_count"),
            "cost_cny": row.get("cost_cny"),
        }
    return out


def rows_from_collected() -> dict[tuple[str, str], dict[str, Any]]:
    data = load_json(ROOT / "artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json", {})
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for key, benchmark_id, expected_max in [("eupho", "eupho2025", 30), ("apho", "apho2025", 30)]:
        for row in data.get(key, []):
            model = row.get("model")
            if model not in MODELS:
                continue
            status = row.get("status") or "invalid"
            score = row.get("score")
            max_score = row.get("max_score")
            if status == "valid" and (score is None or max_score is None or abs(float(max_score) - expected_max) > 1e-6):
                status = "invalid"
            note = row.get("note", "")
            if row.get("images_sent") == 0:
                note = (note + "; text-only").strip("; ")
                if status == "valid":
                    status = "caveated"
            out[(benchmark_id, model)] = {
                "score": score,
                "max_score": max_score,
                "status": status,
                "note": note,
                "source": "artifacts/runs/candidate_benchmarks/new_benchmark_results_20260524_collected.json",
                "images_sent": row.get("images_sent"),
                "cost_cny": row.get("cost_cny"),
            }
    return out


def rows_from_nbpho_dirs() -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted((ROOT / "artifacts/runs/candidate_benchmarks").glob("nbpho2025_full_allmodels_20260524*/summary.json")):
        data = load_json(path, {})
        rejudged_path = path.parent / "summary.rejudged_full_exam.json"
        rejudged = load_json(rejudged_path, {})
        model = data.get("model")
        if model not in MODELS:
            continue
        pj = (rejudged.get("parsed_judge") if rejudged else None) or data.get("parsed_judge") or {}
        score = pj.get("total_score")
        max_score = pj.get("max_score")
        error = data.get("generation_error") or ""
        finish = data.get("generation_finish_reason")
        status = "valid"
        note_parts: list[str] = []
        if data.get("sent_image_count") == 0:
            note_parts.append("text-only")
            status = "caveated"
        if error:
            status = "invalid"
            note_parts.append(str(error)[:120])
        if score is None or max_score is None:
            status = "invalid"
        elif abs(float(max_score) - 72) > 1e-6:
            status = "invalid"
            note_parts.append(f"judge max_score {max_score} != 72")
        if isinstance(finish, str) and "content_filter" in finish:
            status = "invalid"
            note_parts.append("content_filter")
        elif finish == "length" and status == "valid":
            status = "caveated"
            note_parts.append("generation length stop")
        out[model] = {
            "score": score,
            "max_score": max_score,
            "status": status,
            "note": "; ".join(note_parts + (["rejudged full exam"] if rejudged else [])),
            "source": str((rejudged_path if rejudged else path).relative_to(ROOT)),
            "images_sent": data.get("sent_image_count"),
            "cost_cny": float(data.get("total_cost_cny") or 0) + float(rejudged.get("total_cost_cny") or 0),
        }
    return out


def fmt_cell(entry: dict[str, Any] | None) -> str:
    if not entry:
        return "pending"
    status = entry.get("status")
    score = entry.get("score")
    max_score = entry.get("max_score")
    if status == "invalid":
        return "invalid"
    if score is None:
        return "pending"
    suffix = "*" if status == "caveated" else ""
    return f"{float(score):.3g}{suffix}"


def main() -> None:
    matrix: dict[str, dict[str, dict[str, Any]]] = {model: {} for model in MODELS}
    for model, scores in ORIGINAL5.items():
        for benchmark_id, score in scores.items():
            status = "caveated" if isinstance(score, str) and score.endswith("*") else "valid"
            numeric = float(str(score).rstrip("*"))
            max_score = dict((bid, mx) for bid, _, mx in BENCHMARKS)[benchmark_id]
            matrix[model][benchmark_id] = {
                "score": numeric,
                "max_score": max_score,
                "status": status,
                "note": "text-only/image-caveated" if status == "caveated" else "",
                "source": "docs/latest_model_checkpoint_20260524_zh.md",
            }

    for (benchmark_id, model), entry in rows_from_collected().items():
        matrix[model][benchmark_id] = entry

    for path, benchmark_id, expected_max in [
        ("artifacts/runs/candidate_benchmarks/incho2026_full_allmodels_20260524_v1_mm_summary.json", "incho2026", 106),
        ("artifacts/runs/candidate_benchmarks/incho2026_full_allmodels_20260524_v2_textonly_summary.json", "incho2026", 106),
        ("artifacts/runs/candidate_benchmarks/usapho2025_full_allmodels_20260524_v1_mm_summary.json", "usapho2025", 150),
        ("artifacts/runs/candidate_benchmarks/usapho2025_full_allmodels_20260524_v2_textonly_summary.json", "usapho2025", 150),
    ]:
        for model, entry in rows_from_batch(path, benchmark_id, expected_max).items():
            matrix[model][benchmark_id] = entry

    for (benchmark_id, model), entry in MANUAL_CANDIDATE.items():
        matrix[model][benchmark_id] = entry

    for model, entry in rows_from_nbpho_dirs().items():
        matrix[model]["nbpho2025"] = entry
    for model, entry in rows_from_batch(
        "artifacts/runs/candidate_benchmarks/nbpho2025_full_allmodels_20260524_v2_textonly_summary.json",
        "nbpho2025",
        72,
    ).items():
        matrix[model]["nbpho2025"] = entry

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": "2026-05-24",
        "models": MODELS,
        "benchmarks": [
            {"benchmark_id": bid, "name": name, "max_score": max_score, "status": BENCHMARK_STATUS[bid]}
            for bid, name, max_score in BENCHMARKS
        ],
        "matrix": matrix,
        "cell_note": "A trailing * means caveated. invalid means generation/API/judge focus failed and is not a capability score.",
    }
    (OUT_ROOT / "full_benchmark_table.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (OUT_ROOT / "full_benchmark_table.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model"] + [f"{name} /{mx}" for _, name, mx in BENCHMARKS])
        for model in MODELS:
            writer.writerow([model] + [fmt_cell(matrix[model].get(bid)) for bid, _, _ in BENCHMARKS])

    lines: list[str] = []
    lines.append("# Full Science Arena Benchmark Table, 2026-05-24")
    lines.append("")
    lines.append("`*` means caveated; `invalid` is not a model-capability score; `pending` means no usable artifact yet.")
    lines.append("")
    lines.append("## Benchmark Set")
    lines.append("")
    lines.append("| Benchmark | Max | Status |")
    lines.append("|---|---:|---|")
    for bid, name, max_score in BENCHMARKS:
        lines.append(f"| {name} | {max_score:g} | {BENCHMARK_STATUS[bid]} |")
    lines.append("")
    lines.append("## Model Matrix")
    lines.append("")
    header = "| Model | " + " | ".join(f"{name} /{mx:g}" for _, name, mx in BENCHMARKS) + " |"
    sep = "|---|" + "|".join("---:" for _ in BENCHMARKS) + "|"
    lines.append(header)
    lines.append(sep)
    for model in MODELS:
        cells = [fmt_cell(matrix[model].get(bid)) for bid, _, _ in BENCHMARKS]
        lines.append("| `" + model + "` | " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("## Invalid Or Caveated Candidate Rows")
    lines.append("")
    lines.append("| Benchmark | Model | Status | Note | Source |")
    lines.append("|---|---|---|---|---|")
    for model in MODELS:
        for bid, _, _ in BENCHMARKS:
            entry = matrix[model].get(bid)
            if not entry or entry.get("status") == "valid":
                continue
            note = str(entry.get("note") or "").replace("|", "/")
            source = str(entry.get("source") or "").replace("|", "/")
            lines.append(f"| {bid} | `{model}` | {entry.get('status')} | {note} | {source} |")
    (OUT_ROOT / "full_benchmark_table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps({"out_dir": str(OUT_ROOT), "models": len(MODELS), "benchmarks": len(BENCHMARKS)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
