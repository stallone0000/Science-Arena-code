#!/usr/bin/env python3
"""Recompute saved IBO 2023 model totals without external API calls."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "results/llm_human_evaluation/IBO 23"
SCORING_KEY = ROOT / "data/science_arena/ibo2023/scoring_key.json"


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def expected_total() -> float:
    rows = load_json(SCORING_KEY)
    return float(sum(sum(float(x) for x in row.get("sub_scores", [])) for row in rows))


def model_rows(path: Path) -> list[dict]:
    data = load_json(path)
    return [row for row in data if isinstance(row, dict)]


def saved_score_for(stem: str, saved: dict) -> float | None:
    aliases = {
        "claude-opus-4.1": "amazon/claude-opus-4.1",
        "claude-opus-4.5": "anthropic/claude-opus-4.5",
        "gemini-2.5-pro": "google/gemini-2.5-pro",
        "gemini-3-pro-preview": "google/gemini-3-pro-preview",
        "gpt-5": "openai/gpt-5",
        "gpt-5.1": "openai/gpt-5.1",
        "qwen3-235b-a22b": "huaweicloud/qwen3-235b-a22b",
        "grok-4": "xai/grok-4",
    }
    for key in (stem, aliases.get(stem, "")):
        if key and key in saved:
            return float(saved[key])
    return None


def main() -> None:
    expected_items = len(load_json(SCORING_KEY))
    max_total = expected_total()
    all_score_path = RESULT_DIR / "all_score_list.json"
    saved = load_json(all_score_path) if all_score_path.exists() else {}
    report = {
        "exam": "IBO 2023",
        "source_result_dir": str(RESULT_DIR.relative_to(ROOT)),
        "expected_items": expected_items,
        "max_total": max_total,
        "models": {},
    }
    for path in sorted(RESULT_DIR.glob("*.json")):
        if path.name == "all_score_list.json":
            continue
        rows = model_rows(path)
        total = float(sum(float(row.get("new_score", 0.0)) for row in rows))
        saved_score = saved_score_for(path.stem, saved)
        missing = [idx for idx, row in enumerate(rows) if "new_score" not in row]
        tagless = [
            row.get("quiz_index")
            for row in rows
            if "<boa>" not in str(row.get("model_output", "")) or "<eoa>" not in str(row.get("model_output", ""))
        ]
        report["models"][path.stem] = {
            "items": len(rows),
            "score": total,
            "saved_score": saved_score,
            "matches_saved": saved_score is None or abs(saved_score - total) < 1e-9,
            "missing_new_score": missing,
            "tagless_items_requiring_saved_llm_judge_subresults": tagless,
        }
    out = ROOT / "experiments/reproducibility/ibo2023_saved_score_reproduction.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
