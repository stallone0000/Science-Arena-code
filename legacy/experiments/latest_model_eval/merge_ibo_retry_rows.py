#!/usr/bin/env python3
"""Merge successful IBO retry rows into a base run without API calls."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from run_ibo2023_latest_models import (
    DEFAULT_ITEMS_PATH,
    DEFAULT_SCORING_KEY_PATH,
    model_tag,
    quiz_index,
    score_answer,
    scoring_map_from_key,
    summarize_rows,
)


ERROR_STATUSES = {"api_error", "api_exception"}
UNUSABLE_SCORE_STATUSES = {"api_error", "api_exception", "missing_tags", "tagless_fallback_missing_tokens"}
IDENTITY_FIELDS = (
    "quiz_index",
    "source",
    "source_q_idx",
    "item_id",
    "model",
    "correct_answer",
    "image_urls",
    "sent_image_urls",
    "use_multimodal",
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_models(raw: str) -> list[str]:
    return [part.strip() for part in (raw or "").split(",") if part.strip()]


def is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    return False


def id_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value).strip()


def compact_space(value: Any) -> str:
    return " ".join(id_text(value).split())


def row_qid(row: dict[str, Any]) -> str:
    qid = compact_space(row.get("quiz_index"))
    if qid:
        return qid

    source = compact_space(row.get("source"))
    source_q_idx = compact_space(row.get("source_q_idx"))
    if source and source_q_idx:
        return f"{source} {source_q_idx}"
    return ""


def row_aliases(row: dict[str, Any]) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    qid = row_qid(row)
    if qid:
        aliases.append(("qid", qid))

    source = compact_space(row.get("source"))
    source_q_idx = compact_space(row.get("source_q_idx"))
    if source and source_q_idx:
        aliases.append(("qid", f"{source} {source_q_idx}"))

    item_id = compact_space(row.get("item_id"))
    if item_id:
        aliases.append(("item_id", item_id))

    deduped: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for alias in aliases:
        if alias not in seen:
            seen.add(alias)
            deduped.append(alias)
    return deduped


def output_text(row: dict[str, Any]) -> str:
    for key in ("llm_output", "model_output", "answer"):
        value = row.get(key)
        if not is_blank(value):
            return str(value)
    return ""


def is_successful_retry(row: Any) -> bool:
    if not isinstance(row, dict):
        return False
    if not is_blank(row.get("error")):
        return False
    if str(row.get("finish_reason") or "").strip().lower() == "content_filter":
        return False
    if row.get("score_status") in UNUSABLE_SCORE_STATUSES:
        return False
    return bool(output_text(row).strip())


def ensure_model_output(row: dict[str, Any]) -> dict[str, Any]:
    updated = dict(row)
    if is_blank(updated.get("model_output")):
        text = output_text(updated)
        if text:
            updated["model_output"] = text
    return updated


def overlay_retry_row(base_row: dict[str, Any], retry_row: dict[str, Any]) -> dict[str, Any]:
    merged = ensure_model_output(retry_row)
    for field in IDENTITY_FIELDS:
        if is_blank(merged.get(field)) and not is_blank(base_row.get(field)):
            merged[field] = base_row[field]
    return merged


def default_model_tags(base_root: Path) -> list[str]:
    return sorted(
        path.name
        for path in base_root.iterdir()
        if path.is_dir() and (path / "solutions.json").exists()
    )


def resolve_model_dir_name(raw_model: str, base_root: Path, retry_root: Path) -> str:
    raw_model = raw_model.strip()
    candidates = [raw_model, model_tag(raw_model)]
    for candidate in candidates:
        if (base_root / candidate / "solutions.json").exists() or (retry_root / candidate / "solutions.json").exists():
            return candidate
    return model_tag(raw_model)


def load_items_by_qid() -> dict[str, dict[str, Any]]:
    if not DEFAULT_ITEMS_PATH.exists():
        return {}
    items = load_json(DEFAULT_ITEMS_PATH)
    return {
        quiz_index(item): item
        for item in items
        if isinstance(item, dict) and item.get("source") and item.get("source_q_idx") is not None
    }


def item_from_row(row: dict[str, Any], items_by_qid: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    qid = row_qid(row)
    if qid in items_by_qid:
        return items_by_qid[qid]

    source = row.get("source")
    source_q_idx = row.get("source_q_idx")
    if (is_blank(source) or is_blank(source_q_idx)) and qid and " " in qid:
        source, source_q_idx = qid.rsplit(" ", 1)

    answer = row.get("correct_answer")
    if is_blank(answer):
        return None
    if is_blank(source) or is_blank(source_q_idx):
        return None
    return {"source": source, "source_q_idx": source_q_idx, "answer": answer}


def rescore_row(
    row: dict[str, Any],
    *,
    items_by_qid: dict[str, dict[str, Any]],
    scoring_map: dict[str, list[float]],
) -> dict[str, Any]:
    updated = ensure_model_output(row)
    item = item_from_row(updated, items_by_qid)
    if item is None:
        return updated

    qid = quiz_index(item)
    sub_scores = scoring_map.get(qid)
    if sub_scores is None:
        return updated

    try:
        score_row = score_answer(
            item=item,
            sub_scores=sub_scores,
            model_output=str(updated.get("model_output", "")),
            allow_tagless_parse=True,
        )
    except Exception as exc:
        updated["score_status"] = f"score_exception:{type(exc).__name__}"
        updated["score_error"] = str(exc)
        return updated

    updated["quiz_index"] = qid
    updated["source"] = item.get("source", updated.get("source", ""))
    updated["source_q_idx"] = item.get("source_q_idx", updated.get("source_q_idx"))
    updated["correct_answer"] = item.get("answer", updated.get("correct_answer", ""))
    updated["score"] = score_row["score"]
    updated["max_score"] = score_row["max_score"]
    updated["sub_results"] = score_row["sub_results"]
    updated["sub_scores_config"] = sub_scores
    updated["score_status"] = score_row["status"]
    updated["extracted_answer"] = score_row["extracted_answer"]
    updated["extracted_tokens"] = score_row["tokens"]
    return updated


def index_base_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], int]:
    index: dict[tuple[str, str], int] = {}
    for position, row in enumerate(rows):
        for alias in row_aliases(row):
            index.setdefault(alias, position)
    return index


def find_base_position(row: dict[str, Any], index: dict[tuple[str, str], int]) -> int | None:
    for alias in row_aliases(row):
        if alias in index:
            return index[alias]
    return None


def merge_model(
    *,
    model_name: str,
    base_root: Path,
    retry_root: Path,
    run_dir: Path,
    items_by_qid: dict[str, dict[str, Any]],
    scoring_map: dict[str, list[float]],
) -> dict[str, Any]:
    tag = resolve_model_dir_name(model_name, base_root, retry_root)
    base_path = base_root / tag / "solutions.json"
    retry_path = retry_root / tag / "solutions.json"
    if not base_path.exists():
        raise FileNotFoundError(base_path)
    if not retry_path.exists():
        raise FileNotFoundError(retry_path)

    base_rows = [row for row in load_json(base_path) if isinstance(row, dict)]
    retry_rows = [row for row in load_json(retry_path) if isinstance(row, dict)]
    merged_rows = [dict(row) for row in base_rows]
    base_index = index_base_rows(base_rows)

    replaced_items: list[str] = []
    skipped_items: list[dict[str, str]] = []
    for retry_row in retry_rows:
        label = row_qid(retry_row) or compact_space(retry_row.get("item_id")) or "<unknown>"
        if not is_successful_retry(retry_row):
            skipped_items.append({"item": label, "reason": "retry_not_successful"})
            continue

        position = find_base_position(retry_row, base_index)
        if position is None:
            skipped_items.append({"item": label, "reason": "no_matching_base_row"})
            continue

        merged_rows[position] = overlay_retry_row(base_rows[position], retry_row)
        replaced_items.append(row_qid(merged_rows[position]) or label)

    rescored_rows = [
        rescore_row(row, items_by_qid=items_by_qid, scoring_map=scoring_map)
        for row in merged_rows
    ]

    model_dir = run_dir / tag
    solutions_path = model_dir / "solutions.json"
    save_json(solutions_path, rescored_rows)
    save_json(model_dir / "scores.json", rescored_rows)

    summary = {
        "model": next((str(row.get("model")) for row in rescored_rows if row.get("model")), model_name),
        "model_tag": tag,
        "run_dir": str(model_dir),
        "solutions_path": str(solutions_path),
        "base_solutions_path": str(base_path),
        "retry_solutions_path": str(retry_path),
        "base_items": len(base_rows),
        "retry_items": len(retry_rows),
        "replaced_count": len(replaced_items),
        "replaced_items": replaced_items,
        "skipped_retry_count": len(skipped_items),
        "skipped_retry_items": skipped_items,
        **summarize_rows(rescored_rows),
    }
    save_json(model_dir / "summary.json", summary)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_root", type=Path, required=True)
    parser.add_argument("--retry_root", type=Path, required=True)
    parser.add_argument("--output_root", type=Path, required=True)
    parser.add_argument("--run_name", required=True)
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model names or model_tag directories. Defaults to all model dirs under --base_root.",
    )
    parser.add_argument("--scoring_key", type=Path, default=DEFAULT_SCORING_KEY_PATH)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    base_root = args.base_root.expanduser().resolve()
    retry_root = args.retry_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    run_dir = output_root / args.run_name

    models = parse_models(args.models)
    if not models:
        models = default_model_tags(base_root)
    if not models:
        raise RuntimeError(f"No model solutions found under {base_root}")

    scoring_map = scoring_map_from_key(load_json(args.scoring_key.expanduser().resolve()))
    items_by_qid = load_items_by_qid()
    created_at = datetime.now().isoformat(timespec="seconds")

    model_summaries = [
        merge_model(
            model_name=model,
            base_root=base_root,
            retry_root=retry_root,
            run_dir=run_dir,
            items_by_qid=items_by_qid,
            scoring_map=scoring_map,
        )
        for model in models
    ]

    aggregate = {
        "run_dir": str(run_dir),
        "created_at": created_at,
        "base_root": str(base_root),
        "retry_root": str(retry_root),
        "output_root": str(output_root),
        "run_name": args.run_name,
        "scoring_key": str(args.scoring_key.expanduser().resolve()),
        "models_requested": models,
        "models": model_summaries,
        "totals": {
            "cost_cny": sum(float(row.get("cost_cny", 0) or 0) for row in model_summaries),
            "prompt_tokens": sum(float(row.get("prompt_tokens", 0) or 0) for row in model_summaries),
            "completion_tokens": sum(float(row.get("completion_tokens", 0) or 0) for row in model_summaries),
            "total_tokens": sum(float(row.get("total_tokens", 0) or 0) for row in model_summaries),
            "replaced_count": sum(int(row.get("replaced_count", 0) or 0) for row in model_summaries),
        },
    }
    save_json(run_dir / "summary.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
