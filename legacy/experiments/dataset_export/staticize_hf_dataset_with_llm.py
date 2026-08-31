#!/usr/bin/env python3
"""Staticize the HF-style Science Arena export into text-only interleaved rows.

The source export preserves original image assets.  This script uses one fixed
multimodal model to convert each row into a canonical text-only problem,
solution, and rubric while keeping the original image metadata for auditability.
Downstream model evaluation should use only the staticized text fields.
"""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
INTERLEAVED_DIR = ROOT / "new-things/interleaved"
if str(INTERLEAVED_DIR) not in sys.path:
    sys.path.insert(0, str(INTERLEAVED_DIR))

from api_client_interleaved import UnifiedChatClient, build_user_message  # noqa: E402


DEFAULT_ENV_FILE = Path("/home/jovyan/zhaoguangxiang-data-zzzc/zhaoguangxiang/.config/science_arena/360.env")
DEFAULT_PRICES = ROOT / "experiments/judge_pipeline_v2/model_prices_360.json"
DEFAULT_IN = ROOT / "datasets/science_arena_hf"
DEFAULT_OUT = ROOT / "datasets/science_arena_text_interleaved_v1"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


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


def file_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def usage_cost(usage: dict[str, Any], model: str, prices: dict[str, Any]) -> dict[str, float]:
    price = prices.get(model, {})
    prompt = float(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion = float(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total = float(usage.get("total_tokens") or prompt + completion)
    input_price = float(price.get("input_price_cny_per_1m", 0) or 0)
    output_price = float(price.get("output_price_cny_per_1m", 0) or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
        "cost_cny": prompt / 1_000_000 * input_price + completion / 1_000_000 * output_price,
    }


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        import re

        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError:
        pass
    import re

    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    for candidate in reversed(fenced):
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else {"raw": parsed}
        except json.JSONDecodeError:
            continue
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return {"parse_error": "no_json_object", "raw_text": text[:4000]}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError as exc:
        return {"parse_error": f"json_decode_error: {exc}", "raw_text": text[:4000]}


def asset_local_path(asset: dict[str, Any]) -> Path | None:
    local = str(asset.get("local_path") or "").strip()
    if not local:
        return None
    path = Path(local)
    return path if path.exists() else None


def truncate_text(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n\n[TRUNCATED_BY_STATICIZER_INPUT_LIMIT]"


def build_staticization_prompt(row: dict[str, Any], max_chars: int) -> str:
    question_assets = [
        {
            "asset_index": idx,
            "asset_id": asset.get("asset_id"),
            "original": asset.get("original"),
            "hf_path": asset.get("hf_path"),
            "mime_type": asset.get("mime_type"),
            "bytes": asset.get("bytes"),
            "status": asset.get("status"),
        }
        for idx, asset in enumerate(row.get("images", []) or [], start=1)
    ]
    answer_assets = [
        {
            "asset_index": idx,
            "asset_id": asset.get("asset_id"),
            "original": asset.get("original"),
            "hf_path": asset.get("hf_path"),
            "mime_type": asset.get("mime_type"),
            "bytes": asset.get("bytes"),
            "status": asset.get("status"),
        }
        for idx, asset in enumerate(row.get("official_answer_images", []) or [], start=1)
    ]
    payload = {
        "benchmark_id": row.get("benchmark_id"),
        "discipline": row.get("discipline"),
        "sequence_id": row.get("sequence_id"),
        "turn_id": row.get("turn_id"),
        "turn_index": row.get("turn_index"),
        "max_score": row.get("max_score"),
        "grading_type": row.get("grading_type"),
        "release_date": row.get("release_date"),
        "source_quality": row.get("source_quality"),
        "status": row.get("status"),
        "caveats": row.get("caveats", []),
        "original_prompt_text": truncate_text(row.get("prompt", ""), max_chars),
        "official_answer_text": truncate_text(row.get("official_answer", ""), max_chars),
        "official_rubric_text": truncate_text(row.get("official_rubric", ""), max_chars),
        "question_image_assets_in_order": question_assets,
        "official_answer_image_assets_in_order": answer_assets,
    }
    return (
        "You are converting a public Science Arena benchmark row into a canonical text-only row. "
        "The downstream solver models will NOT receive images, so every necessary diagram, graph, table, "
        "spectrum, molecular structure, apparatus drawing, and visual label must be transcribed into text. "
        "Do not solve the problem beyond faithfully preserving the official answer/rubric already provided. "
        "Be conservative: if a visual detail is unreadable, flag it rather than inventing it. "
        "Return only valid JSON with exactly these keys:\n"
        "- question_text_staticized: complete standalone text-only prompt for this interleaved turn.\n"
        "- official_answer_staticized: official final answer in text.\n"
        "- official_detailed_solution_staticized: official worked solution/explanation in text.\n"
        "- official_rubric_staticized: detailed scoring rubric, including partial-credit point allocations.\n"
        "- question_figures: list of {asset_index, asset_id, description, extracted_values, uncertainty_notes}.\n"
        "- solution_figures: list of {asset_index, asset_id, description, extracted_values, uncertainty_notes}.\n"
        "- completeness: {question_complete, figures_complete, answer_complete, rubric_complete, max_score_reconciled, issues}.\n"
        "- reviewer_notes: concise notes for a human dataset auditor.\n\n"
        "For sketches requested from students, state exactly what features a correct sketch should contain in the rubric. "
        "Preserve original units, constants, labels, coordinates, and point values. "
        "If the original row is already text-only, still normalize formatting and keep the same meaning.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def fallback_staticized_row(row: dict[str, Any], *, reason: str) -> dict[str, Any]:
    return {
        "question_text_staticized": row.get("prompt", ""),
        "official_answer_staticized": row.get("official_answer", ""),
        "official_detailed_solution_staticized": row.get("official_answer", ""),
        "official_rubric_staticized": row.get("official_rubric", ""),
        "question_figures": [],
        "solution_figures": [],
        "completeness": {
            "question_complete": bool(str(row.get("prompt", "")).strip()),
            "figures_complete": not (row.get("images") or row.get("official_answer_images")),
            "answer_complete": bool(str(row.get("official_answer", "")).strip()),
            "rubric_complete": bool(str(row.get("official_rubric", "")).strip()),
            "max_score_reconciled": row.get("max_score") is not None,
            "issues": [reason] if reason else [],
        },
        "reviewer_notes": reason,
    }


def staticize_one(
    row: dict[str, Any],
    *,
    client: UnifiedChatClient,
    model: str,
    max_tokens: int,
    thinking_mode: str,
    thinking_effort: str,
    text_max_chars: int,
    prices: dict[str, Any],
    force_llm: bool,
) -> dict[str, Any]:
    has_images = bool(row.get("images") or row.get("official_answer_images"))
    if not has_images and not force_llm:
        parsed = fallback_staticized_row(row, reason="no_image_row_passthrough")
        response: dict[str, Any] = {"usage": {}, "finish_reason": "passthrough", "client_meta": {}}
    else:
        image_urls: list[str] = []
        image_assets: list[dict[str, Any]] = []
        for asset in list(row.get("images", []) or []) + list(row.get("official_answer_images", []) or []):
            path = asset_local_path(asset)
            if path:
                image_urls.append(file_to_data_url(path))
                image_assets.append(asset)
        prompt = build_staticization_prompt(row, text_max_chars)
        if has_images and not image_urls:
            parsed = fallback_staticized_row(row, reason="image_assets_missing_or_unreadable")
            response = {"usage": {}, "finish_reason": "missing_images", "client_meta": {}}
        else:
            response = client.chat(
                model=model,
                messages=[build_user_message(text=prompt, image_urls=image_urls, use_multimodal=bool(image_urls))],
                max_tokens=max_tokens,
                thinking_mode=thinking_mode,
                thinking_effort=thinking_effort,
            )
            text = response.get("content") or response.get("llm_output") or ""
            parsed = extract_json_object(text)
            if parsed.get("parse_error"):
                fallback = fallback_staticized_row(row, reason=f"staticizer_parse_error:{parsed.get('parse_error')}")
                fallback["raw_staticizer_text"] = text[:4000]
                parsed = fallback
    usage = usage_cost(response.get("usage", {}) or {}, model, prices)
    output = {
        **row,
        "original_prompt": row.get("prompt", ""),
        "question_text_staticized": str(parsed.get("question_text_staticized") or row.get("prompt", "")),
        "official_answer_staticized": str(parsed.get("official_answer_staticized") or row.get("official_answer", "")),
        "official_detailed_solution_staticized": str(
            parsed.get("official_detailed_solution_staticized")
            or parsed.get("official_answer_staticized")
            or row.get("official_answer", "")
        ),
        "official_rubric_staticized": str(parsed.get("official_rubric_staticized") or row.get("official_rubric", "")),
        "question_figures_staticized": parsed.get("question_figures") or [],
        "solution_figures_staticized": parsed.get("solution_figures") or [],
        "staticization_completeness": parsed.get("completeness") or {},
        "staticization_reviewer_notes": parsed.get("reviewer_notes", ""),
        "staticization_model": model,
        "staticization_finish_reason": response.get("finish_reason"),
        "staticization_error": response.get("error", ""),
        "staticization_usage": usage,
        "staticization_client_meta": response.get("client_meta", {}),
        "evaluation_protocol": "text_only_interleaved_staticized_v1",
    }
    output["prompt"] = output["question_text_staticized"]
    output["official_answer"] = output["official_answer_staticized"]
    output["official_rubric"] = output["official_rubric_staticized"]
    return output


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('benchmark_id')}::{row.get('sequence_id')}::{row.get('turn_id')}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=Path, default=DEFAULT_IN)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--model", default="google/google/gemini-3.1-pro-preview")
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_delay", type=float, default=5.0)
    parser.add_argument("--thinking_mode", default="auto")
    parser.add_argument("--thinking_effort", default="auto")
    parser.add_argument("--text_max_chars", type=int, default=60000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--benchmark_id", action="append", default=[])
    parser.add_argument("--force_llm_all_rows", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    os.environ.update(load_env_file(args.env_file))
    prices = load_json(DEFAULT_PRICES, {})
    rows = iter_jsonl(args.input_dir / "benchmark_items.jsonl")
    if args.benchmark_id:
        allowed = set(args.benchmark_id)
        rows = [row for row in rows if row.get("benchmark_id") in allowed]
    if args.limit:
        rows = rows[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_jsonl = args.output_dir / "benchmark_items.jsonl"
    done: dict[str, dict[str, Any]] = {}
    if args.resume and out_jsonl.exists():
        for row in iter_jsonl(out_jsonl):
            done[row_key(row)] = row

    pending = [row for row in rows if row_key(row) not in done]
    client = UnifiedChatClient(
        api_key=api_key_from_env(),
        backend="requests",
        max_retries=args.max_retries,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
    )

    results = dict(done)
    started = time.time()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {
            pool.submit(
                staticize_one,
                row,
                client=client,
                model=args.model,
                max_tokens=args.max_tokens,
                thinking_mode=args.thinking_mode,
                thinking_effort=args.thinking_effort,
                text_max_chars=args.text_max_chars,
                prices=prices,
                force_llm=args.force_llm_all_rows,
            ): row
            for row in pending
        }
        for index, future in enumerate(as_completed(futures), start=1):
            row = futures[future]
            key = row_key(row)
            try:
                results[key] = future.result()
            except Exception as exc:  # noqa: BLE001
                failed = fallback_staticized_row(row, reason=f"staticizer_exception:{type(exc).__name__}:{exc}")
                results[key] = {
                    **row,
                    "original_prompt": row.get("prompt", ""),
                    "question_text_staticized": failed["question_text_staticized"],
                    "official_answer_staticized": failed["official_answer_staticized"],
                    "official_detailed_solution_staticized": failed["official_detailed_solution_staticized"],
                    "official_rubric_staticized": failed["official_rubric_staticized"],
                    "question_figures_staticized": [],
                    "solution_figures_staticized": [],
                    "staticization_completeness": failed["completeness"],
                    "staticization_reviewer_notes": failed["reviewer_notes"],
                    "staticization_model": args.model,
                    "staticization_finish_reason": "exception",
                    "staticization_error": str(exc),
                    "staticization_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0, "cost_cny": 0},
                    "evaluation_protocol": "text_only_interleaved_staticized_v1",
                }
            ordered = [results[row_key(item)] for item in rows if row_key(item) in results]
            write_jsonl(out_jsonl, ordered)
            if index % 10 == 0 or index == len(pending):
                print(
                    json.dumps(
                        {
                            "completed_pending": index,
                            "pending_total": len(pending),
                            "row_key": key,
                            "elapsed_seconds": round(time.time() - started, 1),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    final_rows = [results[row_key(row)] for row in rows if row_key(row) in results]
    write_jsonl(out_jsonl, final_rows)

    summaries: dict[str, dict[str, Any]] = {}
    for row in final_rows:
        summary = summaries.setdefault(
            row["benchmark_id"],
            {
                "benchmark_id": row["benchmark_id"],
                "discipline": row.get("discipline", ""),
                "release_date": row.get("release_date", ""),
                "source_quality": row.get("source_quality", ""),
                "item_count": 0,
                "sequence_count": 0,
                "max_score_sum": 0.0,
                "staticization_errors": 0,
                "incomplete_rows": 0,
                "staticization_cost_cny": 0.0,
            },
        )
        summary["item_count"] += 1
        summary["max_score_sum"] += float(row.get("max_score") or 0)
        summary["staticization_cost_cny"] += float((row.get("staticization_usage") or {}).get("cost_cny") or 0)
        if row.get("staticization_error"):
            summary["staticization_errors"] += 1
        comp = row.get("staticization_completeness") or {}
        if any(comp.get(key) is False for key in ["question_complete", "answer_complete", "rubric_complete"]):
            summary["incomplete_rows"] += 1
    for benchmark_id, summary in summaries.items():
        summary["sequence_count"] = len({row["sequence_id"] for row in final_rows if row["benchmark_id"] == benchmark_id})

    write_jsonl(args.output_dir / "benchmark_summary.jsonl", sorted(summaries.values(), key=lambda row: row["benchmark_id"]))
    info = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "source_input_dir": str(args.input_dir),
        "schema": "science_arena_text_only_interleaved_staticized_v1",
        "staticization_model": args.model,
        "row_count": len(final_rows),
        "benchmark_count": len(summaries),
        "total_staticization_cost_cny": sum(float(row.get("staticization_usage", {}).get("cost_cny") or 0) for row in final_rows),
        "evaluation_rule": "Solver APIs receive only question_text_staticized plus prior interleaved text turns; no image blocks and no image URLs are sent during model evaluation.",
    }
    (args.output_dir / "dataset_info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "README.md").write_text(
        "# Science Arena Text-Only Interleaved Staticized Dataset\n\n"
        "This dataset is derived from `datasets/science_arena_hf` by a fixed multimodal staticizer. "
        "Original images and source references are retained for audit, but downstream model evaluation uses "
        "`question_text_staticized` only. This makes every API route comparable under the same text-only "
        "interleaved protocol.\n",
        encoding="utf-8",
    )
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
