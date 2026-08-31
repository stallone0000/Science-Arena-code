#!/usr/bin/env python3
"""Collect latest-model Science Arena results from existing artifacts.

The collector is intentionally read-only unless --output is supplied. It does
not call model or judge APIs; it only reads batch summaries, generation
summaries/logs, judge aggregate/partial files, and IBO per-model summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
JUDGE_RUNS = ROOT / "experiments/judge_pipeline_v2/runs"
PRICES_PATH = ROOT / "experiments/judge_pipeline_v2/model_prices_360.json"

DEFAULT_BATCH_ROOTS = [
    ROOT / "artifacts/runs/latest_360_20260522_calibrated_v1",
    ROOT / "artifacts/runs/latest_360_20260523_qwen36_clean_v1",
    ROOT / "artifacts/runs/latest_360_20260523_qwen36_icho_retry_v1",
    ROOT / "artifacts/runs/latest_360_20260523_tail_v1",
    ROOT / "artifacts/runs/latest_360_20260523_tail2_v1",
    ROOT / "artifacts/runs/latest_360_20260523_mimo_icho_clean_v1",
    ROOT / "artifacts/runs/latest_360_20260523_dsv4pro_icho_retry_v1",
    ROOT / "artifacts/runs/latest_360_20260523_dsv4pro_ipho_s2_retry_v1",
    ROOT / "artifacts/runs/latest_360_20260523_ccho_cpho_v1",
    ROOT / "artifacts/runs/latest_360_20260523_ccho_cpho_tail_v1",
    ROOT / "artifacts/runs/latest_360_20260523_ccho_cpho_tail2_v1",
    ROOT / "artifacts/runs/latest_360_20260523_dsv4pro_ccho_retry_v1",
    ROOT / "artifacts/runs/latest_360_20260523_dsv4pro_nopersist_v1",
    ROOT / "artifacts/runs/latest_360_20260523_dsv4pro_singleuser_v1",
    ROOT / "artifacts/runs/latest_360_20260523_dsv4pro_icho_img1_v1",
    ROOT / "artifacts/runs/latest_360_20260523_dsv4pro_ccho_img1_v1",
    ROOT / "artifacts/runs/latest_360_20260523_dsv4pro_targeted_merged_v1",
    ROOT / "artifacts/runs/latest_360_20260523_cloudsway_ccho_cpho_retry_v1",
    ROOT / "artifacts/runs/latest_360_20260523_qwen36_ccho_cpho_retry_v1",
]
DEFAULT_IBO_ROOTS = [
    ROOT / "experiments/latest_model_eval/ibo2023_runs/latest_360_20260523_v1",
    ROOT / "experiments/latest_model_eval/ibo2023_runs/latest_360_20260523_mimo_v1",
]

EXAM_ORDER = ["ipho", "icho", "cpho", "ccho", "ibo"]
EXAM_LABELS = {
    "ipho": "IPhO",
    "icho": "IChO",
    "cpho": "CPhO",
    "ccho": "CChO",
    "ibo": "IBO",
}
EXAM_MAX_SCORES = {
    "ipho": 30.0,
    "icho": 60.0,
    "cpho": 320.0,
    "ccho": 100.0,
    "ibo": 453.0,
}
IPHO_SECTION_MAX = {"S1": 10.0, "S2": 10.0, "S3": 10.0}
STANDARD_EXAM_CONFIGS = {
    "ccho": {
        "proportion": [0, 5, 9, 10, 10, 14, 9, 9, 7, 13, 14],
        "max_point": [0, 14, 27, 28, 28, 40, 25, 25, 18, 29, 33],
    },
    "cpho": {
        "proportion": [0, 45, 45, 45, 45, 45, 50, 45],
        "max_point": [0, 45, 45, 45, 45, 45, 50, 45],
    },
    "icho": {
        "proportion": [0, 6, 6.5, 7.5, 6, 6.5, 6, 7, 7.5, 7],
        "max_point": [0, 34, 67, 111, 40, 16, 22, 22, 35, 38],
    },
}

OK_STATUSES = {"ok", "skipped_existing"}
MODEL_WIDE_SUPERSEDE = {
    "latest_360_20260523_qwen36_clean_v1": {"qwen/qwen3.6-plus"},
}
REPLACEABLE_STATUSES = {"failed", "unknown"}


@dataclass
class Piece:
    model: str
    model_tag: str
    exam: str
    section: str
    source_root: str
    source_priority: int
    source_kind: str = "batch"
    score: float | None = None
    max_score: float | None = None
    status: str = "unknown"
    generation_status: str = ""
    judge_status: str = ""
    items: int = 0
    attempted_items: int = 0
    errors: int = 0
    missing_tags: int = 0
    generation_prompt_tokens: float = 0.0
    generation_completion_tokens: float = 0.0
    generation_total_tokens: float = 0.0
    judge_prompt_tokens: float = 0.0
    judge_completion_tokens: float = 0.0
    judge_total_tokens: float = 0.0
    generation_cost_cny: float = 0.0
    judge_cost_cny: float = 0.0
    note: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def total_tokens(self) -> float:
        return self.generation_total_tokens + self.judge_total_tokens

    @property
    def total_cost_cny(self) -> float:
        return self.generation_cost_cny + self.judge_cost_cny


def load_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default
    except json.JSONDecodeError as exc:
        return {"_read_error": f"{type(exc).__name__}: {exc}", "_path": str(path)}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(value, dict):
                    rows.append(value)
    except FileNotFoundError:
        pass
    return rows


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def model_tag(model: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "__", model.strip())
    return tag.strip("_") or "model"


def model_from_tag(tag: str) -> str:
    return tag.replace("__", "/", 1)


def cost_from_tokens(model: str, prompt_tokens: float, completion_tokens: float, prices: dict[str, Any]) -> float:
    price = prices.get(model, {}) if isinstance(prices, dict) else {}
    input_price = as_float(price.get("input_price_cny_per_1m"))
    output_price = as_float(price.get("output_price_cny_per_1m"))
    return prompt_tokens / 1_000_000 * input_price + completion_tokens / 1_000_000 * output_price


def usage_cost(usage: dict[str, Any], model: str, prices: dict[str, Any]) -> dict[str, float]:
    prompt_tokens = as_float(usage.get("prompt_tokens") or usage.get("input_tokens"))
    completion_tokens = as_float(usage.get("completion_tokens") or usage.get("output_tokens"))
    total_tokens = as_float(usage.get("total_tokens"), prompt_tokens + completion_tokens)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_cny": cost_from_tokens(model, prompt_tokens, completion_tokens, prices),
    }


def summarize_solution_usage(solution_path: Path, model: str, prices: dict[str, Any]) -> dict[str, Any]:
    rows = load_json(solution_path, [])
    if not isinstance(rows, list):
        return {"units": 0, "errors": 0, "prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0, "cost_cny": 0.0}
    prompt_tokens = 0.0
    completion_tokens = 0.0
    total_tokens = 0.0
    cost_cny = 0.0
    errors = 0
    for row in rows:
        if not isinstance(row, dict):
            continue
        if row.get("error"):
            errors += 1
        cost = usage_cost(row.get("usage", {}) or {}, model, prices)
        prompt_tokens += cost["prompt_tokens"]
        completion_tokens += cost["completion_tokens"]
        total_tokens += cost["total_tokens"]
        cost_cny += cost["cost_cny"]
    return {
        "units": len(rows),
        "errors": errors,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_cny": cost_cny,
    }


def log_exit_status(path: Path) -> tuple[str, int | None]:
    if not path.exists():
        return "", None
    text = path.read_text(encoding="utf-8", errors="replace")
    matches = list(re.finditer(r"\[exit_code\]\s+(-?\d+)", text))
    last_command = max(text.rfind("\n$ "), text.rfind("\n/usr/bin/python"), text.rfind("\npython "))
    if matches and last_command > matches[-1].end():
        return "running", None
    if not matches:
        return "running", None
    code = int(matches[-1].group(1))
    return ("ok" if code == 0 else "failed"), code


def parse_log_model(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    for pattern in (
        r"--pre_model\s+(\S+)",
        r"--model_under_test\s+(\S+)",
        r"\[CFG\]\s+pre_model=(\S+)",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
    return ""


def parse_judge_run_name(path: Path) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r"--run_name\s+(\S+)", text)
    if match:
        return match.group(1).strip()
    match = re.search(r"/experiments/judge_pipeline_v2/runs/([^/\s]+)", text)
    if match:
        return match.group(1).strip()
    return ""


def parse_log_exam_section(log_path: Path, suffix: str) -> tuple[str, str] | None:
    name = log_path.name
    if not name.endswith(suffix):
        return None
    stem = name[: -len(suffix)]
    if stem.startswith("ipho_"):
        return "ipho", stem.split("_", 1)[1].upper()
    return stem, ""


def standard_weighted_score(rows: list[dict[str, Any]], exam: str) -> float:
    config = STANDARD_EXAM_CONFIGS.get(exam)
    if not config:
        return sum(as_float(row.get("score")) for row in rows if not row.get("error") and row.get("score") is not None)
    by_problem: dict[int, float] = {}
    for row in rows:
        if row.get("error") or row.get("score") is None:
            continue
        qid = as_int(as_float(row.get("question_id")))
        by_problem[qid] = by_problem.get(qid, 0.0) + as_float(row.get("score"))
    total = 0.0
    for qid, score in by_problem.items():
        if qid < len(config["max_point"]) and config["max_point"][qid]:
            total += score / config["max_point"][qid] * config["proportion"][qid]
    return total


def aggregate_judge_rows(rows: list[dict[str, Any]], exam: str, model: str, status: str, expected_items: int = 0) -> dict[str, Any]:
    valid_rows = [row for row in rows if not row.get("error") and row.get("score") is not None]
    score = standard_weighted_score(valid_rows, exam) if exam in STANDARD_EXAM_CONFIGS else sum(as_float(row.get("score")) for row in valid_rows)
    prompt_tokens = 0.0
    completion_tokens = 0.0
    total_tokens = 0.0
    cost_cny = 0.0
    for row in rows:
        cost = row.get("cost") or row.get("usage_cost") or {}
        if cost:
            prompt_tokens += as_float(cost.get("prompt_tokens"))
            completion_tokens += as_float(cost.get("completion_tokens"))
            total_tokens += as_float(cost.get("total_tokens"))
            cost_cny += as_float(cost.get("cost_cny"))
    errors = sum(1 for row in rows if row.get("error"))
    return {
        "model": model,
        "score": score,
        "items": len(valid_rows),
        "attempted_items": len(rows),
        "expected_items": expected_items,
        "cost_cny": cost_cny,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "errors": errors,
        "status": status,
    }


def judge_metrics_from_dir(judge_dir: Path, model: str, exam: str) -> dict[str, Any]:
    aggregate = load_json(judge_dir / "aggregate.json", {})
    if isinstance(aggregate, dict) and aggregate:
        row = aggregate.get(model)
        if row is None and len(aggregate) == 1:
            row = next(iter(aggregate.values()))
        if isinstance(row, dict):
            row_with_totals = dict(row)
            row_with_totals["total_tokens"] = as_float(
                row_with_totals.get("total_tokens"),
                as_float(row_with_totals.get("prompt_tokens")) + as_float(row_with_totals.get("completion_tokens")),
            )
            return {**row_with_totals, "status": "ok"}

    final_rows = load_json(judge_dir / "judge_results.json", None)
    if isinstance(final_rows, list):
        return aggregate_judge_rows(final_rows, exam, model, "ok")

    partial_rows = load_json(judge_dir / "judge_results_partial.json", None)
    if isinstance(partial_rows, list):
        cases = load_json(judge_dir / "cases_manifest.json", [])
        expected = len(cases) if isinstance(cases, list) else 0
        status = "partial" if expected and len(partial_rows) < expected else "partial"
        return aggregate_judge_rows(partial_rows, exam, model, status, expected_items=expected)

    return {}


def piece_status(generation_status: str, judge_status: str, errors: int, score: float | None) -> str:
    if generation_status == "running" or judge_status == "running":
        return "running"
    if generation_status == "failed" or judge_status == "skipped_generation_failed":
        return "failed"
    if judge_status == "partial":
        return "partial"
    if judge_status == "failed":
        return "failed" if score is None else "partial"
    if score is None:
        if generation_status in OK_STATUSES:
            return "generated"
        return "unknown"
    if errors:
        return "complete_with_errors"
    if generation_status in OK_STATUSES and judge_status in OK_STATUSES:
        return "ok"
    return "partial"


def piece_from_summary_result(result: dict[str, Any], root: Path, priority: int) -> Piece | None:
    model = str(result.get("model") or "")
    exam = str(result.get("exam") or "")
    if not model or exam not in {"ipho", "icho", "cpho", "ccho"}:
        return None
    section = str(result.get("section") or "").upper()
    generation = result.get("generation") or {}
    judge = result.get("judge") or {}
    generation_status = str(result.get("generation_status") or "")
    judge_status = str(result.get("judge_status") or "")
    gen_errors = as_int(generation.get("errors"))
    judge_errors = as_int(judge.get("errors"))
    score = judge.get("score")
    score_value = as_float(score) if score is not None else None
    generation_prompt_tokens = as_float(generation.get("prompt_tokens"))
    generation_completion_tokens = as_float(generation.get("completion_tokens"))
    generation_total_tokens = as_float(generation.get("total_tokens"), generation_prompt_tokens + generation_completion_tokens)
    judge_prompt_tokens = as_float(judge.get("prompt_tokens"))
    judge_completion_tokens = as_float(judge.get("completion_tokens"))
    judge_total_tokens = as_float(judge.get("total_tokens"), judge_prompt_tokens + judge_completion_tokens)
    piece = Piece(
        model=model,
        model_tag=str(result.get("model_tag") or model_tag(model)),
        exam=exam,
        section=section,
        source_root=rel(root),
        source_priority=priority,
        score=score_value,
        max_score=IPHO_SECTION_MAX.get(section, EXAM_MAX_SCORES.get(exam)) if exam == "ipho" else EXAM_MAX_SCORES.get(exam),
        generation_status=generation_status,
        judge_status=judge_status,
        items=as_int(judge.get("items")),
        attempted_items=as_int(judge.get("attempted_items")),
        errors=gen_errors + judge_errors,
        generation_prompt_tokens=generation_prompt_tokens,
        generation_completion_tokens=generation_completion_tokens,
        generation_total_tokens=generation_total_tokens,
        judge_prompt_tokens=judge_prompt_tokens,
        judge_completion_tokens=judge_completion_tokens,
        judge_total_tokens=judge_total_tokens,
        generation_cost_cny=as_float(generation.get("cost_cny")),
        judge_cost_cny=as_float(judge.get("cost_cny")),
        details={"run_dir": result.get("run_dir"), "judge_run_dir": result.get("judge_run_dir")},
    )
    piece.status = piece_status(generation_status, judge_status, piece.errors, score_value)
    return piece


def generation_piece_from_dir(summary_path: Path, root: Path, priority: int, prices: dict[str, Any]) -> Piece | None:
    run_dir = summary_path.parent
    parts = run_dir.relative_to(root).parts
    if len(parts) < 3 or parts[1] != "interleaved":
        return None
    tag = parts[0]
    exam = parts[2]
    if exam not in {"ipho", "icho", "cpho", "ccho"}:
        return None
    section = parts[3].upper() if exam == "ipho" and len(parts) > 3 else ""
    summary = load_json(summary_path, {})
    if not isinstance(summary, dict):
        return None
    generation = summary.get("generation") if isinstance(summary.get("generation"), dict) else summary
    config = load_json(run_dir / "config.json", {})
    model = ""
    if isinstance(config, dict):
        model = str(config.get("pre_model") or config.get("generation", {}).get("pre_model") or "")
    model = model or model_from_tag(tag)
    solution_path = run_dir / "solutions.json"
    has_token_totals = any(key in generation for key in ("prompt_tokens", "completion_tokens", "total_tokens"))
    usage = summarize_solution_usage(solution_path, model, prices) if solution_path.exists() and not has_token_totals else {}
    prompt_tokens = as_float(generation.get("prompt_tokens"), as_float(usage.get("prompt_tokens")))
    completion_tokens = as_float(generation.get("completion_tokens"), as_float(usage.get("completion_tokens")))
    total_tokens = as_float(generation.get("total_tokens"), prompt_tokens + completion_tokens)
    gen_log_name = f"ipho_{section}_generation.log" if exam == "ipho" else f"{exam}_generation.log"
    judge_log_name = f"ipho_{section}_judge.log" if exam == "ipho" else f"{exam}_judge.log"
    generation_status, _ = log_exit_status(root / "_logs" / tag / gen_log_name)
    if not generation_status:
        generation_status = "ok" if solution_path.exists() else ""
    judge_status, _ = log_exit_status(root / "_logs" / tag / judge_log_name)
    judge_run_name = parse_judge_run_name(root / "_logs" / tag / judge_log_name)
    judge = judge_metrics_from_dir(JUDGE_RUNS / judge_run_name, model, exam) if judge_run_name else {}
    if judge:
        judge_status = str(judge.get("status") or judge_status or "ok")
    score = judge.get("score") if isinstance(judge, dict) else None
    score_value = as_float(score) if score is not None else None
    piece = Piece(
        model=model,
        model_tag=tag,
        exam=exam,
        section=section,
        source_root=rel(root),
        source_priority=priority,
        score=score_value,
        max_score=IPHO_SECTION_MAX.get(section, EXAM_MAX_SCORES.get(exam)) if exam == "ipho" else EXAM_MAX_SCORES.get(exam),
        generation_status=generation_status,
        judge_status=judge_status,
        items=as_int(judge.get("items")),
        attempted_items=as_int(judge.get("attempted_items")),
        errors=as_int(usage.get("errors")) + as_int(judge.get("errors")),
        generation_prompt_tokens=prompt_tokens,
        generation_completion_tokens=completion_tokens,
        generation_total_tokens=total_tokens,
        judge_prompt_tokens=as_float(judge.get("prompt_tokens")),
        judge_completion_tokens=as_float(judge.get("completion_tokens")),
        judge_total_tokens=as_float(judge.get("total_tokens")),
        generation_cost_cny=cost_from_tokens(model, prompt_tokens, completion_tokens, prices),
        judge_cost_cny=as_float(judge.get("cost_cny")),
        details={"run_dir": rel(run_dir), "judge_run_name": judge_run_name},
    )
    if judge and judge.get("expected_items"):
        piece.details["expected_judge_items"] = judge.get("expected_items")
    piece.status = piece_status(generation_status, judge_status, piece.errors, score_value)
    return piece


def log_only_piece(log_path: Path, root: Path, priority: int, prices: dict[str, Any]) -> Piece | None:
    parsed = parse_log_exam_section(log_path, "_generation.log")
    if not parsed:
        return None
    exam, section = parsed
    if exam not in {"ipho", "icho", "cpho", "ccho"}:
        return None
    tag = log_path.parent.name
    model = parse_log_model(log_path) or model_from_tag(tag)
    gen_status, _ = log_exit_status(log_path)
    judge_log_name = f"ipho_{section}_judge.log" if exam == "ipho" else f"{exam}_judge.log"
    judge_log = log_path.parent / judge_log_name
    judge_status, _ = log_exit_status(judge_log)
    judge_run_name = parse_judge_run_name(judge_log)
    judge = judge_metrics_from_dir(JUDGE_RUNS / judge_run_name, model, exam) if judge_run_name else {}
    if judge:
        judge_status = str(judge.get("status") or judge_status or "ok")
    score = judge.get("score") if isinstance(judge, dict) else None
    score_value = as_float(score) if score is not None else None
    solution_path = root / tag / "interleaved" / exam / (section if exam == "ipho" else "") / "solutions.json"
    usage = summarize_solution_usage(solution_path, model, prices) if solution_path.exists() else {}
    piece = Piece(
        model=model,
        model_tag=tag,
        exam=exam,
        section=section,
        source_root=rel(root),
        source_priority=priority,
        score=score_value,
        max_score=IPHO_SECTION_MAX.get(section, EXAM_MAX_SCORES.get(exam)) if exam == "ipho" else EXAM_MAX_SCORES.get(exam),
        generation_status=gen_status,
        judge_status=judge_status,
        items=as_int(judge.get("items")),
        attempted_items=as_int(judge.get("attempted_items")),
        errors=as_int(usage.get("errors")) + as_int(judge.get("errors")),
        generation_prompt_tokens=as_float(usage.get("prompt_tokens")),
        generation_completion_tokens=as_float(usage.get("completion_tokens")),
        generation_total_tokens=as_float(usage.get("total_tokens")),
        generation_cost_cny=as_float(usage.get("cost_cny")),
        judge_prompt_tokens=as_float(judge.get("prompt_tokens")),
        judge_completion_tokens=as_float(judge.get("completion_tokens")),
        judge_total_tokens=as_float(judge.get("total_tokens")),
        judge_cost_cny=as_float(judge.get("cost_cny")),
        details={"log": rel(log_path), "judge_run_name": judge_run_name},
    )
    if not solution_path.exists() and gen_status == "ok":
        piece.note = "generation log exited 0 but solutions.json is not present"
    piece.status = piece_status(gen_status, judge_status, piece.errors, score_value)
    return piece


def collect_batch_root(root: Path, priority: int, prices: dict[str, Any], summary_only: bool = False) -> tuple[list[Piece], dict[str, Any]]:
    pieces_by_key: dict[tuple[str, str, str], Piece] = {}
    info: dict[str, Any] = {
        "path": rel(root),
        "priority": priority,
        "exists": root.exists(),
        "summary_results": 0,
        "generation_summaries": 0,
        "log_only": 0,
        "manual_overrides": 0,
        "skipped": [],
    }
    if not root.exists():
        return [], info

    summary = load_json(root / "latest_eval_summary.json", {})
    if isinstance(summary, dict):
        info["skipped"] = summary.get("skipped", []) if isinstance(summary.get("skipped"), list) else []
        for result in summary.get("results", []) if isinstance(summary.get("results"), list) else []:
            if not isinstance(result, dict):
                continue
            piece = piece_from_summary_result(result, root, priority)
            if piece:
                pieces_by_key[(piece.model, piece.exam, piece.section)] = piece
                info["summary_results"] += 1

    if not summary_only:
        for summary_path in sorted(root.glob("*/interleaved/**/summary.json")):
            piece = generation_piece_from_dir(summary_path, root, priority, prices)
            if not piece:
                continue
            key = (piece.model, piece.exam, piece.section)
            old = pieces_by_key.get(key)
            if old is None or (old.status in REPLACEABLE_STATUSES and piece.status not in {"unknown"}):
                pieces_by_key[key] = piece
                info["generation_summaries"] += 1

        for log_path in sorted((root / "_logs").glob("*/*_generation.log")):
            piece = log_only_piece(log_path, root, priority, prices)
            if not piece:
                continue
            key = (piece.model, piece.exam, piece.section)
            old = pieces_by_key.get(key)
            if old is None or (old.status in REPLACEABLE_STATUSES and piece.status not in {"unknown"}):
                pieces_by_key[key] = piece
                info["log_only"] += 1

    overrides = load_json(root / "manual_status_overrides.json", [])
    if isinstance(overrides, list):
        for row in overrides:
            if not isinstance(row, dict):
                continue
            model = str(row.get("model") or "")
            exam = str(row.get("exam") or "")
            if not model or exam not in {"ipho", "icho", "cpho", "ccho"}:
                continue
            section = str(row.get("section") or "").upper()
            piece = Piece(
                model=model,
                model_tag=str(row.get("model_tag") or model_tag(model)),
                exam=exam,
                section=section,
                source_root=rel(root),
                source_priority=priority,
                source_kind="manual_override",
                score=as_float(row.get("score")) if row.get("score") is not None else None,
                max_score=IPHO_SECTION_MAX.get(section, EXAM_MAX_SCORES.get(exam)) if exam == "ipho" else EXAM_MAX_SCORES.get(exam),
                status=str(row.get("status") or "failed"),
                generation_status=str(row.get("generation_status") or row.get("status") or "failed"),
                judge_status=str(row.get("judge_status") or ""),
                errors=as_int(row.get("errors"), 1),
                note=str(row.get("note") or ""),
            )
            pieces_by_key[(piece.model, piece.exam, piece.section)] = piece
            info["manual_overrides"] += 1

    return list(pieces_by_key.values()), info


def aggregate_ibo_rows(rows: list[dict[str, Any]], model: str, root: Path, prices: dict[str, Any]) -> Piece:
    score = sum(as_float(row.get("score")) for row in rows if row.get("score") is not None)
    max_score = sum(as_float(row.get("max_score")) for row in rows if row.get("max_score") is not None) or EXAM_MAX_SCORES["ibo"]
    prompt_tokens = 0.0
    completion_tokens = 0.0
    total_tokens = 0.0
    cost_cny = 0.0
    errors = 0
    missing_tags = 0
    for row in rows:
        if row.get("error"):
            errors += 1
        if row.get("score_status") and row.get("score_status") != "tagged":
            missing_tags += 1
        cost = row.get("usage_cost") or usage_cost(row.get("usage", {}) or {}, model, prices)
        prompt_tokens += as_float(cost.get("prompt_tokens"))
        completion_tokens += as_float(cost.get("completion_tokens"))
        total_tokens += as_float(cost.get("total_tokens"))
        cost_cny += as_float(cost.get("cost_cny"))
    official_max = max(EXAM_MAX_SCORES["ibo"], max_score)
    status = "partial" if len(rows) < 87 or errors or missing_tags else "ok"
    return Piece(
        model=model,
        model_tag=model_tag(model),
        exam="ibo",
        section="",
        source_root=rel(root),
        source_priority=0,
        source_kind="ibo",
        score=score,
        max_score=official_max,
        status=status,
        items=len([row for row in rows if row.get("score") is not None]),
        attempted_items=len(rows),
        errors=errors,
        missing_tags=missing_tags,
        generation_prompt_tokens=prompt_tokens,
        generation_completion_tokens=completion_tokens,
        generation_total_tokens=total_tokens,
        generation_cost_cny=cost_cny,
    )


def ibo_piece_from_summary(summary_path: Path, root: Path, priority: int, prices: dict[str, Any]) -> Piece | None:
    summary = load_json(summary_path, {})
    if not isinstance(summary, dict) or not summary.get("model"):
        return None
    model = str(summary["model"])
    errors = as_int(summary.get("errors")) + as_int(summary.get("missing_tags"))
    incomplete = bool(summary.get("incomplete"))
    status = "incomplete" if incomplete else ("complete_with_errors" if errors else "ok")
    piece = Piece(
        model=model,
        model_tag=str(summary.get("model_tag") or model_tag(model)),
        exam="ibo",
        section="",
        source_root=rel(root),
        source_priority=priority,
        source_kind="ibo",
        score=as_float(summary.get("score")) if summary.get("score") is not None else None,
        max_score=as_float(summary.get("max_score"), EXAM_MAX_SCORES["ibo"]),
        status=status,
        items=as_int(summary.get("items")),
        attempted_items=as_int(summary.get("selected_items") or summary.get("items")),
        errors=as_int(summary.get("errors")),
        missing_tags=as_int(summary.get("missing_tags")),
        generation_prompt_tokens=as_float(summary.get("prompt_tokens")),
        generation_completion_tokens=as_float(summary.get("completion_tokens")),
        generation_total_tokens=as_float(summary.get("total_tokens")),
        generation_cost_cny=as_float(summary.get("cost_cny")),
        note=str(summary.get("note") or ""),
        details={
            "run_dir": summary.get("run_dir"),
            "selected_items": summary.get("selected_items"),
            "generated_items": summary.get("generated_items"),
            "reused_items": summary.get("reused_items"),
            "attempted_max_score": summary.get("attempted_max_score"),
            "missing_items": summary.get("missing_items"),
        },
    )
    return piece


def collect_ibo_root(root: Path, priority: int, prices: dict[str, Any]) -> tuple[list[Piece], dict[str, Any]]:
    pieces: list[Piece] = []
    info: dict[str, Any] = {"path": rel(root), "priority": priority, "exists": root.exists(), "model_summaries": 0, "fallback_rows": 0}
    if not root.exists():
        return pieces, info
    for model_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        summary_path = model_dir / "summary.json"
        piece = ibo_piece_from_summary(summary_path, root, priority, prices) if summary_path.exists() else None
        if piece:
            pieces.append(piece)
            info["model_summaries"] += 1
            continue
        rows = load_json(model_dir / "scores.json", None)
        if not isinstance(rows, list):
            rows = load_json(model_dir / "solutions.json", None)
        if isinstance(rows, list) and rows and isinstance(rows[0], dict) and "score" in rows[0]:
            config = load_json(model_dir / "config.json", {})
            model = str(config.get("models") or rows[0].get("model") or model_from_tag(model_dir.name)) if isinstance(config, dict) else model_from_tag(model_dir.name)
            if "," in model:
                model = model.split(",", 1)[0]
            piece = aggregate_ibo_rows(rows, model, root, prices)
            piece.model_tag = model_dir.name
            piece.source_priority = priority
            piece.details["run_dir"] = rel(model_dir)
            pieces.append(piece)
            info["fallback_rows"] += 1
    return pieces, info


def select_active_batch_pieces(all_pieces: list[Piece]) -> tuple[list[Piece], list[dict[str, Any]]]:
    highest_model_wide: dict[str, int] = {}
    for piece in all_pieces:
        root_name = Path(piece.source_root).name
        if piece.model in MODEL_WIDE_SUPERSEDE.get(root_name, set()):
            highest_model_wide[piece.model] = max(highest_model_wide.get(piece.model, -1), piece.source_priority)

    filtered: list[Piece] = []
    superseded: list[dict[str, Any]] = []
    for piece in all_pieces:
        min_priority = highest_model_wide.get(piece.model)
        if min_priority is not None and piece.source_priority < min_priority:
            superseded.append(
                {
                    "model": piece.model,
                    "exam": piece.exam,
                    "section": piece.section,
                    "source_root": piece.source_root,
                    "reason": "model-wide superseded by cleaner root",
                }
            )
            continue
        filtered.append(piece)

    def selection_rank(piece: Piece) -> tuple[int, int]:
        status_rank = {
            "ok": 6,
            "complete_with_errors": 5,
            "partial": 4,
            "generated": 3,
            "running": 2,
            "failed": 1,
            "unknown": 0,
        }.get(piece.status, 0)
        return (status_rank, piece.source_priority)

    best_by_exam_section: dict[tuple[str, str, str], tuple[int, int]] = {}
    for piece in filtered:
        key = (piece.model, piece.exam, piece.section)
        best_by_exam_section[key] = max(best_by_exam_section.get(key, (-1, -1)), selection_rank(piece))

    active: list[Piece] = []
    for piece in filtered:
        if selection_rank(piece) == best_by_exam_section[(piece.model, piece.exam, piece.section)]:
            active.append(piece)
        else:
            superseded.append(
                {
                    "model": piece.model,
                    "exam": piece.exam,
                    "section": piece.section,
                    "source_root": piece.source_root,
                    "reason": "older duplicate for model/exam/section",
                }
            )
    return active, superseded


def select_active_ibo_pieces(pieces: list[Piece]) -> tuple[list[Piece], list[dict[str, Any]]]:
    best: dict[str, Piece] = {}
    superseded: list[dict[str, Any]] = []
    for piece in sorted(pieces, key=lambda item: item.source_priority):
        old = best.get(piece.model)
        if old is not None:
            superseded.append(
                {
                    "model": old.model,
                    "exam": "ibo",
                    "source_root": old.source_root,
                    "reason": "older duplicate IBO root",
                }
            )
        best[piece.model] = piece
    return list(best.values()), superseded


def worst_status(statuses: list[str], has_score: bool, missing: list[str]) -> str:
    if not statuses:
        return "not_run"
    if any(status == "running" for status in statuses):
        return "running"
    if missing and has_score:
        return "partial"
    if missing:
        return "incomplete"
    if any(status == "failed" for status in statuses):
        return "failed" if not has_score else "partial"
    if any(status == "partial" for status in statuses):
        return "partial"
    if any(status == "incomplete" for status in statuses):
        return "incomplete"
    if any(status == "complete_with_errors" for status in statuses):
        return "complete_with_errors"
    if all(status == "ok" for status in statuses):
        return "ok"
    if has_score:
        return "partial"
    return statuses[0]


def aggregate_exam(pieces: list[Piece], exam: str) -> dict[str, Any]:
    if not pieces:
        return {
            "exam": exam,
            "label": EXAM_LABELS[exam],
            "score": None,
            "max_score": EXAM_MAX_SCORES[exam],
            "status": "not_run",
            "sections": {},
            "source_roots": [],
            "notes": [],
            "items": 0,
            "attempted_items": 0,
            "errors": 0,
            "missing_tags": 0,
            "tokens": {"prompt": 0.0, "completion": 0.0, "total": 0.0},
            "cost_cny": 0.0,
        }

    source_roots = sorted({piece.source_root for piece in pieces})
    score = sum(piece.score for piece in pieces if piece.score is not None)
    has_score = any(piece.score is not None for piece in pieces)
    expected_sections = ["S1", "S2", "S3"] if exam == "ipho" else [""]
    present_sections = {piece.section for piece in pieces if piece.score is not None or piece.status not in {"unknown"}}
    missing = [section for section in expected_sections if section not in present_sections]
    statuses = [piece.status for piece in pieces]
    notes = [piece.note for piece in pieces if piece.note]
    if missing:
        label = ", ".join(section or "all" for section in missing)
        notes.append(f"missing expected section(s): {label}")

    sections: dict[str, Any] = {}
    for piece in sorted(pieces, key=lambda item: item.section or "all"):
        key = piece.section or "all"
        sections[key] = {
            "score": piece.score,
            "max_score": piece.max_score,
            "status": piece.status,
            "generation_status": piece.generation_status,
            "judge_status": piece.judge_status,
            "items": piece.items,
            "attempted_items": piece.attempted_items,
            "errors": piece.errors,
            "tokens": {
                "generation": piece.generation_total_tokens,
                "judge": piece.judge_total_tokens,
                "total": piece.total_tokens,
            },
            "cost_cny": round(piece.total_cost_cny, 8),
            "source_root": piece.source_root,
            "details": {key: value for key, value in piece.details.items() if value not in (None, "", [])},
        }

    prompt_tokens = sum(piece.generation_prompt_tokens + piece.judge_prompt_tokens for piece in pieces)
    completion_tokens = sum(piece.generation_completion_tokens + piece.judge_completion_tokens for piece in pieces)
    total_tokens = sum(piece.total_tokens for piece in pieces)
    cost_cny = sum(piece.total_cost_cny for piece in pieces)
    max_score = EXAM_MAX_SCORES[exam]
    if exam == "ibo" and pieces[0].max_score:
        max_score = pieces[0].max_score
    return {
        "exam": exam,
        "label": EXAM_LABELS[exam],
        "score": score if has_score else None,
        "max_score": max_score,
        "score_fraction": (score / max_score if has_score and max_score else None),
        "status": worst_status(statuses, has_score, missing),
        "sections": sections,
        "source_roots": source_roots,
        "notes": notes,
        "items": sum(piece.items for piece in pieces),
        "attempted_items": sum(piece.attempted_items for piece in pieces),
        "errors": sum(piece.errors for piece in pieces),
        "missing_tags": sum(piece.missing_tags for piece in pieces),
        "tokens": {
            "prompt": prompt_tokens,
            "completion": completion_tokens,
            "total": total_tokens,
        },
        "cost_cny": cost_cny,
    }


def build_report(batch_roots: list[Path], ibo_roots: list[Path], dry_run: bool = False, summary_only: bool = False) -> dict[str, Any]:
    prices = load_json(PRICES_PATH, {})
    batch_pieces: list[Piece] = []
    ibo_pieces: list[Piece] = []
    source_info: dict[str, list[dict[str, Any]]] = {"batch_roots": [], "ibo_roots": []}

    for priority, root in enumerate(batch_roots, start=1):
        pieces, info = collect_batch_root(root, priority, prices, summary_only=summary_only)
        batch_pieces.extend(pieces)
        source_info["batch_roots"].append(info)

    for priority, root in enumerate(ibo_roots, start=1):
        pieces, info = collect_ibo_root(root, priority, prices)
        ibo_pieces.extend(pieces)
        source_info["ibo_roots"].append(info)

    active_batch, superseded_batch = select_active_batch_pieces(batch_pieces)
    active_ibo, superseded_ibo = select_active_ibo_pieces(ibo_pieces)
    all_active = active_batch + active_ibo
    models = sorted({piece.model for piece in all_active})

    rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for model in models:
        model_pieces = [piece for piece in all_active if piece.model == model]
        row = {
            "model": model,
            "model_tag": model_tag(model),
            "exams": {},
            "total_tokens": 0.0,
            "total_cost_cny": 0.0,
            "overall_status": "not_run",
        }
        statuses: list[str] = []
        for exam in EXAM_ORDER:
            exam_pieces = [piece for piece in model_pieces if piece.exam == exam]
            exam_row = aggregate_exam(exam_pieces, exam)
            row["exams"][exam] = exam_row
            row["total_tokens"] += as_float(exam_row["tokens"]["total"])
            row["total_cost_cny"] += as_float(exam_row["cost_cny"])
            statuses.append(str(exam_row["status"]))
            flat_rows.append(
                {
                    "model": model,
                    "exam": exam,
                    "score": exam_row["score"],
                    "max_score": exam_row["max_score"],
                    "status": exam_row["status"],
                    "tokens": exam_row["tokens"]["total"],
                    "cost_cny": exam_row["cost_cny"],
                    "items": exam_row["items"],
                    "attempted_items": exam_row["attempted_items"],
                    "errors": exam_row["errors"],
                    "source_roots": exam_row["source_roots"],
                    "notes": exam_row["notes"],
                }
            )
        row["overall_status"] = worst_status([status for status in statuses if status != "not_run"], True, [])
        rows.append(row)

    return {
        "created_at": datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        "project_root": str(ROOT),
        "dry_run": dry_run,
        "summary_only": summary_only,
        "max_scores": {EXAM_LABELS[key]: value for key, value in EXAM_MAX_SCORES.items()},
        "sources": source_info,
        "superseded": superseded_batch + superseded_ibo,
        "rows": rows,
        "flat_rows": flat_rows,
    }


def format_score_cell(exam: dict[str, Any]) -> str:
    score = exam.get("score")
    max_score = exam.get("max_score")
    status = exam.get("status")
    if score is None:
        return f"- ({status})"
    return f"{score:.3g}/{max_score:g} ({status})"


def format_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Latest Model Eval Summary ({report['created_at']})",
        "",
        "| model | IPhO | IChO | CPhO | CChO | IBO | total tokens | total cost CNY |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["rows"]:
        exams = row["exams"]
        lines.append(
            "| {model} | {ipho} | {icho} | {cpho} | {ccho} | {ibo} | {tokens:.0f} | {cost:.4f} |".format(
                model=row["model"],
                ipho=format_score_cell(exams["ipho"]),
                icho=format_score_cell(exams["icho"]),
                cpho=format_score_cell(exams["cpho"]),
                ccho=format_score_cell(exams["ccho"]),
                ibo=format_score_cell(exams["ibo"]),
                tokens=as_float(row["total_tokens"]),
                cost=as_float(row["total_cost_cny"]),
            )
        )
    notes: list[str] = []
    for row in report["rows"]:
        for exam in EXAM_ORDER:
            exam_row = row["exams"][exam]
            for note in exam_row.get("notes", []):
                notes.append(f"- {row['model']} {EXAM_LABELS[exam]}: {note}")
    if notes:
        lines.extend(["", "## Notes", *notes])
    return "\n".join(lines) + "\n"


def format_tsv(report: dict[str, Any]) -> str:
    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["model", "exam", "score", "max_score", "status", "tokens", "cost_cny", "items", "attempted_items", "errors", "source_roots", "notes"],
        delimiter="\t",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in report["flat_rows"]:
        out = dict(row)
        out["source_roots"] = ",".join(row.get("source_roots") or [])
        out["notes"] = " | ".join(row.get("notes") or [])
        writer.writerow(out)
    return output.getvalue()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", action="append", type=Path, default=None, help="Batch artifact root to read. Repeat to override defaults.")
    parser.add_argument("--ibo-root", action="append", type=Path, default=None, help="IBO artifact root to read. Repeat to override defaults.")
    parser.add_argument("--format", choices=["markdown", "json", "tsv"], default="markdown", help="Output format.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path. Without this, prints to stdout only.")
    parser.add_argument("--dry-run", action="store_true", help="Read artifacts and print report, but do not write --output.")
    parser.add_argument("--summary-only", action="store_true", help="Read batch latest_eval_summary.json plus manual overrides only; skip fallback scans of run directories and logs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    batch_roots = [path if path.is_absolute() else ROOT / path for path in (args.batch_root or DEFAULT_BATCH_ROOTS)]
    ibo_roots = [path if path.is_absolute() else ROOT / path for path in (args.ibo_root or DEFAULT_IBO_ROOTS)]
    report = build_report(batch_roots, ibo_roots, dry_run=args.dry_run, summary_only=args.summary_only)
    if args.format == "json":
        text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    elif args.format == "tsv":
        text = format_tsv(report)
    else:
        text = format_markdown(report)

    if args.output and not args.dry_run:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
