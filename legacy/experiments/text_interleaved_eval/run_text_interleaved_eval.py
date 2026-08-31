#!/usr/bin/env python3
"""Run model generation and rubric judging on the staticized text-only dataset."""

from __future__ import annotations

import argparse
import json
import os
import re
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

from api_client_interleaved import UnifiedChatClient  # noqa: E402


DEFAULT_ENV_FILE = Path("/home/jovyan/zhaoguangxiang-data-zzzc/zhaoguangxiang/.config/science_arena/360.env")
DEFAULT_PRICES = ROOT / "experiments/judge_pipeline_v2/model_prices_360.json"
DEFAULT_DATASET = ROOT / "datasets/science_arena_text_interleaved_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/runs/text_interleaved_eval"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


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


def row_key(row: dict[str, Any]) -> str:
    return f"{row.get('benchmark_id')}::{row.get('sequence_id')}::{row.get('turn_id')}"


def is_bad_finish_reason(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return False
    good = {"stop", "completed", "complete", "objective_key"}
    if text in good:
        return False
    bad_fragments = (
        "content_filter",
        "finish_content_filter",
        "length",
        "max_token",
        "max_output",
        "truncat",
    )
    return any(fragment in text for fragment in bad_fragments)


def safe_model_name(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [jsonable(item) for item in value]
    return value


def extract_json_object(text: str) -> dict[str, Any]:
    def parse_json_maybe_sanitize(candidate: str) -> Any:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Judge evidence often contains LaTeX such as \dot or \theta.
            # Escape only singleton backslashes; do not turn valid JSON "\\dot"
            # into an invalid odd-length backslash run.
            sanitized = re.sub(r"(?<!\\)\\(?![\"\\/bfnrtu])", r"\\\\", candidate)
            return json.loads(sanitized)

    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = parse_json_maybe_sanitize(cleaned)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError:
        pass
    fenced = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S | re.I)
    for candidate in reversed(fenced):
        try:
            parsed = parse_json_maybe_sanitize(candidate)
            return parsed if isinstance(parsed, dict) else {"raw": parsed}
        except json.JSONDecodeError:
            continue
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return {"parse_error": "no_json_object", "raw_text": text[:4000]}
    try:
        parsed = parse_json_maybe_sanitize(match.group(0))
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError as exc:
        return {"parse_error": f"json_decode_error: {exc}", "raw_text": text[:4000]}


def build_solver_prompt(row: dict[str, Any], solver_prompt_variant: str = "standard") -> str:
    question_text = str(row.get("question_text_staticized") or "").strip()
    if not question_text:
        raise ValueError(f"Missing question_text_staticized for {row_key(row)}")
    if solver_prompt_variant not in {"standard", "answer_first", "final_only"}:
        raise ValueError(f"Unknown solver_prompt_variant: {solver_prompt_variant}")
    objective_instruction = ""
    if row.get("grading_type") == "objective_key":
        objective_instruction = (
            "\n\nFINAL ANSWER CONTRACT FOR AUTOMATED SCORING:\n"
            "- You MUST include exactly one final answer span using this format: <boa>A B C D<eoa>\n"
            "- The scorer reads only the tokens inside <boa> and <eoa>. If the span is missing, the item may score zero.\n"
            "- Inside the span, write only the final entries in order, separated by spaces.\n"
            "- For true/false entries, use T or F. For multiple choice, use option letters. For numbers, use plain numeric values without units.\n"
            "- Put any explanation outside the tags, and end the response with the tagged final answer span."
        )
    if solver_prompt_variant == "answer_first":
        if row.get("grading_type") == "objective_key":
            output_control = (
                "\n\nOUTPUT CONTROL FOR API STABILITY:\n"
                "- Start the response with exactly one <boa>...</eoa> final answer span on the first non-empty line.\n"
                "- Then provide only a short justification if useful.\n"
                "- Do not include <think> blocks, hidden/private chain-of-thought, or long exploratory reasoning.\n"
                "- If uncertain, still give the best final answer in the required tag format."
            )
        else:
            output_control = (
                "\n\nOUTPUT CONTROL FOR API STABILITY:\n"
                "- Start the response with a line beginning `FINAL ANSWER:` for the current turn.\n"
                "- Then give only the essential equations, numerical steps, or facts needed to justify that answer.\n"
                "- Do not include <think> blocks, hidden/private chain-of-thought, or long exploratory reasoning.\n"
                "- If the problem is difficult, still provide your best final answer; do not stop with analysis only."
            )
        return (
            "You are solving a released public science benchmark item for research evaluation. "
            "This is not a live exam, contest, school assignment, or proctored assessment. "
            "Use only the text in the current prompt and prior conversation. "
            "All original figures have already been converted to text descriptions. "
            "Give the final answer first, then a concise supporting solution for the current turn.\n\n"
            f"Benchmark: {row.get('benchmark_id')} | Turn: {row.get('turn_id')} | Max score: {row.get('max_score')}\n"
            f"{objective_instruction}{output_control}\n\n{question_text}"
        )
    if solver_prompt_variant == "final_only":
        if row.get("grading_type") == "objective_key":
            output_control = (
                "Return exactly one line and nothing else:\n"
                "<boa>YOUR FINAL TOKENS IN ORDER</eoa>\n"
                "Do not include <think>, reasoning, analysis, explanation, markdown, or any text outside the tags."
            )
        else:
            output_control = (
                "Return only the final answer for the current turn. "
                "Start with `FINAL ANSWER:`. Do not include <think>, hidden/private chain-of-thought, "
                "or exploratory reasoning. Include only essential final expressions or values."
            )
        return (
            "You are answering a released public science benchmark item for research evaluation. "
            "This is not a live exam, contest, school assignment, or proctored assessment. "
            "All original figures have already been converted to text descriptions. "
            f"{objective_instruction}\n\n"
            f"Benchmark: {row.get('benchmark_id')} | Turn: {row.get('turn_id')} | Max score: {row.get('max_score')}\n\n"
            f"{output_control}\n\n{question_text}"
        )
    return (
        "You are solving a released public science benchmark item for research evaluation. "
        "This is not a live exam, contest, school assignment, or proctored assessment. "
        "Use only the text in the current prompt and prior conversation. "
        "All original figures have already been converted to text descriptions. "
        "Show reasoning and give a clear final answer for the current turn.\n\n"
        f"Benchmark: {row.get('benchmark_id')} | Turn: {row.get('turn_id')} | Max score: {row.get('max_score')}\n\n"
        f"{objective_instruction}\n\n{question_text}{objective_instruction}"
    )


def build_judge_prompt(row: dict[str, Any], candidate_answer: str) -> str:
    question_text = str(row.get("question_text_staticized") or "").strip()
    official_answer = str(row.get("official_answer_staticized") or "").strip()
    detailed_solution = str(row.get("official_detailed_solution_staticized") or "").strip()
    rubric = str(row.get("official_rubric_staticized") or "").strip()
    missing = [
        name
        for name, value in [
            ("question_text_staticized", question_text),
            ("official_answer_staticized", official_answer),
            ("official_detailed_solution_staticized", detailed_solution),
            ("official_rubric_staticized", rubric),
        ]
        if not value
    ]
    if missing:
        raise ValueError(f"Missing judge fields for {row_key(row)}: {missing}")
    payload = {
        "benchmark_id": row.get("benchmark_id"),
        "sequence_id": row.get("sequence_id"),
        "turn_id": row.get("turn_id"),
        "turn_index": row.get("turn_index"),
        "max_score": row.get("max_score"),
        "question_text": question_text,
        "official_answer": official_answer,
        "official_detailed_solution": detailed_solution,
        "official_rubric": rubric,
        "candidate_answer": candidate_answer,
    }
    return (
        "You are a strict olympiad examiner. Grade the candidate answer against the official answer, "
        "detailed solution, and rubric. Use the raw max_score for this turn. Award partial credit; "
        "do not give credit for unsupported or contradictory work. If the official rubric includes point "
        "allocations, follow them exactly. Return only valid JSON with keys: total_score, max_score, "
        "sub_scores, evidence, deductions, confidence.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def score_objective(row: dict[str, Any], candidate_answer: str) -> dict[str, Any] | None:
    if row.get("grading_type") != "objective_key":
        return None
    answer_text = str(row.get("official_answer_staticized") or row.get("official_answer") or "")
    has_tag = bool(re.search(r"<boa>", str(candidate_answer or ""), flags=re.I))
    predicted_text = extract_boa_content(candidate_answer)
    key_text = extract_boa_content(answer_text)
    key_text = re.split(r"\bSub-?scores?\s*[:：]", key_text, maxsplit=1, flags=re.I)[0]
    key_text = re.sub(r"^(?:answer|key)\s*[:：]\s*", "", key_text.strip(), flags=re.I)
    key_tokens = normalize_objective_tokens(key_text)
    if not key_tokens:
        return None
    if not has_tag:
        predicted_text = fallback_objective_answer(candidate_answer, len(key_tokens))
    pred_tokens = normalize_objective_tokens(predicted_text)
    sub_scores = parse_sub_scores(answer_text, len(key_tokens), float(row.get("max_score") or 0))
    earned = 0.0
    comparisons = []
    for index, key_token in enumerate(key_tokens):
        predicted_token = pred_tokens[index] if index < len(pred_tokens) else ""
        item_score = sub_scores[index] if objective_token_equal(predicted_token, key_token) else 0.0
        earned += item_score
        comparisons.append(
            {
                "index": index + 1,
                "predicted": predicted_token,
                "key": key_token,
                "max_score": sub_scores[index],
                "score": item_score,
            }
        )
    return {
        "total_score": min(earned, float(row.get("max_score") or earned)),
        "max_score": float(row.get("max_score") or 0),
        "sub_scores": comparisons,
        "evidence": "Deterministic objective-key comparison using official key and sub-scores.",
        "deductions": {},
        "confidence": "high",
        "extraction_status": "tagged" if has_tag else "tagless_fallback",
    }


def extract_boa_content(text: str) -> str:
    match = re.search(r"<boa>(.*?)(?:</?eoa>|</boa>)", str(text or ""), flags=re.S | re.I)
    return match.group(1).strip() if match else str(text or "").strip()


def fallback_objective_answer(text: str, expected_count: int) -> str:
    raw = str(text or "")
    marker_matches = list(re.finditer(r"(?i)(final\s+answer|answers?|answer)\s*(?:is|are)?\s*[:：]", raw))
    if marker_matches:
        tail = raw[marker_matches[-1].end() :]
        tokens = normalize_objective_tokens(tail)
        if tokens:
            return " ".join(tokens[:expected_count])
    explicit_answers = re.findall(
        r"(?i)(?:\*\*)?\s*answer\s*(?:\*\*)?\s*[:：]\s*(?:\*\*)?\s*([A-Z]|T|F|[-+]?\d+(?:\.\d+)?)",
        raw,
    )
    if len(explicit_answers) >= expected_count:
        return " ".join(explicit_answers[:expected_count])
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if lines:
        last_tokens = normalize_objective_tokens(lines[-1])
        if len(last_tokens) >= expected_count:
            return " ".join(last_tokens[:expected_count])
    return raw


def normalize_objective_tokens(text: str) -> list[str]:
    text = str(text or "")
    text = re.sub(r"</?(?:boa|eoa)>", " ", text, flags=re.I)
    text = text.replace(",", " ")
    text = re.sub(r"[\[\]();]", " ", text)
    return [token.strip().upper() for token in text.split() if token.strip()]


def parse_sub_scores(answer_text: str, expected_count: int, max_score: float) -> list[float]:
    match = re.search(r"Sub-?scores?\s*[:：]\s*(\[[^\]]+\])", answer_text, flags=re.I)
    if match:
        try:
            values = json.loads(match.group(1))
            scores = [float(value) for value in values]
            if len(scores) == expected_count:
                return scores
        except Exception:
            pass
    if expected_count <= 0:
        return []
    equal = max_score / expected_count if max_score else 0.0
    return [equal for _ in range(expected_count)]


def objective_token_equal(predicted: str, key: str) -> bool:
    if predicted == key:
        return True
    try:
        return abs(float(predicted) - float(key)) <= 1e-6
    except Exception:
        return False


def run_sequence(
    *,
    model: str,
    request_model: str,
    rows: list[dict[str, Any]],
    client: UnifiedChatClient,
    judge_model: str,
    max_tokens: int,
    judge_max_tokens: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    solver_prompt_variant: str,
    timeout_label: str,
    thinking_mode: str,
    thinking_effort: str,
    judge_thinking_mode: str,
    judge_thinking_effort: str,
    prices: dict[str, Any],
    partial_dir: Path | None = None,
) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    evaluation_protocol = (
        "text_only_interleaved_staticized_v1"
        if solver_prompt_variant == "standard"
        else f"text_only_interleaved_staticized_v1_{solver_prompt_variant}_diagnostic"
    )
    for row in sorted(rows, key=lambda item: int(item.get("turn_index") or 0)):
        prompt = build_solver_prompt(row, solver_prompt_variant=solver_prompt_variant)
        user_message = {"role": "user", "content": prompt}
        call_messages = messages + [user_message]
        started = time.time()
        generation_response = client.chat(
            model=request_model,
            messages=call_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            thinking_mode=thinking_mode,
            thinking_effort=thinking_effort,
        )
        elapsed = time.time() - started
        answer = generation_response.get("content") or generation_response.get("llm_output") or ""
        generation_usage = usage_cost(generation_response.get("usage", {}) or {}, request_model, prices)
        if not generation_usage["cost_cny"] and request_model != model:
            generation_usage = usage_cost(generation_response.get("usage", {}) or {}, model, prices)
        generation_finish_reason = generation_response.get("finish_reason")
        generation_error = str(generation_response.get("error") or "")
        if is_bad_finish_reason(generation_finish_reason) and not generation_error:
            generation_error = f"bad_finish_reason:{generation_finish_reason}"
        if answer.strip() and not generation_error:
            messages.append(user_message)
            messages.append({"role": "assistant", "content": answer})
        else:
            messages.append(user_message)
            messages.append(
                {
                    "role": "assistant",
                    "content": "[GENERATION_FAILED: no answer was produced for this turn.]",
                }
            )
        if generation_error or not answer.strip():
            judge_response = {"finish_reason": "generation_failed", "usage": {}, "client_meta": {}}
            judge_result = {
                "parse_error": "generation_failed",
                "generation_error": generation_error or "empty_generation",
                "total_score": 0.0,
                "max_score": float(row.get("max_score") or 0),
            }
            judge_usage = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0, "cost_cny": 0.0}
        else:
            objective = score_objective(row, answer)
            if objective is not None:
                judge_response = {"finish_reason": "objective_key", "usage": {}, "client_meta": {}}
                judge_result = objective
                judge_usage = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0, "cost_cny": 0.0}
            else:
                judge_prompt = build_judge_prompt(row, answer)
                judge_response = client.chat(
                    model=judge_model,
                    messages=[{"role": "user", "content": judge_prompt}],
                    max_tokens=judge_max_tokens,
                    thinking_mode=judge_thinking_mode,
                    thinking_effort=judge_thinking_effort,
                )
                judge_text = judge_response.get("content") or judge_response.get("llm_output") or ""
                judge_result = extract_json_object(judge_text)
                judge_usage = usage_cost(judge_response.get("usage", {}) or {}, judge_model, prices)
            if is_bad_finish_reason(judge_response.get("finish_reason")) and not judge_response.get("error"):
                judge_response["error"] = f"bad_finish_reason:{judge_response.get('finish_reason')}"
                judge_result = {
                    "parse_error": "judge_bad_finish_reason",
                    "judge_error": judge_response["error"],
                    "total_score": 0.0,
                    "max_score": float(row.get("max_score") or 0),
                }
        outputs.append(
            {
                "benchmark_id": row.get("benchmark_id"),
                "sequence_id": row.get("sequence_id"),
                "turn_id": row.get("turn_id"),
                "turn_index": row.get("turn_index"),
                "model": model,
                "request_model": request_model,
                "judge_model": judge_model,
                "evaluation_protocol": evaluation_protocol,
                "solver_prompt_variant": solver_prompt_variant,
                "generation_finish_reason": generation_finish_reason,
                "generation_error": generation_error,
                "generation_elapsed_seconds": elapsed,
                "generation_usage": generation_usage,
                "generation_client_meta": generation_response.get("client_meta", {}),
                "generation_sampling": {
                    "temperature": temperature,
                    "top_p": top_p,
                    "top_k": top_k,
                },
                "candidate_answer": answer,
                "judge_finish_reason": judge_response.get("finish_reason"),
                "judge_error": judge_response.get("error", ""),
                "judge_usage": judge_usage,
                "judge_result": judge_result,
                "score": judge_result.get("total_score"),
                "max_score": judge_result.get("max_score") or row.get("max_score"),
                "total_cost_cny": generation_usage["cost_cny"] + judge_usage["cost_cny"],
                "timeout_label": timeout_label,
            }
        )
        if partial_dir is not None:
            sequence_id = str(row.get("sequence_id"))
            write_jsonl(partial_dir / f"{sequence_id}.jsonl", outputs)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument(
        "--request_model",
        default="",
        help="Optional API model name to request while recording results under --model. Use only with a single --model.",
    )
    parser.add_argument("--model_config", type=Path, default=None)
    parser.add_argument("--judge_model", default="google/gemini-3.5-flash")
    parser.add_argument("--benchmark_id", action="append", default=[])
    parser.add_argument(
        "--sequence_id",
        action="append",
        default=[],
        help="Optional sequence ids to run. Useful for targeted repair of dirty or partial interleaved sequences.",
    )
    parser.add_argument("--limit_sequences", type=int, default=0)
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--judge_max_tokens", type=int, default=65536)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument(
        "--solver_prompt_variant",
        choices=["standard", "answer_first", "final_only"],
        default="standard",
        help="Prompt style for model generation. Non-standard variants are diagnostic and should not be merged silently.",
    )
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_delay", type=float, default=5.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--thinking_mode", default="auto")
    parser.add_argument("--thinking_effort", default="auto")
    parser.add_argument("--judge_thinking_mode", default="on")
    parser.add_argument("--judge_thinking_effort", default="high")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force_rerun_selected_sequences",
        action="store_true",
        help="With --sequence_id and --resume, drop selected existing sequences even if they look clean.",
    )
    args = parser.parse_args()

    os.environ.update(load_env_file(args.env_file))
    if args.model_config:
        model_config = load_json(args.model_config, {})
        args.model = list(dict.fromkeys(list(args.model or []) + list(model_config.get("models", []))))
    if not args.model:
        raise SystemExit("At least one --model or a --model_config with models is required.")
    if args.request_model and len(args.model) != 1:
        raise SystemExit("--request_model can only be used with exactly one logical --model.")
    prices = load_json(DEFAULT_PRICES, {})
    rows = iter_jsonl(args.dataset_dir / "benchmark_items.jsonl")
    if args.benchmark_id:
        allowed = set(args.benchmark_id)
        rows = [row for row in rows if row.get("benchmark_id") in allowed]
    output_rows = list(rows)
    output_rows_by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in output_rows:
        output_rows_by_sequence.setdefault(str(row.get("sequence_id")), []).append(row)
    output_sequences = sorted(output_rows_by_sequence)
    rows_by_sequence: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_sequence.setdefault(str(row.get("sequence_id")), []).append(row)
    sequences = sorted(rows_by_sequence)
    if args.sequence_id:
        requested_sequences = set(args.sequence_id)
        unknown_sequences = sorted(requested_sequences - set(rows_by_sequence))
        if unknown_sequences:
            raise SystemExit(f"Unknown sequence_id(s) after benchmark filtering: {unknown_sequences}")
        sequences = [seq for seq in sequences if seq in requested_sequences]
    if args.limit_sequences:
        sequences = sequences[: args.limit_sequences]

    run_name = args.run_name or f"text_interleaved_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "run_config.json", jsonable(vars(args) | {"models": args.model, "sequence_count": len(sequences)}))

    for model in args.model:
        request_model = args.request_model or model
        model_dir = run_dir / safe_model_name(model)
        output_path = model_dir / "turn_results.jsonl"
        existing: dict[str, dict[str, Any]] = {}
        bad_existing_sequences: set[str] = set()
        if args.resume and output_path.exists():
            selected_sequence_set = set(args.sequence_id)
            for row in iter_jsonl(output_path):
                existing[row_key(row)] = row
                sequence_id = str(row.get("sequence_id"))
                if (
                    row.get("generation_error")
                    or row.get("judge_error")
                    or (row.get("judge_result") or {}).get("parse_error")
                    or is_bad_finish_reason(row.get("generation_finish_reason"))
                    or is_bad_finish_reason(row.get("judge_finish_reason"))
                ) and (not args.sequence_id or sequence_id in set(args.sequence_id)):
                    bad_existing_sequences.add(sequence_id)
                if args.force_rerun_selected_sequences and selected_sequence_set and sequence_id in selected_sequence_set:
                    bad_existing_sequences.add(sequence_id)
            if bad_existing_sequences:
                existing = {
                    key: value
                    for key, value in existing.items()
                    if str(value.get("sequence_id")) not in bad_existing_sequences
                }
        pending_sequences = [
            seq
            for seq in sequences
            if any(row_key(row) not in existing for row in rows_by_sequence[seq])
        ]
        pending_set = set(pending_sequences)
        client = UnifiedChatClient(
            api_key=api_key_from_env(),
            backend="requests",
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            timeout=args.timeout,
        )
        # Resume is atomic at sequence level: a partially completed sequence is
        # rerun from turn 1 so the interleaved history is reproducible.
        all_results = {
            key: value
            for key, value in existing.items()
            if str(value.get("sequence_id")) not in pending_set
        }
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
            futures = {
                pool.submit(
                    run_sequence,
                    model=model,
                    request_model=request_model,
                    rows=rows_by_sequence[seq],
                    client=client,
                    judge_model=args.judge_model,
                    max_tokens=args.max_tokens,
                    judge_max_tokens=args.judge_max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    solver_prompt_variant=args.solver_prompt_variant,
                    timeout_label=str(args.timeout),
                    thinking_mode=args.thinking_mode,
                    thinking_effort=args.thinking_effort,
                    judge_thinking_mode=args.judge_thinking_mode,
                    judge_thinking_effort=args.judge_thinking_effort,
                    prices=prices,
                    partial_dir=model_dir / "partial_sequences",
                ): seq
                for seq in pending_sequences
            }
            for index, future in enumerate(as_completed(futures), start=1):
                seq = futures[future]
                try:
                    sequence_results = future.result()
                    for result in sequence_results:
                        all_results[row_key(result)] = result
                except Exception as exc:  # noqa: BLE001
                    for row in rows_by_sequence[seq]:
                        all_results[row_key(row)] = {
                            "benchmark_id": row.get("benchmark_id"),
                            "sequence_id": row.get("sequence_id"),
                            "turn_id": row.get("turn_id"),
                            "turn_index": row.get("turn_index"),
                            "model": model,
                            "request_model": request_model,
                            "judge_model": args.judge_model,
                            "evaluation_protocol": "text_only_interleaved_staticized_v1",
                            "solver_prompt_variant": args.solver_prompt_variant,
                            "generation_finish_reason": "sequence_exception",
                            "generation_error": f"{type(exc).__name__}: {exc}",
                            "generation_usage": {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0, "cost_cny": 0.0},
                            "judge_usage": {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0, "cost_cny": 0.0},
                            "candidate_answer": "",
                            "judge_finish_reason": "sequence_exception",
                            "judge_error": f"{type(exc).__name__}: {exc}",
                            "judge_result": {
                                "parse_error": "sequence_exception",
                                "total_score": 0.0,
                                "max_score": float(row.get("max_score") or 0),
                            },
                            "score": 0.0,
                            "max_score": float(row.get("max_score") or 0),
                            "total_cost_cny": 0.0,
                        }
                ordered = [
                    all_results[row_key(row)]
                    for seq2 in output_sequences
                    for row in output_rows_by_sequence[seq2]
                    if row_key(row) in all_results
                ]
                write_jsonl(output_path, ordered)
                print(
                    json.dumps(
                        {
                            "model": model,
                            "completed_sequences": index,
                            "pending_sequences": len(pending_sequences),
                            "sequence_id": seq,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        ordered = [
            all_results[row_key(row)]
            for seq in output_sequences
            for row in output_rows_by_sequence[seq]
            if row_key(row) in all_results
        ]
        write_jsonl(output_path, ordered)
        summary_by_benchmark: dict[str, dict[str, Any]] = {}
        report_scales = {
            bench: float(next((row.get("report_scale") for row in rows if row.get("benchmark_id") == bench), 0) or 0)
            for bench in {row.get("benchmark_id") for row in rows}
        }
        for result in ordered:
            bench = str(result.get("benchmark_id"))
            summary = summary_by_benchmark.setdefault(
                bench,
                {
                    "benchmark_id": bench,
                    "model": model,
                    "score": 0.0,
                    "max_score": 0.0,
                    "reported_score": 0.0,
                    "report_scale": report_scales.get(bench) or 0.0,
                    "turn_count": 0,
                    "generation_error_count": 0,
                    "judge_error_count": 0,
                    "bad_finish_reason_count": 0,
                    "cost_cny": 0.0,
                },
            )
            summary["score"] += float(result.get("score") or 0)
            summary["max_score"] += float(result.get("max_score") or 0)
            summary["turn_count"] += 1
            summary["cost_cny"] += float(result.get("total_cost_cny") or 0)
            if result.get("generation_error"):
                summary["generation_error_count"] += 1
            if result.get("judge_error") or (result.get("judge_result") or {}).get("parse_error"):
                summary["judge_error_count"] += 1
            if is_bad_finish_reason(result.get("generation_finish_reason")) or is_bad_finish_reason(result.get("judge_finish_reason")):
                summary["bad_finish_reason_count"] += 1
        for summary in summary_by_benchmark.values():
            if summary["max_score"] and summary["report_scale"]:
                summary["reported_score"] = summary["score"] / summary["max_score"] * summary["report_scale"]
        save_json(model_dir / "benchmark_summary.json", sorted(summary_by_benchmark.values(), key=lambda row: row["benchmark_id"]))
    print(json.dumps({"run_dir": str(run_dir), "models": args.model, "sequence_count": len(sequences)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
