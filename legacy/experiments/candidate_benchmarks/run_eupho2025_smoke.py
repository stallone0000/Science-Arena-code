#!/usr/bin/env python3
"""Smoke-test EuPhO 2025 candidate benchmark generation and rubric judging."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
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

from api_client_interleaved import UnifiedChatClient, build_user_message  # noqa: E402


DEFAULT_ENV_FILE = Path("/home/jovyan/zhaoguangxiang-data-zzzc/zhaoguangxiang/.config/science_arena/360.env")
DEFAULT_DRAFT = ROOT / "data/science_arena/candidate_benchmarks/structured/eupho2025/eupho2025_theory_draft.json"
DEFAULT_PRICES = ROOT / "experiments/judge_pipeline_v2/model_prices_360.json"


def load_json(path: Path, default: Any = None) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", cleaned, flags=re.S)
    if not match:
        return {"parse_error": "no_json_object", "raw_text": text}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError as exc:
        return {"parse_error": f"json_decode_error: {exc}", "raw_text": text}


def find_problem(draft: dict[str, Any], problem_id: str) -> dict[str, Any]:
    for problem in draft.get("problems", []):
        if problem.get("id") == problem_id:
            return problem
    raise KeyError(f"Problem not found: {problem_id}")


def build_generation_prompt(problem: dict[str, Any]) -> str:
    lines = [
        "You are solving a public olympiad physics benchmark problem.",
        "Give a self-contained solution. Show the derivation and final answer for every subquestion.",
        "",
        f"Problem: {problem.get('title')} ({problem.get('total_points')} points)",
    ]
    for item in problem.get("question_context", []):
        lines.append(f"- {item}")
    lines.append("")
    for sub in problem.get("subquestions", []):
        lines.append(f"{sub.get('id')} [{sub.get('points')} pts]: {sub.get('prompt')}")
    return "\n".join(lines).strip()


def build_judge_prompt(problem: dict[str, Any], candidate_answer: str) -> str:
    rubric = problem.get("marking_scheme", {})
    payload = {
        "problem_id": problem.get("id"),
        "title": problem.get("title"),
        "total_points": problem.get("total_points"),
        "answer_key_summary": problem.get("answer_key_summary"),
        "marking_scheme": rubric.get("subquestions"),
        "candidate_answer": candidate_answer,
    }
    return (
        "You are grading a public olympiad physics solution using the official marking scheme. "
        "Award partial credit criterion by criterion. Do not give credit for unsupported final answers. "
        "Return only valid JSON with keys: total_score, max_score, subquestion_scores, rationale.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--draft", type=Path, default=DEFAULT_DRAFT)
    parser.add_argument("--problem_id", default="eupho2025_t1")
    parser.add_argument("--model", default="google/gemini-3.5-flash")
    parser.add_argument("--judge_model", default="google/gemini-3.5-flash")
    parser.add_argument("--output_root", type=Path, default=ROOT / "artifacts/runs/candidate_benchmarks")
    parser.add_argument("--run_name", default="")
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--api_backend", default="requests", choices=["requests", "openai"])
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--max_try", type=int, default=2)
    parser.add_argument("--retry_delay", type=float, default=5.0)
    args = parser.parse_args()

    os.environ.update(load_env_file(args.env_file))
    draft = load_json(args.draft, {})
    problem = find_problem(draft, args.problem_id)
    prices = load_json(DEFAULT_PRICES, {})
    run_name = args.run_name or f"eupho2025_smoke_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.output_root / run_name

    assets: list[dict[str, Any]] = []
    image_urls: list[str] = []
    for asset in problem.get("assets", []):
        path = ROOT / asset.get("path", "")
        if not path.exists():
            assets.append({**asset, "status": "missing"})
            continue
        image_urls.append(file_to_data_url(path))
        assets.append({**asset, "status": "sent_as_data_url", "bytes": path.stat().st_size})

    client = UnifiedChatClient(
        api_key=api_key_from_env(),
        backend=args.api_backend,
        max_retries=args.max_try,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
    )

    generation_prompt = build_generation_prompt(problem)
    generation_message = build_user_message(text=generation_prompt, image_urls=image_urls, use_multimodal=True)
    generation_response = client.chat(
        model=args.model,
        messages=[generation_message],
        max_tokens=args.max_tokens,
        thinking_mode="auto",
        thinking_effort="minimal",
    )
    candidate_answer = generation_response.get("content") or generation_response.get("llm_output") or ""

    judge_prompt = build_judge_prompt(problem, candidate_answer)
    judge_response = client.chat(
        model=args.judge_model,
        messages=[{"role": "user", "content": judge_prompt}],
        max_tokens=args.max_tokens,
        thinking_mode="auto",
        thinking_effort="minimal",
    )
    judge_text = judge_response.get("content") or judge_response.get("llm_output") or ""
    parsed_judge = extract_json_object(judge_text)

    generation_usage = usage_cost(generation_response.get("usage", {}) or {}, args.model, prices)
    judge_usage = usage_cost(judge_response.get("usage", {}) or {}, args.judge_model, prices)
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_id": draft.get("benchmark_id"),
        "problem_id": args.problem_id,
        "model": args.model,
        "judge_model": args.judge_model,
        "asset_count": len(assets),
        "sent_image_count": len(image_urls),
        "generation_finish_reason": generation_response.get("finish_reason"),
        "judge_finish_reason": judge_response.get("finish_reason"),
        "parsed_judge": parsed_judge,
        "generation_usage": generation_usage,
        "judge_usage": judge_usage,
        "total_cost_cny": generation_usage["cost_cny"] + judge_usage["cost_cny"],
    }

    save_json(run_dir / "problem.json", problem)
    save_json(run_dir / "assets.json", assets)
    save_json(run_dir / "prompts.json", {"generation": generation_prompt, "judge": judge_prompt})
    save_json(run_dir / "generation_response.json", generation_response)
    save_json(run_dir / "judge_response.json", judge_response)
    save_json(run_dir / "judge_result.json", parsed_judge)
    save_json(run_dir / "summary.json", summary)
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
