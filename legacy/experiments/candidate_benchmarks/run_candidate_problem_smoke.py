#!/usr/bin/env python3
"""Run a page-range candidate benchmark smoke test.

This is intentionally lightweight: it reads extracted PDF page text and images,
sends a single public problem to a solver model, then asks a judge model to grade
against the official solution/marking-scheme pages.
"""

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
    def parse_json_maybe_sanitize(candidate: str) -> Any:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Some judges accidentally emit LaTeX-like fragments such as
            # "\u00tilde" inside JSON strings. Treat invalid unicode escapes
            # as literal text rather than failing the whole score parse.
            sanitized = re.sub(r"\\u(?![0-9a-fA-F]{4})", r"\\\\u", candidate)
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
        return {"parse_error": "no_json_object", "raw_text": text}
    try:
        parsed = parse_json_maybe_sanitize(match.group(0))
        return parsed if isinstance(parsed, dict) else {"raw": parsed}
    except json.JSONDecodeError as exc:
        return {"parse_error": f"json_decode_error: {exc}", "raw_text": text}


def selected_pages(pages_json: dict[str, Any], page_numbers: list[int]) -> list[dict[str, Any]]:
    wanted = {int(page) for page in page_numbers}
    return [page for page in pages_json.get("pages", []) if int(page.get("page", -1)) in wanted]


def page_text_blob(pages: list[dict[str, Any]], label: str) -> str:
    chunks = []
    for page in pages:
        chunks.append(f"[{label} page {page.get('page')}]\n{str(page.get('text') or '').strip()}")
    return "\n\n".join(chunks).strip()


def page_image_paths(
    pages_json_path: Path,
    pages: list[dict[str, Any]],
    *,
    include_page_renders: bool = False,
    image_asset_mode: str = "all",
) -> list[Path]:
    base_dir = pages_json_path.parent
    paths: list[Path] = []
    for page in pages:
        if include_page_renders and image_asset_mode in {"all", "page_renders"}:
            render = page.get("page_render") or {}
            rel = render.get("file")
            if rel:
                path = base_dir / str(rel)
                if path.exists():
                    paths.append(path)
        if image_asset_mode == "page_renders":
            continue
        for image in page.get("images", []) or []:
            rel = image.get("file")
            if not rel:
                continue
            path = base_dir / str(rel)
            if path.exists():
                paths.append(path)
    return paths


def build_generation_prompt(config: dict[str, Any], question_text: str, *, public_benchmark_context: bool = False) -> str:
    if public_benchmark_context:
        intro = (
            "You are working with a released public science benchmark item for research evaluation. "
            "This is not a live exam, contest, assignment, or proctored assessment. "
            "Provide an educational, self-contained solution."
        )
    else:
        intro = "You are solving a public olympiad physics benchmark problem."
    return f"""{intro}
Give a self-contained solution. Show the derivation and final answer for every subquestion.
For any requested sketch/diagram, describe the required graphical features in words.

Problem: {config.get('title')} ({config.get('total_points')} points)

Official problem text:
{question_text}
"""


def build_judge_prompt(config: dict[str, Any], question_text: str, solution_text: str, candidate_answer: str) -> str:
    payload = {
        "benchmark_id": config.get("benchmark_id"),
        "problem_id": config.get("problem_id"),
        "title": config.get("title"),
        "total_points": config.get("total_points"),
        "official_problem_text": question_text,
        "official_solution_and_marking_scheme": solution_text,
        "candidate_answer": candidate_answer,
    }
    return (
        "You are a strict olympiad examiner. Grade the candidate answer only against the official solution and "
        "marking-scheme text below. Award partial credit criterion by criterion using the official raw points. "
        "Return only valid JSON with keys: total_score, max_score, sub_scores, evidence, deductions.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
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
    parser.add_argument("--pre_multimodal_mode", choices=["auto", "on", "off"], default="on")
    parser.add_argument("--thinking_mode", default="auto")
    parser.add_argument("--thinking_effort", default="auto")
    parser.add_argument("--judge_thinking_mode", default="")
    parser.add_argument("--judge_thinking_effort", default="")
    parser.add_argument("--image_asset_mode", choices=["all", "page_renders", "embedded"], default="all")
    parser.add_argument("--system_prompt", default="")
    parser.add_argument("--public_benchmark_context", action="store_true")
    args = parser.parse_args()

    os.environ.update(load_env_file(args.env_file))
    config = load_json(args.config, {})
    prices = load_json(DEFAULT_PRICES, {})
    run_name = args.run_name or f"{config.get('problem_id', 'candidate')}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.output_root / run_name

    question_pages_json_path = ROOT / config["source_files"]["question_pages_json"]
    solution_pages_json_path = ROOT / config["source_files"]["solution_pages_json"]
    question_pages = selected_pages(load_json(question_pages_json_path, {}), config.get("question_pages", []))
    solution_pages = selected_pages(load_json(solution_pages_json_path, {}), config.get("solution_pages", []))
    question_text = page_text_blob(question_pages, "question")
    solution_text = page_text_blob(solution_pages, "solution")

    image_paths = page_image_paths(
        question_pages_json_path,
        question_pages,
        include_page_renders=bool(config.get("include_question_page_renders", False)),
        image_asset_mode=args.image_asset_mode,
    )
    requested_image_urls = [file_to_data_url(path) for path in image_paths]
    image_urls = [] if args.pre_multimodal_mode == "off" else requested_image_urls
    assets = [{"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size} for path in image_paths]

    client = UnifiedChatClient(
        api_key=api_key_from_env(),
        backend=args.api_backend,
        max_retries=args.max_try,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
    )
    generation_prompt = build_generation_prompt(
        config,
        question_text,
        public_benchmark_context=args.public_benchmark_context,
    )
    if args.system_prompt.strip():
        generation_prompt = args.system_prompt.strip() + "\n\n" + generation_prompt
    generation_response = client.chat(
        model=args.model,
        messages=[
            build_user_message(
                text=generation_prompt,
                image_urls=image_urls,
                use_multimodal=args.pre_multimodal_mode != "off",
            )
        ],
        max_tokens=args.max_tokens,
        thinking_mode=args.thinking_mode,
        thinking_effort=args.thinking_effort,
    )
    candidate_answer = generation_response.get("content") or generation_response.get("llm_output") or ""

    generation_usage = usage_cost(generation_response.get("usage", {}) or {}, args.model, prices)
    generation_error = str(generation_response.get("error") or "").strip()
    if generation_error or not candidate_answer.strip():
        judge_prompt = ""
        judge_response = {}
        parsed_judge = {
            "parse_error": "generation_failed",
            "generation_error": generation_error or "empty_generation",
        }
        judge_usage = {"prompt_tokens": 0.0, "completion_tokens": 0.0, "total_tokens": 0.0, "cost_cny": 0.0}
    else:
        judge_prompt = build_judge_prompt(config, question_text, solution_text, candidate_answer)
        judge_response = client.chat(
            model=args.judge_model,
            messages=[{"role": "user", "content": judge_prompt}],
            max_tokens=args.max_tokens,
            thinking_mode=args.judge_thinking_mode or args.thinking_mode,
            thinking_effort=args.judge_thinking_effort or args.thinking_effort,
        )
        judge_text = judge_response.get("content") or judge_response.get("llm_output") or ""
        parsed_judge = extract_json_object(judge_text)
        judge_usage = usage_cost(judge_response.get("usage", {}) or {}, args.judge_model, prices)

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "benchmark_id": config.get("benchmark_id"),
        "problem_id": config.get("problem_id"),
        "model": args.model,
        "judge_model": args.judge_model,
        "question_pages": config.get("question_pages", []),
        "solution_pages": config.get("solution_pages", []),
        "pre_multimodal_mode": args.pre_multimodal_mode,
        "image_asset_mode": args.image_asset_mode,
        "available_image_count": len(requested_image_urls),
        "sent_image_count": len(image_urls),
        "generation_finish_reason": generation_response.get("finish_reason"),
        "generation_error": generation_error,
        "judge_finish_reason": judge_response.get("finish_reason"),
        "parsed_judge": parsed_judge,
        "generation_usage": generation_usage,
        "judge_usage": judge_usage,
        "total_cost_cny": generation_usage["cost_cny"] + judge_usage["cost_cny"],
    }

    save_json(run_dir / "config.json", config)
    save_json(run_dir / "assets.json", assets)
    save_json(run_dir / "prompts.json", {"generation": generation_prompt, "judge": judge_prompt})
    save_json(run_dir / "generation_response.json", generation_response)
    save_json(run_dir / "judge_response.json", judge_response)
    save_json(run_dir / "judge_result.json", parsed_judge)
    save_json(run_dir / "summary.json", summary)
    print(json.dumps({"run_dir": str(run_dir), **summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    run()
