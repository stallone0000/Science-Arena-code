#!/usr/bin/env python3
"""Smoke-test target 360 model routes with tiny prompts and usage/cost logging."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[2]
API_URL = "https://api.360.cn/v1/chat/completions"
API_KEY_ENV_VARS = ("QIHOO_API_KEY", "QIHOO360_API_KEY", "360_API_KEY")

TARGET_MODELS = [
    "cloudsway/gemini-3-flash-preview",
    "google/gemini-3.5-flash",
    "qwen/qwen3.7-max",
    "qwen/qwen3.6-plus",
    "bytedance/doubao-seed-2-0-pro",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "moonshotai/kimi-k2.6",
    "openai/gpt-5.5",
    "anthropic/claude-opus-4.7",
    "minimax/MiniMax-M2.7",
    "z-ai/glm-5.1",
    "xiaomi/mimo-v2.5-pro",
]


def get_api_key() -> str:
    for env in API_KEY_ENV_VARS:
        value = os.environ.get(env)
        if value and value.strip():
            return value.strip()
    raise RuntimeError(f"Missing API key. Set one of: {', '.join(API_KEY_ENV_VARS)}")


def load_prices() -> dict[str, Any]:
    path = ROOT / "experiments/judge_pipeline_v2/model_prices_360.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def usage_cost(usage: dict[str, Any], model: str, prices: dict[str, Any]) -> dict[str, float]:
    row = prices.get(model, {})
    prompt_tokens = float(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = float(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    input_price = float(row.get("input_price_cny_per_1m", 0) or 0)
    output_price = float(row.get("output_price_cny_per_1m", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": float(usage.get("total_tokens") or prompt_tokens + completion_tokens),
        "cost_cny": prompt_tokens / 1_000_000 * input_price + completion_tokens / 1_000_000 * output_price,
        "input_price_cny_per_1m": input_price,
        "output_price_cny_per_1m": output_price,
    }


def call_model(model: str, args: argparse.Namespace, api_key: str, prices: dict[str, Any]) -> dict[str, Any]:
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "Host": "api.360.cn"}
    base_payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply exactly with: ok"}],
        "stream": False,
        "num_beams": 1,
        "repetition_penalty": 1.05,
    }
    errors = []
    start = time.time()
    for token_param in ("max_completion_tokens", "max_tokens"):
        payload = dict(base_payload)
        payload[token_param] = args.max_tokens
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=args.timeout)
            if response.status_code >= 400:
                errors.append({"token_param": token_param, "status_code": response.status_code, "body": response.text[:1000]})
                continue
            data = response.json()
            text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            usage = data.get("usage", {})
            return {
                "model": model,
                "ok": True,
                "token_param": token_param,
                "latency_seconds": time.time() - start,
                "content_excerpt": text[:200],
                "usage": usage,
                "cost": usage_cost(usage, model, prices),
            }
        except Exception as exc:
            errors.append({"token_param": token_param, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "model": model,
        "ok": False,
        "latency_seconds": time.time() - start,
        "errors": errors,
        "usage": {},
        "cost": usage_cost({}, model, prices),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(TARGET_MODELS))
    parser.add_argument("--max_workers", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--run_name", default="")
    args = parser.parse_args()
    models = [item.strip() for item in args.models.split(",") if item.strip()]
    api_key = get_api_key()
    prices = load_prices()
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = args.run_name or f"target_model_smoke_{stamp}"
    out_dir = ROOT / "experiments/judge_pipeline_v2/api_smoke" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(call_model, model, args, api_key, prices) for model in models]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    results.sort(key=lambda row: row["model"].lower())
    summary = {
        "run_name": run_name,
        "api_url": API_URL,
        "created_at": stamp,
        "max_tokens": args.max_tokens,
        "models": results,
        "ok_count": sum(1 for row in results if row["ok"]),
        "error_count": sum(1 for row in results if not row["ok"]),
        "total_cost_cny": sum(float(row.get("cost", {}).get("cost_cny", 0)) for row in results),
        "total_prompt_tokens": sum(float(row.get("cost", {}).get("prompt_tokens", 0)) for row in results),
        "total_completion_tokens": sum(float(row.get("cost", {}).get("completion_tokens", 0)) for row in results),
    }
    (out_dir / "smoke_results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
