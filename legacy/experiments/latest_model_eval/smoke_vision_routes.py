#!/usr/bin/env python3
"""Smoke-test whether current 360 model routes accept image_url blocks."""

from __future__ import annotations

import argparse
import base64
import concurrent.futures
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = Path("/home/jovyan/zhaoguangxiang-data-zzzc/zhaoguangxiang/.config/science_arena/360.env")
DEFAULT_OUTPUT_ROOT = ROOT / "experiments/latest_model_eval/vision_smoke"
DEFAULT_API_URL = "https://api.360.cn/v1/chat/completions"
DEFAULT_IMAGE_URL = (
    "https://science-data-360-1368061776.cos.ap-beijing.myqcloud.com/"
    "Biology/hlf/IBO2023/019a011b-60a3-70b3-9568-1f06a3623f08_2_165_634_1313_808_0.jpg"
)
DEFAULT_MODELS = [
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


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\s,]+", raw or "") if item.strip()]


def model_tag(model: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "__", model).strip("_") or "model"


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def api_key_from_env(env_file: Path) -> str:
    for key, value in load_env_file(env_file).items():
        os.environ.setdefault(key, value)
    for name in ("QIHOO_API_KEY", "QIHOO360_API_KEY", "360_API_KEY", "API_KEY"):
        if os.environ.get(name):
            return str(os.environ[name])
    raise RuntimeError(f"Missing API key. Source env or provide --env_file. Not found: {env_file}")


def red_dot_data_url() -> str:
    # 1x1 red PNG.
    data = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADUlEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
    )
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def token_param_for(model: str) -> str:
    canonical = model.lower().split("/")[-1]
    if canonical.startswith("gpt-") or canonical.startswith(("o1-", "o3-", "o4-")):
        return "max_completion_tokens"
    return "max_tokens"


def smoke_one(args: argparse.Namespace, model: str, image_mode: str, api_key: str) -> dict[str, Any]:
    image_url = red_dot_data_url() if image_mode == "data" else args.image_url
    prompt = (
        "This is a route capability smoke test. If you can process the attached image block, "
        "answer with exactly: IMAGE_BLOCK_ACCEPTED"
    )
    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }
        ],
        "stream": False,
        token_param_for(model): args.max_tokens,
    }
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.post(
            args.api_url,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "Host": "api.360.cn",
            },
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=args.timeout,
        )
        status_code = response.status_code
        try:
            body = response.json()
        except Exception:
            body = {"raw_text": response.text[:1000]}
        choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
        message = choice.get("message") if isinstance(choice, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        return {
            "model": model,
            "image_mode": image_mode,
            "ok": status_code < 400 and bool(content),
            "status_code": status_code,
            "content_excerpt": str(content)[:300],
            "usage": body.get("usage", {}) if isinstance(body, dict) else {},
            "error": "" if status_code < 400 else json.dumps(body, ensure_ascii=False)[:1200],
        }
    except Exception as exc:
        return {
            "model": model,
            "image_mode": image_mode,
            "ok": False,
            "status_code": 0,
            "content_excerpt": "",
            "usage": {},
            "error": f"{type(exc).__name__}: {exc}",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--image_modes", default="url,data", help="Comma/space separated: url,data")
    parser.add_argument("--image_url", default=DEFAULT_IMAGE_URL)
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=180)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    api_key = api_key_from_env(args.env_file)
    run_name = args.run_name or f"vision_smoke_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = args.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    models = parse_csv(args.models)
    image_modes = parse_csv(args.image_modes)
    tasks = [(model, mode) for model in models for mode in image_modes]
    results: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
        futures = {executor.submit(smoke_one, args, model, mode, api_key): (model, mode) for model, mode in tasks}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            results.append(row)
            (out_dir / "results_partial.json").write_text(
                json.dumps(sorted(results, key=lambda r: (r["model"], r["image_mode"])), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[VISION] {row['model']} {row['image_mode']}: ok={row['ok']} status={row['status_code']}")

    results.sort(key=lambda r: (r["model"], r["image_mode"]))
    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "api_url": args.api_url,
        "max_tokens": args.max_tokens,
        "results": results,
        "ok_by_model": {
            model: {row["image_mode"]: row["ok"] for row in results if row["model"] == model}
            for model in models
        },
    }
    (out_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["ok_by_model"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
