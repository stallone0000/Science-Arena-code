#!/usr/bin/env python3
"""Verify real image visibility for requested 360 model routes.

This is stricter than a route-acceptance smoke: the model must recover text or
shape counts from the image, not merely accept an image_url block.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import requests
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ENV_FILE = Path("/home/jovyan/zhaoguangxiang-data-zzzc/zhaoguangxiang/.config/science_arena/360.env")
DEFAULT_OUTPUT_ROOT = ROOT / "artifacts/runs/vision_visibility_smoke_20260528"
DEFAULT_API_URL = "https://api.360.cn/v1/chat/completions"
REMOTE_TEXT_IMAGE_URL = "https://dummyimage.com/480x220/ffffff/000000.png&text=VISION-730"
DEFAULT_MODELS = [
    "moonshotai/kimi-k2.6",
    "moonshot/moonshotai/kimi-k2.6",
    "xiaomi/mimo-v2.5",
    "openrouter/xiaomi/mimo-v2.5",
    "anthropic/claude-sonnet-4-6",
    "xai/grok-4",
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
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError(f"Missing API key. Not found in environment or {env_file}")


def make_visibility_png() -> bytes:
    image = Image.new("RGB", (640, 300), "white")
    draw = ImageDraw.Draw(image)
    try:
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 52)
        font_small = ImageFont.truetype("DejaVuSans.ttf", 26)
    except Exception:
        font_big = ImageFont.load_default()
        font_small = ImageFont.load_default()
    draw.text((42, 34), "VISION-730", fill="black", font=font_big)
    draw.text((42, 112), "Count the shapes below.", fill="black", font=font_small)
    draw.rectangle((72, 175, 142, 245), fill=(220, 20, 60), outline="black", width=3)
    draw.ellipse((260, 175, 330, 245), fill=(30, 90, 220), outline="black", width=3)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def make_data_url(png_bytes: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")


def token_budget_for(model: str, requested: int) -> int:
    if model.lower().endswith("/grok-4") or model.lower() == "grok-4":
        return min(requested, 512)
    return requested


def build_payload(model: str, image_url: str, mode: str, max_tokens: int, temperature: float | None) -> dict[str, Any]:
    if mode == "data":
        prompt = (
            "Read the attached image. Return only compact JSON with keys "
            "code, red_squares, blue_circles. The code is the exact large text in the image."
        )
    else:
        prompt = (
            "Read the attached image. Return only compact JSON with key code. "
            "The code is the exact visible text in the image."
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
        "max_tokens": token_budget_for(model, max_tokens),
    }
    if temperature is not None:
        payload["temperature"] = float(temperature)
    return payload


def call_model(args: argparse.Namespace, model: str, mode: str, image_url: str, api_key: str) -> dict[str, Any]:
    payload = build_payload(model, image_url, mode, args.max_tokens, args.temperature)
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
            body: Any = response.json()
        except Exception:
            body = {"raw_text": response.text[:2000]}
        choice = (body.get("choices") or [{}])[0] if isinstance(body, dict) else {}
        message = choice.get("message") if isinstance(choice, dict) else {}
        content = message.get("content", "") if isinstance(message, dict) else ""
        text = str(content or "")
        visible_code = "VISION-730" in text.upper()
        visible_shapes = bool(re.search(r"red[^0-9]{0,20}1", text, flags=re.I)) and bool(
            re.search(r"blue[^0-9]{0,20}1", text, flags=re.I)
        )
        visibility_pass = visible_code and (visible_shapes if mode == "data" else True)
        return {
            "model": model,
            "mode": mode,
            "status_code": status_code,
            "route_ok": status_code < 400 and bool(text.strip()),
            "visibility_pass": visibility_pass,
            "visible_code": visible_code,
            "visible_shapes": visible_shapes if mode == "data" else None,
            "finish_reason": choice.get("finish_reason") if isinstance(choice, dict) else None,
            "usage": body.get("usage", {}) if isinstance(body, dict) else {},
            "content_excerpt": text[:1000],
            "error": "" if status_code < 400 else json.dumps(body, ensure_ascii=False)[:2000],
            "payload_image_mode": mode,
            "payload_token_budget": payload["max_tokens"],
        }
    except Exception as exc:
        return {
            "model": model,
            "mode": mode,
            "status_code": 0,
            "route_ok": False,
            "visibility_pass": False,
            "visible_code": False,
            "visible_shapes": False if mode == "data" else None,
            "finish_reason": None,
            "usage": {},
            "content_excerpt": "",
            "error": f"{type(exc).__name__}: {exc}",
            "payload_image_mode": mode,
            "payload_token_budget": token_budget_for(model, args.max_tokens),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS))
    parser.add_argument("--modes", default="data,url")
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args()

    api_key = api_key_from_env(args.env_file)
    run_name = args.run_name or f"requested_model_vision_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir = args.output_root / run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    png_bytes = make_visibility_png()
    (out_dir / "visibility_probe.png").write_bytes(png_bytes)
    data_url = make_data_url(png_bytes)
    image_urls = {"data": data_url, "url": REMOTE_TEXT_IMAGE_URL}

    results: list[dict[str, Any]] = []
    for model in parse_csv(args.models):
        for mode in parse_csv(args.modes):
            if mode not in image_urls:
                raise SystemExit(f"Unsupported mode: {mode}")
            row = call_model(args, model, mode, image_urls[mode], api_key)
            results.append(row)
            (out_dir / "results.partial.json").write_text(
                json.dumps({"results": results}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "model": model,
                        "mode": mode,
                        "route_ok": row["route_ok"],
                        "visibility_pass": row["visibility_pass"],
                        "status_code": row["status_code"],
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "api_url": args.api_url,
        "remote_text_image_url": REMOTE_TEXT_IMAGE_URL,
        "image_asset": str(out_dir / "visibility_probe.png"),
        "results": results,
        "visibility_by_model": {
            model: {
                row["mode"]: row["visibility_pass"]
                for row in results
                if row["model"] == model
            }
            for model in parse_csv(args.models)
        },
    }
    (out_dir / "results.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary["visibility_by_model"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
