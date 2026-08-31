#!/usr/bin/env python3
"""Streaming probe for Kimi K2.6 on IChO Q1.3.

The non-streaming route can read the Q1.3 image but stalls when asked to solve.
This diagnostic checks whether streaming exposes any visible content or only
hidden reasoning/empty deltas before the route stalls.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import requests


DEFAULT_OUT = Path(
    "artifacts/runs/latest_360_20260529_kimi_context_checks_v1/"
    "raw_q1_3_streaming_matrix_v1"
)
DEFAULT_IMAGE_DIR = Path(
    "artifacts/runs/latest_360_20260529_kimi_context_checks_v1/"
    "raw_q1_3_image_payload_matrix_v1"
)
DEFAULT_PROMPTS = Path(
    "artifacts/runs/latest_360_20260529_kimi_context_checks_v1/"
    "latest_360_20260529_kimi_direct_icho_q1_3_multimodal_nosampling_30k_v1/"
    "interleaved/icho/prompts.json"
)


def api_key() -> str:
    for name in ("QIHOO_API_KEY", "QIHOO360_API_KEY", "360_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError("No 360 API key env var is set")


def image_data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def load_prompt_text(path: Path) -> str:
    row = json.loads(path.read_text(encoding="utf-8"))[0]
    user_content = row["messages"][1]["content"]
    return next(block["text"] for block in user_content if block.get("type") == "text")


def build_cases(model: str) -> list[dict[str, Any]]:
    return [
        {
            "id": "direct_stream_read_w480_nosampling_omitthinking_512",
            "model": model,
            "task": "read",
            "use_image": True,
            "sampling": None,
            "thinking": "omit",
            "max_tokens": 512,
            "timeout": 300,
        },
        {
            "id": "direct_stream_solve_brief_w480_nosampling_omitthinking_512",
            "model": model,
            "task": "solve_brief",
            "use_image": True,
            "sampling": None,
            "thinking": "omit",
            "max_tokens": 512,
            "timeout": 900,
        },
        {
            "id": "direct_stream_solve_brief_w480_nosampling_reasoningsplit_false_512",
            "model": model,
            "task": "solve_brief",
            "use_image": True,
            "sampling": None,
            "thinking": "omit",
            "extra": {"reasoning_split": False},
            "max_tokens": 512,
            "timeout": 900,
        },
        {
            "id": "direct_stream_solve_brief_w480_temp1_top095_reasoningsplit_false_512",
            "model": model,
            "task": "solve_brief",
            "use_image": True,
            "sampling": {"temperature": 1, "top_p": 0.95},
            "thinking": "omit",
            "extra": {"reasoning_split": False},
            "max_tokens": 512,
            "timeout": 900,
        },
        {
            "id": "direct_stream_solve_brief_w480_nosampling_enablethinking_false_reasoningsplit_false_512",
            "model": model,
            "task": "solve_brief",
            "use_image": True,
            "sampling": None,
            "thinking": "omit",
            "extra": {"enable_thinking": False, "reasoning_split": False},
            "max_tokens": 512,
            "timeout": 900,
        },
        {
            "id": "direct_stream_solve_brief_w480_temp1_top095_omitthinking_512",
            "model": model,
            "task": "solve_brief",
            "use_image": True,
            "sampling": {"temperature": 1, "top_p": 0.95},
            "thinking": "omit",
            "max_tokens": 512,
            "timeout": 900,
        },
        {
            "id": "direct_stream_solve_brief_w480_nosampling_off_512",
            "model": model,
            "task": "solve_brief",
            "use_image": True,
            "sampling": None,
            "thinking": "off",
            "max_tokens": 512,
            "timeout": 900,
        },
        {
            "id": "direct_stream_solve_brief_textonly_nosampling_omitthinking_512",
            "model": model,
            "task": "solve_brief",
            "use_image": False,
            "sampling": None,
            "thinking": "omit",
            "max_tokens": 512,
            "timeout": 600,
        },
    ]


def parse_stream(resp: requests.Response, raw_path: Path) -> tuple[dict[str, Any], str, str]:
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    event_count = 0
    finish_reason = ""
    raw_events: list[str] = []
    with raw_path.open("w", encoding="utf-8") as raw:
        for line in resp.iter_lines(decode_unicode=True):
            if not line:
                continue
            raw.write(line + "\n")
            raw.flush()
            if line.startswith("data:"):
                payload = line[5:].strip()
            else:
                payload = line.strip()
            if payload == "[DONE]":
                break
            raw_events.append(payload[:1000])
            event_count += 1
            try:
                data = json.loads(payload)
            except Exception:
                continue
            choice = (data.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            message = choice.get("message") or {}
            for obj in (delta, message):
                text = obj.get("content")
                if text:
                    content_parts.append(text)
                reasoning = obj.get("reasoning_content") or obj.get("reasoning")
                if reasoning:
                    reasoning_parts.append(str(reasoning))
            if choice.get("finish_reason"):
                finish_reason = choice.get("finish_reason")
    return (
        {
            "event_count": event_count,
            "finish_reason": finish_reason,
            "raw_event_samples": raw_events[:5],
        },
        "".join(content_parts),
        "".join(reasoning_parts),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moonshotai/kimi-k2.6")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--prompts_json", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--case_ids", default="")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    prompt_text = load_prompt_text(args.prompts_json)
    read_prompt = (
        "The attached image is a chemistry reaction scheme from IChO 2025 Q1. "
        "If the image is visible, start with IMAGE_VISIBLE. Then transcribe labels and arrows. "
        "Do not solve. Keep under 200 words."
    )
    solve_brief_prompt = (
        "This is IChO 2025 Q1.3. Use the attached reaction scheme if present. "
        "Question: Draw/describe compounds E and F. Stereochemistry is not required. "
        "Return only two lines, exactly: E: ... and F: ... . Do not explain."
    )
    # Keep full prompt available in artifacts for reproducibility.
    (args.out_dir / "source_prompt.txt").write_text(prompt_text, encoding="utf-8")

    image_path = args.image_dir / "w480q70.jpg"
    image_url = image_data_url(image_path)
    asset = {
        "path": str(image_path),
        "bytes_len": image_path.stat().st_size,
        "data_url_len": len(image_url),
    }
    cases = build_cases(args.model)
    if args.case_ids:
        wanted = {case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()}
        cases = [case for case in cases if case["id"] in wanted]
        missing = sorted(wanted - {case["id"] for case in cases})
        if missing:
            raise ValueError(f"Unknown case ids: {missing}")

    headers = {
        "Authorization": f"Bearer {api_key()}",
        "Content-Type": "application/json",
        "Host": "api.360.cn",
    }
    url = "https://api.360.cn/v1/chat/completions"
    jsonl_path = args.out_dir / "results.jsonl"
    results: list[dict[str, Any]] = []

    for case in cases:
        print(f"RUN {case['id']}", flush=True)
        text = read_prompt if case["task"] == "read" else solve_brief_prompt
        user_content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if case["use_image"]:
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        payload: dict[str, Any] = {
            "model": case["model"],
            "messages": [
                {"role": "system", "content": "Be concise. Do not write <think>."},
                {"role": "user", "content": user_content},
            ],
            "stream": True,
            "max_completion_tokens": case["max_tokens"],
        }
        if case.get("extra"):
            payload.update(case["extra"])
        if case["sampling"]:
            payload.update(case["sampling"])
        if case["thinking"] == "off":
            payload["thinking"] = {"type": "disabled"}

        started = time.time()
        record: dict[str, Any] = {
            **case,
            "asset": asset if case["use_image"] else None,
        }
        raw_path = args.out_dir / f"{case['id']}.stream.txt"
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=case["timeout"], stream=True)
            record["http_status"] = resp.status_code
            if resp.status_code >= 400:
                record["status"] = "http_error"
                record["error"] = resp.text[:1000]
                record["content"] = ""
                raw_path.write_text(resp.text[:5000], encoding="utf-8")
            else:
                stream_meta, content, reasoning = parse_stream(resp, raw_path)
                record.update(stream_meta)
                record["status"] = "ok"
                record["error"] = ""
                record["content"] = content
                record["content_len"] = len(content)
                record["reasoning_len"] = len(reasoning)
                record["visible_marker"] = "IMAGE_VISIBLE" in content.upper()
        except Exception as exc:
            record["status"] = "exception"
            record["error"] = repr(exc)
            record["traceback"] = traceback.format_exc()[-2000:]
            record["content"] = ""
        record["elapsed_s"] = round(time.time() - started, 3)

        compact = {
            key: record.get(key)
            for key in [
                "id",
                "model",
                "task",
                "use_image",
                "sampling",
                "thinking",
                "extra",
                "max_tokens",
                "status",
                "http_status",
                "finish_reason",
                "event_count",
                "content_len",
                "reasoning_len",
                "visible_marker",
                "elapsed_s",
                "error",
            ]
        }
        compact["content_head"] = (record.get("content") or "")[:500]
        compact["content_tail"] = (record.get("content") or "")[-500:]
        results.append(compact)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        (args.out_dir / "summary.json").write_text(
            json.dumps({"asset": asset, "results": results}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(json.dumps(compact, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
