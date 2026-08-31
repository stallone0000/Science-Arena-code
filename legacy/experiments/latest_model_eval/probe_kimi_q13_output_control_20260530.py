#!/usr/bin/env python3
"""Small 2026-05-30 Kimi K2.6 Q1.3 visible-output control probe.

This is a diagnostic only. It records whether the 360 Kimi route emits visible
assistant content for IChO Q1.3 solving prompts, while separately counting
hidden reasoning deltas. Hidden reasoning is never treated as the answer.
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
    "artifacts/runs/latest_360_20260530_kimi_q13_output_control_v1"
)
DEFAULT_IMAGE_DIR = Path(
    "artifacts/runs/latest_360_20260529_kimi_context_checks_v1/"
    "raw_q1_3_image_payload_matrix_v1"
)


def api_key() -> str:
    for name in ("QIHOO_API_KEY", "QIHOO360_API_KEY", "360_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError("No 360 API key env var is set")


def image_data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def parse_error_code(text: str) -> str:
    try:
        data = json.loads(text)
    except Exception:
        return ""
    error = data.get("error") if isinstance(data, dict) else None
    if isinstance(error, dict):
        return str(error.get("code") or "")
    return ""


def build_cases(model: str, max_tokens: int, timeout: float) -> list[dict[str, Any]]:
    suffix = str(max_tokens)
    route_tag = "prefixed" if model.startswith("moonshot/") else "direct"
    system_strict = (
        "You are solving a released public chemistry olympiad benchmark. "
        "Do not use hidden reasoning. Put the final answer in the visible "
        "assistant content. Output only the requested final answer lines."
    )
    prompt_strict = (
        "IChO 2025 Q1.3. Use the attached reaction scheme image. "
        "Determine compounds E and F. Stereochemistry is not required.\n"
        "Visible answer only. Return exactly two lines and nothing else:\n"
        "E: <structure or concise description>\n"
        "F: <structure or concise description>"
    )
    prompt_contract = (
        "The attached image is visible input for IChO 2025 Q1.3. "
        "You must answer in the normal visible assistant message, not in a "
        "reasoning field. Do not explain. Do not write analysis. "
        "Only output:\nE: ...\nF: ..."
    )
    return [
        {
            "id": f"{route_tag}_strict_visible_nosampling_{suffix}",
            "model": model,
            "system": system_strict,
            "prompt": prompt_strict,
            "sampling": None,
            "extra": {},
            "max_tokens": max_tokens,
            "timeout": timeout,
        },
        {
            "id": f"{route_tag}_strict_visible_temp1_top095_{suffix}",
            "model": model,
            "system": system_strict,
            "prompt": prompt_strict,
            "sampling": {"temperature": 1, "top_p": 0.95},
            "extra": {},
            "max_tokens": max_tokens,
            "timeout": timeout,
        },
        {
            "id": f"{route_tag}_strict_visible_temp08_top095_{suffix}",
            "model": model,
            "system": system_strict,
            "prompt": prompt_strict,
            "sampling": {"temperature": 0.8, "top_p": 0.95},
            "extra": {},
            "max_tokens": max_tokens,
            "timeout": min(timeout, 300),
        },
        {
            "id": f"{route_tag}_contract_nosampling_reasoningsplit_false_{suffix}",
            "model": model,
            "system": system_strict,
            "prompt": prompt_contract,
            "sampling": None,
            "extra": {"reasoning_split": False},
            "max_tokens": max_tokens,
            "timeout": timeout,
        },
    ]


def parse_stream(resp: requests.Response) -> dict[str, Any]:
    content_parts: list[str] = []
    reasoning_len = 0
    event_count = 0
    finish_reason = ""
    chunk_field_counts = {
        "content": 0,
        "reasoning_content": 0,
        "reasoning": 0,
    }
    raw_event_samples: list[dict[str, Any]] = []

    for line in resp.iter_lines(decode_unicode=True):
        if not line:
            continue
        payload = line[5:].strip() if line.startswith("data:") else line.strip()
        if payload == "[DONE]":
            break
        event_count += 1
        try:
            data = json.loads(payload)
        except Exception:
            continue
        choice = (data.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        message = choice.get("message") or {}
        if len(raw_event_samples) < 5:
            raw_event_samples.append(
                {
                    "choice_keys": sorted(choice.keys()),
                    "delta_keys": sorted(delta.keys()) if isinstance(delta, dict) else [],
                    "message_keys": sorted(message.keys()) if isinstance(message, dict) else [],
                    "finish_reason": choice.get("finish_reason"),
                }
            )
        for obj in (delta, message):
            if not isinstance(obj, dict):
                continue
            text = obj.get("content")
            if text:
                chunk_field_counts["content"] += 1
                content_parts.append(str(text))
            reasoning_content = obj.get("reasoning_content")
            if reasoning_content:
                chunk_field_counts["reasoning_content"] += 1
                reasoning_len += len(str(reasoning_content))
            reasoning = obj.get("reasoning")
            if reasoning:
                chunk_field_counts["reasoning"] += 1
                reasoning_len += len(str(reasoning))
        if choice.get("finish_reason"):
            finish_reason = str(choice.get("finish_reason"))

    content = "".join(content_parts)
    return {
        "event_count": event_count,
        "finish_reason": finish_reason,
        "content": content,
        "content_len": len(content),
        "reasoning_content_len": reasoning_len,
        "chunk_field_counts": chunk_field_counts,
        "raw_event_samples": raw_event_samples,
    }


def write_summary(out_dir: Path, payload: dict[str, Any]) -> None:
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moonshotai/kimi-k2.6")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--image_name", default="w480q70.jpg")
    parser.add_argument("--max_tokens", type=int, default=131072)
    parser.add_argument("--timeout", type=float, default=1200)
    parser.add_argument("--case_ids", default="")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    image_path = args.image_dir / args.image_name
    image_url = image_data_url(image_path)
    image_meta = {
        "path": str(image_path),
        "bytes_len": image_path.stat().st_size,
        "data_url_len": len(image_url),
    }
    cases = build_cases(args.model, args.max_tokens, args.timeout)
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
    results: list[dict[str, Any]] = []
    summary: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "artifact_dir": str(args.out_dir),
        "model": args.model,
        "token_clamp_disabled_env": os.environ.get("SCIENCE_ARENA_DISABLE_TOKEN_CLAMP") == "1",
        "image": image_meta,
        "results": results,
    }
    write_summary(args.out_dir, summary)

    jsonl_path = args.out_dir / "results.jsonl"
    for case in cases:
        print(f"RUN {case['id']}", flush=True)
        user_content = [
            {"type": "text", "text": case["prompt"]},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
        request_payload: dict[str, Any] = {
            "model": case["model"],
            "messages": [
                {"role": "system", "content": case["system"]},
                {"role": "user", "content": user_content},
            ],
            "stream": True,
            "max_completion_tokens": case["max_tokens"],
        }
        if case["sampling"]:
            request_payload.update(case["sampling"])
        request_payload.update(case["extra"])

        record: dict[str, Any] = {
            "id": case["id"],
            "model": case["model"],
            "sampling": case["sampling"],
            "extra": case["extra"],
            "max_completion_tokens": case["max_tokens"],
            "timeout": case["timeout"],
            "payload_image_summary": {
                "image_url_block_count": 1,
                "data_url_image_count": 1,
                "remote_url_image_count": 0,
            },
        }
        started = time.time()
        try:
            resp = requests.post(
                url,
                headers=headers,
                json=request_payload,
                timeout=case["timeout"],
                stream=True,
            )
            record["http_status"] = resp.status_code
            if resp.status_code >= 400:
                text = resp.text[:4000]
                record.update(
                    {
                        "status": "http_error",
                        "error": text[:1000],
                        "error_code": parse_error_code(text),
                        "finish_reason": "",
                        "content": "",
                        "content_len": 0,
                        "reasoning_content_len": 0,
                    }
                )
            else:
                stream_result = parse_stream(resp)
                content = stream_result.pop("content")
                record.update(stream_result)
                record.update(
                    {
                        "status": "ok",
                        "error": "",
                        "error_code": "",
                        "content_head": content[:500],
                        "content_tail": content[-500:],
                    }
                )
        except Exception as exc:
            record.update(
                {
                    "status": "exception",
                    "error": repr(exc),
                    "error_code": "",
                    "traceback": traceback.format_exc()[-2000:],
                    "finish_reason": "",
                    "content_len": 0,
                    "reasoning_content_len": 0,
                }
            )
        record["elapsed_s"] = round(time.time() - started, 3)
        results.append(record)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_summary(args.out_dir, summary)
        print(
            json.dumps(
                {
                    "id": record["id"],
                    "status": record.get("status"),
                    "http": record.get("http_status"),
                    "error_code": record.get("error_code"),
                    "finish": record.get("finish_reason"),
                    "content_len": record.get("content_len"),
                    "reasoning_content_len": record.get("reasoning_content_len"),
                    "elapsed_s": record.get("elapsed_s"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_summary(args.out_dir, summary)
    print(f"DONE {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
