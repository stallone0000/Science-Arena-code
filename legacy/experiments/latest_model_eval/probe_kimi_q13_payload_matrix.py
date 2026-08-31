#!/usr/bin/env python3
"""Probe Kimi K2.6 IChO Q1.3 image-payload and sampling behavior.

This script is intentionally small and diagnostic. It does not participate in
canonical benchmark scoring; it writes raw request summaries so failed Kimi
turns can be separated into image-payload, sampling, or long-output failures.
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
    "raw_q1_3_image_payload_matrix_v2"
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


def compact_result(record: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "model",
        "asset",
        "task",
        "sampling",
        "thinking",
        "max_tokens",
        "status",
        "http_status",
        "finish_reason",
        "error",
        "elapsed_s",
        "output_len",
        "visible_marker",
        "usage",
        "asset_bytes",
        "asset_data_url_len",
    ]
    item = {key: record.get(key) for key in keys}
    output = record.get("output") or ""
    item["output_head"] = output[:500]
    item["output_tail"] = output[-500:]
    return item


def write_summary(out_dir: Path, assets: dict[str, Any], results: list[dict[str, Any]]) -> None:
    payload = {
        "assets": assets,
        "results": [compact_result(record) for record in results],
    }
    (out_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def build_cases(model: str, include_orig: bool) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = [
        {
            "id": "direct_text_ok_nosampling_omitthinking_64",
            "model": model,
            "asset": "w480q70",
            "task": "ok",
            "sampling": None,
            "thinking": "omit",
            "max_tokens": 64,
            "timeout": 120,
            "use_image": False,
        },
        {
            "id": "direct_text_ok_temp1_top095_omitthinking_64",
            "model": model,
            "asset": "w480q70",
            "task": "ok",
            "sampling": {"temperature": 1, "top_p": 0.95},
            "thinking": "omit",
            "max_tokens": 64,
            "timeout": 120,
            "use_image": False,
        },
        {
            "id": "direct_w480_read_nosampling_off_512",
            "model": model,
            "asset": "w480q70",
            "task": "read",
            "sampling": None,
            "thinking": "off",
            "max_tokens": 512,
            "timeout": 240,
            "use_image": True,
        },
        {
            "id": "direct_w480_read_temp1_top095_off_512",
            "model": model,
            "asset": "w480q70",
            "task": "read",
            "sampling": {"temperature": 1, "top_p": 0.95},
            "thinking": "off",
            "max_tokens": 512,
            "timeout": 240,
            "use_image": True,
        },
        {
            "id": "direct_w480_solve_nosampling_off_2048",
            "model": model,
            "asset": "w480q70",
            "task": "solve",
            "sampling": None,
            "thinking": "off",
            "max_tokens": 2048,
            "timeout": 300,
            "use_image": True,
        },
        {
            "id": "direct_w480_solve_nosampling_omitthinking_512",
            "model": model,
            "asset": "w480q70",
            "task": "solve",
            "sampling": None,
            "thinking": "omit",
            "max_tokens": 512,
            "timeout": 300,
            "use_image": True,
        },
        {
            "id": "direct_w480_solve_nosampling_omitthinking_2048",
            "model": model,
            "asset": "w480q70",
            "task": "solve",
            "sampling": None,
            "thinking": "omit",
            "max_tokens": 2048,
            "timeout": 420,
            "use_image": True,
        },
        {
            "id": "direct_w480_solve_brief_nosampling_omitthinking_512",
            "model": model,
            "asset": "w480q70",
            "task": "solve_brief",
            "sampling": None,
            "thinking": "omit",
            "max_tokens": 512,
            "timeout": 300,
            "use_image": True,
        },
        {
            "id": "direct_w480_solve_temp1_top095_off_2048",
            "model": model,
            "asset": "w480q70",
            "task": "solve",
            "sampling": {"temperature": 1, "top_p": 0.95},
            "thinking": "off",
            "max_tokens": 2048,
            "timeout": 300,
            "use_image": True,
        },
        {
            "id": "direct_w640_read_nosampling_off_512",
            "model": model,
            "asset": "w640q75",
            "task": "read",
            "sampling": None,
            "thinking": "off",
            "max_tokens": 512,
            "timeout": 240,
            "use_image": True,
        },
        {
            "id": "direct_w640_solve_temp1_top095_off_4096",
            "model": model,
            "asset": "w640q75",
            "task": "solve",
            "sampling": {"temperature": 1, "top_p": 0.95},
            "thinking": "off",
            "max_tokens": 4096,
            "timeout": 360,
            "use_image": True,
        },
        {
            "id": "direct_w480_read_nosampling_omitthinking_512",
            "model": model,
            "asset": "w480q70",
            "task": "read",
            "sampling": None,
            "thinking": "omit",
            "max_tokens": 512,
            "timeout": 240,
            "use_image": True,
        },
        {
            "id": "direct_w480_read_temp1_top095_omitthinking_512",
            "model": model,
            "asset": "w480q70",
            "task": "read",
            "sampling": {"temperature": 1, "top_p": 0.95},
            "thinking": "omit",
            "max_tokens": 512,
            "timeout": 240,
            "use_image": True,
        },
        {
            "id": "direct_q13_textonly_solve_nosampling_off_512",
            "model": model,
            "asset": "w480q70",
            "task": "solve",
            "sampling": None,
            "thinking": "off",
            "max_tokens": 512,
            "timeout": 180,
            "use_image": False,
        },
        {
            "id": "direct_q13_textonly_solve_nosampling_omitthinking_512",
            "model": model,
            "asset": "w480q70",
            "task": "solve",
            "sampling": None,
            "thinking": "omit",
            "max_tokens": 512,
            "timeout": 240,
            "use_image": False,
        },
        {
            "id": "direct_q13_textonly_solve_temp1_top095_off_512",
            "model": model,
            "asset": "w480q70",
            "task": "solve",
            "sampling": {"temperature": 1, "top_p": 0.95},
            "thinking": "off",
            "max_tokens": 512,
            "timeout": 180,
            "use_image": False,
        },
    ]
    if include_orig:
        cases.append(
            {
                "id": "direct_orig_read_nosampling_off_512",
                "model": model,
                "asset": "orig",
                "task": "read",
                "sampling": None,
                "thinking": "off",
                "max_tokens": 512,
                "timeout": 300,
                "use_image": True,
            }
        )
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="moonshotai/kimi-k2.6")
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--image_dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--prompts_json", type=Path, default=DEFAULT_PROMPTS)
    parser.add_argument("--include_orig", action="store_true")
    parser.add_argument(
        "--case_ids",
        default="",
        help="Comma-separated case ids to run. Empty means all default cases.",
    )
    parser.add_argument(
        "--timeout_override",
        type=float,
        default=0.0,
        help="If positive, override per-case request timeout in seconds.",
    )
    parser.add_argument(
        "--max_tokens_override",
        type=int,
        default=0,
        help="If positive, override per-case max_completion_tokens.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    full_prompt_text = load_prompt_text(args.prompts_json)
    read_prompt = (
        "The attached image is a chemistry reaction scheme from IChO 2025 Q1. "
        "If the image is visible, start with IMAGE_VISIBLE. Then transcribe the visible "
        "compound labels, molecular formulas, reagents, arrows, and any relation involving "
        "D, E, F, G, H. Do not solve the chemistry question. Keep the answer under 300 words."
    )
    solve_prompt = (
        "Start with FINAL ANSWER in one short paragraph. Then give only concise supporting steps. "
        "Do not write hidden-thought tags or long analysis. If the attached image is visible, "
        "mention IMAGE_VISIBLE once.\n\n"
        + full_prompt_text
    )
    solve_brief_prompt = (
        "This is IChO 2025 Q1.3. Use the attached reaction scheme. "
        "Question: Draw/describe the structures of compounds E and F. "
        "Stereochemistry is not required. Start with FINAL ANSWER, then at most 5 concise bullet points. "
        "If the image is visible, include IMAGE_VISIBLE."
    )

    assets: dict[str, Any] = {}
    for name in ("w480q70", "w640q75", "orig"):
        path = args.image_dir / f"{name}.jpg"
        if path.exists():
            assets[name] = {
                "path": str(path),
                "bytes_len": path.stat().st_size,
                "data_url_len": len(image_data_url(path)),
            }

    cases = build_cases(args.model, args.include_orig)
    if args.case_ids:
        wanted = {case_id.strip() for case_id in args.case_ids.split(",") if case_id.strip()}
        cases = [case for case in cases if case["id"] in wanted]
        missing = sorted(wanted - {case["id"] for case in cases})
        if missing:
            raise ValueError(f"Unknown case ids: {missing}")
    if args.timeout_override > 0:
        for case in cases:
            case["timeout"] = args.timeout_override
    if args.max_tokens_override > 0:
        for case in cases:
            case["max_tokens"] = args.max_tokens_override
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
        image_url = image_data_url(Path(assets[case["asset"]]["path"]))
        if case["task"] == "read":
            text = read_prompt
        elif case["task"] == "ok":
            text = "Return exactly: OK"
        elif case["task"] == "solve_brief":
            text = solve_brief_prompt
        else:
            text = solve_prompt
        user_content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        if case.get("use_image", True):
            user_content.append({"type": "image_url", "image_url": {"url": image_url}})
        payload: dict[str, Any] = {
            "model": case["model"],
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are solving or reading a released public olympiad benchmark. "
                        "Be concise. Do not write <think>."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "stream": False,
            "max_completion_tokens": case["max_tokens"],
        }
        if case["sampling"]:
            payload.update(case["sampling"])
        if case["thinking"] == "off":
            payload["thinking"] = {"type": "disabled"}

        started = time.time()
        record: dict[str, Any] = {
            **case,
            "asset_bytes": assets[case["asset"]]["bytes_len"],
            "asset_data_url_len": assets[case["asset"]]["data_url_len"],
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=case["timeout"])
            record["elapsed_s"] = round(time.time() - started, 3)
            record["http_status"] = resp.status_code
            raw_path = args.out_dir / f"{case['id']}.raw.json"
            record["raw_response_path"] = str(raw_path)
            try:
                data: Any = resp.json()
            except Exception:
                data = {"raw_text": resp.text[:5000]}
            raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            if resp.status_code >= 400:
                record["status"] = "http_error"
                record["error"] = resp.text[:1000]
                record["output"] = ""
            else:
                choice = (data.get("choices") or [{}])[0]
                message = choice.get("message") or {}
                content = message.get("content") or ""
                record["status"] = "ok"
                record["finish_reason"] = choice.get("finish_reason")
                record["output"] = content
                record["output_len"] = len(content)
                record["visible_marker"] = "IMAGE_VISIBLE" in content.upper()
                record["usage"] = data.get("usage")
                record["error"] = ""
        except Exception as exc:
            record["elapsed_s"] = round(time.time() - started, 3)
            record["status"] = "exception"
            record["error"] = repr(exc)
            record["traceback"] = traceback.format_exc()[-2000:]
            record["output"] = ""

        results.append(record)
        with jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_summary(args.out_dir, assets, results)
        print(
            json.dumps(
                {
                    "id": record["id"],
                    "status": record.get("status"),
                    "http": record.get("http_status"),
                    "finish": record.get("finish_reason"),
                    "out_len": record.get("output_len"),
                    "elapsed": record.get("elapsed_s"),
                    "err": (record.get("error") or "")[:160],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    write_summary(args.out_dir, assets, results)
    print(f"DONE {args.out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
