#!/usr/bin/env python3
"""LLM-assisted completeness audit for a page-range candidate benchmark."""

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
        env[key.strip().removeprefix("export").strip()] = value.strip().strip('"').strip("'")
    return env


def api_key_from_env() -> str:
    key = os.environ.get("QIHOO_API_KEY") or os.environ.get("QIHOO360_API_KEY") or os.environ.get("360_API_KEY")
    if not key:
        raise RuntimeError("Missing 360 API key in QIHOO_API_KEY/QIHOO360_API_KEY/360_API_KEY.")
    return key


def selected_pages(pages_json: dict[str, Any], page_numbers: list[int]) -> list[dict[str, Any]]:
    wanted = {int(page) for page in page_numbers}
    return [page for page in pages_json.get("pages", []) if int(page.get("page", -1)) in wanted]


def page_text_blob(pages: list[dict[str, Any]], label: str, max_chars: int) -> str:
    chunks = []
    for page in pages:
        text = str(page.get("text") or "").strip()
        chunks.append(f"[{label} page {page.get('page')}]\n{text}")
    blob = "\n\n".join(chunks)
    return blob[:max_chars]


def file_to_data_url(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0] or "image/png"
    return f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode('ascii')}"


def page_image_paths(pages_json_path: Path, pages: list[dict[str, Any]], include_page_renders: bool) -> list[Path]:
    base_dir = pages_json_path.parent
    paths: list[Path] = []
    for page in pages:
        if include_page_renders:
            render = page.get("page_render") or {}
            rel = render.get("file")
            if rel and (base_dir / rel).exists():
                paths.append(base_dir / rel)
        for image in page.get("images", []) or []:
            rel = image.get("file")
            if rel and (base_dir / rel).exists():
                paths.append(base_dir / rel)
    return paths


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", default="google/gemini-3.5-flash")
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--text_max_chars", type=int, default=90000)
    args = parser.parse_args()

    os.environ.update(load_env_file(args.env_file))
    config = load_json(args.config, {})
    question_pages_json_path = ROOT / config["source_files"]["question_pages_json"]
    solution_pages_json_path = ROOT / config["source_files"]["solution_pages_json"]
    question_pages = selected_pages(load_json(question_pages_json_path, {}), config.get("question_pages", []))
    solution_pages = selected_pages(load_json(solution_pages_json_path, {}), config.get("solution_pages", []))
    question_text = page_text_blob(question_pages, "question", args.text_max_chars)
    solution_text = page_text_blob(solution_pages, "solution", args.text_max_chars)
    image_paths = page_image_paths(
        question_pages_json_path,
        question_pages,
        include_page_renders=bool(config.get("include_question_page_renders", False)),
    )
    image_urls = [file_to_data_url(path) for path in image_paths]
    prompt = (
        "You are auditing a candidate Science Arena benchmark before open-source release. "
        "Check whether the benchmark is complete and fair for multimodal LLM evaluation. "
        "Return only valid JSON with keys: overall_status, total_points_reconciled, "
        "problems, missing_question_content, missing_images, missing_answers, missing_rubrics, "
        "static_evaluation_risks, required_fixes, ready_for_all_model_smoke. "
        "For every problem/subproblem you can identify, state whether question text, necessary figures, answer, "
        "and scoring details appear complete. Be strict; do not call it ready if any required diagram or rubric is absent.\n\n"
        + json.dumps(
            {
                "benchmark_id": config.get("benchmark_id"),
                "problem_id": config.get("problem_id"),
                "title": config.get("title"),
                "declared_total_points": config.get("total_points"),
                "question_pages": config.get("question_pages", []),
                "solution_pages": config.get("solution_pages", []),
                "image_count_sent": len(image_urls),
                "question_text": question_text,
                "solution_text": solution_text,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    client = UnifiedChatClient(api_key=api_key_from_env(), timeout=args.timeout, max_retries=2, retry_delay=5)
    response = client.chat(
        model=args.model,
        messages=[build_user_message(text=prompt, image_urls=image_urls, use_multimodal=True)],
        max_tokens=args.max_tokens,
        thinking_mode="auto",
        thinking_effort="auto",
    )
    text = response.get("content") or response.get("llm_output") or ""
    parsed = extract_json_object(text)
    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(args.config),
        "model": args.model,
        "sent_image_count": len(image_urls),
        "image_assets": [str(path.relative_to(ROOT)) for path in image_paths],
        "response_finish_reason": response.get("finish_reason"),
        "response_error": response.get("error"),
        "client_meta": response.get("client_meta", {}),
        "usage": response.get("usage", {}),
        "parsed_audit": parsed,
        "raw_response": text,
    }
    save_json(args.output_dir / "audit_response.json", response)
    save_json(args.output_dir / "audit.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
