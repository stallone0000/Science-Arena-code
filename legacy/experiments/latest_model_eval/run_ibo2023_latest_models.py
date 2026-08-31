#!/usr/bin/env python3
"""Generate and objectively score IBO 2023 answers through the 360 API.

The script is intentionally self-contained at the experiment layer. It reuses
the shared 360 chat client/model registry but writes only under
experiments/latest_model_eval/.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import html
import json
import os
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


ROOT = Path(__file__).resolve().parents[2]
ALLOWED_OUTPUT_ROOT = ROOT / "experiments/latest_model_eval"
DEFAULT_OUTPUT_ROOT = ALLOWED_OUTPUT_ROOT / "ibo2023_runs"
DEFAULT_ENV_FILE = Path("/home/jovyan/zhaoguangxiang-data-zzzc/zhaoguangxiang/.config/science_arena/360.env")
DEFAULT_ITEMS_PATH = ROOT / "data/science_arena/ibo2023/items_data.json"
DEFAULT_SCORING_KEY_PATH = ROOT / "data/science_arena/ibo2023/scoring_key.json"
DEFAULT_PRICES_PATH = ROOT / "experiments/judge_pipeline_v2/model_prices_360.json"

sys.path.insert(0, str(ROOT / "new-things/interleaved"))
sys.path.insert(0, str(ROOT / "code"))

from api_client_interleaved import (  # noqa: E402
    DEFAULT_API_URL,
    DEFAULT_HOST,
    UnifiedChatClient,
    build_user_message,
    resolve_multimodal_mode,
)
from model_registry import resolve_model_profile  # noqa: E402
from runtime_utils import get_api_key  # noqa: E402


DEFAULT_MODELS = [
    "cloudsway/gemini-3-flash-preview",
    "google/gemini-3.5-flash",
    "qwen/qwen3.7-max",
    "qwen/qwen3.6-plus",
    "bytedance/doubao-seed-2-0-pro",
    "deepseek/deepseek-v4-pro",
    "deepseek/deepseek-v4-flash",
    "moonshotai/kimi-k2.6",
    "anthropic/claude-opus-4.7",
    "minimax/MiniMax-M2.7",
    "z-ai/glm-5.1",
    "xiaomi/mimo-v2.5-pro",
]

ANSWER_CONTRACT = """

Final answer contract for automated scoring:
- Put the final answer sequence inside exactly one <boa>...</eoa> block.
- Inside the block, provide only the final entries in order, separated by spaces.
- For table-style questions, list only the filled cells in row order; omit row labels, column labels, and explanations.
- For true/false entries, use T or F. For multiple choice, use the option letters. For numbers, output plain numeric values without units.
"""


@dataclass(frozen=True)
class AnswerSpec:
    kind: str
    raw: str
    accepted: tuple[str, ...] = ()
    low: Optional[float] = None
    high: Optional[float] = None
    group: tuple["AnswerSpec", ...] = ()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    ensure_allowed_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def append_jsonl(path: Path, data: dict[str, Any]) -> None:
    ensure_allowed_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def ensure_allowed_output_path(path: Path) -> None:
    resolved = path.expanduser().resolve()
    allowed = ALLOWED_OUTPUT_ROOT.resolve()
    try:
        resolved.relative_to(allowed)
    except ValueError as exc:
        raise ValueError(f"Refusing to write outside {allowed}: {resolved}") from exc


def parse_csv(raw: str) -> list[str]:
    return [item.strip() for item in re.split(r"[\s,]+", raw or "") if item.strip()]


def parse_comma_list(raw: str) -> list[str]:
    return [item.strip() for item in (raw or "").split(",") if item.strip()]


def model_tag(model: str) -> str:
    tag = re.sub(r"[^A-Za-z0-9_.-]+", "__", model.strip())
    return tag.strip("_") or "model"


def quiz_index(item: dict[str, Any]) -> str:
    return f"{item['source']} {item['source_q_idx']}"


def load_env_file(path: Path) -> dict[str, str]:
    env: dict[str, str] = {}
    if not path or not path.exists():
        return env
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip().strip('"').strip("'")
        if key:
            env[key] = value
    return env


def apply_env_file(path: Path) -> None:
    for key, value in load_env_file(path).items():
        os.environ.setdefault(key, value)


def clean_text(value: str) -> str:
    text = html.unescape(str(value))
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("©", "C").replace("✘", "X").replace("×", "X")
    return text


def compact_answer_token(value: str) -> str:
    text = clean_text(value).strip()
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\([^)]*\)", "", text)
    text = text.strip().strip("`'\"“”‘’.,;:()[]{}")
    text = re.sub(r"\s+", "", text)
    upper = text.upper()
    if upper in {"TRUE", "T"}:
        return "T"
    if upper in {"FALSE", "F"}:
        return "F"
    return upper


def extract_numbers(value: str) -> list[float]:
    return [float(x) for x in re.findall(r"[-+]?\d+(?:\.\d+)?", clean_text(value))]


def make_spec(raw: str) -> AnswerSpec:
    token = compact_answer_token(raw)
    token = re.sub(r"(?i)BOTHARERIGHT", "", token)
    if "/" in token:
        accepted = tuple(part for part in (compact_answer_token(x) for x in token.split("/")) if part)
        return AnswerSpec(kind="alternatives", raw=raw, accepted=accepted)

    range_match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?)\s*[-–]\s*([-+]?\d+(?:\.\d+)?)", token)
    if range_match:
        low = float(range_match.group(1))
        high = float(range_match.group(2))
        return AnswerSpec(kind="range", raw=raw, low=min(low, high), high=max(low, high))

    return AnswerSpec(kind="exact", raw=raw, accepted=(token,))


def group_spec(tokens: Iterable[str], raw: str) -> AnswerSpec:
    return AnswerSpec(kind="group", raw=raw, group=tuple(make_spec(token) for token in tokens))


SPECIAL_SPECS: dict[str, list[AnswerSpec]] = {
    "2023_IBO_theory_1 8": [
        group_spec(["B", "A", "B"], "8.1 gram stain table"),
        group_spec(["C", "B"], "8.2 monophyly table"),
        make_spec("T"),
        make_spec("F"),
        make_spec("F"),
    ],
    "2023_IBO_theory_1 24": [
        group_spec(["0", "100", "50", "50", "50"], "IBD1 fractions"),
        group_spec(["100", "0", "25", "0", "0"], "IBD2 fractions"),
        group_spec(["F", "F", "T", "T", "T"], "relationship true/false table"),
        make_spec("T"),
        make_spec("F"),
        make_spec("T"),
    ],
}


TOKEN_RE = re.compile(
    r"(?i)\btrue\b|\bfalse\b|[-+]?\d+(?:\.\d+)?(?:\s*(?:/|-|–)\s*[-+]?\d+(?:\.\d+)?)*|\b[A-Za-z]\b"
)


def strip_reference_label(segment: str) -> str:
    text = segment.strip()
    # Handles answer-key fragments such as "16.2 63.0-63.5" without
    # removing legitimate decimal answers from non-fragment keys.
    return re.sub(
        r"^\s*(?:Q\.?\s*)?\d+(?:\.\d+)+\s*(?:[:.)-]\s*)?\s+",
        "",
        text,
        flags=re.IGNORECASE,
    )


def answer_tokens(text: str) -> list[str]:
    cleaned = clean_text(text)
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = cleaned.replace("$$$", " ")
    return [compact_answer_token(match.group(0)) for match in TOKEN_RE.finditer(cleaned)]


def generic_specs(answer: str, expected_count: int) -> list[AnswerSpec]:
    if "$$$" in answer:
        parts = []
        for segment in answer.split("$$$"):
            tokens = answer_tokens(strip_reference_label(segment))
            if tokens:
                parts.extend(tokens)
    else:
        parts = answer_tokens(answer)
    if len(parts) > expected_count:
        parts = parts[:expected_count]
    return [make_spec(part) for part in parts]


def expected_specs(item: dict[str, Any], sub_scores: list[float]) -> list[AnswerSpec]:
    qid = quiz_index(item)
    if qid in SPECIAL_SPECS:
        return SPECIAL_SPECS[qid]
    return generic_specs(str(item.get("answer", "")), len(sub_scores))


def is_numeric_spec(spec: AnswerSpec) -> bool:
    if spec.kind == "range":
        return True
    if spec.kind == "group":
        return all(is_numeric_spec(child) for child in spec.group)
    if spec.kind in {"exact", "alternatives"}:
        return all(re.fullmatch(r"[-+]?\d+(?:\.\d+)?", value or "") for value in spec.accepted)
    return False


def filter_label_tokens(tokens: list[str], item_index: int, specs: list[AnswerSpec]) -> list[str]:
    if not tokens:
        return tokens
    filtered: list[str] = []
    prefix = str(item_index)
    for token in tokens:
        if re.fullmatch(rf"{re.escape(prefix)}\.\d+(?:\.\d+)*", token, flags=re.IGNORECASE):
            continue
        if re.fullmatch(rf"Q?{re.escape(prefix)}\.\d+(?:\.\d+)*", token, flags=re.IGNORECASE):
            continue
        filtered.append(token)

    flat_expected = flatten_specs(specs)
    if len(filtered) > len(flat_expected) and not any(is_numeric_spec(spec) for spec in flat_expected):
        maybe_labels = {str(i) for i in range(1, len(flat_expected) + 1)}
        filtered = [token for token in filtered if token not in maybe_labels]
    return filtered


def flatten_specs(specs: list[AnswerSpec]) -> list[AnswerSpec]:
    flat: list[AnswerSpec] = []
    for spec in specs:
        if spec.kind == "group":
            flat.extend(spec.group)
        else:
            flat.append(spec)
    return flat


def token_matches_spec(predicted: str, spec: AnswerSpec) -> bool:
    token = compact_answer_token(predicted)
    if spec.kind == "range":
        numbers = extract_numbers(token)
        return any(spec.low is not None and spec.high is not None and spec.low <= number <= spec.high for number in numbers)

    if spec.kind == "alternatives":
        return any(token_matches_spec(token, AnswerSpec(kind="exact", raw=alt, accepted=(alt,))) for alt in spec.accepted)

    if spec.kind == "exact":
        expected = spec.accepted[0] if spec.accepted else ""
        pred_numbers = extract_numbers(token)
        exp_numbers = extract_numbers(expected)
        if pred_numbers and exp_numbers and re.fullmatch(r"[-+]?\d+(?:\.\d+)?", expected):
            return any(abs(number - exp_numbers[0]) <= 1e-9 for number in pred_numbers)
        return token == expected

    raise ValueError(f"Unsupported non-leaf spec kind: {spec.kind}")


def extract_boa_answer(text: str) -> tuple[Optional[str], str]:
    matches = re.findall(r"<boa>(.*?)(?:</?eoa>|</boa>)", text or "", flags=re.IGNORECASE | re.DOTALL)
    if matches:
        return matches[-1].strip(), "tagged"
    return None, "missing_tags"


def score_answer(
    *,
    item: dict[str, Any],
    sub_scores: list[float],
    model_output: str,
    allow_tagless_parse: bool = False,
) -> dict[str, Any]:
    specs = expected_specs(item, sub_scores)
    qid = quiz_index(item)
    if len(specs) != len(sub_scores):
        return {
            "quiz_index": qid,
            "score": 0.0,
            "max_score": float(sum(sub_scores)),
            "sub_results": [0] * len(sub_scores),
            "status": f"invalid_expected_spec_count:{len(specs)}",
            "extracted_answer": "",
            "tokens": [],
        }

    extracted, extraction_status = extract_boa_answer(model_output)
    if extracted is None:
        if not allow_tagless_parse:
            return {
                "quiz_index": qid,
                "score": 0.0,
                "max_score": float(sum(sub_scores)),
                "sub_results": [0] * len(sub_scores),
                "status": extraction_status,
                "extracted_answer": "",
                "tokens": [],
            }
        extracted = fallback_tagless_answer(model_output)
        extraction_status = "tagless_fallback"

    tokens = filter_label_tokens(answer_tokens(extracted), int(item["source_q_idx"]), specs)
    cursor = 0
    sub_results: list[int] = []
    score = 0.0
    for spec, points in zip(specs, sub_scores):
        if spec.kind == "group":
            group_tokens = tokens[cursor : cursor + len(spec.group)]
            cursor += len(spec.group)
            ok = len(group_tokens) == len(spec.group) and all(
                token_matches_spec(token, child) for token, child in zip(group_tokens, spec.group)
            )
        else:
            token = tokens[cursor] if cursor < len(tokens) else ""
            cursor += 1
            ok = bool(token and token_matches_spec(token, spec))
        sub_results.append(1 if ok else 0)
        if ok:
            score += float(points)

    status = extraction_status
    if cursor < len(tokens):
        status = f"{status}_extra_tokens"
    elif cursor > len(tokens):
        status = f"{status}_missing_tokens"

    return {
        "quiz_index": qid,
        "score": float(score),
        "max_score": float(sum(sub_scores)),
        "sub_results": sub_results,
        "status": status,
        "extracted_answer": extracted,
        "tokens": tokens,
    }


def fallback_tagless_answer(model_output: str) -> str:
    text = (model_output or "").strip()
    matches = list(re.finditer(r"(?i)(final answer|answer)\s*[:：]", text))
    if matches:
        return text[matches[-1].end() :].strip()
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text[-1000:]


def validate_scoring(items: list[dict[str, Any]], scoring_map: dict[str, list[float]]) -> list[str]:
    warnings: list[str] = []
    for item in items:
        qid = quiz_index(item)
        sub_scores = scoring_map.get(qid, [])
        specs = expected_specs(item, sub_scores)
        if len(specs) != len(sub_scores):
            warnings.append(f"{qid}: expected {len(sub_scores)} sub-scores but built {len(specs)} specs")
    return warnings


def item_image_urls(item: dict[str, Any]) -> list[str]:
    urls = []
    for image in item.get("image", []) or []:
        if isinstance(image, dict) and image.get("url"):
            urls.append(str(image["url"]).strip())
    return [url for url in urls if url]


def build_prompt(item: dict[str, Any], *, answer_contract: str) -> str:
    blocks = [str(item.get("prompt", "")).strip(), str(item.get("question", "")).strip()]
    if answer_contract:
        blocks.append(answer_contract.strip())
    return "\n\n".join(block for block in blocks if block)


def usage_cost(usage: dict[str, Any], model: str, prices: dict[str, Any]) -> dict[str, float]:
    prompt_tokens = float(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
    completion_tokens = float(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
    total_tokens = float(usage.get("total_tokens") or prompt_tokens + completion_tokens)
    price = prices.get(model) or prices.get(resolve_model_profile(model).profile.preferred_model) or {}
    input_price = float(price.get("input_price_cny_per_1m", 0) or 0)
    output_price = float(price.get("output_price_cny_per_1m", 0) or 0)
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_cny": prompt_tokens / 1_000_000 * input_price + completion_tokens / 1_000_000 * output_price,
    }


def scoring_map_from_key(rows: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {
        str(row["quiz_index"]): [float(value) for value in row.get("sub_scores", [])]
        for row in rows
        if isinstance(row, dict) and row.get("quiz_index")
    }


def selector_matches(item: dict[str, Any], selectors: set[str]) -> bool:
    if not selectors:
        return True
    qid = quiz_index(item)
    normalized = qid.lower()
    theory = str(item.get("source", "")).replace("2023_IBO_", "").lower()
    source_idx = str(item.get("source_q_idx", "")).strip()
    aliases = {
        qid,
        normalized,
        source_idx,
        f"{theory}:{source_idx}",
        f"{theory.replace('theory_', 't')}:{source_idx}",
        f"{item.get('source')}:{source_idx}",
    }
    return any(selector.lower() in aliases for selector in selectors)


def select_items(items: list[dict[str, Any]], items_arg: str, limit: int) -> list[dict[str, Any]]:
    selectors = set(parse_comma_list(items_arg))
    selected = [item for item in items if selector_matches(item, selectors)]
    if limit and limit > 0:
        selected = selected[:limit]
    return selected


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    score = sum(float(row.get("score", 0) or 0) for row in rows)
    max_score = sum(float(row.get("max_score", 0) or 0) for row in rows)
    prompt_tokens = sum(float(row.get("usage_cost", {}).get("prompt_tokens", 0) or 0) for row in rows)
    completion_tokens = sum(float(row.get("usage_cost", {}).get("completion_tokens", 0) or 0) for row in rows)
    total_tokens = sum(float(row.get("usage_cost", {}).get("total_tokens", 0) or 0) for row in rows)
    cost_cny = sum(float(row.get("usage_cost", {}).get("cost_cny", 0) or 0) for row in rows)
    errors = sum(1 for row in rows if row.get("error"))
    missing_tags = sum(1 for row in rows if str(row.get("score_status", "")).startswith("missing_tags"))
    return {
        "items": len(rows),
        "score": score,
        "max_score": max_score,
        "score_fraction": score / max_score if max_score else 0.0,
        "errors": errors,
        "missing_tags": missing_tags,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cost_cny": cost_cny,
    }


def api_call_log_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "logged_at": datetime.now().isoformat(timespec="seconds"),
        "quiz_index": row.get("quiz_index", ""),
        "model": row.get("model", ""),
        "score": row.get("score", 0.0),
        "max_score": row.get("max_score", 0.0),
        "score_status": row.get("score_status", ""),
        "usage": row.get("usage", {}),
        "usage_cost": row.get("usage_cost", {}),
        "error": row.get("error", ""),
        "finish_reason": row.get("finish_reason", ""),
        "elapsed_seconds": row.get("elapsed_seconds", 0.0),
        "use_multimodal": row.get("use_multimodal", False),
        "image_count": len(row.get("sent_image_urls") or []),
        "client_meta": row.get("client_meta", {}),
    }


def run_one_item(
    *,
    client: UnifiedChatClient,
    args: argparse.Namespace,
    model: str,
    item: dict[str, Any],
    sub_scores: list[float],
    prices: dict[str, Any],
    use_multimodal: bool,
) -> dict[str, Any]:
    qid = quiz_index(item)
    prompt_text = build_prompt(item, answer_contract=ANSWER_CONTRACT if args.append_answer_contract else "")
    image_urls = item_image_urls(item)
    sent_image_urls = image_urls if use_multimodal else []
    message = build_user_message(text=prompt_text, image_urls=sent_image_urls, use_multimodal=use_multimodal)
    started = time.time()
    response = client.chat(
        model=model,
        messages=[message],
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        thinking_mode=args.thinking_mode,
        thinking_effort=args.thinking_effort,
        thinking_budget=args.thinking_budget,
        reasoning_effort=(args.reasoning_effort or None),
        print_curl=args.print_curl,
    )
    elapsed = time.time() - started
    model_output = response.get("llm_output") or response.get("content") or ""
    score_row = score_answer(
        item=item,
        sub_scores=sub_scores,
        model_output=model_output,
        allow_tagless_parse=args.allow_tagless_parse,
    )
    usage = response.get("usage") or {}
    cost = usage_cost(usage, model, prices)
    return {
        "quiz_index": qid,
        "source": item.get("source", ""),
        "source_q_idx": item.get("source_q_idx"),
        "model": model,
        "model_output": model_output,
        "correct_answer": item.get("answer", ""),
        "score": score_row["score"],
        "max_score": score_row["max_score"],
        "sub_results": score_row["sub_results"],
        "sub_scores_config": sub_scores,
        "score_status": score_row["status"],
        "extracted_answer": score_row["extracted_answer"],
        "extracted_tokens": score_row["tokens"],
        "usage": usage,
        "usage_cost": cost,
        "error": response.get("error", ""),
        "finish_reason": response.get("finish_reason", ""),
        "client_meta": response.get("client_meta", {}),
        "elapsed_seconds": elapsed,
        "image_urls": image_urls,
        "sent_image_urls": sent_image_urls,
        "use_multimodal": use_multimodal,
    }


def rescore_existing_row(
    *,
    row: dict[str, Any],
    item: dict[str, Any],
    sub_scores: list[float],
    prices: dict[str, Any],
    model: str,
    allow_tagless_parse: bool,
) -> dict[str, Any]:
    updated = dict(row)
    score_row = score_answer(
        item=item,
        sub_scores=sub_scores,
        model_output=str(updated.get("model_output", "")),
        allow_tagless_parse=allow_tagless_parse,
    )
    updated["score"] = score_row["score"]
    updated["max_score"] = score_row["max_score"]
    updated["sub_results"] = score_row["sub_results"]
    updated["sub_scores_config"] = sub_scores
    updated["score_status"] = score_row["status"]
    updated["extracted_answer"] = score_row["extracted_answer"]
    updated["extracted_tokens"] = score_row["tokens"]
    updated["usage_cost"] = usage_cost(updated.get("usage") or {}, model, prices)
    return updated


def run_model(
    *,
    args: argparse.Namespace,
    model: str,
    items: list[dict[str, Any]],
    scoring_map: dict[str, list[float]],
    prices: dict[str, Any],
    run_dir: Path,
    api_key: str,
) -> dict[str, Any]:
    tag = model_tag(model)
    model_dir = run_dir / tag
    ensure_allowed_output_path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    use_multimodal = resolve_multimodal_mode(model=model, mode=args.multimodal_mode, explicit_override=None)
    resolved_profile = resolve_model_profile(model)
    client = UnifiedChatClient(
        api_key=api_key,
        backend=args.api_backend,
        api_url=args.api_url,
        host=args.api_host,
        max_retries=args.max_try,
        retry_delay=args.retry_delay,
        timeout=args.timeout,
        model_tag=tag,
        verbose=args.verbose,
    )

    prompt_rows = []
    for item in items:
        prompt_rows.append(
            {
                "quiz_index": quiz_index(item),
                "source": item.get("source", ""),
                "source_q_idx": item.get("source_q_idx"),
                "prompt_text": build_prompt(
                    item,
                    answer_contract=ANSWER_CONTRACT if args.append_answer_contract else "",
                ),
                "image_urls": item_image_urls(item),
                "use_multimodal": use_multimodal,
                "model_profile": resolved_profile.to_summary_dict(),
            }
        )
    save_json(model_dir / "prompts.json", prompt_rows)

    output_path = model_dir / "solutions.json"
    api_calls_path = model_dir / "api_calls.jsonl"
    existing_rows: dict[str, dict[str, Any]] = {}
    if output_path.exists() and not args.force:
        items_by_qid = {quiz_index(item): item for item in items}
        for row in load_json(output_path):
            if isinstance(row, dict) and row.get("quiz_index"):
                qid = str(row["quiz_index"])
                if args.retry_errors and (row.get("error") or row.get("score_status") in {"api_error", "api_exception"}):
                    continue
                item = items_by_qid.get(qid)
                if item is not None and qid in scoring_map:
                    row = rescore_existing_row(
                        row=row,
                        item=item,
                        sub_scores=scoring_map[qid],
                        prices=prices,
                        model=model,
                        allow_tagless_parse=args.allow_tagless_parse,
                    )
                existing_rows[qid] = row

    rows: list[dict[str, Any]] = []
    rows_by_qid: dict[str, dict[str, Any]] = dict(existing_rows)
    pending = [item for item in items if args.force or quiz_index(item) not in existing_rows]
    lock = threading.Lock()

    def worker(item: dict[str, Any]) -> dict[str, Any]:
        qid = quiz_index(item)
        sub_scores = scoring_map[qid]
        return run_one_item(
            client=client,
            args=args,
            model=model,
            item=item,
            sub_scores=sub_scores,
            prices=prices,
            use_multimodal=use_multimodal,
        )

    if pending:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.workers, 1)) as executor:
            futures = {executor.submit(worker, item): quiz_index(item) for item in pending}
            for future in concurrent.futures.as_completed(futures):
                qid = futures[future]
                try:
                    row = future.result()
                except Exception as exc:
                    item = next(item for item in pending if quiz_index(item) == qid)
                    sub_scores = scoring_map[qid]
                    row = {
                        "quiz_index": qid,
                        "source": item.get("source", ""),
                        "source_q_idx": item.get("source_q_idx"),
                        "model": model,
                        "model_output": "",
                        "correct_answer": item.get("answer", ""),
                        "score": 0.0,
                        "max_score": float(sum(sub_scores)),
                        "sub_results": [0] * len(sub_scores),
                        "sub_scores_config": sub_scores,
                        "score_status": "api_exception",
                        "extracted_answer": "",
                        "extracted_tokens": [],
                        "usage": {},
                        "usage_cost": usage_cost({}, model, prices),
                        "error": f"{type(exc).__name__}: {exc}",
                        "finish_reason": "",
                        "client_meta": {},
                        "elapsed_seconds": 0.0,
                        "image_urls": item_image_urls(item),
                        "sent_image_urls": item_image_urls(item) if use_multimodal else [],
                        "use_multimodal": use_multimodal,
                    }
                with lock:
                    rows_by_qid[qid] = row
                    append_jsonl(api_calls_path, api_call_log_record(row))
                    ordered = [rows_by_qid[quiz_index(item)] for item in items if quiz_index(item) in rows_by_qid]
                    save_json(output_path, ordered)
                print(f"[IBO] {model} {qid}: score={row.get('score')}/{row.get('max_score')} status={row.get('score_status')}")

    rows = [rows_by_qid[quiz_index(item)] for item in items if quiz_index(item) in rows_by_qid]
    save_json(output_path, rows)
    save_json(model_dir / "scores.json", rows)
    summary = {
        "model": model,
        "model_tag": tag,
        "model_profile": resolved_profile.to_summary_dict(),
        "run_dir": str(model_dir),
        "solutions_path": str(output_path),
        "use_multimodal": use_multimodal,
        "selected_items": len(items),
        "generated_items": len(pending),
        "reused_items": len(existing_rows),
        **summarize_rows(rows),
    }
    save_json(model_dir / "summary.json", summary)
    return summary


def dry_run_report(
    *,
    args: argparse.Namespace,
    items: list[dict[str, Any]],
    scoring_map: dict[str, list[float]],
    warnings: list[str],
) -> dict[str, Any]:
    models = parse_csv(args.models)
    return {
        "dry_run": True,
        "models": models,
        "items": [
            {
                "quiz_index": quiz_index(item),
                "source": item.get("source", ""),
                "source_q_idx": item.get("source_q_idx"),
                "sub_scores": scoring_map.get(quiz_index(item), []),
                "image_count": len(item_image_urls(item)),
            }
            for item in items
        ],
        "selected_items": len(items),
        "selected_max_score": sum(sum(scoring_map.get(quiz_index(item), [])) for item in items),
        "scoring_warnings": warnings,
        "output_root": str(args.output_root),
        "will_append_answer_contract": args.append_answer_contract,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(DEFAULT_MODELS), help="Comma/space-separated provider-qualified model names.")
    parser.add_argument("--items", default="", help="Comma-separated quiz selectors, e.g. 'theory_1:1,2023_IBO_theory_2 5'.")
    parser.add_argument("--limit", type=int, default=0, help="Limit selected items after --items filtering.")
    parser.add_argument("--items_path", type=Path, default=DEFAULT_ITEMS_PATH)
    parser.add_argument("--scoring_key_path", type=Path, default=DEFAULT_SCORING_KEY_PATH)
    parser.add_argument("--prices_path", type=Path, default=DEFAULT_PRICES_PATH)
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run_name", default="", help="Output run name. Defaults to ibo2023_<timestamp>.")
    parser.add_argument("--env_file", type=Path, default=DEFAULT_ENV_FILE, help="Optional external env file; never copied to outputs.")
    parser.add_argument("--dry_run", action="store_true", help="Validate data/scoring selection without API calls.")
    parser.add_argument("--force", action="store_true", help="Regenerate items even when solutions.json already contains them.")
    parser.add_argument("--retry_errors", action="store_true", help="When resuming, regenerate rows whose existing solution has an API/error status.")
    parser.add_argument("--workers", type=int, default=1, help="Per-model API concurrency.")
    parser.add_argument("--model_workers", type=int, default=1, help="Parallel models to evaluate.")

    parser.add_argument("--api_backend", choices=["requests", "openai"], default="requests")
    parser.add_argument("--api_url", default=DEFAULT_API_URL)
    parser.add_argument("--api_host", default=DEFAULT_HOST)
    parser.add_argument("--max_tokens", type=int, default=65536)
    parser.add_argument("--max_try", type=int, default=3)
    parser.add_argument("--retry_delay", type=float, default=5)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--multimodal_mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--thinking_mode", choices=["auto", "on", "off"], default="auto")
    parser.add_argument(
        "--thinking_effort",
        choices=["auto", "none", "minimal", "low", "medium", "high", "xhigh", "max"],
        default="auto",
    )
    parser.add_argument("--thinking_budget", type=int, default=None)
    parser.add_argument("--reasoning_effort", default="")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--append_answer_contract", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--allow_tagless_parse", action="store_true")
    parser.add_argument("--print_curl", action="store_true", help="Print redacted curl commands.")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    args.output_root = args.output_root.expanduser().resolve()
    ensure_allowed_output_path(args.output_root)

    items = load_json(args.items_path)
    scoring_rows = load_json(args.scoring_key_path)
    scoring_map = scoring_map_from_key(scoring_rows)
    selected_items = select_items(items, args.items, args.limit)
    missing_scoring = [quiz_index(item) for item in selected_items if quiz_index(item) not in scoring_map]
    if missing_scoring:
        raise RuntimeError(f"Missing scoring key rows: {missing_scoring[:5]}")

    warnings = validate_scoring(selected_items, scoring_map)
    if args.dry_run:
        report = dry_run_report(args=args, items=selected_items, scoring_map=scoring_map, warnings=warnings)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    apply_env_file(args.env_file)
    api_key = get_api_key()
    prices = load_json(args.prices_path) if args.prices_path.exists() else {}

    run_name = args.run_name or f"ibo2023_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir = args.output_root / run_name
    ensure_allowed_output_path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "run_dir": str(run_dir),
        "models": parse_csv(args.models),
        "items_path": str(args.items_path),
        "scoring_key_path": str(args.scoring_key_path),
        "prices_path": str(args.prices_path),
        "selected_items": [quiz_index(item) for item in selected_items],
        "selected_max_score": sum(sum(scoring_map[quiz_index(item)]) for item in selected_items),
        "args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key not in {"env_file"}
        },
        "env_file": str(args.env_file),
        "scoring_warnings": warnings,
    }
    save_json(run_dir / "config.json", config)

    models = parse_csv(args.models)
    model_summaries: list[dict[str, Any]] = []
    if args.model_workers <= 1:
        for model in models:
            summary = run_model(
                args=args,
                model=model,
                items=selected_items,
                scoring_map=scoring_map,
                prices=prices,
                run_dir=run_dir,
                api_key=api_key,
            )
            model_summaries.append(summary)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(args.model_workers, 1)) as executor:
            futures = {
                executor.submit(
                    run_model,
                    args=args,
                    model=model,
                    items=selected_items,
                    scoring_map=scoring_map,
                    prices=prices,
                    run_dir=run_dir,
                    api_key=api_key,
                ): model
                for model in models
            }
            for future in concurrent.futures.as_completed(futures):
                model = futures[future]
                try:
                    summary = future.result()
                except Exception as exc:
                    summary = {
                        "model": model,
                        "model_tag": model_tag(model),
                        "error": f"{type(exc).__name__}: {exc}",
                        "items": 0,
                        "score": 0.0,
                        "max_score": config["selected_max_score"],
                        "errors": len(selected_items),
                        "prompt_tokens": 0.0,
                        "completion_tokens": 0.0,
                        "total_tokens": 0.0,
                        "cost_cny": 0.0,
                    }
                model_summaries.append(summary)
                save_json(
                    run_dir / "summary_partial.json",
                    {
                        "run_dir": str(run_dir),
                        "created_at": config["created_at"],
                        "selected_items": len(selected_items),
                        "selected_max_score": config["selected_max_score"],
                        "models": model_summaries,
                    },
                )
                print(f"[IBO-MODEL] {model}: score={summary.get('score')}/{summary.get('max_score')}")

    aggregate = {
        "run_dir": str(run_dir),
        "created_at": config["created_at"],
        "selected_items": len(selected_items),
        "selected_max_score": config["selected_max_score"],
        "models": model_summaries,
        "totals": {
            "cost_cny": sum(float(row.get("cost_cny", 0) or 0) for row in model_summaries),
            "prompt_tokens": sum(float(row.get("prompt_tokens", 0) or 0) for row in model_summaries),
            "completion_tokens": sum(float(row.get("completion_tokens", 0) or 0) for row in model_summaries),
            "total_tokens": sum(float(row.get("total_tokens", 0) or 0) for row in model_summaries),
        },
    }
    save_json(run_dir / "summary.json", aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
