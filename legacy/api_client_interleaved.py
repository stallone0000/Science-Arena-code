# -*- coding: utf-8 -*-
"""Unified API client for ScienceArena interleaved pipelines.

Features:
- Default backend: openai SDK (compatible with OpenAI-style gateways)
- Fallback backend: requests
- Curl command builder for quick debugging
- Unified response shape with `llm_output` preserved
- Model-aware thinking controls (on/off/effort/budget) with safe fallback
"""

from __future__ import annotations

import base64
import copy
import json
import mimetypes
import os
import random
import re
import shlex
import time
import importlib
from typing import Any, Dict, List, Optional

from model_registry import (
    ResolvedModelProfile,
    canonical_model_name,
    normalize_model_name,
    resolve_model_profile,
    resolve_thinking_capability,
)

requests = importlib.import_module("requests")

DEFAULT_API_URL = "https://api.360.cn/v1/chat/completions"
FALLBACK_API_URL = "http://api.zhinao.qihoo.net/v1/chat/completions"
DEFAULT_HOST = "api.360.cn"

PROXY_ENV_VARS = (
    "http_proxy",
    "https_proxy",
    "all_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
)


for _proxy_var in PROXY_ENV_VARS:
    os.environ.pop(_proxy_var, None)


SESSION = requests.Session()
SESSION.trust_env = False


def ensure_api_key(api_key: Optional[str] = None) -> str:
    key = api_key or os.environ.get("API_KEY") or os.environ.get("QIHOO_API_KEY")
    if not key:
        raise ValueError(
            "Missing API key. Please pass --api_key or set API_KEY / QIHOO_API_KEY."
        )
    return key


def normalize_openai_base_url(api_url: str) -> str:
    url = (api_url or "").strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url[: -len("/chat/completions")]
    return url


def build_api_url_candidates(api_url: str) -> List[str]:
    primary = (api_url or "").strip()
    if not primary:
        primary = DEFAULT_API_URL

    candidates = [primary]
    if primary == DEFAULT_API_URL:
        candidates.append(FALLBACK_API_URL)
    elif primary == FALLBACK_API_URL:
        candidates.append(DEFAULT_API_URL)
    return candidates


def normalize_max_tokens(model: str, requested_max_tokens: int) -> int:
    model_l = model.lower()
    max_tokens = int(requested_max_tokens)
    if "claude-opus-4.1" in model_l:
        return min(max_tokens, 31999)
    if "claude-opus-4.5" in model_l:
        return min(max_tokens, 64000)
    return max_tokens


THINKING_MODE_VALUES = {"auto", "on", "off"}
THINKING_EFFORT_VALUES = {
    "auto",
    "max",
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
}

THINKING_PAYLOAD_KEYS = (
    "thinking",
    "effort",
    "reasoning_effort",
    "thinking_budget",
    "thinking_level",
    "enable_thinking",
    "reasoning_split",
)

def normalize_thinking_mode(value: Optional[str]) -> str:
    mode = (value or "auto").strip().lower()
    if mode not in THINKING_MODE_VALUES:
        raise ValueError(f"Unsupported thinking_mode: {value}")
    return mode


def normalize_thinking_effort(value: Optional[str]) -> str:
    effort = (value or "auto").strip().lower()
    if effort not in THINKING_EFFORT_VALUES:
        raise ValueError(f"Unsupported thinking_effort: {value}")
    return effort


def extract_thinking_fields(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: copy.deepcopy(payload[key])
        for key in THINKING_PAYLOAD_KEYS
        if key in payload
    }


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        result.append(item)
    return result


def compare_thinking_fields(
    derived_fields: Dict[str, Any],
    final_fields: Dict[str, Any],
) -> Optional[bool]:
    if not derived_fields:
        return None
    return final_fields == derived_fields


class ThinkingSemanticsError(ValueError):
    def __init__(
        self,
        message: str,
        *,
        supported_thinking_controls: Dict[str, Any],
        recommended_benchmark_thinking_request: Dict[str, Any],
        normalized_provider_thinking_request: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.supported_thinking_controls = supported_thinking_controls
        self.recommended_benchmark_thinking_request = recommended_benchmark_thinking_request
        self.normalized_provider_thinking_request = normalized_provider_thinking_request or {}


def build_supported_thinking_controls(
    resolved_profile: ResolvedModelProfile,
) -> Dict[str, Any]:
    capability = resolve_thinking_capability(resolved_profile)
    return {
        "family": capability.family,
        "provider_request_fields": list(capability.provider_request_fields),
        "compat_request_fields": list(capability.compat_request_fields),
        "off_support": capability.off_support,
        "supported_modes": list(capability.supported_modes),
        "supported_efforts": list(capability.supported_efforts),
        "benchmark_mode": capability.benchmark_mode,
        "benchmark_effort": capability.benchmark_effort,
        "benchmark_notes": capability.benchmark_notes,
        "notes": capability.notes,
    }


def _family_supports_thinking_budget(
    capability: Any,
    canonical_name: str,
) -> bool:
    if capability.family == "claude_budget_legacy":
        return True
    if capability.family == "gemini" and "2.5" in canonical_name:
        return True
    if capability.family == "qwen" and capability.current_client_mapping != "likely_wrong":
        return True
    return False


def _effective_claude_budget(model: str, max_tokens: int, requested_budget: Optional[int], effort: str) -> int:
    normalized_limit = normalize_max_tokens(model, max_tokens)
    if requested_budget is not None:
        base_budget = int(requested_budget)
    else:
        base_budget = _claude_budget_for_effort(effort)
    # Leave some room for visible answer tokens.
    upper_bound = max(normalized_limit - 1024, 1024)
    return max(min(base_budget, upper_bound), 1024)


def _effective_gemini_budget(model_l: str, requested_budget: Optional[int], effort: str) -> int:
    if requested_budget is not None:
        return int(requested_budget)
    budget = _gemini_budget_for_effort(model_l, effort)
    if budget is None:
        default_budget = _gemini_default_budget(model_l)
        if default_budget is None:
            raise ValueError(f"No Gemini thinking budget mapping for model: {model_l}")
        budget = default_budget
    return int(budget)


def _effective_qwen_budget(requested_budget: Optional[int], effort: str) -> Optional[int]:
    if requested_budget is not None:
        return int(requested_budget)
    return _qwen_budget_for_effort(effort)


def _recommended_benchmark_request(
    resolved_profile: ResolvedModelProfile,
) -> Dict[str, Any]:
    capability = resolve_thinking_capability(resolved_profile)
    return {
        "thinking_mode": capability.benchmark_mode,
        "thinking_effort": capability.benchmark_effort,
        "notes": capability.benchmark_notes,
    }


def _raise_thinking_error(
    message: str,
    *,
    resolved_profile: ResolvedModelProfile,
    normalized_provider_request: Optional[Dict[str, Any]] = None,
) -> None:
    raise ThinkingSemanticsError(
        message,
        supported_thinking_controls=build_supported_thinking_controls(resolved_profile),
        recommended_benchmark_thinking_request=_recommended_benchmark_request(resolved_profile),
        normalized_provider_thinking_request=normalized_provider_request,
    )


def resolve_thinking_request_semantics(
    *,
    model: str,
    max_tokens: int,
    reasoning_effort: Optional[str] = None,
    thinking_mode: str = "auto",
    thinking_effort: str = "auto",
    thinking_budget: Optional[int] = None,
    resolved_profile: Optional[ResolvedModelProfile] = None,
) -> Dict[str, Any]:
    resolved_profile = resolved_profile or resolve_model_profile(model)
    capability = resolve_thinking_capability(resolved_profile)
    model_l = resolved_profile.canonical_name

    mode_l = normalize_thinking_mode(thinking_mode)
    explicit_effort = normalize_thinking_effort(thinking_effort)
    legacy_effort = normalize_thinking_effort(reasoning_effort) if reasoning_effort else "auto"
    if explicit_effort == "auto" and legacy_effort != "auto":
        explicit_effort = legacy_effort

    supported_controls = build_supported_thinking_controls(resolved_profile)
    recommended_request = _recommended_benchmark_request(resolved_profile)

    semantic_mode = mode_l
    if semantic_mode == "auto":
        semantic_mode = "on" if explicit_effort != "auto" else capability.benchmark_mode

    if semantic_mode not in capability.supported_modes:
        _raise_thinking_error(
            (
                f"Model `{model}` does not support thinking_mode={semantic_mode!r}. "
                f"Supported modes: {list(capability.supported_modes)}. "
                f"Benchmark strongest: {recommended_request}."
            ),
            resolved_profile=resolved_profile,
        )

    if thinking_budget is not None and not _family_supports_thinking_budget(capability, model_l):
        _raise_thinking_error(
            (
                f"Model `{model}` does not support explicit thinking_budget on the current provider route. "
                f"Supported controls: {supported_controls}."
            ),
            resolved_profile=resolved_profile,
        )

    if thinking_budget is not None and explicit_effort != "auto":
        _raise_thinking_error(
            (
                f"Model `{model}` should not receive both thinking_budget={thinking_budget} and "
                f"thinking_effort={explicit_effort!r}. Use one control path only."
            ),
            resolved_profile=resolved_profile,
        )

    if semantic_mode == "off":
        if explicit_effort != "auto":
            _raise_thinking_error(
                (
                    f"Model `{model}` should use thinking_mode='off' without an explicit thinking_effort. "
                    f"Supported controls: {supported_controls}."
                ),
                resolved_profile=resolved_profile,
            )
        if thinking_budget not in (None, 0):
            _raise_thinking_error(
                (
                    f"Model `{model}` should not set thinking_budget={thinking_budget} while disabling thinking. "
                    f"Use thinking_mode='off' only."
                ),
                resolved_profile=resolved_profile,
            )
        normalized_provider_request: Dict[str, Any]
        if capability.family == "gpt5":
            normalized_provider_request = {"reasoning_effort": "none"}
        elif capability.family == "gemini" and "2.5" in model_l:
            normalized_provider_request = {"thinking_budget": 0}
        elif capability.family in {"claude_budget_legacy", "claude_adaptive"}:
            normalized_provider_request = {"thinking": {"type": "disabled"}}
        elif capability.family in {"glm", "kimi", "deepseek", "doubao"}:
            normalized_provider_request = {"thinking": {"type": "disabled"}}
        elif capability.family == "qwen":
            normalized_provider_request = {"enable_thinking": False, "thinking_budget": 0}
        else:
            _raise_thinking_error(
                (
                    f"Model `{model}` does not provide a strict off control on the current route. "
                    f"Supported controls: {supported_controls}. "
                    f"Benchmark strongest: {recommended_request}."
                ),
                resolved_profile=resolved_profile,
            )
        return {
            "request_semantics_valid": True,
            "semantic_failure_reason": "",
            "supported_thinking_controls": supported_controls,
            "recommended_benchmark_thinking_request": recommended_request,
            "normalized_provider_thinking_request": normalized_provider_request,
            "normalized_mode": "off",
            "normalized_effort": "off",
        }

    normalized_effort = explicit_effort
    if normalized_effort == "auto":
        normalized_effort = capability.benchmark_effort

    if capability.supported_efforts:
        if normalized_effort == "auto":
            _raise_thinking_error(
                f"Model `{model}` is missing a benchmark strongest thinking effort mapping.",
                resolved_profile=resolved_profile,
            )
        if normalized_effort not in capability.supported_efforts:
            if normalized_effort in {"none", "minimal"} and "off" in capability.supported_modes:
                _raise_thinking_error(
                    (
                        f"Model `{model}` treats thinking_effort={normalized_effort!r} as an off-like request. "
                        f"Use thinking_mode='off' instead. Supported efforts while thinking is on: "
                        f"{list(capability.supported_efforts)}."
                    ),
                    resolved_profile=resolved_profile,
                )
            _raise_thinking_error(
                (
                    f"Model `{model}` does not support thinking_effort={normalized_effort!r}. "
                    f"Supported efforts: {list(capability.supported_efforts)}. "
                    f"Benchmark strongest: {recommended_request}."
                ),
                resolved_profile=resolved_profile,
            )
    else:
        if normalized_effort != "auto":
            _raise_thinking_error(
                (
                    f"Model `{model}` does not expose adjustable thinking effort on the current route. "
                    f"Use thinking_mode='on' with the model's default strongest path. "
                    f"Supported controls: {supported_controls}."
                ),
                resolved_profile=resolved_profile,
            )

    normalized_provider_request: Dict[str, Any] = {}
    if capability.family == "gpt5":
        normalized_provider_request["reasoning_effort"] = normalized_effort
    elif capability.family == "gemini":
        if "2.5" in model_l:
            normalized_provider_request["thinking_budget"] = _effective_gemini_budget(
                model_l,
                thinking_budget,
                normalized_effort,
            )
        else:
            normalized_provider_request["thinking_level"] = normalized_effort
    elif capability.family == "claude_budget_legacy":
        normalized_provider_request["thinking"] = {
            "type": "enabled",
            "budget_tokens": _effective_claude_budget(model, max_tokens, thinking_budget, normalized_effort),
        }
        normalized_provider_request["effort"] = normalized_effort
    elif capability.family == "claude_adaptive":
        normalized_provider_request["thinking"] = {"type": "adaptive"}
        normalized_provider_request["effort"] = normalized_effort
    elif capability.family == "glm":
        normalized_provider_request["thinking"] = {"type": "enabled"}
    elif capability.family == "qwen":
        if capability.off_support == "toggle_not_supported":
            normalized_provider_request = {}
        else:
            normalized_provider_request["enable_thinking"] = True
            budget = _effective_qwen_budget(thinking_budget, normalized_effort)
            if budget is not None:
                normalized_provider_request["thinking_budget"] = int(budget)
    elif capability.family == "kimi":
        normalized_provider_request["thinking"] = {"type": "enabled"}
    elif capability.family == "deepseek":
        normalized_provider_request["thinking"] = {"type": "enabled"}
        normalized_provider_request["reasoning_effort"] = normalized_effort
    elif capability.family == "doubao":
        normalized_provider_request["thinking"] = {"type": "enabled"}
        normalized_provider_request["reasoning_effort"] = normalized_effort
    elif capability.family == "minimax":
        normalized_provider_request = {}
    else:
        normalized_provider_request = {}

    return {
        "request_semantics_valid": True,
        "semantic_failure_reason": "",
        "supported_thinking_controls": supported_controls,
        "recommended_benchmark_thinking_request": recommended_request,
        "normalized_provider_thinking_request": normalized_provider_request,
        "normalized_mode": "on",
        "normalized_effort": normalized_effort,
    }


def _is_gpt5_reasoning_model(model_l: str) -> bool:
    return model_l.startswith("gpt-5") or model_l.startswith("o1") or model_l.startswith("o3") or model_l.startswith("o4")


def _gpt_supports_none_effort(model_l: str) -> bool:
    match = re.search(r"gpt-5\.(\d+)", model_l)
    if match:
        return int(match.group(1)) >= 1
    return False


def _gpt_supports_minimal_effort(model_l: str) -> bool:
    # Minimal is the old GPT-5-era downshift; GPT-5.1+ uses none instead.
    # This keeps original GPT-5 models (e.g. gpt-5, gpt-5-mini) on minimal-capable path.
    if "gpt-5-codex" in model_l:
        return False
    return not _gpt_supports_none_effort(model_l)


def _gpt_supports_xhigh_effort(model_l: str) -> bool:
    if "codex-max" in model_l:
        return True
    match = re.search(r"gpt-5\.(\d+)", model_l)
    return bool(match and int(match.group(1)) >= 2)


def _default_gpt_effort(model_l: str) -> str:
    if _gpt_supports_xhigh_effort(model_l):
        return "xhigh"
    if "gpt-5-pro" in model_l:
        return "high"
    return "high"


def _normalize_gpt_effort_for_model(model_l: str, effort: str) -> str:
    x = effort
    if x == "xhigh" and not _gpt_supports_xhigh_effort(model_l):
        x = "high"
    if x == "none" and not _gpt_supports_none_effort(model_l):
        x = "minimal"
    if x == "minimal" and not _gpt_supports_minimal_effort(model_l):
        x = "none" if _gpt_supports_none_effort(model_l) else "low"
    return x


def _claude_budget_for_effort(effort: str) -> int:
    if effort == "max":
        return 48000
    if effort in {"none", "minimal"}:
        return 0
    if effort == "low":
        return 4000
    if effort == "medium":
        return 12000
    return 30000


def _qwen_budget_for_effort(effort: str) -> Optional[int]:
    if effort in {"none", "minimal"}:
        return 0
    if effort == "low":
        return 1024
    if effort == "medium":
        return 4096
    if effort in {"high", "xhigh"}:
        return 16384
    return None


def _gemini_default_budget(model_l: str) -> Optional[int]:
    if "2.5-pro" in model_l:
        return 32768
    if "2.5-flash" in model_l:
        return 24576
    return None


def _gemini_budget_for_effort(model_l: str, effort: str) -> Optional[int]:
    if effort in {"none", "minimal"}:
        return 0
    if effort == "low":
        return 1024
    if effort == "medium":
        return 4096
    if effort in {"high", "xhigh"}:
        return _gemini_default_budget(model_l) or 16384
    return None


def _gemini_level_for_effort(effort: str) -> str:
    if effort in {"none", "minimal"}:
        return "minimal"
    if effort == "low":
        return "low"
    if effort == "medium":
        return "medium"
    return "high"


def model_requires_data_url_images(model: str) -> bool:
    return resolve_model_profile(model).profile.require_data_url_images


def is_glm4_payload_strict(model: str) -> bool:
    return resolve_model_profile(model).profile.strict_payload


def model_disables_image_blocks(model: str) -> bool:
    return resolve_model_profile(model).profile.disable_image_blocks


def is_known_text_only_model(model: str) -> bool:
    profile = resolve_model_profile(model).profile
    return profile.multimodal_default is False


def is_known_multimodal_model(model: str) -> bool:
    profile = resolve_model_profile(model).profile
    if profile.disable_image_blocks:
        return False
    return profile.multimodal_default is True


def default_use_multimodal(model: str) -> bool:
    """Default multimodal behavior by model capability on current gateway."""
    profile = resolve_model_profile(model).profile
    if profile.disable_image_blocks:
        return False
    if profile.multimodal_default is not None:
        return profile.multimodal_default
    return True


def resolve_multimodal_mode(
    *,
    model: str,
    mode: str = "auto",
    explicit_override: Optional[bool] = None,
) -> bool:
    """Resolve final multimodal switch.

    Priority:
    1) explicit_override (legacy bool flags)
    2) mode=on/off
    3) mode=auto -> model-based default
    """
    if explicit_override is not None:
        return bool(explicit_override)

    mode_l = (mode or "auto").strip().lower()
    if mode_l == "on":
        return True
    if mode_l == "off":
        return False
    if mode_l != "auto":
        raise ValueError(f"Unsupported multimodal mode: {mode}")
    return default_use_multimodal(model)


def describe_model_modality(model: str) -> str:
    resolved = resolve_model_profile(model)
    profile = resolved.profile
    if profile.disable_image_blocks:
        return "single-modal (image blocks disabled by gateway route)"
    if profile.multimodal_default is False:
        return "single-modal"
    if profile.multimodal_default is True:
        return "multimodal"
    return "unknown (default multimodal)"


def url_to_data_url(url: str, timeout: float = 60.0) -> str:
    if not url or url.startswith("data:"):
        return url
    resp = SESSION.get(url, timeout=timeout)
    resp.raise_for_status()
    data = resp.content
    mime = resp.headers.get("Content-Type")
    if not mime or not str(mime).startswith("image/"):
        mime = mimetypes.guess_type(url)[0] or "image/png"
    b64 = base64.b64encode(data).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def infer_extra_options(
    model: str,
    model_tag: str = "",
    reasoning_effort: Optional[str] = None,
    thinking_mode: str = "auto",
    thinking_effort: str = "auto",
    thinking_budget: Optional[int] = None,
    resolved_profile: Optional[ResolvedModelProfile] = None,
    max_tokens: int = 81920,
) -> Dict[str, Any]:
    resolved_profile = resolved_profile or resolve_model_profile(model)
    semantics = resolve_thinking_request_semantics(
        model=model,
        max_tokens=max_tokens,
        reasoning_effort=reasoning_effort,
        thinking_mode=thinking_mode,
        thinking_effort=thinking_effort,
        thinking_budget=thinking_budget,
        resolved_profile=resolved_profile,
    )
    return copy.deepcopy(semantics["normalized_provider_thinking_request"])


def build_chat_payload(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    max_tokens: int,
    model_tag: str = "",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    reasoning_effort: Optional[str] = None,
    thinking_mode: str = "auto",
    thinking_effort: str = "auto",
    thinking_budget: Optional[int] = None,
    token_param_override: str = "auto",
    use_max_completion_tokens: bool = False,
    resolved_profile: Optional[ResolvedModelProfile] = None,
    normalized_provider_thinking_request: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    resolved_profile = resolved_profile or resolve_model_profile(model)
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
    }

    # GLM 4.5/4.7: keep payload minimal to avoid 1210 parameter-format errors.
    if not resolved_profile.profile.strict_payload:
        payload["repetition_penalty"] = 1.05
        payload["num_beams"] = 1
        payload["user"] = "sciencearena"

    normalized_max_tokens = normalize_max_tokens(model, max_tokens)
    if token_param_override not in {"auto", "max_tokens", "max_completion_tokens"}:
        raise ValueError(f"Unsupported token_param_override: {token_param_override}")

    effective_use_max_completion_tokens = use_max_completion_tokens or (
        resolved_profile.profile.token_limit_mode == "max_completion_tokens"
    )
    if token_param_override == "max_tokens":
        effective_use_max_completion_tokens = False
    elif token_param_override == "max_completion_tokens":
        effective_use_max_completion_tokens = True
    if effective_use_max_completion_tokens:
        payload["max_completion_tokens"] = normalized_max_tokens
    else:
        payload["max_tokens"] = normalized_max_tokens

    if temperature is not None:
        payload["temperature"] = float(temperature)
    if top_p is not None:
        payload["top_p"] = float(top_p)
    if top_k is not None:
        payload["top_k"] = int(top_k)

    if normalized_provider_thinking_request is not None:
        payload.update(copy.deepcopy(normalized_provider_thinking_request))
    else:
        payload.update(
            infer_extra_options(
                model,
                model_tag=model_tag,
                reasoning_effort=reasoning_effort,
                thinking_mode=thinking_mode,
                thinking_effort=thinking_effort,
                thinking_budget=thinking_budget,
                resolved_profile=resolved_profile,
                max_tokens=max_tokens,
            )
        )
    return payload


def build_curl_command(
    *,
    api_url: str,
    api_key: str,
    host: str,
    payload: Dict[str, Any],
) -> str:
    body = json.dumps(payload, ensure_ascii=False)
    return " ".join(
        [
            "curl",
            "-sS",
            shlex.quote(api_url),
            "-H",
            shlex.quote(f"Authorization: Bearer {api_key}"),
            "-H",
            shlex.quote("Content-Type: application/json"),
            "-H",
            shlex.quote(f"Host: {host}"),
            "-d",
            shlex.quote(body),
        ]
    )


def should_retry_with_max_completion_tokens(error_message: str) -> bool:
    msg = (error_message or "").lower()
    return (
        "unsupported parameter" in msg
        and "max_tokens" in msg
        and "max_completion_tokens" in msg
    )


def should_retry_with_max_tokens(error_message: str) -> bool:
    msg = (error_message or "").lower()
    return (
        "unsupported parameter" in msg
        and "max_completion_tokens" in msg
        and "max_tokens" in msg
    )


def drop_unsupported_payload_field(payload: Dict[str, Any], error_message: str) -> List[str]:
    msg = (error_message or "").lower()
    if not any(
        token in msg
        for token in (
            "unsupported parameter",
            "unknown parameter",
            "invalid parameter",
            "unrecognized request argument",
            "is not allowed",
            "extra fields",
        )
    ):
        return []

    if "budget_tokens" in msg:
        thinking = payload.get("thinking")
        if isinstance(thinking, dict) and "budget_tokens" in thinking:
            thinking.pop("budget_tokens", None)
            if not thinking:
                payload.pop("thinking", None)
            return ["thinking.budget_tokens"]

    direct_keys = (
        "effort",
        "reasoning_effort",
        "thinking_budget",
        "thinking_level",
        "enable_thinking",
        "reasoning_split",
    )
    for key in direct_keys:
        if key in msg and key in payload:
            payload.pop(key, None)
            return [key]

    if "thinking" in msg and "thinking" in payload:
        payload.pop("thinking", None)
        return ["thinking"]

    return []


def is_connectivity_error(error_message: str) -> bool:
    msg = (error_message or "").lower()
    return any(
        token in msg
        for token in (
            "connection error",
            "connecttimeout",
            "read timed out",
            "timed out",
            "max retries exceeded",
            "failed to establish a new connection",
            "name or service not known",
            "temporary failure",
            "connection refused",
        )
    )


HTTP_STATUS_RE = re.compile(r"\bhttp\s+(\d{3})\b", re.IGNORECASE)
TRANSIENT_HTTP_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}
TRANSIENT_ERROR_HINTS = (
    "temporarily unavailable",
    "temporary failure",
    "please retry",
    "try again",
    "upstream",
    "backend",
    "bad gateway",
    "gateway timeout",
    "service unavailable",
    "rate limit",
    "too many requests",
    "overloaded",
)


def extract_http_status(error_message: str) -> Optional[int]:
    match = HTTP_STATUS_RE.search(error_message or "")
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def has_image_blocks(messages: List[Dict[str, Any]]) -> bool:
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if isinstance(block, dict) and block.get("type") == "image_url":
                return True
    return False


def is_retryable_error(
    error_message: str,
    *,
    resolved_profile: ResolvedModelProfile,
    messages: List[Dict[str, Any]],
) -> bool:
    msg = (error_message or "").lower()
    if not msg:
        return True
    if is_connectivity_error(msg):
        return True

    status = extract_http_status(msg)
    if status in TRANSIENT_HTTP_STATUSES:
        return True
    if any(token in msg for token in TRANSIENT_ERROR_HINTS):
        return True

    # GPT-5 multimodal routes have shown intermittent 403 behavior on the 360
    # gateway. Retrying the exact same payload a few times is worthwhile before
    # concluding the alias/route is unusable.
    if (
        status == 403
        and resolved_profile.profile.thinking_family == "gpt5"
        and has_image_blocks(messages)
    ):
        return True

    return status not in {400, 401, 403, 404, 422}


def compute_retry_sleep_seconds(retry_delay: float, attempt_index: int) -> float:
    base = float(retry_delay) * (2 ** max(attempt_index, 0))
    jitter = random.uniform(0.85, 1.15)
    return max(base * jitter, 0.0)


def _safe_model_dump(obj: Any) -> Any:
    if obj is None:
        return {}
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if hasattr(obj, "to_dict"):
        return obj.to_dict()
    if hasattr(obj, "dict"):
        return obj.dict()  # type: ignore[no-any-return]
    return obj


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks: List[str] = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") in {"text", "output_text"} and part.get("text"):
                    chunks.append(str(part["text"]))
                elif "content" in part and isinstance(part["content"], str):
                    chunks.append(part["content"])
                elif "value" in part and isinstance(part["value"], str):
                    chunks.append(part["value"])
            else:
                text_attr = getattr(part, "text", None)
                if isinstance(text_attr, str) and text_attr:
                    chunks.append(text_attr)
        return "\n".join([x for x in chunks if x])
    return str(content)


def merge_reasoning_and_content(reasoning: str, content: str) -> str:
    reasoning = (reasoning or "").strip()
    content = content or ""
    if reasoning and content:
        return f"<think>\n{reasoning}\n</think>\n\n{content}"
    if reasoning:
        return f"<think>\n{reasoning}\n</think>"
    return content


class UnifiedChatClient:
    def __init__(
        self,
        *,
        api_key: str,
        backend: str = "openai",
        api_url: str = DEFAULT_API_URL,
        host: str = DEFAULT_HOST,
        max_retries: int = 5,
        retry_delay: float = 30.0,
        timeout: float = 900.0,
        model_tag: str = "",
        verbose: bool = False,
    ) -> None:
        self.api_key = ensure_api_key(api_key)
        self.backend = backend.lower().strip()
        self.api_url = api_url
        self.api_url_candidates = build_api_url_candidates(api_url)
        self.request_url = self.api_url_candidates[0]
        self.openai_base_url = normalize_openai_base_url(self.request_url)
        self.host = host
        self.max_retries = int(max_retries)
        self.retry_delay = float(retry_delay)
        self.timeout = float(timeout)
        self.model_tag = model_tag
        self.verbose = verbose
        self._image_url_cache: Dict[str, str] = {}
        self._openai_client = None
        self._openai_clients: Dict[str, Any] = {}

        if self.backend not in {"openai", "requests"}:
            raise ValueError("backend must be one of: openai, requests")

        if self.backend == "openai":
            try:
                from openai import OpenAI  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "openai package is required for backend=openai. Please install: pip install openai"
                ) from exc

            self._openai_client = OpenAI(
                api_key=self.api_key,
                base_url=self.openai_base_url,
                timeout=self.timeout,
                default_headers={"Host": self.host},
            )
            self._openai_clients[self.openai_base_url] = self._openai_client

    def _request_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json; charset=utf-8",
            "Host": self.host,
        }

    def chat(
        self,
        *,
        model: str,
        messages: List[Dict[str, Any]],
        max_tokens: int = 81920,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        thinking_mode: str = "auto",
        thinking_effort: str = "auto",
        thinking_budget: Optional[int] = None,
        token_param_override: str = "auto",
        print_curl: bool = False,
    ) -> Dict[str, Any]:
        resolved_profile = resolve_model_profile(model)
        thinking_capability = resolve_thinking_capability(resolved_profile)
        semantics = resolve_thinking_request_semantics(
            model=model,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
            thinking_mode=thinking_mode,
            thinking_effort=thinking_effort,
            thinking_budget=thinking_budget,
            resolved_profile=resolved_profile,
        )
        adapted_messages = self._prepare_messages_for_model(resolved_profile, messages)

        default_token_param_name = (
            "max_completion_tokens"
            if resolved_profile.profile.token_limit_mode == "max_completion_tokens"
            else "max_tokens"
        )
        requested_token_param_name = (
            token_param_override if token_param_override != "auto" else default_token_param_name
        )
        use_max_completion_tokens = requested_token_param_name == "max_completion_tokens"
        retried_with_max_completion_tokens = False
        retried_with_max_tokens = False
        token_param_fallback_used = False
        token_param_fallback_warning = ""
        payload = build_chat_payload(
            model=model,
            messages=adapted_messages,
            max_tokens=max_tokens,
            model_tag=self.model_tag,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            reasoning_effort=reasoning_effort,
            thinking_mode=thinking_mode,
            thinking_effort=thinking_effort,
            thinking_budget=thinking_budget,
            token_param_override=token_param_override,
            use_max_completion_tokens=use_max_completion_tokens,
            resolved_profile=resolved_profile,
            normalized_provider_thinking_request=semantics["normalized_provider_thinking_request"],
        )
        derived_thinking_options = extract_thinking_fields(payload)
        dropped_thinking_fields: List[str] = []
        thinking_drop_events: List[Dict[str, Any]] = []
        semantic_failure_reason = ""

        if print_curl:
            print(build_curl_command(api_url=self.api_url, api_key=self.api_key, host=self.host, payload=payload))

        last_error = ""
        error_history: List[Dict[str, Any]] = []
        attempt = 0
        while attempt < self.max_retries:
            try:
                response = self._chat_with_endpoint_fallback(payload)
                response["llm_output"] = merge_reasoning_and_content(
                    response.get("reasoning_content", ""),
                    response.get("content", ""),
                )
                final_thinking_fields = extract_thinking_fields(payload)
                response["client_meta"] = {
                    "resolved_profile": resolved_profile.to_summary_dict(),
                    "thinking_capability": thinking_capability.to_summary_dict(),
                    "requested_thinking_mode": thinking_mode,
                    "requested_thinking_effort": thinking_effort,
                    "requested_thinking_budget": thinking_budget,
                    "resolved_thinking_family": resolved_profile.profile.thinking_family,
                    "request_semantics_valid": semantics["request_semantics_valid"],
                    "semantic_failure_reason": semantic_failure_reason,
                    "supported_thinking_controls": semantics["supported_thinking_controls"],
                    "recommended_benchmark_thinking_request": semantics["recommended_benchmark_thinking_request"],
                    "normalized_provider_thinking_request": copy.deepcopy(
                        semantics["normalized_provider_thinking_request"]
                    ),
                    "derived_thinking_options": copy.deepcopy(derived_thinking_options),
                    "final_payload_thinking_fields": final_thinking_fields,
                    "dropped_thinking_fields": _dedupe_preserve_order(dropped_thinking_fields),
                    "thinking_drop_events": thinking_drop_events,
                    "thinking_fields_preserved_in_final_success": compare_thinking_fields(
                        derived_thinking_options,
                        final_thinking_fields,
                    ),
                    "token_limit_mode": resolved_profile.profile.token_limit_mode,
                    "requested_token_param_variant": token_param_override,
                    "requested_token_param_name": requested_token_param_name,
                    "effective_token_param_name": (
                        "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
                    ),
                    "token_param_used": (
                        "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
                    ),
                    "token_param_fallback_used": token_param_fallback_used,
                    "token_param_fallback_warning": token_param_fallback_warning,
                    "token_param_error_history": error_history,
                    "retried_with_max_completion_tokens": retried_with_max_completion_tokens,
                    "retried_with_max_tokens": retried_with_max_tokens,
                    "finish_reason": response.get("finish_reason", "unknown"),
                    "api_url_used": response.get("api_url_used", self.request_url),
                    "attempt_count": attempt + 1,
                    "error_history": error_history,
                }
                return response
            except Exception as exc:  # pragma: no cover - runtime network path
                last_error = str(exc)
                error_history.append(
                    {
                        "attempt": attempt + 1,
                        "error": last_error,
                    }
                )
                if should_retry_with_max_completion_tokens(last_error) and not use_max_completion_tokens:
                    use_max_completion_tokens = True
                    retried_with_max_completion_tokens = True
                    token_param_fallback_used = True
                    token_param_fallback_warning = (
                        f"Requested token parameter `{requested_token_param_name}` was rejected; "
                        "fell back to `max_completion_tokens`."
                    )
                    payload = build_chat_payload(
                        model=model,
                        messages=adapted_messages,
                        max_tokens=max_tokens,
                        model_tag=self.model_tag,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        reasoning_effort=reasoning_effort,
                        thinking_mode=thinking_mode,
                        thinking_effort=thinking_effort,
                        thinking_budget=thinking_budget,
                        token_param_override="max_completion_tokens",
                        use_max_completion_tokens=use_max_completion_tokens,
                        resolved_profile=resolved_profile,
                    )
                    continue
                if should_retry_with_max_tokens(last_error) and use_max_completion_tokens:
                    use_max_completion_tokens = False
                    retried_with_max_tokens = True
                    token_param_fallback_used = True
                    token_param_fallback_warning = (
                        f"Requested token parameter `{requested_token_param_name}` was rejected; "
                        "fell back to `max_tokens`."
                    )
                    payload = build_chat_payload(
                        model=model,
                        messages=adapted_messages,
                        max_tokens=max_tokens,
                        model_tag=self.model_tag,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        reasoning_effort=reasoning_effort,
                        thinking_mode=thinking_mode,
                        thinking_effort=thinking_effort,
                        thinking_budget=thinking_budget,
                        token_param_override="max_tokens",
                        use_max_completion_tokens=use_max_completion_tokens,
                        resolved_profile=resolved_profile,
                    )
                    continue

                dropped_fields = drop_unsupported_payload_field(payload, last_error)
                if dropped_fields:
                    semantic_failure_reason = (
                        f"Final request would require dropping thinking fields: {dropped_fields}. "
                        f"Original error: {last_error}"
                    )
                    dropped_thinking_fields.extend(dropped_fields)
                    thinking_drop_events.append(
                        {
                            "attempt": attempt + 1,
                            "error": last_error,
                            "dropped_fields": dropped_fields,
                            "remaining_thinking_fields": extract_thinking_fields(payload),
                        }
                    )
                    break

                if self.verbose:
                    print(
                        f"[WARN] chat failed attempt {attempt + 1}/{self.max_retries}: {exc}"
                    )
                should_retry = is_retryable_error(
                    last_error,
                    resolved_profile=resolved_profile,
                    messages=adapted_messages,
                )
                if attempt < self.max_retries - 1 and should_retry:
                    time.sleep(compute_retry_sleep_seconds(self.retry_delay, attempt))
                else:
                    attempt += 1
                    break
                attempt += 1

        if semantic_failure_reason:
            raise RuntimeError(
                f"Thinking semantic failure for model `{model}`: {semantic_failure_reason}. "
                f"Supported controls: {semantics['supported_thinking_controls']}. "
                f"Benchmark strongest: {semantics['recommended_benchmark_thinking_request']}."
            )

        return {
            "content": "",
            "reasoning_content": "",
            "llm_output": "",
            "usage": {},
            "raw": {},
            "error": last_error,
            "client_meta": {
                "resolved_profile": resolved_profile.to_summary_dict(),
                "thinking_capability": thinking_capability.to_summary_dict(),
                "requested_thinking_mode": thinking_mode,
                "requested_thinking_effort": thinking_effort,
                "requested_thinking_budget": thinking_budget,
                "resolved_thinking_family": resolved_profile.profile.thinking_family,
                "request_semantics_valid": semantics["request_semantics_valid"],
                "semantic_failure_reason": semantic_failure_reason,
                "supported_thinking_controls": semantics["supported_thinking_controls"],
                "recommended_benchmark_thinking_request": semantics["recommended_benchmark_thinking_request"],
                "normalized_provider_thinking_request": copy.deepcopy(
                    semantics["normalized_provider_thinking_request"]
                ),
                "derived_thinking_options": copy.deepcopy(derived_thinking_options),
                "final_payload_thinking_fields": extract_thinking_fields(payload),
                "dropped_thinking_fields": _dedupe_preserve_order(dropped_thinking_fields),
                "thinking_drop_events": thinking_drop_events,
                "thinking_fields_preserved_in_final_success": None,
                "token_limit_mode": resolved_profile.profile.token_limit_mode,
                "requested_token_param_variant": token_param_override,
                "requested_token_param_name": requested_token_param_name,
                "effective_token_param_name": (
                    "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
                ),
                "token_param_used": (
                    "max_completion_tokens" if "max_completion_tokens" in payload else "max_tokens"
                ),
                "token_param_fallback_used": token_param_fallback_used,
                "token_param_fallback_warning": token_param_fallback_warning,
                "token_param_error_history": error_history,
                "retried_with_max_completion_tokens": retried_with_max_completion_tokens,
                "retried_with_max_tokens": retried_with_max_tokens,
                "finish_reason": "unknown",
                "api_url_used": self.request_url,
                "attempt_count": attempt,
                "error_history": error_history,
            },
        }

    def _chat_with_endpoint_fallback(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        last_error = ""
        for index, api_url in enumerate(list(self.api_url_candidates)):
            try:
                if self.backend == "openai":
                    response = self._chat_with_openai(payload, api_url=api_url)
                else:
                    response = self._chat_with_requests(payload, api_url=api_url)
                if index > 0:
                    self.api_url_candidates = [api_url] + [x for x in self.api_url_candidates if x != api_url]
                    self.request_url = self.api_url_candidates[0]
                    self.openai_base_url = normalize_openai_base_url(self.request_url)
                return response
            except Exception as exc:
                last_error = str(exc)
                if not is_connectivity_error(last_error):
                    raise
                if self.verbose:
                    print(f"[WARN] connectivity issue on {api_url}: {last_error}")
        raise RuntimeError(last_error or "Unknown connectivity error")

    def _prepare_messages_for_model(
        self,
        resolved_profile: ResolvedModelProfile,
        messages: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        adapted = copy.deepcopy(messages)

        if resolved_profile.profile.disable_image_blocks:
            for msg in adapted:
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                filtered = [
                    block
                    for block in content
                    if not (isinstance(block, dict) and block.get("type") == "image_url")
                ]
                if not filtered:
                    filtered = [{"type": "text", "text": "[Image omitted due to model input constraints.]"}]
                msg["content"] = filtered

        if not resolved_profile.profile.require_data_url_images:
            return adapted

        for msg in adapted:
            content = msg.get("content")
            if not isinstance(content, list):
                continue
            for block in content:
                if not isinstance(block, dict):
                    continue
                if block.get("type") != "image_url":
                    continue
                image_obj = block.get("image_url")
                if not isinstance(image_obj, dict):
                    continue
                url = str(image_obj.get("url", "")).strip()
                if not url or url.startswith("data:"):
                    continue
                try:
                    if url in self._image_url_cache:
                        image_obj["url"] = self._image_url_cache[url]
                    else:
                        data_url = url_to_data_url(url)
                        self._image_url_cache[url] = data_url
                        image_obj["url"] = data_url
                except Exception:
                    # If conversion fails, keep original URL and let server decide.
                    pass
        return adapted

    def _chat_with_requests(self, payload: Dict[str, Any], *, api_url: str) -> Dict[str, Any]:
        response = SESSION.post(
            api_url,
            headers=self._request_headers(),
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}: {response.text[:1200]}")
        data = response.json()

        if data.get("error"):
            raise RuntimeError(str(data["error"]))

        choices = data.get("choices") or []
        if not choices:
            raise RuntimeError(f"Invalid API response without choices: {str(data)[:300]}")

        message = choices[0].get("message") or {}
        content = _extract_text(message.get("content"))
        reasoning = _extract_text(message.get("reasoning_content"))
        finish_reason = str(choices[0].get("finish_reason") or "unknown")

        return {
            "content": content,
            "reasoning_content": reasoning,
            "usage": data.get("usage") or {},
            "raw": data,
            "finish_reason": finish_reason,
            "api_url_used": api_url,
        }

    def _chat_with_openai(self, payload: Dict[str, Any], *, api_url: str) -> Dict[str, Any]:
        client = self._get_openai_client(api_url)

        request_kwargs: Dict[str, Any] = {
            "model": payload["model"],
            "messages": payload["messages"],
            "stream": False,
            "timeout": self.timeout,
        }

        if "max_tokens" in payload:
            request_kwargs["max_tokens"] = payload["max_tokens"]
        if "max_completion_tokens" in payload:
            request_kwargs["max_completion_tokens"] = payload["max_completion_tokens"]

        if "temperature" in payload:
            request_kwargs["temperature"] = payload["temperature"]
        if "top_p" in payload:
            request_kwargs["top_p"] = payload["top_p"]

        extra_body: Dict[str, Any] = {}
        for key in (
            "thinking",
            "reasoning_effort",
            "thinking_budget",
            "thinking_level",
            "enable_thinking",
            "reasoning_split",
            "repetition_penalty",
            "num_beams",
            "top_k",
            "user",
        ):
            if key in payload:
                extra_body[key] = payload[key]

        if extra_body:
            request_kwargs["extra_body"] = extra_body

        response = client.chat.completions.create(**request_kwargs)

        choices = getattr(response, "choices", None) or []
        if not choices:
            raise RuntimeError("OpenAI response has no choices")

        message = choices[0].message
        content = _extract_text(getattr(message, "content", ""))
        reasoning = _extract_text(getattr(message, "reasoning_content", ""))
        finish_reason = str(getattr(choices[0], "finish_reason", "") or "unknown")

        return {
            "content": content,
            "reasoning_content": reasoning,
            "usage": _safe_model_dump(getattr(response, "usage", {})),
            "raw": _safe_model_dump(response),
            "finish_reason": finish_reason,
            "api_url_used": api_url,
        }

    def _get_openai_client(self, api_url: str) -> Any:
        base_url = normalize_openai_base_url(api_url)
        client = self._openai_clients.get(base_url)
        if client is not None:
            return client
        if self._openai_client is None:
            raise RuntimeError("openai client is not initialized")

        from openai import OpenAI  # type: ignore

        client = OpenAI(
            api_key=self.api_key,
            base_url=base_url,
            timeout=self.timeout,
            default_headers={"Host": self.host},
        )
        self._openai_clients[base_url] = client
        return client


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Cannot parse bool value: {value}")


def clean_image_urls(image_urls: Optional[List[Any]]) -> List[str]:
    if not image_urls:
        return []
    result: List[str] = []
    for url in image_urls:
        text = str(url).strip()
        if text:
            result.append(text)
    return result


def build_user_message(
    *,
    text: str,
    image_urls: Optional[List[str]] = None,
    use_multimodal: bool = True,
) -> Dict[str, Any]:
    image_urls = clean_image_urls(image_urls)
    if use_multimodal and image_urls:
        content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
        for image_url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": image_url}})
        return {"role": "user", "content": content}
    return {"role": "user", "content": text}
