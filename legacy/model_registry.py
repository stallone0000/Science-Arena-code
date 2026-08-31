# -*- coding: utf-8 -*-
"""Explicit model registry for ScienceArena API compatibility.

This module is the single source of truth for model capability and routing
assumptions. Add new models here first, then rerun smoke tests.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, Optional, Tuple


TokenLimitMode = str
ThinkingFamily = str


MODEL_FAMILY_HINTS: Tuple[str, ...] = (
    "gemini-",
    "claude-",
    "gpt-",
    "o1-",
    "o3-",
    "o4-",
    "deepseek",
    "qwen",
    "glm",
    "doubao",
    "minimax",
    "kimi",
)


def normalize_model_name(model: str) -> str:
    return (model or "").strip().lower()


def canonical_model_name(model: str) -> str:
    model_l = normalize_model_name(model)
    if not model_l:
        return ""

    parts = [part for part in model_l.split("/") if part]
    if not parts:
        return model_l

    compact = []
    for part in parts:
        if not compact or compact[-1] != part:
            compact.append(part)

    for part in reversed(compact):
        if any(hint in part for hint in MODEL_FAMILY_HINTS):
            return part
    return compact[-1]


@dataclass(frozen=True)
class ModelProfile:
    profile_id: str
    aliases: Tuple[str, ...] = ()
    canonical_exact: Tuple[str, ...] = ()
    canonical_prefixes: Tuple[str, ...] = ()
    preferred_model: str = ""
    multimodal_default: Optional[bool] = None
    disable_image_blocks: bool = False
    require_data_url_images: bool = False
    strict_payload: bool = False
    token_limit_mode: TokenLimitMode = "max_tokens"
    thinking_family: ThinkingFamily = "generic"
    backend_notes: Dict[str, str] = field(default_factory=dict)
    route_notes: str = ""


@dataclass(frozen=True)
class ResolvedModelProfile:
    requested_model: str
    normalized_model: str
    canonical_name: str
    matched_via: str
    profile: ModelProfile

    def to_summary_dict(self) -> Dict[str, object]:
        profile_dict = asdict(self.profile)
        profile_dict["aliases"] = list(self.profile.aliases)
        profile_dict["canonical_exact"] = list(self.profile.canonical_exact)
        profile_dict["canonical_prefixes"] = list(self.profile.canonical_prefixes)
        return {
            "requested_model": self.requested_model,
            "normalized_model": self.normalized_model,
            "canonical_name": self.canonical_name,
            "matched_via": self.matched_via,
            "profile": profile_dict,
        }


@dataclass(frozen=True)
class ThinkingCapability:
    family: ThinkingFamily
    provider_request_fields: Tuple[str, ...]
    compat_request_fields: Tuple[str, ...]
    off_support: str
    observability: str
    supported_modes: Tuple[str, ...] = ("on",)
    supported_efforts: Tuple[str, ...] = ()
    benchmark_mode: str = "on"
    benchmark_effort: str = "auto"
    current_client_mapping: str = "supported"
    benchmark_notes: str = ""
    notes: str = ""

    def to_summary_dict(self) -> Dict[str, object]:
        return {
            "family": self.family,
            "provider_request_fields": list(self.provider_request_fields),
            "compat_request_fields": list(self.compat_request_fields),
            "off_support": self.off_support,
            "observability": self.observability,
            "supported_modes": list(self.supported_modes),
            "supported_efforts": list(self.supported_efforts),
            "benchmark_mode": self.benchmark_mode,
            "benchmark_effort": self.benchmark_effort,
            "current_client_mapping": self.current_client_mapping,
            "benchmark_notes": self.benchmark_notes,
            "notes": self.notes,
        }


PROFILES: Tuple[ModelProfile, ...] = (
    ModelProfile(
        profile_id="gpt5_bare_alias",
        aliases=("gpt-5",),
        preferred_model="openai/gpt-5",
        multimodal_default=True,
        token_limit_mode="max_completion_tokens",
        thinking_family="gpt5",
        backend_notes={
            "requests": "Requests multimodal routing has shown route drift across runs; rerun targeted smoke before relying on this alias.",
        },
        route_notes="Prefer provider-qualified openai/gpt-5 in docs and new onboarding, but still rerun targeted smoke because routes can drift.",
    ),
    ModelProfile(
        profile_id="openai_gpt5",
        aliases=("openai/gpt-5",),
        preferred_model="openai/gpt-5",
        multimodal_default=True,
        token_limit_mode="max_completion_tokens",
        thinking_family="gpt5",
        backend_notes={
            "requests": "Provider-qualified alias has also shown intermittent Forbidden on some multimodal routes; rerun targeted smoke on the current endpoint.",
        },
    ),
    ModelProfile(
        profile_id="openai_gpt51",
        aliases=("openai/gpt-5.1",),
        preferred_model="openai/gpt-5.1",
        multimodal_default=True,
        token_limit_mode="max_completion_tokens",
        thinking_family="gpt5",
    ),
    ModelProfile(
        profile_id="openai_gpt52",
        aliases=("openai/gpt-5.2",),
        preferred_model="openai/gpt-5.2",
        multimodal_default=True,
        token_limit_mode="max_completion_tokens",
        thinking_family="gpt5",
    ),
    ModelProfile(
        profile_id="openai_gpt54",
        aliases=("openai/gpt-5.4",),
        preferred_model="openai/gpt-5.4",
        multimodal_default=True,
        token_limit_mode="max_completion_tokens",
        thinking_family="gpt5",
    ),
    ModelProfile(
        profile_id="openai_gpt55",
        aliases=("openai/gpt-5.5",),
        preferred_model="openai/gpt-5.5",
        multimodal_default=True,
        token_limit_mode="max_completion_tokens",
        thinking_family="gpt5",
    ),
    ModelProfile(
        profile_id="qiniu_openai_gpt54",
        aliases=("qiniu/openai/gpt-5.4",),
        preferred_model="qiniu/openai/gpt-5.4",
        multimodal_default=True,
        token_limit_mode="max_completion_tokens",
        thinking_family="gpt5",
        route_notes="Explicit onboarding example profile. This model would also match gpt5_family via canonical name, but the alias keeps docs and preferred naming explicit.",
    ),
    ModelProfile(
        profile_id="claude_budget_legacy_data_url_variants",
        canonical_exact=("claude-opus-4.5",),
        preferred_model="anthropic/claude-opus-4.5",
        multimodal_default=True,
        require_data_url_images=True,
        token_limit_mode="max_tokens",
        thinking_family="claude_budget_legacy",
    ),
    ModelProfile(
        profile_id="claude_adaptive_data_url_variants",
        canonical_exact=("claude-sonnet-4-6", "claude-opus-4.6", "claude-opus-4.7"),
        preferred_model="anthropic/claude-sonnet-4-6",
        multimodal_default=True,
        require_data_url_images=True,
        token_limit_mode="max_tokens",
        thinking_family="claude_adaptive",
        route_notes="Claude 4.6/4.7 use adaptive thinking with effort; legacy budget_tokens mode is deprecated on 4.6 and superseded on 4.7.",
    ),
    ModelProfile(
        profile_id="glm45",
        canonical_exact=("glm-4.5",),
        preferred_model="bigmodel/glm-4.5",
        multimodal_default=False,
        disable_image_blocks=True,
        strict_payload=True,
        token_limit_mode="max_tokens",
        thinking_family="glm",
        route_notes="Current 360 route rejects image blocks; keep payload minimal.",
    ),
    ModelProfile(
        profile_id="glm47",
        canonical_exact=("glm-4.7",),
        preferred_model="bigmodel/glm-4.7",
        multimodal_default=False,
        disable_image_blocks=True,
        strict_payload=True,
        token_limit_mode="max_tokens",
        thinking_family="glm",
        route_notes="Current 360 route rejects image blocks; keep payload minimal.",
    ),
    ModelProfile(
        profile_id="glm5",
        canonical_exact=("glm-5",),
        preferred_model="z-ai/glm-5",
        multimodal_default=True,
        token_limit_mode="max_tokens",
        thinking_family="glm",
    ),
    ModelProfile(
        profile_id="deepseek_text_only",
        canonical_exact=("deepseek-v3.1", "deepseek-v3.2"),
        preferred_model="deepseek-v3.2",
        multimodal_default=False,
        token_limit_mode="max_tokens",
        thinking_family="deepseek",
    ),
    ModelProfile(
        profile_id="qwen_text_only",
        canonical_exact=("qwen3-235b-a22b-thinking-2507", "qwen3.5-397b-a17b"),
        preferred_model="alibaba/qwen3-235b-a22b-thinking-2507",
        multimodal_default=False,
        token_limit_mode="max_tokens",
        thinking_family="qwen",
    ),
    ModelProfile(
        profile_id="kimi_text_only",
        canonical_exact=("kimi-k2.5",),
        preferred_model="moonshot/kimi-k2.5",
        multimodal_default=False,
        token_limit_mode="max_tokens",
        thinking_family="kimi",
    ),
    ModelProfile(
        profile_id="gemini_family",
        canonical_prefixes=("gemini-",),
        preferred_model="google/gemini-3-pro-preview",
        multimodal_default=True,
        token_limit_mode="max_tokens",
        thinking_family="gemini",
    ),
    ModelProfile(
        profile_id="claude_budget_legacy_family",
        canonical_prefixes=("claude-",),
        preferred_model="anthropic/claude-opus-4.5",
        multimodal_default=True,
        token_limit_mode="max_tokens",
        thinking_family="claude_budget_legacy",
    ),
    ModelProfile(
        profile_id="gpt5_family",
        canonical_prefixes=("gpt-5",),
        preferred_model="openai/gpt-5",
        multimodal_default=True,
        token_limit_mode="max_completion_tokens",
        thinking_family="gpt5",
        route_notes="Prefer provider-qualified model names when onboarding new GPT-5 routes.",
    ),
    ModelProfile(
        profile_id="o_series_family",
        canonical_prefixes=("o1-", "o3-", "o4-"),
        preferred_model="o3-mini",
        multimodal_default=True,
        token_limit_mode="max_completion_tokens",
        thinking_family="gpt5",
    ),
    ModelProfile(
        profile_id="doubao_family",
        canonical_prefixes=("doubao-",),
        preferred_model="volcengine/doubao-seed-2-0-pro",
        multimodal_default=True,
        token_limit_mode="max_tokens",
        thinking_family="doubao",
    ),
    ModelProfile(
        profile_id="minimax_family",
        canonical_prefixes=("minimax-",),
        preferred_model="minimax/MiniMax-M2.5",
        multimodal_default=True,
        token_limit_mode="max_tokens",
        thinking_family="minimax",
    ),
    ModelProfile(
        profile_id="generic_multimodal_fallback",
        preferred_model="",
        multimodal_default=True,
        token_limit_mode="max_tokens",
        thinking_family="generic",
        route_notes="Unknown model family; review and add an explicit profile before production use.",
    ),
)


def resolve_model_profile(model: str) -> ResolvedModelProfile:
    normalized = normalize_model_name(model)
    canonical = canonical_model_name(model)

    for profile in PROFILES:
        if normalized and normalized in tuple(normalize_model_name(alias) for alias in profile.aliases):
            return ResolvedModelProfile(
                requested_model=model,
                normalized_model=normalized,
                canonical_name=canonical,
                matched_via="alias",
                profile=profile,
            )

    for profile in PROFILES:
        if canonical and canonical in tuple(normalize_model_name(name) for name in profile.canonical_exact):
            return ResolvedModelProfile(
                requested_model=model,
                normalized_model=normalized,
                canonical_name=canonical,
                matched_via="canonical_exact",
                profile=profile,
            )

    for profile in PROFILES:
        if canonical and any(canonical.startswith(prefix) for prefix in profile.canonical_prefixes):
            return ResolvedModelProfile(
                requested_model=model,
                normalized_model=normalized,
                canonical_name=canonical,
                matched_via="canonical_prefix",
                profile=profile,
            )

    fallback = PROFILES[-1]
    return ResolvedModelProfile(
        requested_model=model,
        normalized_model=normalized,
        canonical_name=canonical,
        matched_via="fallback",
        profile=fallback,
    )


def resolve_thinking_capability(model: str | ResolvedModelProfile) -> ThinkingCapability:
    resolved = model if isinstance(model, ResolvedModelProfile) else resolve_model_profile(model)
    canonical = resolved.canonical_name
    family = resolved.profile.thinking_family

    gpt_minor_version: Optional[int] = None
    if canonical.startswith("gpt-5."):
        try:
            gpt_minor_version = int(canonical.split(".", 2)[1])
        except (IndexError, ValueError):
            gpt_minor_version = None

    gpt_xhigh_supported = bool(
        "codex-max" in canonical
        or (gpt_minor_version is not None and gpt_minor_version >= 2)
    )

    if family == "gpt5":
        supports_strict_off = bool(gpt_minor_version is not None and gpt_minor_version >= 1)
        if "gpt-5-pro" in canonical:
            return ThinkingCapability(
                family=family,
                provider_request_fields=("reasoning.effort",),
                compat_request_fields=("reasoning_effort",),
                off_support="off_not_supported",
                observability="usage_reasoning_tokens",
                supported_modes=("on",),
                supported_efforts=("high",),
                benchmark_mode="on",
                benchmark_effort="high",
                benchmark_notes="GPT-5 Pro only supports high reasoning effort.",
                notes="OpenAI Chat Completions exposes reasoning token usage but not raw reasoning by default.",
            )
        return ThinkingCapability(
            family=family,
            provider_request_fields=("reasoning.effort",),
            compat_request_fields=("reasoning_effort",),
            off_support="strict_supported" if supports_strict_off else "best_effort_only",
            observability="usage_reasoning_tokens",
            supported_modes=("on", "off") if supports_strict_off else ("on",),
            supported_efforts=("low", "medium", "high") + (("xhigh",) if gpt_xhigh_supported else ()),
            benchmark_mode="on",
            benchmark_effort="xhigh" if gpt_xhigh_supported else "high",
            benchmark_notes="Benchmark strongest for GPT reasoning models is the highest provider-supported reasoning_effort.",
            notes=(
                "OpenAI Chat Completions exposes reasoning token usage, but raw reasoning is not "
                "returned by default. GPT-5.1+ routes support strict none/off; older GPT-5 routes only support best-effort downshift."
            ),
        )

    if family == "gemini":
        if "2.5-pro" in canonical:
            return ThinkingCapability(
                family=family,
                provider_request_fields=("thinkingConfig.thinkingBudget",),
                compat_request_fields=("thinking_budget",),
                off_support="off_not_supported",
                observability="usage_reasoning_tokens",
                supported_modes=("on",),
                supported_efforts=("low", "medium", "high"),
                benchmark_mode="on",
                benchmark_effort="high",
                benchmark_notes="Benchmark strongest uses the maximum documented thinkingBudget for Gemini 2.5 Pro.",
                notes="Gemini 2.5 Pro uses thinkingBudget and does not support a strict thinking-off mode.",
            )
        if "2.5" in canonical:
            return ThinkingCapability(
                family=family,
                provider_request_fields=("thinkingConfig.thinkingBudget",),
                compat_request_fields=("thinking_budget",),
                off_support="strict_supported",
                observability="usage_reasoning_tokens",
                supported_modes=("on", "off"),
                supported_efforts=("low", "medium", "high"),
                benchmark_mode="on",
                benchmark_effort="high",
                benchmark_notes="Benchmark strongest uses the maximum documented thinkingBudget for Gemini 2.5 non-Pro models.",
                notes="Gemini 2.5 non-Pro variants can use thinkingBudget=0 to disable visible thinking.",
            )
        if "gemini-3-pro" in canonical:
            return ThinkingCapability(
                family=family,
                provider_request_fields=("thinkingConfig.thinkingLevel",),
                compat_request_fields=("thinking_level",),
                off_support="off_not_supported",
                observability="usage_reasoning_tokens",
                supported_modes=("on",),
                supported_efforts=("low", "medium", "high"),
                benchmark_mode="on",
                benchmark_effort="high",
                benchmark_notes='Gemini 3 Pro defaults to dynamic "high"; benchmark strongest pins high explicitly.',
                notes="Gemini 3 Pro uses thinkingLevel and does not support a strict full-off mode.",
            )
        return ThinkingCapability(
            family=family,
            provider_request_fields=("thinkingConfig.thinkingLevel",),
            compat_request_fields=("thinking_level",),
            off_support="best_effort_only",
            observability="usage_reasoning_tokens",
            supported_modes=("on",),
            supported_efforts=("minimal", "low", "medium", "high"),
            benchmark_mode="on",
            benchmark_effort="high",
            benchmark_notes='Gemini 3 Flash-family strongest request pins thinkingLevel="high".',
            notes="Gemini 3 Flash-family models can use minimal as a weak-thinking setting, but they still do not provide a strict full-off switch.",
        )

    if family == "claude_budget_legacy":
        return ThinkingCapability(
            family=family,
            provider_request_fields=("thinking.type", "thinking.budget_tokens", "effort"),
            compat_request_fields=("thinking.type", "thinking.budget_tokens", "effort"),
            off_support="strict_supported",
            observability="provider_specific_blocks",
            supported_modes=("on", "off"),
            supported_efforts=("low", "medium", "high"),
            benchmark_mode="on",
            benchmark_effort="high",
            benchmark_notes="Legacy Claude strongest uses enabled thinking with a large budget_tokens value plus high effort.",
            notes="Claude 4.1/4.5 style routes support manual thinking.type=enabled with budget_tokens, and effort can be set alongside it.",
        )

    if family == "claude_adaptive":
        return ThinkingCapability(
            family=family,
            provider_request_fields=("thinking.type", "effort"),
            compat_request_fields=("thinking.type", "effort"),
            off_support="strict_supported",
            observability="provider_specific_blocks",
            supported_modes=("on", "off"),
            supported_efforts=("low", "medium", "high", "max"),
            benchmark_mode="on",
            benchmark_effort="max",
            benchmark_notes='Claude 4.6/4.7 strongest uses adaptive thinking with effort="max".',
            notes="Claude 4.6/4.7 use adaptive thinking; manual budget_tokens is deprecated on 4.6 and no longer the recommended control surface.",
        )

    if family == "glm":
        return ThinkingCapability(
            family=family,
            provider_request_fields=("thinking.type",),
            compat_request_fields=("thinking.type",),
            off_support="strict_supported",
            observability="provider_specific_blocks",
            supported_modes=("on", "off"),
            supported_efforts=(),
            benchmark_mode="on",
            benchmark_effort="auto",
            benchmark_notes='GLM benchmark strongest uses thinking={"type":"enabled"} with no extra effort knob.',
            notes="GLM 4.7/5 default to thinking-on and support thinking.type=disabled for explicit off.",
        )

    if family == "qwen":
        if canonical in {"qwen3-235b-a22b-thinking-2507", "qwen3.5-397b-a17b"}:
            return ThinkingCapability(
                family=family,
                provider_request_fields=("enable_thinking", "thinking_budget"),
                compat_request_fields=("enable_thinking", "thinking_budget"),
                off_support="toggle_not_supported",
                observability="reasoning_content",
                supported_modes=("on",),
                supported_efforts=(),
                benchmark_mode="on",
                benchmark_effort="auto",
                current_client_mapping="supported",
                benchmark_notes="This model is already always-thinking; benchmark strongest should rely on the model identity rather than a toggle field.",
                notes=(
                    "Aliyun docs mark enable_thinking ineffective for qwen3-235b-a22b-thinking-2507 and related always-thinking models; "
                    "treat toggle results as non-authoritative."
                ),
            )
        return ThinkingCapability(
            family=family,
            provider_request_fields=("enable_thinking", "thinking_budget"),
            compat_request_fields=("enable_thinking", "thinking_budget"),
            off_support="strict_supported",
            observability="reasoning_content",
            supported_modes=("on", "off"),
            supported_efforts=("low", "medium", "high"),
            benchmark_mode="on",
            benchmark_effort="high",
            benchmark_notes="Benchmark strongest maps to enable_thinking=true plus the highest supported thinking_budget.",
            notes="Standard Qwen3 OpenAI-compatible routes use enable_thinking plus optional thinking_budget.",
        )

    if family == "kimi":
        return ThinkingCapability(
            family=family,
            provider_request_fields=("thinking.type",),
            compat_request_fields=("thinking.type",),
            off_support="strict_supported",
            observability="reasoning_content",
            supported_modes=("on", "off"),
            supported_efforts=(),
            benchmark_mode="on",
            benchmark_effort="auto",
            benchmark_notes='Kimi K2.5 strongest request uses thinking={"type":"enabled"}; disable must use thinking.type=disabled.',
            notes="Kimi K2.5 uses thinking.type=disabled for non-thinking mode and preserves reasoning_content during tool-use turns.",
        )

    if family == "deepseek":
        return ThinkingCapability(
            family=family,
            provider_request_fields=("thinking.type",),
            compat_request_fields=("thinking.type", "reasoning_effort"),
            off_support="strict_supported",
            observability="reasoning_content",
            supported_modes=("on", "off"),
            supported_efforts=(),
            benchmark_mode="on",
            benchmark_effort="auto",
            benchmark_notes='Benchmark strongest maps to thinking={"type":"enabled"}; official docs do not expose a public effort ladder.',
            notes="DeepSeek documents thinking.type=enabled/disabled and returns reasoning_content. Current public docs do not define a stable effort ladder for the OpenAI-compatible route.",
        )

    if family == "doubao":
        return ThinkingCapability(
            family=family,
            provider_request_fields=("thinking.type", "reasoning_effort"),
            compat_request_fields=("thinking.type", "reasoning_effort"),
            off_support="strict_supported",
            observability="reasoning_content",
            supported_modes=("on", "off"),
            supported_efforts=("low", "medium", "high"),
            benchmark_mode="on",
            benchmark_effort="high",
            benchmark_notes='Benchmark strongest uses thinking={"type":"enabled"} plus reasoning_effort="high".',
            notes=(
                "Volcengine docs expose strict enabled/disabled semantics, but compatibility layers may rename the knob. "
                "Treat preserved payload evidence as the main source of truth on the 360 gateway."
            ),
        )

    if family == "minimax":
        return ThinkingCapability(
            family=family,
            provider_request_fields=(),
            compat_request_fields=(),
            off_support="toggle_not_supported",
            observability="content_think_tag",
            supported_modes=("on",),
            supported_efforts=(),
            benchmark_mode="on",
            benchmark_effort="auto",
            current_client_mapping="no_public_knob",
            benchmark_notes="Benchmark strongest can only rely on the model's default reasoning behavior on the current OpenAI-compatible route.",
            notes="MiniMax OpenAI-compatible docs do not expose a reliable public thinking toggle; reasoning may appear inline in <think> tags.",
        )

    return ThinkingCapability(
        family=family,
        provider_request_fields=(),
        compat_request_fields=(),
        off_support="unknown",
        observability="unknown",
        supported_modes=("on",),
        supported_efforts=(),
        benchmark_mode="on",
        benchmark_effort="auto",
        current_client_mapping="unknown",
        benchmark_notes="Unknown family; no benchmark strongest mapping is defined.",
        notes="Unknown family. Add an explicit profile and verify provider docs before relying on thinking controls.",
    )
