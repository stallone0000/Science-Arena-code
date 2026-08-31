# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest
from unittest import mock

from api_client_interleaved import (
    ThinkingSemanticsError,
    UnifiedChatClient,
    build_chat_payload,
    compare_thinking_fields,
    drop_unsupported_payload_field,
    extract_thinking_fields,
    resolve_thinking_request_semantics,
)
from model_registry import resolve_thinking_capability


class ApiClientThinkingTest(unittest.TestCase):
    def test_gpt54_strongest_auto_maps_to_xhigh(self) -> None:
        payload = build_chat_payload(
            model="qiniu/openai/gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            thinking_mode="on",
            thinking_effort="auto",
        )
        self.assertEqual(extract_thinking_fields(payload), {"reasoning_effort": "xhigh"})

    def test_token_param_override_changes_payload(self) -> None:
        payload_max = build_chat_payload(
            model="qiniu/openai/gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            token_param_override="max_tokens",
        )
        payload_completion = build_chat_payload(
            model="qiniu/openai/gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            token_param_override="max_completion_tokens",
        )
        self.assertIn("max_tokens", payload_max)
        self.assertNotIn("max_completion_tokens", payload_max)
        self.assertIn("max_completion_tokens", payload_completion)
        self.assertNotIn("max_tokens", payload_completion)

    def test_gpt54_off_maps_to_none(self) -> None:
        payload = build_chat_payload(
            model="qiniu/openai/gpt-5.4",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            thinking_mode="off",
            thinking_effort="auto",
        )
        self.assertEqual(extract_thinking_fields(payload), {"reasoning_effort": "none"})

    def test_gpt51_off_maps_to_none(self) -> None:
        payload = build_chat_payload(
            model="openai/gpt-5.1",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            thinking_mode="off",
            thinking_effort="auto",
        )
        self.assertEqual(extract_thinking_fields(payload), {"reasoning_effort": "none"})

    def test_gemini25_pro_off_is_rejected(self) -> None:
        with self.assertRaises(ThinkingSemanticsError):
            build_chat_payload(
                model="google/gemini-2.5-pro",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=1024,
                thinking_mode="off",
                thinking_effort="auto",
            )

    def test_gemini25_pro_strongest_uses_max_budget(self) -> None:
        payload = build_chat_payload(
            model="google/gemini-2.5-pro",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            thinking_mode="on",
            thinking_effort="auto",
        )
        self.assertEqual(extract_thinking_fields(payload), {"thinking_budget": 32768})

    def test_kimi_off_uses_thinking_disabled(self) -> None:
        payload = build_chat_payload(
            model="moonshot/kimi-k2.5",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=1024,
            thinking_mode="off",
            thinking_effort="auto",
        )
        self.assertEqual(extract_thinking_fields(payload), {"thinking": {"type": "disabled"}})

    def test_drop_unsupported_payload_field_reports_budget_tokens(self) -> None:
        payload = {"thinking": {"type": "enabled", "budget_tokens": 30000}}
        dropped = drop_unsupported_payload_field(payload, "unsupported parameter: budget_tokens")
        self.assertEqual(dropped, ["thinking.budget_tokens"])
        self.assertEqual(payload, {"thinking": {"type": "enabled"}})

    def test_drop_unsupported_payload_field_reports_direct_key(self) -> None:
        payload = {"reasoning_effort": "high"}
        dropped = drop_unsupported_payload_field(payload, "unsupported parameter: reasoning_effort")
        self.assertEqual(dropped, ["reasoning_effort"])
        self.assertEqual(payload, {})

    def test_drop_unsupported_payload_field_reports_effort(self) -> None:
        payload = {"effort": "max"}
        dropped = drop_unsupported_payload_field(payload, "unsupported parameter: effort")
        self.assertEqual(dropped, ["effort"])
        self.assertEqual(payload, {})

    def test_qwen_always_thinking_rejects_toggle(self) -> None:
        with self.assertRaises(ThinkingSemanticsError):
            resolve_thinking_request_semantics(
                model="alibaba/qwen3-235b-a22b-thinking-2507",
                max_tokens=1024,
                thinking_mode="off",
                thinking_effort="auto",
            )

    def test_qwen_always_thinking_strongest_requires_no_extra_payload(self) -> None:
        semantics = resolve_thinking_request_semantics(
            model="alibaba/qwen3-235b-a22b-thinking-2507",
            max_tokens=1024,
            thinking_mode="on",
            thinking_effort="auto",
        )
        self.assertEqual(semantics["normalized_provider_thinking_request"], {})

    def test_deepseek_rejects_fake_low_effort(self) -> None:
        with self.assertRaises(ThinkingSemanticsError):
            resolve_thinking_request_semantics(
                model="deepseek-v3.2",
                max_tokens=1024,
                thinking_mode="on",
                thinking_effort="low",
            )

    def test_claude_legacy_uses_budget_and_effort(self) -> None:
        payload = build_chat_payload(
            model="anthropic/claude-opus-4.5",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8192,
            thinking_mode="on",
            thinking_effort="high",
        )
        self.assertEqual(
            extract_thinking_fields(payload),
            {"thinking": {"type": "enabled", "budget_tokens": 7168}, "effort": "high"},
        )

    def test_claude_adaptive_uses_adaptive_and_max_effort(self) -> None:
        payload = build_chat_payload(
            model="anthropic/claude-sonnet-4-6",
            messages=[{"role": "user", "content": "hello"}],
            max_tokens=8192,
            thinking_mode="on",
            thinking_effort="auto",
        )
        self.assertEqual(
            extract_thinking_fields(payload),
            {"thinking": {"type": "adaptive"}, "effort": "max"},
        )

    def test_semantic_failure_when_provider_would_drop_thinking_fields(self) -> None:
        client = UnifiedChatClient(
            api_key="test-key",
            backend="requests",
            max_retries=1,
            retry_delay=0.0,
            timeout=1.0,
        )
        with mock.patch.object(
            client,
            "_chat_with_endpoint_fallback",
            side_effect=RuntimeError("unsupported parameter: reasoning_effort"),
        ):
            with self.assertRaises(RuntimeError) as exc_ctx:
                client.chat(
                    model="qiniu/openai/gpt-5.4",
                    messages=[{"role": "user", "content": "hello"}],
                    max_tokens=1024,
                    thinking_mode="on",
                    thinking_effort="high",
                )
        self.assertIn("Thinking semantic failure", str(exc_ctx.exception))

    def test_token_param_fallback_warning_to_max_completion_tokens(self) -> None:
        client = UnifiedChatClient(
            api_key="test-key",
            backend="requests",
            max_retries=2,
            retry_delay=0.0,
            timeout=1.0,
        )
        responses = [
            RuntimeError("unsupported parameter: max_tokens; use max_completion_tokens"),
            {
                "content": "ok",
                "reasoning_content": "",
                "usage": {},
                "raw": {},
                "finish_reason": "stop",
                "api_url_used": client.request_url,
            },
        ]

        def fake_endpoint_fallback(payload):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with mock.patch.object(client, "_chat_with_endpoint_fallback", side_effect=fake_endpoint_fallback):
            response = client.chat(
                model="qiniu/openai/gpt-5.4",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=128,
                token_param_override="max_tokens",
            )
        meta = response["client_meta"]
        self.assertTrue(meta["token_param_fallback_used"])
        self.assertEqual(meta["requested_token_param_name"], "max_tokens")
        self.assertEqual(meta["effective_token_param_name"], "max_completion_tokens")
        self.assertEqual(meta["finish_reason"], "stop")

    def test_token_param_fallback_warning_to_max_tokens(self) -> None:
        client = UnifiedChatClient(
            api_key="test-key",
            backend="requests",
            max_retries=2,
            retry_delay=0.0,
            timeout=1.0,
        )
        responses = [
            RuntimeError("unsupported parameter: max_completion_tokens; use max_tokens"),
            {
                "content": "ok",
                "reasoning_content": "",
                "usage": {},
                "raw": {},
                "finish_reason": "length",
                "api_url_used": client.request_url,
            },
        ]

        def fake_endpoint_fallback(payload):
            item = responses.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

        with mock.patch.object(client, "_chat_with_endpoint_fallback", side_effect=fake_endpoint_fallback):
            response = client.chat(
                model="google/gemini-3-pro-preview",
                messages=[{"role": "user", "content": "hello"}],
                max_tokens=128,
                token_param_override="max_completion_tokens",
            )
        meta = response["client_meta"]
        self.assertTrue(meta["token_param_fallback_used"])
        self.assertEqual(meta["requested_token_param_name"], "max_completion_tokens")
        self.assertEqual(meta["effective_token_param_name"], "max_tokens")
        self.assertEqual(meta["finish_reason"], "length")

    def test_compare_thinking_fields_returns_none_for_empty_derived(self) -> None:
        self.assertIsNone(compare_thinking_fields({}, {}))

    def test_compare_thinking_fields_detects_preserved_payload(self) -> None:
        derived = {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}
        self.assertTrue(compare_thinking_fields(derived, derived))
        self.assertFalse(compare_thinking_fields(derived, {"thinking": {"type": "enabled"}}))

    def test_special_capabilities_are_marked_correctly(self) -> None:
        qwen = resolve_thinking_capability("alibaba/qwen3-235b-a22b-thinking-2507").to_summary_dict()
        gemini = resolve_thinking_capability("google/gemini-2.5-pro").to_summary_dict()
        gpt5 = resolve_thinking_capability("qiniu/openai/gpt-5.4").to_summary_dict()
        doubao = resolve_thinking_capability("volcengine/doubao-seed-2-0-pro").to_summary_dict()
        claude46 = resolve_thinking_capability("anthropic/claude-sonnet-4-6").to_summary_dict()
        self.assertEqual(qwen["off_support"], "toggle_not_supported")
        self.assertEqual(qwen["current_client_mapping"], "supported")
        self.assertEqual(gemini["off_support"], "off_not_supported")
        self.assertEqual(gpt5["off_support"], "strict_supported")
        self.assertEqual(doubao["supported_efforts"], ["low", "medium", "high"])
        self.assertEqual(claude46["supported_efforts"], ["low", "medium", "high", "max"])


if __name__ == "__main__":
    unittest.main()
