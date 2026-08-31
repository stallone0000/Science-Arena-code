# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

from model_registry import canonical_model_name, resolve_model_profile


class ModelRegistryTest(unittest.TestCase):
    def test_qiniu_openai_gpt54_is_explicit_alias(self) -> None:
        resolved = resolve_model_profile("qiniu/openai/gpt-5.4")
        self.assertEqual(canonical_model_name("qiniu/openai/gpt-5.4"), "gpt-5.4")
        self.assertEqual(resolved.matched_via, "alias")
        self.assertEqual(resolved.profile.profile_id, "qiniu_openai_gpt54")
        self.assertEqual(resolved.profile.preferred_model, "qiniu/openai/gpt-5.4")
        self.assertEqual(resolved.profile.token_limit_mode, "max_completion_tokens")
        self.assertTrue(resolved.profile.multimodal_default)

    def test_openai_gpt54_and_gpt55_are_explicit_aliases(self) -> None:
        resolved54 = resolve_model_profile("openai/gpt-5.4")
        resolved55 = resolve_model_profile("openai/gpt-5.5")
        self.assertEqual(resolved54.profile.profile_id, "openai_gpt54")
        self.assertEqual(resolved55.profile.profile_id, "openai_gpt55")
        self.assertEqual(resolved54.profile.preferred_model, "openai/gpt-5.4")
        self.assertEqual(resolved55.profile.preferred_model, "openai/gpt-5.5")

    def test_claude_adaptive_models_are_explicit(self) -> None:
        resolved_sonnet = resolve_model_profile("anthropic/claude-sonnet-4-6")
        resolved_opus46 = resolve_model_profile("anthropic/claude-opus-4.6")
        resolved_opus47 = resolve_model_profile("anthropic/claude-opus-4.7")
        self.assertEqual(resolved_sonnet.profile.profile_id, "claude_adaptive_data_url_variants")
        self.assertEqual(resolved_opus46.profile.profile_id, "claude_adaptive_data_url_variants")
        self.assertEqual(resolved_opus47.profile.profile_id, "claude_adaptive_data_url_variants")

    def test_token_limit_mode_expectations(self) -> None:
        self.assertEqual(resolve_model_profile("qiniu/openai/gpt-5.4").profile.token_limit_mode, "max_completion_tokens")
        self.assertEqual(resolve_model_profile("openai/gpt-5.5").profile.token_limit_mode, "max_completion_tokens")
        self.assertEqual(resolve_model_profile("anthropic/claude-sonnet-4-6").profile.token_limit_mode, "max_tokens")


if __name__ == "__main__":
    unittest.main()
