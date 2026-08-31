# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

from image_policy import (
    resolve_generation_image_policy,
    resolve_ipho_interleaved_turn_images,
    resolve_ipho_iterative_question_images,
    resolve_standard_interleaved_background_images,
    resolve_standard_interleaved_turn_images,
    resolve_standard_iterative_question_images,
)


class ImagePolicyTest(unittest.TestCase):
    def test_unified_standard_iterative_defaults_to_global_current(self) -> None:
        policy = resolve_generation_image_policy(
            preset="unified",
            method="iterative",
            exam="cpho",
            iterative_image_mode="inherit",
            interleaved_global_image_replay="inherit",
            interleaved_replay_global_persist="inherit",
        )
        self.assertEqual(policy.iterative_image_mode, "global_current")
        resolved = resolve_standard_iterative_question_images(
            policy,
            global_question_images=["g1"],
            history_local_images=["h1"],
            current_local_images=["c1"],
        )
        self.assertEqual(resolved, ["g1", "c1"])

    def test_legacy_standard_iterative_defaults_follow_exam(self) -> None:
        cpho_policy = resolve_generation_image_policy(
            preset="legacy",
            method="iterative",
            exam="cpho",
            iterative_image_mode="inherit",
            interleaved_global_image_replay="inherit",
            interleaved_replay_global_persist="inherit",
        )
        icho_policy = resolve_generation_image_policy(
            preset="legacy",
            method="iterative",
            exam="icho",
            iterative_image_mode="inherit",
            interleaved_global_image_replay="inherit",
            interleaved_replay_global_persist="inherit",
        )
        self.assertEqual(cpho_policy.iterative_image_mode, "cumulative")
        self.assertEqual(icho_policy.iterative_image_mode, "current_only")

    def test_unified_standard_interleaved_replays_global_without_persisting(self) -> None:
        policy = resolve_generation_image_policy(
            preset="unified",
            method="interleaved",
            exam="cpho",
            iterative_image_mode="inherit",
            interleaved_global_image_replay="inherit",
            interleaved_replay_global_persist="inherit",
        )
        self.assertEqual(
            resolve_standard_interleaved_background_images(
                policy,
                global_question_images=["g1", "g2"],
            ),
            [],
        )
        turn = resolve_standard_interleaved_turn_images(
            policy,
            global_question_images=["g1", "g2"],
            current_local_images=["c1"],
        )
        self.assertEqual(turn["call_image_urls"], ["g1", "g2", "c1"])
        self.assertEqual(turn["persisted_image_urls"], ["c1"])
        self.assertEqual(turn["replay_global_image_urls"], ["g1", "g2"])

    def test_legacy_standard_interleaved_keeps_global_once_in_history(self) -> None:
        policy = resolve_generation_image_policy(
            preset="legacy",
            method="interleaved",
            exam="cpho",
            iterative_image_mode="inherit",
            interleaved_global_image_replay="inherit",
            interleaved_replay_global_persist="inherit",
        )
        self.assertEqual(
            resolve_standard_interleaved_background_images(
                policy,
                global_question_images=["g1"],
            ),
            ["g1"],
        )
        turn = resolve_standard_interleaved_turn_images(
            policy,
            global_question_images=["g1"],
            current_local_images=["c1"],
        )
        self.assertEqual(turn["call_image_urls"], ["c1"])
        self.assertEqual(turn["persisted_image_urls"], ["c1"])
        self.assertEqual(turn["replay_global_image_urls"], [])

    def test_ipho_iterative_and_interleaved_use_item_local_images(self) -> None:
        parsed_item = {
            "background_images": ["b1"],
            "question_images": ["q1", "q2"],
        }
        iterative = resolve_ipho_iterative_question_images(parsed_item)
        interleaved = resolve_ipho_interleaved_turn_images(parsed_item)
        self.assertEqual(iterative, ["b1", "q1", "q2"])
        self.assertEqual(interleaved["call_image_urls"], ["b1", "q1", "q2"])
        self.assertEqual(interleaved["persisted_image_urls"], ["b1", "q1", "q2"])


if __name__ == "__main__":
    unittest.main()
