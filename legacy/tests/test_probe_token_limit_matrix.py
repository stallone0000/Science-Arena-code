# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

from probe_token_limit_matrix import truncated_by_length


class ProbeTokenLimitMatrixTest(unittest.TestCase):
    def test_truncated_by_length_prefers_finish_reason(self) -> None:
        self.assertTrue(
            truncated_by_length(
                finish_reason="length",
                content="done",
                completion_tokens=10,
                requested_token_cap=100,
            )
        )

    def test_truncated_by_length_uses_completion_cap_proximity(self) -> None:
        self.assertTrue(
            truncated_by_length(
                finish_reason="stop",
                content="complete enough",
                completion_tokens=98,
                requested_token_cap=100,
            )
        )

    def test_truncated_by_length_uses_unfinished_tail(self) -> None:
        self.assertTrue(
            truncated_by_length(
                finish_reason="stop",
                content="Step 19: The final constraint implies",
                completion_tokens=40,
                requested_token_cap=100,
            )
        )


if __name__ == "__main__":
    unittest.main()
