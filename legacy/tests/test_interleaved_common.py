# -*- coding: utf-8 -*-

from __future__ import annotations

import unittest

from interleaved_common import build_invocation_metadata, redact_argv


class InterleavedCommonTest(unittest.TestCase):
    def test_redact_argv_supports_equals_and_split_forms(self) -> None:
        argv = [
            "python",
            "tool.py",
            "--api_key=secret-a",
            "--other=x",
            "--api_key",
            "secret-b",
        ]
        self.assertEqual(
            redact_argv(argv),
            [
                "python",
                "tool.py",
                "--api_key=<redacted>",
                "--other=x",
                "--api_key",
                "<redacted>",
            ],
        )

    def test_build_invocation_metadata_redacts_argv(self) -> None:
        meta = build_invocation_metadata(
            argv=["python", "tool.py", "--api_key=secret-value"],
            entrypoint="/tmp/tool.py",
            cwd="/tmp",
        )
        self.assertEqual(meta["argv"], ["python", "tool.py", "--api_key=<redacted>"])
        self.assertEqual(meta["entrypoint"], "/tmp/tool.py")
        self.assertEqual(meta["cwd"], "/tmp")


if __name__ == "__main__":
    unittest.main()
