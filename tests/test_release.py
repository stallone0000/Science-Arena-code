from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from science_arena import pipeline


ROOT = Path(__file__).resolve().parents[1]


class ReleaseTests(unittest.TestCase):
    def test_api_key_is_environment_only(self) -> None:
        import os
        from unittest import mock

        with mock.patch.dict(os.environ, {"QIHOO_API_KEY": "test-sentinel"}, clear=True):
            self.assertEqual(pipeline.load_api_key(), "test-sentinel")
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                pipeline.load_api_key()

    def test_direct_png_payload_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.png"
            raw = b"\x89PNG\r\n\x1a\n" + b"official-pixel-fixture"
            path.write_bytes(raw)
            data_url = pipeline.image_data_url(path)
            import base64

            self.assertEqual(base64.b64decode(data_url.split(",", 1)[1]), raw)

    def test_offline_replay(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            completed = subprocess.run(
                [sys.executable, "scripts/reproduce_tables.py", "--output", directory],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            report = json.loads((Path(directory) / "validation.json").read_text())
            self.assertEqual(report["status"], "PASS")
            self.assertEqual(report["checks"]["table6_model_count"], 14)
            self.assertEqual(report["checks"]["cross_family_judge_count"], 6)


if __name__ == "__main__":
    unittest.main()
