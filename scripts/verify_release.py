#!/usr/bin/env python3
"""Fail closed on credentials, hidden reasoning, broken JSON, and large blobs."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", "artifacts", "reproduced", ".venv"}
BINARY = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".zst"}
PATTERNS = {
    "360 API key": re.compile(rb"fk\d{6,}\.[A-Za-z0-9_-]{12,}"),
    "OpenAI-style key": re.compile(rb"sk-[A-Za-z0-9_-]{20,}"),
    "GitHub token": re.compile(rb"gh[opusr]_[A-Za-z0-9]{20,}"),
    "private key": re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}
FORBIDDEN_REASONING_KEYS = re.compile(
    rb'"(?:reasoning_content|redacted_reasoning|chain_of_thought|thinking_content)"\s*:'
)


def main() -> int:
    errors: list[str] = []
    files = [
        path
        for path in ROOT.rglob("*")
        if path.is_file() and not (set(path.relative_to(ROOT).parts) & SKIP_PARTS)
    ]
    for path in files:
        relative = path.relative_to(ROOT)
        if relative.name in {".env", "API_KEY.sh"}:
            errors.append(f"forbidden credential file: {relative}")
        if path.stat().st_size >= 95 * 1024 * 1024:
            errors.append(f"oversized Git candidate: {relative}")
        if path.suffix.lower() in BINARY:
            continue
        raw = path.read_bytes()
        for label, pattern in PATTERNS.items():
            if pattern.search(raw):
                errors.append(f"{label} pattern in {relative}")
        if relative.parts and relative.parts[0] == "results" and FORBIDDEN_REASONING_KEYS.search(raw):
            errors.append(f"hidden reasoning field in released result: {relative}")
        if path.suffix in {".json", ".jsonl"}:
            try:
                if path.suffix == ".json":
                    json.loads(raw)
                else:
                    for line in raw.splitlines():
                        if line.strip():
                            json.loads(line)
            except Exception as exc:
                errors.append(f"invalid {path.suffix}: {relative}: {exc}")
    if errors:
        print("FAIL")
        for error in errors:
            print(f"- {error}")
        return 2
    print(f"PASS: scanned {len(files)} repository files; no credential or release-policy violation")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
