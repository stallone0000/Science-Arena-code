#!/usr/bin/env python3
"""Parse 360-model.md into machine-readable model pricing."""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRICE_RE = re.compile(r"¥\s*([0-9.]+)\s*/\s*1M tokens", re.I)


def parse_number(text: str) -> int:
    return int(text.strip().replace(",", ""))


def parse_price(text: str) -> float:
    match = PRICE_RE.search(text)
    if not match:
        raise ValueError(f"Cannot parse price: {text}")
    return float(match.group(1))


def main() -> None:
    path = ROOT / "360-model.md"
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = {}
    i = 0
    while i + 4 < len(lines):
        name = lines[i]
        if "/" not in name and not name.startswith("360"):
            i += 1
            continue
        try:
            tpm = parse_number(lines[i + 1])
            rpm = parse_number(lines[i + 2])
            input_price = parse_price(lines[i + 3])
            output_price = parse_price(lines[i + 4])
        except Exception:
            i += 1
            continue
        rows[name] = {
            "tpm": tpm,
            "rpm": rpm,
            "input_price_cny_per_1m": input_price,
            "output_price_cny_per_1m": output_price,
        }
        i += 5
    out = ROOT / "experiments/judge_pipeline_v2/model_prices_360.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {out} with {len(rows)} models")


if __name__ == "__main__":
    main()
