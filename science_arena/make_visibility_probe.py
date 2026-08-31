#!/usr/bin/env python3
"""Create the deterministic, training-time-impossible image visibility probe.

This image is used only to verify that a provider route actually exposes image
pixels to the model.  Formal benchmark turns always use mechanically rendered
official competition pages instead.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1200
HEIGHT = 700
TOKEN = "Q7MX-4829"
BLUE_TRIANGLES = 4
RED_ARROW_DIRECTION = "northwest"


def font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    )
    for candidate in candidates:
        if candidate.is_file():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    output_dir = Path(__file__).resolve().parent / "assets" / "probes"
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "synthetic_visibility_q7mx_4829.png"

    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((35, 35, WIDTH - 35, HEIGHT - 35), radius=24, outline="black", width=8)
    draw.text((90, 70), TOKEN, fill="black", font=font(96))
    draw.text((90, 205), "Count the blue triangles and read the red arrow.", fill="black", font=font(38))

    triangle_x = (120, 300, 480, 660)
    for x in triangle_x:
        draw.polygon(((x, 550), (x + 70, 390), (x + 140, 550)), fill=(20, 90, 220), outline="black")

    # Red northwest arrow: shaft from southeast to northwest, with two head wings.
    draw.line((1080, 570, 850, 340), fill=(220, 25, 25), width=24)
    draw.line((850, 340, 880, 440), fill=(220, 25, 25), width=24)
    draw.line((850, 340, 950, 370), fill=(220, 25, 25), width=24)
    image.save(output, format="PNG", optimize=False)

    manifest = {
        "schema_version": "science_arena_synthetic_visibility_probe_v1",
        "purpose": "transport/content visibility only; never a formal benchmark question asset",
        "path": str(output.relative_to(Path.cwd())) if output.is_relative_to(Path.cwd()) else str(output),
        "sha256": sha256(output),
        "width_px": WIDTH,
        "height_px": HEIGHT,
        "expected": {
            "token": TOKEN,
            "blue_triangles": BLUE_TRIANGLES,
            "red_arrow_direction": RED_ARROW_DIRECTION,
        },
        "generation": "deterministic Pillow vector primitives; no model generation",
    }
    manifest_path = output.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
