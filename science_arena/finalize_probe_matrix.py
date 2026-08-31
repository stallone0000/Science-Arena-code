#!/usr/bin/env python3
"""Re-evaluate saved visibility probes without issuing any API requests.

The transport runner deliberately saves the verbatim visible answer and raw
provider response.  This small offline pass applies answer-equivalence rules
that are broader than a single regular expression (for example ``M N O`` vs
``M,N,O`` and ``1/2 a`` vs ``\\frac{1}{2}a``) while leaving the raw evidence
untouched.
"""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def compact_math(text: str) -> str:
    value = text.casefold()
    value = value.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    value = re.sub(r"\\frac\s*\{\s*1\s*\}\s*\{\s*2\s*\}", "1/2", value)
    value = value.replace("½", "1/2")
    value = re.sub(r"[\s,;|:_`$\\(){}\[\].]+", "", value)
    return value


def ipho_wall_probe_passed(text: str) -> bool:
    normalized = compact_math(text)
    return "mno" in normalized and ("1/2a" in normalized or "a/2" in normalized)


def classify(proof: dict[str, Any]) -> dict[str, Any]:
    text_probe = proof.get("text_probe") or {}
    vision_probe = proof.get("vision_probe") or {}
    visible = str(vision_probe.get("visible_content") or "")
    http_status = vision_probe.get("http_status")
    vision_content_passed = bool(http_status == 200 and ipho_wall_probe_passed(visible))
    text_content_passed = bool(text_probe.get("passed"))
    return {
        "route": proof.get("route"),
        "text_transport_passed": text_content_passed,
        "text_probe_http_status": text_probe.get("http_status"),
        "text_probe_error": text_probe.get("error") or "",
        "vision_transport_passed": http_status == 200,
        "vision_content_passed": vision_content_passed,
        "vision_probe_http_status": http_status,
        "vision_probe_finish_reason": vision_probe.get("finish_reason") or "",
        "vision_visible_content": visible,
        "vision_probe_error": vision_probe.get("error") or "",
        "raw_runner_regex_passed": bool(vision_probe.get("passed")),
        "normalization_rule": "ipho_wall_labels_MNO_and_half_a_v1",
        "provisional_input_mode": "vision" if vision_content_passed else "text_staticized",
        "final_gate_passed": False,
        "final_gate_note": (
            "P1 real-image content pass only; P2/P3 multi-image and no-image negative controls still required"
            if vision_content_passed
            else "No usable visible answer on P1; route remains text_staticized unless later full gate passes"
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("capability_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    capability_dir = args.capability_dir.resolve()
    rows = []
    for path in sorted(capability_dir.glob("*.json")):
        value = read_json(path)
        if isinstance(value, dict) and value.get("route"):
            rows.append(classify(value))
    if not rows:
        raise RuntimeError(f"No per-route proof JSON found in {capability_dir}")
    output = args.output or capability_dir / "normalized_p1_matrix.json"
    document = {
        "schema_version": "science_arena_route_probe_normalized_p1_v1",
        "created_at": utc_now(),
        "source_capability_dir": str(capability_dir),
        "model_count": len(rows),
        "vision_content_pass_count": sum(bool(row["vision_content_passed"]) for row in rows),
        "rows": rows,
    }
    write_json(output.resolve(), document)
    print(
        json.dumps(
            {
                "model_count": document["model_count"],
                "vision_content_pass_count": document["vision_content_pass_count"],
                "output": str(output.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
