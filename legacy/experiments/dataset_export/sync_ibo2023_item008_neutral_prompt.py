#!/usr/bin/env python3
"""Sync the solver-visible prompt for IBO 2023 item 008.

The row had already been neutralized in ``question_text_staticized`` but the
evaluator reads ``prompt``. Keep source fidelity in ``original_prompt`` and
metadata while ensuring downstream solver calls see only the neutral text.
"""

from __future__ import annotations

import json
from pathlib import Path


DATASET = Path("datasets/science_arena_text_interleaved_v1/benchmark_items.jsonl")


def main() -> None:
    rows = []
    changed = 0
    for line in DATASET.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("benchmark_id") == "ibo2023" and row.get("sequence_id") == "ibo2023_item_008":
            neutral = row.get("question_text_staticized")
            if not neutral:
                raise SystemExit("IBO item 008 has no question_text_staticized neutral text")
            if row.get("prompt") != neutral:
                row["prompt"] = neutral
                notes = list(row.get("staticization_audit_notes") or [])
                note = (
                    "Public text-only repair: prompt was synced to the neutralized "
                    "question_text_staticized because the evaluator uses prompt as "
                    "the solver-visible field."
                )
                if note not in notes:
                    notes.append(note)
                row["staticization_audit_notes"] = notes
                changed += 1
        rows.append(row)
    DATASET.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"dataset": str(DATASET), "changed": changed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
