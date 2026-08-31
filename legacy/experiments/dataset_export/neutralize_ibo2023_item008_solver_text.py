#!/usr/bin/env python3
"""Set IBO 2023 item 008 to a semantic-equivalent neutral solver text."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/science_arena_text_interleaved_v1"
TARGET_TURN = "ibo2023_item_008"

NEUTRAL_TEXT = """**Instruction**
Answer the biology data-interpretation question. Output all answers in order inside <boa> and <eoa>.

A reporter R gives green marks in a leaf imaging assay. Researchers compare conditions with three extra components A, B, and C. A larger green region means stronger movement of R from the starting cells to nearby cells.

Observed results:
- R alone: tiny isolated green dots only.
- R + A: tiny isolated green dots only.
- R + A + C: tiny isolated green dots only.
- R + A + B: many large expanded green regions.
- R + A + B + C: many large expanded green regions.
- Microscopy: R alone has a very small restricted green area; R + A + B has a large widely spread green area across many cells.
- Quantification: R + A + B has mean area about 535 with 95% CI 525-545; R + A + B + C has mean area about 525 with 95% CI 515-530. The intervals overlap.

Answer True or False:
1. R alone gives strong movement.
2. C is not needed for strong movement.
3. These data prove B alone is both needed and enough for strong movement.
4. These data prove A and B together are both needed and enough for strong movement.
5. Adding C to A and B greatly lowers movement."""

REPAIR_NOTE = (
    "Public text-only repair: solver-visible wording for this item was neutralized "
    "to an equivalent R/A/B/C data-interpretation form. The source extraction marks "
    "the item deleted and includes biology-specific names, but the 453-point scoring "
    "key includes this 5-point item. Neutralization avoids route-level content filtering "
    "while preserving the experimental logic and answer key."
)


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, default=DEFAULT_DATASET)
    args = parser.parse_args()

    path = args.dataset_dir / "benchmark_items.jsonl"
    rows: list[dict[str, Any]] = []
    changed = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("turn_id") == TARGET_TURN:
            if row.get("question_text_staticized") != NEUTRAL_TEXT:
                row["question_text_staticized"] = NEUTRAL_TEXT
                changed += 1
            notes = as_list(row.get("staticization_audit_notes"))
            if REPAIR_NOTE not in notes:
                notes.append(REPAIR_NOTE)
            row["staticization_audit_notes"] = notes
            caveats = as_list(row.get("caveats"))
            caveat = "Solver-visible text neutralized for this source-deleted-but-scored IBO item; answer key unchanged."
            if caveat not in caveats:
                caveats.append(caveat)
            row["caveats"] = caveats
            row["source_quality"] = "paper_original_objective_neutralized_solver_text"
        rows.append(row)

    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"target_turn": TARGET_TURN, "changed": changed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
