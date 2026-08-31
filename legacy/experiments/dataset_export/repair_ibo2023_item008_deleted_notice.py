#!/usr/bin/env python3
"""Remove the deleted-question notice from IBO 2023 item 008 solver text.

The notice is preserved in metadata because it exists in the source extraction,
but it conflicts with the current 453-point scoring key where this item is
included and answerable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/science_arena_text_interleaved_v1"
TARGET_TURN = "ibo2023_item_008"
NOTICE = "THIS QUESTION HAS BEEN DELETED. PLEASE DO NOT ATTEMPT TO ANSWER IT."
REPAIR_NOTE = (
    "Public text-only repair: the source extraction contains the notice "
    "'THIS QUESTION HAS BEEN DELETED. PLEASE DO NOT ATTEMPT TO ANSWER IT.', "
    "but this item remains included in the 453-point answer key. The notice "
    "was removed from solver-visible question_text_staticized to avoid a "
    "non-scientific refusal/filtering artifact; source fidelity is preserved "
    "in this metadata note."
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
    repaired = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("turn_id") == TARGET_TURN:
            text = str(row.get("question_text_staticized") or "")
            if NOTICE in text:
                row["question_text_staticized"] = text.replace(NOTICE + "\n", "").replace(NOTICE, "").replace("\n\n\n", "\n\n")
                repaired += 1
            notes = as_list(row.get("staticization_audit_notes"))
            if REPAIR_NOTE not in notes:
                notes.append(REPAIR_NOTE)
            row["staticization_audit_notes"] = notes
            caveats = as_list(row.get("caveats"))
            caveat = "Source extraction marks this item deleted, but the current scoring key includes it; solver-visible deleted notice removed."
            if caveat not in caveats:
                caveats.append(caveat)
            row["caveats"] = caveats
            reviewer = str(row.get("staticization_reviewer_notes") or "")
            if "solver-visible deleted notice removed" not in reviewer:
                row["staticization_reviewer_notes"] = reviewer + (
                    " Public release repair: solver-visible deleted notice removed; source warning retained in audit notes."
                )
        rows.append(row)

    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    print(json.dumps({"target_turn": TARGET_TURN, "notice_removed": repaired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
