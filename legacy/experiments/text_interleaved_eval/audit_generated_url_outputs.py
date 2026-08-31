#!/usr/bin/env python3
"""Audit generated answers for URL/image-markdown strings.

This is separate from input-payload image leakage. A model can hallucinate a
URL in its own answer even when the benchmark prompt was text-only. We keep
that as an auditable quality flag rather than conflating it with API payload
fairness.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


PATTERNS = [
    "http://",
    "https://",
    "data:image",
    "image_url",
    "myqcloud.com",
    "cos.ap-",
]
MARKDOWN_IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]+\)")


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_result_files(path: Path):
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(path.glob("*/turn_results.jsonl"))


def audit_answer(answer: str) -> tuple[list[str], list[str]]:
    hits = [pattern for pattern in PATTERNS if pattern in answer]
    md_images = MARKDOWN_IMAGE_RE.findall(answer)
    if md_images and "markdown_image" not in hits:
        hits.append("markdown_image")
    snippets: list[str] = []
    for pattern in PATTERNS:
        index = answer.find(pattern)
        if index >= 0:
            start = max(0, index - 80)
            end = min(len(answer), index + 180)
            snippets.append(answer[start:end].replace("\n", " "))
    for md_image in md_images[:3]:
        snippets.append(md_image[:240])
    return hits, snippets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_root", type=Path, default=Path("artifacts/runs/text_interleaved_eval"))
    parser.add_argument("--glob", default="text_staticized_*_20260524_v2/*/turn_results.jsonl")
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for matched in sorted(args.run_root.glob(args.glob)):
        for path in iter_result_files(matched):
            if not path.exists():
                continue
            for row in iter_jsonl(path):
                answer = str(row.get("candidate_answer") or "")
                hits, snippets = audit_answer(answer)
                if not hits:
                    continue
                rows.append(
                    {
                        "run_file": str(path),
                        "benchmark_id": row.get("benchmark_id"),
                        "model": row.get("model"),
                        "request_model": row.get("request_model"),
                        "sequence_id": row.get("sequence_id"),
                        "turn_id": row.get("turn_id"),
                        "hits": hits,
                        "snippets": snippets,
                    }
                )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "generated_url_audit.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    with (args.output_dir / "generated_url_audit.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["benchmark_id", "model", "request_model", "sequence_id", "turn_id", "hits", "snippet"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "benchmark_id": row["benchmark_id"],
                    "model": row["model"],
                    "request_model": row["request_model"],
                    "sequence_id": row["sequence_id"],
                    "turn_id": row["turn_id"],
                    "hits": ";".join(row["hits"]),
                    "snippet": " || ".join(row["snippets"]),
                }
            )
    by_cell: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (str(row.get("benchmark_id")), str(row.get("model")))
        by_cell[key] = by_cell.get(key, 0) + 1
    summary = {
        "row_count": len(rows),
        "cell_counts": [
            {"benchmark_id": bench, "model": model, "count": count}
            for (bench, model), count in sorted(by_cell.items())
        ],
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
