#!/usr/bin/env python3
"""Rebuild paper-facing tables from frozen ScienceArena result artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def paper_main_rows() -> list[list[float]]:
    tex = (ROOT / "paper/source/latex/science-arena-CR-final.tex").read_text(
        encoding="utf-8"
    )
    block = tex[tex.index(r"\begin{table*}[t]", tex.index("tab:main_results")) :]
    block = block[: block.index(r"\label{tab:latest-all-benchmarks}")]
    rows: list[list[float]] = []
    for line in block.splitlines():
        if (
            r"\textbf{" not in line
            or " & " not in line
            or not re.search(r"\(\d{2}-\d{2}\)\s*&", line)
            or not line.rstrip().endswith(r"\\")
        ):
            continue
        cells = line.split(" & ")[1:]
        if len(cells) != 14:
            continue
        values = []
        for cell in cells[:13]:
            match = re.search(r"-?\d+(?:\.\d+)?", cell)
            if not match:
                raise RuntimeError(f"cannot parse paper table cell: {cell}")
            values.append(float(match.group()))
        rows.append(values)
    if len(rows) != 14:
        raise RuntimeError(f"expected 14 main-table model rows, found {len(rows)}")
    return rows


def build_table6() -> tuple[list[dict[str, Any]], list[str]]:
    data = load_json("results/2026/table6/table6_scores.json")
    paper = paper_main_rows()
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, source in enumerate(data["rows"]):
        contests = source["competitions"]
        row = {
            "order": source["table6_order"],
            "model": source["display_name"],
            "route": source["route"],
            "input_mode": source["input_mode"],
            "nbpho2026": contests["nbpho2026"]["score"],
            "usapho2026": contests["usapho2026"]["score"],
            "ipho2026": contests["ipho2026"]["score"],
            "icho2026": contests["icho2026"]["score"],
            "four_contest_macro_percent": source["macro_mean_percent"],
        }
        rows.append(row)
        expected = paper[index][9:13]
        actual = [row[k] for k in ("nbpho2026", "usapho2026", "ipho2026", "icho2026")]
        for contest, left, right in zip(
            ("NBPhO", "USAPhO", "IPhO", "IChO"), actual, expected
        ):
            if round(float(left) + 1e-10, 2) != round(float(right) + 1e-10, 2):
                errors.append(
                    f"row {index + 1} {contest}: artifact {left:.6f} != paper {right:.2f}"
                )
    return rows, errors


def build_stability() -> list[dict[str, Any]]:
    data = load_json("results/2026/solver_stability_all_models.json")
    rows = []
    for model in data["table8_stability"]:
        row = {"model": model["display_name"], "route": model["route"]}
        for contest in ("ipho2026", "icho2026"):
            values = model["contests"][contest]
            row[f"{contest}_mean"] = values["mean"]
            row[f"{contest}_sample_sd"] = values["sample_sd"]
            row[f"{contest}_range"] = values["range"]
        rows.append(row)
    return rows


def build_crossjudge() -> list[dict[str, Any]]:
    data = load_json("results/2026/crossjudge/summary.json")
    totals = data["contest_total_mean_sample_sd"]
    reliability = {
        row["judge_route"]: row
        for row in data["repeat_reliability"]
        if row["subset"] == "overall"
    }
    by_route: dict[str, dict[str, Any]] = {}
    for source in totals:
        route = source["judge_route"]
        row = by_route.setdefault(
            route,
            {
                "judge_route": route,
                "input_mode": reliability[route]["judge_input_mode"],
                "within_judge_mae_pp": 100
                * reliability[route]["mean_pairwise_repeat_mae_normalized"],
                "exact_three_repeat_agreement_percent": 100
                * reliability[route]["exact_three_repeat_agreement_rate"],
            },
        )
        contest = source["contest"]
        row[f"{contest}_mean"] = source["mean_total"]
        row[f"{contest}_sample_sd"] = source["sample_sd_total"]
    order = [j["route"] for j in load_json("configs/judges_cross_family.json")["judges"]]
    return [by_route[route] for route in order]


def build_calibration() -> list[dict[str, Any]]:
    data = load_json(
        "results/archived/calibration_replay/archived_two_gemini_calibration_replay_20260829.json"
    )["item_level_expert_agreement"]["by_run"]
    display = {
        "gemini_3_1_pro_preview": "Gemini 3.1 Pro",
        "gemini_3_5_flash": "Gemini 3.5 Flash",
    }
    rows = []
    for key in sorted(data):
        judge, exam = key.split("__", 1)
        raw = data[key]["raw_points"]
        norm = data[key]["fraction_of_archived_row_max"]
        rows.append(
            {
                "judge": display[judge],
                "exam": exam,
                "n": raw["denominator"],
                "raw_mae": raw["mean_absolute_error"],
                "normalized_mae": norm["mean_absolute_error"],
                "pearson_r": norm["pearson_r"],
            }
        )
    return rows


def build_modality() -> list[dict[str, Any]]:
    text = (ROOT / "results/2026/modality_ablation.md").read_text(encoding="utf-8")
    labels = {
        "native": "Native image",
        "caption": "Faithful textual description",
        "blind": "No visual information",
    }
    rows = []
    for key, label in labels.items():
        match = re.search(rf"\| {re.escape(key)} \| (\d+) \| ([0-9.]+) \|", text)
        if not match:
            raise RuntimeError(f"cannot parse modality arm {key}")
        rows.append(
            {"condition": label, "paired_cells": int(match.group(1)), "mean_percent": float(match.group(2))}
        )
    return rows


def markdown_table(rows: list[dict[str, Any]]) -> str:
    fields = list(rows[0])
    lines = ["| " + " | ".join(fields) + " |", "|" + "---|" * len(fields)]
    for row in rows:
        values = []
        for field in fields:
            value = row[field]
            values.append(f"{value:.4f}" if isinstance(value, float) else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "reproduced")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    table6, errors = build_table6()
    products = {
        "main_table_2026": table6,
        "four_contest_macro_ci": load_json(
            "results/2026/table6/table6_bootstrap_macro_ci.json"
        )["rows"],
        "solver_stability": build_stability(),
        "cross_family_judges": build_crossjudge(),
        "judge_calibration": build_calibration(),
        "modality_ablation": build_modality(),
    }
    for name, rows in products.items():
        write_csv(output / f"{name}.csv", rows)

    validation = {
        "status": "PASS" if not errors else "FAIL",
        "checks": {
            "four_2026_columns_match_final_latex_at_two_decimals": not errors,
            "table6_model_count": len(table6),
            "stability_model_count": len(products["solver_stability"]),
            "cross_family_judge_count": len(products["cross_family_judges"]),
            "calibration_row_count": len(products["judge_calibration"]),
        },
        "errors": errors,
    }
    (output / "validation.json").write_text(
        json.dumps(validation, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report = ["# ScienceArena offline replay", ""]
    for name, rows in products.items():
        report.extend([f"## {name.replace('_', ' ').title()}", "", markdown_table(rows), ""])
    report.extend(["## Validation", "", f"Status: **{validation['status']}**", ""])
    if errors:
        report.extend([f"- {error}" for error in errors])
    (output / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"{validation['status']}: wrote {len(products)} tables to {output}")
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
