#!/usr/bin/env python3
"""Draw EMNLP judge-calibration bar figures."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent

RUNS = {
    "ipho": {
        "Gemini 3.5 Flash": ROOT
        / "experiments/judge_pipeline_v2/runs/merged_ipho_gemini35_v6/comparison_to_expert.json",
        "Gemini 3.1 Pro": ROOT
        / "experiments/judge_pipeline_v2/runs/merged_ipho_gemini31pro_strictnotes_v2_reparsed/comparison_to_expert.json",
    },
    "icho": {
        "Gemini 3.5 Flash": ROOT
        / "experiments/judge_pipeline_v2/runs/merged_icho_gemini35_v8_reparsed/comparison_to_expert.json",
        "Gemini 3.1 Pro": ROOT
        / "experiments/judge_pipeline_v2/runs/fullcalib_icho_gemini31pro_strictnotes_v4_reparsed/comparison_to_expert.json",
    },
}

MODEL_ORDER = [
    "gemini-2.5-pro",
    "gpt-5",
    "glm-4.5",
    "deepseek-v3.1",
    "qwen3-235b-a22b",
]

DISPLAY_NAMES = {
    "gemini-2.5-pro": "Gemini\n2.5 Pro",
    "gpt-5": "GPT-5",
    "glm-4.5": "GLM\n4.5",
    "deepseek-v3.1": "DeepSeek\nV3.1",
    "qwen3-235b-a22b": "Qwen3\n235B",
}

SERIES = [
    ("Human Ground Truth", "C0"),
    ("Gemini 3.1 Pro", "C1"),
    ("Gemini 3.5 Flash", "C2"),
]


def set_style() -> None:
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rc("font", family="DejaVu Sans")
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42
    plt.rcParams["axes.linewidth"] = 1.3


def load_exam(exam: str) -> dict[str, dict[str, float]]:
    by_series: dict[str, dict[str, float]] = {}
    human: dict[str, float] = {}
    for judge_name, path in RUNS[exam].items():
        data = json.loads(path.read_text(encoding="utf-8"))
        judge_scores = {row["model"]: float(row["judge"]) for row in data["rows"]}
        by_series[judge_name] = judge_scores
        for row in data["rows"]:
            human.setdefault(row["model"], float(row["expert"]))
    by_series["Human Ground Truth"] = human
    return by_series


def draw_exam(exam: str, title: str, ylabel: str, ymax: float, output: str) -> None:
    set_style()
    values = load_exam(exam)
    x = np.arange(len(MODEL_ORDER))
    width = 0.24

    fig, ax = plt.subplots(1, 1, figsize=(7.2, 3.6))
    offsets = [-width, 0.0, width]
    for (label, color), offset in zip(SERIES, offsets, strict=True):
        heights = [values[label][model] for model in MODEL_ORDER]
        bars = ax.bar(
            x + offset,
            heights,
            width=width,
            color=color,
            edgecolor="black",
            linewidth=1.0,
            label=label,
            zorder=3,
        )
        for bar, height in zip(bars, heights, strict=True):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + ymax * 0.015,
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold",
                rotation=90,
            )

    ax.set_title(title, fontweight="bold", fontsize=20, pad=8)
    ax.set_ylabel(ylabel, fontweight="bold", fontsize=22)
    ax.set_xlabel("Archived Model Answer", fontweight="bold", fontsize=21)
    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES[model] for model in MODEL_ORDER], fontweight="bold")
    ax.tick_params(labelsize=15)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.6, len(MODEL_ORDER) - 0.4)
    ax.grid(True, linestyle="--", axis="y", zorder=0)
    ax.grid(True, linestyle="--", axis="x", zorder=0)
    plt.tight_layout()
    fig.savefig(OUT_DIR / output, bbox_inches="tight")
    plt.close(fig)


def draw_legend(output: str) -> None:
    set_style()
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 0.58))
    handles = [
        Patch(facecolor=color, edgecolor="black", linewidth=1.0, label=label)
        for label, color in SERIES
    ]
    legend = ax.legend(
        handles=handles,
        loc="center",
        ncol=3,
        frameon=False,
        handlelength=1.8,
        columnspacing=1.5,
    )
    plt.setp(legend.get_texts(), fontweight="bold", fontsize=14)
    ax.axis("off")
    fig.savefig(OUT_DIR / output, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    draw_legend("judge_calibration_caption.pdf")
    draw_exam("ipho", "IPhO 2025", "Score / 30", 30.0, "judge_calibration_ipho.pdf")
    draw_exam("icho", "IChO 2025", "Score / 60", 60.0, "judge_calibration_icho.pdf")


if __name__ == "__main__":
    main()
