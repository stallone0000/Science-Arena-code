#!/usr/bin/env python3
"""Draw the Section 3 subfield diagnostics as small-multiple radar plots."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


OUT = Path(__file__).resolve().parent / "subfield_diagnostics_radar.pdf"


plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rc("font", family="DejaVu Sans")
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
plt.rcParams["axes.linewidth"] = 1.3


PANELS = [
    (
        "Physics",
        "C0",
        [
            ("Thermo", 74.1),
            ("Astro", 76.8),
            ("E&M", 78.2),
            ("Mech.", 81.3),
            ("Quantum\nAtomic", 83.7),
        ],
    ),
    (
        "Chemistry",
        "C1",
        [
            ("Stereo", 34.7),
            ("Inorg.\nCoord.", 56.8),
            ("Organic\nStruct.", 71.8),
            ("Physical\nChem.", 78.4),
            ("Analytical\nSpec.", 86.6),
        ],
    ),
    (
        "Biology",
        "C2",
        [
            ("Exp.\nData", 64.3),
            ("Bioinfo.\nStats", 76.6),
            ("Animal\nNeuro", 78.6),
            ("Mol.\nGenetics", 85.8),
            ("Biochem.", 91.4),
        ],
    ),
]


def draw_panel(ax, title: str, color: str, values: list[tuple[str, float]]) -> None:
    labels = [x[0] for x in values]
    scores = [x[1] for x in values]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    closed_angles = np.r_[angles, angles[0]]
    closed_scores = np.r_[scores, scores[0]]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(
        closed_angles,
        closed_scores,
        color=color,
        linewidth=2.5,
        marker="o",
        markersize=7,
        markeredgewidth=2,
        markerfacecolor="none",
    )
    ax.fill(closed_angles, closed_scores, color=color, alpha=0.09)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=9, fontweight="bold")
    ax.set_rlabel_position(210)
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
    ax.tick_params(pad=15)
    ax.grid(True, linestyle="--", linewidth=0.9)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=34)


def main() -> None:
    fig, axes = plt.subplots(1, 3, subplot_kw={"polar": True}, figsize=(12.0, 5.1))
    for ax, (title, color, values) in zip(axes, PANELS):
        draw_panel(ax, title, color, values)
    plt.tight_layout(pad=2.0)
    plt.savefig(OUT, bbox_inches="tight")


if __name__ == "__main__":
    main()
