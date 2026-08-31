#!/usr/bin/env python3
"""Convenience wrapper around the sealed 2026 ScienceArena runner."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = ROOT / "artifacts/inputs-2026"
ITEMS = {
    "ipho2026": "experiments/2026_reval/staticizations/qonly_gemini31_reviewed_frozen_final_v2_20260829/datasets/ipho2026/items.jsonl",
    "icho2026": "experiments/2026_reval/staticizations/qonly_gemini31_reviewed_frozen_final_v2_20260829/datasets/icho2026_theory/items.jsonl",
    "nbpho2026": "experiments/2026_reval/staticizations/qonly_gemini31_reviewed_frozen_final_v2_20260829/datasets/nbpho2026/items.jsonl",
    "usapho2026": "experiments/2026_reval/staticizations/qonly_gemini31_reviewed_frozen_final_v2_20260829/datasets/usapho2026/items.jsonl",
}
MODELS = "experiments/2026_reval/manifests/table6_models_frozen_20260828.json"
FROZEN_CAPABILITIES = "experiments/2026_reval/capabilities/table6_frozen_20260828"
PROBE_META = "experiments/2026_reval/assets/probes/official_ipho2026_t1a_fig1a.json"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["validate", "probe", "solve", "judge", "aggregate"])
    parser.add_argument("--contest", required=True, choices=sorted(ITEMS))
    parser.add_argument("--workspace", type=Path, default=DEFAULT_WORKSPACE)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--model-routes", default="")
    parser.add_argument("--fresh-capabilities", action="store_true")
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    if not workspace.is_dir():
        raise SystemExit(
            f"missing input workspace {workspace}; run download_artifacts.py --name inputs-2026 --extract"
        )
    run_root = args.run_root or Path("runs") / args.contest
    resolved_run_root = (
        Path(run_root).resolve()
        if Path(run_root).is_absolute()
        else (workspace / run_root).resolve()
    )
    if workspace != resolved_run_root and workspace not in resolved_run_root.parents:
        raise SystemExit("--run-root must remain inside --workspace")
    resolved_run_root.parent.mkdir(parents=True, exist_ok=True)
    capability_dir = (
        Path(run_root) / "capabilities" if args.fresh_capabilities else Path(FROZEN_CAPABILITIES)
    )
    command = [
        sys.executable,
        "-m",
        "science_arena",
        args.command,
        "--workspace",
        str(workspace),
        "--items",
        ITEMS[args.contest],
        "--models",
        MODELS,
        "--run-root",
        str(run_root),
        "--capability-dir",
        str(capability_dir),
        "--workers",
        str(args.workers),
    ]
    if args.model_routes:
        command.extend(["--model-routes", args.model_routes])
    if args.command == "judge":
        command.extend(["--judge-route", "google/gemini-3.5-flash"])
    if args.command == "probe":
        meta = json.loads((workspace / PROBE_META).read_text(encoding="utf-8"))
        command.extend(
            [
                "--probe-image",
                meta["asset_path"],
                "--probe-expected-regex",
                meta["accepted_response_patterns"][0],
                "--probe-question",
                meta["probe_question"],
            ]
        )
    return subprocess.run(command, cwd=ROOT, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
