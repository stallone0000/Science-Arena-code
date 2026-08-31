#!/usr/bin/env python3
"""Run three independent IPhO/IChO 2026 solver+judge replicates.

This wrapper intentionally invokes the hash-frozen Table-6 pipeline snapshot.
Every contest/replicate uses a distinct run root; existing terminal records are
only resumed inside that same root and are never borrowed from the published
Table-6 roots.  The API key remains owned by the invoked pipeline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import subprocess
import sys
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
PIPELINE = WORKSPACE / "experiments/2026_reval/snapshots/pipeline_table6_ipho_icho_formal_no_cot_20260830.py"
MODELS = WORKSPACE / "experiments/2026_reval/manifests/table6_models_frozen_20260828.json"
CAPABILITIES = WORKSPACE / "experiments/2026_reval/capabilities/table6_frozen_20260828"
ITEMS = {
    "ipho2026": WORKSPACE / "experiments/2026_reval/staticizations/qonly_gemini31_reviewed_frozen_final_v2_20260829/datasets/ipho2026/items.jsonl",
    "icho2026": WORKSPACE / "experiments/2026_reval/staticizations/qonly_gemini31_reviewed_frozen_final_v2_20260829/datasets/icho2026_theory/items.jsonl",
}
EXPECTED_SHA256 = {
    PIPELINE: "f891879b6a2e1cb9b9ff0989b35e07ed5fccadab15499e54fd7f2183243c8051",
    MODELS: "d761e8d508c6677ee3c6452123e4a3c3e8327f8dad8afde616b0277e19c54d4c",
    ITEMS["ipho2026"]: "3d9256334674a913be301e809d5baa04ae9228fda6767ce3e2cc4fed54c4a205",
    ITEMS["icho2026"]: "a0f0ef68489395e0ead74636d79f60b913d3c1ff3d9b015680d93493078f0837",
}
REPLICATES = ("repA", "repB", "repC")
JUDGE_ROUTE = "google/gemini-3.5-flash"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_root(contest: str, replicate: str) -> Path:
    return WORKSPACE / f"experiments/2026_reval/runs/{contest}_repeat_{replicate}_gemini35_judge_20260830"


def verify_frozen_inputs() -> None:
    for path, expected in EXPECTED_SHA256.items():
        actual = sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"frozen input changed: {path}: {actual} != {expected}")
    manifest = json.loads(MODELS.read_text(encoding="utf-8"))
    models = manifest["models"]
    if len(models) != 14:
        raise RuntimeError(f"expected 14 routes, found {len(models)}")
    calibrated = [row["route"] for row in models if row.get("calibrated_production_judge") is True]
    if calibrated != [JUDGE_ROUTE]:
        raise RuntimeError(f"unexpected calibrated judge: {calibrated}")
    modes = {"vision": 0, "text_staticized": 0}
    for row in models:
        modes[row["input_mode"]] += 1
        proof = CAPABILITIES / (row["route"].replace("/", "__") + ".json")
        if not proof.is_file():
            raise RuntimeError(f"missing frozen capability proof: {proof}")
    if modes != {"vision": 8, "text_staticized": 6}:
        raise RuntimeError(f"unexpected modality split: {modes}")


def common_args(contest: str, replicate: str, workers: int) -> list[str]:
    return [
        sys.executable,
        str(PIPELINE),
        "PLACEHOLDER_STAGE",
        "--workspace", str(WORKSPACE),
        "--items", str(ITEMS[contest]),
        "--models", str(MODELS),
        "--run-root", str(run_root(contest, replicate)),
        "--capability-dir", str(CAPABILITIES),
        "--workers", str(workers),
    ]


def run_stage(contest: str, replicate: str, stage: str, workers: int) -> None:
    command = common_args(contest, replicate, workers)
    command[2] = stage
    if stage == "judge":
        command.extend(["--judge-route", JUDGE_ROUTE])
    print(f"[{contest}/{replicate}] START {stage}", flush=True)
    completed = subprocess.run(command, cwd=WORKSPACE, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"{contest}/{replicate} {stage} failed with {completed.returncode}")
    print(f"[{contest}/{replicate}] DONE {stage}", flush=True)


def run_one(contest: str, replicate: str, workers: int) -> str:
    root = run_root(contest, replicate)
    root.mkdir(parents=True, exist_ok=True)
    for stage in ("validate", "solve", "judge", "aggregate"):
        run_stage(contest, replicate, stage, workers)
    aggregate = json.loads((root / "aggregate.json").read_text(encoding="utf-8"))
    expected_rows = 14 * (23 if contest == "ipho2026" else 68)
    if not aggregate.get("complete") or aggregate.get("missing_count") != 0:
        raise RuntimeError(f"{contest}/{replicate}: incomplete aggregate")
    if aggregate.get("row_count") != expected_rows:
        raise RuntimeError(
            f"{contest}/{replicate}: row_count={aggregate.get('row_count')} != {expected_rows}"
        )
    return f"{contest}/{replicate}: complete"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--replicates", default=",".join(REPLICATES))
    parser.add_argument("--contests", default="ipho2026,icho2026")
    parser.add_argument("--workers", type=int, default=28)
    parser.add_argument("--parallel-jobs", type=int, default=6)
    args = parser.parse_args()
    verify_frozen_inputs()
    replicates = [value.strip() for value in args.replicates.split(",") if value.strip()]
    contests = [value.strip() for value in args.contests.split(",") if value.strip()]
    if any(value not in REPLICATES for value in replicates):
        raise ValueError(f"replicates must be selected from {REPLICATES}")
    if any(value not in ITEMS for value in contests):
        raise ValueError(f"contests must be selected from {tuple(ITEMS)}")
    jobs = [(contest, replicate) for replicate in replicates for contest in contests]
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.parallel_jobs, len(jobs))) as executor:
        futures = {
            executor.submit(run_one, contest, replicate, args.workers): (contest, replicate)
            for contest, replicate in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            contest, replicate = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                print(f"{contest}/{replicate}: FAILED: {exc}", file=sys.stderr, flush=True)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
