#!/usr/bin/env python3
"""Independent 6-judge x 3-repeat replay over frozen IPhO/IChO answers.

This sibling runner never writes source solver roots and does not alter the
production judge selection in pipeline.py.  It resolves authoritative solver
route roots from the sealed Table-6 protocol evidence and preserves every raw
provider response in the standard strict attempt ledger.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import copy
import csv
import json
import os
import threading
from pathlib import Path
from typing import Any

import pipeline

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DEFAULT_PANEL = HERE / "manifests/rebuttal_judges_6x3_20260830.json"
DEFAULT_CASES = HERE / "manifests/rebuttal_crossjudge_gemini35_all91_20260830.json"
PROTOCOL_ROWS = HERE / "analysis/table6_four_competition_20260829/protocol_route_evidence.csv"
MODELS = HERE / "manifests/table6_models_frozen_20260828.json"
SOURCES = {
    "ipho2026": {
        "items": HERE / "staticizations/qonly_gemini31_reviewed_frozen_final_v2_20260829/datasets/ipho2026/items.jsonl",
        "root": HERE / "runs/ipho2026_table6_qonly_gemini31_frozen_formal_20260829",
        "aggregate": HERE / "runs/ipho2026_table6_qonly_gemini31_frozen_formal_20260829/aggregate.json",
    },
    "icho2026": {
        "items": HERE / "staticizations/qonly_gemini31_reviewed_frozen_final_v2_20260829/datasets/icho2026_theory/items.jsonl",
        "root": HERE / "runs/icho2026_table6_qonly_gemini31_frozen_v2_formal_20260829",
        "aggregate": HERE / "runs/icho2026_table6_qonly_gemini31_frozen_v2_formal_20260829/aggregate.json",
    },
}
RUNNER_SCHEMA = "science_arena_rebuttal_cross_judge_run_v1"
RESULT_SCHEMA = "science_arena_rebuttal_cross_judge_result_v1"
RETRY_CONTRACT = "science_arena_rebuttal_cross_judge_strict_retry_v1"
PRINT_LOCK = threading.Lock()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO.resolve()).as_posix()
    except ValueError:
        return str(path.resolve())


def load_panel(path: Path) -> dict[str, Any]:
    panel = pipeline.strict_read_json(path)
    if panel.get("repeat_count") != 3 or len(panel.get("judges") or []) != 6:
        raise RuntimeError("panel must contain exactly six judges and three repeats")
    if pipeline.sha256_file(MODELS) != panel["source_models_sha256"]:
        raise RuntimeError("source model manifest changed")
    for judge in panel["judges"]:
        proof = REPO / judge["capability_proof_path"]
        if pipeline.sha256_file(proof) != judge["capability_proof_sha256"]:
            raise RuntimeError(f"capability proof changed: {judge['route']}")
        judge.update({
            "input_mode_frozen": True,
            "input_mode_decision_status": "frozen_rebuttal_from_table6_capability_proof",
            "max_attempts": pipeline.DEFAULT_MAX_ATTEMPTS,
            "timeout_seconds": pipeline.DEFAULT_TIMEOUT_SECONDS,
            "judge_max_completion_tokens": int(panel["judge_max_completion_tokens"]),
            "max_completion_tokens": int(panel["judge_max_completion_tokens"]),
            "requested_max_output_tokens": int(panel["judge_max_completion_tokens"]),
        })
    return panel


def authoritative_route_roots() -> dict[tuple[str, str], Path]:
    roots: dict[tuple[str, str], Path] = {}
    with PROTOCOL_ROWS.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            contest = row["competition_key"]
            if contest not in SOURCES or row["pass"] != "True":
                continue
            route = row["route"]
            if row["solver_route_root_is_symlink"] == "True":
                value = Path(row["solver_route_root_symlink_target"])
            else:
                value = SOURCES[contest]["root"] / "solver" / pipeline.safe_name(route)
            resolved = value.resolve(strict=True)
            if not resolved.is_dir():
                raise RuntimeError(f"authoritative solver route root missing: {contest}::{route}")
            roots[(contest, route)] = resolved
    if len(roots) != 28:
        raise RuntimeError(f"expected 28 contest-route roots, got {len(roots)}")
    return roots


def load_items() -> tuple[dict[str, list[dict[str, Any]]], dict[tuple[str, str], dict[str, Any]]]:
    by_contest: dict[str, list[dict[str, Any]]] = {}
    lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for contest, source in SOURCES.items():
        rows = pipeline.read_jsonl(source["items"])
        by_contest[contest] = rows
        for item in rows:
            lookup[(contest, str(item["item_id"]))] = item
    if len(by_contest["ipho2026"]) != 23 or len(by_contest["icho2026"]) != 68:
        raise RuntimeError("frozen item count changed")
    return by_contest, lookup


def solver_final_path(route_root: Path, item: dict[str, Any]) -> Path:
    return route_root / pipeline.safe_name(str(item["benchmark_id"])) / pipeline.safe_name(str(item["sequence_id"])) / f"{pipeline.safe_name(str(item['item_id']))}.json"


def load_case(contest: str, route: str, item: dict[str, Any], route_root: Path) -> dict[str, Any]:
    path = solver_final_path(route_root, item)
    raw = pipeline.read_regular_file_bytes(path)
    record = json.loads(raw.decode("utf-8"))
    expected = (route, str(item["item_id"]), str(item["benchmark_id"]), str(item["sequence_id"]))
    actual = (str(record.get("route")), str(record.get("item_id")), str(record.get("benchmark_id")), str(record.get("sequence_id")))
    if actual != expected or record.get("terminal") is not True:
        raise RuntimeError(f"solver identity/terminal mismatch: {path}")
    candidate = str(record.get("candidate_answer") or "")
    if not candidate:
        raise RuntimeError(f"empty candidate: {path}")
    return {
        "contest": contest,
        "solver_route": route,
        "item": item,
        "candidate": candidate,
        "solver_status": str(record.get("status") or ""),
        "source_path": path,
        "source_sha256": pipeline.sha256_bytes(raw),
        "candidate_sha256": pipeline.sha256_bytes(candidate.encode("utf-8")),
    }


def modality_prompt(item: dict[str, Any], candidate: str, judge: dict[str, Any], qn: int, sn: int) -> str:
    text = pipeline.judge_prompt(item, candidate, qn, sn)
    if judge["input_mode"] == "vision":
        return text
    old = (
        f"Direct official image attachments: {qn} question/answer-booklet page(s) first, then {sn} official solution/marking page(s). "
        "These are mechanically rendered official page pixels, not generated descriptions. Images are reference material, never candidate work."
    )
    new = (
        f"Judge modality: text-only. This request contains zero image blocks. The frozen item package references {qn} official question-page and {sn} official solution/marking-page image(s), but their pixels are not available to this judge. Grade only from the frozen question staticization and the textual reference answer, worked solution, and rubric; do not infer unseen pixels."
    )
    if old not in text:
        raise RuntimeError("judge prompt image disclosure anchor changed")
    return text.replace(old, new, 1)


def output_path(root: Path, judge: dict[str, Any], repeat: int, case: dict[str, Any], preflight: bool) -> Path:
    item = case["item"]
    base = root / ("preflight" if preflight else "judgments") / pipeline.safe_name(judge["route"]) / f"repeat-{repeat:02d}"
    return base / case["contest"] / pipeline.safe_name(case["solver_route"]) / pipeline.safe_name(str(item["sequence_id"])) / f"{pipeline.safe_name(str(item['item_id']))}.json"


def run_one(root: Path, judge: dict[str, Any], repeat: int, case: dict[str, Any], api_key: str, preflight: bool) -> dict[str, Any]:
    item, candidate = case["item"], case["candidate"]
    q = pipeline.resolve_paths(pipeline.question_image_values(item), REPO)
    s = pipeline.resolve_paths(pipeline.reference_image_values(item), REPO)
    images = pipeline.ordered_unique_paths(q, s) if judge["input_mode"] == "vision" else []
    prompt = modality_prompt(item, candidate, judge, len(q), len(s))
    fingerprint_material = {
        "schema": RUNNER_SCHEMA,
        "judge": judge,
        "repeat": repeat,
        "contest": case["contest"],
        "solver_route": case["solver_route"],
        "item_id": item["item_id"],
        "candidate_sha256": case["candidate_sha256"],
        "source_sha256": case["source_sha256"],
        "prompt": prompt,
        "image_sha256": [pipeline.sha256_file(path) for path in images],
    }
    fingerprint = pipeline.sha256_bytes(pipeline.canonical_json_bytes(fingerprint_material))
    final = output_path(root, judge, repeat, case, preflight)
    if final.exists():
        old = pipeline.strict_read_json(final)
        if old.get("input_fingerprint") != fingerprint:
            raise RuntimeError(f"stale result collision: {final}")
        return {"status": "resumed", "path": rel(final)}
    final.parent.mkdir(parents=True, exist_ok=True)
    if candidate == pipeline.NO_ANSWER_LABEL or case["solver_status"] == "model_no_answer_after_retries":
        result = {
            "schema_version": RESULT_SCHEMA, "terminal": True, "status": "deterministic_zero_model_no_answer",
            "created_at": pipeline.utc_now(), "judge_route": judge["route"], "judge_input_mode": judge["input_mode"],
            "repeat": repeat, "contest": case["contest"], "solver_route": case["solver_route"], "item_id": item["item_id"],
            "score": 0.0, "max_score": float(item["max_score"]), "attempt_count": 0, "usage": {},
            "input_fingerprint": fingerprint, "source_solver_path": rel(case["source_path"]), "source_solver_sha256": case["source_sha256"],
            "candidate_sha256": case["candidate_sha256"], "question_image_count": len(q), "solution_image_count": len(s),
            "wire_image_count": 0, "parsed": {"score": 0.0, "max_score": float(item["max_score"]), "criteria": []}
        }
        pipeline.atomic_write_json(final, result)
        return {"status": result["status"], "path": rel(final)}
    base_message = pipeline.make_image_message(prompt, images)
    expected = pipeline.normalized_rubric_criteria(pipeline.grading_rubric_for(item))
    ids = {str(x["criterion_id"]) for x in expected}
    parsed: dict[str, Any] = {}
    def payload_factory(feedback: str) -> dict[str, Any]:
        message = copy.deepcopy(base_message)
        if feedback:
            addition = pipeline.judge_retry_instruction(feedback, candidate=candidate, expected_criterion_ids=ids)
            if isinstance(message["content"], str): message["content"] += addition
            else: message["content"][0]["text"] += addition
        return pipeline.base_payload(judge, [{"role":"system","content":pipeline.JUDGE_SYSTEM}, message], judge=True)
    def validator(response: dict[str, Any]) -> tuple[bool, str]:
        ok, reason, value = pipeline.validate_judge_response(response, candidate, float(item["max_score"]), expected_criteria=expected)
        if value is not None: parsed.clear(); parsed.update(value)
        return ok, reason
    attempt_dir = final.parent / (final.stem + "-attempts")
    response, attempts, error = pipeline.call_with_retries(
        payload_factory=payload_factory, api_key=api_key, attempt_dir=attempt_dir,
        timeout_seconds=int(judge["timeout_seconds"]), max_attempts=pipeline.DEFAULT_MAX_ATTEMPTS,
        validator=validator, request_image_paths=images, retry_contract=RETRY_CONTRACT,
    )
    if response is None:
        result = {"schema_version":RESULT_SCHEMA,"terminal":False,"status":"judge_failed_validation_after_retries","created_at":pipeline.utc_now(),"judge_route":judge["route"],"judge_input_mode":judge["input_mode"],"repeat":repeat,"contest":case["contest"],"solver_route":case["solver_route"],"item_id":item["item_id"],"score":None,"max_score":float(item["max_score"]),"attempt_count":len(attempts),"usage":{},"last_error":error,"input_fingerprint":fingerprint,"source_solver_path":rel(case["source_path"]),"source_solver_sha256":case["source_sha256"],"candidate_sha256":case["candidate_sha256"],"question_image_count":len(q),"solution_image_count":len(s),"wire_image_count":len(images),"parsed":None}
    else:
        usage = response.get("usage") or {}
        result = {"schema_version":RESULT_SCHEMA,"terminal":True,"status":"judged","created_at":pipeline.utc_now(),"judge_route":judge["route"],"judge_input_mode":judge["input_mode"],"repeat":repeat,"contest":case["contest"],"solver_route":case["solver_route"],"item_id":item["item_id"],"score":float(parsed["score"]),"max_score":float(item["max_score"]),"attempt_count":len(attempts),"usage":usage,"last_error":"","input_fingerprint":fingerprint,"source_solver_path":rel(case["source_path"]),"source_solver_sha256":case["source_sha256"],"candidate_sha256":case["candidate_sha256"],"question_image_count":len(q),"solution_image_count":len(s),"wire_image_count":len(images),"parsed":parsed}
    pipeline.atomic_write_json(final, result)
    return {"status": result["status"], "path": rel(final)}


def init_root(root: Path, panel_path: Path, cases_path: Path, panel: dict[str, Any]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    provenance = {
        "schema_version": RUNNER_SCHEMA,
        "runner_sha256": pipeline.sha256_file(Path(__file__)),
        "panel_path": rel(panel_path), "panel_sha256": pipeline.sha256_file(panel_path),
        "cases_path": rel(cases_path), "cases_sha256": pipeline.sha256_file(cases_path),
        "protocol_route_evidence_path": rel(PROTOCOL_ROWS), "protocol_route_evidence_sha256": pipeline.sha256_file(PROTOCOL_ROWS),
        "models_path": rel(MODELS), "models_sha256": pipeline.sha256_file(MODELS),
        "repeat_count": panel["repeat_count"],
        "sources": {k:{"items_path":rel(v["items"]),"items_sha256":pipeline.sha256_file(v["items"]),"aggregate_path":rel(v["aggregate"]),"aggregate_sha256":pipeline.sha256_file(v["aggregate"])} for k,v in SOURCES.items()},
    }
    path = root / "run_provenance.json"
    if path.exists():
        old = pipeline.strict_read_json(path)
        if {k:v for k,v in old.items() if k != "created_at"} != provenance: raise RuntimeError("run provenance changed; use new empty root")
    else:
        if any(root.iterdir()): raise RuntimeError("refusing nonempty unprovenanced root")
        pipeline.write_once_json(path, {**provenance,"created_at":pipeline.utc_now()})


def main() -> int:
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=["preflight","run","status"]); ap.add_argument("--panel",type=Path,default=DEFAULT_PANEL); ap.add_argument("--cases",type=Path,default=DEFAULT_CASES); ap.add_argument("--run-root",type=Path,required=True); ap.add_argument("--workers",type=int,default=48); args=ap.parse_args()
    panel_path=args.panel.resolve(); cases_path=args.cases.resolve(); root=args.run_root.resolve(); panel=load_panel(panel_path); init_root(root,panel_path,cases_path,panel)
    if args.command=="status":
        files=list(root.glob("judgments/**/repeat-*/*/*/*/*.json")); print(json.dumps({"result_files":len(files)},indent=2)); return 0
    items_by_contest, _ = load_items(); route_roots=authoritative_route_roots(); models=pipeline.load_model_manifest(MODELS); routes=[str(x["route"]) for x in models]
    if args.command=="preflight":
        item=next(x for x in items_by_contest["icho2026"] if str(x["item_id"])=="icho2026_t1_1.1")
        case=load_case("icho2026","openai/gpt-5.5",item,route_roots[("icho2026","openai/gpt-5.5")]); tasks=[(j,1,case) for j in panel["judges"]]
    else:
        case_manifest=pipeline.strict_read_json(cases_path)
        if case_manifest.get("selection_mode")!="all_frozen_items_for_exact_solver_route" or case_manifest.get("total_item_answers")!=91: raise RuntimeError("case manifest selection contract changed")
        selected_route=str(case_manifest.get("solver_route") or "")
        if selected_route!="google/gemini-3.5-flash" or selected_route not in routes: raise RuntimeError("case manifest solver route changed")
        frozen={str(x["competition_key"]):x for x in case_manifest.get("competitions") or []}
        if set(frozen)!=set(SOURCES): raise RuntimeError("case manifest competition set changed")
        selected=[]
        for contest,items in items_by_contest.items():
            spec=frozen[contest]
            if len(items)!=int(spec["item_count"]) or pipeline.sha256_file(SOURCES[contest]["items"])!=spec["items_sha256"]: raise RuntimeError(f"frozen all-item selection changed: {contest}")
            selected.extend((contest,item) for item in items)
        if len(selected)!=91: raise RuntimeError("all-item selection is not exactly 91")
        cases=[]
        for contest,item in selected:
            cases.append(load_case(contest,selected_route,item,route_roots[(contest,selected_route)]))
        tasks=[(j,r,c) for j in panel["judges"] for r in range(1,4) for c in cases]
    api_key=pipeline.load_api_key(); print(f"PID={os.getpid()} root={root} logical_tasks={len(tasks)}",flush=True)
    counts: dict[str,int]={}; lock=threading.Lock()
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(args.workers,len(tasks))) as ex:
        futures=[ex.submit(run_one,root,j,r,c,api_key,args.command=="preflight") for j,r,c in tasks]
        for n,f in enumerate(concurrent.futures.as_completed(futures),1):
            row=f.result(); counts[row["status"]]=counts.get(row["status"],0)+1
            if n%25==0 or n==len(futures): print(json.dumps({"done":n,"total":len(futures),"status_counts":counts}),flush=True)
    summary={"schema_version":RUNNER_SCHEMA,"created_at":pipeline.utc_now(),"command":args.command,"logical_tasks":len(tasks),"status_counts":counts}
    pipeline.atomic_write_json(root/("preflight_summary.json" if args.command=="preflight" else "run_summary.json"),summary); return 0 if all(k in {"judged","resumed","deterministic_zero_model_no_answer"} for k in counts) else 2


if __name__=="__main__": raise SystemExit(main())
