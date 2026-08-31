#!/usr/bin/env python3
"""Build a Hugging Face-ready Science Arena dataset export.

The export is intentionally conservative: every row is an interleaved turn.
Rows from the same large problem share `sequence_id` and increasing
`turn_index`; a runner can replay prior turns in sequence order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
from concurrent.futures import ThreadPoolExecutor


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = ROOT / "datasets/science_arena_hf"


LONG_FORM_JSONS = {
    "icho2025": ROOT / "data/science_arena/icho2025/fix_ICHO-2025-latex.json",
    "ccho2025": ROOT / "data/science_arena/ccho2025/fix_CCHO-2025-latex.json",
    "cpho2025": ROOT / "data/science_arena/cpho2025/fix_CPHO-2025-latex.json",
    "eupho2025": ROOT / "data/science_arena/eupho2025/fix_EuPhO-2025-latex.json",
    "apho2025": ROOT / "data/science_arena/apho2025/fix_APHO-2025-latex.json",
    "nbpho2025": ROOT / "data/science_arena/nbpho2025/fix_NBPhO-2025-latex.json",
}

PAGE_RANGE_CONFIGS = {
    "incho2026": [
        ROOT / f"data/science_arena/candidate_benchmarks/structured/hbcse2026/incho2026_p{i}_render_v1.json"
        for i in range(1, 6)
    ],
    "usapho2025": [
        ROOT / f"data/science_arena/candidate_benchmarks/structured/usapho2025/usapho2025_{pid}_render_v1.json"
        for pid in ["a1", "a2", "a3", "b1", "b2", "b3"]
    ],
}

IBO_ITEMS = ROOT / "data/science_arena/ibo2023/items_data.json"
IBO_KEY = ROOT / "data/science_arena/ibo2023/scoring_key.json"

IPHO_SECTIONS = {
    "ipho2025_s1": ROOT / "data/science_arena/ipho2025/IPhO_2025_S1_solution.json",
    "ipho2025_s2": ROOT / "data/science_arena/ipho2025/IPhO_2025_S2_solution.json",
    "ipho2025_s3": ROOT / "data/science_arena/ipho2025/IPhO_2025_S3_solution.json",
}

BENCHMARK_META = {
    "ipho2025": {
        "discipline": "physics",
        "status": "caveat_remote_images",
        "report_scale": 30,
        "release_date": "2025-07",
        "source_quality": "paper_original",
    },
    "icho2025": {
        "discipline": "chemistry",
        "status": "caveat_remote_images",
        "report_scale": 60,
        "release_date": "2025-07",
        "source_quality": "paper_original",
    },
    "ccho2025": {
        "discipline": "chemistry",
        "status": "caveat_missing_q7_image",
        "report_scale": 100,
        "release_date": "2025",
        "source_quality": "paper_original_caveat",
    },
    "cpho2025": {
        "discipline": "physics",
        "status": "caveat_remote_images",
        "report_scale": 320,
        "release_date": "2025",
        "source_quality": "paper_original",
    },
    "ibo2023": {
        "discipline": "biology",
        "status": "objective_key_caveat_remote_images",
        "report_scale": 453,
        "release_date": "2023-07",
        "source_quality": "paper_original_objective",
    },
    "eupho2025": {
        "discipline": "physics",
        "status": "image_enabled_complete",
        "report_scale": 30,
        "release_date": "2025-06",
        "source_quality": "new_official_pdf_static",
    },
    "apho2025": {
        "discipline": "physics",
        "status": "draft_human_review",
        "report_scale": 30,
        "release_date": "2025-05",
        "source_quality": "new_official_pdf_page_range",
    },
    "nbpho2025": {
        "discipline": "physics",
        "status": "caveat_staticization",
        "report_scale": 72,
        "release_date": "2025",
        "source_quality": "new_official_pdf_experimental_caveat",
    },
    "incho2026": {
        "discipline": "chemistry",
        "status": "draft_official_rubric",
        "report_scale": 106,
        "release_date": "2026-01",
        "source_quality": "new_official_pdf_page_range",
    },
    "usapho2025": {
        "discipline": "physics",
        "status": "derived_rubric_auxiliary",
        "report_scale": 150,
        "release_date": "2025-04",
        "source_quality": "new_official_pdf_derived_rubric",
    },
}

SOURCE_STUBS = {
    "usnco2025": "source_stub_download_blocked",
    "ukcho2026": "source_stub_download_blocked",
    "ukcho2025": "source_stub_download_blocked",
    "ioaa2025": "raw_extracted_not_normalized",
    "eoes2025": "lab_task_not_staticized",
}


@dataclass
class Asset:
    original: str
    kind: str
    local_path: str = ""
    hf_path: str = ""
    sha256: str = ""
    bytes: int = 0
    mime_type: str = ""
    status: str = "pending"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    return [value]


def join_text(parts: list[Any]) -> str:
    return "\n\n".join(str(part).strip() for part in parts if str(part).strip())


def scalar(value: Any) -> str:
    if isinstance(value, list) and value:
        return str(value[0])
    if value is None:
        return ""
    return str(value)


def score_float(value: Any) -> float | None:
    text = scalar(value)
    if not text:
        return None
    match = re.search(r"-?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def safe_id(text: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return text.strip("_") or "item"


def is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def resolve_local(path: str) -> Path | None:
    if is_url(path) or not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    return p if p.exists() else None


def selected_pages(pages_json: dict[str, Any], page_numbers: list[int]) -> list[dict[str, Any]]:
    wanted = {int(page) for page in page_numbers}
    return [page for page in pages_json.get("pages", []) if int(page.get("page", -1)) in wanted]


def page_text_blob(pages: list[dict[str, Any]], label: str) -> str:
    return "\n\n".join(
        f"[{label} page {page.get('page')}]\n{str(page.get('text') or '').strip()}" for page in pages
    ).strip()


def page_assets(pages_json_path: Path, pages: list[dict[str, Any]]) -> list[str]:
    base = pages_json_path.parent
    refs: list[str] = []
    for page in pages:
        render = page.get("page_render") or {}
        if render.get("file"):
            refs.append(str((base / render["file"]).relative_to(ROOT)))
        for image in page.get("images", []) or []:
            if image.get("file"):
                refs.append(str((base / image["file"]).relative_to(ROOT)))
    return refs


def make_row(
    *,
    benchmark_id: str,
    sequence_id: str,
    turn_id: str,
    turn_index: int,
    prompt: str,
    answer: str,
    rubric: str,
    max_score: float | None,
    image_refs: list[str],
    answer_image_refs: list[str] | None = None,
    source_refs: list[str],
    grading_type: str,
    caveats: list[str] | None = None,
) -> dict[str, Any]:
    meta = BENCHMARK_META.get(benchmark_id, {})
    answer_image_refs = answer_image_refs or []
    if not answer.strip() and answer_image_refs:
        answer = "Official answer is image-only; see `official_answer_image_refs`."
    if not rubric.strip() and answer_image_refs:
        rubric = "Official rubric is image-only or implicit; judge against the official answer image and the listed max_score."
    return {
        "benchmark_id": benchmark_id,
        "discipline": meta.get("discipline", ""),
        "sequence_id": sequence_id,
        "turn_id": turn_id,
        "turn_index": turn_index,
        "interleaved_policy": "include_prior_turns_in_same_sequence",
        "grading_type": grading_type,
        "max_score": max_score,
        "report_scale": meta.get("report_scale"),
        "release_date": meta.get("release_date", ""),
        "source_quality": meta.get("source_quality", ""),
        "prompt": prompt,
        "official_answer": answer,
        "official_rubric": rubric,
        "image_refs": sorted(dict.fromkeys(ref for ref in image_refs if ref)),
        "official_answer_image_refs": sorted(dict.fromkeys(ref for ref in answer_image_refs if ref)),
        "source_refs": sorted(dict.fromkeys(ref for ref in source_refs if ref)),
        "status": meta.get("status", ""),
        "caveats": caveats or [],
    }


def rows_from_long_form(benchmark_id: str, path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    data = load_json(path)
    for problem_index, problem in enumerate(data, start=1):
        qid = scalar(problem.get("question_id")) or str(problem_index)
        sequence_id = f"{benchmark_id}_{safe_id(qid)}"
        overall = join_text(as_list(problem.get("overall_instructions")))
        overall_figs = [str(x) for x in as_list(problem.get("overall_instructions_fig"))]
        details = problem.get("detailed_question") or []
        for turn_index, sub in enumerate(details, start=1):
            sid = scalar(sub.get("sub_question_id")) or f"{qid}.{turn_index}"
            instructions = join_text(as_list(sub.get("sub_question_instructions")))
            content = join_text(as_list(sub.get("sub_question_content")))
            prompt = join_text([overall, instructions, content])
            answer = join_text(as_list(sub.get("sub_question_solution")))
            rubric = join_text(as_list(sub.get("scoring_rubric"))) or answer
            image_refs = (
                overall_figs
                + [str(x) for x in as_list(sub.get("sub_question_instructions_fig"))]
                + [str(x) for x in as_list(sub.get("sub_question_content_fig"))]
            )
            solution_refs = [str(x) for x in as_list(sub.get("sub_question_solution_fig"))]
            rows.append(
                make_row(
                    benchmark_id=benchmark_id,
                    sequence_id=sequence_id,
                    turn_id=f"{sequence_id}_{safe_id(sid)}",
                    turn_index=turn_index,
                    prompt=prompt,
                    answer=answer,
                    rubric=rubric,
                    max_score=score_float(sub.get("score")),
                    image_refs=image_refs,
                    answer_image_refs=solution_refs,
                    source_refs=[str(path.relative_to(ROOT))] + solution_refs,
                    grading_type="llm_judge_rubric",
                    caveats=[],
                )
            )
    return rows


def rows_from_ibo() -> list[dict[str, Any]]:
    items = load_json(IBO_ITEMS)
    keys = {row["quiz_index"]: row for row in load_json(IBO_KEY)}
    rows = []
    for idx, item in enumerate(items, start=1):
        source = str(item.get("source") or "").strip()
        source_q_idx = str(item.get("source_q_idx") or "").strip()
        qid = item.get("quiz_index") or (f"{source} {source_q_idx}".strip() if source and source_q_idx else str(idx))
        key = keys.get(str(qid), {})
        image_refs: list[str] = []
        for image in as_list(item.get("image")) + as_list(item.get("image_urls")) + as_list(item.get("images")):
            if isinstance(image, dict):
                image_refs.append(str(image.get("url") or image.get("path") or ""))
            else:
                image_refs.append(str(image))
        prompt = join_text([item.get("prompt", ""), item.get("question", "")])
        answer = str(item.get("answer") or "")
        if key.get("sub_scores"):
            answer = join_text([answer, "Sub-scores: " + json.dumps(key.get("sub_scores", []), ensure_ascii=False)])
        rows.append(
            make_row(
                benchmark_id="ibo2023",
                sequence_id=f"ibo2023_item_{idx:03d}",
                turn_id=f"ibo2023_item_{idx:03d}",
                turn_index=1,
                prompt=prompt,
                answer=answer,
                rubric="Objective answer key; score each listed choice by the official key.",
                max_score=sum(float(x) for x in key.get("sub_scores", []) if isinstance(x, (int, float))),
                image_refs=image_refs,
                answer_image_refs=[],
                source_refs=[str(IBO_ITEMS.relative_to(ROOT)), str(IBO_KEY.relative_to(ROOT))],
                grading_type="objective_key",
                caveats=["Some canonical prompt text contains HTML image URLs; strict text-only runs need sanitized prompts."],
            )
        )
    return rows


def rows_from_ipho() -> list[dict[str, Any]]:
    rows = []
    for section_index, (section_id, path) in enumerate(IPHO_SECTIONS.items(), start=1):
        data = load_json(path)
        sequence_id = section_id
        context = ""
        turn_index = 0
        for block_index, block in enumerate(data, start=1):
            if not isinstance(block, list) or len(block) < 2:
                continue
            if len(block) >= 3:
                intro = block[0][0] if block and isinstance(block[0], list) else ""
                question = block[1][0] if len(block) > 1 and isinstance(block[1], list) else ""
                solution = block[2][0] if len(block) > 2 and isinstance(block[2], list) else ""
            else:
                intro = ""
                question = block[0][0] if block and isinstance(block[0], list) else ""
                solution = block[1][0] if len(block) > 1 and isinstance(block[1], list) else ""
            if "Determine" not in question and "Calculate" not in question and not re.search(r"\b[A-Z]\.\s*\d+", question):
                context = join_text([context, intro])
            turn_index += 1
            score_match = re.search(r"total score is\s*:?\s*([0-9.]+)\s*pt", solution, flags=re.I)
            max_score = float(score_match.group(1)) if score_match else None
            if max_score is None:
                inline_score = re.search(r"\{([0-9.]+)\}\s*\\mathrm\{p\}", question)
                if inline_score:
                    max_score = float(inline_score.group(1))
            image_refs: list[str] = []
            for part in block:
                if isinstance(part, list) and len(part) > 1:
                    image_refs += [str(x) for x in as_list(part[1])]
            rows.append(
                make_row(
                    benchmark_id="ipho2025",
                    sequence_id=sequence_id,
                    turn_id=f"{section_id}_{turn_index:03d}",
                    turn_index=turn_index,
                    prompt=join_text([context, intro, question]),
                    answer=solution,
                    rubric=solution,
                    max_score=max_score,
                    image_refs=image_refs,
                    answer_image_refs=[],
                    source_refs=[str(path.relative_to(ROOT))],
                    grading_type="llm_judge_rubric",
                    caveats=["IPhO source was reconstructed from solution JSON; verify against official PDFs for release."],
                )
            )
    return rows


def rows_from_page_configs(benchmark_id: str, paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for turn_index, path in enumerate(paths, start=1):
        cfg = load_json(path)
        q_pages_path = ROOT / cfg["source_files"]["question_pages_json"]
        s_pages_path = ROOT / cfg["source_files"]["solution_pages_json"]
        q_pages = selected_pages(load_json(q_pages_path), cfg.get("question_pages", []))
        s_pages = selected_pages(load_json(s_pages_path), cfg.get("solution_pages", []))
        prompt = page_text_blob(q_pages, "question")
        answer = page_text_blob(s_pages, "solution")
        assets = page_assets(q_pages_path, q_pages)
        answer_assets = page_assets(s_pages_path, s_pages)
        rows.append(
            make_row(
                benchmark_id=benchmark_id,
                sequence_id=f"{benchmark_id}_full",
                turn_id=cfg.get("problem_id", f"{benchmark_id}_{turn_index}"),
                turn_index=turn_index,
                prompt=prompt,
                answer=answer,
                rubric=answer,
                max_score=float(cfg.get("total_points")) if cfg.get("total_points") is not None else None,
                image_refs=assets,
                answer_image_refs=answer_assets,
                source_refs=[
                    cfg["source_files"].get("question_pdf", ""),
                    cfg["source_files"].get("solution_pdf", ""),
                    str(path.relative_to(ROOT)),
                ],
                grading_type="llm_judge_rubric" if benchmark_id != "usapho2025" else "derived_rubric_llm_judge",
                caveats=cfg.get("known_limitations", []),
            )
        )
    return rows


def collect_asset(ref: str, out_dir: Path, *, download_remote: bool, timeout: float) -> Asset:
    kind = "remote" if is_url(ref) else "local"
    asset = Asset(original=ref, kind=kind)
    try:
        if is_url(ref):
            if not download_remote:
                asset.status = "remote_not_downloaded"
                return asset
            parsed = urlparse(ref)
            suffix = Path(parsed.path).suffix or ".jpg"
            digest = hashlib.sha256(ref.encode("utf-8")).hexdigest()[:20]
            dest = out_dir / "assets" / "remote" / f"{digest}{suffix}"
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                response = requests.get(ref, timeout=timeout)
                response.raise_for_status()
                dest.write_bytes(response.content)
            source_path = dest
        else:
            source_path = resolve_local(ref)
            if not source_path:
                asset.status = "missing_local"
                return asset
            rel = source_path.relative_to(ROOT) if source_path.is_relative_to(ROOT) else Path(source_path.name)
            dest = out_dir / "assets" / "local" / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists():
                shutil.copy2(source_path, dest)
        data = dest.read_bytes()
        asset.local_path = str(dest)
        asset.hf_path = str(dest.relative_to(out_dir))
        asset.sha256 = hashlib.sha256(data).hexdigest()
        asset.bytes = len(data)
        asset.mime_type = mimetypes.guess_type(str(dest))[0] or "application/octet-stream"
        asset.status = "ok"
    except Exception as exc:  # noqa: BLE001
        asset.status = f"error:{type(exc).__name__}:{exc}"
    return asset


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--download_remote_assets", action="store_true")
    parser.add_argument("--remote_timeout", type=float, default=30.0)
    parser.add_argument("--asset_workers", type=int, default=16)
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    rows += rows_from_ipho()
    for benchmark_id, path in LONG_FORM_JSONS.items():
        rows += rows_from_long_form(benchmark_id, path)
    rows += rows_from_ibo()
    for benchmark_id, paths in PAGE_RANGE_CONFIGS.items():
        rows += rows_from_page_configs(benchmark_id, paths)

    asset_refs = sorted(
        {
            ref
            for row in rows
            for ref in list(row.get("image_refs", [])) + list(row.get("official_answer_image_refs", []))
        }
    )
    asset_usage: dict[str, list[dict[str, Any]]] = {ref: [] for ref in asset_refs}
    for row in rows:
        for ref in list(row.get("image_refs", [])) + list(row.get("official_answer_image_refs", [])):
            asset_usage.setdefault(ref, []).append(
                {
                    "benchmark_id": row["benchmark_id"],
                    "sequence_id": row["sequence_id"],
                    "turn_id": row["turn_id"],
                }
            )

    workers = max(1, int(args.asset_workers))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        collected = list(
            pool.map(
                lambda ref: collect_asset(
                    ref,
                    out_dir,
                    download_remote=args.download_remote_assets,
                    timeout=args.remote_timeout,
                ),
                asset_refs,
            )
        )

    assets: list[dict[str, Any]] = []
    asset_lookup: dict[str, dict[str, Any]] = {}
    for ref, asset in zip(asset_refs, collected, strict=True):
        payload = asset.__dict__
        usage = asset_usage.get(ref, [])
        payload["asset_id"] = hashlib.sha256(ref.encode("utf-8")).hexdigest()[:20]
        payload["benchmark_ids"] = sorted({item["benchmark_id"] for item in usage})
        payload["used_by"] = usage
        assets.append(payload)
        asset_lookup[ref] = payload

    for row in rows:
        row["images"] = [asset_lookup.get(ref, {"original": ref, "status": "unseen"}) for ref in row["image_refs"]]
        row["official_answer_images"] = [
            asset_lookup.get(ref, {"original": ref, "status": "unseen"})
            for ref in row["official_answer_image_refs"]
        ]

    benchmark_summary: dict[str, dict[str, Any]] = {}
    for row in rows:
        summary = benchmark_summary.setdefault(
            row["benchmark_id"],
            {
                "benchmark_id": row["benchmark_id"],
                "discipline": row["discipline"],
                "status": row["status"],
                "report_scale": row["report_scale"],
                "release_date": row.get("release_date", ""),
                "source_quality": row.get("source_quality", ""),
                "item_count": 0,
                "sequence_count": 0,
                "max_score_sum": 0.0,
                "image_count": 0,
            },
        )
        summary["item_count"] += 1
        summary["max_score_sum"] += float(row["max_score"] or 0)
        summary["image_count"] += len(row["image_refs"])
        summary["official_answer_image_count"] = summary.get("official_answer_image_count", 0) + len(
            row.get("official_answer_image_refs", [])
        )
    for benchmark_id, summary in benchmark_summary.items():
        summary["sequence_count"] = len({row["sequence_id"] for row in rows if row["benchmark_id"] == benchmark_id})

    non_evaluable = [
        {"benchmark_id": key, "status": status, "reason": "not included in benchmark_items.jsonl"}
        for key, status in SOURCE_STUBS.items()
    ]

    write_jsonl(out_dir / "benchmark_items.jsonl", rows)
    write_jsonl(out_dir / "assets_manifest.jsonl", assets)
    write_jsonl(out_dir / "benchmark_summary.jsonl", sorted(benchmark_summary.values(), key=lambda x: x["benchmark_id"]))
    write_jsonl(out_dir / "non_evaluable_sources.jsonl", non_evaluable)

    dataset_info = {
        "created_at": "2026-05-24",
        "schema": "science_arena_interleaved_turn_v1",
        "row_count": len(rows),
        "benchmark_count": len(benchmark_summary),
        "asset_ref_count": len(asset_refs),
        "asset_ok_count": sum(1 for asset in assets if asset.get("status") == "ok"),
        "asset_error_count": sum(1 for asset in assets if asset.get("status") not in {"ok", "remote_not_downloaded"}),
        "remote_assets_downloaded": bool(args.download_remote_assets),
        "asset_workers": workers,
        "non_evaluable_sources": SOURCE_STUBS,
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(dataset_info, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "README.md").write_text(
        "# Science Arena HF Export\n\n"
        "This export uses `science_arena_interleaved_turn_v1`. Each row in "
        "`benchmark_items.jsonl` is one interleaved turn. Rows sharing "
        "`sequence_id` should be presented sequentially with previous turns in context.\n\n"
        "Primary columns: `benchmark_id`, `discipline`, `sequence_id`, `turn_id`, "
        "`turn_index`, `prompt`, `official_answer`, `official_rubric`, "
        "`max_score`, `image_refs`, `images`, `official_answer_image_refs`, "
        "`official_answer_images`, `grading_type`, `status`, "
        "`release_date`, and `source_quality`.\n\n"
        "`assets_manifest.jsonl` stores local/remote asset hashes, MIME types, "
        "HF-relative paths, and `used_by` row references. The export includes "
        "caveat fields and does not silently mark source-stub benchmarks as final. "
        "See `non_evaluable_sources.jsonl`.\n",
        encoding="utf-8",
    )
    print(json.dumps(dataset_info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
