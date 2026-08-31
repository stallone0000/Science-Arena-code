# -*- coding: utf-8 -*-
"""Shared constants and helpers for the unified ScienceArena runners."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from api_client_interleaved import clean_image_urls


CHEMISTRY_INTRO = """
The following is a chemistry test question to test your chemistry ability.
You are now a chemistry competition candidate. Answer the question.
[Answering guidelines for structure depiction]
- If the question asks you to draw or depict one or more chemical structures,
  describe each structure in clear, evaluator-understandable text and do not rely
  on images.
- Make structure descriptions self-contained so each structure can be graded in
  isolation.
- If multiple structures are required, list them separately.
- If atom labels or mapping are part of the task, include a short legend.
- If the prompt does not request a structure, answer normally.
""".strip()

PHYSICS_INTRO = (
    "The following is a physics test question to test your physics ability. "
    "You are now a physics competition candidate. Answer the question."
)

BACKGROUND_PROMPT = "The following background is known:"
ADDITIONAL_BACKGROUND_PROMPT = "We also know this background:"
IPHO_BACKGROUND_PROMPT = "The following background is known:"

ANSWER_QUERY = """
Based on the above background or past information, now answer the following question.
NOTICE: indicate your confidence for this answer. Confidence score must be between
0% and 100%. Put confidence at the end in this format: Confidence: \\boxed{}.
For example: Confidence: \\boxed{85%}.
The question you need to answer is:
""".strip()

INTERLEAVED_SELF_CONTAINED_REMINDER = """
IMPORTANT: answer THIS question in a self-contained way so it can be graded in isolation.
If you rely on a previous result, restate the needed formula or conclusion explicitly here.
Do not answer only with references like "as above" or "from the previous answer".
""".strip()

STANDARD_PREVIOUS_ANSWER_PROMPT = "AND, your solution to the above question is:"
IPHO_PREVIOUS_ANSWER_PROMPT = "Your solution to the above question is:"

FULL_STANDARD_SUFFIX = "Please answer EACH sub-question separately and label the sub-question ID clearly."
FULL_IPHO_SUFFIX_TEMPLATE = "Please answer ALL sub-questions in Part {part} separately and label them clearly."

EVAL_PROMPT6_STANDARD = """
Based on the above scoring rules and the following solution,
strictly compare the final result and score step by step.
NOTICE: give confidence for your judgment between 0% and 100%, format:
Confidence: \\boxed{}.
Also output final score in this format: Score: \\boxed{}.
The solution you need to judge is:
""".strip()

EVAL_PROMPT7_STANDARD = "Below is total score:"
EVAL_PROMPT_SUB_SOLUTION = "Here are the standard answers:"

EVAL_PROMPT5_CHEM = (
    "You are a strict examiner. Compare the standard answer and the solution strictly. "
    "All images provided to you are from standard answers only. Below is scoring details:"
)

EVAL_PROMPT5_PHYS = (
    "You are a strict examiner. Compare the standard answer and the solution strictly. "
    "Below is scoring details:"
)

IPHO_EVAL_PROMPT5 = "Below is the total score and scoring details.\n"
IPHO_EVAL_PROMPT6 = """
Based on the above scoring rules and the following solution,
strictly compare the final result and score step by step.
NOTICE: give confidence for your judgment between 0% and 100%, format:
Confidence: \\boxed{}.
Also output final score in this format: Score: \\boxed{}.
The solution you need to judge is:
""".strip()


@dataclass(frozen=True)
class StandardExamSpec:
    exam: str
    data_file: str
    answer_intro: str
    eval_prompt5: str
    proportion: Tuple[float, ...]
    max_point: Tuple[float, ...]
    default_pre_model: str = "google/gemini-3-pro-preview"
    default_judge_model: str = "google/gemini-3-pro-preview"


STANDARD_EXAM_SPECS: Dict[str, StandardExamSpec] = {
    "ccho": StandardExamSpec(
        exam="ccho",
        data_file="fix_CCHO-2025-latex.json",
        answer_intro=CHEMISTRY_INTRO,
        eval_prompt5=EVAL_PROMPT5_CHEM,
        proportion=(0, 5, 9, 10, 10, 14, 9, 9, 7, 13, 14),
        max_point=(0, 14, 27, 28, 28, 40, 25, 25, 18, 29, 33),
    ),
    "cpho": StandardExamSpec(
        exam="cpho",
        data_file="fix_CPHO-2025-latex.json",
        answer_intro=PHYSICS_INTRO,
        eval_prompt5=EVAL_PROMPT5_PHYS,
        proportion=(0, 45, 45, 45, 45, 45, 50, 45),
        max_point=(0, 45, 45, 45, 45, 45, 50, 45),
    ),
    "icho": StandardExamSpec(
        exam="icho",
        data_file="fix_ICHO-2025-latex.json",
        answer_intro=CHEMISTRY_INTRO,
        eval_prompt5=EVAL_PROMPT5_CHEM,
        proportion=(0, 6, 6.5, 7.5, 6, 6.5, 6, 7, 7.5, 7),
        max_point=(0, 34, 67, 111, 40, 16, 22, 22, 35, 36),
    ),
}


PART_ID_RE = re.compile(r"^\s*([A-Z])\.")
IPHO_TOTAL_SCORE_RE = re.compile(r"total score\s*is\s*:?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
BOXED_SCORE_RE = re.compile(r"(?:\\?box(?:ed)?|\x08oxed)\{\s*([-+]?\d+(?:\.\d+)?)\s*\}", re.IGNORECASE)
PLAIN_SCORE_LINE_RE = re.compile(
    r"(?:^|\n)\s*(?:\*\*)?(?:final\s+)?score(?:\*\*)?\s*[:：]\s*\(?\s*([-+]?\d+(?:\.\d+)?)\s*(?:/|\)|$)",
    re.IGNORECASE,
)
SUBQUESTION_SCORE_RE = re.compile(
    r"(?:^|\n)\s*(?:[-*]\s*)?(?:\*\*)?score\s+for\s+[^:\n]{1,80}:\s*\(?\s*([-+]?\d+(?:\.\d+)?)\s*/\s*([-+]?\d+(?:\.\d+)?)",
    re.IGNORECASE,
)


def interleaved_root() -> Path:
    return Path(__file__).resolve().parent


def toacl_root() -> Path:
    return interleaved_root().parent / "ToACL"


def default_data_root() -> Path:
    return interleaved_root() / "data"


def default_artifacts_root() -> Path:
    return interleaved_root() / "artifacts"


def default_runs_root() -> Path:
    return default_artifacts_root() / "runs"


def default_smoke_root() -> Path:
    return default_artifacts_root() / "smoke"


def default_logs_root() -> Path:
    return default_artifacts_root() / "logs"


def resolve_data_root(raw_path: str | Path | None = None) -> Path:
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return default_data_root()


def resolve_output_root(raw_path: str | Path | None = None) -> Path:
    if raw_path:
        return Path(raw_path).expanduser().resolve()
    return default_runs_root()


def build_run_dir(
    *,
    run_name: str,
    method: str,
    exam: str,
    output_root: str | Path | None = None,
    question_num_index: str | None = None,
) -> Path:
    root = resolve_output_root(output_root)
    path = root / run_name / method / exam
    if exam == "ipho":
        path = path / f"S{question_num_index or '1'}"
    return path


def get_standard_spec(exam: str) -> StandardExamSpec:
    exam_l = (exam or "").strip().lower()
    if exam_l not in STANDARD_EXAM_SPECS:
        raise KeyError(f"Unsupported standard exam: {exam}")
    return STANDARD_EXAM_SPECS[exam_l]


def model_tag_basename(model_tag: str) -> str:
    tag = (model_tag or "").strip()
    if not tag:
        return "model"
    return Path(tag).name


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def maybe_load_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return load_json(path)


def save_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(path)


def update_named_json(path: Path, key: str, value: Any) -> None:
    current = maybe_load_json(path, {})
    if not isinstance(current, dict):
        current = {}
    current[key] = value
    save_json(path, current)


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def safe_join_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "".join(str(x) for x in value)
    return str(value)


def parse_csv_items(raw: str | Sequence[str] | None) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw]
    else:
        items = re.split(r"[\s,]+", str(raw).strip())
    return [item for item in items if item]


def parse_question_ids(raw: str | Sequence[str] | None) -> List[str]:
    return [item.strip() for item in parse_csv_items(raw)]


def parse_item_indices(raw: str | Sequence[str] | None) -> List[int]:
    result: List[int] = []
    for item in parse_csv_items(raw):
        result.append(int(item))
    return result


def parse_part_ids(raw: str | Sequence[str] | None) -> List[str]:
    return [item.strip().upper() for item in parse_csv_items(raw)]


def normalize_text(value: str) -> str:
    text = unicodedata.normalize("NFKC", value or "")
    text = text.casefold()
    text = text.replace("\x08", "\\b")
    text = re.sub(r"[ \t]+", " ", text)
    return text


def extract_boxed_score(text: str, *, allow_flexible: bool = False) -> float:
    normalized = normalize_text(text)
    match = BOXED_SCORE_RE.findall(normalized)
    if not match and not allow_flexible:
        return 0.0
    if not match:
        plain_matches = PLAIN_SCORE_LINE_RE.findall(normalized)
        if plain_matches:
            try:
                return float(plain_matches[-1])
            except ValueError:
                return 0.0

        subquestion_matches = SUBQUESTION_SCORE_RE.findall(normalized)
        if subquestion_matches:
            total = 0.0
            for earned, _ in subquestion_matches:
                try:
                    total += float(earned)
                except ValueError:
                    continue
            return total
        return 0.0
    try:
        return float(match[-1])
    except ValueError:
        return 0.0


def extract_ipho_total_score(reference_text: str) -> float:
    match = IPHO_TOTAL_SCORE_RE.search(reference_text or "")
    if not match:
        return 0.0
    try:
        return float(match.group(1))
    except ValueError:
        return 0.0


def flatten_usage(usage: Any) -> Dict[str, Any]:
    if usage is None:
        return {}
    if isinstance(usage, dict):
        return usage
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if hasattr(usage, "to_dict"):
        return usage.to_dict()
    return {}


def namespace_to_dict(args: Any) -> Dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, dict):
        return dict(args)
    return {k: getattr(args, k) for k in vars(args)}


def redact_sensitive_fields(data: Dict[str, Any], *, keys: Sequence[str] = ("api_key",)) -> Dict[str, Any]:
    result = dict(data)
    for key in keys:
        if key in result and result[key]:
            result[key] = "<redacted>"
    return result


def redact_argv(
    argv: Sequence[Any] | None,
    *,
    option_keys: Sequence[str] = ("api_key",),
) -> List[str]:
    if not argv:
        return []

    normalized_keys = {str(key).lstrip("-").replace("-", "_") for key in option_keys}
    raw = [str(item) for item in argv]
    result: List[str] = []
    skip_next = False
    for index, token in enumerate(raw):
        if skip_next:
            skip_next = False
            result.append("<redacted>")
            continue

        if token.startswith("--"):
            name, sep, value = token[2:].partition("=")
            normalized_name = name.replace("-", "_")
            if normalized_name in normalized_keys:
                if sep:
                    result.append(f"--{name}=<redacted>")
                else:
                    result.append(token)
                    if index + 1 < len(raw):
                        skip_next = True
                continue
        result.append(token)
    return result


def build_invocation_metadata(
    *,
    argv: Sequence[Any] | None = None,
    entrypoint: str | Path | None = None,
    cwd: str | Path | None = None,
) -> Dict[str, Any]:
    resolved_argv = redact_argv(argv)
    resolved_entrypoint = Path(entrypoint).resolve() if entrypoint else None
    resolved_cwd = Path(cwd).resolve() if cwd else Path.cwd().resolve()
    return {
        "argv": resolved_argv,
        "entrypoint": str(resolved_entrypoint) if resolved_entrypoint else "",
        "cwd": str(resolved_cwd),
    }


def build_standard_background_text(spec: StandardExamSpec, item: Dict[str, Any]) -> str:
    overall_text = safe_join_text(item.get("overall_instructions", []))
    return f"{spec.answer_intro}\n\n{BACKGROUND_PROMPT}\n{overall_text}".strip()


def build_standard_subquestion_text(sub_question: Dict[str, Any]) -> str:
    sid = safe_join_text(sub_question.get("sub_question_id", [])).strip()
    instructions = safe_join_text(sub_question.get("sub_question_instructions", [])).strip()
    question = safe_join_text(sub_question.get("sub_question_content", [])).strip()
    sections: List[str] = []
    if sid:
        sections.append(f"Sub-question ID: {sid}")
    if instructions:
        sections.append(instructions)
    sections.append(ANSWER_QUERY)
    if question:
        sections.append(question)
    return "\n\n".join([section for section in sections if section])


def build_standard_iterative_append_text(sub_question: Dict[str, Any]) -> str:
    instructions = safe_join_text(sub_question.get("sub_question_instructions", [])).strip()
    question = safe_join_text(sub_question.get("sub_question_content", [])).strip()
    parts: List[str] = []
    if instructions:
        parts.append(f"{ADDITIONAL_BACKGROUND_PROMPT}\n{instructions}")
    parts.append(ANSWER_QUERY)
    if question:
        parts.append(question)
    return "\n".join([part for part in parts if part]).strip()


def apply_interleaved_answer_style(question_text: str, *, answer_style: str = "self_contained") -> str:
    text = (question_text or "").strip()
    style = (answer_style or "self_contained").strip().lower()
    if style == "contextual":
        return text
    if style != "self_contained":
        raise ValueError(f"Unsupported interleaved answer style: {answer_style}")
    return "\n\n".join([part for part in (text, INTERLEAVED_SELF_CONTAINED_REMINDER) if part]).strip()


def build_standard_full_prompt(spec: StandardExamSpec, question: Dict[str, Any]) -> Tuple[str, List[str]]:
    blocks: List[str] = []
    overall = safe_join_text(question.get("overall_instructions", [])).strip()
    if overall:
        blocks.append(f"Overall instructions:\n{overall}")

    image_urls = clean_image_urls(question.get("overall_instructions_fig", []))
    for sub_question in question.get("detailed_question", []):
        sid = safe_join_text(sub_question.get("sub_question_id", [])).strip()
        block_lines = [f"Sub-question {sid}:" if sid else "Sub-question:"]
        instructions = safe_join_text(sub_question.get("sub_question_instructions", [])).strip()
        content = safe_join_text(sub_question.get("sub_question_content", [])).strip()
        if instructions:
            block_lines.append(instructions)
        if content:
            block_lines.append(content)
        for idx, fig in enumerate(clean_image_urls(sub_question.get("sub_question_instructions_fig", [])), start=1):
            block_lines.append(f"[Figure {sid}-{idx}]")
            image_urls.append(fig)
        blocks.append("\n".join([line for line in block_lines if line]))

    prompt = (
        f"{spec.answer_intro}\n\n{BACKGROUND_PROMPT}\n"
        + "\n\n".join(blocks)
        + f"\n\n{ANSWER_QUERY}\n{FULL_STANDARD_SUFFIX}"
    ).strip()
    return prompt, image_urls


def build_standard_reference_map(data: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    ref_map: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for question in data:
        qid = safe_join_text(question.get("question_id", [])).strip()
        for sub_question in question.get("detailed_question", []):
            sid = safe_join_text(sub_question.get("sub_question_id", [])).strip()
            ref_map[(qid, sid)] = {
                "score": safe_join_text(sub_question.get("score", [])).strip() or "0",
                "rubric": safe_join_text(sub_question.get("scoring_rubric", [])),
                "solution": safe_join_text(sub_question.get("sub_question_solution", [])),
                "solution_fig": clean_image_urls(sub_question.get("sub_question_solution_fig", [])),
            }
    return ref_map


def build_standard_question_map(data: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    result: Dict[str, Dict[str, Any]] = {}
    for question in data:
        qid = safe_join_text(question.get("question_id", [])).strip()
        result[qid] = question
    return result


def build_standard_full_reference(question: Dict[str, Any]) -> Dict[str, Any]:
    total_score = 0.0
    refs: List[str] = []
    image_urls: List[str] = []
    for sub_question in question.get("detailed_question", []):
        sid = safe_join_text(sub_question.get("sub_question_id", [])).strip()
        score = safe_join_text(sub_question.get("score", [])).strip() or "0"
        solution = safe_join_text(sub_question.get("sub_question_solution", [])).strip()
        rubric = safe_join_text(sub_question.get("scoring_rubric", [])).strip()
        try:
            total_score += float(score)
        except ValueError:
            pass

        refs.append(f"Sub-question {sid} (Score {score}):")
        if solution:
            refs.append(f"Standard answer: {solution}")
        for idx, fig in enumerate(clean_image_urls(sub_question.get("sub_question_solution_fig", [])), start=1):
            refs.append(f"[Standard Answer Figure {sid}-{idx}]")
            image_urls.append(fig)
        if rubric:
            refs.append(f"Scoring rubric: {rubric}")

    return {
        "total_score": total_score,
        "reference_text": "\n".join(refs),
        "image_urls": image_urls,
    }


def coerce_answer_from_solution_row(row: Any) -> str:
    if isinstance(row, dict):
        answer = row.get("answer")
        if answer is None:
            return ""
        return str(answer)
    return str(row)


def parse_ipho_item(item: Any) -> Dict[str, Any]:
    if not isinstance(item, list):
        return {
            "background_text": "",
            "background_images": [],
            "question_text": str(item),
            "question_images": [],
            "judge_text": "",
            "judge_images": [],
        }

    def _entry_text(entry: Any) -> str:
        if isinstance(entry, list) and entry:
            return str(entry[0])
        return ""

    def _entry_images(entry: Any) -> List[str]:
        if isinstance(entry, list) and len(entry) > 1 and isinstance(entry[1], list):
            return clean_image_urls(entry[1])
        return []

    background_entry = None
    question_entry = None
    judge_entry = None

    if len(item) >= 3:
        background_entry = item[0]
        question_entry = item[1]
        judge_entry = item[-1]
    elif len(item) == 2:
        question_entry = item[0]
        judge_entry = item[1]
    elif len(item) == 1:
        question_entry = item[0]

    return {
        "background_text": _entry_text(background_entry),
        "background_images": _entry_images(background_entry),
        "question_text": _entry_text(question_entry),
        "question_images": _entry_images(question_entry),
        "judge_text": _entry_text(judge_entry),
        "judge_images": _entry_images(judge_entry),
    }


def build_ipho_question_text(parsed_item: Dict[str, Any], item_index: int) -> str:
    blocks: List[str] = [f"Item index: {item_index}"]
    background_text = parsed_item.get("background_text", "")
    if background_text:
        blocks.append(f"{IPHO_BACKGROUND_PROMPT}\n{background_text}")
    blocks.append(ANSWER_QUERY)
    question_text = parsed_item.get("question_text", "")
    if question_text:
        blocks.append(question_text)
    return "\n\n".join([block for block in blocks if block])


def build_ipho_iterative_append_text(parsed_item: Dict[str, Any], item_index: int, is_first: bool) -> str:
    question_text = parsed_item.get("question_text", "").strip()
    background_text = parsed_item.get("background_text", "").strip()
    blocks: List[str] = []
    if background_text:
        label = IPHO_BACKGROUND_PROMPT if is_first else ADDITIONAL_BACKGROUND_PROMPT
        blocks.append(f"{label}\n{background_text}")
    elif item_index == 0 and not is_first:
        blocks.append(IPHO_BACKGROUND_PROMPT)
    blocks.append(ANSWER_QUERY)
    if question_text:
        blocks.append(question_text)
    return "\n".join([block for block in blocks if block]).strip()


def extract_ipho_part_id(question_text: str) -> str:
    match = PART_ID_RE.match(question_text or "")
    if match:
        return match.group(1).upper()
    return "Misc"


def group_ipho_parts(
    data: Sequence[Any],
    *,
    part_ids: Optional[Sequence[str]] = None,
    item_indices: Optional[Sequence[int]] = None,
) -> List[Dict[str, Any]]:
    wanted_parts = {part.upper() for part in (part_ids or [])}
    wanted_items = set(item_indices or [])
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for idx, raw_item in enumerate(data):
        if wanted_items and idx not in wanted_items:
            continue
        parsed = parse_ipho_item(raw_item)
        part_id = extract_ipho_part_id(parsed.get("question_text", ""))
        if wanted_parts and part_id not in wanted_parts:
            continue
        groups.setdefault(part_id, []).append(
            {
                "item_index": idx,
                "parsed": parsed,
            }
        )
    return [
        {
            "part_id": part_id,
            "items": items,
        }
        for part_id, items in groups.items()
    ]


def build_ipho_full_prompt(part_group: Dict[str, Any]) -> Tuple[str, List[str]]:
    part_id = str(part_group.get("part_id", "Misc"))
    blocks: List[str] = []
    image_urls: List[str] = []
    for item_pos, item in enumerate(part_group.get("items", []), start=1):
        parsed = item["parsed"]
        background_text = parsed.get("background_text", "").strip()
        question_text = parsed.get("question_text", "").strip()
        if background_text:
            blocks.append(background_text)
            image_urls.extend(clean_image_urls(parsed.get("background_images", [])))
        if question_text:
            blocks.append(question_text)
        for fig_idx, fig in enumerate(clean_image_urls(parsed.get("question_images", [])), start=1):
            blocks.append(f"[Fig-{part_id}-{item_pos}-{fig_idx}]")
            image_urls.append(fig)

    prompt = (
        f"{IPHO_BACKGROUND_PROMPT}\n"
        + "\n\n".join(blocks)
        + f"\n\n{ANSWER_QUERY}\n{FULL_IPHO_SUFFIX_TEMPLATE.format(part=part_id)}"
    ).strip()
    return prompt, image_urls


def build_ipho_full_reference(part_group: Dict[str, Any]) -> Dict[str, Any]:
    refs: List[str] = []
    image_urls: List[str] = []
    total_score = 0.0
    for item in part_group.get("items", []):
        parsed = item["parsed"]
        question_text = parsed.get("question_text", "").strip()
        judge_text = parsed.get("judge_text", "").strip()
        if question_text:
            refs.append(f"Question: {question_text}")
        if judge_text:
            refs.append(judge_text)
            total_score += extract_ipho_total_score(judge_text)
        for fig_idx, fig in enumerate(clean_image_urls(parsed.get("judge_images", [])), start=1):
            refs.append(f"[Standard Answer Figure {item['item_index']}-{fig_idx}]")
            image_urls.append(fig)

    if total_score:
        refs.insert(0, f"Part total score: {total_score}")
    return {
        "total_score": total_score,
        "reference_text": "\n".join(refs),
        "image_urls": image_urls,
    }


def select_standard_questions(
    data: Sequence[Dict[str, Any]],
    question_ids: Optional[Sequence[str]] = None,
    *,
    limit: int = 0,
) -> List[Dict[str, Any]]:
    wanted = {qid.strip() for qid in (question_ids or []) if qid.strip()}
    selected: List[Dict[str, Any]] = []
    for question in data:
        qid = safe_join_text(question.get("question_id", [])).strip()
        if wanted and qid not in wanted:
            continue
        selected.append(question)
        if limit and len(selected) >= limit:
            break
    return selected


def filter_standard_solution_rows(
    rows: Sequence[Any],
    question_ids: Optional[Sequence[str]] = None,
    *,
    limit_by_question: bool = False,
    limit: int = 0,
) -> List[Any]:
    wanted = {qid.strip() for qid in (question_ids or []) if qid.strip()}
    filtered: List[Any] = []
    seen_question_ids: List[str] = []
    for row in rows:
        qid = ""
        if isinstance(row, dict):
            qid = str(row.get("question_id", "")).strip()
        if wanted and qid not in wanted:
            continue
        if limit and limit_by_question and qid:
            if qid not in seen_question_ids:
                if len(seen_question_ids) >= limit:
                    break
                seen_question_ids.append(qid)
        filtered.append(row)
        if limit and not limit_by_question and len(filtered) >= limit:
            break
    return filtered


def select_ipho_items(
    data: Sequence[Any],
    item_indices: Optional[Sequence[int]] = None,
    *,
    limit: int = 0,
) -> List[Tuple[int, Any]]:
    wanted = set(item_indices or [])
    selected: List[Tuple[int, Any]] = []
    for idx, item in enumerate(data):
        if wanted and idx not in wanted:
            continue
        selected.append((idx, item))
        if limit and len(selected) >= limit:
            break
    return selected


def filter_ipho_solution_rows(
    rows: Sequence[Any],
    *,
    item_indices: Optional[Sequence[int]] = None,
    part_ids: Optional[Sequence[str]] = None,
    limit: int = 0,
) -> List[Any]:
    wanted_items = set(item_indices or [])
    wanted_parts = {part.upper() for part in (part_ids or [])}
    filtered: List[Any] = []
    for row in rows:
        if isinstance(row, dict):
            if wanted_items and "item_index" in row and int(row["item_index"]) not in wanted_items:
                continue
            if wanted_parts and "part_id" in row and str(row["part_id"]).upper() not in wanted_parts:
                continue
        filtered.append(row)
        if limit and len(filtered) >= limit:
            break
    return filtered


def compute_standard_weighted_score(spec: StandardExamSpec, question_id: str, parsed_score: float) -> float:
    try:
        q_index = int(float(question_id))
    except ValueError:
        return 0.0
    if q_index < 0 or q_index >= len(spec.max_point):
        return 0.0
    max_point = spec.max_point[q_index]
    if max_point == 0:
        return 0.0
    return parsed_score / max_point * spec.proportion[q_index]


def enumerate_standard_subquestions(data: Sequence[Dict[str, Any]]) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    for question in data:
        for sub_question in question.get("detailed_question", []):
            yield question, sub_question
