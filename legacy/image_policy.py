# -*- coding: utf-8 -*-
"""Unified image policy helpers for ScienceArena generation and evaluation."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence

from api_client_interleaved import clean_image_urls


def unique_image_urls(*groups: Iterable[str]) -> List[str]:
    seen = set()
    result: List[str] = []
    for group in groups:
        for url in clean_image_urls(list(group)):
            if url in seen:
                continue
            seen.add(url)
            result.append(url)
    return result


def extract_message_image_urls(messages: Sequence[Dict[str, Any]]) -> List[str]:
    result: List[str] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "image_url":
                continue
            image_obj = block.get("image_url")
            if not isinstance(image_obj, dict):
                continue
            url = str(image_obj.get("url", "")).strip()
            if url:
                result.append(url)
    return unique_image_urls(result)


def extract_user_only_messages(messages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        filtered.append(copy.deepcopy(message))
    return filtered


def standard_question_image_categories(
    question: Dict[str, Any],
    sub_question: Dict[str, Any] | None = None,
) -> Dict[str, List[str]]:
    global_question_images = clean_image_urls(question.get("overall_instructions_fig", []))
    local_question_images = clean_image_urls(sub_question.get("sub_question_instructions_fig", [])) if sub_question else []
    return {
        "global_question_images": global_question_images,
        "local_question_images": local_question_images,
    }


def standard_reference_image_categories(sub_question: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        "reference_images": clean_image_urls(sub_question.get("sub_question_solution_fig", [])),
    }


def ipho_image_categories(parsed_item: Dict[str, Any]) -> Dict[str, List[str]]:
    return {
        "background_images": clean_image_urls(parsed_item.get("background_images", [])),
        "question_images": clean_image_urls(parsed_item.get("question_images", [])),
        "reference_images": clean_image_urls(parsed_item.get("judge_images", [])),
    }


def standard_full_question_images(question: Dict[str, Any]) -> List[str]:
    image_urls = clean_image_urls(question.get("overall_instructions_fig", []))
    for sub_question in question.get("detailed_question", []):
        image_urls = unique_image_urls(image_urls, clean_image_urls(sub_question.get("sub_question_instructions_fig", [])))
    return image_urls


def standard_full_reference_images(question: Dict[str, Any]) -> List[str]:
    image_urls: List[str] = []
    for sub_question in question.get("detailed_question", []):
        image_urls = unique_image_urls(image_urls, clean_image_urls(sub_question.get("sub_question_solution_fig", [])))
    return image_urls


def ipho_item_question_images(parsed_item: Dict[str, Any]) -> List[str]:
    return unique_image_urls(parsed_item.get("background_images", []), parsed_item.get("question_images", []))


def ipho_part_question_images(part_group: Dict[str, Any]) -> List[str]:
    image_urls: List[str] = []
    for item in part_group.get("items", []):
        parsed = item["parsed"]
        image_urls = unique_image_urls(
            image_urls,
            parsed.get("background_images", []),
            parsed.get("question_images", []),
        )
    return image_urls


def ipho_part_reference_images(part_group: Dict[str, Any]) -> List[str]:
    image_urls: List[str] = []
    for item in part_group.get("items", []):
        parsed = item["parsed"]
        image_urls = unique_image_urls(image_urls, parsed.get("judge_images", []))
    return image_urls


@dataclass(frozen=True)
class GenerationImagePolicy:
    preset: str
    method: str
    exam: str
    iterative_image_mode: str
    interleaved_global_image_replay: bool
    interleaved_replay_global_persist: bool
    interleaved_background_global_persist: bool

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "method": self.method,
            "exam": self.exam,
            "iterative_image_mode": self.iterative_image_mode,
            "interleaved_global_image_replay": self.interleaved_global_image_replay,
            "interleaved_replay_global_persist": self.interleaved_replay_global_persist,
            "interleaved_background_global_persist": self.interleaved_background_global_persist,
        }


@dataclass(frozen=True)
class JudgeImagePolicy:
    preset: str
    method: str
    exam: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "preset": self.preset,
            "method": self.method,
            "exam": self.exam,
        }


def _resolve_bool_override(raw: str, default: bool) -> bool:
    if raw == "inherit":
        return default
    if raw == "on":
        return True
    if raw == "off":
        return False
    raise ValueError(f"Unsupported on/off/inherit value: {raw}")


def _default_iterative_image_mode(*, preset: str, exam: str) -> str:
    if preset == "legacy":
        return "current_only" if exam == "icho" else "cumulative"
    return "global_current"


def _default_interleaved_global_image_replay(*, preset: str) -> bool:
    return preset == "unified"


def _default_interleaved_replay_global_persist(*, preset: str) -> bool:
    return False


def resolve_generation_image_policy(
    *,
    preset: str,
    method: str,
    exam: str,
    iterative_image_mode: str,
    interleaved_global_image_replay: str,
    interleaved_replay_global_persist: str,
) -> GenerationImagePolicy:
    preset_l = (preset or "unified").strip().lower()
    if preset_l not in {"unified", "legacy"}:
        raise ValueError(f"Unsupported generation_image_preset: {preset}")

    method_l = (method or "").strip().lower()
    exam_l = (exam or "").strip().lower()

    resolved_iterative_mode = (
        _default_iterative_image_mode(preset=preset_l, exam=exam_l)
        if iterative_image_mode == "inherit"
        else iterative_image_mode
    )
    if resolved_iterative_mode not in {"current_only", "global_current", "cumulative"}:
        raise ValueError(f"Unsupported iterative_image_mode: {iterative_image_mode}")

    if method_l == "interleaved" and exam_l != "ipho":
        replay = _resolve_bool_override(
            interleaved_global_image_replay,
            _default_interleaved_global_image_replay(preset=preset_l),
        )
        persist = _resolve_bool_override(
            interleaved_replay_global_persist,
            _default_interleaved_replay_global_persist(preset=preset_l),
        )
        background_global_persist = (not replay) or persist
    else:
        replay = False
        persist = False
        background_global_persist = method_l == "interleaved" and exam_l != "ipho"

    return GenerationImagePolicy(
        preset=preset_l,
        method=method_l,
        exam=exam_l,
        iterative_image_mode=resolved_iterative_mode,
        interleaved_global_image_replay=replay,
        interleaved_replay_global_persist=persist,
        interleaved_background_global_persist=background_global_persist,
    )


def resolve_judge_image_policy(
    *,
    preset: str,
    method: str,
    exam: str,
) -> JudgeImagePolicy:
    preset_l = (preset or "reference_only").strip().lower()
    if preset_l not in {"reference_only", "question_plus_reference"}:
        raise ValueError(f"Unsupported judge_image_preset: {preset}")
    return JudgeImagePolicy(
        preset=preset_l,
        method=(method or "").strip().lower(),
        exam=(exam or "").strip().lower(),
    )


def resolve_standard_iterative_question_images(
    policy: GenerationImagePolicy,
    *,
    global_question_images: Sequence[str],
    history_local_images: Sequence[str],
    current_local_images: Sequence[str],
) -> List[str]:
    mode = policy.iterative_image_mode
    if mode == "current_only":
        return unique_image_urls(current_local_images)
    if mode == "global_current":
        return unique_image_urls(global_question_images, current_local_images)
    if mode == "cumulative":
        return unique_image_urls(global_question_images, history_local_images, current_local_images)
    raise ValueError(f"Unsupported resolved iterative image mode: {mode}")


def resolve_standard_interleaved_background_images(
    policy: GenerationImagePolicy,
    *,
    global_question_images: Sequence[str],
) -> List[str]:
    if policy.interleaved_background_global_persist:
        return unique_image_urls(global_question_images)
    return []


def resolve_standard_interleaved_turn_images(
    policy: GenerationImagePolicy,
    *,
    global_question_images: Sequence[str],
    current_local_images: Sequence[str],
) -> Dict[str, List[str]]:
    replay_global_images = unique_image_urls(global_question_images) if policy.interleaved_global_image_replay else []
    call_image_urls = unique_image_urls(replay_global_images, current_local_images)
    persisted_image_urls = unique_image_urls(
        replay_global_images if policy.interleaved_replay_global_persist else [],
        current_local_images,
    )
    return {
        "call_image_urls": call_image_urls,
        "persisted_image_urls": persisted_image_urls,
        "replay_global_image_urls": replay_global_images,
    }


def resolve_ipho_iterative_question_images(parsed_item: Dict[str, Any]) -> List[str]:
    return ipho_item_question_images(parsed_item)


def resolve_ipho_interleaved_turn_images(parsed_item: Dict[str, Any]) -> Dict[str, List[str]]:
    images = ipho_item_question_images(parsed_item)
    return {
        "call_image_urls": images,
        "persisted_image_urls": images,
        "replay_global_image_urls": [],
    }
