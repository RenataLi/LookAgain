"""Matched document-detail controls; custom metrics, not official TAT-DQA scoring."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lookagain.actions import CROP_BOXES, SYSTEM_INSTRUCTION, extract_answer, pixel_box

REGIONS = ("tl", "tr", "bl", "br")
ACTIONS = ["direct", "highres", "repeat",
           *(f"{condition}_{region}" for condition in ("native", "degraded") for region in REGIONS)]
ANSWER_INSTRUCTION = "Answer by copying the relevant single text span from the page, including any units. Do not explain."
CROP_PROMPT = "The additional image is an enlarged region of the original image. Reconsider your answer using the available visual evidence. " + ANSWER_INSTRUCTION
REPEAT_PROMPT = "The additional image repeats the original image. Reconsider your answer using the available visual evidence. " + ANSWER_INSTRUCTION


def bounded_size(size, tokens):
    """32-pixel-grid dimensions within both source size and token ceiling."""
    if (len(size) != 2 or any(type(value) is not int or value < 32 for value in size)
            or type(tokens) is not int or tokens < 1):
        raise ValueError("size needs two integer dimensions >=32; tokens must be a positive integer")
    width, height = size
    scale = min(1.0, math.sqrt(tokens * 32 * 32 / (width * height)))
    new_width = max(32, int(width * scale / 32) * 32)
    new_height = max(32, int(height * scale / 32) * 32)
    while (new_width // 32) * (new_height // 32) > tokens:
        if new_width >= new_height and new_width > 32:
            new_width -= 32
        elif new_height > 32:
            new_height -= 32
        else:
            raise ValueError("cannot satisfy visual-token ceiling")
    return new_width, new_height


def image_digest(image):
    """Hash RGB pixels plus dimensions, independent of file encoding."""
    rgb = image if image.mode == "RGB" else image.convert("RGB")
    return hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode() + rgb.tobytes()).hexdigest()


def build_request(source, question, action, config, initial_answer=None):
    """Fresh chat, ordered PIL images, and realized pixel provenance; no labels."""
    if action not in ACTIONS:
        raise ValueError(f"unknown native-detail action: {action}")
    if not isinstance(question, str):
        raise TypeError("question must be a string")
    standalone = action in ("direct", "highres")
    if not standalone and not isinstance(initial_answer, str):
        raise ValueError("follow-up requires the fresh direct response string")
    source = source if source.mode == "RGB" else source.convert("RGB")
    budget = config["highres_visual_tokens"] if action == "highres" else config["base_visual_tokens"]
    overview = source.resize(bounded_size(source.size, budget), Image.Resampling.BICUBIC)
    question_prompt = question + "\n" + ANSWER_INSTRUCTION
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION},
                {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": question_prompt}]}]
    images = [overview]
    condition, region = (action, None) if action in ("direct", "highres", "repeat") else action.split("_")
    geometry = {
        "condition": condition, "region": region,
        "source_size": list(source.size), "overview_size": list(overview.size),
        "additional_size": None, "crop_size": None,
        "source_roi_pixels": None, "projected_overview_roi_pixels": None,
        "crop_box_pixels": None, "crop_box_normalized": None,
        "assigned_region_box": None, "actual_second_view_box": None,
        "additional_view_kind": None, "resampling": "BICUBIC",
        "overview_visual_token_budget": budget,
        "system_prompt": SYSTEM_INSTRUCTION, "question_prompt": question_prompt,
        "followup_prompt": None,
    }
    if not standalone:
        messages.append({"role": "assistant", "content": initial_answer})
        if action == "repeat":
            additional = overview.copy()
            prompt = REPEAT_PROMPT
            geometry.update(additional_view_kind="exact_overview_repeat",
                            actual_second_view_box=[0.0, 0.0, 1.0, 1.0])
        else:
            assigned = CROP_BOXES["crop_" + region]
            box = pixel_box(source.size, assigned)
            x0, y0, x1, y1 = box
            target_size = bounded_size((x1 - x0, y1 - y0), config["crop_visual_tokens"])
            projected = (x0 * overview.width / source.width, y0 * overview.height / source.height,
                         x1 * overview.width / source.width, y1 * overview.height / source.height)
            if condition == "native":
                additional = source.crop(box).resize(target_size, Image.Resampling.BICUBIC)
            else:
                additional = overview.resize(target_size, Image.Resampling.BICUBIC, box=projected)
            prompt = CROP_PROMPT
            geometry.update(
                source_roi_pixels=list(box), projected_overview_roi_pixels=list(projected),
                crop_box_pixels=list(box), crop_box_normalized=list(assigned),
                assigned_region_box=list(assigned), crop_size=list(target_size),
                actual_second_view_box=[x0 / source.width, y0 / source.height,
                                        x1 / source.width, y1 / source.height],
                additional_view_kind="source_render_crop" if condition == "native" else "base_overview_crop",
            )
        images.append(additional)
        messages.append({"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]})
        geometry.update(additional_size=list(additional.size), followup_prompt=prompt)
    geometry["image_rgb_sha256"] = [image_digest(image) for image in images]
    geometry["messages_sha256"] = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    geometry["initial_answer_sha256"] = None if standalone else hashlib.sha256(initial_answer.encode("utf-8")).hexdigest()
    return messages, images, geometry


def _levenshtein(first, second):
    if len(first) < len(second):
        first, second = second, first
    previous = list(range(len(second) + 1))
    for row, char_a in enumerate(first, 1):
        current = [row]
        for column, char_b in enumerate(second, 1):
            current.append(min(current[-1] + 1, previous[column] + 1,
                               previous[column - 1] + (char_a != char_b)))
        previous = current
    return previous[-1]


def score_response(response, target):
    """Custom EM and diagnostic ANLS for ONE text span, not official TAT-DQA.

    EM lowercases and collapses whitespace; punctuation/articles/numbers stay.
    Diagnostic ANLS preserves internal whitespace and uses strict NL < 0.5.
    """
    if not isinstance(response, str):
        raise TypeError("response must be a string")
    if not isinstance(target, str) or not target.strip():
        raise ValueError("target must be one nonempty string, not a list of spans")
    answer = extract_answer(response)
    normalized_target = " ".join(target.lower().split())
    normalized_prediction = " ".join(answer.lower().split()) if answer is not None else None
    correct = normalized_prediction is not None and normalized_prediction == normalized_target
    anls = 0.0
    if answer is not None and answer.strip():
        pred_text, target_text = answer.lower().strip(), target.lower().strip()
        distance = _levenshtein(pred_text, target_text) / max(len(pred_text), len(target_text))
        anls = 1.0 - distance if distance < 0.5 else 0.0
    return {
        "predicted_answer": answer, "parse_valid": answer is not None,
        "answer_format": "marked" if re.search(r"(?im)^ANSWER:", response) else "bare",
        "normalized_prediction": normalized_prediction, "normalized_target": normalized_target,
        "correct": correct, "conservative_text_em": float(correct),
        "anls": anls, "diagnostic_anls": anls,
    }
