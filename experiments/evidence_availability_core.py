"""V5 label-privileged ROI controls; targets never enter the language prompt."""
from __future__ import annotations

import hashlib
import json

from PIL import Image

from native_detail_core import bounded_size, image_digest, score_response, ANSWER_INSTRUCTION
from history_context_core import FRESH_DESCRIPTION
from lookagain.actions import SYSTEM_INSTRUCTION

BUDGETS = (256, 512, 1024)
FIDELITIES = ("native", "degraded")
ACTIONS = [*(f"{kind}_{budget}" for budget in BUDGETS for kind in ("direct", *FIDELITIES)), "highres"]


def parse_action(action):
    if action not in ACTIONS:
        raise ValueError(f"Unknown evidence-availability action: {action}")
    return ("highres", 4096) if action == "highres" else (action.split("_")[0], int(action.split("_")[1]))


def validate_roi(roi, size):
    if (not isinstance(roi, (list, tuple)) or len(roi) != 4
            or any(type(x) is not int for x in roi)):
        raise ValueError("ROI must contain four integer source-pixel coordinates")
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]):
        raise ValueError("ROI is outside the source image")
    if min(x1 - x0, y1 - y0) < 256:
        raise ValueError("Context ROI sides must be at least 256 pixels")
    return tuple(roi)


def build_request(source, question, action, config, roi):
    """Fresh single-turn requests; annotation selects geometry only, never text."""
    kind, budget = parse_action(action)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Nonempty question string required")
    source = source if source.mode == "RGB" else source.convert("RGB")
    box = validate_roi(roi, source.size)
    overview = source.resize(bounded_size(source.size, budget), Image.Resampling.BICUBIC)
    images = [overview]
    text = question + "\n" + ANSWER_INSTRUCTION
    content = [{"type": "image"}, {"type": "text", "text": text}]
    geometry = dict(condition=kind, overview_budget=budget, history_mode="fresh", previous_answer_in_prompt=False,
                    source_size=list(source.size), overview_size=list(overview.size), additional_size=None,
                    source_roi_pixels=None, projected_overview_roi_pixels=None, crop_box_normalized=None,
                    roi_used=False, roi_localizer="reference_annotation_privileged", resampling="BICUBIC",
                    system_prompt=SYSTEM_INSTRUCTION, question_prompt=text, additional_view_kind=None,
                    crop_visual_token_budget=config["crop_visual_tokens"])
    if kind in FIDELITIES:
        x0, y0, x1, y1 = box
        target = bounded_size((x1 - x0, y1 - y0), config["crop_visual_tokens"])
        if target[0] * target[1] < 65536:
            raise ValueError("ROI would trigger hidden processor minimum-pixel upsampling")
        projected = (x0 * overview.width / source.width, y0 * overview.height / source.height,
                     x1 * overview.width / source.width, y1 * overview.height / source.height)
        additional = (source.crop(box).resize(target, Image.Resampling.BICUBIC) if kind == "native"
                      else overview.resize(target, Image.Resampling.BICUBIC, box=projected))
        images.append(additional)
        text = FRESH_DESCRIPTION + "\n" + question + "\n" + ANSWER_INSTRUCTION
        content = [{"type": "image"}, {"type": "image"}, {"type": "text", "text": text}]
        geometry.update(additional_size=list(target), source_roi_pixels=list(box),
                        projected_overview_roi_pixels=list(projected), roi_used=True, question_prompt=text,
                        crop_box_normalized=[x0/source.width, y0/source.height, x1/source.width, y1/source.height],
                        additional_view_kind="source_render_roi" if kind == "native" else "overview_derived_roi")
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": content}]
    geometry["image_rgb_sha256"] = [image_digest(image) for image in images]
    geometry["messages_sha256"] = hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    return messages, images, geometry
