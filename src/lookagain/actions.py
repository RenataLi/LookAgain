"""Action construction. Never uses answers, scene graphs or bounding-box labels."""
from __future__ import annotations

import math
import re

CROP_BOXES = {
    "crop_tl": (0.0, 0.0, 0.6, 0.6),
    "crop_tr": (0.4, 0.0, 1.0, 0.6),
    "crop_bl": (0.0, 0.4, 0.6, 1.0),
    "crop_br": (0.4, 0.4, 1.0, 1.0),
}
SYSTEM_INSTRUCTION = "You answer questions accurately based on the provided images."
ANSWER_INSTRUCTION = "Answer the question using a single word or short phrase. Do not explain."


def resize_for_tokens(image, tokens: int, factor: int = 32):
    """Use a fixed approximate visual-token budget, preserving aspect ratio.

    Explicit bicubic upsampling is allowed and logged; it adds no source detail.
    The 32-pixel factor is Qwen3-VL's patch size times spatial merge size.
    """
    from PIL import Image

    if tokens < 4 or tokens > 16384:
        raise ValueError("visual token budget must be between 4 and 16384")
    width, height = image.size
    scale = math.sqrt(tokens * factor * factor / (width * height))
    nw = max(factor, int(width * scale / factor) * factor)
    nh = max(factor, int(height * scale / factor) * factor)
    while (nw // factor) * (nh // factor) > tokens:
        if nw >= nh and nw > factor:
            nw -= factor
        elif nh > factor:
            nh -= factor
        else:
            raise ValueError("cannot fit image into token budget")
    return image.resize((nw, nh), Image.Resampling.BICUBIC)


def pixel_box(size, normalized_box):
    width, height = size
    x0, y0, x1, y1 = normalized_box
    return (int(x0 * width), int(y0 * height), min(width, math.ceil(x1 * width)), min(height, math.ceil(y1 * height)))


def build_request(source, question: str, action: str, config: dict, initial_answer: str | None = None):
    """Return chat messages, ordered PIL images and auditable geometry."""
    if action not in {"direct", "highres", "recheck", "think", *CROP_BOXES}:
        raise ValueError(f"unknown action: {action}")
    standalone = action in {"direct", "highres"}
    if not standalone and initial_answer is None:
        raise ValueError("follow-up action requires the original answer")
    budget = config["highres_visual_tokens"] if action == "highres" else config["base_visual_tokens"]
    overview = resize_for_tokens(source, budget)
    images = [overview]
    messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}, {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"{question}\n{ANSWER_INSTRUCTION}"}]}]
    geometry = {"source_size": list(source.size), "overview_size": list(overview.size), "crop_box_normalized": None, "crop_box_pixels": None}
    if not standalone:
        messages.append({"role": "assistant", "content": initial_answer})
        content = []
        if action in CROP_BOXES:
            box = pixel_box(source.size, CROP_BOXES[action])
            crop = resize_for_tokens(source.crop(box), config["crop_visual_tokens"])
            images.append(crop)
            geometry.update(crop_box_normalized=list(CROP_BOXES[action]), crop_box_pixels=list(box), crop_size=list(crop.size))
            content.append({"type": "image"})
            location = {"crop_tl": "upper-left", "crop_tr": "upper-right", "crop_bl": "lower-left", "crop_br": "lower-right"}[action]
            prompt = f"The additional image is an enlarged {location} region of the original image. Reconsider your answer using the available visual evidence. " + ANSWER_INSTRUCTION
        elif action == "recheck":
            images.append(overview.copy())
            prompt = "The additional image repeats the original image. Reconsider your answer using the available visual evidence. " + ANSWER_INSTRUCTION
            content.append({"type": "image"})
        else:
            prompt = "Reconsider your answer carefully using only the image already provided. Give a brief reasoning, then give the actual short answer on a final line beginning with ANSWER:."
        content.append({"type": "text", "text": prompt})
        messages.append({"role": "user", "content": content})
    return messages, images, geometry


def extract_answer(text: str):
    """Accept a unique final marker or a bare one-line answer.

    A bare response is compared in its entirety, never searched for a target.
    Multi-line reasoning requires a final marker; no semantic extraction.
    """
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    markers = sum(bool(re.match(r"(?i)^ANSWER:", line)) for line in lines)
    if len(lines) == 1 and markers == 0:
        return lines[0] if "<" not in lines[0] and ">" not in lines[0] else None
    if not lines or markers != 1:
        return None
    match = re.fullmatch(r"(?i)ANSWER:[ \t]*(\S.*)", lines[-1])
    answer = match.group(1).strip() if match else None
    return answer if answer and "<" not in answer and ">" not in answer else None


def normalize_answer(text: str):
    # Deliberately conservative: no semantic judge, article/number substitutions.
    return " ".join(text.strip().casefold().rstrip(".!?").split())


def grade(text: str, target: str):
    answer = extract_answer(text)
    return {"predicted_answer": answer, "parse_valid": answer is not None, "answer_format": "marked" if re.search(r"(?im)^ANSWER:", text) else "bare", "correct": answer is not None and normalize_answer(answer) == normalize_answer(target)}
