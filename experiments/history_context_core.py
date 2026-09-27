"""V4 history intervention using immutable v3 pixels and answer scoring."""
from __future__ import annotations

import hashlib
import json

from native_detail_core import (
    REGIONS, ANSWER_INSTRUCTION, CROP_PROMPT, REPEAT_PROMPT,
    build_request as build_v3_request, score_response,
)

HISTORIES = ("actual", "fresh", "placeholder")
FIDELITIES = ("native", "degraded")
ACTIONS = ["direct", "highres", "repeat", *(
    f"{history}_{fidelity}_{region}"
    for history in HISTORIES for fidelity in FIDELITIES for region in REGIONS
)]
PLACEHOLDER = "I have viewed the page."
FRESH_DESCRIPTION = "The first image is the full page. The second image is an enlarged region of the first image."


def build_request(source, question, action, config, initial_answer=None):
    """Build independent branches; neither target nor other branch answers enter."""
    if action not in ACTIONS:
        raise ValueError(f"Unknown history-context action: {action}")
    if action in ("direct", "highres", "repeat"):
        history = "actual" if action == "repeat" else "none"
        base_action = action
    else:
        history, fidelity, region = action.split("_")
        base_action = f"{fidelity}_{region}"
    if history == "actual" and not isinstance(initial_answer, str):
        raise ValueError("Actual history requires the fresh direct response")
    inserted = initial_answer if history == "actual" else PLACEHOLDER
    messages, images, geometry = build_v3_request(source, question, base_action, config, inserted)
    if history == "fresh":
        prompt = FRESH_DESCRIPTION + "\n" + question + "\n" + ANSWER_INSTRUCTION
        messages = [messages[0], {"role": "user", "content": [
            {"type": "image"}, {"type": "image"}, {"type": "text", "text": prompt}]}]
        geometry.update(question_prompt=prompt, followup_prompt=None)
    actual_inserted = inserted if history in ("actual", "placeholder") else None
    geometry.update(
        history_mode=history,
        previous_answer_in_prompt=history == "actual",
        inserted_history_text=actual_inserted,
        initial_answer_sha256=(hashlib.sha256(actual_inserted.encode("utf-8")).hexdigest()
                               if actual_inserted is not None else None),
        messages_sha256=hashlib.sha256(json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest(),
    )
    return messages, images, geometry
