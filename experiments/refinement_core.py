"""One prospective extraction instruction; frozen geometry and answer parser."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import random

from region_selection_core import (Observation, candidate_boxes, build_selector_request,
                                   build_answer_request, canonical_hash)
from native_detail_core import ANSWER_INSTRUCTION

STRICT_ANSWER_INSTRUCTION = (
    'Return only the shortest text span that directly answers the question, copied from the page. '
    'Keep required units, currency signs, punctuation and number formatting as printed. '
    'Do not add a sentence, explanation, field label, adjacent value or identifier unless the question asks for it. '
    'Write one answer on a single line.'
)
NEW_ACTIONS = ('new_direct', 'new_highres', *(f'new_native_{i}' for i in range(1, 10)))
GENERATION_ACTIONS = ('legacy_selector', 'legacy_selected', *NEW_ACTIONS)
PREFLIGHT_ACTIONS = ('legacy_selector', *NEW_ACTIONS, *(f'legacy_native_{i}' for i in range(1, 10)))


def make_request(source, question, action, config, region=None):
    if action not in (*PREFLIGHT_ACTIONS, 'legacy_selected'):
        raise ValueError('Unknown refinement action')
    if action == 'legacy_selector':
        return build_selector_request(source, question, config)
    if action == 'legacy_selected':
        if type(region) is not int or region not in range(1, 10):
            raise ValueError('Legacy selection must be a committed region ID')
        action = f'legacy_native_{region}'
    if action.startswith('legacy_native_'):
        return build_answer_request(source, question, 'native', config, int(action.rsplit('_', 1)[1]))
    if action not in NEW_ACTIONS:
        raise ValueError('Unknown refinement action')
    if action.startswith('new_native_'):
        kind, region = 'native', int(action.rsplit('_', 1)[1])
    else:
        kind, region = action.removeprefix('new_'), None
    messages, images, geometry = build_answer_request(source, question, kind, config, region)
    old_text = messages[-1]['content'][-1]['text']
    if not old_text.endswith(ANSWER_INSTRUCTION):
        raise ValueError('Frozen instruction suffix differs')
    text = old_text[:-len(ANSWER_INSTRUCTION)] + STRICT_ANSWER_INSTRUCTION
    messages[-1]['content'][-1]['text'] = text
    geometry.update(question_prompt=text, extraction_instruction_version='minimal_span_v1',
                    messages_sha256=canonical_hash(messages))
    return messages, images, geometry


def action_order(example_id, seed=20260927):
    actions = list(NEW_ACTIONS)
    rng = random.Random(int.from_bytes(hashlib.sha256(f'refinement-order:{seed}:{example_id}'.encode()).digest(), 'big'))
    rng.shuffle(actions)
    return ['legacy_selector', 'legacy_selected', *actions]


def preflight_key(action, region=None):
    return f'legacy_native_{region}' if action == 'legacy_selected' else action


def normalized_boxes(size):
    w, h = size
    return [[x0/w, y0/h, x1/w, y1/h] for x0,y0,x1,y1 in candidate_boxes(size).values()]
