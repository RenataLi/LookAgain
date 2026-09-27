"""Given-page region selection with an explicit annotation-free input boundary."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import random
import re
from pathlib import Path
import sys

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/'src'), str(PROJECT/'experiments')]
from native_detail_core import bounded_size, image_digest
from evidence_availability_core import build_request as evidence_request

OBSERVATION_KEYS = {'example_id','question','image_path','image_sha256','source_cluster_id'}
SELECTOR_SYSTEM = ('Select one viewing window for a question about a document image. '
                   'Return exactly REGION: followed by one digit from 1 to 9. '
                   'Do not answer the question or explain.')
SELECTOR_INSTRUCTION = '''Choose the window most likely to contain the evidence needed to answer the question.
The image is the full page overview. Each window is half the page width and half the page height.
Coordinates below are normalized to the full page, with (0,0) at top left and (1,1) at bottom right.
1: left 0.00, top 0.00, right 0.50, bottom 0.50
2: left 0.25, top 0.00, right 0.75, bottom 0.50
3: left 0.50, top 0.00, right 1.00, bottom 0.50
4: left 0.00, top 0.25, right 0.50, bottom 0.75
5: left 0.25, top 0.25, right 0.75, bottom 0.75
6: left 0.50, top 0.25, right 1.00, bottom 0.75
7: left 0.00, top 0.50, right 0.50, bottom 1.00
8: left 0.25, top 0.50, right 0.75, bottom 1.00
9: left 0.50, top 0.50, right 1.00, bottom 1.00
Output exactly REGION: <digit>.'''

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

@dataclass(frozen=True)
class Observation:
    example_id: str
    question: str
    image_path: str
    image_sha256: str
    source_cluster_id: str

    @classmethod
    def from_manifest(cls, row):
        if not isinstance(row, dict) or set(row) != OBSERVATION_KEYS:
            raise ValueError('Inference observations must contain exactly the five allowed fields; labels/ROI are forbidden')
        if any(not isinstance(v, str) or not v.strip() for v in row.values()):
            raise ValueError('Observation fields must be nonempty strings')
        if not re.fullmatch('[0-9a-f]{64}', row['image_sha256']):
            raise ValueError('Invalid source image SHA256')
        return cls(**row)

def candidate_boxes(size):
    if len(size) != 2 or any(type(v) is not int or v < 512 for v in size):
        raise ValueError('Source dimensions must be integer values at least 512')
    w,h = size
    cw,ch = w//2,h//2
    xs = [round((w-cw)*i/2) for i in range(3)]
    ys = [round((h-ch)*i/2) for i in range(3)]
    return {1+3*j+i:(x,y,x+cw,y+ch) for j,y in enumerate(ys) for i,x in enumerate(xs)}

def parse_region(response, truncated=False):
    if not isinstance(response, str) or type(truncated) is not bool:
        raise TypeError('Selector response and truncation need string/bool values')
    match = None if truncated else re.fullmatch(r'\s*REGION:\s*([1-9])\s*', response)
    return (int(match.group(1)), True) if match else (5, False)

def random_region(example_id, seed):
    if not isinstance(example_id, str) or not example_id or type(seed) is not int:
        raise ValueError('A nonempty ID and integer seed are required')
    digest = hashlib.sha256(f'region-random:{seed}:{example_id}'.encode()).digest()
    return random.Random(int.from_bytes(digest,'big')).choice(range(1,10))

def build_selector_request(source, question, config):
    if not isinstance(question,str) or not question.strip():
        raise ValueError('Nonempty question required')
    source = source if source.mode=='RGB' else source.convert('RGB')
    boxes = candidate_boxes(source.size)
    overview = source.resize(bounded_size(source.size,config['base_visual_tokens']),Image.Resampling.BICUBIC)
    text = 'Question: '+question+'\n'+SELECTOR_INSTRUCTION
    messages = [{'role':'system','content':SELECTOR_SYSTEM},
                {'role':'user','content':[{'type':'image'},{'type':'text','text':text}]}]
    geometry = {'source_size':list(source.size),'overview_size':list(overview.size),
                'image_rgb_sha256':[image_digest(overview)], 'messages_sha256':canonical_hash(messages),
                'candidate_boxes':{str(k):list(v) for k,v in boxes.items()},
                'roi_localizer':'fixed_geometry_no_annotations','question_prompt':text,
                'system_prompt':SELECTOR_SYSTEM,'selector_sees_high_resolution_crops':False}
    return messages,[overview],geometry

def build_answer_request(source, question, action, config, region_id=None):
    if action not in ('direct','highres','native','degraded'):
        raise ValueError('Unknown answer action')
    boxes = candidate_boxes(source.size)
    if action in ('native','degraded'):
        if type(region_id) is not int or region_id not in boxes:
            raise ValueError('A region ID from 1 to 9 is required')
    elif region_id is not None:
        raise ValueError('Standalone actions do not take a region')
    mapped = 'highres' if action=='highres' else f'{action}_256'
    if config['base_visual_tokens'] != 256 or config['highres_visual_tokens'] != 4096:
        raise ValueError('This protocol fixes overview/high-resolution ceilings at 256/4096')
    messages,images,geometry = evidence_request(source,question,mapped,config,boxes[region_id or 5])
    geometry['roi_localizer'] = 'fixed_geometry_no_annotations'
    geometry['region_id'] = region_id
    geometry['answer_page_privileged'] = True
    geometry['roi_annotation_used'] = False
    return messages,images,geometry
