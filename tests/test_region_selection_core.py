"""Independent source-only tests for the new fixed-grid selector interface.

No VLM, source labels, real model responses or GPU execution are used.
"""
from dataclasses import FrozenInstanceError
import random
from pathlib import Path
import sys

import pytest
from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'experiments'),
               str(Path(__file__).resolve().parents[1] / 'src')]
import region_selection_core as core
from native_detail_core import bounded_size, image_digest


CONFIG = {'base_visual_tokens': 256, 'crop_visual_tokens': 1024,
          'highres_visual_tokens': 4096, 'selector_max_new_tokens': 16,
          'answer_max_new_tokens': 64, 'seed': 20260927, 'attention': 'sdpa'}


@pytest.fixture
def observation_row():
    return {'example_id': 'source-only-example', 'question': 'Which code is shown?',
            'image_path': 'images/page.png', 'image_sha256': 'a' * 64,
            'source_cluster_id': 'cluster:source-only'}


@pytest.fixture(scope='module')
def source():
    # Dense vertical patterns retain real fine detail that the overview loses.
    row = bytes(channel for x in range(1280)
                for channel in (255 * (x % 2), 255 * ((x // 3) % 2), (x * 17) % 256))
    return Image.frombytes('RGB', (1280, 1536), row * 1536)


def test_observation_copies_exact_inference_fields_and_is_frozen(observation_row):
    obj = core.Observation.from_manifest(observation_row)
    assert {key: getattr(obj, key) for key in observation_row} == observation_row
    observation_row['question'] = 'Mutated external mapping'
    assert obj.question == 'Which code is shown?'
    with pytest.raises(FrozenInstanceError):
        obj.question = 'No mutable observation'


@pytest.mark.parametrize('key', ['answer', 'original_answers', 'validated_primary_answers',
                               'roi_pixels', 'bbox', 'source_semantic_review',
                               'predicted_answer', 'correct', 'extra'])
def test_annotated_or_augmented_manifest_is_rejected(observation_row, key):
    with pytest.raises((ValueError, TypeError)):
        core.Observation.from_manifest({**observation_row, key: 'must not enter inference'})


@pytest.mark.parametrize('key', ['example_id', 'question', 'image_path', 'image_sha256', 'source_cluster_id'])
def test_missing_inference_field_rejected(observation_row, key):
    observation_row.pop(key)
    with pytest.raises((ValueError, TypeError, KeyError)):
        core.Observation.from_manifest(observation_row)


@pytest.mark.parametrize('region', range(1, 10))
def test_strict_region_accepts_single_uppercase_marker(region):
    assert core.parse_region(f'REGION: {region}') == (region, True)
    assert core.parse_region(f' \n\tREGION:\t{region} \n') == (region, True)


@pytest.mark.parametrize('response', ['', '5', 'R5', 'region: 5', 'Region: 5',
                                     'REGION: 0', 'REGION: 10', 'REGION: 05',
                                     'REGION: 5.', 'REGION: 5 because it is relevant',
                                     'REGION: 2\nREGION: 5', 'ANSWER: REGION: 5',
                                     'Reasoning first\nREGION: 5', 'REGION: 5\nANSWER: x',
                                     'REGION: <5>', 'REGION: ５'])
def test_invalid_region_falls_back_without_hidden_text_extraction(response):
    assert core.parse_region(response) == (5, False)


@pytest.mark.parametrize('region', [1, 5, 9])
def test_truncation_always_invalidates_even_valid_region(region):
    assert core.parse_region(f'REGION: {region}', truncated=True) == (5, False)


def test_center_parse_and_center_fallback_are_distinguishable():
    assert core.parse_region('REGION: 5') == (5, True)
    assert core.parse_region('') == (5, False)


def test_even_dimensions_have_independently_calculated_row_major_boxes():
    assert core.candidate_boxes((1600, 2000)) == {
        1: (0, 0, 800, 1000), 2: (400, 0, 1200, 1000), 3: (800, 0, 1600, 1000),
        4: (0, 500, 800, 1500), 5: (400, 500, 1200, 1500), 6: (800, 500, 1600, 1500),
        7: (0, 1000, 800, 2000), 8: (400, 1000, 1200, 2000), 9: (800, 1000, 1600, 2000)}


def test_odd_dimensions_use_equal_floor_sized_boxes_and_half_even_starts():
    boxes = core.candidate_boxes((1273, 1501))
    assert boxes[1] == (0, 0, 636, 750)
    assert boxes[2] == (318, 0, 954, 750)  # round(637/2) = 318
    assert boxes[5] == (318, 376, 954, 1126)  # round(751/2) = 376
    assert boxes[9] == (637, 751, 1273, 1501)
    assert {(x1-x0)*(y1-y0) for x0,y0,x1,y1 in boxes.values()} == {477000}
    assert 477000 <= 1273*1501/4


@pytest.mark.parametrize('size', [(512, 512), (513, 515), (1654, 2339), (4000, 1600)])
def test_all_boxes_are_equal_area_bounded_and_reach_outer_edges(size):
    boxes = core.candidate_boxes(size)
    assert set(boxes) == set(range(1, 10))
    widths, heights = set(), set()
    for box in boxes.values():
        assert isinstance(box, tuple) and len(box) == 4
        assert all(type(v) is int for v in box)
        x0,y0,x1,y1 = box
        assert 0 <= x0 < x1 <= size[0] and 0 <= y0 < y1 <= size[1]
        assert x1-x0 >= 256 and y1-y0 >= 256
        widths.add(x1-x0); heights.add(y1-y0)
    assert widths == {size[0]//2} and heights == {size[1]//2}
    assert boxes[1][:2] == (0, 0) and boxes[9][2:] == size


@pytest.mark.parametrize('size', [(511, 512), (512, 511), (True, 1000),
                                 (1024.0, 1024), (0, 1000), (1000,), (1000, 1000, 1000)])
def test_unsupported_candidate_dimensions_rejected(size):
    with pytest.raises((ValueError, TypeError)):
        core.candidate_boxes(size)


def test_random_region_is_source_stable_and_does_not_consume_global_rng():
    state = random.getstate()
    first = [core.random_region(f'example-{i}', 20260927) for i in range(80)]
    assert random.getstate() == state
    assert first == [core.random_region(f'example-{i}', 20260927) for i in range(80)]
    assert all(type(value) is int and 1 <= value <= 9 for value in first)
    assert len(set(first)) == 9
    assert first != [core.random_region(f'example-{i}', 20260928) for i in range(80)]


def test_selector_image_is_exact_unannotated_overview(source):
    question = 'Which code is shown?'
    messages, images, geometry = core.build_selector_request(source, question, CONFIG)
    expected = source.resize(bounded_size(source.size, 256), Image.Resampling.BICUBIC)
    assert len(images) == 1 and images[0].mode == 'RGB'
    assert images[0].size == expected.size and images[0].tobytes() == expected.tobytes()
    assert [message['role'] for message in messages] == ['system', 'user']
    assert question in str(messages)
    assert geometry['image_rgb_sha256'] == [image_digest(expected)]


@pytest.mark.parametrize('region', range(1, 10))
def test_answer_pairs_have_identical_language_geometry_and_original_overview(source, region):
    question = 'Which code is shown?'
    native = core.build_answer_request(source, question, 'native', CONFIG, region_id=region)
    degraded = core.build_answer_request(source, question, 'degraded', CONFIG, region_id=region)
    direct = core.build_answer_request(source, question, 'direct', CONFIG)
    assert native[0] == degraded[0]
    assert [message['role'] for message in native[0]] == ['system', 'user']
    assert [image.size for image in native[1]] == [image.size for image in degraded[1]]
    assert len(native[1]) == len(degraded[1]) == 2
    assert native[1][0].tobytes() == degraded[1][0].tobytes() == direct[1][0].tobytes()
    assert native[2]['messages_sha256'] == degraded[2]['messages_sha256']
    assert native[2]['source_roi_pixels'] == degraded[2]['source_roi_pixels'] == list(core.candidate_boxes(source.size)[region])
    assert native[2]['region_id'] == degraded[2]['region_id'] == region
    assert native[2]['roi_localizer'] == degraded[2]['roi_localizer'] == 'fixed_geometry_no_annotations'
    assert native[2]['image_rgb_sha256'][1] != degraded[2]['image_rgb_sha256'][1]
    assert native[2]['previous_answer_in_prompt'] is False
    # No selected ID/candidate description/history is inserted into answer text.
    assert 'REGION:' not in str(native[0])


@pytest.mark.parametrize('region', [1, 5, 9])
def test_native_and_degraded_pixels_match_independent_pillow_construction(source, region):
    box = core.candidate_boxes(source.size)[region]
    overview = source.resize(bounded_size(source.size, 256), Image.Resampling.BICUBIC)
    x0,y0,x1,y1 = box
    target = bounded_size((x1-x0, y1-y0), 1024)
    projected = (x0*overview.width/source.width, y0*overview.height/source.height,
                 x1*overview.width/source.width, y1*overview.height/source.height)
    expected_native = source.crop(box).resize(target, Image.Resampling.BICUBIC)
    expected_degraded = overview.resize(target, Image.Resampling.BICUBIC, box=projected)
    for action, expected in [('native', expected_native), ('degraded', expected_degraded)]:
        _, images, geometry = core.build_answer_request(source, 'Which code?', action, CONFIG, region_id=region)
        assert images[1].size == expected.size and images[1].tobytes() == expected.tobytes()
        assert geometry['projected_overview_roi_pixels'] == list(projected)


def test_degraded_depends_only_on_overview_under_controlled_overview_injection(monkeypatch):
    """Dataflow check, not a claim that arbitrary real sources resize identically."""
    first = Image.new('RGB', (1280, 1536), 'black')
    second = Image.new('RGB', (1280, 1536), 'white')
    overview = Image.new('RGB', bounded_size(first.size, 256), (33, 127, 201))
    for source_image in (first, second):
        def fixed_overview(size, resample=None, box=None, reducing_gap=None):
            assert size == overview.size and box is None
            return overview.copy()
        monkeypatch.setattr(source_image, 'resize', fixed_overview)
    degraded = [core.build_answer_request(im, 'Which code?', 'degraded', CONFIG, region_id=5)[1][1]
                for im in (first, second)]
    native = [core.build_answer_request(im, 'Which code?', 'native', CONFIG, region_id=5)[1][1]
              for im in (first, second)]
    assert degraded[0].tobytes() == degraded[1].tobytes()
    assert native[0].tobytes() != native[1].tobytes()


@pytest.mark.parametrize('region', [None, 0, 10, True, 1.0, '1'])
def test_regional_request_rejects_missing_or_invalid_candidate(source, region):
    with pytest.raises((ValueError, TypeError)):
        core.build_answer_request(source, 'Q?', 'native', CONFIG, region_id=region)


def test_standalone_controls_use_one_image_and_no_realized_roi(source):
    for action in ('direct', 'highres'):
        messages, images, geometry = core.build_answer_request(source, 'Q?', action, CONFIG)
        assert len(images) == 1
        assert [message['role'] for message in messages] == ['system', 'user']
        assert geometry['source_roi_pixels'] is None and geometry['roi_used'] is False
