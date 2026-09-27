"""CPU-only adversarial adapter checks; no weights, labels or real responses."""
import copy
import hashlib
from pathlib import Path
import struct
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import refinement_features as f

BOXES = [[x, y, x + .5, y + .5] for y in (0., .25, .5) for x in (0., .25, .5)]
MESSAGES = [{"role": "system", "content": "Select a region."},
            {"role": "user", "content": [{"type": "image"},
                                            {"type": "text", "text": "Question: What is shown? Fixed instructions."}]}]


class FakeProcessor:
    def __init__(self, inputs):
        self.inputs = inputs
        self.expected = {"verified_cpu_tensors": "fixture"}
        self.last = None
        self.seen_keys = None
        self.bad_capture = False

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True}
        assert messages == MESSAGES
        return "<fixed synthetic prompt>"

    def __call__(self, **kwargs):
        assert kwargs["text"] == ["<fixed synthetic prompt>"]
        assert kwargs["return_tensors"] == "pt"
        self.seen_keys = set(self.inputs)
        self.last = {"tampered": True} if self.bad_capture else copy.deepcopy(self.expected)
        return {key: value.clone() for key, value in self.inputs.items()}


class FakeDecoder:
    def __init__(self, hidden, processor):
        self.hidden = hidden
        self.processor = processor
        self.rope_deltas = "stale-from-another-source"
        self.calls = []
        self.fail = False

    def __call__(self, **kwargs):
        assert self.processor.last == self.processor.expected
        assert self.rope_deltas is None
        assert kwargs["use_cache"] is False and kwargs["past_key_values"] is None
        assert kwargs["return_dict"] is True
        assert kwargs["output_hidden_states"] is False and kwargs["output_attentions"] is False
        assert "token_type_ids" not in kwargs and "labels" not in kwargs
        self.calls.append(kwargs)
        self.rope_deltas = "memo-from-current-source"
        if self.fail:
            raise RuntimeError("synthetic decoder failure")
        return SimpleNamespace(last_hidden_state=self.hidden)


class FakeModel:
    training = False
    device = torch.device("cpu")

    def __init__(self, decoder):
        self.model = decoder

    def __call__(self, *args, **kwargs):
        raise AssertionError("The language-model head must not be used")

    def generate(self, *args, **kwargs):
        raise AssertionError("Generation must not be used")


def backend(padding="none"):
    ids = [42, f.VISION_START_TOKEN_ID] + [f.IMAGE_TOKEN_ID] * 16 + [f.VISION_END_TOKEN_ID, 90, 91]
    attention = [1] * len(ids)
    if padding == "right":
        ids += [151643, 151643]
        attention += [0, 0]
    elif padding == "left":
        ids = [151643, 151643] + ids
        attention = [0, 0] + attention
    positions = [i for i, token in enumerate(ids) if token == f.IMAGE_TOKEN_ID]
    query = max(i for i, a in enumerate(attention) if a)
    hidden = torch.zeros((1, len(ids), 2560), dtype=torch.float32)
    for n, pos in enumerate(positions):
        row, col = divmod(n, 4)
        hidden[0, pos, :3] = torch.tensor([10 * row + col, 2 * (10 * row + col), 1.])
    hidden[0, query, :] = 777.
    if padding == "right":
        hidden[0, -1, :] = -999.
    inputs = {"input_ids": torch.tensor([ids]), "attention_mask": torch.tensor([attention]),
              "pixel_values": torch.zeros((64, 1536)), "image_grid_thw": torch.tensor([[1, 8, 8]]),
              "token_type_ids": torch.zeros((1, len(ids)), dtype=torch.int64)}
    processor = FakeProcessor(inputs)
    decoder = FakeDecoder(hidden, processor)
    result = SimpleNamespace(torch=torch, processor=processor, model=FakeModel(decoder))
    return result


def extract(b):
    return f.extract_vectors(b, copy.deepcopy(MESSAGES), [object()], copy.deepcopy(BOXES))


@pytest.mark.parametrize("padding", ["none", "right", "left"])
def test_known_row_major_region_means_query_padding_and_one_forward(padding):
    b = backend(padding)
    result = extract(b)
    # Hand-computed 2x2 means of the 4x4 matrix 10*row+column.
    means = np.array([5.5, 6.5, 7.5, 15.5, 16.5, 17.5, 25.5, 26.5, 27.5], dtype=np.float32)
    np.testing.assert_array_equal(result["region_vectors"][:, 0], means)
    np.testing.assert_array_equal(result["region_vectors"][:, 1], 2 * means)
    np.testing.assert_array_equal(result["region_vectors"][:, 2], np.ones(9))
    np.testing.assert_array_equal(result["query_vector"], np.full(2560, 777., dtype=np.float32))
    assert result["region_vectors"].dtype == np.float32 and result["query_vector"].dtype == np.float32
    assert result["region_vectors"].shape == (9, 2560)
    assert result["image_region_token_counts"] == [4] * 9
    assert result["visual_tokens"] == 16
    assert result["query_token_semantics"] == "final_nonpadding_prompt_token"
    assert len(b.model.model.calls) == 1
    assert b.model.model.rope_deltas is None
    assert "token_type_ids" in b.processor.seen_keys
    assert "token_type_ids" not in b.model.model.calls[0]
    assert result["raw_feature_sha256"] == f.raw_feature_digest(result["region_vectors"], result["query_vector"])


def test_box_center_boundary_is_half_open_and_rectangular_grid_is_not_transposed():
    # Merged grid is 2 rows x 4 columns. Token centers x=.125,.375,.625,.875.
    vectors = np.array([[0.], [1.], [2.], [3.], [10.], [11.], [12.], [13.]], dtype=np.float32)
    boxes = [[.125, .25, .625, .75]] * 9
    pooled, counts = f.pool_image_tokens(vectors, [1, 4, 8], boxes)
    # Includes upper row columns0,1; x=.625 and y=.75 are excluded.
    np.testing.assert_array_equal(pooled[:, 0], np.full(9, .5))
    assert counts == [2] * 9


def test_digest_has_independently_specified_shape_order_and_float32_bytes():
    regions = np.arange(9 * 2560, dtype=np.float32).reshape(9, 2560)
    query = np.arange(2560, dtype=np.float32) * -1
    expected = hashlib.sha256(b"lookagain-refinement-raw-features-v1\0" + struct.pack("<III", 9, 2560, 2560)
                              + regions.astype("<f4").tobytes() + query.astype("<f4").tobytes()).hexdigest()
    assert f.raw_feature_digest(regions, query) == expected
    assert f.raw_feature_digest(regions[::-1], query) != expected
    query[-1] += 1
    assert f.raw_feature_digest(regions, query) != expected


def test_bfloat16_hidden_state_is_copied_to_detached_float32():
    b = backend()
    b.model.model.hidden = b.model.model.hidden.to(torch.bfloat16)
    result = extract(b)
    assert result["region_vectors"].dtype == np.float32
    # 777 is not exactly representable in BF16; do not pretend extraction restores it.
    assert result["query_vector"][0] == float(torch.tensor(777., dtype=torch.bfloat16))
    b.model.model.hidden.zero_()
    assert result["region_vectors"][0, 0] == 5.5


@pytest.mark.parametrize("defect", ["capture_mismatch", "capture_missing", "training", "second_grid",
                                    "temporal_grid", "odd_grid", "float_grid", "wrong_count",
                                    "noncontiguous_images", "image_in_padding", "mask_hole",
                                    "mask_nonbinary", "float_ids", "extra_label_tensor", "bad_frame"])
def test_malformed_input_fails_before_any_model_forward(defect):
    b = backend()
    x = b.processor.inputs
    if defect == "capture_mismatch": b.processor.bad_capture = True
    elif defect == "capture_missing": b.processor.expected = None
    elif defect == "training": b.model.training = True
    elif defect == "second_grid": x["image_grid_thw"] = torch.tensor([[1, 8, 8], [1, 8, 8]])
    elif defect == "temporal_grid": x["image_grid_thw"][0, 0] = 2
    elif defect == "odd_grid": x["image_grid_thw"][0, 1] = 7
    elif defect == "float_grid": x["image_grid_thw"] = x["image_grid_thw"].float()
    elif defect == "wrong_count": x["image_grid_thw"][0, 1] = 6
    elif defect == "noncontiguous_images": x["input_ids"][0, 5], x["input_ids"][0, 19] = 90, f.IMAGE_TOKEN_ID
    elif defect == "image_in_padding": x["attention_mask"][0, :3] = 0
    elif defect == "mask_hole": x["attention_mask"][0, 7] = 0
    elif defect == "mask_nonbinary": x["attention_mask"][0, 7] = 2
    elif defect == "float_ids": x["input_ids"] = x["input_ids"].float()
    elif defect == "extra_label_tensor": x["labels"] = torch.ones_like(x["input_ids"])
    elif defect == "bad_frame": x["input_ids"][0, 1] = 90
    with pytest.raises(ValueError):
        extract(b)
    assert b.model.model.calls == []


@pytest.mark.parametrize("box", [[False, 0, .5, .5], [0, 0, float("nan"), .5],
                                 [0, 0, .5, float("inf")], [0, 0, 0, .5], [-.1, 0, .5, .5]])
def test_invalid_normalized_boxes_rejected_before_forward(box):
    b = backend()
    boxes = copy.deepcopy(BOXES)
    boxes[0] = box
    with pytest.raises(ValueError):
        f.extract_vectors(b, MESSAGES, [object()], boxes)
    assert b.model.model.calls == []


def test_empty_token_region_rejected_instead_of_silent_nan_pool():
    with pytest.raises(ValueError, match="no merged-token center"):
        f.pool_image_tokens(np.zeros((16, 3), dtype=np.float32), [1, 8, 8], [[0, 0, .01, .01]] * 9)


@pytest.mark.parametrize("position", [2, 20])
def test_nonfinite_image_or_query_hidden_state_rejected(position):
    b = backend()
    b.model.model.hidden[0, position, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        extract(b)
    assert b.model.model.rope_deltas is None


def test_decoder_failure_clears_mutable_rope_memo():
    b = backend()
    b.model.model.fail = True
    with pytest.raises(RuntimeError, match="synthetic decoder failure"):
        extract(b)
    assert b.model.model.rope_deltas is None


def test_wrong_hidden_dimension_rejected():
    b = backend()
    b.model.model.hidden = b.model.model.hidden[:, :, :128]
    with pytest.raises(ValueError, match="shape"):
        extract(b)


@pytest.mark.parametrize("defect", ["history", "text_before_image", "extra_annotation", "two_images"])
def test_message_history_and_nonobservation_payloads_rejected(defect):
    b = backend()
    messages = copy.deepcopy(MESSAGES)
    images = [object()]
    if defect == "history": messages.append({"role": "assistant", "content": "Old answer"})
    elif defect == "text_before_image": messages[1]["content"].reverse()
    elif defect == "extra_annotation": messages[1]["content"][1]["answer"] = "Forbidden"
    elif defect == "two_images": images.append(object())
    with pytest.raises(ValueError):
        f.extract_vectors(b, messages, images, BOXES)
    assert b.model.model.calls == []
