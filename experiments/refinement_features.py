"""One frozen Qwen3-VL decoder forward for label-free region features.

No generation, language-model head, learned projection, scoring, or labels.
The query is the final attended *prompt* token, after question and fixed
instructions; it is not a pure question embedding. With the fixed image-first
chat template, image-token states cannot attend to the later question.

The caller owns source loading/resizing, locked CPU-input validation, timing,
allocator measurement, and immutable feature persistence.
"""
from __future__ import annotations

import hashlib
import math
import struct

import numpy as np

IMAGE_TOKEN_ID = 151655
VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653
HIDDEN_SIZE = 2560
QUERY_TOKEN_SEMANTICS = "final_nonpadding_prompt_token"


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_boxes(boxes_normalized):
    _require(isinstance(boxes_normalized, (list, tuple)) and len(boxes_normalized) == 9,
             "Exactly nine normalized candidate boxes are required")
    rows = []
    for box in boxes_normalized:
        _require(isinstance(box, (list, tuple)) and len(box) == 4, "Each box must contain four coordinates")
        _require(all(type(x) in (int, float) and math.isfinite(x) for x in box),
                 "Box coordinates must be finite numeric values, not booleans")
        x0, y0, x1, y1 = map(float, box)
        _require(0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1, "Invalid normalized box bounds")
        rows.append((x0, y0, x1, y1))
    return rows


def _grid_shape(grid_thw):
    _require(isinstance(grid_thw, (list, tuple)) and len(grid_thw) == 3,
             "One image grid triple is required")
    _require(all(type(x) is int and x > 0 for x in grid_thw), "Grid entries must be positive integers")
    t, h, w = grid_thw
    _require(t == 1 and h % 2 == 0 and w % 2 == 0, "Require a still image with a 2x2 spatial merge")
    return h // 2, w // 2


def pool_image_tokens(image_vectors, grid_thw, boxes_normalized):
    """Mean final decoder image states whose merged-cell centers lie in a box.

Merged cells are in (row, column) order. Bounds are half-open on right/bottom.
These spatially indexed states have a globally contextualized visual encoder
and causal decoder context; they are not strictly local pixel descriptors.
"""
    boxes = validate_boxes(boxes_normalized)
    mh, mw = _grid_shape(grid_thw)
    image_vectors = np.asarray(image_vectors)
    _require(image_vectors.ndim == 2 and image_vectors.shape[0] == mh * mw
             and image_vectors.shape[1] > 0, "Image-state count differs from the merged grid")
    _require(np.issubdtype(image_vectors.dtype, np.floating) and np.isfinite(image_vectors).all(),
             "Image states must be finite floating-point values")
    vectors = np.asarray(image_vectors, dtype=np.float32)
    _require(np.isfinite(vectors).all(), "Image-state conversion overflowed float32")
    cx = np.tile((np.arange(mw, dtype=np.float64) + 0.5) / mw, mh)
    cy = np.repeat((np.arange(mh, dtype=np.float64) + 0.5) / mh, mw)
    pooled, counts = [], []
    for x0, y0, x1, y1 in boxes:
        mask = (cx >= x0) & (cx < x1) & (cy >= y0) & (cy < y1)
        n = int(mask.sum())
        _require(n > 0, "A candidate box contains no merged-token center")
        counts.append(n)
        # Explicit float64 accumulation avoids order-dependent low-precision sums.
        pooled.append(vectors[mask].mean(axis=0, dtype=np.float64).astype(np.float32))
    result = np.ascontiguousarray(np.stack(pooled), dtype=np.float32)
    _require(np.isfinite(result).all(), "Region pooling produced nonfinite values")
    return result, counts


def raw_feature_digest(region_vectors, query_vector):
    """SHA256 of versioned, shape-bound, little-endian float32 row-major arrays."""
    regions = np.asarray(region_vectors)
    query = np.asarray(query_vector)
    _require(regions.shape == (9, HIDDEN_SIZE) and query.shape == (HIDDEN_SIZE,),
             "Raw feature shapes must be (9,2560) and (2560,)")
    _require(np.issubdtype(regions.dtype, np.floating) and np.issubdtype(query.dtype, np.floating)
             and np.isfinite(regions).all() and np.isfinite(query).all(), "Raw features must be finite floats")
    r, q = np.ascontiguousarray(regions, dtype="<f4"), np.ascontiguousarray(query, dtype="<f4")
    _require(np.isfinite(r).all() and np.isfinite(q).all(), "Raw feature conversion overflowed float32")
    digest = hashlib.sha256(b"lookagain-refinement-raw-features-v1\0")
    digest.update(struct.pack("<III", 9, HIDDEN_SIZE, HIDDEN_SIZE))
    digest.update(r.tobytes(order="C"))
    digest.update(q.tobytes(order="C"))
    return digest.hexdigest()


def _validate_messages(messages):
    _require(isinstance(messages, list) and len(messages) == 2,
             "Feature extraction requires exactly system and user messages, with no history")
    system, user = messages
    _require(isinstance(system, dict) and set(system) == {"role", "content"}
             and system["role"] == "system" and isinstance(system["content"], str)
             and bool(system["content"].strip()), "Invalid feature system message")
    _require(isinstance(user, dict) and set(user) == {"role", "content"} and user["role"] == "user",
             "Invalid feature user message")
    content = user["content"]
    _require(isinstance(content, list) and len(content) == 2 and content[0] == {"type": "image"},
             "Exactly one image must precede the query text")
    text = content[1]
    _require(isinstance(text, dict) and set(text) == {"type", "text"} and text["type"] == "text"
             and isinstance(text["text"], str) and bool(text["text"].strip()), "Invalid feature query text")


def extract_vectors(backend, messages, images, boxes_normalized):
    """Extract nine raw region vectors and one prompt-query vector.

`backend.processor` must be the existing ProcessorCapture with a non-null
`expected` CPU preflight capture. Its normal call checks all tensors before any
device transfer. Batch size is deliberately one; padding is supported only as
a contiguous attended span. No answer or annotation argument is accepted.
"""
    _validate_messages(messages)
    validate_boxes(boxes_normalized)
    _require(isinstance(images, (list, tuple)) and len(images) == 1, "Exactly one overview image is required")
    processor = backend.processor
    _require(getattr(processor, "expected", None) is not None,
             "A frozen expected CPU processor capture is required before the forward")
    torch = backend.torch
    model = backend.model
    _require(not model.training, "The frozen feature model must be in evaluation mode")
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=images, return_tensors="pt")
    _require(getattr(processor, "last", None) == processor.expected,
             "Actual CPU processor capture does not match the frozen preflight")
    required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
    _require(set(inputs) in (required, required | {"token_type_ids"}), "Unexpected processor tensor keys")
    _require(all(t.device.type == "cpu" for t in inputs.values()), "Processor validation must precede CUDA transfer")
    ids, attention = inputs["input_ids"], inputs["attention_mask"]
    _require(ids.ndim == 2 and ids.shape[0] == 1 and attention.shape == ids.shape,
             "Exactly one token sequence with an aligned attention mask is required")
    _require(ids.dtype == torch.int64 and attention.dtype in (torch.int64, torch.int32, torch.bool),
             "Input IDs and attention mask must retain integer dtypes")
    id_list = ids[0].tolist()
    mask_list = attention[0].tolist()
    _require(all(x in (0, 1) for x in mask_list), "Attention mask must be binary")
    attended = [i for i, value in enumerate(mask_list) if value == 1]
    _require(bool(attended) and attended == list(range(attended[0], attended[-1] + 1)),
             "The attended token span must be nonempty and contiguous")
    query_index = attended[-1]
    grids = inputs["image_grid_thw"].tolist()
    _require(len(grids) == 1, "Exactly one image grid is required")
    mh, mw = _grid_shape(grids[0])
    positions = [i for i, token in enumerate(id_list) if token == IMAGE_TOKEN_ID]
    _require(len(positions) == mh * mw and positions == list(range(positions[0], positions[-1] + 1)),
             "Image placeholders must be a single contiguous complete merged grid")
    _require(all(mask_list[i] == 1 for i in positions), "Image placeholders cannot occupy padding")
    _require(positions[0] > 0 and positions[-1] + 1 < query_index
             and id_list[positions[0] - 1] == VISION_START_TOKEN_ID
             and id_list[positions[-1] + 1] == VISION_END_TOKEN_ID,
             "Image placeholders must be framed and precede the final prompt token")
    _require(sum(x == VISION_START_TOKEN_ID for x in id_list) == 1
             and sum(x == VISION_END_TOKEN_ID for x in id_list) == 1, "Unexpected additional vision framing")
    # All metadata validation and the wrapper's tensor/hash comparison are CPU-only.
    inputs.pop("token_type_ids", None)
    device_inputs = {key: tensor.to(model.device) for key, tensor in inputs.items()}
    decoder = model.model
    # Full uncached forward already recomputes RoPE; clear its mutable memo too.
    _require(hasattr(decoder, "rope_deltas"), "Expected Qwen3-VL multimodal decoder API")
    decoder.rope_deltas = None
    try:
        with torch.inference_mode():
            output = decoder(**device_inputs, use_cache=False, past_key_values=None,
                             return_dict=True, output_hidden_states=False, output_attentions=False)
            hidden = output.last_hidden_state
            _require(hidden.ndim == 3 and tuple(hidden.shape) == (1, len(id_list), HIDDEN_SIZE),
                     "Unexpected final Qwen hidden-state shape")
            _require(hidden.dtype in (torch.float32, torch.bfloat16, torch.float16),
                     "Expected floating-point final hidden states")
            image_states = hidden[0, positions, :].detach().to(device="cpu", dtype=torch.float32).numpy().copy()
            query = hidden[0, query_index, :].detach().to(device="cpu", dtype=torch.float32).numpy().copy()
    finally:
        decoder.rope_deltas = None
    _require(np.isfinite(query).all(), "Final prompt-query vector must be finite")
    regions, counts = pool_image_tokens(image_states, grids[0], boxes_normalized)
    result = {
        "region_vectors": regions,
        "query_vector": np.ascontiguousarray(query, dtype=np.float32),
        "input_tokens": len(id_list),
        "visual_tokens": len(positions),
        "image_grid_thw": grids,
        "image_region_token_counts": counts,
        "query_token_index": query_index,
        "query_token_semantics": QUERY_TOKEN_SEMANTICS,
        "image_token_positions": positions,
        "raw_feature_sha256": raw_feature_digest(regions, query),
    }
    return result
