"""Reject altered resume contents using synthetic requests, without model inference."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
import history_context as runner
import history_context_core as core


@pytest.fixture(scope="module")
def saved_case(tmp_path_factory):
    directory = tmp_path_factory.mktemp("history_resume_synthetic")
    source = Image.new("RGB", (1280, 1792), "white")
    draw = ImageDraw.Draw(source)
    for x in range(19, source.width, 37):
        draw.line((x, 11, x, source.height - 21), fill=(x % 251, 29, 117), width=1)
    image_path = directory / "synthetic.png"
    source.save(image_path)
    row = {
        "example_id": "synthetic-question", "image_id": "synthetic-page",
        "source_id": "synthetic-report", "question": "Which year is shown?",
        "answer": "2019", "image_path": image_path.name,
        "image_sha256": runner.sha256_file(image_path),
    }
    manifest = directory / "manifest.jsonl"
    manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
    config = {"base_visual_tokens": 1024, "crop_visual_tokens": 1024,
              "highres_visual_tokens": 4096, "answer_max_tokens": 64}
    records = {}
    direct_response = "ANSWER: 2019"
    for action in core.ACTIONS:
        _, images, geometry = core.build_request(source, row["question"], action, config, direct_response)
        response = "ANSWER: 2020" if "degraded" in action else direct_response
        scores = core.score_response(response, row["answer"])
        scores.update(runner.score_official(scores["predicted_answer"] or "", row["answer"]))
        grids = [[1, image.height // 16, image.width // 16] for image in images]
        visual_tokens = sum(t * h * w // 4 for t, h, w in grids)
        # These are declared synthetic measurements, not tokenization/model results.
        record = {
            **{key: row[key] for key in ("example_id", "image_id", "source_id", "question")},
            "target_answer": row["answer"], "source_image_sha256": row["image_sha256"],
            "action": action, "status": "ok", "response": response,
            "observed_direct_response_sha256": hashlib.sha256(direct_response.encode()).hexdigest(),
            "raw_continuation": response[len("ANSWER:"):], "answer_prefix_prefilled": True,
            "generated_tokens": 3, "generation_truncated": False,
            "image_grid_thw": grids, "visual_tokens": visual_tokens,
            "input_tokens": visual_tokens + 80, "mean_token_logprob": -0.5,
            "elapsed_s": 1.25, "peak_memory_gib": 0.5, "peak_reserved_gib": 0.75,
            **geometry, **scores,
        }
        records[row["example_id"], action] = record
    return row, manifest, config, records


def selected_records(saved_case, *actions):
    row, _, _, records = saved_case
    return {key: deepcopy(records[key]) for key in (
        (row["example_id"], action) for action in ("direct", *actions)
    )}


def validate(saved_case, records):
    row, manifest, config, _ = saved_case
    runner.validate_saved_records(records, [row], manifest, config)


def test_accepts_complete_and_incomplete_valid_saved_cohort(saved_case):
    validate(saved_case, saved_case[3])
    validate(saved_case, selected_records(saved_case))
    validate(saved_case, selected_records(saved_case, "fresh_native_tl", "placeholder_degraded_br"))
    validate(saved_case, {})


@pytest.mark.parametrize("field,replacement", [
    ("question", "Which month is shown?"),
    ("target_answer", "2020"),
    ("image_id", "another-page"),
    ("source_id", "another-report"),
    ("source_image_sha256", "f" * 64),
    ("condition", "degraded"),
    ("region", "br"),
    ("history_mode", "placeholder"),
    ("previous_answer_in_prompt", False),
    ("inserted_history_text", "ANSWER: 2020"),
    ("initial_answer_sha256", "f" * 64),
    ("observed_direct_response_sha256", "f" * 64),
    ("messages_sha256", "f" * 64),
    ("image_rgb_sha256", ["f" * 64, "f" * 64]),
    ("correct", False),
    ("conservative_text_em", 0.0),
    ("official_em", 0.0),
    ("official_f1", 0.0),
    ("anls", 0.0),
    ("parse_valid", False),
    ("normalized_prediction", "2020"),
])
def test_rejects_changed_content_and_recomputed_fields(saved_case, field, replacement):
    records = selected_records(saved_case, "actual_native_tl")
    records[saved_case[0]["example_id"], "actual_native_tl"][field] = replacement
    with pytest.raises(ValueError, match="Saved record content mismatch"):
        validate(saved_case, records)


@pytest.mark.parametrize("action", ["actual_native_tl", "fresh_native_tl", "placeholder_native_tl", "repeat"])
def test_rejects_missing_fresh_direct_even_for_answer_free_branch(saved_case, action):
    records = selected_records(saved_case, action)
    del records[saved_case[0]["example_id"], "direct"]
    with pytest.raises(ValueError, match="no fresh direct"):
        validate(saved_case, records)


def test_rejects_coherently_rehashed_placeholder_answer_injection(saved_case):
    records = selected_records(saved_case, "placeholder_native_tl")
    record = records[saved_case[0]["example_id"], "placeholder_native_tl"]
    record["inserted_history_text"] = "ANSWER: 2019"
    record["initial_answer_sha256"] = hashlib.sha256(record["inserted_history_text"].encode()).hexdigest()
    with pytest.raises(ValueError, match="Saved record content mismatch"):
        validate(saved_case, records)


def test_rejects_direct_response_change_that_leaves_stale_actual_history(saved_case):
    records = selected_records(saved_case, "actual_native_tl")
    direct = records[saved_case[0]["example_id"], "direct"]
    direct["response"] = "ANSWER: 2020"
    direct["raw_continuation"] = " 2020"
    for record in records.values():
        record["observed_direct_response_sha256"] = hashlib.sha256(direct["response"].encode()).hexdigest()
    scores = core.score_response(direct["response"], saved_case[0]["answer"])
    scores.update(runner.score_official(scores["predicted_answer"], saved_case[0]["answer"]))
    direct.update(scores)
    with pytest.raises(ValueError, match="Saved record content mismatch"):
        validate(saved_case, records)


@pytest.mark.parametrize("field,value,message", [
    ("raw_continuation", " 2020", "continuation/prefix"),
    ("answer_prefix_prefilled", False, "continuation/prefix"),
    ("generated_tokens", 0, "generation length"),
    ("generated_tokens", 65, "generation length"),
    ("elapsed_s", 0, "measurement"),
    ("input_tokens", 0, "measurement"),
])
def test_rejects_invalid_saved_generation_measurements(saved_case, field, value, message):
    records = selected_records(saved_case, "actual_native_tl")
    records[saved_case[0]["example_id"], "actual_native_tl"][field] = value
    with pytest.raises(ValueError, match=message):
        validate(saved_case, records)


def test_rejects_native_degraded_token_count_mismatch_on_resume(saved_case):
    records = selected_records(saved_case, "actual_native_tl", "actual_degraded_tl")
    records[saved_case[0]["example_id"], "actual_degraded_tl"]["input_tokens"] += 1
    with pytest.raises(ValueError, match="Mismatched native/degraded input_tokens"):
        validate(saved_case, records)
