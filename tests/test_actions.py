from PIL import Image
import pytest

from lookagain.actions import build_request, grade, pixel_box, resize_for_tokens

CONFIG = dict(base_visual_tokens=64, highres_visual_tokens=256, crop_visual_tokens=64)


def test_crops_preserve_overview_and_need_no_labels():
    image = Image.new("RGB", (300, 200))
    messages, images, geometry = build_request(image, "Which color?", "crop_br", CONFIG, "ANSWER: red")
    assert len(images) == 2
    assert geometry["crop_box_pixels"] == [120, 80, 300, 200]
    assert messages[2]["content"] == "ANSWER: red"
    assert "target_answer" not in str(messages)


def test_think_does_not_add_image_and_independent_highres():
    image = Image.new("RGB", (300, 200))
    messages, images, _ = build_request(image, "Which color?", "think", CONFIG, "ANSWER: red")
    assert len(images) == 1 and len(messages) == 4
    messages, images, _ = build_request(image, "Which color?", "highres", CONFIG)
    assert len(messages) == 2 and len(images) == 1


@pytest.mark.parametrize("size", [(1, 10000), (10000, 1), (300, 200), (200, 300)])
def test_visual_budget(size):
    resized = resize_for_tokens(Image.new("RGB", size), 64)
    assert resized.width % 32 == resized.height % 32 == 0
    assert resized.width * resized.height // 1024 <= 64


def test_grader_rejects_unstructured_or_semantically_similar_answer():
    assert grade("ANSWER: RED.", "red")["correct"]
    assert not grade("It might be red", "red")["correct"]
    assert grade("red", "red")["correct"]
    assert not grade("ANSWER: <short answer>", "red")["parse_valid"]
    assert not grade("ANSWER: dark red", "red")["correct"]
    assert grade("It seems blue.\nANSWER: red", "red")["correct"]


def test_followup_requires_observed_answer():
    with pytest.raises(ValueError):
        build_request(Image.new("RGB", (64, 64)), "Q", "recheck", CONFIG)


@pytest.mark.parametrize("text", ["ANSWER: red\nActually no", "ANSWER: blue\nANSWER: red", "ANSWER:\nred", "ANSWER:   "])
def test_answer_marker_must_be_unique_and_final(text):
    assert not grade(text, "red")["parse_valid"]
