"""CPU tests of geometry, pre-action information boundaries and text metrics."""
import importlib.util
from pathlib import Path

from PIL import Image
import pytest

PROJECT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("native_detail_core_tests", PROJECT / "experiments" / "native_detail_core.py")
core = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(core)


@pytest.fixture
def config():
    return {"base_visual_tokens": 12, "crop_visual_tokens": 20, "highres_visual_tokens": 64}


@pytest.fixture
def source():
    image = Image.new("RGB", (251, 193))
    image.putdata([(x, y, (x * 71 + y * 37) % 256) for y in range(193) for x in range(251)])
    return image


def pixels(image):
    return image.mode, image.size, image.tobytes()


def test_bounded_size_obeys_source_and_grid_without_upsampling():
    assert core.bounded_size((63, 95), 4096) == (32, 64)
    for size in [(251, 193), (2401, 3507), (32, 5000), (5000, 32)]:
        for budget in [1, 12, 1024, 4096]:
            width, height = core.bounded_size(size, budget)
            assert 32 <= width <= size[0] and 32 <= height <= size[1]
            assert width % 32 == height % 32 == 0
            assert width * height // 1024 <= budget
    for size, budget in [((31, 80), 5), ((100, 100), 0), ((100, 100), True)]:
        with pytest.raises(ValueError):
            core.bounded_size(size, budget)


def test_odd_native_boxes_project_without_independent_rounding(config, source):
    expected = {"tl": [0, 0, 151, 116], "tr": [100, 0, 251, 116],
                "bl": [0, 77, 151, 193], "br": [100, 77, 251, 193]}
    centers = {}
    for region, box in expected.items():
        _, images, geometry = core.build_request(source, "Read?", "native_" + region, config, "ANSWER: first")
        assert geometry["source_roi_pixels"] == box
        assert geometry["projected_overview_roi_pixels"] == [
            box[0] * images[0].width / 251, box[1] * images[0].height / 193,
            box[2] * images[0].width / 251, box[3] * images[0].height / 193]
        assert pixels(images[1]) == pixels(source.crop(box).resize(images[1].size, Image.Resampling.BICUBIC))
        centers[region] = images[1].getpixel((images[1].width // 2, images[1].height // 2))
    assert centers["tr"][0] > centers["tl"][0]
    assert centers["br"][0] > centers["bl"][0]
    assert centers["bl"][1] > centers["tl"][1]
    assert centers["br"][1] > centers["tr"][1]


def test_pairs_have_identical_messages_first_pixels_and_final_sizes(config, source):
    direct_messages, direct_images, _ = core.build_request(source, "Which total?", "direct", config)
    for region in core.REGIONS:
        am, ai, ag = core.build_request(source, "Which total?", "native_" + region, config, "ANSWER: fresh")
        bm, bi, bg = core.build_request(source, "Which total?", "degraded_" + region, config, "ANSWER: fresh")
        assert am == bm and am[:2] == direct_messages
        assert pixels(ai[0]) == pixels(bi[0]) == pixels(direct_images[0])
        assert ai[1].size == bi[1].size
        assert ag["source_roi_pixels"] == bg["source_roi_pixels"]
        assert ag["projected_overview_roi_pixels"] == bg["projected_overview_roi_pixels"]
        assert ag["followup_prompt"] == core.CROP_PROMPT
        assert not any(word in core.CROP_PROMPT for word in ("upper-left", "upper-right", "lower-left", "lower-right"))
        assert ag["image_rgb_sha256"][0] == bg["image_rgb_sha256"][0]


def test_degraded_reads_only_overview_pixels(config, source, monkeypatch):
    """Distinct original pixels with identical overview must yield identical degraded views."""
    original_resize = Image.Image.resize
    original_crop = Image.Image.crop
    other = Image.new("RGB", source.size, (240, 10, 25))
    overview = original_resize(source, core.bounded_size(source.size, config["base_visual_tokens"]), Image.Resampling.BICUBIC)
    native_ids = {id(source), id(other)}

    def controlled_resize(image, size, *args, **kwargs):
        if id(image) in native_ids:
            assert size == overview.size
            return overview.copy()
        return original_resize(image, size, *args, **kwargs)

    def forbid_native_crop(image, *args, **kwargs):
        if id(image) in native_ids:
            raise AssertionError("degraded action read native crop pixels")
        return original_crop(image, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "resize", controlled_resize)
    monkeypatch.setattr(Image.Image, "crop", forbid_native_crop)
    for region in core.REGIONS:
        _, first, geometry = core.build_request(source, "Read?", "degraded_" + region, config, "ANSWER: first")
        _, second, _ = core.build_request(other, "Read?", "degraded_" + region, config, "ANSWER: first")
        assert pixels(first[1]) == pixels(second[1])
        expected = original_resize(overview, first[1].size, Image.Resampling.BICUBIC,
                                   box=tuple(geometry["projected_overview_roi_pixels"]))
        assert pixels(first[1]) == pixels(expected)


def test_truthful_repeat_is_exact_separate_overview_copy(config, source):
    _, images, geometry = core.build_request(source, "Read?", "repeat", config, "ANSWER: first")
    assert pixels(images[0]) == pixels(images[1]) and images[0] is not images[1]
    assert geometry["image_rgb_sha256"][0] == geometry["image_rgb_sha256"][1]
    assert geometry["followup_prompt"] == core.REPEAT_PROMPT
    assert "repeats the original image" in geometry["followup_prompt"]
    assert geometry["source_roi_pixels"] is None
    assert geometry["actual_second_view_box"] == [0.0, 0.0, 1.0, 1.0]
    first_pixel = images[0].getpixel((0, 0))
    images[1].putpixel((0, 0), (255, 0, 0))
    assert images[0].getpixel((0, 0)) == first_pixel


def test_standalone_and_independent_followup_history(config, source):
    original = pixels(source)
    for action in core.ACTIONS:
        messages, images, geometry = core.build_request(source, "Read?", action, config, "ANSWER: only direct")
        assistants = [message["content"] for message in messages if message["role"] == "assistant"]
        if action in ("direct", "highres"):
            assert assistants == [] and len(images) == 1
            assert geometry["additional_size"] is None
        else:
            assert assistants == ["ANSWER: only direct"] and len(images) == 2
            messages[2]["content"] = "must not reach next branch"
        assert pixels(source) == original
    _, highres, _ = core.build_request(source, "Read?", "highres", config)
    assert highres[0].size == (224, 192)
    with pytest.raises(ValueError, match="fresh direct"):
        core.build_request(source, "Read?", "native_tl", config)
    with pytest.raises(ValueError, match="unknown"):
        core.build_request(source, "Read?", "oracle", config)


def test_custom_em_preserves_units_punctuation_articles_and_number_forms():
    assert core.score_response("ANSWER:  USD   1,000 million ", "usd 1,000 million")["correct"]
    for predicted, target in [("USD 1,000", "USD 1000"), ("5 million", "5"),
                              ("the bank", "bank"), ("12.", "12"), ("1.0", "1")]:
        result = core.score_response("ANSWER: " + predicted, target)
        assert not result["correct"] and result["conservative_text_em"] == 0.0
    assert core.score_response("ANSWER: coca  cola", "coca cola")["correct"]
    assert core.score_response("ANSWER: coca  cola", "coca cola")["anls"] < 1


def test_invalid_answers_and_target_type_are_not_silently_coerced():
    for response in ["", "ANSWER:", "ANSWER: one\nANSWER: two", "reasoning\nno final marker"]:
        result = core.score_response(response, "one")
        assert result["parse_valid"] is False
        assert result["correct"] is False and result["anls"] == 0
    for target in [[], ["one"], "   ", 12]:
        with pytest.raises(ValueError):
            core.score_response("one", target)


def test_diagnostic_anls_strict_boundary_and_full_levenshtein():
    assert core.score_response("ANSWER: ac", "ab")["anls"] == 0
    assert core.score_response("ANSWER: abce", "abcd")["anls"] == .75
    assert core.score_response("ANSWER: kitten", "sitting")["anls"] == pytest.approx(4 / 7)
    assert core.score_response("ANSWER: USD 5", "usd 5")["anls"] == 1
    assert core.score_response("ANSWER: a", "xyz")["anls"] == 0

