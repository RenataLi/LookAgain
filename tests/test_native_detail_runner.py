"""Critical v3 provenance and paired-input failure modes, CPU only."""
import importlib.util
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

EXPERIMENTS = Path(__file__).resolve().parents[1] / "experiments"
sys.path.insert(0, str(EXPERIMENTS))


def runner():
    # Imports after the independent scorer implementation is available.
    import native_detail
    return native_detail


def manifest(tmp_path):
    module = runner()
    image = tmp_path / "page.png"
    Image.new("RGB", (1500, 2000), "white").save(image)
    row = dict(example_id="q1", image_id="page1", source_id="report1", question="Which year?",
               answer="2019", image_path="page.png", image_sha256=module.sha256_file(image))
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path, row


def test_manifest_detects_tampering_and_repeated_source(tmp_path):
    module = runner()
    path, row = manifest(tmp_path)
    assert len(module.validate_manifest(path)) == 1
    repeated = {**row, "example_id": "q2", "image_id": "page2"}
    path.write_text(json.dumps(row) + "\n" + json.dumps(repeated), encoding="utf-8")
    with pytest.raises(ValueError, match="source_id"):
        module.validate_manifest(path)
    path.write_text(json.dumps({**row, "image_sha256": "0" * 64}), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        module.validate_manifest(path)


def test_manifest_rejects_escape_and_answer_arrays(tmp_path):
    module = runner()
    path, row = manifest(tmp_path)
    for changes, reason in (({"image_path": "../outside.png"}, "escapes"), ({"answer": ["2019"]}, "answer")):
        path.write_text(json.dumps({**row, **changes}), encoding="utf-8")
        with pytest.raises(ValueError, match=reason):
            module.validate_manifest(path)


def test_pair_rejects_prompt_and_actual_grid_mismatch():
    module = runner()
    base = dict(followup_prompt="same", overview_size=[864, 1184], additional_size=[864, 1184],
                image_grid_thw=[[1, 74, 54], [1, 74, 54]], input_tokens=4100, visual_tokens=3996,
                image_rgb_sha256=["overview", "addition"])
    records = {("q", "native_tl"): base, ("q", "degraded_tl"): base.copy()}
    module.check_pair(records, "q", "tl")
    for key, value in (("followup_prompt", "different"), ("image_grid_thw", [[1, 2, 2]]), ("input_tokens", 4099)):
        records[("q", "degraded_tl")] = {**base, key: value}
        with pytest.raises(ValueError, match=key):
            module.check_pair(records, "q", "tl")


def test_records_fail_on_duplicates_or_unsuccessful_rows(tmp_path):
    module = runner()
    path = tmp_path / "records.jsonl"
    row = dict(example_id="q", action="direct", status="ok")
    path.write_text((json.dumps(row) + "\n") * 2, encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        module.read_records(path)
    path.write_text(json.dumps({**row, "status": "error"}), encoding="utf-8")
    with pytest.raises(ValueError, match="failed"):
        module.read_records(path)
