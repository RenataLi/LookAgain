import hashlib
import io
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from lookagain.data import image_split, prepare_gqa, validate_manifest


def _encoded_image(color="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (12, 8), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _manifest(tmp_path):
    encoded = _encoded_image()
    (tmp_path / "image.png").write_bytes(encoded)
    row = {
        "example_id": "q1", "image_id": "i1", "image_path": "image.png",
        "question": "What color?", "answer": "red", "dataset": "fixture",
        "dataset_revision": "a" * 40, "source_split": "train",
        "image_sha256": hashlib.sha256(encoded).hexdigest(),
    }
    path = tmp_path / "manifest.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path, row


def test_manifest_rejects_modified_image_bytes(tmp_path):
    path, row = _manifest(tmp_path)
    assert validate_manifest(path) == [row]
    (tmp_path / "image.png").write_bytes(_encoded_image("blue"))
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_manifest(path)


def test_manifest_rejects_duplicate_source_image(tmp_path):
    path, row = _manifest(tmp_path)
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "example_id": "q2"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate image_id"):
        validate_manifest(path)


@pytest.mark.parametrize("unsafe_path", ["../outside.png", "C:\\outside.png"])
def test_manifest_rejects_paths_outside_sample(tmp_path, unsafe_path):
    path, row = _manifest(tmp_path)
    row["image_path"] = unsafe_path
    path.write_text(json.dumps(row), encoding="utf-8")
    with pytest.raises(ValueError, match="(escapes|relative)"):
        validate_manifest(path)


def test_image_partitions_are_order_and_question_independent():
    ids = [f"original-{i}" for i in range(1000)]
    before = {key: image_split(key) for key in ids}
    after = {key: image_split(key) for key in reversed(ids)}
    assert before == after
    assert set(before.values()) == {"train", "dev", "test"}
    assert image_split("same-source", seed=7) == image_split("same-source", seed=7)
    assert image_split("one", train_fraction=1, dev_fraction=0) == "train"
    with pytest.raises(ValueError):
        image_split("one", train_fraction=0.9, dev_fraction=0.2)


def test_preparation_pins_streams_and_joins_by_image_id(tmp_path, monkeypatch):
    calls = []
    revision = "b" * 40
    short_cache = str(tmp_path / "short-cache")
    monkeypatch.setenv("HF_DATASETS_CACHE", short_cache)

    class Stream(list):
        def cast_column(self, *_args, **_kwargs):
            return self

        def select_columns(self, _columns):
            return self

    image_rows = Stream([
        {"id": "image-b", "image": {"bytes": _encoded_image("blue")}},
        {"id": "image-b", "image": {"bytes": _encoded_image("blue")}},
        {"id": "image-a", "image": {"bytes": _encoded_image("red")}},
    ])
    questions = Stream([
        {"id": "other", "imageId": "not-selected", "question": "Unused?", "answer": "yes"},
        {"id": "qa", "imageId": "image-a", "question": "Color?", "answer": "red", "types": {"detailed": "queryAttr"}},
        {"id": "qb", "imageId": "image-b", "question": "Color?", "answer": "blue", "types": {}},
        {"id": "later", "imageId": "image-a", "question": "Later?", "answer": "yes"},
    ])

    def fake_load_dataset(**kwargs):
        calls.append(kwargs)
        return image_rows if kwargs["name"].endswith("_images") else questions

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(load_dataset=fake_load_dataset, Image=lambda **_: None))
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=lambda: SimpleNamespace(dataset_info=lambda *a, **kw: SimpleNamespace(sha=revision))))
    path = prepare_gqa(tmp_path, limit=2)
    rows = validate_manifest(path)
    assert [row["image_id"] for row in rows] == ["image-b", "image-a"]
    assert [row["answer"] for row in rows] == ["blue", "red"]
    assert all(call["revision"] == revision and call["streaming"] for call in calls)
    assert all(call["cache_dir"] == short_cache for call in calls)
    assert all(row["source_split"] == "train" for row in rows)
    metadata = json.loads((tmp_path / "sample_metadata.json").read_text())
    assert metadata["sample_role"] == "development_only"
    assert metadata["representative_random_sample"] is False
    assert metadata["questions_scanned"] == 3
    # Reuse validates local bytes and does not read remote rows a second time.
    assert prepare_gqa(tmp_path, limit=2) == path
    assert len(calls) == 2
    with pytest.raises(FileExistsError, match="settings differ"):
        prepare_gqa(tmp_path, limit=1)
