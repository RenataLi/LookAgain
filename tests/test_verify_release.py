"""Isolated public inventory checks; no model or source-document dependency."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "verify_public_release", Path(__file__).resolve().parents[1] / "scripts/verify_release.py")
inventory = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inventory)


def save(root, entries):
    (root / "release_manifest.json").write_text(json.dumps({"files": entries}), encoding="utf-8")


@pytest.fixture
def release(tmp_path):
    entries = []
    for name, content in (("payload.txt", b"original bytes\n"), ("nested/values.json", b'{"n": 3}\n')):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append({"path": name, "sha256": hashlib.sha256(content).hexdigest()})
    save(tmp_path, entries)
    return tmp_path, entries


def test_valid_inventory_verifies_every_listed_file(release):
    root, entries = release
    assert inventory.verify(root) == len(entries) == 2


def test_modified_file_rejected(release):
    root, _ = release
    (root / "payload.txt").write_bytes(b"modified bytes\n")
    with pytest.raises(ValueError, match="digest differs"):
        inventory.verify(root)


def test_missing_file_rejected(release):
    root, entries = release
    entries[0]["path"] = "absent.txt"
    save(root, entries)
    with pytest.raises(ValueError, match="Missing"):
        inventory.verify(root)


@pytest.mark.parametrize("name", ["../outside.txt", "nested/../../outside.txt", "/outside.txt", "C:/outside.txt", "nested\\values.json"])
def test_traversal_and_absolute_paths_rejected(release, name):
    root, entries = release
    entries[0]["path"] = name
    save(root, entries)
    with pytest.raises(ValueError, match="Invalid manifest path"):
        inventory.verify(root)


def test_duplicate_inventory_path_rejected(release):
    root, entries = release
    save(root, entries + [dict(entries[0])])
    with pytest.raises(ValueError, match="Duplicate"):
        inventory.verify(root)


@pytest.mark.parametrize("name,index", [("./payload.txt", 0), ("nested//values.json", 1), ("nested/./values.json", 1)])
def test_path_alias_cannot_double_count_same_file(release, name, index):
    root, entries = release
    alias = {**entries[index], "path": name}
    save(root, entries + [alias])
    with pytest.raises(ValueError):
        inventory.verify(root)


def test_empty_inventory_rejected(tmp_path):
    save(tmp_path, [])
    with pytest.raises(ValueError, match="Empty"):
        inventory.verify(tmp_path)


def test_wrong_digest_rejected_even_when_file_exists(release):
    root, entries = release
    entries[0]["sha256"] = "0" * 64
    save(root, entries)
    with pytest.raises(ValueError, match="digest differs"):
        inventory.verify(root)
