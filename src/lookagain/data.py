"""Small, pinned GQA development samples; never an OOD benchmark downloader.

The sample is the first N distinct images in the pinned training-image stream,
with the first matching question in the pinned training-question stream. It is
a reproducible convenience sample for debugging, NOT a representative estimate
of GQA performance. ``seed`` records the seed for future image-level partitions;
it does not turn this prefix into a random sample.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
from pathlib import Path, PureWindowsPath
import re
from typing import Any

DATASET_ID = "lmms-lab-encoder/GQA"
IMAGE_CONFIG = "train_balanced_images"
QUESTION_CONFIG = "train_balanced_instructions"
SOURCE_SPLIT = "train"
DEFAULT_SEED = 20260925
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"[0-9a-f]{40,64}\Z")
_REQUIRED = (
    "example_id", "image_id", "image_path", "question", "answer", "dataset",
    "dataset_revision", "source_split", "image_sha256",
)


def file_sha256(path: Path) -> str:
    """Hash a file without loading its entire contents into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def image_split(
    image_key: str,
    seed: int = DEFAULT_SEED,
    train_fraction: float = 0.8,
    dev_fraction: float = 0.1,
) -> str:
    """Stable image-level partition, independent of order or question count.

    Use the same canonical original-image key for every question/crop. A source
    image content hash is preferable if duplicate images have different IDs.
    This utility creates project partitions, not official benchmark splits.
    Tiny development samples need not populate all three partitions.
    """
    if not isinstance(image_key, str) or not image_key:
        raise ValueError("image_key must be a nonempty string")
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (train_fraction, dev_fraction)):
        raise ValueError("split fractions must be finite and between zero and one")
    if train_fraction + dev_fraction > 1:
        raise ValueError("train_fraction + dev_fraction must not exceed one")
    value = int.from_bytes(
        hashlib.sha256(f"{seed}\0{image_key}".encode("utf-8")).digest()[:8], "big"
    )
    scale = 1 << 64
    if value < int(train_fraction * scale):
        return "train"
    if value < int((train_fraction + dev_fraction) * scale):
        return "dev"
    return "test"


def validate_manifest(path: Path) -> list[dict[str, Any]]:
    """Read JSONL and verify provenance, one-question/image, bytes, and decoding.

    Image paths are relative to the manifest and must remain inside its folder.
    Returned rows retain relative paths; callers resolve them against path.parent.
    """
    from PIL import Image

    path = Path(path).resolve()
    rows: list[dict[str, Any]] = []
    image_ids: set[str] = set()
    example_ids: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at manifest line {number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Manifest line {number} must be an object")
            for key in _REQUIRED:
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"Manifest line {number}: {key} must be a nonempty string")
            if not _REVISION.fullmatch(row["dataset_revision"]):
                raise ValueError(f"Manifest line {number}: dataset_revision must be a commit hash")
            if not _SHA256.fullmatch(row["image_sha256"]):
                raise ValueError(f"Manifest line {number}: invalid image_sha256")
            if row["image_id"] in image_ids:
                raise ValueError(f"Duplicate image_id: {row['image_id']}")
            if row["example_id"] in example_ids:
                raise ValueError(f"Duplicate example_id: {row['example_id']}")
            relative = Path(row["image_path"])
            if relative.is_absolute() or PureWindowsPath(row["image_path"]).drive:
                raise ValueError(f"Image path must be relative: {row['image_path']}")
            image_path = (path.parent / relative).resolve()
            if not image_path.is_relative_to(path.parent):
                raise ValueError(f"Image path escapes manifest directory: {row['image_path']}")
            if not image_path.is_file():
                raise ValueError(f"Missing image: {image_path}")
            if file_sha256(image_path) != row["image_sha256"]:
                raise ValueError(f"Image checksum mismatch: {row['image_id']}")
            try:
                with Image.open(image_path) as img:
                    img.verify()
                with Image.open(image_path) as img:
                    img.load()
                    if min(img.size) <= 0:
                        raise ValueError("Image has empty dimensions")
            except Exception as exc:
                raise ValueError(f"Invalid image {row['image_id']}: {exc}") from exc
            image_ids.add(row["image_id"])
            example_ids.add(row["example_id"])
            rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    return rows


def _write_image(value: Any, stem: Path) -> Path:
    """Preserve original JPEG/PNG bytes where the source supplies them."""
    from PIL import Image

    original: bytes | None = None
    if isinstance(value, dict):
        original = value.get("bytes")
        if original is None and value.get("path"):
            original = Path(value["path"]).read_bytes()
    if original is not None:
        with Image.open(io.BytesIO(original)) as img:
            img.load()
            suffix = {"JPEG": ".jpg", "PNG": ".png"}.get(img.format)
            if suffix:
                destination = stem.with_suffix(suffix)
                destination.write_bytes(original)
                return destination
            image = img.convert("RGB")
    elif isinstance(value, Image.Image):
        image = value.convert("RGB")
    else:
        raise ValueError("GQA image is neither encoded bytes nor a PIL image")
    destination = stem.with_suffix(".png")
    image.save(destination, format="PNG")
    return destination


def prepare_gqa(
    output_dir: Path,
    limit: int = 32,
    seed: int = DEFAULT_SEED,
    revision: str | None = None,
) -> Path:
    """Stream a small GQA train prefix and write manifest.jsonl + provenance.

    Only the image-stream prefix is decoded. The text-only question stream is
    scanned until one question has been found for every selected image. Remote
    Parquet row groups can still transfer more bytes than the final sample.
    Existing samples are validated/reused only when all requested settings match.
    """
    from datasets import Image as DatasetImage, load_dataset
    from huggingface_hub import HfApi

    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    metadata_path = output_dir / "sample_metadata.json"
    # Resolve mutable names once; both streams use the same immutable commit.
    resolved_revision = HfApi().dataset_info(DATASET_ID, revision=revision).sha
    if not resolved_revision or not _REVISION.fullmatch(resolved_revision):
        raise RuntimeError("Hugging Face did not return an immutable dataset revision")
    settings = {
        "dataset": DATASET_ID,
        "dataset_revision": resolved_revision,
        "source_split": SOURCE_SPLIT,
        "image_config": IMAGE_CONFIG,
        "question_config": QUESTION_CONFIG,
        "limit": limit,
        "seed": seed,
        "sample_role": "development_only",
        "sampling": "first_unique_images_then_first_matching_question_in_pinned_stream_order",
        "representative_random_sample": False,
        "seed_usage": "reserved_for_image_level_project_partitions; prefix_selection_is_seed_independent",
    }
    if manifest_path.exists():
        if not metadata_path.exists():
            raise FileExistsError("Existing manifest has no sample_metadata.json; use a new output directory")
        stored = json.loads(metadata_path.read_text(encoding="utf-8"))
        if any(stored.get(key) != value for key, value in settings.items()):
            raise FileExistsError("Existing sample settings differ; use a new output directory")
        if stored.get("manifest_sha256") != file_sha256(manifest_path):
            raise ValueError("Existing manifest checksum differs from sample metadata")
        existing_rows = validate_manifest(manifest_path)
        if len(existing_rows) != limit:
            raise ValueError("Existing manifest row count does not match requested limit")
        return manifest_path

    # A short cache root avoids Windows MAX_PATH failures in HF lock filenames.
    cache_dir = os.environ.get("HF_DATASETS_CACHE") or str(output_dir / ".cache")
    common = {"path": DATASET_ID, "split": SOURCE_SPLIT, "revision": resolved_revision,
              "streaming": True, "cache_dir": cache_dir}
    print(f"Streaming up to {limit} GQA training images at {resolved_revision}...", flush=True)
    image_stream = load_dataset(name=IMAGE_CONFIG, **common).cast_column("image", DatasetImage(decode=False))
    images: dict[str, Path] = {}
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    for item in image_stream:
        image_id = str(item["id"])
        if image_id in images:
            continue
        # No upstream IDs enter a path; numbering keeps filenames readable.
        stem = images_dir / f"{len(images):05d}"
        images[image_id] = _write_image(item["image"], stem)
        if len(images) == limit:
            break
    if len(images) != limit:
        raise ValueError(f"Requested {limit} images but training stream supplied {len(images)}")
    print(f"Saved {len(images)} unique images; scanning training questions for matches...", flush=True)

    instructions = load_dataset(name=QUESTION_CONFIG, **common)
    # Avoid decoding unused nested annotation/semantic columns when supported.
    instructions = instructions.select_columns(["id", "imageId", "question", "answer", "types"])
    selected: dict[str, dict[str, Any]] = {}
    questions_scanned = 0
    for question in instructions:
        questions_scanned += 1
        if questions_scanned % 100_000 == 0:
            print(f"Scanned {questions_scanned:,} questions; matched {len(selected)}/{limit} images.", flush=True)
        image_id = str(question["imageId"])
        if image_id not in images or image_id in selected:
            continue
        if not isinstance(question.get("answer"), str) or not question["answer"].strip():
            continue
        image_path = images[image_id]
        row = {
            "example_id": f"gqa:{question['id']}",
            "image_id": image_id,
            "image_path": image_path.relative_to(output_dir).as_posix(),
            "question": question["question"],
            "answer": question["answer"],
            "dataset": DATASET_ID,
            "dataset_revision": resolved_revision,
            "source_split": SOURCE_SPLIT,
            "image_sha256": file_sha256(image_path),
        }
        question_types = question.get("types") or {}
        if question_types.get("detailed"):
            row["question_type"] = question_types["detailed"]
        selected[image_id] = row
        if len(selected) == limit:
            break
    if len(selected) != limit:
        missing = sorted(set(images) - set(selected))
        raise ValueError(f"No training question found for selected images: {missing[:10]}")
    print(f"Matched {len(selected)} images after {questions_scanned:,} training questions.", flush=True)

    pending = output_dir / "manifest.pending.jsonl"
    with pending.open("w", encoding="utf-8", newline="\n") as stream:
        for image_id in images:
            stream.write(json.dumps(selected[image_id], ensure_ascii=False, sort_keys=True) + "\n")
    validate_manifest(pending)
    metadata = {**settings, "questions_scanned": questions_scanned, "count": len(selected),
                "manifest_sha256": file_sha256(pending)}
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    pending.replace(manifest_path)
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="Directory for this development sample")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--revision", default=None, help="Dataset commit hash (resolved to SHA when omitted)")
    args = parser.parse_args()
    manifest = prepare_gqa(args.output, args.limit, args.seed, args.revision)
    print(f"Development convenience sample: {manifest}")
    print("Not a representative GQA evaluation sample; no final OOD benchmark was accessed.")


if __name__ == "__main__":
    main()
