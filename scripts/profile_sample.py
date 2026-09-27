"""Profile a saved GQA development sample without model outputs or embeddings."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path, PureWindowsPath
import statistics

import PIL
from PIL import Image




def sha256(content):
    return hashlib.sha256(content).hexdigest()


def percentile(values, p):
    values = sorted(values)
    position = (len(values) - 1) * p
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def summary(values):
    return {
        "min": min(values), "p25": percentile(values, 0.25),
        "median": statistics.median(values), "p75": percentile(values, 0.75),
        "p95": percentile(values, 0.95), "max": max(values),
        "mean": statistics.mean(values),
    }


def dhash(image):
    # Difference hash: 64 horizontal comparisons, no learned features.
    pixels = list(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes())
    value = 0
    for y in range(8):
        for x in range(8):
            value = (value << 1) | (pixels[y * 9 + x] > pixels[y * 9 + x + 1])
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="Saved GQA manifest beside its images directory")
    parser.add_argument("--metadata", type=Path, help="Defaults to sample_metadata.json beside the manifest")
    parser.add_argument("--output", type=Path, required=True, help="Directory for sample_profile.json and sample_profile.md")
    args = parser.parse_args()

    def counts(counter, limit=None):
        ordered = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
        if limit is not None:
            ordered = ordered[:limit]
        return [{"value": key, "count": value, "percent": round(100 * value / N, 2)}
                for key, value in ordered]


    manifest = args.manifest.resolve()
    SAMPLE = manifest.parent
    REPORTS = args.output.resolve()
    metadata_path = args.metadata.resolve() if args.metadata else SAMPLE / "sample_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    N = len(rows)
    if not N:
        raise ValueError("Manifest is empty")
    if sha256(manifest.read_bytes()) != metadata["manifest_sha256"]:
        raise ValueError("Manifest checksum differs from sample metadata")
    if metadata.get("count") != N:
        raise ValueError("Manifest row count differs from sample metadata")
    if metadata.get("sampling") != "first_unique_images_then_first_matching_question_in_pinned_stream_order":
        raise ValueError("This generator describes the pinned GQA development-prefix protocol only")
    if metadata.get("source_split") != "train" or metadata.get("sample_role") != "development_only":
        raise ValueError("Expected a training-source development sample")
    for field in ("image_id", "example_id"):
        values = [row.get(field) for row in rows]
        if any(not isinstance(value, str) or not value for value in values) or len(set(values)) != N:
            raise ValueError(f"Manifest {field} values must be nonempty and unique")

    widths, heights, lengths, areas, aspects, byte_sizes = [], [], [], [], [], []
    byte_groups, pixel_groups = defaultdict(list), defaultdict(list)
    descriptors = []
    dimensions = Counter()
    formats = Counter()
    answer_counts = Counter()
    type_counts = Counter()
    first_word_counts = Counter()
    question_lengths = []
    question_duplicate_counts = Counter()
    answer_lengths = Counter()

    for row in rows:
        relative = row["image_path"]
        if not isinstance(relative, str) or Path(relative).is_absolute() or PureWindowsPath(relative).drive:
            raise ValueError("Image paths must be relative to the manifest")
        path = (SAMPLE / relative).resolve()
        if not path.is_relative_to(SAMPLE):
            raise ValueError("Image path escapes the manifest directory")
        for field in ("dataset", "dataset_revision", "source_split"):
            if row.get(field) != metadata.get(field):
                raise ValueError(f"Manifest {field} disagrees with metadata")
        for field in ("question", "answer"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Manifest {field} must be a nonempty string")
        encoded = path.read_bytes()
        actual_sha = sha256(encoded)
        if actual_sha != row["image_sha256"]:
            raise ValueError(f"Image checksum mismatch: {row['image_id']}")
        byte_groups[actual_sha].append(row["image_id"])
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            rgb = image.convert("RGB")
            pixel_sha = sha256(f"RGB:{width}:{height}:".encode("ascii") + rgb.tobytes())
            pixel_groups[pixel_sha].append(row["image_id"])
            descriptors.append((row["image_id"], dhash(rgb), width, height))
            formats[image.format] += 1
        widths.append(width)
        heights.append(height)
        lengths.append(max(width, height))
        areas.append(width * height)
        aspects.append(width / height)
        byte_sizes.append(len(encoded))
        dimensions[f"{width}x{height}"] += 1
        answer = row["answer"].strip().casefold()
        answer_counts[answer] += 1
        answer_lengths[len(answer.split())] += 1
        type_counts[row.get("question_type", "missing")] += 1
        question = row["question"].strip()
        question_lengths.append(len(question.split()))
        question_duplicate_counts[question] += 1
        first_word_counts[question.split()[0].casefold()] += 1

    near_pairs = []
    for index, (image_id, code, width, height) in enumerate(descriptors):
        for other_id, other_code, other_width, other_height in descriptors[index + 1:]:
            distance = (code ^ other_code).bit_count()
            if distance <= 4:
                near_pairs.append({
                    "image_id_a": image_id, "image_id_b": other_id,
                    "dhash64_hamming_distance": distance,
                    "dimensions_a": [width, height], "dimensions_b": [other_width, other_height],
                })
    near_pairs.sort(key=lambda item: (item["dhash64_hamming_distance"], item["image_id_a"], item["image_id_b"]))
    exact_duplicates = [group for group in byte_groups.values() if len(group) > 1]
    pixel_duplicates = [group for group in pixel_groups.values() if len(group) > 1]
    yes_no = answer_counts["yes"] + answer_counts["no"]

    profile = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "profile_version": 1,
        "pillow_version": PIL.__version__,
        "sample": {**metadata, "dataset_url": f"https://huggingface.co/datasets/{metadata['dataset']}/tree/{metadata['dataset_revision']}"},
        "validation": {
            "manifest_hash_matches_metadata": True,
            "all_image_hashes_match_manifest": True,
            "all_images_decoded": True,
            "unique_image_ids": len({row["image_id"] for row in rows}),
            "unique_example_ids": len({row["example_id"] for row in rows}),
            "one_question_per_image": len({row["image_id"] for row in rows}) == N,
        },
        "images": {
            "count": N,
            "width_pixels": summary(widths), "height_pixels": summary(heights),
            "long_edge_pixels": summary(lengths), "area_pixels": summary(areas),
            "width_over_height": summary(aspects), "file_bytes": summary(byte_sizes),
            "total_file_bytes": sum(byte_sizes), "formats": dict(formats),
            "portrait_count": sum(w < h for w, h in zip(widths, heights)),
            "landscape_count": sum(w > h for w, h in zip(widths, heights)),
            "square_count": sum(w == h for w, h in zip(widths, heights)),
            "long_edge_at_most_640_count": sum(value <= 640 for value in lengths),
            "long_edge_at_most_1024_count": sum(value <= 1024 for value in lengths),
            "most_common_dimensions": counts(dimensions, 15),
        },
        "questions": {
            "detailed_types": counts(type_counts), "first_words": counts(first_word_counts),
            "length_whitespace_words": summary(question_lengths),
            "unique_exact_question_strings": len(question_duplicate_counts),
            "repeated_exact_question_string_groups": sum(value > 1 for value in question_duplicate_counts.values()),
        },
        "answers": {
            "unique_normalized_strings": len(answer_counts), "normalization": "strip + Unicode casefold",
            "yes_count": answer_counts["yes"], "no_count": answer_counts["no"],
            "yes_no_count": yes_no, "yes_no_percent": 100 * yes_no / N,
            "most_common_30": counts(answer_counts, 30),
            "answer_length_whitespace_words": dict(sorted(answer_lengths.items())),
        },
        "duplicate_screening": {
            "exact_encoded_sha256_duplicate_groups": exact_duplicates,
            "decoded_rgb_pixel_sha256_duplicate_groups": pixel_duplicates,
            "dhash_method": "64-bit horizontal difference hash: grayscale, Lanczos resize9x8, adjacent comparisons",
            "dhash_candidate_threshold": 4, "pairs_compared": N * (N - 1) // 2,
            "dhash_candidate_pairs": near_pairs,
            "interpretation": "Heuristic candidates only; no candidates does not exclude crops, edits, or semantically similar scenes. No learned embeddings used.",
        },
        "limitations": [
            "Deterministic first-image prefix and first matching question; not a representative or random sample.",
            "Selected from GQA training split for development; not a final held-out benchmark.",
            "One question per image avoids within-sample repeated-image weighting but does not establish balanced answer or question-type coverage.",
            "No comparison against the complete GQA distribution was performed.",
            "Byte and pixel duplicate checks cover this saved sample only; they cannot establish absence from VLM pretraining or other datasets.",
            "Small native image sizes limit claims about newly acquired visual detail: resizing or cropping cannot create missing source pixels.",
            "This profile contains no VLM predictions, measured model results, or image embeddings.",
        ],
    }

    def fmt_number(value):
        return f"{value:,.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else f"{value:,}"

    lines = [
        "# GQA development sample profile", "",
        f"The saved sample contains **{N} images and {N} questions**, one question per image. All source image bytes and the manifest match their recorded SHA-256 checksums; all images decode successfully.", "",
        "This is a **development convenience sample**, not a representative GQA evaluation. Images are the first distinct images in the pinned training-image stream; each uses its first matching question in the pinned training-question stream. The seed does not randomize this prefix. No model outputs were used in this profile.", "",
        "## Provenance", "",
        f"- Dataset: [{metadata['dataset']}]({profile['sample']['dataset_url']}).",
        f"- Revision: `{metadata['dataset_revision']}`.",
        f"- Configurations: `{metadata['image_config']}` and `{metadata['question_config']}`; source split: `{metadata['source_split']}`.",
        f"- Training questions scanned to join the selected images: {metadata['questions_scanned']:,}.",
        f"- Manifest SHA-256: `{metadata['manifest_sha256']}`.",
        f"- Encoded image files total: {sum(byte_sizes) / 1024 / 1024:.2f} MiB; formats: {', '.join(f'{key}: {value}' for key, value in formats.items())}.",
        "", "## Native image sizes", "",
        "| Measure | Minimum | Median | 95th percentile | Maximum |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for key, label in [("width_pixels", "Width, px"), ("height_pixels", "Height, px"), ("long_edge_pixels", "Long edge, px"), ("area_pixels", "Area, pixels")]:
        values = profile["images"][key]
        lines.append(f"| {label} | {fmt_number(values['min'])} | {fmt_number(values['median'])} | {fmt_number(values['p95'])} | {fmt_number(values['max'])} |")
    lines.extend([
        "", f"Landscape: {profile['images']['landscape_count']}; portrait: {profile['images']['portrait_count']}; square: {profile['images']['square_count']}. Long edge ≤640px: {profile['images']['long_edge_at_most_640_count']}/{N}; ≤1024px: {profile['images']['long_edge_at_most_1024_count']}/{N}.",
        "", "These are the original saved image dimensions, not the model processor's resized dimensions. A larger inference canvas does not add source pixels. Crop benefits here may come from re-encoding and attention allocation; this sample alone cannot establish benefits on genuinely high-resolution scenes.",
        "", "## Answer and question composition", "",
        f"There are {len(answer_counts)} distinct normalized answer strings. Yes/no answers account for **{yes_no}/{N} ({100 * yes_no / N:.1f}%)**: yes={answer_counts['yes']}, no={answer_counts['no']}. The median question length is {statistics.median(question_lengths):g} whitespace-separated words. There are {len(question_duplicate_counts)} distinct exact question strings.",
        "", "| Detailed GQA question type | Count | Share |", "| --- | ---: | ---: |",
    ])
    for item in profile["questions"]["detailed_types"]:
        lines.append(f"| {item['value']} | {item['count']} | {item['percent']:.2f}% |")
    lines.extend(["", "Most frequent answers:", "", "| Answer | Count | Share |", "| --- | ---: | ---: |"])
    for item in profile["answers"]["most_common_30"][:15]:
        lines.append(f"| {item['value']} | {item['count']} | {item['percent']:.2f}% |")
    lines.extend([
        "", "## Duplicate screening", "",
        f"Exact encoded-file duplicates: **{len(exact_duplicates)} groups**. Identical decoded RGB images with matching dimensions: **{len(pixel_duplicates)} groups**. These checks cover only the saved sample.",
        "", f"A simple 64-bit difference-hash screen compared {N * (N - 1) // 2:,} image pairs and found **{len(near_pairs)} candidate pairs** with Hamming distance ≤4. This is a coarse visual similarity heuristic, not proof of duplication or uniqueness; it can miss crops and edits or flag unrelated low-detail scenes. No learned embeddings were computed.",
    ])
    if near_pairs:
        lines.extend(["", "| Image A | Image B | Difference-hash distance |", "| --- | --- | ---: |"])
        for item in near_pairs:
            lines.append(f"| {item['image_id_a']} | {item['image_id_b']} | {item['dhash64_hamming_distance']} |")
    lines.extend(["", "## Interpretation limits", ""])
    lines.extend(f"- {text}" for text in profile["limitations"][:6])
    lines.extend(["", "The accompanying `sample_profile.json` contains the complete count tables, summary statistics, screening settings, and provenance. No inference-result claim follows from this profile.", ""])

    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "sample_profile.json").write_text(json.dumps(profile, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (REPORTS / "sample_profile.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"count": N, "long_edge_pixels": summary(lengths), "yes_no_percent": 100 * yes_no / N,
                      "unique_answers": len(answer_counts), "exact_duplicate_groups": len(exact_duplicates),
                      "pixel_duplicate_groups": len(pixel_duplicates), "dhash_candidates": len(near_pairs),
                      "type_counts": dict(type_counts)}, indent=2))


if __name__ == "__main__":
    main()
