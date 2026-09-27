"""Render a self-contained local gallery of the fixed direction-control study.

Usage: python experiments/render_direction_gallery.py --run RUN_DIRECTORY
       --manifest MANIFEST_JSONL --output OUTSIDE_PROJECT/gallery.html

This viewer neither runs inference nor changes scores. Its outcome-selected
examples illustrate recorded interventions; they are not a learned policy.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
import math
from pathlib import Path, PureWindowsPath


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = ("named", "neutral", "sham", "frame")
REGIONS = ("tl", "tr", "bl", "br")
ACTIONS = ("direct", *(f"{condition}_{region}" for condition in CONDITIONS for region in REGIONS))
BOXES = {
    "tl": (0.0, 0.0, 0.6, 0.6), "tr": (0.4, 0.0, 1.0, 0.6),
    "bl": (0.0, 0.4, 0.6, 1.0), "br": (0.4, 0.4, 1.0, 1.0),
}
REGION_LABELS = {"tl": "Upper left", "tr": "Upper right", "bl": "Lower left", "br": "Lower right"}
REGION_COLORS = {"tl": "#2563eb", "tr": "#c77806", "bl": "#0f766e", "br": "#9333ea"}
CONDITION_LABELS = {
    "named": "Named crop", "neutral": "Neutral crop",
    "sham": "Named sham", "frame": "Original-frame instruction",
}
CONDITION_NOTES = {
    "named": "True crop; quadrant explicitly named.",
    "neutral": "Same true crop; quadrant name omitted.",
    "sham": "Full overview repeated; its description incorrectly claims the named quadrant.",
    "frame": "Same true crop; directions explicitly refer to the original image.",
}
STRATA = (
    "Frame improves on named", "Frame worsens relative to named",
    "Neutral improves on named", "Named/sham direction answer changes",
    "All answers unchanged from direct", "Other complete example", "Incomplete example",
)


def escape(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def finite_time(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) and value >= 0 else None


def seconds(value: object) -> str:
    number = finite_time(value)
    return f"{number:.3f} s" if number is not None else "unavailable"


def normalize(value: object) -> str:
    # Matches the study's conservative string normalization, not a new grader.
    return " ".join(str(value).strip().casefold().rstrip(".!?").split()) if value is not None else ""


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSON at {path.name}:{number}") from error
        if not isinstance(row, dict):
            raise ValueError(f"Expected an object at {path.name}:{number}")
        rows.append(row)
    return rows


def load_manifest(path: Path) -> dict[str, dict]:
    """Check every manifest path before selecting or opening any image."""
    base = path.parent.resolve()
    result = {}
    for row in read_jsonl(path):
        example_id = row.get("example_id")
        image_path = row.get("image_path")
        if not isinstance(example_id, str) or not example_id or example_id in result:
            raise ValueError("Manifest needs unique, nonempty string example IDs")
        if not isinstance(image_path, str) or not image_path:
            raise ValueError(f"Missing image_path for {example_id}")
        relative = Path(image_path)
        if relative.is_absolute() or PureWindowsPath(image_path).drive or PureWindowsPath(image_path).root:
            raise ValueError(f"Manifest image path must be relative: {example_id}")
        resolved = (base / relative).resolve()
        if not resolved.is_relative_to(base):
            raise ValueError(f"Manifest image escapes its directory: {example_id}")
        if not resolved.is_file():
            raise ValueError(f"Manifest image is missing: {example_id}")
        result[example_id] = {**row, "_resolved_image": resolved}
    return result


def valid_record(row: dict | None) -> bool:
    return bool(row and row.get("status") == "ok" and isinstance(row.get("correct"), bool))


def classify(rows: dict[str, dict]) -> str:
    """Assign the first matching, disclosed stratum; never cherry-pick quadrants."""
    if any(not valid_record(rows.get(action)) for action in ACTIONS):
        return "Incomplete example"
    means = {condition: sum(rows[f"{condition}_{region}"]["correct"] for region in REGIONS) / 4
             for condition in CONDITIONS}
    if means["frame"] > means["named"]:
        return "Frame improves on named"
    if means["frame"] < means["named"]:
        return "Frame worsens relative to named"
    if means["neutral"] > means["named"]:
        return "Neutral improves on named"
    for region in REGIONS:
        named = normalize(rows[f"named_{region}"].get("predicted_answer"))
        sham = normalize(rows[f"sham_{region}"].get("predicted_answer"))
        if named in {"left", "right"} and sham in {"left", "right"} and named != sham:
            return "Named/sham direction answer changes"
    direct = normalize(rows["direct"].get("predicted_answer"))
    if direct and all(normalize(rows[action].get("predicted_answer")) == direct for action in ACTIONS):
        return "All answers unchanged from direct"
    return "Other complete example"


def choose_examples(groups: dict[str, dict[str, dict]], limit: int) -> tuple[list[tuple[str, str]], dict[str, list[str]]]:
    buckets = {stratum: [] for stratum in STRATA}
    for example_id in sorted(groups):
        buckets[classify(groups[example_id])].append(example_id)
    chosen = []
    position = 0
    while len(chosen) < limit:
        added = False
        for stratum in STRATA:
            if position < len(buckets[stratum]) and len(chosen) < limit:
                chosen.append((stratum, buckets[stratum][position]))
                added = True
        if not added:
            break
        position += 1
    return chosen, buckets


def jpeg_data(image, bounds: tuple[int, int]) -> str:
    from PIL import Image

    thumbnail = image.copy()
    thumbnail.thumbnail(bounds, Image.Resampling.LANCZOS)
    buffer = io.BytesIO()
    thumbnail.save(buffer, format="JPEG", quality=77, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def image_panel(metadata: dict) -> str:
    from PIL import Image, ImageOps

    path = metadata["_resolved_image"]
    expected_hash = metadata.get("image_sha256")
    if expected_hash and hashlib.sha256(path.read_bytes()).hexdigest() != expected_hash:
        raise ValueError(f"Image hash differs from manifest: {metadata['example_id']}")
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened).convert("RGB")
    width, height = image.size
    overview = jpeg_data(image, (720, 520))
    overlays = []
    previews = []
    for region in REGIONS:
        x0, y0, x1, y1 = BOXES[region]
        color = REGION_COLORS[region]
        overlays.append(
            f'<rect x="{x0*width:.2f}" y="{y0*height:.2f}" width="{(x1-x0)*width:.2f}" '
            f'height="{(y1-y0)*height:.2f}" fill="none" stroke="{color}" stroke-width="2.5" '
            'vector-effect="non-scaling-stroke" />'
        )
        pixel_box = (int(x0*width), int(y0*height), min(width, math.ceil(x1*width)), min(height, math.ceil(y1*height)))
        crop = jpeg_data(image.crop(pixel_box), (190, 140))
        previews.append(
            f'<figure class="crop-preview"><img src="{crop}" alt="Source content of the {escape(REGION_LABELS[region].lower())} crop">'
            f'<figcaption style="color:{color}">{escape(REGION_LABELS[region])}</figcaption></figure>'
        )
    sham = jpeg_data(image, (190, 140))
    return (
        '<section class="image-panel"><h3>Original image and fixed regions</h3>'
        f'<svg class="overview" viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" '
        'role="img" aria-label="Original image with four fixed candidate rectangles">'
        f'<image width="{width}" height="{height}" href="{overview}" />' + "".join(overlays) + '</svg>'
        f'<p class="caption">Source: {width} × {height} pixels after orientation correction. Rectangles are fixed candidate regions, not learned localization.</p>'
        '<h3>Real-crop conditions show these regions</h3><div class="crop-previews">' + "".join(previews) + '</div>'
        '<div class="sham-preview"><img src="' + sham + '" alt="Full original image repeated by every sham action">'
        '<div><strong>Sham: the whole overview, four times</strong><p>The second image repeats the original overview unchanged. Only its claimed quadrant changes. It is not the crop illustrated above.</p></div></div>'
        '<p class="caption">These are compact previews of source content. Model views used the recorded resizing and token budgets; bicubic upsampling creates no new source detail.</p></section>'
    )


def grade_label(row: dict | None, direct: dict | None = None) -> tuple[str, str]:
    if not valid_record(row):
        return "Unavailable", "unavailable"
    if row["correct"]:
        return ("Correct · repair", "correct") if valid_record(direct) and not direct["correct"] else ("Correct", "correct")
    return ("Incorrect · harm", "incorrect") if valid_record(direct) and direct["correct"] else ("Incorrect", "incorrect")


def pixel_checks(rows: dict[str, dict]) -> str:
    """Expose recorded image equality without inferring it from answers."""
    direct_hashes = rows.get("direct", {}).get("image_rgb_sha256")
    direct_hash = direct_hashes[0] if isinstance(direct_hashes, list) and direct_hashes else None
    sham_ok = direct_hash is not None
    real_ok = True
    for region in REGIONS:
        hashes = rows.get(f"sham_{region}", {}).get("image_rgb_sha256")
        sham_ok = sham_ok and isinstance(hashes, list) and len(hashes) == 2 and hashes[0] == hashes[1] == direct_hash
        candidates = [rows.get(f"{condition}_{region}", {}).get("image_rgb_sha256") for condition in ("named", "neutral", "frame")]
        if not all(isinstance(item, list) and len(item) == 2 for item in candidates):
            real_ok = False
        elif len({item[1] for item in candidates}) != 1:
            real_ok = False
    if sham_ok and real_ok:
        return '<p class="pixel-note">Recorded RGB hashes agree: every sham repeats the first overview; each true crop is identical across named, neutral, and frame conditions.</p>'
    return '<p class="pixel-note warning">Pixel-equality checks are incomplete or disagree. Inspect the recorded image hashes before interpreting this example.</p>'


def result_cell(row: dict | None, direct: dict | None) -> str:
    if row is None:
        return '<td class="missing">No record</td>'
    prediction = row.get("predicted_answer")
    prediction = "[no parsed answer]" if prediction is None else prediction
    label, kind = grade_label(row, direct)
    elapsed = finite_time(row.get("elapsed_s"))
    initial = finite_time(direct.get("elapsed_s")) if direct else None
    total = elapsed + initial if elapsed is not None and initial is not None else None
    details = (
        '<details><summary>Recorded output &amp; view</summary>'
        f'<pre>{escape(row.get("response", ""))}</pre>'
        f'<p>Additional image: {escape(row.get("additional_view_kind", "unknown"))}<br>'
        f'Actual source box: {escape(row.get("actual_second_view_box"))}<br>'
        f'Claimed box: {escape(row.get("claimed_crop_box"))}<br>'
        f'Processed size: {escape(row.get("additional_size"))}<br>'
        f'Visual tokens: {escape(row.get("visual_tokens"))}</p>'
        f'<p class="prompt">{escape(row.get("followup_prompt", ""))}</p></details>'
    )
    return (
        f'<td><div class="prediction">{escape(prediction)}</div><div class="grade {kind}">{label}</div>'
        f'<div class="timing">Total {seconds(total)}<span>Extra {seconds(elapsed)}</span></div>{details}</td>'
    )


def example_card(stratum: str, example_id: str, rows: dict[str, dict], metadata: dict) -> str:
    direct = rows.get("direct")
    prediction = direct.get("predicted_answer") if direct else None
    label, kind = grade_label(direct)
    headers = ''.join(
        f'<th scope="col"><span class="dot" style="background:{REGION_COLORS[region]}"></span>{escape(REGION_LABELS[region])}</th>'
        for region in REGIONS
    )
    table_rows = []
    for condition in CONDITIONS:
        tag = "FULL OVERVIEW REPEAT" if condition == "sham" else "TRUE CROP"
        tag_class = "view-tag sham" if condition == "sham" else "view-tag"
        complete = all(valid_record(rows.get(f"{condition}_{region}")) for region in REGIONS)
        count = sum(rows[f"{condition}_{region}"]["correct"] for region in REGIONS) if complete else None
        score = f"{count}/4 correct" if count is not None else "Incomplete row"
        table_rows.append(
            f'<tr class="condition-{condition}"><th scope="row"><span class="{tag_class}">{tag}</span>'
            f'<strong>{escape(CONDITION_LABELS[condition])}</strong><span class="condition-note">{escape(CONDITION_NOTES[condition])}</span>'
            f'<span class="row-score">{score}</span></th>'
            + ''.join(result_cell(rows.get(f"{condition}_{region}"), direct) for region in REGIONS) + '</tr>'
        )
    return (
        '<article class="example"><div class="example-meta">'
        f'<span class="stratum">{escape(stratum)}</span><span>{escape(example_id)} · image {escape(metadata.get("image_id"))}</span></div>'
        f'<h2>{escape(metadata.get("question", ""))}</h2>'
        '<div class="answer-strip">'
        f'<div><span>Reference answer</span><strong>{escape(metadata.get("answer", ""))}</strong></div>'
        f'<div><span>Fresh direct answer</span><strong>{escape(prediction if prediction is not None else "[no parsed answer]")}</strong></div>'
        f'<div><span>Recorded direct grade</span><strong class="{kind}">{label}</strong></div>'
        f'<div><span>Direct latency</span><strong>{seconds(direct.get("elapsed_s") if direct else None)}</strong></div></div>'
        '<div class="example-content">' + image_panel(metadata)
        + '<section class="results"><h3>Four conditions × four regions</h3>'
        '<p class="caption">Every cell is an independent follow-up from the same fresh direct answer. Original recorded grades are preserved.</p>'
        '<div class="table-wrap"><table><thead><tr><th scope="col">Condition</th>' + headers
        + '</tr></thead><tbody>' + ''.join(table_rows) + '</tbody></table></div>'
        + pixel_checks(rows) + '</section></div></article>'
    )


CSS = """
:root{color-scheme:light;font-family:Inter,system-ui,-apple-system,'Segoe UI',sans-serif;color:#172b4d;background:#f3f6fa}
*{box-sizing:border-box}body{margin:0}main{max-width:1640px;margin:auto;padding:42px 30px 60px}
header{max-width:1180px;margin-bottom:30px}.eyebrow{text-transform:uppercase;letter-spacing:.13em;font-size:12px;font-weight:700;color:#0f766e}
h1{font-size:clamp(29px,4vw,44px);letter-spacing:-.035em;margin:10px 0 14px}h2{font-size:23px;line-height:1.4;margin:18px 0 14px}
h3{font-size:14px;margin:0 0 10px}p{line-height:1.65}.lead{font-size:17px}.note,.caption{font-size:12px;color:#596a80}.caption{margin:8px 0 16px}
.scope{border-left:4px solid #c77806;background:#fff6e9;padding:12px 16px;font-size:14px;line-height:1.6;border-radius:0 7px 7px 0}
.selection{margin-top:16px;border:1px solid #dce4ee;background:#fff;padding:13px 17px;border-radius:9px}.selection summary{font-size:13px;font-weight:600}
.selection ol{padding-left:22px;font-size:12px;line-height:1.8;color:#52637a}.example{background:#fff;border:1px solid #dce4ee;border-radius:15px;padding:26px;margin:28px 0;box-shadow:0 2px 6px #13294206}
.example-meta{display:flex;gap:12px;align-items:center;flex-wrap:wrap;color:#627187;font-size:12px;overflow-wrap:anywhere}.stratum{color:#245064;font-weight:650;background:#eaf2f7;padding:7px 10px;border-radius:6px}
.answer-strip{display:flex;gap:20px 35px;flex-wrap:wrap;padding:14px 17px;border-radius:8px;background:#f7f9fc;margin-bottom:26px}.answer-strip div{display:flex;flex-direction:column;gap:5px}
.answer-strip span{font-size:11px;color:#617089}.answer-strip strong{font-size:15px;overflow-wrap:anywhere}.example-content{display:grid;grid-template-columns:minmax(260px,.7fr) minmax(720px,1.6fr);gap:30px;align-items:start}
.overview{width:100%;height:auto;max-height:440px;background:#f4f6f9;display:block}.crop-previews{display:grid;grid-template-columns:repeat(4,1fr);gap:7px}.crop-preview{margin:0;min-width:0}
.crop-preview img{display:block;max-width:100%;height:78px;object-fit:contain;background:#f4f6f9;width:100%}.crop-preview figcaption{font-size:10px;line-height:1.4;text-align:center;margin-top:5px;font-weight:600}
.sham-preview{display:flex;gap:12px;align-items:center;border:1px dashed #c89b50;background:#fffaf0;border-radius:7px;margin-top:16px;padding:10px}.sham-preview img{width:85px;height:66px;object-fit:contain;flex-shrink:0}
.sham-preview strong{font-size:12px}.sham-preview p{font-size:11px;line-height:1.5;margin:4px 0 0;color:#765b2f}.table-wrap{overflow:auto}table{width:100%;border-collapse:collapse;table-layout:fixed;min-width:710px;font-size:12px}
th,td{padding:12px 9px;border:1px solid #e2e8f0;text-align:left;vertical-align:top;overflow-wrap:anywhere}thead th{background:#eef3f8;font-size:11px;font-weight:650}
thead th:first-child{width:25%}tbody th{font-weight:400;background:#f8fafc}tbody th strong{display:block;font-size:12px;line-height:1.4}.condition-note{display:block;font-size:10px;line-height:1.5;color:#617089;margin-top:6px}
.view-tag{display:inline-block;font-size:8px;font-weight:750;letter-spacing:.055em;background:#e9f1f4;color:#245064;padding:4px 5px;border-radius:3px;margin-bottom:7px}.view-tag.sham{background:#fff0d4;color:#865809}
.condition-sham td,.condition-sham th{background:#fffbf3}.row-score{display:block;font-size:10px;font-weight:650;margin-top:8px;color:#52637a}.prediction{font-size:14px;font-weight:650;line-height:1.4;min-height:20px}
.grade{font-size:10px;margin-top:5px;line-height:1.4}.correct{color:#0f766e}.incorrect{color:#b03942}.unavailable,.missing{color:#748196}.timing{font-size:10px;margin-top:9px;font-variant-numeric:tabular-nums;line-height:1.55}
.timing span{display:block;color:#748196}.dot{display:inline-block;height:7px;width:7px;border-radius:50%;margin-right:5px}details{margin-top:8px}summary{cursor:pointer;font-size:10px;color:#617089}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;font-size:11px;line-height:1.5;background:#f2f5f8;padding:7px;margin-bottom:6px}td details p{font-size:10px;line-height:1.6;margin:6px 0}
.pixel-note{font-size:11px;line-height:1.6;color:#52716a;background:#f1f7f4;border-radius:5px;padding:9px 11px}.pixel-note.warning{background:#fff4e5;color:#815d1b}
footer{border-top:1px solid #dce4ee;margin-top:35px;padding-top:16px;font-size:12px;line-height:1.7;color:#617089}code{font-size:11px;overflow-wrap:anywhere}
@media(max-width:1180px){.example-content{grid-template-columns:1fr}.image-panel{max-width:620px}.overview{max-height:440px}main{padding:25px 15px}.example{padding:20px}.crop-preview img{height:100px}}
@media(max-width:560px){.answer-strip{gap:15px;display:grid;grid-template-columns:1fr 1fr}.answer-strip strong{font-size:13px}.example{padding:15px}.crop-preview img{height:63px}.crop-preview figcaption{font-size:9px}}
@media print{body{background:white}main{padding:0}.example{box-shadow:none;break-inside:avoid}.selection{display:none}td details{display:none}.example-content{grid-template-columns:1fr}.image-panel{max-width:520px}}
"""


def render_gallery(run: Path, manifest: Path, output: Path, max_examples: int = 12) -> Path:
    """Render logged outcomes only; never read another run or fetch remote data."""
    if isinstance(max_examples, bool) or not isinstance(max_examples, int) or not 1 <= max_examples <= 100:
        raise ValueError("max_examples must be an integer between 1 and 100")
    run, manifest, output = Path(run).resolve(), Path(manifest).resolve(), Path(output).resolve()
    if output.is_relative_to(PROJECT_ROOT):
        raise ValueError("Write the image-containing gallery outside the code project directory")
    if output.suffix.lower() not in {".html", ".htm"}:
        raise ValueError("Output must be an .html or .htm file")
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    if metadata.get("experiment") != "direction_controls_v2":
        raise ValueError("Expected a direction_controls_v2 run")
    manifest_hash = hashlib.sha256(manifest.read_bytes()).hexdigest()
    if metadata.get("manifest_sha256") != manifest_hash:
        raise ValueError("Manifest hash does not match the run")
    examples = load_manifest(manifest)
    expected_ids = metadata.get("example_ids")
    if not isinstance(expected_ids, list) or not expected_ids or len(set(expected_ids)) != len(expected_ids):
        raise ValueError("Run metadata must declare unique example_ids")
    if set(expected_ids) - set(examples):
        raise ValueError("Run declares examples absent from the manifest")
    groups = {example_id: {} for example_id in expected_ids}
    records = read_jsonl(run / "records.jsonl")
    for record in records:
        example_id, action = record.get("example_id"), record.get("action")
        if example_id not in groups or action not in ACTIONS:
            raise ValueError("Record has an undeclared example or action")
        if action in groups[example_id]:
            raise ValueError(f"Duplicate example/action: {example_id}, {action}")
        if str(record.get("image_id")) != str(examples[example_id].get("image_id")):
            raise ValueError(f"Image identity differs from manifest: {example_id}")
        groups[example_id][action] = record
    chosen, buckets = choose_examples(groups, max_examples)
    complete = sum(all(valid_record(rows.get(action)) for action in ACTIONS) for rows in groups.values())
    expected_count = len(expected_ids)
    primary_target = metadata.get("config", {}).get("protocol", {}).get("expected_examples", 400)
    small = expected_count < primary_target
    scope = (
        "Engineering smoke / partial-panel diagnostic only. This small run is not sufficient for a scientific conclusion."
        if small else "Development-panel diagnostic. The same images were used during method development; this is not held-out confirmation."
    )
    if complete != expected_count:
        scope += " Coverage is incomplete; missing cells are displayed and no complete-study conclusion is available."
    strata_list = ''.join(f'<li>{escape(stratum)}: {len(buckets[stratum])} examples</li>' for stratum in STRATA)
    cards = ''.join(example_card(stratum, example_id, groups[example_id], examples[example_id]) for stratum, example_id in chosen)
    errors_path = run / "errors.jsonl"
    error_count = len(read_jsonl(errors_path)) if errors_path.exists() else 0
    document = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src data:; style-src \'unsafe-inline\'; base-uri \'none\'; form-action \'none\'">'
        '<title>LookAgain · direction-control diagnostic</title><style>' + CSS + '</style></head><body><main><header>'
        '<div class="eyebrow">LookAgain · local experiment viewer</div><h1>Which view — and which coordinate frame?</h1>'
        '<p class="lead">A frozen model receives the same original image, then a fixed crop or a repeated overview. '
        'This gallery compares quadrant naming, neutral wording, a deliberately mismatched sham, and an explicit original-frame instruction. '
        '<strong>No learned controller or learned region selection is shown.</strong></p>'
        f'<div class="scope">{escape(scope)}<br>{complete}/{expected_count} complete examples; '
        f'{len(records)}/{17*expected_count} recorded action slots; {error_count} logged execution errors. Showing {len(chosen)} examples.</div>'
        '<p class="note">The real-crop conditions use identical regional pixels. The sham repeats the first overview byte-for-byte while incorrectly claiming a quadrant. '
        'This is not a full content × naming factorial design: neutral sham is absent. The sham contrast changes image/text congruity and does not isolate a pure wording effect.</p>'
        '<p class="note">Total latency = fresh direct call + selected follow-up call. Extra latency is the follow-up alone. '
        'Correctness is the original recorded pilot score, not a new semantic judgment. Source resizing may upsample small images; it creates no new source detail.</p>'
        '<details class="selection" open><summary>Selection rule and run provenance — illustrative, outcome-selected, not representative</summary>'
        '<p class="note">Assign each example to the first matching group below. Frame and neutral comparisons use mean correctness over all four quadrants. '
        'The direction-change group requires different named/sham predictions that are exactly left and right for at least one quadrant. '
        'Within each group, sort example IDs lexically; select round-robin across groups up to the display limit. No best quadrant is selected.</p>'
        '<ol>' + strata_list + '</ol>'
        f'<p class="note">Run created: {escape(metadata.get("created_utc", "unknown"))}<br>'
        f'Run fingerprint: <code>{escape(metadata.get("fingerprint", "unavailable"))}</code><br>'
        f'Manifest SHA-256: <code>{escape(manifest_hash)}</code></p></details></header>'
        + cards + '<footer>Self-contained local diagnostic export: only embedded JPEG thumbnails and SVG overlays; no scripts or remote assets. '
        'Images retain their original rights. This file makes no claim that source images may be publicly redistributed; review the dataset terms before sharing it. '
        'These examples illustrate fixed experimental interventions and cannot establish population accuracy, transfer, or controller performance.</footer></main></body></html>'
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Directory containing run.json and records.jsonl")
    parser.add_argument("--manifest", type=Path, required=True, help="The original image manifest, matching the run hash")
    parser.add_argument("--output", type=Path, required=True, help="Local HTML path outside the code project")
    parser.add_argument("--max-examples", type=int, default=12)
    args = parser.parse_args()
    path = render_gallery(args.run, args.manifest, args.output, args.max_examples)
    print(f"Wrote local, outcome-selected direction gallery: {path}")


if __name__ == "__main__":
    main()
