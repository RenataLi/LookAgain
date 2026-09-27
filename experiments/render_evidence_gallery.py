"""Render an offline gallery only from a completed, strictly analyzed v5 panel.

Creates exact reconstructed input PNGs and a provenance sidecar. No model is
loaded and no source annotations or inputs are changed. Do not run before the
separate condition-blind semantic review has been locked.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path, PureWindowsPath
import sys

from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
from evidence_availability_core import ACTIONS, build_request
from evidence_availability import experiment_digest
from native_detail_core import image_digest

PRIMARY_ACTIONS = ("direct_256", "native_256", "degraded_256")
IDENTITY_KEYS = ("config", "manifest_sha256", "code_sha256", "model", "runtime", "role", "example_ids")
SELECTION_RULE = ("All reports with unequal primary native_256/degraded_256 parsed answers, plus raw-response differences "
                  "when either parse is invalid, in lexicographic example-ID order; then the first remaining report in "
                  "lexicographic order as an unchanged comparison. No accuracy-based ranking or cap is applied.")


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def escape(value):
    return html.escape(str(value), quote=True)


def require_relative_file(base, relative):
    require(isinstance(relative, str) and relative != "", "Missing relative asset path")
    windows = PureWindowsPath(relative)
    require(not windows.is_absolute() and not windows.drive and not Path(relative).is_absolute(), "Absolute asset path")
    require(":" not in relative and "\\" not in relative, "Nonportable asset path")
    target = (base / relative).resolve()
    require(target.is_relative_to(base.resolve()) and target.is_file(), "Missing or escaping asset")
    return target


def load_completed(run, manifest, summary_path):
    """Check completion and strict-summary bindings before reading any responses."""
    run, manifest, summary_path = map(lambda p: Path(p).resolve(), (run, manifest, summary_path))
    completion = read_json(run / "completed.json")
    summary = read_json(summary_path)
    require(summary.get("status") == "complete" and summary.get("integrity", {}).get("passed") is True,
            "A completed strict main-panel summary is required")
    require(summary.get("experiment") == "evidence_availability_v5" and summary.get("role") == "main",
            "Engineering smoke or another experiment cannot produce this gallery")
    metadata = read_json(run / "run.json")
    require(metadata.get("role") == "main" and metadata.get("experiment") == "evidence_availability_v5", "Wrong run identity")
    identity = {key: metadata[key] for key in IDENTITY_KEYS}
    expected_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    require(expected_fingerprint == metadata["fingerprint"] == summary["bindings"]["run_fingerprint"], "Run fingerprint mismatch")
    config = metadata["config"]
    require(metadata["code_sha256"] == experiment_digest() == config["protocol"]["inference_code_sha256"],
            "Inference/preparation code changed after the run")
    bindings = (("run_json_sha256", run / "run.json"), ("records_sha256", run / "records.jsonl"),
        ("completion_sha256", run / "completed.json"), ("manifest_sha256", manifest),
        ("preparation_metadata_sha256", manifest.parent / "selection_metadata.json"),
        ("roi_audit_sha256", manifest.parent / "roi_audit.jsonl"))
    for key, path in bindings:
        require(summary["bindings"][key] == digest(path), f"Strict-summary binding mismatch: {key}")
    require(completion["records_sha256"] == summary["bindings"]["records_sha256"], "Completion hash mismatch")
    require(digest(manifest) == config["protocol"]["main_manifest_sha256"] == metadata["manifest_sha256"], "Manifest pin mismatch")
    source_rows = read_jsonl(manifest)
    sources = {r["example_id"]: r for r in source_rows}
    require(len(sources) == len(source_rows) == config["protocol"]["main_examples"] == 85, "Need the complete 85-report cohort")
    require(len({r["source_id"] for r in source_rows}) == len(sources), "Repeated original reports")
    require(metadata["example_ids"] == [r["example_id"] for r in source_rows], "Source order or IDs changed")
    expected = len(sources) * len(ACTIONS)
    require(expected == completion["records"] == summary["coverage"]["records"] == 850, "Need all 850 completed calls")
    # Validate every supplied path, even when its report will not be displayed.
    for source in source_rows:
        source_path = require_relative_file(manifest.parent, source["image_path"])
        require(digest(source_path) == source["image_sha256"], "Source image hash mismatch")
    groups = {key: {} for key in sources}
    records = read_jsonl(run / "records.jsonl")
    require(len(records) == expected, "Incomplete/excess records")
    for row in records:
        key, action = row["example_id"], row["action"]
        require(key in groups and action in ACTIONS and action not in groups[key] and row.get("status") == "ok",
                "Unexpected, duplicate or failed record")
        for field, source_field in (("question", "question"), ("target_answer", "answer"), ("image_id", "image_id"),
                ("source_id", "source_id"), ("source_image_sha256", "image_sha256"), ("roi_provenance_sha256", "roi_provenance_sha256")):
            require(row[field] == sources[key][source_field], f"Record/source mismatch: {field}")
        groups[key][action] = row
    require(all(set(g) == set(ACTIONS) for g in groups.values()), "Incomplete action groups")
    return metadata, summary, sources, groups


def select_cases(groups):
    changed = []
    for key in sorted(groups):
        a, b = groups[key]["native_256"], groups[key]["degraded_256"]
        if a["predicted_answer"] != b["predicted_answer"] or (
                a["response"] != b["response"] and not (a["parse_valid"] and b["parse_valid"])):
            changed.append(key)
    stable = next((key for key in sorted(groups) if key not in changed), None)
    return changed, stable


def save_input(image, assets):
    rgb_hash = image_digest(image)
    target = assets / (rgb_hash + ".png")
    if not target.exists():
        image.save(target, format="PNG")
    with Image.open(target) as saved:
        require(image_digest(saved) == rgb_hash, "Lossless saved PNG differs from reconstructed RGB input")
    return target.name, {"rgb_sha256": rgb_hash, "file_sha256": digest(target), "size": list(image.size), "format": "PNG"}


def score_table(group, actions):
    lines = []
    for action in actions:
        row = group[action]
        flags = []
        if not row["parse_valid"]:
            flags.append("INVALID PARSE")
        if row["generation_truncated"]:
            flags.append("TRUNCATED")
        lines.append(f'<tr><th scope="row">{escape(action)}</th><td><pre>{escape(row["response"])}</pre>'
            f'<small>Parsed: {escape(row["predicted_answer"])} · {escape(", ".join(flags) or "valid; not truncated")}</small></td>'
            f'<td>{100*row["official_em"]:.0f}%</td><td>{100*row["conservative_text_em"]:.0f}%</td>'
            f'<td>{row["visual_tokens"]}</td><td>{row["input_tokens"]}</td><td>{row["elapsed_s"]:.3f}</td></tr>')
    return ('<div class="scroll"><table><thead><tr><th>Action</th><th>Complete raw response</th><th>Official EM</th>'
            '<th>Conservative EM</th><th>Actual visual tokens</th><th>Total input tokens</th><th>Standalone seconds</th>'
            '</tr></thead><tbody>' + "".join(lines) + '</tbody></table></div>')


def render_case(key, role, source, group, config, manifest, assets, include_source=False):
    source_path = require_relative_file(manifest.parent, source["image_path"])
    with Image.open(source_path) as raw:
        page = ImageOps.exif_transpose(raw).convert("RGB")
    saved, requests = {}, {}
    for action in PRIMARY_ACTIONS:
        messages, images, geometry = build_request(page, source["question"], action, config, source["roi_pixels"])
        row = group[action]
        require(all(row.get(k) == value for k, value in geometry.items()), f"Reconstructed input differs: {key}/{action}")
        require(row["image_grid_thw"] == [[1, im.height//16, im.width//16] for im in images], "Saved grid differs")
        paths = []
        for image in images:
            name, info = save_input(image, assets)
            relative = assets.name + "/" + name
            paths.append(relative)
            saved[relative] = info
        requests[action] = {"input_files": paths, "messages": messages, "messages_sha256": geometry["messages_sha256"],
            "image_rgb_sha256": geometry["image_rgb_sha256"], "source_roi_pixels": geometry["source_roi_pixels"],
            "actual_image_grid_thw": row["image_grid_thw"], "actual_visual_tokens": row["visual_tokens"]}
    require(requests["direct_256"]["input_files"][0] == requests["native_256"]["input_files"][0] == requests["degraded_256"]["input_files"][0],
            "Primary overview inputs differ")
    figures = []
    for label, description, asset in (
        ("Same overview · budget cap 256", "Input 1 for all three primary actions", requests["direct_256"]["input_files"][0]),
        ("Native evidence region", "Input 2 from the original 200-DPI render", requests["native_256"]["input_files"][1]),
        ("Degraded evidence region", "Input 2 reconstructed only from the overview", requests["degraded_256"]["input_files"][1])):
        width, height = saved[asset]["size"]
        figures.append(f'<figure><a href="{escape(asset)}"><img src="{escape(asset)}" alt="{escape(label)}" loading="lazy"></a>'
            f'<figcaption><strong>{escape(label)}</strong><br>{escape(description)}<br>{width}×{height} RGB pixels · lossless PNG</figcaption></figure>')
    content = f'<section><p class="eyebrow">{escape(role)}</p><h2>{escape(source["question"])}</h2>'
    content += f'<p><code>{escape(key)}</code> · {escape(source["source_id"])} · source flag: {escape(source.get("source_quality_category", "unknown"))}</p>'
    content += f'<p>Unchanged reference: <strong>{escape(source["answer"])}</strong></p><div class="images">' + "".join(figures) + '</div>'
    content += '<p class="note">Click a view to inspect the exact reconstructed input PNG. Browser display scaling may differ; no annotation boxes or labels were drawn into these inputs.</p>'
    content += score_table(group, [*PRIMARY_ACTIONS, "highres"])
    content += '<details><summary>Other overview budgets: all remaining recorded responses</summary>'
    content += score_table(group, [a for a in ACTIONS if a not in (*PRIMARY_ACTIONS, "highres")]) + '</details>'
    source_asset = None
    if include_source:
        name, info = save_input(page, assets)
        source_asset = assets.name + "/" + name
        saved[source_asset] = info
        x0, y0, x1, y1 = source["roi_pixels"]
        content += ('<details><summary>Source page and annotation-guided region (illustration; not a model input)</summary>'
            f'<svg class="source" viewBox="0 0 {page.width} {page.height}" role="img" aria-label="Source page with privileged region overlay">'
            f'<image href="{escape(source_asset)}" width="{page.width}" height="{page.height}"/>'
            f'<rect x="{x0}" y="{y0}" width="{x1-x0}" height="{y1-y0}" fill="none" stroke="#e03131" stroke-width="6"/>'
            '</svg></details>')
    content += '<details><summary>Exact prompts and input provenance</summary><pre>'
    content += escape(json.dumps(requests, ensure_ascii=False, indent=2)) + '</pre></details></section>'
    return content, {"example_id": key, "source_id": source["source_id"], "selection_role": role,
        "source_image_sha256": source["image_sha256"], "roi_provenance_sha256": source["roi_provenance_sha256"],
        "roi_pixels": source["roi_pixels"], "requests": requests, "assets": saved, "optional_source_asset": source_asset}


class AssetChecker(HTMLParser):
    """Reject remote/escaping references and assert all local assets exist."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.references = []

    def handle_starttag(self, tag, attrs):
        require(tag.lower() not in ("script", "iframe", "object", "embed"), "Active/embedded remote content is not permitted")
        for key, value in attrs:
            require(not key.lower().startswith("on"), "Inline event handler is not permitted")
            if key in ("src", "href"):
                target = require_relative_file(self.base, value)
                self.references.append({"reference": value, "sha256": digest(target)})


def render(run, manifest, summary_path, output, *, include_source=False):
    output, manifest, summary_path = map(lambda p: Path(p).resolve(), (output, manifest, summary_path))
    require(output.suffix.lower() == ".html", "Output must be a local HTML file")
    require(not output.is_relative_to(PROJECT), "Export local image galleries outside the publishable code repository")
    metadata, summary, sources, groups = load_completed(run, manifest, summary_path)
    changed, stable = select_cases(groups)
    declared = summary["paired_native_degraded_response_agreement"]["256"]["semantic_audit_required_example_ids"]
    require(changed == sorted(declared), "Renderer selection disagrees with strict-summary audit coverage")
    assets = output.parent / (output.stem + "_assets")
    provenance_path = output.with_suffix(".provenance.json")
    qa_path = output.with_suffix(".qa.json")
    require(not any(path.exists() for path in (output, assets, provenance_path, qa_path)), "Use fresh output paths; gallery assets are not overwritten")
    assets.mkdir(parents=True)
    cases, sections = [], []
    selected = [(key, "Primary-pair answer change") for key in changed]
    if stable is not None:
        selected.append((stable, "First unchanged primary-pair comparison"))
    for key, role in selected:
        section, case = render_case(key, role, sources[key], groups[key], metadata["config"], manifest, assets, include_source)
        sections.append(section)
        cases.append(case)
    style = ('body{font:16px/1.5 system-ui,sans-serif;margin:auto;max-width:1280px;padding:28px;background:#f4f6f8;color:#162434}'
        'h1,h2{line-height:1.2}h1{font-size:32px}h2{font-size:22px}header,.banner{padding:20px;background:#e5edf3;border:1px solid #bacbd9;border-radius:8px}'
        'section{background:white;margin:28px 0;padding:24px;border:1px solid #ced7df;border-radius:8px}.eyebrow{font-size:12px;font-weight:700;text-transform:uppercase;color:#375d75}'
        '.images{display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;align-items:start}figure{margin:0;min-width:0}img{width:100%;max-height:540px;object-fit:contain;background:#eef1f4}'
        'figcaption,.note,small{font-size:12px;color:#48596b}small{display:block;margin-top:6px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}'
        'th,td{padding:9px;border-bottom:1px solid #d7dfe5;vertical-align:top;text-align:left}th{white-space:nowrap}pre{font:13px/1.4 monospace;white-space:pre-wrap;overflow-wrap:anywhere;margin:0}'
        'details{margin-top:18px}summary{cursor:pointer;font-weight:600}.source{display:block;max-width:750px;width:100%;margin:12px auto}code{overflow-wrap:anywhere}'
        '@media(max-width:760px){body{padding:12px}.images{grid-template-columns:1fr}section{padding:16px}}')
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'; connect-src \'none\'; frame-src \'none\'; base-uri \'none\'">'
        '<title>LookAgain: is the visual evidence available?</title><style>' + style + '</style></head><body><header>'
        '<h1>Is the visual evidence available?</h1><p>LookAgain · completed 85-report development diagnostic · 850 calls · frozen Qwen3-VL-4B.</p>'
        '<p><strong>The region comes from reference-linked annotation.</strong> Its location is privileged; reference/OCR text is never supplied in the language prompt. These examples show conditional evidence availability, not a learned selector or an achieved policy.</p></header>'
        f'<p class="banner"><strong>Outcome-selected illustrations.</strong> Showing all {len(changed)} primary-pair answer changes and {int(stable is not None)} unchanged comparison. '
        'These cases are not a representative sample. The same reports have known earlier development outcomes; no held-out, transfer or controller claim is made.</p>'
        '<p>' + escape(SELECTION_RULE) + '</p><p>Every action is a fresh single-turn request. Native uses the source-rendered region; degraded reconstructs the same region from the corresponding overview. Dimensions and token grids match within each pair. '
        'Reported token counts are realized values; 256/512/1024 are overview ceilings, not guaranteed equal total computation. High resolution is the standalone full-page comparator with a 4096-token cap.</p>'
        '<p>Standalone latency includes source decoding, view construction/hashing, processing/transfers and generation/decoding. It excludes model loading, warmup, PDF rendering and privileged localization. All recorded outliers, invalid answers and truncations are retained. '
        'Official EM and conservative text EM are original automatic grades; equivalent wording can have different grades, and neither substitutes for the separate semantic audit.</p>'
        + "".join(sections) + '<footer><p>PNGs are reconstructed from frozen source pixels and match logged RGB hashes; they are not captured activations. '
        'The local gallery has no JavaScript or network dependencies. It is an inspection artifact and makes no image redistribution-rights claim.</p>'
        f'<p><a href="{escape(provenance_path.name)}">Selection, prompts and asset provenance</a></p></footer></body></html>')
    output.write_text(document, encoding="utf-8")
    provenance = {"created_utc": datetime.now(timezone.utc).isoformat(), "experiment": "evidence_availability_v5",
        "selection_rule": SELECTION_RULE, "selection_is_outcome_dependent": True, "representative_claim": False,
        "changed_example_ids": changed, "stable_example_id": stable, "cases": cases,
        "bindings": summary["bindings"], "summary_sha256": digest(summary_path),
        "renderer_sha256": digest(__file__), "gallery_html_sha256": digest(output), "include_source_illustrations": include_source,
        "semantic_review": "Separate condition-blind artifact; this gallery must be viewed only after that review is locked.",
        "limits": summary["caveats"]}
    provenance_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checker = AssetChecker(output.parent)
    checker.feed(document)
    all_assets = {name: info for case in cases for name, info in case["assets"].items()}
    for relative, info in all_assets.items():
        target = require_relative_file(output.parent, relative)
        require(digest(target) == info["file_sha256"], "Asset file changed during rendering")
        with Image.open(target) as saved:
            require(image_digest(saved) == info["rgb_sha256"] and list(saved.size) == info["size"], "Asset RGB hash or dimensions changed")
    qa = {"status": "PASS", "gallery_html_sha256": digest(output), "provenance_sha256": digest(provenance_path),
        "case_count": len(cases), "unique_png_assets": len(all_assets), "checked_local_references": len(checker.references),
        "relative_assets_exist_and_stay_within_gallery_directory": True, "all_saved_input_pixels_match_hashes": True,
        "external_network_dependencies": False, "script_tags_or_event_handlers": False,
        "browser_visual_review_performed": False, "note": "Programmatic asset/input QA only; browser visual review is separate."}
    qa_path.write_text(json.dumps(qa, indent=2) + "\n", encoding="utf-8")
    return qa


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest", "summary", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--include-source", action="store_true", help="Include full source PNGs with an HTML-only ROI overlay, clearly separate from model inputs")
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.manifest, args.summary, args.output, include_source=args.include_source), indent=2))
