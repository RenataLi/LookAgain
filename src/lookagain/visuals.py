"""Static research figures and an offline, outcome-stratified example viewer.

Figures consume measured analysis summaries; galleries embed only thumbnails
from local manifest images. Neither output implies a learned crop controller.
"""

from __future__ import annotations

import base64
import html
import io
import json
import math
from collections import defaultdict
from pathlib import Path, PureWindowsPath


ACTION_ORDER = (
    "direct", "highres", "recheck", "think", "crop_tl", "crop_tr",
    "crop_bl", "crop_br", "uniform_random_crop", "hindsight_oracle",
)
LABELS = {
    "direct": "Direct / stop", "highres": "High-res direct",
    "recheck": "Repeat image", "think": "Extra text reasoning",
    "crop_tl": "Crop TL", "crop_tr": "Crop TR",
    "crop_bl": "Crop BL", "crop_br": "Crop BR",
    "uniform_random_crop": "Random crop",
    "hindsight_oracle": "Oracle (privileged)",
}
CROP_COLORS = {
    "crop_tl": "#2563eb", "crop_tr": "#d97706",
    "crop_bl": "#0f766e", "crop_br": "#9333ea",
}
COLORS = {
    "direct": "#172b4d", "highres": "#516784", "recheck": "#ad526d",
    "think": "#677b38", **CROP_COLORS,
    "uniform_random_crop": "#78716c", "hindsight_oracle": "#111827",
}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return float(value) if math.isfinite(value) else None


def _ordered(mapping: dict) -> list[str]:
    return [name for name in ACTION_ORDER if name in mapping] + sorted(set(mapping) - set(ACTION_ORDER))


def render_figures(summary: dict, output_dir: Path) -> list[Path]:
    """Write one two-panel measured-results figure as SVG and PNG.

    Accuracy uses a full 0–100% axis. Gain/harm bars share the same complete
    example denominator. Oracle cost charges only its selected action and is
    explicitly labeled as a privileged bound, not achievable policy latency.
    """
    import matplotlib as mpl
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = {**summary.get("actions", {}), **summary.get("baselines", {})}
    names = _ordered(methods)
    for name in names:
        row = methods[name]
        values = [row.get("accuracy"), row.get("total_latency_s", {}).get("mean"),
                  row.get("wrong_to_right_fraction"), row.get("right_to_wrong_fraction")]
        if any(_finite(value) is None or value < 0 for value in values):
            raise ValueError(f"Invalid measured figure values for {name}")
        if any(values[index] > 1 for index in (0, 2, 3)):
            raise ValueError(f"Accuracy/transition fractions outside [0, 1] for {name}")
    coverage = summary.get("coverage", {})
    n = coverage.get("complete_examples", 0)
    images = coverage.get("complete_images", 0)
    excluded = coverage.get("excluded_examples", 0)

    with mpl.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.labelcolor": "#334155", "text.color": "#172b4d",
        "xtick.color": "#475569", "ytick.color": "#475569",
        "axes.edgecolor": "#a8b3c2", "svg.fonttype": "none",
    }):
        figure = Figure(figsize=(14, 8.2), facecolor="white")
        FigureCanvasAgg(figure)
        left, right = figure.subplots(1, 2, gridspec_kw={"width_ratios": [1.08, 1]})
        figure.subplots_adjust(left=0.065, right=0.975, top=0.77, bottom=0.33, wspace=0.47)
        figure.text(0.065, 0.95, "LookAgain  |  Development diagnostic", fontsize=19, weight="bold", va="top")
        figure.text(0.065, 0.89,
                    f"{n} complete examples · {images} source images · {excluded} excluded examples\n"
                    "Frozen VLM · fixed crop grid · no learned selector", fontsize=10.5, color="#475569", va="top")
        left.set_title("Answer accuracy versus measured total latency", loc="left", pad=13, fontsize=12)
        left.set_xlabel("Mean total latency per question (seconds)")
        left.set_ylabel("Pilot normalized exact match (%)")
        left.set_ylim(0, 100)
        left.set_yticks([0, 20, 40, 60, 80, 100])
        left.grid(axis="both", color="#e2e8f0", linewidth=0.7)
        left.set_axisbelow(True)
        right.set_title("Repairs and damage relative to the initial answer", loc="left", pad=13, fontsize=11)
        right.set_xlabel("Fraction of the same complete examples (%)")
        right.grid(axis="x", color="#e2e8f0", linewidth=0.7)
        right.set_axisbelow(True)

        if names:
            max_latency = max(methods[name]["total_latency_s"]["mean"] for name in names)
            left.set_xlim(0, max(1.0, max_latency * 1.12))
            for name in names:
                row = methods[name]
                oracle = name == "hindsight_oracle"
                left.scatter(
                    row["total_latency_s"]["mean"], 100 * row["accuracy"],
                    s=90 if oracle else 56, marker="D" if oracle else "o",
                    facecolors="none" if oracle else COLORS.get(name, "#64748b"),
                    edgecolors=COLORS.get(name, "#64748b"), linewidths=1.8 if oracle else 0.8,
                    label=LABELS.get(name, name), zorder=3, clip_on=False,
                )
            left.legend(loc="upper left", bbox_to_anchor=(0, -0.19), fontsize=8,
                        ncol=2, frameon=False, labelspacing=0.7, borderaxespad=0)
            positions = list(range(len(names)))
            gains = [100 * methods[name]["wrong_to_right_fraction"] for name in names]
            harms = [100 * methods[name]["right_to_wrong_fraction"] for name in names]
            right.barh([value - 0.18 for value in positions], gains, height=0.32,
                       color="#0f766e", label="Wrong → right")
            right.barh([value + 0.18 for value in positions], harms, height=0.32,
                       color="#c2414b", label="Right → wrong")
            max_rate = max(gains + harms)
            right.set_xlim(0, min(115, max(5.0, max_rate * 1.18)))
            right.set_yticks(positions, [LABELS.get(name, name) for name in names])
            right.invert_yaxis()
            offset = max(0.08, max_rate * 0.018)
            for index, name in enumerate(names):
                for rate, count, shift, color in (
                    (gains[index], methods[name]["wrong_to_right_count"], -0.18, "#0f766e"),
                    (harms[index], methods[name]["right_to_wrong_count"], 0.18, "#b03942"),
                ):
                    if count:
                        right.text(rate + offset, index + shift, f"{count}", va="center", fontsize=8, color=color)
            right.legend(loc="upper left", bbox_to_anchor=(0, -0.11), ncol=2,
                         frameon=False, fontsize=9)
        else:
            left.set_xlim(0, 1)
            right.set_xlim(0, 1)
            right.set_yticks([])
            for axes in (left, right):
                axes.text(0.5, 0.5, "No complete paired examples", ha="center", va="center", transform=axes.transAxes)

        figure.text(0.065, 0.025,
                    "The hollow oracle is a privileged upper bound using reference-answer correctness; it is not a deployable policy.\n"
                    "Its latency charges only the selected action, not outcome search. High-res is standalone; follow-ups include the initial call.\n"
                    "Bars show transition rates; labels show counts. Complete-case coverage and sampling limitations apply.",
                    fontsize=9, linespacing=1.6, color="#475569", va="bottom")
        paths = [output_dir / "accuracy_cost_and_transitions.svg", output_dir / "accuracy_cost_and_transitions.png"]
        figure.savefig(paths[0], format="svg", facecolor="white")
        figure.savefig(paths[1], format="png", dpi=180, facecolor="white")
        figure.clear()
    return paths


def _escape(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _seconds(value: object) -> str:
    number = _finite(value)
    return f"{number:.3f}" if number is not None and number >= 0 else "—"


def _read_manifest(path: Path) -> list[dict]:
    rows = []
    seen = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or not isinstance(row.get("example_id"), str):
            raise ValueError(f"Invalid gallery manifest row {number}")
        if row["example_id"] in seen:
            raise ValueError(f"Duplicate manifest example: {row['example_id']}")
        seen.add(row["example_id"])
        rows.append(row)
    return rows


def _thumbnail(row: dict, directory: Path) -> tuple[str, int, int]:
    from PIL import Image

    relative = Path(row["image_path"])
    if relative.is_absolute() or PureWindowsPath(str(relative)).drive:
        raise ValueError("Gallery images must be relative to the manifest")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory):
        raise ValueError("Gallery image escapes the manifest directory")
    with Image.open(path) as original:
        picture = original.convert("RGB")
        # Match the runner's image orientation; no implicit EXIF transform.
        picture.thumbnail((720, 520), Image.Resampling.LANCZOS)
        stream = io.BytesIO()
        picture.save(stream, format="JPEG", quality=78, optimize=True)
        return base64.b64encode(stream.getvalue()).decode("ascii"), picture.width, picture.height


def _category(rows: list[dict]) -> str:
    direct = [row for row in rows if row.get("action") == "direct" and row.get("status") == "ok"]
    if len(direct) != 1 or not isinstance(direct[0].get("correct"), bool):
        return "Other / incomplete"
    baseline = direct[0]["correct"]
    valid = [row for row in rows if row.get("status") == "ok" and isinstance(row.get("correct"), bool)]
    crop_rows = [row for row in valid if str(row.get("action", "")).startswith("crop_")]
    if len(valid) == len(rows) and all(row["correct"] for row in valid):
        return "All recorded answers correct"
    if not baseline and any(row["correct"] for row in crop_rows):
        return "A fixed crop repairs the answer"
    if baseline and any(not row["correct"] for row in crop_rows):
        return "A fixed crop damages the answer"
    return "Other / incomplete"


def _image_svg(encoded: str, width: int, height: int, rows: list[dict]) -> str:
    from .actions import CROP_BOXES

    boxes = dict(CROP_BOXES)
    for row in rows:
        action = row.get("action")
        box = row.get("crop_box_normalized")
        if action in boxes and isinstance(box, (list, tuple)) and len(box) == 4:
            numbers = [_finite(value) for value in box]
            if all(value is not None and 0 <= value <= 1 for value in numbers):
                if numbers[0] < numbers[2] and numbers[1] < numbers[3]:
                    boxes[action] = tuple(numbers)
    overlays = []
    for action, (x0, y0, x1, y1) in boxes.items():
        color = CROP_COLORS[action]
        x, y = x0 * width, y0 * height
        box_width, box_height = (x1 - x0) * width, (y1 - y0) * height
        overlays.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{box_width:.2f}" height="{box_height:.2f}" '
            f'fill="none" stroke="{color}" stroke-width="3" vector-effect="non-scaling-stroke" />'
        )
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="Source image with four fixed crop rectangles" '
        f'xmlns="http://www.w3.org/2000/svg">'
        f'<image width="{width}" height="{height}" href="data:image/jpeg;base64,{encoded}" />'
        + "".join(overlays) + "</svg>"
    )


def render_gallery(records: list[dict], manifest: Path, output: Path, max_examples: int = 12) -> Path:
    """Write a self-contained local HTML gallery with escaped data and thumbnails.

    Examples are picked by round-robin across declared outcome strata, sorted
    by example ID within each stratum. This is illustrative selection, not an
    unbiased sample or evidence of a learned policy. No network assets or scripts
    are included. The caller chooses the output location and sharing policy.
    """
    if isinstance(max_examples, bool) or not isinstance(max_examples, int) or max_examples < 1:
        raise ValueError("max_examples must be a positive integer")
    manifest = Path(manifest).resolve()
    output = Path(output)
    manifest_rows = _read_manifest(manifest)
    by_id = {row["example_id"]: row for row in manifest_rows}
    grouped = defaultdict(list)
    for row in records:
        if not isinstance(row, dict) or not isinstance(row.get("example_id"), str):
            raise ValueError("Every gallery record requires a string example_id")
        grouped[row["example_id"]].append(row)
    categories = (
        "All recorded answers correct", "A fixed crop repairs the answer",
        "A fixed crop damages the answer", "Other / incomplete",
    )
    buckets = {category: [] for category in categories}
    for example_id in sorted(set(grouped) & set(by_id)):
        buckets[_category(grouped[example_id])].append(example_id)
    chosen = []
    position = 0
    while len(chosen) < max_examples:
        added = False
        for category in categories:
            if position < len(buckets[category]) and len(chosen) < max_examples:
                chosen.append((category, buckets[category][position]))
                added = True
        if not added:
            break
        position += 1

    cards = []
    for category, example_id in chosen:
        metadata = by_id[example_id]
        rows = sorted(grouped[example_id], key=lambda row: (
            ACTION_ORDER.index(row.get("action")) if row.get("action") in ACTION_ORDER else len(ACTION_ORDER),
            str(row.get("action", "")),
        ))
        direct_rows = [row for row in rows if row.get("action") == "direct" and row.get("status") == "ok"]
        baseline = direct_rows[0] if len(direct_rows) == 1 else None
        direct_time = _finite(baseline.get("elapsed_s")) if baseline else None
        encoded, width, height = _thumbnail(metadata, manifest.parent)
        action_rows = []
        for row in rows:
            action = str(row.get("action", "unknown"))
            standalone = action in ("direct", "highres")
            elapsed = _finite(row.get("elapsed_s"))
            total = elapsed if standalone else (
                elapsed + direct_time if elapsed is not None and direct_time is not None else None
            )
            valid = row.get("status") == "ok" and isinstance(row.get("correct"), bool)
            if not valid:
                result, result_class = "Unavailable", "unknown"
            elif row["correct"]:
                result, result_class = "Correct", "correct"
                if baseline and not baseline.get("correct") and not standalone:
                    result = "Correct · repair"
            else:
                result, result_class = "Incorrect", "incorrect"
                if baseline and baseline.get("correct") and not standalone:
                    result = "Incorrect · harm"
            predicted = row.get("predicted_answer")
            if predicted is None:
                predicted = "[no parsed answer]"
            target = row.get("target_answer", metadata.get("answer"))
            incremental = "—" if standalone else _seconds(elapsed)
            color = COLORS.get(action, "#64748b")
            raw_details = ""
            if row.get("response"):
                raw_details = f'<details><summary>Raw output</summary><pre>{_escape(row["response"])}</pre></details>'
            action_rows.append(
                f'<tr><th scope="row"><span class="dot" style="background:{color}"></span>{_escape(LABELS.get(action, action))}</th>'
                f'<td>{_escape(predicted)}{raw_details}</td><td>{_escape(target)}</td>'
                f'<td class="{result_class}">{result}</td><td class="number">{incremental}</td>'
                f'<td class="number">{_seconds(total)}</td></tr>'
            )
        target = metadata.get("answer", rows[0].get("target_answer", ""))
        image_id = metadata.get("image_id", rows[0].get("image_id", ""))
        cards.append(
            '<article class="card">'
            f'<div class="card-heading"><span class="tag">{_escape(category)}</span>'
            f'<span class="identity">{_escape(example_id)} · image {_escape(image_id)}</span></div>'
            f'<h2>{_escape(metadata.get("question", rows[0].get("question", "")))}</h2>'
            f'<p class="reference">Reference answer: <strong>{_escape(target)}</strong></p>'
            '<div class="example-layout"><figure>' + _image_svg(encoded, width, height, rows)
            + '<figcaption>Fixed candidate regions: <span class="tl">TL</span> · <span class="tr">TR</span> · '
            '<span class="bl">BL</span> · <span class="br">BR</span>. Rectangles show candidate views, not learned localization.</figcaption>'
            '</figure><div class="table-scroll"><table><thead><tr><th>Action</th><th>Prediction</th><th>Reference</th>'
            '<th>Result</th><th>Extra s</th><th>Total s</th></tr></thead><tbody>'
            + "".join(action_rows) + '</tbody></table></div></div></article>'
        )
    missing_manifest = len(set(grouped) - set(by_id))
    content = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>LookAgain — development diagnostic gallery</title>
<style>
:root{{color-scheme:light;font-family:system-ui,-apple-system,"Segoe UI",sans-serif;color:#172b4d;background:#f3f6fa}}
*{{box-sizing:border-box}}body{{margin:0}}main{{max-width:1440px;margin:auto;padding:40px 28px 60px}}
header{{max-width:1100px;margin-bottom:30px}}.eyebrow{{text-transform:uppercase;letter-spacing:.13em;font-size:12px;font-weight:700;color:#0f766e}}
h1{{font-size:clamp(28px,4vw,42px);letter-spacing:-.035em;margin:10px 0 14px}}h2{{font-size:21px;line-height:1.4;margin:14px 0 8px}}
p{{line-height:1.65}}.lead{{font-size:17px}}.note{{font-size:13px;color:#52637a}}.card{{background:#fff;border:1px solid #dce4ee;border-radius:15px;padding:24px;margin:24px 0;box-shadow:0 2px 5px #13294206}}
.card-heading{{display:flex;gap:12px;align-items:center;flex-wrap:wrap}}.tag{{font-size:12px;font-weight:650;background:#eaf2f7;padding:6px 9px;border-radius:6px}}
.identity{{color:#627187;font-size:12px;overflow-wrap:anywhere}}.reference{{font-size:14px;margin-top:0}}.example-layout{{display:grid;grid-template-columns:minmax(260px,.8fr) minmax(550px,1.4fr);gap:26px;align-items:start}}
figure{{margin:0}}svg{{display:block;width:100%;height:auto;max-height:520px;background:#f4f6f9}}figcaption{{font-size:12px;line-height:1.6;color:#52637a;margin-top:9px}}
.tl{{color:#2563eb}}.tr{{color:#b86b04}}.bl{{color:#0f766e}}.br{{color:#9333ea}}.table-scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:12px;line-height:1.4}}
th,td{{text-align:left;padding:10px 8px;border-bottom:1px solid #e5ebf2;vertical-align:top}}thead th{{font-size:11px;color:#52637a;background:#f6f8fb}}tbody th{{font-weight:600;min-width:115px}}td{{overflow-wrap:anywhere}}
.number{{font-variant-numeric:tabular-nums;white-space:nowrap}}.correct{{color:#0f766e}}.incorrect{{color:#b03942}}.unknown{{color:#64748b}}.dot{{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:6px}}
details{{margin-top:6px}}summary{{cursor:pointer;color:#617089;font-size:11px}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit;font-size:11px;max-width:360px;background:#f6f8fb;padding:8px}}
footer{{font-size:12px;color:#617089;border-top:1px solid #dce4ee;margin-top:32px;padding-top:15px;line-height:1.7}}
@media(max-width:1050px){{.example-layout{{grid-template-columns:1fr}}figure{{max-width:720px}}main{{padding:22px 14px}}.card{{padding:18px}}}}
@media print{{body{{background:white}}main{{padding:0}}.card{{break-inside:avoid;box-shadow:none}}details{{display:none}}}}
</style></head><body><main><header><div class="eyebrow">LookAgain · local research viewer</div>
<h1>When does another view change the answer?</h1>
<p class="lead">Development diagnostics from a frozen vision-language model. Four fixed overlapping crops are compared with the initial answer, a repeated image, additional text reasoning, and a standalone higher-resolution answer. <strong>No learned controller is shown.</strong></p>
<p class="note">Selection rule: round-robin through all-correct, crop-repair, crop-harm, and other/incomplete examples; sorted example IDs within each group. This outcome-stratified gallery is illustrative and is not a representative performance sample. Showing {len(chosen)} of {len(set(grouped) & set(by_id))} recorded examples matched to the manifest; {missing_manifest} unmatched record groups omitted.</p>
<p class="note">“Extra s” is the additional action time; “Total s” includes the initial call for follow-ups. Direct and high-res are standalone. Correctness is the recorded pilot answer score. A successful crop here was selected for inspection after observing outcomes, not chosen by a learned policy.</p>
<p class="note">The available GQA source images can be small. Fixed-budget bicubic resizing may upsample both the overview and crops; it creates no new source detail. Answer changes can reflect re-encoding, attention, or representation effects and do not by themselves demonstrate recovery of previously unavailable visual information.</p>
</header>{''.join(cards) if cards else '<p>No recorded examples matched the manifest.</p>'}
<footer>Local, self-contained diagnostic export. Thumbnails are embedded solely to inspect the locally available examples; this file makes no claim of public redistribution rights for source images. Review dataset terms before sharing it. No remote assets or scripts are loaded.</footer>
</main></body></html>'''
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(content, encoding="utf-8")
    return output
