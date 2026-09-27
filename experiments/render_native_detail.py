"""Completed-run figure and local pixel gallery; no inference or metric tuning."""
from __future__ import annotations
import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from native_detail import validate_manifest
from native_detail_core import ACTIONS, REGIONS, build_request


def png_uri(image):
    stream = io.BytesIO()
    image.save(stream, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(stream.getvalue()).decode()


def render(run, manifest, figure, gallery):
    if not (run / "completed.json").exists() or not (run / "summary.json").exists():
        raise ValueError("A completed, analyzed run is required")
    metadata = json.loads((run / "run.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    records = [json.loads(line) for line in (run / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    sources = validate_manifest(manifest)
    groups = {row["example_id"]: {} for row in sources}
    for row in records:
        groups[row["example_id"]][row["action"]] = row
    if len(records) != len(sources) * len(ACTIONS) or any(set(group) != set(ACTIONS) for group in groups.values()):
        raise ValueError("Incomplete panel")
    names = ["direct", "highres", "repeat", "native", "degraded"]
    labels = ["Direct", "High-res\nfull page", "Repeat", "Source\nregion", "Overview\nregion"]
    scores, official, times = [], [], []
    for name in names:
        actions = [name] if name in ACTIONS else [f"{name}_{region}" for region in REGIONS]
        scores.append(np.mean([[group[a]["conservative_text_em"] for a in actions] for group in groups.values()]) * 100)
        official.append(np.mean([[group[a]["official_em"] for a in actions] for group in groups.values()]) * 100)
        times.append(np.mean([[group[a]["elapsed_s"] + (group["direct"]["elapsed_s"] if a not in ("direct", "highres") else 0) for a in actions] for group in groups.values()]))
    native = np.array([np.mean([g[f"native_{r}"]["conservative_text_em"] for r in REGIONS]) for g in groups.values()])
    degraded = np.array([np.mean([g[f"degraded_{r}"]["conservative_text_em"] for r in REGIONS]) for g in groups.values()])
    delta = native - degraded
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.5), gridspec_kw={"width_ratios": [1.35, 1, 1]})
    x = np.arange(len(names))
    axes[0].bar(x-.18, scores, width=.36, color="#176b76", label="Conservative EM (primary)")
    axes[0].bar(x+.18, official, width=.36, color="#95c9ca", label="Official-normalized EM")
    axes[0].set_xticks(x, labels)
    axes[0].set_ylabel("Exact match (%)")
    axes[0].set_ylim(0, 100)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[0].set_title("Quality: all fixed conditions", loc="left", fontweight="bold")
    for ax in axes[:2]:
        ax.grid(axis="y", alpha=.18)
        ax.set_axisbelow(True)
    axes[1].bar(x, times, color=["#385474", "#97763a", "#7e8791", "#176b76", "#95c9ca"])
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("Mean total policy latency (s)")
    axes[1].set_title("Cost: includes initial answer", loc="left", fontweight="bold")
    for i, value in enumerate(times):
        axes[1].text(i, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    values, counts = np.unique(delta * 100, return_counts=True)
    axes[2].bar(values, counts, width=16, color=["#b56d59" if v<0 else "#176b76" if v>0 else "#a7afb9" for v in values])
    axes[2].axvline(0, color="#495160", linewidth=.8)
    axes[2].set_xticks([-100, -50, 0, 50, 100])
    axes[2].set_xlabel("Source minus overview region (pp)\nMean of four regions per report")
    axes[2].set_ylabel("Number of source reports")
    axes[2].set_title("Paired fidelity effect", loc="left", fontweight="bold")
    fig.suptitle(f"LookAgain v3 · {len(sources)} financial-document pages · frozen Qwen3-VL-4B", x=.04, ha="left", fontweight="bold", fontsize=15)
    fig.text(.04,.015,"Development panel; region averages are a uniform single-region expectation. No learned selector. PDF render: 200 DPI.",fontsize=9,color="#515b65")
    fig.tight_layout(rect=[.01,.065,1,.92])
    figure.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(figure,dpi=190)
    fig.savefig(figure.with_suffix('.svg'))
    plt.close(fig)

    # Outcome-selected illustrations, with all branches shown and selection disclosed.
    strata = {"Source-region advantage": [], "Overview-region advantage": [], "Tied, highres repair": [], "Tied other": []}
    by_id = {row["example_id"]: row for row in sources}
    for example_id in sorted(groups):
        group = groups[example_id]
        a = sum(group[f"native_{r}"]["correct"] for r in REGIONS)
        b = sum(group[f"degraded_{r}"]["correct"] for r in REGIONS)
        category = "Source-region advantage" if a>b else "Overview-region advantage" if a<b else "Tied, highres repair" if group["highres"]["correct"] and not group["direct"]["correct"] else "Tied other"
        strata[category].append(example_id)
    chosen = []
    for index in range(3):
        for category, identifiers in strata.items():
            if index < len(identifiers):
                chosen.append((category, identifiers[index]))
    esc = lambda value: html.escape(str(value), quote=True)
    sections = []
    for category, example_id in chosen:
        row, group = by_id[example_id], groups[example_id]
        with Image.open(manifest.parent / row["image_path"]) as opened:
            source = opened.convert("RGB")
        _, direct_images, _ = build_request(source,row["question"],"direct",config)
        table = []
        for action in ACTIONS:
            result = group[action]
            cost = result["elapsed_s"] + (group["direct"]["elapsed_s"] if action not in ("direct","highres") else 0)
            table.append(f'<tr><td>{esc(action)}</td><td>{esc(result["predicted_answer"])}</td><td>{int(result["conservative_text_em"])}</td><td>{result["official_em"]:.0f}</td><td>{cost:.3f}s</td></tr>')
        pairs = []
        for region in REGIONS:
            frames = []
            for condition in ("native","degraded"):
                action = f"{condition}_{region}"
                _, images, geometry = build_request(source,row["question"],action,config,group["direct"]["response"])
                if geometry["image_rgb_sha256"] != group[action]["image_rgb_sha256"]:
                    raise ValueError("Gallery pixel reproduction mismatch")
                frames.append(f'<figure><img src="{png_uri(images[1])}" alt="{esc(action)} input"><figcaption>{esc(action)} · {images[1].width}×{images[1].height}</figcaption></figure>')
            pairs.append(f'<details><summary>{esc(region.upper())}: compare exact second-view pixels</summary><div class="pair">{"".join(frames)}</div></details>')
        sections.append(f'<article><p class="tag">{esc(category)}</p><h2>{esc(row["question"])}</h2><p>Reference: <strong>{esc(row["answer"])}</strong> · {esc(row["source_id"])}<br><small>{esc(example_id)}</small></p><div class="case"><figure><img src="{png_uri(direct_images[0])}" alt="Exact direct overview input"><figcaption>Exact overview pixels, displayed smaller. Open an image to inspect full size.</figcaption></figure><div><table><thead><tr><th>Action</th><th>Answer</th><th>Primary EM</th><th>Official EM</th><th>Total cost</th></tr></thead><tbody>{"".join(table)}</tbody></table></div></div>{"".join(pairs)}</article>')
    payload = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LookAgain v3 · Pixel intervention gallery</title><style>
    body{{font:16px/1.5 system-ui,sans-serif;margin:0;background:#f2f5f7;color:#20313c}} main{{max-width:1320px;margin:auto;padding:30px}}h1{{font-size:34px;line-height:1.15}}h2{{font-size:22px}}article{{background:white;padding:26px;border-radius:14px;margin:26px 0;border:1px solid #dce3e7}}.tag{{color:#176b76;font-weight:700}}.case{{display:grid;grid-template-columns:1fr 1.8fr;gap:24px}}figure{{margin:0}}img{{max-width:100%;height:auto;border:1px solid #dce3e7}}figcaption,small{{color:#60717e;font-size:12px}}table{{width:100%;border-collapse:collapse;font-size:13px}}td,th{{padding:9px;text-align:left;border-bottom:1px solid #e4e9ed;overflow-wrap:anywhere}}th{{background:#edf4f5}}.pair{{display:grid;grid-template-columns:1fr 1fr;gap:15px;margin-top:15px}}details{{margin:12px 0;padding:12px;border:1px solid #dce3e7}}summary{{cursor:pointer;font-weight:600}}.note{{background:#e5f0f1;padding:18px;border-radius:10px}}@media(max-width:800px){{.case,.pair{{grid-template-columns:1fr}}main{{padding:16px}}}}
    </style><main><h1>LookAgain · Does another view add detail?</h1><p>Local inspection gallery for the completed 100-report development panel.</p><div class="note">These examples are selected <strong>after observing outcomes</strong>: up to three examples per disclosed stratum, sorted by question ID. They illustrate successes and failures; they do not estimate performance. Every listed example shows all eleven actions. Region crops use fixed locations, not a trained controller.</div><p>Source PDFs: author-released TAT-DQA training data, rendered at 200 DPI. The source-region and overview-region inputs share prompts, geometry and output size; only their pixel derivation differs. Conservative EM is the primary score. Official EM additionally normalizes punctuation, articles and numbers. Latency includes the direct answer for follow-ups.</p><p>Stratum counts: {esc(json.dumps({k:len(v) for k,v in strata.items()}))}. Run fingerprint: <code>{esc(metadata["fingerprint"])}</code>.</p>{"".join(sections)}<p>Images are embedded for local research inspection. Source documents/images are not included in the source-code release archive. Dataset: <a href="https://nextplusplus.github.io/TAT-DQA/">TAT-DQA authors</a>.</p></main></html>'''
    gallery.parent.mkdir(parents=True,exist_ok=True)
    gallery.write_text(payload,encoding="utf-8")
    (gallery.parent / "gallery_manifest.json").write_text(json.dumps({"run_fingerprint":metadata["fingerprint"],"selected":chosen,"stratum_counts":{k:len(v) for k,v in strata.items()},"selection":"first three sorted question IDs per outcome stratum; post hoc illustration only"},indent=2),encoding="utf-8")
    print(json.dumps({"figure":str(figure),"gallery":str(gallery),"examples":len(chosen)}))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--figure",type=Path,required=True)
    parser.add_argument("--gallery",type=Path,required=True)
    args=parser.parse_args()
    render(args.run.resolve(),args.manifest.resolve(),args.figure.resolve(),args.gallery.resolve())
