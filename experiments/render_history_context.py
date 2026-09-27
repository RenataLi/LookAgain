"""Offline v4 case gallery and scientific figures, requiring completed strict analysis."""
from __future__ import annotations

import argparse
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path
import sys

from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "experiments"))
from history_context_core import ACTIONS, HISTORIES, FIDELITIES, REGIONS, build_request
from native_detail_core import image_digest

METRIC = "conservative_text_em"
CATEGORIES = (
    "Native-specific rescue after new history", "Degraded-only rescue",
    "Joint rescue after new history", "New-history harm in both fidelities",
    "History changes both fidelity answers equally", "Stable correct answer",
    "Error persists in every condition", "High-resolution-only rescue",
)
RULES = (
    "Direct and actual native/degraded wrong; fresh or placeholder native correct and matched degraded wrong.",
    "Direct wrong; at least one history has degraded correct and matched native wrong.",
    "Direct and actual native/degraded wrong; fresh or placeholder native and degraded both correct.",
    "Direct and actual native/degraded correct; fresh or placeholder native and degraded both wrong.",
    "Valid native/degraded predictions match within actual and within a new history, but differ between these histories.",
    "Direct and all 24 crop conditions correct; the selected region is the first fixed quadrant.",
    "Direct, high-resolution, repeat and all 24 crop conditions wrong.",
    "High-resolution correct; direct, repeat and all 24 crop conditions wrong.",
)


def require(test, message):
    if not test:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def escape(value):
    return html.escape(str(value), quote=True)


def category_matches(group, region):
    def em(history, fidelity):
        return bool(group[f"{history}_{fidelity}_{region}"][METRIC])
    def equal_pair(history):
        rows = [group[f"{history}_{f}_{region}"] for f in FIDELITIES]
        values = [r.get("normalized_prediction") for r in rows]
        return values[0] if all(r["parse_valid"] for r in rows) and values[0] == values[1] else None
    direct = bool(group["direct"][METRIC])
    actual = [em("actual", f) for f in FIDELITIES]
    new = [[em(h, f) for f in FIDELITIES] for h in HISTORIES[1:]]
    crops = [bool(group[a][METRIC]) for a in ACTIONS[3:]]
    anchor = equal_pair("actual")
    return (
        not direct and not any(actual) and any(n and not d for n, d in new),
        not direct and any(not em(h, "native") and em(h, "degraded") for h in HISTORIES),
        not direct and not any(actual) and any(all(pair) for pair in new),
        direct and all(actual) and any(not any(pair) for pair in new),
        anchor is not None and any(equal_pair(h) is not None and equal_pair(h) != anchor for h in HISTORIES[1:]),
        direct and all(crops),
        not any(bool(group[a][METRIC]) for a in ACTIONS),
        bool(group["highres"][METRIC]) and not any(bool(group[a][METRIC]) for a in ACTIONS if a != "highres"),
    )


def select_cases(groups):
    """Fixed category order, lexicographic IDs, fixed quadrant order; no duplicate pages."""
    selected, coverage, used = [], [], set()
    for index, (category, rule) in enumerate(zip(CATEGORIES, RULES)):
        candidates = [(key, region) for key in sorted(groups) for region in REGIONS
                      if category_matches(groups[key], region)[index]]
        choice = next(((key, region) for key, region in candidates if key not in used), None)
        coverage.append({"category": category, "rule": rule, "matching_page_region_pairs": len(candidates),
                         "matching_pages": len({key for key, _ in candidates}), "selected": choice,
                         "skip_reason": None if choice else ("not observed" if not candidates else "all matching pages already selected")})
        if choice:
            used.add(choice[0])
            selected.append((category, *choice))
    return selected, coverage


def load_completed(run, manifest, summary_path, mask_path):
    # Gate before even reading raw answers; this renderer must never inspect interim outcomes.
    completion, summary = read_json(run / "completed.json"), read_json(summary_path)
    require(completion.get("records") == 2700, "Need completed 100 x 27 main calls")
    require(summary.get("status") == "complete" and summary.get("integrity", {}).get("passed") is True,
            "Need a successful strict analysis")
    metadata = read_json(run / "run.json")
    require(metadata.get("role") == "main", "Gallery requires the main development cohort")
    for key, path in (("run_json_sha256", run / "run.json"), ("records_sha256", run / "records.jsonl"),
                      ("completion_sha256", run / "completed.json"), ("manifest_sha256", manifest),
                      ("source_quality_mask_sha256", mask_path)):
        require(summary["bindings"][key] == digest(path), f"Summary binding mismatch: {key}")
    require(completion["records_sha256"] == summary["bindings"]["records_sha256"], "Completion hash mismatch")
    require(metadata["fingerprint"] == summary["bindings"]["run_fingerprint"], "Run fingerprint mismatch")
    sources = {row["example_id"]: row for row in read_jsonl(manifest)}
    require(len(sources) == 100 and len({r["source_id"] for r in sources.values()}) == 100, "Need 100 unique reports")
    require(set(sources) == set(metadata["example_ids"]), "Planned IDs differ")
    groups = {key: {} for key in sources}
    for row in read_jsonl(run / "records.jsonl"):
        key, action = row["example_id"], row["action"]
        require(key in groups and action in ACTIONS and action not in groups[key] and row["status"] == "ok", "Unexpected/duplicate/failed row")
        for field, source_field in (("question", "question"), ("target_answer", "answer"), ("source_id", "source_id")):
            require(row[field] == sources[key][source_field], f"Source identity mismatch: {key}/{field}")
        groups[key][action] = row
    require(all(set(g) == set(ACTIONS) for g in groups.values()), "Incomplete paired panel")
    return metadata, summary, sources, groups, read_json(mask_path)


def plot_summary(summary, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    paths = [output / f"history_context_{name}.{ext}" for name in ("quality", "cost") for ext in ("png", "svg")]
    require(not any(path.exists() for path in paths), "Refusing to overwrite scientific figures")
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    for offset, fidelity, color in ((-.18, "native", "#237a79"), (.18, "degraded", "#b2762e")):
        axes[0].bar([i + offset for i in range(3)], [100 * summary["conditions"][f"{h}_{fidelity}"]["metrics"][METRIC] for h in HISTORIES],
                    width=.34, color=color, label=fidelity.title())
    for action, style in (("direct", "--"), ("highres", ":")):
        axes[0].axhline(100 * summary["conditions"][action]["metrics"][METRIC], color="#555555", linestyle=style, label=action)
    axes[0].set(xticks=range(3), xticklabels=HISTORIES, ylabel="Conservative text EM (%)", ylim=(0, 100), title="A  Four-region means; 100 reports")
    axes[0].legend(ncol=2, fontsize=8)
    stats = [summary["detail_effects"][h][METRIC] for h in HISTORIES]
    stats.append(summary["contrasts"][summary["primary_contrast"]]["metrics"][METRIC])
    for i, stat in enumerate(stats):
        axes[1].plot(stat["ci95_pp"], [i, i], color="#237a79", linewidth=2)
        axes[1].plot(stat["difference_pp"], i, "o", color="#237a79")
    axes[1].axvline(0, color="#777777", linewidth=.8)
    axes[1].set(yticks=range(4), yticklabels=[*HISTORIES, "placeholder − actual\n(primary interaction)"], xlabel="Native − degraded effect / interaction (pp)", title="B  Paired bootstrap, exploratory 95% CI")
    axes[1].invert_yaxis()
    fig.suptitle("Development diagnostic; no held-out or mechanism claim", fontsize=12)
    for ext in ("png", "svg"):
        fig.savefig(output / f"history_context_quality.{ext}", dpi=180)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(10, 5.8), layout="constrained")
    names = list(summary["conditions"])
    for offset, mode, color in ((-.18, "decision_state", "#237a79"), (.18, "standalone", "#b2762e")):
        values = [summary["conditions"][name][mode + "_latency_s"]["mean"] for name in names]
        bars = axis.barh([i + offset for i in range(len(names))], values, height=.34,
                         color=color, label=mode.replace("_", " ").title())
        axis.bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axis.set(yticks=range(len(names)), yticklabels=[name.replace("_", " ") for name in names],
             xlabel="Mean measured latency (s); observed calls, no exclusions",
             title="Two cost perspectives; region means average one fixed-region invocation")
    axis.set_ylim(len(names) - .5, -1.2)
    axis.margins(x=.18)
    axis.legend(loc="upper right", ncol=2, frameon=False)
    axis.grid(axis="x", alpha=.2)
    for ext in ("png", "svg"):
        fig.savefig(output / f"history_context_cost.{ext}", dpi=180)
    plt.close(fig)
    return [{"file": path.name, "sha256": digest(path)} for path in paths]


def render_case(category, key, region, source, group, config, mask, manifest, assets):
    path = (manifest.parent / source["image_path"]).resolve()
    require(path.is_relative_to(manifest.parent.resolve()) and digest(path) == source["image_sha256"], "Source path/hash mismatch")
    with Image.open(path) as opened:
        page = ImageOps.exif_transpose(opened).convert("RGB")
    actions = ["direct", "highres", "repeat", *(f"{h}_{f}_{region}" for h in HISTORIES for f in FIDELITIES)]
    audit, saved, chats, rows = {}, {}, {}, []
    for action in actions:
        row = group[action]
        messages, images, geometry = build_request(page, source["question"], action, config, group["direct"]["response"])
        require(all(geometry[k] == row[k] for k in ("image_rgb_sha256", "messages_sha256", "initial_answer_sha256")), "Reconstructed inputs differ from logged inputs")
        names = []
        for image, rgb_hash in zip(images, geometry["image_rgb_sha256"]):
            filename = rgb_hash + ".png"
            target = assets / filename
            if not target.exists():
                image.save(target, format="PNG")
            with Image.open(target) as check:
                require(image_digest(check) == rgb_hash, "Saved PNG pixels differ")
            names.append("assets/" + filename)
            saved[names[-1]] = {"rgb_sha256": rgb_hash, "file_sha256": digest(target), "size": list(image.size)}
        audit[action] = {"input_files": names, "messages_sha256": geometry["messages_sha256"], "source_roi_pixels": geometry["source_roi_pixels"]}
        chats[action] = messages
        added = action not in ("direct", "highres")
        decision = row["elapsed_s"] + (group["direct"]["elapsed_s"] if added else 0)
        standalone = decision if action == "repeat" or action.startswith("actual_") else row["elapsed_s"]
        flags = ("invalid " if not row["parse_valid"] else "") + ("truncated" if row["generation_truncated"] else "")
        rows.append(f"<tr><th>{escape(action)}</th><td><pre>{escape(row['response'])}</pre><small>{escape(flags)}</small></td><td>{100*row[METRIC]:.0f}%</td><td>{100*row['official_f1']:.1f}%</td><td>{row['elapsed_s']:.3f}</td><td>{decision:.3f}</td><td>{standalone:.3f}</td></tr>")
    figures = []
    for label, name in (("Input 1: same overview", audit["direct"]["input_files"][0]), ("Input 2: native PDF-render crop", audit[f"actual_native_{region}"]["input_files"][1]), ("Input 2: crop from degraded overview", audit[f"actual_degraded_{region}"]["input_files"][1])):
        figures.append(f'<figure><a href="{name}"><img src="{name}" alt="{escape(label)}" loading="lazy"></a><figcaption>{label} · {saved[name]["size"]}</figcaption></figure>')
    flag = mask["mask"][key]
    content = f'<section><h2>{escape(category)}</h2><p><code>{escape(key)}</code> · quadrant {region} · source flag: {escape(flag["category"])}</p><h3>{escape(source["question"])}</h3><p>Reference span: <strong>{escape(source["answer"])}</strong></p><p>{escape(flag.get("reason", ""))}</p><div class="images">' + "".join(figures)
    content += '</div><p>All three histories use this same pair of pixels within each fidelity. Click an image for its full-resolution lossless input. No crop boundary is drawn into the inputs.</p>'
    content += '<div class="scroll"><table><thead><tr><th>Condition</th><th>Complete response</th><th>Text EM</th><th>Official F1</th><th>Call (s)</th><th>Decision state (s)</th><th>Standalone (s)</th></tr></thead><tbody>' + "".join(rows) + '</tbody></table></div>'
    content += f'<details><summary>Exact chat templates and input provenance</summary><pre>{escape(json.dumps({"messages": chats, "inputs": audit}, ensure_ascii=False, indent=2))}</pre></details></section>'
    return content, {"category": category, "example_id": key, "source_id": source["source_id"], "region": region, "quality_flag": flag, "actions": audit, "assets": saved}


class AssetChecker(HTMLParser):
    def __init__(self, base):
        super().__init__()
        self.base = base
    def handle_starttag(self, tag, attrs):
        for key, value in attrs:
            if key in ("src", "href"):
                require(value is not None and ":" not in value and not value.startswith("/"), "Non-local HTML resource")
                path = (self.base / value).resolve()
                require(path.is_relative_to(self.base.resolve()) and path.is_file(), "Missing/escaping HTML asset")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "manifest", "summary", "quality-mask", "output", "plots"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    metadata, summary, sources, groups, mask = load_completed(args.run, args.manifest, args.summary, args.quality_mask)
    require(not args.output.parent.exists(), "Use a new gallery directory; existing deliverables are never overwritten")
    assets = args.output.parent / "assets"
    assets.mkdir(parents=True)
    selected, coverage = select_cases(groups)
    sections, cases = [], []
    for category, key, region in selected:
        section, case = render_case(category, key, region, sources[key], groups[key], metadata["config"], mask, args.manifest, assets)
        sections.append(section)
        cases.append(case)
    counts = "".join(f'<li>{escape(row["category"])}: {row["matching_pages"]} pages / {row["matching_page_region_pairs"]} page–region pairs; {escape(row["skip_reason"] or "shown")}. <small>{escape(row["rule"])}</small></li>' for row in coverage)
    style = 'body{font:16px/1.5 system-ui,sans-serif;max-width:1280px;margin:auto;padding:28px;background:#f5f6f8;color:#19232f}h1,h2,h3{line-height:1.2}section{background:white;padding:24px;margin:28px 0;border:1px solid #cdd5de;border-radius:8px}.images{display:flex;gap:16px;align-items:start}figure{flex:1;margin:0;min-width:0}img{width:100%;max-height:520px;object-fit:contain;background:#eee}figcaption,small{font-size:12px;color:#465466}small{display:block}table{border-collapse:collapse;width:100%;font-size:13px}th,td{text-align:left;vertical-align:top;border-bottom:1px solid #d7dce2;padding:8px}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.4 monospace;margin:0}details{margin-top:18px}li{margin:8px 0}@media(max-width:700px){.images{display:block}body{padding:10px}}'
    document = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="Content-Security-Policy" content="default-src \'none\'; img-src \'self\'; style-src \'unsafe-inline\'; base-uri \'none\'"><title>LookAgain — history and visual detail</title><style>' + style + '</style><body><h1>History and visual detail</h1><p>100 development reports · 2,700 completed calls · frozen Qwen3-VL-4B. This is an <strong>outcome-selected illustration</strong>, not a representative estimate or a learned policy.</p><p>Categories were coded before main outcomes were inspected. Selection uses category order, lexicographic example ID and fixed quadrant order tl/tr/bl/br, without repeated pages. A displayed quadrant is selected using outcomes; aggregate conclusions require all four regions and all 100 reports.</p><ul>' + counts + '</ul><p>Native means a crop from the fixed-DPI PDF render. Degraded means the same region reconstructed from the low-resolution overview, with matched output dimensions. Fresh changes chat layout and wording; the placeholder preserves turns but changes answer content and token length. Neither identifies an internal mechanism by itself.</p><p>Decision-state cost includes direct plus branch; standalone fresh/placeholder excludes direct, while actual/repeat needs both calls. Direct and high-resolution are standalone controls. Times retain observed variability and exclude loading, warmup and PDF rendering. Reference spans may be ambiguous; original labels and the frozen source-only quality flags are preserved. Exact-match failures need not mean semantic errors.</p>' + "".join(sections) + '<p>Saved images reconstruct the logged PIL inputs and match their pixel hashes; they are not captured model activations. Full prompts and hashes are available above. No held-out validation, generalization, novelty or controller-training claim follows from these cases.</p><p><a href="provenance.json">Provenance and deterministic selection</a></p></body></html>'
    figures = plot_summary(summary, args.plots)
    document = document.replace('</body>', '<p>Zero or narrow empirical bootstrap intervals can arise from few or no observed discordances; they do not establish population equivalence or rule out effects on unseen reports.</p></body>')
    provenance = {"experiment": "history_context_v4", "selection": "Outcome-selected, illustrative; deterministic rules fixed in renderer before inspecting main outcomes", "bindings": summary["bindings"], "summary_sha256": digest(args.summary), "renderer_sha256": digest(__file__), "categories": coverage, "cases": cases, "figures": figures, "limitations": summary["caveats"]}
    (args.output.parent / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.write_text(document, encoding="utf-8")
    AssetChecker(args.output.parent).feed(document)
    print(f"Wrote {len(cases)} illustrative cases, exact input PNGs, provenance and PNG/SVG scientific figures; offline asset validation passed.")


if __name__ == "__main__":
    main()
