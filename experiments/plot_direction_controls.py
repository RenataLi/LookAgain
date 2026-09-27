"""Plot measured direction controls; no model access or outcome selection."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

LABELS = {"direct": "Direct / stop", "named": "Named real crop", "neutral": "Neutral real crop",
          "sham": "Repeated overview + crop label", "frame": "Real crop + original-frame rule"}
COLORS = {"direct": "#243B5A", "named": "#087F8C", "neutral": "#3772D8", "sham": "#C98A16", "frame": "#8653B6"}
CONTRASTS = ("neutral_minus_named", "named_minus_sham", "frame_minus_named")
CONTRAST_LABELS = ("Remove quadrant naming", "True crop minus repeated overview", "Add original-frame instruction")


def plot(summary, output, allow_smoke=False):
    if not summary.get("completed_primary_cohort") and not allow_smoke:
        raise ValueError("Incomplete or small engineering runs require --allow-smoke")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=True)
    n = summary["coverage"]["complete_images"]
    main = summary["main"]
    rows = {"direct": main["direct"], **main["conditions"]}
    methods = list(rows)
    contrasts = [main["primary_contrasts"][key] for key in CONTRASTS]
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.spines.left": False, "axes.edgecolor": "#9BAABC", "text.color": "#20344F",
                         "axes.labelcolor": "#344A65", "xtick.color": "#546880", "ytick.color": "#344A65"})
    fig = plt.figure(figsize=(14, 8), facecolor="white")
    left = fig.add_axes([.255, .31, .27, .48])
    right = fig.add_axes([.60, .31, .355, .48])
    title = "LookAgain v2 | Separating pixels, crop names and coordinate frames"
    if not summary.get("completed_primary_cohort"):
        title = "ENGINEERING SMOKE | Layout check, not a research result"
    fig.text(.055, .94, title, size=18, weight="bold")
    fig.text(.055, .888, f"{n} source images · {summary['coverage']['records']:,} completed calls · four quadrants per condition\n"
             "Frozen Qwen3-VL-4B-Instruct · reused development panel · no learned controller", size=11, color="#536780", linespacing=1.6)
    for index, key in enumerate(methods):
        value = 100 * rows[key]["expected_accuracy"]
        left.barh(index, value, color=COLORS[key], height=.54, alpha=.88)
        left.text(min(value + 1.6, 94), index, f"{value:.2f}%", va="center", fontsize=10, weight="bold")
    left.set(yticks=range(len(methods)), yticklabels=[LABELS[key] for key in methods], xlim=(0, 105), ylim=(-.7, len(methods)-.3))
    left.invert_yaxis()
    left.set_xlabel("Pilot exact match (%)")
    left.set_title("Expected accuracy for a uniform quadrant", loc="left", pad=16, fontsize=11)
    left.xaxis.grid(True, color="#E1E8F0", linewidth=.7)
    left.set_axisbelow(True)
    left.tick_params(axis="y", length=0, pad=8)
    endpoints = [abs(value) for row in contrasts for value in (row.get("delta_ci_95_pp") or [row["delta_expected_accuracy_pp"]])]
    extent = max(2.0, max(endpoints, default=2.0) * 1.2)
    right.axvline(0, color="#9AA9BA", linewidth=1, linestyle="--")
    for index, row in enumerate(contrasts):
        point = row["delta_expected_accuracy_pp"]
        interval = row.get("delta_ci_95_pp")
        if interval:
            right.plot(interval, [index, index], color="#087F8C", linewidth=2.5)
            right.plot(interval, [index, index], linestyle="none", marker="|", markersize=10, color="#087F8C")
        right.scatter([point], [index], s=60, color="#243B5A", zorder=3)
        ci = f"[{interval[0]:+.2f}, {interval[1]:+.2f}]" if interval else "interval unavailable"
        right.text(.03, index - .28, f"{CONTRAST_LABELS[index]}\n{point:+.2f} pp  {ci}",
                   transform=right.get_yaxis_transform(), ha="left", va="bottom", fontsize=10, linespacing=1.5)
    right.set(xlim=(-extent, extent), ylim=(2.6, -.75), yticks=[])
    right.set_xlabel("Paired difference (percentage points)")
    right.set_title("Three planned contrasts · 95% image bootstrap", loc="left", pad=16, fontsize=11)
    right.xaxis.grid(True, color="#E1E8F0", linewidth=.7)
    right.set_axisbelow(True)
    fig.text(.055, .18,
             "Each condition averages four paired views within each image, then averages images. Direct uses one answer per image.\n"
             "Intervals use 10,000 paired image resamples and are unadjusted for multiple comparisons. Four views are not four independent units.\n"
             "The repeated-overview control receives a deliberately incorrect crop description; its contrast also changes image/text congruity.\n"
             "No neutral-overview condition was run: this is not a full content-by-naming factorial experiment.",
             size=10, color="#536780", linespacing=1.7, va="top")
    paths = []
    for extension in ("png", "svg"):
        path = output / f"direction_conditions_and_contrasts.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        paths.append(path)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5), layout="constrained", facecolor="white")
    conditions = list(main["conditions"])
    for offset, field, color, label in ((-.18, "fix_fraction_all_presentations", "#087F8C", "Wrong to right"),
                                       (.18, "harm_fraction_all_presentations", "#BD4558", "Right to wrong")):
        values = [100 * main["conditions"][key][field] for key in conditions]
        ax.barh([i+offset for i in range(len(conditions))], values, height=.32, color=color, label=label)
        for i, value in enumerate(values):
            ax.text(value+.2, i+offset, f"{value:.2f}%", va="center", fontsize=9)
    maximum = max(100*main["conditions"][c][f] for c in conditions for f in ("fix_fraction_all_presentations", "harm_fraction_all_presentations"))
    ax.set(yticks=range(len(conditions)), yticklabels=[LABELS[key] for key in conditions], xlim=(0,max(3,maximum+3)))
    ax.invert_yaxis()
    ax.set_xlabel(f"Fraction of {4*n:,} paired image-quadrant presentations (%)")
    ax.set_title("Repairs and harms relative to the fresh direct answer" + (" | SMOKE ONLY" if not summary.get("completed_primary_cohort") else ""), pad=16)
    ax.xaxis.grid(True, color="#E1E8F0")
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    for extension in ("png", "svg"):
        path = output / f"direction_fixes_and_harms.{extension}"
        fig.savefig(path, dpi=180, facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-smoke", action="store_true")
    args = parser.parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    for path in plot(summary, args.output, allow_smoke=args.allow_smoke):
        print(path)


if __name__ == "__main__":
    main()
