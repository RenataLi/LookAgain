"""Build a measured, four-page pilot note. Requires the optional report extras."""
from __future__ import annotations

import argparse
from datetime import datetime
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image

NAVY = colors.HexColor("#162A46")
TEAL = colors.HexColor("#087F8C")
GRAY = colors.HexColor("#52647A")
LABELS = {"direct":"Direct / stop", "highres":"High-resolution direct", "recheck":"Repeat image", "think":"Extra text reasoning", "crop_tl":"Crop: upper left", "crop_tr":"Crop: upper right", "crop_bl":"Crop: lower left", "crop_br":"Crop: lower right"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scratch", type=Path, required=True)
    args = p.parse_args()
    summary = json.loads((args.report / "summary.json").read_text(encoding="utf-8"))
    run = json.loads((args.report / "run.json").read_text(encoding="utf-8"))
    # The qualitative interpretation below was authored for this exact study.
    # A new experiment needs a new interpretation, not only a refreshed table.
    if run["fingerprint"] != "c62bd536119edcf7674daf46b10ff1923d26d9c991acc05896d75d64779973aa":
        raise ValueError("This research note is specific to the archived 400-image study")
    diagnostics = json.loads((args.report / "diagnostics.json").read_text(encoding="utf-8"))
    if diagnostics["run_fingerprint"] != run["fingerprint"]:
        raise ValueError("Secondary diagnostics belong to a different run")
    profile = json.loads(args.profile.read_text(encoding="utf-8"))
    if summary["status"] != "complete" or summary["coverage"]["complete_examples"] != len(run["example_ids"]):
        raise ValueError("The research note requires complete planned coverage")
    n = summary["coverage"]["complete_examples"]
    if n < 300:
        raise ValueError("A small engineering run is not a completed diagnostic study")
    if (profile["sample"]["count"] != n
            or profile["sample"]["manifest_sha256"] != run["manifest_sha256"]):
        raise ValueError("Image profile must describe this complete study manifest")
    if summary.get("run_fingerprint") != run["fingerprint"]:
        raise ValueError("Main summary belongs to a different run")
    if (diagnostics.get("status") != "complete"
            or diagnostics["coverage"]["complete_examples"] != n
            or diagnostics["coverage"]["planned_examples"] != n
            or diagnostics["crop_exclusive_rescues"]["n_examples"] != n):
        raise ValueError("Secondary diagnostics must cover the entire planned run")
    actions = summary["actions"]
    oracle = summary["baselines"]["hindsight_oracle"]
    direct = actions["direct"]
    crop_metrics = [m for key, m in actions.items() if key.startswith("crop_")]
    args.scratch.mkdir(parents=True, exist_ok=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":11, "axes.spines.top":False, "axes.spines.right":False})
    fig, ax = plt.subplots(figsize=(7.4,3.6), layout="constrained")
    for i, (key, m) in enumerate(actions.items()):
        ax.scatter(m["total_latency_s"]["mean"], 100*m["accuracy"], s=55, label=LABELS[key])
    ax.scatter(oracle["total_latency_s"]["mean"], 100*oracle["accuracy"], marker="D", facecolors="none", edgecolors="black", s=95, label="Hindsight oracle*")
    ax.set(xlabel="Mean full-path latency (seconds)", ylabel="Pilot exact match (%)", ylim=(0,100), xlim=(0,None))
    ax.grid(alpha=.18)
    ax.legend(loc="center left", bbox_to_anchor=(1.01,.5), frameon=False, fontsize=9)
    chart = args.scratch / "note_cost.png"
    fig.savefig(chart,dpi=200)
    plt.close(fig)

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleCustom",fontName="Helvetica-Bold",fontSize=27,leading=31,textColor=NAVY,spaceAfter=10))
    styles.add(ParagraphStyle(name="Sub",fontName="Helvetica",fontSize=12,leading=17,textColor=GRAY,spaceAfter=16))
    styles.add(ParagraphStyle(name="BodyCustom",fontName="Helvetica",fontSize=10,leading=14,spaceAfter=8,textColor=NAVY))
    styles.add(ParagraphStyle(name="Head",fontName="Helvetica-Bold",fontSize=14,leading=19,spaceBefore=10,spaceAfter=8,textColor=TEAL))
    styles.add(ParagraphStyle(name="SmallCustom",fontName="Helvetica",fontSize=8,leading=11,spaceAfter=6,textColor=GRAY))
    story=[]
    def para(text,style="BodyCustom"):
        return Paragraph(text,styles[style])
    def add(text,style="BodyCustom"):
        story.append(para(text,style))
    def table(rows,widths):
        content=[[para(str(cell),"SmallCustom") for cell in row] for row in rows]
        t=Table(content,colWidths=widths,repeatRows=1,hAlign="LEFT")
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#E9F2F4")),("VALIGN",(0,0),(-1,-1),"TOP"),("LINEBELOW",(0,0),(-1,0),.6,TEAL),("LINEBELOW",(0,1),(-1,-1),.25,colors.HexColor("#DDE4EC")),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.append(t)

    add("LookAgain", "TitleCustom")
    add("The marginal value of another look<br/>A diagnostic study of frozen vision-language inference", "Sub")
    add("RenataLi  |  Independent research engineering  |  25 September 2026", "SmallCustom")
    add("Abstract", "Head")
    add(f"We evaluate eight paired inference conditions for a frozen Qwen3-VL-4B-Instruct on {n} distinct GQA training images. The conditions separate a first answer, higher-resolution input, repeated pixels, additional text reasoning, and four fixed image regions. The direct condition achieves {100*direct['accuracy']:.2f}% pilot normalized exact match. A privileged hindsight selector over stop and the follow-up actions reaches {100*oracle['accuracy']:.2f}%, an optimistic {oracle['delta_accuracy_pp']:.2f}-percentage-point opportunity. These are development diagnostics, not held-out benchmark results or evidence for a trained controller.")
    add(f"Always inspecting a fixed quadrant scores {100*min(m['accuracy'] for m in crop_metrics):.2f}% to {100*max(m['accuracy'] for m in crop_metrics):.2f}%. The measured damage and limited oracle headroom motivate revising the task and action design before training a selector.")
    add("Research question", "Head")
    add("After an initial answer, can a model benefit more from inspecting a particular region than from another answer attempt or additional text reasoning? A future controller should predict the change in correctness, including harmful interventions, using only information available before the action.")
    add("Scope and contribution", "Head")
    add("This version contributes a reproducible paired experimental pipeline and measured diagnostic evidence. It does not introduce a validated new learning method. Adaptive cropping, lightweight external crop models, value-of-information selection and necessity-aware tool use already have close precedents [1-5].")
    add("The protocol includes strong full-image input, no-new-region controls, real inference costs and explicit failure reporting. Learned selection and out-of-distribution transfer remain future work, conditional on the measured headroom.")
    add("Study status", "Head")
    add(f"{n} complete image-question pairs; {n*len(actions)} recorded model calls. One question per image. Deterministic development prefix, not a representative random sample. No official test split or transfer benchmark was used.")

    story.append(PageBreak())
    add("Protocol and reproducibility", "TitleCustom")
    table([
        ["Component","Specification"],
        ["Model","Qwen3-VL-4B-Instruct; frozen BF16 weights; SDPA; greedy generation"],
        ["Data",f"{n}-image prefix from GQA training images, joined to first matching training question; source revision pinned"],
        ["Images","Approximately 1,024 visual tokens per overview/crop; 4,096 for independent high-resolution direct answering; actual grids logged"],
        ["Crops","Four overlapping quadrants, each 60% of width and height; known location disclosed to model; no object-label access"],
        ["Responses","Short branches: 32-token cap with prefilled ANSWER: prefix; think: 128-token cap and generated final marker"],
        ["Hardware",escape(run["gpu"])+"; PyTorch "+escape(run["packages"]["torch"])+"; Transformers "+escape(run["packages"]["transformers"])],
    ],[36*mm,134*mm])
    add("Timing and scoring", "Head")
    add("Warm synchronized wall time includes image decoding/resizing, processing, host-to-device transfer, generation, decoding and batched diagnostic log-probability extraction. It excludes model loading and synthetic warm-up. No cross-call KV-cache reuse is used. Every follow-up policy is charged the original direct call plus its own call; high-resolution direct is independent.")
    add("Correctness is conservative normalized exact match. A bare one-line answer is compared in its entirety; multi-line reasoning needs a unique final ANSWER: line. Invalid formats count as errors. There is no LLM judge, yes/no rescue rule, or target-dependent extraction. Raw responses are retained for audit.")
    add("Engineering preflight", "Head")
    add("Small engineering checks exposed placeholder copying and verbose answer formats. Those runs were excluded. The response contract was frozen after a 16-image / 128-call format check with no malformed or truncated outputs. The main study was restarted from the beginning; this development process is documented, not externally preregistered.")
    add("Native image detail", "Head")
    add(f"The sample's median long edge is {profile['images']['long_edge_pixels']['median']:g} pixels; {profile['images']['long_edge_at_most_640_count']} of {n} images have a long edge no greater than 640 pixels. Bicubic upsampling creates no new source detail. This pilot chiefly probes re-encoding and attention allocation, and cannot establish performance on true high-resolution visual search.")

    story.append(PageBreak())
    add("Measured outcomes", "TitleCustom")
    rows=[["Condition","Exact match","Delta [95% CI], pp","Fix / harm","Mean s"]]
    for key,m in actions.items():
        ci=m["delta_ci_95_pp"]
        rows.append([LABELS[key],f"{100*m['accuracy']:.2f}%",f"{m['delta_accuracy_pp']:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]",f"{m['wrong_to_right_count']} / {m['right_to_wrong_count']}",f"{m['total_latency_s']['mean']:.3f}"])
    table(rows,[43*mm,25*mm,54*mm,25*mm,23*mm])
    add("Changes are paired against direct. Fix = wrong to right; harm = right to wrong. 95% percentile intervals use 10,000 image-cluster bootstrap samples; they are exploratory and not adjusted for multiple comparisons.","SmallCustom")
    story.append(Image(str(chart),width=170*mm,height=83*mm))
    add("*Hindsight oracle: correctness is known for every candidate, only the selected action is charged, and high-resolution direct is excluded from its action set. This is an optimistic diagnostic bound, not a measured deployable policy. The full report includes separate crop and no-new-region oracle diagnostics.","SmallCustom")
    format_counts=summary["format_diagnostics"]
    add(f"Main-run diagnostics: {sum(v['invalid_answers'] for v in format_counts.values())} invalid answer formats; {sum(v['truncated_generations'] for v in format_counts.values())} truncated generations. Peak allocated GPU memory across recorded actions: {max(m['policy_peak_memory_gib']['max'] for m in actions.values()):.2f} GiB.","SmallCustom")

    story.append(PageBreak())
    add("What the pilot tells us", "TitleCustom")
    add("What can be concluded", "Head")
    add("Every fixed crop condition reduces average exact match: each repairs only 2-6 examples while damaging 32-55. Even the privileged selector offers limited headroom. The decision for this version is to revise the task/action design before controller training, not to claim a successful adaptive policy.")
    exclusive = diagnostics["crop_exclusive_rescues"]
    ci = diagnostics["comparisons_vs_nonvisual_oracle"]["oracle_all"]["delta_ci_95_pp"]
    add(f"There are <b>{exclusive['count']} crop-exclusive rescues out of {n}</b> ({100*exclusive['fraction_all_examples']:.2f}%; paired 95% interval {ci[0]:.2f} to {ci[1]:.2f} percentage points). High-resolution direct already answers {exclusive['highres_correct_among_exclusive']} of these correctly. This is the measured upper-bound opportunity beyond the available no-new-region alternatives.")
    add("Six of the nine exclusive rescues change left to right and succeed only with right-side crops. The prompt explicitly names the crop location. Textual direction cues or a shifted coordinate frame are competing explanations; the recorded gains do not establish acquisition of new visual detail. Illustrative case inspection also finds ambiguous object references. Original grades are retained.")
    add("Limitations", "Head")
    add("This is a deterministic training-data prefix with unknown backbone pretraining exposure. Most images are small; the overview resolution was not optimized in a separate sweep. Exact match and implicit referents limit interpretation. A separate eight-image repeat preserved all answers but showed mean latencies 12-58% higher; background activity and sparse synthetic warm-up limit timing claims. One backbone and task family cannot establish transfer.")
    add("Decision before controller training", "Head")
    add("First test direction-word placebos, neutral crop descriptions, and explicit original-image coordinates on a new development panel with unambiguous referents and native fine detail. Compare visibility and global-context preservation. Only then, if complementary benefit survives, collect image-disjoint controller training data. Keep the planned transfer benchmarks untouched. These controls are planned, not completed ablations.")
    add("Selected primary references", "Head")
    refs=[("1","Adaptive Chain-of-Focus","2505.15436"),("2","CropVLM","2511.19820"),("3","VOILA","2602.03007"),("4","Beacon","2607.28595"),("5","VisLens","2608.30705")]
    for num,title,aid in refs:
        add(f'[{num}] {title}. <link href="https://arxiv.org/abs/{aid}" color="#087F8C">arxiv.org/abs/{aid}</link>',"SmallCustom")
    add('<link href="https://cs.stanford.edu/people/dorarad/gqa/about.html" color="#087F8C">GQA dataset</link> | <link href="https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct" color="#087F8C">Qwen3-VL model card</link>',"SmallCustom")
    add("Provenance", "Head")
    add("Run fingerprint: "+run["fingerprint"],"SmallCustom")
    add("Model revision: "+run["model"]["revision"],"SmallCustom")
    add("Source digest: "+run["code_sha256"],"SmallCustom")

    def footer(canvas,doc):
        canvas.setStrokeColor(colors.HexColor("#DDE4EC")); canvas.line(20*mm,17*mm,190*mm,17*mm)
        canvas.setFont("Helvetica",8); canvas.setFillColor(GRAY)
        canvas.drawString(20*mm,12*mm,"LOOKAGAIN  /  DEVELOPMENT RESEARCH NOTE")
        canvas.drawRightString(190*mm,12*mm,str(doc.page))
    doc=SimpleDocTemplate(str(args.output),pagesize=(210*mm,297*mm),rightMargin=20*mm,leftMargin=20*mm,topMargin=19*mm,bottomMargin=23*mm,title="LookAgain: Diagnostic Pilot",author="RenataLi")
    doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print(args.output)


if __name__ == "__main__":
    main()
