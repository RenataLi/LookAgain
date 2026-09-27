"""Three-page v5 research note from completed data and run-bound interpretation."""
import argparse
from html import escape
import json
from pathlib import Path
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle,getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image


def build(summary_path, output):
    s=json.loads(summary_path.read_text(encoding="utf-8"))
    a=json.loads((summary_path.parent/"interpretation.json").read_text(encoding="utf-8"))
    assert s["status"]=="complete" and s["integrity"]["passed"]
    assert a["run_fingerprint"]==s["bindings"]["run_fingerprint"]
    styles=getSampleStyleSheet()
    for name,size,leading,after,font in [("Title5",25,29,13,"Helvetica-Bold"),("Deck5",12,16,11,"Helvetica"),
        ("Body5",10,14,8,"Helvetica"),("Head5",13,17,9,"Helvetica-Bold"),("Small5",8,11,6,"Helvetica"),("Cell5",8.5,11,0,"Helvetica")]:
        styles.add(ParagraphStyle(name=name,fontName=font,fontSize=size,leading=leading,spaceAfter=after,textColor=colors.HexColor("#174f5b" if name in ("Title5","Head5") else "#263b48")))
    p=lambda text,style="Body5":Paragraph(text,styles[style])
    def table(rows,widths):
        t=Table([[p(escape(str(x)),"Cell5") for x in row] for row in rows],colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#e5f0f1")),("ROWBACKGROUNDS",(0,1),(-1,-1),[colors.white,colors.HexColor("#f4f7f8")]),
            ("VALIGN",(0,0),(-1,-1),"TOP"),("LEFTPADDING",(0,0),(-1,-1),7),("RIGHTPADDING",(0,0),(-1,-1),7),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        return t
    def effect(r):return f"{r['difference_pp']:+.2f} pp [{r['ci95_pp'][0]:+.2f}, {r['ci95_pp'][1]:+.2f}]"
    primary=s["detail_effects"]["256"]["official_em"]
    story=[p("LOOKAGAIN / RESEARCH NOTE 05","Small5"),p("When does another view<br/>supply useful evidence?","Title5"),
        p("Overview resolution and a reference-guided region in a frozen VLM","Deck5"),
        p("Primary detail advantage: <b>"+effect(primary)+"</b>","Deck5"),p(escape(a["conclusion"])),
        p("85 reused TAT-DQA reports; 850 independent calls. The region is supplied by reference-linked annotation. This diagnostic does not implement a learned or deployable region selector."),p("Accuracy and measured standalone cost","Head5")]
    rows=[["Condition","Official EM","Strict EM","Mean seconds"]]
    for action,r in s["actions"].items():
        rows.append([action.replace("_"," "),f"{100*r['metrics']['official_em']:.2f}%",f"{100*r['metrics']['conservative_text_em']:.2f}%",f"{r['standalone_latency_s']['mean']:.3f}"])
    story += [table(rows,[185,105,100,105]),Spacer(1,9),
        p("Official span EM is primary for v5; strict text EM remains secondary. This is an explicit prospective metric and cohort change from v3/v4, not a like-for-like improvement over their headline scores. Invalid and truncated responses remain in all denominators.","Small5"),
        p("Times include image decoding/construction, hashes, processing, transfers, generation and decoding. Model loading, warmup, rendering and reference-based localization are excluded. One active-desktop run is not a stable throughput benchmark.","Small5"),PageBreak(),
        p("Resolution, region and cost","Title5"),
        p("At each overview cap (256, 512, 1024 visual tokens), compare the full page alone, the same page plus a source-render crop, and the page plus a degraded crop rebuilt from that overview. A separate full-page 4096-token baseline completes ten calls per report. Every call uses a fresh conversation."),
        p("The source crop is identical across budgets. Within each native/degraded pair, region coordinates, output size, chat, overview pixels, image grids and input tokens match. The degraded crop cannot recover detail lost in the overview. Native regions contain 64-424 actual visual tokens, below the 1024 cap."),
        p("Matched detail effects","Head5")]
    rows=[["Overview cap","Official EM advantage (95% CI)","Strict EM advantage (95% CI)"]]
    for b in ("256","512","1024"):
        rows.append([b,effect(s["detail_effects"][b]["official_em"]),effect(s["detail_effects"][b]["conservative_text_em"])])
    story += [table(rows,[80,208,207]),Spacer(1,10),
        Image(str(summary_path.parent/"evidence_cost.png"),width=495,height=291),Spacer(1,8),
        p("Paired intervals use 10,000 whole-report resamples, seed 20260926. Only official native256 minus degraded256 is primary; remaining intervals are exploratory and unadjusted. A resolution interaction can result from degrading the control, so absolute quality matters.","Small5"),
        p("An alternative decision-state accounting adds the matching direct call to a regional invocation and takes the maximum memory peak. The previous answer is never fed to the regional call. Neither accounting includes a deployable selector.","Small5"),PageBreak(),
        p("Content audit and next decision","Title5"),p(escape(a["semantic_summary"])),
        p("Every primary-pair parsed-answer change, plus raw-response differences with an invalid parse, was selected for qualitative review. Review packets contain source pages, questions, references and randomized A/B responses with condition identities and metric scores withheld. These descriptive judgments are separate from independent expert validation. Official scores remain unchanged."),
        p("Inspected examples","Head5")]
    for case in a["case_notes"][:3]:story.append(p(escape(case)))
    story += [p("What this result can support","Head5"),p(escape(a["next_step"])),p(escape(a["uncertainty_note"])),
        p("Integrity and scope","Head5"),p("The source-only preparation retained 85 of 100 prior development reports after quality, mapping and PDF-frame checks. It corrects CropBox-to-MediaBox coordinates and excludes unsupported cases without replacements. Eligibility uses reference annotations and therefore selects a privileged cohort. Full source/pixel/chat/score reconstruction and a separate arithmetic/bootstrap check accompany the results."),
        p("Frozen Qwen3-VL-4B-Instruct, BF16/SDPA, greedy 64-token output, RTX 5090. No controller, second-model replication, transfer or held-out performance is established. The generic idea of utility-supervised crop routing already has close prior work."),
        p("Primary sources and reproducibility","Head5"),
        p('<link href="https://nextplusplus.github.io/TAT-DQA/" color="#176b76">TAT-DQA authors</link> | <link href="https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct" color="#176b76">Qwen3-VL-4B-Instruct</link><br/><link href="https://arxiv.org/abs/2608.21762v2" color="#176b76">GapSight: loss-gap supervision for crop routing</link> | <link href="https://arxiv.org/abs/2602.06566v3" color="#176b76">SPARC: perception and reasoning</link>',"Small5"),
        p("Repository: docs/evidence_availability_protocol.md; configs/evidence_availability.json; reports/evidence85. Raw calls, source audit, complete content review, input/record hashes and execution instructions are included. Full source pages and model weights are excluded from the code archive.","Small5")]
    output.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=42,bottomMargin=43,title="LookAgain: evidence availability and overview resolution",author="LookAgain research project")
    def footer(canvas,document):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor("#d3dfe4"));canvas.line(50,35,A4[0]-50,35)
        canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor("#647785"))
        canvas.drawString(50,23,"LOOKAGAIN / V5 DEVELOPMENT DIAGNOSTIC / 26 SEPTEMBER 2026")
        canvas.drawRightString(A4[0]-50,23,str(document.page));canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--summary",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args();build(a.summary,a.output)
