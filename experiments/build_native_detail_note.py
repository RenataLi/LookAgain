"""Render the completed, hash-bound v3 research note from its analysis JSON."""
from __future__ import annotations
import argparse
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether


def build(summary, output):
    data=json.loads(summary.read_text(encoding="utf-8"))
    interpretation=json.loads((summary.parent/'interpretation.json').read_text(encoding='utf-8'))
    if interpretation['run_fingerprint'] != data['bindings']['run_fingerprint']:
        raise ValueError('Authored interpretation belongs to another run')
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="TitleV3",fontName="Helvetica-Bold",fontSize=27,leading=31,textColor=colors.HexColor("#174f5b"),spaceAfter=16))
    styles.add(ParagraphStyle(name="DeckV3",fontName="Helvetica",fontSize=12,leading=17,textColor=colors.HexColor("#435864"),spaceAfter=15))
    styles.add(ParagraphStyle(name="BodyV3",fontName="Helvetica",fontSize=10,leading=14,spaceAfter=9))
    styles.add(ParagraphStyle(name="SmallV3",fontName="Helvetica",fontSize=8,leading=11,textColor=colors.HexColor("#546975"),spaceAfter=6))
    styles.add(ParagraphStyle(name="HeadV3",fontName="Helvetica-Bold",fontSize=13,leading=17,textColor=colors.HexColor("#174f5b"),spaceBefore=12,spaceAfter=8))
    styles.add(ParagraphStyle(name="CellV3",fontName="Helvetica",fontSize=8.6,leading=11))
    para=lambda text,style="BodyV3":Paragraph(text,styles[style])
    table=lambda rows,widths:make_table(rows,widths,styles)
    story=[]
    story+=[para("LOOKAGAIN / RESEARCH NOTE 03","SmallV3"),para("Does another view<br/>add useful detail?","TitleV3"),para("A paired PDF-resolution intervention in frozen Qwen3-VL-4B","DeckV3")]
    contrast=data["contrasts"]["native_minus_degraded"]["metrics"]["conservative_text_em"]
    lo,hi=contrast["ci95_pp"]
    headline=f"Source-region minus overview-region exact match: <b>{contrast['difference_pp']:+.2f} percentage points</b> (95% paired interval [{lo:+.2f}, {hi:+.2f}])."
    story+=[para(headline,"DeckV3"),para("We test whether pixels discarded by the initial overview carry useful answer information. The two regional branches have the same question, initial response, neutral prompt, field of view, output size and token grid. One branch samples the fixed 200-DPI source rendering; the other uses only the already-seen overview.")]
    rows=[["Condition", "Strict EM", "Official EM", "Total mean time"]]
    names={"direct":"Direct overview","highres":"High-resolution full page","repeat":"Repeat exact overview","native":"Source region (mean of 4)","degraded":"Overview region (mean of 4)"}
    for key,label in names.items():
        item=data["conditions"][key]
        rows.append([label,f"{100*item['metrics']['conservative_text_em']:.2f}%",f"{100*item['metrics']['official_em']:.2f}%",f"{item['full_path_latency_s']['mean']:.3f} s"])
    story+=[table(rows,[220,80,80,115]),Spacer(1,9),para("100 source reports, one page/question each; 1,100 model calls. All are TAT-DQA training examples used for development. Four-region averages describe a uniformly chosen single region in expectation, not a four-view deployed policy.","SmallV3")]
    story+=[para("What this result permits","HeadV3"),para(escape(interpretation['conclusion'])),para("The oracle results below use answer labels after the fact. They measure possible headroom; this stage does not train a controller or establish that a useful policy can be learned.")]
    story.append(PageBreak())
    story+=[para("Controls, cost and headroom","TitleV3"),para("All eleven actions are evaluated on every report. Each follow-up independently starts from the same fresh direct answer; high-resolution direct is a separate single call.")]
    oracle=data["privileged_oracles"]
    rows=[["Label-aware candidate set", "Strict EM", "Official EM"]]
    for key,row in oracle["oracles"].items():
        rows.append([key.replace("_"," "),f"{100*row['metrics']['conservative_text_em']:.2f}%",f"{100*row['metrics']['official_em']:.2f}%"])
    story+=[table(rows,[325,85,85])]
    oc=oracle["matched_native_minus_degraded"]["conservative_text_em"]
    story+=[Spacer(1,9),para(f"Matched native-minus-degraded oracle gap: <b>{oc['difference_pp']:+.2f} pp</b>, 95% interval [{oc['ci95_pp'][0]:+.2f}, {oc['ci95_pp'][1]:+.2f}]. Both sets include direct, repeat and four regions; highres is excluded from their selection.")]
    story+=[para("Repairs and harms both matter","HeadV3")]
    rows=[["Condition", "Repairs", "Harms", "Presentations"]]
    for name in ("native","degraded","repeat","highres"):
        r=data["changes_vs_direct"][name]
        rows.append([names[name],str(r['fixes']),str(r['harms']),str(r['n_presentations'])])
    story+=[table(rows,[265,70,70,90]),Spacer(1,9),para("For regional conditions, 400 presentations remain clustered within 100 reports. A repair changes a wrong direct answer to correct; a harm changes a correct one to wrong. Invalid and truncated answers remain in the denominator.","SmallV3")]
    story+=[para("Measured computation","HeadV3"),para("Overview and region ceilings are 1,024 visual tokens; highres is at most 4,096. Source/highres views are never enlarged above source dimensions. Realized grids are recorded. Total follow-up latency adds the initial direct call, with no cross-call KV reuse. Measurements include PNG decoding, resizing, pixel hashes, processing, transfers, generation and answer extraction; model load, warmup, logging and PDF rasterization are separate.")]
    story+=[para("These are local RTX 5090 measurements, not a hardware-independent speed guarantee. The interventions do not have equal total computation. Native/degraded pairs match actual input-token counts and image grids, but generation lengths and time can differ.")]
    story.append(PageBreak())
    story+=[para("Data quality and limits","TitleV3"),para("The source audit was frozen before main inference. The primary estimate retains all 100 original labels. A prespecified sensitivity analysis removes two confirmed label errors and four unresolved or unsupported question/reference pairs; mapping-only problems remain included.")]
    quality=data.get("quality_sensitivity") or data.get("source_quality_sensitivity")
    if quality:
        qc=quality["contrasts"]["native_minus_degraded"]["metrics"]["conservative_text_em"]
        story.append(para(f"On the 94 retained reports, the primary native-minus-degraded contrast is <b>{qc['difference_pp']:+.2f} pp</b> (95% interval [{qc['ci95_pp'][0]:+.2f}, {qc['ci95_pp'][1]:+.2f}]). Source-only exclusions are documented by question ID and a pinned quality-mask hash; they were not chosen from model mistakes."))
    story+=[para("Design and uncertainty","HeadV3"),para("Original reports and eligible questions are ordered by seeded hashes. Each selected PDF/OCR unit has one page, one short text-span reference, no scale and no comparison requirement. Full pages are rendered at 200 DPI from original PDFs; the distributed 224x224 PNG thumbnails are not used. Text availability and resolution are technical screens, not guarantees of perfect annotations or intrinsic vector detail."),para("The primary metric preserves punctuation, articles and numeric formatting after lowercasing and whitespace normalization. Secondary official EM/F1 follow a pinned author evaluator; diagnostic ANLS is reported separately. Intervals use 10,000 paired resamples of source reports, seed 20260925, and are exploratory without multiplicity correction.")]
    story+=[para("Interpretation","HeadV3"),para("The study measures a financial-document development panel in one model family. Related report templates can remain correlated, and two-stage resampling is part of the degraded intervention. Scoring against one reference can penalize valid paraphrases or unit formatting. Qualitative source checks describe label support; they do not provide independent expert validation. Conclusions apply to the fixed-region visual comparison on this panel."),para(escape(interpretation['uncertainty_note'])),para(escape(interpretation['next_step']))]
    story+=[para("Sources and reproduction","HeadV3"),para('<link href="https://nextplusplus.github.io/TAT-DQA/" color="#176b76">TAT-DQA authors: dataset, task and access</link><br/><link href="https://github.com/fengbinzhu/Doc2SoarGraph/tree/71a02715849942b79c1f74934a139e4b56fc6826" color="#176b76">Pinned author evaluator: Doc2SoarGraph</link><br/><link href="https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct" color="#176b76">Frozen backbone: Qwen3-VL-4B-Instruct</link>',"SmallV3"),para("Repository artifacts: docs/native_detail_protocol.md; configs/native_detail.json; reports/native100/{run.json, records.jsonl, summary.json, report.md, source_audit.md, source_quality_mask.json}. Exact model, source, input-pixel and code hashes are retained there.","SmallV3")]
    output.parent.mkdir(parents=True,exist_ok=True)
    doc=SimpleDocTemplate(str(output),pagesize=A4,rightMargin=50,leftMargin=50,topMargin=44,bottomMargin=43,title="LookAgain: Does another view add useful detail?",author="LookAgain research project")
    def footer(canvas,document):
        canvas.saveState();canvas.setStrokeColor(colors.HexColor("#d3dfe4"));canvas.line(50,35,A4[0]-50,35);canvas.setFont("Helvetica",8);canvas.setFillColor(colors.HexColor("#647785"));canvas.drawString(50,23,"LOOKAGAIN / V3 DEVELOPMENT STUDY / 25 SEPTEMBER 2026");canvas.drawRightString(A4[0]-50,23,str(document.page));canvas.restoreState()
    doc.build(story,onFirstPage=footer,onLaterPages=footer)


def make_table(rows,widths,styles):
    values=[[Paragraph(escape(str(value)),styles['CellV3'])for value in row]for row in rows]
    item=Table(values,colWidths=widths,repeatRows=1,hAlign="LEFT")
    item.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e5f0f1')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f5f8f9')]),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),8),('RIGHTPADDING',(0,0),(-1,-1),8),('TOPPADDING',(0,0),(-1,-1),9),('BOTTOMPADDING',(0,0),(-1,-1),9),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#c3d5dc'))]))
    return item


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--summary',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args();build(args.summary,args.output)
