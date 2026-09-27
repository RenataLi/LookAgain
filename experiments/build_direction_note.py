"""Render the authored, run-bound v2 note from verified aggregate evidence.

This command does not infer conclusions for a new experiment. The interpretation
JSON must identify the same completed run, and its prose needs a human case audit.
Install the optional `report` dependencies before using this script.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
    TableStyle,
)

NAVY = colors.HexColor("#18354B")
TEAL = colors.HexColor("#007F88")
PALE = colors.HexColor("#EEF5F6")
GRAY = colors.HexColor("#526270")
ORDER = ("named", "neutral", "sham", "frame")
LABELS = {
    "direct": "Fresh direct answer",
    "named": "Named real crop",
    "neutral": "Neutral real crop",
    "sham": "Repeated overview + crop label",
    "frame": "Real crop + original-frame rule",
}


def build(summary_path: Path, interpretation_path: Path, output: Path) -> None:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    prose = json.loads(interpretation_path.read_text(encoding="utf-8"))
    if not summary.get("completed_primary_cohort") or summary["coverage"]["complete_images"] != 400:
        raise ValueError("The published note requires the complete 400-image panel")
    if prose.get("run_fingerprint") != summary["run_fingerprint"]:
        raise ValueError("Authored interpretation belongs to another run")
    for field in ("headline", "finding", "secondary_findings", "case_audit", "decision", "execution"):
        if not isinstance(prose.get(field), str) or not prose[field].strip():
            raise ValueError(f"Missing authored interpretation: {field}")
    if any("TODO" in prose[field] for field in prose if isinstance(prose[field], str)):
        raise ValueError("Unfinished interpretation")
    output.parent.mkdir(parents=True, exist_ok=True)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name="LA_Title", fontName="Helvetica-Bold", fontSize=26,
                              leading=29, textColor=NAVY, spaceAfter=9))
    styles.add(ParagraphStyle(name="LA_Subtitle", fontSize=11.5, leading=16,
                              textColor=GRAY, spaceAfter=13))
    styles.add(ParagraphStyle(name="LA_H2", fontName="Helvetica-Bold", fontSize=13,
                              leading=17, textColor=NAVY, spaceBefore=13, spaceAfter=7))
    styles.add(ParagraphStyle(name="LA_Body", fontSize=9.5, leading=13.5,
                              textColor=NAVY, spaceAfter=8))
    styles.add(ParagraphStyle(name="LA_Small", fontSize=8, leading=11,
                              textColor=GRAY, spaceAfter=6))
    styles.add(ParagraphStyle(name="LA_Cell", fontSize=8.1, leading=11,
                              textColor=NAVY, alignment=TA_LEFT))
    width = 7.1 * inch

    def p(text: str, style: str = "LA_Body") -> Paragraph:
        return Paragraph(escape(text), styles[style])

    def table(rows: list[list[str]], widths: list[float]) -> Table:
        rendered = [[p(str(cell), "LA_Cell") for cell in row] for row in rows]
        t = Table(rendered, colWidths=widths, hAlign="LEFT", repeatRows=1, spaceAfter=6)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), PALE),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7),
            ("LEFTPADDING", (0, 0), (-1, -1), 7),
            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("LINEBELOW", (0, 0), (-1, 0), 0.7, TEAL),
            ("LINEBELOW", (0, 1), (-1, -1), 0.3, colors.HexColor("#DCE5E8")),
        ]))
        return t

    def figure(name: str, height: float) -> Image:
        path = summary_path.parent / name
        if not path.is_file():
            raise ValueError(f"Missing measured figure: {path}")
        result = Image(str(path), width=width, height=height, kind="proportional")
        result.hAlign = "CENTER"
        return result

    story = [p("LookAgain", "LA_Title"), p(
        "Do crops help, or do their descriptions change the answer?\n"
        "Direction controls on a frozen vision-language model", "LA_Subtitle"),
        p("Development diagnostic v2 | 25 September 2026 | RenataLi", "LA_Small"),
        table([[prose["headline"]]], [width]),
        p("Question and intervention", "LA_H2"),
        p("The original pilot contained crop-only answer repairs involving left and right. "
          "This follow-up separates three observable effects: removing a crop's directional "
          "label, adding an instruction to use the original image's coordinates, and replacing "
          "the crop with a repeated overview under the same named-crop instruction."),
        table([
            ["Condition", "Second image", "Instruction"],
            ["Named", "Actual quadrant crop", "Names the quadrant"],
            ["Neutral", "Same actual crop", "Omits the quadrant name"],
            ["Sham", "Exact copy of overview", "Same named-crop text; deliberately inaccurate"],
            ["Original frame", "Same actual crop", "Adds original-image coordinate rule"],
        ], [0.95 * inch, 2.05 * inch, 4.1 * inch]),
        p("400 distinct GQA training images; one question per image. One fresh direct answer "
          "and 16 independent follow-ups per image: 4 conditions x 4 overlapping quadrants. "
          "All 6,800 planned generation records are retained. Each follow-up starts from the "
          "same fresh direct conversation, and branch order is shuffled within image."),
        p("Measured main result", "LA_H2"), p(prose["finding"]),
    ]
    rows = [["Condition", "Mean accuracy", "Change vs direct", "Fixes / harms"]]
    direct = summary["main"]["direct"]
    rows.append([LABELS["direct"], f"{100 * direct['expected_accuracy']:.2f}%", "Reference", "-"])
    for key in ORDER:
        row = summary["main"]["conditions"][key]
        rows.append([LABELS[key], f"{100 * row['expected_accuracy']:.2f}%",
                     f"{row['delta_vs_direct_pp']:+.2f} pp",
                     f"{row['fix_presentations']} / {row['harm_presentations']}"])
    story.extend([table(rows, [3.1 * inch, 1.1 * inch, 1.35 * inch, 1.55 * inch]),
                  p("Condition scores average four quadrants per image, then all 400 images. "
                    "Each fix/harm count has 1,600 image-quadrant presentations as denominator; "
                    "these are not 1,600 independent images or an implemented region selector.", "LA_Small"),
                  PageBreak(), p("Paired evidence", "LA_Title"),
                  p("Prespecified contrasts and answer transitions", "LA_Subtitle"),
                  figure("direction_conditions_and_contrasts.png", 4.05 * inch),
                  p("Intervals: 10,000 paired source-image bootstrap draws, seed 20260925; "
                    "95% percentile intervals, unadjusted for multiple comparisons. All four "
                    "quadrants and all conditions stay together when an image is resampled.", "LA_Small"),
                  figure("direction_fixes_and_harms.png", 3.9 * inch),
                  p("Wrong-to-right and right-to-wrong transitions are measured against the same "
                    "fresh direct answer. Their difference equals the change in average accuracy. "
                    "An intervention can repair some answers and still reduce overall accuracy.", "LA_Small"),
                  PageBreak(), p("Interpretation and scope", "LA_Title"),
                  p("Observed behavior, not a claim about hidden reasoning", "LA_Subtitle"),
                  p("Secondary diagnostics", "LA_H2"), p(prose["secondary_findings"]),
                  p("Case inspection", "LA_H2"), p(prose["case_audit"]),
                  p("Limits on the conclusion", "LA_H2"),
                  p("The sham incorrectly calls a full-image copy a particular crop. Named minus "
                    "sham changes image content and image/text congruity; it cannot isolate the "
                    "benefit of new pixels. A neutral sham is absent, so the design does not "
                    "identify an image-content by naming interaction. Identical-pixel contrasts "
                    "show prompt sensitivity, not an internal reasoning mechanism."),
                  p("This is the same convenience development panel used in v1, not held-out "
                    "confirmation. Its median source long edge is 500 pixels; enlarging these "
                    "images creates no new source detail. Exact match can penalize alternative "
                    "valid answers, and referent ambiguity remains. No controller has been "
                    "trained and no domain or backbone transfer has been measured."),
                  p("Execution and reproducibility", "LA_H2"), p(prose["execution"]),
                  p("Qwen3-VL-4B-Instruct, frozen BF16, SDPA, greedy decoding, 32 output tokens; "
                    "approximately 1,024 visual tokens per view. The raw run, source/manifest "
                    "hashes, prompts, output continuations, grades, actual grids, timings and "
                    "memory measurements accompany this note."),
                  p("Next decision", "LA_H2"), p(prose["decision"]),
                  p("Evidence map", "LA_H2"),
                  p("reports/direction400/report.md: complete tables, intervals, subgroups and "
                    "v1 comparison. docs/direction_controls_protocol.md: rules locked before "
                    "generation. reports/direction400/decision_and_cases.md: authored case "
                    "audit. docs/development_data_options.md: conditional data candidates.", "LA_Small"),
                  p("Run fingerprint: " + summary["run_fingerprint"], "LA_Small"),
                  p("Sources: Qwen model card - huggingface.co/Qwen/Qwen3-VL-4B-Instruct; "
                    "GQA - cs.stanford.edu/people/dorarad/gqa/about.html. Upstream terms apply; "
                    "weights and source images are not included in the source archive.", "LA_Small")])

    def page_frame(canvas, doc):
        canvas.saveState()
        page_width, page_height = doc.pagesize
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(1.1)
        canvas.line(doc.leftMargin, page_height - 26, page_width - doc.rightMargin, page_height - 26)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(GRAY)
        canvas.drawString(doc.leftMargin, 23, "LookAgain | Development diagnostic v2")
        canvas.drawRightString(page_width - doc.rightMargin, 23, str(doc.page))
        canvas.restoreState()

    doc = SimpleDocTemplate(str(output), pagesize=(8.3 * inch, 11.7 * inch),
                            leftMargin=0.6 * inch, rightMargin=0.6 * inch,
                            topMargin=0.65 * inch, bottomMargin=0.55 * inch,
                            title="LookAgain: direction controls on a frozen VLM",
                            author="RenataLi", subject="Development diagnostic, 400 images")
    doc.build(story, onFirstPage=page_frame, onLaterPages=page_frame)
    print(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--interpretation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.summary, args.interpretation, args.output)


if __name__ == "__main__":
    main()
