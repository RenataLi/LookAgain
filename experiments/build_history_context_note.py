"""Three-page note from a completed v4 analysis and run-bound interpretation."""
import argparse
from html import escape
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak


LABELS = {"direct": "Direct overview", "highres": "High-resolution full page", "repeat": "Actual: repeat overview",
          "actual_native": "Actual: source region", "actual_degraded": "Actual: degraded region",
          "fresh_native": "Fresh: source region", "fresh_degraded": "Fresh: degraded region",
          "placeholder_native": "Placeholder: source region", "placeholder_degraded": "Placeholder: degraded region"}


def build(summary_path, output):
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    authored = json.loads((summary_path.parent / "interpretation.json").read_text(encoding="utf-8"))
    if data["status"] != "complete" or authored["run_fingerprint"] != data["bindings"]["run_fingerprint"]:
        raise ValueError("Incomplete analysis or interpretation belongs to another run")
    styles = getSampleStyleSheet()
    for name, size, leading, after, color, font in (
        ("Title4", 26, 30, 14, "#174f5b", "Helvetica-Bold"),
        ("Deck4", 12, 16, 12, "#435864", "Helvetica"),
        ("Body4", 10, 14, 8, "#263b48", "Helvetica"),
        ("Head4", 13, 17, 8, "#174f5b", "Helvetica-Bold"),
        ("Small4", 8, 11, 6, "#546975", "Helvetica"),
        ("Cell4", 8.5, 11, 0, "#263b48", "Helvetica"),
    ):
        styles.add(ParagraphStyle(name=name, fontName=font, fontSize=size, leading=leading,
                                  spaceAfter=after, textColor=colors.HexColor(color)))
    p = lambda text, style="Body4": Paragraph(text, styles[style])

    def table(rows, widths):
        item = Table([[p(escape(str(x)), "Cell4") for x in row] for row in rows], colWidths=widths, repeatRows=1)
        item.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e5f0f1")),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f8f9")]),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 7), ("RIGHTPADDING", (0, 0), (-1, -1), 7),
            ("TOPPADDING", (0, 0), (-1, -1), 7), ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ]))
        return item

    def effect(stat):
        lo, hi = stat["ci95_pp"]
        return f"{stat['difference_pp']:+.2f} pp [{lo:+.2f}, {hi:+.2f}]"

    primary = data["contrasts"][data["primary_contrast"]]["metrics"]["conservative_text_em"]
    story = [p("LOOKAGAIN / RESEARCH NOTE 04", "Small4"),
             p("Do previous answers limit<br/>the value of another view?", "Title4"),
             p("History content, request format and visual detail in a frozen VLM", "Deck4"),
             p("Primary interaction: <b>" + effect(primary) + "</b>", "Deck4"),
             p("This interaction measures how replacing the actual previous answer with a fixed acknowledgement changes the accuracy advantage of source-render detail over an overview-derived control. A positive value is not, by itself, proof that the model uses new detail more effectively."),
             p(escape(authored["conclusion"])), p("Absolute quality and measured cost", "Head4")]
    rows = [["Condition", "Strict EM", "Official EM", "Two-stage time", "Standalone time"]]
    for name, label in LABELS.items():
        item = data["conditions"][name]
        rows.append([label, f"{100*item['metrics']['conservative_text_em']:.2f}%", f"{100*item['metrics']['official_em']:.2f}%",
                     f"{item['decision_state_latency_s']['mean']:.3f} s", f"{item['standalone_latency_s']['mean']:.3f} s"])
    story += [table(rows, [184, 64, 74, 86, 87]), Spacer(1, 9),
              p("100 reused development reports; 2,700 calls. Every crop condition averages four fixed regions within each report: expected uniform single-region selection, not a learned selector. All 100 original references remain in the primary result.", "Small4"),
              p("Two-stage time charges the initial direct answer plus the chosen branch. Standalone fresh/placeholder use only one call; actual/repeat require two. Highres and direct are standalone comparators in both columns. Times are local full-path means, not equal-compute comparisons.", "Small4"),
              PageBreak(), p("What changes, what stays fixed", "Title4"),
              p("A complete 3 x 2 x 4 intervention crosses history, regional pixel source and region. Three standalone/control actions bring the total to 27 calls per report. All regional branches use the same overview and matched region geometry.")]
    rows = [["History", "Previous assistant content", "Request structure"],
            ["Actual", "Fresh direct response, unchanged", "Overview/question; assistant answer; region and reconsider instruction"],
            ["Placeholder", "I have viewed the page.", "Same roles and user messages as Actual"],
            ["Fresh", "No previous answer", "One user turn: overview + region + description/question"]]
    story += [table(rows, [80, 160, 255]), Spacer(1, 9),
              p("Native regions sample the fixed 200-DPI source rendering. Degraded regions are rebuilt only from the already-seen overview. Within each history, native/degraded messages, view sizes, grids and input-token counts match. Across histories the pixels match, while text and turn structure intentionally differ."),
              p("Absolute source-detail effects", "Head4")]
    rows = [["History", "Native minus degraded", "Source-region fixes / harms"]]
    for history in ("actual", "fresh", "placeholder"):
        stat = data["detail_effects"][history]["conservative_text_em"]
        changes = data["changes_vs_direct"][history + "_native"]
        rows.append([history.capitalize(), effect(stat), f"{changes['fixes']} / {changes['harms']} (400 presentations)"])
    story += [table(rows, [85, 195, 215]), Spacer(1, 9),
              p("Fixes and harms compare each branch with direct. The 400 presentations are clustered within 100 source reports. Metric changes can reflect punctuation, units or paraphrases; selected cases are inspected separately."),
              p("Prior-answer persistence is descriptive", "Head4"),
              p("Matching a previous answer does not prove copying, and changing it does not prove the new pixels caused the change. Fresh versus actual bundles changes in history, wording and image/text placement. The placeholder preserves layout but changes content and length; it is not a semantically inert, token-matched intervention."),
              PageBreak(), p("Evidence, limits and next decision", "Title4"),
              p("A reused development panel", "Head4"),
              p("The same TAT-DQA training reports were studied in v3, and those outcomes motivated v4. This is an outcome-informed diagnostic, not independent confirmation or held-out transfer. The source-only quality mask was fixed before v3 inference and is unchanged here.")]
    sensitivity = data["source_quality_sensitivity"]
    sp = sensitivity["contrasts"][data["primary_contrast"]]["metrics"]["conservative_text_em"]
    story += [p("On the 94-report quality sensitivity, the primary interaction is <b>" + effect(sp) + "</b>. Six previously flagged label-error/ambiguous examples are excluded only from this secondary result."),
              p("Analysis and integrity", "Head4"),
              p("Strict EM lowercases and collapses whitespace while preserving punctuation and numeric formatting. Secondary official single-span EM/F1 use the unchanged pinned author evaluator; ANLS is diagnostic. Invalid/truncated responses remain in denominators. Intervals use 10,000 paired bootstrap samples of whole reports, seed 20260925; secondary analyses are exploratory and unadjusted."),
              p("The completed analysis reconstructs every input's pixel/chat provenance and re-scores every raw response. Smoke-source CPU checks verify both image blocks and distinct native/degraded tensors. These checks do not capture production activations or establish an internal attention mechanism."),
              p(escape(authored["uncertainty_note"])),
              p("Advancement rule and practical interpretation", "Head4"),
              p("A history qualifies for untouched-report validation only if its native-minus-degraded strict-EM effect is at least 2 pp, its paired interval lower endpoint is positive, and its native mean is at least direct. A positive history interaction alone is insufficient. The rule does not authorize automatic controller training."),
              p(escape(authored["next_step"])),
              p("Sources and reproducibility", "Head4"),
              p('<link href="https://nextplusplus.github.io/TAT-DQA/" color="#176b76">TAT-DQA authors</link> | '
                '<link href="https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct" color="#176b76">Qwen3-VL-4B-Instruct</link><br/>'
                '<link href="https://arxiv.org/html/2605.15864v1" color="#176b76">VisualSwap / visual re-examination (2026 preprint)</link><br/>'
                '<link href="https://arxiv.org/html/2608.01930v1" color="#176b76">Recompute or Reuse? (2026 preprint)</link>', "Small4"),
              p("Repository: docs/history_context_protocol.md; docs/history_context_execution.md; docs/history_context_related_work.md; configs/history_context.json; reports/history100/{run.json,records.jsonl,summary.json,interpretation.json}. Exact model, code, input and record hashes are included. No novelty, learned localization or generalization is established.", "Small4")]
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(output), pagesize=A4, rightMargin=50, leftMargin=50, topMargin=42, bottomMargin=43,
                            title="LookAgain: History and the value of another view", author="LookAgain research project")

    def footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d3dfe4"))
        canvas.line(50, 35, A4[0]-50, 35)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#647785"))
        canvas.drawString(50, 23, "LOOKAGAIN / V4 DEVELOPMENT DIAGNOSTIC / 26 SEPTEMBER 2026")
        canvas.drawRightString(A4[0]-50, 23, str(document.page))
        canvas.restoreState()
    doc.build(story, onFirstPage=footer, onLaterPages=footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    build(args.summary, args.output)
