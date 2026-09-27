"""Render the completed DUDE Markdown note as a five-page research PDF.

No inference or source-statistic modification. Requires reportlab, Pillow and
pypdf; --font-dir may point to Arial TTFs or DejaVu Sans TTFs. Relative document
links are shown as readable text: the accompanying repository is authoritative.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from html import escape
from io import BytesIO
import json
from pathlib import Path
import re

from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


INK = colors.HexColor('#172D42')
MUTED = colors.HexColor('#506176')
TEAL = colors.HexColor('#146C75')
LINE = colors.HexColor('#D6E0E6')
PAPER_WIDTH, PAPER_HEIGHT = A4
MARGIN = 43
WIDTH = PAPER_WIDTH - 2 * MARGIN


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fonts(font_dir):
    candidates = [Path(font_dir)] if font_dir else [Path('C:/Windows/Fonts'), Path('/usr/share/fonts/truetype/dejavu')]
    for directory in candidates:
        for files in [('arial.ttf', 'arialbd.ttf', 'ariali.ttf', 'arialbi.ttf'),
                      ('DejaVuSans.ttf', 'DejaVuSans-Bold.ttf', 'DejaVuSans-Oblique.ttf', 'DejaVuSans-BoldOblique.ttf')]:
            if all((directory / f).is_file() for f in files):
                names = ('Note', 'Note-Bold', 'Note-Italic', 'Note-BoldItalic')
                for name, filename in zip(names, files):
                    pdfmetrics.registerFont(TTFont(name, str(directory / filename)))
                pdfmetrics.registerFontFamily('Note', normal=names[0], bold=names[1], italic=names[2], boldItalic=names[3])
                return {filename: sha(directory / filename) for filename in files}
    raise ValueError('Provide --font-dir containing Arial or DejaVu Sans regular/bold/italic/bold-italic TTF files')


def inline(value):
    # Retain source meaning while making punctuation portable in PDF typography.
    value = value.replace('\u2013', '-').replace('\u2014', '-').replace('\u2011', '-').replace('\u2212', '-')
    value = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', value)
    value = escape(value)
    value = re.sub(r'`([^`]+)`', r'<font size="8.7">\1</font>', value)
    value = re.sub(r'\*\*([^*]+)\*\*', r'<b>\1</b>', value)
    value = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'<i>\1</i>', value)
    return value


def build(source, output, font_dir=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    font_hashes = fonts(font_dir)
    text = source.read_text(encoding='utf-8')
    source_hash = sha(source)
    required = ['## Abstract', '## Model and conditions', '## Primary result',
                '## Descriptive comparisons and response audit', '## Compute and completion',
                '## Interpretation and reproducibility']
    if any(text.count(marker) != 1 for marker in required):
        raise ValueError('Expected completed DUDE note structure is absent or duplicated')

    body = ParagraphStyle('Body', fontName='Note', fontSize=9.6, leading=13.7,
                          textColor=INK, spaceAfter=7.8, allowWidows=0, allowOrphans=0)
    small = ParagraphStyle('Small', parent=body, fontSize=8.2, leading=11.3, textColor=MUTED, spaceAfter=5)
    caption = ParagraphStyle('Caption', parent=small, fontSize=8.7, leading=12.2, spaceAfter=11)
    heading = ParagraphStyle('Heading', parent=body, fontName='Note-Bold', fontSize=12.2,
                             leading=16, textColor=TEAL, spaceBefore=10, spaceAfter=8, keepWithNext=True)
    title = ParagraphStyle('Title', parent=body, fontName='Note-Bold', fontSize=22.5,
                           leading=27, textColor=INK, spaceAfter=13, keepWithNext=True)
    table_style = ParagraphStyle('Cell', parent=body, fontSize=8.3, leading=11.1, spaceAfter=0)
    story = []
    figure_hashes = {}
    lines = text.splitlines()
    i = 0
    page_labels = {1: 'Question and source cohort', 2: 'Conditions and primary analysis',
                   3: 'Quality and comparison baseline', 4: 'Paired evidence and semantic audit',
                   5: 'Compute, interpretation and limits'}
    page_break_headings = {'Model and conditions', 'Descriptive comparisons and response audit', 'Compute and completion'}

    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('# '):
            story += [Paragraph('LOOKAGAIN  /  RESEARCH NOTE', small),
                      Paragraph(inline(line[2:]), title),
                      Paragraph('660 source clusters  |  2,640 main invocations  |  Frozen Qwen3-VL-4B', small)]
        elif line.startswith('## '):
            label = line[3:]
            if label in page_break_headings:
                story.append(PageBreak())
            story.append(Paragraph(inline(label), heading))
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [cell.strip() for cell in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\-\s]+', cell) for cell in cells):
                    rows.append(cells)
                i += 1
            n = len(rows[0])
            if any(len(row) != n for row in rows):
                raise ValueError('Inconsistent Markdown table width')
            if n == 2:
                widths = [WIDTH * .57, WIDTH * .43]
            elif n == 3:
                widths = [WIDTH * .16, WIDTH * .57, WIDTH * .27]
            elif n == 4 and 'Latency' in rows[0][1]:
                widths = [WIDTH * .17, WIDTH * .39, WIDTH * .22, WIDTH * .22]
            elif n == 4:
                widths = [WIDTH * .23, WIDTH * .23, WIDTH * .22, WIDTH * .32]
            else:
                widths = [WIDTH / n] * n
            cells = [[Paragraph('<b>' + inline(c) + '</b>' if k == 0 else inline(c), table_style)
                      for c in row] for k, row in enumerate(rows)]
            table = Table(cells, colWidths=widths, repeatRows=1, hAlign='LEFT')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#EAF2F4')),
                ('LINEBELOW', (0,0), (-1,0), .7, TEAL),
                ('LINEBELOW', (0,1), (-1,-1), .25, LINE),
                ('VALIGN', (0,0), (-1,-1), 'TOP'),
                ('LEFTPADDING', (0,0), (-1,-1), 7), ('RIGHTPADDING', (0,0), (-1,-1), 7),
                ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6),
            ]))
            story += [table, Spacer(1, 10)]
            continue
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)', line)
            if not match:
                raise ValueError('Unsupported image syntax')
            relative = match.group(2)
            path = (source.parent / relative).resolve()
            if not path.is_relative_to(source.parent):
                raise ValueError('Figure escapes report directory')
            if path.name == 'paired_contrasts.png':
                story += [PageBreak(), Paragraph('Paired differences and response review', heading)]
            figure_hashes[relative] = sha(path)
            with PILImage.open(path) as image:
                width, height = image.size
            rendered_width = WIDTH
            rendered_height = WIDTH * height / width
            if rendered_height > 330:
                rendered_width *= 330 / rendered_height
                rendered_height = 330
            story += [Spacer(1, 6), Image(str(path), rendered_width, rendered_height), Spacer(1, 5)]
        else:
            paragraph = [line]
            while i + 1 < len(lines) and lines[i+1].strip() and not lines[i+1].lstrip().startswith(('#', '|', '![')):
                i += 1
                paragraph.append(lines[i].strip())
            value = ' '.join(paragraph)
            story.append(Paragraph(inline(value), caption if value.startswith('*Figure') else body))
        i += 1

    story += [Spacer(1, 7), Paragraph('Provenance: this PDF is a layout rendering of the accompanying research_note.md. '
                                    'All quantitative and semantic source artifacts remain unchanged. Source SHA-256: '
                                    + source_hash, small)]
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE)
        canvas.setLineWidth(.6)
        canvas.line(MARGIN, PAPER_HEIGHT-29, PAPER_WIDTH-MARGIN, PAPER_HEIGHT-29)
        canvas.setFont('Note', 7.6)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, PAPER_HEIGHT-22, 'LOOKAGAIN / DUDE MATCHED-DETAIL REPLICATION')
        canvas.drawRightString(PAPER_WIDTH-MARGIN, PAPER_HEIGHT-22, 'RESEARCH NOTE')
        canvas.line(MARGIN, 32, PAPER_WIDTH-MARGIN, 32)
        canvas.drawString(MARGIN, 20, page_labels.get(doc.page, 'Research note'))
        canvas.drawRightString(PAPER_WIDTH-MARGIN, 20, f'{doc.page} / 5')
        canvas.restoreState()

    output.parent.mkdir(parents=True, exist_ok=True)
    with BytesIO() as draft:
        doc = SimpleDocTemplate(draft, pagesize=A4, rightMargin=MARGIN, leftMargin=MARGIN,
                                topMargin=43, bottomMargin=43, title='LookAgain: a prospective matched-detail replication',
                                author='LookAgain research project', subject=f'Source Markdown SHA256 {source_hash}',
                                pageCompression=1)
        doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
        reader = PdfReader(draft)
        if len(reader.pages) != 5:
            raise ValueError(f'Expected five inspected-layout pages; generated {len(reader.pages)}')
        writer = PdfWriter()
        writer.clone_document_from_reader(reader)
        writer.add_metadata({'/SourceMarkdownSHA256': source_hash, '/BuilderSHA256': sha(__file__),
                             '/FigureSHA256': json.dumps(figure_hashes, sort_keys=True),
                             '/Subject': 'Completed prospective DUDE replication. Privileged answer page and ROI; no learned controller.'})
        with output.open('wb') as stream:
            writer.write(stream)
    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'source_markdown_sha256': source_hash,
                'builder_sha256': sha(__file__), 'figure_sha256': figure_hashes,
                'font_sha256': font_hashes, 'pdf_sha256': sha(output), 'pages': 5,
                'new_model_calls': 0, 'visual_qa': 'Must be rendered and inspected separately after generation.'}
    output.with_suffix('.build.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metadata, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--font-dir', type=Path)
    args = parser.parse_args()
    build(args.source, args.output, args.font_dir)


if __name__ == '__main__':
    main()
