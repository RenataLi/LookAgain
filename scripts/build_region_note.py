"""Render a five-page region-selection note, without inference or metric changes.

Production requires a completed evaluation summary and its bound real figures.
--synthetic is only for an explicitly marked work-directory layout fixture.
Run the PDF skill's authoring marker before a new PDF authoring operation.
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
TEAL = colors.HexColor('#096D81')
LINE = colors.HexColor('#D6E0E6')
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN = 41
WIDTH = PAGE_WIDTH - 2*MARGIN
REQUIRED = ('Abstract', 'Design and inputs', 'Primary evaluation', 'Costs and confidence gates',
            'Selector behavior and controls', 'Limits and reproducibility')
BREAK_BEFORE = {'Primary evaluation', 'Costs and confidence gates', 'Selector behavior and controls', 'Limits and reproducibility'}
FIGURES = {'region_quality.png': 287, 'region_quality_cost.png': 385, 'region_selector_choices.png': 305}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def register_fonts(font_dir):
    candidates = [Path(font_dir)] if font_dir else [Path('C:/Windows/Fonts'), Path('/usr/share/fonts/truetype/dejavu')]
    for directory in candidates:
        for files in [('arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf'),
                      ('DejaVuSans.ttf','DejaVuSans-Bold.ttf','DejaVuSans-Oblique.ttf','DejaVuSans-BoldOblique.ttf')]:
            if all((directory/f).is_file() for f in files):
                names = ('RegionNote','RegionNote-Bold','RegionNote-Italic','RegionNote-BoldItalic')
                for name, filename in zip(names, files):
                    pdfmetrics.registerFont(TTFont(name,str(directory/filename)))
                pdfmetrics.registerFontFamily('RegionNote',normal=names[0],bold=names[1],italic=names[2],boldItalic=names[3])
                return {filename:sha(directory/filename) for filename in files}
    raise ValueError('Provide Arial or DejaVu Sans regular/bold/italic/bold-italic TTF files with --font-dir')


def inline(value):
    value = value.replace('\u2013','-').replace('\u2014','-').replace('\u2011','-').replace('\u2212','-')
    links = []
    def link(match):
        label, target = match.group(1), match.group(2)
        if target.startswith(('https://','http://')):
            token = 'REGIONLINKPLACEHOLDER'+str(len(links))
            links.append((token,'<link href="'+escape(target,quote=True)+'" color="#096D81">'+escape(label)+'</link>'))
            return token
        return label
    value = re.sub(r'\[([^\]]+)\]\(([^)]+)\)',link,value)
    value = escape(value)
    value = re.sub(r'`([^`]+)`',r'<font size="8.0">\1</font>',value)
    value = re.sub(r'\*\*([^*]+)\*\*',r'<b>\1</b>',value)
    value = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)',r'<i>\1</i>',value)
    for token, rendered in links:
        value = value.replace(token,rendered)
    return value


def build(source, output, summary_path=None, font_dir=None, synthetic=False):
    source, output = Path(source).resolve(), Path(output).resolve()
    require(not output.exists() and not output.with_suffix('.build.json').exists(),'Use a new output path')
    text = source.read_text(encoding='utf-8')
    require(all(text.count('## '+heading+'\n') == 1 for heading in REQUIRED),'Unexpected note section structure')
    require(len(re.findall(r'^## ',text,re.MULTILINE)) == len(REQUIRED),'Unexpected extra section')
    if synthetic:
        require('SYNTHETIC LAYOUT' in text,'Synthetic fixture must be explicitly marked')
        summary_hash, source_bindings = None, None
    else:
        require(summary_path is not None,'Production requires --summary')
        require(not re.search(r'\[PENDING|SYNTHETIC LAYOUT|TODO|TBD',text),'Unfinished content in production note')
        summary = json.loads(Path(summary_path).read_text(encoding='utf-8'))
        require(summary['status']=='completed_evaluation_pilot' and summary['role']=='evaluation'
                and summary['n_sources']==60 and summary['n_records']==780,'Completed N60 evaluation required')
        require(summary['new_model_calls']==0 and summary['automatic_controller_training'] is False,'Report scope differs')
        summary_hash, source_bindings = sha(summary_path), summary['bindings']
    figure_meta_path = source.parent/'figures/figures_metadata.json'
    figure_meta = json.loads(figure_meta_path.read_text(encoding='utf-8'))
    require(figure_meta['synthetic'] is synthetic,'Figure reality status differs')
    if not synthetic:
        require(figure_meta['summary_sha256']==summary_hash,'Figures refer to a different summary')
    font_hashes = register_fonts(font_dir)
    body = ParagraphStyle('Body',fontName='RegionNote',fontSize=9.35,leading=13.05,
                          textColor=INK,spaceAfter=7.2,allowWidows=0,allowOrphans=0)
    small = ParagraphStyle('Small',parent=body,fontSize=7.9,leading=10.8,textColor=MUTED,spaceAfter=5)
    caption = ParagraphStyle('Caption',parent=body,fontSize=8.1,leading=11.25,textColor=MUTED,spaceAfter=8)
    heading = ParagraphStyle('Heading',parent=body,fontName='RegionNote-Bold',fontSize=12.5,leading=16,
                             textColor=TEAL,spaceBefore=8,spaceAfter=8,keepWithNext=True)
    title = ParagraphStyle('Title',parent=body,fontName='RegionNote-Bold',fontSize=22,leading=26.3,
                           spaceAfter=12,keepWithNext=True)
    cell_style = ParagraphStyle('Cell',parent=body,fontSize=8.1,leading=10.8,spaceAfter=0)
    story, figure_hashes = [], {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1; continue
        if line.startswith('# '):
            story.extend([Paragraph('LOOKAGAIN / GIVEN-PAGE REGION SELECTION',small),
                          Paragraph(inline(line[2:]),title),
                          Paragraph('SYNTHETIC LAYOUT - NOT EXPERIMENTAL RESULTS' if synthetic else
                                    '60 evaluation source clusters | Frozen Qwen3-VL-4B | Geometry-only region bank',small)])
        elif line.startswith('## '):
            label = line[3:]
            if label in BREAK_BEFORE:
                story.append(PageBreak())
            story.append(Paragraph(inline(label),heading))
        elif line.startswith('|'):
            rows = []
            while i < len(lines) and lines[i].strip().startswith('|'):
                cells = [c.strip() for c in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r'[:\-\s]+',c) for c in cells):
                    rows.append(cells)
                i += 1
            require(rows and all(len(row)==len(rows[0]) for row in rows),'Bad Markdown table')
            n = len(rows[0])
            widths = {2:[.62,.38],3:[.45,.27,.28],4:[.43,.18,.18,.21],5:[.36,.16,.16,.16,.16]}.get(n,[1/n]*n)
            table = Table([[Paragraph(('<b>'+inline(c)+'</b>') if k==0 else inline(c),cell_style) for c in row]
                           for k,row in enumerate(rows)],colWidths=[WIDTH*w for w in widths],repeatRows=1,hAlign='LEFT')
            table.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#EAF2F4')),
                ('LINEBELOW',(0,0),(-1,0),.7,TEAL),('LINEBELOW',(0,1),(-1,-1),.25,LINE),
                ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),6),
                ('RIGHTPADDING',(0,0),(-1,-1),6),('TOPPADDING',(0,0),(-1,-1),4.5),
                ('BOTTOMPADDING',(0,0),(-1,-1),4.5)]))
            story.extend([table,Spacer(1,8)])
            continue
        elif line.startswith('!['):
            match = re.fullmatch(r'!\[([^\]]*)\]\(([^)]+)\)',line)
            require(match is not None,'Unsupported image syntax')
            relative = match.group(2)
            image_path = (source.parent/relative).resolve()
            require(image_path.is_relative_to(source.parent/'figures') and image_path.name in FIGURES,'Unexpected figure')
            require(image_path.name not in {Path(x).name for x in figure_hashes},'Duplicate figure')
            digest = sha(image_path)
            require(digest==figure_meta['figures'][image_path.name],'Figure hash differs')
            figure_hashes[relative] = digest
            with PILImage.open(image_path) as im:
                w,h = im.size
            display_w, display_h = WIDTH, WIDTH*h/w
            cap = FIGURES[image_path.name]
            if display_h>cap:
                display_w *= cap/display_h; display_h = cap
            story.extend([Spacer(1,3),Image(str(image_path),display_w,display_h),Spacer(1,4)])
        elif line.startswith('```'):
            code = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code.append(lines[i]); i += 1
            require(i < len(lines),'Unclosed code fence')
            story.append(Paragraph('<br/>'.join(escape(x) for x in code),small))
        else:
            paragraph = [line]
            while i+1<len(lines) and lines[i+1].strip() and not lines[i+1].lstrip().startswith(('#','|','![','```','- ')):
                i += 1; paragraph.append(lines[i].strip())
            value = ' '.join(paragraph)
            bullet = value.startswith('- ')
            story.append(Paragraph(inline(value[2:] if bullet else value),
                                   caption if value.startswith('*Figure') else body,
                                   bulletText='-' if bullet else None))
        i += 1
    require({Path(x).name for x in figure_hashes}==set(FIGURES),'All three figures must appear exactly once')
    story.extend([Spacer(1,4),Paragraph('Document provenance: layout rendering of research_note.md; '
                  'the source statistics and frozen artifacts are unchanged. Source SHA-256: '+sha(source),small)])
    page_labels = {1:'Question, cohorts and input privilege',2:'Primary quality comparison',3:'Measured cost components',
                   4:'Selection behavior and matched control',5:'Interpretation and reproduction boundaries'}
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(LINE); canvas.setLineWidth(.6)
        canvas.line(MARGIN,PAGE_HEIGHT-29,PAGE_WIDTH-MARGIN,PAGE_HEIGHT-29)
        canvas.setFont('RegionNote',7.3); canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN,PAGE_HEIGHT-22,'LOOKAGAIN / GIVEN-PAGE REGION SELECTION')
        canvas.drawRightString(PAGE_WIDTH-MARGIN,PAGE_HEIGHT-22,'SYNTHETIC LAYOUT' if synthetic else 'RESEARCH NOTE')
        canvas.line(MARGIN,31,PAGE_WIDTH-MARGIN,31)
        canvas.drawString(MARGIN,20,page_labels.get(doc.page,'Research note'))
        canvas.drawRightString(PAGE_WIDTH-MARGIN,20,f'{doc.page} / 5')
        canvas.restoreState()
    with BytesIO() as stream:
        doc = SimpleDocTemplate(stream,pagesize=A4,rightMargin=MARGIN,leftMargin=MARGIN,topMargin=43,bottomMargin=43,
                                title='LookAgain: given-page region selection',author='LookAgain research project',
                                subject='Synthetic layout only' if synthetic else 'Small outcome-held-out document-QA pilot; page remains privileged',pageCompression=1)
        doc.build(story,onFirstPage=decorate,onLaterPages=decorate)
        reader = PdfReader(stream)
        require(len(reader.pages)==5,f'Expected five pages; generated {len(reader.pages)}. Check text length and page layout.')
        writer = PdfWriter(); writer.clone_document_from_reader(reader)
        writer.add_metadata({'/SourceMarkdownSHA256':sha(source),'/BuilderSHA256':sha(__file__),
                             '/SummarySHA256':summary_hash or 'SYNTHETIC',
                             '/FigureSHA256':json.dumps(figure_hashes,sort_keys=True),'/SyntheticLayout':str(synthetic)})
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('xb') as target:
            writer.write(target)
    metadata = {'created_utc':datetime.now(timezone.utc).isoformat(),'source_markdown_sha256':sha(source),
                'summary_sha256':summary_hash,'source_bindings':source_bindings,'builder_sha256':sha(__file__),
                'figure_sha256':figure_hashes,'figures_metadata_sha256':sha(figure_meta_path),'font_sha256':font_hashes,
                'pdf_sha256':sha(output),'pages':5,'synthetic':synthetic,'new_model_calls':0,
                'visual_qa':'Page rendering and visual inspection are separate from the structural PDF checks.'}
    with output.with_suffix('.build.json').open('x',encoding='utf-8',newline='\n') as target:
        json.dump(metadata,target,ensure_ascii=False,indent=2); target.write('\n')
    print(json.dumps(metadata,ensure_ascii=False,indent=2))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--summary',type=Path)
    parser.add_argument('--font-dir',type=Path)
    parser.add_argument('--synthetic',action='store_true')
    args = parser.parse_args()
    build(args.source,args.output,args.summary,args.font_dir,args.synthetic)


if __name__=='__main__':
    main()
