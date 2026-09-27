"""Build a wholly synthetic four-step illustration, never a model result.

Requirements: Pillow. Optional --font-dir points to Segoe UI or DejaVu fonts.
The synthetic page is drawn here from invented company/financial content.
No dataset, grades, model, scores, or network access is used.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import os

from PIL import Image, ImageDraw, ImageFont

W, H = 1200, 680
NAVY = '#0b1928'
PANEL = '#142a3b'
INK = '#173349'
TEXT = '#edf5fb'
MUTED = '#a2bacb'
TEAL = '#73e1ce'
AMBER = '#efbd73'


def font(size: int, bold: bool = False, font_dir: Path | None = None):
    roots = ([font_dir] if font_dir else []) + [
        Path(os.environ.get('WINDIR', 'C:/Windows')) / 'Fonts',
        Path('/usr/share/fonts/truetype/dejavu'),
        Path('/usr/share/fonts/truetype/liberation2'),
    ]
    names = ['segoeuib.ttf', 'DejaVuSans-Bold.ttf', 'LiberationSans-Bold.ttf'] if bold else [
        'segoeui.ttf', 'DejaVuSans.ttf', 'LiberationSans-Regular.ttf']
    for root in roots:
        for name in names:
            path = root / name
            if path.is_file():
                return ImageFont.truetype(str(path), size)
    raise RuntimeError('A Segoe UI, DejaVu Sans, or Liberation Sans font is required; use --font-dir.')


def build(font_dir=None):
    f = lambda size, bold=False: font(size, bold, font_dir)
    # The answer occupies the center half-page window, with its row and year.
    source = Image.new('RGB', (720, 1000), '#fbfcfb')
    d = ImageDraw.Draw(source)
    d.rectangle((0, 0, 720, 16), fill='#2c827c')
    d.text((65, 65), 'NORTHLINE', font=f(30, True), fill=INK)
    d.text((65, 115), 'Annual report', font=f(42, True), fill=INK)
    d.text((65, 179), 'Illustrative company · synthetic page', font=f(20), fill='#68808c')
    d.line((65, 229, 655, 229), fill='#cbd8de', width=2)
    d.text((194, 278), 'FINANCIAL SUMMARY', font=f(20, True), fill='#2c827c')
    d.text((194, 319), 'Fiscal year 2025', font=f(28, True), fill=INK)
    d.rounded_rectangle((181, 378, 541, 414), radius=5, fill='#e3eeef')
    d.text((194, 384), 'Metric', font=f(19, True), fill='#35596a')
    d.text((462, 384), '2025', font=f(19, True), fill='#35596a')
    rows = [('Operating revenue', '$24.8M'), ('Operating costs', '$17.2M'), ('Operating income', '$7.6M')]
    for index, (label, value) in enumerate(rows):
        y = 444 + index * 73
        d.text((194, y), label, font=f(21, index == 0), fill=INK)
        d.text((447, y), value, font=f(21, index == 0), fill=INK)
        d.line((194, y + 45, 533, y + 45), fill='#dce5e7', width=2)
    d.text((194, 697), 'All amounts shown in USD millions.', font=f(17), fill='#68808c')
    d.line((65, 801, 655, 801), fill='#cbd8de', width=2)
    for y, length in [(838, 553), (861, 497), (884, 536)]:
        d.rounded_rectangle((65, y, 65 + length, y + 6), radius=3, fill='#c5d2d8')
    d.text((65, 946), 'SCHEMATIC · NOT A SOURCE DOCUMENT', font=f(15, True), fill='#8396a0')

    # Three starts per axis, each crop exactly half the original width/height.
    windows = [(x, y, x + 360, y + 500) for y in (0, 250, 500) for x in (0, 180, 360)]
    assert windows[4] == (180, 250, 540, 750)
    chosen_crop = source.crop(windows[4])
    titles = ['Start with the overview', 'Compare nine windows', 'Commit region 5', 'Read the selected detail']
    notes = [
        ['A page and a question enter', 'the visual workflow.'],
        ['Fixed half-page windows overlap.', 'The overview provides context.'],
        ['This illustration chooses the', 'center window before answering.'],
        ['Combine page context with', 'the native-resolution crop.'],
    ]
    frames = []

    def paste_page(im, xy, size, overview=False):
        page = source
        if overview:
            # Merely a visual shorthand for a low-resolution overview, not a
            # simulation of an actual VLM processor or its token allocation.
            page = page.resize((180, 250), Image.Resampling.LANCZOS)
        page = page.resize(size, Image.Resampling.LANCZOS)
        shadow = ImageDraw.Draw(im)
        shadow.rounded_rectangle((xy[0] + 6, xy[1] + 8, xy[0] + size[0] + 6, xy[1] + size[1] + 8), radius=8, fill='#06121e')
        im.paste(page, xy)

    for step in range(4):
        im = Image.new('RGB', (W, H), NAVY)
        draw = ImageDraw.Draw(im)
        for x in range(0, W, 40):
            draw.line((x, 0, x, H), fill='#102334')
        for y in range(0, H, 40):
            draw.line((0, y, W, y), fill='#102334')
        draw.text((38, 24), 'LOOKAGAIN', font=f(18, True), fill=TEAL)
        draw.text((38, 59), 'Where should the model look?', font=f(31, True), fill=TEXT)
        draw.rounded_rectangle((701, 27, 1162, 65), radius=18, fill='#302b25', outline='#6f5b40')
        draw.text((719, 34), 'Schematic walkthrough · synthetic document', font=f(17, True), fill=AMBER)
        draw.line((38, 113, 1162, 113), fill='#284254', width=1)
        draw.rounded_rectangle((38, 146, 479, 568), radius=17, fill=PANEL, outline='#355064')
        draw.text((60, 169), f'STEP 0{step + 1} / 04', font=f(15, True), fill=TEAL)
        draw.text((60, 206), titles[step], font=f(28, True), fill=TEXT)
        for i, line in enumerate(notes[step]):
            draw.text((60, 253 + 27 * i), line, font=f(21), fill=MUTED)
        draw.line((60, 320, 456, 320), fill='#355064')
        draw.text((60, 341), 'QUESTION', font=f(14, True), fill='#80a4b8')
        draw.text((60, 368), 'What was operating revenue', font=f(22), fill=TEXT)
        draw.text((60, 398), 'in 2025?', font=f(22), fill=TEXT)
        if step == 3:
            draw.rounded_rectangle((59, 449, 457, 543), radius=12, fill='#1b403f', outline='#3a8579')
            draw.text((77, 460), 'ILLUSTRATIVE ANSWER', font=f(14, True), fill=TEAL)
            draw.text((77, 481), '$24.8M', font=f(42, True), fill='#e3fff7')
        else:
            lower = ['Page context first', '9 candidates · no scores shown', 'One selected crop'][step]
            draw.rounded_rectangle((60, 466, 457, 527), radius=11, fill='#1d3a4d')
            draw.text((78, 483), lower, font=f(20, True), fill='#b7d0dc')

        if step < 3:
            x, y, pw, ph = 704, 142, 302, 420
            draw.text((704, 118), 'OVERVIEW', font=f(13, True), fill='#87a9bb')
            paste_page(im, (x, y), (pw, ph), overview=True)
            draw = ImageDraw.Draw(im)
            if step == 1:
                overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
                od = ImageDraw.Draw(overlay)
                for i, box in enumerate(windows):
                    xx0, yy0, xx1, yy1 = x + box[0] * pw / 720, y + box[1] * ph / 1000, x + box[2] * pw / 720, y + box[3] * ph / 1000
                    od.rectangle((xx0, yy0, xx1, yy1), fill=(47, 174, 161, 11), outline=(25, 129, 117, 230), width=2)
                im = Image.alpha_composite(im.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(im)
                for i, box in enumerate(windows):
                    xx, yy = x + box[0] * pw / 720 + 8, y + box[1] * ph / 1000 + 8
                    draw.rounded_rectangle((xx, yy, xx + 26, yy + 28), radius=5, fill='#176c67')
                    draw.text((xx + 7, yy + 2), str(i + 1), font=f(18, True), fill='#e8fff8')
            elif step == 2:
                xx0, yy0, xx1, yy1 = x + pw / 4, y + ph / 4, x + pw * .75, y + ph * .75
                overlay = Image.new('RGBA', im.size, (0, 0, 0, 0))
                od = ImageDraw.Draw(overlay)
                for box in [(x, y, x + pw, yy0), (x, yy1, x + pw, y + ph), (x, yy0, xx0, yy1), (xx1, yy0, x + pw, yy1)]:
                    od.rectangle(box, fill=(4, 19, 33, 103))
                im = Image.alpha_composite(im.convert('RGBA'), overlay).convert('RGB')
                draw = ImageDraw.Draw(im)
                draw.rectangle((xx0, yy0, xx1, yy1), outline=AMBER, width=4)
                draw.rounded_rectangle((xx0 + 8, yy0 + 8, xx0 + 88, yy0 + 36), radius=5, fill='#805d2e')
                draw.text((xx0 + 17, yy0 + 10), 'REGION 5', font=f(13, True), fill='#fff0ce')
                draw.line((xx1 + 4, yy0 + 110, 1079, yy0 + 110), fill=AMBER, width=2)
                draw.text((1028, yy0 + 124), 'center', font=f(18), fill=AMBER)
                draw.text((1028, yy0 + 150), 'window', font=f(18), fill=AMBER)
        else:
            paste_page(im, (530, 192), (202, 280), overview=True)
            draw = ImageDraw.Draw(im)
            draw.text((530, 162), 'OVERVIEW', font=f(15, True), fill='#87a9bb')
            draw.rectangle((580.5, 262, 681.5, 402), outline=AMBER, width=3)
            draw.line((739, 331, 788, 331), fill=TEAL, width=3)
            draw.polygon([(791, 331), (778, 324), (778, 338)], fill=TEAL)
            draw.text((806, 133), 'NATIVE CROP · REGION 5', font=f(16, True), fill=TEAL)
            crop = chosen_crop.resize((315, 437), Image.Resampling.LANCZOS)
            im.paste(crop, (806, 163))
            draw = ImageDraw.Draw(im)
            draw.rectangle((804, 161, 1122, 601), outline=TEAL, width=2)
            # Highlight the exact operating-revenue row in the enlarged crop.
            row_y0 = 163 + int((431 - 250) * 437 / 500)
            row_y1 = 163 + int((489 - 250) * 437 / 500)
            draw.rounded_rectangle((813, row_y0, 1114, row_y1), radius=5, outline='#2b9a82', width=3)
            draw.text((538, 502), 'context', font=f(18), fill='#87a9bb')
            draw.text((538, 532), '+ detail', font=f(18, True), fill=TEAL)

        draw = ImageDraw.Draw(im)
        labels = ['OVERVIEW', 'CANDIDATE WINDOWS', 'SELECT A REGION', 'REVISIT + ANSWER']
        for index, label in enumerate(labels):
            start = 39 + index * 288
            active = index == step
            draw.rounded_rectangle((start, 609, start + 273, 643), radius=8, fill='#285c58' if active else '#132b3d')
            draw.text((start + 12, 616), f'{index + 1}  {label}', font=f(14, True), fill='#ccfff0' if active else '#718fa2')
        draw.text((39, 657), 'Overview + native crop · one selected region · frozen vision-language model', font=f(13), fill='#7b98aa')
        frames.append(im)
    return frames


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1] / 'assets')
    parser.add_argument('--font-dir', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    frames = build(args.font_dir)
    frames[-1].save(args.output / 'demo_poster.png')
    frames[0].save(args.output / 'lookagain_demo.gif', save_all=True, append_images=frames[1:],
                   duration=[2400, 3000, 2600, 3600], loop=0, disposal=2, optimize=False)
    sheet = Image.new('RGB', (1200, 680))
    for i, frame in enumerate(frames):
        sheet.paste(frame.resize((600, 340), Image.Resampling.LANCZOS), ((i % 2) * 600, (i // 2) * 340))
    sheet.save(args.output.parent / 'demo_preview.png')
    print('Created 1200x680 schematic GIF (4 steps, 11.6 seconds) and final-frame poster.')


if __name__ == '__main__':
    main()
