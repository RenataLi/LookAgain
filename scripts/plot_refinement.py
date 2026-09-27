"""Two static plots from a completed exposed-cohort refinement summary.

No inference, fitting, confidence intervals for cross-fitted policies, or source
media. --synthetic marks layout-only input explicitly; never use it for results.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

POLICIES = (
    ('direct', 'Overview answer'),
    ('center', 'Center window'),
    ('random', 'Seeded random window'),
    ('uniform', 'Uniform expected window'),
    ('prompted', 'Frozen prompted selector'),
    ('best_fixed', 'Best fixed window (OOF)'),
    ('ranker_position', 'Position ridge (OOF)'),
    ('ranker_image_position', 'Image + position ridge (OOF)'),
    ('ranker_full', 'Full ridge (OOF)'),
    ('highres', 'High-resolution answer'),
)
COLORS = {'direct': '#64748B', 'center': '#64748B', 'random': '#64748B', 'uniform': '#64748B',
          'prompted': '#BA4A32', 'best_fixed': '#967545', 'ranker_position': '#8460A8',
          'ranker_image_position': '#448AA8', 'ranker_full': '#174E78', 'highres': '#287D61'}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(value, message):
    if not value:
        raise ValueError(message)


def quality(summary, policy, version):
    return 100 * summary['strategies'][policy + '_' + version]['metrics']['primary_em']['mean']


def theme():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.labelcolor': '#243545', 'text.color': '#243545',
                         'xtick.color': '#46586A', 'ytick.color': '#46586A',
                         'axes.edgecolor': '#B6C3CE', 'grid.color': '#DEE5EB',
                         'grid.linewidth': .6, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'savefig.facecolor': 'white'})


def validate_summary(summary, synthetic=False):
    require(summary['status'] == 'completed_development_diagnostic' and summary['role'] == 'main', 'Completed development summary required')
    require(summary['n_sources'] == 124 or synthetic, 'Expected all 124 exposed development sources')
    require(summary['primary_contrast'] == 'ranker_full_new_minus_prompted_new', 'Unexpected primary contrast')
    for policy, _ in POLICIES:
        for version in ('old', 'new'):
            q = quality(summary, policy, version)
            require(math.isfinite(q) and 0 <= q <= 100, 'Invalid EM value')
        latency = summary['strategies'][policy + '_new']['cost']['elapsed_s']['mean']
        require(math.isfinite(latency) and latency > 0, 'Invalid measured component latency')
    require('ci95' not in json.dumps(summary['primary']).lower(), 'Primary cross-fit CI is not supported')


def banner(fig, summary, synthetic):
    label = ('SYNTHETIC LAYOUT CHECK — no experimental results' if synthetic else
             f"{summary['n_sources']} previously exposed sources · development OOF · provided answer page")
    fig.text(.02, .985, label, ha='left', va='top', fontsize=9, color='#9E3B32' if synthetic else '#576B7B')


def save(fig, output, name, source_hash, synthetic):
    files = []
    for suffix in ('png', 'pdf'):
        path = output / (name + '.' + suffix)
        metadata = ({'Title': 'LookAgain exposed-cohort refinement: ' + name,
                     'Subject': 'Development diagnostic; summary SHA256 ' + source_hash,
                     'Author': 'LookAgain', 'Keywords': 'development, OOF, no cross-fit CI' + (', synthetic' if synthetic else '')}
                    if suffix == 'pdf' else {'Description': 'Summary SHA256: ' + source_hash + '; development OOF; synthetic=' + str(synthetic)})
        fig.savefig(path, dpi=180, metadata=metadata)
        files.append({'path': path.name, 'sha256': sha(path), 'bytes': path.stat().st_size})
    plt.close(fig)
    return files


def render(summary_path, output, synthetic=False):
    summary_path, output = Path(summary_path), Path(output)
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    validate_summary(summary, synthetic)
    output.mkdir(parents=True, exist_ok=True)
    require(not (output / 'figures_metadata.json').exists(), 'Use a fresh figure directory')
    theme()
    source_hash = sha(summary_path)
    artifacts = []
    primary = 100 * summary['primary']['primary_em']['mean_difference']

    fig, ax = plt.subplots(figsize=(10.4, 7.3))
    fig.subplots_adjust(left=.31, right=.91, bottom=.19, top=.85)
    y = np.arange(len(POLICIES))
    old = [quality(summary, name, 'old') for name, _ in POLICIES]
    new = [quality(summary, name, 'new') for name, _ in POLICIES]
    ax.barh(y - .18, old, height=.31, color='#CDD6DF', label='Archived instruction')
    bars = ax.barh(y + .18, new, height=.31, color=[COLORS[name] for name, _ in POLICIES], label='Revised extraction instruction')
    ax.set_yticks(y, [label for _, label in POLICIES])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xticks(np.arange(0, 101, 20))
    ax.set_xlabel('Exact match (%) · unchanged automatic scorer')
    ax.grid(axis='x', zorder=0)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, new):
        ax.text(value + 1 if value < 94 else value - 1, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}', va='center', ha='left' if value < 94 else 'right', fontsize=8.5)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(.305, .909), frameon=False, ncols=2, fontsize=8.7)
    fig.suptitle('Extraction instruction and frozen region choices', x=.31, y=.94, ha='left', fontsize=15, fontweight='bold')
    banner(fig, summary, synthetic)
    fig.text(.02, .047, 'Old/new comparisons use identical per-source region IDs. Learned and best-fixed policies use five-fold predictions trained on old-bank labels.\n'
             'All sources were already examined; prompt design used these data. No cross-fit confidence intervals or independent generalization claim.',
             ha='left', va='bottom', fontsize=8.6, linespacing=1.6)
    artifacts.extend(save(fig, output, 'quality_bars', source_hash, synthetic))

    fig, ax = plt.subplots(figsize=(10.4, 7.3))
    fig.subplots_adjust(left=.10, right=.70, bottom=.25, top=.85)
    latencies = [summary['strategies'][name + '_new']['cost']['elapsed_s']['mean'] for name, _ in POLICIES]
    # Numbered labels and a side key remain readable even when cheap controls coincide.
    offsets = [(6, -12), (24, 3), (22, -19), (-27, -26), (7, 9), (-4, 25),
               (-27, 10), (25, 0), (7, 10), (7, 8)]
    for index, ((name, label), latency, value, offset) in enumerate(zip(POLICIES, latencies, new, offsets), 1):
        marker = 'D' if name == 'ranker_full' else 's' if name == 'highres' else 'o'
        ax.scatter(latency, value, s=68 if name in ('ranker_full', 'highres', 'prompted') else 45,
                   marker=marker, color=COLORS[name], edgecolor='white', linewidth=.7, zorder=3, clip_on=False)
        if value < 3:
            offset = (offset[0], 9)
        elif value > 97:
            offset = (offset[0], -14)
        ax.annotate(str(index), (latency, value), xytext=offset, textcoords='offset points',
                    fontsize=8.5, color=COLORS[name], fontweight='bold',
                    arrowprops={'arrowstyle': '-', 'color': COLORS[name], 'lw': .5, 'alpha': .65,
                                'shrinkA': 2, 'shrinkB': 5})
        fig.text(.735, .825 - (index - 1)*.040, f'{index:>2}  {label}', fontsize=8.8, color=COLORS[name], va='top')
    ax.set_xlim(0, max(latencies)*1.13)
    ax.set_ylim(0, 100)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_xlabel('Mean assembled per-source latency (seconds)')
    ax.set_ylabel('Exact match (%)')
    ax.grid(True)
    fig.suptitle('Answer quality and measured component cost', x=.10, y=.94, ha='left', fontsize=15, fontweight='bold')
    banner(fig, summary, synthetic)
    fig.text(.735, .36, f'Primary OOF contrast\nFull ridge − prompted\n{primary:+.2f} percentage points\n\nDescriptive; no naive CI',
             va='top', fontsize=10, linespacing=1.5, color='#243545')
    fig.text(.02, .065,
             'Full/image rankers: feature forward + pooling/projection + measured CPU ranking + selected answer. Position/cheap controls: CPU choice + answer.\n'
             'Prompted: current selector replay + answer at the archived selected ID (a timing proxy if replay changes ID). Uniform: expected one-crop cost.\n'
             'Measured components are assembled, not a timed end-to-end deployment. Warmups and offline fitting are separate. Hindsight oracles are omitted.\n'
             'Provided answer page; reused development data; string metrics are not a semantic correctness audit.',
             ha='left', va='bottom', fontsize=8.4, linespacing=1.6)
    artifacts.extend(save(fig, output, 'quality_cost', source_hash, synthetic))

    metadata = {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
                'summary_sha256': source_hash, 'plot_code_sha256': sha(__file__),
                'summary_path': str(summary_path), 'synthetic_layout_only': synthetic,
                'n_sources': summary['n_sources'], 'scope': 'reused_development_OOF_no_crossfit_CI',
                'quality_metric': 'unchanged primary exact match', 'new_model_calls': 0,
                'figures': artifacts,
                'limits': ['No independent evaluation is implied.', 'Cost is assembled from measured components.',
                           'OOF policies share training sources across folds; no inferential error bars are plotted.',
                           'All answer-page candidates use geometry, with no answer-ROI annotations at inference.']}
    (output / 'figures_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--synthetic', action='store_true', help='Explicitly mark layout-only synthetic input')
    args = parser.parse_args()
    result = render(args.summary, args.output, args.synthetic)
    print(json.dumps({'figures': len(result['figures']), 'synthetic': result['synthetic_layout_only']}))


if __name__ == '__main__':
    main()
