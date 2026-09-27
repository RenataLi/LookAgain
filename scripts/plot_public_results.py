"""Build the README figure directly from the released numerical results."""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / 'reports/benchmark/expected_summary.json'


def main():
    results = json.loads(SUMMARY.read_text(encoding='utf-8'))['strategies']
    rows = [('direct_new', 'Direct overview'), ('uniform_new', 'Uniform crop'),
            ('prompted_new', 'Prompted selector'), ('ranker_image_position_new', 'Image + position'),
            ('ranker_full_new', 'LookAgain ranker'), ('highres_new', 'High-resolution page')]
    quality = [100 * results[k]['metrics']['primary_em']['mean'] for k, _ in rows]
    latency = [results[k]['cost']['elapsed_s']['mean'] for k, _ in rows]
    colors = ['#8091a8', '#a7b4c6', '#596d88', '#65bdb8', '#008e88', '#dd9831']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.spines.left': False, 'axes.edgecolor': '#d4dce7',
                         'text.color': '#172c43', 'axes.labelcolor': '#53677f',
                         'xtick.color': '#53677f', 'ytick.color': '#172c43'})
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), gridspec_kw={'width_ratios': [1.14, 1]})
    fig.patch.set_facecolor('#f8fafc')
    for ax in axes: ax.set_facecolor('#f8fafc')
    y = np.arange(len(rows))
    for ax, values, bound, title, formatter in [
        (axes[0], quality, 100, 'ANSWER QUALITY', lambda v: f'{v:.2f}%'),
        (axes[1], latency, 1.92, 'MEASURED COMPUTE', lambda v: f'{v:.3f} s')]:
        ax.barh(y, values, color=colors, height=.61, zorder=3)
        ax.set_xlim(0, bound)
        ax.set_ylim(-.7, len(rows)-.3)
        ax.invert_yaxis()
        ax.set_yticks(y, [label for _, label in rows] if ax is axes[0] else [])
        ax.tick_params(axis='both', length=0, pad=9)
        ax.grid(axis='x', color='#e4eaf1', linewidth=.8, zorder=0)
        ax.set_title(title, loc='left', fontsize=11, fontweight='bold', pad=15)
        for yy, value in zip(y, values):
            ax.text(value + bound*.025, yy, formatter(value), va='center', fontsize=11,
                    fontweight='bold' if yy == 4 else 'normal')
    axes[0].set_xlabel('Custom exact match (%)', labelpad=12)
    axes[1].set_xlabel('Mean assembled latency per source (s)', labelpad=12)
    fig.text(.025, .95, 'LookAgain  /  quality and compute', fontsize=21, fontweight='bold')
    fig.text(.025, .895, '124 reused development sources  ·  five-fold OOF region ranking  ·  frozen Qwen3-VL-4B',
             fontsize=11, color='#53677f')
    fig.text(.025, .035, 'RTX 5090 · Latency sums measured policy components. Uniform crop is the within-source expectation.',
             fontsize=9, color='#53677f')
    fig.subplots_adjust(left=.205, right=.97, top=.77, bottom=.19, wspace=.16)
    target = ROOT / 'assets'
    target.mkdir(exist_ok=True)
    fig.savefig(target/'results.png', dpi=180, facecolor=fig.get_facecolor())
    fig.savefig(target/'results.svg', facecolor=fig.get_facecolor(), metadata={'Date': None})
    plt.close(fig)
    print('Built assets/results.png and assets/results.svg from the released summary.')


if __name__ == '__main__':
    main()
