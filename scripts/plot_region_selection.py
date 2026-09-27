"""Static plots of completed region-selection summaries; never loads a VLM.

All plotted policy costs are assembled from measured invocation components.
Use --synthetic only for an explicitly synthetic, non-evidence layout fixture.
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


COLORS = {'direct': '#607181', 'center_native': '#8499A4',
          'uniform_expected_native': '#B58344', 'selected_native': '#096D81',
          'selected_degraded': '#CD713E', 'highres': '#7953A5',
          'confidence_gate_0.25': '#237D6B', 'confidence_gate_0.5': '#329A7D',
          'confidence_gate_0.75': '#69AF93'}
LABELS = {'direct': 'Direct overview', 'center_native': 'Fixed center',
          'uniform_expected_native': 'Uniform expectation', 'selected_native': 'Selected native',
          'selected_degraded': 'Selected degraded', 'highres': 'High-resolution page',
          'confidence_gate_0.25': 'Gate q25', 'confidence_gate_0.5': 'Gate q50',
          'confidence_gate_0.75': 'Gate q75'}
QUALITY = ('direct', 'center_native', 'uniform_expected_native', 'selected_native', 'selected_degraded', 'highres')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate(summary, synthetic):
    require(summary.get('role') in ('development', 'evaluation'), 'Engineering quality plots are prohibited')
    require(summary.get('status') == 'completed_' + summary['role'] + '_pilot', 'Completed development/evaluation summary required')
    require(type(summary.get('n_sources')) is int and summary['n_sources'] > 0, 'Invalid source count')
    require(summary.get('n_records') == 13 * summary['n_sources'], 'Incomplete thirteen-action panel')
    require(summary.get('synthetic', False) is synthetic, 'Synthetic mode must be explicitly and consistently declared')
    if not synthetic:
        bindings = summary['bindings']
        for key in ('manifest_sha256', 'lock_sha256', 'run_sha256', 'records_sha256', 'completed_sha256', 'analysis_code_sha256'):
            require(isinstance(bindings.get(key), str) and len(bindings[key]) == 64
                    and all(c in '0123456789abcdef' for c in bindings[key]), 'Missing summary binding: ' + key)
    for name in QUALITY:
        require(name in summary['strategies'], 'Missing policy: ' + name)
    for name, value in summary['strategies'].items():
        if name not in LABELS:
            continue
        score = value['metrics']['primary_em']['mean']
        latency = value['costs']['elapsed_s']['mean']
        require(number(score) and 0 <= score <= 1, 'Invalid score for ' + name)
        require(number(latency) and latency > 0, 'Invalid cost for ' + name)
    primary = summary['primary_result']
    require(primary['contrast'] == 'selected_native_minus_uniform_expected_native' and primary['metric'] == 'primary_em', 'Wrong primary contrast')
    require(number(primary['difference']) and -1 <= primary['difference'] <= 1, 'Invalid primary difference')
    require(isinstance(primary['ci95'], list) and len(primary['ci95']) == 2
            and all(number(x) and -1 <= x <= 1 for x in primary['ci95'])
            and primary['ci95'][0] <= primary['ci95'][1], 'Invalid paired interval')
    counts = summary['selector_region_counts']
    require(set(counts) == {str(i) for i in range(1,10)} and all(type(x) is int and x >= 0 for x in counts.values()), 'Invalid selector histogram')
    require(sum(counts.values()) == summary['n_sources'], 'Selector histogram does not cover every source')
    fallback = summary['selector_invalid_or_truncated']
    require(type(fallback) is int and 0 <= fallback <= counts['5'], 'Fallback count incompatible with center decisions')
    for key, gate in summary.get('confidence_gates', {}).items():
        require('confidence_gate_' + key in summary['strategies'], 'Missing gate strategy')
        require(number(gate['zoom_fraction']) and 0 <= gate['zoom_fraction'] <= 1, 'Invalid observed zoom fraction')


def decorate(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#B2BBC3')
    ax.spines['bottom'].set_color('#B2BBC3')
    ax.tick_params(colors='#314659')
    ax.set_axisbelow(True)


def header(summary, synthetic):
    scope = 'development diagnostic' if summary['role'] == 'development' else 'small outcome-held-out evaluation'
    prefix = 'SYNTHETIC LAYOUT - NOT RESULTS\n' if synthetic else ''
    return prefix + f"Given-page region selection: {summary['n_sources']} source clusters\n{scope}; annotation-selected answer page"


def render(summary_path, output, synthetic=False):
    summary_path, output = Path(summary_path).resolve(), Path(output).resolve()
    summary = json.loads(summary_path.read_text(encoding='utf-8'))
    validate(summary, synthetic)
    require(not output.exists(), 'Use a fresh figure directory; no overwrite of archived figures')
    output.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 10.5, 'axes.titlesize': 13,
                         'figure.facecolor': 'white', 'axes.facecolor': 'white',
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    files = []
    def save(fig, name):
        for suffix in ('png', 'pdf'):
            path = output / f'{name}.{suffix}'
            kwargs = {'dpi': 190} if suffix == 'png' else {}
            fig.savefig(path, facecolor='white', **kwargs)
            files.append(path)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(9.7, 6.2))
    fig.subplots_adjust(left=.235, right=.935, top=.80, bottom=.30)
    y = np.arange(len(QUALITY))
    values = [100 * summary['strategies'][name]['metrics']['primary_em']['mean'] for name in QUALITY]
    ax.barh(y, values, color=[COLORS[name] for name in QUALITY], height=.6)
    ax.set_yticks(y, [LABELS[name] for name in QUALITY])
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel('Custom source-validated exact match (%)')
    ax.grid(axis='x', color='#E2E6EA', linewidth=.7)
    decorate(ax)
    for position, value in zip(y, values):
        inside = value > 91
        ax.text(value - 1.3 if inside else value + 1.1, position, f'{value:.2f}%',
                va='center', ha='right' if inside else 'left', color='white' if inside else '#203547', fontsize=10)
    fig.suptitle(header(summary, synthetic), y=.965, fontsize=13, linespacing=1.5)
    p = summary['primary_result']
    fig.text(.235, .115,
             f"Primary selected - uniform: {100*p['difference']:+.2f} pp\n"
             f"95% paired source-bootstrap CI [{100*p['ci95'][0]:+.2f}, {100*p['ci95'][1]:+.2f}] pp",
             fontsize=11, color='#096D81', linespacing=1.5)
    fig.text(.235, .035, 'Bars show point estimates. The paired contrast supplies inference.\n'
             'Uniform = per-source mean over nine crops; not nine independent observations.',
             fontsize=8.8, color='#506176', linespacing=1.45)
    save(fig, 'region_quality')

    names = [name for name in LABELS if name in summary['strategies']]
    fig, ax = plt.subplots(figsize=(10.1, 7.1))
    fig.subplots_adjust(left=.10, right=.96, top=.82, bottom=.34)
    markers = {'direct': 's', 'center_native': 'D', 'uniform_expected_native': 'X',
               'selected_native': 'o', 'selected_degraded': 'o', 'highres': '*',
               'confidence_gate_0.25': '^', 'confidence_gate_0.5': 'v', 'confidence_gate_0.75': 'P'}
    latencies = []
    for name in names:
        value = summary['strategies'][name]
        latency = value['costs']['elapsed_s']['mean']; latencies.append(latency)
        label = LABELS[name]
        if name.startswith('confidence_gate_'):
            key = name.removeprefix('confidence_gate_')
            label += f" (zoom {100*summary['confidence_gates'][key]['zoom_fraction']:.1f}%)"
        ax.scatter(latency, 100*value['metrics']['primary_em']['mean'],
                   s=130 if name == 'highres' else 78, marker=markers[name],
                   color=COLORS[name], edgecolor='white', linewidth=.55, label=label, zorder=3)
    ax.set_xlim(0, max(latencies)*1.12)
    ax.set_ylim(0,100)
    ax.set_xlabel('Mean policy latency from measured components (seconds)')
    ax.set_ylabel('Custom source-validated exact match (%)')
    ax.grid(color='#E2E6EA', linewidth=.7)
    decorate(ax)
    ax.legend(loc='upper center', bbox_to_anchor=(.5,-.22), ncol=3, frameon=False,
              fontsize=9, columnspacing=1.8, handletextpad=.7, labelspacing=.8)
    fig.suptitle(header(summary, synthetic), y=.965, fontsize=13, linespacing=1.5)
    fig.text(.10,.09, 'Selected policies: full selector + crop. Gates: direct, then selector + crop only when zooming.\n'
             'Fixed policies have no VLM selector charge. Hindsight oracles are excluded from this figure.',
             fontsize=8.8, color='#506176', linespacing=1.45)
    fig.text(.10,.026, 'Component sums are not directly timed deployment pipelines; active-desktop measurements.\n'
             'Gate q25/q50/q75 denote development quantiles, not guaranteed evaluation zoom fractions.',
             fontsize=8.8, color='#506176', linespacing=1.45)
    save(fig, 'region_quality_cost')

    fig, ax = plt.subplots(figsize=(9.7,5.8))
    fig.subplots_adjust(left=.09, right=.965, top=.80, bottom=.30)
    total = np.asarray([summary['selector_region_counts'][str(i)] for i in range(1,10)])
    fallback = np.zeros(9, dtype=int); fallback[4] = summary['selector_invalid_or_truncated']
    valid = total - fallback
    ax.bar(range(1,10), valid, width=.66, color='#096D81', label='Valid selector response')
    ax.bar(range(1,10), fallback, bottom=valid, width=.66, color='#CE7D41', label='Invalid/truncated -> center fallback')
    ax.set_xticks(range(1,10), ['1\nTop left','2\nTop center','3\nTop right',
                               '4\nMiddle left','5\nCenter','6\nMiddle right',
                               '7\nBottom left','8\nBottom center','9\nBottom right'], fontsize=8.7)
    ax.set_ylabel('Source clusters')
    ax.set_ylim(0, max(1,int(total.max())) * 1.19)
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.grid(axis='y', color='#E2E6EA', linewidth=.7)
    decorate(ax)
    for region, count in enumerate(total,1):
        ax.text(region, count + max(1,total.max())*.025, str(int(count)), ha='center', va='bottom', fontsize=9.5)
    fig.suptitle(header(summary, synthetic), y=.965, fontsize=13, linespacing=1.5)
    ax.legend(loc='upper center', bbox_to_anchor=(.5,-.25), ncol=2, frameon=False, fontsize=9.1)
    count = summary['selector_invalid_or_truncated']
    fig.text(.09,.037, f'Fallback: {count}/{summary["n_sources"]} sources ({100*count/summary["n_sources"]:.1f}%). '
             'Valid center choices remain distinct.\nNine overlapping half-width, half-height windows (at most 25% of page area); row-major IDs.\n'
             'Counts do not measure evidence localization accuracy.',
             fontsize=8.8, color='#506176', linespacing=1.45)
    save(fig, 'region_selector_choices')

    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'summary_sha256': sha(summary_path),
                'script_sha256': sha(__file__), 'role': summary['role'], 'n_sources': summary['n_sources'],
                'synthetic': synthetic, 'source_bindings': summary.get('bindings'),
                'figures': {path.name: sha(path) for path in files},
                'new_model_calls': 0,
                'scope': 'Presentation of a supplied completed-summary artifact, not an independent reconstruction of raw records/pixels.',
                'cost_scope': 'Paired sums/means of measured components; no direct end-to-end deployment timing.',
                'inference_scope': 'Primary source-paired CI annotated; policy point estimates are not independent-bar hypothesis tests.'}
    (output/'figures_metadata.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'status':'rendered','figures':len(files),'synthetic':synthetic,'new_model_calls':0}))
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--synthetic', action='store_true')
    args = parser.parse_args()
    render(args.summary, args.output, args.synthetic)


if __name__ == '__main__':
    main()
