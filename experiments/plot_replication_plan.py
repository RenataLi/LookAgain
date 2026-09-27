"""Plot prospective source availability and exact power; no VLM results."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, default=Path('reports/replication_plan/power_plan.json'))
    parser.add_argument('--sources', type=Path, default=Path('reports/replication_plan/source_feasibility.json'))
    parser.add_argument('--output', type=Path, default=Path('reports/replication_plan'))
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    sources = json.loads(args.sources.read_text(encoding='utf-8'))
    assert plan['status'] == 'prospective_planning_only_no_v6_inference_or_model_results'
    available = plan['feasibility']['at_most_original_reports_before_semantic_audit']
    assert available == 62, 'This figure describes the audited TAT-DQA source pool'
    assert sources['remaining']['strict_eligible_reports'] == available
    rows = [row for row in plan['scenarios'] if row['true_delta'] == .05]
    assert len(rows) == 3
    args.output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, (left, right) = plt.subplots(1, 2, figsize=(12.6, 5.7), gridspec_kw={'width_ratios': [1, 1.3]})
    fig.subplots_adjust(left=.23, right=.975, top=.79, bottom=.20, wspace=.40)
    fig.suptitle('Independent replication needs a larger source pool', x=.035, ha='left', y=.96,
                 fontsize=18, fontweight='bold')
    fig.text(.035, .895, 'Prospective planning only  |  No new model outcomes  |  One question per original report',
             fontsize=10.5, color='#526070')

    labels = ['Training release', 'Unused source IDs', 'Metadata eligible', 'Strict ROI candidate']
    values = [sources['release']['original_report_source_count'],
              sources['remaining']['source_upper_bound_before_filters'],
              sources['remaining']['metadata_eligible_reports'],
              sources['remaining']['strict_eligible_reports']]
    left.barh(range(4), values, height=.57, color=['#CDD5DF', '#8EA2BA', '#5B789C', '#244A70'])
    left.set_yticks(range(4), labels)
    left.invert_yaxis()
    left.set_xlim(0, 195)
    left.set_xlabel('Original source reports')
    left.set_title('A. Local TAT-DQA inventory', loc='left', pad=16, fontsize=12, fontweight='bold')
    left.spines['left'].set_visible(False)
    left.tick_params(axis='y', length=0, pad=10)
    for i, value in enumerate(values):
        left.text(value + 3, i, str(value), va='center', fontweight='bold')
    left.grid(axis='x', alpha=.16)
    left.set_axisbelow(True)

    colors = ['#187E85', '#315EA7', '#BF6736']
    for row, color in zip(rows, colors):
        curve = row['sample_size_search']['positive_direction_power_curve']
        xs = list(map(int, curve))
        ys = list(curve.values())
        q = row['total_discordance_q']
        n80 = row['sample_size_search']['minimum_n_for_80pct']
        right.plot(xs, ys, color=color, lw=2, label=f'Discordance {q:.0%}  |  N80 = {n80}')
        right.scatter([available], [curve[str(available)]], color=color, s=30, zorder=5)
    right.axhline(.8, color='#576371', ls='--', lw=1)
    right.text(880, .81, '80% target', ha='right', fontsize=9, color='#576371')
    right.axvline(available, color='#576371', ls=':', lw=1)
    right.text(available + 11, .40, 'At most 62\nnew reports', fontsize=9, color='#384351')
    right.set(xlim=(0, 900), ylim=(0, 1.02), xlabel='Independent original reports',
              ylabel='Probability of detecting benefit')
    right.set_title('B. True gain +5 percentage points', loc='left', pad=16, fontsize=12, fontweight='bold')
    right.yaxis.set_major_formatter(PercentFormatter(1))
    right.grid(alpha=.16)
    right.legend(loc='lower right', fontsize=8.5, frameon=False)

    fig.text(.035, .092,
        'Power: positive-direction rejection by a two-sided exact McNemar test, alpha = .05. Curves assume independent pairs.',
        fontsize=9, color='#526070')
    fig.text(.035, .055,
        'The 62-report ceiling precedes semantic review. N80 values are planning requirements, not available cohorts or inference budgets.',
        fontsize=9, color='#526070')
    for suffix in ('png', 'svg'):
        fig.savefig(args.output / f'replication_feasibility.{suffix}', dpi=180, facecolor='white')
    plt.close(fig)
    provenance = {'purpose': 'Prospective planning figure; not new model results',
        'power_plan_sha256': hashlib.sha256(args.plan.read_bytes()).hexdigest(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'source_counts': dict(zip(labels, values)),
        'source_audit_sha256': hashlib.sha256(args.sources.read_bytes()).hexdigest(),
        'files': {f'replication_feasibility.{s}': hashlib.sha256((args.output / f'replication_feasibility.{s}').read_bytes()).hexdigest()
                  for s in ('png', 'svg')}}
    (args.output / 'figure_provenance.json').write_text(json.dumps(provenance, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
