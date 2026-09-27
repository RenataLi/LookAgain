"""Media-free independent arithmetic audit of the completed N60 evaluation.

Shares only the frozen DUDE scorer/answer parser. Does not import the main
analyzer, runner, selector core, NumPy, model weights, or an image processor.
It validates record/summary bindings and arithmetic, not source pixels or the
original execution lock. Run the main strict validator first.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / 'experiments'), str(PROJECT / 'src')]
from dude_metrics import score_response, validate_reference_sets

ACTIONS = ('direct', 'selector', *(f'native_{i}' for i in range(1, 10)), 'selected_degraded', 'highres')
METRICS = ('primary_em', 'official_anls')
ADDITIVE = ('elapsed_s', 'input_tokens', 'visual_tokens', 'generated_tokens', 'processor_observer_elapsed_s')
PEAKS = ('peak_memory_gib', 'peak_reserved_gib')
RAW_COSTS = ADDITIVE[:-1] + PEAKS
ALL_COSTS = ADDITIVE + PEAKS
FRACTIONS = ('0.25', '0.5', '0.75')
PRIMARY = 'selected_native_minus_uniform_expected_native'
OBSERVATION_KEYS = {'example_id', 'source_cluster_id', 'question', 'image_path', 'image_sha256'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def decode(text):
    return json.loads(text, object_pairs_hook=no_duplicate_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def read(path):
    return decode(Path(path).read_text(encoding='utf-8'))


def jsonlines(path):
    return [decode(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Timezone-aware timestamp required')
    return result


def quantile(values, probability):
    """Standalone scalar linear interpolation, not NumPy or analyzer code."""
    ordered = sorted(values)
    require(bool(ordered) and all(finite(v) for v in ordered), 'Empty/nonfinite quantile data')
    require(0 <= probability <= 1, 'Bad quantile probability')
    position = (len(ordered) - 1) * probability
    lo = math.floor(position)
    hi = math.ceil(position)
    return float(ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo))


def describe(values):
    values = list(values)
    return {'n': len(values), 'mean': math.fsum(values) / len(values),
            'median': quantile(values, .5), 'p95': quantile(values, .95),
            'min': min(values), 'max': max(values)}


def strict_region(record):
    require(isinstance(record['response'], str) and type(record['generation_truncated']) is bool,
            'Invalid selector metadata')
    matched = None if record['generation_truncated'] else re.fullmatch(r'\s*REGION:\s*([1-9])\s*', record['response'])
    region = int(matched[1]) if matched else 5
    valid = matched is not None
    require(type(record['selector_region']) is int and record['selector_region'] == region
            and record['selector_valid'] is valid, 'Selector response/decision mismatch')
    return region, valid


def random_id(example_id, seed):
    raw = hashlib.sha256(('region-random:' + str(seed) + ':' + example_id).encode()).digest()
    return random.Random(int.from_bytes(raw, 'big')).choice(range(1, 10))


def validate_panel(rows, labels, records, expected_n):
    require(type(expected_n) is int and expected_n > 0 and len(rows) == expected_n, 'Incorrect source count')
    require(all(set(r) == OBSERVATION_KEYS for r in rows), 'Inference manifest contains missing/extra fields')
    require(all(all(isinstance(v, str) and v.strip() for v in row.values()) for row in rows), 'Invalid observation field')
    require(all(re.fullmatch('[0-9a-f]{64}', r['image_sha256']) for r in rows), 'Invalid image hash')
    ids = [r['example_id'] for r in rows]
    require(len(set(ids)) == expected_n, 'Duplicate example ID')
    require(len({r['source_cluster_id'] for r in rows}) == expected_n, 'Duplicate source cluster')
    label_map = {r['example_id']: r for r in labels}
    require(len(label_map) == len(labels) == expected_n and set(label_map) == set(ids), 'Label coverage mismatch')
    for row in rows:
        label = label_map[row['example_id']]
        require(all(label[k] == row[k] for k in OBSERVATION_KEYS), 'Observation/label identity mismatch')
        validate_reference_sets(label['original_answers'], label['validated_primary_answers'], label['original_answer_variants'])
        require(label['answer'] == label['original_answers'][0], 'Canonical answer changed')
    table = {(r['example_id'], r['action']): r for r in records}
    require(len(table) == len(records) == expected_n * 13, 'Incomplete or duplicate records')
    require(set(table) == {(eid, action) for eid in ids for action in ACTIONS}, 'Wrong action/source coverage')
    for row in rows:
        for action in ACTIONS:
            record = table[row['example_id'], action]
            require(record['status'] == 'ok' and record['source_cluster_id'] == row['source_cluster_id'], 'Wrong source or failed record')
            require(isinstance(record['response'], str) and type(record['generation_truncated']) is bool, 'Invalid answer metadata')
            for key in ALL_COSTS:
                require(finite(record[key]) and record[key] >= 0, 'Invalid measurement: ' + key)
            require(record['elapsed_s'] > 0 and 0 < record['peak_memory_gib'] <= record['peak_reserved_gib'], 'Invalid latency/memory')
            require(record['processor_observer_elapsed_s'] <= record['elapsed_s'], 'Observer time exceeds action')
            for key in ('input_tokens', 'visual_tokens', 'generated_tokens'):
                require(type(record[key]) is int and record[key] > 0, 'Invalid token count')
            lp = record.get('mean_token_logprob')
            require(lp is None or finite(lp) and lp <= 1e-5, 'Invalid confidence')
        strict_region(table[row['example_id'], 'selector'])
    return table, label_map


def verify_calibration(calibration, expected_development_n, evaluation_ids=()):
    require(calibration['schema_version'] == 1 and calibration['method'] == 'numpy_linear_quantile', 'Calibration schema/rule differs')
    require(calibration['missing_confidence_rule'] == 'zoom' and calibration['tie_rule'] == 'confidence <= threshold zooms', 'Calibration gate rule differs')
    timestamp(calibration['created_utc'])
    entries = calibration['development_direct_confidence']
    require(len(entries) == expected_development_n, 'Development confidence coverage differs')
    ids = [r['example_id'] for r in entries]
    require(len(set(ids)) == len(ids) and not set(ids).intersection(evaluation_ids), 'Duplicate/overlapping calibration IDs')
    values = [r['mean_token_logprob'] for r in entries]
    require(all(v is None or finite(v) and v <= 1e-5 for v in values), 'Invalid calibration contributor')
    finite_values = [v for v in values if v is not None]
    require(bool(finite_values), 'No finite confidence contributors')
    require(calibration['development_sources'] == expected_development_n
            and calibration['finite_development_sources'] == len(finite_values)
            and calibration['missing_development_sources'] == values.count(None), 'Calibration counts differ')
    require(set(calibration['thresholds']) == set(FRACTIONS), 'Threshold set differs')
    expected = {q: quantile(finite_values, float(q)) for q in FRACTIONS}
    for q in FRACTIONS:
        require(finite(calibration['thresholds'][q])
                and math.isclose(calibration['thresholds'][q], expected[q], rel_tol=1e-12, abs_tol=1e-12), 'Quantile recomputation differs: ' + q)
    require(calibration['uses_reference_labels'] is False and calibration['new_model_calls'] == 0, 'Calibration scope differs')
    return expected


def rebuild(rows, labels, records, thresholds, *, expected_n=60, samples=10000, seed=20260927):
    """Pure arithmetic for completed records. Small N is for synthetic tests only."""
    table, label_map = validate_panel(rows, labels, records, expected_n)
    require(set(thresholds) == set(FRACTIONS) and all(finite(v) for v in thresholds.values()), 'Bad thresholds')
    require(type(samples) is int and samples > 0 and type(seed) is int, 'Bad bootstrap settings')
    n = len(rows)
    strategies, per_source = {}, []
    gates = {q: [] for q in FRACTIONS}
    invalid = {a: 0 for a in ACTIONS if a != 'selector'}

    def add(name, score, measurements, invalid_answer, truncated):
        out = strategies.setdefault(name, {'scores': {m: [] for m in METRICS},
            'cost_observations': {c: [] for c in ALL_COSTS}, 'invalid_answer_per_source': [], 'truncated_answer_per_source': []})
        for m in METRICS: out['scores'][m].append(float(score[m]))
        for c in ALL_COSTS: out['cost_observations'][c].append(measurements[c])
        out['invalid_answer_per_source'].append(invalid_answer)
        out['truncated_answer_per_source'].append(truncated)

    for row in rows:
        eid = row['example_id']; label = label_map[eid]
        rr = {a: table[eid, a] for a in ACTIONS}
        grades = {a: score_response(rr[a]['response'], label['original_answers'], label['validated_primary_answers'], label['original_answer_variants'])
                  for a in ACTIONS if a != 'selector'}
        for a in grades: invalid[a] += not grades[a]['parse_valid']
        selected, valid = strict_region(rr['selector'])
        selected_action = 'native_' + str(selected)
        randomized_id = random_id(eid, seed)
        oracle = next(j for j in range(1, 10) if grades['native_' + str(j)]['primary_em'] == max(grades['native_' + str(k)]['primary_em'] for k in range(1, 10)))
        # Map a policy to the one answer it returns and all sequential calls it pays.
        plans = {a: (a, [a]) for a in ('direct', *(f'native_{j}' for j in range(1, 10)), 'highres')}
        plans.update(selected_native=(selected_action, ['selector', selected_action]),
                     selected_degraded=('selected_degraded', ['selector', 'selected_degraded']),
                     center_native=('native_5', ['native_5']),
                     seeded_random_native=(f'native_{randomized_id}', [f'native_{randomized_id}']),
                     hindsight_oracle_native=(f'native_{oracle}', [f'native_{oracle}']))
        lp = rr['direct'].get('mean_token_logprob')
        for q in FRACTIONS:
            zoom = lp is None or lp <= thresholds[q]
            gates[q].append(zoom)
            plans['confidence_gate_' + q] = (selected_action, ['direct', 'selector', selected_action]) if zoom else ('direct', ['direct'])
        for policy, (answer_action, calls) in plans.items():
            measurements = {}
            for field in ALL_COSTS:
                parts = [rr[a][field] for a in calls]
                measurements[field] = max(parts) if field in PEAKS else math.fsum(parts)
            add(policy, grades[answer_action], measurements, not grades[answer_action]['parse_valid'], rr[answer_action]['generation_truncated'])
        native = [f'native_{j}' for j in range(1, 10)]
        add('uniform_expected_native', {m: math.fsum(grades[a][m] for a in native) / 9 for m in METRICS},
            {c: math.fsum(rr[a][c] for a in native) / 9 for c in ALL_COSTS},
            sum(not grades[a]['parse_valid'] for a in native) / 9,
            sum(rr[a]['generation_truncated'] for a in native) / 9)
        per_source.append({'example_id': eid, 'source_cluster_id': row['source_cluster_id'],
            'selector_region': selected, 'selector_valid': valid, 'random_region': randomized_id,
            'oracle_primary_region': oracle, 'direct_logprob': lp,
            'native_scores': {str(j): {m: grades[f'native_{j}'][m] for m in METRICS} for j in range(1, 10)},
            'scores': {name: {m: v['scores'][m][-1] for m in METRICS} for name, v in strategies.items()}})
    rng = random.Random(seed)
    draws = [[rng.choice(range(n)) for _ in range(n)] for _ in range(samples)]
    cached_intervals = {}

    def confidence_interval(values):
        key = tuple(values)
        if key not in cached_intervals:
            means = [math.fsum(values[j] for j in draw) / n for draw in draws]
            cached_intervals[key] = [quantile(means, .025), quantile(means, .975)]
        return cached_intervals[key]

    for values in strategies.values():
        values['metrics'] = {m: {'mean': math.fsum(v) / n, 'ci95': confidence_interval(v)} for m, v in values['scores'].items()}
        values['costs'] = {c: describe(v) for c, v in values['cost_observations'].items()}
        values['format_rates'] = {key: math.fsum(values[key + '_per_source']) / n for key in ('invalid_answer', 'truncated_answer')}
    strategies['uniform_expected_native']['marginal_single_crop_costs'] = {
        c: describe(table[row['example_id'], f'native_{j}'][c] for row in rows for j in range(1, 10)) for c in RAW_COSTS}
    pairs = [('selected_native', negative) for negative in ('uniform_expected_native', 'center_native', 'seeded_random_native', 'selected_degraded', 'direct', 'highres')]
    pairs += [('confidence_gate_' + q, base) for q in FRACTIONS for base in ('direct', 'highres')]
    contrasts = {}
    for positive, negative in pairs:
        result = {}
        for metric in METRICS:
            delta = [a - b for a, b in zip(strategies[positive]['scores'][metric], strategies[negative]['scores'][metric])]
            result[metric] = {'difference': math.fsum(delta) / n, 'ci95': confidence_interval(delta), 'per_source_difference': delta}
        if negative != 'uniform_expected_native':
            pairs_binary = list(zip(strategies[positive]['scores']['primary_em'], strategies[negative]['scores']['primary_em']))
            fixes = pairs_binary.count((1, 0)); harms = pairs_binary.count((0, 1))
            wrong = sum(b == 0 for _, b in pairs_binary); correct = n - wrong
            result['automatic_transitions'] = {'fixes': fixes, 'harms': harms, 'baseline_wrong': wrong, 'baseline_correct': correct,
                'fix_rate': fixes / wrong if wrong else None, 'harm_rate': harms / correct if correct else None}
        contrasts[positive + '_minus_' + negative] = result
    return {'n_sources': n, 'n_records': len(records), 'strategies': strategies, 'contrasts': contrasts,
        'primary_result': {'contrast': PRIMARY, 'metric': 'primary_em', **contrasts[PRIMARY]['primary_em']},
        'per_source': per_source, 'invalid_answers': invalid,
        'selector_invalid_or_truncated': sum(not r['selector_valid'] for r in per_source),
        'selector_region_counts': {str(j): sum(r['selector_region'] == j for r in per_source) for j in range(1, 10)},
        'truncated_generations': {a: sum(table[row['example_id'], a]['generation_truncated'] for row in rows) for a in ACTIONS},
        'raw_action_costs': {a: {c: describe(table[row['example_id'], a][c] for row in rows) for c in RAW_COSTS} for a in ACTIONS},
        'physical_experiment_measured_time_sum_s': math.fsum(r['elapsed_s'] for r in records),
        'confidence_gates': {q: {'threshold': thresholds[q], 'zoom_count': sum(v), 'zoom_fraction': sum(v) / n,
            'requested_development_quantile': float(q), 'zoom_per_source': v} for q, v in gates.items()},
        'bootstrap': {'samples': samples, 'seed': seed, 'unit': 'source_cluster'}}


def compare(summary, expected):
    """Every numeric/identity leaf rebuilt above is required and checked."""
    failures, comparisons = [], 0

    def walk(actual, wanted, path):
        nonlocal comparisons
        if isinstance(wanted, dict):
            if not isinstance(actual, dict):
                failures.append({'path': path, 'reason': 'Expected object'}); return
            for key, value in wanted.items():
                if key not in actual:
                    failures.append({'path': path + '/' + key, 'reason': 'Missing field'})
                else: walk(actual[key], value, path + '/' + key)
        elif isinstance(wanted, list):
            if not isinstance(actual, list) or len(actual) != len(wanted):
                failures.append({'path': path, 'reason': 'List length/type mismatch'}); return
            for i, value in enumerate(wanted): walk(actual[i], value, path + '/' + str(i))
        else:
            comparisons += 1
            if type(wanted) is bool or wanted is None:
                okay = actual is wanted
            elif finite(wanted):
                okay = finite(actual) and math.isclose(actual, wanted, rel_tol=1e-10, abs_tol=1e-10)
            else: okay = actual == wanted
            if not okay: failures.append({'path': path, 'actual': actual, 'expected': wanted})
    walk(summary, expected, '')
    for group in ('strategies', 'contrasts'):
        if set(summary.get(group, {})) != set(expected[group]):
            failures.append({'path': '/' + group, 'reason': 'Unexpected or missing strategy/contrast'})
    if 'automatic_transitions' in summary.get('contrasts', {}).get(PRIMARY, {}):
        failures.append({'path': '/contrasts/' + PRIMARY, 'reason': 'Fractional uniform contrast cannot have binary transitions'})
    return comparisons, failures


def check_files(manifest, labels, run_dir, summary_path, calibration_path):
    run_dir = Path(run_dir)
    # This guard is deliberately before reading any raw response or summary.
    completed_path = run_dir / 'completed.json'
    require(completed_path.is_file(), 'Refusing incomplete run: completed.json is absent')
    completed = read(completed_path)
    require(completed['records'] == 780 and not (run_dir / 'error.json').exists(), 'Requires complete error-free N60 evaluation')
    run = read(run_dir / 'run.json')
    require(run['role'] == 'evaluation', 'Only held-out evaluation is accepted by this CLI')
    config = run['config']
    require(config['source_counts'] == {'engineering': 2, 'development': 64, 'evaluation': 60}, 'Cohort specification differs')
    require(config['bootstrap_samples'] == 10000 and config['bootstrap_seed'] == config['seed'] == 20260927, 'Bootstrap/RNG differs')
    require(config['primary_contrast'] == PRIMARY and config['generation_calls_per_source'] == 13, 'Statistical protocol differs')
    summary = read(summary_path)
    require(summary['status'] == 'completed_evaluation_pilot' and summary['role'] == 'evaluation', 'Summary is not completed evaluation')
    paths = {'manifest_sha256': Path(manifest), 'labels_sha256': Path(labels), 'run_sha256': run_dir / 'run.json',
             'records_sha256': run_dir / 'records.jsonl', 'completed_sha256': completed_path, 'calibration_sha256': Path(calibration_path)}
    for key, path in paths.items(): require(summary['bindings'][key] == digest(path), 'Summary binding differs: ' + key)
    require(completed['records_sha256'] == digest(paths['records_sha256']), 'Completion/record digest differs')
    require(run['manifest_sha256'] == digest(manifest), 'Run observation identity differs')
    require(run['execution_lock_sha256'] == summary['bindings']['lock_sha256'], 'Run/summary lock identity differs')
    for relative in ('experiments/dude_metrics.py', 'src/lookagain/actions.py'):
        require(run['code_bindings'][relative] == digest(PROJECT / relative), 'Shared frozen scorer/parser differs: ' + relative)
    require(summary['bindings']['analysis_code_sha256'] == run['code_bindings']['experiments/analyze_region_selection.py'], 'Reported analyzer differs from run code binding')
    require(timestamp(completed['finished_utc']) >= timestamp(run['created_utc']), 'Completion precedes run')
    rows, label_rows = jsonlines(manifest), jsonlines(labels)
    calibration = read(calibration_path)
    require(timestamp(calibration['created_utc']) <= timestamp(run['created_utc']), 'Calibration postdates evaluation')
    thresholds = verify_calibration(calibration, 64, [r['example_id'] for r in rows])
    records = jsonlines(paths['records_sha256'])
    expected = rebuild(rows, label_rows, records, thresholds)
    require(math.isclose(completed['sum_measured_action_elapsed_s'], expected['physical_experiment_measured_time_sum_s'], rel_tol=1e-10, abs_tol=1e-10), 'Completion time sum differs')
    comparisons, failures = compare(summary, expected)
    require(summary['new_model_calls'] == 0 and summary['automatic_controller_training'] is False, 'Summary scope differs')
    return {'schema_version': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'PASS' if not failures else 'FAIL', 'comparisons': comparisons, 'failures': failures,
        'sources': 60, 'physical_records': 780, 'new_model_calls': 0,
        'primary_result_independently_recomputed': expected['primary_result'],
        'policy_summary_independently_recomputed': {name: {'metrics': values['metrics'], 'costs': values['costs'], 'format_rates': values['format_rates']}
                                                  for name, values in expected['strategies'].items()},
        'confidence_gates_independently_recomputed': expected['confidence_gates'],
        'bindings': {**{key: digest(path) for key, path in paths.items()}, 'summary_sha256': digest(summary_path),
            'checker_sha256': digest(__file__), 'shared_scorer_sha256': digest(PROJECT / 'experiments/dude_metrics.py'),
            'shared_answer_parser_sha256': digest(PROJECT / 'src/lookagain/actions.py')},
        'scope': {'independent_arithmetic': True, 'imports_main_analyzer': False,
            'shared_frozen_dude_scorer_and_answer_parser': True, 'images_required': False,
            'source_pixels_or_execution_lock_independently_validated': False,
            'development_quantiles_recomputed_from_bound_contributor_list': True,
            'raw_development_records_independently_read': False},
        'limitations': ['Run the main strict source/pixel/processor validator separately; matching public hashes does not revalidate source images.',
            'Scorer/parser code is shared and explicitly hash-bound; this is not independent semantic adjudication.',
            'Calibration contributor values are bound and re-quantiled here, not reloaded from raw development responses.',
            'Policy latency is a component reconstruction; unmeasured cheap gate/random arithmetic is excluded.',
            'Uniform marginal costs describe nine equally weighted single crops, never 9N independent inferential sources.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'labels', 'run', 'summary', 'calibration', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Use a new report path; existing audit reports are immutable')
    report = check_files(args.manifest, args.labels, args.run, args.summary, args.calibration)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'status': report['status'], 'comparisons': report['comparisons'], 'new_model_calls': 0}))
    raise SystemExit(report['status'] != 'PASS')


if __name__ == '__main__':
    main()
