"""Completed-run analysis of the given-page, fixed-grid region-selection pilot.

No generation. Engineering reports withhold all answer scores. --calibrate reads
only completed development direct log probabilities, never reference answers.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / 'experiments'), str(PROJECT / 'src')]
from region_selection_core import parse_region, random_region
from dude_metrics import score_response, validate_reference_sets

ACTIONS = ('direct', 'selector', *(f'native_{i}' for i in range(1, 10)), 'selected_degraded', 'highres')
PRIMARY = 'selected_native_minus_uniform_expected_native'
METRICS = ('primary_em', 'official_anls')
SUM_COSTS = ('elapsed_s', 'input_tokens', 'visual_tokens', 'generated_tokens')
PEAK_COSTS = ('peak_memory_gib', 'peak_reserved_gib')
FRACTIONS = ('0.25', '0.5', '0.75')


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON: ' + value)))


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(parsed.tzinfo is not None, 'Timezone-aware timestamp required')
    return parsed


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def distribution(values):
    values = list(values)
    require(bool(values) and all(finite(v) for v in values), 'Finite nonempty observations required')
    return {'n': len(values), 'mean': math.fsum(values) / len(values),
            'median': float(np.quantile(values, .5, method='linear')),
            'p95': float(np.quantile(values, .95, method='linear')),
            'min': min(values), 'max': max(values)}


def record_index(observations, records, config, role):
    rows = [asdict(row) if is_dataclass(row) else dict(row) for row in observations]
    require(role in ('engineering', 'development', 'evaluation'), 'Unknown analysis role')
    require(len(rows) == config['source_counts'][role] > 0, 'Incorrect complete-cohort denominator')
    ids = [row['example_id'] for row in rows]
    require(len(set(ids)) == len(ids), 'Duplicate example ID')
    require(len({row['source_cluster_id'] for row in rows}) == len(rows), 'Duplicate source cluster')
    records = list(records.values()) if isinstance(records, dict) else list(records)
    indexed = {(record['example_id'], record['action']): record for record in records}
    require(len(indexed) == len(records), 'Duplicate action record')
    require(set(indexed) == {(eid, action) for eid in ids for action in ACTIONS}, 'Incomplete or extra action panel')
    for row in rows:
        eid = row['example_id']
        for action in ACTIONS:
            record = indexed[eid, action]
            require(record['status'] == 'ok' and record['source_cluster_id'] == row['source_cluster_id'], 'Failed record or wrong source')
            require(isinstance(record['response'], str) and type(record['generation_truncated']) is bool, 'Invalid response metadata')
            for field in (*SUM_COSTS, *PEAK_COSTS):
                require(finite(record[field]) and record[field] >= 0, 'Invalid cost ' + field)
            require(record['elapsed_s'] > 0 and record['peak_memory_gib'] > 0
                    and record['peak_reserved_gib'] >= record['peak_memory_gib'], 'Invalid latency or memory')
            for field in ('input_tokens', 'visual_tokens', 'generated_tokens'):
                require(type(record[field]) is int and record[field] > 0, 'Invalid token count')
            confidence = record.get('mean_token_logprob')
            require(confidence is None or finite(confidence) and confidence <= 1e-5, 'Invalid log probability')
        selector = indexed[eid, 'selector']
        region, valid = parse_region(selector['response'], selector['generation_truncated'])
        require(selector['selector_region'] == region and type(selector['selector_region']) is int
                and selector['selector_valid'] is valid, 'Saved selector decision differs from strict parser')
    return rows, indexed


def validate_labels(rows, labels):
    labels = list(labels)
    indexed = {row['example_id']: row for row in labels}
    require(len(indexed) == len(labels) and set(indexed) == {r['example_id'] for r in rows}, 'Label coverage differs')
    for row in rows:
        label = indexed[row['example_id']]
        for field in ('question', 'source_cluster_id', 'image_sha256', 'image_path'):
            require(label[field] == row[field], 'Observation/label identity differs: ' + field)
        validate_reference_sets(label['original_answers'], label['validated_primary_answers'], label['original_answer_variants'])
        require(label['answer'] == label['original_answers'][0], 'Canonical reference changed')
    return indexed


def validate_calibration(calibration, config, config_sha256=None):
    require(isinstance(calibration, dict) and calibration.get('schema_version') == 1, 'Invalid calibration schema')
    timestamp(calibration['created_utc'])
    require(calibration['method'] == 'numpy_linear_quantile' and calibration['missing_confidence_rule'] == 'zoom', 'Calibration rule differs')
    require(set(calibration['thresholds']) == set(FRACTIONS), 'Calibration threshold coverage differs')
    thresholds = [calibration['thresholds'][key] for key in FRACTIONS]
    require(all(finite(value) and value <= 1e-5 for value in thresholds) and thresholds == sorted(thresholds), 'Invalid confidence thresholds')
    require(type(calibration['finite_development_sources']) is int
            and 0 < calibration['finite_development_sources'] <= config['source_counts']['development'], 'Invalid development confidence denominator')
    for field in ('development_records_sha256', 'development_lock_sha256', 'config_sha256'):
        value = calibration[field]
        require(isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value), 'Invalid calibration binding ' + field)
    if config_sha256 is not None:
        require(calibration['config_sha256'] == config_sha256, 'Calibration scientific config differs')


def calibrate_confidence(observations, records, config, bindings):
    rows, indexed = record_index(observations, records, config, 'development')
    values = [indexed[row['example_id'], 'direct'].get('mean_token_logprob') for row in rows]
    finite_values = [value for value in values if value is not None]
    require(bool(finite_values), 'No finite development confidence observations')
    result = {'schema_version': 1, 'created_utc': now(),
              **{key: bindings[key] for key in ('development_records_sha256', 'development_lock_sha256', 'config_sha256')},
              'thresholds': {key: float(np.quantile(finite_values, float(key), method='linear')) for key in FRACTIONS},
              'missing_confidence_rule': 'zoom', 'method': 'numpy_linear_quantile',
              'finite_development_sources': len(finite_values), 'development_sources': len(rows),
              'missing_development_sources': len(values) - len(finite_values),
              'development_direct_confidence': [{'example_id': row['example_id'], 'mean_token_logprob': value}
                                                for row, value in zip(rows, values)],
              'analysis_code_sha256': sha(__file__), 'uses_reference_labels': False,
              'tie_rule': 'confidence <= threshold zooms', 'new_model_calls': 0}
    validate_calibration(result, config, bindings['config_sha256'])
    return result


def cost(record):
    result = {field: record[field] for field in (*SUM_COSTS, *PEAK_COSTS)}
    if 'processor_observer_elapsed_s' in record:
        value = record['processor_observer_elapsed_s']
        require(finite(value) and 0 <= value <= record['elapsed_s'], 'Invalid processor observer time')
        result['processor_observer_elapsed_s'] = value
    return result


def sequential_cost(*records):
    costs = [cost(record) for record in records]
    keys = set.intersection(*(set(value) for value in costs))
    return {key: max(value[key] for value in costs) if key in PEAK_COSTS
            else math.fsum(value[key] for value in costs) for key in keys}


def mean_cost(records):
    costs = [cost(record) for record in records]
    keys = set.intersection(*(set(value) for value in costs))
    return {key: math.fsum(value[key] for value in costs) / len(costs) for key in keys}


def paired_draws(n, samples, seed):
    require(type(samples) is int and samples > 0, 'Positive bootstrap sample count required')
    rng = random.Random(seed)
    return np.asarray([[rng.randrange(n) for _ in range(n)] for _ in range(samples)], dtype=np.int64)


def interval(values, draws):
    array = np.asarray(values, dtype=float)
    means = array[draws].mean(axis=1)
    return [float(value) for value in np.quantile(means, [.025, .975], method='linear')]


def compute_statistics(observations, records, labels, config, role, calibration=None):
    rows, indexed = record_index(observations, records, config, role)
    technical = {'n_sources': len(rows), 'n_records': len(indexed),
                 'selector_invalid_or_truncated': sum(not indexed[r['example_id'], 'selector']['selector_valid'] for r in rows),
                 'selector_region_counts': {str(region): sum(indexed[r['example_id'], 'selector']['selector_region'] == region for r in rows) for region in range(1,10)},
                 'truncated_generations': {action: sum(indexed[r['example_id'], action]['generation_truncated'] for r in rows) for action in ACTIONS},
                 'raw_action_costs': {action: {field: distribution(indexed[r['example_id'], action][field] for r in rows)
                                             for field in (*SUM_COSTS, *PEAK_COSTS)} for action in ACTIONS},
                 'physical_experiment_measured_time_sum_s': math.fsum(record['elapsed_s'] for record in indexed.values())}
    if role == 'engineering':
        return {'status': 'completed_engineering_technical_only', 'role': role, **technical,
                'answer_scores_withheld': True, 'confidence_calibration': None,
                'automatic_controller_training': False}
    label_map = validate_labels(rows, labels)
    if calibration is not None:
        validate_calibration(calibration, config)
    require(role != 'evaluation' or calibration is not None, 'Evaluation requires frozen development calibration')
    strategies = {}
    per_source = []
    invalid_answers = {action: 0 for action in ACTIONS if action != 'selector'}
    zooms = {fraction: [] for fraction in FRACTIONS} if calibration is not None else {}
    def append(name, metric, costs):
        value = strategies.setdefault(name, {'scores': {key: [] for key in METRICS}, 'cost_observations': {}})
        for key in METRICS:
            value['scores'][key].append(float(metric[key]))
        for key, amount in costs.items():
            value['cost_observations'].setdefault(key, []).append(amount)
    for row in rows:
        eid = row['example_id']; label = label_map[eid]
        answer_records = {action: indexed[eid, action] for action in ACTIONS if action != 'selector'}
        scores = {action: score_response(record['response'], label['original_answers'], label['validated_primary_answers'], label['original_answer_variants'])
                  for action, record in answer_records.items()}
        for action, score in scores.items():
            invalid_answers[action] += not score['parse_valid']
        selector = indexed[eid, 'selector']; region = selector['selector_region']
        selected = f'native_{region}'; random_id = random_region(eid, config['seed'])
        native_records = [answer_records[f'native_{i}'] for i in range(1,10)]
        expected_scores = {key: math.fsum(scores[f'native_{i}'][key] for i in range(1,10))/9 for key in METRICS}
        for action, score in scores.items():
            if action.startswith('native_') or action in ('direct', 'highres'):
                append(action, score, cost(answer_records[action]))
        append('selected_native', scores[selected], sequential_cost(selector, answer_records[selected]))
        append('selected_degraded', scores['selected_degraded'], sequential_cost(selector, answer_records['selected_degraded']))
        append('uniform_expected_native', expected_scores, mean_cost(native_records))
        append('center_native', scores['native_5'], cost(answer_records['native_5']))
        append('seeded_random_native', scores[f'native_{random_id}'], cost(answer_records[f'native_{random_id}']))
        oracle_id = min(range(1,10), key=lambda i: (-scores[f'native_{i}']['primary_em'], i))
        append('hindsight_oracle_native', scores[f'native_{oracle_id}'], cost(answer_records[f'native_{oracle_id}']))
        confidence = answer_records['direct'].get('mean_token_logprob')
        chosen_actions = {action: action for action in answer_records if action.startswith('native_') or action in ('direct', 'highres')}
        chosen_actions.update(selected_native=selected, selected_degraded='selected_degraded', center_native='native_5',
                              seeded_random_native=f'native_{random_id}', hindsight_oracle_native=f'native_{oracle_id}')
        for fraction in zooms:
            zoom = confidence is None or confidence <= calibration['thresholds'][fraction]
            zooms[fraction].append(zoom)
            append('confidence_gate_' + fraction, scores[selected] if zoom else scores['direct'],
                   sequential_cost(answer_records['direct'], selector, answer_records[selected]) if zoom else cost(answer_records['direct']))
            chosen_actions['confidence_gate_' + fraction] = selected if zoom else 'direct'
        for name, action in chosen_actions.items():
            strategies[name].setdefault('invalid_answer_per_source', []).append(not scores[action]['parse_valid'])
            strategies[name].setdefault('truncated_answer_per_source', []).append(answer_records[action]['generation_truncated'])
        strategies['uniform_expected_native'].setdefault('invalid_answer_per_source', []).append(
            sum(not scores[f'native_{i}']['parse_valid'] for i in range(1,10))/9)
        strategies['uniform_expected_native'].setdefault('truncated_answer_per_source', []).append(
            sum(answer_records[f'native_{i}']['generation_truncated'] for i in range(1,10))/9)
        per_source.append({'example_id': eid, 'source_cluster_id': row['source_cluster_id'],
                           'selector_region': region, 'selector_valid': selector['selector_valid'],
                           'random_region': random_id, 'oracle_primary_region': oracle_id,
                           'direct_logprob': confidence,
                           'native_scores': {str(i): {key: scores[f'native_{i}'][key] for key in METRICS} for i in range(1,10)},
                           'scores': {name: {key: data['scores'][key][-1] for key in METRICS} for name,data in strategies.items()}})
    draws = paired_draws(len(rows), config['bootstrap_samples'], config['bootstrap_seed'])
    for data in strategies.values():
        data['metrics'] = {key: {'mean': math.fsum(values)/len(values), 'ci95': interval(values, draws)} for key,values in data['scores'].items()}
        data['costs'] = {key: distribution(values) for key,values in data['cost_observations'].items()}
        data['format_rates'] = {key: math.fsum(data[key + '_per_source'])/len(rows)
                                for key in ('invalid_answer', 'truncated_answer')}
    strategies['uniform_expected_native']['cost_distribution_scope'] = 'Per-source expected single-crop costs; these percentiles are not the marginal randomized-policy latency percentiles.'
    strategies['uniform_expected_native']['marginal_single_crop_costs'] = {
        field: distribution(indexed[row['example_id'], f'native_{i}'][field] for row in rows for i in range(1,10))
        for field in (*SUM_COSTS, *PEAK_COSTS)}
    strategies['uniform_expected_native']['marginal_distribution_scope'] = 'Equal-weight nine times N measured single crops; descriptive costs only, never an inferential sample size.'
    strategies['hindsight_oracle_native']['selection_criterion'] = 'Best primary EM among nine crops; lowest region ID breaks ties. ANLS uses that same selected action and is not an ANLS oracle.'
    strategies['hindsight_oracle_native']['label_privileged'] = True
    pairs = [('selected_native', name) for name in ('uniform_expected_native', 'center_native', 'seeded_random_native', 'selected_degraded', 'direct', 'highres')]
    pairs += [('confidence_gate_' + key, baseline) for key in zooms for baseline in ('direct', 'highres')]
    contrasts = {}
    for positive, negative in pairs:
        key = positive + '_minus_' + negative
        contrast = {}
        for metric in METRICS:
            delta = [a-b for a,b in zip(strategies[positive]['scores'][metric], strategies[negative]['scores'][metric])]
            contrast[metric] = {'difference': math.fsum(delta)/len(delta), 'ci95': interval(delta, draws), 'per_source_difference': delta}
        if negative != 'uniform_expected_native':
            a,b = strategies[positive]['scores']['primary_em'], strategies[negative]['scores']['primary_em']
            fixes = sum(x==1 and y==0 for x,y in zip(a,b)); harms = sum(x==0 and y==1 for x,y in zip(a,b))
            wrong = sum(y==0 for y in b); correct = sum(y==1 for y in b)
            contrast['automatic_transitions'] = {'fixes': fixes, 'harms': harms, 'baseline_wrong': wrong, 'baseline_correct': correct,
                                                  'fix_rate': fixes/wrong if wrong else None, 'harm_rate': harms/correct if correct else None}
        contrasts[key] = contrast
    result = {'status': 'completed_' + role + '_pilot', 'role': role, **technical,
              'analysis_scope': 'Prespecified small outcome-held-out evaluation pilot' if role == 'evaluation' else 'Development diagnostic; not held-out confirmation',
              'invalid_answers': invalid_answers, 'strategies': strategies, 'contrasts': contrasts,
              'primary_result': {'contrast': PRIMARY, 'metric': 'primary_em', **contrasts[PRIMARY]['primary_em'],
                                 'test': 'paired source-cluster percentile bootstrap; no McNemar for fractional uniform baseline'},
              'per_source': per_source, 'confidence_gates': {key: {'threshold': calibration['thresholds'][key],
                                                                  'zoom_count': sum(values), 'zoom_fraction': sum(values)/len(values),
                                                                  'requested_development_quantile': float(key), 'zoom_per_source': values}
                                                               for key,values in zooms.items()},
              'bootstrap': {'samples': config['bootstrap_samples'], 'seed': config['bootstrap_seed'],
                            'unit': 'source_cluster', 'method': 'paired percentile 95%, numpy linear endpoints; shared Python Random draws'},
              'automatic_controller_training': False,
              'cost_definitions': {'selected': 'Full selector plus one selected answer; selector cost charged once to each alternative strategy. These are paired component-sum reconstructions, not a separately timed deployment pipeline.',
                                   'uniform': 'Mean of nine measured single-crop costs, expected uniform random selection; not cost of generating all nine.',
                                   'fixed_proposals': 'Fixed geometry construction is included in each measured crop invocation; separate random-ID arithmetic is unmeasured.',
                                   'confidence_gate': 'Direct only when confidence exceeds fixed threshold; otherwise direct plus selector plus selected native.',
                                   'sequential_memory': 'Maximum component peak, never sum. Uniform memory is expected single-action peak, not nine-action maximum.',
                                   'oracle': 'Label-privileged best primary-EM native action, smallest ID tie-break. Charges chosen answer only, not an implemented policy.'},
              'limitations': ['N60 evaluation is a small outcome-held-out pilot, not a powered confirmation.',
                              'The answer page remains annotation-privileged; source-only annotations were previously audited.',
                              'Primary selected-versus-uniform contrast is fractional source-paired data; binary-pair significance tests do not apply.',
                              'Only the declared primary EM contrast is primary; other intervals are descriptive and unadjusted.',
                              'Confidence is uncalibrated token log probability. Development quantiles need not yield exact evaluation zoom fractions.',
                              'Allocator-reserved counters are retained separately and are not a claim of required physical VRAM.',
                              'A positive interval does not establish a trained controller, model-family transfer or deployment benefit.']}
    return result


def analyze(manifest, labels_path, lock_path, run_dir, calibration_path=None, engineering=False, calibrate=False):
    from run_region_selection import validate_completed
    observations, records, run, lock = validate_completed(Path(manifest), Path(lock_path), Path(run_dir))
    role = lock['role']; config = lock['config']
    require(run['role'] == role, 'Run/lock role differs')
    require(engineering == (role == 'engineering'), 'Engineering flag must match locked role')
    if calibrate:
        require(role == 'development' and not engineering, 'Calibration requires completed development only')
        return calibrate_confidence(observations, records, config,
                                    {'development_records_sha256': sha(Path(run_dir)/'records.jsonl'),
                                     'development_lock_sha256': sha(lock_path), 'config_sha256': lock['config_sha256']})
    calibration = read(calibration_path) if calibration_path else None
    if calibration is not None:
        validate_calibration(calibration, config, lock['config_sha256'])
    if role == 'evaluation':
        require(calibration_path is not None and sha(calibration_path) == lock['calibration_sha256'], 'Evaluation calibration binding differs')
        require(calibration == lock['calibration'], 'Evaluation calibration snapshot differs')
        require(timestamp(calibration['created_utc']) <= timestamp(lock['locked_at_utc']), 'Calibration postdates evaluation lock')
    labels = None
    if role != 'engineering':
        require(labels_path is not None and sha(labels_path) == lock['labels_sha256'], 'Label artifact differs from execution lock')
        labels = lines(labels_path)
    result = compute_statistics(observations, records, labels, config, role, calibration)
    result['created_utc'] = now()
    result['bindings'] = {'manifest_sha256': sha(manifest), 'labels_sha256': sha(labels_path) if labels_path and role != 'engineering' else None,
                          'lock_sha256': sha(lock_path), 'run_sha256': sha(Path(run_dir)/'run.json'),
                          'records_sha256': sha(Path(run_dir)/'records.jsonl'), 'completed_sha256': sha(Path(run_dir)/'completed.json'),
                          'calibration_sha256': sha(calibration_path) if calibration_path else None,
                          'analysis_code_sha256': sha(__file__)}
    result['new_model_calls'] = 0
    return result


def report_markdown(summary):
    text = ['# Given-page region-selection pilot', '', f"Status: **{summary['status']}**. Sources: **{summary['n_sources']}**; records: **{summary['n_records']}**.", '',
            'The original answer page is privileged. The nine candidate regions use geometry only. No trained controller is evaluated.', '']
    if summary['role'] == 'engineering':
        text += ['Engineering-only technical report: answer quality and intervals are deliberately withheld.',
                 f"Invalid/truncated selector decisions: {summary['selector_invalid_or_truncated']} (center fallback)."]
        return '\n'.join(text) + '\n'
    primary = summary['primary_result']
    text += [f"Primary selected-native minus uniform-expected-native EM: **{100*primary['difference']:+.3f} pp**, paired95% CI **[{100*primary['ci95'][0]:+.3f}, {100*primary['ci95'][1]:+.3f}] pp**.",
             'The uniform baseline is a per-source mean over nine answers; this is not a binary McNemar comparison. N60 is not a powered confirmation.', '',
             '| Strategy | Primary EM,% | Official ANLS | Mean strategy latency,s | Mean sequential allocated peak,GiB |',
             '|---|---:|---:|---:|---:|']
    for name, value in summary['strategies'].items():
        text.append(f"| {name} | {100*value['metrics']['primary_em']['mean']:.3f} | {value['metrics']['official_anls']['mean']:.5f} | {value['costs']['elapsed_s']['mean']:.3f} | {value['costs']['peak_memory_gib']['mean']:.3f} |")
    text += ['', 'Selected regional strategies pay the full selector cost. Uniform cost is an expected one-crop cost. Confidence gates pay direct, then selector plus crop only when zooming. Oracle values are privileged upper bounds, not deployed policies.', '', '## Confidence gates', '']
    for key, value in summary['confidence_gates'].items():
        text.append(f"- Development quantile {key}: threshold {value['threshold']:.8g}; observed zoom {value['zoom_count']}/{summary['n_sources']} ({100*value['zoom_fraction']:.2f}%).")
    text += ['', f"Selector invalid/truncated: {summary['selector_invalid_or_truncated']}; fallback to center retained.", '', '## Limits', '']
    text += ['- ' + value for value in summary['limitations']]
    return '\n'.join(text) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'lock', 'run', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--labels', type=Path)
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--calibrate', action='store_true', help='Write calibration JSON to --output, using completed development only')
    args = parser.parse_args()
    result = analyze(args.manifest, args.labels, args.lock, args.run, args.calibration, args.engineering, args.calibrate)
    require(not args.output.exists(), 'Use a new output path; archived artifacts must not be overwritten')
    if args.calibrate:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    else:
        args.output.mkdir(parents=True)
        (args.output/'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        (args.output/'report.md').write_text(report_markdown(result), encoding='utf-8')
    print(json.dumps({'status': result.get('status', 'calibration_complete'), 'new_model_calls': 0}))


if __name__ == '__main__':
    main()
