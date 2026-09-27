"""Media-free independent arithmetic replay of a completed refinement summary.

Uses the unchanged shared scorer, but no analyzer, NumPy, training module,
images, model or tensor reconstruction. This checks saved result arithmetic and
explicit artifact hashes; it is not the strict execution/source validator.
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
from dude_metrics import score_response

METRICS = ('primary_em', 'official_anls')
ADDITIVE = ('elapsed_s', 'input_tokens', 'visual_tokens', 'generated_tokens', 'processor_observer_elapsed_s')
PEAK = ('peak_memory_gib', 'peak_reserved_gib')
KINDS = ('full', 'image_position', 'position', 'best_fixed', 'center', 'random', 'prompted')
NEW = ('legacy_selector', 'legacy_selected', 'new_direct', 'new_highres', *(f'new_native_{i}' for i in range(1, 10)))
OLD = ('selector', 'direct', 'highres', 'selected_degraded', *(f'native_{i}' for i in range(1, 10)))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def invalid(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=invalid)


def lines(path):
    return [json.loads(s) for s in Path(path).read_text(encoding='utf-8').splitlines() if s.strip()]


def canonical(rows):
    ordered = sorted(rows, key=lambda r: (r['example_id'], r.get('action', '')))
    return hashlib.sha256(json.dumps(ordered, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def average(values):
    values = list(values)
    require(bool(values), 'Empty arithmetic denominator')
    return math.fsum(values) / len(values)


def quantile(values, probability):
    ordered = sorted(values)
    require(bool(ordered), 'Empty quantile')
    index = (len(ordered) - 1) * probability
    low = math.floor(index)
    high = math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def stats(values):
    values = list(values)
    return {'n': len(values), 'mean': average(values), 'median': quantile(values, .5),
            'p95': quantile(values, .95), 'min': min(values), 'max': max(values)}


def cost_components(records, cpu=0.0):
    result = {key: math.fsum(r.get(key, 0) for r in records) for key in ADDITIVE}
    result.update({key: max(r[key] for r in records) for key in PEAK})
    result['elapsed_s'] += cpu
    result['cpu_policy_elapsed_s'] = cpu
    return result


class Comparisons:
    def __init__(self):
        self.count = 0

    def check(self, name, actual, expected):
        if isinstance(expected, dict):
            require(isinstance(actual, dict), name + ': missing mapping')
            for key, value in expected.items():
                require(key in actual, name + ': missing ' + key)
                self.check(name + '.' + str(key), actual[key], value)
        elif isinstance(expected, (list, tuple)):
            require(isinstance(actual, (list, tuple)) and len(actual) == len(expected), name + ': length differs')
            for index, value in enumerate(expected):
                self.check(name + '[' + str(index) + ']', actual[index], value)
        elif type(expected) in (int, float):
            require(type(actual) in (int, float) and math.isfinite(actual)
                    and math.isclose(actual, expected, rel_tol=1e-10, abs_tol=1e-12),
                    name + ': numerical mismatch ' + repr(actual) + ' != ' + repr(expected))
            self.count += 1
        else:
            require(actual == expected, name + ': value differs')
            self.count += 1


def selector(record):
    match = None if record['generation_truncated'] else re.fullmatch(r'\s*REGION:\s*([1-9])\s*', record['response'])
    region = int(match.group(1)) if match else 5
    require(record['selector_region'] == region and record['selector_valid'] is bool(match), 'Selector parse differs')
    return region


def check_statistics(summary, rows, labels, records, archived, features, decisions, config):
    """Second implementation of saved arithmetic; does not recompute learned fits."""
    check = Comparisons()
    ids = [r['example_id'] for r in rows]
    require(len(set(ids)) == len(ids) == config['development_sources'], 'Source denominator differs')
    require(len({r['source_cluster_id'] for r in rows}) == len(ids), 'Repeated source cluster')
    maps = []
    for values, actions in ((records, NEW), (archived, OLD)):
        table = {(r['example_id'], r['action']): r for r in values}
        require(len(table) == len(values) and set(table) == {(eid, a) for eid in ids for a in actions}, 'Incomplete/duplicate panel')
        maps.append(table)
    new, old = maps
    lab = {r['example_id']: r for r in labels}
    feature = {r['example_id']: r for r in features}
    policy = {r['example_id']: r for r in decisions['predictions']}
    for values, table in ((labels, lab), (features, feature), (decisions['predictions'], policy)):
        require(len(values) == len(table) and set(table) == set(ids), 'Missing/duplicate side artifact')
    for row in rows:
        eid = row['example_id']
        for table in (lab, feature, policy):
            require(table[eid]['source_cluster_id'] == row['source_cluster_id'], 'Side artifact source differs')
        require(lab[eid]['question'] == row['question'] and lab[eid]['image_sha256'] == row['image_sha256'], 'Label identity differs')
        require(policy[eid]['regions']['prompted'] == selector(old[eid, 'selector']), 'Frozen prompted ID differs')
        require(policy[eid]['regions']['center'] == 5, 'Wrong center region')
    check.check('scope', summary, {'status': 'completed_development_diagnostic', 'role': 'main',
                                 'n_sources': len(ids), 'generation_calls': len(records),
                                 'feature_forward_calls': len(features), 'new_model_calls_by_analyzer': 0})
    for key in ('primary', 'factorial_interaction', 'descriptive_policy_contrasts', 'strategies'):
        encoded = json.dumps(summary[key]).lower()
        require('ci95' not in encoded and 'confidence_interval' not in encoded, 'Unsupported cross-fit interval in ' + key)
    grades = {'new': {}, 'old': {}}
    expected = {}
    mixture = []
    replays = []

    def append(name, score, cost=None, historical=None, invalid=None, truncated=None):
        value = expected.setdefault(name, {'scores': {m: [] for m in METRICS}, 'cost': [], 'historical': [], 'invalid': [], 'truncated': []})
        for m in METRICS:
            value['scores'][m].append(score[m])
        if cost is not None:
            value['cost'].append(cost)
        if historical is not None:
            value['historical'].append(historical)
        value['invalid'].append(invalid)
        value['truncated'].append(truncated)

    for eid in ids:
        gold = lab[eid]
        for domain, table, actions in (('new', new, NEW), ('old', old, OLD)):
            for action in actions:
                if action in ('selector', 'legacy_selector'):
                    continue
                record = table[eid, action]
                grades[domain][eid, action] = score_response(record['response'], gold['original_answers'], gold['validated_primary_answers'], gold['original_answer_variants'])
        for kind in KINDS:
            region = policy[eid]['regions'][kind]
            require(type(region) is int and 1 <= region <= 9, 'Invalid selected region')
            name = 'ranker_' + kind if kind in ('full', 'image_position', 'position') else kind
            nr, ore = new[eid, 'new_native_' + str(region)], old[eid, 'native_' + str(region)]
            ns, os = grades['new'][eid, nr['action']], grades['old'][eid, ore['action']]
            if kind == 'prompted':
                components, cpu = [new[eid, 'legacy_selector'], nr], 0.0
            else:
                components = [feature[eid], nr] if kind in ('full', 'image_position') else [nr]
                cpu = policy[eid]['cpu_elapsed_s'][kind]
            append(name + '_new', ns, cost_components(components, cpu), invalid=not ns['parse_valid'], truncated=nr['generation_truncated'])
            append(name + '_old', os, historical=cost_components([ore]), invalid=not os['parse_valid'], truncated=ore['generation_truncated'])
        for domain, table, prefix in (('new', new, 'new_'), ('old', old, '')):
            for action in ('direct', 'highres'):
                r = table[eid, prefix + action]
                grade = grades[domain][eid, prefix + action]
                kwargs = {'cost' if domain == 'new' else 'historical': cost_components([r])}
                append(action + '_' + domain, grade, invalid=not grade['parse_valid'], truncated=r['generation_truncated'], **kwargs)
            bank = [grades[domain][eid, prefix + f'native_{i}'] for i in range(1, 10)]
            bank_costs = [cost_components([table[eid, prefix + f'native_{i}']], policy[eid]['cpu_elapsed_s']['random'] if domain == 'new' else 0.) for i in range(1, 10)]
            expected_cost = {key: average(c[key] for c in bank_costs) for key in bank_costs[0]}
            kwargs = {'cost' if domain == 'new' else 'historical': expected_cost}
            append('uniform_' + domain, {m: average(g[m] for g in bank) for m in METRICS},
                   invalid=average(float(not g['parse_valid']) for g in bank),
                   truncated=average(float(table[eid, prefix + f'native_{i}']['generation_truncated']) for i in range(1, 10)), **kwargs)
            append('oracle9_' + domain, {m: max(g[m] for g in bank) for m in METRICS})
            if domain == 'new':
                mixture.extend(bank_costs)
        chosen = selector(new[eid, 'legacy_selector'])
        rs = grades['new'][eid, 'legacy_selected']
        og = grades['old'][eid, f'native_{chosen}']
        replays.append({'selector_region_equal': chosen == policy[eid]['regions']['prompted'],
                        'selector_raw_equal': new[eid, 'legacy_selector']['response'] == old[eid, 'selector']['response'],
                        'selected_raw_equal_at_replayed_region': new[eid, 'legacy_selected']['response'] == old[eid, f'native_{chosen}']['response'],
                        'selected_primary_em_equal_at_replayed_region': rs['primary_em'] == og['primary_em'],
                        'selected_anls_equal_at_replayed_region': rs['official_anls'] == og['official_anls'],
                        'selected_normalized_equal_at_replayed_region': bool(rs['parse_valid'] and og['parse_valid'] and rs['normalized_prediction'] == og['normalized_prediction'])})

    require(set(expected) == set(summary['strategies']), 'Policy coverage differs')
    for name, value in expected.items():
        actual = summary['strategies'][name]
        check.check(name + '.scores', actual['scores'], value['scores'])
        check.check(name + '.metrics', actual['metrics'], {m: {'mean': average(values), 'n_sources': len(ids)} for m, values in value['scores'].items()})
        for source_key, observation_key, distribution_key in (('cost', 'cost_observations', 'cost'), ('historical', 'historical_answer_cost_observations', 'historical_answer_cost')):
            values = value[source_key]
            check.check(name + '.' + observation_key, actual[observation_key], values)
            check.check(name + '.' + distribution_key, actual[distribution_key], {k: stats(r[k] for r in values) for k in values[0]} if values else None)
        for field in ('invalid', 'truncated'):
            count = math.fsum(float(x) for x in value[field]) if all(x is not None for x in value[field]) else None
            check.check(name + '.' + field, actual[field + '_count_or_expected_count'], count)
        check.check(name + '.deployed_policy', actual['deployed_policy'], not name.startswith('oracle9_'))
    check.check('uniform_marginal', summary['strategies']['uniform_new']['marginal_single_crop_cost'], {k: stats(r[k] for r in mixture) for k in mixture[0]})

    folds = [policy[eid]['fold'] for eid in ids]
    require(set(folds) == set(range(config['folds'])), 'Missing fold')
    def contrast_expected(aa, bb):
        delta = [aa[i] - bb[i] for i in range(len(ids))]
        return {'n_sources': len(ids), 'mean_difference': average(delta), 'per_source_differences': delta,
                'folds': [{'fold': f, 'n_sources': folds.count(f),
                           'a_mean': average(aa[i] for i in range(len(ids)) if folds[i] == f),
                           'b_mean': average(bb[i] for i in range(len(ids)) if folds[i] == f),
                           'mean_difference': average(delta[i] for i in range(len(ids)) if folds[i] == f)} for f in range(config['folds'])]}
    def counts(aa, bb):
        cells = {(a, b): sum(x == a and y == b for x, y in zip(aa, bb)) for a in (0, 1) for b in (0, 1)}
        return {'repairs': cells[1, 0], 'harms': cells[0, 1], 'both_correct': cells[1, 1], 'both_wrong': cells[0, 0]}
    for strategy in ('ranker_full', 'ranker_image_position', 'ranker_position', 'best_fixed', 'center', 'random', 'uniform', 'direct', 'highres'):
        name = strategy + '_new_minus_prompted_new'
        for metric in METRICS:
            aa = expected[strategy + '_new']['scores'][metric]
            bb = expected['prompted_new']['scores'][metric]
            check.check(name + '.' + metric, summary['descriptive_policy_contrasts'][name][metric], contrast_expected(aa, bb))
            if metric == 'primary_em' and all(x in (0, 1) for x in aa):
                check.check(name + '.cells', summary['descriptive_policy_contrasts'][name][metric], counts(aa, bb))
    check.check('primary_alias', summary['primary'], summary['descriptive_policy_contrasts']['ranker_full_new_minus_prompted_new'])
    for metric in METRICS:
        new_difference = [x-y for x,y in zip(expected['ranker_full_new']['scores'][metric], expected['prompted_new']['scores'][metric])]
        old_difference = [x-y for x,y in zip(expected['ranker_full_old']['scores'][metric], expected['prompted_old']['scores'][metric])]
        check.check('interaction.' + metric, summary['factorial_interaction'][metric], contrast_expected(new_difference, old_difference))
        aa, bb = expected['prompted_new']['scores'][metric], expected['prompted_old']['scores'][metric]
        delta = [x-y for x,y in zip(aa, bb)]
        rng = random.Random(config['bootstrap_seed'])
        means = [math.fsum(delta[rng.randrange(len(ids))] for _ in ids) / len(ids) for _ in range(config['bootstrap_samples'])]
        check.check('fixed_prompt.' + metric, summary['fixed_policy_prompt_effect'][metric],
                    {'n_sources': len(ids), 'mean_difference': average(delta), 'per_source_differences': delta,
                     'development_paired_bootstrap_ci95': [quantile(means, .025), quantile(means, .975)]})
        if metric == 'primary_em':
            check.check('fixed_prompt.cells', summary['fixed_policy_prompt_effect'][metric], counts(aa, bb))
    for key in replays[0]:
        check.check('replay.' + key, summary['replay_agreement'][key + '_count'], sum(r[key] for r in replays))
    for domain, table, actions in (('new', new, NEW), ('old', old, OLD)):
        for action in actions:
            if action in ('selector', 'legacy_selector'):
                continue
            values = [grades[domain][eid, action] for eid in ids]
            check.check('raw.' + domain + '.' + action, summary['raw_action_quality'][domain][action],
                        {'scores': {m: [g[m] for g in values] for m in METRICS},
                         'means': {m: average(g[m] for g in values) for m in METRICS},
                         'invalid': sum(not g['parse_valid'] for g in values),
                         'truncated': sum(table[eid, action]['generation_truncated'] for eid in ids)})
    check.check('feature_cost', summary['feature_cost'], {key: stats(r[key] for r in features) for key in (*ADDITIVE[:4], *PEAK)})
    check.check('generation_time', summary['physical_generation_time_sum_s'], math.fsum(r['elapsed_s'] for r in records))
    check.check('feature_time', summary['physical_feature_time_sum_s'], math.fsum(r['elapsed_s'] for r in features))
    check.check('historical_selector', summary['historical_selector_cost'], {key: stats(old[eid, 'selector'][key] for eid in ids) for key in (*ADDITIVE[:4], *PEAK)})
    return {'status': 'PASS', 'comparisons': check.count, 'n_sources': len(ids),
            'primary_mean_difference': summary['primary']['primary_em']['mean_difference'],
            'primary_crossfit_ci_reported': False, 'new_model_calls': 0,
            'limitations': ['This is a second arithmetic implementation by the same project, not an independent human replication.',
                           'The frozen response parser/scorer is shared; this check does not establish semantic correctness.',
                           'Media-free metadata replay does not reload source pixels, inspect tensors, verify model weights, or refit ridge models. Use the strict analyzer for those source/run/fit bindings.']}


def run_check(summary_path, manifest, labels_path, records_path, archived_paths, features_path, decisions_path, lock_path, output):
    summary, lock = load(summary_path), load(lock_path)
    rows, labels, records, features = lines(manifest), lines(labels_path), lines(records_path), lines(features_path)
    archived = [r for path in archived_paths for r in lines(path)]
    decisions = load(decisions_path)
    bindings = summary['bindings']
    for key, path in (('manifest_sha256', manifest), ('lock_sha256', lock_path), ('records_sha256', records_path),
                      ('features_records_sha256', features_path), ('decisions_sha256', decisions_path)):
        require(bindings[key] == sha(path), 'Summary artifact binding differs: ' + key)
    require(bindings['labels_canonical_sha256'] == canonical(labels), 'Label binding differs')
    require(bindings['archived_records_canonical_sha256'] == canonical(archived), 'Archived bank binding differs')
    require(lock['role'] == 'main' and lock['decisions']['sha256'] == sha(decisions_path), 'Main decision lock differs')
    result = check_statistics(summary, rows, labels, records, archived, features, decisions, lock['config'])
    result.update(created_utc=datetime.now(timezone.utc).isoformat(), summary_sha256=sha(summary_path), checker_code_sha256=sha(__file__),
                  input_bindings=[{'path': str(path), 'sha256': sha(path)} for path in (summary_path, manifest, labels_path, records_path,
                                   *archived_paths, features_path, decisions_path, lock_path)])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('summary', 'manifest', 'labels', 'records', 'features', 'decisions', 'lock', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--old-records', required=True, nargs='+', type=Path)
    args = parser.parse_args()
    result = run_check(args.summary, args.manifest, args.labels, args.records, args.old_records,
                       args.features, args.decisions, args.lock, args.output)
    print(json.dumps({'status': result['status'], 'comparisons': result['comparisons'], 'n_sources': result['n_sources']}))


if __name__ == '__main__':
    main()
