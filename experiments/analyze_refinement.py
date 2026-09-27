"""Completed exposed-cohort refinement analysis; no generation or score changes.

Cross-fitted policies share training data across folds. Their source-weighted
means and five held-out fold results are descriptive: no naive paired-bootstrap
confidence interval is assigned to their contrasts. Only the fixed archived
selector policy's old/new instruction contrast receives a development interval.
"""
from __future__ import annotations

import argparse
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
from dude_metrics import score_response, validate_reference_sets
from region_selection_core import parse_region

NEW_ACTIONS = ('legacy_selector', 'legacy_selected', 'new_direct', 'new_highres',
               *(f'new_native_{i}' for i in range(1, 10)))
OLD_ACTIONS = ('direct', 'selector', *(f'native_{i}' for i in range(1, 10)),
               'selected_degraded', 'highres')
KINDS = ('full', 'image_position', 'position', 'best_fixed', 'center', 'random', 'prompted')
CPU_KINDS = tuple(k for k in KINDS if k != 'prompted')
METRICS = ('primary_em', 'official_anls')
SUM_FIELDS = ('elapsed_s', 'input_tokens', 'visual_tokens', 'generated_tokens')
PEAK_FIELDS = ('peak_memory_gib', 'peak_reserved_gib')
PRIMARY = 'ranker_full_new_minus_prompted_new'


def require(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def timestamp(value):
    result = datetime.fromisoformat(value.replace('Z', '+00:00'))
    require(result.tzinfo is not None, 'Timezone-aware timestamp required')
    return result


def canonical_records_sha256(records):
    ordered = sorted(records, key=lambda r: (r['example_id'], r.get('action', '')))
    return hashlib.sha256(json.dumps(ordered, sort_keys=True, ensure_ascii=False, allow_nan=False).encode('utf-8')).hexdigest()


def read(path):
    def reject(value):
        raise ValueError('Nonfinite JSON: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject)


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def mean(values):
    values = list(values)
    require(bool(values) and all(finite(x) for x in values), 'Finite nonempty values required')
    return math.fsum(values) / len(values)


def distribution(values):
    values = list(values)
    return {'n': len(values), 'mean': mean(values),
            'median': float(np.quantile(values, .5, method='linear')),
            'p95': float(np.quantile(values, .95, method='linear')),
            'min': min(values), 'max': max(values)}


def validate_cost(record, *, feature=False):
    for field in (*SUM_FIELDS, *PEAK_FIELDS):
        require(finite(record[field]) and record[field] >= 0, 'Invalid cost: ' + field)
    require(record['elapsed_s'] > 0 and record['peak_memory_gib'] > 0
            and record['peak_reserved_gib'] >= record['peak_memory_gib'], 'Invalid latency/memory')
    for field in ('input_tokens', 'visual_tokens', 'generated_tokens'):
        require(type(record[field]) is int, 'Nonintegral token count')
    require(record['input_tokens'] > 0 and record['visual_tokens'] > 0, 'Empty model inputs')
    require(record['generated_tokens'] == 0 if feature else record['generated_tokens'] > 0,
            'Incorrect generated-token count')
    observer = record.get('processor_observer_elapsed_s', 0.0)
    require(finite(observer) and 0 <= observer <= record['elapsed_s'], 'Invalid observer time')


def cost(record):
    return {**{key: record[key] for key in (*SUM_FIELDS, *PEAK_FIELDS)},
            'processor_observer_elapsed_s': record.get('processor_observer_elapsed_s', 0.0),
            'cpu_policy_elapsed_s': 0.0}


def sequential_cost(*records, cpu=0.0):
    require(finite(cpu) and cpu >= 0, 'Invalid CPU policy latency')
    parts = [cost(r) for r in records]
    require(bool(parts), 'No measured components')
    value = {key: max(p[key] for p in parts) if key in PEAK_FIELDS
             else math.fsum(p[key] for p in parts) for key in parts[0]}
    value['elapsed_s'] += cpu
    value['cpu_policy_elapsed_s'] = cpu
    return value


def average_cost(costs):
    return {key: mean(c[key] for c in costs) for key in costs[0]}


def index_records(rows, records, actions):
    records = list(records.values()) if isinstance(records, dict) else list(records)
    indexed = {(r['example_id'], r['action']): r for r in records}
    require(len(indexed) == len(records), 'Duplicate action record')
    require(set(indexed) == {(r['example_id'], a) for r in rows for a in actions}, 'Incomplete/extra action panel')
    for row in rows:
        for action in actions:
            r = indexed[row['example_id'], action]
            require(r['source_cluster_id'] == row['source_cluster_id'] and r['status'] == 'ok', 'Source/status mismatch')
            require(isinstance(r['response'], str) and type(r['generation_truncated']) is bool, 'Invalid answer record')
            validate_cost(r)
    return indexed


def selector(record):
    region, valid = parse_region(record['response'], record['generation_truncated'])
    require(type(record.get('selector_region')) is int and record['selector_region'] == region
            and record.get('selector_valid') is valid, 'Selector parser metadata differs')
    return region, valid


def validate_inputs(rows, labels, new_records, old_records, feature_records, decisions, config):
    rows = [dict(row) for row in rows]
    ids = [r['example_id'] for r in rows]
    require(len(ids) == config['development_sources'] > 0, 'Incorrect complete source denominator')
    require(len(set(ids)) == len(ids), 'Duplicate example ID')
    require(len({r['source_cluster_id'] for r in rows}) == len(rows), 'Repeated source cluster')
    new = index_records(rows, new_records, NEW_ACTIONS)
    old = index_records(rows, old_records, OLD_ACTIONS)
    label_map = {r['example_id']: r for r in labels}
    require(len(label_map) == len(labels) and set(label_map) == set(ids), 'Label coverage differs')
    features = {r['example_id']: r for r in feature_records}
    require(len(features) == len(feature_records) and set(features) == set(ids), 'Feature coverage differs')
    predictions = decisions['predictions']
    decision_map = {r['example_id']: r for r in predictions}
    require(len(decision_map) == len(predictions) and set(decision_map) == set(ids), 'Decision coverage differs')
    for row in rows:
        eid = row['example_id']
        label = label_map[eid]
        for field in ('source_cluster_id', 'question', 'image_sha256'):
            require(label[field] == row[field], 'Observation/reference identity differs: ' + field)
        validate_reference_sets(label['original_answers'], label['validated_primary_answers'], label['original_answer_variants'])
        require(label['answer'] == label['original_answers'][0], 'Canonical reference changed')
        feature = features[eid]
        require(feature['source_cluster_id'] == row['source_cluster_id'], 'Feature source differs')
        validate_cost(feature, feature=True)
        projected = np.asarray(feature['projected_features'], dtype=float)
        require(projected.shape == (9, config['feature_dimensions']) and np.isfinite(projected).all(), 'Invalid feature matrix')
        decision = decision_map[eid]
        require(decision['source_cluster_id'] == row['source_cluster_id'], 'Decision source differs')
        require(type(decision['fold']) is int and 0 <= decision['fold'] < config['folds'], 'Invalid fold')
        require(set(decision['regions']) == set(KINDS), 'Policy decision coverage differs')
        require(set(decision['cpu_elapsed_s']) == set(CPU_KINDS), 'CPU policy timing coverage differs')
        for kind in KINDS:
            require(type(decision['regions'][kind]) is int and 1 <= decision['regions'][kind] <= 9, 'Invalid region ID')
        require(decision['regions']['center'] == 5, 'Center baseline differs')
        for kind in CPU_KINDS:
            require(finite(decision['cpu_elapsed_s'][kind]) and decision['cpu_elapsed_s'][kind] >= 0, 'Invalid policy CPU cost')
        for kind in ('full', 'image_position', 'position'):
            scores = decision['scores'][kind]
            require(len(scores) == 9 and all(finite(s) for s in scores), 'Invalid ranker scores')
            require(decision['regions'][kind] == max(range(1, 10), key=lambda i: (scores[i - 1], -i)), 'Ranker argmax differs')
        archived_region, _ = selector(old[eid, 'selector'])
        selector(new[eid, 'legacy_selector'])
        require(decision['regions']['prompted'] == archived_region, 'Prompted policy changed archived region')
    require({d['fold'] for d in predictions} == set(range(config['folds'])), 'Empty/missing source fold')
    return rows, label_map, new, old, features, decision_map


def paired_interval(values, *, samples, seed):
    require(type(samples) is int and samples > 0, 'Positive bootstrap count required')
    values = np.asarray(values, dtype=float)
    require(values.ndim == 1 and len(values) > 0 and np.isfinite(values).all(), 'Invalid paired differences')
    rng = random.Random(seed)
    draws = np.asarray([[rng.randrange(len(values)) for _ in values] for _ in range(samples)], dtype=np.int64)
    return [float(v) for v in np.quantile(values[draws].mean(axis=1), [.025, .975], method='linear')]


def contrast(a, b, folds):
    delta = [x - y for x, y in zip(a, b)]
    require(len(a) == len(b) == len(folds) > 0, 'Unpaired contrast')
    return {'n_sources': len(delta), 'mean_difference': mean(delta),
            'per_source_differences': delta,
            'folds': [{'fold': f, 'n_sources': folds.count(f),
                       'a_mean': mean(x for x, k in zip(a, folds) if k == f),
                       'b_mean': mean(x for x, k in zip(b, folds) if k == f),
                       'mean_difference': mean(x for x, k in zip(delta, folds) if k == f)}
                      for f in sorted(set(folds))],
            'uncertainty': 'Descriptive source-weighted OOF mean; no naive bootstrap CI for overlapping cross-fit training sets.'}


def transition(a, b):
    """a relative to b; grades unchanged, invalid predictions remain in denominator."""
    return {'repairs': sum(x == 1 and y == 0 for x, y in zip(a, b)),
            'harms': sum(x == 0 and y == 1 for x, y in zip(a, b)),
            'both_correct': sum(x == y == 1 for x, y in zip(a, b)),
            'both_wrong': sum(x == y == 0 for x, y in zip(a, b))}


def compute_statistics(rows, labels, new_records, old_records, feature_records, decisions, config):
    rows, labels, new, old, features, decisions = validate_inputs(
        rows, labels, new_records, old_records, feature_records, decisions, config)
    strategies = {}
    per_source = []
    replay_rows = []
    folds = [decisions[r['example_id']]['fold'] for r in rows]
    uniform_mixture = []
    raw_quality = {'new': {}, 'old': {}}

    def append(name, scores, costs=None, historical=None, invalid=None, truncated=None):
        item = strategies.setdefault(name, {'scores': {m: [] for m in METRICS},
                                            'cost_observations': [], 'historical_answer_cost_observations': [],
                                            'invalid': [], 'truncated': []})
        for metric in METRICS:
            item['scores'][metric].append(scores[metric])
        if costs is not None:
            item['cost_observations'].append(costs)
        if historical is not None:
            item['historical_answer_cost_observations'].append(historical)
        item['invalid'].append(invalid)
        item['truncated'].append(truncated)

    for row in rows:
        eid = row['example_id']
        label = labels[eid]
        decision = decisions[eid]
        def grade(record):
            return score_response(record['response'], label['original_answers'], label['validated_primary_answers'], label['original_answer_variants'])
        nq = {a: grade(new[eid, a]) for a in NEW_ACTIONS if a != 'legacy_selector'}
        oq = {a: grade(old[eid, a]) for a in OLD_ACTIONS if a != 'selector'}
        for domain, grades in (('new', nq), ('old', oq)):
            for action, score in grades.items():
                item = raw_quality[domain].setdefault(action, {'scores': {m: [] for m in METRICS}, 'invalid': 0, 'truncated': 0})
                for metric in METRICS:
                    item['scores'][metric].append(score[metric])
                item['invalid'] += not score['parse_valid']
                item['truncated'] += (new if domain == 'new' else old)[eid, action]['generation_truncated']
        source_scores = {}
        source_costs = {}
        for kind in KINDS:
            region = decision['regions'][kind]
            nr, ore = new[eid, f'new_native_{region}'], old[eid, f'native_{region}']
            ns, os = nq[f'new_native_{region}'], oq[f'native_{region}']
            name = 'ranker_' + kind if kind in ('full', 'image_position', 'position') else kind
            if kind in ('full', 'image_position'):
                combined = sequential_cost(features[eid], nr, cpu=decision['cpu_elapsed_s'][kind])
            elif kind == 'prompted':
                combined = sequential_cost(new[eid, 'legacy_selector'], nr)
            else:
                combined = sequential_cost(nr, cpu=decision['cpu_elapsed_s'][kind])
            append(name + '_new', ns, costs=combined, invalid=not ns['parse_valid'], truncated=nr['generation_truncated'])
            append(name + '_old', os, historical=cost(ore), invalid=not os['parse_valid'], truncated=ore['generation_truncated'])
            source_scores[name + '_new'] = {m: ns[m] for m in METRICS}
            source_scores[name + '_old'] = {m: os[m] for m in METRICS}
            source_costs[name + '_new'] = combined
        for domain, index, grades, prefix in (('new', new, nq, 'new_'), ('old', old, oq, '')):
            for base in ('direct', 'highres'):
                a = prefix + base
                record, score = index[eid, a], grades[a]
                append(base + '_' + domain, score, costs=cost(record) if domain == 'new' else None,
                       historical=cost(record) if domain == 'old' else None,
                       invalid=not score['parse_valid'], truncated=record['generation_truncated'])
            bank = [grades[prefix + f'native_{i}'] for i in range(1, 10)]
            uniform_scores = {m: mean(s[m] for s in bank) for m in METRICS}
            oracle_scores = {m: max(s[m] for s in bank) for m in METRICS}
            bank_costs = [cost(index[eid, prefix + f'native_{i}']) for i in range(1, 10)]
            if domain == 'new':
                bank_costs = [sequential_cost(index[eid, prefix + f'native_{i}'], cpu=decision['cpu_elapsed_s']['random']) for i in range(1, 10)]
                uniform_mixture.extend(bank_costs)
            append('uniform_' + domain, uniform_scores,
                   costs=average_cost(bank_costs) if domain == 'new' else None,
                   historical=average_cost(bank_costs) if domain == 'old' else None,
                   invalid=mean(float(not s['parse_valid']) for s in bank),
                   truncated=mean(float(index[eid, prefix + f'native_{i}']['generation_truncated']) for i in range(1, 10)))
            append('oracle9_' + domain, oracle_scores)
        archived_id, archived_valid = selector(old[eid, 'selector'])
        replay_id, replay_valid = selector(new[eid, 'legacy_selector'])
        # Replay follows the current selector; compare it to that same archived bank region.
        replay = new[eid, 'legacy_selected']
        archived_same = old[eid, f'native_{replay_id}']
        replay_score, same_score = nq['legacy_selected'], oq[f'native_{replay_id}']
        replay_rows.append({'example_id': eid, 'archived_region': archived_id, 'replayed_region': replay_id,
                            'archived_selector_valid': archived_valid, 'replayed_selector_valid': replay_valid,
                            'selector_region_equal': replay_id == archived_id,
                            'selector_raw_equal': new[eid, 'legacy_selector']['response'] == old[eid, 'selector']['response'],
                            'selected_raw_equal_at_replayed_region': replay['response'] == archived_same['response'],
                            'selected_primary_em_equal_at_replayed_region': replay_score['primary_em'] == same_score['primary_em'],
                            'selected_anls_equal_at_replayed_region': replay_score['official_anls'] == same_score['official_anls'],
                            'selected_normalized_equal_at_replayed_region': bool(replay_score['parse_valid'] and same_score['parse_valid']
                                and replay_score['normalized_prediction'] == same_score['normalized_prediction']),
                            'replayed_selected_primary_em': replay_score['primary_em'],
                            'archived_same_region_primary_em': same_score['primary_em']})
        per_source.append({'example_id': eid, 'source_cluster_id': row['source_cluster_id'], 'fold': decision['fold'],
                           'regions': decision['regions'], 'scores': source_scores, 'new_policy_costs': source_costs})

    for name, value in strategies.items():
        value['metrics'] = {m: {'mean': mean(scores), 'n_sources': len(scores)} for m, scores in value['scores'].items()}
        for collection, key in (('cost_observations', 'cost'), ('historical_answer_cost_observations', 'historical_answer_cost')):
            values = value[collection]
            value[key] = {field: distribution(x[field] for x in values) for field in values[0]} if values else None
        value['invalid_count_or_expected_count'] = (math.fsum(float(x) for x in value['invalid'])
                                                     if all(x is not None for x in value['invalid']) else None)
        value['truncated_count_or_expected_count'] = (math.fsum(float(x) for x in value['truncated'])
                                                       if all(x is not None for x in value['truncated']) else None)
        value['deployed_policy'] = not name.startswith('oracle9_')
        value['cost_scope'] = ('Current measured sequential components plus measured CPU policy time; not an end-to-end policy invocation.'
                               if name.endswith('_new') and not name.startswith('oracle9_') else
                               'Archived answer-only observations; feature/CPU cost is not mixed with this historical timing.'
                               if name.endswith('_old') and not name.startswith('oracle9_') else
                               'Label-aware hindsight upper bound among nine candidates; no deployment cost assigned.')
    strategies['prompted_new']['cost_scope'] += ' Selector component is the current replay; if its ID changes, this is a latency proxy for the fixed archived-ID policy.'
    strategies['uniform_new']['marginal_single_crop_cost'] = {field: distribution(x[field] for x in uniform_mixture) for field in uniform_mixture[0]}
    strategies['uniform_new']['cost_scope'] += ' Mean includes one measured random-choice CPU cost. Cost quantiles above describe per-source expected costs; marginal_single_crop_cost uses the equally weighted 9N components, not 9N independent sources.'

    descriptive = {}
    for a, b in ((name + '_new', 'prompted_new') for name in ('ranker_full', 'ranker_image_position', 'ranker_position', 'best_fixed', 'center', 'random', 'uniform', 'direct', 'highres')):
        key = a + '_minus_' + b
        descriptive[key] = {m: contrast(strategies[a]['scores'][m], strategies[b]['scores'][m], folds) for m in METRICS}
        if all(x in (0, 1) for x in strategies[a]['scores']['primary_em']):
            descriptive[key]['primary_em'].update(transition(strategies[a]['scores']['primary_em'], strategies[b]['scores']['primary_em']))
    interaction = {}
    for metric in METRICS:
        new_delta = [a - b for a, b in zip(strategies['ranker_full_new']['scores'][metric], strategies['prompted_new']['scores'][metric])]
        old_delta = [a - b for a, b in zip(strategies['ranker_full_old']['scores'][metric], strategies['prompted_old']['scores'][metric])]
        interaction[metric] = contrast(new_delta, old_delta, folds)
    fixed_prompt = {}
    for metric in METRICS:
        a, b = strategies['prompted_new']['scores'][metric], strategies['prompted_old']['scores'][metric]
        diffs = [x - y for x, y in zip(a, b)]
        fixed_prompt[metric] = {'n_sources': len(rows), 'mean_difference': mean(diffs), 'per_source_differences': diffs,
                                'development_paired_bootstrap_ci95': paired_interval(diffs, samples=config['bootstrap_samples'], seed=config['bootstrap_seed']),
                                'interval_scope': 'Paired source-resampling description of this reused development cohort; not a new held-out result or a prompt-selection-adjusted interval.'}
    fixed_prompt['primary_em'].update(transition(strategies['prompted_new']['scores']['primary_em'], strategies['prompted_old']['scores']['primary_em']))
    for domain in raw_quality.values():
        for value in domain.values():
            value['means'] = {m: mean(v) for m, v in value['scores'].items()}

    return {'schema_version': 1, 'status': 'completed_development_diagnostic', 'role': 'main',
            'n_sources': len(rows), 'generation_calls': len(new), 'feature_forward_calls': len(features),
            'primary_contrast': PRIMARY, 'primary': descriptive[PRIMARY],
            'strategies': strategies, 'descriptive_policy_contrasts': descriptive,
            'factorial_interaction': {'definition': '(full_new - prompted_new) - (full_old - prompted_old); identical frozen per-source region IDs across instruction versions.', **interaction},
            'fixed_policy_prompt_effect': {'definition': 'New instruction minus archived old answer at the archived prompted region; current legacy_selected replay is not used as the baseline.', **fixed_prompt},
            'replay_agreement': {'per_source': replay_rows,
                                 **{key + '_count': sum(r[key] for r in replay_rows) for key in ('selector_region_equal', 'selector_raw_equal', 'selected_raw_equal_at_replayed_region', 'selected_primary_em_equal_at_replayed_region', 'selected_anls_equal_at_replayed_region', 'selected_normalized_equal_at_replayed_region')},
                                 'n_sources': len(rows), 'mismatch_changes_frozen_policy': False},
            'raw_action_quality': raw_quality, 'per_source': per_source,
            'feature_cost': {field: distribution(r[field] for r in features.values()) for field in (*SUM_FIELDS, *PEAK_FIELDS)},
            'physical_generation_time_sum_s': math.fsum(r['elapsed_s'] for r in new.values()),
            'physical_feature_time_sum_s': math.fsum(r['elapsed_s'] for r in features.values()),
            'historical_selector_cost': {field: distribution(old[r['example_id'], 'selector'][field] for r in rows) for field in (*SUM_FIELDS, *PEAK_FIELDS)},
            'bootstrap': {'only_fixed_policy_prompt_effect': True, 'samples': config['bootstrap_samples'], 'seed': config['bootstrap_seed'], 'unit': 'source', 'generator': 'Python random.Random randrange', 'quantile_method': 'numpy linear'},
            'caveats': [
                'All 124 sources were previously exposed. The new prompt was informed by their saved outputs; there is no new held-out evaluation.',
                'Ridge labels are only frozen old native-bank grades. Cross-fitting with shared training sets is not independent training replication; primary and policy interactions have no naive bootstrap CI.',
                'Source-weighted OOF means retain unequal fold sizes. Fold deltas are descriptive and are not five independent experimental replications.',
                'Answer page is provided. Candidate regions use fixed geometry without answer-ROI annotations at inference.',
                'EM/ANLS remain automatic string metrics. Format containment is not semantic correctness; invalid/truncated answers remain in denominators.',
                'CPU policy inference and the complete feature path are charged where used, including pooling, projection and projection-matrix creation. Only offline fitting is excluded from per-query policy cost and reported separately.',
                'Latency combines measured sequential components on an active desktop; it is not a timed end-to-end deployed policy. Sequential peak memory is max, not sum; reserved memory is not physical device capacity.',
                'Uniform expected accuracy averages nine candidate grades within each source. A nine-candidate oracle uses labels and is not a deployed strategy.',
                'This is a region-ranking and answer-extraction diagnostic, not a trained gain/cost controller or a transfer result.'],
            'new_model_calls_by_analyzer': 0}


def render_markdown(summary):
    if summary.get('answer_scores_withheld'):
        return '# Refinement engineering check\n\nComplete technical run; answer scores are withheld.\n'
    lines_out = ['# Exposed-cohort extraction and region-ranking diagnostic', '',
                 f"All {summary['n_sources']} sources were previously exposed. This report does not constitute a new held-out evaluation.", '',
                 '## Primary descriptive result', '']
    primary = summary['primary']['primary_em']
    lines_out += [f"Full ranker minus frozen prompted policy: **{100 * primary['mean_difference']:+.3f} percentage points**. No naive bootstrap interval is assigned to overlapping cross-fit predictions.", '',
                  '| Fold | Sources | Full EM | Prompted EM | Difference (pp) |', '|---|---:|---:|---:|---:|']
    for f in primary['folds']:
        lines_out.append(f"| {f['fold']} | {f['n_sources']} | {100*f['a_mean']:.2f}% | {100*f['b_mean']:.2f}% | {100*f['mean_difference']:+.2f} |")
    lines_out += ['', '## Current instruction strategies', '', '| Strategy | EM | Official ANLS | Mean assembled seconds |', '|---|---:|---:|---:|']
    for name in ('direct_new', 'center_new', 'random_new', 'uniform_new', 'prompted_new', 'ranker_position_new', 'ranker_image_position_new', 'ranker_full_new', 'best_fixed_new', 'highres_new', 'oracle9_new'):
        item = summary['strategies'][name]
        timing = f"{item['cost']['elapsed_s']['mean']:.4f}" if item['cost'] else 'hindsight; not deployed'
        lines_out.append(f"| {name} | {100*item['metrics']['primary_em']['mean']:.2f}% | {item['metrics']['official_anls']['mean']:.4f} | {timing} |")
    effect = summary['fixed_policy_prompt_effect']['primary_em']
    lo, hi = effect['development_paired_bootstrap_ci95']
    lines_out += ['', '## Instruction and ranking factors', '',
                  f"At the fixed archived prompted region, new minus old instruction EM is {100*effect['mean_difference']:+.3f} pp; paired development interval [{100*lo:+.3f}, {100*hi:+.3f}] pp. This is not a held-out or prompt-selection-adjusted interval.", '',
                  f"The descriptive full-versus-prompted interaction is {100*summary['factorial_interaction']['primary_em']['mean_difference']:+.3f} pp. Both instruction versions use the same frozen OOF region IDs.", '',
                  f"Current selector replay retains the archived region on {summary['replay_agreement']['selector_region_equal_count']}/{summary['n_sources']} sources. Replays never silently replace the frozen policy or old-bank baseline.", '',
                  '## Interpretation and cost limits', '']
    lines_out.extend('- ' + c for c in summary['caveats'])
    lines_out += ['', 'Full per-source metrics, fold means, replay agreements, raw-action failures, historical answer-only costs and uniform latency-mixture observations are retained in summary.json.', '']
    return '\n'.join(lines_out)


def analyze(manifest, lock_path, run_dir, feature_lock_path, feature_run_dir, decisions_path, output):
    # Lazy import keeps numerical tests independent of model/runtime modules.
    from refinement_execution import validate_generation, validate_features, load_prepared
    rows, records, run, lock = validate_generation(Path(manifest), Path(lock_path), Path(run_dir))
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    role = lock['role']
    if role == 'engineering':
        indexed = index_records(rows, records, NEW_ACTIONS)
        selector_states = [selector(indexed[row['example_id'], 'legacy_selector']) for row in rows]
        summary = {'schema_version': 1, 'status': 'completed_engineering_technical_only', 'role': role,
                   'n_sources': len(rows), 'generation_calls': len(records), 'answer_scores_withheld': True,
                   'selector_invalid_or_truncated': sum(not valid for _, valid in selector_states),
                   'truncated_generations': sum(r['generation_truncated'] for r in records),
                   'raw_action_costs': {action: {field: distribution(indexed[r['example_id'], action][field] for r in rows)
                                                for field in (*SUM_FIELDS, *PEAK_FIELDS)} for action in NEW_ACTIONS},
                   'new_model_calls_by_analyzer': 0}
    else:
        require(role == 'main', 'Unsupported refinement role')
        require(all(p is not None for p in (feature_lock_path, feature_run_dir, decisions_path)), 'Main analysis requires feature run, feature lock and decisions')
        from train_refinement import validate_decisions
        prepared, labels, old_records = load_prepared(role)
        require(prepared == rows, 'Validated preparation rows differ')
        require(sha(decisions_path) == lock['decisions']['sha256'], 'Supplied decisions differ from generation lock')
        feature_result = validate_features(Path(manifest), Path(feature_lock_path), Path(feature_run_dir))
        require(feature_result[0] == rows, 'Feature/generation source order differs')
        require(feature_result[3]['config'] == lock['config'], 'Feature/generation configs differ')
        require(any(p['kind'] == 'features' and p['role'] == 'main'
                    and p['lock']['sha256'] == sha(feature_lock_path)
                    and p['records']['sha256'] == sha(Path(feature_run_dir) / 'records.jsonl')
                    for p in lock['prerequisites']), 'Supplied feature evidence is not the generation prerequisite')
        # Exact execution return contract is deliberately explicit.
        feature_records = feature_result[1]
        decisions = validate_decisions(Path(decisions_path), Path(feature_lock_path), Path(feature_run_dir), Path(manifest))
        require(timestamp(decisions['created_utc']) <= timestamp(lock['locked_at_utc']), 'Decisions were not committed before generation lock')
        summary = compute_statistics(rows, labels, records, old_records, feature_records, decisions, lock['config'])
        summary['training_metadata'] = {key: decisions[key] for key in ('training_elapsed_s', 'sources_with_nonconstant_old_native_grades',
                                       'deployment_model_has_independent_performance_estimate')}
    summary['created_utc'] = now()
    summary['bindings'] = {'manifest_sha256': sha(manifest), 'lock_sha256': sha(lock_path),
                           'run_sha256': sha(Path(run_dir) / 'run.json'),
                           'records_sha256': sha(Path(run_dir) / 'records.jsonl'),
                           'completed_sha256': sha(Path(run_dir) / 'completed.json'),
                           'features_lock_sha256': sha(feature_lock_path) if feature_lock_path else None,
                           'decisions_sha256': sha(decisions_path) if decisions_path else None, 'analysis_code_sha256': sha(__file__)}
    if feature_run_dir is not None:
        summary['bindings']['features_run_sha256'] = sha(Path(feature_run_dir) / 'run.json')
        summary['bindings']['features_records_sha256'] = sha(Path(feature_run_dir) / 'records.jsonl')
        summary['bindings']['features_completed_sha256'] = sha(Path(feature_run_dir) / 'completed.json')
    if role == 'main':
        summary['bindings']['labels_canonical_sha256'] = canonical_records_sha256(labels)
        summary['bindings']['archived_records_canonical_sha256'] = canonical_records_sha256(old_records)
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    (output / 'report.md').write_text(render_markdown(summary), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('manifest', 'lock', 'run', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    for name in ('features-lock', 'features-run', 'decisions'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    summary = analyze(args.manifest, args.lock, args.run, args.features_lock, args.features_run, args.decisions, args.output)
    print(json.dumps({'status': summary['status'], 'n_sources': summary['n_sources'], 'output': str(args.output)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
