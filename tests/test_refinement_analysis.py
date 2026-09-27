"""Independent small-cohort arithmetic and fail-closed analysis regressions."""
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import analyze_refinement as a


def fixture():
    rows = [{'example_id': f'e{i}', 'source_cluster_id': f's{i}', 'question': 'Which word?',
             'image_sha256': str(i) * 64, 'image_path': f'{i}.png'} for i in range(6)]
    labels = [{**r, 'answer': 'answer', 'original_answers': ['answer'],
               'validated_primary_answers': ['answer'], 'original_answer_variants': []} for r in rows]
    nc = [{1}, {1, 2}, set(), {2}, {4}, {5}]
    oc = [{2}, {2}, {2}, {3}, set(), {2}]
    full_regions = [1, 1, 2, 3, 4, 5]
    folds = [0, 0, 1, 2, 3, 4]
    new, old, features, decisions = [], [], [], []
    def record(row, action, correct, elapsed=1.0):
        return {**{k: row[k] for k in ('example_id', 'source_cluster_id')}, 'action': action, 'status': 'ok',
                'response': 'ANSWER: ' + ('answer' if correct else 'wrong'), 'raw_continuation': 'answer' if correct else 'wrong',
                'generation_truncated': False, 'elapsed_s': elapsed,
                'peak_memory_gib': 4.0, 'peak_reserved_gib': 7.0,
                'input_tokens': 100, 'visual_tokens': 80, 'generated_tokens': 2,
                'processor_observer_elapsed_s': .05}
    for i, row in enumerate(rows):
        current_region = 3 if i == 0 else 2
        for action in a.NEW_ACTIONS:
            if action.startswith('new_native_'):
                region = int(action.rsplit('_', 1)[1])
                r = record(row, action, region in nc[i], float(region))
            elif action == 'legacy_selected':
                r = record(row, action, current_region in oc[i], .75)
            else:
                r = record(row, action, action == 'new_highres', .5 if action == 'legacy_selector' else 1.0)
            if action == 'legacy_selector':
                r.update(response=f'REGION: {current_region}', selector_region=current_region, selector_valid=True)
            new.append(r)
        for action in a.OLD_ACTIONS:
            region = int(action.rsplit('_', 1)[1]) if action.startswith('native_') else None
            r = record(row, action, region in oc[i] if region else action == 'highres', float(2 * region) if region else 2.0)
            if action == 'selector':
                r.update(response='REGION: 2', selector_region=2, selector_valid=True)
            old.append(r)
        features.append({**{k: row[k] for k in ('example_id', 'source_cluster_id')},
                         'elapsed_s': 10.0, 'peak_memory_gib': 8.0, 'peak_reserved_gib': 11.0,
                         'input_tokens': 50, 'visual_tokens': 40, 'generated_tokens': 0,
                         'processor_observer_elapsed_s': .1, 'projected_features': [[0.] * 73 for _ in range(9)]})
        regions = {'full': full_regions[i], 'image_position': full_regions[i], 'position': 1,
                   'best_fixed': 3, 'center': 5, 'random': 9, 'prompted': 2}
        decisions.append({**{k: row[k] for k in ('example_id', 'source_cluster_id')}, 'fold': folds[i],
                          'regions': regions,
                          'scores': {k: [float(j == regions[k]) for j in range(1, 10)] for k in ('full', 'image_position', 'position')},
                          'cpu_elapsed_s': dict(zip(a.CPU_KINDS, (.1, .2, .3, .4, .5, .6)))})
    config = {'development_sources': 6, 'feature_dimensions': 73, 'folds': 5,
              'bootstrap_samples': 113, 'bootstrap_seed': 20260927}
    return rows, labels, new, old, features, {'predictions': decisions}, config


def compute(data=None):
    return a.compute_statistics(*(fixture() if data is None else data))


def test_source_weighted_oof_mean_and_fold_means_are_not_equal():
    result = compute()
    p = result['primary']['primary_em']
    assert p['mean_difference'] == pytest.approx(float(Fraction(1, 3)))
    assert [x['n_sources'] for x in p['folds']] == [2, 1, 1, 1, 1]
    assert [x['mean_difference'] for x in p['folds']] == [.5, 0., -1., 1., 1.]
    assert sum(x['mean_difference'] for x in p['folds']) / 5 != pytest.approx(p['mean_difference'])
    assert (p['repairs'], p['harms'], p['both_correct'], p['both_wrong']) == (3, 1, 1, 1)


def test_no_naive_crossfit_intervals_anywhere():
    result = compute()
    forbidden = ['ci95', 'confidence_interval', 'bootstrap_ci']
    for section in ('primary', 'descriptive_policy_contrasts', 'factorial_interaction', 'strategies'):
        encoded = json.dumps(result[section])
        assert all(s not in encoded for s in forbidden)
    assert result['bootstrap']['only_fixed_policy_prompt_effect'] is True


def test_same_oof_ids_factorial_interaction_matches_fraction_arithmetic():
    result = compute()
    strategies = result['strategies']
    assert strategies['ranker_full_new']['scores']['primary_em'] == [1, 1, 0, 0, 1, 1]
    assert strategies['ranker_full_old']['scores']['primary_em'] == [0, 0, 1, 1, 0, 0]
    assert strategies['prompted_new']['scores']['primary_em'] == [0, 1, 0, 1, 0, 0]
    assert strategies['prompted_old']['scores']['primary_em'] == [1, 1, 1, 0, 0, 1]
    assert result['factorial_interaction']['primary_em']['mean_difference'] == pytest.approx(float(Fraction(2, 3)))


def test_prompt_effect_uses_archived_id_not_current_selector_or_replay_grade():
    data = fixture()
    for record in data[2]:
        if record['action'] == 'legacy_selected':
            record['response'] = 'ANSWER: answer'
    result = compute(data)
    effect = result['fixed_policy_prompt_effect']['primary_em']
    assert effect['per_source_differences'] == [-1, 0, -1, 1, 0, -1]
    assert effect['mean_difference'] == pytest.approx(-1 / 3)
    assert (effect['repairs'], effect['harms']) == (1, 3)
    assert result['replay_agreement']['selector_region_equal_count'] == 5
    assert result['replay_agreement']['per_source'][0]['replayed_region'] == 3


def test_fixed_policy_ci_matches_separate_python_draws_and_scalar_quantile():
    result = compute()
    values = [-1, 0, -1, 1, 0, -1]
    rng = random.Random(20260927)
    means = sorted(float(sum(Fraction(values[rng.randrange(6)], 6) for _ in range(6))) for _ in range(113))
    def scalar_quantile(p):
        index = (len(means) - 1) * p
        lower = int(index)
        return means[lower] + (index - lower) * (means[min(lower + 1, len(means) - 1)] - means[lower])
    assert result['fixed_policy_prompt_effect']['primary_em']['development_paired_bootstrap_ci95'] == pytest.approx([scalar_quantile(.025), scalar_quantile(.975)])


def test_feature_cpu_cost_and_sequential_memory_are_charged_correctly():
    strategies = compute()['strategies']
    c = strategies['ranker_full_new']['cost_observations'][0]
    assert c['elapsed_s'] == pytest.approx(11.1)
    assert c['cpu_policy_elapsed_s'] == .1
    assert (c['input_tokens'], c['visual_tokens'], c['generated_tokens']) == (150, 120, 2)
    assert (c['peak_memory_gib'], c['peak_reserved_gib']) == (8, 11)
    assert c['processor_observer_elapsed_s'] == pytest.approx(.15)
    assert strategies['ranker_image_position_new']['cost_observations'][0]['elapsed_s'] == pytest.approx(11.2)
    p = strategies['ranker_position_new']['cost_observations'][0]
    assert p['elapsed_s'] == pytest.approx(1.3)
    assert p['input_tokens'] == 100 and p['peak_memory_gib'] == 4
    assert strategies['prompted_new']['cost_observations'][0]['elapsed_s'] == pytest.approx(2.5)
    assert strategies['prompted_new']['cost_observations'][0]['generated_tokens'] == 4


def test_old_costs_remain_historical_answer_only():
    result = compute()['strategies']
    assert result['ranker_full_old']['cost'] is None
    assert result['ranker_full_old']['historical_answer_cost_observations'][0]['elapsed_s'] == 2
    assert result['prompted_old']['historical_answer_cost_observations'][0]['elapsed_s'] == 4
    assert result['highres_old']['cost'] is None
    assert result['ranker_full_old']['historical_answer_cost_observations'][0]['cpu_policy_elapsed_s'] == 0


def test_uniform_expected_accuracy_and_latency_mixture_denominators():
    result = compute()
    uniform = result['strategies']['uniform_new']
    assert uniform['scores']['primary_em'] == pytest.approx([1/9, 2/9, 0, 1/9, 1/9, 1/9])
    assert uniform['metrics']['primary_em']['mean'] == pytest.approx(1/9)
    assert uniform['cost']['elapsed_s']['n'] == 6
    assert uniform['marginal_single_crop_cost']['elapsed_s']['n'] == 54
    assert uniform['cost']['elapsed_s']['mean'] == pytest.approx(5.6)
    assert uniform['cost']['elapsed_s']['p95'] == pytest.approx(5.6)
    assert uniform['marginal_single_crop_cost']['elapsed_s']['p95'] == pytest.approx(9.6)
    assert result['n_sources'] == 6


def test_oracle_is_per_source_and_metricwise_and_not_deployed():
    item = compute()['strategies']['oracle9_new']
    assert item['scores']['primary_em'] == [1, 1, 0, 1, 1, 1]
    assert item['metrics']['primary_em']['mean'] == pytest.approx(5/6)
    assert item['cost'] is None and item['historical_answer_cost'] is None
    assert item['deployed_policy'] is False


def test_invalid_and_truncated_answers_are_retained_in_denominator():
    data = fixture()
    record = next(r for r in data[2] if r['example_id'] == 'e0' and r['action'] == 'new_native_1')
    record['response'] = 'ANSWER: answer\nextra line'
    record['generation_truncated'] = True
    result = compute(data)
    assert result['n_sources'] == 6
    item = result['strategies']['ranker_full_new']
    assert item['scores']['primary_em'][0] == 0
    assert item['invalid_count_or_expected_count'] == item['truncated_count_or_expected_count'] == 1
    assert result['raw_action_quality']['new']['new_native_1']['invalid'] == 1


@pytest.mark.parametrize('mutation,message', [
    (lambda d: d[2].pop(), 'Incomplete'),
    (lambda d: d[3].append(deepcopy(d[3][0])), 'Duplicate'),
    (lambda d: d[0][1].update(source_cluster_id='s0'), 'Repeated'),
    (lambda d: d[1][0].update(question='Changed question'), 'identity'),
    (lambda d: d[1][0].update(answer='new label'), 'Canonical'),
    (lambda d: d[4][0].update(generated_tokens=1), 'generated'),
    (lambda d: d[4][0].update(projected_features=[[0.0]*72]*9), 'matrix'),
    (lambda d: d[4][0]['projected_features'][0].__setitem__(0, float('nan')), 'matrix'),
    (lambda d: d[5]['predictions'][0]['regions'].update(prompted=1), 'archived'),
    (lambda d: d[5]['predictions'][0]['regions'].update(full=2), 'argmax'),
    (lambda d: d[5]['predictions'][0]['regions'].update(center=4), 'Center'),
    (lambda d: d[5]['predictions'][0]['cpu_elapsed_s'].update(full=-.1), 'CPU'),
    (lambda d: d[5]['predictions'][0].update(fold=True), 'fold'),
    (lambda d: d[5]['predictions'][-1].update(fold=3), 'fold'),
    (lambda d: d[6].update(development_sources=5), 'denominator'),
    (lambda d: d[2][0].update(selector_region=7), 'Selector'),
])
def test_fail_closed_panels_and_decisions(mutation, message):
    data = fixture()
    mutation(data)
    with pytest.raises(ValueError, match=message):
        compute(data)


def test_tied_ranker_scores_require_smallest_region():
    data = fixture()
    decision = data[5]['predictions'][0]
    decision['scores']['full'] = [0.0]*9
    compute(data)
    decision['regions']['full'] = 2
    with pytest.raises(ValueError, match='argmax'):
        compute(data)


def test_markdown_reports_development_scope_and_all_baselines():
    text = a.render_markdown(compute())
    assert 'previously exposed' in text and 'No naive bootstrap interval' in text
    assert 'prompt-selection-adjusted' in text
    for name in ('ranker_full_new', 'ranker_image_position_new', 'ranker_position_new', 'best_fixed_new', 'highres_new'):
        assert name in text


def cli_fixture(tmp_path, monkeypatch, role='main'):
    data = fixture()
    rows, labels, new, old, features, decisions, config = data
    manifest = tmp_path / 'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    run_dir, feature_dir = tmp_path / 'generation', tmp_path / 'features'
    for folder, records in ((run_dir, new), (feature_dir, features)):
        folder.mkdir()
        (folder / 'run.json').write_text('{}', encoding='utf-8')
        (folder / 'completed.json').write_text('{}', encoding='utf-8')
        (folder / 'records.jsonl').write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
    feature_lock = tmp_path / 'feature_lock.json'
    feature_lock.write_text('{}', encoding='utf-8')
    decisions.update(created_utc='2026-09-27T00:01:00+00:00', training_elapsed_s=.123,
                     sources_with_nonconstant_old_native_grades=5,
                     deployment_model_has_independent_performance_estimate=False)
    decisions_path = tmp_path / 'decisions.json'
    decisions_path.write_text(json.dumps(decisions), encoding='utf-8')
    lock = {'role': role, 'config': config, 'locked_at_utc': '2026-09-27T00:02:00+00:00',
            'decisions': {'sha256': a.sha(decisions_path)},
            'prerequisites': [{'kind': 'features', 'role': 'main', 'lock': {'sha256': a.sha(feature_lock)},
                               'records': {'sha256': a.sha(feature_dir / 'records.jsonl')}}]}
    lock_path = tmp_path / 'generation_lock.json'
    lock_path.write_text(json.dumps(lock), encoding='utf-8')
    feature_lock_value = {'config': config}
    seen = []
    def validate_features(m, lp, rp):
        assert (m, lp, rp) == (manifest, feature_lock, feature_dir)
        return rows, features, {}, feature_lock_value
    def validate_decisions(p, lp, rp, m):
        assert (p, lp, rp, m) == (decisions_path, feature_lock, feature_dir, manifest)
        seen.append('decisions_revalidated')
        return decisions
    monkeypatch.setitem(sys.modules, 'refinement_execution', SimpleNamespace(
        validate_generation=lambda *_: (rows, new, {}, lock), validate_features=validate_features,
        load_prepared=lambda _: (rows, labels, old)))
    monkeypatch.setitem(sys.modules, 'train_refinement', SimpleNamespace(validate_decisions=validate_decisions))
    args = (manifest, lock_path, run_dir, feature_lock, feature_dir, decisions_path, tmp_path / 'analysis')
    return args, data, lock, feature_lock_value, seen


def test_cli_binds_components_and_passes_actual_manifest_to_training_validator(tmp_path, monkeypatch):
    args, data, lock, feature_lock, seen = cli_fixture(tmp_path, monkeypatch)
    result = a.analyze(*args)
    assert seen == ['decisions_revalidated']
    assert result['status'] == 'completed_development_diagnostic'
    assert result['bindings']['labels_canonical_sha256'] == a.canonical_records_sha256(data[1])
    assert result['bindings']['archived_records_canonical_sha256'] == a.canonical_records_sha256(data[3])
    assert result['training_metadata']['training_elapsed_s'] == .123
    assert (args[-1] / 'summary.json').is_file()


@pytest.mark.parametrize('mutation,error', [
    (lambda data, lock, fl: lock['decisions'].update(sha256='0'*64), 'decisions differ'),
    (lambda data, lock, fl: lock['prerequisites'][0]['records'].update(sha256='0'*64), 'prerequisite'),
    (lambda data, lock, fl: fl.update(config={}), 'configs differ'),
    (lambda data, lock, fl: data[5].update(created_utc='2026-09-27T00:03:00+00:00'), 'before generation lock'),
    (lambda data, lock, fl: data[5].update(created_utc='2026-09-27T00:01:00'), 'Timezone'),
])
def test_cli_rejects_alternative_or_late_artifacts(tmp_path, monkeypatch, mutation, error):
    args, data, lock, feature_lock, seen = cli_fixture(tmp_path, monkeypatch)
    mutation(data, lock, feature_lock)
    with pytest.raises(ValueError, match=error):
        a.analyze(*args)
    assert not (args[-1] / 'summary.json').exists()


def test_engineering_cli_never_loads_labels_or_calls_scorer(tmp_path, monkeypatch):
    args, data, lock, feature_lock, seen = cli_fixture(tmp_path, monkeypatch, role='engineering')
    def forbidden(*_):
        raise AssertionError('Engineering must not read labels or calculate answer scores')
    sys.modules['refinement_execution'].load_prepared = forbidden
    sys.modules['refinement_execution'].validate_features = forbidden
    monkeypatch.setattr(a, 'score_response', forbidden)
    # Engineering is meaningful before main training decisions exist.
    args = (*args[:3], None, None, None, args[-1])
    result = a.analyze(*args)
    assert result['answer_scores_withheld'] is True
    assert 'strategies' not in result and 'primary' not in result
    assert result['generation_calls'] == 78
    assert seen == []
