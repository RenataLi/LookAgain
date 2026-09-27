"""Synthetic independent checks of fractional paired estimates and real costs."""
import copy
from fractions import Fraction
import itertools
from pathlib import Path
import random
import sys

import numpy as np
import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'experiments'),
               str(Path(__file__).resolve().parents[1] / 'src')]
import analyze_region_selection as analysis
from region_selection_core import Observation


@pytest.fixture
def panel():
    config = {'source_counts': {'engineering': 4, 'development': 4, 'evaluation': 4},
              'seed': 20260927, 'bootstrap_samples': 256, 'bootstrap_seed': 20260927}
    rows, labels, records = [], [], []
    correct_sets = [set(range(1,10)), {1}, {1,2,3,4}, set()]
    selected = [2,1,9,5]
    lp = [-2., -.25, -1., None]
    for i in range(4):
        row = {'example_id': f'example-{i}', 'question': 'Which word?',
               'image_path': f'images/{i}.png', 'image_sha256': str(i)*64,
               'source_cluster_id': f'cluster-{i}'}
        rows.append(Observation.from_manifest(row))
        labels.append({**row, 'answer': 'yes', 'original_answers': ['yes'],
                       'original_answer_variants': ['YES'], 'validated_primary_answers': ['yes']})
        for action in analysis.ACTIONS:
            if action.startswith('native_'):
                region = int(action.split('_')[1]); correct = region in correct_sets[i]
                elapsed, memory, tokens = float(region), 5+region/10, 300+region
            elif action == 'selector':
                correct = False; elapsed, memory, tokens = 3.,8.,200
            elif action == 'direct':
                correct = i == 2; elapsed, memory, tokens = 2.,10.,100
            elif action == 'selected_degraded':
                correct = i in (1,2); elapsed, memory, tokens = 7.,6.,400
            else:
                correct = True; elapsed, memory, tokens = 15.,12.,1000
            response = 'ANSWER: '+('yes' if correct else 'no')
            if action == 'direct' and i == 1:
                response = 'ANSWER:'  # Invalid but relatively confident: must not force zoom.
            if action == 'selector':
                response = f'REGION: {selected[i]}' if i != 3 else 'region: 5'
            record = {'example_id': row['example_id'], 'source_cluster_id': row['source_cluster_id'],
                      'action': action, 'status': 'ok', 'response': response,
                      'generation_truncated': False, 'elapsed_s': elapsed,
                      'peak_memory_gib': memory, 'peak_reserved_gib': memory+1,
                      'input_tokens': tokens, 'visual_tokens': tokens-20, 'generated_tokens': 2,
                      'processor_observer_elapsed_s': .01, 'mean_token_logprob': lp[i] if action=='direct' else -.2}
            if action == 'selector':
                record.update(selector_region=selected[i], selector_valid=i != 3)
            records.append(record)
    bindings = {'development_records_sha256': 'a'*64, 'development_lock_sha256': 'b'*64, 'config_sha256': 'c'*64}
    calibration = analysis.calibrate_confidence(rows, records, config, bindings)
    calibration['created_utc'] = '2026-09-25T00:00:00+00:00'
    return rows, records, labels, config, calibration


def result(panel, **kwargs):
    rows, records, labels, config, calibration = panel
    return analysis.compute_statistics(rows, records, labels, config, 'evaluation', calibration, **kwargs)


def test_fractional_primary_hand_counts_and_nine_way_weights(panel):
    summary = result(panel)
    selected = summary['strategies']['selected_native']
    uniform = summary['strategies']['uniform_expected_native']
    assert selected['scores']['primary_em'] == [1.,1.,0.,0.]
    assert uniform['scores']['primary_em'] == pytest.approx([1.,1/9,4/9,0.])
    assert selected['metrics']['primary_em']['mean'] == .5
    assert uniform['metrics']['primary_em']['mean'] == pytest.approx(7/18)
    assert summary['primary_result']['difference'] == pytest.approx(1/9)
    assert summary['primary_result']['per_source_difference'] == pytest.approx([0.,8/9,-4/9,0.])
    assert 'automatic_transitions' not in summary['contrasts'][analysis.PRIMARY]
    assert 'mcnemar' not in summary['primary_result']
    assert summary['n_sources'] == 4 and summary['n_records'] == 52


def test_ci_matches_complete_exact_paired_enumeration(monkeypatch, panel):
    draws = np.asarray(list(itertools.product(range(4), repeat=4)))
    monkeypatch.setattr(analysis, 'paired_draws', lambda n,samples,seed: draws)
    summary = result(panel)
    delta = [Fraction(0), Fraction(8,9), Fraction(-4,9), Fraction(0)]
    exact = sorted(sum((delta[i] for i in draw), Fraction(0))/4 for draw in itertools.product(range(4), repeat=4))
    def linear_quantile(fraction):
        index = fraction*(len(exact)-1); lo = index.numerator//index.denominator
        weight = index-lo
        return float(exact[lo]*(1-weight)+exact[lo+1]*weight)
    expected = [linear_quantile(Fraction(1,40)), linear_quantile(Fraction(39,40))]
    assert summary['primary_result']['ci95'] == pytest.approx(expected)
    assert expected[0] < 0 < expected[1]


def test_draws_are_complete_paired_sources_in_fixed_python_rng_order():
    actual = analysis.paired_draws(4, 11, 20260927)
    rng = random.Random(20260927)
    expected = [[rng.randrange(4) for _ in range(4)] for _ in range(11)]
    np.testing.assert_array_equal(actual, expected)


def test_selector_is_charged_fully_to_both_selected_alternatives(panel):
    summary = result(panel)
    selected = summary['strategies']['selected_native']
    degraded = summary['strategies']['selected_degraded']
    assert selected['cost_observations']['elapsed_s'] == [5.,4.,12.,8.]
    assert selected['costs']['elapsed_s']['mean'] == 7.25
    assert degraded['cost_observations']['elapsed_s'] == [10.]*4
    assert selected['cost_observations']['input_tokens'] == [502.,501.,509.,505.]
    assert selected['cost_observations']['peak_memory_gib'] == [8.]*4
    assert selected['cost_observations']['processor_observer_elapsed_s'] == [.02]*4


def test_uniform_expected_and_randomized_marginal_cost_quantiles_differ(panel):
    uniform = result(panel)['strategies']['uniform_expected_native']
    assert uniform['costs']['elapsed_s']['mean'] == 5.
    assert uniform['costs']['elapsed_s']['median'] == 5.
    assert uniform['costs']['elapsed_s']['p95'] == 5.
    assert uniform['marginal_single_crop_costs']['elapsed_s']['n'] == 36
    assert uniform['marginal_single_crop_costs']['elapsed_s']['p95'] == 9.
    assert 'not the marginal' in uniform['cost_distribution_scope']


def test_fixed_center_random_and_oracle_are_separate_policies(panel):
    summary = result(panel)
    assert summary['strategies']['center_native']['scores']['primary_em'] == [1.,0.,0.,0.]
    assert summary['strategies']['hindsight_oracle_native']['scores']['primary_em'] == [1.,1.,1.,0.]
    assert [row['oracle_primary_region'] for row in summary['per_source']] == [1,1,1,1]
    assert summary['strategies']['hindsight_oracle_native']['costs']['elapsed_s']['mean'] == 1.
    assert summary['strategies']['seeded_random_native']['cost_observations']['elapsed_s'] == [float(row['random_region']) for row in summary['per_source']]


def test_gate_ties_missing_and_invalid_answer_do_not_change_locked_rule(panel):
    summary = result(panel)
    assert summary['confidence_gates']['0.25']['zoom_per_source'] == [True,False,False,True]
    assert summary['confidence_gates']['0.5']['zoom_per_source'] == [True,False,True,True]
    assert summary['confidence_gates']['0.25']['zoom_fraction'] == .5
    assert summary['strategies']['confidence_gate_0.25']['scores']['primary_em'] == [1.,0.,1.,0.]
    assert summary['strategies']['confidence_gate_0.5']['scores']['primary_em'] == [1.,0.,0.,0.]
    assert summary['strategies']['confidence_gate_0.25']['cost_observations']['elapsed_s'] == [7.,2.,2.,10.]
    assert summary['strategies']['confidence_gate_0.5']['cost_observations']['elapsed_s'] == [7.,2.,14.,10.]
    assert summary['strategies']['confidence_gate_0.5']['cost_observations']['peak_memory_gib'] == [10.]*4
    assert summary['invalid_answers']['direct'] == 1


def test_calibration_uses_finite_logprobs_not_labels_or_answer_correctness(monkeypatch, panel):
    rows, records, _, config, _ = panel
    def forbidden(*args, **kwargs):
        raise AssertionError('Calibration must never score answers')
    monkeypatch.setattr(analysis, 'score_response', forbidden)
    bindings = {'development_records_sha256': 'a'*64, 'development_lock_sha256': 'b'*64, 'config_sha256': 'c'*64}
    calibration = analysis.calibrate_confidence(rows, records, config, bindings)
    assert calibration['thresholds'] == {'0.25': -1.5, '0.5': -1., '0.75': -.625}
    assert calibration['finite_development_sources'] == 3
    assert calibration['missing_development_sources'] == 1
    assert calibration['uses_reference_labels'] is False
    assert calibration['development_direct_confidence'] == [
        {'example_id':f'example-{i}', 'mean_token_logprob':value}
        for i,value in enumerate([-2.,-.25,-1.,None])]


def test_all_missing_confidence_fails_calibration(panel):
    rows, records, _, config, _ = copy.deepcopy(panel)
    for record in records:
        if record['action'] == 'direct':
            record['mean_token_logprob'] = None
    with pytest.raises(ValueError, match='No finite'):
        analysis.calibrate_confidence(rows, records, config, {'development_records_sha256': 'a'*64, 'development_lock_sha256': 'b'*64, 'config_sha256': 'c'*64})


def test_engineering_withholds_quality_and_never_reads_labels(monkeypatch, panel):
    rows, records, _, config, _ = panel
    monkeypatch.setattr(analysis, 'score_response', lambda *args: (_ for _ in ()).throw(AssertionError('No engineering scoring')))
    summary = analysis.compute_statistics(rows, records, object(), config, 'engineering')
    assert summary['answer_scores_withheld'] is True
    assert 'primary_result' not in summary and 'strategies' not in summary and 'invalid_answers' not in summary
    assert summary['selector_invalid_or_truncated'] == 1


def test_discrete_fix_harm_denominators_are_not_primary_fractional_pairs(panel):
    transitions = result(panel)['contrasts']['selected_native_minus_selected_degraded']['automatic_transitions']
    assert transitions == {'fixes':1, 'harms':1, 'baseline_wrong':2, 'baseline_correct':2, 'fix_rate':.5, 'harm_rate':.5}


def test_policy_format_rates_follow_reused_answer_and_uniform_fraction(panel):
    rows, records, labels, config, calibration = copy.deepcopy(panel)
    changed = next(record for record in records if record['example_id']=='example-0' and record['action']=='native_2')
    changed['response'] = 'ANSWER:'
    changed['generation_truncated'] = True
    summary = analysis.compute_statistics(rows,records,labels,config,'evaluation',calibration)
    assert summary['strategies']['selected_native']['format_rates'] == {'invalid_answer':.25,'truncated_answer':.25}
    assert summary['strategies']['uniform_expected_native']['format_rates'] == pytest.approx({'invalid_answer':1/36,'truncated_answer':1/36})
    assert summary['strategies']['center_native']['format_rates'] == {'invalid_answer':0.,'truncated_answer':0.}
    assert summary['strategies']['confidence_gate_0.25']['format_rates']['invalid_answer'] == .5


@pytest.mark.parametrize('tamper', ['missing', 'duplicate', 'source', 'selector', 'role_count', 'confidence_nan'])
def test_incomplete_or_corrupt_panel_is_rejected(panel, tamper):
    rows, records, labels, config, calibration = copy.deepcopy(panel)
    if tamper == 'missing': records.pop()
    if tamper == 'duplicate': records.append(copy.deepcopy(records[0]))
    if tamper == 'source': records[0]['source_cluster_id'] = 'other'
    if tamper == 'selector': records[1]['selector_region'] = 3
    if tamper == 'role_count': config['source_counts']['evaluation'] = 5
    if tamper == 'confidence_nan': records[0]['mean_token_logprob'] = float('nan')
    with pytest.raises(ValueError):
        analysis.compute_statistics(rows, records, labels, config, 'evaluation', calibration)


@pytest.mark.parametrize('field', ['question', 'source_cluster_id', 'image_path', 'image_sha256', 'answer'])
def test_labels_cannot_change_observation_identity_or_canonical_reference(panel, field):
    rows, records, labels, config, calibration = copy.deepcopy(panel)
    labels[0][field] = 'changed'
    with pytest.raises(ValueError):
        analysis.compute_statistics(rows, records, labels, config, 'evaluation', calibration)


def test_evaluation_requires_frozen_calibration(panel):
    rows, records, labels, config, _ = panel
    with pytest.raises(ValueError, match='requires frozen'):
        analysis.compute_statistics(rows, records, labels, config, 'evaluation')


@pytest.mark.parametrize('tamper', ['unordered', 'missing', 'nan', 'naive', 'wrong_config'])
def test_calibration_schema_and_config_binding(panel, tamper):
    *_, config, calibration = copy.deepcopy(panel)
    if tamper == 'unordered': calibration['thresholds']['0.25'] = 0.
    if tamper == 'missing': calibration['thresholds'].pop('0.5')
    if tamper == 'nan': calibration['thresholds']['0.5'] = float('nan')
    if tamper == 'naive': calibration['created_utc'] = '2026-09-25T00:00:00'
    if tamper == 'wrong_config': calibration['config_sha256'] = 'f'*64
    with pytest.raises(ValueError):
        analysis.validate_calibration(calibration, config, 'c'*64)


def mocked_bound_analyze(monkeypatch, panel, tamper=None):
    import run_region_selection as runner
    rows, records, labels, config, calibration = copy.deepcopy(panel)
    lock = {'role':'evaluation', 'config':config, 'config_sha256':'c'*64,
            'labels_sha256':'d'*64, 'calibration_sha256':'e'*64, 'calibration':copy.deepcopy(calibration),
            'locked_at_utc':'2026-09-25T00:01:00+00:00'}
    if tamper == 'labels_hash': lock['labels_sha256'] = 'a'*64
    if tamper == 'calibration_hash': lock['calibration_sha256'] = 'a'*64
    if tamper == 'snapshot': lock['calibration']['thresholds']['0.25'] = -1.6
    if tamper == 'late_calibration': lock['locked_at_utc'] = '2026-09-24T00:00:00+00:00'
    monkeypatch.setattr(runner, 'validate_completed', lambda *args: (rows,records,{'role':'evaluation'},lock))
    monkeypatch.setattr(analysis, 'sha', lambda path: {'labels.jsonl':'d'*64, 'calibration.json':'e'*64}.get(Path(path).name,'a'*64))
    monkeypatch.setattr(analysis, 'read', lambda path: calibration)
    monkeypatch.setattr(analysis, 'lines', lambda path: labels)
    return analysis.analyze(Path('manifest.jsonl'), Path('labels.jsonl'), Path('lock.json'), Path('run'), Path('calibration.json'))


def test_completed_validator_integration_and_binding_fields(monkeypatch, panel):
    summary = mocked_bound_analyze(monkeypatch, panel)
    assert summary['bindings']['labels_sha256'] == 'd'*64
    assert summary['bindings']['calibration_sha256'] == 'e'*64
    assert summary['new_model_calls'] == 0


@pytest.mark.parametrize('tamper', ['labels_hash', 'calibration_hash', 'snapshot', 'late_calibration'])
def test_completed_validator_composition_rejects_changed_artifacts(monkeypatch, panel, tamper):
    with pytest.raises(ValueError):
        mocked_bound_analyze(monkeypatch, panel, tamper)
