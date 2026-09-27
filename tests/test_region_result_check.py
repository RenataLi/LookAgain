"""Independent checker smoke: synthetic arithmetic only, no model/run files."""
from copy import deepcopy
from fractions import Fraction
import itertools
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import check_region_results as check


@pytest.fixture
def panel():
    bits = [[1,0,0,0,0,0,0,0,0], [1,1,1,1,0,0,0,0,0], [1,0,1,0,1,0,1,0,1]]
    rows, labels, records = [], [], []
    for i in range(3):
        row = {'example_id': f'synthetic:{i}', 'source_cluster_id': f'cluster:{i}',
               'question': 'Is the target present?', 'image_path': f'{i}.png', 'image_sha256': str(i)*64}
        rows.append(row)
        labels.append({**row, 'answer': 'YES', 'original_answers': ['YES'],
                       'original_answer_variants': [], 'validated_primary_answers': ['YES']})
        for action in check.ACTIONS:
            if action == 'direct':
                elapsed, allocated = 10*(i+1), [8,20,4][i]
                response = ['ANSWER: NO', 'ANSWER:', 'ANSWER: YES'][i]
            elif action == 'selector':
                elapsed, allocated = i+1, [12,6,30][i]
                response = ['REGION: 1', 'REGION: 5', 'REGION: 9 extra'][i]
            elif action.startswith('native_'):
                j = int(action.split('_')[1])
                elapsed, allocated = 100*i+j, 2+i+j
                response = 'ANSWER: ' + ('YES' if bits[i][j-1] else 'NO')
            elif action == 'selected_degraded':
                elapsed, allocated = 20+i, 5+i
                response = 'ANSWER: ' + ('YES' if i==1 else 'NO')
            else:
                elapsed, allocated, response = 50+i, 40+i, 'ANSWER: YES'
            record = {'example_id': row['example_id'], 'source_cluster_id': row['source_cluster_id'],
                      'action': action, 'status': 'ok', 'response': response, 'generation_truncated': False,
                      'elapsed_s': elapsed, 'peak_memory_gib': allocated, 'peak_reserved_gib': allocated+20,
                      'input_tokens': 100+elapsed, 'visual_tokens': 64, 'generated_tokens': 3,
                      'processor_observer_elapsed_s': .01*(i+1),
                      'mean_token_logprob': [-2.,-1.,None][i] if action=='direct' else -.5}
            if action == 'selector':
                record.update(selector_region=[1,5,5][i], selector_valid=i!=2)
            records.append(record)
    return rows, labels, records


def rebuilt(panel, samples=101):
    return check.rebuild(*panel, {'0.25':-2., '0.5':-1., '0.75':-.5}, expected_n=3, samples=samples)


def test_closed_form_primary_and_seeded_10k_interval(panel):
    result = rebuilt(panel, 10000)
    assert result['primary_result']['difference'] == pytest.approx(Fraction(8,27))
    assert result['primary_result']['per_source_difference'] == pytest.approx([Fraction(8,9),Fraction(-4,9),Fraction(4,9)])
    assert result['primary_result']['ci95'] == pytest.approx([Fraction(-4,9),Fraction(8,9)])
    assert result['strategies']['selected_native']['metrics']['primary_em']['mean'] == pytest.approx(Fraction(2,3))
    assert result['strategies']['uniform_expected_native']['metrics']['primary_em']['mean'] == pytest.approx(Fraction(10,27))
    assert 'automatic_transitions' not in result['contrasts'][check.PRIMARY]


def test_scalar_percentile_matches_exhaustive_27_fraction_reference():
    delta = [Fraction(8,9),Fraction(-4,9),Fraction(4,9)]
    means = [float(sum((delta[j] for j in draw), Fraction()) / 3)
             for draw in itertools.product(range(3), repeat=3)]
    assert check.quantile(means,.025) == pytest.approx(Fraction(-34,135))
    assert check.quantile(means,.975) == pytest.approx(Fraction(107,135))


def test_gate_ties_missing_invalid_and_sequential_full_cost(panel):
    result = rebuilt(panel)
    strategies = result['strategies']
    assert result['confidence_gates']['0.25']['zoom_per_source'] == [True,False,True]
    assert result['confidence_gates']['0.5']['zoom_per_source'] == [True,True,True]
    assert strategies['confidence_gate_0.25']['scores']['primary_em'] == [1.,0.,1.]
    assert strategies['confidence_gate_0.25']['cost_observations']['elapsed_s'] == [12.,20.,238.]
    assert strategies['confidence_gate_0.5']['cost_observations']['elapsed_s'] == [12.,127.,238.]
    assert strategies['selected_native']['cost_observations']['elapsed_s'] == [2.,107.,208.]
    assert strategies['selected_degraded']['cost_observations']['elapsed_s'] == [21.,23.,25.]
    assert strategies['selected_native']['cost_observations']['peak_memory_gib'] == [12,8,30]
    assert strategies['confidence_gate_0.25']['cost_observations']['processor_observer_elapsed_s'] == pytest.approx([.03,.02,.09])


def test_uniform_marginal_percentile_is_not_percentile_of_means(panel):
    expected = rebuilt(panel)['strategies']['uniform_expected_native']
    assert expected['costs']['elapsed_s']['p95'] == pytest.approx(195.)
    assert expected['marginal_single_crop_costs']['elapsed_s']['p95'] == pytest.approx(207.7)
    assert expected['cost_observations']['peak_memory_gib'] == [7.,8.,9.]
    assert expected['marginal_single_crop_costs']['elapsed_s']['n'] == 27


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_source', 'bad_selector', 'nonfinite'])
def test_strict_panel_rejects_consequential_damage(panel, mutation):
    rows, labels, records = deepcopy(panel)
    if mutation=='missing': records.pop()
    if mutation=='duplicate': records.append(deepcopy(records[0]))
    if mutation=='wrong_source': records[0]['source_cluster_id']='other'
    if mutation=='bad_selector': next(r for r in records if r['action']=='selector')['selector_region']=9
    if mutation=='nonfinite': records[0]['mean_token_logprob']=float('nan')
    with pytest.raises(ValueError):
        check.rebuild(rows, labels, records, {'0.25':-2.,'0.5':-1.,'0.75':-.5}, expected_n=3, samples=5)


def test_comparator_detects_grade_and_cost_tampering(panel):
    correct = rebuilt(panel)
    changed = deepcopy(correct)
    changed['strategies']['selected_native']['cost_observations']['peak_memory_gib'][0] += 3
    changed['primary_result']['difference'] = 0
    comparisons, failures = check.compare(changed, correct)
    assert comparisons > 1000
    assert len(failures)==2
    assert any(f['path'].endswith('/difference') for f in failures)


def test_fractional_primary_cannot_add_binary_transitions(panel):
    correct = rebuilt(panel)
    changed = deepcopy(correct)
    changed['contrasts'][check.PRIMARY]['automatic_transitions'] = {'fixes':2}
    assert check.compare(changed, correct)[1]


def test_calibration_recomputed_from_contributors_without_labels():
    calibration = {'schema_version':1, 'method':'numpy_linear_quantile',
        'missing_confidence_rule':'zoom', 'tie_rule':'confidence <= threshold zooms',
        'created_utc':'2026-09-27T00:00:00+00:00', 'development_sources':3,
        'finite_development_sources':2, 'missing_development_sources':1,
        'development_direct_confidence':[{'example_id':f'dev{i}','mean_token_logprob':v}
                                          for i,v in enumerate([-2.,-1.,None])],
        'thresholds':{'0.25':-1.75,'0.5':-1.5,'0.75':-1.25},
        'uses_reference_labels':False, 'new_model_calls':0}
    assert check.verify_calibration(calibration, 3, ['eval0']) == calibration['thresholds']
    calibration['thresholds']['0.5']=-1.4
    with pytest.raises(ValueError,match='Quantile recomputation'):
        check.verify_calibration(calibration, 3)


def test_incomplete_run_guard_precedes_response_and_summary_reads():
    with patch.object(Path, 'is_file', return_value=False), patch.object(check,'read',side_effect=AssertionError('Read before completion')):
        with pytest.raises(ValueError,match='completed.json'):
            check.check_files('manifest','labels','run','summary','calibration')


def test_json_duplicate_keys_are_rejected():
    with pytest.raises(ValueError,match='Duplicate JSON key'):
        check.decode('{"records":780,"records":1}')
