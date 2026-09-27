"""Synthetic integration of OLD response parsing with source-held-out fitting."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import train_refinement as training


@pytest.fixture
def panel(monkeypatch):
    config = {'ridge_alpha':1.0, 'ranker_kinds':['full','image_position','position'], 'seed':20260927}
    monkeypatch.setattr(training, 'read_json', lambda path:deepcopy(config))
    rng = np.random.default_rng(357)
    rows = [{'example_id':f'synthetic:{i}', 'source_cluster_id':f'cluster:{i}'} for i in range(15)]
    labels = [{'example_id':row['example_id'],'original_answers':[f'gold{i}'],
               'validated_primary_answers':[f'gold{i}'],'original_answer_variants':[]}
              for i,row in enumerate(rows)]
    features = rng.normal(size=(15,9,73))
    features[:,:,64:] = np.eye(9)
    feature_records = [{'example_id':row['example_id'],'projected_features':features[i].tolist()}
                       for i,row in enumerate(rows)]
    correct = (rng.random((15,9)) > .5)
    correct[0] = False; correct[1] = True
    records=[]
    for i,row in enumerate(rows):
        for region in range(1,10):
            records.append({'example_id':row['example_id'],'action':f'native_{region}',
                            'response':f"ANSWER: {'gold'+str(i) if correct[i,region-1] else 'wrong'}"})
        records.append({'example_id':row['example_id'],'action':'selector','selector_region':i%9+1})
    return rows, labels, records, feature_records, correct


def test_heldout_response_flips_cannot_change_own_fold_policy(panel):
    rows,labels,records,features,_ = panel
    original = training.scientific_decisions(rows,labels,records,features)
    heldout_ids = set(original['fold_models'][2]['test_example_ids'])
    altered = deepcopy(records)
    for record in altered:
        if record['example_id'] in heldout_ids and record['action'].startswith('native_'):
            gold = 'gold'+record['example_id'].split(':')[1]
            record['response'] = 'ANSWER: '+('wrong' if record['response']=='ANSWER: '+gold else gold)
    result = training.scientific_decisions(rows,labels,altered,features)
    assert original['fold_models'][2] == result['fold_models'][2]
    before={p['example_id']:training.strip_timings(p) for p in original['predictions']}
    after={p['example_id']:training.strip_timings(p) for p in result['predictions']}
    assert all(before[eid]==after[eid] for eid in heldout_ids)
    # Other folds use these responses as training labels, and may legitimately change.
    assert any(original['fold_models'][f]['rankers'] != result['fold_models'][f]['rankers']
               for f in range(5) if f != 2)


def test_old_targets_are_derived_from_raw_responses_with_frozen_parser(panel):
    rows,labels,records,features,correct = panel
    result = training.scientific_decisions(rows,labels,records,features)
    np.testing.assert_array_equal([x['native_em'] for x in result['old_targets']], correct.astype(float))
    # No updated score field may override the frozen parser applied to raw text.
    enriched=deepcopy(records)
    for row in enriched:
        row['primary_em'] = 123
        row['correct'] = True
        row['new_answer_response'] = 'ANSWER: fabricated'
    assert training.strip_timings(result) == training.strip_timings(training.scientific_decisions(rows,labels,enriched,features))


def test_all_sources_predicted_once_and_each_model_excludes_its_sources(panel):
    rows,labels,records,features,correct = panel
    result = training.scientific_decisions(rows,labels,records,features)
    ids = [row['example_id'] for row in rows]
    assert [row['example_id'] for row in result['predictions']] == ids
    assert result['sources_with_nonconstant_old_native_grades'] == int(np.sum(np.ptp(correct.astype(float),axis=1)>0))
    for fold in result['fold_models']:
        assert set(fold['test_example_ids']).isdisjoint(fold['training_example_ids'])
        assert set(fold['test_example_ids']) | set(fold['training_example_ids']) == set(ids)
        for model in fold['rankers'].values():
            assert [ids[i] for i in model['training_indices']] == fold['training_example_ids']
            assert model['n_training_rows'] == 9*len(fold['training_example_ids'])
    assert result['deployment_model_has_independent_performance_estimate'] is False
    for model in result['deployment_full_data_models'].values():
        assert model['n_training_sources'] == len(rows)


def test_label_and_feature_record_order_cannot_change_decisions(panel):
    rows,labels,records,features,_ = panel
    a=training.scientific_decisions(rows,labels,records,features)
    b=training.scientific_decisions(rows,labels[::-1],records[::-1],features[::-1])
    assert training.strip_timings(a) == training.strip_timings(b)


def test_selector_archive_id_is_preserved_and_training_timing_is_finite(panel):
    rows,labels,records,features,_ = panel
    result=training.scientific_decisions(rows,labels,records,features)
    for i,prediction in enumerate(result['predictions']):
        assert prediction['regions']['prompted'] == i%9+1
        assert set(prediction['cpu_elapsed_s']) == {'full','image_position','position','best_fixed','center','random'}
        assert all(np.isfinite(t) and t >= 0 for t in prediction['cpu_elapsed_s'].values())
    assert np.isfinite(result['training_elapsed_s']) and result['training_elapsed_s'] >= 0


def test_missing_feature_coverage_fails(panel):
    rows,labels,records,features,_ = panel
    with pytest.raises((ValueError, AssertionError)):
        training.scientific_decisions(rows,labels,records,features[:-1])
