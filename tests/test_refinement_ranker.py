"""Independent numerical and leakage tests; synthetic CPU arrays only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import refinement_ranker as ranker


def fixture_arrays(n=15):
    rng = np.random.default_rng(119)
    X = rng.normal(size=(n, 9, 73))
    X[:, :, 64:] = np.eye(9)
    Y = (rng.random((n, 9)) > .55).astype(float)
    Y[0] = 0
    Y[1] = 1
    return X, Y


def test_projection_matches_independent_definition_and_normalization():
    rng = np.random.default_rng(811)
    regions = rng.normal(size=(9, 17)).astype(np.float32)
    query = rng.normal(size=17).astype(np.float32)
    matrix = np.random.default_rng(20260927).standard_normal((17, 32)) / np.sqrt(17)
    pr = regions.astype(float) @ matrix
    pq = query.astype(float) @ matrix
    pr /= np.linalg.norm(pr, axis=1)[:, None]
    pq /= np.linalg.norm(pq)
    expected = np.column_stack((pr, pr * pq, np.eye(9)))
    actual = ranker.project_vectors(regions, query)
    assert actual.dtype == np.float64 and actual.shape == (9, 73)
    np.testing.assert_allclose(actual, expected, atol=1e-15, rtol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(actual[:, :32], axis=1), 1)
    np.testing.assert_array_equal(actual[:, 64:], np.eye(9))
    np.testing.assert_array_equal(actual, ranker.project_vectors(regions, query))


def test_query_changes_only_interaction_features():
    rng = np.random.default_rng(8)
    regions = rng.normal(size=(9, 19)).astype(np.float32)
    query = rng.normal(size=19).astype(np.float32)
    first = ranker.project_vectors(regions, query)
    second = ranker.project_vectors(regions, -query)
    np.testing.assert_array_equal(first[:, :32], second[:, :32])
    np.testing.assert_allclose(first[:, 32:64], -second[:, 32:64])
    np.testing.assert_array_equal(first[:, 64:], second[:, 64:])


@pytest.mark.parametrize('bad', ['zero_region', 'zero_query', 'nan', 'inf', 'wrong_regions', 'wrong_query'])
def test_projection_rejects_unusable_inputs(bad):
    regions = np.ones((9, 4), dtype=np.float32)
    query = np.ones(4, dtype=np.float32)
    if bad == 'zero_region': regions[3] = 0
    elif bad == 'zero_query': query[:] = 0
    elif bad == 'nan': regions[0, 0] = np.nan
    elif bad == 'inf': query[1] = np.inf
    elif bad == 'wrong_regions': regions = regions[:8]
    elif bad == 'wrong_query': query = query[:3]
    with pytest.raises(ValueError): ranker.project_vectors(regions, query)


def test_zero_projected_vector_is_fatal(monkeypatch):
    class ZeroProjection:
        def standard_normal(self, shape): return np.zeros(shape)
    monkeypatch.setattr(ranker.np.random, 'default_rng', lambda seed: ZeroProjection())
    with pytest.raises(ValueError, match='projected'):
        ranker.project_vectors(np.ones((9, 4)), np.ones(4))


@pytest.mark.parametrize('kind,columns', [
    ('full', list(range(73))),
    ('image_position', list(range(32)) + list(range(64, 73))),
    ('position', list(range(64, 73))),
])
def test_ridge_matches_augmented_least_squares_and_stationarity(kind, columns):
    X, Y = fixture_arrays()
    train = [0, 1, 3, 6, 8, 10, 12]
    fitted = ranker.fit_ranker(X, Y, train, kind=kind)
    x = X[train][:, :, columns]
    x = (x-x.mean(axis=1, keepdims=True)).reshape(-1, len(columns))
    y = Y[train]
    y = (y-y.mean(axis=1, keepdims=True)).ravel()
    rms = np.maximum(np.sqrt(np.mean(x**2, axis=0)), 1e-8)
    z = x / rms
    # Independent formulation: least squares on data and a regularizer block.
    augmented_X = np.vstack((z / np.sqrt(len(z)), np.eye(len(columns))))
    augmented_y = np.r_[y / np.sqrt(len(y)), np.zeros(len(columns))]
    expected, *_ = np.linalg.lstsq(augmented_X, augmented_y, rcond=None)
    np.testing.assert_allclose(fitted['scales'], rms, atol=1e-14)
    np.testing.assert_allclose(fitted['weights'], expected, atol=2e-14, rtol=1e-12)
    w = np.asarray(fitted['weights'])
    gradient = 2*z.T@(z@w-y)/len(y)+2*w
    np.testing.assert_allclose(gradient, 0, atol=2e-14)
    assert fitted['n_training_sources'] == len(train)
    assert fitted['n_training_rows'] == len(train)*9
    assert fitted['n_flat_label_training_sources'] == 2
    assert fitted['feature_columns'] == columns
    json.dumps(fitted, allow_nan=False)


def test_heldout_features_and_labels_cannot_change_fit_or_training_scale():
    X, Y = fixture_arrays()
    train = [0, 1, 4, 7, 9]
    holdout = [i for i in range(len(X)) if i not in train]
    initial = ranker.fit_ranker(X, Y, train)
    X_changed = X.copy(); Y_changed = Y.copy()
    X_changed[holdout] *= 1e8
    X_changed[holdout] += 1e5
    Y_changed[holdout] = 1-Y_changed[holdout]
    assert ranker.fit_ranker(X_changed, Y_changed, train) == initial


@pytest.mark.parametrize('kind', ranker.KINDS)
def test_per_source_common_feature_offsets_cancel_in_fit_and_prediction(kind):
    X, Y = fixture_arrays()
    offsets = np.random.default_rng(11).normal(size=(len(X), 1, 73))*5
    train = [0, 2, 3, 6, 8, 10]
    a = ranker.fit_ranker(X, Y, train, kind=kind)
    b = ranker.fit_ranker(X+offsets, Y, train, kind=kind)
    np.testing.assert_allclose(a['weights'], b['weights'], atol=1e-14)
    np.testing.assert_allclose(a['scales'], b['scales'], atol=1e-14)
    np.testing.assert_allclose(ranker.predict_scores(a, X), ranker.predict_scores(a, X+offsets), atol=1e-14)
    np.testing.assert_allclose(ranker.predict_scores(a, X).mean(axis=1), 0, atol=1e-15)


def test_duplicated_sources_preserve_mean_loss_regularization():
    X, Y = fixture_arrays(10)
    single = ranker.fit_ranker(X, Y, list(range(10)))
    duplicated = ranker.fit_ranker(np.concatenate((X,X)), np.concatenate((Y,Y)), list(range(20)))
    np.testing.assert_allclose(single['weights'], duplicated['weights'], atol=1e-14)
    assert duplicated['n_training_rows'] == 180
    # A sum-loss ridge fit would change its effective regularization here.


@pytest.mark.parametrize('kind', ranker.KINDS)
@pytest.mark.parametrize('label_type', ['zeros', 'ones', 'mixed_flat'])
def test_flat_label_sources_retained_and_yield_exact_zero_scores(kind, label_type):
    X, _ = fixture_arrays(10)
    Y = np.zeros((10,9))
    if label_type == 'ones': Y[:] = 1
    if label_type == 'mixed_flat': Y[::2] = 1
    fitted = ranker.fit_ranker(X, Y, list(range(10)), kind=kind)
    assert fitted['n_training_sources'] == fitted['n_flat_label_training_sources'] == 10
    np.testing.assert_array_equal(fitted['weights'], 0)
    scores = ranker.predict_scores(fitted, X)
    np.testing.assert_array_equal(scores, np.zeros((10,9)))
    np.testing.assert_array_equal(ranker.select_regions(scores), np.ones(10))


def test_no_between_region_feature_information_has_no_predictive_power():
    X, Y = fixture_arrays(10)
    X[:] = X[:, :1, :]
    fitted = ranker.fit_ranker(X, Y, list(range(10)))
    # Choose exactly representable common values to avoid numerical centering noise.
    X[:] = np.arange(73)
    fitted = ranker.fit_ranker(X, Y, list(range(10)))
    np.testing.assert_array_equal(fitted['scales'], np.full(73, 1e-8))
    np.testing.assert_array_equal(fitted['weights'], np.zeros(73))
    np.testing.assert_array_equal(ranker.select_regions(ranker.predict_scores(fitted, X)), np.ones(10))


def test_ablations_really_ignore_omitted_features():
    X, Y = fixture_arrays()
    train = list(range(10))
    for kind, omitted in [('image_position', list(range(32,64))), ('position', list(range(64)))]:
        a = ranker.fit_ranker(X, Y, train, kind=kind)
        changed = X.copy(); changed[:,:,omitted] *= -1e6
        b = ranker.fit_ranker(changed, Y, train, kind=kind)
        assert a == b
        np.testing.assert_array_equal(ranker.predict_scores(a, X), ranker.predict_scores(a, changed))


def test_fold_assignment_matches_hash_sort_and_survives_input_reordering():
    ids = [f'cluster:{i}' for i in range(124)]
    actual = ranker.source_folds(ids)
    expected = {}
    for position, cid in enumerate(sorted(ids, key=lambda x:hashlib.sha256(('refinement-fold:20260927:'+x).encode()).hexdigest())):
        expected[cid] = position % 5
    assert actual.tolist() == [expected[cid] for cid in ids]
    reverse = ranker.assign_folds(ids[::-1])
    assert reverse.tolist() == actual[::-1].tolist()
    assert sorted(np.bincount(actual).tolist()) == [24,25,25,25,25]


@pytest.mark.parametrize('ids', [['a']*5, ['a','b','c','d',''], ['a','b','c','d',None], ['a','b']])
def test_invalid_source_identifiers_are_rejected(ids):
    with pytest.raises(ValueError): ranker.assign_folds(ids)


def test_best_fixed_is_training_only_and_ties_choose_lowest_region():
    Y = np.zeros((10,9))
    Y[:5,2] = 1; Y[:5,6] = 1
    Y[5:,8] = 1
    assert ranker.training_best_fixed(Y, list(range(5))) == 3
    Y[5:] = 1-Y[5:]
    assert ranker.training_best_fixed(Y, list(range(5))) == 3
    assert ranker.training_best_fixed(np.ones((2,9)), [0,1]) == 1
    np.testing.assert_array_equal(ranker.select_regions([[0,2,2,0,0,0,0,0,0]]), [2])


def test_oof_training_proofs_and_each_fold_no_own_labels():
    X, Y = fixture_arrays()
    ids = [f'source:{i}' for i in range(len(X))]
    result = ranker.fit_oof(X, Y, ids)
    assert result['n_sources'] == 15 and result['n_flat_label_sources'] == 2
    assert result['uses_new_answer_labels'] is False
    all_test = []
    for entry in result['folds']:
        train, test = entry['training_indices'], entry['heldout_indices']
        assert set(train).isdisjoint(test) and set(train)|set(test) == set(range(15))
        assert entry['training_source_ids'] == [ids[i] for i in train]
        assert entry['heldout_source_ids'] == [ids[i] for i in test]
        all_test += test
        alternate = Y.copy(); alternate[test] = 1-alternate[test]
        for kind in ranker.KINDS:
            # The model for this fold cannot depend on this fold's labels.
            assert ranker.fit_ranker(X, alternate, train, kind=kind) == entry['models'][kind]
            scores = ranker.predict_scores(entry['models'][kind], X[test])
            np.testing.assert_array_equal(scores, np.asarray(result['oof_scores'][kind])[test])
            np.testing.assert_array_equal(ranker.select_regions(scores), np.asarray(result['oof_selected_regions'][kind])[test])
    assert sorted(all_test) == list(range(15))
    json.dumps(result, allow_nan=False)


def test_source_order_permutation_preserves_oof_predictions():
    X, Y = fixture_arrays()
    ids = [f'source:{i}' for i in range(len(X))]
    first = ranker.fit_oof(X, Y, ids)
    permutation = np.random.default_rng(39).permutation(len(X))
    second = ranker.fit_oof(X[permutation], Y[permutation], [ids[i] for i in permutation])
    inverse = np.argsort(permutation)
    for kind in ranker.KINDS:
        np.testing.assert_allclose(first['oof_scores'][kind], np.asarray(second['oof_scores'][kind])[inverse], atol=2e-14)
        np.testing.assert_array_equal(first['oof_selected_regions'][kind], np.asarray(second['oof_selected_regions'][kind])[inverse])


@pytest.mark.parametrize('indices', [[], [0,0], [-1], [15], [True], [1.5]])
def test_invalid_training_indices_are_rejected(indices):
    X, Y = fixture_arrays()
    with pytest.raises(ValueError): ranker.fit_ranker(X, Y, indices)


@pytest.mark.parametrize('alpha', [0, -.5, 2, True, np.nan, np.inf])
def test_no_regularization_tuning_supported(alpha):
    X, Y = fixture_arrays()
    with pytest.raises(ValueError): ranker.fit_ranker(X, Y, [0,1], alpha=alpha)


@pytest.mark.parametrize('bad', ['nonbinary_labels','nan_labels','nan_features','wrong_shape','unknown_kind'])
def test_malformed_training_inputs_fail_closed(bad):
    X, Y = fixture_arrays()
    kind = 'full'
    if bad == 'nonbinary_labels': Y[2,1] = .5
    elif bad == 'nan_labels': Y[2,1] = np.nan
    elif bad == 'nan_features': X[2,1,1] = np.nan
    elif bad == 'wrong_shape': X = X[:,:,:72]
    elif bad == 'unknown_kind': kind = 'tuned'
    with pytest.raises(ValueError): ranker.fit_ranker(X,Y,[0,1,2],kind=kind)


@pytest.mark.parametrize('bad', ['wrong_columns','negative_scale','nan_weight','intercept','alpha'])
def test_corrupted_serialized_model_rejected(bad):
    X, Y = fixture_arrays()
    model = deepcopy(ranker.fit_ranker(X,Y,[0,1,2]))
    if bad == 'wrong_columns': model['feature_columns'][0] = 1
    elif bad == 'negative_scale': model['scales'][0] = -1
    elif bad == 'nan_weight': model['weights'][0] = np.nan
    elif bad == 'intercept': model['intercept'] = 1
    elif bad == 'alpha': model['alpha'] = 2
    with pytest.raises(ValueError): ranker.predict_scores(model, X)


def test_single_source_prediction_and_json_roundtrip():
    X, Y = fixture_arrays()
    model = ranker.fit_ranker(X,Y,[0,1,2])
    restored = json.loads(json.dumps(model,allow_nan=False))
    np.testing.assert_array_equal(ranker.predict_scores(restored,X[4]),ranker.predict_scores(model,X[4:5]))
