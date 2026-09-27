"""Fixed, CPU-only region ranker for the 124-source development experiment.

No images, answers, model calls or filesystem access occur here. Feature vectors
must come from the frozen low-resolution adapter. Nine windows are observations
within a source, never nine independent sources. All arithmetic is float64.
"""
from __future__ import annotations

import hashlib
from numbers import Integral, Real
from typing import Sequence

import numpy as np

REGIONS = 9
PROJECTION_DIM = 32
FEATURE_DIM = 73
KINDS = ('full', 'image_position', 'position')
FOLD_PREFIX = 'refinement-fold:20260927:'
SCALE_FLOOR = 1e-8


def _finite_array(value, name):
    a = np.asarray(value)
    if a.dtype.kind not in 'fiu' or not np.all(np.isfinite(a)):
        raise ValueError(f'{name} must be finite real numeric values')
    return a.astype(np.float64, copy=False)


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def project_vectors(region_vectors, query_vector, projection_seed=20260927, dim=32):
    """Return [9,73] = normalized regions, query×region, nine position bits.

    Adapter outputs are normally float32; conversion to float64 precedes the
    fixed Gaussian projection. Zero raw or projected vectors are fatal.
    """
    if _integer(dim, 'dim', 1) != PROJECTION_DIM:
        raise ValueError('The locked projection dimension is 32')
    seed = _integer(projection_seed, 'projection_seed')
    regions = _finite_array(region_vectors, 'region_vectors')
    query = _finite_array(query_vector, 'query_vector')
    if regions.ndim != 2 or regions.shape[0] != REGIONS or regions.shape[1] < 1:
        raise ValueError('region_vectors must have shape [9,D], D >= 1')
    if query.shape != (regions.shape[1],):
        raise ValueError('query_vector must have shape [D] matching regions')
    raw = np.vstack((regions, query[None, :]))
    if np.any(np.linalg.norm(raw, axis=1) == 0):
        raise ValueError('Zero raw feature vector')
    projection = np.random.default_rng(seed).standard_normal((regions.shape[1], dim))
    projection /= np.sqrt(regions.shape[1])
    projected = raw @ projection
    norms = np.linalg.norm(projected, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError('Zero or nonfinite projected feature vector')
    projected /= norms[:, None]
    result = np.concatenate((projected[:REGIONS],
                             projected[:REGIONS] * projected[REGIONS],
                             np.eye(REGIONS, dtype=np.float64)), axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError('Nonfinite projected features')
    return result


def feature_columns(kind):
    """Return the fixed column indices for one declared ablation."""
    if kind == 'full':
        return list(range(FEATURE_DIM))
    if kind == 'image_position':
        return list(range(32)) + list(range(64, 73))
    if kind == 'position':
        return list(range(64, 73))
    raise ValueError(f'Unknown ranker kind: {kind!r}')


def _features(X, allow_single=False):
    a = _finite_array(X, 'X')
    if allow_single and a.shape == (REGIONS, FEATURE_DIM):
        a = a[None, :, :]
    if a.ndim != 3 or a.shape[1:] != (REGIONS, FEATURE_DIM) or len(a) == 0:
        raise ValueError('X must have nonempty shape [n,9,73]')
    return a


def _labels(Y, n=None):
    a = _finite_array(Y, 'Y')
    if a.ndim != 2 or a.shape[1] != REGIONS or len(a) == 0:
        raise ValueError('Y must have nonempty shape [n,9]')
    if n is not None and len(a) != n:
        raise ValueError('X and Y source counts differ')
    if not np.all((a == 0) | (a == 1)):
        raise ValueError('Y must contain only binary frozen legacy EM labels')
    return a


def _training_indices(indices_train, n):
    indices = [_integer(i, 'training index') for i in indices_train]
    if not indices or len(set(indices)) != len(indices) or any(i >= n for i in indices):
        raise ValueError('Training indices must be nonempty, unique and in bounds')
    return indices


def fit_ranker(X, Y, indices_train, kind='full', alpha=1.0):
    """Fit only indexed sources and return a JSON-serializable model.

    X and Y are centered across the nine windows separately for each source.
    Scales are RMS of centered training rows, floored at 1e-8. The normal
    equations minimize mean_rows((X_scaled w - y_centered)^2)+alpha*||w||².
    Flat-label sources remain in both the row count and normalization.
    """
    features = _features(X)
    labels = _labels(Y, len(features))
    indices = _training_indices(indices_train, len(features))
    columns = feature_columns(kind)
    if isinstance(alpha, (bool, np.bool_)) or not isinstance(alpha, Real) or alpha != 1.0:
        raise ValueError('The locked ridge alpha is 1.0; no tuning is supported')
    train = features[indices][:, :, columns]
    train = train - train.mean(axis=1, keepdims=True)
    target = labels[indices]
    target = target - target.mean(axis=1, keepdims=True)
    rows = train.reshape(-1, len(columns))
    scales = np.maximum(np.sqrt(np.mean(rows * rows, axis=0)), SCALE_FLOOR)
    scaled = rows / scales
    y = target.reshape(-1)
    gram = scaled.T @ scaled / len(scaled)
    rhs = scaled.T @ y / len(scaled)
    weights = np.linalg.solve(gram + float(alpha) * np.eye(len(columns)), rhs)
    if not all(np.all(np.isfinite(a)) for a in (scales, weights)):
        raise ValueError('Nonfinite fitted model')
    prediction = scaled @ weights
    mse = float(np.mean((prediction-y)**2))
    return {
        'schema_version': 1, 'kind': kind, 'input_features': FEATURE_DIM,
        'feature_columns': columns, 'alpha': float(alpha), 'intercept': 0.0,
        'centering': 'within_source_across_all_nine_windows',
        'scale_method': 'training_centered_rows_RMS_floor_1e-8',
        'scale_floor': SCALE_FLOOR, 'scales': scales.tolist(),
        'weights': weights.tolist(), 'training_indices': indices,
        'n_training_sources': len(indices), 'n_training_rows': len(scaled),
        'n_flat_label_training_sources': int(np.sum(np.ptp(labels[indices], axis=1) == 0)),
        'training_mse_centered': mse,
        'training_objective': mse + float(alpha) * float(weights @ weights),
        'objective': 'mean_rows_squared_residual_plus_alpha_weight_squared_norm',
        'training_metrics_are_not_independent_performance': True,
    }


def predict_scores(model, X):
    """Return [n,9] scores; a single [9,73] source is accepted as n=1."""
    features = _features(X, allow_single=True)
    columns = feature_columns(model.get('kind'))
    if model.get('feature_columns') != columns or model.get('input_features') != FEATURE_DIM:
        raise ValueError('Model feature contract differs')
    if (model.get('alpha') != 1.0 or model.get('intercept') != 0.0
            or model.get('centering') != 'within_source_across_all_nine_windows'
            or model.get('scale_floor') != SCALE_FLOOR):
        raise ValueError('Model training contract differs')
    scales = _finite_array(model.get('scales'), 'model scales')
    weights = _finite_array(model.get('weights'), 'model weights')
    if scales.shape != (len(columns),) or weights.shape != scales.shape or np.any(scales < SCALE_FLOOR):
        raise ValueError('Invalid model coefficient or scaling dimensions')
    used = features[:, :, columns]
    centered = used - used.mean(axis=1, keepdims=True)
    result = (centered / scales) @ weights
    if not np.all(np.isfinite(result)):
        raise ValueError('Nonfinite prediction scores')
    return result


def select_regions(scores):
    """Return one-based IDs; exact ties use the smallest region ID."""
    a = _finite_array(scores, 'scores')
    if a.shape == (REGIONS,):
        a = a[None, :]
    if a.ndim != 2 or a.shape[1] != REGIONS or len(a) == 0:
        raise ValueError('scores must have nonempty shape [n,9]')
    return np.argmax(a, axis=1).astype(np.int64) + 1


def assign_folds(source_cluster_ids: Sequence[str], n_folds=5, prefix=FOLD_PREFIX):
    """Hash-sort unique source IDs, round-robin; output follows input order."""
    ids = list(source_cluster_ids)
    count = _integer(n_folds, 'n_folds', 2)
    if not isinstance(prefix, str) or not prefix:
        raise ValueError('Fold prefix must be a nonempty string')
    if any(not isinstance(x, str) or not x.strip() for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Source IDs must be unique nonempty strings')
    if len(ids) < count:
        raise ValueError('Every fold requires at least one source')
    ordered = sorted(range(len(ids)), key=lambda i: (hashlib.sha256((prefix+ids[i]).encode('utf-8')).hexdigest(), ids[i]))
    folds = np.empty(len(ids), dtype=np.int64)
    for rank, index in enumerate(ordered):
        folds[index] = rank % count
    return folds


def training_best_fixed(Y, indices_train):
    """Best constant candidate from training sources only, lowest-ID tie."""
    labels = _labels(Y)
    indices = _training_indices(indices_train, len(labels))
    return int(np.argmax(labels[indices].mean(axis=0))) + 1


# Execution entry point; both names implement exactly the same fold assignment.
source_folds = assign_folds


def fit_oof(X, Y, source_cluster_ids, n_folds=5, alpha=1.0):
    """Return JSON OOF coefficients, exact training/heldout IDs and predictions.

    New answer labels have no argument here. This helper does not compute a
    cross-fit confidence interval or claim independent held-out evaluation.
    """
    features = _features(X)
    labels = _labels(Y, len(features))
    ids = list(source_cluster_ids)
    if len(ids) != len(features):
        raise ValueError('Source IDs and feature source counts differ')
    folds = assign_folds(ids, n_folds=n_folds)
    scores = {kind: np.empty((len(ids), REGIONS), dtype=np.float64) for kind in KINDS}
    fixed = np.empty(len(ids), dtype=np.int64)
    entries = []
    for fold in range(n_folds):
        train = np.flatnonzero(folds != fold).tolist()
        heldout = np.flatnonzero(folds == fold).tolist()
        models = {kind: fit_ranker(features, labels, train, kind=kind, alpha=alpha) for kind in KINDS}
        best = training_best_fixed(labels, train)
        for kind, model in models.items():
            scores[kind][heldout] = predict_scores(model, features[heldout])
        fixed[heldout] = best
        entries.append({'fold': fold, 'training_indices': train, 'heldout_indices': heldout,
            'training_source_ids': [ids[i] for i in train],
            'heldout_source_ids': [ids[i] for i in heldout],
            'models': models, 'training_best_fixed_region': best})
    return {'schema_version': 1, 'scope': 'reused_source_development_out_of_fold',
        'source_cluster_ids': ids, 'fold_assignment': folds.tolist(),
        'fold_prefix': FOLD_PREFIX, 'n_folds': n_folds, 'n_sources': len(ids),
        'n_flat_label_sources': int(np.sum(np.ptp(labels, axis=1) == 0)),
        'training_labels': 'frozen_legacy_nine_native_custom_em',
        'uses_new_answer_labels': False, 'alpha': float(alpha), 'folds': entries,
        'oof_scores': {kind: value.tolist() for kind, value in scores.items()},
        'oof_selected_regions': {kind: select_regions(value).tolist() for kind, value in scores.items()},
        'oof_training_best_fixed_regions': fixed.tolist(),
        'new_held_out_generalization_claim': False,
        'crossfit_uncertainty': 'No naive independent paired-bootstrap CI for learned-policy comparisons'}
