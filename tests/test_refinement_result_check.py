"""Second-implementation replay and adversarial summary/binding checks."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'experiments'), str(Path(__file__).resolve().parent)]
import analyze_refinement as analysis
import check_refinement_results as checker
from test_refinement_analysis import fixture


def ready():
    data = fixture()
    return data, analysis.compute_statistics(*data)


def test_second_implementation_replays_all_arithmetic():
    data, summary = ready()
    result = checker.check_statistics(summary, *data)
    assert result['status'] == 'PASS'
    assert result['comparisons'] > 2500
    assert result['primary_mean_difference'] == pytest.approx(1/3)
    assert result['primary_crossfit_ci_reported'] is False
    assert result['new_model_calls'] == 0


@pytest.mark.parametrize('mutation', [
    lambda s: s['primary']['primary_em'].update(mean_difference=.3),
    lambda s: s['factorial_interaction']['primary_em'].update(mean_difference=.5),
    lambda s: s['primary']['primary_em'].update(ci95=[0, 1]),
    lambda s: s['strategies']['ranker_full_new']['cost_observations'][0].update(elapsed_s=1.1),
    lambda s: s['strategies']['ranker_full_new']['cost_observations'][0].update(peak_memory_gib=12),
    lambda s: s['strategies']['ranker_position_new']['cost_observations'][0].update(elapsed_s=11.3),
    lambda s: s['strategies']['uniform_new']['marginal_single_crop_cost']['elapsed_s'].update(p95=5.6),
    lambda s: s['strategies']['ranker_full_old']['scores']['primary_em'].__setitem__(0, 1),
    lambda s: s['fixed_policy_prompt_effect']['primary_em'].update(mean_difference=0),
    lambda s: s['replay_agreement'].update(selector_region_equal_count=6),
    lambda s: s['raw_action_quality']['new']['new_native_1'].update(invalid=1),
    lambda s: s['strategies']['oracle9_new'].update(deployed_policy=True),
    lambda s: s['feature_cost']['generated_tokens'].update(mean=1),
])
def test_detects_wrong_summary_arithmetic(mutation):
    data, summary = ready()
    mutation(summary)
    with pytest.raises(ValueError):
        checker.check_statistics(summary, *data)


def saved_fixture(tmp_path):
    data, summary = ready()
    rows, labels, records, old, features, decisions, config = data
    paths = {}
    for name, values in [('manifest', rows), ('labels', labels), ('records', records), ('old', old), ('features', features)]:
        paths[name] = tmp_path / (name + '.jsonl')
        paths[name].write_text('\n'.join(json.dumps(r) for r in values) + '\n', encoding='utf-8')
    paths['decisions'] = tmp_path / 'decisions.json'
    paths['decisions'].write_text(json.dumps(decisions), encoding='utf-8')
    paths['lock'] = tmp_path / 'lock.json'
    lock = {'role': 'main', 'config': config, 'decisions': {'sha256': checker.sha(paths['decisions'])}}
    paths['lock'].write_text(json.dumps(lock), encoding='utf-8')
    summary['bindings'] = {
        'manifest_sha256': checker.sha(paths['manifest']), 'lock_sha256': checker.sha(paths['lock']),
        'records_sha256': checker.sha(paths['records']), 'features_records_sha256': checker.sha(paths['features']),
        'decisions_sha256': checker.sha(paths['decisions']),
        'labels_canonical_sha256': analysis.canonical_records_sha256(labels),
        'archived_records_canonical_sha256': analysis.canonical_records_sha256(old),
    }
    paths['summary'] = tmp_path / 'summary.json'
    paths['summary'].write_text(json.dumps(summary), encoding='utf-8')
    paths['output'] = tmp_path / 'check.json'
    return paths


def call(paths):
    return checker.run_check(paths['summary'], paths['manifest'], paths['labels'], paths['records'], [paths['old']],
                             paths['features'], paths['decisions'], paths['lock'], paths['output'])


def test_portable_replay_writes_sha_bound_check(tmp_path):
    paths = saved_fixture(tmp_path)
    result = call(paths)
    assert result['status'] == 'PASS'
    assert result['summary_sha256'] == checker.sha(paths['summary'])
    assert len(result['input_bindings']) == 8
    assert json.loads(paths['output'].read_text())['checker_code_sha256'] == checker.sha(checker.__file__)


@pytest.mark.parametrize('name', ['manifest', 'labels', 'records', 'old', 'features', 'decisions', 'lock'])
def test_rejects_changed_input_artifact(tmp_path, name):
    paths = saved_fixture(tmp_path)
    text = paths[name].read_text(encoding='utf-8')
    if name in ('labels', 'old'):
        # Canonical bindings intentionally tolerate whitespace, but never altered content.
        text = text.replace('answer', 'changed', 1)
    else:
        text += '\n'
    paths[name].write_text(text, encoding='utf-8')
    with pytest.raises(ValueError, match='binding'):
        call(paths)
    assert not paths['output'].exists()


def test_checker_runs_without_site_packages_or_images(tmp_path):
    paths = saved_fixture(tmp_path)
    command = [sys.executable, '-S', checker.__file__]
    for key in ('summary', 'manifest', 'labels', 'records', 'features', 'decisions', 'lock', 'output'):
        command.extend(['--' + key, str(paths[key])])
    command.extend(['--old-records', str(paths['old'])])
    run = subprocess.run(command, capture_output=True, text=True, check=False)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout)['status'] == 'PASS'


def test_scalar_quantile_matches_fixed_interpolation_without_numpy():
    assert checker.quantile([0, 10, 20, 30], .95) == pytest.approx(28.5)
    assert checker.quantile([5], .95) == 5
    assert checker.stats([1, 2, 9])['median'] == 2
