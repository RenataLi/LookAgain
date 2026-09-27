"""Adversarial execution and projected split checks; no source text or model generation."""
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image, ImageDraw
import pytest

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'experiments'))
import run_region_selection as runner
from region_selection_core import Observation


def write_lines(path, records):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in records), encoding='utf-8')


def save(case):
    write_lines(case['run_dir'] / 'records.jsonl', case['records'])
    case['completion']['records_sha256'] = runner.sha_file(case['run_dir'] / 'records.jsonl')
    case['completion']['records'] = len(case['records'])
    runner.write_json(case['run_dir'] / 'completed.json', case['completion'])
    runner.write_json(case['run_dir'] / 'run.json', case['run'])


@pytest.fixture
def case(tmp_path, monkeypatch):
    """Real pixel/request/parser checks; only the external lock is stubbed.

    Captures are explicit synthetic fixtures, not claims of processor execution.
    A second fixture hook can supply them to a CPU-preflight contract.
    """
    config = runner.read_json(runner.CONFIG)
    config = deepcopy(config)
    config['source_counts']['engineering'] = 1
    source = Image.new('RGB', (544, 672), 'white')
    draw = ImageDraw.Draw(source)
    for x in range(5, source.width, 23):
        draw.line((x, 0, x, source.height), fill=(x % 255, 34, 91), width=3)
    source.save(tmp_path / 'page.png')
    observation = Observation('synthetic-question', 'What year is shown?', 'page.png',
                              runner.sha_file(tmp_path / 'page.png'), 'synthetic-source')
    manifest = tmp_path / 'observation_manifest.jsonl'
    write_lines(manifest, [asdict(observation)])
    captures, records = {}, []
    for action in runner.action_order(observation.example_id, config['seed']):
        _, images, geometry = runner.make_request(source, observation.question, action, config, 2)
        grids = [[1, im.height // 16, im.width // 16] for im in images]
        visual = sum(t * h * w // 4 for t, h, w in grids)
        capture = {'input_tokens': visual + 90, 'visual_tokens': visual, 'image_grid_thw': grids,
                   'tensor_sha256': {key: hashlib.sha256((action + key).encode()).hexdigest()
                                     for key in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')},
                   'formatted_prompt_sha256': hashlib.sha256((action + '-prompt').encode()).hexdigest()}
        captures[action] = deepcopy(capture)
        response = 'REGION: 2' if action == 'selector' else 'ANSWER: 2019'
        record = {'example_id': observation.example_id, 'source_cluster_id': observation.source_cluster_id,
                  'action': action, 'status': 'ok', 'observation_sha256': runner.canonical_hash(asdict(observation)),
                  'geometry': geometry, 'response': response,
                  'raw_continuation': response if action == 'selector' else response[len('ANSWER:'):],
                  'answer_prefix_prefilled': action != 'selector', 'generated_tokens': 4,
                  'generation_truncated': False, 'input_tokens': capture['input_tokens'],
                  'visual_tokens': capture['visual_tokens'], 'image_grid_thw': capture['image_grid_thw'],
                  'processor_capture': capture, 'processor_observer_elapsed_s': .01,
                  'elapsed_s': .5, 'peak_memory_gib': 8., 'peak_reserved_gib': 9., 'mean_token_logprob': -.3}
        if action == 'selector':
            record.update(selector_region=2, selector_valid=True)
        records.append(record)
    lock = {'locked_at_utc': '2026-09-26T00:00:00Z', 'run_id': 'synthetic-engineering', 'role': 'engineering',
            'config': config, 'manifest_sha256': runner.sha_file(manifest), 'expected_records': 13,
            'runtime': {'synthetic': True}, 'code_bindings': {'synthetic': 'c' * 64},
            'model': {'model_id': 'synthetic'}, 'model_file_sha256': {'weights': 'a' * 64}}
    lock_path = tmp_path / 'lock.json'
    runner.write_json(lock_path, lock)
    monkeypatch.setattr(runner, 'validate_lock', lambda *args, **kwargs: (lock, config))
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    run = {key: lock[key] for key in ('config', 'role', 'run_id', 'runtime', 'code_bindings', 'model', 'model_file_sha256', 'manifest_sha256')}
    run.update(execution_lock_sha256=runner.sha_file(lock_path), created_utc='2026-09-26T00:01:00Z',
               inference_labels_loaded=False, answer_page_privileged=True, roi_annotation_used=False,
               warmup_calls=[{'source':'synthetic_white','size':list(size),'action':a,
                              'max_new_tokens':2,'generated_tokens':2}
                             for size in ((1024,768),(768,1024),(1024,1024)) for a in ('direct','selector')])
    preflight = {'example_id':observation.example_id, 'observation_sha256':runner.canonical_hash(asdict(observation)), 'actions':{}}
    for r in records:
        key = 'degraded_2' if r['action'] == 'selected_degraded' else r['action']
        preflight['actions'][key] = {'geometry_sha256':runner.canonical_hash(r['geometry']),
                                     'processor_capture':deepcopy(r['processor_capture'])}
    for selected in range(1,10):
        if selected == 2:
            continue
        _,_,geometry = runner.make_request(source,observation.question,'selected_degraded',config,selected)
        capture = deepcopy(captures['selected_degraded'])
        capture['tensor_sha256']['pixel_values'] = hashlib.sha256(('degraded-'+str(selected)).encode()).hexdigest()
        preflight['actions']['degraded_'+str(selected)] = {
            'geometry_sha256':runner.canonical_hash(geometry),'processor_capture':capture}
    monkeypatch.setattr(runner,'preflight_rows',lambda role:{observation.example_id:preflight})
    completion = {'records': 13, 'finished_utc': '2026-09-26T00:02:00Z', 'elapsed_s': 9.,
                  'sum_measured_action_elapsed_s': 6.5,
                  'elapsed_s_scope': 'Single uninterrupted segment after synthetic warmup; loading/preparation excluded'}
    result = dict(observation=observation, source=source, config=config, captures=captures, records=records,
                  lock=lock, lock_path=lock_path, manifest=manifest, run_dir=run_dir, run=run, completion=completion,
                  preflight=preflight)
    save(result)
    return result


def validate(case):
    return runner.validate_completed(case['manifest'], case['lock_path'], case['run_dir'])


def record(case, action):
    return next(r for r in case['records'] if r['action'] == action)


def test_complete_synthetic_pairing(case):
    observations, records, _, _ = validate(case)
    assert len(observations) == 1 and len(records) == 13


@pytest.mark.parametrize('field,value', [
    ('raw_continuation', ' 2020'), ('answer_prefix_prefilled', False), ('generated_tokens', True),
    ('generated_tokens', 65), ('generation_truncated', 1), ('generation_truncated', True),
    ('elapsed_s', False), ('elapsed_s', -1), ('processor_observer_elapsed_s', 1.),
    ('peak_reserved_gib', 7.), ('mean_token_logprob', .7), ('source_cluster_id', 'other'),
    ('observation_sha256', '0' * 64), ('status', 'failed'),
])
def test_record_corruption_rejected_even_when_completion_rehashed(case, field, value):
    record(case, 'native_2')[field] = value
    save(case)
    with pytest.raises(ValueError):
        validate(case)


def test_missing_action_never_becomes_smaller_complete_subset(case):
    case['records'] = [r for r in case['records'] if r['action'] != 'native_8']
    save(case)
    with pytest.raises(ValueError):
        validate(case)


def test_duplicate_and_unknown_actions_rejected(case):
    record(case, 'native_8')['action'] = 'native_9'
    save(case)
    with pytest.raises(ValueError):
        validate(case)


def test_tampered_geometry_rejected(case):
    record(case, 'native_2')['geometry']['region_id'] = 9
    save(case)
    with pytest.raises(ValueError):
        validate(case)


@pytest.mark.parametrize('invalid,truncated', [('REGION: 9 because...', False), ('REGION: 2', True), ('', False)])
def test_invalid_or_truncated_selector_kept_with_real_center_degraded_pixels(case, invalid, truncated):
    selector = record(case, 'selector')
    selector.update(response=invalid, raw_continuation=invalid, selector_region=5, selector_valid=False,
                    generation_truncated=truncated,
                    generated_tokens=case['config']['selector_max_new_tokens'] if truncated else 4)
    _, _, geometry = runner.make_request(case['source'], case['observation'].question,
                                         'selected_degraded', case['config'], 5)
    record(case, 'selected_degraded')['geometry'] = geometry
    record(case, 'selected_degraded')['processor_capture'] = deepcopy(case['preflight']['actions']['degraded_5']['processor_capture'])
    save(case)
    assert len(validate(case)[1]) == 13
    record(case, 'selected_degraded')['geometry']['region_id'] = 2
    save(case)
    with pytest.raises(ValueError):
        validate(case)


@pytest.mark.parametrize('field,value', [('run_id', 'another-run'), ('inference_labels_loaded', True), ('roi_annotation_used', True)])
def test_identity_and_annotation_boundary_claims_checked(case, field, value):
    case['run'][field] = value
    save(case)
    with pytest.raises(ValueError):
        validate(case)


@pytest.mark.parametrize('field,value', [('finished_utc', '2026-09-25T23:00:00Z'),
                                       ('elapsed_s', -1.), ('sum_measured_action_elapsed_s', 99.)])
def test_completion_chronology_and_costs_checked(case, field, value):
    case['completion'][field] = value
    save(case)
    with pytest.raises(ValueError):
        validate(case)


def test_existing_failed_run_cannot_silently_resume(case, monkeypatch):
    from types import SimpleNamespace
    def forbidden(*args, **kwargs):
        raise AssertionError('Must not construct a model for an existing run directory')
    monkeypatch.setitem(sys.modules, 'lookagain.backend', SimpleNamespace(QwenBackend=forbidden))
    monkeypatch.setattr(runner, 'configure_runtime', lambda *args: case['lock']['runtime'])
    with pytest.raises(ValueError, match='new run directory'):
        runner.execute(SimpleNamespace(lock=case['lock_path'], manifest=case['manifest'], role='engineering',
                                       model_dir=case['run_dir'], output=case['run_dir']))


@pytest.mark.parametrize('extra', ['answer', 'validated_primary_answers', 'roi_pixels', 'ocr', 'source_urls'])
def test_annotation_fields_cannot_enter_observation_loader(case, extra):
    row = asdict(case['observation'])
    row[extra] = 'forbidden'
    write_lines(case['manifest'], [row])
    with pytest.raises(ValueError, match='five allowed fields'):
        runner.observations(case['manifest'], 'engineering', case['config'])


def test_numeric_split_projection_counts_and_exclusion():
    """Rechecks exported membership arithmetic, not original source provenance.

    Texts, source identifiers and original content hashes are not distributed.
    Schema/approval booleans describe checks made during the local export; this
    test cannot independently repeat the underlying source audit.
    """
    path = PROJECT / 'reports/cohort_splits.json'
    assert runner.sha_file(path) == '2eea42f7cb3175778f99f7e4bdbf2b05e6d942e0c496dde7c838237651736f83'
    projection = runner.read_json(path)
    assert projection['original_source_provenance_revalidated_by_public_fixture'] is False
    assert projection['contains_original_source_identifiers'] is False
    assert projection['contains_questions_answers_or_media'] is False
    assert 'original_file_commitments' not in projection
    prior = projection['prior_main_source_indices']
    engineering = projection['prior_engineering_source_indices']
    historical = projection['historical_outcome_source_indices']
    for indices, n in ((prior, 660), (engineering, 11), (historical, 670)):
        assert all(type(x) is int and x >= 0 for x in indices)
        assert len(indices) == len(set(indices)) == n
    observed = set()
    for role, n in [('development', 64), ('evaluation', 60), ('engineering', 2)]:
        rows = projection['cohorts'][role]
        assert len(rows) == n
        for row in rows:
            assert set(row) == {'source_index', 'observation_schema_valid',
                                'observation_fields_match_label', 'source_audit_approved'}
            assert type(row['source_index']) is int and row['source_index'] >= 0
            assert row['observation_schema_valid'] is True
            assert row['observation_fields_match_label'] is True
            assert row['source_audit_approved'] is True
            assert row['source_index'] not in observed
            observed.add(row['source_index'])
            if role == 'evaluation':
                assert row['source_index'] not in set(prior) | set(engineering) | set(historical)
    assert len(observed) == 126


@pytest.mark.parametrize('change', ['input_tokens', 'tensor_sha256', 'formatted_prompt_sha256', 'empty_tensor_map'])
def test_coherently_rehashed_actual_processor_corruption_rejected(case, change):
    row=record(case,'native_2')
    capture=row['processor_capture']
    if change=='input_tokens':
        capture['input_tokens']+=1
        row['input_tokens']=capture['input_tokens']
    elif change=='empty_tensor_map':
        capture['tensor_sha256']={}
    elif change=='tensor_sha256':
        capture['tensor_sha256']['pixel_values']='0'*64
    else:
        capture['formatted_prompt_sha256']='0'*64
    save(case)
    with pytest.raises(ValueError,match='processor capture'):
        validate(case)


def test_capture_returns_original_cpu_batch_and_rejects_before_cuda():
    import torch
    batch={'input_ids':torch.tensor([[1,2,3]]), 'attention_mask':torch.tensor([[1,1,1]]),
           'pixel_values':torch.zeros((16,3)), 'image_grid_thw':torch.tensor([[1,4,4]])}
    capture=runner.ProcessorCapture(lambda **kwargs:batch)
    assert capture(text=['synthetic prompt']) is batch
    capture.expected=deepcopy(capture.last)
    assert capture(text=['synthetic prompt']) is batch
    batch['pixel_values'][0,0]=1
    with pytest.raises(ValueError,match='Actual CPU processor inputs'):
        capture(text=['synthetic prompt'])


@pytest.fixture
def locked_case(tmp_path,monkeypatch):
    """Real lock guards on synthetic reservation bytes; external source audit is stubbed.

    The real production validator still checks byte hashes, ordered identities,
    cohort disjointness, chronology, calibration and label mutation. Only this
    fixture's expected reservation digest replaces the historical digest.
    """
    project=tmp_path/'project'; preparation=project/'reports/region_selection_preparation'
    preparation.mkdir(parents=True)
    reservation={'schema_version':1,'reserved_at_utc':'2026-09-26T00:00:00Z',
                 'cohorts':{},'historical_outcome_source_ids':['synthetic-prior-source']}
    for role,n in [('engineering',2),('development',64),('evaluation',60)]:
        observations=[asdict(Observation(f'synthetic-{role}-question-{i}',
                                        'What is shown in the synthetic fixture?',
                                        f'synthetic-{role}-{i}.png','a'*64,
                                        f'synthetic-{role}-source-{i}')) for i in range(n)]
        labels=[{**row,'answer':'synthetic answer','source_semantic_audit_status':'approved'}
                for row in observations]
        op=preparation/(role+'_observation_manifest.jsonl')
        lp=preparation/(role+'_labels.jsonl')
        write_lines(op,observations);write_lines(lp,labels)
        reservation['cohorts'][role]={'n':n,'example_ids':[r['example_id'] for r in observations],
                                     'source_cluster_ids':[r['source_cluster_id'] for r in observations],
                                     'observation_manifest_sha256':runner.sha_file(op),
                                     'labels_sha256':runner.sha_file(lp)}
    reservation_path=preparation/'split_reservation.json'
    runner.write_json(reservation_path,reservation)
    monkeypatch.setattr(runner,'RESERVATION_SHA',runner.sha_file(reservation_path))
    config=runner.read_json(runner.CONFIG)
    config_path=project/'configs/region_selection.json';config_path.parent.mkdir(parents=True)
    runner.write_json(config_path,config)
    (project/'experiments').mkdir()
    shutil.copyfile(PROJECT/'experiments/analyze_region_selection.py',project/'experiments/analyze_region_selection.py')
    input_report=project/'reports/region_selection_inputs';input_report.mkdir()
    for role in ('engineering','development','evaluation'):
        for extension in ('.jsonl','.meta.json'):
            runner.write_json(input_report/(role+extension),{'explicit_synthetic_fixture':role})
    monkeypatch.setattr(runner,'PROJECT',project)
    monkeypatch.setattr(runner,'CONFIG',config_path)
    monkeypatch.setattr(runner,'PREPARATION',preparation)
    monkeypatch.setattr(runner,'INPUT_REPORT',input_report)
    monkeypatch.setattr(runner,'validate_source_design',lambda:None)
    codes={'synthetic.py':'c'*64}
    monkeypatch.setattr(runner,'code_bindings',lambda:codes)
    model={'model_id':config['model_id'],'revision':config['model_revision']}
    files={'synthetic.safetensors':'a'*64};runtime={'synthetic':True}
    evidence={'code_bindings':codes,'model_file_sha256':files,'runtime':runtime,
              'records_sha256':'b'*64,'lock_sha256':'d'*64}
    engineering={**evidence,'records':26,'run_id':'synthetic-engineering','finished_utc':'2026-09-27T01:00:00Z'}
    development={**evidence,'records':832,'run_id':'synthetic-development','finished_utc':'2026-09-27T02:00:00Z',
                 'direct_confidence':[{'example_id':f'synthetic-{i}','mean_token_logprob':-float(i+1) if i<63 else None}
                                      for i in range(64)]}
    calibration={'schema_version':1,'created_utc':'2026-09-27T03:00:00Z',
                 'config_sha256':runner.sha_file(config_path),'method':'numpy_linear_quantile','missing_confidence_rule':'zoom',
                 'thresholds':{'0.25':-47.5,'0.5':-32.,'0.75':-16.5},'finite_development_sources':63,'development_sources':64,
                 'missing_development_sources':1,'uses_reference_labels':False,
                 'development_direct_confidence':deepcopy(development['direct_confidence']),
                 'analysis_code_sha256':runner.sha_file(project/'experiments/analyze_region_selection.py'),
                 'development_records_sha256':development['records_sha256'],'development_lock_sha256':development['lock_sha256']}
    manifest=runner.public_manifest('evaluation')
    lock={'schema_version':1,'execution_locked':True,'role':'evaluation','run_id':'synthetic-evaluation',
          'locked_at_utc':'2026-09-27T04:00:00Z','config':config,'config_sha256':runner.sha_file(config_path),
          'code_bindings':codes,'manifest_sha256':runner.sha_file(manifest),'labels_sha256':runner.sha_file(runner.public_labels('evaluation')),
          'expected_records':780,'model':model,'model_file_sha256':files,'runtime':runtime,
          'verification':{'status':'PASS','code_bindings':codes},'source_bindings':runner.source_bindings(),
          'input_bindings':runner.input_bindings(),'engineering_evidence':engineering,'development_evidence':development,
          'calibration':calibration,'calibration_canonical_sha256':runner.canonical_hash(calibration)}
    lock_path=tmp_path/'execution-lock.json';runner.write_json(lock_path,lock)
    return {'lock':lock,'lock_path':lock_path,'manifest':manifest,'preparation':preparation}


def check_lock(case):
    runner.write_json(case['lock_path'],case['lock'])
    return runner.validate_lock(case['lock_path'],case['manifest'],'evaluation')


def test_accepts_complete_synthetic_lock_chain(locked_case):
    assert check_lock(locked_case)[0]['role']=='evaluation'


@pytest.mark.parametrize('field,value',[('source_bindings',{}),('input_bindings',{}),
                                     ('labels_sha256','0'*64),('expected_records',779),('run_id',' ')])
def test_lock_cannot_drop_source_input_or_label_bindings(locked_case,field,value):
    locked_case['lock'][field]=value
    with pytest.raises(ValueError):check_lock(locked_case)


def test_lock_cannot_substitute_other_roles_manifest_even_with_updated_hash(locked_case):
    locked_case['manifest']=runner.public_manifest('development')
    locked_case['lock']['manifest_sha256']=runner.sha_file(locked_case['manifest'])
    with pytest.raises(ValueError,match='reserved manifest'):check_lock(locked_case)


@pytest.mark.parametrize('which,field,value',[
    ('engineering_evidence','runtime',{'different':True}),
    ('engineering_evidence','model_file_sha256',{'other':'0'*64}),
    ('engineering_evidence','code_bindings',{'other':'0'*64}),
    ('engineering_evidence','run_id','synthetic-evaluation'),
    ('engineering_evidence','finished_utc','2026-09-27T05:00:00Z'),
    ('development_evidence','runtime',{'different':True}),
    ('development_evidence','model_file_sha256',{'other':'0'*64}),
    ('development_evidence','records',831),
    ('development_evidence','run_id','synthetic-engineering'),
    ('development_evidence','finished_utc','2026-09-27T03:01:00Z'),
])
def test_prerequisite_identity_and_chronology_not_declarations_only(locked_case,which,field,value):
    locked_case['lock'][which][field]=value
    with pytest.raises(ValueError):check_lock(locked_case)


@pytest.mark.parametrize('field,value',[('created_utc','2026-09-27T05:00:00Z'),
                                     ('finite_development_sources',True),('finite_development_sources',1.5),
                                     ('development_records_sha256','0'*64)])
def test_calibration_source_and_chronology_are_strict(locked_case,field,value):
    lock=locked_case['lock'];lock['calibration'][field]=value
    lock['calibration_canonical_sha256']=runner.canonical_hash(lock['calibration'])
    with pytest.raises(ValueError):check_lock(locked_case)


def test_reserved_labels_cannot_change_even_with_rehashed_lock(locked_case):
    path=runner.public_labels('evaluation')
    labels=runner.read_jsonl(path);labels[0]['answer']='invented'
    write_lines(path,labels)
    locked_case['lock']['labels_sha256']=runner.sha_file(path)
    locked_case['lock']['source_bindings']=runner.source_bindings()
    with pytest.raises(ValueError,match='Reserved label'):check_lock(locked_case)


@pytest.mark.parametrize('field,value',[
    ('thresholds',{'0.25':-48.,'0.5':-32.,'0.75':-16.5}),
    ('finite_development_sources',64),('missing_development_sources',0),
    ('uses_reference_labels',True),('analysis_code_sha256','0'*64)])
def test_prescribed_development_quantiles_recomputed_from_hand_panel(locked_case,field,value):
    lock=locked_case['lock']
    # Hand panel -63..-1 plus None: linear Q1=-47.5, median=-32, Q3=-16.5.
    runner.validate_development_thresholds(lock['calibration'],lock['development_evidence'])
    lock['calibration'][field]=value
    lock['calibration_canonical_sha256']=runner.canonical_hash(lock['calibration'])
    with pytest.raises(ValueError):check_lock(locked_case)


def test_development_confidence_cannot_duplicate_source(locked_case):
    evidence=locked_case['lock']['development_evidence']
    evidence['direct_confidence'][1]['example_id']=evidence['direct_confidence'][0]['example_id']
    with pytest.raises(ValueError,match='coverage'):check_lock(locked_case)


def test_cpu_preflight_observation_identity_is_checked(case):
    case['preflight']['observation_sha256']='0'*64
    with pytest.raises(ValueError,match='preflight observation'):validate(case)


def test_calibration_cannot_claim_different_direct_values_with_same_quantiles(locked_case):
    lock=locked_case['lock']
    # Changing an interior non-quantile datum would leave these three quantiles unchanged.
    lock['calibration']['development_direct_confidence'][1]['mean_token_logprob']=-2.1
    lock['calibration_canonical_sha256']=runner.canonical_hash(lock['calibration'])
    with pytest.raises(ValueError):check_lock(locked_case)
