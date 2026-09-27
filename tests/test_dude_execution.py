"""Synthetic adversarial execution/resume tests: no model or corpus loading."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image, ImageDraw
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'experiments'))
import dude_replication as runner


def write_lines(path, rows):
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding='utf-8')


@pytest.fixture
def cohort(tmp_path, monkeypatch):
    """Small synthetic cohort; only external source/lock boundary is stubbed.

    Scores, geometry, fingerprints, raw records, completion and resume validation
    remain real. Token counts/digests are declared fixtures, not model evidence.
    """
    image=Image.new('RGB',(960,1280),'white'); draw=ImageDraw.Draw(image)
    for x in range(17,960,29): draw.line((x,0,x,1280),fill=(x%255,34,91),width=3)
    image.save(tmp_path/'page.png')
    row={'example_id':'synthetic-q','image_id':'synthetic-p','source_id':'synthetic-s',
         'source_cluster_id':'synthetic-s','question':'What year is shown?',
         'answer':'2019','original_answers':['2019'],'original_answer_variants':[' 2019 '],
         'validated_primary_answers':['2019'],'image_path':'page.png',
         'image_sha256':runner.sha_file(tmp_path/'page.png'),'roi_pixels':[31,89,351,409],
         'roi_provenance_sha256':'r'*64,'source_semantic_review_sha256':'v'*64}
    config={'seed':20260927,'crop_visual_tokens':1024,'highres_visual_tokens':4096}
    manifest=tmp_path/'manifest.jsonl';write_lines(manifest,[row])
    records={};preflight={'example_id':row['example_id'],'actions':{}}
    for action in runner.ACTIONS:
        _,images,geometry=runner.build_request(image,row['question'],action,config,row['roi_pixels'])
        grids=[[1,im.height//16,im.width//16] for im in images]
        visual=sum(t*h*w//4 for t,h,w in grids)
        hashes={key:hashlib.sha256((action+key).encode()).hexdigest() for key in ('input_ids','attention_mask','pixel_values','image_grid_thw')}
        p={'image_sizes':[list(im.size) for im in images],
           'image_rgb_sha256':geometry['image_rgb_sha256'],'messages_sha256':geometry['messages_sha256'],
           'input_tokens':visual+70,'visual_tokens':visual,'image_grid_thw':grids,
           'tensor_sha256':hashes,'formatted_prompt_sha256':'p'*64}
        preflight['actions'][action]=p
        response='ANSWER: 2020' if action=='degraded_256' else 'ANSWER: 2019'
        record={key:deepcopy(row[key]) for key in ('example_id','image_id','source_id','source_cluster_id','question','original_answers','original_answer_variants','validated_primary_answers')}
        record.update(action=action,status='ok',target_answer=row['answer'],source_image_sha256=row['image_sha256'],
                      roi_provenance_sha256=row['roi_provenance_sha256'],source_semantic_review_sha256=row['source_semantic_review_sha256'],
                      **geometry,**runner.score_response(response,row['original_answers'],row['validated_primary_answers'],row['original_answer_variants']),
                      response=response,raw_continuation=response[len('ANSWER:'):],answer_prefix_prefilled=True,
                      generated_tokens=3,generation_truncated=False,input_tokens=p['input_tokens'],visual_tokens=visual,image_grid_thw=grids,
                      processor_input_tensor_sha256=hashes,formatted_prompt_sha256=p['formatted_prompt_sha256'],
                      elapsed_s=1.5,peak_memory_gib=8.,peak_reserved_gib=9.,processor_observer_elapsed_s=.01,mean_token_logprob=-.5)
        records[(row['example_id'],action)]=record
    lock={'locked_at_utc':'2026-09-27T00:00:00Z','run_id':'synthetic-main','config':config,
          'manifest_sha256':runner.sha_file(manifest),'code_bindings':{'synthetic':'c'*64},
          'model':{'model_id':'fixture','revision':'fixture'},'model_file_sha256':{'weights':'w'*64},
          'runtime':{'fixture':True},'role':'main'}
    lock_path=tmp_path/'lock.json';runner.write_json(lock_path,lock)
    monkeypatch.setattr(runner,'validate_lock',lambda *a,**k:(lock,config))
    monkeypatch.setattr(runner,'validate_manifest',lambda *a,**k:[row])
    monkeypatch.setattr(runner,'preflight_rows',lambda:{row['example_id']:preflight})
    run_dir=tmp_path/'run';run_dir.mkdir()
    identity=runner.make_identity(lock_path,lock,[row])
    run={**identity,'fingerprint':runner.canonical_hash(identity),'created_utc':'2026-09-27T00:01:00Z'}
    runner.write_json(run_dir/'run.json',run)
    result={'row':row,'rows':[row],'config':config,'manifest':manifest,'records':records,'preflight':preflight,
            'lock':lock,'lock_path':lock_path,'run':run,'run_dir':run_dir}
    save_records(result)
    return result


def save_records(case, completed=True):
    path=case['run_dir']/'records.jsonl';write_lines(path,list(case['records'].values()))
    completion={'records':len(case['records']),'records_sha256':runner.sha_file(path),
                'fingerprint':case['run']['fingerprint'],'finished_utc':'2026-09-27T00:02:00Z',
                'elapsed_s':10.,'sum_measured_action_elapsed_s':sum(r['elapsed_s'] for r in case['records'].values())}
    marker=case['run_dir']/'completed.json'
    if completed: runner.write_json(marker,completion)
    elif marker.exists(): marker.unlink()


def validate(case, complete=True):
    return runner.validate_run_files(case['manifest'],case['lock_path'],case['run_dir'],'main',complete=complete)


def test_accepts_complete_and_strictly_separates_partial_resume(cohort):
    rows,records,run,completion=runner.validate_completed_run(cohort['manifest'],cohort['lock_path'],cohort['run_dir'])
    assert len(rows)==1 and len(records)==4 and completion['records']==4
    cohort['records'].pop(('synthetic-q','highres'));save_records(cohort,completed=False)
    assert len(validate(cohort,complete=False)[1])==3
    with pytest.raises(ValueError,match='Completed run required'): validate(cohort)


def test_coherently_rehashed_missing_record_still_rejected(cohort):
    cohort['records'].pop(('synthetic-q','highres'));save_records(cohort)
    with pytest.raises(ValueError,match='Incomplete paired run'): validate(cohort)


def test_recover_completion_and_reuse_completed_run_without_loading_model(cohort,monkeypatch):
    from types import SimpleNamespace
    def forbidden(*a,**k):raise AssertionError('Completed run must not construct a model')
    monkeypatch.setitem(sys.modules,'lookagain.backend',SimpleNamespace(QwenBackend=forbidden))
    monkeypatch.setattr(runner,'configure_runtime',lambda *a:cohort['lock']['runtime'])
    monkeypatch.setattr(runner,'now',lambda:'2026-09-27T00:03:00Z')
    (cohort['run_dir']/'completed.json').unlink()
    runner.run(cohort['manifest'],cohort['run_dir'],cohort['lock_path'],cohort['run_dir'],'main')
    marker=runner.read_json(cohort['run_dir']/'completed.json')
    assert marker['elapsed_s'] is None and 'recovered' in marker['elapsed_s_scope']
    validate(cohort)
    digest=runner.sha_file(cohort['run_dir']/'records.jsonl')
    runner.run(cohort['manifest'],cohort['run_dir'],cohort['lock_path'],cohort['run_dir'],'main')
    assert runner.sha_file(cohort['run_dir']/'records.jsonl')==digest


def test_orphan_records_are_never_adopted(cohort,monkeypatch):
    monkeypatch.setattr(runner,'configure_runtime',lambda *a:cohort['lock']['runtime'])
    (cohort['run_dir']/'run.json').unlink()
    with pytest.raises(ValueError,match='Nonempty unbound'):
        runner.run(cohort['manifest'],cohort['run_dir'],cohort['lock_path'],cohort['run_dir'],'main')


@pytest.mark.parametrize('field,value',[
    ('primary_em',0.),('official_anls',0.),('target_answer','2020'),
    ('original_answers',['2020']),('validated_primary_answers',['2020']),
    ('source_semantic_review_sha256','x'*64),('image_id','other-page'),
    ('formatted_prompt_sha256','x'*64),('processor_input_tensor_sha256',{}),
    ('raw_continuation',' 2020'),('answer_prefix_prefilled',False),
    ('generated_tokens',True),('generation_truncated',True),
    ('elapsed_s',False),('processor_observer_elapsed_s',2.),('peak_reserved_gib',7.),
])
def test_tampered_record_rejected_even_with_new_completion_hash(cohort,field,value):
    cohort['records']['synthetic-q','native_256'][field]=value;save_records(cohort)
    with pytest.raises(ValueError): validate(cohort)


def test_pair_equal_wrong_input_tokens_do_not_pass_preflight(cohort):
    for action in ('native_256','degraded_256'):
        cohort['records']['synthetic-q',action]['input_tokens']+=1
    save_records(cohort)
    with pytest.raises(ValueError,match='input_tokens'):validate(cohort)


def test_invalid_and_truncated_answers_remain_in_complete_denominator(cohort):
    r=cohort['records']['synthetic-q','native_256'];r['response']='ANSWER:';r['raw_continuation']=''
    r.update(runner.score_response(r['response'],['2019'],['2019'],[' 2019 ']))
    r['generated_tokens']=64;r['generation_truncated']=True;save_records(cohort)
    records=validate(cohort)[1]
    assert len(records)==4 and records['synthetic-q','native_256']['parse_valid'] is False
    assert records['synthetic-q','native_256']['primary_em']==0


@pytest.mark.parametrize('field,value',[
    ('records_sha256','x'*64),('fingerprint','x'*64),('records',3),
    ('finished_utc','2026-09-26T23:59:00Z'),('finished_utc','2026-09-27T00:02:00'),
    ('sum_measured_action_elapsed_s',7.),('elapsed_s',-1.),
])
def test_completion_binding_and_chronology(cohort,field,value):
    path=cohort['run_dir']/'completed.json';r=runner.read_json(path);r[field]=value;runner.write_json(path,r)
    with pytest.raises(ValueError):validate(cohort)


@pytest.mark.parametrize('field,value',[
    ('role','engineering'),('runtime',{'changed':True}),('model_file_sha256',{'weights':'x'*64}),
    ('example_ids',['another-source']),('execution_lock_sha256','x'*64),
])
def test_rehashed_run_identity_still_bound_to_original_lock(cohort,field,value):
    path=cohort['run_dir']/'run.json';r=runner.read_json(path);r[field]=value
    original=runner.make_identity(cohort['lock_path'],cohort['lock'],cohort['rows'])
    r['fingerprint']=runner.canonical_hash({key:r[key] for key in original});runner.write_json(path,r)
    with pytest.raises(ValueError,match='identity'):validate(cohort)


def test_error_log_blocks_retry_and_unexpected_action_cannot_enter(cohort):
    path=cohort['run_dir']/'errors.jsonl';path.write_text('{"error":"synthetic"}\n',encoding='utf-8')
    with pytest.raises(ValueError,match='Errored run'):validate(cohort)
    path.unlink();r=deepcopy(cohort['records']['synthetic-q','direct_256']);r['action']='extra'
    cohort['records']['synthetic-q','extra']=r;save_records(cohort)
    with pytest.raises(ValueError,match='Unexpected source/action'):validate(cohort)


@pytest.mark.parametrize('text',[
    '{"example_id":"x","action":"direct_256","status":"ok","elapsed_s":NaN}\n',
    '{"example_id":"x","action":"direct_256","status":"ok"}\n{"unfinished":',
    '{"example_id":"x","action":"direct_256","status":"error"}\n',
    '{"example_id":"x","action":"direct_256","status":"ok"}\n'*2,
])
def test_malformed_failed_duplicate_or_nonfinite_raw_records_fail_closed(tmp_path,text):
    path=tmp_path/'records.jsonl';path.write_text(text,encoding='utf-8')
    with pytest.raises((ValueError,json.JSONDecodeError)):runner.read_records(path)


def test_processor_observer_returns_exact_input_objects_and_rejects_wrong_evidence():
    torch=pytest.importorskip('torch')
    batch={'input_ids':torch.tensor([[2,3,4]]),'attention_mask':torch.tensor([[1,1,1]]),
           'image_grid_thw':torch.tensor([[1,2,2]]),'pixel_values':torch.arange(12).reshape(4,3).float(),
           'token_type_ids':torch.tensor([[0,0,0]])}
    snapshots={key:t.clone() for key,t in batch.items()}
    class Processor:
        def __call__(self,*a,**k):return batch
    observer=runner.ProcessorObserver(Processor());actual=observer(text=['synthetic prompt'])
    assert actual is batch and all(torch.equal(batch[k],v) for k,v in snapshots.items())
    assert 'token_type_ids' not in observer.last['tensor_sha256']
    observer.expected={k:v for k,v in observer.last.items() if k!='observer_elapsed_s'}
    assert observer(text=['synthetic prompt']) is batch
    observer.expected=deepcopy(observer.expected);observer.expected['tensor_sha256']['pixel_values']='f'*64
    with pytest.raises(ValueError,match='tensor_sha256'):observer(text=['synthetic prompt'])


@pytest.mark.parametrize('changed_field',[None,'model','model_file_sha256','runtime','run_id'])
def test_main_lock_requires_same_engineering_environment_and_distinct_identity(tmp_path,monkeypatch,changed_field):
    config={'model_id':'fixture','model_revision':'fixed','seed':20260927}
    codes={'synthetic':'c'*64};model={'model_id':'fixture','revision':'fixed'}
    hashes={'weights':'w'*64};runtime={'gpu':'fixture','packages':{'torch':'fixed'}}
    model_dir=tmp_path/'model';model_dir.mkdir();runner.write_json(model_dir/'lookagain-model.json',model)
    manifest=tmp_path/'main_manifest.jsonl';manifest.write_text('{}\n',encoding='utf-8')
    proof=tmp_path/'proof.json';runner.write_json(proof,{'status':'PASS','code_bindings':codes})
    eng_dir=tmp_path/'eng';eng_dir.mkdir();eng_lock=tmp_path/'engineering-lock.json'
    for path in [eng_lock,*(eng_dir/name for name in ('run.json','records.jsonl','completed.json'))]:path.write_text('{}\n',encoding='utf-8')
    eng_run={'code_bindings':codes,'model':deepcopy(model),'model_file_sha256':deepcopy(hashes),'runtime':deepcopy(runtime),'run_id':'fixture-engineering'}
    if changed_field=='run_id':eng_run['run_id']='fixture-main'
    elif changed_field:eng_run[changed_field]={'changed':'engineering-used-different-value'}
    monkeypatch.setattr(runner,'validate_source_design',lambda:({},config))
    monkeypatch.setattr(runner,'validate_manifest',lambda *a:[{'example_id':'fixture'}]*660)
    monkeypatch.setattr(runner,'code_bindings',lambda:codes)
    monkeypatch.setattr(runner,'model_files',lambda *a:hashes)
    monkeypatch.setattr(runner,'configure_runtime',lambda *a:runtime)
    monkeypatch.setattr(runner,'validate_completed_run',lambda *a,**k:([],{i:{} for i in range(40)},eng_run,{}))
    if changed_field is None:
        runner.create_lock(manifest,model_dir,tmp_path/'main-lock.json','main','fixture-main',proof,eng_dir,eng_lock)
        lock=runner.read_json(tmp_path/'main-lock.json')
        assert lock['engineering_evidence']['records']==40 and lock['role']=='main'
    else:
        with pytest.raises(ValueError,match='[Mm]odel|[Rr]untime|[Ee]ngineering'):
            runner.create_lock(manifest,model_dir,tmp_path/'main-lock.json','main','fixture-main',proof,eng_dir,eng_lock)
