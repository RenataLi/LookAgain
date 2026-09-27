"""CPU guard tests with synthetic requests and an in-memory result filesystem."""
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'experiments'),str(PROJECT/'src')]
import refinement_execution as ex
import refinement_ranker as ranker
from refinement_core import PREFLIGHT_ACTIONS,action_order,make_request,normalized_boxes
from refinement_features import raw_feature_digest
from region_selection_core import Observation,canonical_hash

CFG={'base_visual_tokens':256,'crop_visual_tokens':1024,'highres_visual_tokens':4096,
     'seed':20260927,'selector_max_new_tokens':16,'answer_max_new_tokens':64}


def hashed(obj):return hashlib.sha256(json.dumps(obj,sort_keys=True).encode()).hexdigest()


@pytest.fixture
def panel(monkeypatch):
    """Build both complete roles, without ever opening annotations or a real file."""
    def build(kind='generation'):
        source=Image.new('RGB',(1024,1024),(17,31,57))
        rows=[Observation(f'source:{i}',f'Which symbol {i}?',f'images/{i}.png','a'*64,f'cluster:{i}') for i in range(2)]
        pf={}; raw={}; records=[]
        for row in rows:
            actions={}
            for action in PREFLIGHT_ACTIONS:
                _,images,g=make_request(source,row.question,action,CFG)
                grids=[[1,im.height//16,im.width//16] for im in images]
                nv=sum(t*h*w//4 for t,h,w in grids)
                capture={'input_tokens':nv+10,'visual_tokens':nv,'image_grid_thw':grids,
                         'formatted_prompt_sha256':hashed([row.example_id,action]),
                         'tensor_sha256':{'input_ids':hashed([row.example_id,action,'ids'])}}
                actions[action]={'geometry_sha256':canonical_hash(g),'processor_capture':capture}
                if action=='legacy_selector':
                    ids=[42,151652]+[151655]*nv+[151653]+[44]*7
                    tensors={'input_ids':torch.tensor([ids]),'attention_mask':torch.ones((1,len(ids)),dtype=torch.int64),
                             'image_grid_thw':torch.tensor(grids)}
                    actions[action]['feature_layout']=ex.feature_layout(tensors,normalized_boxes(source.size))
            pf[row.example_id]={'observation_sha256':canonical_hash(asdict(row)),'actions':actions}
            for action in (['features'] if kind=='features' else action_order(row.example_id)):
                request='legacy_selector' if action=='features' else action
                _,_,g=make_request(source,row.question,request,CFG,5)
                key='legacy_native_5' if action=='legacy_selected' else request
                capture=actions[key]['processor_capture']
                r={'example_id':row.example_id,'source_cluster_id':row.source_cluster_id,'action':action,'status':'ok',
                   'observation_sha256':canonical_hash(asdict(row)),'geometry':g,'processor_capture':copy.deepcopy(capture),
                   **{k:copy.deepcopy(capture[k]) for k in ('input_tokens','visual_tokens','image_grid_thw')},
                   'elapsed_s':.1,'processor_observer_elapsed_s':.01,'peak_memory_gib':8.,'peak_reserved_gib':9.}
                if action=='features':
                    rng=np.random.default_rng(100+int(row.example_id[-1]))
                    regions=rng.normal(size=(9,2560)).astype(np.float32);query=rng.normal(size=2560).astype(np.float32)
                    projected=ranker.project_vectors(regions,query)
                    relative='raw_features/'+hashlib.sha256(row.example_id.encode()).hexdigest()+'.npz'
                    raw[relative]={'region_vectors':regions,'query_vector':query}
                    r.update(**copy.deepcopy(actions['legacy_selector']['feature_layout']),generated_tokens=0,
                             projected_features=projected.tolist(),projected_sha256=canonical_hash(projected.tolist()),
                             raw_features_path=relative,raw_features_sha256='f'*64,raw_feature_sha256=raw_feature_digest(regions,query))
                else:
                    selector=action=='legacy_selector';text='REGION: 5' if selector else 'synthetic'
                    r.update(response=text if selector else 'ANSWER:'+text,raw_continuation=text,
                             answer_prefix_prefilled=not selector,generated_tokens=2,generation_truncated=False,mean_token_logprob=-.5)
                    if selector:r.update(selector_region=5,selector_valid=True)
                records.append(r)
        lock={'role':'engineering','kind':kind,'run_id':'synthetic','config':copy.deepcopy(CFG),'runtime':{},
              'code_bindings':{'x':'b'*64},'model':{'x':1},'model_file_sha256':{'x':'c'*64},
              'manifest_sha256':'d'*64,'expected_records':len(records),'locked_at_utc':'2026-09-27T00:00:00+00:00'}
        run={k:copy.deepcopy(v) for k,v in lock.items() if k not in ('expected_records','locked_at_utc')}
        n_warm=6 if kind=='generation' else 3
        run.update(execution_lock_sha256='e'*64,created_utc='2026-09-27T00:01:00+00:00',warmup_calls=[{'synthetic':True}]*n_warm,
                   inference_labels_loaded=False,roi_annotation_used=False,answer_page_privileged=True)
        startup={'role':'engineering','kind':kind,'run_id':'synthetic','status':'complete','model_load_completed':True,
                 'created_utc':'2026-09-27T00:00:30+00:00','finished_utc':'2026-09-27T00:01:59+00:00',
                 'planned_warmup_invocations':n_warm,'warmup_invocations_started':n_warm,'warmup_invocations_completed':n_warm,
                 'warmup_calls':copy.deepcopy(run['warmup_calls']),'current_warmup':None,
                 'planned_main_invocations':len(records),'main_invocations_started':len(records),
                 'main_invocations_completed':len(records),'records_completed':len(records)}
        done={'records':len(records),'finished_utc':'2026-09-27T00:02:00+00:00','elapsed_s':4.,
              'sum_measured_action_elapsed_s':sum(r['elapsed_s'] for r in records),'startup_sha256':'s'*64}
        store={'records':records,'run':run,'startup':startup,'done':done,'lock':lock,'pf':pf,'raw':raw,'rows':rows}
        def sync():done['records_sha256']=hashed(records)
        sync();store['sync']=sync
        def read_json(p):
            key={'run.json':'run','completed.json':'done','startup.json':'startup'}[Path(p).name]
            return copy.deepcopy(store[key])
        def read_jsonl(p):
            assert Path(p).name=='records.jsonl','Annotation reads are forbidden in completion validation'
            return copy.deepcopy(records)
        def sha(p):
            name=Path(p).name
            return hashed(records) if name=='records.jsonl' else ('s'*64 if name=='startup.json' else ('f'*64 if name.endswith('.npz') else 'e'*64))
        class Archive:
            def __init__(self,data):self.data=data;self.files=list(data)
            def __enter__(self):return self
            def __exit__(self,*a):pass
            def __getitem__(self,k):return self.data[k]
        def load(p,allow_pickle):
            assert allow_pickle is False
            return Archive(raw['raw_features/'+Path(p).name])
        monkeypatch.setattr(ex,'read_json',read_json);monkeypatch.setattr(ex,'read_jsonl',read_jsonl)
        monkeypatch.setattr(ex,'sha_file',sha)
        monkeypatch.setattr(ex,'validate_lock',lambda *a,**k:(copy.deepcopy(lock),copy.deepcopy(CFG)))
        monkeypatch.setattr(ex,'observations',lambda *a,**k:rows);monkeypatch.setattr(ex,'preflight_rows',lambda role:pf)
        monkeypatch.setattr(Image,'open',lambda p:source.copy());monkeypatch.setattr(np,'load',load)
        store['validate']=lambda:ex.validate_completed(Path('synthetic/manifest.jsonl'),Path('synthetic/lock.json'),
                                                        Path('__synthetic_refinement_no_real_files__'),kind)
        return store
    return build


@pytest.mark.parametrize('kind',['generation','features'])
def test_complete_run_reconstructs_geometry_and_returns_dict_rows(panel,kind):
    rows,records,run,lock=panel(kind)['validate']()
    assert len(rows)==2 and all(isinstance(r,dict) for r in rows)
    assert len(records)==(26 if kind=='generation' else 2)
    assert run['kind']==lock['kind']==kind


@pytest.mark.parametrize('defect',['missing','duplicate','order','source','observation','pixels','prompt','capture','tokens',
    'region','selector_valid','raw','prefix','zero_generated','over_cap','bool_generated','truncation','nan_cost',
    'zero_time','observer_time','reserved','logprob','sum','status'])
def test_record_tampering_fails_even_with_refreshed_completion_hash(panel,defect):
    s=panel();r=s['records'][0]
    if defect=='missing':s['records'].pop();s['done']['records']-=1
    elif defect=='duplicate':s['records'][-1]=copy.deepcopy(r)
    elif defect=='order':s['records'][0],s['records'][1]=s['records'][1],s['records'][0]
    elif defect=='source':r['source_cluster_id']='other'
    elif defect=='observation':r['observation_sha256']='0'*64
    elif defect=='pixels':r['geometry']['image_rgb_sha256']=['0'*64]
    elif defect=='prompt':r['geometry']['question_prompt']='Changed'
    elif defect=='capture':r['processor_capture']['input_tokens']=1
    elif defect=='tokens':r['visual_tokens']+=1
    elif defect=='region':r['selector_region']=9
    elif defect=='selector_valid':r['selector_valid']=False
    elif defect=='raw':r['raw_continuation']='REGION: 9'
    elif defect=='prefix':s['records'][1]['answer_prefix_prefilled']=False
    elif defect=='zero_generated':r['generated_tokens']=0
    elif defect=='over_cap':r['generated_tokens']=17
    elif defect=='bool_generated':r['generated_tokens']=True
    elif defect=='truncation':r['generation_truncated']=0
    elif defect=='nan_cost':r['peak_memory_gib']=float('nan')
    elif defect=='zero_time':r['elapsed_s']=0
    elif defect=='observer_time':r['processor_observer_elapsed_s']=.2
    elif defect=='reserved':r['peak_reserved_gib']=7
    elif defect=='logprob':r['mean_token_logprob']=.5
    elif defect=='sum':s['done']['sum_measured_action_elapsed_s']+=.5
    elif defect=='status':r['status']='error'
    s['sync']()
    with pytest.raises((ValueError,KeyError)):s['validate']()


@pytest.mark.parametrize('key,value',[('runtime',{'changed':True}),('code_bindings',{}),('model',{}),('model_file_sha256',{}),
    ('manifest_sha256','0'*64),('execution_lock_sha256','0'*64),('inference_labels_loaded',True),('roi_annotation_used',True),
    ('answer_page_privileged',False),('created_utc','2026-09-26T00:00:00+00:00'),('warmup_calls',[])])
def test_run_identity_guards(panel,key,value):
    s=panel();s['run'][key]=value
    with pytest.raises(ValueError):s['validate']()


@pytest.mark.parametrize('defect',['query','positions','counts','raw_digest','raw_path','raw_file_hash','raw_dtype',
                                  'raw_nan','raw_keys','projection','interaction','zero_generated','onehot','startup'])
def test_raw_feature_and_projection_replay_guards(panel,defect):
    s=panel('features');r=s['records'][0];raw=s['raw'][r['raw_features_path']]
    if defect=='query':r['query_token_index']-=1
    elif defect=='positions':r['image_token_positions'][0]+=1
    elif defect=='counts':r['image_region_token_counts'][0]+=1
    elif defect=='raw_digest':r['raw_feature_sha256']='0'*64
    elif defect=='raw_path':r['raw_features_path']='../outside.npz'
    elif defect=='raw_file_hash':r['raw_features_sha256']='0'*64
    elif defect=='raw_dtype':raw['region_vectors']=raw['region_vectors'].astype(np.float64)
    elif defect=='raw_nan':raw['query_vector'][0]=np.nan
    elif defect=='raw_keys':raw['labels']=np.zeros(9)
    elif defect in ('projection','interaction'):
        r['projected_features'][0][32]+=.1
        r['projected_sha256']=canonical_hash(r['projected_features'])
    elif defect=='zero_generated':r['generated_tokens']=1
    elif defect=='onehot':r['projected_features'][0][64]=0;r['projected_sha256']=canonical_hash(r['projected_features'])
    elif defect=='startup':s['startup']['warmup_invocations_started']+=1
    s['sync']()
    with pytest.raises((ValueError,KeyError)):s['validate']()


def test_layout_known_grid_and_last_prompt_position():
    ids=[1,151652]+[151655]*16+[151653,9,10]
    x={'input_ids':torch.tensor([ids]),'attention_mask':torch.ones((1,len(ids)),dtype=torch.int64),
       'image_grid_thw':torch.tensor([[1,8,8]])}
    out=ex.feature_layout(x,normalized_boxes((1024,1024)))
    assert out['image_token_positions']==list(range(2,18))
    assert out['query_token_index']==20 and out['image_region_token_counts']==[4]*9
    x['attention_mask'][0,-1]=0
    with pytest.raises(ValueError,match='unpadded'):ex.feature_layout(x,normalized_boxes((1024,1024)))


@pytest.mark.parametrize('failure',['load','second_warmup'])
def test_failed_loading_or_warmup_is_retained_in_incremental_ledger(monkeypatch,failure):
    import lookagain.backend as backend_module
    records={};history=[]
    class Output:
        def exists(self):return False
        def mkdir(self,**kwargs):pass
        def __truediv__(self,name):return Path('__in_memory_failed_run__')/name
    class Backend:
        def __init__(self,*a):
            if failure=='load':raise RuntimeError('synthetic loading failure')
            self.processor=object();self.calls=0
        def generate(self,*a,**k):
            self.calls+=1
            if self.calls==2:raise RuntimeError('synthetic warmup failure')
            return {'generated_tokens':2}
    def write(path,value):records[Path(path).name]=copy.deepcopy(value);history.append(copy.deepcopy(value))
    lock={'role':'engineering','kind':'generation','run_id':'failed-fixture','runtime':{},'expected_records':26}
    monkeypatch.setattr(ex,'validate_lock',lambda *a:(lock,CFG))
    monkeypatch.setattr(ex,'configure_runtime',lambda seed:{})
    monkeypatch.setattr(ex,'observations',lambda *a:[]);monkeypatch.setattr(ex,'preflight_rows',lambda *a:{})
    monkeypatch.setattr(ex,'write_json',write);monkeypatch.setattr(ex,'sha_file',lambda p:'x'*64)
    monkeypatch.setattr(backend_module,'QwenBackend',Backend)
    args=SimpleNamespace(lock=Path('lock'),manifest=Path('manifest'),model_dir=Path('model'),output=Output())
    with pytest.raises(RuntimeError,match='synthetic'):ex.execute(args)
    assert history[0]['status']=='loading'
    assert records['startup.json']['status']=='failed'
    assert records['error.json']['records_completed']==0
    assert 'completed.json' not in records and 'run.json' not in records
    assert records['error.json']['warmup_invocations_started']==(0 if failure=='load' else 2)
    assert records['error.json']['warmup_invocations_completed']==(0 if failure=='load' else 1)


@pytest.mark.parametrize('defect',[None,'config','code_bindings','runtime','model','model_file_sha256',
                                  'role','kind','completion_time','run_lock_hash','records_hash'])
def test_prerequisite_artifacts_must_share_locked_execution_identity(monkeypatch,defect):
    codes={'synthetic.py':'h'}
    dep={'role':'engineering','kind':'generation','finished_utc':'2026-09-27T00:01:00+00:00'}
    for key in ('lock','run','records','completed'):
        dep[key]={'path':f'reports/synthetic/prerequisite_{key}.json','sha256':'h'}
    lock={'schema_version':1,'execution_locked':True,'role':'engineering','kind':'features','config':CFG,
          'config_sha256':'h','code_bindings':codes,'manifest_sha256':'h','source_bindings':{},'input_bindings':{},
          'verification':{'status':'PASS','code_bindings':codes},'expected_records':2,
          'locked_at_utc':'2026-09-27T00:02:00+00:00','prerequisites':[dep],
          'runtime':{'seed':1},'model':{'id':'fixture'},'model_file_sha256':{'weights':'h'}}
    child={k:copy.deepcopy(lock[k]) for k in ('config','code_bindings','runtime','model','model_file_sha256')}
    child.update(role='engineering',kind='generation')
    child_run=copy.deepcopy(child);child_run['execution_lock_sha256']='h'
    child_done={'finished_utc':dep['finished_utc'],'records_sha256':'h'}
    if defect in ('config','code_bindings','runtime','model','model_file_sha256'):child[defect]={'different':True}
    elif defect=='role':child['role']='main'
    elif defect=='kind':child_run['kind']='features'
    elif defect=='completion_time':child_done['finished_utc']='2026-09-27T00:01:01+00:00'
    elif defect=='run_lock_hash':child_run['execution_lock_sha256']='different'
    elif defect=='records_hash':child_done['records_sha256']='different'
    def read(p):
        if Path(p)==ex.CONFIG:return CFG
        return {'lock.json':lock,'reservation.json':{'reserved_at_utc':'2026-09-27T00:00:00+00:00'},
                'prerequisite_lock.json':child,'prerequisite_run.json':child_run,
                'prerequisite_completed.json':child_done}[Path(p).name]
    monkeypatch.setattr(ex,'validate_reservation',lambda:None);monkeypatch.setattr(ex,'read_json',read)
    monkeypatch.setattr(ex,'sha_file',lambda p:'h');monkeypatch.setattr(ex,'code_bindings',lambda:codes)
    monkeypatch.setattr(ex,'file_bindings',lambda p:{})
    if defect is None:assert ex.validate_lock(Path('lock.json'),Path('manifest.jsonl'))[0] is lock
    else:
        with pytest.raises(ValueError,match='Prerequisite'):ex.validate_lock(Path('lock.json'),Path('manifest.jsonl'))


def test_main_generation_lock_replays_decisions_before_writing(monkeypatch):
    import train_refinement as training
    cfg={**CFG,'model_id':'fixture','model_revision':'revision'};codes={'x':'h'}
    featuredep={'role':'main','kind':'features','lock':{'path':'reports/synthetic/features_lock.json','sha256':'h'},
                'run':{'path':'reports/synthetic/features_run/run.json','sha256':'h'}}
    def read(p):
        if Path(p)==ex.CONFIG:return cfg
        if Path(p).name=='verification.json':return {'status':'PASS','code_bindings':codes}
        if Path(p).name.endswith('.meta.json'):
            role=Path(p).name.split('.')[0];n=2 if role=='engineering' else 124
            return {'code_bindings':codes,'config_sha256':'h','preflight_sha256':'h','sources':n,
                    'requests':n*21,'manifest_sha256':'h'}
        if Path(p).name=='lookagain-model.json':return {'model_id':'fixture','revision':'revision'}
        raise AssertionError('Decision replay should precede this read: '+str(p))
    seen=[]
    def reject(*args):seen.append(args);raise ValueError('Synthetic decision replay rejected')
    monkeypatch.setattr(ex,'validate_reservation',lambda:None);monkeypatch.setattr(ex,'read_json',read)
    monkeypatch.setattr(ex,'observations',lambda *a:[{}]*124);monkeypatch.setattr(ex,'code_bindings',lambda:codes)
    monkeypatch.setattr(ex,'sha_file',lambda p:'h');monkeypatch.setattr(ex,'file_bindings',lambda p:{})
    monkeypatch.setattr(ex,'model_files',lambda p:{});monkeypatch.setattr(ex,'configure_runtime',lambda seed:{})
    monkeypatch.setattr(ex,'evidence',lambda *a:featuredep);monkeypatch.setattr(training,'validate_decisions',reject)
    monkeypatch.setattr(ex,'write_json',lambda *a:pytest.fail('Must not write a lock after failed decision replay'))
    args=SimpleNamespace(manifest=Path('manifest'),role='main',kind='generation',model_dir=Path('model'),
         run_id='main',verification=Path('verification.json'),decisions=PROJECT/'reports/synthetic/decisions.json',
         prerequisite=[('manifest','features_lock','features_run')],output=Path('newlock.json'))
    with pytest.raises(ValueError,match='Synthetic decision'):ex.lock_execution(args)
    assert len(seen)==1 and seen[0][0]==args.decisions and seen[0][3]==args.manifest
