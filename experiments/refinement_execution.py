"""Prospectively locked extraction/ranker experiment on reused development data."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'experiments')]
from dude_replication import (require,read_json,read_jsonl,write_json,now,utc,safe_path,
                              model_files,configure_runtime)
from check_dude_requests import sha_file
from region_selection_core import Observation,canonical_hash,parse_region
from run_region_selection import ProcessorCapture,code_bindings as old_code_bindings
from refinement_core import make_request,action_order,preflight_key,normalized_boxes,PREFLIGHT_ACTIONS

REPORT=PROJECT/'reports/refinement124'
PREPARATION=REPORT/'preparation'
INPUTS=REPORT/'inputs'
CONFIG=PROJECT/'configs/refinement.json'
NEW_CODE=('experiments/prepare_refinement.py','experiments/refinement_core.py',
          'experiments/refinement_execution.py','experiments/refinement_features.py',
          'experiments/refinement_ranker.py','experiments/train_refinement.py',
          'experiments/analyze_refinement.py','docs/refinement_protocol.md','configs/refinement.json')


def code_bindings():
    return {**old_code_bindings(),**{name:sha_file(PROJECT/name) for name in NEW_CODE}}


def public_manifest(role):return PREPARATION/(role+'_observation_manifest.jsonl')
def public_labels(role):return PREPARATION/(role+'_labels.jsonl')


def count_sources(role):
    require(role in ('engineering','main'),'Unknown role')
    return 2 if role=='engineering' else 124


def validate_reservation():
    r=read_json(PREPARATION/'reservation.json')
    require(r['main_is_reused_development_data'] is True and r['new_holdout_sources']==0,'Invalid scope')
    sources=[]
    for role in ('engineering','main'):
        c=r['cohorts'][role]
        require(c['observation_manifest_sha256']==sha_file(public_manifest(role)),'Reserved manifest changed')
        require(c['labels_sha256']==sha_file(public_labels(role)),'Reserved label bytes changed')
        obs=[Observation.from_manifest(x) for x in read_jsonl(public_manifest(role))]
        require(len(obs)==count_sources(role),'Wrong reservation size')
        require([x.example_id for x in obs]==c['example_ids'],'Example order differs')
        require([x.source_cluster_id for x in obs]==c['source_cluster_ids'],'Source order differs')
        require(len(set(c['source_cluster_ids']))==len(obs) and len(set(c['example_ids']))==len(obs),'Duplicate source')
        sources.append(set(c['source_cluster_ids']))
    require(sources[0].isdisjoint(sources[1]),'Engineering overlap')
    for name,digest in r['upstream_bindings'].items():
        require(sha_file(safe_path(PROJECT,name))==digest,'Old source/results changed: '+name)
    return r


def observations(manifest,role):
    manifest=Path(manifest)
    require(sha_file(manifest)==sha_file(public_manifest(role)),'Observation bytes not reserved')
    rows=[Observation.from_manifest(x) for x in read_jsonl(manifest)]
    require(len(rows)==count_sources(role),'Wrong observation count')
    for row in rows:
        require(sha_file(safe_path(manifest.parent,row.image_path))==row.image_sha256,'Raster changed')
    return rows


def load_prepared(role='main'):
    """Analysis/training only: never called by inference."""
    reservation=validate_reservation()
    rows=read_jsonl(public_manifest(role));labels=read_jsonl(public_labels(role))
    oldroles=['engineering'] if role=='engineering' else ['development','evaluation']
    records=[]
    for oldrole in oldroles:
        records.extend(read_jsonl(PROJECT/'reports/region_selection60/raw_runs'/oldrole/'records.jsonl'))
    require([r['example_id'] for r in rows]==[r['example_id'] for r in labels],'Label order differs')
    return rows,labels,records


def file_bindings(folder):
    return {p.relative_to(PROJECT).as_posix():sha_file(p) for p in sorted(folder.rglob('*')) if p.is_file()}


def preflight_rows(role):
    rows=read_jsonl(INPUTS/(role+'.jsonl'))
    table={r['example_id']:r for r in rows}
    require(len(table)==len(rows)==count_sources(role),'Preflight coverage mismatch')
    return table


def format_prompt(processor,messages,selector):
    if selector:return processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
    return processor.apply_chat_template([*messages,{'role':'assistant','content':'ANSWER:'}],
        tokenize=False,continue_final_message=True)


def feature_layout(inputs, boxes):
    """CPU preflight layout, bound alongside the exact input tensor hashes."""
    import numpy as np
    from refinement_features import (IMAGE_TOKEN_ID, VISION_START_TOKEN_ID, VISION_END_TOKEN_ID,
                                     QUERY_TOKEN_SEMANTICS, pool_image_tokens)
    ids=inputs['input_ids'].tolist();mask=inputs['attention_mask'].tolist()
    grids=inputs['image_grid_thw'].tolist()
    require(len(ids)==len(mask)==len(grids)==1 and len(ids[0])==len(mask[0]),'Feature batch must be one')
    require(all(v==1 for v in mask[0]),'Feature preflight must be unpadded')
    positions=[i for i,value in enumerate(ids[0]) if value==IMAGE_TOKEN_ID]
    t,h,w=grids[0]
    require(t==1 and h%2==w%2==0 and len(positions)==h*w//4,'Feature grid differs')
    require(positions and positions==list(range(positions[0],positions[-1]+1)),'Image sequence is not contiguous')
    require(positions[0]>0 and positions[-1]+1<len(ids[0])-1
            and ids[0][positions[0]-1]==VISION_START_TOKEN_ID
            and ids[0][positions[-1]+1]==VISION_END_TOKEN_ID,'Image frame/query order differs')
    _,counts=pool_image_tokens(np.zeros((len(positions),1),dtype=np.float32),grids[0],boxes)
    return {'query_token_index':len(ids[0])-1,'query_token_semantics':QUERY_TOKEN_SEMANTICS,
            'image_token_positions':positions,'image_region_token_counts':counts}


def preflight(args):
    from PIL import Image
    import torch
    from transformers import AutoProcessor
    validate_reservation();rows=observations(args.manifest,args.role);config=read_json(CONFIG)
    require(not args.output.exists(),'New preflight path required')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,trust_remote_code=False)
    capture=ProcessorCapture(processor);torch.set_num_threads(4)
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        for index,row in enumerate(rows,1):
            with Image.open(safe_path(args.manifest.parent,row.image_path)) as im:source=im.convert('RGB')
            actions={}
            for action in PREFLIGHT_ACTIONS:
                messages,images,geometry=make_request(source,row.question,action,config)
                prompt=format_prompt(processor,messages,action=='legacy_selector')
                tensors=capture(text=[prompt],images=images,return_tensors='pt')
                actions[action]={'geometry_sha256':canonical_hash(geometry),'processor_capture':capture.last}
                if action=='legacy_selector':
                    actions[action]['feature_layout']=feature_layout(tensors,normalized_boxes(source.size))
            stream.write(json.dumps({'example_id':row.example_id,'observation_sha256':canonical_hash(asdict(row)),
                                    'actions':actions},ensure_ascii=False)+'\n');stream.flush()
            if index%8==0 or index==len(rows):print(json.dumps({'preflight_sources':index,'requests':index*21}),flush=True)
    write_json(args.output.with_suffix('.meta.json'),{'created_utc':now(),'role':args.role,
        'sources':len(rows),'requests':len(rows)*21,'preflight_sha256':sha_file(args.output),
        'manifest_sha256':sha_file(args.manifest),'code_bindings':code_bindings(),'config_sha256':sha_file(CONFIG),
        'vlm_calls':0,'processor_files':{p.name:sha_file(p) for p in args.model_dir.iterdir()
                                             if p.is_file() and p.suffix in ('.json','.txt')}})


def validate_lock(lock_path,manifest,model_dir=None):
    validate_reservation();lock=read_json(lock_path);config=read_json(CONFIG)
    require(lock['schema_version']==1 and lock['execution_locked'] is True,'Execution lock missing')
    role=lock['role'];count_sources(role)
    require(lock['kind'] in ('features','generation'),'Unknown execution kind')
    require(lock['config']==config and lock['config_sha256']==sha_file(CONFIG),'Config changed')
    require(lock['code_bindings']==code_bindings(),'Locked code changed')
    require(lock['manifest_sha256']==sha_file(manifest)==sha_file(public_manifest(role)),'Manifest changed')
    require(lock['source_bindings']==file_bindings(PREPARATION),'Preparation changed')
    require(lock['input_bindings']==file_bindings(INPUTS),'CPU preflight changed')
    require(lock['verification']['status']=='PASS' and lock['verification']['code_bindings']==code_bindings(),'Unverified code')
    require(lock['expected_records']==count_sources(role)*(13 if lock['kind']=='generation' else 1),'Record count differs')
    require(utc(lock['locked_at_utc'])>=utc(read_json(PREPARATION/'reservation.json')['reserved_at_utc']),'Lock predates reservation')
    for dep in lock['prerequisites']:
        for key in ('lock','run','records','completed'):
            require(sha_file(safe_path(PROJECT,dep[key]['path']))==dep[key]['sha256'],'Prerequisite artifact changed')
        require(utc(dep['finished_utc'])<=utc(lock['locked_at_utc']),'Prerequisite postdates lock')
        prerequisite_lock=read_json(safe_path(PROJECT,dep['lock']['path']))
        prerequisite_run=read_json(safe_path(PROJECT,dep['run']['path']))
        prerequisite_done=read_json(safe_path(PROJECT,dep['completed']['path']))
        require(prerequisite_lock['role']==prerequisite_run['role']==dep['role']
                and prerequisite_lock['kind']==prerequisite_run['kind']==dep['kind'],'Prerequisite identity differs')
        require(prerequisite_done['finished_utc']==dep['finished_utc'],'Prerequisite completion time differs')
        require(prerequisite_run['execution_lock_sha256']==dep['lock']['sha256'],'Prerequisite run/lock binding differs')
        require(prerequisite_done['records_sha256']==dep['records']['sha256'],'Prerequisite completion/records binding differs')
        for key in ('config','code_bindings','runtime','model','model_file_sha256'):
            require(prerequisite_lock[key]==prerequisite_run[key]==lock[key],'Prerequisite differs in '+key)
    if role=='main':
        require({(d['role'],d['kind']) for d in lock['prerequisites']} >= {('engineering','features'),('engineering','generation')},'Engineering prerequisites missing')
    if role=='main' and lock['kind']=='generation':
        dec=lock['decisions'];require(sha_file(safe_path(PROJECT,dec['path']))==dec['sha256'],'Committed OOF decisions changed')
        require(utc(dec['created_utc'])<=utc(lock['locked_at_utc']),'Decisions postdate generation lock')
        require(('main','features') in {(d['role'],d['kind']) for d in lock['prerequisites']},'Main features prerequisite missing')
    if model_dir is not None:
        require(model_files(model_dir)==lock['model_file_sha256'],'Model/processor bytes changed')
        require(read_json(model_dir/'lookagain-model.json')==lock['model'],'Model identity changed')
    return lock,config


def evidence(manifest,lock_path,run_dir):
    lock=read_json(lock_path)
    fn=validate_generation if lock['kind']=='generation' else validate_features
    _,records,run,_=fn(manifest,lock_path,run_dir)
    result={'role':lock['role'],'kind':lock['kind'],'records_count':len(records),
            'finished_utc':read_json(run_dir/'completed.json')['finished_utc']}
    for key,path in [('lock',lock_path),('run',run_dir/'run.json'),('records',run_dir/'records.jsonl'),('completed',run_dir/'completed.json')]:
        result[key]={'path':path.resolve().relative_to(PROJECT).as_posix(),'sha256':sha_file(path)}
    return result


def lock_execution(args):
    validate_reservation();config=read_json(CONFIG);rows=observations(args.manifest,args.role)
    verification=read_json(args.verification)
    require(verification['status']=='PASS' and verification['code_bindings']==code_bindings(),'Verification differs')
    for role in ('engineering','main'):
        meta=read_json(INPUTS/(role+'.meta.json'))
        require(meta['code_bindings']==code_bindings() and meta['config_sha256']==sha_file(CONFIG),'Preflight code differs')
        require(meta['preflight_sha256']==sha_file(INPUTS/(role+'.jsonl')),'Preflight hash differs')
        require(meta['sources']==count_sources(role) and meta['requests']==count_sources(role)*21,'Preflight incomplete')
        require(meta['manifest_sha256']==sha_file(public_manifest(role)),'Preflight source differs')
    model=read_json(args.model_dir/'lookagain-model.json')
    require(model['model_id']==config['model_id'] and model['revision']==config['model_revision'],'Wrong model')
    lock={'schema_version':1,'execution_locked':True,'role':args.role,'kind':args.kind,'run_id':args.run_id,
          'locked_at_utc':now(),'config':config,'config_sha256':sha_file(CONFIG),'code_bindings':code_bindings(),
          'manifest_sha256':sha_file(args.manifest),'source_bindings':file_bindings(PREPARATION),
          'input_bindings':file_bindings(INPUTS),'verification':verification,'model':model,
          'model_file_sha256':model_files(args.model_dir),'runtime':configure_runtime(config['seed']),
          'expected_records':len(rows)*(13 if args.kind=='generation' else 1),'prerequisites':[]}
    for item in args.prerequisite or []:
        manifest,lp,rp=item;lock['prerequisites'].append(evidence(Path(manifest),Path(lp),Path(rp)))
    if args.role=='main' and args.kind=='generation':
        require(args.decisions is not None,'Committed decisions required')
        from train_refinement import validate_decisions
        main_features=[dep for dep in lock['prerequisites'] if dep['role']=='main' and dep['kind']=='features']
        require(len(main_features)==1,'Exactly one completed main-feature prerequisite required')
        dep=main_features[0]
        validate_decisions(args.decisions,safe_path(PROJECT,dep['lock']['path']),
                           safe_path(PROJECT,dep['run']['path']).parent,args.manifest)
        decisions=read_json(args.decisions)
        require(decisions['code_bindings']==code_bindings(),'Decision code differs')
        lock['decisions']={'path':args.decisions.resolve().relative_to(PROJECT).as_posix(),
                           'sha256':sha_file(args.decisions),'created_utc':decisions['created_utc']}
    require(not args.output.exists(),'New lock path required');args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,lock);validate_lock(args.output,args.manifest)
    print(json.dumps({'locked':args.kind,'role':args.role,'records':lock['expected_records']}))


def _expected(pf,row,action,geometry,region=None):
    require(pf[row.example_id]['observation_sha256']==canonical_hash(asdict(row)),'Preflight observation differs')
    expected=pf[row.example_id]['actions'][preflight_key(action,region)]
    require(expected['geometry_sha256']==canonical_hash(geometry),'Preflight geometry differs')
    return expected['processor_capture']


def execute(args):
    from PIL import Image
    from lookagain.backend import QwenBackend
    from refinement_features import extract_vectors
    from refinement_ranker import project_vectors
    lock,config=validate_lock(args.lock,args.manifest,args.model_dir)
    require(configure_runtime(config['seed'])==lock['runtime'],'Runtime changed')
    rows=observations(args.manifest,lock['role']);pf=preflight_rows(lock['role'])
    require(not args.output.exists(),'Use a new run; interrupted runs must not be silently resumed')
    args.output.mkdir(parents=True)
    startup={'schema_version':1,'role':lock['role'],'kind':lock['kind'],'run_id':lock['run_id'],
             'created_utc':now(),'status':'loading','model_load_completed':False,
             'planned_warmup_invocations':3 if lock['kind']=='features' else 6,
             'warmup_invocations_started':0,'warmup_invocations_completed':0,'warmup_calls':[],
             'current_warmup':None,'planned_main_invocations':lock['expected_records']}
    write_json(args.output/'startup.json',startup)
    count=0;measured=0;main_attempts=0;main_model_completed=0;current_action=None
    try:
        load_start=time.perf_counter()
        backend=QwenBackend(args.model_dir,config)
        capture=ProcessorCapture(backend.processor);backend.processor=capture
        load_elapsed=time.perf_counter()-load_start;warm_start=time.perf_counter()
        startup.update(model_load_completed=True,status='warmup',load_elapsed_s=load_elapsed)
        write_json(args.output/'startup.json',startup)
        for size in [(1024,768),(768,1024),(1024,1024)]:
            source=Image.new('RGB',size,'white')
            for action in (['features'] if lock['kind']=='features' else ['new_direct','legacy_selector']):
                startup['current_warmup']={'source':'synthetic_white','size':list(size),'action':action}
                startup['warmup_invocations_started']+=1
                write_json(args.output/'startup.json',startup)
                capture.expected=None
                if action=='features':
                    messages,images,_=make_request(source,'Where is the text?','legacy_selector',config)
                    capture(text=[format_prompt(capture,messages,True)],images=images,return_tensors='pt')
                    capture.expected=capture.last
                    vectors=extract_vectors(backend,messages,images,normalized_boxes(size))
                    startup['warmup_invocations_completed']+=1
                    project_vectors(vectors['region_vectors'],vectors['query_vector'])
                    item={'source':'synthetic_white','size':list(size),'kind':'features','generated_tokens':0}
                else:
                    messages,images,_=make_request(source,'Where is the text?',action,config)
                    result=backend.generate(messages,images,2,answer_prefix=action!='legacy_selector')
                    startup['warmup_invocations_completed']+=1
                    item={'source':'synthetic_white','size':list(size),'action':action,'max_new_tokens':2,
                          'generated_tokens':result['generated_tokens']}
                startup['warmup_calls'].append(item);startup['current_warmup']=None
                write_json(args.output/'startup.json',startup)
        backend.torch.cuda.synchronize()
        run={k:lock[k] for k in ('role','kind','run_id','config','runtime','code_bindings','model','model_file_sha256','manifest_sha256')}
        run.update(execution_lock_sha256=sha_file(args.lock),created_utc=now(),warmup_calls=startup['warmup_calls'],
                   load_elapsed_s=load_elapsed,warmup_elapsed_s=time.perf_counter()-warm_start,
                   inference_labels_loaded=False,roi_annotation_used=False,answer_page_privileged=True,
                   raw_feature_persistence_timing='After measured action elapsed_s; included in run wall time')
        write_json(args.output/'run.json',run)
        startup.update(status='running',warmup_finished_utc=now())
        write_json(args.output/'startup.json',startup)
        started=time.perf_counter()
        with (args.output/'records.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for index,row in enumerate(rows,1):
                selected=None
                actions=['features'] if lock['kind']=='features' else action_order(row.example_id,config['seed'])
                for action in actions:
                    current_action={'example_id':row.example_id,'action':action}
                    backend.torch.cuda.synchronize();backend.torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
                    with Image.open(safe_path(args.manifest.parent,row.image_path)) as im:source=im.convert('RGB')
                    request_action='legacy_selector' if action=='features' else action
                    messages,images,geometry=make_request(source,row.question,request_action,config,selected)
                    capture.expected=_expected(pf,row,request_action,geometry,selected)
                    if action=='features':
                        main_attempts+=1
                        vectors=extract_vectors(backend,messages,images,normalized_boxes(source.size))
                        main_model_completed+=1
                        projected=project_vectors(vectors['region_vectors'],vectors['query_vector'])
                        result={k:v for k,v in vectors.items() if k not in ('region_vectors','query_vector')}
                        result.update(projected_features=projected.tolist(),projected_sha256=canonical_hash(projected.tolist()),generated_tokens=0)
                    else:
                        selector=action=='legacy_selector'
                        cap=config['selector_max_new_tokens'] if selector else config['answer_max_new_tokens']
                        main_attempts+=1
                        result=backend.generate(messages,images,cap,answer_prefix=not selector)
                        main_model_completed+=1
                        if selector:
                            selected,valid=parse_region(result['response'],result['generation_truncated'])
                            result.update(selector_region=selected,selector_valid=valid)
                    backend.torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    record={'example_id':row.example_id,'source_cluster_id':row.source_cluster_id,'action':action,'status':'ok',
                            'observation_sha256':canonical_hash(asdict(row)),'geometry':geometry,**result,
                            'processor_capture':capture.last,'processor_observer_elapsed_s':capture.elapsed_s,
                            'elapsed_s':elapsed,'peak_memory_gib':backend.torch.cuda.max_memory_allocated()/1024**3,
                            'peak_reserved_gib':backend.torch.cuda.max_memory_reserved()/1024**3}
                    if action=='features':
                        import numpy as np
                        relative='raw_features/'+hashlib.sha256(row.example_id.encode()).hexdigest()+'.npz'
                        path=safe_path(args.output,relative);path.parent.mkdir(exist_ok=True)
                        require(not path.exists(),'Raw feature file already exists')
                        np.savez(path,region_vectors=vectors['region_vectors'],query_vector=vectors['query_vector'])
                        record.update(raw_features_path=relative,raw_features_sha256=sha_file(path))
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    count+=1;measured+=elapsed
                    current_action=None
                os.fsync(stream.fileno())
                print(json.dumps({'kind':lock['kind'],'completed_sources':index,'total_sources':len(rows),
                                  'records':count,'elapsed_s':round(time.perf_counter()-started,2)}),flush=True)
        require(count==lock['expected_records'],'Wrong execution count')
        startup.update(status='complete',finished_utc=now(),main_invocations_started=main_attempts,
                       main_invocations_completed=main_model_completed,records_completed=count)
        write_json(args.output/'startup.json',startup)
        write_json(args.output/'completed.json',{'records':count,'records_sha256':sha_file(args.output/'records.jsonl'),
             'finished_utc':now(),'elapsed_s':time.perf_counter()-started,'sum_measured_action_elapsed_s':measured,
             'elapsed_s_scope':'Single uninterrupted post-warmup segment; includes raw-feature and record persistence',
             'startup_sha256':sha_file(args.output/'startup.json')})
    except BaseException as error:
        failed_phase=startup['status']
        startup.update(status='failed',failed_phase=failed_phase,failed_utc=now(),
                       main_invocations_started=main_attempts,main_invocations_completed=main_model_completed,
                       records_completed=count,current_main_action=current_action)
        write_json(args.output/'startup.json',startup)
        write_json(args.output/'error.json',{'created_utc':now(),'type':type(error).__name__,'message':str(error),
             'records_completed':count,'failed_phase':failed_phase,'startup_sha256':sha_file(args.output/'startup.json'),
             'warmup_invocations_started':startup['warmup_invocations_started'],
             'warmup_invocations_completed':startup['warmup_invocations_completed'],
             'main_invocations_started':main_attempts,'main_invocations_completed':main_model_completed,
             'attempt_count_semantics':'Started invocation attempts can fail before, during, or after GPU work; do not silently discard them'})
        raise


def validate_completed(manifest,lock_path,run_dir,kind):
    from PIL import Image
    import numpy as np
    manifest,run_dir=Path(manifest),Path(run_dir)
    lock,config=validate_lock(lock_path,manifest);require(lock['kind']==kind,'Wrong run kind')
    rows=observations(manifest,lock['role']);pf=preflight_rows(lock['role'])
    run=read_json(run_dir/'run.json');done=read_json(run_dir/'completed.json')
    startup=read_json(run_dir/'startup.json')
    require(not (run_dir/'error.json').exists(),'Run error must be resolved explicitly')
    for key in ('role','kind','run_id','config','runtime','code_bindings','model','model_file_sha256','manifest_sha256'):
        require(run[key]==lock[key],'Run identity changed: '+key)
    require(run['execution_lock_sha256']==sha_file(lock_path),'Run lock hash differs')
    require(run['inference_labels_loaded'] is False and run['roi_annotation_used'] is False and run['answer_page_privileged'] is True,'Inference isolation differs')
    require(utc(run['created_utc'])>=utc(lock['locked_at_utc']) and utc(done['finished_utc'])>=utc(run['created_utc']),'Invalid time ordering')
    require(len(run['warmup_calls'])==(6 if kind=='generation' else 3),'Unexpected warmup count')
    require(sha_file(run_dir/'startup.json')==done['startup_sha256'],'Startup ledger hash differs')
    require(startup['status']=='complete' and startup['model_load_completed'] is True,'Startup did not complete')
    for key in ('role','kind','run_id'):
        require(startup[key]==run[key],'Startup identity differs: '+key)
    expected_warmups=6 if kind=='generation' else 3
    require(startup['planned_warmup_invocations']==startup['warmup_invocations_started']
            ==startup['warmup_invocations_completed']==len(startup['warmup_calls'])==expected_warmups,
            'Startup warmup count differs')
    require(startup['warmup_calls']==run['warmup_calls'] and startup['current_warmup'] is None,'Warmup ledger differs')
    require(startup['planned_main_invocations']==startup['main_invocations_started']
            ==startup['main_invocations_completed']==startup['records_completed']==lock['expected_records'],
            'Main invocation ledger differs')
    require(utc(lock['locked_at_utc'])<=utc(startup['created_utc'])<=utc(run['created_utc'])
            <=utc(startup['finished_utc'])<=utc(done['finished_utc']),'Startup time ordering differs')
    for key in ('elapsed_s','sum_measured_action_elapsed_s'):
        require(type(done[key]) in (int,float) and math.isfinite(done[key]) and done[key]>0,'Invalid completed timing')
    require(done['elapsed_s']>=done['sum_measured_action_elapsed_s'],'Run wall time below measured sum')
    records=read_jsonl(run_dir/'records.jsonl')
    require(sha_file(run_dir/'records.jsonl')==done['records_sha256'],'Raw completion hash differs')
    require(len(records)==done['records']==lock['expected_records'],'Incomplete record count')
    expected_pairs=[(r.example_id,a) for r in rows for a in (['features'] if kind=='features' else action_order(r.example_id,config['seed']))]
    require([(r['example_id'],r['action']) for r in records]==expected_pairs,'Record order/coverage differs')
    table={(r['example_id'],r['action']):r for r in records}
    for row in rows:
        with Image.open(safe_path(manifest.parent,row.image_path)) as im:source=im.convert('RGB')
        selected=None
        for action in (['features'] if kind=='features' else action_order(row.example_id,config['seed'])):
            record=table[row.example_id,action]
            require(record['status']=='ok' and record['source_cluster_id']==row.source_cluster_id,'Record source/status differs')
            require(record['observation_sha256']==canonical_hash(asdict(row)),'Observation record hash differs')
            request_action='legacy_selector' if action=='features' else action
            _,_,geometry=make_request(source,row.question,request_action,config,selected)
            require(record['geometry']==geometry,'Actual geometry differs')
            expected=_expected(pf,row,request_action,geometry,selected)
            require(record['processor_capture']==expected,'Actual CPU tensors differ')
            for key in ('input_tokens','visual_tokens','image_grid_thw'):
                require(record[key]==expected[key],'Token accounting differs: '+key)
            for key in ('elapsed_s','peak_memory_gib','peak_reserved_gib','processor_observer_elapsed_s'):
                require(type(record[key]) in (int,float) and math.isfinite(record[key]) and record[key]>0,'Invalid cost '+key)
            require(record['peak_reserved_gib']>=record['peak_memory_gib'] and record['elapsed_s']>=record['processor_observer_elapsed_s'],'Impossible memory/time')
            if action=='features':
                from refinement_features import raw_feature_digest,pool_image_tokens
                from refinement_ranker import project_vectors
                x=np.asarray(record['projected_features'],dtype=np.float64)
                require(x.shape==(9,73) and np.isfinite(x).all(),'Invalid projected features')
                require(canonical_hash(x.tolist())==record['projected_sha256'],'Projected feature hash differs')
                require(record['generated_tokens']==0 and record['query_token_semantics']=='final_nonpadding_prompt_token','Feature query/generation differs')
                require(np.array_equal(x[:,64:],np.eye(9)),'Position features differ')
                require(np.allclose(np.linalg.norm(x[:,:32],axis=1),1,atol=1e-12),'Region features not normalized')
                layout=pf[row.example_id]['actions']['legacy_selector']['feature_layout']
                for key in ('query_token_index','query_token_semantics','image_token_positions','image_region_token_counts'):
                    require(record[key]==layout[key],'Frozen feature layout differs: '+key)
                require(type(record['query_token_index']) is int and record['query_token_index']==record['input_tokens']-1,
                        'Feature query is not the final unpadded token')
                positions=record['image_token_positions']
                require(isinstance(positions,list) and all(type(p) is int for p in positions)
                        and len(positions)==record['visual_tokens'] and positions
                        and positions==list(range(positions[0],positions[-1]+1))
                        and positions[0]>0 and positions[-1]+1<record['query_token_index'],'Invalid feature image positions')
                _,counts=pool_image_tokens(np.zeros((record['visual_tokens'],1),dtype=np.float32),
                    record['image_grid_thw'][0],normalized_boxes(source.size))
                require(record['image_region_token_counts']==counts
                        and all(type(n) is int for n in record['image_region_token_counts']),'Incorrect feature pooling counts')
                relative='raw_features/'+hashlib.sha256(row.example_id.encode()).hexdigest()+'.npz'
                require(record['raw_features_path']==relative,'Raw feature path differs')
                raw_path=safe_path(run_dir,relative)
                require(sha_file(raw_path)==record['raw_features_sha256'],'Raw feature archive changed')
                with np.load(raw_path,allow_pickle=False) as raw:
                    require(set(raw.files)=={'region_vectors','query_vector'},'Unexpected raw feature archive keys')
                    regions=raw['region_vectors'];query=raw['query_vector']
                    require(regions.dtype==query.dtype==np.dtype('float32'),'Raw features must preserve float32 dtype')
                    require(raw_feature_digest(regions,query)==record['raw_feature_sha256'],'Raw vector digest differs')
                    replay=project_vectors(regions,query)
                require(np.array_equal(x,replay) and canonical_hash(replay.tolist())==record['projected_sha256'],
                        'Projected features do not reproduce from raw vectors')
            else:
                selector=action=='legacy_selector';cap=config['selector_max_new_tokens'] if selector else config['answer_max_new_tokens']
                require(type(record['generated_tokens']) is int and 0<record['generated_tokens']<=cap,'Invalid generated count')
                require(type(record['generation_truncated']) is bool,'Missing truncation flag')
                require(record['answer_prefix_prefilled'] is (not selector),'Answer prefix differs')
                require(record['response']==(('ANSWER:' if not selector else '')+record['raw_continuation']),'Raw continuation differs')
                require(record['mean_token_logprob'] is None or (math.isfinite(record['mean_token_logprob']) and record['mean_token_logprob']<=1e-5),'Invalid logprob')
                if selector:
                    selected,valid=parse_region(record['response'],record['generation_truncated'])
                    require(record['selector_region']==selected and record['selector_valid']==valid,'Selector parse differs')
    require(math.isclose(sum(r['elapsed_s'] for r in records),done['sum_measured_action_elapsed_s'],rel_tol=1e-10),'Measured total differs')
    return [asdict(r) for r in rows],records,run,lock


def validate_generation(manifest,lock_path,run_dir):return validate_completed(manifest,lock_path,run_dir,'generation')
def validate_features(manifest,lock_path,run_dir):return validate_completed(manifest,lock_path,run_dir,'features')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    for command in ('preflight','lock','run','validate'):
        p=sub.add_parser(command);p.add_argument('--manifest',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        if command in ('preflight','lock'):
            p.add_argument('--role',choices=['engineering','main'],required=True)
        if command!='validate':p.add_argument('--model-dir',type=Path,required=True)
        if command in ('run','validate'):p.add_argument('--lock',type=Path,required=True)
        if command=='lock':
            p.add_argument('--kind',choices=['features','generation'],required=True);p.add_argument('--run-id',required=True)
            p.add_argument('--verification',type=Path,required=True);p.add_argument('--decisions',type=Path)
            p.add_argument('--prerequisite',nargs=3,action='append',metavar=('MANIFEST','LOCK','RUN'))
    args=parser.parse_args()
    if args.command=='preflight':preflight(args)
    elif args.command=='lock':lock_execution(args)
    elif args.command=='run':execute(args)
    else:
        lock=read_json(args.lock)
        _,records,_,_=validate_completed(args.manifest,args.lock,args.output,lock['kind'])
        print(json.dumps({'validation':'PASS','records':len(records)}))


if __name__=='__main__':main()
