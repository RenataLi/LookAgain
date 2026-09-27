"""Frozen given-page selection experiment; inference reads only observation fields."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/'src'), str(PROJECT/'experiments')]
from region_selection_core import (Observation, candidate_boxes, build_selector_request,
    build_answer_request, parse_region, canonical_hash)
from dude_replication import (require, read_json, read_jsonl, write_json, now, utc,
    safe_path, model_files, configure_runtime, validate_source_design)
from check_dude_requests import sha_file, tensor_hash

CONFIG = PROJECT/'configs/region_selection.json'
PREPARATION = PROJECT/'reports/region_selection_preparation'
INPUT_REPORT = PROJECT/'reports/region_selection_inputs'
RESERVATION_SHA = '9fbaded61d06702c3752f3ac0d3c69d64d6459e95619c563ee15bc1f8905f09f'
ACTIONS = ('direct','selector',*(f'native_{i}' for i in range(1,10)),'selected_degraded','highres')
NEW_FILES = ('experiments/region_selection_core.py','experiments/run_region_selection.py',
             'experiments/analyze_region_selection.py','configs/region_selection.json',
             'docs/region_selection_protocol.md','experiments/prepare_region_selection.py')

def public_manifest(role):return PREPARATION/(role+'_observation_manifest.jsonl')
def public_labels(role):return PREPARATION/(role+'_labels.jsonl')

def source_bindings():
    return {p.relative_to(PROJECT).as_posix():sha_file(p) for p in sorted(PREPARATION.rglob('*')) if p.is_file()}

def input_bindings():
    return {p.relative_to(PROJECT).as_posix():sha_file(p) for role in ('engineering','development','evaluation')
            for p in (INPUT_REPORT/(role+'.jsonl'),INPUT_REPORT/(role+'.meta.json'))}

def validate_reservation():
    path=PREPARATION/'split_reservation.json'
    require(sha_file(path)==RESERVATION_SHA,'Original source reservation changed')
    reserved=read_json(path)
    cohorts=[]
    for role,cohort in reserved['cohorts'].items():
        require(sha_file(public_manifest(role))==cohort['observation_manifest_sha256'],'Reserved observation bytes changed')
        require(sha_file(public_labels(role))==cohort['labels_sha256'],'Reserved label bytes changed')
        rows=[Observation.from_manifest(r) for r in read_jsonl(public_manifest(role))]
        require([r.example_id for r in rows]==cohort['example_ids'],'Reserved example order changed')
        require([r.source_cluster_id for r in rows]==cohort['source_cluster_ids'],'Reserved source order changed')
        cohorts.append(set(r.source_cluster_id for r in rows))
    require(sum(map(len,cohorts))==len(set.union(*cohorts)),'Cohort source overlap')
    require(set(reserved['cohorts']['evaluation']['source_cluster_ids']).isdisjoint(reserved['historical_outcome_source_ids']),'Evaluation overlaps earlier outcomes')
    return reserved

def code_bindings():
    from dude_replication import code_bindings as previous_bindings
    return {**previous_bindings(), **{p:sha_file(PROJECT/p) for p in NEW_FILES}}

def observations(manifest, role, config):
    path=Path(manifest).resolve()
    rows=[Observation.from_manifest(r) for r in read_jsonl(path)]
    require(len(rows)==config['source_counts'][role], 'Unexpected cohort size')
    for key in ('example_id','source_cluster_id'):
        require(len({getattr(r,key) for r in rows})==len(rows), 'Duplicate '+key)
    for row in rows:
        require(sha_file(safe_path(path.parent,row.image_path))==row.image_sha256, 'Image hash differs')
    return rows

def validate_calibration(calibration, config):
    require(calibration['schema_version']==1,'Invalid calibration version')
    utc(calibration['created_utc'])
    require(calibration['config_sha256']==sha_file(CONFIG),'Calibration config differs')
    require(calibration['method']=='numpy_linear_quantile' and calibration['missing_confidence_rule']=='zoom','Calibration rule differs')
    keys=[str(q) for q in config['confidence_zoom_fractions']]
    require(set(calibration['thresholds'])==set(keys),'Missing confidence thresholds')
    values=[calibration['thresholds'][k] for k in keys]
    require(all(type(v) in (int,float) and math.isfinite(v) and v<=1e-5 for v in values),'Invalid threshold')
    require(values==sorted(values),'Unordered thresholds')
    require(type(calibration['finite_development_sources']) is int and 0<calibration['finite_development_sources']<=64,'Invalid calibration sample count')

def validate_development_thresholds(calibration,evidence):
    import numpy as np
    direct=evidence['direct_confidence']
    require(len(direct)==64 and len({r['example_id'] for r in direct})==64,'Development confidence coverage differs')
    values=[r['mean_token_logprob'] for r in direct if r['mean_token_logprob'] is not None]
    require(values and all(type(v) in (int,float) and math.isfinite(v) and v<=1e-5 for v in values),'Invalid development confidence values')
    expected={str(q):float(np.quantile(values,q,method='linear')) for q in (.25,.5,.75)}
    require(calibration['thresholds']==expected,'Thresholds differ from the prescribed development-only quantiles')
    require(calibration['finite_development_sources']==len(values),'Calibration finite sample count differs')
    require(calibration.get('development_sources')==64 and calibration.get('missing_development_sources')==64-len(values),'Calibration source count differs')
    require(calibration.get('uses_reference_labels') is False,'Confidence calibration must not use references')
    require(calibration.get('development_direct_confidence')==direct,'Calibration contributing observations differ')
    require(calibration.get('analysis_code_sha256')==sha_file(PROJECT/'experiments/analyze_region_selection.py'),'Calibration code differs')

def validate_lock(lock_path, manifest, role=None, model_dir=None):
    validate_source_design()
    reservation=validate_reservation()
    lock=read_json(lock_path); config=read_json(CONFIG)
    require(lock['schema_version']==1 and lock['execution_locked'] is True,'Execution lock required')
    require(role is None or role==lock['role'],'Role mismatch')
    role=lock['role']
    require(role in config['source_counts'],'Unknown cohort role')
    require(lock['config']==config and lock['config_sha256']==sha_file(CONFIG),'Config mismatch')
    require(lock['code_bindings']==code_bindings(),'Code changed after execution lock')
    require(lock['manifest_sha256']==sha_file(manifest),'Observation manifest changed')
    require(lock['manifest_sha256']==sha_file(public_manifest(role)),'Role differs from reserved manifest')
    require(lock['labels_sha256']==sha_file(public_labels(role)),'Evaluation labels changed')
    require(lock['source_bindings']==source_bindings(),'Missing, extra or changed source preparation bindings')
    require(lock['input_bindings']==input_bindings(),'Missing, extra or changed CPU preflight bindings')
    for rel,digest in lock['source_bindings'].items():
        require(sha_file(safe_path(PROJECT,rel))==digest,'Source preparation changed: '+rel)
    for rel,digest in lock['input_bindings'].items():
        require(sha_file(safe_path(PROJECT,rel))==digest,'CPU preflight changed: '+rel)
    require(lock['verification']['status']=='PASS' and lock['verification']['code_bindings']==lock['code_bindings'],'Unverified code')
    require(lock['expected_records']==config['source_counts'][role]*13,'Record target differs')
    require(isinstance(lock['run_id'],str) and lock['run_id'].strip(),'Run ID missing')
    utc(lock['locked_at_utc'])
    require(utc(lock['locked_at_utc'])>=utc(reservation['reserved_at_utc']),'Execution predates reservation')
    if role!='engineering':
        require(lock['engineering_evidence']['records']==26 and lock['engineering_evidence']['code_bindings']==lock['code_bindings'],'Completed matching engineering prerequisite required')
        require(lock['engineering_evidence']['model_file_sha256']==lock['model_file_sha256'],'Engineering model files differ')
        require(lock['engineering_evidence']['runtime']==lock['runtime'],'Engineering runtime differs')
        require(lock['engineering_evidence']['run_id']!=lock['run_id'],'Prerequisite/current run IDs must differ')
        require(utc(lock['engineering_evidence']['finished_utc'])<=utc(lock['locked_at_utc']),'Engineering not finished before current lock')
    if role=='evaluation':
        validate_calibration(lock['calibration'],config)
        require(utc(lock['calibration']['created_utc'])<=utc(lock['locked_at_utc']),'Calibration postdates evaluation lock')
        require(canonical_hash(lock['calibration'])==lock['calibration_canonical_sha256'],'Calibration changed')
        require(lock['development_evidence']['records']==832,'Development prerequisite missing')
        require(lock['calibration']['development_records_sha256']==lock['development_evidence']['records_sha256'],'Calibration used different development data')
        require(lock['calibration']['development_lock_sha256']==lock['development_evidence']['lock_sha256'],'Calibration development lock differs')
        require(lock['development_evidence']['code_bindings']==lock['code_bindings'],'Development code differs')
        require(lock['development_evidence']['model_file_sha256']==lock['model_file_sha256'],'Development model differs')
        require(lock['development_evidence']['runtime']==lock['runtime'],'Development runtime differs')
        require(len({lock['run_id'],lock['development_evidence']['run_id'],lock['engineering_evidence']['run_id']})==3,'Execution stage run IDs must differ')
        require(utc(lock['development_evidence']['finished_utc'])<=utc(lock['calibration']['created_utc']),'Calibration precedes completed development')
        validate_development_thresholds(lock['calibration'],lock['development_evidence'])
    if model_dir is not None:
        require(read_json(Path(model_dir)/'lookagain-model.json')==lock['model'],'Model provenance changed')
        require(model_files(model_dir)==lock['model_file_sha256'],'Model/processor bytes changed')
    return lock,config

class ProcessorCapture:
    """Observe CPU inputs without changing the BatchFeature sent to CUDA."""
    def __init__(self,processor):
        self.processor=processor; self.last=None; self.elapsed_s=0; self.expected=None
    def __getattr__(self,name):
        return getattr(self.processor,name)
    def __call__(self,*args,**kwargs):
        inputs=self.processor(*args,**kwargs)
        start=time.perf_counter()
        grids=inputs['image_grid_thw'].tolist()
        self.last={'tensor_sha256':{k:tensor_hash(v) for k,v in inputs.items()},
            'formatted_prompt_sha256':hashlib.sha256(kwargs['text'][0].encode()).hexdigest(),
            'input_tokens':int(inputs['input_ids'].shape[-1]),'image_grid_thw':grids,
            'visual_tokens':sum(t*h*w//4 for t,h,w in grids)}
        if self.expected is not None:
            require(self.last==self.expected,'Actual CPU processor inputs differ from pre-inference check')
        self.elapsed_s=time.perf_counter()-start
        return inputs

def make_request(source,question,action,config,selected=None):
    if action=='selector':return build_selector_request(source,question,config)
    if action.startswith('native_'):return build_answer_request(source,question,'native',config,int(action.split('_')[1]))
    if action=='selected_degraded':return build_answer_request(source,question,'degraded',config,selected)
    return build_answer_request(source,question,action,config)

def action_order(example_id,seed):
    rest=list(ACTIONS[2:])
    rng=random.Random(int.from_bytes(hashlib.sha256(f'region-order:{seed}:{example_id}'.encode()).digest(),'big'))
    rng.shuffle(rest)
    return ['direct','selector',*rest]

def preflight_key(action,selected):
    return f'degraded_{selected}' if action=='selected_degraded' else action

def preflight_rows(role):
    rows=read_jsonl(INPUT_REPORT/(role+'.jsonl'))
    table={r['example_id']:r for r in rows}
    require(len(table)==len(rows),'Duplicate CPU preflight source')
    return table

def preflight(args):
    from PIL import Image
    import torch
    from transformers import AutoProcessor
    config=read_json(CONFIG); rows=observations(args.manifest,args.role,config)
    validate_reservation()
    require(sha_file(args.manifest)==sha_file(public_manifest(args.role)),'Wrong source reservation')
    require(not args.output.exists(),'Use a new CPU preflight path')
    processor=AutoProcessor.from_pretrained(str(args.model_dir),local_files_only=True,trust_remote_code=False)
    capture=ProcessorCapture(processor);torch.set_num_threads(4)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    count=0
    with args.output.open('x',encoding='utf-8',newline='\n') as stream:
        for index,row in enumerate(rows,1):
            with Image.open(safe_path(Path(args.manifest).parent,row.image_path)) as im:source=im.convert('RGB')
            cases={}
            for key in ('direct','selector',*(f'native_{i}' for i in range(1,10)),*(f'degraded_{i}' for i in range(1,10)),'highres'):
                action='selected_degraded' if key.startswith('degraded_') else key
                selected=int(key.split('_')[1]) if action=='selected_degraded' else None
                messages,images,geometry=make_request(source,row.question,action,config,selected)
                if action=='selector':
                    prompt=processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
                else:
                    prompt=processor.apply_chat_template([*messages,{'role':'assistant','content':'ANSWER:'}],tokenize=False,continue_final_message=True)
                capture(text=[prompt],images=images,return_tensors='pt')
                cases[key]={'geometry_sha256':canonical_hash(geometry),'processor_capture':capture.last}
                count+=1
            stream.write(json.dumps({'example_id':row.example_id,'observation_sha256':canonical_hash(asdict(row)),'actions':cases},ensure_ascii=False)+'\n');stream.flush()
            if index%8==0 or index==len(rows):print(json.dumps({'role':args.role,'preflight_sources':index,'requests':count}),flush=True)
    write_json(args.output.with_suffix('.meta.json'),{'role':args.role,'created_utc':now(),'requests':count,
        'sources':len(rows),'manifest_sha256':sha_file(args.manifest),'config_sha256':sha_file(CONFIG),
        'preflight_sha256':sha_file(args.output),'code_bindings':code_bindings(),'new_model_calls':0,
        'processor_files':{p.name:sha_file(p) for p in args.model_dir.iterdir() if p.is_file() and p.suffix in ('.json','.txt')},
        'note':'Actual CPU processor calls for every possible action including all nine degraded windows; no VLM weights loaded.'})

def validate_completed(manifest,lock_path,run_dir):
    from PIL import Image
    manifest,run_dir=Path(manifest).resolve(),Path(run_dir).resolve()
    lock,config=validate_lock(lock_path,manifest)
    rows=observations(manifest,lock['role'],config)
    run=read_json(run_dir/'run.json'); completed=read_json(run_dir/'completed.json')
    expected={'execution_lock_sha256':sha_file(lock_path),'manifest_sha256':sha_file(manifest),
              'config':config,'role':lock['role'],'run_id':lock['run_id'],'runtime':lock['runtime'],
              'code_bindings':lock['code_bindings'],'model':lock['model'],
              'model_file_sha256':lock['model_file_sha256']}
    require(all(run.get(k)==v for k,v in expected.items()),'Run identity differs')
    require(run.get('inference_labels_loaded') is False and run.get('roi_annotation_used') is False,
            'Inference label/ROI isolation declaration differs')
    require(run.get('answer_page_privileged') is True,'Answer-page privilege must remain explicit')
    require(utc(run['created_utc'])>=utc(lock['locked_at_utc']),'Run precedes lock')
    require(not (run_dir/'error.json').exists(),'Run has execution error')
    records=read_jsonl(run_dir/'records.jsonl')
    require(sha_file(run_dir/'records.jsonl')==completed['records_sha256'],'Completed raw hash mismatch')
    require(completed['records']==len(records)==lock['expected_records'],'Incomplete records')
    require(utc(completed['finished_utc'])>=utc(run['created_utc']),'Completion precedes run')
    for field in ('elapsed_s','sum_measured_action_elapsed_s'):
        require(type(completed[field]) in (int,float) and math.isfinite(completed[field]) and completed[field]>0,'Invalid completed timing')
    require(len(run['warmup_calls'])==6,'Synthetic warmup count differs')
    require([(r['example_id'],r['action']) for r in records]==[(r.example_id,a) for r in rows for a in action_order(r.example_id,config['seed'])],'Action order differs')
    require(math.isclose(sum(r['elapsed_s'] for r in records),completed['sum_measured_action_elapsed_s'],rel_tol=1e-10,abs_tol=1e-8),'Measured timing sum differs')
    table={(r['example_id'],r['action']):r for r in records}
    require(len(table)==len(records),'Duplicate record')
    require(set(table)=={(r.example_id,a) for r in rows for a in ACTIONS},'Wrong action coverage')
    pf=preflight_rows(lock['role'])
    require(set(pf)=={r.example_id for r in rows},'CPU preflight coverage differs')
    for row in rows:
        require(pf[row.example_id]['observation_sha256']==canonical_hash(asdict(row)),'CPU preflight observation differs')
        selector=table[row.example_id,'selector']
        selected,valid=parse_region(selector['response'],selector['generation_truncated'])
        require(selector['selector_region']==selected and selector['selector_valid']==valid,'Selector decision mismatch')
        with Image.open(safe_path(manifest.parent,row.image_path)) as im:source=im.convert('RGB')
        for action in ACTIONS:
            r=table[row.example_id,action]
            require(r['status']=='ok' and r['source_cluster_id']==row.source_cluster_id,'Source/status mismatch')
            require(r['observation_sha256']==canonical_hash(asdict(row)),'Observation identity differs')
            _,_,geometry=make_request(source,row.question,action,config,selected)
            require(r['geometry']==geometry,'Reconstructed request differs')
            expected=pf[row.example_id]['actions'][preflight_key(action,selected)]
            require(expected['geometry_sha256']==canonical_hash(geometry),'CPU preflight geometry differs')
            require(r['processor_capture']==expected['processor_capture'],'Production processor capture differs from frozen preflight')
            require(r['answer_prefix_prefilled']==(action!='selector'),'Answer prefix mismatch')
            require(r['response']==('' if action=='selector' else 'ANSWER:')+r['raw_continuation'],'Response continuation mismatch')
            cap=config['selector_max_new_tokens'] if action=='selector' else config['answer_max_new_tokens']
            require(type(r['generated_tokens']) is int and 1<=r['generated_tokens']<=cap,'Bad generated length')
            require(type(r['generation_truncated']) is bool and (not r['generation_truncated'] or r['generated_tokens']==cap),'Bad truncation')
            for name in ('elapsed_s','peak_memory_gib','peak_reserved_gib','processor_observer_elapsed_s'):
                require(type(r[name]) in (int,float) and math.isfinite(r[name]) and r[name]>=0,'Bad measured '+name)
            require(r['elapsed_s']>0 and r['processor_observer_elapsed_s']<=r['elapsed_s'],'Bad timing')
            require(r['peak_reserved_gib']>=r['peak_memory_gib']>0,'Bad memory telemetry')
            lp=r['mean_token_logprob']
            require(lp is None or type(lp) in (int,float) and math.isfinite(lp) and lp<=1e-5,'Bad confidence')
            for name in ('input_tokens','visual_tokens','image_grid_thw'):
                require(r[name]==r['processor_capture'][name],'Captured processor metadata differs')
            require(all(repr(k) and re_hash(v) for k,v in r['processor_capture']['tensor_sha256'].items()),'Bad tensor digests')
            require(re_hash(r['processor_capture']['formatted_prompt_sha256']),'Bad prompt digest')
    return rows,records,run,lock

def re_hash(value):
    import re
    return isinstance(value,str) and re.fullmatch('[0-9a-f]{64}',value) is not None

def completed_evidence(manifest,lock_path,run_dir,role):
    rows,records,run,lock=validate_completed(manifest,lock_path,run_dir)
    require(lock['role']==role,'Wrong prerequisite role')
    result={'records':len(records),'records_sha256':sha_file(Path(run_dir)/'records.jsonl'),
            'lock_sha256':sha_file(lock_path),'run_id':lock['run_id'],
            'code_bindings':lock['code_bindings'],'model_file_sha256':lock['model_file_sha256'],'runtime':run['runtime'],
            'finished_utc':read_json(Path(run_dir)/'completed.json')['finished_utc']}
    if role=='development':
        table={r['example_id']:r['mean_token_logprob'] for r in records if r['action']=='direct'}
        result['direct_confidence']=[{'example_id':r.example_id,'mean_token_logprob':table[r.example_id]} for r in rows]
    return result

def lock_execution(args):
    validate_source_design()
    validate_reservation()
    config=read_json(CONFIG); rows=observations(args.manifest,args.role,config)
    verification=read_json(args.verification)
    require(verification['status']=='PASS' and verification['code_bindings']==code_bindings(),'Verification missing/current code not checked')
    source_files=sorted(p for p in PREPARATION.rglob('*') if p.is_file())
    require(source_files,'Source reservation must exist before locking execution')
    canonical_manifest=public_manifest(args.role)
    require(sha_file(args.manifest)==sha_file(canonical_manifest),'Observation manifest differs from public source reservation')
    input_files=[]
    for role in config['source_counts']:
        meta_path=INPUT_REPORT/(role+'.meta.json');meta=read_json(meta_path)
        require(meta['code_bindings']==code_bindings() and meta['config_sha256']==sha_file(CONFIG),'CPU preflight code differs')
        require(meta['sources']==config['source_counts'][role] and meta['requests']==meta['sources']*21,'CPU preflight incomplete')
        require(meta['manifest_sha256']==sha_file(public_manifest(role)),'CPU preflight source differs')
        require(meta['preflight_sha256']==sha_file(INPUT_REPORT/(role+'.jsonl')),'CPU preflight record hash differs')
        input_files.extend([meta_path,INPUT_REPORT/(role+'.jsonl')])
    model=read_json(args.model_dir/'lookagain-model.json')
    require(model['model_id']==config['model_id'] and model['revision']==config['model_revision'],'Wrong model')
    lock={'schema_version':1,'execution_locked':True,'role':args.role,'run_id':args.run_id,
          'locked_at_utc':now(),'config':config,'config_sha256':sha_file(CONFIG),
          'code_bindings':code_bindings(),'manifest_sha256':sha_file(args.manifest),
          'labels_sha256':sha_file(public_labels(args.role)),
          'expected_records':len(rows)*13,'model':model,'model_file_sha256':model_files(args.model_dir),
          'runtime':configure_runtime(config['seed']),'verification':verification,
          'source_bindings':{p.relative_to(PROJECT).as_posix():sha_file(p) for p in source_files},
          'input_bindings':{p.relative_to(PROJECT).as_posix():sha_file(p) for p in input_files}}
    if args.role!='engineering':
        require(all([args.engineering_manifest,args.engineering_lock,args.engineering_run]),'Engineering prerequisite paths required')
        lock['engineering_evidence']=completed_evidence(args.engineering_manifest,args.engineering_lock,args.engineering_run,'engineering')
    if args.role=='evaluation':
        require(all([args.development_manifest,args.development_lock,args.development_run,args.calibration]),'Development/calibration prerequisite required')
        lock['development_evidence']=completed_evidence(args.development_manifest,args.development_lock,args.development_run,'development')
        calibration=read_json(args.calibration);validate_calibration(calibration,config)
        lock.update(calibration=calibration,calibration_sha256=sha_file(args.calibration),calibration_canonical_sha256=canonical_hash(calibration))
    require(not args.output.exists(),'Use a new lock path')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    write_json(args.output,lock)
    validate_lock(args.output,args.manifest,args.role)
    print(json.dumps({'status':'locked','role':args.role,'sources':len(rows),'records':len(rows)*13,'lock_sha256':sha_file(args.output)}))

def execute(args):
    from PIL import Image
    from lookagain.backend import QwenBackend
    lock,config=validate_lock(args.lock,args.manifest,args.role,args.model_dir)
    runtime=configure_runtime(config['seed']);require(runtime==lock['runtime'],'Runtime changed')
    rows=observations(args.manifest,args.role,config)
    pf=preflight_rows(args.role)
    require(not args.output.exists(),'Use a new run directory; failed/interrupted runs are preserved without silent retry')
    args.output.mkdir(parents=True)
    backend=QwenBackend(args.model_dir,config); capture=ProcessorCapture(backend.processor); backend.processor=capture
    warmups=[]
    for size in ((1024,768),(768,1024),(1024,1024)):
        im=Image.new('RGB',size,'white')
        for action in ('direct','selector'):
            messages,images,_=make_request(im,'Where is the text?',action,config)
            result=backend.generate(messages,images,2,answer_prefix=action!='selector')
            warmups.append({'source':'synthetic_white','size':list(size),'action':action,'max_new_tokens':2,'generated_tokens':result['generated_tokens']})
    run={k:lock[k] for k in ('config','role','run_id','runtime','code_bindings','model','model_file_sha256','manifest_sha256')}
    run.update(execution_lock_sha256=sha_file(args.lock),created_utc=now(),warmup_calls=warmups,
               inference_labels_loaded=False,answer_page_privileged=True,roi_annotation_used=False)
    write_json(args.output/'run.json',run)
    started=time.perf_counter();count=0;measured=0
    try:
        with (args.output/'records.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for index,row in enumerate(rows,1):
                require(pf[row.example_id]['observation_sha256']==canonical_hash(asdict(row)),'CPU preflight observation differs')
                selected=None
                for action in action_order(row.example_id,config['seed']):
                    backend.torch.cuda.synchronize();backend.torch.cuda.reset_peak_memory_stats()
                    start=time.perf_counter()
                    with Image.open(safe_path(Path(args.manifest).parent,row.image_path)) as im:source=im.convert('RGB')
                    messages,images,geometry=make_request(source,row.question,action,config,selected)
                    expected=pf[row.example_id]['actions'][preflight_key(action,selected)]
                    require(expected['geometry_sha256']==canonical_hash(geometry),'Frozen CPU preflight geometry differs')
                    capture.expected=expected['processor_capture']
                    cap=config['selector_max_new_tokens'] if action=='selector' else config['answer_max_new_tokens']
                    result=backend.generate(messages,images,cap,answer_prefix=action!='selector')
                    if action=='selector':
                        selected,valid=parse_region(result['response'],result['generation_truncated'])
                    backend.torch.cuda.synchronize();elapsed=time.perf_counter()-start
                    record={'example_id':row.example_id,'source_cluster_id':row.source_cluster_id,'action':action,'status':'ok',
                            'observation_sha256':canonical_hash(asdict(row)),'geometry':geometry,**result,
                            'processor_capture':capture.last,'processor_observer_elapsed_s':capture.elapsed_s,
                            'elapsed_s':elapsed,'peak_memory_gib':backend.torch.cuda.max_memory_allocated()/1024**3,
                            'peak_reserved_gib':backend.torch.cuda.max_memory_reserved()/1024**3}
                    if action=='selector':
                        record.update(selector_region=selected,selector_valid=valid)
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');stream.flush()
                    count+=1;measured+=elapsed
                import os
                os.fsync(stream.fileno())
                print(json.dumps({'completed_sources':index,'total_sources':len(rows),'records':count,'elapsed_s':round(time.perf_counter()-started,2)}),flush=True)
        require(count==lock['expected_records'],'Wrong execution count')
        write_json(args.output/'completed.json',{'records':count,'records_sha256':sha_file(args.output/'records.jsonl'),
                   'finished_utc':now(),'elapsed_s':time.perf_counter()-started,
                   'elapsed_s_scope':'Single uninterrupted segment after synthetic warmup; loading/preparation excluded',
                   'sum_measured_action_elapsed_s':measured})
    except BaseException as error:
        write_json(args.output/'error.json',{'created_utc':now(),'type':type(error).__name__,'message':str(error),'records_completed':count})
        raise

def main():
    parser=argparse.ArgumentParser(description=__doc__); sub=parser.add_subparsers(dest='command',required=True)
    for command in ('lock','run','preflight'):
        p=sub.add_parser(command);p.add_argument('--role',choices=['engineering','development','evaluation'],required=True)
        p.add_argument('--manifest',type=Path,required=True);p.add_argument('--model-dir',type=Path,required=True)
        p.add_argument('--output',type=Path,required=True)
        if command=='lock':
            p.add_argument('--verification',type=Path,required=True);p.add_argument('--run-id',required=True)
            for item in ('engineering-manifest','engineering-lock','engineering-run','development-manifest','development-lock','development-run','calibration'):
                p.add_argument('--'+item,type=Path)
        elif command=='run':p.add_argument('--lock',type=Path,required=True)
    args=parser.parse_args()
    if args.command=='lock':lock_execution(args)
    elif args.command=='preflight':preflight(args)
    else:execute(args)

if __name__=='__main__':main()
