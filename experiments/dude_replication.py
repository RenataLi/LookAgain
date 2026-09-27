"""Locked DUDE execution and completed-run integrity validation.

The old request builder/backend/scorer are imported unchanged. A processor
observer hashes the actual CPU inputs immediately before their CUDA transfer.
No source/reference text beyond the original question enters the model prompt.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path, PureWindowsPath
import platform
import random
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT / 'src'), str(PROJECT / 'experiments')]
from check_dude_requests import ACTIONS, FROZEN_FILES, sha_file, tensor_hash
from evidence_availability_core import build_request, validate_roi
from dude_metrics import score_response, validate_reference_sets

SOURCE_LOCK_SHA = 'c684b0b59af32e68dd30d72e04b3d61b1d28f16e78d250d8a983021316cb86fd'
CONFIG_SHA = 'd174347176b24fe00d5a4c7fcce069179429092132cd371f421eb69f02b046b5'
SOURCE_REPORT = PROJECT / 'reports/dude_preparation'
NEW_CODE = ['experiments/dude_replication.py', 'experiments/analyze_dude_replication.py',
            'experiments/prepare_dude_review.py']
PACKAGES = ('torch','torchvision','transformers','Pillow','numpy','huggingface-hub','safetensors')

def require(value, message):
    if not value: raise ValueError(message)

def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))

def read_jsonl(path):
    return [json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))
            for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')

def utc(value):
    result = datetime.fromisoformat(value.replace('Z','+00:00'))
    require(result.tzinfo is not None, 'Timezone-aware timestamp required')
    return result

def now(): return datetime.now(timezone.utc).isoformat()

def safe_path(root, relative):
    root = Path(root).resolve(); rel = Path(relative)
    require(not rel.is_absolute() and not PureWindowsPath(relative).drive, 'Absolute asset path rejected')
    result = (root/rel).resolve()
    require(result.is_relative_to(root), 'Asset path escapes root')
    return result

def unique(rows, field='example_id'):
    result = {row[field]:row for row in rows}
    require(len(result)==len(rows), 'Duplicate '+field)
    return result

def code_bindings():
    names = set(NEW_CODE) | set(FROZEN_FILES) | {
        'experiments/check_dude_requests.py', 'experiments/dude_metrics.py',
        'experiments/prepare_dude_replication.py', 'docs/dude_execution.md'}
    return {name:sha_file(PROJECT/name) for name in sorted(names)}

def validate_source_design():
    path=SOURCE_REPORT/'source_design_lock.json'
    require(sha_file(path)==SOURCE_LOCK_SHA, 'Prepared source/design lock changed')
    lock=read_json(path)
    require(lock['source_design_locked'] is True and lock['execution_locked'] is False, 'Invalid source lock state')
    for rel,digest in lock['public_file_bindings'].items():
        require(sha_file(safe_path(PROJECT,rel))==digest, 'Frozen source/design file changed: '+rel)
    for rel,digest in FROZEN_FILES.items():
        require(sha_file(PROJECT/rel)==digest, 'Frozen inference dependency changed: '+rel)
    require(sha_file(PROJECT/'configs/dude_replication.json')==CONFIG_SHA, 'Preparation configuration changed')
    return lock,read_json(PROJECT/'configs/dude_replication.json')

def validate_manifest(manifest, role):
    from PIL import Image
    from prepare_dude_replication import validate_source_review
    require(role in ('main','engineering'), 'Unknown cohort role')
    manifest=Path(manifest).resolve()
    expected=SOURCE_REPORT/(role+'_manifest.jsonl')
    require(sha_file(manifest)==sha_file(expected), 'Manifest differs from frozen cohort')
    rows=read_jsonl(manifest); unique(rows)
    require(len(rows)==(660 if role=='main' else 10), 'Incorrect cohort count')
    require(len({r['source_cluster_id'] for r in rows})==len(rows), 'Duplicate source cluster')
    reviews=unique(read_jsonl(SOURCE_REPORT/'source_review_ledger.jsonl'))
    audits=unique(read_jsonl(SOURCE_REPORT/'source_audit_metadata.jsonl'))
    other=read_jsonl(SOURCE_REPORT/(('engineering' if role=='main' else 'main')+'_manifest.jsonl'))
    require(set(r['source_cluster_id'] for r in rows).isdisjoint(r['source_cluster_id'] for r in other), 'Engineering/main overlap')
    selection=read_json(SOURCE_REPORT/'selection_metadata.json')
    if role=='main':
        require(set(r['source_cluster_id'] for r in rows).isdisjoint(selection['engineering_source_reservations']), 'Reserved engineering source in main')
    for row in rows:
        eid=row['example_id']; review=reviews[eid]; audit=audits[eid]
        require(row['cohort']==role and row['source_split']=='train', 'Wrong role/split')
        require(row['source_id']==row['source_cluster_id'], 'Source identifiers disagree')
        require(row['source_semantic_audit_status']=='approved', 'Unapproved source')
        require(row['answer_page_privileged'] is True and row['roi_privileged'] is True, 'Missing privilege declaration')
        require(row['answer']==row['original_answers'][0], 'Canonical answer changed')
        validate_reference_sets(row['original_answers'],row['validated_primary_answers'],row['original_answer_variants'])
        validate_source_review(row,review)
        require(canonical_hash(review)==row['source_semantic_review_sha256']==audit['semantic_review_sha256'], 'Semantic audit binding differs')
        require(row['roi_provenance_sha256']==audit['technical_audit_sha256'], 'Technical audit binding differs')
        path=safe_path(manifest.parent,row['image_path'])
        require(sha_file(path)==row['image_sha256']==row['rendered_png_sha256'], 'Source PNG hash differs')
        with Image.open(path) as im:
            require(im.mode=='RGB' and list(im.size)==row['pixel_dimensions'], 'Source raster differs')
            box=validate_roi(row['roi_pixels'],im.size)
            require((box[2]-box[0])*(box[3]-box[1])<=.25*im.width*im.height,'ROI exceeds locked area')
    return rows

def preflight_rows():
    return unique(read_jsonl(SOURCE_REPORT/'preflight_selected_metadata.jsonl'))

def model_files(model_dir):
    model_dir=Path(model_dir).resolve()
    names={'lookagain-model.json','config.json','generation_config.json','model.safetensors.index.json',
           'preprocessor_config.json','video_preprocessor_config.json','tokenizer_config.json','chat_template.json','tokenizer.json','vocab.json','merges.txt'}
    index=read_json(model_dir/'model.safetensors.index.json')
    names.update(index['weight_map'].values())
    return {name:sha_file(safe_path(model_dir,name)) for name in sorted(names)}

def configure_runtime(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    import torch
    require(torch.cuda.is_available(),'CUDA required')
    torch.set_num_threads(4)
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    return {'gpu':torch.cuda.get_device_name(0),'cuda':torch.version.cuda,
            'packages':{p:importlib.metadata.version(p) for p in PACKAGES},
            'python':platform.python_version(),'platform':platform.platform(),
            'torch_cpu_threads':torch.get_num_threads(),'cublas_workspace_config':os.environ['CUBLAS_WORKSPACE_CONFIG'],
            'cudnn_benchmark':False,'tf32':False}

def validate_lock(lock_path, manifest, role, model_dir=None):
    _,config=validate_source_design(); lock=read_json(lock_path)
    require(lock['schema_version']==1 and lock['execution_locked'] is True,'Execution lock required')
    require(lock['role']==role and lock['expected_examples']==(660 if role=='main' else 10),'Wrong locked execution role/count')
    require(lock['source_design_lock_sha256']==SOURCE_LOCK_SHA and lock['config_sha256']==CONFIG_SHA,'Wrong source/config lock')
    require(lock['config']==config,'Locked scientific configuration changed')
    require(lock['code_bindings']==code_bindings(),'Execution code/document changed; new engineering verification required')
    require(sha_file(manifest)==lock['manifest_sha256'],'Wrong locked manifest')
    require(lock['expected_records']==lock['expected_examples']*4,'Wrong record target')
    require(isinstance(lock['run_id'],str) and lock['run_id'].strip(),'Missing run identity')
    utc(lock['locked_at_utc'])
    require(lock['verification']['status']=='PASS' and lock['verification']['code_bindings']==lock['code_bindings'],'Unverified execution sources')
    if role=='main':
        require(lock.get('engineering_evidence',{}).get('validated_complete') is True,'Completed engineering verification required')
        require(lock['engineering_evidence']['records']==40,'Engineering coverage incomplete')
        require(lock['engineering_evidence']['code_bindings']==lock['code_bindings'],'Main code differs from engineering code')
    if model_dir is not None:
        meta=read_json(Path(model_dir)/'lookagain-model.json')
        require(meta==lock['model'] and meta['model_id']==config['model_id'] and meta['revision']==config['model_revision'],'Model identity differs')
        require(model_files(model_dir)==lock['model_file_sha256'],'Actual model/processor files differ')
    return lock,config

def make_identity(lock_path, lock, rows):
    return {'execution_lock_sha256':sha_file(lock_path),'run_id':lock['run_id'],'config':lock['config'],
            'manifest_sha256':lock['manifest_sha256'],'code_bindings':lock['code_bindings'],
            'model':lock['model'],'model_file_sha256':lock['model_file_sha256'],'runtime':lock['runtime'],
            'role':lock['role'],'example_ids':[r['example_id'] for r in rows]}

def read_records(path):
    records={}
    if Path(path).exists():
        for row in read_jsonl(path):
            key=(row['example_id'],row['action'])
            require(key not in records,'Duplicate execution record')
            require(row.get('status')=='ok','Failed execution record')
            records[key]=row
    return records

def validate_record(row, record, geometry, expected_processor):
    expected={key:row[key] for key in ('example_id','image_id','source_id','source_cluster_id','question',
                 'original_answers','original_answer_variants','validated_primary_answers')}
    expected.update(target_answer=row['answer'],source_image_sha256=row['image_sha256'],
                    roi_provenance_sha256=row['roi_provenance_sha256'],
                    source_semantic_review_sha256=row['source_semantic_review_sha256'])
    expected.update(geometry)
    expected.update(score_response(record['response'],row['original_answers'],row['validated_primary_answers'],row['original_answer_variants']))
    for key in ('input_tokens','visual_tokens','image_grid_thw'):
        expected[key]=expected_processor[key]
    expected['processor_input_tensor_sha256']=expected_processor['tensor_sha256']
    expected['formatted_prompt_sha256']=expected_processor['formatted_prompt_sha256']
    for key,value in expected.items():
        require(record.get(key)==value,'Record differs in '+key+': '+row['example_id']+'/'+record.get('action','?'))
    for key in ('input_tokens','visual_tokens'):
        require(type(record.get(key)) is int and record[key]>0,'Invalid integer '+key)
    for key in ('parse_valid','correct','previous_answer_in_prompt','roi_used'):
        require(type(record.get(key)) is bool,'Invalid boolean '+key)
    for key in ('primary_em','strict_em','official_anls','anls'):
        require(type(record.get(key)) in (int,float) and math.isfinite(record[key]),'Invalid numeric score '+key)
    require(record.get('status')=='ok','Unsuccessful record')
    require(record.get('answer_prefix_prefilled') is True and record['response']=='ANSWER:'+record['raw_continuation'],'Answer prefix differs')
    require(type(record.get('generated_tokens')) is int and 1<=record['generated_tokens']<=64,'Invalid token count')
    require(type(record.get('generation_truncated')) is bool,'Missing truncation flag')
    require(not record['generation_truncated'] or record['generated_tokens']==64,'Impossible truncated length')
    for key in ('elapsed_s','peak_memory_gib','peak_reserved_gib','processor_observer_elapsed_s'):
        require(type(record.get(key)) in (int,float) and math.isfinite(record[key]) and record[key]>=0,'Invalid measurement '+key)
    require(record['elapsed_s']>0 and record['peak_memory_gib']>0 and record['peak_reserved_gib']>=record['peak_memory_gib'],'Invalid latency/memory')
    require(record['processor_observer_elapsed_s']<=record['elapsed_s'],'Observer exceeds measured call')
    value=record.get('mean_token_logprob')
    require(value is None or type(value) in (int,float) and math.isfinite(value) and value<=1e-5,'Invalid diagnostic log probability')

def validate_saved_records(records, rows, manifest, config):
    from PIL import Image
    pf=preflight_rows(); expected={(r['example_id'],a) for r in rows for a in ACTIONS}
    require(set(records)<=expected,'Unexpected source/action')
    for row in rows:
        actions=[a for a in ACTIONS if (row['example_id'],a) in records]
        if not actions: continue
        with Image.open(safe_path(Path(manifest).parent,row['image_path'])) as raw: source=raw.convert('RGB')
        for action in actions:
            _,images,geometry=build_request(source,row['question'],action,config,row['roi_pixels'])
            p=pf[row['example_id']]['actions'][action]
            require([list(im.size) for im in images]==p['image_sizes'],'Reconstructed sizes differ from CPU preflight')
            require(geometry['image_rgb_sha256']==p['image_rgb_sha256'] and geometry['messages_sha256']==p['messages_sha256'],'Reconstructed inputs differ from source lock')
            validate_record(row,records[(row['example_id'],action)],geometry,p)
    return expected

def validate_run_files(manifest,lock_path,run_dir,role,complete=False):
    manifest,lock_path,run_dir=map(lambda p:Path(p).resolve(),(manifest,lock_path,run_dir))
    lock,config=validate_lock(lock_path,manifest,role); rows=validate_manifest(manifest,role)
    run=read_json(run_dir/'run.json'); identity=make_identity(lock_path,lock,rows)
    require({k:run.get(k) for k in identity}==identity and run.get('fingerprint')==canonical_hash(identity),'Run identity/fingerprint differs')
    require(utc(run['created_utc'])>=utc(lock['locked_at_utc']),'Run precedes lock')
    errors=run_dir/'errors.jsonl'
    require(not errors.exists() or not errors.read_text(encoding='utf-8').strip(),'Errored run requires documented new engineering lock, no silent retries')
    records=read_records(run_dir/'records.jsonl')
    expected=validate_saved_records(records,rows,manifest,config)
    completion=None
    if (run_dir/'completed.json').exists():
        completion=read_json(run_dir/'completed.json')
        require(set(records)==expected and completion['records']==len(expected),'Incomplete paired run')
        require(completion['records_sha256']==sha_file(run_dir/'records.jsonl'),'Completion record hash differs')
        require(completion['fingerprint']==run['fingerprint'],'Completion identity differs')
        require(utc(completion['finished_utc'])>=utc(run['created_utc']),'Completion precedes run')
        require(math.isclose(completion['sum_measured_action_elapsed_s'],sum(r['elapsed_s'] for r in records.values()),rel_tol=1e-12),'Completion latency sum differs')
        require(completion['elapsed_s'] is None or type(completion['elapsed_s']) in (int,float) and math.isfinite(completion['elapsed_s']) and completion['elapsed_s']>0,'Invalid completion elapsed time')
    if complete: require(completion is not None,'Completed run required; interim analysis prohibited')
    return rows,records,run,completion

def validate_completed_run(manifest,lock_path,run_dir,role='main'):
    return validate_run_files(manifest,lock_path,run_dir,role,complete=True)

class ProcessorObserver:
    """Read-only input observer; returns the exact BatchFeature from the processor."""
    def __init__(self,processor): self.processor=processor; self.expected=None; self.last=None
    def __getattr__(self,name): return getattr(self.processor,name)
    def __call__(self,*args,**kwargs):
        import torch
        result=self.processor(*args,**kwargs)
        started=time.perf_counter()
        hashes={k:tensor_hash(v) for k,v in result.items() if isinstance(v,torch.Tensor) and k!='token_type_ids'}
        prompt=kwargs['text'][0]
        measured={'tensor_sha256':hashes,'formatted_prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                  'input_tokens':int(result['input_ids'].shape[-1]),'image_grid_thw':result['image_grid_thw'].tolist()}
        measured['visual_tokens']=sum(math.prod(grid)//4 for grid in measured['image_grid_thw'])
        if self.expected is not None:
            for key,value in measured.items(): require(value==self.expected[key],'Actual processor input differs from preflight: '+key)
        self.last={**measured,'observer_elapsed_s':time.perf_counter()-started}
        return result

def run(manifest,model_dir,lock_path,output,role):
    manifest,model_dir,lock_path,output=map(lambda p:Path(p).resolve(),(manifest,model_dir,lock_path,output))
    lock,config=validate_lock(lock_path,manifest,role,model_dir)
    require(configure_runtime(config['seed'])==lock['runtime'],'Runtime differs from execution lock')
    rows=validate_manifest(manifest,role); identity=make_identity(lock_path,lock,rows)
    output.mkdir(parents=True,exist_ok=True)
    meta_path=output/'run.json'; record_path=output/'records.jsonl'
    if meta_path.exists():
        _,records,_,completion=validate_run_files(manifest,lock_path,output,role)
        if completion is not None:
            print('All locked actions already complete.',flush=True); return
    else:
        require(not any(output.iterdir()),'Nonempty unbound run directory')
        write_json(meta_path,{**identity,'fingerprint':canonical_hash(identity),'created_utc':now(),
            'experiment':'dude_matched_detail_replication','answer_page_privileged':True,'roi_privileged':True,
            'cost_definition':'Synchronized standalone decode, view construction, RGB hashing, processor, actual CPU input observation/hash validation, transfer, generation, decode and diagnostic logprob. Excludes PDF preparation, model loading, warmup and record I/O. Observer time reported separately; active-desktop hardware, no isolated benchmark claim.',
            'warmup':'Six synthetic direct_256/native_256 calls at landscape/square/portrait, two generated tokens each; excluded from experimental counts.',
            'durability':'Flush every action and fsync every source; errors abort, no automatic retries; resume requires full immutable identity and saved-record validation.'})
        records={}
    expected={(r['example_id'],a) for r in rows for a in ACTIONS}
    if set(records)==expected:
        write_json(output/'completed.json',{'records':len(records),'records_sha256':sha_file(record_path),'fingerprint':canonical_hash(identity),
            'elapsed_s':None,'elapsed_s_scope':'Completion recovered from durable records; total wall time unavailable.',
            'sum_measured_action_elapsed_s':sum(r['elapsed_s'] for r in records.values()),'finished_utc':now()})
        return
    import torch
    from PIL import Image
    from lookagain.backend import QwenBackend
    backend=QwenBackend(model_dir,config); observer=ProcessorObserver(backend.processor); backend.processor=observer
    for size in ((2400,1800),(2000,2000),(1800,2400)):
        source=Image.new('RGB',size,(128,128,128))
        for action in ('direct_256','native_256'):
            messages,images,_=build_request(source,'What color is the image?',action,config,[128,128,384,384])
            backend.generate(messages,images,2,answer_prefix=True)
    torch.cuda.synchronize(); started=time.perf_counter(); pf=preflight_rows()
    with record_path.open('a',encoding='utf-8',newline='\n') as stream:
        for index,row in enumerate(rows):
            actions=list(ACTIONS)
            seed=int(hashlib.sha256(f'{config["seed"]}:{row["example_id"]}'.encode()).hexdigest()[:16],16)
            random.Random(seed).shuffle(actions)
            for action in actions:
                key=(row['example_id'],action)
                if key in records: continue
                try:
                    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); action_started=time.perf_counter()
                    with Image.open(safe_path(manifest.parent,row['image_path'])) as raw: source=raw.convert('RGB')
                    messages,images,geometry=build_request(source,row['question'],action,config,row['roi_pixels'])
                    p=pf[row['example_id']]['actions'][action]
                    require(geometry['image_rgb_sha256']==p['image_rgb_sha256'] and geometry['messages_sha256']==p['messages_sha256'],'Request differs before generation')
                    observer.expected=p
                    result=backend.generate(messages,images,64,answer_prefix=True)
                    torch.cuda.synchronize(); elapsed=time.perf_counter()-action_started
                    record={k:row[k] for k in ('example_id','image_id','source_id','source_cluster_id','question',
                                'original_answers','original_answer_variants','validated_primary_answers')}
                    record.update(action=action,status='ok',target_answer=row['answer'],source_image_sha256=row['image_sha256'],
                        roi_provenance_sha256=row['roi_provenance_sha256'],source_semantic_review_sha256=row['source_semantic_review_sha256'],
                        **result,**geometry,**score_response(result['response'],row['original_answers'],row['validated_primary_answers'],row['original_answer_variants']),
                        elapsed_s=elapsed,peak_memory_gib=torch.cuda.max_memory_allocated()/1024**3,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
                        processor_input_tensor_sha256=observer.last['tensor_sha256'],formatted_prompt_sha256=observer.last['formatted_prompt_sha256'],
                        processor_observer_elapsed_s=observer.last['observer_elapsed_s'])
                    validate_record(row,record,geometry,p)
                    stream.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n'); stream.flush(); records[key]=record
                except Exception as error:
                    with (output/'errors.jsonl').open('a',encoding='utf-8') as errors:
                        errors.write(json.dumps({'example_id':row['example_id'],'action':action,'error':repr(error),'utc':now()})+'\n')
                    raise
            os.fsync(stream.fileno())
            print(f'[{index+1}/{len(rows)}] {len(records)}/{len(expected)} actions complete; {(time.perf_counter()-started)/60:.1f} min after warmup',flush=True)
    require(set(records)==expected,'Execution ended incomplete')
    write_json(output/'completed.json',{'records':len(records),'records_sha256':sha_file(record_path),'fingerprint':canonical_hash(identity),
        'elapsed_s':time.perf_counter()-started,'elapsed_s_scope':'Current segment after warmup; on resume excludes earlier segments.',
        'sum_measured_action_elapsed_s':sum(r['elapsed_s'] for r in records.values()),'finished_utc':now()})

def create_lock(manifest,model_dir,output,role,run_id,verification,engineering_run=None,engineering_lock=None):
    require(not Path(output).exists(),'Do not overwrite an execution lock')
    _,config=validate_source_design(); rows=validate_manifest(manifest,role); codes=code_bindings()
    proof=read_json(verification)
    require(proof['status']=='PASS' and proof['code_bindings']==codes,'Tests/review must bind final execution code')
    meta=read_json(Path(model_dir)/'lookagain-model.json')
    require(meta['model_id']==config['model_id'] and meta['revision']==config['model_revision'],'Wrong model')
    lock={'schema_version':1,'execution_locked':True,'role':role,'run_id':run_id,'locked_at_utc':now(),
          'source_design_lock_sha256':SOURCE_LOCK_SHA,'config_sha256':CONFIG_SHA,'config':config,
          'manifest_sha256':sha_file(manifest),'expected_examples':len(rows),'expected_records':len(rows)*4,
          'code_bindings':codes,'model':meta,'model_file_sha256':model_files(model_dir),
          'runtime':configure_runtime(config['seed']),'verification':proof,'verification_sha256':sha_file(verification)}
    if role=='main':
        require(engineering_run is not None and engineering_lock is not None,'Engineering evidence required')
        eng_manifest=Path(manifest).parent/'engineering_manifest.jsonl'
        _,records,eng_run,_=validate_completed_run(eng_manifest,engineering_lock,engineering_run,role='engineering')
        require(eng_run['code_bindings']==codes,'Code changed since engineering; verify again')
        for field in ('model','model_file_sha256','runtime'):
            require(eng_run[field]==lock[field],'Main '+field+' differs from completed engineering')
        require(eng_run['run_id']!=run_id,'Engineering and main require distinct run IDs')
        lock['engineering_evidence']={'validated_complete':True,'records':len(records),'code_bindings':codes,
            'execution_lock_sha256':sha_file(engineering_lock),'run_sha256':sha_file(Path(engineering_run)/'run.json'),
            'records_sha256':sha_file(Path(engineering_run)/'records.jsonl'),'completed_sha256':sha_file(Path(engineering_run)/'completed.json')}
    write_json(output,lock)
    print(json.dumps({'role':role,'execution_lock_sha256':sha_file(output),'examples':len(rows),'generation_calls':0}))

def main():
    parser=argparse.ArgumentParser(description=__doc__); commands=parser.add_subparsers(dest='command',required=True)
    for name in ('lock','run','validate'):
        p=commands.add_parser(name); p.add_argument('--manifest',type=Path,required=True)
        p.add_argument('--role',choices=['main','engineering'],default='main')
        if name=='lock':
            p.add_argument('--model-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
            p.add_argument('--run-id',required=True);p.add_argument('--verification',type=Path,required=True)
            p.add_argument('--engineering-run',type=Path);p.add_argument('--engineering-lock',type=Path)
        else:
            p.add_argument('--lock',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
            if name=='run':p.add_argument('--model-dir',type=Path,required=True)
    a=parser.parse_args()
    if a.command=='lock':create_lock(a.manifest,a.model_dir,a.output,a.role,a.run_id,a.verification,a.engineering_run,a.engineering_lock)
    elif a.command=='run':run(a.manifest,a.model_dir,a.lock,a.output,a.role)
    else:
        rows,records,_,_=validate_completed_run(a.manifest,a.lock,a.output,a.role)
        print(json.dumps({'status':'PASS','sources':len(rows),'records':len(records),'generation_calls':0}))
if __name__=='__main__':main()
