"""Independent post-freeze DUDE result audit; no model loading or generation.

Does not import the runner/analyzer or their statistical/scoring functions.
Only the explicitly frozen response parser is shared. A real-run invocation
requires a caller-supplied blinded-review freeze artifact and completed run.
This supplemental post-lock checker does not change the frozen primary analysis.
Until separately authorized, use --self-test only.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys

import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
SOURCE=PROJECT/'reports/dude_preparation'
sys.path.insert(0,str(PROJECT/'src'))
from lookagain.actions import extract_answer

ACTIONS=('direct_256','native_256','degraded_256','highres')
FIELDS=('elapsed_s','processor_observer_elapsed_s','peak_memory_gib','peak_reserved_gib',
        'input_tokens','visual_tokens','generated_tokens')
PAIRS=(('native_256','degraded_256'),('native_256','direct_256'),
       ('degraded_256','direct_256'),('highres','direct_256'),
       ('native_256','highres'),('degraded_256','highres'))
PRIMARY='native_256_minus_degraded_256'
SOURCE_SHA='c684b0b59af32e68dd30d72e04b3d61b1d28f16e78d250d8a983021316cb86fd'
CONFIG_SHA='d174347176b24fe00d5a4c7fcce069179429092132cd371f421eb69f02b046b5'
PARSER_SHA='3277f7c95d26d7e45fd2ed889dafc3c9b507e675456644e25a8f19d6d6bbcd41'
SAMPLES=10000
SEED=20260927

def require(value,message):
    if not value: raise ValueError(message)

def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()

def reject_constant(value):raise ValueError('Nonfinite JSON: '+value)
def read(path):return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=reject_constant)
def lines(path):return [json.loads(x,parse_constant=reject_constant) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]
def canonical(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
def utc(value):
    result=datetime.fromisoformat(value.replace('Z','+00:00'))
    require(result.tzinfo is not None,'Naive timestamp')
    return result
def safe(root,relative):
    p=(Path(root)/relative).resolve()
    require(p.is_relative_to(Path(root).resolve()),'Path escapes artifact root')
    return p
def normalize(text):return ' '.join(text.casefold().split())

def edit_distance(first,second):
    previous=list(range(len(second)+1))
    for i,a in enumerate(first,1):
        current=[i]
        for j,b in enumerate(second,1):current.append(min(current[-1]+1,previous[j]+1,previous[j-1]+(a!=b)))
        previous=current
    return previous[-1]

def rescore(response,row):
    original=row['original_answers'];variants=row['original_answer_variants'];valid=row['validated_primary_answers']
    require(len(original)==1 and original[0]==row['answer'] and original[0] in valid,'Canonical reference is missing or changed')
    require(bool(valid) and all(isinstance(x,str) and x.strip() and x in original+variants for x in valid),'Unapproved invented reference')
    prediction=extract_answer(response)
    em=int(prediction is not None and normalize(prediction) in {normalize(x) for x in valid})
    if prediction is None:anls=0.
    else:
        # Independent scalar reproduction of the pinned upstream semantics:
        # normalized edit distance, RAW uppercase-length denominator, retain .5.
        distances=[]
        for gold in original:
            a=' '.join(gold.strip().lower().split());b=' '.join(prediction.strip().lower().split())
            denominator=max(len(gold.upper()),len(prediction.upper()))
            distances.append(0. if denominator==0 else edit_distance(a,b)/denominator)
        similarity=1-min(distances);anls=0. if similarity<.5 else similarity
    return {'predicted_answer':prediction,'parse_valid':prediction is not None,
            'normalized_prediction':normalize(prediction) if prediction is not None else None,
            'primary_em':em,'strict_em':em,'correct':bool(em),'official_anls':anls,'anls':anls}

def exact_binomial(fixes,harms):
    """Conditional two-sided exact test, independent recurrence (not comb sum)."""
    n=fixes+harms
    require(type(fixes) is int and type(harms) is int and min(fixes,harms)>=0,'Invalid discordances')
    if n==0:return 1.
    probability=math.ldexp(1.,-n);terms=[probability]
    for k in range(1,min(fixes,harms)+1):
        probability*=((n-k+1)/k);terms.append(probability)
    return min(1.,2*math.fsum(terms))

def distribution(values):
    return {'n':len(values),'mean':math.fsum(values)/len(values),'median':statistics.median(values),
            'min':min(values),'p95':float(np.quantile(values,.95,method='linear')),'max':max(values)}

def compute(rows,records,samples=SAMPLES,seed=SEED):
    """Independent numerical results; caller separately establishes provenance."""
    n=len(rows);ids=[x['example_id'] for x in rows]
    require(n>0 and len(set(ids))==n and len({x['source_cluster_id'] for x in rows})==n,'Invalid source denominator')
    require(set(records)=={(eid,a) for eid in ids for a in ACTIONS},'Incomplete/extra four-action table')
    scored={key:rescore(record['response'],rows[ids.index(key[0])]) for key,record in records.items()}
    vectors={metric:{a:[scored[eid,a][metric] for eid in ids] for a in ACTIONS} for metric in ('primary_em','official_anls')}
    rng=random.Random(seed)
    draws=np.fromiter((rng.randrange(n) for _ in range(samples*n)),dtype=np.int64,count=samples*n).reshape(samples,n)
    def interval(values):
        delta=np.asarray(values,dtype=float)
        means=delta[draws].sum(axis=1)/n
        ci=np.quantile(means,[.025,.975],method='linear').tolist();mean=math.fsum(values)/n
        return {'n_source_clusters':n,'difference':mean,'difference_pp':100*mean,'ci95':ci,
                'ci95_pp':[100*x for x in ci],'degenerate_empirical_interval':ci[0]==ci[1]}
    actions={}
    for a in ACTIONS:
        selected=[records[eid,a] for eid in ids]
        costs={mode:{} for mode in ('standalone','decision_state')}
        sums={mode:{} for mode in costs}
        for mode in costs:
            for field in FIELDS:
                values=[]
                for eid in ids:
                    value=records[eid,a][field]
                    if mode=='decision_state' and a in ('native_256','degraded_256'):
                        direct=records[eid,'direct_256'][field]
                        value=max(value,direct) if field.startswith('peak_') else value+direct
                    values.append(value)
                costs[mode][field]=distribution(values);sums[mode][field]=math.fsum(values)
        actions[a]={'n_source_clusters':n,'n_invocations':n,'successful_records':n,'failed_records':0,
                    'invalid_answers':sum(not scored[eid,a]['parse_valid'] for eid in ids),
                    'truncated_generations':sum(r['generation_truncated'] for r in selected),
                    'valid_truncated_answers':sum(r['generation_truncated'] and scored[eid,a]['parse_valid'] for eid,r in zip(ids,selected)),
                    'primary_correct':sum(vectors['primary_em'][a]),
                    'metrics':{m:math.fsum(vectors[m][a])/n for m in vectors},
                    'metric_ci95':{m:interval(vectors[m][a])['ci95'] for m in vectors},
                    'costs':costs,'independent_cost_sums':sums,
                    'observations':{f:[r[f] for r in selected] for f in FIELDS}}
    contrasts={}
    for positive,negative in PAIRS:
        after=vectors['primary_em'][positive];before=vectors['primary_em'][negative]
        cells=Counter(zip(before,after));fixes=cells[0,1];harms=cells[1,0]
        correct=cells[1,0]+cells[1,1];wrong=n-correct
        contrasts[positive+'_minus_'+negative]={'positive':positive,'negative':negative,
            'metrics':{m:interval([a-b for a,b in zip(vectors[m][positive],vectors[m][negative])]) for m in vectors},
            'primary_em_transitions':{'metric':'primary_em','positive':positive,'negative':negative,
                'n_source_clusters':n,'fixes':fixes,'harms':harms,'negative_correct':correct,'negative_wrong':wrong,
                'correct_to_correct':cells[1,1],'wrong_to_wrong':cells[0,0],
                'fix_rate_among_negative_wrong':fixes/wrong if wrong else None,
                'harm_rate_among_negative_correct':harms/correct if correct else None,
                'fix_fraction_all_sources':fixes/n,'harm_fraction_all_sources':harms/n}}
    primary=contrasts[PRIMARY]['metrics']['primary_em'];transitions=contrasts[PRIMARY]['primary_em_transitions']
    fixes,harms=transitions['fixes'],transitions['harms'];p=exact_binomial(fixes,harms)
    test={'native_only_correct':fixes,'degraded_only_correct':harms,'discordant_pairs':fixes+harms,
          'p_two_sided':p,'alpha':.05,'positive_direction':fixes>harms,'positive_direction_rejection':fixes>harms and p<=.05}
    checks={'mean_difference_at_least_0_02':primary['difference']>=.02,'exact_two_sided_p_at_most_0_05':p<=.05,
            'paired_ci95_lower_positive':primary['ci95'][0]>0,
            'native_mean_at_least_direct_mean':actions['native_256']['metrics']['primary_em']>=actions['direct_256']['metrics']['primary_em']}
    required=[eid for eid in ids if scored[eid,'native_256']['predicted_answer']!=scored[eid,'degraded_256']['predicted_answer'] or
              (records[eid,'native_256']['response']!=records[eid,'degraded_256']['response'] and
               not(scored[eid,'native_256']['parse_valid'] and scored[eid,'degraded_256']['parse_valid']))]
    result={'n_source_clusters':n,'n_records':len(records),'example_ids':ids,'source_cluster_ids':[r['source_cluster_id'] for r in rows],
            'actions':actions,'contrasts':contrasts,'primary_metric':'primary_em','primary_contrast':PRIMARY,
            'primary_result':{**primary,'exact_mcnemar':test},'per_source_action_scores':vectors,
            'quantitative_advancement':{'checks':checks,'positive_observed_direction':primary['difference']>0,
                'quantitative_rule_met':all(checks.values()) and primary['difference']>0,'automatic_controller_training':False,'controller_ready':False},
            'native_degraded_response_agreement':{'semantic_audit_required_ids':required},
            'standalone_measured_action_time_sum_s':math.fsum(r['elapsed_s'] for r in records.values()),
            'bootstrap':{'samples':samples,'seed':seed,'unit':'source_cluster','n_source_clusters':n,'all_four_actions_paired':True}}
    return result,scored

def compare(expected,actual,path='summary',failures=None):
    """Check every independently recomputed field; tolerate roundoff only."""
    failures=[] if failures is None else failures
    if isinstance(expected,dict):
        if not isinstance(actual,dict):failures.append(path+': expected object');return failures
        for key,value in expected.items():
            if key=='independent_cost_sums':continue
            if key not in actual:failures.append(path+'.'+key+': missing')
            else:compare(value,actual[key],path+'.'+key,failures)
    elif isinstance(expected,list):
        if not isinstance(actual,list) or len(expected)!=len(actual):failures.append(path+': length/type differs')
        else:
            for i,(left,right) in enumerate(zip(expected,actual)):compare(left,right,f'{path}[{i}]',failures)
    elif type(expected) in (int,float):
        if type(actual) not in (int,float) or not math.isfinite(actual) or not math.isclose(expected,actual,rel_tol=1e-11,abs_tol=1e-14):failures.append(path+': numerical difference')
    elif expected!=actual or type(expected) is not type(actual):failures.append(path+': value/type differs')
    return failures

def verify(args):
    # Refuse incomplete artifacts before reading any model response/statistic.
    run_dir=args.run.resolve();summary_path=args.summary.resolve()
    require(args.blind_freeze.is_file(),'Missing separately authorized blinded-review freeze artifact')
    require((run_dir/'completed.json').is_file(),'Run completion missing; no interim audit')
    require(summary_path.is_file(),'Completed analysis summary missing')
    source_lock=read(SOURCE/'source_design_lock.json')
    require(sha(SOURCE/'source_design_lock.json')==SOURCE_SHA,'Frozen source lock changed')
    for relative,value in source_lock['public_file_bindings'].items():require(sha(safe(PROJECT,relative))==value,'Source binding changed: '+relative)
    require(sha(PROJECT/'src/lookagain/actions.py')==PARSER_SHA,'Frozen parser changed')
    require(sha(PROJECT/'configs/dude_replication.json')==CONFIG_SHA,'Scientific config changed')
    manifest=args.manifest.resolve();engineering=args.engineering_manifest.resolve()
    require(sha(manifest)==sha(SOURCE/'main_manifest.jsonl'),'Main manifest changed')
    require(sha(engineering)==sha(SOURCE/'engineering_manifest.jsonl'),'Engineering manifest changed')
    rows=lines(manifest);eng=lines(engineering);ids=[r['example_id'] for r in rows]
    clusters=[r['source_cluster_id'] for r in rows]
    require(len(rows)==660 and len(set(ids))==len(set(clusters))==660,'Main source denominator is not 660')
    require(len(eng)==10,'Engineering source denominator differs')
    require(set(ids).isdisjoint(x['example_id'] for x in eng),'Engineering example entered main')
    require(set(clusters).isdisjoint(source_lock['reserved_engineering_source_clusters']),'Reserved engineering cluster entered main')
    lock=read(args.lock);run=read(run_dir/'run.json');completion=read(run_dir/'completed.json')
    require(lock['role']==run['role']=='main' and lock['execution_locked'] is True,'Missing main execution authority')
    require(lock['expected_examples']==660 and lock['expected_records']==2640,'Execution target changed')
    require(lock.get('engineering_evidence',{}).get('validated_complete') is True and lock['engineering_evidence']['records']==40,'Missing completed engineering prerequisite')
    require(lock['engineering_evidence']['code_bindings']==lock['code_bindings'],'Engineering/main code differs')
    require(lock['source_design_lock_sha256']==SOURCE_SHA and lock['config_sha256']==CONFIG_SHA,'Source/scientific lock differs')
    require(lock['manifest_sha256']==sha(manifest) and lock['config']==read(PROJECT/'configs/dude_replication.json'),'Execution manifest/config differs')
    for relative,value in lock['code_bindings'].items():require(sha(safe(PROJECT,relative))==value,'Execution source changed: '+relative)
    identity={'execution_lock_sha256':sha(args.lock),'run_id':lock['run_id'],'config':lock['config'],
              'manifest_sha256':lock['manifest_sha256'],'code_bindings':lock['code_bindings'],'model':lock['model'],
              'model_file_sha256':lock['model_file_sha256'],'runtime':lock['runtime'],'role':'main','example_ids':ids}
    require({key:run.get(key) for key in identity}==identity and run['fingerprint']==canonical(identity),'Run identity differs')
    require(utc(completion['finished_utc'])>=utc(run['created_utc'])>=utc(lock['locked_at_utc']),'Invalid lock/run chronology')
    require(completion['fingerprint']==run['fingerprint'] and completion['records']==2640,'Completion identity/count differs')
    records_path=run_dir/'records.jsonl'
    require(completion['records_sha256']==sha(records_path),'Completion raw hash differs')
    errors=run_dir/'errors.jsonl';require(not errors.exists() or not errors.read_text(encoding='utf-8').strip(),'Execution errors present')
    raw=lines(records_path);records={(r['example_id'],r['action']):r for r in raw}
    require(len(raw)==len(records)==2640,'Duplicate or incomplete record count')
    require(set(records)=={(eid,a) for eid in ids for a in ACTIONS},'Missing/unexpected main action')
    pf={r['example_id']:r for r in lines(SOURCE/'preflight_selected_metadata.jsonl')}
    reviews={r['example_id']:r for r in lines(SOURCE/'source_review_ledger.jsonl')}
    for row in rows:
        eid=row['example_id'];review=reviews[eid]
        require(row['cohort']=='main' and row['source_semantic_audit_status']=='approved' and review['semantic_approved'] is True,'Unapproved main source')
        require(canonical(review)==row['source_semantic_review_sha256'] and review['source_audit_sha256']==row['roi_provenance_sha256'],'Source-review binding differs')
        require(sha(safe(manifest.parent,row['image_path']))==row['image_sha256']==pf[eid]['source_image_sha256'],'Source raster hash differs')
        for action in ACTIONS:
            record=records[eid,action];p=pf[eid]['actions'][action]
            require(record['status']=='ok','Unsuccessful record')
            for key in ('example_id','image_id','source_id','source_cluster_id','question','original_answers','original_answer_variants','validated_primary_answers','roi_provenance_sha256','source_semantic_review_sha256'):
                require(record[key]==row[key],'Source/reference mismatch: '+key)
            require(record['target_answer']==row['answer'] and record['source_image_sha256']==row['image_sha256'],'Wrong target/source pixels')
            for key in ('image_rgb_sha256','messages_sha256','input_tokens','visual_tokens','image_grid_thw','formatted_prompt_sha256'):
                require(record[key]==p[key],'Actual request/preflight differs: '+key)
            require(record['processor_input_tensor_sha256']==p['tensor_sha256'],'Actual input tensors differ')
            require(record['source_size']==row['pixel_dimensions'] and record['overview_size']==p['image_sizes'][0],'Source/overview geometry differs')
            require(record['source_roi_pixels']==(row['roi_pixels'] if action in ('native_256','degraded_256') else None),'ROI differs')
            require(record['previous_answer_in_prompt'] is False and record['history_mode']=='fresh','Unexpected answer history')
            require(record['response']=='ANSWER:'+record['raw_continuation'] and record['answer_prefix_prefilled'] is True,'Raw/prefix differs')
            for field in FIELDS:require(type(record[field]) in (int,float) and math.isfinite(record[field]) and record[field]>=0,'Invalid measurement')
            require(record['elapsed_s']>0 and record['peak_reserved_gib']>=record['peak_memory_gib']>0,'Invalid elapsed/memory')
            require(0<=record['processor_observer_elapsed_s']<=record['elapsed_s'],'Observer exceeds call')
            require(all(type(record[k]) is int and record[k]>0 for k in ('input_tokens','visual_tokens','generated_tokens')) and record['generated_tokens']<=64,'Invalid token counts')
            require(type(record['generation_truncated']) is bool and (not record['generation_truncated'] or record['generated_tokens']==64),'Invalid truncation')
    recomputed,scored=compute(rows,records)
    grade_failures=[]
    for key,score in scored.items():compare(score,records[key],f'record[{key[0]}/{key[1]}]',grade_failures)
    require(not grade_failures,'Independent grade mismatch: '+str(grade_failures[:5]))
    require(math.isclose(completion['sum_measured_action_elapsed_s'],recomputed['standalone_measured_action_time_sum_s'],rel_tol=1e-12),'Completion measured sum differs')
    summary=read(summary_path)
    provenance={'manifest_sha256':sha(manifest),'execution_lock_sha256':sha(args.lock),'run_json_sha256':sha(run_dir/'run.json'),
                'records_sha256':sha(records_path),'completed_sha256':sha(run_dir/'completed.json'),
                'analysis_script_sha256':sha(PROJECT/'experiments/analyze_dude_replication.py'),
                'completion_validator_sha256':sha(PROJECT/'experiments/dude_replication.py'),
                'metric_adapter_sha256':sha(PROJECT/'experiments/dude_metrics.py'),'errors_sha256':sha(errors) if errors.exists() else None,'numpy_version':np.__version__}
    failures=compare(recomputed,summary)
    compare({'status':'completed_primary_replication','role':'main','fingerprint':run['fingerprint'],'completion':completion,'provenance':provenance},summary,failures=failures)
    for relative,value in summary.get('figures',{}).get('files',{}).items():
        require(sha(safe(summary_path.parent,relative))==value,'Figure digest differs')
    return {'status':'PASS' if not failures else 'FAIL','created_utc':datetime.now(timezone.utc).isoformat(),
            'audit_script_sha256':sha(__file__),'method':'Independent raw response rescoring and source-paired numerical reconstruction. Shared frozen parser only; runner/analyzer not imported.',
            'bindings':{**provenance,'source_design_lock_sha256':SOURCE_SHA,'summary_sha256':sha(summary_path),
                        'engineering_manifest_sha256':sha(engineering),'blind_freeze_sha256':sha(args.blind_freeze)},
            'counts':{'main_source_clusters':660,'records':2640,'engineering_clusters_excluded':len(source_lock['reserved_engineering_source_clusters']),
                      'invalid_answers':{a:recomputed['actions'][a]['invalid_answers'] for a in ACTIONS}},
            'recomputed':recomputed,'failures':failures,'new_model_calls':0,
            'limitations':['Caller must authorize this audit only after the actual condition-blinded review freeze; its supplied artifact is hash-bound, not re-adjudicated here.',
                           'This verifies automatic statistics and source/request identity, not semantic judgments or model internals.',
                           'The four quantitative criteria do not alone establish full advancement or a deployable selector.',
                           'Provided answer page and ROI remain annotation-privileged; residual source dependence/pretraining exposure remain unknown.']}

def self_test():
    rows=[];records={};native=[1,1,1,1,1,0,0,0];degraded=[1,0,0,1,0,1,0,0]
    for i in range(8):
        row={'example_id':f'synthetic-{i}','source_cluster_id':f'source-{i}','answer':'2019',
             'original_answers':['2019'],'original_answer_variants':[' 2019 '],'validated_primary_answers':['2019']};rows.append(row)
        for j,a in enumerate(ACTIONS):
            correct={'direct_256':i<4,'native_256':native[i],'degraded_256':degraded[i],'highres':i<6}[a]
            records[row['example_id'],a]={'response':'ANSWER: '+('2019' if correct else '2020'),
                'generation_truncated':False,'elapsed_s':float(j+2),'processor_observer_elapsed_s':.01,
                'peak_memory_gib':10. if a=='direct_256' else 8.,'peak_reserved_gib':11.,
                'input_tokens':100,'visual_tokens':64,'generated_tokens':3}
    expected,scored=compute(rows,records)
    p=expected['primary_result'];t=expected['contrasts'][PRIMARY]['primary_em_transitions']
    assert (p['difference'],t['fixes'],t['harms'],p['exact_mcnemar']['p_two_sided'])==(.25,3,1,.625)
    assert expected['standalone_measured_action_time_sum_s']==112
    assert expected['actions']['native_256']['costs']['decision_state']['elapsed_s']['mean']==5.
    assert expected['actions']['native_256']['costs']['decision_state']['peak_memory_gib']['mean']==10.
    assert expected['quantitative_advancement']['quantitative_rule_met'] is False
    assert exact_binomial(6,0)==.03125 and exact_binomial(0,0)==1 and exact_binomial(3,3)==1
    assert rescore('ANSWER: 2019.',rows[0])['primary_em']==0
    assert rescore('ANSWER:',rows[0])['parse_valid'] is False
    assert rescore('ANSWER:  2019  ',rows[0])['primary_em']==1
    assert not compare(expected,expected)
    changed=json.loads(json.dumps(expected));changed['primary_result']['difference']+=.01
    assert compare(expected,changed)
    # Independently materialize the specified RNG stream to check draw ordering.
    rng=random.Random(SEED);delta=np.asarray([a-b for a,b in zip(native,degraded)])
    values=[sum(delta[rng.randrange(8)] for _ in range(8))/8 for _ in range(SAMPLES)]
    assert np.quantile(values,[.025,.975],method='linear').tolist()==p['ci95']
    try:compute(rows,{k:v for k,v in records.items() if k!=('synthetic-0','highres')})
    except ValueError:pass
    else:raise AssertionError('Incomplete paired data accepted')
    return {'status':'PASS','synthetic_only':True,'main_outcomes_read':False,'samples':SAMPLES,'seed':SEED,
            'checks':['hand-counted EM/discordance/exact p','zero and extreme exact binomial cases','shared fixed-seed bootstrap order',
                      'standalone and decision-state sum/max cost semantics','strict punctuation/invalid parsing',
                      'summary tamper detection','incomplete-pair rejection'],'new_model_calls':0}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test',action='store_true')
    for name in ('manifest','engineering-manifest','lock','run','summary','blind-freeze','output'):parser.add_argument('--'+name,type=Path)
    args=parser.parse_args()
    if args.self_test:
        result=self_test();result['audit_script_sha256']=sha(__file__)
    else:
        require(all(getattr(args,name) is not None for name in ('manifest','engineering_manifest','lock','run','summary','blind_freeze','output')),'All explicit real-run paths including blinded freeze are required')
        try:result=verify(args)
        except (ValueError,KeyError,TypeError,OSError,IndexError) as error:
            result={'status':'FAIL','created_utc':datetime.now(timezone.utc).isoformat(),
                    'audit_script_sha256':sha(__file__),'failure_type':type(error).__name__,
                    'failure':str(error),'recomputation_completed':False,'new_model_calls':0}
    if args.output:
        require(not args.output.exists(),'Output directory already exists; choose a new path')
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('status','new_model_calls') if k in result}))
    if result['status']!='PASS':raise SystemExit(1)

if __name__=='__main__':main()
