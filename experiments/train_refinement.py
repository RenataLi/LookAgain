"""Commit five-fold predictions from OLD labels before any new main answers."""
from __future__ import annotations
import argparse
import hashlib
import math
from pathlib import Path
import sys
import time
import numpy as np

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'src'),str(PROJECT/'experiments')]
from refinement_execution import (validate_features,load_prepared,code_bindings,CONFIG,
                                  PREPARATION,public_manifest)
from refinement_ranker import fit_ranker,predict_scores,source_folds,training_best_fixed
from dude_replication import require,read_json,write_json,now,utc,safe_path
from check_dude_requests import sha_file
from dude_metrics import score_response


def scientific_decisions(rows,labels,oldrecords,feature_records):
    """No new-answer argument exists. Replaying this function verifies every fit."""
    config=read_json(CONFIG);ids=[r['example_id'] for r in rows]
    require(len(set(ids))==len(rows),'Duplicate rows')
    labelmap={r['example_id']:r for r in labels};old={(r['example_id'],r['action']):r for r in oldrecords}
    fm={r['example_id']:r for r in feature_records}
    require(set(fm)==set(ids)==set(labelmap),'Training inputs disagree')
    X=np.asarray([fm[eid]['projected_features'] for eid in ids],dtype=np.float64)
    Y=[]
    for eid in ids:
        label=labelmap[eid]
        Y.append([score_response(old[eid,f'native_{j}']['response'],label['original_answers'],
            label['validated_primary_answers'],label['original_answer_variants'])['primary_em'] for j in range(1,10)])
    Y=np.asarray(Y,dtype=np.float64)
    folds=source_folds([r['source_cluster_id'] for r in rows])
    models=[];predictions=[None]*len(rows);training_start=time.perf_counter()
    for fold in range(5):
        train=np.flatnonzero(folds!=fold).tolist();test=np.flatnonzero(folds==fold).tolist()
        fitted={kind:fit_ranker(X,Y,train,kind=kind,alpha=config['ridge_alpha']) for kind in config['ranker_kinds']}
        fixed=training_best_fixed(Y,train)
        models.append({'fold':fold,'training_example_ids':[ids[i] for i in train],
            'training_source_cluster_ids':[rows[i]['source_cluster_id'] for i in train],
            'test_example_ids':[ids[i] for i in test],'best_fixed':int(fixed),'rankers':fitted})
        for i in test:
            row=rows[i];eid=ids[i];regions={};scores={};timings={}
            for kind in config['ranker_kinds']:
                start=time.perf_counter();values=np.asarray(predict_scores(fitted[kind],X[i:i+1]))[0]
                region=int(np.argmax(values))+1;elapsed=time.perf_counter()-start
                regions[kind]=region;scores[kind]=values.tolist();timings[kind]=elapsed
            start=time.perf_counter();regions['best_fixed']=int(fixed);timings['best_fixed']=time.perf_counter()-start
            start=time.perf_counter();regions['center']=5;timings['center']=time.perf_counter()-start
            start=time.perf_counter()
            digest=hashlib.sha256(f"refinement-random:{config['seed']}:{eid}".encode()).digest()
            regions['random']=int.from_bytes(digest,'big')%9+1
            timings['random']=time.perf_counter()-start
            regions['prompted']=old[eid,'selector']['selector_region']
            predictions[i]={'example_id':eid,'source_cluster_id':row['source_cluster_id'],'fold':int(fold),
                            'regions':regions,'scores':scores,'cpu_elapsed_s':timings}
    final={kind:fit_ranker(X,Y,list(range(len(rows))),kind=kind,alpha=config['ridge_alpha']) for kind in config['ranker_kinds']}
    return {'predictions':predictions,'fold_models':models,'deployment_full_data_models':final,
            'old_targets':[{'example_id':eid,'native_em':Y[i].tolist()} for i,eid in enumerate(ids)],
            'training_elapsed_s':time.perf_counter()-training_start,
            'sources_with_nonconstant_old_native_grades':int(np.sum(np.ptp(Y,axis=1)>0)),
            'deployment_model_has_independent_performance_estimate':False}


def strip_timings(value):
    if isinstance(value,dict):return {k:strip_timings(v) for k,v in value.items() if k not in ('cpu_elapsed_s','training_elapsed_s')}
    if isinstance(value,list):return [strip_timings(v) for v in value]
    return value


def train(args):
    rows,features,run,lock=validate_features(args.manifest,args.features_lock,args.features_run)
    require(lock['role']=='main','OOF needs main development features')
    prepared,labels,oldrecords=load_prepared('main');require(rows==prepared,'Prepared order differs')
    result=scientific_decisions(rows,labels,oldrecords,features)
    require(not args.output.exists(),'Use a new decisions artifact')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result.update(schema_version=1,created_utc=now(),code_bindings=code_bindings(),config_sha256=sha_file(CONFIG),
        reservation_sha256=sha_file(PREPARATION/'reservation.json'),
        old_labels_sha256=sha_file(PREPARATION/'main_labels.jsonl'),training_new_answers_loaded=False,
        main_is_reused_development_data=True,features_lock_sha256=sha_file(args.features_lock),
        features_records_sha256=sha_file(args.features_run/'records.jsonl'),
        features_completed_sha256=sha_file(args.features_run/'completed.json'))
    write_json(args.output,result)
    validate_decisions(args.output,args.features_lock,args.features_run,args.manifest)
    print({'decisions_committed':len(rows),'nonconstant_old_sources':result['sources_with_nonconstant_old_native_grades'],
           'decisions_sha256':sha_file(args.output)})


def validate_decisions(path,features_lock,features_run,manifest=None):
    manifest=Path(manifest) if manifest is not None else public_manifest('main')
    rows,features,run,lock=validate_features(manifest,features_lock,features_run)
    require(lock['role']=='main','Wrong feature scope')
    value=read_json(path)
    require(value['code_bindings']==code_bindings() and value['config_sha256']==sha_file(CONFIG),'Decision code/config changed')
    require(value['reservation_sha256']==sha_file(PREPARATION/'reservation.json'),'Decision reservation changed')
    require(value['old_labels_sha256']==sha_file(PREPARATION/'main_labels.jsonl'),'Decision labels changed')
    require(value['training_new_answers_loaded'] is False and value['main_is_reused_development_data'] is True,'Wrong training scope')
    require(value['features_lock_sha256']==sha_file(features_lock),'Decision features lock differs')
    require(value['features_records_sha256']==sha_file(features_run/'records.jsonl'),'Decision feature records differ')
    require(value['features_completed_sha256']==sha_file(features_run/'completed.json'),'Decision completion differs')
    require(utc(value['created_utc'])>=utc(read_json(features_run/'completed.json')['finished_utc']),'Decisions predate complete features')
    _,labels,oldrecords=load_prepared('main');expected=scientific_decisions(rows,labels,oldrecords,features)
    for key in expected:
        if key!='training_elapsed_s':require(strip_timings(value[key])==strip_timings(expected[key]),'Decision replay differs: '+key)
    for row in value['predictions']:
        require(set(row['cpu_elapsed_s'])=={'full','image_position','position','best_fixed','center','random'},'CPU timing coverage differs')
        require(all(type(v) in (float,int) and math.isfinite(v) and v>=0 for v in row['cpu_elapsed_s'].values()),'Invalid CPU selection cost')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('manifest','features-lock','features-run','output'):p.add_argument('--'+key,type=Path,required=True)
    train(p.parse_args())
