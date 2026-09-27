"""Reserve only the 124 already-exposed sources; copy exact existing rasters."""
from __future__ import annotations
import argparse
from dataclasses import asdict
from pathlib import Path
import shutil
import sys
import json

PROJECT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PROJECT/'src'),str(PROJECT/'experiments')]
from dude_replication import read_jsonl, write_json, now, require, safe_path
from check_dude_requests import sha_file
from region_selection_core import Observation


def prepare(previous_data, output):
    report=PROJECT/'reports/refinement124/preparation'
    require(not report.exists() and not output.exists(),'Use a new preparation path')
    report.mkdir(parents=True); output.mkdir(parents=True)
    old=PROJECT/'reports/region_selection_preparation'
    cohorts={}; previous={}; source_sets=[]
    for role,upstreams in [('engineering',['engineering']),('main',['development','evaluation'])]:
        folder=output/role; (folder/'images').mkdir(parents=True)
        rows=[];labels=[];origins=[]
        for upstream in upstreams:
            mp=old/(upstream+'_observation_manifest.jsonl');lp=old/(upstream+'_labels.jsonl')
            previous[mp.relative_to(PROJECT).as_posix()]=sha_file(mp)
            previous[lp.relative_to(PROJECT).as_posix()]=sha_file(lp)
            oldrows=read_jsonl(mp);oldlabels=read_jsonl(lp)
            require([r['example_id'] for r in oldrows]==[r['example_id'] for r in oldlabels],'Old panel order changed')
            for row in oldrows:
                Observation.from_manifest(row)
                src=safe_path(previous_data/upstream,row['image_path'])
                require(sha_file(src)==row['image_sha256'],'Old raster differs')
                dest=safe_path(folder,row['image_path']);shutil.copyfile(src,dest)
                rows.append(row);origins.append({'example_id':row['example_id'],'previous_role':upstream})
            labels.extend(oldlabels)
        require(len(rows)==(2 if role=='engineering' else 124),'Wrong reserved count')
        require(len({r['example_id'] for r in rows})==len(rows),'Duplicate example')
        sources={r['source_cluster_id'] for r in rows};require(len(sources)==len(rows),'Duplicate cluster')
        source_sets.append(sources)
        for name,values in [('observation_manifest',rows),('labels',labels)]:
            text=''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in values)
            (folder/(name+'.jsonl')).write_text(text,encoding='utf-8',newline='\n')
            (report/(role+'_'+name+'.jsonl')).write_text(text,encoding='utf-8',newline='\n')
        cohorts[role]={'example_ids':[r['example_id'] for r in rows],
                      'source_cluster_ids':[r['source_cluster_id'] for r in rows],
                      'observation_manifest_sha256':sha_file(folder/'observation_manifest.jsonl'),
                      'labels_sha256':sha_file(folder/'labels.jsonl'),'origins':origins}
    require(source_sets[0].isdisjoint(source_sets[1]),'Engineering/main overlap')
    oldrun=PROJECT/'reports/region_selection60'
    for role in ['engineering','development','evaluation']:
        for p in [oldrun/(role+'_lock.json'),*(oldrun/'raw_runs'/role).glob('*.json*')]:
            previous[p.relative_to(PROJECT).as_posix()]=sha_file(p)
    write_json(report/'reservation.json',{'schema_version':1,'reserved_at_utc':now(),
        'cohorts':cohorts,'upstream_bindings':previous,'new_holdout_sources':0,
        'main_is_reused_development_data':True,'engineering_excluded_from_training':True,
        'answer_page_privileged':True,'roi_annotations_available_at_inference':False})
    print(json.dumps({'reserved_sources':126,'training_development_sources':124,'new_holdout_sources':0}))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--previous-data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();prepare(a.previous_data,a.output)
