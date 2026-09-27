"""Deterministic unblinded post-run case selection; no grade changes or models.

Require the complete main run AND its bound analysis before accessing answers.
Public output is metadata only. Source page/crop views stay under workspace work/.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

from PIL import Image

PROJECT=Path(__file__).resolve().parents[1]
WORK=PROJECT.parent.parent/'work'
sys.path[:0]=[str(PROJECT/'experiments'),str(PROJECT/'src')]
CATEGORY_QUOTAS=(('prompt_gain',2),('prompt_harm',2),('ranker_gain',1),('ranker_harm',1))


def require(condition,message):
    if not condition:raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def read(path):
    def reject(value):raise ValueError('Nonfinite JSON: '+value)
    return json.loads(Path(path).read_text(encoding='utf-8'),parse_constant=reject)


def canonical_records_sha(records):
    ordered=sorted(records,key=lambda r:(r['example_id'],r.get('action','')))
    return hashlib.sha256(json.dumps(ordered,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def source_order(row):
    return hashlib.sha256(('refinement-case:'+row['source_cluster_id']).encode()).hexdigest(),row['example_id']


def select_cases(per_source):
    """Fixed category order, source-hash order, global dedup, no replacement."""
    require(len({r['example_id'] for r in per_source})==len(per_source),'Duplicate example IDs')
    require(len({r['source_cluster_id'] for r in per_source})==len(per_source),'Duplicate source clusters')
    def qualifies(row,category):
        s=row['scores']
        old=s['prompted_old']['primary_em'];new=s['prompted_new']['primary_em']
        full=s['ranker_full_new']['primary_em']
        require(all(type(v) in (int,float) and v in (0,1) for v in (old,new,full)),'Expected binary primary grades')
        return {'prompt_gain':old==0 and new==1,'prompt_harm':old==1 and new==0,
                'ranker_gain':new==0 and full==1,'ranker_harm':new==1 and full==0}[category]
    chosen=[];used=set();categories=[]
    for category,quota in CATEGORY_QUOTAS:
        eligible=sorted((r for r in per_source if qualifies(r,category)),key=source_order)
        available=[r for r in eligible if r['source_cluster_id'] not in used]
        picked=available[:quota]
        categories.append({'category':category,'requested':quota,'eligible_before_dedup':len(eligible),
                           'available_after_dedup':len(available),'selected':len(picked)})
        for row in picked:
            used.add(row['source_cluster_id'])
            chosen.append({'case_id':f'case{len(chosen)+1:03d}','category':category,
                           'example_id':row['example_id'],'source_cluster_id':row['source_cluster_id'],
                           'source_order_sha256':source_order(row)[0]})
    return chosen,categories


def prepare(manifest,lock_path,run_dir,summary_path,decisions_path,output,views):
    manifest,lock_path,run_dir,summary_path,decisions_path,output,views=map(
        lambda p:Path(p).resolve(),(manifest,lock_path,run_dir,summary_path,decisions_path,output,views))
    # These existence/count checks deliberately precede all record/summary reads.
    completed_path=run_dir/'completed.json'
    require(completed_path.is_file() and summary_path.is_file(),'Complete main run and finished analysis are both required')
    completed=read(completed_path)
    require(type(completed['records']) is int and completed['records']==124*13,'Main generation is incomplete')
    require(not (run_dir/'error.json').exists(),'Failed runs are not reviewable as completed data')
    require(views.is_relative_to(WORK.resolve()),'Source views must remain under workspace work/')
    require(output.is_relative_to((PROJECT/'reports').resolve()),'Public metadata must remain in project reports/')
    require(not output.exists() and not views.exists(),'Use new metadata and view directories; no silent overwrite')

    from refinement_execution import validate_generation,load_prepared
    from refinement_core import make_request
    from dude_metrics import score_response
    from dude_replication import safe_path
    from native_detail_core import image_digest

    rows,records,run,lock=validate_generation(manifest,lock_path,run_dir)
    require(lock['role']=='main' and len(rows)==124 and len(records)==1612,'Only complete main development runs are supported')
    summary=read(summary_path)
    require(summary['status']=='completed_development_diagnostic' and summary['role']=='main'
            and summary['n_sources']==124 and summary['generation_calls']==1612,'Analysis is not complete main development')
    bound={'manifest_sha256':manifest,'lock_sha256':lock_path,'run_sha256':run_dir/'run.json',
           'records_sha256':run_dir/'records.jsonl','completed_sha256':completed_path,
           'decisions_sha256':decisions_path,'analysis_code_sha256':PROJECT/'experiments/analyze_refinement.py'}
    for field,path in bound.items():require(summary['bindings'][field]==sha(path),'Analysis binding differs: '+field)
    require(sha(decisions_path)==lock['decisions']['sha256'],'Decisions differ from generation lock')
    decisions=read(decisions_path);decisionmap={r['example_id']:r for r in decisions['predictions']}
    prepared,labels,oldrecords=load_prepared('main')
    require(prepared==rows,'Prepared source rows differ')
    require(canonical_records_sha(labels)==summary['bindings']['labels_canonical_sha256'],'Reference metadata differs')
    require(canonical_records_sha(oldrecords)==summary['bindings']['archived_records_canonical_sha256'],'Archived records differ')
    rowmap={r['example_id']:r for r in rows};labelmap={r['example_id']:r for r in labels}
    smap={r['example_id']:r for r in summary['per_source']}
    require(set(rowmap)==set(labelmap)==set(smap)==set(decisionmap),'Source/decision/summary coverage differs')
    require(len(smap)==len(summary['per_source'])==124,'Duplicate summary entries')
    new={(r['example_id'],r['action']):r for r in records}
    old={(r['example_id'],r['action']):r for r in oldrecords}
    for eid in rowmap:
        require(smap[eid]['source_cluster_id']==rowmap[eid]['source_cluster_id'],'Source cluster differs')
        require(smap[eid]['regions']==decisionmap[eid]['regions'],'Committed policy regions differ')
        require(labelmap[eid]['question']==rowmap[eid]['question'],'Question and reference source differ')
        label=labelmap[eid];regions=smap[eid]['regions']
        for policy,region in (('prompted',regions['prompted']),('ranker_full',regions['full'])):
            for domain,index,prefix in (('old',old,''),('new',new,'new_')):
                grade=score_response(index[eid,f'{prefix}native_{region}']['response'],label['original_answers'],
                                     label['validated_primary_answers'],label['original_answer_variants'])
                for metric in ('primary_em','official_anls'):
                    require(grade[metric]==smap[eid]['scores'][policy+'_'+domain][metric],'Selection score differs from raw response')
    selected,categories=select_cases(summary['per_source'])
    output.mkdir(parents=True);views.mkdir(parents=True)
    cases=[]
    for chosen in selected:
        eid=chosen['example_id'];row=rowmap[eid];label=labelmap[eid];regions=smap[eid]['regions']
        source_path=safe_path(manifest.parent,row['image_path'])
        require(sha(source_path)==row['image_sha256'],'Source raster changed before rendering')
        folder=views/chosen['case_id'];folder.mkdir()
        fullpage=folder/'fullpage.png';shutil.copyfile(source_path,fullpage)
        assets={'fullpage':{'workspace_relative_path':fullpage.relative_to(WORK.parent).as_posix(),
                            'sha256':sha(fullpage),'kind':'exact_source_page_file'}}
        with Image.open(source_path) as loaded:source=loaded.convert('RGB')
        for policy,region in (('prompted',regions['prompted']),('ranker_full',regions['full'])):
            _,images,geometry=make_request(source,row['question'],f'new_native_{region}',lock['config'])
            expected=new[eid,f'new_native_{region}']['geometry']['image_rgb_sha256']
            require([image_digest(im) for im in images]==expected,'Rendered review crop differs from actual model input')
            require(old[eid,f'native_{region}']['geometry']['image_rgb_sha256']==expected,'Old/new crop pixels differ')
            crop=folder/(policy+'_native_crop.png');images[1].save(crop,format='PNG')
            assets[policy+'_native_crop']={'workspace_relative_path':crop.relative_to(WORK.parent).as_posix(),
                'sha256':sha(crop),'image_rgb_sha256':image_digest(images[1]),'size':list(images[1].size),
                'region':region,'source_roi_pixels':geometry['source_roi_pixels'],'kind':'exact_resized_native_model_crop'}
        answers={}
        lookups=[('prompted_old',old,f"native_{regions['prompted']}"),
                 ('prompted_new',new,f"new_native_{regions['prompted']}"),
                 ('ranker_full_old',old,f"native_{regions['full']}"),
                 ('ranker_full_new',new,f"new_native_{regions['full']}"),
                 ('highres_old',old,'highres'),('highres_new',new,'new_highres')]
        for name,index,action in lookups:
            record=index[eid,action]
            grade=score_response(record['response'],label['original_answers'],label['validated_primary_answers'],label['original_answer_variants'])
            answers[name]={'action':action,'response':record['response'],'raw_continuation':record['raw_continuation'],
                           'generation_truncated':record['generation_truncated'],
                           **{key:grade[key] for key in ('predicted_answer','parse_valid','primary_em','official_anls')}}
        cases.append({**chosen,'question':row['question'],'fold':smap[eid]['fold'],
                      'regions':{'archived_prompted':regions['prompted'],'oof_ranker_full':regions['full']},
                      'original_answers':label['original_answers'],'original_answer_variants':label['original_answer_variants'],
                      'validated_primary_answers':label['validated_primary_answers'],
                      'source_semantic_audit_status':label.get('source_semantic_audit_status'),
                      'source_image_sha256':row['image_sha256'],'source_size':list(source.size),
                      'answers':answers,'views':assets})
    result={'schema_version':1,'created_utc':datetime.now(timezone.utc).isoformat(),
            'status':'completed_run_post_outcome_unblinded_selection','source_population':124,
            'selection_rule':{'category_order_and_quotas':list(CATEGORY_QUOTAS),
                'sort':'SHA256(UTF-8("refinement-case:" + source_cluster_id)); example_id is the tie-breaker',
                'global_source_deduplication':True,'replace_short_categories':False,
                'prompt_contrast':'archived prompted region: new minus old primary EM',
                'ranker_contrast':'OOF full ranker new minus archived prompted new primary EM'},
            'category_counts':categories,'selected_cases':len(cases),'cases':cases,
            'bindings':{**{field:sha(path) for field,path in bound.items()},'summary_sha256':sha(summary_path),
                        'case_script_sha256':sha(Path(__file__))},
            'review_completed':False,'grades_changed':False,'new_model_calls':0,
            'limitations':['Cases are selected by observed automatic outcomes and are not a representative semantic audit.',
                           'Conditions and references are visible; qualitative judgments describe the selected cases without independent expert validation.',
                           'The 124 sources are reused development data; no population semantic accuracy can be inferred.',
                           'Full-page and crop media are local work assets, excluded from the public metadata package.']}
    with (output/'selection.json').open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False);stream.write('\n')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('manifest','lock','run','summary','decisions'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--output',type=Path,default=PROJECT/'reports/refinement124/qualitative_cases')
    parser.add_argument('--views',type=Path,default=WORK/'refinement_case_review')
    args=parser.parse_args()
    result=prepare(args.manifest,args.lock,args.run,args.summary,args.decisions,args.output,args.views)
    print(json.dumps({'selected_cases':result['selected_cases'],'category_counts':result['category_counts'],
                      'new_model_calls':0,'metadata':str(args.output/'selection.json')}))


if __name__=='__main__':main()
