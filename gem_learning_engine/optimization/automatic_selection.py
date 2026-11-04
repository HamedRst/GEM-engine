"""Frozen, one-feature Surrogate-QC SPARC automatic optimizer selection.

The selector is not a new optimization algorithm: it dispatches to unchanged
native gsMOBO+SPARC or NSGA-II+SPARC, before any exact optimizer FBA calls.
Its deployment model is a separately documented fit on 297 historical rows;
retrospective 297 held-out predictions used different training folds.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

AUTO_ID='AUTO_QC_SPARC'
GSMOBO_SPARC='DEV_GSMOBO50K_SPARC_A'
NSGA_SPARC='DEV_NSGA50K_SPARC_LEGACY_OPERATOR'
MODEL=Path(__file__).with_name('auto_selector_model.json')
DECISION_NAME='automatic_selection.json'


def hash_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def model_and_hash():
    if not MODEL.is_file() or MODEL.is_symlink():
        raise RuntimeError('Frozen Automatic selection coefficients are unavailable')
    obj=json.loads(MODEL.read_text(encoding='utf-8'))
    if (obj.get('model_kind')!='one_feature_standard_scaler_ridge' or obj.get('n_train_rows')!=297 or
            obj.get('ridge_alpha')!=6.0 or obj.get('target_clip_pp')!=[-20,20] or
            obj.get('switch_margin_pp')!=0.5 or obj.get('feature_name')!='qc_nrmse_max'):
        raise RuntimeError('Automatic selection model metadata is not the documented final QC-only method')
    for k in ('raw_intercept_pp','raw_coefficient_pp_per_nrmse'):
        if not math.isfinite(float(obj[k])):raise RuntimeError('Non-finite selector coefficient')
    return obj,hash_file(MODEL)


def fingerprint(job):
    # The selected method must never change on resume or when source input changes.
    data={k:job.get(k) for k in ('case','prior_rows','seed','budget','problem_hash','exact_solver','general_binding_id')}
    for key in ('model_path','landscape_path'):
        path=job.get(key)
        if not path or not Path(path).is_file(): raise RuntimeError('Source file missing for selector: '+key)
        data[key+'_sha256']=hash_file(path)
    data['auto_id']=AUTO_ID
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def feature_from_qc(qc):
    if qc.get('status')!='PASS' or qc.get('live_exact_calls')!=0:
        raise RuntimeError('Automatic selection requires PASS surrogate QC with zero optimizer FBA calls')
    metrics=qc.get('objective_metrics')
    if not isinstance(metrics,dict) or not metrics:
        raise RuntimeError('Objective-specific surrogate QC metrics missing')
    values=[]
    for key,metric in metrics.items():
        v=metric.get('range_nrmse') if isinstance(metric,dict) else None
        if v is None or not math.isfinite(float(v)) or float(v)<0:
            raise RuntimeError('Unavailable objective-specific range_nrmse: '+str(key))
        values.append(float(v))
    return max(values)


def predict(qc):
    model,model_sha=model_and_hash()
    x=feature_from_qc(qc)
    prediction=model['raw_intercept_pp']+model['raw_coefficient_pp_per_nrmse']*x
    family='gsmobo' if prediction>model['switch_margin_pp'] else 'nsga'
    return {'feature_qc_nrmse_max':x,'predicted_gsmobo_minus_nsga_hv_pp':prediction,
            'selected_family':family,'selected_method':GSMOBO_SPARC if family=='gsmobo' else NSGA_SPARC,
            'selector_model_id':model['id'],'selector_model_sha256':model_sha,
            'switch_margin_pp':model['switch_margin_pp'],
            'outside_historical_qc_range':not (model['training_qc_min']<=x<=model['training_qc_max']),
            'deployment_refit_not_cross_validated_figure3_predictions':True,
            'feature_source':'pre-run objective-specific surrogate QC only'}


def _atomic_json(path,data):
    from .worker import atomic_json
    atomic_json(path,data)


def preflight(job_path):
    """Child process, isolated from subsequent selected optimizer runtime imports."""
    job_path=Path(job_path).resolve();job=json.loads(job_path.read_text());out=job_path.parent
    if job.get('method')!=AUTO_ID:raise RuntimeError('Expected Automatic selection job')
    stage=out/'automatic_selection_preflight';stage.mkdir(exist_ok=True)
    provisional={**job,'family':'nsga','method':NSGA_SPARC}
    from .worker import prepare
    if job.get('general_binding'):
        from .general_runtime import prepare_general
        cfg,parent,baseline,runtime,prov=prepare_general(provisional,stage)
    else:
        cfg,parent,baseline,runtime,prov=prepare(provisional,stage)
    import surrogate_qc  # Loaded from the provisional NSGA runtime in this child process.
    qc=surrogate_qc.validate(cfg,job['case'])
    if qc.get('rows')!=int(job['prior_rows']):
        raise RuntimeError('QC cohort size inconsistent with selected landscape prior')
    selected=predict(qc)
    qcpath=Path(cfg['paths']['output_root'])/job['case']/f"seed_{job['seed']}"/'surrogate_qc'/'surrogate_qc_summary.json'
    if not qcpath.is_file():raise RuntimeError('Surrogate QC summary missing after preflight')
    selected.update({'requested_method':AUTO_ID,
        'job_fingerprint':fingerprint(job),'preoptimization_qc_summary':str(qcpath.relative_to(out)),
        'preoptimization_qc_sha256':hash_file(qcpath), 'preoptimization_qc_live_exact_calls':0,
        'selection_timestamp_utc':time.time(),'preflight_runtime_key':'nsga_parametric'})
    _atomic_json(out/DECISION_NAME,selected)
    print('Automatic selection:',selected['selected_family'],selected['selected_method'],
          'QC max NRMSE',selected['feature_qc_nrmse_max'],
          'predicted difference pp',selected['predicted_gsmobo_minus_nsga_hv_pp'],flush=True)


def check_decision(job,out,decision):
    if (decision.get('requested_method')!=AUTO_ID or
            decision.get('job_fingerprint')!=fingerprint(job)):
        raise RuntimeError('Saved selector decision does not match original job; refuse replay')
    model,sha=model_and_hash()
    if decision.get('selector_model_sha256')!=sha or decision.get('selector_model_id')!=model['id']:
        raise RuntimeError('Selector model changed since the original decision; refuse replay')
    family=decision.get('selected_family');method=decision.get('selected_method')
    if (family,method) not in (('gsmobo',GSMOBO_SPARC),('nsga',NSGA_SPARC)):
        raise RuntimeError('Unsupported selected frozen optimizer')
    qcpath=Path(out)/decision['preoptimization_qc_summary']
    if (not qcpath.is_file() or qcpath.is_symlink() or not qcpath.resolve().is_relative_to(Path(out).resolve()) or
            hash_file(qcpath)!=decision.get('preoptimization_qc_sha256')):
        raise RuntimeError('Pre-run QC evidence missing or altered; refuse replay')
    verified=predict(json.loads(qcpath.read_text()))
    if verified['selected_method']!=method:
        raise RuntimeError('Selector decision inconsistent with verified QC and frozen coefficients')
    return decision


def resolve(job,out):
    out=Path(out).resolve();decision_path=out/DECISION_NAME
    if not decision_path.is_file():
        log=out/'automatic_selection_preflight.log'
        with log.open('a',encoding='utf-8') as f:
            proc=subprocess.run([sys.executable,'-m','gem_learning_engine.optimization.automatic_selection',
                                 '--preflight','--job',str(out/'job.json')],
                                 stdout=f,stderr=subprocess.STDOUT,check=False)
        if proc.returncode!=0:
            raise RuntimeError('Automatic selection QC preflight failed; no optimizer FBA started. See '+str(log))
    if not decision_path.is_file():raise RuntimeError('Automatic selection decision was not persisted')
    decision=check_decision(job,out,json.loads(decision_path.read_text()))
    selected={**job,'family':decision['selected_family'],'method':decision['selected_method']}
    selected['automatic_selection_source_method']=AUTO_ID
    selected['selector_model_sha256']=decision['selector_model_sha256']
    return selected,decision


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--preflight',action='store_true');ap.add_argument('--job',required=True)
    a=ap.parse_args()
    if not a.preflight:raise SystemExit('Use --preflight')
    preflight(a.job)

if __name__=='__main__':main()
