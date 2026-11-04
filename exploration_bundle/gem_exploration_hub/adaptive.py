"""Query-driven exact-FBA rescue following GEM Engine Exploration paper pipeline.

Original Calculator scientific operators, surrogate fit/QC and Atlas rendering are
unmodified. Only opt-in, verified CSV-based evaluator/landscape combinations can
acquire new evidence. Failed quality gates remain failed after budget exhaustion.
"""
from __future__ import annotations
import json
import math
import fcntl
import re
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc

from gem_exploration.engine import prepare_problem
from gem_exploration.operators import OPERATOR_BACKENDS
from gem_exploration.surrogate import fit_surrogate
from .acquisition import extend_archive, evaluator_identity, digest


class InsufficientEvidence(RuntimeError):
    """QC not met after authorized evidence acquisition; never a validated analysis."""
    def __init__(self, reason, record):
        self.record = record
        super().__init__(reason)


QC_MARKERS = ('Insufficient finite rows for surrogate',
              'Surrogate regression QC failed', 'Surrogate classifier QC failed',
              'Scope leaves too few rows for')


def assess_quality(problem: dict, landscape: str, seed: int = 20260822) -> dict:
    """Use original v0.5.0 QC, same preprocessed target, scope and variables.

    No new surrogates or replacement thresholds. Raw CV scores are retained.
    """
    op = (problem.get('operator') or {}).get('name')
    backend = OPERATOR_BACKENDS.get(op)
    if backend == 'landscape':
        return {'status':'not_required','operator':op,'backend':backend,
                'reason':'Exact-landscape operator does not require surrogate QC'}
    if backend not in ('surrogate_assisted','hybrid'):
        raise ValueError(f'Unknown operator/backend: {op}/{backend}')
    try:
        context = prepare_problem(problem,landscape,seed)
    except RuntimeError as exc:
        if any(m in str(exc) for m in QC_MARKERS):
            return {'status':'insufficient','operator':op,'backend':backend,'reason':str(exc),'metrics':{}}
        raise
    spec=context['spec']; df=context['scoped']; vars_=context['variables']; cfg=spec['surrogate']
    goals=[]
    if op == 'feasibility':
        col=(spec['operator'].get('parameters') or {}).get('column','feasible')
        if col not in df.columns:raise ValueError(f'Feasibility column absent: {col}')
        y=df[col].fillna(False).astype(bool).astype(int)
        if y.nunique()<2:
            # Match the original feasibility operator: no classifier with one class.
            return {'status':'not_required','operator':op,'backend':backend,
                    'reason':'Single feasibility class: original exact-only frequency with no classifier'}
        d=df.assign(_feasible_numeric=y)
        goals=[(d,'_feasible_numeric','classification')]
    else:
        targets=[str(t['name']) for t in spec.get('targets',[])]
        if not targets:raise ValueError('Surrogate operator needs at least one target')
        goals=[(df,t,'regression') for t in targets]
    metrics={}; failures={}
    for frame,target,task in goals:
        try:
            fitted=fit_surrogate(frame,vars_,target,cfg,task=task,seed=seed)
            metrics[target]=fitted.metrics
        except RuntimeError as exc:
            if any(m in str(exc) for m in QC_MARKERS):failures[target]=str(exc)
            else:raise
    return {'status':'insufficient' if failures else 'pass','operator':op,'backend':backend,
            'rows_in_scope':len(df),'variables':vars_,'metrics':metrics,'failures':failures,
            'reason':'; '.join(f'{k}: {v}' for k,v in failures.items()) if failures else 'Original Calculator surrogate QC passed'}


def select_candidate_designs(problem:dict, landscape:str, evaluator:dict, n:int, seed:int)->list[dict]:
    """Maximin novelty with optional query relevance; no unreliable QC model is trusted.

    Half the selection remains global coverage, so focusing on a local question
    does not entirely collapse design-space support. Tree dispersion is not treated
    as calibrated uncertainty. Variable levels are always within evaluator bounds.
    """
    if n<1:return []
    vars_=evaluator['variables']; dim=len(vars_)
    if dim>64:raise ValueError('Too many evaluator variables')
    cols=[v['column'] for v in vars_]
    frame=pd.read_csv(landscape,usecols=cols)
    X=frame[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    bounds=np.array([[v['min'],v['max']] for v in vars_],dtype=float)
    extent=bounds[:,1]-bounds[:,0]
    X=(X-bounds[:,0])/extent
    X=X[np.isfinite(X).all(axis=1)]
    tree=cKDTree(X) if len(X) else None
    m=int(math.ceil(math.log2(max(128,n*64))))
    candidates=qmc.Sobol(d=dim,scramble=True,seed=seed).random_base2(m=m)
    # Estimate which dimensions are explicitly varied by this operator.
    p=(problem.get('operator') or {}).get('parameters') or {}
    focus=[]
    for field in ('variable','variable_x','variable_y','x_variable','y_variable'):
        val=p.get(field)
        if isinstance(val,str) and val in cols and val not in focus:focus.append(val)
    for field in ('variables','pair'):
        val=p.get(field)
        if isinstance(val,(list,tuple)):
            for item in val:
                if item in cols and item not in focus:focus.append(item)
    if focus and dim>1:
        # Explicitly vary queried dimension(s); hold other dimensions near the
        # empirical median. Still include the original Sobol global candidates.
        anchor=np.nanmedian(X,axis=0) if len(X) else np.full(dim,0.5)
        anchor=np.clip(anchor,0,1)
        local=np.tile(anchor,(len(candidates),1))
        for col in focus:local[:,cols.index(col)]=candidates[:,cols.index(col)]
        candidates=np.vstack([candidates,local])
    candidates=np.clip(candidates,0,1)
    # Farthest-first sequential novelty. Batch cannot choose duplicates.
    if tree is None:nearest=np.full(len(candidates),math.sqrt(dim),dtype=float)
    else:nearest=tree.query(candidates,k=1)[0]
    chosen=[]
    for _ in range(n):
        idx=int(np.argmax(nearest))
        if nearest[idx]<=1e-8:break
        point=candidates[idx]
        chosen.append({v['name']:float(v['min']+point[j]*extent[j]) for j,v in enumerate(vars_)})
        nearest=np.minimum(nearest,np.linalg.norm(candidates-point,axis=1))
    return chosen


def _run_adaptive_locked(problem:dict, original_csv:str, acq:dict,workspace:Path,
                 seed:int=20260822)->dict:
    """QC -> budgeted exact-FBA batches -> landscape-copy -> QC retry.

    Budget is in *new design points*, not LP solver calls. Each point can take two
    exact LP solves. A stalled QC is not silently declared successful.
    """
    original_csv=str(Path(original_csv).expanduser().resolve())
    model_path=acq['model_path']; evaluator=acq['evaluator']
    total=int(acq.get('max_new_designs_total',0))
    batch=int(acq.get('batch_size',4))
    rounds=int(acq.get('max_rounds',5))
    if not 0<=total<=10000 or not 1<=batch<=1000 or not 1<=rounds<=100:
        raise ValueError('Invalid adaptive budget, batch size, or rounds')
    if not acq.get('query_group'):raise ValueError('Query group required')
    manifest=json.loads(Path(acq['source_manifest']).read_text(encoding='utf-8'))
    evaluator_sha,_=evaluator_identity(model_path,evaluator)
    if (manifest.get('source_sha256')!=digest(original_csv) or
        manifest.get('evaluator_sha256')!=evaluator_sha or
        manifest.get('schema_version')!='1.0'):
        raise ValueError('Verified source/model/evaluator manifest mismatch')
    dest=Path(workspace).expanduser().resolve()/'landscapes'/acq['query_group']/'extended_landscape.csv'
    source_rows=len(pd.read_csv(original_csv,usecols=['growth']))
    previously_acquired=0
    if dest.exists():
        # extend_archive validates both source and working SHA/evaluator before use.
        checked=extend_archive(original_csv,acq['source_manifest'],model_path,evaluator,
                               workspace,acq['query_group'],source_rows,0)
        if Path(checked['working_landscape']).resolve()!=dest.resolve():
            raise ValueError('Working group points to an unexpected archive')
        previously_acquired=checked['working_rows']-source_rows
    current_landscape=str(dest) if dest.exists() else original_csv
    baseline=assess_quality(problem,current_landscape,seed)
    record={'policy':'adaptive_qc','source_landscape':original_csv,'initial_working_surrogate_qc':baseline,
            'history':[{'round':0,'status':baseline['status'],'qc':baseline}],
            'new_designs':0,'new_fba_solves':0,'budget_designs':total,
            'batch_size':batch,'max_rounds':rounds,'evidence_backend':baseline['backend'],
            'previously_acquired_designs':previously_acquired}
    if baseline['status'] in ('pass','not_required'):
        record.update({'status':'pass','working_landscape':current_landscape,'source_reused_without_copy':not dest.exists(),
                       'final_surrogate_qc':baseline})
        return record
    # This evaluator appends exactly growth, product and feasibility. Refuse to
    # spend FBA on a surrogate target that the new rows could not populate.
    if total>previously_acquired and baseline['status']=='insufficient':
        supported={'growth','product','feasible'}
        for target in problem.get('targets',[]):
            raw=str(target.get('value',target.get('name','')))
            col=raw.split(':',1)[1] if raw.startswith('result:') else raw
            if col not in supported:
                raise ValueError('Exact evaluator cannot generate target '+repr(raw)+
                    '; FBA acquisition requires a compatible target-flux/deterministic evaluator mapping')
        available=[c for c in pd.read_csv(current_landscape,nrows=0).columns if c.startswith('requested_medium__')]
        expected=[v['column'] for v in evaluator['variables']]
        if set(available)!=set(expected):
            raise ValueError('Exact evaluator does not cover all existing design columns; unsafe landscape extension')
    landscape=current_landscape
    for i in range(1,rounds+1):
        allowance=min(batch,total-previously_acquired-record['new_designs'])
        if allowance<=0:break
        # Reuse verified prior extension for this exact group, if present.
        if dest.exists():landscape=str(dest)
        candidates=select_candidate_designs(problem,landscape,evaluator,allowance,
                                            int(acq.get('seed',42))+i-1)
        if not candidates:break
        rows=pd.read_csv(landscape,usecols=['growth'])
        result=extend_archive(original_csv,acq['source_manifest'],model_path,evaluator,workspace,
                              acq['query_group'],len(rows)+len(candidates),allowance,
                              seed=int(acq.get('seed',42))+i-1,candidate_designs=candidates)
        landscape=result['working_landscape']
        record['new_designs']+=int(result['new_designs'])
        record['new_fba_solves']+=int(result['new_fba_solves'])
        qc=assess_quality(problem,landscape,seed)
        record['history'].append({'round':i,'acquisition':result,'qc':qc,'status':qc['status']})
        if qc['status'] in ('pass','not_required'):
            record.update({'status':'pass','working_landscape':landscape,'final_surrogate_qc':qc})
            return record
        if int(result['new_designs'])==0:break
    record.update({'status':'insufficient','working_landscape':landscape,
                   'final_surrogate_qc':record['history'][-1]['qc']})
    raise InsufficientEvidence('Surrogate remains insufficient after authorized exact-FBA acquisition budget',record)


def run_adaptive(problem:dict,original_csv:str,acq:dict,workspace:Path,seed:int=20260822)->dict:
    """Serialize adaptive operations for one group, preventing duplicated concurrent solves."""
    group=acq.get('query_group')
    if not isinstance(group,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',group):
        raise ValueError('Invalid working landscape group')
    lock_dir=Path(workspace).expanduser().resolve()/'.acquisition_locks'
    lock_dir.mkdir(parents=True,exist_ok=True)
    with (lock_dir/(group+'.lock')).open('a+') as f:
        try:fcntl.flock(f.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('Acquisition for this landscape group is already running')
        try:return _run_adaptive_locked(problem,original_csv,acq,workspace,seed)
        finally:fcntl.flock(f.fileno(),fcntl.LOCK_UN)
