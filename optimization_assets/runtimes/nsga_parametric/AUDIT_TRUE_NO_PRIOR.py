#!/usr/bin/env python3
from __future__ import annotations
import json, math, re
from pathlib import Path
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent
RESULTS=HERE/'results_M5PRO'
CFG=__import__('yaml').safe_load((HERE/'worker_state/M5PRO/study_config.yaml').read_text()) or {}
N=int(CFG['study']['landscape_prior_rows'])
SEED=f"seed_{CFG['study']['seed']}"
METHOD='BASE_NSGA2_NO50K'
CHECKS=(16,32,64,128)

def bools(s):
    if s.dtype==bool: return s.fillna(False)
    return s.astype(str).str.lower().isin(['true','1','yes','y'])

def objcols(df):
    prefs=[
        [c for c in df.columns if str(c).startswith('objective_norm__')],
        [c for c in df.columns if str(c).startswith('exact_objective_norm__')],
        [c for c in df.columns if str(c).lower().startswith('exact_') and str(c).lower().endswith('_norm')],
    ]
    for cols in prefs:
        if cols: return cols
    # Last resort: normalized objective columns containing both objective and norm.
    cols=[c for c in df.columns if 'objective' in str(c).lower() and 'norm' in str(c).lower()]
    if cols: return cols
    raise RuntimeError('Could not identify normalized objective columns in calls.csv')

def union_box_volume(P):
    P=np.asarray(P,float)
    if P.ndim!=2 or len(P)==0: return 0.0
    P=np.maximum(P,0.0)
    P=P[np.all(np.isfinite(P),axis=1)]
    if len(P)==0: return 0.0
    # Remove dominated points for maximization.
    keep=np.ones(len(P),dtype=bool)
    for i in range(len(P)):
        if not keep[i]: continue
        dom=np.all(P>=P[i],axis=1)&np.any(P>P[i],axis=1)
        if dom.any(): keep[i]=False
    P=P[keep]
    d=P.shape[1]
    if d==1: return float(P[:,0].max(initial=0.0))
    xs=np.unique(P[:,0]); xs=xs[xs>0]
    prev=0.0; v=0.0
    for x in xs:
        sub=P[P[:,0]>=x,1:]
        v+=(float(x)-prev)*union_box_volume(sub)
        prev=float(x)
    return float(v)

def hv_from_calls(df, prefer_runtime=True):
    if len(df)==0: return float('nan')
    # Official true-NO-PRIOR metric when emitted by runtime: self-front only.
    # For sliced subsets (e.g. adaptive-only 17..128), disable cumulative runtime fields.
    for cc in (('self_hv_after','method_only_hv_after','method_only_hv') if prefer_runtime else ()):
        if cc in df.columns:
            s=pd.to_numeric(df[cc],errors='coerce').dropna()
            if len(s): return float(s.iloc[-1])
    cols=objcols(df)
    x=df.copy()
    feas=None
    for c in ('exact_feasible','feasible','solver_feasible'):
        if c in x.columns:
            feas=bools(x[c]); break
    if feas is not None: x=x[feas]
    if len(x)==0: return 0.0
    Y=x[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    Y=Y[np.all(np.isfinite(Y),axis=1)]
    return union_box_volume(Y)

def prior_hv(case, seed_dir):
    fp=seed_dir/'frozen_pareto_front.csv'
    if fp.is_file():
        try: return hv_from_calls(pd.read_csv(fp))
        except Exception: pass
    # Metadata fallback.
    prep=HERE/'worker_state'/'M5PRO'/'prepared'/case
    for q in prep.rglob('*.json'):
        try:
            j=json.loads(q.read_text())
        except Exception: continue
        for k in ('computed_prior_hv','computed_prior10k_hv','prior_hv_exact','landscape_hv','baseline_hv'):
            if isinstance(j,dict) and k in j:
                try: return float(j[k])
                except Exception: pass
    return float('nan')

def full50_hv(case):
    prep=HERE/'worker_state'/'M5PRO'/'prepared'/case
    for q in prep.rglob('*.json'):
        try: j=json.loads(q.read_text())
        except Exception: continue
        if isinstance(j,dict):
            for k in ('computed_full50k_hv','full50k_reference_hv','landscape_50k_hv','reference_hv_50k','full50k_hv_qc_reference_rounded'):
                if k in j:
                    try: return float(j[k])
                    except Exception: pass
    ref=HERE/'REFERENCE_5K_50K_HV_BOUNDS.json'
    if ref.is_file():
        try:
            r=json.loads(ref.read_text()); return float(r['cases'][case]['full50k_hv'])
        except Exception: pass
    return float('nan')

rows=[]
if RESULTS.is_dir():
    for case_dir in sorted([d for d in RESULTS.iterdir() if d.is_dir() and '__' in d.name]):
        case=case_dir.name
        calls=case_dir/SEED/METHOD/'calls.csv'
        row={'case':case,'status':'NOT_STARTED'}
        if calls.is_file():
            df=pd.read_csv(calls)
            # Preserve file order as call order; if a call index exists, sort it.
            for c in ('call','logical_call','call_index'):
                if c in df.columns:
                    df=df.sort_values(c); break
            for n in CHECKS:
                row[f'no_prior_hv_{n}']=hv_from_calls(df.iloc[:min(n,len(df))]) if len(df)>=n else float('nan')
            row['adaptive_only_hv_17_128']=hv_from_calls(df.iloc[16:128], prefer_runtime=False) if len(df)>=17 else float('nan')
            row['status']='COMPLETE_128' if len(df)>=128 else f'PARTIAL_{len(df)}'
        seed_dir=case_dir/SEED
        row['prior_N_hv']=prior_hv(case, seed_dir)
        row['landscape_50k_hv']=full50_hv(case)
        x=row.get('no_prior_hv_128',float('nan'))
        p=row['prior_N_hv']; f=row['landscape_50k_hv']
        row['delta_128_vs_prior_N']=x-p if np.isfinite(x) and np.isfinite(p) else float('nan')
        row['delta_128_vs_50k']=x-f if np.isfinite(x) and np.isfinite(f) else float('nan')
        row['beats_prior_N_landscape']=bool(np.isfinite(row['delta_128_vs_prior_N']) and row['delta_128_vs_prior_N']>0)
        row['beats_50k_landscape']=bool(np.isfinite(row['delta_128_vs_50k']) and row['delta_128_vs_50k']>0)
        rows.append(row)
out=pd.DataFrame(rows)
outfile=HERE/f'TRUE_NO_PRIOR_SELF_FRONT_N{N}_AUDIT.csv'
out.to_csv(outfile,index=False)
print('='*160)
print('NSGA PARAMETRIC-PRIOR STUDY — TRUE NO-PRIOR SELF-FRONT AUDIT')
print('NO N-ROW OR 50K LANDSCAPE FRONTIER IS UNIONED INTO THESE NO-PRIOR SCORES')
print('='*160)
if len(out): print(out.to_string(index=False))
complete=out[out.status=='COMPLETE_128'] if 'status' in out else out.iloc[0:0]
print('\nCompleted @128:',len(complete))
if len(complete):
    print('NO-PRIOR @128 beats N-row landscape :',f"{int(complete.beats_prior_N_landscape.sum())}/{len(complete)}")
    print('NO-PRIOR @128 beats 50K landscape:',f"{int(complete.beats_50k_landscape.sum())}/{len(complete)}")
print('\nSaved:',outfile)
print('\nDefinitions:')
print('prior_N_hv               = frozen N-row Sobol-prefix landscape alone')
print('landscape_50k_hv        = original 50K reference landscape alone when metadata is available')
print('no_prior_hv_*           = BASE_NSGA2_NO50K live calls ONLY')
print('adaptive_only_hv_17_128 = NSGA no-prior calls 17-128 ONLY')
