#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,yaml
from generic_core import hypervolume_exact
from gsmobo_algorithms import BASE_NO50K

def b(s):
 if pd.api.types.is_bool_dtype(s):return s.astype(bool).to_numpy(bool)
 return s.astype(str).str.lower().isin(['true','1','yes','optimal','feasible','optimal_generic_problem']).to_numpy(bool)
def hv_rows(df,cols,ref):
 if not len(df):return 0.0
 f=b(df['exact_feasible']);Y=df[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float);ok=f&np.isfinite(Y).all(axis=1)
 return float(hypervolume_exact(Y[ok],ref)) if ok.any() else 0.0
def first_call(h,cols,ref,target):
 for k in range(1,len(h)+1):
  if hv_rows(h.iloc[:k],cols,ref)>float(target)+1e-12:return k
 return None
def validate_history(h,case):
 req={'call','exact_feasible','archive_hv_after'}
 miss=sorted(req-set(h.columns))
 if miss:raise RuntimeError(f'{case}: NO-PRIOR calls.csv missing columns {miss}')
 c=pd.to_numeric(h['call'],errors='coerce')
 if c.isna().any() or not np.allclose(c.to_numpy(float),np.round(c.to_numpy(float)),rtol=0,atol=0):raise RuntimeError(f'{case}: invalid NO-PRIOR call indices')
 ints=c.to_numpy(int)
 if len(ints)!=128 or len(np.unique(ints))!=128 or not np.array_equal(np.sort(ints),np.arange(1,129)):raise RuntimeError(f'{case}: NO-PRIOR history must contain exactly calls 1..128; got {len(ints)} rows')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study-config',required=True);a=ap.parse_args();sp=Path(a.study_config).resolve();cfg=yaml.safe_load(sp.read_text()) or {};outroot=Path(cfg['paths']['output_root']);rows=[]
 for case in cfg['cases']:
  prep=yaml.safe_load((Path(cfg['paths']['prepared_root'])/case/'problem_config.yaml').read_text()) or {};p=prep['problem'];comp=p['compiled_problem'];cols=list(comp['objective_norm_columns']);ref=tuple(float(x) for x in comp.get('hv_reference_normalized',[0.0]*len(cols)));d=outroot/case/f"seed_{cfg['study']['seed']}";cp=d/BASE_NO50K/'calls.csv'
  if not cp.exists():raise RuntimeError(f'{case}: missing NO-PRIOR calls.csv')
  h=pd.read_csv(cp).sort_values('call').reset_index(drop=True);validate_history(h,case)
  missing=[x for x in cols if x not in h.columns]
  if missing:raise RuntimeError(f'{case}: NO-PRIOR history missing objective columns {missing}')
  prior=float(p['prior_hv_exact']);full=float(p['full50k_reference_hv']);rec={'case':case,'prior_label':cfg['study']['prior_label'],'landscape_prior_rows':cfg['study']['landscape_prior_rows'],'landscape_prior_hv':prior,'landscape_50k_hv':full}
  for cpv in (16,32,64,128):rec[f'no_prior_self_hv_{cpv}']=hv_rows(h[pd.to_numeric(h.call,errors='coerce')<=cpv],cols,ref)
  rec['adaptive_only_self_hv_17_128']=hv_rows(h[(pd.to_numeric(h.call,errors='coerce')>=17)&(pd.to_numeric(h.call,errors='coerce')<=128)],cols,ref);rec['old_union_hv_128_diagnostic_only']=float(pd.to_numeric(h.iloc[127]['archive_hv_after'],errors='coerce'));rec['delta_128_vs_prior']=rec['no_prior_self_hv_128']-prior;rec['delta_128_vs_50k']=rec['no_prior_self_hv_128']-full;rec['beats_prior_landscape']=bool(rec['no_prior_self_hv_128']>prior+1e-12);rec['beats_50k_landscape']=bool(rec['no_prior_self_hv_128']>full+1e-12);rec['first_call_beating_prior']=first_call(h,cols,ref,prior);rec['first_call_beating_50k']=first_call(h,cols,ref,full);rec['scoring_contract']='LIVE_SELF_FRONT_ONLY_NO_LANDSCAPE_UNION';rows.append(rec)
 if len(rows)!=25:raise RuntimeError(f'NO-PRIOR audit expected 25 cases, got {len(rows)}')
 df=pd.DataFrame(rows);tag=str(cfg['study']['prior_tag']);out=Path(__file__).resolve().parent/f'NO_PRIOR_SELF_FRONT_{tag}_AUDIT.csv';df.to_csv(out,index=False);print('='*150);print(f'{cfg["study"]["prior_label"]} STUDY — TRUE NO-PRIOR SELF-FRONT AUDIT');print('NO CURRENT-PRIOR OR 50K LANDSCAPE FRONTIER IS UNIONED INTO THESE NO-PRIOR SCORES');print('='*150);print(df.to_string(index=False));print(f'\nAUDIT CSV: {out}')
if __name__=='__main__':main()
