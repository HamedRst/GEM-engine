#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,yaml
from selected_panel import SELECTED_METHODS
from gsmobo_algorithms import BASE_NO50K

CHECKPOINTS=[32,64,128]

def fail(msg): raise SystemExit('POSTRUN VERIFY FAIL: '+msg)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study-config',required=True);a=ap.parse_args();sp=Path(a.study_config).resolve();cfg=yaml.safe_load(sp.read_text()) or {}
 worker=str(cfg['worker']);tag=str(cfg['study']['prior_tag']);label=str(cfg['study']['prior_label']);n=int(cfg['study']['landscape_prior_rows']);here=Path(__file__).resolve().parent
 csv=here/f'ALL_RESULTS_{worker}_{tag}_SELECTED3.csv';audit=here/f'NO_PRIOR_SELF_FRONT_{tag}_AUDIT.csv'
 if not csv.exists():fail(f'missing consolidated CSV {csv}')
 if not audit.exists():fail(f'missing NO-PRIOR audit {audit}')
 df=pd.read_csv(csv);ad=pd.read_csv(audit)
 expected_rows=25*len(SELECTED_METHODS)*len(CHECKPOINTS)
 if len(df)!=expected_rows:fail(f'consolidated rows={len(df)} expected={expected_rows}')
 if df['case_key'].nunique()!=25:fail(f'case count={df.case_key.nunique()} expected=25')
 if set(df['method'].astype(str))!=set(SELECTED_METHODS):fail('method set mismatch')
 if set(pd.to_numeric(df['checkpoint'],errors='coerce').dropna().astype(int))!=set(CHECKPOINTS):fail('checkpoint set mismatch')
 if df['experiment_row_id'].duplicated().any():fail('duplicate experiment_row_id')
 combos=df[['case_key','method','checkpoint']]
 if combos.duplicated().any():fail('duplicate case/method/checkpoint combination')
 if not (df['run_status'].astype(str)=='COMPLETE').all():fail('one or more rows are not run_status=COMPLETE')
 ci=df['common_initialization_identity_pass']
 ci_ok=ci.astype(bool) if pd.api.types.is_bool_dtype(ci) else ci.astype(str).str.lower().isin(['true','1','yes'])
 if not ci_ok.all():fail('common initialization identity failed')
 if df['error'].fillna('').astype(str).str.strip().ne('').any():fail('nonempty method error field')
 # prior metadata contract
 prior=df[df.method.astype(str)!=BASE_NO50K]
 if not (pd.to_numeric(prior.landscape_prior_rows,errors='coerce')==n).all():fail('prior rows metadata mismatch')
 if not (prior.prior_label.astype(str)==label).all():fail('prior label metadata mismatch')
 no=df[df.method.astype(str)==BASE_NO50K].copy()
 if not (no.evaluation_mode.astype(str)=='LIVE_SELF_FRONT').all():fail('NO-PRIOR evaluation mode not LIVE_SELF_FRONT')
 if not np.allclose(pd.to_numeric(no.baseline_hv,errors='coerce'),0.0,rtol=0,atol=0,equal_nan=False):fail('NO-PRIOR baseline_hv is not zero')
 if not np.allclose(pd.to_numeric(no.final_primary_hv,errors='coerce'),pd.to_numeric(no.final_self_hv,errors='coerce'),rtol=0,atol=1e-12,equal_nan=False):fail('NO-PRIOR primary HV != self-front HV')
 corr=no['no_prior_self_front_correction_applied'];corr_ok=corr.astype(bool) if pd.api.types.is_bool_dtype(corr) else corr.astype(str).str.lower().isin(['true','1','yes'])
 if not corr_ok.all():fail('NO-PRIOR correction flag is not true')
 if len(ad)!=25 or ad['case'].nunique()!=25:fail(f'audit rows/cases invalid: rows={len(ad)} cases={ad.case.nunique()}')
 if not (ad['scoring_contract'].astype(str)=='LIVE_SELF_FRONT_ONLY_NO_LANDSCAPE_UNION').all():fail('audit scoring contract mismatch')
 # audit values must agree with consolidated NO-PRIOR values at 32/64/128
 for cp in CHECKPOINTS:
  amap=ad.set_index('case')[f'no_prior_self_hv_{cp}']
  dcp=no[pd.to_numeric(no.checkpoint,errors='coerce').astype(int)==cp].set_index('case_key')['final_primary_hv']
  if set(amap.index)!=set(dcp.index):fail(f'audit/consolidated case mismatch at checkpoint {cp}')
  aa=pd.to_numeric(amap.loc[dcp.index],errors='coerce').to_numpy(float);bb=pd.to_numeric(dcp,errors='coerce').to_numpy(float)
  if not np.allclose(aa,bb,rtol=0,atol=1e-12,equal_nan=False):fail(f'audit/consolidated NO-PRIOR mismatch at checkpoint {cp}')
 rec={'status':'PASS','worker':worker,'prior_label':label,'landscape_prior_rows':n,'rows':len(df),'cases':25,'methods':len(SELECTED_METHODS),'checkpoints':CHECKPOINTS,'no_prior_audit_rows':len(ad)}
 (here/f'POSTRUN_VERIFY_{worker}_{tag}.json').write_text(json.dumps(rec,indent=2),encoding='utf-8')
 print(json.dumps(rec,indent=2));print('POSTRUN VERIFY: PASS')
if __name__=='__main__':main()
