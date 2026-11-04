#!/usr/bin/env python3
"""Fresh-process canary for serialized global surrogate handoff.

This deliberately reloads the exact joblib artifact that run_case.py will load,
then performs predictions on real prepared prior rows. It catches missing-module,
pickle/joblib relocation, schema, and prediction-shape failures before live work.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, pandas as pd, yaml
from generic_problem import feature_bounds, normalize_design


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study-config',required=True);ap.add_argument('--case',required=True);a=ap.parse_args()
    sp=Path(a.study_config).resolve();cfg=yaml.safe_load(sp.read_text()) or {};case=str(a.case)
    if case not in cfg.get('cases',{}):raise SystemExit(f'SURROGATE LOAD CANARY FAIL: unknown case {case}')
    prep=Path(cfg['paths']['prepared_root']).resolve()/case
    out=Path(cfg['paths']['output_root']).resolve()/case/f"seed_{cfg['study']['seed']}"/'surrogate_qc'
    fm=out/'final_global_surrogate.joblib';oo=out/'oof_prior_predictions.npz';sm=out/'surrogate_qc_summary.json'
    for p in (fm,oo,sm,prep/'baseline_compact.csv.gz',prep/'problem_config.yaml'):
        if not p.is_file():raise SystemExit(f'SURROGATE LOAD CANARY FAIL: missing {p}')
    summary=json.loads(sm.read_text(encoding='utf-8'))
    if summary.get('status')!='PASS':raise SystemExit(f'SURROGATE LOAD CANARY FAIL: QC summary is not PASS for {case}')
    parent=yaml.safe_load((prep/'problem_config.yaml').read_text(encoding='utf-8')) or {}
    features,bounds=feature_bounds(parent['problem']);m=int(parent['problem']['compiled_problem']['objective_dim'])
    df=pd.read_csv(prep/'baseline_compact.csv.gz',nrows=4)
    if len(df)<1:raise SystemExit(f'SURROGATE LOAD CANARY FAIL: empty prepared prior for {case}')
    X=normalize_design(df,features,bounds)
    # Critical operation: deserialize in this new Python process.
    model=joblib.load(fm)
    pred,pinf=model.predict(X)
    pred=np.asarray(pred,float);pinf=np.asarray(pinf,float).reshape(-1)
    if pred.shape!=(len(X),m):raise SystemExit(f'SURROGATE LOAD CANARY FAIL: prediction shape={pred.shape}, expected={(len(X),m)} for {case}')
    if pinf.shape!=(len(X),):raise SystemExit(f'SURROGATE LOAD CANARY FAIL: infeasibility shape={pinf.shape}, expected={(len(X),)} for {case}')
    if not np.isfinite(pred).all() or not np.isfinite(pinf).all():raise SystemExit(f'SURROGATE LOAD CANARY FAIL: nonfinite prediction for {case}')
    z=np.load(oo)
    oof=np.asarray(z['oof'],float)
    expected_rows=int(cfg['study']['landscape_prior_rows'])
    if oof.shape!=(expected_rows,m) or not np.isfinite(oof).all():raise SystemExit(f'SURROGATE LOAD CANARY FAIL: OOF shape/finite contract failed for {case}: {oof.shape}')
    print(f'[{case}] SURROGATE SERIALIZATION/HANDOFF CANARY PASS | fresh joblib load + predict | rows={expected_rows} | m={m}')

if __name__=='__main__':main()
