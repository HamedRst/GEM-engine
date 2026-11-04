#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import joblib, numpy as np, yaml
HERE=Path(__file__).resolve().parent

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study-config',default=str(HERE/'worker_state/M5PRO/study_config.yaml'));a=ap.parse_args(); cfg=yaml.safe_load(Path(a.study_config).read_text()) or {};N=int(cfg['study']['landscape_prior_rows']);out=Path(cfg['paths']['output_root']);bad=[]
 for c in cfg['cases']:
  q=out/c/f"seed_{cfg['study']['seed']}"/'surrogate_qc'; fm=q/'final_global_surrogate.joblib'; oo=q/'oof_prior_predictions.npz'; sm=q/'surrogate_qc_summary.json'
  try:
   if not (fm.is_file() and oo.is_file() and sm.is_file()): raise RuntimeError('missing surrogate artifact')
   model=joblib.load(fm); z=np.load(oo); arr=np.asarray(z['oof']); summary=json.loads(sm.read_text())
   if arr.ndim!=2 or arr.shape[0]!=N: raise RuntimeError(f'OOF shape={arr.shape}, expected first dimension {N}')
   if not np.isfinite(arr).all(): raise RuntimeError('OOF has nonfinite entries')
   if int(summary.get('rows',-1))!=N or summary.get('status')!='PASS': raise RuntimeError('summary row/status mismatch')
  except Exception as e: bad.append((c,str(e)))
 if bad: raise SystemExit('SURROGATE HANDOFF VERIFY FAIL: '+repr(bad))
 print('SURROGATE HANDOFF VERIFY PASS')
 print(f'  fitted surrogate models: {len(cfg["cases"])}/{len(cfg["cases"])} loadable')
 print(f'  OOF training rows       : exactly {N}/{N} every case')
 return 0
if __name__=='__main__': raise SystemExit(main())
