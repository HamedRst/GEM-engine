#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent
PACKAGE_WORKER='MACSTUDIO'; PRIOR_LABEL='10K'; PRIOR_TAG='PRIOR10K'; VERSION='GEM_GSMOBO_Systematic_Development_v0.4.2'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(cmd,check=True): print('\n$ '+' '.join(map(str,cmd)),flush=True); return subprocess.run(list(map(str,cmd)),check=check)
def marker_path(worker):return HERE/f'PREFLIGHT_PASS_{worker}_{PRIOR_TAG}.json'
def study_path(worker):return HERE/'worker_state'/f'{worker}_{PRIOR_TAG}'/'study_config.yaml'

def preflight(worker):
 print('\nPHASE 0: process guard + package integrity + offline contracts (ZERO FBA)');run([sys.executable,HERE/'process_guard.py','--worker',worker]);run([sys.executable,HERE/'verify_package.py']);run([sys.executable,HERE/'PACKAGE_SELFTEST.py'])
 print('\nPHASE 1: real BoTorch/GPyTorch runtime smoke (ZERO FBA)');run([sys.executable,HERE/'botorch_preflight.py'])
 print('\nPHASE 2: compile all 25 fixed-semantic problems with reduced Sobol prior (ZERO new landscape FBA)');run([sys.executable,HERE/'prepare_worker.py','--worker',worker]);sp=study_path(worker);cfg=yaml.safe_load(sp.read_text()) or {};cases=list(cfg['cases'])
 if len(cases)!=25:raise SystemExit(f'PRECHECK FAIL: expected 25 cases, got {len(cases)}')
 print('\nPHASE 3: all-row OOF + final surrogate QC on the reduced prior (ZERO live FBA)')
 for i,c in enumerate(cases,1):
  print(f'QC {i}/25: {c}');run([sys.executable,HERE/'surrogate_qc.py','--study-config',sp,'--case',c]);run([sys.executable,HERE/'surrogate_load_canary.py','--study-config',sp,'--case',c])
 print('\nPHASE 4: exact-backend canary for every cell')
 for i,c in enumerate(cases,1): print(f'CANARY {i}/25: {c}');run([sys.executable,HERE/'exact_backend_canary.py','--study-config',sp,'--case',c])
 rec={'status':'PASS','worker':worker,'prior_label':PRIOR_LABEL,'prior_tag':PRIOR_TAG,'study_config':str(sp),'study_config_sha256':sha(sp),'package_hash_manifest_sha256':sha(HERE/'PACKAGE_HASHES.sha256'),'cases':len(cases),'time':time.time()};marker_path(worker).write_text(json.dumps(rec,indent=2),encoding='utf-8');print('\nDEEP PREFLIGHT PASS — no live campaign calls have started.');return sp,cfg

def load_preflight(worker):
 mp=marker_path(worker);sp=study_path(worker)
 if not mp.exists() or not sp.exists():raise SystemExit('LIVE START REFUSED: deep preflight marker/study config missing. Run --mode preflight or full first.')
 rec=json.loads(mp.read_text())
 if rec.get('status')!='PASS':raise SystemExit('LIVE START REFUSED: preflight marker is not PASS')
 if rec.get('study_config_sha256')!=sha(sp):raise SystemExit('LIVE START REFUSED: study_config changed after preflight')
 if rec.get('package_hash_manifest_sha256')!=sha(HERE/'PACKAGE_HASHES.sha256'):raise SystemExit('LIVE START REFUSED: package hash manifest changed after preflight')
 run([sys.executable,HERE/'verify_package.py']);run([sys.executable,HERE/'process_guard.py','--worker',worker]);cfg=yaml.safe_load(sp.read_text()) or {};return sp,cfg

def live(worker,sp,cfg,t0):
 cases=list(cfg['cases']);print('\nALL PREFLIGHTS PASS. BEGINNING LIVE @128 CAMPAIGN.');failures=[]
 for i,c in enumerate(cases,1):
  print('\n'+'#'*112);print(f'{worker} LIVE CELL {i}/25: {c} | prior={PRIOR_LABEL}');print('#'*112)
  p=run([sys.executable,HERE/'run_case.py','--study-config',sp,'--case',c,'--budget','128'],check=False)
  if p.returncode!=0:failures.append({'case':c,'returncode':p.returncode});print('TECHNICAL CELL FAILURE: stopping immediately; completed results remain resumable.',flush=True);break
 agg=run([sys.executable,HERE/'aggregate_worker.py','--study-config',sp],check=False)
 audit_rc=99;post_rc=99
 if agg.returncode==0:
  audit=run([sys.executable,HERE/'AUDIT_NO_PRIOR_SELF_FRONT.py','--study-config',sp],check=False);audit_rc=int(audit.returncode)
  if audit_rc==0:
   post=run([sys.executable,HERE/'POSTRUN_VERIFY.py','--study-config',sp],check=False);post_rc=int(post.returncode)
 csv=HERE/f'ALL_RESULTS_{worker}_{PRIOR_TAG}_SELECTED3.csv';audit_csv=HERE/f'NO_PRIOR_SELF_FRONT_{PRIOR_TAG}_AUDIT.csv'
 complete=(not failures and agg.returncode==0 and audit_rc==0 and post_rc==0 and csv.exists() and audit_csv.exists())
 status={'worker':worker,'prior_label':PRIOR_LABEL,'package_version':VERSION,'status':'COMPLETE' if complete else 'FAILED_OR_INCOMPLETE','failures':failures,'aggregation_returncode':int(agg.returncode),'audit_returncode':audit_rc,'postrun_verify_returncode':post_rc,'wall_seconds':time.time()-t0,'consolidated_csv':str(csv) if csv.exists() else None,'no_prior_audit_csv':str(audit_csv) if audit_csv.exists() else None,'resume_instruction':'Run RUN_THIS.command again. Completed calls are never deleted.'};(HERE/f'RUN_STATUS_{worker}_{PRIOR_TAG}.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print('\n'+json.dumps(status,indent=2))
 if not complete:raise SystemExit(3)
 print(f'\nSUCCESS — upload these files to ChatGPT:\n{csv}\n{audit_csv}')

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--worker',choices=['M5PRO','MACSTUDIO'],required=True);ap.add_argument('--mode',choices=['full','preflight','live'],default='full');a=ap.parse_args();worker=a.worker
 if worker!=PACKAGE_WORKER:raise SystemExit(f'This package is frozen for {PACKAGE_WORKER}; got {worker}')
 t0=time.time();print('='*112);print(f'{VERSION} — LANDSCAPE-SIZE ABLATION | prior={PRIOR_LABEL} | ALL 25 CELLS');print(f'WORKER={worker} ONLY | 3 chosen heuristic implementations + 2 scientific references; canonical gsMOBO algorithm source unchanged | seed=20260822 | endpoint=128 | mode={a.mode}');print('='*112)
 if a.mode in ('full','preflight'):
  sp,cfg=preflight(worker)
  if a.mode=='preflight':return
 else:sp,cfg=load_preflight(worker)
 if a.mode in ('full','live'):live(worker,sp,cfg,t0)
if __name__=='__main__':main()
