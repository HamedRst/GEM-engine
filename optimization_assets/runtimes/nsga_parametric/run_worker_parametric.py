#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, json, shutil, subprocess, sys, time
from pathlib import Path
import yaml
HERE=Path(__file__).resolve().parent
CFG=HERE/'worker_state/M5PRO/study_config.yaml'

def run(cmd,check=True):
 print('$',' '.join(map(str,cmd)),flush=True); return subprocess.run(list(map(str,cmd)),cwd=str(HERE),check=check)

def lineage_proof(cfg,cache):
 cache=Path(cache); missing=[]; unique=[]
 for c in cfg['cases'].values():
  key=(Path(str(c['source_benchmark'])).name,str(c['source_case']))
  if key not in unique: unique.append(key)
 for bench,sc in unique:
  hits=list((cache/bench/sc).glob('*/raw_exact_cache.csv')) if (cache/bench/sc).is_dir() else []
  if not hits: missing.append((bench,sc))
 if len(unique)!=15: raise RuntimeError(f'expected 15 unique exact-source lineages, found {len(unique)}')
 if missing: raise RuntimeError(f'physical exact-source lineage proof missing: {missing}')
 print('PHYSICAL EXACT-SOURCE LINEAGE PROOF: 15/15 PASS',flush=True)

def main():
 cfg=yaml.safe_load(CFG.read_text()) or {};N=int(cfg['study']['landscape_prior_rows']);cases=list(cfg['cases']);t0=time.time();fail=[]
 print('='*118);print(f'GEM NSGA-II PARAMETRIC PRIOR — N={N} — M5PRO ALL25');print('50K semantics frozen | 11 methods | endpoint 128 | staged certification before adaptive campaign');print('='*118)
 run([sys.executable,HERE/'verify_parametric_package.py','--study-config',CFG]);run([sys.executable,HERE/'verify_surrogate_handoff.py','--study-config',CFG])
 # Exact certification uses a disposable runtime/cache tree so preflight cannot alter production cache accounting.
 cert_state=HERE/'_certification_state'
 if cert_state.exists(): shutil.rmtree(cert_state)
 cert_runtime=cert_state/'runtime_sources'; cert_runtime.mkdir(parents=True,exist_ok=True)
 cert_cfg=copy.deepcopy(cfg);cert_cfg['paths']=dict(cert_cfg['paths']);cert_cfg['paths']['runtime_sources_root']=str(cert_runtime)
 cert_cfgp=cert_state/'study_config.yaml';cert_cfgp.write_text(yaml.safe_dump(cert_cfg,sort_keys=False),encoding='utf-8')
 try:
  for i,c in enumerate(cases,1):
   print(f'EXACT CANARY {i}/25: {c}',flush=True);run([sys.executable,HERE/'exact_backend_canary.py','--study-config',cert_cfgp,'--case',c])
  lineage_proof(cert_cfg,cert_state/'source_exact_cache')
 except Exception:
  print(f'Certification evidence retained at {cert_state}',flush=True)
  raise
 else:
  shutil.rmtree(cert_state)
  print('DISPOSABLE EXACT CERTIFICATION STATE REMOVED; production source cache remains untouched.',flush=True)
 # Stage 1 is valid production work and is retained: all cases/all methods through matched 16-call initialization.
 for i,c in enumerate(cases,1):
  print(f'CERTIFIED WARMUP {i}/25: {c} ->16',flush=True);run([sys.executable,HERE/'run_case.py','--study-config',CFG,'--case',c,'--budget','16'])
 # Stage 2 proves post-warmup adaptive/operator execution on the hardest distinct semantics.
 reps=['cho__product_carbon_oxygen','cglutamicum__growth_robustness','yeast__product_cost']
 for c in reps:
  print(f'ADAPTIVE CERTIFICATION: {c} ->20',flush=True);run([sys.executable,HERE/'run_case.py','--study-config',CFG,'--case',c,'--budget','20'])
 print('ALL CERTIFICATION GATES PASS. CONTINUING/RESUMING PRODUCTION TO 128.',flush=True)
 for i,c in enumerate(cases,1):
  cp=run([sys.executable,HERE/'run_case.py','--study-config',CFG,'--case',c,'--budget','128'],check=False)
  if cp.returncode:
   fail.append({'case':c,'returncode':cp.returncode});print(f'FAIL-CLOSED at {c}; saved progress is resumable.',flush=True);break
 postprocess=[]
 if fail:
  # Best-effort partial diagnostics are useful, but an incomplete campaign can never be marked PASS.
  for cmd,label in [([sys.executable,HERE/'AUDIT_TRUE_NO_PRIOR.py'],'true_no_prior_audit'),([sys.executable,HERE/'aggregate_worker.py','--study-config',CFG],'aggregation')]:
   cp=run(cmd,check=False)
   if cp.returncode: postprocess.append({'stage':label,'returncode':cp.returncode})
 else:
  # A completed 128-call campaign is only PASS if both required post-processing stages also succeed.
  for cmd,label in [([sys.executable,HERE/'AUDIT_TRUE_NO_PRIOR.py'],'true_no_prior_audit'),([sys.executable,HERE/'aggregate_worker.py','--study-config',CFG],'aggregation')]:
   cp=run(cmd,check=False)
   if cp.returncode: postprocess.append({'stage':label,'returncode':cp.returncode})
 status='PASS' if (not fail and not postprocess) else 'FAILED_OR_INCOMPLETE'
 rec={'status':status,'landscape_prior_rows':N,'reference_semantics_rows':50000,'failures':fail,'postprocess_failures':postprocess,'wall_seconds':time.time()-t0}
 (HERE/f'RUN_STATUS_N{N}_M5PRO.json').write_text(json.dumps(rec,indent=2)+'\n')
 if postprocess: print(f'POST-PROCESS FAIL-CLOSED: {postprocess}',flush=True)
 return 0 if status=='PASS' else 3
if __name__=='__main__':raise SystemExit(main())
