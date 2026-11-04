#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np, pandas as pd, yaml
from robustness_provenance import verify_robustness_source, semantic_problem_fingerprint, CONTRACT as ROBUSTNESS_CONTRACT

HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--study-config',default=str(HERE/'worker_state/M5PRO/study_config.yaml')); ap.add_argument('--case',default=None); a=ap.parse_args()
 cfgp=Path(a.study_config).resolve(); cfg=yaml.safe_load(cfgp.read_text()) or {}; st=cfg['study']; N=int(st['landscape_prior_rows'])
 if not 5000<=N<=50000: raise SystemExit(f'VERIFY FAIL prior rows out of supported range: {N}')
 expected={'prior_rows':N,'prior_design_index_start':0,'prior_design_index_end':N-1,'reference_semantics_rows':50000,'budget':128}
 for k,v in expected.items():
  if int(st.get(k,-1))!=int(v): raise SystemExit(f'VERIFY FAIL study.{k}={st.get(k)!r}, expected={v!r}')
 if st.get('no_prior_evaluation')!='self_front_live_calls_only_no_landscape_union': raise SystemExit('VERIFY FAIL no-prior contract')
 contract=json.loads((HERE/'AUTHORITATIVE_GSMOBO25_CONTRACT.json').read_text()); cases=list(contract['case_keys']); methods=list(contract['nsga_methods'])
 if len(cases)!=25 or len(methods)!=11: raise SystemExit('VERIFY FAIL authoritative panel dimensions')
 if a.case is not None and a.case not in cases: raise SystemExit(f'VERIFY FAIL unknown case: {a.case}')
 prep=Path(cfg['paths']['prepared_root']).resolve(); dirs=sorted(d.name for d in prep.iterdir() if d.is_dir() and '__' in d.name)
 if a.case is None:
  if sorted(cases)!=dirs: raise SystemExit('VERIFY FAIL prepared case set is not exact authoritative 25')
 else:
  if a.case not in dirs: raise SystemExit(f'VERIFY FAIL prepared case missing: {a.case}')
 # Code/provenance manifest.
 cm=json.loads((HERE/'PARAMETRIC_CODE_MANIFEST.json').read_text())
 for rel,dig in cm['sha256'].items():
  p=HERE/rel
  if not p.is_file() or sha(p)!=dig: raise SystemExit(f'VERIFY FAIL code/config SHA mismatch: {rel}')
 for case in ([a.case] if a.case is not None else cases):
  d=prep/case; bp=d/'baseline_compact.csv.gz'; pp=d/'problem_config.yaml'; mp=d/'PRIOR_LOCAL_MANIFEST.json'
  if not (bp.is_file() and pp.is_file() and mp.is_file()): raise SystemExit(f'VERIFY FAIL incomplete prepared case: {case}')
  df=pd.read_csv(bp); di=pd.to_numeric(df['design_index'],errors='coerce').to_numpy()
  if len(df)!=N or not np.array_equal(di,np.arange(N)): raise SystemExit(f'VERIFY FAIL exact prefix rows/index: {case}')
  pc=yaml.safe_load(pp.read_text()) or {}; p=pc['problem']; comp=p['compiled_problem']; pm=p.get('parametric_prior',{})
  if int(comp.get('rows',-1))!=50000: raise SystemExit(f'VERIFY FAIL frozen compile rows changed from 50K: {case}')
  if int(pm.get('landscape_prior_rows',-1))!=N or int(pm.get('reference_semantics_rows',-1))!=50000: raise SystemExit(f'VERIFY FAIL parametric metadata: {case}')
  man=json.loads(mp.read_text())
  if int(man.get('landscape_prior_rows',-1))!=N or int(man.get('design_index_end',-2))!=N-1 or not man.get('zero_new_landscape_fba_calls'): raise SystemExit(f'VERIFY FAIL local prior manifest: {case}')
  if man.get('prefix_core_sha256')!=sha(bp): raise SystemExit(f'VERIFY FAIL prepared prefix checksum mismatch: {case}')
  # Revalidate the current authoritative source against the frozen 50K provenance on every build/resume.
  cc=cfg['cases'][case]; runtime_bench=Path(str(cc['source_benchmark'])).expanduser(); canonical_bench=Path(str(cc.get('canonical_source_benchmark',cc['source_benchmark']))).expanduser()
  root=Path(cfg['paths']['gem_hamed_root']).resolve()
  if not runtime_bench.is_absolute(): runtime_bench=(root/runtime_bench).resolve()
  if not canonical_bench.is_absolute(): canonical_bench=(root/canonical_bench).resolve()
  sc=str(cc['source_case']); sb=canonical_bench/'prepared'/sc/'baseline_compact.csv.gz'; sp=canonical_bench/'prepared'/sc/'problem_config.yaml'; eng=canonical_bench/'engine_bridge.py'
  prov=p.get('source_provenance',{}); is_robust=str(cc.get('source_kind','')) in {'robustness','growth_robustness'}
  if is_robust:
   try: cproof=verify_robustness_source(canonical_bench,sc,require_paths_exist=True)
   except Exception as e: raise SystemExit(f'VERIFY FAIL authoritative robustness provenance: {case}: {e}')
   exp=ROBUSTNESS_CONTRACT['cases'][sc]
   if prov.get('source_problem_semantic_sha256')!=exp['semantic_problem_fingerprint_sha256']: raise SystemExit(f'VERIFY FAIL bundled robustness semantic contract: {case}')
   if man.get('canonical_source_problem_semantic_sha256')!=cproof['problem_config_semantic_sha256']: raise SystemExit(f'VERIFY FAIL manifest/canonical robustness semantic fingerprint: {case}')
  else:
   checks=[(sb,prov.get('source_baseline_sha256'),'source baseline'),(sp,prov.get('source_problem_sha256'),'source problem'),(eng,prov.get('source_engine_bridge_original_sha256'),'source engine')]
   for src,expected,label in checks:
    if not src.is_file(): raise SystemExit(f'VERIFY FAIL missing authoritative {label}: {case}: {src}')
    if not expected or sha(src)!=expected: raise SystemExit(f'VERIFY FAIL authoritative {label} SHA mismatch: {case}')
  if not sb.is_file() or man.get('source_baseline_sha256')!=sha(sb): raise SystemExit(f'VERIFY FAIL manifest/source baseline SHA mismatch: {case}')
  # Runtime-copy integrity is separate from canonical scientific provenance.
  rprov=p.get('parametric_runtime_provenance',{})
  rb=runtime_bench/'prepared'/sc/'baseline_compact.csv.gz'; rp=runtime_bench/'prepared'/sc/'problem_config.yaml'; re=runtime_bench/'engine_bridge.py'
  for src,key,label in [(rb,'runtime_baseline_sha256','runtime baseline'),(rp,'runtime_problem_sha256','runtime problem'),(re,'runtime_engine_sha256','runtime engine')]:
   expected=rprov.get(key)
   if not src.is_file() or not expected or sha(src)!=expected: raise SystemExit(f'VERIFY FAIL {label} integrity mismatch: {case}')
  if str(cc.get('source_kind','')) in {'robustness','growth_robustness'}:
   if canonical_bench.resolve()==runtime_bench.resolve(): raise SystemExit(f'VERIFY FAIL robustness canonical/runtime roots unexpectedly identical: {case}')
   if rprov.get('mode')!='relocation_invariant_semantics_plus_deterministic_runtime_rewrite': raise SystemExit(f'VERIFY FAIL robustness runtime provenance mode: {case}')
   try: rproof=verify_robustness_source(runtime_bench,sc,require_paths_exist=True)
   except Exception as e: raise SystemExit(f'VERIFY FAIL runtime robustness provenance: {case}: {e}')
   if rprov.get('runtime_problem_semantic_sha256')!=rproof['problem_config_semantic_sha256']: raise SystemExit(f'VERIFY FAIL runtime robustness semantic fingerprint record: {case}')
  elif canonical_bench.resolve()!=runtime_bench.resolve():
   raise SystemExit(f'VERIFY FAIL non-robustness canonical/runtime roots differ: {case}')
  pdef=Path(str(cc['problem_definition'])).resolve(); expected_pdef=prov.get('problem_definition_sha256')
  if expected_pdef and (not pdef.is_file() or sha(pdef)!=expected_pdef): raise SystemExit(f'VERIFY FAIL problem-definition SHA mismatch: {case}')
  if int(df['feasible'].astype(bool).sum())<int(cfg.get('preflight',{}).get('min_problem_feasible_rows',50)): raise SystemExit(f'VERIFY FAIL insufficient feasible prior rows: {case}')
 print('PARAMETRIC PACKAGE VERIFY PASS' + (f' [case={a.case}]' if a.case else ''))
 print(f'  prior rows          : exact design_index 0..{N-1} ({N} rows)')
 print('  reference semantics : frozen from authoritative 50K')
 print('  cases / methods     : 25 / 11')
 print('  endpoint            : 128')
 print('  landscape FBA calls : ZERO')
 print('  NO-PRIOR            : live self-front only; no N/50K union')
 return 0
if __name__=='__main__': raise SystemExit(main())
