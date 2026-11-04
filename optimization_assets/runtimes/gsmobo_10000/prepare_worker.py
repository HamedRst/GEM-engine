#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os, subprocess, sys, hashlib
from pathlib import Path
import pandas as pd, yaml
from selected_panel import SELECTED_METHODS,validate_selected_panel

HERE=Path(__file__).resolve().parent
PRIOR_ROWS=10000
PRIOR_LABEL='10K'
PRIOR_TAG='PRIOR10K'
PACKAGE_WORKER='MACSTUDIO'
PACKAGE_VERSION='GEM_GSMOBO_Systematic_Development_v0.4.2'
CANONICAL_ALGORITHM_SHA='df02dd309c8712525c7e9c7f915ce8a43d4b34a4e7cfeeb0fb271e855e9665b7'

def sha256_file(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def load_yaml(p): return yaml.safe_load(Path(p).read_text(encoding='utf-8')) or {}

def find_root():
    env=os.environ.get('GEM_HAMED_ROOT')
    if env and Path(env).expanduser().is_dir(): return Path(env).expanduser().resolve()
    required=['GEM_5Species_Growth_Product_ResourceCost_Benchmark_v0.1.0','GEM_5Species_Product_Carbon_Oxygen_Efficiency_Benchmark_v0.1.0','GEM_5Species_Growth_Environmental_Robustness_Benchmark_v0.2.0']
    for p in [HERE.parent,*HERE.parents]:
        if all((p/x).is_dir() for x in required): return p.resolve()
    home_candidate=Path.home()/'Desktop'/'GEM-Hamed'
    if home_candidate.is_dir(): return home_candidate.resolve()
    return HERE.parent.resolve()

def discover_source_case(src:Path,prefix:str):
    prep=src/'prepared'; cand=[]
    if not prep.is_dir(): raise RuntimeError(f'Source has no prepared/: {src}')
    for p in prep.iterdir():
        if p.is_dir() and p.name.startswith(prefix) and (p/'baseline_compact.csv.gz').exists() and (p/'problem_config.yaml').exists(): cand.append(p.name)
    if len(cand)!=1: raise RuntimeError(f'Expected exactly one prepared source case in {src.name} with prefix {prefix!r}; found {cand}')
    return cand[0]

def main():
    validate_selected_panel()
    ap=argparse.ArgumentParser(); ap.add_argument('--worker',choices=['M5PRO','MACSTUDIO'],required=True); args=ap.parse_args(); worker=args.worker
    if worker!=PACKAGE_WORKER: raise SystemExit(f'This package is frozen for {PACKAGE_WORKER}; got --worker {worker}')
    if sha256_file(HERE/'gsmobo_algorithms.py')!=CANONICAL_ALGORITHM_SHA: raise SystemExit('ALGORITHM LINEAGE HASH FAIL: gsmobo_algorithms.py differs from canonical v0.3.0')
    root=find_root(); cat=load_yaml(HERE/'problem_catalog.yaml'); mf=HERE/'MANIFEST_ALL25.csv'; jobs=pd.read_csv(mf)
    if len(jobs)!=25 or jobs.case_key.nunique()!=25 or set(jobs.worker.astype(str))!={worker}: raise SystemExit('ALL25 manifest contract FAIL')
    missing=[]
    for pk in sorted(set(jobs['problem'].astype(str))):
        src=root/cat['problems'][pk]['source_benchmark']
        if not src.is_dir(): missing.append(str(src))
    if missing: raise SystemExit('MISSING REQUIRED SOURCE BENCHMARKS BEFORE LIVE RUN:\n'+'\n'.join(sorted(set(missing))))
    state=HERE/'worker_state'/f'{worker}_{PRIOR_TAG}'; prepared=(state/'prepared').resolve(); results=(HERE/f'results_{worker}_{PRIOR_TAG}').resolve(); runtime_sources=(state/'runtime_sources').resolve(); state.mkdir(parents=True,exist_ok=True); prepared.mkdir(parents=True,exist_ok=True); results.mkdir(parents=True,exist_ok=True); runtime_sources.mkdir(parents=True,exist_ok=True)
    cases={}; source_map={}
    for _,j in jobs.iterrows():
        s=str(j.species); pr=str(j.problem); sd=cat['species'][s]; pdx=cat['problems'][pr]; src=(root/pdx['source_benchmark']).resolve(); key=(pdx['source_benchmark'],sd['prefix'])
        if key not in source_map: source_map[key]=discover_source_case(src,str(sd['prefix']))
        source_case=source_map[key]
        cases[str(j.case_key)]={'job_id':str(j.job_id),'worker_assignment':worker,'predicted_weight':float(j.predicted_weight),'species':s,'problem_key':pr,'source_benchmark':pdx['source_benchmark'],'source_kind':pdx['source_kind'],'source_case':source_case,'problem_definition':str((HERE/pdx['definition']).resolve()),'study_config_path':str((state/'study_config.yaml').resolve())}
    cfg={
      'version':'gsmobo-landscape-size-ablation/v0.4.2-selected3','worker':worker,
      'provenance':{'package_version':PACKAGE_VERSION,'experiment':'nested_sobol_prior_size_ablation','prior_label':PRIOR_LABEL,'landscape_prior_rows':PRIOR_ROWS,'reference_semantics_rows':50000,'canonical_gsmobo_algorithms_sha256':CANONICAL_ALGORITHM_SHA,'worker_manifest_sha256':sha256_file(mf),'package_hash_manifest_sha256':sha256_file(HERE/'PACKAGE_HASHES.sha256') if (HERE/'PACKAGE_HASHES.sha256').exists() else 'PRE_FREEZE','core_source_fingerprint':(HERE/'CORE_SOURCE_FINGERPRINT.txt').read_text(encoding='utf-8').strip() if (HERE/'CORE_SOURCE_FINGERPRINT.txt').exists() else 'PRE_FREEZE'},
      'paths':{'gem_hamed_root':str(root),'prepared_root':str(prepared),'output_root':str(results),'runtime_sources_root':str(runtime_sources)},
      'study':{'seed':20260822,'batch_size':4,'screen_total_calls':128,'common_initial_calls':16,'landscape_prior_rows':PRIOR_ROWS,'prior_label':PRIOR_LABEL,'prior_tag':PRIOR_TAG,'reference_semantics_rows':50000,'exact_cache_round_decimals':12,'exact_source_batch_timeout_seconds':30.0,'exact_source_candidate_timeout_seconds':30.0,'exact_source_fallback_timeout_seconds':60.0,'exact_source_fallback_solver':'glpk_exact','min_distance_to_frozen_50k':1e-8,'min_distance_within_method':1e-8,'pool_rows':4096,'gp_subset_rows':1024},
      'preflight':{'required_source_archive_rows':50000,'min_problem_feasible_rows':50,'min_problem_feasible_fraction':0.001},
      'surrogate':{'generic_n_estimators':160,'generic_min_samples_leaf':2,'generic_max_features':1.0,'generic_n_jobs':-1,'generic_oof_folds':5,'infeasible_probability_threshold':0.10},
      'gsmobo_dev':{'initial_calls':16,'min_feasible_live_for_no50k_gp':2,'num_restarts':5,'raw_samples':512,'maxiter':200,'safe_pool_rows':2048,'structure_archive_rows':6000,'structure_pool_rows':640,'structure_top_restarts':10,'structure_refine_maxiter':120,'protected_parent_slots':1,'guided_acq_noninferiority_tolerance':1e-6,'legacy_vb_min_live_feasible':2,'legacy_vb_infeasibility_penalty':0.35,'legacy_vb_gate_tolerance':1e-8,'legacy_structure_pool_rows':256},
      'methods_primary':SELECTED_METHODS,'cases':cases,
    }
    sp=state/'study_config.yaml'; sp.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
    (state/'resolved_source_cases.json').write_text(json.dumps([{'source_benchmark':k[0],'species_prefix':k[1],'source_case':v} for k,v in sorted(source_map.items())],indent=2),encoding='utf-8')
    print(f'GEM-Hamed root: {root}'); print(f'Worker: {worker} | ALL 25 cells | prior={PRIOR_LABEL} ({PRIOR_ROWS} exact Sobol-prefix rows)'); print('Problem semantics remain frozen from the original 50K archive. New landscape FBA calls: ZERO.')
    for i,k in enumerate(cases,1):
        print(f'[{i}/25] {k}',flush=True); subprocess.run([sys.executable,str(HERE/'multi_problem_bridge.py'),'inspect','--study-config',str(sp),'--case-key',k],check=True)
    print(f'WORKER PREPARE PASS: {worker} | 25 cells | prior={PRIOR_LABEL} | ZERO new landscape FBA')
    print(sp)
if __name__=='__main__': main()
