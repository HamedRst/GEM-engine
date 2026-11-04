#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import hashlib
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys
import tempfile
import time

import pandas as pd
import numpy as np
import yaml

from generic_problem import load_yaml, compile_frozen_archive, compile_live_exact, feature_bounds, normalize_design


def _sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def load_cfg(path): return yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}


def _adapt_robustness_semantics(df: pd.DataFrame, source_kind: str, *, live: bool) -> pd.DataFrame:
    """Expose the intended oxygen-limitation retention metric from the environmental-robustness source.

    The frozen environmental-robustness benchmark stores the oxygen scenario as
    ``prior_oxygen_stressed_growth`` in its 50K prior and ``oxygen_stressed_growth``
    in live exact output.  Round-1 P3 is specifically Growth × Oxygen-Limitation
    Robustness, whose second objective is stressed/nominal growth retention.

    This adapter creates the historical generic name ``stress_tolerance`` without
    changing the source benchmark or using any species-specific branch.  It fails
    closed if the required oxygen channel is absent.
    """
    if str(source_kind) != 'robustness':
        return df
    out=df.copy()
    if 'stress_tolerance' in out.columns:
        return out
    stressed='oxygen_stressed_growth' if live else 'prior_oxygen_stressed_growth'
    required={'maximum_growth',stressed}
    missing=sorted(required-set(out.columns))
    if missing:
        raise RuntimeError(
            f'Robustness source cannot expose oxygen-limitation retention; missing={missing}; '
            f'available={list(out.columns)[:120]}'
        )
    nominal=pd.to_numeric(out['maximum_growth'],errors='coerce').to_numpy(float)
    oxygen=pd.to_numeric(out[stressed],errors='coerce').to_numpy(float)
    ratio=np.full(len(out),np.nan,float)
    ok=np.isfinite(nominal)&(nominal>1e-12)&np.isfinite(oxygen)
    ratio[ok]=np.clip(oxygen[ok]/nominal[ok],0.0,1.0+1e-8)
    out['stress_tolerance']=ratio
    return out


def _source_cache_paths(cfg:dict, c:dict, src:Path, source_problem_path:Path) -> tuple[Path,Path]:
    """Hash-qualified raw exact cache shared by generic problems on one worker.

    The cache stores only source-backend exact outputs before generic objective
    compilation.  Therefore Growth×Product, Product×Cost, and constrained Product
    may reuse the same biological solve for the same source/species/design while
    still compiling different objective/constraint semantics afterward.
    """
    original_bridge=src/'engine_bridge.py'
    token='|'.join([
        str(c.get('source_case','')),
        _sha256(source_problem_path),
        _sha256(original_bridge),
    ])
    cache_id=hashlib.sha256(token.encode('utf-8')).hexdigest()[:20]
    runtime_sources=Path(cfg['paths']['runtime_sources_root']).resolve()
    root=runtime_sources.parent/'source_exact_cache'/src.name/str(c['source_case'])/cache_id
    return root/'raw_exact_cache.csv', root/'cache_meta.json'


def _raw_key(x01,decimals:int):
    return tuple(np.round(np.asarray(x01,float),int(decimals)).tolist())


def _load_raw_cache(path:Path, xcols:list[str], decimals:int) -> dict:
    if not path.exists(): return {}
    df=pd.read_csv(path)
    if not all(c in df.columns for c in xcols):
        raise RuntimeError(f'Raw source exact cache has incompatible key columns: {path}')
    out={}
    for _,r in df.iterrows():
        k=_raw_key([r[c] for c in xcols],decimals)
        if k in out: raise RuntimeError(f'Duplicate key in raw source exact cache: {path}')
        out[k]=r.to_dict()
    return out


def _write_raw_cache_atomic(path:Path, records:dict):
    path.parent.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(list(records.values()))
    tmp=path.with_suffix(path.suffix+'.tmp')
    df.to_csv(tmp,index=False)
    tmp.replace(path)

def resolve_case(cfg,case_key):
    if case_key not in cfg.get('cases',{}): raise KeyError(f'Unknown case {case_key}')
    c=dict(cfg['cases'][case_key] or {})
    required=['source_benchmark','source_case','problem_definition']
    miss=[x for x in required if not c.get(x)]
    if miss: raise RuntimeError(f'{case_key}: missing case fields {miss}')
    return c

def source_root(cfg,c):
    parent=Path(cfg['paths']['gem_hamed_root']).expanduser().resolve()
    p=parent/str(c['source_benchmark'])
    if not p.is_dir(): raise FileNotFoundError(f'Source benchmark missing: {p}')
    return p

def source_prepared(src,source_case):
    b=src/'prepared'/source_case/'baseline_compact.csv.gz'; p=src/'prepared'/source_case/'problem_config.yaml'
    if not b.exists(): raise FileNotFoundError(b)
    if not p.exists(): raise FileNotFoundError(p)
    return b,p

def active_spec(study_config,c):
    p=(Path(study_config).resolve().parent/str(c['problem_definition'])).resolve()
    if not p.exists(): raise FileNotFoundError(p)
    return load_yaml(p),p

def _compat_bridge(src:Path, cfg:dict, case:dict) -> Path:
    """Create a package-local exact-bridge copy, integrating known solver compatibility before live work.

    The source benchmark is read-only. Only the Growth/Product source has one known GLPK
    stage-2 basis-state hazard. If the source already contains the reset, the copy is
    unchanged. If it matches the known pre-fix block, we integrate the reset into the
    local copy. Unknown lineages fail closed rather than receiving a blind patch.
    """
    source=src/'engine_bridge.py'
    if not source.exists(): raise FileNotFoundError(source)
    runtime_root=Path(cfg['paths']['runtime_sources_root']).resolve()/Path(str(case['source_benchmark'])).name
    runtime_root.mkdir(parents=True,exist_ok=True); target=runtime_root/'engine_bridge.py'
    text=source.read_text(encoding='utf-8')
    if str(case.get('source_kind',''))=='growth_product':
        helper='def _reset_glpk_basis(model):'
        call='_reset_glpk_basis(m) if "glpk" in solver_iface else None'
        if helper in text and call in text:
            patched=text
        else:
            old=('m.objective=m.reactions.get_by_id(desc["product"]); '
                 'm.objective_direction="max"; '
                 's2=m.optimize(); st2=str(s2.status); '
                 'ok2=st2.lower() in desc["feasible_statuses"]')
            anchor='def _solve_one(desc,xphys):\n'
            if text.count(old)!=1 or text.count(anchor)!=1:
                raise RuntimeError('Growth source engine_bridge.py is neither known basis-reset lineage nor known pre-fix lineage; refusing unsafe compatibility transform')
            h='''def _reset_glpk_basis(model):\n    """Reset GLPK simplex basis only; mathematical LP remains unchanged."""\n    try:\n        import swiglpk as glp\n    except Exception as exc:\n        raise RuntimeError("GLPK basis reset requested but swiglpk is unavailable") from exc\n    glp.glp_std_basis(model.solver.problem)\n\n'''
            new=('m.objective=m.reactions.get_by_id(desc["product"]); '
                 'm.objective_direction="max"; '
                 'solver_iface=str(getattr(m.solver.interface,"__name__","")).lower(); '
                 '(_reset_glpk_basis(m) if "glpk" in solver_iface else None); '
                 's2=m.optimize(); st2=str(s2.status); '
                 'ok2=st2.lower() in desc["feasible_statuses"]')
            patched=text.replace(anchor,h+anchor,1).replace(old,new,1)
    else:
        patched=text
    if (not target.exists()) or target.read_text(encoding='utf-8')!=patched:
        target.write_text(patched,encoding='utf-8')
    py_compile.compile(str(target),doraise=True)
    return target

def inspect_case(study_config,case_key):
    cfg=load_cfg(study_config); c=resolve_case(cfg,case_key); src=source_root(cfg,c); bpath,ppath=source_prepared(src,c['source_case']); spec,spec_path=active_spec(study_config,c)
    source_df=pd.read_csv(bpath,compression='infer'); source_df=_adapt_robustness_semantics(source_df,str(c.get('source_kind','')),live=False); required=int(cfg.get('preflight',{}).get('required_archive_rows',50000))
    if len(source_df)!=required: raise RuntimeError(f'{case_key}: source frozen archive rows={len(source_df)} required={required}')
    source_problem=load_yaml(ppath); baseline,meta=compile_frozen_archive(source_df,source_problem,spec)
    # Semantic validation for any objective that is declared deterministic from
    # design but has a historical/source normalized metric available.  This is a
    # zero-FBA equality check and prevents a generic problem definition from
    # silently changing PCO carbon/oxygen or resource-cost semantics.
    semantic=[]
    for obj in meta.get('objectives',[]):
        metric=obj.get('validation_normalized_metric')
        if not metric: continue
        if metric not in source_df.columns:
            raise RuntimeError(f'{case_key}: deterministic objective {obj.get("name")} requires validation metric {metric!r}, missing from source archive')
        col=obj['norm_column']; a=pd.to_numeric(baseline[col],errors='coerce').to_numpy(float); b=pd.to_numeric(source_df[metric],errors='coerce').to_numpy(float)
        ok=np.isfinite(a)&np.isfinite(b)
        if not ok.any(): raise RuntimeError(f'{case_key}: no finite rows for deterministic semantic validation {obj.get("name")} vs {metric}')
        err=float(np.max(np.abs(a[ok]-b[ok])))
        if err>5e-10: raise RuntimeError(f'{case_key}: deterministic objective semantic mismatch {obj.get("name")} vs {metric}: max_abs_error={err:.3e}')
        semantic.append({'objective':obj.get('name'),'source_metric':metric,'max_abs_error':err,'rows':int(ok.sum())})
    outdir=Path(cfg['paths']['prepared_root']).resolve()/case_key; outdir.mkdir(parents=True,exist_ok=True); local_b=outdir/'baseline_compact.csv.gz'; baseline.to_csv(local_b,index=False,compression='gzip')
    bridge=_compat_bridge(src,cfg,c)
    source_prov={'source_benchmark_name':src.name,'source_case':str(c['source_case']),'source_baseline_sha256':_sha256(bpath),'source_problem_sha256':_sha256(ppath),'source_engine_bridge_original_sha256':_sha256(src/'engine_bridge.py'),'source_engine_bridge_runtime_sha256':_sha256(bridge),'problem_definition_sha256':_sha256(spec_path)}
    pp=dict(source_problem.get('problem',source_problem)); pp.update({'case_name':case_key,'case_key':case_key,'baseline_table':str(local_b),'compiled_problem':meta,'objective_names':meta['objective_names'],'objective_directions':meta['objective_directions'],'objective_norm_columns':meta['objective_norm_columns'],'objective_raw_columns':meta['objective_raw_columns'],'objective_dim':meta['objective_dim'],'hv_reference_normalized':meta['hv_reference_normalized'],'general_problem_definition':str(spec_path),'source_problem_config':str(ppath),'source_benchmark':str(src),'source_case':c['source_case'],'source_provenance':source_prov})
    (outdir/'problem_config.yaml').write_text(yaml.safe_dump({'problem':pp},sort_keys=False),encoding='utf-8')
    report={'status':'PASS','exact_replay_qc_pass':True,'rows':len(baseline),'source_feasible_rows':meta['source_feasible_rows'],'problem_feasible_rows':meta['problem_feasible_rows'],'dimension':len(meta['features']),'problem_id':meta['problem_id'],'problem_hash':meta['problem_hash'],'objective_names':meta['objective_names'],'objective_directions':meta['objective_directions'],'objective_dim':meta['objective_dim'],'hv_reference_normalized':meta['hv_reference_normalized'],'constraints':meta['constraints'],'deterministic_semantic_validation':semantic,'archive_relabel_fba_solves':0,'source_baseline':str(bpath),'source_problem':str(ppath),'source_engine_bridge_original':str(src/'engine_bridge.py'),'source_engine_bridge_runtime':str(bridge),**source_prov}
    minrows=int(cfg.get('preflight',{}).get('min_problem_feasible_rows',50)); minfrac=float(cfg.get('preflight',{}).get('min_problem_feasible_fraction',0.001)); frac=meta['problem_feasible_rows']/max(1,len(baseline)); report['problem_feasible_fraction']=frac
    if meta['problem_feasible_rows']<minrows or frac<minfrac: raise RuntimeError(f'{case_key}: generic problem is degenerate: feasible={meta["problem_feasible_rows"]}/{len(baseline)} ({frac:.4g})')
    (outdir/'landscape_preflight.json').write_text(json.dumps(report,indent=2),encoding='utf-8'); return report

def _source_evaluate_cmd(runtime_bridge:Path, src:Path, c:dict, source_problem:dict, run_dir:Path, model_path:Path, proposal:Path, raw_path:Path):
    """Build the exact-source command from the source benchmark's declared contract.

    Growth/Product and PCO sources expose the standard study-config/case-key CLI.
    The frozen robustness source predates that generic interface and exposes
    stress-factor/stress-reaction directly.  This dispatch is based only on the
    catalogued source kind, never on species or optimization objective count.
    """
    base=[sys.executable,str(runtime_bridge),'evaluate','--run-dir',str(run_dir),'--model-path',str(model_path),'--proposal',str(proposal),'--output',str(raw_path)]
    kind=str(c.get('source_kind',''))
    if kind=='robustness':
        source_study=src/'study_config.yaml'
        if not source_study.exists(): raise FileNotFoundError(source_study)
        return base+['--study-config',str(source_study),'--case-key',str(c['source_case'])]
    source_study=src/'study_config.yaml'
    if not source_study.exists(): raise FileNotFoundError(source_study)
    return base+['--study-config',str(source_study),'--case-key',str(c['source_case'])]



def _exact_solver_policy(cfg: dict) -> dict:
    # Uniform fail-closed timeout/fallback policy for source exact evaluation.
    study=dict(cfg.get('study',{}) or {})
    out={
        'batch_timeout_seconds':float(study.get('exact_source_batch_timeout_seconds',30.0)),
        'candidate_timeout_seconds':float(study.get('exact_source_candidate_timeout_seconds',30.0)),
        'fallback_timeout_seconds':float(study.get('exact_source_fallback_timeout_seconds',600.0)),
        'fallback_solver':str(study.get('exact_source_fallback_solver','glpk_exact')).strip(),
    }
    if any(out[k] <= 0 for k in ('batch_timeout_seconds','candidate_timeout_seconds','fallback_timeout_seconds')):
        raise RuntimeError(f'Invalid exact source timeout policy: {out}')
    if out['fallback_solver'] != 'glpk_exact':
        raise RuntimeError(f"This frozen hotfix requires exact_source_fallback_solver='glpk_exact'; got {out['fallback_solver']!r}")
    return out


_FORCED_SOLVER_WRAPPER = r"""\
import runpy, sys
solver=sys.argv[1]
bridge=sys.argv[2]
bridge_args=sys.argv[3:]
import cobra
orig_read=cobra.io.read_sbml_model
def patched_read(*args, **kwargs):
    model=orig_read(*args, **kwargs)
    model.solver=solver
    return model
cobra.io.read_sbml_model=patched_read
try:
    cobra.Configuration().solver=solver
except Exception:
    pass
sys.argv=[bridge, *bridge_args]
runpy.run_path(bridge, run_name='__main__')
"""


def _run_source_command(cmd, *, src: Path, env: dict, timeout_seconds: float, force_solver: str | None = None):
    actual=list(map(str,cmd))
    if force_solver is not None:
        if len(actual) < 2:
            raise RuntimeError('Source exact command is malformed')
        actual=[actual[0],'-c',_FORCED_SOLVER_WRAPPER,str(force_solver),*actual[1:]]
    t0=time.monotonic()
    try:
        proc=subprocess.run(actual,text=True,capture_output=True,cwd=str(src),env=env,timeout=float(timeout_seconds))
        return proc,False,time.monotonic()-t0
    except subprocess.TimeoutExpired:
        return None,True,time.monotonic()-t0


def _require_source_success(proc, raw_path: Path, context: str) -> pd.DataFrame:
    if proc is None:
        raise RuntimeError(f'{context}: internal error: missing subprocess result')
    if proc.returncode!=0:
        raise RuntimeError(
            f'{context}: source exact bridge failed rc={proc.returncode}.\\n'
            f'STDOUT:\\n{proc.stdout}\\nSTDERR:\\n{proc.stderr}'
        )
    if not raw_path.exists():
        raise RuntimeError(f'{context}: source exact bridge returned success but no output')
    return pd.read_csv(raw_path)


def _evaluate_source_missing(runtime_bridge: Path, src: Path, c: dict, source_problem: dict,
                             run_dir: Path, model_path: Path, source_rows: pd.DataFrame,
                             cfg: dict) -> pd.DataFrame:
    # Fast path: source-default batch. Recovery: same candidates isolated; only
    # an individually timed-out candidate is retried with glpk_exact.
    policy=_exact_solver_policy(cfg)
    env=os.environ.copy(); env['PYTHONPATH']=str(src)+os.pathsep+env.get('PYTHONPATH','')

    with tempfile.TemporaryDirectory(prefix='gem_gsmobo_dev_batch_') as td:
        td=Path(td); raw_path=td/'source_exact.csv'; source_prop=td/'source_proposal.csv'
        source_rows.to_csv(source_prop,index=False)
        cmd=_source_evaluate_cmd(runtime_bridge,src,c,source_problem,run_dir,model_path,source_prop,raw_path)
        proc,timed_out,wall=_run_source_command(
            cmd,src=src,env=env,timeout_seconds=policy['batch_timeout_seconds']
        )
        if not timed_out:
            fresh=_require_source_success(proc,raw_path,'Source exact batch')
            if len(fresh)!=len(source_rows):
                raise RuntimeError(f'Source exact bridge row mismatch {len(fresh)} != {len(source_rows)}')
            fresh=fresh.copy()
            fresh['source_solver_route']='default_batch'
            fresh['source_solver_fallback_used']=False
            fresh['source_solver_isolated_retry']=False
            fresh['source_primary_timeout_seconds']=float(policy['batch_timeout_seconds'])
            fresh['source_fallback_solver']=''
            fresh['source_exact_attempt_wall_seconds']=float(wall)
            return fresh

    print(
        f"[exact-policy] source batch exceeded {policy['batch_timeout_seconds']:.0f}s; "
        f"isolating {len(source_rows)} candidate(s) with fail-closed per-candidate retries",
        flush=True,
    )

    parts=[]
    for j in range(len(source_rows)):
        one=source_rows.iloc[[j]].reset_index(drop=True)
        pid=str(one.iloc[0].get('proposal_id',f'candidate_{j+1}'))
        with tempfile.TemporaryDirectory(prefix=f'gem_gsmobo_dev_candidate_{j+1:03d}_') as td:
            td=Path(td); raw_path=td/'source_exact.csv'; source_prop=td/'source_proposal.csv'
            one.to_csv(source_prop,index=False)
            cmd=_source_evaluate_cmd(runtime_bridge,src,c,source_problem,run_dir,model_path,source_prop,raw_path)
            proc,timed_out,primary_wall=_run_source_command(
                cmd,src=src,env=env,timeout_seconds=policy['candidate_timeout_seconds']
            )
            if not timed_out:
                row=_require_source_success(proc,raw_path,f'{pid} default isolated exact')
                if len(row)!=1: raise RuntimeError(f'{pid}: isolated source exact returned {len(row)} rows')
                row=row.copy()
                row['source_solver_route']='default_isolated_after_batch_timeout'
                row['source_solver_fallback_used']=False
                row['source_solver_isolated_retry']=True
                row['source_primary_timeout_seconds']=float(policy['candidate_timeout_seconds'])
                row['source_fallback_solver']=''
                row['source_exact_attempt_wall_seconds']=float(primary_wall)
                parts.append(row)
                continue

            print(
                f"[exact-policy] {pid}: default solver exceeded "
                f"{policy['candidate_timeout_seconds']:.0f}s; retrying same candidate with "
                f"{policy['fallback_solver']}",
                flush=True,
            )
            fallback_raw=td/'source_exact_fallback.csv'
            fallback_cmd=_source_evaluate_cmd(runtime_bridge,src,c,source_problem,run_dir,model_path,source_prop,fallback_raw)
            fproc,ftimed_out,fallback_wall=_run_source_command(
                fallback_cmd,src=src,env=env,
                timeout_seconds=policy['fallback_timeout_seconds'],
                force_solver=policy['fallback_solver'],
            )
            if ftimed_out:
                # NSGA_TIMEOUT_POLICY_HOTFIX_V011: terminal timeout is censored, counted, and continued.
                timeout_rec=one.iloc[0].to_dict()
                timeout_rec.update({
                    'maximum_growth':0.0,
                    'product_flux':0.0,
                    'growth_constrained_product_flux':0.0,
                    'robust_growth':0.0,
                    'robustness_retention':0.0,
                    'stress_tolerance':0.0,
                    'carbon_efficiency':0.0,
                    'oxygen_efficiency':0.0,
                    'resource_efficiency':0.0,
                    'normalized_resource_cost':1.0,
                    'carbon_demand':0.0,
                    'oxygen_demand':0.0,
                    'status':'technical_timeout',
                    'stored_status':'technical_timeout',
                    'feasible':False,
                    'solver_feasible':False,
                    'growth_viable':False,
                    'fba_solves':0,
                    'solver_wall_seconds':float(primary_wall+fallback_wall),
                    'source_timeout_censored':True,
                    'source_solver_route':f'default_timeout_then_{policy["fallback_solver"]}_timeout',
                    'source_solver_fallback_used':True,
                    'source_solver_isolated_retry':True,
                    'source_primary_timeout_seconds':float(policy['candidate_timeout_seconds']),
                    'source_fallback_timeout_seconds':float(policy['fallback_timeout_seconds']),
                    'source_fallback_solver':policy['fallback_solver'],
                    'source_exact_attempt_wall_seconds':float(primary_wall+fallback_wall),
                })
                parts.append(pd.DataFrame([timeout_rec]))
                print(
                    f'[exact-policy] {pid}: {policy["fallback_solver"]} also exceeded '
                    f'{policy["fallback_timeout_seconds"]:.0f}s; recording technical_timeout, '
                    'consuming this logical call, and continuing with no replacement',
                    flush=True,
                )
                continue
            row=_require_source_success(
                fproc,fallback_raw,f'{pid} {policy["fallback_solver"]} fallback exact'
            )
            if len(row)!=1: raise RuntimeError(f'{pid}: fallback source exact returned {len(row)} rows')
            row=row.copy()
            row['source_solver_route']=f'default_timeout_then_{policy["fallback_solver"]}'
            row['source_solver_fallback_used']=True
            row['source_solver_isolated_retry']=True
            row['source_primary_timeout_seconds']=float(policy['candidate_timeout_seconds'])
            row['source_fallback_solver']=policy['fallback_solver']
            row['source_exact_attempt_wall_seconds']=float(primary_wall+fallback_wall)
            parts.append(row)

    fresh=pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
    if len(fresh)!=len(source_rows):
        raise RuntimeError(f'Isolated source exact row mismatch {len(fresh)} != {len(source_rows)}')
    return fresh

def evaluate(study_config,case_key,proposal,output):
    cfg=load_cfg(study_config); c=resolve_case(cfg,case_key); src=source_root(cfg,c); _,ppath=source_prepared(src,c['source_case']); source_problem=load_yaml(ppath).get('problem',load_yaml(ppath)); spec,_=active_spec(study_config,c)
    runtime_bridge=_compat_bridge(src,cfg,c); source_study=src/'study_config.yaml'
    # Resolve paths portably. Prepared configs may contain absolute paths from a
    # different Mac/user account, so never require those absolute paths to survive
    # transfer. Prefer an existing stored path; otherwise rebuild from the source
    # benchmark's own study config / standard inputs layout.
    run_dir=Path(str(source_problem.get('run_dir',''))).expanduser()
    model_path=Path(str(source_problem.get('model_path',''))).expanduser()
    source_cfg=load_cfg(source_study); source_case_cfg=dict((source_cfg.get('cases',{}) or {}).get(str(c['source_case']),{}) or {})
    if not run_dir.exists():
        rr=source_case_cfg.get('run_dir')
        if rr:
            rp=Path(str(rr)).expanduser(); run_dir=rp if rp.is_absolute() else (src/rp).resolve()
        if not run_dir.exists():
            guess=src/'inputs'/'runs'/str(c['source_case']); run_dir=guess.resolve()
    if not model_path.exists():
        mp=source_case_cfg.get('model_path')
        if mp:
            xp=Path(str(mp)).expanduser(); model_path=xp if xp.is_absolute() else (src/xp).resolve()
        if not model_path.exists():
            globpat=source_case_cfg.get('model_glob') or source_problem.get('model_glob')
            candidates=[]
            if globpat:
                candidates=list((src/'inputs'/'models').glob(str(globpat)))+list(src.glob(str(globpat)))
            if not candidates:
                candidates=list((src/'inputs'/'models').glob(str(c['source_case'])+'.*'))
            candidates=[x for x in candidates if x.is_file()]
            if len(candidates)==1: model_path=candidates[0].resolve()
            elif len(candidates)>1:
                candidates.sort(key=lambda x:(x.suffix.lower() not in {'.xml','.sbml'},len(x.name)))
                model_path=candidates[0].resolve()
    if not run_dir.exists(): raise FileNotFoundError(f'{case_key}: portable source run_dir resolution failed: {run_dir}')
    if not model_path.exists(): raise FileNotFoundError(f'{case_key}: portable source model_path resolution failed: {model_path}')
    proposal=Path(proposal).resolve(); output=Path(output).resolve(); prop=pd.read_csv(proposal)

    # Source-level exact reuse across generic problem formulations. This is a
    # computational cache only: the generic problem is recompiled every time,
    # so Product×Cost cannot inherit Growth×Product objective semantics, etc.
    features,bounds=feature_bounds(source_problem)
    X01=normalize_design(prop,features,bounds)
    decimals=int(cfg.get('study',{}).get('exact_cache_round_decimals',12))
    xcols=[f'x01__{f}' for f in features]
    cache_path,meta_path=_source_cache_paths(cfg,c,src,ppath)
    meta_expected={
        'source_benchmark':src.name,'source_case':str(c['source_case']),
        'source_problem_sha256':_sha256(ppath),'source_engine_bridge_original_sha256':_sha256(src/'engine_bridge.py'),
        'features':features,'exact_cache_round_decimals':decimals,
    }
    if meta_path.exists():
        old=json.loads(meta_path.read_text(encoding='utf-8'))
        if old!=meta_expected:
            raise RuntimeError(f'Raw source exact cache metadata mismatch; refusing stale reuse: {meta_path}')
    else:
        meta_path.parent.mkdir(parents=True,exist_ok=True); meta_path.write_text(json.dumps(meta_expected,indent=2),encoding='utf-8')
    cache=_load_raw_cache(cache_path,xcols,decimals)
    ordered=[None]*len(prop); missing=[]
    for i,x in enumerate(X01):
        rec=cache.get(_raw_key(x,decimals))
        if rec is None: missing.append(i)
        else:
            rec=dict(rec); rec['source_raw_cache_hit']=True
            rec['source_original_fba_solves']=int(float(rec.get('source_original_fba_solves',rec.get('fba_solves',0)) or 0))
            rec['source_original_solver_wall_seconds']=float(rec.get('source_original_solver_wall_seconds',rec.get('solver_wall_seconds',0.0)) or 0.0)
            rec['fba_solves']=0; rec['solver_wall_seconds']=0.0; ordered[i]=rec

    if missing:
        psub=prop.iloc[np.asarray(missing,int)].reset_index(drop=True)
        fresh=_evaluate_source_missing(runtime_bridge,src,c,source_problem,run_dir,model_path,psub,cfg)
        if len(fresh)!=len(missing): raise RuntimeError(f'Source exact bridge row mismatch {len(fresh)} != {len(missing)}')
        for j,ii in enumerate(missing):
            rec=fresh.iloc[j].to_dict(); orig_solves=int(float(rec.get('fba_solves',0) or 0)); orig_wall=float(rec.get('solver_wall_seconds',0.0) or 0.0)
            for k,f in enumerate(features): rec[f]=float(prop.iloc[ii][f]); rec[xcols[k]]=float(X01[ii,k])
            rec['source_original_fba_solves']=orig_solves; rec['source_original_solver_wall_seconds']=orig_wall
            # Store the true source result, not the current-call cache marker.
            stored=dict(rec); stored.pop('source_raw_cache_hit',None); cache[_raw_key(X01[ii],decimals)]=stored
            current=dict(rec); current['source_raw_cache_hit']=False; ordered[ii]=current
        _write_raw_cache_atomic(cache_path,cache)
    raw=pd.DataFrame(ordered)
    raw=_adapt_robustness_semantics(raw,str(c.get('source_kind','')),live=True)
    print(f'[source-cache] {case_key}: hits={len(prop)-len(missing)}/{len(prop)} | physical_source_evals={len(missing)}',flush=True)
    prepared=load_yaml(Path(cfg['paths']['prepared_root']).resolve()/case_key/'problem_config.yaml')
    # NSGA_TIMEOUT_POLICY_HOTFIX_V011:
    # Terminal timeouts are censored technical evaluations, not biological solutions.
    _timeout_series=raw.get('source_timeout_censored',pd.Series(False,index=raw.index))
    _timeout_mask=pd.Series(_timeout_series,index=raw.index).fillna(False).astype(bool).to_numpy()
    if _timeout_mask.any():
        _good_idx=np.flatnonzero(~_timeout_mask)
        _compiled_by_index={}
        _compiled_columns=[]
        if len(_good_idx):
            _good_out=compile_live_exact(
                raw.iloc[_good_idx].reset_index(drop=True),
                prop.iloc[_good_idx].reset_index(drop=True),
                prepared,
                spec,
            )
            _compiled_columns=list(_good_out.columns)
            for _j,_orig_i in enumerate(_good_idx):
                _compiled_by_index[int(_orig_i)]=_good_out.iloc[_j].to_dict()

        _problem=prepared.get('problem',prepared) if isinstance(prepared,dict) else {}
        _objective_names=list(_problem.get('objective_names',[]) or [])
        _rows=[]
        for _i in range(len(prop)):
            if not bool(_timeout_mask[_i]):
                _rows.append(_compiled_by_index[int(_i)])
                continue

            _r={_c:np.nan for _c in _compiled_columns}
            for _c,_v in prop.iloc[_i].to_dict().items():
                _r[_c]=_v

            _r.update({
                'maximum_growth':0.0,
                'product_flux':0.0,
                'growth_constrained_product_flux':0.0,
                'robust_growth':0.0,
                'robustness_retention':0.0,
                'stress_tolerance':0.0,
                'carbon_efficiency':0.0,
                'oxygen_efficiency':0.0,
                'resource_efficiency':0.0,
                'normalized_resource_cost':1.0,
                'carbon_demand':0.0,
                'oxygen_demand':0.0,
                'feasible':False,
                'solver_feasible':False,
                'growth_viable':False,
                'status':'technical_timeout',
                'stored_status':'technical_timeout',
            })

            for _name in _objective_names:
                _r[f'objective_norm__{_name}']=0.0
                _r[f'objective_raw__{_name}']=(1.0 if _name=='normalized_resource_cost' else 0.0)
            for _c in list(_r):
                _lc=str(_c).lower()
                if _lc.startswith('objective_norm__') or (_lc.startswith('exact_') and _lc.endswith('_norm')):
                    _r[_c]=0.0

            _rawr=raw.iloc[_i].to_dict()
            for _c in (
                'proposal_id',
                'source_timeout_censored',
                'source_solver_route',
                'source_solver_fallback_used',
                'source_solver_isolated_retry',
                'source_primary_timeout_seconds',
                'source_fallback_timeout_seconds',
                'source_fallback_solver',
                'source_exact_attempt_wall_seconds',
            ):
                if _c in _rawr:
                    _r[_c]=_rawr[_c]
            _rows.append(_r)

        out=pd.DataFrame(_rows)

        _final_timeout=out.get('source_timeout_censored',pd.Series(False,index=out.index))
        _final_timeout=pd.Series(_final_timeout,index=out.index).fillna(False).astype(bool)
        if _final_timeout.any():
            if 'status' in out.columns:
                out.loc[_final_timeout,'status']='technical_timeout'
            else:
                out['status']=np.where(_final_timeout,'technical_timeout','')
            for _c in ('feasible','solver_feasible','growth_viable'):
                if _c in out.columns:
                    out.loc[_final_timeout,_c]=False
            for _c in out.columns:
                _lc=str(_c).lower()
                if _lc.startswith('objective_norm__') or (_lc.startswith('exact_') and _lc.endswith('_norm')):
                    out.loc[_final_timeout,_c]=0.0
    else:
        out=compile_live_exact(raw,prop,prepared,spec)

    output.parent.mkdir(parents=True,exist_ok=True)
    out.to_csv(output,index=False)

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('inspect'); a.add_argument('--study-config',required=True); a.add_argument('--case-key',required=True)
    e=sub.add_parser('evaluate'); e.add_argument('--study-config',required=True); e.add_argument('--case-key',required=True); e.add_argument('--proposal',required=True); e.add_argument('--output',required=True)
    args=ap.parse_args()
    if args.cmd=='inspect': print(json.dumps(inspect_case(Path(args.study_config),args.case_key),indent=2))
    else: evaluate(Path(args.study_config),args.case_key,Path(args.proposal),Path(args.output))

if __name__=='__main__': main()
