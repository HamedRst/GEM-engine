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
    runtime_root=Path(cfg['paths']['runtime_sources_root']).resolve()/str(case['source_benchmark'])
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



def _resolve_design_cache(src:Path, source_case:str, source_problem:dict) -> Path:
    psrc=source_problem.get('problem',source_problem)
    stored=psrc.get('design_cache')
    if stored:
        p=Path(str(stored)).expanduser()
        if p.exists() and p.is_file():
            return p.resolve()
    dc=src/'inputs'/'runs'/str(source_case)/'design_cache'
    cand=sorted(dc.glob('sobol_*.npy')) if dc.is_dir() else []
    if len(cand)!=1:
        raise RuntimeError(
            f'{source_case}: cannot resolve unique authoritative Sobol design cache; '
            f'stored={stored!r}; candidates={[str(x) for x in cand]}'
        )
    return cand[0].resolve()


def _select_sobol_prefix(source_df:pd.DataFrame, src:Path, source_case:str, source_problem:dict, prior_rows:int):
    """Return the exact first N Sobol designs, verified against the original .npy cache.

    This deliberately refuses arbitrary CSV row prefixes.  The source archive must
    expose a unique integer design_index sequence and the first N designs must match
    the authoritative Sobol cache either in normalized [0,1] coordinates or after
    conversion to physical feature bounds.
    """
    from generic_problem import feature_bounds
    n=int(prior_rows)
    if n<=0 or n>len(source_df):
        raise RuntimeError(f'Invalid landscape_prior_rows={n} for source archive rows={len(source_df)}')
    if 'design_index' not in source_df.columns:
        raise RuntimeError(f'{source_case}: design_index missing; refusing arbitrary row-order prefix')
    di=pd.to_numeric(source_df['design_index'],errors='coerce')
    if di.isna().any() or not np.allclose(di.to_numpy(float),np.round(di.to_numpy(float)),rtol=0,atol=0):
        raise RuntimeError(f'{source_case}: design_index is not finite integer-valued')
    if di.duplicated().any():
        raise RuntimeError(f'{source_case}: duplicate design_index values; refusing prefix')
    order=np.argsort(di.to_numpy(np.int64),kind='stable')
    ordered=source_df.iloc[order].reset_index(drop=True)
    idx=pd.to_numeric(ordered['design_index'],errors='raise').to_numpy(np.int64)
    expected=np.arange(idx[0],idx[0]+len(idx),dtype=np.int64)
    if not np.array_equal(idx,expected):
        raise RuntimeError(f'{source_case}: design_index is not a contiguous Sobol sequence; first={idx[:8].tolist()}')
    prefix=ordered.iloc[:n].copy().reset_index(drop=True)

    cache=_resolve_design_cache(src,source_case,source_problem)
    A=np.load(cache,allow_pickle=False)
    features,bounds=feature_bounds(source_problem.get('problem',source_problem))
    if A.ndim!=2 or A.shape[0]<n or A.shape[1]!=len(features):
        raise RuntimeError(f'{source_case}: Sobol cache shape {A.shape} incompatible with N={n}, d={len(features)}')
    X=prefix[features].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    if not np.isfinite(X).all():
        raise RuntimeError(f'{source_case}: nonfinite feature value in selected prefix')
    C=np.asarray(A[:n],float)
    lo=bounds[:,0]; hi=bounds[:,1]
    err_direct=float(np.max(np.abs(X-C)))
    err_phys=float(np.max(np.abs(X-(lo[None,:]+C*(hi-lo)[None,:]))))
    mode='physical_cache' if err_direct<=err_phys else 'normalized_cache'
    err=min(err_direct,err_phys)
    if not np.isfinite(err) or err>1e-8:
        raise RuntimeError(
            f'{source_case}: first {n} design_index rows do not match Sobol cache; '
            f'direct_err={err_direct:.3e}, normalized_to_physical_err={err_phys:.3e}'
        )
    manifest={
        'selection':'first_N_authoritative_sobol_designs_by_design_index',
        'landscape_prior_rows':n,
        'source_archive_rows':int(len(source_df)),
        'design_index_start':int(idx[0]),
        'design_index_end':int(prefix['design_index'].iloc[-1]),
        'design_cache':str(cache),
        'design_cache_sha256':_sha256(cache),
        'cache_coordinate_mode':mode,
        'sobol_prefix_max_abs_error':err,
        'prefix_design_index_sha256':hashlib.sha256(prefix['design_index'].to_numpy(np.int64).tobytes()).hexdigest(),
        'prefix_feature_matrix_sha256':hashlib.sha256(np.asarray(X,dtype=np.float64).tobytes()).hexdigest(),
        'zero_new_landscape_fba_calls':True,
    }
    return prefix,manifest


def _archive_hv(df:pd.DataFrame, meta:dict) -> float:
    from generic_core import hypervolume_exact
    cols=list(meta['objective_norm_columns'])
    if 'feasible' not in df.columns:
        return 0.0
    f=df['feasible'].astype(str).str.lower().isin(['true','1','yes','optimal_generic_problem']).to_numpy(bool)
    Y=df[cols].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    ok=f & np.isfinite(Y).all(axis=1)
    return float(hypervolume_exact(Y[ok],meta.get('hv_reference_normalized',[0.0]*len(cols)))) if ok.any() else 0.0


def inspect_case(study_config,case_key):
    cfg=load_cfg(study_config); c=resolve_case(cfg,case_key); src=source_root(cfg,c); bpath,ppath=source_prepared(src,c['source_case']); spec,spec_path=active_spec(study_config,c)
    source_df=pd.read_csv(bpath,compression='infer'); source_df=_adapt_robustness_semantics(source_df,str(c.get('source_kind','')),live=False)
    required=int(cfg.get('preflight',{}).get('required_source_archive_rows',50000))
    prior_rows=int(cfg.get('study',{}).get('landscape_prior_rows',0))
    if len(source_df)!=required: raise RuntimeError(f'{case_key}: source frozen archive rows={len(source_df)} required={required}')
    if prior_rows<=0 or prior_rows>required: raise RuntimeError(f'{case_key}: invalid landscape_prior_rows={prior_rows}')

    source_problem=load_yaml(ppath)
    # Freeze the problem definition on the original full 50K archive.  This keeps
    # objective normalization, qXX constraints, viability thresholds, and HV
    # coordinates identical to the v0.3.0 50K experiment.  Only the optimizer's
    # accessible prior/training archive is reduced to N Sobol points.
    full_baseline,meta=compile_frozen_archive(source_df,source_problem,spec)
    prefix_source,prefix_manifest=_select_sobol_prefix(source_df,src,str(c['source_case']),source_problem,prior_rows)
    prepared_semantics={'problem':{**dict(source_problem.get('problem',source_problem)),'compiled_problem':meta}}
    features,_=feature_bounds(source_problem.get('problem',source_problem))
    baseline=compile_live_exact(prefix_source,prefix_source[features].copy(),prepared_semantics,spec)

    semantic=[]
    for obj in meta.get('objectives',[]):
        metric=obj.get('validation_normalized_metric')
        if not metric: continue
        if metric not in prefix_source.columns:
            raise RuntimeError(f'{case_key}: deterministic objective {obj.get("name")} requires validation metric {metric!r}, missing from source archive')
        col=obj['norm_column']; a=pd.to_numeric(baseline[col],errors='coerce').to_numpy(float); b=pd.to_numeric(prefix_source[metric],errors='coerce').to_numpy(float)
        ok=np.isfinite(a)&np.isfinite(b)
        if not ok.any(): raise RuntimeError(f'{case_key}: no finite rows for deterministic semantic validation {obj.get("name")} vs {metric}')
        err=float(np.max(np.abs(a[ok]-b[ok])))
        if err>5e-10: raise RuntimeError(f'{case_key}: deterministic objective semantic mismatch {obj.get("name")} vs {metric}: max_abs_error={err:.3e}')
        semantic.append({'objective':obj.get('name'),'source_metric':metric,'max_abs_error':err,'rows':int(ok.sum())})

    local_feasible=int(baseline['feasible'].astype(str).str.lower().isin(['true','1','yes','optimal_generic_problem']).sum())
    full_hv=_archive_hv(full_baseline,meta); prior_hv=_archive_hv(baseline,meta)
    outdir=Path(cfg['paths']['prepared_root']).resolve()/case_key; outdir.mkdir(parents=True,exist_ok=True)
    local_b=outdir/'baseline_compact.csv.gz'; baseline.to_csv(local_b,index=False,compression='gzip')
    (outdir/'prior_prefix_manifest.json').write_text(json.dumps(prefix_manifest,indent=2),encoding='utf-8')
    bridge=_compat_bridge(src,cfg,c)
    source_prov={'source_benchmark_name':src.name,'source_case':str(c['source_case']),'source_baseline_sha256':_sha256(bpath),'source_problem_sha256':_sha256(ppath),'source_engine_bridge_original_sha256':_sha256(src/'engine_bridge.py'),'source_engine_bridge_runtime_sha256':_sha256(bridge),'problem_definition_sha256':_sha256(spec_path),'landscape_prior_rows':prior_rows,'reference_semantics_rows':required,'sobol_prefix_manifest':prefix_manifest}
    pp=dict(source_problem.get('problem',source_problem)); pp.update({'case_name':case_key,'case_key':case_key,'baseline_table':str(local_b),'compiled_problem':meta,'objective_names':meta['objective_names'],'objective_directions':meta['objective_directions'],'objective_norm_columns':meta['objective_norm_columns'],'objective_raw_columns':meta['objective_raw_columns'],'objective_dim':meta['objective_dim'],'hv_reference_normalized':meta['hv_reference_normalized'],'general_problem_definition':str(spec_path),'source_problem_config':str(ppath),'source_benchmark':str(src),'source_case':c['source_case'],'source_provenance':source_prov,'landscape_prior_rows':prior_rows,'reference_semantics_rows':required,'full50k_reference_hv':full_hv,'prior_hv_exact':prior_hv})
    (outdir/'problem_config.yaml').write_text(yaml.safe_dump({'problem':pp},sort_keys=False),encoding='utf-8')
    frac=local_feasible/max(1,len(baseline))
    report={'status':'PASS','exact_replay_qc_pass':True,'rows':len(baseline),'landscape_prior_rows':prior_rows,'reference_semantics_rows':required,'source_full_archive_rows':len(source_df),'source_feasible_rows_full50k':meta['source_feasible_rows'],'problem_feasible_rows_full50k':meta['problem_feasible_rows'],'problem_feasible_rows':local_feasible,'problem_feasible_fraction':frac,'dimension':len(meta['features']),'problem_id':meta['problem_id'],'problem_hash':meta['problem_hash'],'objective_names':meta['objective_names'],'objective_directions':meta['objective_directions'],'objective_dim':meta['objective_dim'],'hv_reference_normalized':meta['hv_reference_normalized'],'full50k_reference_hv':full_hv,'prior_hv_exact':prior_hv,'prior_hv_fraction_of_full50k':(prior_hv/full_hv if full_hv>1e-15 else None),'constraints':meta['constraints'],'deterministic_semantic_validation':semantic,'archive_relabel_fba_solves':0,'new_landscape_fba_solves':0,'source_baseline':str(bpath),'source_problem':str(ppath),'source_engine_bridge_original':str(src/'engine_bridge.py'),'source_engine_bridge_runtime':str(bridge),'sobol_prefix':prefix_manifest,**source_prov}
    minrows=int(cfg.get('preflight',{}).get('min_problem_feasible_rows',50)); minfrac=float(cfg.get('preflight',{}).get('min_problem_feasible_fraction',0.001))
    if local_feasible<minrows or frac<minfrac: raise RuntimeError(f'{case_key}: N={prior_rows} generic problem is degenerate under frozen 50K semantics: feasible={local_feasible}/{len(baseline)} ({frac:.4g})')
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
    # Continue-on-single-point-failure policy.
    # Scientific semantics are unchanged. We try the canonical/default solver,
    # then glpk_exact. If that one candidate still cannot produce an exact row,
    # only that logical point is marked SKIPPED_EXACT_FAILURE and the campaign
    # continues. No surrogate/predicted value is substituted.
    study=dict(cfg.get('study',{}) or {})
    out={
        'batch_timeout_seconds':max(float(study.get('exact_source_batch_timeout_seconds',30.0)),60.0),
        'candidate_timeout_seconds':max(float(study.get('exact_source_candidate_timeout_seconds',30.0)),120.0),
        'fallback_timeout_seconds':max(float(study.get('exact_source_fallback_timeout_seconds',60.0)),300.0),
        'fallback_solver':str(study.get('exact_source_fallback_solver','glpk_exact')).strip(),
    }
    if any(out[k] <= 0 for k in ('batch_timeout_seconds','candidate_timeout_seconds','fallback_timeout_seconds')):
        raise RuntimeError(f'Invalid exact source timeout policy: {out}')
    if out['fallback_solver'] != 'glpk_exact':
        raise RuntimeError(f"This repair requires exact_source_fallback_solver='glpk_exact'; got {out['fallback_solver']!r}")
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
    # Fast path: source-default batch.
    # Recovery: isolate each candidate -> canonical/default -> glpk_exact.
    # If a single candidate still fails, return an explicit skip marker for that
    # candidate only. The caller preserves the logical call but excludes it from
    # exact training, feasibility statistics, Pareto fronts, and HV.
    policy=_exact_solver_policy(cfg)
    env=os.environ.copy(); env['PYTHONPATH']=str(src)+os.pathsep+env.get('PYTHONPATH','')

    def _proc_ok(proc, raw_path: Path):
        return proc is not None and proc.returncode==0 and raw_path.exists()

    def _proc_diag(proc):
        if proc is None:
            return 'timeout'
        out=(proc.stdout or '').strip(); err=(proc.stderr or '').strip()
        txt=f'rc={proc.returncode}'
        if out: txt+=f' stdout={out[-1200:]}'
        if err: txt+=f' stderr={err[-1200:]}'
        return txt

    def _skip_row(one: pd.DataFrame, pid: str, primary_reason: str, fallback_reason: str,
                  primary_wall: float, fallback_wall: float) -> pd.DataFrame:
        row={
            'proposal_id':pid,
            'source_exact_skipped':True,
            'exact_skipped':True,
            'source_exact_skip_reason':f'default={primary_reason}; {policy["fallback_solver"]}={fallback_reason}',
            'source_solver_route':'SKIPPED_AFTER_EXACT_FAILURE',
            'source_solver_fallback_used':True,
            'source_solver_isolated_retry':True,
            'source_primary_timeout_seconds':float(policy['candidate_timeout_seconds']),
            'source_fallback_solver':policy['fallback_solver'],
            'source_solver_retry_attempts':2,
            'source_exact_attempt_wall_seconds':float(primary_wall+fallback_wall),
            'solver_wall_seconds':float(primary_wall+fallback_wall),
            'fba_solves':0,
            'source_problem_feasible':False,
            'feasible':False,
            'status':'SKIPPED_EXACT_FAILURE',
        }
        for col,val in one.iloc[0].items():
            if col not in row:
                row[col]=val
        print(
            f"[exact-policy] {pid}: exact evaluation still failed after default + "
            f"{policy['fallback_solver']}; SKIPPING THIS POINT ONLY and continuing",
            flush=True,
        )
        return pd.DataFrame([row])

    with tempfile.TemporaryDirectory(prefix='gem_gsmobo_dev_batch_') as td:
        td=Path(td); raw_path=td/'source_exact.csv'; source_prop=td/'source_proposal.csv'
        source_rows.to_csv(source_prop,index=False)
        cmd=_source_evaluate_cmd(runtime_bridge,src,c,source_problem,run_dir,model_path,source_prop,raw_path)
        proc,timed_out,wall=_run_source_command(
            cmd,src=src,env=env,timeout_seconds=policy['batch_timeout_seconds']
        )
        if (not timed_out) and _proc_ok(proc,raw_path):
            fresh=_require_source_success(proc,raw_path,'Source exact batch')
            if len(fresh)!=len(source_rows):
                raise RuntimeError(f'Source exact bridge row mismatch {len(fresh)} != {len(source_rows)}')
            fresh=fresh.copy()
            fresh['source_solver_route']='default_batch'
            fresh['source_solver_fallback_used']=False
            fresh['source_solver_isolated_retry']=False
            fresh['source_primary_timeout_seconds']=float(policy['batch_timeout_seconds'])
            fresh['source_fallback_solver']=''
            fresh['source_solver_retry_attempts']=0
            fresh['source_exact_attempt_wall_seconds']=float(wall)
            fresh['source_exact_skipped']=False
            fresh['exact_skipped']=False
            return fresh
        batch_reason='timeout' if timed_out else _proc_diag(proc)

    print(
        f"[exact-policy] source batch did not complete under the fast-path "
        f"({policy['batch_timeout_seconds']:.0f}s; {batch_reason}); isolating "
        f"{len(source_rows)} candidate(s)",
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
            if (not timed_out) and _proc_ok(proc,raw_path):
                row=_require_source_success(proc,raw_path,f'{pid} default isolated exact')
                if len(row)!=1: raise RuntimeError(f'{pid}: isolated source exact returned {len(row)} rows')
                row=row.copy()
                row['source_solver_route']='default_isolated_after_batch_recovery'
                row['source_solver_fallback_used']=False
                row['source_solver_isolated_retry']=True
                row['source_primary_timeout_seconds']=float(policy['candidate_timeout_seconds'])
                row['source_fallback_solver']=''
                row['source_solver_retry_attempts']=1
                row['source_exact_attempt_wall_seconds']=float(primary_wall)
                row['source_exact_skipped']=False
                row['exact_skipped']=False
                parts.append(row)
                continue

            primary_reason='timeout' if timed_out else _proc_diag(proc)
            print(
                f"[exact-policy] {pid}: default isolated attempt failed "
                f"({policy['candidate_timeout_seconds']:.0f}s; {primary_reason}); trying "
                f"{policy['fallback_solver']} for up to {policy['fallback_timeout_seconds']:.0f}s",
                flush=True,
            )

            fallback_raw=td/'source_exact_fallback.csv'
            fallback_cmd=_source_evaluate_cmd(runtime_bridge,src,c,source_problem,run_dir,model_path,source_prop,fallback_raw)
            fproc,ftimed_out,fallback_wall=_run_source_command(
                fallback_cmd,src=src,env=env,
                timeout_seconds=policy['fallback_timeout_seconds'],
                force_solver=policy['fallback_solver'],
            )
            if (not ftimed_out) and _proc_ok(fproc,fallback_raw):
                row=_require_source_success(
                    fproc,fallback_raw,f'{pid} {policy["fallback_solver"]} fallback exact'
                )
                if len(row)!=1: raise RuntimeError(f'{pid}: fallback source exact returned {len(row)} rows')
                row=row.copy()
                row['source_solver_route']=f'default_recovery_then_{policy["fallback_solver"]}'
                row['source_solver_fallback_used']=True
                row['source_solver_isolated_retry']=True
                row['source_primary_timeout_seconds']=float(policy['candidate_timeout_seconds'])
                row['source_fallback_solver']=policy['fallback_solver']
                row['source_solver_retry_attempts']=2
                row['source_exact_attempt_wall_seconds']=float(primary_wall+fallback_wall)
                row['source_exact_skipped']=False
                row['exact_skipped']=False
                parts.append(row)
                continue

            fallback_reason='timeout' if ftimed_out else _proc_diag(fproc)
            parts.append(_skip_row(one,pid,primary_reason,fallback_reason,primary_wall,fallback_wall))

    fresh=pd.concat(parts,ignore_index=True,sort=False) if parts else pd.DataFrame()
    if len(fresh)!=len(source_rows):
        raise RuntimeError(f'Isolated source exact row mismatch {len(fresh)} != {len(source_rows)}')
    return fresh

def evaluate(study_config,case_key,proposal,output):
    cfg=load_cfg(study_config); c=resolve_case(cfg,case_key); src=source_root(cfg,c); bpath,ppath=source_prepared(src,c['source_case']); source_problem=load_yaml(ppath).get('problem',load_yaml(ppath)); spec,_=active_spec(study_config,c)
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
            rec=fresh.iloc[j].to_dict()
            skipped=str(rec.get('source_exact_skipped',rec.get('exact_skipped',False))).strip().lower() in {'true','1','yes'}
            for k,f in enumerate(features):
                rec[f]=float(prop.iloc[ii][f]); rec[xcols[k]]=float(X01[ii,k])
            if skipped:
                rec['source_exact_skipped']=True;rec['exact_skipped']=True
                rec['source_raw_cache_hit']=False
                ordered[ii]=rec
                continue
            orig_solves=int(float(rec.get('fba_solves',0) or 0)); orig_wall=float(rec.get('solver_wall_seconds',0.0) or 0.0)
            rec['source_original_fba_solves']=orig_solves; rec['source_original_solver_wall_seconds']=orig_wall
            stored=dict(rec); stored.pop('source_raw_cache_hit',None); cache[_raw_key(X01[ii],decimals)]=stored
            current=dict(rec); current['source_raw_cache_hit']=False; current['source_exact_skipped']=False; current['exact_skipped']=False; ordered[ii]=current
        _write_raw_cache_atomic(cache_path,cache)

    skipped_mask=np.array([
        str((r or {}).get('source_exact_skipped',(r or {}).get('exact_skipped',False))).strip().lower() in {'true','1','yes'}
        for r in ordered
    ],dtype=bool)
    success_idx=np.where(~skipped_mask)[0]
    prepared=load_yaml(Path(cfg['paths']['prepared_root']).resolve()/case_key/'problem_config.yaml')
    out_rows=[None]*len(prop)
    if len(success_idx):
        raw_success=pd.DataFrame([ordered[int(i)] for i in success_idx]).reset_index(drop=True)
        raw_success=_adapt_robustness_semantics(raw_success,str(c.get('source_kind','')),live=True)
        prop_success=prop.iloc[success_idx].reset_index(drop=True)
        compiled_success=compile_live_exact(raw_success,prop_success,prepared,spec).reset_index(drop=True)
        for j,ii in enumerate(success_idx):
            rr=compiled_success.iloc[j].to_dict()
            rr['source_exact_skipped']=False;rr['exact_skipped']=False
            out_rows[int(ii)]=rr

    comp=prepared.get('problem',prepared).get('compiled_problem',{})
    raw_cols=list(comp.get('objective_raw_columns',[]));norm_cols=list(comp.get('objective_norm_columns',[]))
    problem_hash=str(comp.get('problem_hash',''))
    for ii in np.where(skipped_mask)[0]:
        rr=dict(ordered[int(ii)] or {})
        for col in raw_cols+norm_cols: rr[col]=np.nan
        rr['source_problem_feasible']=False
        rr['feasible']=False
        rr['status']='SKIPPED_EXACT_FAILURE'
        rr['source_exact_skipped']=True
        rr['exact_skipped']=True
        rr['source_raw_cache_hit']=False
        rr['problem_hash']=problem_hash
        out_rows[int(ii)]=rr

    out=pd.DataFrame(out_rows)
    if len(out)!=len(prop): raise RuntimeError(f'Compiled exact row mismatch {len(out)} != {len(prop)}')
    print(
        f'[source-cache] {case_key}: hits={len(prop)-len(missing)}/{len(prop)} | '
        f'physical_source_evals={len(missing)} | skipped_exact={int(skipped_mask.sum())}',
        flush=True,
    )
    output.parent.mkdir(parents=True,exist_ok=True); out.to_csv(output,index=False)

def main():
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('inspect'); a.add_argument('--study-config',required=True); a.add_argument('--case-key',required=True)
    e=sub.add_parser('evaluate'); e.add_argument('--study-config',required=True); e.add_argument('--case-key',required=True); e.add_argument('--proposal',required=True); e.add_argument('--output',required=True)
    args=ap.parse_args()
    if args.cmd=='inspect': print(json.dumps(inspect_case(Path(args.study_config),args.case_key),indent=2))
    else: evaluate(Path(args.study_config),args.case_key,Path(args.proposal),Path(args.output))

if __name__=='__main__': main()
