"""Dispatch a single GUI job to unchanged, hash-verified scientific modules."""
from __future__ import annotations
import argparse
import copy
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import platform
import shutil
import signal
import sys
import time
import traceback
import yaml
import numpy as np
import pandas as pd

from .catalog import ASSETS, canonical_cases, runtime_key, sha, verify_sources
from .problems import template_bytes, validate_content

def atomic_json(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, default=str, allow_nan=False) + '\n')
    tmp.replace(path)

def atomic_csv(frame,path):
    path=Path(path);tmp=path.with_suffix('.csv.tmp')
    frame.to_csv(tmp,index=False);tmp.replace(path)

def relocate(value, root):
    if isinstance(value, dict): return {k: relocate(v, root) for k, v in value.items()}
    if isinstance(value, list): return [relocate(v, root) for v in value]
    if isinstance(value, str):
        for source in (ASSETS / 'benchmarks').iterdir():
            marker = source.name
            if marker in value and ('/Users/' in value or value.startswith(str(ASSETS))):
                tail = value.split(marker, 1)[1].lstrip('/')
                return str(root / marker / tail)
    return value

def runtime_sources(root):
    """Copy source packages; relocate configuration paths only, never optimizer code."""
    for src in (ASSETS / 'benchmarks').iterdir():
        dst = root / src.name
        if not dst.exists(): shutil.copytree(src, dst)
        for original in src.rglob('*'):
            if original.is_file() and original.suffix in ('.yaml', '.yml', '.json'):
                target = dst / original.relative_to(src)
                try:
                    obj = yaml.safe_load(original.read_text()) if original.suffix != '.json' else json.loads(original.read_text())
                    obj = relocate(obj, root)
                    target.write_text(json.dumps(obj, indent=2) if original.suffix == '.json' else yaml.safe_dump(obj, sort_keys=False))
                except (UnicodeError, yaml.YAMLError, json.JSONDecodeError): pass

def prepare(job, out):
    verify_sources()
    family = job.get('family', 'gsmobo'); size = int(job['prior_rows'])
    key = runtime_key(family, size)
    runtime = ASSETS / 'runtimes' / key
    sys.path.insert(0, str(runtime))
    gp = importlib.import_module('generic_problem')
    cfg = yaml.safe_load((runtime / 'original_study_config.yaml').read_text())
    case = job['case']; source = job.get('binding') or next((c for c in canonical_cases() if c['case'] == case), None)
    if source is None: raise ValueError('Unknown canonical source binding')
    root = out.parent / '_source_bindings'; root.mkdir(parents=True, exist_ok=True)
    with (root / 'initialization.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not (root / 'ready.json').exists():
            runtime_sources(root)
            atomic_json(root / 'ready.json', {'assets_sha256': sha(ASSETS / 'SOURCE_HASHES.json')})
    src = Path(source['source_benchmark']) if job.get('binding') else root / Path(source['source_benchmark']).name
    solver=job.get('exact_solver','glpk')
    if solver not in ('glpk','glpk_presolve'):raise ValueError('Unsupported exact solver route')
    original = yaml.safe_load((src / 'prepared' / source['source_case'] / 'problem_config.yaml').read_text())
    source_problem = original['problem']
    # A custom model may never silently reuse another model's landscape.
    if job.get('model_path') and sha(job['model_path']) != source['model_sha256']:
        raise ValueError('Selected model does not match the authoritative landscape model')
    bp = src / 'prepared' / source['source_case'] / 'baseline_compact.csv.gz'
    if job.get('landscape_path') and sha(job['landscape_path']) != sha(Path(source['landscape_path'])):
        raise ValueError('Selected landscape does not match this canonical source binding')
    spec_text = job.get('problem_yaml') or template_bytes(source['problem_key']).decode()
    spec = validate_content(spec_text)
    spec_path = out / 'problem.gemproblem.yaml'; spec_path.write_text(spec_text)
    raw = pd.read_csv(bp)
    if len(raw) != 50000: raise ValueError('Canonical semantics require the complete 50K archive')
    # Source adaptation is the frozen historical oxygen-retention adapter.
    bridge = importlib.import_module('multi_problem_bridge')
    raw = bridge._adapt_robustness_semantics(raw, source.get('source_kind', ''), live=False)
    full, comp = gp.compile_frozen_archive(raw, original, spec)
    if not np.array_equal(pd.to_numeric(raw['design_index']).to_numpy(), np.arange(len(raw))):
        raise ValueError('Authoritative design-index ordering failed')
    features, bounds = gp.feature_bounds(source_problem)
    # Verify saved Sobol coordinate ordering before selecting the prefix.
    prefix_module_path = ASSETS / 'runtimes/gsmobo_10000/multi_problem_bridge.py'
    m_spec = importlib.util.spec_from_file_location('_prefix_bridge', prefix_module_path)
    prefix_module = importlib.util.module_from_spec(m_spec); m_spec.loader.exec_module(prefix_module)
    prefix, prefix_manifest = prefix_module._select_sobol_prefix(raw, src, source['source_case'], original, size)
    parent = {'problem': {**source_problem, 'compiled_problem': comp}}
    baseline = gp.compile_live_exact(prefix, prefix[features], parent, spec)
    prepared = out / 'prepared' / case; prepared.mkdir(parents=True, exist_ok=True)
    local_bp = prepared / 'baseline_compact.csv.gz'; baseline.to_csv(local_bp, index=False, compression='gzip')
    pp = {**source_problem, 'compiled_problem': comp, 'baseline_table': str(local_bp),
          'source_benchmark': str(src), 'source_case': source['source_case'],
          'general_problem_definition': str(spec_path), 'landscape_prior_rows': size,
          'reference_semantics_rows': 50000}
    parent = {'problem': pp}
    (prepared / 'problem_config.yaml').write_text(yaml.safe_dump(parent, sort_keys=False))
    atomic_json(prepared / 'landscape_preflight.json', {'status': 'PASS', 'archive_relabel_fba_solves': 0,
                'problem_feasible_rows': int(gp.as_bool(baseline['feasible']).sum()),
                'rows': size, 'problem_hash': comp['problem_hash'], 'prefix': prefix_manifest})
    cfg['paths'] = {'gem_hamed_root': str(root), 'prepared_root': str(out / 'prepared'),
                    'output_root': str(out / 'scientific_results'), 'runtime_sources_root': str(out.parent / '_exact_bridge_caches/runtime_sources')}
    cfg['study'].update(seed=int(job['seed']), landscape_prior_rows=size, prior_label=f'{size // 1000}K',
                        reference_semantics_rows=50000)
    cfg['preflight']['required_source_archive_rows'] = 50000
    cfg['cases'] = {case: {**source, 'source_benchmark': str(src), 'problem_definition': str(spec_path),
                         'prepared_dir': str(prepared), 'study_config_path': str(out / 'study_config.yaml'), 'case_key': case}}
    cfg['methods_primary'] = [job['method']]
    sp = out / 'study_config.yaml'; sp.write_text(yaml.safe_dump(cfg, sort_keys=False))
    gc = importlib.import_module('generic_core')
    def hv(df):
        Y = df[comp['objective_norm_columns']].to_numpy(float)
        f = gp.as_bool(df['feasible']) & np.isfinite(Y).all(1)
        return float(gc.hypervolume_exact(Y[f], comp['hv_reference_normalized'])) if f.any() else 0.
    provenance = {'runtime_key': key, 'algorithm_sha256': sha(runtime / ('gsmobo_algorithms.py' if family == 'gsmobo' else 'nsga_algorithms.py')),
                  'source_model_sha256': source['model_sha256'], 'source_archive_sha256': sha(Path(source['landscape_path'])),
                  'source_engine_sha256': sha(Path(source['source_benchmark']) / 'engine_bridge.py'),
                  'source_problem_sha256': sha(Path(source['source_benchmark']) / 'prepared' / source['source_case'] / 'problem_config.yaml'),
                  'problem_hash': comp['problem_hash'], 'prior_rows': size, 'prefix': prefix_manifest,
                  'full50k_reference_hv': hv(full), 'prior_hv': hv(baseline), 'archive_preparation_fba_solves': 0,
                  'configuration_origin': cfg.get('provenance', {}), 'landscape_seed': source.get('landscape_seed',42),
                  'optimizer_seed': job['seed'], 'method': job['method'], 'case': case}
    provenance['exact_solver']=solver
    atomic_json(out / 'provenance.json', provenance)
    return cfg, parent, baseline, runtime, provenance

def export_metrics(ctx, method, provenance, out, seconds):
    """Generate every exact per-call output ONCE after scientific optimization.

    For prior-union runs, the frozen scientific runtime has already calculated
    archive_hv_after for every call. Recomputing that recursive 3D hypervolume
    2*N times after EVERY batch is unnecessary and can dominate runtime.
    No-prior runs retain exact self-front scoring (not a union with the prior).
    Pareto evolution is retained in full for both modes.
    """
    from generic_core import pareto_points_generic, hypervolume_exact
    from generic_problem import as_bool
    h = ctx.history(method); cols = ctx.objective_norm_columns
    no_prior = 'NO50K' in method
    front = np.empty((0, len(cols))) if no_prior else ctx.baseline_front.copy()
    start_hv = 0. if no_prior else float(ctx.baseline_hv)
    previous_hv = start_hv
    rows = []; evolution = []
    if not no_prior:
        if 'archive_hv_after' not in h.columns:
            raise RuntimeError('Prior-union history missing scientific archive_hv_after; no reported HV was guessed')
    for _, row in h.iterrows():
        before = previous_hv
        valid = bool(as_bool(pd.Series([row['exact_feasible']]))[0]) and np.isfinite(row[cols].to_numpy(float)).all()
        if valid:
            front = pareto_points_generic(np.vstack([front, row[cols].to_numpy(float)]))
        if no_prior:
            # The frozen runtime records prior-union HV even for NO50K runs.
            # Hence NO50K reporting must retain TRUE live self-front HV.
            after = float(hypervolume_exact(front, ctx.hv_ref)) if valid and len(front) else before
        else:
            # Scientific runtime already computed this with the exact same HV helper.
            # Read the committed value instead of recursively recomputing it.
            after = float(row['archive_hv_after'])
            if not np.isfinite(after):
                raise RuntimeError('Scientific history has nonfinite archive_hv_after')
            if after < before - 1e-10:
                raise RuntimeError('Scientific hypervolume decreased unexpectedly; refusing to alter the record')
        previous_hv = after
        rows.append({'call': int(row['call']), 'hv': after, 'delta_hv': after - start_hv,
                     'normalized_gain': (after-start_hv)/max(1e-12, 1-start_hv),
                     'exact_front_extension': after > before + 1e-14})
        for point in front:
            evolution.append({'call': int(row['call']), **dict(zip(cols, map(float, point)))})
    atomic_csv(pd.DataFrame(rows),out / 'hv_trajectory.csv')
    atomic_csv(pd.DataFrame(evolution),out / 'pareto_evolution.csv')
    atomic_csv(pd.DataFrame(front, columns=cols),out / 'pareto_front.csv')
    atomic_csv(h,out / 'calls.csv')
    summary = {'status': 'COMPLETE', 'method': method, 'calls': len(h), 'baseline_hv': start_hv,
               'final_hv': rows[-1]['hv'] if rows else start_hv,
               'final_delta_hv': rows[-1]['delta_hv'] if rows else 0.,
               'normalized_gain': rows[-1]['normalized_gain'] if rows else 0.,
               'feasible_fraction': float(as_bool(h['exact_feasible']).mean()) if len(h) else None,
               'exact_extensions': sum(r['exact_front_extension'] for r in rows),
               'runtime_seconds': seconds, 'scoring': 'LIVE_SELF_FRONT' if no_prior else 'PRIOR_UNION_LIVE_FRONT',
               'full50k_reference_hv': provenance['full50k_reference_hv'],
               'objective_columns': cols, 'objective_names': ctx.objective_names}
    summary['technical_timeout_calls']=int(h['status'].astype(str).str.contains('timeout|technical|skipped',case=False).sum())
    summary['exact_solver']=provenance.get('exact_solver','glpk')
    summary['solver_routes_by_call']=h['exact_solver_route'].value_counts().to_dict() if 'exact_solver_route' in h else {'glpk':len(h)}
    ledger = ctx.outdir / 'exact_runtime_ledger.csv'
    if ledger.exists():
        ll = pd.read_csv(ledger); ll = ll[ll['method'] == method]
        summary['physical_fba_solves'] = int(ll['physical_fba_solves'].sum())
        summary['logical_evaluations'] = int(ll['logical_evals'].sum())
        shutil.copy2(ledger, out / 'exact_runtime_ledger.csv')
    atomic_json(out / 'summary.json', summary)
    return summary


def quick_progress_metrics(ctx, method, provenance, out, seconds, frame):
    """O(1) progress checkpoint: scientific history/RNG have already been committed.

    The comprehensive HV trajectory and Pareto evolution are exported exactly
    once at method completion; the original scientific history retains all calls.
    """
    from generic_core import hypervolume_exact
    from generic_problem import as_bool
    no_prior = 'NO50K' in method
    if no_prior:
        # The archived NSGA-II LiveContext has NO live_self_front method.
        # Compute the true self-front from the committed <=128 scientific
        # calls, without reading or unioning any prior landscape points.
        from generic_core import pareto_points_generic
        Y = frame[ctx.objective_norm_columns].to_numpy(float)
        ok = as_bool(frame['exact_feasible']) & np.isfinite(Y).all(axis=1)
        front = (pareto_points_generic(Y[ok]) if ok.any()
                 else np.empty((0, len(ctx.objective_norm_columns))))
        after = float(hypervolume_exact(front, ctx.hv_ref)) if len(front) else 0.0
        baseline = 0.0
    else:
        baseline = float(ctx.baseline_hv)
        after = float(frame.iloc[-1]['archive_hv_after'])
    delta = after - baseline
    result = {'status':'RUNNING', 'method':method, 'calls':int(len(frame)),
              'baseline_hv':baseline,'final_hv':after,'final_delta_hv':delta,
              'normalized_gain':delta/max(1e-12,1-baseline),
              'feasible_fraction':float(as_bool(frame['exact_feasible']).mean()),
              'exact_extensions':None if no_prior else int(as_bool(frame['exact_front_extension']).sum()),
              'runtime_seconds':float(seconds),
              'scoring':'LIVE_SELF_FRONT' if no_prior else 'PRIOR_UNION_LIVE_FRONT',
              'full50k_reference_hv':provenance['full50k_reference_hv'],
              'objective_columns':ctx.objective_norm_columns, 'objective_names':ctx.objective_names,
              'technical_timeout_calls':int(frame['status'].astype(str).str.contains('timeout|technical|skipped',case=False).sum()),
              'exact_solver':provenance.get('exact_solver','glpk'),
              'solver_routes_by_call':frame['exact_solver_route'].value_counts().to_dict() if 'exact_solver_route' in frame else {'glpk':len(frame)},
              'logical_evaluations':int(len(frame)),
              'reporting_mode':'DEFER_FULL_HV_UNTIL_COMPLETE_v2_4_2'}
    atomic_csv(frame, out/'calls.csv')
    atomic_json(out/'summary.json',result)
    return result

def execute(job, out):
    start = time.monotonic()
    if job['method']=='AUTO_QC_SPARC':
        from .automatic_selection import resolve, AUTO_ID
        selected,decision=resolve(job,out)
        result=execute(selected,out)
        result['requested_method']=AUTO_ID
        result['automatic_selection']={k:v for k,v in decision.items() if k!='selection_timestamp_utc'}
        result['automatic_selection_total_wall_seconds']=time.monotonic()-start
        # Persist in run provenance, not only the UI summary.
        pp=out/'provenance.json'
        provenance=json.loads(pp.read_text()) if pp.exists() else {}
        provenance['automatic_selection']=result['automatic_selection']
        atomic_json(pp,provenance)
        atomic_json(out/'summary.json',result)
        return result
    previous=json.loads((out/'provenance.json').read_text()) if (out/'provenance.json').exists() else {}
    if job.get('exact_solver')=='glpk_presolve':
        compat=out/'solver_configuration';compat.mkdir(exist_ok=True)
        # Numerical solver configuration wrapper; no optimizer or LP changes.
        (compat/'sitecustomize.py').write_text('''import cobra
_original_optimize = cobra.Model.optimize
def _presolve_optimize(model, *args, **kwargs):
    iface = str(getattr(model.solver.interface, "__name__", ""))
    if "glpk" in iface and "glpk_exact" not in iface:
        model.solver.configuration.presolve = True
    return _original_optimize(model, *args, **kwargs)
cobra.Model.optimize = _presolve_optimize
''')
        os.environ['PYTHONPATH']=str(compat)+os.pathsep+os.environ.get('PYTHONPATH','')
    if job.get('general_binding'):
        from .general_runtime import prepare_general
        cfg, parent, baseline, runtime, prov = prepare_general(job, out)
    else:
        cfg, parent, baseline, runtime, prov = prepare(job, out)
    if job['method'] == 'LANDSCAPE_ONLY':
        from generic_core import pareto_points_generic
        from generic_problem import as_bool
        cols = parent['problem']['compiled_problem']['objective_norm_columns']; Y = baseline[cols].to_numpy(float)
        ok = as_bool(baseline['feasible']) & np.isfinite(Y).all(1)
        pd.DataFrame(pareto_points_generic(Y[ok]), columns=cols).to_csv(out / 'pareto_front.csv', index=False)
        baseline.to_csv(out / 'landscape_objectives.csv.gz', index=False, compression='gzip')
        summary = {'status': 'COMPLETE', 'method': job['method'], 'calls': 0, 'physical_fba_solves': 0,
                   'prior_rows': len(baseline), 'final_hv': prov['prior_hv'], 'full50k_reference_hv': prov['full50k_reference_hv'],
                   'objective_columns': cols, 'feasible_fraction': float(ok.mean()), 'runtime_seconds': time.monotonic()-start}
        atomic_json(out / 'summary.json', summary); return summary
    import joblib
    from generic_problem import feature_bounds, normalize_design, as_bool
    from runtime import LiveContext, ExactRunner
    alg = importlib.import_module('gsmobo_algorithms' if job['family'] == 'gsmobo' else 'nsga_algorithms')
    if job['method'] not in alg.RUNNERS: raise ValueError('Method not present in selected frozen runtime')
    import surrogate_qc
    surrogate_qc.validate(cfg, job['case'])
    # Match the original CLI's prepared-archive read boundary exactly.
    # In-memory full-precision values can differ from the saved CSV round trip.
    baseline=pd.read_csv(Path(cfg['paths']['prepared_root'])/job['case']/'baseline_compact.csv.gz')
    p = parent['problem']; comp = p['compiled_problem']; features, bounds = feature_bounds(p)
    X = normalize_design(baseline, features, bounds); Y = baseline[comp['objective_norm_columns']].to_numpy(float); F = as_bool(baseline['feasible'])
    case_seed = int(hashlib.sha256(job['case'].encode()).hexdigest()[:8], 16)
    ids = alg.representative_prior_ids(X, Y, F, int(cfg['study']['gp_subset_rows']), int(job['seed'])+case_seed)
    scientific = Path(cfg['paths']['output_root']) / job['case'] / f"seed_{job['seed']}"
    qc = scientific / 'surrogate_qc'; model = joblib.load(qc / 'final_global_surrogate.joblib'); oof = np.load(qc / 'oof_prior_predictions.npz')['oof']
    c = cfg['cases'][job['case']]
    if job.get('general_binding'):
        from .general_runtime import GenericExactRunner
        er = GenericExactRunner(cfg, parent, scientific, c, job['general_binding'])
    else:
        er = ExactRunner(cfg, parent, scientific, c)
    preflight = er.preflight()
    atomic_json(out / 'exact_bridge_preflight.json', preflight)
    if preflight['status'] != 'PASS': raise ValueError('Exact bridge preflight failed')
    ctx = LiveContext(cfg, parent, baseline, Path(cfg['paths']['prepared_root']) / job['case'] / 'baseline_compact.csv.gz', model, ids, oof[ids], scientific, er, int(job['seed']), int(job['budget']))
    ctx.oof_all = oof
    prov['prior_hv']=float(ctx.baseline_hv)
    # Only persistence is wrapped: proposals, selection, acquisition and FBA are unchanged.
    # Cancellation during a CSV write must preserve the preceding complete history.
    def atomic_history(method, frame):
        path=ctx.method_dir(method)/'calls.csv';tmp=path.with_suffix('.csv.tmp')
        if 'exact_solver_route' not in frame:frame['exact_solver_route']=previous.get('exact_solver',job.get('exact_solver','glpk'))
        old_count=len(pd.read_csv(path)) if path.exists() else 0
        frame.loc[frame.index>=old_count,'exact_solver_route']=job.get('exact_solver','glpk')
        cached=(frame.index>=old_count) & frame['cache_hit'].astype(bool)
        frame.loc[cached,'exact_solver_route']='exact_cache'
        frame.to_csv(tmp,index=False)
        from .rng import snapshot
        # Persist RNG before committing history, so cancellation cannot leave a
        # committed batch without its matching continuation state.
        atomic_json(out/'rng_checkpoints'/f'{len(frame):06d}.json',snapshot(job,len(frame)))
        tmp.replace(path)
        quick_progress_metrics(ctx,method,prov,out,time.monotonic()-start,frame)
    ctx.save_history=atomic_history
    from .rng import begin
    prov['random_state']=begin(job,out,len(ctx.history(job['method'])))
    atomic_json(out/'provenance.json',prov)
    alg.RUNNERS[job['method']](ctx)
    if len(ctx.history(job['method'])) != int(job['budget']): raise RuntimeError('Incomplete logical evaluation budget')
    return export_metrics(ctx, job['method'], prov, out, time.monotonic()-start)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--job', required=True); args = ap.parse_args()
    jp = Path(args.job).resolve(); job = json.loads(jp.read_text()); out = jp.parent
    with (out / 'worker.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        attempts_path=out/'attempts.json'
        attempts=json.loads(attempts_path.read_text()) if attempts_path.exists() else []
        attempt={'attempt':len(attempts)+1,'started_at':time.time(),'exact_solver':job.get('exact_solver','glpk'),'status':'running'}
        attempts.append(attempt);atomic_json(attempts_path,attempts)
        def stop(signum, frame): raise KeyboardInterrupt('Cancellation requested')
        signal.signal(signal.SIGTERM, stop)
        atomic_json(out / 'worker_state.json', {'status': 'running', 'pid': os.getpid(), 'job_id': job['id']})
        try:
            summary = execute(job, out)
            import importlib.metadata
            versions = {name: importlib.metadata.version(name) for name in ['cobra','numpy','pandas','scipy','scikit-learn','torch','botorch','gpytorch']}
            atomic_json(out / 'environment.json', {'python': sys.version, 'platform': platform.platform(), 'versions': versions})
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            summary['peak_worker_memory_mb'] = peak/(1024**2 if sys.platform=='darwin' else 1024)
            summary['peak_memory_scope'] = 'worker process; child solver peak unavailable'
            attempt.update(status='completed',runtime_seconds=time.monotonic()-start)
            atomic_json(attempts_path,attempts)
            summary['cumulative_attempt_seconds']=sum(a.get('runtime_seconds',0.) for a in attempts)
            atomic_json(out / 'summary.json', summary)
            atomic_json(out / 'worker_state.json', {'status': 'completed', 'job_id': job['id'], 'runtime_seconds': time.monotonic()-start})
        except BaseException as e:
            traceback.print_exc()
            status = 'cancelled' if isinstance(e, KeyboardInterrupt) else 'failed'
            attempt.update(status=status,error=str(e),runtime_seconds=time.monotonic()-start)
            atomic_json(attempts_path,attempts)
            atomic_json(out / 'worker_state.json', {'status': status, 'job_id': job['id'], 'error': str(e), 'runtime_seconds': time.monotonic()-start})
            if status == 'failed': raise

if __name__ == '__main__': main()
