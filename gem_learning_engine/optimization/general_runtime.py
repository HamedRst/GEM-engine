"""Independent generic input/FBA adapters for unchanged archived optimizer runners.

This is NOT the canonical replay bridge. Historical benchmark jobs still go through
worker.prepare()/runtime.ExactRunner without adaptation.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import os
from pathlib import Path
import time
import numpy as np
import pandas as pd
import yaml
from .catalog import ASSETS, sha
from .general import load_model, read_landscape, validate_contract, FBA_MODES, referenced_metrics
from .problems import validate_content
from .worker import atomic_json, atomic_csv

class TechnicalFBATimeout(RuntimeError):
    """Technical solver timeout; its benchmark penalty is not biological proof of infeasibility."""




def commit_hard_timeout(job,job_folder,activity):
    """Record only the interrupted pending point; preserve optimizer history/RNG."""
    out=Path(job_folder);b=job['general_binding'];features=b['features']
    x=activity.get('x01');xp=activity.get('xphys')
    if x is None or xp is None or len(x)!=len(features) or len(xp)!=len(features):
        raise RuntimeError('Hard timeout candidate coordinates unavailable; refuse to invent a result')
    x=np.asarray(x,float);xp=np.asarray(xp,float)
    bounds=np.asarray([b['bounds'][f] for f in features],float)
    if not np.isfinite(x).all() or not np.allclose(bounds[:,0]+x*(bounds[:,1]-bounds[:,0]),xp,rtol=1e-12,atol=1e-12):
        raise RuntimeError('Hard timeout candidate coordinate mismatch')
    root=out/'scientific_results'/job['case']/f"seed_{job['seed']}"
    root.mkdir(parents=True,exist_ok=True);cache=root/'exact_cache.csv'
    decimals=12
    config=out/'study_config.yaml'
    if config.exists():decimals=int(yaml.safe_load(config.read_text())['study'].get('exact_cache_round_decimals',12))
    key=tuple(np.round(x,decimals));df=pd.read_csv(cache) if cache.exists() else pd.DataFrame()
    if not df.empty:
        found=np.all(np.round(df[[f'x01__{f}' for f in features]].to_numpy(float),decimals)==np.asarray(key),axis=1)
        if found.any():
            if not df.loc[found,'status'].astype(str).eq('glpk_timeout_assumed_infeasible').all():
                raise RuntimeError('Hard timeout candidate already has a non-timeout cached result')
            return
    elapsed=max(0.,time.time()-float(activity['time']))
    row={'status':'glpk_timeout_assumed_infeasible','fba_solver_status':'glpk_hard_watchdog',
         'technical_timeout_assumed_infeasible':True,'timeout_detail':'Native GLPK failed to return at 300-second limit; external watchdog stopped worker',
         'timeout_seconds':float(activity['timeout_seconds']),'feasible':False,
         'fba_solves':1,'solver_wall_seconds':elapsed,'resource_cost':float(np.mean(x)),
         'resource_efficiency':1-float(np.mean(x))}
    spec=validate_content(job['problem_yaml'])
    for metric in referenced_metrics(spec):
        if metric not in row and metric not in features:row[metric]=float('nan')
    for f,value,norm in zip(features,xp,x):row[f]=float(value);row[f'x01__{f}']=float(norm)
    event={**activity,'status':'glpk_timeout_assumed_infeasible','scientific_infeasibility_proven':False,
           'policy':'SHARED_GLPK_300S_TIMEOUT_AS_INFEASIBLE_V1','recorded_time':time.time(),
           'solver_wall_seconds':elapsed,'source':'external_hard_watchdog','cache_committed':True}
    # Persistent event before the cache is committed; repeated recovery can safely
    # finish a cache commit without touching any previous scientific history.
    with (root/'GLPK_TIMEOUT_EVENTS.jsonl').open('a') as f:
        f.write(json.dumps(event,allow_nan=False,sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())
    atomic_csv(pd.concat([df,pd.DataFrame([row])],ignore_index=True),cache)
    atomic_json(root/'FBA_ACTIVITY.json',{'phase':'timeout_infeasible','pid':None,'time':time.time(),
                'policy':'SHARED_GLPK_300S_TIMEOUT_AS_INFEASIBLE_V1'})

def prepare_general(job,out):
    from .vendor import generic_problem as gp
    from .catalog import runtime_key
    b=job['general_binding']
    if sha(b['model_path'])!=b['model_sha256'] or sha(b['landscape_path'])!=b['landscape_sha256']:
        raise ValueError('Original model/landscape changed since registration; rebind, do not silently use changed inputs')
    if b.get('landscape_config_sha256') and sha(b['landscape_config_path'])!=b['landscape_config_sha256']:
        raise ValueError('Landscape-generation config changed since registration; rebind before running')
    prior=int(job['prior_rows']); full_raw=read_landscape(b['landscape_path'])
    if not 1<=prior<=len(full_raw):raise ValueError('Invalid prefix size for this landscape')
    spec_text=job['problem_yaml'];spec=validate_content(spec_text)
    validate_contract(b,spec)
    pp={'features':b['features'],'bounds':b['bounds'],'feature_labels':{f:f for f in b['features']},
        'biomass_reaction_id':b['biomass_reaction_id'],'product_reaction_id':b['product_reaction_id'],
        'minimum_growth_fraction':b['minimum_growth_fraction'],
        'fba_mode':b['fba_mode'],'objective_reaction_id':b.get('objective_reaction_id',''),
        'objective_direction':b.get('objective_direction','max'),
        'metric_reactions':b.get('metric_reactions',{})}
    # Portable metadata for design-feature objectives, e.g. carbon / oxygen.
    for name in ('carbon_feature','oxygen_feature'):
        if b.get(name): pp[name]=b[name]
    full,comp=gp.compile_frozen_archive(full_raw,pp,spec)
    # Compiled normalization/constraint statistics must come from frozen whole archive.
    baseline=gp.compile_live_exact(full_raw.iloc[:prior],full_raw.iloc[:prior][b['features']],
        {'problem':{**pp,'compiled_problem':comp}},spec)
    key=runtime_key(job['family'],prior,generalized=True)
    runtime=ASSETS/'runtimes'/key
    sys_import_path=str(runtime)
    if sys_import_path not in os.sys.path:os.sys.path.insert(0,sys_import_path)
    case=job['case'];prepared=out/'prepared'/case;prepared.mkdir(parents=True,exist_ok=True)
    bp=prepared/'baseline_compact.csv.gz';baseline.to_csv(bp,index=False,compression="gzip")
    problem={**pp,'compiled_problem':comp,'baseline_table':str(bp),'source_case':case,
             'general_problem_definition':str(out/'problem.gemproblem.yaml'),
             'landscape_prior_rows':prior,'reference_semantics_rows':len(full_raw)}
    parent={'problem':problem}
    (prepared/'problem_config.yaml').write_text(yaml.safe_dump(parent,sort_keys=False))
    (out/'problem.gemproblem.yaml').write_text(spec_text)
    min_feasible=int(gp.as_bool(baseline['feasible']).sum())
    atomic_json(prepared/'landscape_preflight.json',{'status':'PASS' if min_feasible>=4 else 'FAIL',
                'archive_relabel_fba_solves':0,'problem_feasible_rows':min_feasible,
                'prior_rows':prior,'reference_rows':len(full_raw),'problem_hash':comp['problem_hash'],
                'source_order':'existing row order; user must confirm frozen prefix semantics'})
    cfg=yaml.safe_load((runtime/'original_study_config.yaml').read_text())
    cfg['paths']={'prepared_root':str(out/'prepared'),'output_root':str(out/'scientific_results'),
                  'gem_hamed_root':str(out),'runtime_sources_root':str(out/'runtime_sources')}
    cfg['study'].update(seed=int(job['seed']),landscape_prior_rows=prior,prior_label=str(prior),
                        reference_semantics_rows=len(full_raw),screen_total_calls=int(job['budget']))
    cfg['preflight']['required_source_archive_rows']=len(full_raw)
    cfg['preflight']['min_problem_feasible_rows']=4
    cfg['cases']={case:{'case_key':case,'prepared_dir':str(prepared),'model_path':b['model_path'],
                         'study_config_path':str(out/'study_config.yaml'),'source_benchmark':b['landscape_path']}}
    cfg['methods_primary']=[job['method']]
    (out/'study_config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    gc=importlib.import_module('generic_core')
    def hv(df):
        Y=df[comp['objective_norm_columns']].to_numpy(float)
        f=gp.as_bool(df['feasible']) & np.isfinite(Y).all(axis=1)
        return float(gc.hypervolume_exact(Y[f],comp['hv_reference_normalized'])) if f.any() else 0.
    provenance={'runtime_key':key,'algorithm_sha256':sha(runtime/('gsmobo_algorithms.py' if job['family']=='gsmobo' else 'nsga_algorithms.py')),
                'source_model_sha256':b['model_sha256'],'source_archive_sha256':b['landscape_sha256'],
                'problem_hash':comp['problem_hash'],'prior_rows':prior,'reference_rows':len(full_raw),
                'prior_hv':hv(baseline),'full50k_reference_hv':hv(full),
                'reference_label':'full_loaded_landscape_not_necessarily_50k',
                'landscape_seed':b.get('landscape_seed'),'optimizer_seed':job['seed'],'method':job['method'],
                'case':case,'exact_solver':job.get('exact_solver','glpk'),
                'fba_contract':b['fba_mode'],'landscape_config_path':b.get('landscape_config_path',''),
                'is_historical_replay':False,'prefix_is_preserved':True,'archive_preparation_fba_solves':0,
                'exact_evaluator_sha256':sha(__file__),
                'fba_execution_guard':{'policy':'SHARED_GLPK_300S_TIMEOUT_AS_INFEASIBLE_V1',
                    'timeout_seconds':float(os.environ.get('GEM_FBA_TIMEOUT_SECONDS',os.environ.get('GEM_50K_FBA_TIMEOUT_SECONDS','300'))),
                    'retry':False,'timeout_result':'glpk_timeout_assumed_infeasible','scientific_infeasibility_proven':False}}
    atomic_json(out/'provenance.json',provenance)
    return cfg,parent,baseline,runtime,provenance


class GenericExactRunner:
    """Return the unmodified runtime.ExactBatchResult contract.

    Every physical candidate is solved by COBRApy. Cache, ledger, and raw/normalized
    objective columns agree with the parent runner's shape and logical calls.
    """
    def __init__(self,cfg,parent,outroot,case,binding):
        from generic_problem import feature_bounds
        self.cfg=cfg;self.parent=parent;self.outroot=Path(outroot);self.outroot.mkdir(parents=True,exist_ok=True)
        self.binding=binding;self.features,self.bounds=feature_bounds(parent['problem'])
        self.comp=parent['problem']['compiled_problem'];self.decimals=int(cfg['study'].get('exact_cache_round_decimals',12))
        self.cache_path=self.outroot/'exact_cache.csv'
        self.cache={}
        if self.cache_path.exists():
            df=pd.read_csv(self.cache_path)
            cols=[f'x01__{f}' for f in self.features]
            for _,r in df.iterrows():
                self.cache[tuple(np.round([r[k] for k in cols],self.decimals))]=r.to_dict()
        self.model=None
        self.problem_spec=validate_content((self.outroot.parent.parent.parent/'problem.gemproblem.yaml').read_text())
        self.required_metrics=referenced_metrics(self.problem_spec)

    def preflight(self):
        checks={'compiled_problem_valid':int(self.comp['objective_dim'])>=1,
                'model_integrity':sha(self.binding['model_path'])==self.binding['model_sha256'],
                'landscape_integrity':sha(self.binding['landscape_path'])==self.binding['landscape_sha256'],
                'landscape_config_integrity':not self.binding.get('landscape_config_sha256') or sha(self.binding['landscape_config_path'])==self.binding['landscape_config_sha256'],
                'valid_fba_mode':self.binding['fba_mode'] in FBA_MODES}
        try:
            self.model=load_model(self.binding['model_path'])
            rid=set(r.id for r in self.model.reactions)
            needed=set(self.features+[v for v in (self.binding.get('biomass_reaction_id'),self.binding.get('product_reaction_id'),self.binding.get('objective_reaction_id')) if v]+list(self.binding.get('metric_reactions',{}).values()))
            needed|={r for r in self.required_metrics if r not in ('maximum_growth','product_flux','biomass_flux','objective_value','resource_cost','resource_efficiency') and r not in self.features and r not in self.binding.get('metric_reactions',{}) and r not in self.binding.get('derived_metric_recipes',{})}
            checks['model_reaction_ids_valid']=needed<=rid
            if self.binding['feature_mode']=='medium_uptake':
                checks['exchange_feature_ids_valid']=set(self.features)<={r.id for r in self.model.exchanges}
        except Exception as e:checks['model_readable']=False;self.load_error=str(e)
        if not checks.get('model_readable',True):checks['model_error']=self.load_error
        return {'status':'PASS' if all(isinstance(v,bool) and v for v in checks.values()) else 'FAIL',
                'checks':checks,'resolved':{'objective_names':self.comp['objective_names'],'objective_dim':self.comp['objective_dim']}}

    def _solve(self,xphys):
        """Reproduce the FBA protocol recorded for this landscape, not the YAML objective.

        The YAML describes downstream optimization statistics. Changing this
        FBA protocol requires a new measured landscape and its own configuration.
        """
        b=self.binding;mm=self.model.copy();t0=time.perf_counter();solves=0
        # Execution-only guard. The optimizer, GLPK algorithm, presolve setting,
        # model, constraints, basis reset, and FBA scoring are NOT changed.
        timeout = float(os.environ.get('GEM_FBA_TIMEOUT_SECONDS', os.environ.get('GEM_50K_FBA_TIMEOUT_SECONDS', '300')))
        if timeout is not None:
            if not (5 <= timeout <= 3600):
                raise ValueError('GEM_FBA_TIMEOUT_SECONDS must be in [5, 3600]')
            iface = str(getattr(mm.solver.interface, '__name__', '')).lower()
            if 'glpk' not in iface or 'glpk_exact' in iface:
                raise RuntimeError('Expected original optlang GLPK interface; got ' + iface)
            mm.solver.configuration.timeout = int(timeout)
            if mm.solver.configuration.timeout is None:
                raise RuntimeError('Requested GLPK timeout was not configured')

        def bounded_optimize(stage):
            # Native GLPK may be in a C extension: signal progress before and after
            # every physical LP solve, not only after a completed optimizer batch.
            activity = os.environ.get('GEM_50K_HEARTBEAT_PATH') if timeout is not None else None
            if activity: Path(activity).touch()
            began = time.perf_counter()
            def progress(phase,status=None):
                atomic_json(self.outroot/'FBA_ACTIVITY.json',{
                    'phase':phase,'stage':stage,'pid':os.getpid(),'time':time.time(),
                    'timeout_seconds':float(timeout),'hard_watchdog_seconds':float(timeout)+5,
                    'xphys':list(map(float,xphys)),
                    'x01':getattr(self,'_active_candidate_x01',None),
                    'method':getattr(self,'_active_method',None),
                    'batch':getattr(self,'_active_batch',None),
                    'batch_candidate_index':getattr(self,'_active_candidate_index',None),
                    'solver_status':status,'policy':'SHARED_GLPK_300S_TIMEOUT_AS_INFEASIBLE_V1'})
            progress('solving')
            try:
                sol = mm.optimize()
            except Exception as exc:
                elapsed = time.perf_counter() - began
                msg = str(exc).lower()
                if ('time_limit' in msg or 'time limit' in msg or
                    'timelimit' in msg or 'timeout' in msg or 'tmlim' in msg or
                    (timeout is not None and elapsed >= timeout)):
                    raise TechnicalFBATimeout(
                        f'GLPK timeout during {stage}, elapsed={elapsed:.1f}s: {exc}') from exc
                raise
            elapsed = time.perf_counter() - began
            if activity: Path(activity).touch()
            status = str(sol.status).lower().replace(' ', '_')
            progress('returned',status)
            if (status in ('time_limit', 'timelimit', 'timeout', 'tmlim', 'iteration_limit')
                or (timeout is not None and elapsed >= timeout)):
                raise TechnicalFBATimeout(
                    f'GLPK exceeded limit during {stage}, elapsed={elapsed:.1f}s, status={status}')
            return sol
        biomass_id=b.get('biomass_reaction_id') or None
        product_id=b.get('product_reaction_id') or None
        target_id=b.get('objective_reaction_id') or None
        mode=b['fba_mode'];direction=b.get('objective_direction','max')
        try:
            with mm:
                if b['feature_mode']=='medium_uptake':
                    med={} if b.get('close_unlisted_exchanges') else dict(mm.medium)
                    # Fixed controls inherited from the exact sampling catalog.
                    # Users never need to enter these again in Optimization.
                    for fid,val in b.get('fixed_medium',{}).items():med[fid]=float(val)
                    for fid,val in zip(self.features,xphys):med[fid]=max(0,float(val))
                    mm.medium=med
                else:
                    attr='lower_bound' if b['feature_mode']=='reaction_lower_bound' else 'upper_bound'
                    for fid,val in zip(self.features,xphys):setattr(mm.reactions.get_by_id(fid),attr,float(val))
                # A one-stage solve needs neither biomass nor product reaction.
                growth=float('nan')
                if mode in ('growth_constrained_product','growth_constrained_reaction'):
                    biomass=mm.reactions.get_by_id(biomass_id)
                    mm.objective=biomass;mm.objective_direction='max'
                    first=bounded_optimize('growth');solves+=1
                    if first.status!='optimal':
                        return {'feasible':False,'status':str(first.status),'fba_solves':solves,
                                'solver_wall_seconds':time.perf_counter()-t0}
                    growth=float(first.fluxes[biomass_id])
                    # A landscape can additionally impose a global growth-viability
                    # floor before evaluating product; distinct from the local 90% rule.
                    floor=b.get('growth_viability_floor')
                    if floor is not None and growth < float(floor)-1e-12:
                        return {'feasible':False,'status':'below_global_growth_floor',
                                'maximum_growth':growth,'product_flux':float('nan'),
                                'fba_solves':solves,'solver_wall_seconds':time.perf_counter()-t0}
                    biomass.lower_bound=max(float(biomass.lower_bound),float(b['minimum_growth_fraction'])*growth)
                if mode=='maximize_biomass':
                    mm.objective=mm.reactions.get_by_id(biomass_id);mm.objective_direction='max'
                elif mode in ('growth_constrained_product','maximize_product'):
                    mm.objective=mm.reactions.get_by_id(product_id);mm.objective_direction='max'
                elif mode in ('maximize_reaction','growth_constrained_reaction'):
                    mm.objective=mm.reactions.get_by_id(target_id);mm.objective_direction=direction
                # native_objective intentionally retains the GEM's own objective.
                # Preserve the recorded solver-specific *basis* treatment used
                # by the original growth/product bridge. This does not change
                # the mathematical LP or introduce a new optimizer algorithm.
                if (mode=='growth_constrained_product' and b.get('glpk_stage2_basis_reset',False)):
                    solver_iface=str(getattr(mm.solver.interface,'__name__','')).lower()
                    if 'glpk' in solver_iface:
                        import swiglpk as glp
                        glp.glp_std_basis(mm.solver.problem)
                solved=bounded_optimize('primary_or_product');solves+=1
                good=solved.status=='optimal'
                rec={'status':'optimal' if good else str(solved.status),'feasible':bool(good),
                     'fba_solves':solves,'solver_wall_seconds':time.perf_counter()-t0}
                if not good:return rec
                fluxes=solved.fluxes
                if mode=='maximize_biomass':growth=float(fluxes[biomass_id])
                if np.isfinite(growth):rec['maximum_growth']=growth
                if biomass_id:rec['biomass_flux']=float(fluxes[biomass_id])
                if product_id:rec['product_flux']=float(fluxes[product_id])
                rec['objective_value']=float(getattr(solved,'objective_value',float('nan')))
                # Declarative environment stress panel (one exact FBA solve per
                # named stress class). It produces oxygen-growth retention for
                # the v0.2 YAML while retaining the four-scenario *live* FBA
                # contract used by the archived robustness benchmark.
                for metric,recipe in (b.get('derived_metric_recipes') or {}).items():
                    kind=recipe.get('type')
                    if kind not in ('stressed_growth_retention','environmental_stress_panel'):
                        raise ValueError('Unsupported derived exact metric recipe '+str(kind))
                    if not biomass_id or mode!='maximize_biomass':
                        raise ValueError('Stressed-growth evaluation requires a biomass-maximization landscape')
                    groups=(recipe['groups'] if kind=='environmental_stress_panel'
                            else {'oxygen':recipe['reaction_ids']})
                    if not isinstance(groups,dict) or not groups:
                        raise ValueError('Empty environmental stress class definitions')
                    factor=float(recipe['fraction_toward_lower_bound'])
                    if not 0<=factor<=1:raise ValueError('Stress fraction must lie in [0,1]')
                    xmap=dict(zip(self.features,map(float,xphys)))
                    scenario={}
                    for cls,members in groups.items():
                        with mm:
                            medium=dict(mm.medium)
                            for rid in members:
                                if rid not in xmap:raise ValueError('Stress reaction is not a design variable: '+rid)
                                lo=float(self.bounds[self.features.index(rid),0])
                                medium[rid]=max(0.,lo+factor*(xmap[rid]-lo))
                            mm.medium=medium
                            mm.objective=mm.reactions.get_by_id(biomass_id)
                            mm.objective_direction='max'
                            st=bounded_optimize('stress');solves+=1
                            val=float(st.fluxes[biomass_id]) if st.status=='optimal' else 0.0
                            scenario[cls]=val if np.isfinite(val) else 0.0
                            rec[f'{cls}_stressed_growth']=scenario[cls]
                            rec[f'{cls}_stress_status']=str(st.status)
                    oxygen_cls=recipe.get('primary_stress_class','oxygen')
                    if oxygen_cls not in scenario:
                        raise ValueError('Primary stress class not recorded in environmental stress panel')
                    rec['robust_growth']=min(scenario.values())
                    rec['worst_stress_class']=min(scenario,key=scenario.get)
                    rec['oxygen_stressed_growth']=scenario[oxygen_cls]
                    rec[metric]=float(np.clip(scenario[oxygen_cls]/growth,0.,1.+1e-8)) if growth>1e-12 else float('nan')
                    rec['robustness_retention']=float(np.clip(rec['robust_growth']/growth,0.,1.+1e-8)) if growth>1e-12 else float('nan')
                    rec['fba_solves']=solves
                    rec['solver_wall_seconds']=time.perf_counter()-t0
                for key,rid in b.get('metric_reactions',{}).items():rec[key]=float(fluxes[rid])
                # Direct reaction metrics: metric("R_ATP") works when the original
                # landscape recorded the reaction flux as a column under this ID.
                for rid in self.required_metrics:
                    if rid not in rec:
                        try: self.model.reactions.get_by_id(rid)
                        except (KeyError,ValueError): continue
                        else: rec[rid]=float(fluxes[rid])
                return rec
        except TechnicalFBATimeout as exc:
            # User-selected benchmarking policy: consume one logical candidate as
            # infeasible after a 300-second GLPK limit. This is a TECHNICAL
            # PENALTY, *not* proof that its metabolic LP is infeasible.
            enabled=(timeout is not None)
            if not enabled:
                raise
            activity=os.environ.get('GEM_50K_HEARTBEAT_PATH')
            if activity: Path(activity).touch()
            msg=str(exc)
            atomic_json(self.outroot/'FBA_ACTIVITY.json',{'phase':'timeout_infeasible','pid':os.getpid(),'time':time.time(),'detail':msg})
            return {'status':'glpk_timeout_assumed_infeasible',
                    'fba_solver_status':'glpk_time_limit',
                    'technical_timeout_assumed_infeasible':True,
                    'timeout_detail':msg,
                    'timeout_seconds':float(timeout),
                    'feasible':False,
                    'fba_solves':solves+1,
                    'solver_wall_seconds':time.perf_counter()-t0}
        except Exception as e:
            return {'status':'solver_error: '+str(e),'feasible':False,
                    'fba_solves':solves,'solver_wall_seconds':time.perf_counter()-t0}

    def evaluate(self,X01,method,batch_no):
        from runtime import ExactBatchResult,unnormalize_x,key_for_x
        from generic_problem import compile_live_exact,as_bool
        X01=np.atleast_2d(np.asarray(X01,float));Xphys=unnormalize_x(X01,self.bounds)
        rows=[None]*len(X01);hits=np.zeros(len(X01),bool)
        total_wall=0.;solves=0;physical=0
        for i,x in enumerate(X01):
            key=key_for_x(x,self.decimals)
            if key in self.cache:
                rows[i]=dict(self.cache[key]);hits[i]=True;continue
            self._active_candidate_x01=list(map(float,x));self._active_method=method
            self._active_batch=int(batch_no);self._active_candidate_index=int(i)
            physical+=1;ev=self._solve(Xphys[i]);total_wall+=ev['solver_wall_seconds'];solves+=ev['fba_solves']
            if ev.get('technical_timeout_assumed_infeasible'):
                # Persist an independent event immediately, before the batch is
                # committed; record the real technical cause and selected penalty.
                event={'method':method,'batch':int(batch_no),'batch_candidate_index':int(i),
                       'x01':[float(v) for v in x],
                       'status':'glpk_timeout_assumed_infeasible',
                       'scientific_infeasibility_proven':False,
                       'policy':'GLPK_ONLY_300S_TIMEOUT_AS_INFEASIBLE',
                       'timeout_seconds':float(ev['timeout_seconds']),
                       'detail':ev['timeout_detail'],
                       'solver_wall_seconds':float(ev['solver_wall_seconds'])}
                event_path=self.outroot/'GLPK_TIMEOUT_EVENTS.jsonl'
                with event_path.open('a') as out_event:
                    out_event.write(json.dumps(event,allow_nan=False,sort_keys=True)+'\n')
                    out_event.flush()
                    os.fsync(out_event.fileno())
                print('GLPK_TIMEOUT_ASSUMED_INFEASIBLE '
                      f'{method} batch={batch_no} candidate={i} '
                      f'seconds={event["solver_wall_seconds"]:.1f}',flush=True)
            cost=float(np.mean(x));ev.update(resource_cost=cost,resource_efficiency=1-cost)
            # Failed physical solves must retain the full metric schema so that
            # a batch with both feasible and failed designs can be compiled.
            for metric in self.required_metrics:
                if metric not in ev and metric not in self.features:
                    ev[metric]=float('nan')
            for j,f in enumerate(self.features):ev[f]=float(Xphys[i,j]);ev[f'x01__{f}']=float(x[j])
            self.cache[key]=ev;rows[i]=ev
        if physical:atomic_csv(pd.DataFrame(self.cache.values()),self.cache_path)
        raw=pd.DataFrame(rows)
        exact=compile_live_exact(raw,raw[self.features],self.parent,self.problem_spec)
        # Compilation classifies any failed FBA as source_infeasible. Re-label
        # timeout-penalty rows to preserve the distinction in calls.csv and all
        # per-call scientific status lists, while keeping feasible=False.
        timeout_rows=(raw['technical_timeout_assumed_infeasible'].astype(str).str.lower().isin(('true','1','yes'))
                      if 'technical_timeout_assumed_infeasible' in raw else pd.Series(False,index=raw.index))
        if timeout_rows.any():
            exact.loc[timeout_rows,'status']='glpk_timeout_assumed_infeasible'
            exact.loc[timeout_rows,'feasible']=False
        # compile_live_exact returns all compiled columns and source fields.
        cols=self.comp['objective_norm_columns'];rcols=self.comp['objective_raw_columns']
        feasible=as_bool(exact['feasible']);Yraw=exact[rcols].to_numpy(float);Yn=exact[cols].to_numpy(float);Yn[~feasible]=np.nan
        ledger=self.outroot/'exact_runtime_ledger.csv';lr=pd.DataFrame([{
            'method':method,'batch':int(batch_no),'logical_evals':len(X01),'physical_exact_evals':physical,
            'physical_fba_solves':solves,'cell_cache_hits':int(hits.sum()),'source_raw_cache_hits':0,'cache_hits':int(hits.sum()),
            'physical_exact_wall_seconds':total_wall,'exact_bridge_wall_seconds':total_wall}])
        if ledger.exists():lr=pd.concat([pd.read_csv(ledger),lr],ignore_index=True)
        atomic_csv(lr,ledger)
        return ExactBatchResult(X01,Xphys,Yraw,Yn,feasible,exact['status'].astype(str).tolist(),hits,exact,physical,total_wall)
