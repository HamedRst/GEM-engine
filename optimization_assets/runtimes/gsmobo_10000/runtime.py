from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc

from generic_core import pareto_points_generic, hypervolume_exact
from generic_problem import feature_bounds, normalize_design


def ensure_dir(path):
    p=Path(path); p.mkdir(parents=True,exist_ok=True); return p


def key_for_x(x, decimals=12):
    return tuple(np.round(np.asarray(x,float),int(decimals)).tolist())


def sobol_points(n,d,seed):
    """Deterministic scrambled Sobol points without non-power-of-two warnings.

    Generate the next power-of-two net and take the requested prefix.  This keeps
    all call sites dimension-generic while preserving Sobol balance structure.
    """
    n=int(n);d=int(d)
    if n<=0: return np.empty((0,d),float)
    m=int(np.ceil(np.log2(max(1,n))))
    return qmc.Sobol(d=d,scramble=True,seed=int(seed)).random_base2(m)[:n]


def normalize_x(X,bounds):
    X=np.asarray(X,float); b=np.asarray(bounds,float); return (X-b[:,0])/(b[:,1]-b[:,0])


def unnormalize_x(X01,bounds):
    X01=np.asarray(X01,float); b=np.asarray(bounds,float); return b[:,0]+X01*(b[:,1]-b[:,0])


def duration(sec):
    sec=max(0,int(round(float(sec)))); h,r=divmod(sec,3600); m,s=divmod(r,60); return f"{h:02d}:{m:02d}:{s:02d}"


def hv(Y,ref):
    Y=np.asarray(Y,float); ref=np.asarray(ref,float)
    if Y.ndim!=2 or Y.shape[1]!=len(ref): raise RuntimeError(f"HV shape mismatch: {Y.shape}, ref={ref.shape}")
    Y=Y[np.isfinite(Y).all(axis=1)]
    if not len(Y): return 0.0
    return float(hypervolume_exact(Y,ref=ref))


@dataclass
class ExactBatchResult:
    X01: np.ndarray
    Xphys: np.ndarray
    Yraw: np.ndarray
    Yn: np.ndarray
    feasible: np.ndarray
    status: list[str]
    cache_hit: np.ndarray
    raw: pd.DataFrame
    physical_exact_evals: int
    physical_exact_wall_seconds: float


@dataclass
class Pool:
    X: np.ndarray
    pred: np.ndarray
    pinf: np.ndarray
    safe: np.ndarray
    novelty: np.ndarray


class ExactRunner:
    def __init__(self,cfg,parent_cfg,outroot,case):
        self.cfg=cfg; self.parent_cfg=parent_cfg; self.outroot=ensure_dir(outroot); self.case=dict(case)
        p=parent_cfg.get('problem',parent_cfg)
        self.features,self.bounds=feature_bounds(p)
        self.bridge=Path(__file__).with_name('multi_problem_bridge.py').resolve()
        self.study_config=Path(case['study_config_path']).resolve(); self.case_key=str(case['case_key'])
        self.decimals=int(cfg.get('study',{}).get('exact_cache_round_decimals',12))
        self.cache_path=self.outroot/'exact_cache.csv'; self.cache=self._load_cache()

    def _load_cache(self):
        out={}
        if not self.cache_path.exists(): return out
        try: df=pd.read_csv(self.cache_path)
        except Exception: return out
        cols=[f'x01__{f}' for f in self.features]
        if not all(c in df.columns for c in cols): return out
        for _,r in df.iterrows():
            out[key_for_x([r[c] for c in cols],self.decimals)]=r.to_dict()
        return out

    def _compiled(self):
        import yaml
        pp=Path(self.case['prepared_dir'])/'problem_config.yaml'
        obj=yaml.safe_load(pp.read_text(encoding='utf-8')) or {}; p=obj.get('problem',obj); comp=p.get('compiled_problem')
        if not isinstance(comp,dict): raise RuntimeError(f'Prepared case has no compiled_problem: {pp}')
        return p,comp

    def preflight(self):
        problem_path=Path(self.case['prepared_dir'])/'problem_config.yaml'; lp=Path(self.case['prepared_dir'])/'landscape_preflight.json'
        checks={'bridge_exists':self.bridge.exists(),'prepared_problem_exists':problem_path.exists(),'landscape_preflight_exists':lp.exists()}
        try:
            probe=subprocess.run([sys.executable,'-c','import cobra,torch,botorch,gpytorch; print("ok")'],capture_output=True,text=True,timeout=60)
            checks['python_dependencies']=probe.returncode==0
        except Exception: checks['python_dependencies']=False
        resolved={}
        if problem_path.exists():
            try:
                _,c=self._compiled(); m=int(c.get('objective_dim',0)); checks['compiled_problem_valid']=m>=1 and len(c.get('objective_names',[]))==m and len(c.get('hv_reference_normalized',[]))==m
                resolved={'problem_id':c.get('problem_id'),'problem_hash':c.get('problem_hash'),'objective_dim':m,'objective_names':c.get('objective_names')}
            except Exception: checks['compiled_problem_valid']=False
        if lp.exists():
            try:
                o=json.loads(lp.read_text()); checks['landscape_preflight_pass']=o.get('status')=='PASS' and int(o.get('archive_relabel_fba_solves',-1))==0 and int(o.get('problem_feasible_rows',0))>=int(self.cfg.get('preflight',{}).get('min_problem_feasible_rows',50))
            except Exception: checks['landscape_preflight_pass']=False
        return {'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'resolved':resolved}

    def _stream(self,cmd,log_path,timeout=None):
        log_path=Path(log_path); log_path.parent.mkdir(parents=True,exist_ok=True)
        with log_path.open('w',encoding='utf-8') as log:
            proc=subprocess.Popen(list(map(str,cmd)),stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                print(line,end='',flush=True); log.write(line); log.flush()
            rc=proc.wait()
        if rc!=0: raise RuntimeError(f'Exact bridge failed rc={rc}; see {log_path}')

    def evaluate(self,X01,method,batch_no):
        p,comp=self._compiled(); raw_cols=list(comp['objective_raw_columns']); norm_cols=list(comp['objective_norm_columns']); m=int(comp['objective_dim'])
        X01=np.atleast_2d(np.asarray(X01,float)); Xphys=unnormalize_x(X01,self.bounds)
        rows=[None]*len(X01); hits=np.zeros(len(X01),bool); missing=[]
        for i,x in enumerate(X01):
            k=key_for_x(x,self.decimals)
            if k in self.cache: rows[i]=dict(self.cache[k]); hits[i]=True
            else: missing.append(i)
        cell_cache_hits=int(hits.sum()); wall=0.0; exact=pd.DataFrame(); source_cache_hit=np.zeros(0,bool)
        if missing:
            root=ensure_dir(self.outroot/'_exact_runs'/method/f'batch_{int(batch_no):03d}'); attempt=1
            while (root/f'attempt_{attempt:02d}').exists(): attempt+=1
            rd=ensure_dir(root/f'attempt_{attempt:02d}')
            prop=pd.DataFrame(Xphys[np.asarray(missing,int)],columns=self.features); prop.insert(0,'proposal_id',[f'{method}__b{batch_no:03d}__q{i+1:03d}' for i in range(len(prop))])
            pp=rd/'proposal.csv'; rp=rd/'exact_results.csv'; prop.to_csv(pp,index=False)
            cmd=[sys.executable,str(self.bridge),'evaluate','--study-config',str(self.study_config),'--case-key',self.case_key,'--proposal',str(pp),'--output',str(rp)]
            t=time.time(); self._stream(cmd,rd/'exact_bridge.log'); wall=time.time()-t
            exact=pd.read_csv(rp)
            if len(exact)!=len(missing): raise RuntimeError(f'Exact bridge returned {len(exact)} rows for {len(missing)}')
            source_cache_hit=(exact['source_raw_cache_hit'].astype(str).str.lower().isin(['true','1','yes']).to_numpy(bool) if 'source_raw_cache_hit' in exact.columns else np.zeros(len(exact),bool))
            exact_skipped=(exact['exact_skipped'].astype(str).str.lower().isin(['true','1','yes']).to_numpy(bool) if 'exact_skipped' in exact.columns else (exact['source_exact_skipped'].astype(str).str.lower().isin(['true','1','yes']).to_numpy(bool) if 'source_exact_skipped' in exact.columns else np.zeros(len(exact),bool)))
            miss=[c for c in raw_cols+norm_cols+['feasible','status'] if c not in exact.columns]
            if miss: raise RuntimeError(f'Exact bridge missing columns: {miss}')
            for j,ii in enumerate(missing):
                rec=exact.iloc[j].to_dict(); fv=rec.get('feasible',False)
                rec['feasible']=bool(fv) if isinstance(fv,(bool,np.bool_)) else str(fv).strip().lower() in {'true','1','yes'}
                for k,f in enumerate(self.features): rec[f'x01__{f}']=float(X01[ii,k]); rec[f]=float(Xphys[ii,k])
                rec['source_method']=method; rec['source_batch']=int(batch_no); rec['exact_skipped']=bool(exact_skipped[j])
                rows[ii]=rec
                # A computationally skipped exact point is committed to this method's
                # logical history, but never cached as biological truth.
                if not bool(exact_skipped[j]):
                    self.cache[key_for_x(X01[ii],self.decimals)]=dict(rec)
                if bool(source_cache_hit[j]): hits[ii]=True
            pd.DataFrame(list(self.cache.values())).to_csv(self.cache_path,index=False)
            physical_candidates=int((~source_cache_hit).sum())
            nsolve=int(pd.to_numeric(exact.get('fba_solves',pd.Series(dtype=float)),errors='coerce').fillna(0).sum())
            nskipped=int(exact_skipped.sum())
            print(f'[exact] {method} batch {batch_no}: {physical_candidates}/{len(missing)} physical source candidates, {nsolve} FBA solves, skipped_exact={nskipped} in {duration(wall)}',flush=True)
        out=pd.DataFrame(rows); feasible=out['feasible'].astype(bool).to_numpy(); Yraw=out[raw_cols].apply(pd.to_numeric,errors='coerce').to_numpy(float); Yn=out[norm_cols].apply(pd.to_numeric,errors='coerce').to_numpy(float); Yn[~feasible]=np.nan
        ledger=self.outroot/'exact_runtime_ledger.csv'; nsolve=int(pd.to_numeric(exact.get('fba_solves',pd.Series(dtype=float)),errors='coerce').fillna(0).sum()) if len(exact) else 0
        physical_exact_evals=int((~source_cache_hit).sum()) if len(exact) else 0
        physical_solver_wall=float(pd.to_numeric(exact.get('solver_wall_seconds',pd.Series(dtype=float)),errors='coerce').fillna(0).sum()) if len(exact) else 0.0
        source_raw_cache_hits=int(source_cache_hit.sum()) if len(exact) else 0
        exact_skipped_count=int(exact_skipped.sum()) if len(exact) else 0
        lr=pd.DataFrame([{'method':method,'batch':int(batch_no),'logical_evals':len(X01),'physical_exact_evals':physical_exact_evals,'physical_fba_solves':nsolve,'exact_skipped':exact_skipped_count,'cell_cache_hits':cell_cache_hits,'source_raw_cache_hits':source_raw_cache_hits,'cache_hits':int(hits.sum()),'physical_exact_wall_seconds':physical_solver_wall,'exact_bridge_wall_seconds':wall}])
        if ledger.exists(): lr=pd.concat([pd.read_csv(ledger),lr],ignore_index=True)
        lr.to_csv(ledger,index=False)
        return ExactBatchResult(X01,Xphys,Yraw,Yn,feasible,out['status'].astype(str).tolist(),hits,out,physical_exact_evals,physical_solver_wall)


class LiveContext:
    def __init__(self,cfg,parent_cfg,baseline,baseline_path,global_model,subset_local,subset_oof_prior,outdir,exact_runner,seed,budget):
        self.cfg=cfg; self.parent_cfg=parent_cfg; self.baseline=baseline.reset_index(drop=True); self.baseline_path=Path(baseline_path); self.global_model=global_model; self.subset_local=np.asarray(subset_local,int); self.subset_oof_prior=np.asarray(subset_oof_prior,float); self.outdir=ensure_dir(outdir); self.exact_runner=exact_runner; self.seed=int(seed); self.budget=int(budget); self.batch_size=int(cfg['study']['batch_size'])
        p=parent_cfg.get('problem',parent_cfg); self.features,self.bounds=feature_bounds(p); self.feature_labels=dict(p.get('feature_labels',{})); comp=p['compiled_problem']; self.compiled_problem=comp; self.objective_names=list(comp['objective_names']); self.objective_directions=list(comp['objective_directions']); self.objective_norm_columns=list(comp['objective_norm_columns']); self.objective_raw_columns=list(comp['objective_raw_columns']); self.objective_dim=int(comp['objective_dim']); self.hv_ref=tuple(float(x) for x in comp['hv_reference_normalized'])
        self.X0=normalize_design(self.baseline,self.features,self.bounds); self.Y0=self.baseline[self.objective_norm_columns].to_numpy(float); self.F0=self.baseline['feasible'].astype(bool).to_numpy(); self.good0=self.F0 & np.isfinite(self.Y0).all(axis=1); self.baseline_front=pareto_points_generic(self.Y0[self.good0]); self.baseline_hv=hv(self.baseline_front,self.hv_ref); self.baseline_tree=cKDTree(self.X0); self.safe_threshold=float(cfg.get('surrogate',{}).get('infeasible_probability_threshold',0.10)); self.min_d50k=float(cfg['study'].get('min_distance_to_frozen_50k',1e-8)); self.min_dwithin=float(cfg['study'].get('min_distance_within_method',1e-8))
        if len(self.subset_oof_prior) and self.subset_oof_prior.shape!=(len(self.subset_local),self.objective_dim): raise RuntimeError('subset OOF shape mismatch')

    def method_dir(self,m): return ensure_dir(self.outdir/m)
    def history(self,m):
        p=self.method_dir(m)/'calls.csv'; return pd.read_csv(p) if p.exists() else pd.DataFrame()
    def save_history(self,m,df): df.to_csv(self.method_dir(m)/'calls.csv',index=False)
    def current_front(self,m):
        f=self.baseline_front.copy(); h=self.history(m)
        if len(h):
            ok=h['exact_feasible'].astype(bool).to_numpy(); Y=h[self.objective_norm_columns].to_numpy(float); ok &= np.isfinite(Y).all(axis=1)
            if ok.any(): f=pareto_points_generic(np.vstack([f,Y[ok]]))
        return f
    def live_self_front(self,m):
        h=self.history(m)
        if not len(h): return np.empty((0,self.objective_dim),float)
        ok=h['exact_feasible'].astype(bool).to_numpy();Y=h[self.objective_norm_columns].to_numpy(float);ok &= np.isfinite(Y).all(axis=1)
        return pareto_points_generic(Y[ok]) if ok.any() else np.empty((0,self.objective_dim),float)
    def current_delta_hv(self,m):
        if str(m)=='BASE_GSMOBO_QPAREGO_NO50K': return hv(self.live_self_front(m),self.hv_ref)
        return hv(self.current_front(m),self.hv_ref)-self.baseline_hv
    def _filter_new(self,X,m,exclude_frozen_archive=True):
        X=np.atleast_2d(np.asarray(X,float)); keep=np.ones(len(X),bool)
        if exclude_frozen_archive and len(X): keep &= self.baseline_tree.query(X,k=1)[0] > self.min_d50k
        h=self.history(m)
        if len(h):
            cols=[f'x01__{f}' for f in self.features]
            if all(c in h.columns for c in cols): keep &= cKDTree(h[cols].to_numpy(float)).query(X,k=1)[0] > self.min_dwithin
        seen=set()
        for i,x in enumerate(X):
            k=key_for_x(x,12)
            if k in seen: keep[i]=False
            seen.add(k)
        return X[keep]
    def make_pool(self,m,batch_no,rows=None,seed_offset=0):
        n=int(rows or self.cfg['study']['pool_rows']); seed=self.seed+1009*int(batch_no)+int(seed_offset); X=self._filter_new(sobol_points(max(n*2,n+64),len(self.features),seed),m)
        if len(X)<n: X=np.vstack([X,self._filter_new(sobol_points(n*2,len(self.features),seed+17),m)])
        X=X[:n]; pred,pinf=self.global_model.predict(X); pred=np.asarray(pred,float); pinf=np.asarray(pinf,float).reshape(-1); safe=np.isfinite(pred).all(axis=1)&np.isfinite(pinf)&(pinf<=self.safe_threshold); novelty=self.baseline_tree.query(X,k=1)[0]
        return Pool(X,pred,pinf,safe,novelty)
    def evaluate_batch(self,m,X01,modes=None,metadata=None):
        h=self.history(m); start=len(h); b=start//self.batch_size+1; X01=self._filter_new(X01,m)
        if not len(X01): raise RuntimeError(f'{m}: no genuinely new candidates')
        X01=X01[:max(0,self.budget-start)]; modes=list(modes or [m]*len(X01))[:len(X01)]; pred,pinf=self.global_model.predict(X01); pred=np.asarray(pred,float); pinf=np.asarray(pinf,float).reshape(-1); result=self.exact_runner.evaluate(X01,m,b)
        front=self.current_front(m); rows=[]
        for i,x in enumerate(X01):
            before=hv(front,self.hv_ref); y=result.Yn[i]; feas=bool(result.feasible[i] and np.isfinite(y).all())
            if feas: front=pareto_points_generic(np.vstack([front,y]))
            after=hv(front,self.hv_ref); raw=result.raw.iloc[i]
            skipped=bool(str(raw.get('exact_skipped',raw.get('source_exact_skipped',False))).strip().lower() in {'true','1','yes'})
            rec={'method':m,'call':start+i+1,'batch':b,'mode':modes[i],'exact_feasible':feas,'exact_skipped':skipped,'status':result.status[i],'cache_hit':bool(result.cache_hit[i]),'pred_p_infeasible':float(pinf[i]),'exact_hv_gain_this_call':max(0.0,after-before),'exact_front_extension':bool(after>before+1e-14),'archive_hv_after':after,'delta_hv_after':after-self.baseline_hv}
            for j,o in enumerate(self.compiled_problem['objectives']):
                slug=o['slug']; rc=self.objective_raw_columns[j]; nc=self.objective_norm_columns[j]; rec[f'pred_objective__{slug}']=float(pred[i,j]); rec[f'exact_raw__{slug}']=float(result.Yraw[i,j]) if np.isfinite(result.Yraw[i,j]) else np.nan; rec[nc]=float(y[j]) if np.isfinite(y[j]) else np.nan; rec[rc]=float(pd.to_numeric(raw.get(rc,np.nan),errors='coerce'))
            for j,f in enumerate(self.features): rec[f'x01__{f}']=float(x[j]); rec[f]=float(result.Xphys[i,j])
            if metadata:
                for k,v in metadata.items():
                    if isinstance(v,(list,tuple,np.ndarray)) and len(v)==len(X01):
                        vv=v[i]
                        if np.isscalar(vv): rec[k]=vv
                    elif np.isscalar(v): rec[k]=v
            rows.append(rec)
        allh=pd.concat([h,pd.DataFrame(rows)],ignore_index=True) if len(h) else pd.DataFrame(rows); self.save_history(m,allh); return allh
    def summary(self,m,runtime_seconds=None):
        h=self.history(m)
        if not len(h): return {'method':m,'calls':0}
        feas=h['exact_feasible'].astype(bool)
        skipped=(h['exact_skipped'].astype(str).str.lower().isin(['true','1','yes']) if 'exact_skipped' in h.columns else pd.Series(False,index=h.index))
        evaluated=~skipped
        if str(m)=='BASE_GSMOBO_QPAREGO_NO50K':
            final=hv(self.live_self_front(m),self.hv_ref);baseline_report=0.0;delta=final;head=1.0
        else:
            final=float(h.iloc[-1]['archive_hv_after']);baseline_report=self.baseline_hv;delta=final-self.baseline_hv;head=max(1e-12,1.0-self.baseline_hv)
        denom=int(evaluated.sum())
        feasible_fraction=(float((feas & evaluated).sum())/denom) if denom else np.nan
        out={'method':m,'calls':len(h),'objective_dim':self.objective_dim,'objective_names':'|'.join(self.objective_names),'baseline_hv':baseline_report,'final_archive_hv':final,'final_delta_hv':delta,'headroom_normalized_delta_hv':delta/head,'exact_feasible_fraction':feasible_fraction,'exact_skipped':int(skipped.sum()),'exact_successful_calls':denom,'exact_extensions':int(h['exact_front_extension'].astype(bool).sum()),'physical_exact_calls':int((~h['cache_hit'].astype(bool)).sum()),'cache_hits':int(h['cache_hit'].astype(bool).sum())}
        if runtime_seconds is not None: out['runtime_seconds']=float(runtime_seconds)
        return out
