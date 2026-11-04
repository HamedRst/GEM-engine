#!/usr/bin/env python3
"""ZERO-FBA real BoTorch structural smoke; runs before any exact biological call.

Exercises both ordinary all-stochastic qParEGO and mixed problems where known
objective channels are deterministic functions of the candidate design.  This
specifically guards the PCO/carbon-oxygen and resource-cost contracts that have
previously been vulnerable to accidental genericization.
"""
import numpy as np
try:
    import torch,botorch,gpytorch  # noqa: F401
    from generic_core import fit_gsmobo_gp_generic
    from gsmobo_algorithms import _make_acq,_optimize_one,_acq_vals
except Exception as e:
    raise SystemExit(f'BOTORCH PREFLIGHT IMPORT FAIL: {type(e).__name__}: {e}')

rng=np.random.default_rng(20260822)


def check(X,Y,contracts,features,w,seed,label):
    model=fit_gsmobo_gp_generic(X,Y,contracts,features)
    acq=_make_acq(model,X,Y,w,seed,contracts,features)
    x,v=_optimize_one(acq,X.shape[1],3,64,40)
    vals=_acq_vals(acq,np.vstack([x,rng.random((3,X.shape[1]))]))
    assert x.shape==(X.shape[1],) and np.isfinite(x).all()
    assert np.all((x>=0)&(x<=1)) and np.isfinite(v) and len(vals)==4 and np.isfinite(vals).all()
    print(f'  PASS {label}')

# Ordinary all-stochastic 3-objective search.
X=rng.random((40,5))
Y=np.column_stack([np.sin(3*X[:,0])+X[:,1],1-X[:,2]**2,0.3+0.7*X[:,3]])
Y=(Y-Y.min(0))/np.maximum(Y.max(0)-Y.min(0),1e-9)
contracts=[{'name':f'o{i}','direction':'maximize','deterministic_from_design':False} for i in range(3)]
check(X,Y,contracts,[f'x{i}' for i in range(5)],np.asarray([.2,.5,.3]),1234,'all-stochastic m=3')

# PCO-like: exact/stochastic product + deterministic C/O efficiencies.
X=rng.random((48,5)); product=np.clip(.15+.65*X[:,0]+.15*np.sin(5*X[:,3]),0,1)
Y=np.column_stack([product,1-X[:,1],1-X[:,2]])
contracts=[
 {'name':'product','direction':'maximize','deterministic_from_design':False},
 {'name':'carbon_efficiency','direction':'maximize','source':'design_feature','transform':'one_minus_normalized','resolved_feature_index':1,'deterministic_from_design':True,'resolved_normalization':{'method':'fixed_range','min':0.0,'max':1.0}},
 {'name':'oxygen_efficiency','direction':'maximize','source':'design_feature','transform':'one_minus_normalized','resolved_feature_index':2,'deterministic_from_design':True,'resolved_normalization':{'method':'fixed_range','min':0.0,'max':1.0}},
]
check(X,Y,contracts,[f'x{i}' for i in range(5)],np.asarray([.45,.25,.30]),2234,'mixed PCO m=3 (1 stochastic + 2 exact design channels)')

# Cost-like: exact/stochastic product + deterministic resource efficiency after
# minimizing normalized mean resource cost.
X=rng.random((48,7)); product=np.clip(.10+.75*X[:,0]+.10*np.sin(4*X[:,2]),0,1); cost=X.mean(1)
Y=np.column_stack([product,1-cost])
contracts=[
 {'name':'product','direction':'maximize','deterministic_from_design':False},
 {'name':'resource_cost','direction':'minimize','source':'design_metric','metric':'normalized_mean','deterministic_from_design':True,'resolved_normalization':{'method':'fixed_range','min':0.0,'max':1.0}},
]
check(X,Y,contracts,[f'x{i}' for i in range(7)],np.asarray([.65,.35]),3234,'mixed cost m=2 (1 stochastic + exact design cost)')



# Scalar constrained-product path: explicit one-objective qLogEI (no synthetic
# multiobjective scalarizer required).
X=rng.random((40,5)); Y=np.clip(.10+.75*X[:,0]+.08*np.sin(4*X[:,1]),0,1)[:,None]
contracts=[{'name':'product','direction':'maximize','deterministic_from_design':False}]
check(X,Y,contracts,[f'x{i}' for i in range(5)],np.asarray([1.0]),4234,'scalar m=1 constrained-product acquisition')

# End-to-end guided gsMOBO smoke under the mixed PCO contract. This exercises
# actual native-vs-guided acquisition comparison, pending points, and all three
# structural adapters before a biological call is allowed.
from scipy.spatial import cKDTree
import pandas as pd
from gsmobo_algorithms import (_qparego_guided_vb,_qparego_guided_v111,_qparego_guided_A, representative_prior_ids, GHOST_VB,SPARC_VB,TRAIL_VB,GHOST_V111,SPARC_V111,TRAIL_V111,GHOST_A,SPARC_A,TRAIL_A)
from runtime import key_for_x

class _DummyGlobal:
    def predict(self,X):
        X=np.atleast_2d(np.asarray(X,float)); prod=np.clip(.12+.66*X[:,0]+.10*np.sin(5*X[:,3]),0,1)
        return np.column_stack([prod,1-X[:,1],1-X[:,2]]), np.clip(.15+.55*(1-X[:,0]),0,1)

class _GuidedCtx:
    def __init__(self):
        self.features=[f'x{i}' for i in range(5)]; self.objective_dim=3; self.objective_norm_columns=['objective_norm__product','objective_norm__carbon','objective_norm__oxygen']
        self.compiled_problem={'objectives':[
            {'name':'product','slug':'product','direction':'maximize','deterministic_from_design':False},
            {'name':'carbon','slug':'carbon','direction':'maximize','source':'design_feature','transform':'one_minus_normalized','resolved_feature_index':1,'deterministic_from_design':True,'resolved_normalization':{'method':'fixed_range','min':0.0,'max':1.0}},
            {'name':'oxygen','slug':'oxygen','direction':'maximize','source':'design_feature','transform':'one_minus_normalized','resolved_feature_index':2,'deterministic_from_design':True,'resolved_normalization':{'method':'fixed_range','min':0.0,'max':1.0}},
        ]}
        self.cfg={'study':{'exact_cache_round_decimals':12,'min_distance_within_method':1e-8},'gsmobo_dev':{'min_feasible_live_for_no50k_gp':2,'num_restarts':2,'raw_samples':32,'maxiter':30,'structure_archive_rows':256,'structure_pool_rows':64,'structure_top_restarts':3,'structure_refine_maxiter':24,'protected_parent_slots':1,'guided_acq_noninferiority_tolerance':1e-6,'legacy_vb_min_live_feasible':2,'legacy_vb_infeasibility_penalty':0.35,'legacy_vb_gate_tolerance':1e-8,'legacy_structure_pool_rows':64}}
        self.seed=20260822
        self.X0=rng.random((384,5)); prod=np.clip(.12+.66*self.X0[:,0]+.10*np.sin(5*self.X0[:,3]),0,1); self.Y0=np.column_stack([prod,1-self.X0[:,1],1-self.X0[:,2]])
        self.F0=(self.X0[:,0]+.15*self.X0[:,4]>.12); self.good0=self.F0&np.isfinite(self.Y0).all(1); self.baseline_tree=cKDTree(self.X0); self.global_model=_DummyGlobal()
        # GHOST has a real underprediction pocket.
        self.oof_all=self.Y0.copy(); self.oof_all[:80,0]-=.12
        self.subset_local=representative_prior_ids(self.X0,self.Y0,self.F0,128,8891)
        H=rng.random((16,5)); hp=np.clip(.12+.66*H[:,0]+.10*np.sin(5*H[:,3]),0,1); HY=np.column_stack([hp,1-H[:,1],1-H[:,2]])
        d={f'x01__{f}':H[:,j] for j,f in enumerate(self.features)}
        for j,c in enumerate(self.objective_norm_columns): d[c]=HY[:,j]
        d['exact_feasible']=np.ones(16,bool); self._h=pd.DataFrame(d)
    def history(self,method): return self._h.copy()
    def _filter_new(self,X,method,exclude_frozen_archive=True):
        X=np.atleast_2d(np.asarray(X,float)); out=[]; seen=set()
        for x in X:
            if not np.isfinite(x).all() or np.any(x<0) or np.any(x>1): continue
            k=key_for_x(x)
            if k in seen: continue
            seen.add(k); out.append(x)
        return np.asarray(out,float) if out else np.empty((0,len(self.features)),float)

ctx=_GuidedCtx()
variants=[
 ('LEGACY_VB',_qparego_guided_vb,[(GHOST_VB,'ghost'),(SPARC_VB,'sparc'),(TRAIL_VB,'trail')]),
 ('TRUE_NATIVE_V111',_qparego_guided_v111,[(GHOST_V111,'ghost'),(SPARC_V111,'sparc'),(TRAIL_V111,'trail')]),
 ('SYSTEMATIC_A',_qparego_guided_A,[(GHOST_A,'ghost'),(SPARC_A,'sparc'),(TRAIL_A,'trail')]),
]
for vv,(label,fn,pairs) in enumerate(variants,1):
    for k,(meth,mode) in enumerate(pairs,1):
        Xg,modes,md,diag=fn(ctx,meth,mode,4,6234+1000*vv+100*k)
        assert Xg.shape==(4,5) and np.isfinite(Xg).all() and np.all((Xg>=0)&(Xg<=1))
        assert len(modes)==4 and len(md['guided'])==4 and md['guided'][0]==0 and len(md['guidance_reason'])==4
        assert md['implementation_lineage']==label and diag['parent_kind']=='qparego'
        print(f'  PASS {label} end-to-end {mode}: guided={sum(md["guided"])}/4')

print('REAL BOTORCH THREE-LINEAGE STRUCTURAL SMOKE: PASS | ZERO FBA')
