from __future__ import annotations

"""Systematic gsMOBO family development methods.

All methods share the same 16-call exact Sobol initialization.  The first
informative development endpoint is therefore total call 32 (16 common warm-up
+ 16 adaptive calls).  No problem/species branch exists in this module.

Scientific contracts
--------------------
NO50K: live exact history only after the common warm-up.
+50K: one deterministic representative 50K prior + live exact history.
      Infeasible prior/live rows are retained as zero objective outcomes, matching
      the established GEM gsMOBO convention; the frozen feasibility classifier is
      diagnostic only and never a hard candidate veto.
GHOST/SPARC/TRAIL: each scientific heuristic is tested through three family-
      native integration lineages. LEGACY_VB ports the historical parent-first
      problem-aware surrogate-gating behavior; TRUE_NATIVE_V111 ports the
      v1.1.1 qLogEI restart behavior; SYSTEMATIC_A adds advanced preference-
      specific structure and explicit slot-wise native-vs-guided qLogEI
      noninferiority. One slot per adaptive batch remains protected native in
      all heuristic lineages.

SYSTEMATIC_A heuristic geometry is deliberately distinct:
GHOST: preference-specific OOF under-prediction blind spots with local-density-
       adaptive neighborhoods.
SPARC: preference-specific recurrent inferior->elite displacement subspace with
       empirical step-length scale.
TRAIL: preference-specific aligned improving chains with curvature-, segment-,
       and boundary-aware continuation length.
"""

import math
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from generic_core import fit_gsmobo_gp_generic, qparego_scalarization_components_generic
from runtime import sobol_points, key_for_x

BASE_NO50K='BASE_GSMOBO_QPAREGO_NO50K'
BASE_50K='BASE_GSMOBO_QPAREGO_50K'
GHOST_VB='DEV_GSMOBO50K_GHOST_LEGACY_VB'
SPARC_A='DEV_GSMOBO50K_SPARC_A'
TRAIL_V111='DEV_GSMOBO50K_TRAIL_TRUE_NATIVE_V111'
METHODS=[BASE_NO50K,BASE_50K,GHOST_VB,SPARC_A,TRAIL_V111]
HEURISTIC={GHOST_VB:'ghost',SPARC_A:'sparc',TRAIL_V111:'trail'}
LINEAGE={GHOST_VB:'LEGACY_VB',SPARC_A:'SYSTEMATIC_A',TRAIL_V111:'TRUE_NATIVE_V111'}
GHOST=GHOST_VB
SPARC=SPARC_A
TRAIL=TRAIL_V111

def _bool(s):
    if pd.api.types.is_bool_dtype(s): return s.to_numpy(bool)
    return s.astype(str).str.lower().isin(['true','1','yes','optimal','feasible']).to_numpy(bool)


def _history_xy_feasible(ctx,method):
    """Feasible live exact rows for structural mining/anchors only."""
    h=ctx.history(method)
    if not len(h): return np.empty((0,len(ctx.features))),np.empty((0,ctx.objective_dim))
    xc=[f'x01__{f}' for f in ctx.features]
    ok=_bool(h['exact_feasible'])
    X=h.loc[ok,xc].to_numpy(float); Y=h.loc[ok,ctx.objective_norm_columns].to_numpy(float)
    good=np.isfinite(X).all(axis=1)&np.isfinite(Y).all(axis=1)
    return X[good],Y[good]


def _history_training_xy(ctx,method):
    """Native gsMOBO live training convention: unsuccessful rows get objective 0.

    This is intentionally different from structural mining. It prevents sparse-
    feasibility problems from disabling BO and matches the historical gsMOBO
    comparator convention used in GEM: all exact calls teach the GP, while failed
    / constrained-infeasible outcomes are represented as the reference-zero
    outcome vector.
    """
    h=ctx.history(method)
    if not len(h): return np.empty((0,len(ctx.features))),np.empty((0,ctx.objective_dim))
    xc=[f'x01__{f}' for f in ctx.features]
    X=h[xc].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    Y=h[ctx.objective_norm_columns].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    F=_bool(h['exact_feasible'])
    bad=(~F)|(~np.isfinite(Y).all(axis=1))
    Y=Y.copy(); Y[bad]=0.0
    good=np.isfinite(X).all(axis=1)&np.isfinite(Y).all(axis=1)
    return X[good],Y[good]


def zero_outcomes(Y,F):
    """Retain failed/infeasible exact rows as the normalized zero outcome vector."""
    Y=np.asarray(Y,float); F=np.asarray(F,bool).reshape(-1)
    out=np.zeros_like(Y,float); ok=F & np.isfinite(Y).all(axis=1)
    out[ok]=Y[ok]
    return out


def _pareto_representative_ids(Y,good_ids,cap):
    if cap<=0 or len(good_ids)==0:return []
    from generic_core import pareto_points_generic
    Yg=np.asarray(Y,float)[np.asarray(good_ids,int)]
    P=pareto_points_generic(Yg)
    if len(P)>cap:P=P[np.linspace(0,len(P)-1,cap,dtype=int)]
    out=[]
    for q in P:
        j=int(np.argmin(np.sum((Yg-q[None,:])**2,axis=1)));out.append(int(good_ids[j]))
    return list(dict.fromkeys(out))[:cap]


def representative_prior_ids(X,Y,F,n,seed):
    """Deterministic 50K compression: objective extremes + Pareto + Sobol coverage.

    Coverage is drawn from *all* frozen rows so infeasible/bad regions remain in
    the GP prior as zero outcomes.  This is objective-count and species agnostic.
    """
    X=np.asarray(X,float);Y=np.asarray(Y,float);F=np.asarray(F,bool);n=min(int(n),len(X))
    good=np.where(F & np.isfinite(Y).all(axis=1))[0];mandatory=[]
    if len(good):
        per_obj=max(8,min(32,max(8,n//max(8,4*Y.shape[1]))))
        for j in range(Y.shape[1]):
            order=good[np.argsort(Y[good,j],kind='mergesort')];mandatory.extend(map(int,order[-min(per_obj,len(order)):]))
        mandatory.extend(_pareto_representative_ids(Y,good,min(256,max(32,n//4))))
    mandatory=list(dict.fromkeys(mandatory))[:min(n,384)];used=set(mandatory);need=n-len(mandatory)
    if need>0:
        tree=cKDTree(X);targets=sobol_points(max(512,need*6),X.shape[1],int(seed)+17);k=min(32,len(X));nbr=np.asarray(tree.query(targets,k=k)[1],int)
        if nbr.ndim==1:nbr=nbr[:,None]
        for row in nbr:
            for jj in np.atleast_1d(row):
                j=int(jj)
                if j not in used:
                    used.add(j);mandatory.append(j);break
            if len(mandatory)>=n:break
    if len(mandatory)<n:
        for j in range(len(X)):
            if j not in used:used.add(j);mandatory.append(j)
            if len(mandatory)>=n:break
    ids=np.asarray(mandatory[:n],int)
    if len(ids)!=n or len(np.unique(ids))!=n:raise RuntimeError(f'representative prior failed: {len(ids)} rows, unique={len(np.unique(ids))}, target={n}')
    return ids


def _dedup_prior_live(Xp,Yp,Xh,Yh,decimals=12):
    """Combine prior + live history with live exact values winning collisions."""
    Xp=np.asarray(Xp,float);Yp=np.asarray(Yp,float);Xh=np.asarray(Xh,float);Yh=np.asarray(Yh,float)
    if not len(Xp):return Xh.copy(),Yh.copy()
    if not len(Xh):return Xp.copy(),Yp.copy()
    X=np.vstack([Xp,Xh]);Y=np.vstack([Yp,Yh]);last={}
    for i,x in enumerate(X):last[key_for_x(x,decimals)]=i
    keep=np.asarray(sorted(last.values()),int);return X[keep],Y[keep]


def _training(ctx,method,use_archive):
    Xh,Yh=_history_training_xy(ctx,method)
    if use_archive:
        ids=np.asarray(ctx.subset_local,int);ids=ids[(ids>=0)&(ids<len(ctx.X0))]
        Xa=ctx.X0[ids];Ya=zero_outcomes(ctx.Y0[ids],ctx.F0[ids])
        X,Y=_dedup_prior_live(Xa,Ya,Xh,Yh,int(ctx.cfg['study'].get('exact_cache_round_decimals',12)))
    else:
        X,Y=Xh,Yh
    return np.asarray(X,float),np.asarray(Y,float)


def _shared_init(ctx,method,q):
    n=len(ctx.history(method)); b=n//ctx.batch_size+1; seed=ctx.seed+310001+1009*b; X=ctx._filter_new(sobol_points(max(256,16*q),len(ctx.features),seed),method)
    if len(X)<q:
        X=np.vstack([X,ctx._filter_new(sobol_points(max(512,32*q),len(ctx.features),seed+17),method)])
    if len(X)<q: raise RuntimeError(f'{method}: common initialization could only produce {len(X)}/{q} novel points')
    return X[:q],[f'COMMON_SOBOL_INIT::{b}']*q,{'family_stage':'common_initialization'}


def _simplex_weight(m,seed):
    if int(m)==1:return np.ones(1,float)
    import torch
    from botorch.utils.sampling import sample_simplex
    try:
        w=sample_simplex(d=int(m),n=1,qmc=False,seed=int(seed),device=torch.device('cpu'),dtype=torch.double).squeeze(0)
    except TypeError:
        torch.manual_seed(int(seed));w=sample_simplex(int(m),device=torch.device('cpu'),dtype=torch.double).squeeze()
    return np.asarray(w.detach().cpu().numpy(),float).reshape(-1)


def _make_acq(model,X,Y,w,seed,objective_contracts=None,features=None):
    """Build one native qParEGO qLogEI acquisition under the generic objective contract.

    Only stochastic/exact objective channels are GP-modelled.  Any objective
    declared deterministic_from_design (for example PCO carbon/oxygen efficiency
    or normalized resource cost) is inserted exactly from candidate X into the
    same scalarization.  This avoids fitting a surrogate to a known deterministic
    channel and preserves the problem semantics across objective dimensions.
    """
    import torch
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler
    obj,best=qparego_scalarization_components_generic(
        model,np.asarray(X,float),np.asarray(Y,float),np.asarray(w,float),
        objective_contracts=objective_contracts,features=features,
    )
    sampler=SobolQMCNormalSampler(torch.Size([int(256)]),seed=int(seed)+7001)
    return qLogExpectedImprovement(model=model,best_f=best,sampler=sampler,objective=obj)


def _acq_vals(acq,X):
    import torch
    X=np.atleast_2d(np.asarray(X,float)); out=[]
    if not len(X): return np.empty(0)
    with torch.no_grad():
        for s in range(0,len(X),128): out.append(np.asarray(acq(torch.tensor(X[s:s+128],dtype=torch.double).unsqueeze(1)).detach().cpu().numpy(),float).reshape(-1))
    return np.concatenate(out)


def _optimize_one(acq,d,num_restarts,raw_samples,maxiter,starts=None):
    import torch
    from botorch.optim.optimize import optimize_acqf
    bounds=torch.stack([torch.zeros(int(d),dtype=torch.double),torch.ones(int(d),dtype=torch.double)])
    kw=dict(acq_function=acq,bounds=bounds,q=1,num_restarts=int(num_restarts),options={'batch_limit':min(10,int(num_restarts)),'maxiter':int(maxiter)})
    if starts is None:
        kw['raw_samples']=int(raw_samples)
    else:
        starts=np.atleast_2d(np.asarray(starts,float)); kw['raw_samples']=None; kw['num_restarts']=len(starts); kw['batch_initial_conditions']=torch.tensor(starts[:,None,:],dtype=torch.double); kw['options']={'batch_limit':min(10,len(starts)),'maxiter':int(maxiter)}
    cand,val=optimize_acqf(**kw); x=np.asarray(cand.detach().cpu().numpy(),float).reshape(-1,int(d))[0]; v=float(np.asarray(val.detach().cpu().numpy(),float).reshape(-1)[0]) if val is not None else float(_acq_vals(acq,[x])[0]); return np.clip(x,0,1),v


def _qparego_state(ctx,method,use_archive,q,seed):
    """Fit one batch-level native qParEGO state or return a declared Sobol state."""
    X,Y=_training(ctx,method,use_archive)
    live_h=ctx.history(method)
    live_feas=int(_bool(live_h['exact_feasible']).sum()) if len(live_h) and 'exact_feasible' in live_h.columns else 0
    min_live=int(ctx.cfg['gsmobo_dev'].get('min_feasible_live_for_no50k_gp',2))
    if len(X)<2 or ((not use_archive) and live_feas<min_live):
        pool=ctx._filter_new(sobol_points(max(512,64*q),len(ctx.features),int(seed)+701),method)
        if len(pool)<q: raise RuntimeError(f'{method}: insufficient novel feasibility-Sobol points {len(pool)}/{q}')
        return {'kind':'fallback','X':pool[:q],'modes':['LIVE_FEASIBILITY_SOBOL']*q,'slots':[],
                'diag':{'parent_kind':'feasibility_sobol','training_rows':len(X),'live_feasible_rows':live_feas,'gp_fit_fallback':False}}
    try:
        model=fit_gsmobo_gp_generic(X,Y,ctx.compiled_problem['objectives'],ctx.features)
    except Exception as exc:
        pool=ctx._filter_new(sobol_points(max(1024,128*q),len(ctx.features),int(seed)+1701),method)
        if len(pool)<q: raise RuntimeError(f'{method}: GP-fit fallback could only produce {len(pool)}/{q} novel points') from exc
        return {'kind':'fallback','X':pool[:q],'modes':['GP_FIT_NUMERICAL_FALLBACK_SOBOL']*q,'slots':[],
                'diag':{'parent_kind':'gp_fit_fallback','training_rows':len(X),'live_feasible_rows':live_feas,'gp_fit_fallback':True,'gp_fit_error':f'{type(exc).__name__}: {exc}'}}
    return {'kind':'qparego','model':model,'Xtrain':X,'Ytrain':Y,
            'diag':{'parent_kind':'qparego','training_rows':len(X),'live_feasible_rows':live_feas,'gp_fit_fallback':False}}


def _native_qparego_slot(ctx,method,state,seed,slot,selected):
    """Optimize one native slot under the *actual* already-selected batch points.

    This is the core parent-protection contract.  For heuristic methods the
    previous points may themselves be guided, so a stale parent batch generated
    under different pending points would not be a valid comparator.  Native and
    guided candidates for a slot therefore share exactly the same acquisition
    object and X_pending state.
    """
    cfg=ctx.cfg['gsmobo_dev'];d=len(ctx.features);X=state['Xtrain'];Y=state['Ytrain'];model=state['model']
    w=_simplex_weight(ctx.objective_dim,int(seed)+101*int(slot));acq=None;x=None;s=np.nan;pm='QPAREGO_NATIVE';err=''
    try:
        acq=_make_acq(model,X,Y,w,int(seed)+101*int(slot),ctx.compiled_problem['objectives'],ctx.features)
        if selected and hasattr(acq,'set_X_pending'):
            import torch;acq.set_X_pending(torch.tensor(np.asarray(selected,float),dtype=torch.double))
        x,s=_optimize_one(acq,d,cfg['num_restarts'],cfg['raw_samples'],cfg['maxiter'])
    except Exception as exc:
        err=f'{type(exc).__name__}: {exc}';pm='NUMERICAL_FALLBACK_SOBOL'
        fallback=ctx._filter_new(sobol_points(max(1024,64*4),d,int(seed)+8801+int(slot)),method)
        seen={key_for_x(a) for a in selected};fallback=np.asarray([z for z in fallback if key_for_x(z) not in seen],float) if len(fallback) else fallback
        if not len(fallback): raise RuntimeError(f'{method}: numerical fallback has no novel candidate') from exc
        if acq is not None:
            vals=_acq_vals(acq,fallback[:min(len(fallback),512)]);good=np.isfinite(vals)
            if good.any():
                loc=int(np.where(good)[0][int(np.argmax(vals[good]))]);x=np.asarray(fallback[loc],float);s=float(vals[loc])
            else:x=np.asarray(fallback[0],float)
        else:x=np.asarray(fallback[0],float)
    assert x is not None
    xnov=ctx._filter_new([x],method);seen={key_for_x(z) for z in selected}
    if not len(xnov) or key_for_x(x) in seen:
        fallback=ctx._filter_new(sobol_points(2048,d,int(seed)+9801+int(slot)),method)
        fallback=np.asarray([z for z in fallback if key_for_x(z) not in seen],float) if len(fallback) else fallback
        if not len(fallback): raise RuntimeError(f'{method}: qParEGO duplicate recovery failed')
        if acq is not None:
            vals=_acq_vals(acq,fallback[:min(len(fallback),512)]);good=np.isfinite(vals)
            loc=int(np.where(good)[0][int(np.argmax(vals[good]))]) if good.any() else 0
            x=np.asarray(fallback[loc],float);s=float(vals[loc]) if good.any() else np.nan
        else:x=np.asarray(fallback[0],float)
        pm='NUMERICAL_FALLBACK_SOBOL';err=(err+'; ' if err else '')+'duplicate recovery'
    pinf=np.nan
    try:
        _,pp=ctx.global_model.predict(np.asarray([x],float));pinf=float(np.asarray(pp).reshape(-1)[0])
    except Exception:pass
    return {'weight':w,'acq':acq,'parent_x':np.asarray(x,float),'parent_score':s,'parent_mode':pm,'parent_pinf':pinf,'parent_error':err}


def _qparego_parent(ctx,method,use_archive,q,seed):
    """Native qParEGO batch, sequentially conditioned on its own chosen points."""
    state=_qparego_state(ctx,method,use_archive,q,seed)
    if state['kind']=='fallback':return state['X'],state['modes'],state['slots'],state['diag']
    selected=[];modes=[];slots=[]
    for i in range(int(q)):
        sl=_native_qparego_slot(ctx,method,state,seed,i,selected)
        selected.append(sl['parent_x']);modes.append(sl['parent_mode']);slots.append(sl)
    return np.asarray(selected,float),modes,slots,state['diag']


def _qparego_guided_A(ctx,method,mode,q,seed):
    """Guided batch with slot-wise native parent regenerated under actual pending points."""
    state=_qparego_state(ctx,method,True,q,seed)
    if state['kind']=='fallback':
        X=state['X'];bm=state['modes']
        md={'family':'gsmobo','heuristic':mode,'implementation_lineage':'SYSTEMATIC_A','algorithm_stage':'adaptive','guided':np.zeros(len(X),int),'parent_mode':bm,
            'parent_score':[np.nan]*len(X),'custom_score':[np.nan]*len(X),'selected_acq_score':[np.nan]*len(X),
            'guided_minus_parent_acq':[np.nan]*len(X),'guided_minus_parent_gate':[np.nan]*len(X),'custom_pool_rows':[0]*len(X),'candidate_distance_from_parent':[0.0]*len(X),
            'guidance_reason':['PARENT_FALLBACK_PASSTHROUGH']*len(X),'admission_score_kind':['PARENT_FALLBACK']*len(X)}
        return X,[f'{mode.upper()}::PARENT_FALLBACK_PASSTHROUGH']*len(X),md,state['diag']
    rows=[];modes=[];guided=[];pscores=[];cscores=[];pools=[];dists=[];reasons=[];weights=[];parent_modes=[]
    protected=max(1,int(ctx.cfg['gsmobo_dev']['protected_parent_slots']))
    for i in range(int(q)):
        sl=_native_qparego_slot(ctx,method,state,seed,i,rows)
        weights.append(sl['weight']);parent_modes.append(sl['parent_mode'])
        if i<protected:
            x=sl['parent_x'];g=False;ps=sl['parent_score'];cs=np.nan;nr=0;dist=0.0;reason='PROTECTED_PARENT_SLOT'
        elif sl['parent_mode']!='QPAREGO_NATIVE' or sl['acq'] is None:
            x=sl['parent_x'];g=False;ps=sl['parent_score'];cs=np.nan;nr=0;dist=0.0;reason='PARENT_NUMERICAL_FALLBACK_PASSTHROUGH'
        else:
            x,g,ps,cs,nr,dist,reason=_structural_refine(ctx,method,mode,i,sl,seed+6001,rows)
        if key_for_x(x) in {key_for_x(z) for z in rows}:
            # _native_qparego_slot guarantees a novel parent under this pending
            # set, so this recovery is exact and deterministic.
            x=sl['parent_x'];g=False;reason='DUPLICATE_REVERT_PARENT';dist=0.0
        rows.append(np.asarray(x,float));modes.append(f'{mode.upper()}::{reason}');guided.append(int(g));pscores.append(ps);cscores.append(cs);pools.append(nr);dists.append(dist);reasons.append(reason)
    md={'family':'gsmobo','heuristic':mode,'implementation_lineage':'SYSTEMATIC_A','algorithm_stage':'adaptive','guided':guided,'parent_mode':parent_modes,'parent_score':pscores,'custom_score':cscores,
        'selected_acq_score':[cscores[i] if guided[i] else pscores[i] for i in range(len(rows))],
        'guided_minus_parent_acq':[cscores[i]-pscores[i] if np.isfinite(cscores[i]) and np.isfinite(pscores[i]) else np.nan for i in range(len(rows))],
        'guided_minus_parent_gate':[np.nan]*len(rows),'custom_pool_rows':pools,'candidate_distance_from_parent':dists,'guidance_reason':reasons,
        'admission_score_kind':['QLOGEI_NATIVE_NONINFERIOR_GATE']*len(rows)}
    for j,o in enumerate(ctx.compiled_problem['objectives']):md[f'qparego_weight__{o["slug"]}']=[float(weights[i][j]) for i in range(len(rows))]
    return np.asarray(rows,float),modes,md,state['diag']


def _sample_good(ctx,max_rows,seed):
    ids=np.where(ctx.good0 & np.isfinite(ctx.Y0).all(axis=1))[0]
    if len(ids)<=max_rows: return ids
    rng=np.random.default_rng(int(seed)); m=ctx.objective_dim; keep=set()
    for j in range(m):
        order=ids[np.argsort(ctx.Y0[ids,j],kind='mergesort')]
        for x in order[-min(64,len(order)):]: keep.add(int(x))
    rem=np.asarray([x for x in ids if int(x) not in keep],int); need=max(0,int(max_rows)-len(keep))
    chosen=list(sorted(keep));
    if need and len(rem): chosen.extend(rng.choice(rem,size=min(need,len(rem)),replace=False).tolist())
    return np.asarray(chosen[:max_rows],int)


def _u(Y,w):
    """Numpy augmented-Chebyshev preference utility matching qParEGO semantics."""
    Y=np.asarray(Y,float); w=np.asarray(w,float).reshape(-1)
    if Y.ndim!=2 or Y.shape[1]!=len(w): raise RuntimeError(f'preference utility shape mismatch Y={Y.shape}, w={w.shape}')
    wy=Y*w[None,:]
    return np.min(wy,axis=1)+0.05*np.sum(wy,axis=1)


def _robust01(v):
    v=np.asarray(v,float); ok=np.isfinite(v); out=np.zeros(len(v),float)
    if not ok.any(): return out
    lo,hi=np.quantile(v[ok],[0.05,0.95]); out[ok]=0.5 if hi<=lo+1e-15 else np.clip((v[ok]-lo)/(hi-lo),0,1); return out


def _global_guard(ctx,method,rows,seed):
    X=ctx._filter_new(sobol_points(max(int(rows)*2,int(rows)+64),len(ctx.features),int(seed)),method); return X[:int(rows)]




def _productive_displacements(X,Y,w,seed,max_pairs=400):
    U=_u(Y,w); tree=cKDTree(X); elite=np.argsort(U,kind='mergesort')[-min(256,len(X)):][::-1]; vec=[]; mag=[]; gain=[]; m=Y.shape[1]
    for e in elite:
        _,nbr=tree.query(X[e],k=min(48,len(X))); nbr=np.atleast_1d(nbr).astype(int); dy=Y[e][None,:]-Y[nbr]; du=U[e]-U[nbr]; ok=(du>1e-5)&(np.sum(dy>0,axis=1)>=max(1,math.ceil(m/2)))&(np.min(dy,axis=1)>-0.20)
        if not ok.any(): continue
        ids=nbr[ok]; dui=du[ok]; j=int(ids[int(np.argmax(dui))]); v=X[e]-X[j]; L=float(np.linalg.norm(v))
        if L<=1e-8: continue
        vec.append(v/L); mag.append(L); gain.append(float(np.max(dui)))
        if len(vec)>=max_pairs: break
    return np.asarray(vec,float),np.asarray(mag,float),np.asarray(gain,float),elite


def _sparc_pool(ctx,method,w,seed,max_rows):
    ids=_sample_good(ctx,int(ctx.cfg['gsmobo_dev']['structure_archive_rows']),seed+101); X=ctx.X0[ids]; Y=ctx.Y0[ids]; D,L,G,elite=_productive_displacements(X,Y,w,seed)
    if len(D)<8: return _global_guard(ctx,method,max_rows,seed+119)
    A=D*np.sqrt(np.maximum(G,1e-8))[:,None]; _,s,vt=np.linalg.svd(A,full_matrices=False); energy=np.cumsum(s*s)/max(float(np.sum(s*s)),1e-12); rank=int(np.searchsorted(energy,0.90)+1); rank=max(1,min(rank,4,X.shape[1])); basis=vt[:rank].copy(); mean=np.average(D,axis=0,weights=np.maximum(G,1e-8));
    for r in range(rank):
        if np.dot(basis[r],mean)<0: basis[r]*=-1
    U=_u(Y,w); anchors=X[np.argsort(U)[-min(96,len(X)):]]; Xh,Yh=_history_xy_feasible(ctx,method)
    if len(Xh):
        uh=_u(Yh,w); anchors=np.vstack([anchors,Xh[np.argsort(uh)[-min(16,len(Xh)):]]])
    qs=np.quantile(L,[0.30,0.55,0.80]); rng=np.random.default_rng(seed+109); rows=[]; target=max(32,int(max_rows*0.85))
    while len(rows)<target:
        a=anchors[int(rng.integers(0,len(anchors)))]; coef=rng.normal(0,0.35,size=rank); coef[0]=abs(coef[0])+0.65; direction=coef@basis; direction/=max(np.linalg.norm(direction),1e-12); step=float(qs[int(rng.integers(0,3))]); rows.append(np.clip(a+step*direction,0,1))
    return ctx._filter_new(np.vstack([rows,_global_guard(ctx,method,max_rows-len(rows),seed+127)]),method)[:max_rows]





def _legacy_ghost_pool(ctx,method,w,seed,max_rows):
    """Generalized port of the older GHOST restart geometry.

    The historical implementation emphasized valuable surrogate blind spots but
    used simpler fixed-scale neighborhoods than SYSTEMATIC_A.  This port keeps
    that distinction while remaining dimension/problem generic.
    """
    ids=_sample_good(ctx,int(ctx.cfg['gsmobo_dev']['structure_archive_rows']),seed+301)
    X=ctx.X0[ids];Y=ctx.Y0[ids]
    oof=np.asarray(getattr(ctx,'oof_all',np.empty((0,ctx.objective_dim))),float)
    if oof.shape!=ctx.Y0.shape: raise RuntimeError('legacy GHOST requires complete all-row OOF predictions')
    P=oof[ids];ok=np.isfinite(Y).all(axis=1)&np.isfinite(P).all(axis=1);X,Y,P=X[ok],Y[ok],P[ok]
    if not len(X): return _global_guard(ctx,method,max_rows,seed+307)
    ue=_u(Y,w);up=_u(P,w);blind=np.maximum(ue-up,0.0)
    score=0.72*_robust01(blind)+0.28*_robust01(ue)
    ncent=min(96,len(X));centers=X[np.argsort(score,kind='mergesort')[-ncent:][::-1]]
    rng=np.random.default_rng(seed+313);rows=[];target=max(24,int(max_rows*0.80))
    dimscale=float(np.sqrt(5.0/max(1,len(ctx.features))))
    radii=np.asarray([0.025,0.055,0.11])*dimscale
    attempts=0;limit=max(256,16*target)
    while len(rows)<target and attempts<limit:
        attempts+=1;c=centers[int(rng.integers(0,len(centers)))];z=rng.normal(size=len(c));z/=max(np.linalg.norm(z),1e-12)
        r=float(radii[int(rng.integers(0,len(radii)))]);rows.append(np.clip(c+r*z,0,1))
    guard=max(0,int(max_rows)-len(rows));stack=np.asarray(rows,float) if rows else np.empty((0,len(ctx.features)))
    if guard: stack=np.vstack([stack,_global_guard(ctx,method,guard,seed+317)])
    return ctx._filter_new(stack,method)[:max_rows]




def _legacy_trail_pool(ctx,method,w,seed,max_rows):
    """Generalized port of the older pairwise TRAIL continuation geometry."""
    ids=_sample_good(ctx,int(ctx.cfg['gsmobo_dev']['structure_archive_rows']),seed+371);X=ctx.X0[ids];Y=ctx.Y0[ids]
    if len(X)<8:return _global_guard(ctx,method,max_rows,seed+373)
    U=_u(Y,w);tree=cKDTree(X);anchors=np.argsort(U)[-min(144,len(X)):][::-1];segments=[]
    for a in anchors:
        _,nbr=tree.query(X[a],k=min(48,len(X)));nbr=np.atleast_1d(nbr).astype(int);du=U[a]-U[nbr];good=du>1e-5
        if not good.any():continue
        cand=nbr[good];p=int(cand[int(np.argmax(du[good]))]);v=X[a]-X[p];L=float(np.linalg.norm(v))
        if L<=1e-8:continue
        d=v/L;head=np.inf
        for j,z in enumerate(d):
            if z>1e-12:head=min(head,(1-X[a,j])/z)
            elif z<-1e-12:head=min(head,(0-X[a,j])/z)
        if not np.isfinite(head):head=2*L
        segments.append((float(U[a]-U[p]),X[a],d,L,max(0.0,head)))
    if not segments:return _global_guard(ctx,method,max_rows,seed+379)
    segments.sort(key=lambda z:z[0],reverse=True);segments=segments[:96];rng=np.random.default_rng(seed+383);rows=[];target=max(24,int(max_rows*0.80));factors=[0.55,0.90,1.25]
    attempts=0;limit=max(256,16*target)
    while len(rows)<target and attempts<limit:
        attempts+=1;_,a,d,L,head=segments[int(rng.integers(0,len(segments)))];step=min(float(L*factors[int(rng.integers(0,3))]),float(0.85*head))
        if step<=1e-5:continue
        rows.append(np.clip(a+step*d,0,1))
    guard=max(0,int(max_rows)-len(rows));stack=np.asarray(rows,float) if rows else np.empty((0,len(ctx.features)))
    if guard:stack=np.vstack([stack,_global_guard(ctx,method,guard,seed+389)])
    return ctx._filter_new(stack,method)[:max_rows]


def _legacy_structural_pool(ctx,method,mode,w,seed,max_rows):
    if mode=='ghost':return _legacy_ghost_pool(ctx,method,w,seed,max_rows)
    if mode=='trail':return _legacy_trail_pool(ctx,method,w,seed,max_rows)
    raise ValueError(mode)

def _structural_pool(ctx,method,mode,w,seed,max_rows):
    if mode=='sparc': return _sparc_pool(ctx,method,w,seed,max_rows)
    raise ValueError(mode)


def _guided_acq_accept(parent_score,guided_score,distance,tolerance,min_distance):
    """Pure native-authority admission gate used by every heuristic slot."""
    ps=float(parent_score); cs=float(guided_score); dist=float(distance)
    tol=float(tolerance)*(1.0+abs(ps) if np.isfinite(ps) else 1.0)
    distinct=np.isfinite(dist) and dist>float(min_distance)
    better=np.isfinite(cs) and (not np.isfinite(ps) or cs>ps+tol)
    noninferior=np.isfinite(cs) and np.isfinite(ps) and cs>=ps-tol
    accepted=bool(distinct and (better or noninferior))
    reason='GUIDED_BETTER_ACQ' if accepted and better else ('GUIDED_ACQ_NONINFERIOR' if accepted else ('GUIDED_SAME_AS_NATIVE' if not distinct else 'GUIDED_REJECTED_ACQ'))
    return accepted,reason,tol


def _structural_refine(ctx,method,mode,slot,slotinfo,seed,selected):
    acq=slotinfo['acq']; parent=np.asarray(slotinfo['parent_x'],float); ps=float(slotinfo['parent_score']); pm=str(slotinfo['parent_mode'])
    if acq is None or 'FEASIBILITY_EXTENSION' in pm or 'FALLBACK_SOBOL' in pm: return parent,False,ps,np.nan,0,np.nan,'PARENT_FALLBACK_PASSTHROUGH'
    max_rows=int(ctx.cfg['gsmobo_dev']['structure_pool_rows']); pool=_structural_pool(ctx,method,mode,slotinfo['weight'],seed+31*slot,max_rows)
    if selected:
        seen={key_for_x(z) for z in selected}; pool=np.asarray([x for x in pool if key_for_x(x) not in seen],float) if len(pool) else pool
    if not len(pool): return parent,False,ps,np.nan,0,np.nan,'NO_STRUCTURAL_POOL'
    # Do not impose a second hard feasibility gate on top of native qParEGO.
    # The parent GP is trained with infeasible exact calls mapped to reference-zero
    # objectives, so the acquisition itself remains the admission authority.  The
    # frozen classifier is retained only as a diagnostic/tie-break signal.
    try:
        _,pinf=ctx.global_model.predict(pool); pinf=np.asarray(pinf,float).reshape(-1)
    except Exception:
        pinf=np.full(len(pool),np.nan,float)
    P=pool; vals=_acq_vals(acq,P); ok=np.isfinite(vals)
    if not ok.any(): return parent,False,ps,np.nan,len(pool),np.nan,'NONFINITE_STRUCTURAL_ACQ'
    v=vals[ok]; P=P[ok]; pp=pinf[ok] if len(pinf)==len(pool) else np.full(len(P),np.nan)
    # Primary order is acquisition value. Predicted infeasibility only breaks
    # numerically/acquisition-equivalent ties and can never veto a candidate.
    scale=max(float(np.nanquantile(v,0.90)-np.nanquantile(v,0.10)),1e-10)
    tie=np.nan_to_num(pp,nan=1.0,posinf=1.0,neginf=0.0)
    rank_score=v-1e-6*scale*tie
    order=np.argsort(rank_score,kind='mergesort')[::-1]; k=min(int(ctx.cfg['gsmobo_dev']['structure_top_restarts']),len(order)); starts=P[order[:k]]
    if selected and hasattr(acq,'set_X_pending'):
        import torch; acq.set_X_pending(torch.tensor(np.asarray(selected,float),dtype=torch.double))
    try: cand,cs=_optimize_one(acq,len(ctx.features),k,0,ctx.cfg['gsmobo_dev']['structure_refine_maxiter'],starts=starts)
    except Exception: cand=starts[0]; cs=float(_acq_vals(acq,[cand])[0])
    nov=ctx._filter_new([cand],method)
    if not len(nov) or (selected and key_for_x(cand) in {key_for_x(z) for z in selected}): return parent,False,ps,cs,len(pool),np.nan,'STRUCTURAL_DUPLICATE'
    # Strict native-authority gate: guidance is advisory only.  It may replace the
    # independently optimized native point only when the *same* qLogEI acquisition
    # judges it noninferior within a small numerical tolerance.  No percentage
    # allowance is used, so a problem-specific acquisition scale cannot weaken the
    # parent-protection contract.
    dist=float(np.linalg.norm(cand-parent))
    accept,reason,_=_guided_acq_accept(ps,cs,dist,ctx.cfg['gsmobo_dev'].get('guided_acq_noninferiority_tolerance',1e-6),ctx.cfg['study'].get('min_distance_within_method',1e-8))
    if accept: return cand,True,ps,cs,len(pool),dist,reason
    return parent,False,ps,cs,len(pool),dist,reason



def _fallback_guided_metadata(mode,lineage,X,bm,reason):
    return {
        'family':'gsmobo','heuristic':mode,'implementation_lineage':lineage,
        'algorithm_stage':'adaptive','guided':np.zeros(len(X),int),'parent_mode':bm,
        'parent_score':[np.nan]*len(X),'custom_score':[np.nan]*len(X),
        'selected_acq_score':[np.nan]*len(X),'guided_minus_parent_acq':[np.nan]*len(X),
        'guided_minus_parent_gate':[np.nan]*len(X),'custom_pool_rows':[0]*len(X),
        'candidate_distance_from_parent':[0.0]*len(X),'guidance_reason':[reason]*len(X),
        'admission_score_kind':['PARENT_FALLBACK']*len(X),
    }


def _qparego_guided_vb(ctx,method,mode,q,seed):
    """Generalized historical Variant-B integration.

    Contract ported from the validated PCO lineage: obtain the native +50K
    adaptive proposal first, protect the first slot, then allow a problem-aware
    global-surrogate gate to replace non-protected slots with heuristic
    candidates.  The gate is intentionally *not* qLogEI; that is the key
    historical distinction this systematic screen is meant to measure.
    """
    base,bm,slots,diag=_qparego_parent(ctx,method,True,q,seed)
    if not slots:
        md=_fallback_guided_metadata(mode,'LEGACY_VB',base,bm,'PARENT_FALLBACK_PASSTHROUGH')
        return base,[f'{mode.upper()}::VB_PARENT_FALLBACK']*len(base),md,diag
    live_h=ctx.history(method);live_feas=int(_bool(live_h['exact_feasible']).sum()) if len(live_h) and 'exact_feasible' in live_h.columns else 0
    min_live=int(ctx.cfg['gsmobo_dev'].get('legacy_vb_min_live_feasible',2))
    protected=max(1,int(ctx.cfg['gsmobo_dev'].get('protected_parent_slots',1)))
    penalty=float(ctx.cfg['gsmobo_dev'].get('legacy_vb_infeasibility_penalty',0.35))
    gate_tol=float(ctx.cfg['gsmobo_dev'].get('legacy_vb_gate_tolerance',1e-8))
    max_rows=int(ctx.cfg['gsmobo_dev'].get('legacy_structure_pool_rows',256))
    rows=[];modes=[];guided=[];pscores=[];cscores=[];pools=[];dists=[];reasons=[];gate_deltas=[];weights=[];parent_modes=[]
    for i in range(int(q)):
        parent=np.asarray(base[i],float);sl=slots[i];w=np.asarray(sl['weight'],float);weights.append(w);parent_modes.append(sl['parent_mode'])
        x=parent;g=False;cs=np.nan;nr=0;dist=0.0;gd=np.nan;reason='PROTECTED_PARENT_SLOT' if i<protected else 'VB_GATE_NOT_USED'
        if i>=protected and live_feas>=min_live and sl['parent_mode']=='QPAREGO_NATIVE':
            pool=_legacy_structural_pool(ctx,method,mode,w,seed+401+37*i,max_rows)
            if rows and len(pool):
                seen={key_for_x(z) for z in rows};pool=np.asarray([z for z in pool if key_for_x(z) not in seen],float)
            nr=len(pool)
            if len(pool):
                pred,pinf=ctx.global_model.predict(pool);pred=np.asarray(pred,float);pinf=np.asarray(pinf,float).reshape(-1)
                gate=_u(pred,w)-penalty*np.nan_to_num(pinf,nan=1.0,posinf=1.0,neginf=0.0)
                pp,pi=ctx.global_model.predict(parent[None,:]);parent_gate=float(_u(np.asarray(pp,float),w)[0]-penalty*float(np.nan_to_num(np.asarray(pi,float).reshape(-1)[0],nan=1.0,posinf=1.0,neginf=0.0)))
                good=np.isfinite(gate)
                if good.any():
                    valid_idx=np.where(good)[0];jj=int(valid_idx[int(np.argmax(gate[good]))]);cand=np.asarray(pool[jj],float);cg=float(gate[jj]);gd=cg-parent_gate;dist=float(np.linalg.norm(cand-parent));cs=cg
                    distinct=dist>float(ctx.cfg['study'].get('min_distance_within_method',1e-8))
                    if distinct and cg>parent_gate+gate_tol:
                        x=cand;g=True;reason='VB_GLOBAL_SURROGATE_GATE_ACCEPT'
                    else: reason='VB_GLOBAL_SURROGATE_GATE_REJECT'
                else: reason='VB_NONFINITE_GATE'
            else: reason='VB_NO_STRUCTURAL_POOL'
        elif i>=protected and live_feas<min_live:
            reason='VB_SPARSE_LIVE_FEASIBILITY_PARENT'
        if key_for_x(x) in {key_for_x(z) for z in rows}:
            x=parent;g=False;reason='VB_DUPLICATE_REVERT_PARENT';dist=0.0
        rows.append(np.asarray(x,float));modes.append(f'{mode.upper()}::LEGACY_VB::{reason}');guided.append(int(g));pscores.append(float(sl['parent_score']));cscores.append(cs);pools.append(int(nr));dists.append(float(dist));reasons.append(reason);gate_deltas.append(gd)
    md={'family':'gsmobo','heuristic':mode,'implementation_lineage':'LEGACY_VB','algorithm_stage':'adaptive','guided':guided,'parent_mode':parent_modes,
        'parent_score':pscores,'custom_score':cscores,'selected_acq_score':[pscores[i] for i in range(len(rows))],
        'guided_minus_parent_acq':[np.nan]*len(rows),'guided_minus_parent_gate':gate_deltas,'custom_pool_rows':pools,
        'candidate_distance_from_parent':dists,'guidance_reason':reasons,'admission_score_kind':['GLOBAL_SURROGATE_GATE']*len(rows)}
    for j,o in enumerate(ctx.compiled_problem['objectives']):md[f'qparego_weight__{o["slug"]}']=[float(weights[i][j]) for i in range(len(rows))]
    return np.asarray(rows,float),modes,md,diag


def _qparego_guided_v111(ctx,method,mode,q,seed):
    """Generalized v1.1.1 true-native restart integration.

    Native qParEGO/qLogEI is preserved, but non-protected slots may be replaced
    by a qLogEI optimization launched from legacy heuristic restart structure.
    Unlike SYSTEMATIC_A, this lineage intentionally has no explicit
    native-vs-guided acquisition-noninferiority gate; this difference is part of
    the experiment rather than an accidental code-path difference.
    """
    state=_qparego_state(ctx,method,True,q,seed)
    if state['kind']=='fallback':
        X=state['X'];bm=state['modes'];md=_fallback_guided_metadata(mode,'TRUE_NATIVE_V111',X,bm,'PARENT_FALLBACK_PASSTHROUGH')
        return X,[f'{mode.upper()}::V111_PARENT_FALLBACK']*len(X),md,state['diag']
    protected=max(1,int(ctx.cfg['gsmobo_dev'].get('protected_parent_slots',1)));max_rows=int(ctx.cfg['gsmobo_dev'].get('legacy_structure_pool_rows',256))
    rows=[];modes=[];guided=[];pscores=[];cscores=[];pools=[];dists=[];reasons=[];weights=[];parent_modes=[]
    for i in range(int(q)):
        sl=_native_qparego_slot(ctx,method,state,seed,i,rows);weights.append(sl['weight']);parent_modes.append(sl['parent_mode'])
        parent=np.asarray(sl['parent_x'],float);x=parent;g=False;cs=np.nan;nr=0;dist=0.0;reason='PROTECTED_PARENT_SLOT' if i<protected else 'V111_NO_GUIDANCE'
        if i>=protected and sl['parent_mode']=='QPAREGO_NATIVE' and sl['acq'] is not None:
            pool=_legacy_structural_pool(ctx,method,mode,sl['weight'],seed+503+41*i,max_rows)
            if rows and len(pool):
                seen={key_for_x(z) for z in rows};pool=np.asarray([z for z in pool if key_for_x(z) not in seen],float)
            nr=len(pool)
            if len(pool):
                vals=_acq_vals(sl['acq'],pool);good=np.isfinite(vals)
                if good.any():
                    P=pool[good];vv=vals[good];order=np.argsort(vv,kind='mergesort')[::-1];k=min(int(ctx.cfg['gsmobo_dev']['structure_top_restarts']),len(order));starts=P[order[:k]]
                    try:cand,cs=_optimize_one(sl['acq'],len(ctx.features),k,0,ctx.cfg['gsmobo_dev']['structure_refine_maxiter'],starts=starts)
                    except Exception:
                        cand=np.asarray(starts[0],float);cs=float(_acq_vals(sl['acq'],[cand])[0])
                    nov=ctx._filter_new([cand],method);dist=float(np.linalg.norm(cand-parent))
                    if len(nov) and key_for_x(cand) not in {key_for_x(z) for z in rows} and dist>float(ctx.cfg['study'].get('min_distance_within_method',1e-8)) and np.isfinite(cs):
                        x=np.asarray(cand,float);g=True;reason='V111_GUIDED_RESTART_ACCEPT'
                    else:reason='V111_GUIDED_DUPLICATE_OR_INVALID'
                else:reason='V111_NONFINITE_STRUCTURAL_ACQ'
            else:reason='V111_NO_STRUCTURAL_POOL'
        elif i>=protected:reason='V111_PARENT_NUMERICAL_FALLBACK_PASSTHROUGH'
        if key_for_x(x) in {key_for_x(z) for z in rows}:
            x=parent;g=False;reason='V111_DUPLICATE_REVERT_PARENT';dist=0.0
        rows.append(np.asarray(x,float));modes.append(f'{mode.upper()}::TRUE_NATIVE_V111::{reason}');guided.append(int(g));pscores.append(float(sl['parent_score']));cscores.append(float(cs) if np.isfinite(cs) else np.nan);pools.append(int(nr));dists.append(float(dist));reasons.append(reason)
    md={'family':'gsmobo','heuristic':mode,'implementation_lineage':'TRUE_NATIVE_V111','algorithm_stage':'adaptive','guided':guided,'parent_mode':parent_modes,
        'parent_score':pscores,'custom_score':cscores,'selected_acq_score':[cscores[i] if guided[i] else pscores[i] for i in range(len(rows))],
        'guided_minus_parent_acq':[cscores[i]-pscores[i] if np.isfinite(cscores[i]) and np.isfinite(pscores[i]) else np.nan for i in range(len(rows))],
        'guided_minus_parent_gate':[np.nan]*len(rows),'custom_pool_rows':pools,'candidate_distance_from_parent':dists,'guidance_reason':reasons,
        'admission_score_kind':['QLOGEI_RESTART_NO_PARENT_GATE']*len(rows)}
    for j,o in enumerate(ctx.compiled_problem['objectives']):md[f'qparego_weight__{o["slug"]}']=[float(weights[i][j]) for i in range(len(rows))]
    return np.asarray(rows,float),modes,md,state['diag']

def run_method(ctx,method):
    use_archive=method!=BASE_NO50K
    mode=HEURISTIC.get(method)
    lineage=LINEAGE.get(method,'BASELINE')
    initial=int(ctx.cfg['gsmobo_dev']['initial_calls'])
    budget=int(ctx.budget)
    while len(ctx.history(method))<budget:
        n=len(ctx.history(method));q=min(ctx.batch_size,budget-n)
        if n<initial:
            X,modes,meta=_shared_init(ctx,method,q)
            metadata={'family':'gsmobo','heuristic':mode or ('50k' if use_archive else 'no50k'),'implementation_lineage':lineage,
                      'algorithm_stage':'warmup','guided':np.zeros(q,int),'admission_score_kind':['COMMON_INIT']*q}
        else:
            b=n//ctx.batch_size+1;seed=ctx.seed+1009*b+510001
            if mode is not None:
                if lineage=='LEGACY_VB':
                    X,modes,metadata,diag=_qparego_guided_vb(ctx,method,mode,q,seed)
                elif lineage=='TRUE_NATIVE_V111':
                    X,modes,metadata,diag=_qparego_guided_v111(ctx,method,mode,q,seed)
                elif lineage=='SYSTEMATIC_A':
                    X,modes,metadata,diag=_qparego_guided_A(ctx,method,mode,q,seed)
                else:
                    raise RuntimeError(f'{method}: unknown implementation lineage {lineage}')
            else:
                base,bm,slots,diag=_qparego_parent(ctx,method,use_archive,q,seed);X=base;modes=[f'{method}::{x}' for x in bm]
                metadata={'family':'gsmobo','heuristic':('50k' if use_archive else 'no50k'),'implementation_lineage':'BASELINE',
                          'algorithm_stage':'adaptive','guided':np.zeros(q,int),'parent_mode':bm,
                          'parent_score':[np.nan if not slots else slots[i]['parent_score'] for i in range(q)],'custom_score':[np.nan]*q,
                          'selected_acq_score':[np.nan if not slots else slots[i]['parent_score'] for i in range(q)],'guided_minus_parent_acq':[np.nan]*q,
                          'guided_minus_parent_gate':[np.nan]*q,'custom_pool_rows':[0]*q,'candidate_distance_from_parent':[0.0]*q,
                          'guidance_reason':['BASELINE_OR_FALLBACK']*q,'admission_score_kind':['NATIVE_QPAREGO']*q}
                if slots:
                    for j,o in enumerate(ctx.compiled_problem['objectives']):metadata[f'qparego_weight__{o["slug"]}']=[float(slots[i]['weight'][j]) for i in range(q)]
            metadata['parent_training_rows']=int(diag.get('training_rows',0));metadata['parent_live_feasible_rows']=int(diag.get('live_feasible_rows',0));metadata['parent_kind']=str(diag.get('parent_kind',''))
        ctx.evaluate_batch(method,X,modes=modes,metadata=metadata)
        h=ctx.history(method);g=int(pd.to_numeric(h.tail(q).get('guided',pd.Series([0]*q)),errors='coerce').fillna(0).sum()) if len(h) else 0
        print(f'[{method}] {len(h):3d}/{budget} | exact ΔHV={ctx.current_delta_hv(method):.9g} | stage={"warmup" if n<initial else "adaptive"} | lineage={lineage} | guided={g}/{q}',flush=True)
    return ctx.history(method)

RUNNERS={m:(lambda ctx,m=m:run_method(ctx,m)) for m in METHODS}
