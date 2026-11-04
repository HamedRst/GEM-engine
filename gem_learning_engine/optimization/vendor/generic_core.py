from __future__ import annotations

"""Generic objective-space helpers installed into the inherited GEM core module."""

import numpy as np
from generic_surrogate import GenericGlobalSurrogate


def pareto_points_generic(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, float)
    if Y.ndim != 2 or Y.shape[1] < 1:
        raise RuntimeError(f"Pareto input must be (n,m>=1); got {Y.shape}")
    m = Y.shape[1]
    Y = Y[np.isfinite(Y).all(axis=1)]
    if len(Y) == 0:
        return np.empty((0, m), float)
    Y = np.unique(Y, axis=0)
    if m == 1:
        return np.asarray([[float(np.max(Y[:, 0]))]], float)
    if m == 2:
        order = np.lexsort((-Y[:, 1], -Y[:, 0]))
        s = Y[order]; keep=[]; best=-np.inf
        for x,y in s:
            if y > best + 1e-14:
                keep.append((x,y)); best=y
        return np.asarray(keep,float).reshape((-1,2))
    front=[]
    for y in Y:
        if front:
            F=np.asarray(front,float)
            dom=np.all(F >= y[None,:]-1e-14,axis=1) & np.any(F > y[None,:]+1e-14,axis=1)
            if np.any(dom):
                continue
            ydom=np.all(y[None,:] >= F-1e-14,axis=1) & np.any(y[None,:] > F+1e-14,axis=1)
            if np.any(ydom):
                front=[r.copy() for r in F[~ydom]]
        front.append(y.copy())
    return np.asarray(front,float).reshape((-1,m))


def hypervolume_exact(Y: np.ndarray, ref=None) -> float:
    Y=np.asarray(Y,float)
    if Y.ndim != 2 or Y.shape[1] < 1:
        raise RuntimeError(f"HV input must be (n,m>=1); got {Y.shape}")
    r=np.zeros(Y.shape[1],float) if ref is None else np.asarray(ref,float).reshape(-1)
    if len(r) != Y.shape[1]:
        raise RuntimeError(f"HV reference dimension {len(r)} != {Y.shape[1]}")
    Y=Y[np.isfinite(Y).all(axis=1)]
    Y=Y[np.all(Y > r[None,:],axis=1)]
    if len(Y)==0: return 0.0
    Y=pareto_points_generic(Y); m=Y.shape[1]
    if m==1: return float(max(0.0,np.max(Y[:,0])-r[0]))
    xs=np.unique(Y[:,0]); prev=float(r[0]); hv=0.0
    for x in xs:
        x=float(x)
        if x <= prev + 1e-15: continue
        active=Y[Y[:,0] >= x-1e-14,1:]
        if len(active): hv += (x-prev)*hypervolume_exact(active,r[1:])
        prev=x
    return float(max(0.0,hv))


def candidate_hv_gain_generic(front: np.ndarray, candidates: np.ndarray, ref=None) -> np.ndarray:
    C=np.atleast_2d(np.asarray(candidates,float)); m=C.shape[1]
    if m<1: raise RuntimeError("candidate_hv_gain needs >=1 objective")
    r=np.zeros(m,float) if ref is None else np.asarray(ref,float).reshape(-1)
    if len(r)!=m: raise RuntimeError("candidate_hv_gain reference dimension mismatch")
    F=np.empty((0,m),float) if np.size(front)==0 else np.atleast_2d(np.asarray(front,float))
    if F.shape[1]!=m: raise RuntimeError(f"front {F.shape} vs candidates {C.shape}")
    out=np.zeros(len(C),float); valid=np.isfinite(C).all(axis=1)
    if not np.any(valid): return out
    if m==1:
        base=max(float(r[0]),float(np.max(F[:,0])) if len(F) else float(r[0]))
        out[valid]=np.maximum(C[valid,0]-base,0.0); return out
    if m==2:
        Cv=C[valid].copy(); Cv[:,0]=np.maximum(Cv[:,0],r[0]); Cv[:,1]=np.maximum(Cv[:,1],r[1])
        P=pareto_points_generic(F) if len(F) else np.empty((0,2),float)
        if len(P)==0:
            out[valid]=np.maximum(Cv[:,0]-r[0],0)*np.maximum(Cv[:,1]-r[1],0); return out
        P=P[np.argsort(P[:,0])]; xe=P[:,0]; yl=P[:,1]; xs=np.concatenate([[r[0]],xe[:-1]])
        cx=Cv[:,0,None]; cy=Cv[:,1,None]
        gain=np.sum(np.maximum(np.minimum(cx,xe[None,:])-xs[None,:],0)*np.maximum(cy-yl[None,:],0),axis=1)
        gain += np.maximum(Cv[:,0]-xe[-1],0)*np.maximum(Cv[:,1]-r[1],0)
        out[valid]=np.maximum(gain,0); return out
    base=hypervolume_exact(F,r) if len(F) else 0.0
    for i in np.where(valid)[0]:
        U=np.vstack([F,C[i]]) if len(F) else C[i:i+1]
        out[i]=max(0.0,hypervolume_exact(U,r)-base)
    return out


def crossfit_subset_prior_predictions_generic(Xvisible,Yvisible,feasible_visible,subset_local,cfg,seed):
    from sklearn.model_selection import KFold
    X=np.asarray(Xvisible,float); Y=np.asarray(Yvisible,float); F=np.asarray(feasible_visible,bool).reshape(-1); ids=np.asarray(subset_local,int).reshape(-1)
    if X.ndim!=2 or Y.ndim!=2 or len(X)!=len(Y) or len(F)!=len(X):
        raise RuntimeError(f"crossfit shapes X={X.shape} Y={Y.shape} F={F.shape}")
    m=Y.shape[1]; out=np.full((len(ids),m),np.nan,float)
    if len(ids)==0: return out
    folds=max(2,min(int((cfg.get('surrogate_validation',{}) or {}).get('folds',5)),len(X)))
    pos={int(g):i for i,g in enumerate(ids.tolist())}
    for fi,(tr,te) in enumerate(KFold(n_splits=folds,shuffle=True,random_state=int(seed)+277).split(X),1):
        wanted=[int(g) for g in te if int(g) in pos]
        if not wanted: continue
        model=GenericGlobalSurrogate(cfg,int(seed)+1000+fi).fit(X[tr],Y[tr],F[tr]); pred,_=model.predict(X[np.asarray(wanted,int)])
        if np.asarray(pred).shape!=(len(wanted),m): raise RuntimeError("crossfit objective shape mismatch")
        for k,g in enumerate(wanted): out[pos[g]]=pred[k]
    if not np.isfinite(out).all(): raise RuntimeError("crossfit subset predictions incomplete")
    return out


def _objective_contract_parts(objective_contracts, m):
    contracts=list(objective_contracts or [])
    if not contracts:
        contracts=[{'deterministic_from_design':False} for _ in range(int(m))]
    if len(contracts)!=int(m):
        raise RuntimeError(f'objective contract length {len(contracts)} != objective dim {m}')
    stochastic=[i for i,o in enumerate(contracts) if not bool(o.get('deterministic_from_design',False))]
    deterministic=[i for i,o in enumerate(contracts) if bool(o.get('deterministic_from_design',False))]
    if not stochastic:
        raise RuntimeError('gsMOBO requires at least one stochastic/exact objective; all-deterministic problems should use a deterministic optimizer')
    return contracts,stochastic,deterministic


def _torch_normalize_design_objective(raw, objective):
    direction=str(objective.get('direction','maximize')).lower()
    norm=dict(objective.get('resolved_normalization') or {})
    method=str(norm.get('method','frozen_minmax')).lower()
    if method=='none':
        z=raw
        return z if direction=='maximize' else -z
    lo=float(norm['min']); hi=float(norm['max'])
    z=(raw-lo)/(hi-lo)
    return z if direction=='maximize' else 1.0-z


def _torch_design_objective_from_x(X, objective, features):
    import torch
    source=str(objective.get('source','')).lower()
    if source=='design_feature':
        idx=int(objective.get('resolved_feature_index',-1))
        if idx<0 or idx>=X.shape[-1]: raise RuntimeError(f"invalid resolved feature index for {objective.get('name')}")
        z=X[...,idx]
        transform=str(objective.get('transform','normalized')).lower()
        if transform in {'normalized','identity_normalized','demand'}: raw=z
        elif transform in {'one_minus_normalized','efficiency','one_minus_demand'}: raw=1.0-z
        else: raise RuntimeError(f'unsupported deterministic design_feature transform {transform!r}')
    elif source=='design_metric':
        metric=str(objective.get('metric') or objective.get('name')).lower()
        if metric in {'normalized_mean','mean','mean_x01','normalized_resource_cost'}: raw=X.mean(dim=-1)
        elif metric in {'normalized_sum','sum','sum_x01'}: raw=X.sum(dim=-1)
        elif metric in {'normalized_l2','l2'}: raw=torch.sqrt(torch.sum(X*X,dim=-1))
        elif metric in {'normalized_max','max','max_x01'}: raw=X.max(dim=-1).values
        elif metric in {'normalized_min','min','min_x01'}: raw=X.min(dim=-1).values
        else: raise RuntimeError(f'unsupported deterministic design_metric {metric!r}')
    else:
        raise RuntimeError(f'objective {objective.get("name")!r} is not a supported deterministic design objective')
    return _torch_normalize_design_objective(raw,objective)


def _assemble_full_objectives_torch(stochastic_values, X, contracts, features):
    """Insert exact deterministic design objectives beside stochastic GP outputs."""
    import torch
    sv=stochastic_values
    m=len(contracts); stochastic=[i for i,o in enumerate(contracts) if not bool(o.get('deterministic_from_design',False))]
    if sv.shape[-1]!=len(stochastic):
        raise RuntimeError(f'stochastic posterior/sample width {sv.shape[-1]} != expected {len(stochastic)}')
    channels=[None]*m
    for k,j in enumerate(stochastic): channels[j]=sv[...,k]
    for j,o in enumerate(contracts):
        if not bool(o.get('deterministic_from_design',False)): continue
        v=_torch_design_objective_from_x(X,o,features)
        # X lacks MC sample dimensions. Broadcast them in front until v matches
        # stochastic_values without its objective dimension.
        while v.dim()<sv.dim()-1: v=v.unsqueeze(0)
        v=v.expand(*sv.shape[:-1])
        channels[j]=v
    if any(v is None for v in channels): raise RuntimeError('failed assembling full objective tensor')
    return torch.stack(channels,dim=-1)


def fit_gsmobo_gp_generic(Xtrain: np.ndarray, Ytrain: np.ndarray, objective_contracts=None, features=None):
    import torch
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP,SingleTaskGP
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood
    X=np.asarray(Xtrain,float); Y=np.asarray(Ytrain,float)
    if X.ndim!=2 or Y.ndim!=2 or len(X)!=len(Y) or Y.shape[1]<1: raise ValueError(f"gsMOBO expects X=(n,d),Y=(n,m>=1); got {X.shape},{Y.shape}")
    if len(X)<2 or not np.isfinite(X).all() or not np.isfinite(Y).all(): raise RuntimeError("gsMOBO training arrays must be finite and n>=2")
    contracts,stochastic,deterministic=_objective_contract_parts(objective_contracts,Y.shape[1])
    features=list(features or [f'x{i}' for i in range(X.shape[1])])
    if len(features)!=X.shape[1]: raise RuntimeError(f'feature count {len(features)} != X dimension {X.shape[1]}')
    tx=torch.tensor(X,dtype=torch.double)
    models=[SingleTaskGP(tx,torch.tensor(Y[:,[j]],dtype=torch.double)) for j in stochastic]
    model=ModelListGP(*models); fit_gpytorch_mll(SumMarginalLogLikelihood(model.likelihood,model)); model.eval()
    model._gem_objective_dim=int(Y.shape[1]); model._gem_stochastic_objective_indices=list(map(int,stochastic)); model._gem_deterministic_objective_indices=list(map(int,deterministic)); model._gem_objective_contracts=contracts; model._gem_features=features
    return model


def qparego_scalarization_components_generic(model, X: np.ndarray, Y: np.ndarray, weight: np.ndarray, objective_contracts=None, features=None):
    import torch
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    X=np.asarray(X,float); Y=np.asarray(Y,float); m=Y.shape[1]
    contracts,stochastic,_=_objective_contract_parts(objective_contracts or getattr(model,'_gem_objective_contracts',None),m)
    features=list(features or getattr(model,'_gem_features',[f'x{i}' for i in range(X.shape[1])]))
    tx=torch.tensor(X,dtype=torch.double); ty=torch.tensor(Y,dtype=torch.double); model.eval()
    with torch.no_grad(): pm_stoch=model.posterior(tx).mean
    pm_full=_assemble_full_objectives_torch(pm_stoch,tx,contracts,features)
    has_det=any(bool(o.get('deterministic_from_design',False)) for o in contracts)
    if int(m)==1:
        # Scalar constrained problems do not need a synthetic multiobjective
        # scalarizer.  Use ordinary qLogEI on the one maximization-space channel.
        def objective_fn(samples,X=None):
            if X is None and has_det:
                raise RuntimeError('deterministic design objectives require candidate X in scalar acquisition objective')
            full=samples if not has_det else _assemble_full_objectives_torch(samples,X,contracts,features)
            return full[...,0]
        obj=GenericMCObjective(objective_fn)
        with torch.no_grad(): best_f=ty[:,0].max()
        return obj,best_f
    tw=torch.tensor(np.asarray(weight,float),dtype=torch.double)
    scalar=get_chebyshev_scalarization(weights=tw,Y=pm_full)
    def objective_fn(samples,X=None):
        if X is None and has_det:
            raise RuntimeError('deterministic design objectives require candidate X in qParEGO objective')
        full=samples if not has_det else _assemble_full_objectives_torch(samples,X,contracts,features)
        return scalar(full)
    obj=GenericMCObjective(objective_fn)
    with torch.no_grad(): best_f=scalar(ty).max()
    return obj,best_f


def continuous_qparego_select_generic(model,Xtrain,Ytrain,q,seed,mc_samples=256,num_restarts=5,raw_samples=512,maxiter=200,objective_contracts=None,features=None):
    import torch
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.optim.optimize import optimize_acqf_list
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.sampling import sample_simplex
    X=np.asarray(Xtrain,float); Y=np.asarray(Ytrain,float); q=int(q)
    if X.ndim!=2 or Y.ndim!=2 or len(X)!=len(Y) or Y.shape[1]<1: raise ValueError(f"qParEGO expects X=(n,d),Y=(n,m>=1); got {X.shape},{Y.shape}")
    m=Y.shape[1]
    if q<1: return np.empty((0,X.shape[1])),np.empty(0),np.empty((0,m))
    tx=torch.tensor(X,dtype=torch.double); torch.manual_seed(int(seed)); sampler=SobolQMCNormalSampler(torch.Size([int(mc_samples)]),seed=int(seed)+1); acqs=[]; ws=[]
    for i in range(q):
        sd=int(seed)+101+i
        try: w=sample_simplex(d=m,n=1,qmc=False,seed=sd,device=tx.device,dtype=torch.double).squeeze(0)
        except TypeError: torch.manual_seed(sd); w=sample_simplex(m,device=tx.device,dtype=torch.double).squeeze()
        obj,best=qparego_scalarization_components_generic(model,X,Y,w.detach().cpu().numpy(),objective_contracts,features)
        acqs.append(qLogExpectedImprovement(model=model,best_f=best,sampler=sampler,objective=obj)); ws.append(w.detach().cpu().numpy())
    d=X.shape[1]; bounds=torch.stack([torch.zeros(d,dtype=torch.double),torch.ones(d,dtype=torch.double)])
    cand,vals=optimize_acqf_list(acq_function_list=acqs,bounds=bounds,num_restarts=int(num_restarts),raw_samples=int(raw_samples),options={'batch_limit':10,'maxiter':int(maxiter)})
    cn=cand.detach().cpu().numpy(); vn=np.asarray(vals.detach().cpu().numpy(),float).reshape(-1) if vals is not None else np.full(len(cn),np.nan)
    return cn,vn,np.asarray(ws,float).reshape(q,m)


def posterior_mean_std_generic(model,X:np.ndarray,batch:int=1024):
    import torch
    X=np.asarray(X,float); means=[]; stds=[]; model.eval()
    with torch.no_grad():
        for s in range(0,len(X),int(batch)):
            tx=torch.tensor(X[s:s+int(batch)],dtype=torch.double); post=model.posterior(tx)
            means.append(np.asarray(post.mean.detach().cpu().numpy(),float).reshape(len(tx),-1)); var=np.asarray(post.variance.detach().cpu().numpy(),float).reshape(len(tx),-1); stds.append(np.sqrt(np.maximum(var,0)))
    return (np.vstack(means),np.vstack(stds)) if means else (np.empty((0,0)),np.empty((0,0)))


def install_generic_core(ns:dict)->None:
    if ns.get('_gem_generic_core_installed',False) or ns.get('_gem_generic_core_v030',False): return
    ns['pareto_points']=pareto_points_generic
    ns['hypervolume_nd']=hypervolume_exact
    ns['candidate_hv_gain']=candidate_hv_gain_generic
    ns['GlobalSurrogate']=GenericGlobalSurrogate
    ns['crossfit_subset_prior_predictions']=crossfit_subset_prior_predictions_generic
    ns['fit_gsmobo_gp']=fit_gsmobo_gp_generic
    ns['continuous_qparego_select']=continuous_qparego_select_generic
    ns['posterior_mean_std']=posterior_mean_std_generic
    ns['_gem_generic_core_installed']=True
    ns['_gem_generic_core_v030']=True
