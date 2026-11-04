from __future__ import annotations
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import hashlib
import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_predict
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score

_SURROGATE_MEMORY_CACHE: dict[str, "SurrogateBundle"] = {}

def _cache_key(d: pd.DataFrame, variables: list[str], target: str, cfg: dict[str,Any], task: str) -> str:
    # The atlas passes the frozen landscape fingerprint when available.  We also
    # hash the target values + row index so filtered/derived targets cannot silently
    # collide in the persistent cache.
    h=hashlib.sha256()
    h.update(str(cfg.get("_landscape_fingerprint") or "no_fingerprint").encode())
    h.update(b"\0")
    h.update(str(task).encode()); h.update(b"\0")
    h.update(str(target).encode()); h.update(b"\0")
    h.update("|".join(map(str,variables)).encode()); h.update(b"\0")
    relevant={k:v for k,v in cfg.items() if not str(k).startswith("_") and k not in {"cache_dir"}}
    h.update(json.dumps(relevant,sort_keys=True,default=str).encode()); h.update(b"\0")
    hv=pd.util.hash_pandas_object(d[[*variables, target]], index=True).to_numpy(dtype="uint64",copy=False)
    h.update(hv.tobytes())
    return h.hexdigest()

def clear_surrogate_cache() -> None:
    _SURROGATE_MEMORY_CACHE.clear()

@dataclass
class SurrogateBundle:
    model: Any
    variables: list[str]
    target: str
    task: str
    metrics: dict[str,float]
    bounds: np.ndarray

    def predict(self, X:np.ndarray) -> tuple[np.ndarray,np.ndarray|None]:
        pred=np.asarray(self.model.predict(X),float)
        trees=getattr(self.model,'estimators_',None)
        unc=None
        if trees:
            tp=np.vstack([np.asarray(t.predict(X),float) for t in trees]); unc=np.nanstd(tp,axis=0)
        return pred,unc

def fit_surrogate(df:pd.DataFrame, variables:list[str], target:str, cfg:dict[str,Any], task:str='regression', seed:int=20260822) -> SurrogateBundle:
    cols=variables+[target]; d=df[cols].copy()
    for c in cols: d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.replace([np.inf,-np.inf],np.nan).dropna()
    max_rows=int(cfg.get('max_train_rows',50000))
    if len(d)>max_rows: d=d.sample(max_rows,random_state=seed)
    if len(d)<max(50,5*len(variables)): raise RuntimeError(f'Insufficient finite rows for surrogate: {len(d)}')

    key=_cache_key(d,variables,target,cfg,task)
    if key in _SURROGATE_MEMORY_CACHE:
        return _SURROGATE_MEMORY_CACHE[key]
    cache_dir=cfg.get('_cache_dir') or os.environ.get('GEM_EXPLORATION_SURROGATE_CACHE')
    cache_path=None
    if cache_dir:
        cache_path=Path(str(cache_dir)).expanduser().resolve()/f'{key}.joblib'
        if cache_path.exists():
            try:
                bundle=joblib.load(cache_path)
                if isinstance(bundle,SurrogateBundle) and bundle.variables==variables and bundle.target==target and bundle.task==task:
                    _SURROGATE_MEMORY_CACHE[key]=bundle
                    return bundle
            except Exception:
                pass

    X=d[variables].to_numpy(float); y=d[target].to_numpy(float)
    n_estimators=int(cfg.get('n_estimators',300)); model_name=cfg.get('model','extra_trees'); n_jobs=int(cfg.get('n_jobs',-1))
    if task=='classification':
        y=(y>0.5).astype(int)
        cls=RandomForestClassifier if model_name=='random_forest' else ExtraTreesClassifier
        model=cls(n_estimators=n_estimators,random_state=seed,n_jobs=n_jobs,min_samples_leaf=2,class_weight='balanced')
        counts=np.bincount(y) if len(y) else np.array([])
        if len(counts)<2 or counts.min()==0:
            # Deterministic one-class feasibility landscapes are common.  Preserve
            # exact semantics rather than forcing an invalid classifier.
            from sklearn.dummy import DummyClassifier
            constant=int(y[0]) if len(y) else 1
            model=DummyClassifier(strategy='constant',constant=constant)
            metrics={'auc':float('nan'),'n':len(y),'one_class':True,'constant_class':constant}
        else:
            cv=StratifiedKFold(n_splits=min(int(cfg.get('cv_folds',5)),max(2,int(counts.min()))),shuffle=True,random_state=seed)
            try:
                prob=cross_val_predict(model,X,y,cv=cv,method='predict_proba',n_jobs=n_jobs)[:,1]; auc=float(roc_auc_score(y,prob))
            except Exception: auc=float('nan')
            metrics={'auc':auc,'n':len(y),'one_class':False}
            if np.isfinite(auc) and auc<float(cfg.get('min_auc',0.65)): raise RuntimeError(f'Surrogate classifier QC failed: AUC={auc:.4f} < {float(cfg.get("min_auc",0.65)):.4f}')
    else:
        cls=RandomForestRegressor if model_name=='random_forest' else ExtraTreesRegressor
        model=cls(n_estimators=n_estimators,random_state=seed,n_jobs=n_jobs,min_samples_leaf=2,max_features=1.0)
        nfold=min(int(cfg.get('cv_folds',5)),max(2,len(y)//10))
        cv=KFold(n_splits=nfold,shuffle=True,random_state=seed)
        pred=cross_val_predict(model,X,y,cv=cv,n_jobs=n_jobs)
        r2=float(r2_score(y,pred)); rmse=float(mean_squared_error(y,pred)**0.5); mae=float(mean_absolute_error(y,pred))
        metrics={'r2':r2,'rmse':rmse,'mae':mae,'n':len(y),'cv_folds':nfold}
        if r2<float(cfg.get('min_r2',0.50)): raise RuntimeError(f'Surrogate regression QC failed: R2={r2:.4f} < {float(cfg.get("min_r2",0.50)):.4f}')
    model.fit(X,y)
    bounds=np.column_stack([np.nanmin(X,axis=0),np.nanmax(X,axis=0)])
    bundle=SurrogateBundle(model,variables,target,task,metrics,bounds)
    _SURROGATE_MEMORY_CACHE[key]=bundle
    if cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True,exist_ok=True)
            tmp=cache_path.with_suffix('.tmp.joblib')
            joblib.dump(bundle,tmp,compress=3)
            os.replace(tmp,cache_path)
        except Exception:
            pass
    return bundle

def finite_difference_gradient(bundle:SurrogateBundle, point:np.ndarray, step_fraction:float=0.01) -> np.ndarray:
    p=np.asarray(point,float); span=bundle.bounds[:,1]-bundle.bounds[:,0]; span=np.where(span>0,span,1.0); out=[]
    for j in range(len(p)):
        h=max(span[j]*step_fraction,1e-9); lo=p.copy(); hi=p.copy(); lo[j]=max(bundle.bounds[j,0],p[j]-h); hi[j]=min(bundle.bounds[j,1],p[j]+h)
        denom=hi[j]-lo[j]
        if denom<=1e-15: out.append(0.0); continue
        yl,_=bundle.predict(lo.reshape(1,-1)); yh,_=bundle.predict(hi.reshape(1,-1)); out.append(float((yh[0]-yl[0])/denom))
    return np.asarray(out,float)
