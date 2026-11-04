#!/usr/bin/env python3
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import joblib, numpy as np, pandas as pd, yaml
from sklearn.metrics import brier_score_loss, mean_squared_error, r2_score, roc_auc_score
from sklearn.model_selection import KFold
from generic_surrogate import GenericGlobalSurrogate
from generic_problem import feature_bounds, normalize_design

def sha(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def asbool(s): return s.to_numpy(bool) if pd.api.types.is_bool_dtype(s) else s.astype(str).str.lower().isin(['true','1','yes','optimal_generic_problem']).to_numpy(bool)
def metric(y,p):
 ok=np.isfinite(y)&np.isfinite(p)
 if ok.sum()<3:return {'r2':None,'range_nrmse':None,'rank_corr':None}
 span=float(np.max(y[ok])-np.min(y[ok])); rm=math.sqrt(float(mean_squared_error(y[ok],p[ok]))); r=float(r2_score(y[ok],p[ok])) if np.std(y[ok])>1e-15 else None; a=pd.Series(y[ok]).rank().to_numpy();b=pd.Series(p[ok]).rank().to_numpy(); rc=float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else None
 return {'r2':r,'range_nrmse':(rm/span if span>1e-15 else 0.0 if rm<=1e-15 else None),'rank_corr':rc}
def validate(cfg,case,force=False):
 root=Path(cfg['paths']['prepared_root']).resolve()/case; bp=root/'baseline_compact.csv.gz'; pp=root/'problem_config.yaml'; df=pd.read_csv(bp); parent=yaml.safe_load(pp.read_text()) or {}; comp=parent['problem']['compiled_problem']; m=int(comp['objective_dim']); out=Path(cfg['paths']['output_root']).resolve()/case/f"seed_{cfg['study']['seed']}"/'surrogate_qc'; out.mkdir(parents=True,exist_ok=True); summ=out/'surrogate_qc_summary.json'; oo=out/'oof_prior_predictions.npz'; fm=out/'final_global_surrogate.joblib'; bh=sha(bp)
 if not force and summ.exists() and oo.exists() and fm.exists():
  try:
   old=json.loads(summ.read_text()); z=np.load(oo); arr=np.asarray(z['oof']);
   if old.get('status')=='PASS' and old.get('baseline_sha256')==bh and old.get('problem_hash')==comp['problem_hash'] and arr.shape==(len(df),m) and np.isfinite(arr).all(): print(f'[{case}] surrogate QC cache HIT'); return old
  except Exception: pass
 features,bounds=feature_bounds(parent['problem']); X=normalize_design(df,features,bounds); Y=df[comp['objective_norm_columns']].to_numpy(float); F=asbool(df['feasible']); folds=int(cfg['surrogate']['generic_oof_folds']); oof=np.full_like(Y,np.nan); pinf=np.full(len(X),np.nan); foldid=np.full(len(X),-1,int)
 model_cfg=json.loads(json.dumps(cfg)); model_cfg.setdefault('surrogate',{})['objective_contracts']=comp['objectives']; model_cfg['surrogate']['features']=features
 for fi,(tr,te) in enumerate(KFold(n_splits=folds,shuffle=True,random_state=int(cfg['study']['seed'])+177).split(X),1):
  model=GenericGlobalSurrogate(model_cfg,int(cfg['study']['seed'])+5000+fi).fit(X[tr],Y[tr],F[tr]); pr,pi=model.predict(X[te]); oof[te]=pr;pinf[te]=pi;foldid[te]=fi; print(f'[{case}] OOF fold {fi}/{folds}',flush=True)
 if not np.isfinite(oof).all() or not np.isfinite(pinf).all(): raise RuntimeError(f'{case}: incomplete OOF')
 final=GenericGlobalSurrogate(model_cfg,int(cfg['study']['seed'])+9991).fit(X,Y,F); joblib.dump(final,fm); np.savez_compressed(oo,oof=oof,p_infeasible=pinf,fold=foldid)
 good=F&np.isfinite(Y).all(axis=1); metrics={o['slug']:metric(Y[good,j],oof[good,j]) for j,o in enumerate(comp['objectives'])}; det_err={o['slug']:(float(np.max(np.abs(oof[:,j]-Y[:,j]))) if bool(o.get('deterministic_from_design',False)) else None) for j,o in enumerate(comp['objectives'])}; yinf=(~F).astype(int); summary={'status':'PASS','case':case,'problem_id':comp['problem_id'],'problem_hash':comp['problem_hash'],'baseline_sha256':bh,'rows':len(df),'dimension':X.shape[1],'objective_dim':m,'objective_names':comp['objective_names'],'stochastic_objective_indices':comp.get('stochastic_objective_indices',[]),'deterministic_objective_indices':comp.get('deterministic_objective_indices',[]),'feasible_rows':int(F.sum()),'objective_metrics':metrics,'deterministic_oof_max_abs_error':det_err,'feasibility_brier':float(brier_score_loss(yinf,np.clip(pinf,0,1))),'feasibility_roc_auc':float(roc_auc_score(yinf,np.clip(pinf,0,1))) if len(np.unique(yinf))==2 else None,'live_exact_calls':0}; summ.write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(f'[{case}] SURROGATE QC PASS | ZERO live calls'); return summary
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study-config',required=True);ap.add_argument('--case',required=True);ap.add_argument('--force',action='store_true');a=ap.parse_args();cfg=yaml.safe_load(Path(a.study_config).read_text()) or {};validate(cfg,a.case,a.force)
if __name__=='__main__':main()
