#!/usr/bin/env python3
from __future__ import annotations
import argparse,hashlib,json,math
from pathlib import Path
import joblib,numpy as np,pandas as pd,yaml
from sklearn.metrics import brier_score_loss,mean_squared_error,r2_score,roc_auc_score
from sklearn.model_selection import KFold
from core import GlobalSurrogate
from generic_problem import feature_bounds,normalize_design

def sha256(p):
 h=hashlib.sha256();
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def asbool(s): return s.to_numpy(bool) if pd.api.types.is_bool_dtype(s) else s.astype(str).str.lower().isin(['true','1','yes','optimal_generic_problem']).to_numpy(bool)
def r2(y,p):
 ok=np.isfinite(y)&np.isfinite(p); return float(r2_score(y[ok],p[ok])) if ok.sum()>2 and np.std(y[ok])>1e-15 else float('nan')
def nrmse(y,p):
 ok=np.isfinite(y)&np.isfinite(p)
 if not ok.any(): return float('nan')
 span=float(np.max(y[ok])-np.min(y[ok])); rm=math.sqrt(float(mean_squared_error(y[ok],p[ok]))); return 0.0 if span<=1e-15 and rm<=1e-15 else (float(rm/span) if span>1e-15 else float('nan'))
def rankcorr(y,p):
 ok=np.isfinite(y)&np.isfinite(p)
 if ok.sum()<3:return float('nan')
 a=pd.Series(y[ok]).rank().to_numpy(); b=pd.Series(p[ok]).rank().to_numpy(); return float(np.corrcoef(a,b)[0,1]) if np.std(a)>0 and np.std(b)>0 else float('nan')

def validate_case(cfg,case,study_path,force=False):
 root=study_path.parent; seed=int(cfg['study']['seed']); prep=root/Path(cfg['paths']['prepared_root'])/case; bp=prep/'baseline_compact.csv.gz'; pp=prep/'problem_config.yaml'; baseline=pd.read_csv(bp); parent=yaml.safe_load(pp.read_text()) or {}; comp=parent['problem']['compiled_problem']; m=int(comp['objective_dim']); ph=comp['problem_hash']; bh=sha256(bp)
 outdir=root/Path(cfg['paths']['output_root'])/case/f'seed_{seed}'/'surrogate_qc'; summary_path=outdir/'surrogate_qc_summary.json'; oof_path=outdir/'oof_prior_predictions.npz'; model_path=outdir/'final_global_surrogate.joblib'
 if not force and summary_path.exists() and oof_path.exists() and model_path.exists():
  try:
   old=json.loads(summary_path.read_text());
   with np.load(oof_path) as z:o=np.asarray(z['oof'])
   if old.get('status')=='PASS' and old.get('problem_hash')==ph and old.get('baseline_sha256')==bh and o.shape==(len(baseline),m) and np.isfinite(o).all():
    print(f'[{case}] surrogate QC cache HIT | problem hash + archive hash match'); return old
  except Exception: pass
 features,bounds=feature_bounds(parent['problem']); X=normalize_design(baseline,features,bounds); Y=baseline[comp['objective_norm_columns']].to_numpy(float); F=asbool(baseline['feasible'])
 folds=int(cfg.get('surrogate',{}).get('generic_oof_folds',5)); kf=KFold(n_splits=folds,shuffle=True,random_state=seed+177); oof=np.full_like(Y,np.nan); pinf=np.full(len(X),np.nan); foldid=np.full(len(X),-1,int); foldrows=[]
 for fi,(tr,te) in enumerate(kf.split(X),1):
  model=GlobalSurrogate(cfg,seed+5000+fi).fit(X[tr],Y[tr],F[tr]); pred,pprob=model.predict(X[te]); pred=np.asarray(pred,float); pprob=np.asarray(pprob,float).reshape(-1)
  if pred.shape!=(len(te),m): raise RuntimeError(f'{case} fold {fi}: surrogate returned {pred.shape}, expected {(len(te),m)}')
  oof[te]=pred; pinf[te]=pprob; foldid[te]=fi; good=F[te]&np.isfinite(Y[te]).all(axis=1)&np.isfinite(pred).all(axis=1); row={'case':case,'fold':fi,'train_rows':len(tr),'test_rows':len(te),'test_feasible_rows':int(good.sum())}
  for j,o in enumerate(comp['objectives']): row[f"{o['slug']}__r2"]=r2(Y[te][good,j],pred[good,j]); row[f"{o['slug']}__range_nrmse"]=nrmse(Y[te][good,j],pred[good,j]); row[f"{o['slug']}__spearman"]=rankcorr(Y[te][good,j],pred[good,j])
  yinf=(~F[te]).astype(int); row['feasibility_brier']=float(brier_score_loss(yinf,np.clip(pprob,0,1))); row['feasibility_roc_auc']=float(roc_auc_score(yinf,np.clip(pprob,0,1))) if len(np.unique(yinf))==2 else float('nan'); foldrows.append(row); print(f'[{case}] fold {fi}/{folds} complete | objectives={m}',flush=True)
 if np.any(foldid<0) or not np.isfinite(oof).all() or not np.isfinite(pinf).all(): raise RuntimeError(f'{case}: incomplete/nonfinite OOF predictions')
 final=GlobalSurrogate(cfg,seed+9991).fit(X,Y,F); probe,_=final.predict(X[:min(64,len(X))]);
 if np.asarray(probe).shape!=(min(64,len(X)),m): raise RuntimeError(f'{case}: final surrogate objective shape mismatch')
 outdir.mkdir(parents=True,exist_ok=True); joblib.dump(final,model_path); np.savez_compressed(oof_path,oof=oof,p_infeasible=pinf,fold=foldid); pd.DataFrame(foldrows).to_csv(outdir/'surrogate_qc_folds.csv',index=False)
 good=F&np.isfinite(Y).all(axis=1)&np.isfinite(oof).all(axis=1); metrics={}
 for j,o in enumerate(comp['objectives']): metrics[o['slug']]={'r2':r2(Y[good,j],oof[good,j]),'range_nrmse':nrmse(Y[good,j],oof[good,j]),'spearman':rankcorr(Y[good,j],oof[good,j])}
 yinf=(~F).astype(int); summary={'status':'PASS','problem_contract':'gemproblem/v0.2','problem_id':comp['problem_id'],'problem_hash':ph,'baseline_sha256':bh,'case':case,'rows':len(X),'dimension':X.shape[1],'objective_dim':m,'objective_names':comp['objective_names'],'objective_directions':comp['objective_directions'],'folds':folds,'feasible_rows':int(F.sum()),'objective_metrics':metrics,'feasibility_oof_brier':float(brier_score_loss(yinf,np.clip(pinf,0,1))),'feasibility_oof_roc_auc':float(roc_auc_score(yinf,np.clip(pinf,0,1))) if len(np.unique(yinf))==2 else None,'live_exact_calls':0,'pass_rule':'semantic/software gate only; no outcome-tuned accuracy cutoff'}; summary_path.write_text(json.dumps(summary,indent=2),encoding='utf-8'); print(f'[{case}] GENERIC SURROGATE QC PASS | objective_dim={m} | ZERO live calls',flush=True); return summary

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--study-config',default='study_config.yaml'); ap.add_argument('--cases',default=None); ap.add_argument('--force',action='store_true'); args=ap.parse_args(); p=Path(args.study_config).resolve(); cfg=yaml.safe_load(p.read_text()) or {}; cases=list(cfg['cases']);
 if args.cases: cases=[x.strip() for x in args.cases.split(',') if x.strip()]
 fail=[]
 for c in cases:
  try: validate_case(cfg,c,p,args.force)
  except Exception as e: print(f'[{c}] SURROGATE QC FAIL: {type(e).__name__}: {e}'); fail.append((c,str(e)))
 if fail: raise SystemExit(f'Generic surrogate QC failed: {fail}')
 print('\nALL GENERIC SURROGATE QC: PASS | ZERO live calls')
if __name__=='__main__': main()
