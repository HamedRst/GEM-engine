#!/usr/bin/env python3
from __future__ import annotations
import argparse,json,time,traceback
from pathlib import Path
import joblib,numpy as np,pandas as pd,yaml
from generic_problem import feature_bounds, normalize_design
from generic_core import pareto_points_generic
from runtime import ExactRunner,LiveContext,hv
from nsga_algorithms import METHODS,RUNNERS,representative_prior_ids

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--study-config',required=True);ap.add_argument('--case',required=True);ap.add_argument('--budget',type=int,default=128);a=ap.parse_args();sp=Path(a.study_config).resolve();cfg=yaml.safe_load(sp.read_text()) or {};case=a.case
 if case not in cfg['cases']:raise SystemExit(f'Unknown case {case}')
 prep=Path(cfg['paths']['prepared_root'])/case;parent=yaml.safe_load((prep/'problem_config.yaml').read_text()) or {};p=parent['problem'];comp=p['compiled_problem'];baseline=pd.read_csv(prep/'baseline_compact.csv.gz');features,bounds=feature_bounds(p);X=normalize_design(baseline,features,bounds);Y=baseline[comp['objective_norm_columns']].to_numpy(float);F=baseline['feasible'].astype(bool).to_numpy();case_seed=int(__import__('hashlib').sha256(case.encode('utf-8')).hexdigest()[:8],16);ids=representative_prior_ids(X,Y,F,int(cfg['study']['gp_subset_rows']),int(cfg['study']['seed'])+case_seed)
 out=Path(cfg['paths']['output_root'])/case/f"seed_{cfg['study']['seed']}";out.mkdir(parents=True,exist_ok=True);qc=out/'surrogate_qc';model=joblib.load(qc/'final_global_surrogate.joblib');z=np.load(qc/'oof_prior_predictions.npz');oof=np.asarray(z['oof'],float);subset_oof=oof[ids]
 prior_rec={'rows':int(len(ids)),'unique_rows':int(len(np.unique(ids))),'archive_feasible_rows':int(np.sum(F[ids])),'archive_infeasible_rows':int(np.sum(~F[ids])),'ids_sha256':__import__('hashlib').sha256(np.asarray(ids,dtype=np.int64).tobytes()).hexdigest(),'selection':'objective_extremes+pareto_representatives+sobol_nearest_global_coverage','landscape_prior_rows':int(cfg['study']['landscape_prior_rows']),'reference_semantics_rows':int(cfg['study']['reference_semantics_rows']),'population_warmstart':'exact_archive_outcomes_feasibility_aware','surrogate_pseudolabels_used':False}
 (out/'representative_prior.json').write_text(json.dumps(prior_rec,indent=2),encoding='utf-8')
 c=dict(cfg['cases'][case]);c.update({'prepared_dir':str(prep),'study_config_path':str(sp),'case_key':case});er=ExactRunner(cfg,parent,out,c);pf=er.preflight();(out/'exact_bridge_preflight.json').write_text(json.dumps(pf,indent=2),encoding='utf-8')
 if pf['status']!='PASS':raise SystemExit(f'{case}: exact bridge preflight FAIL; no live calls made: {pf}')
 ctx=LiveContext(cfg,parent,baseline,prep/'baseline_compact.csv.gz',model,ids,subset_oof,out,er,int(cfg['study']['seed']),int(a.budget));ctx.oof_all=oof
 print('='*110);print(f'GEM SYSTEMATIC NSGA-II DEVELOPMENT — {case}');print(f'Problem={comp["problem_id"]} | objectives={comp["objective_names"]} | dim={len(features)} | budget={a.budget}');print('='*110,flush=True)
 state_path=out/'runtime_state.json';state=json.loads(state_path.read_text()) if state_path.exists() else {'cumulative_seconds':{}};cum={k:float(v) for k,v in state.get('cumulative_seconds',{}).items()};errors={}
 for m in METHODS:
  before=len(ctx.history(m))
  if before>int(a.budget):raise RuntimeError(f'{m}: existing history {before} exceeds requested budget {a.budget}')
  if before==int(a.budget):print(f'[{m}] already complete {before}/{a.budget}; skip');continue
  print(f'\n[{m}] resume {before}->{a.budget}',flush=True);t=time.time()
  try:
   RUNNERS[m](ctx)
   if len(ctx.history(m))!=int(a.budget):raise RuntimeError(f'{m}: ended at {len(ctx.history(m))}/{a.budget}')
  except Exception as e:
   errors[m]=f'{type(e).__name__}: {e}';traceback.print_exc()
  cum[m]=cum.get(m,0.0)+(time.time()-t);state_path.write_text(json.dumps({'cumulative_seconds':cum},indent=2),encoding='utf-8')
  if m in errors:
   print(f'{case}: fail-closed after technical failure in {m}; later methods in this cell are not started.',flush=True)
   break
 rows=[]
 for m in METHODS:
  s=ctx.summary(m,cum.get(m));s['error']=errors.get(m,'');rows.append(s)
 pd.DataFrame(rows).to_csv(out/'summary.csv',index=False);calls=[]
 for m in METHODS:
  h=ctx.history(m)
  if len(h):g=h.copy();g.insert(0,'case',case);calls.append(g)
 if calls:pd.concat(calls,ignore_index=True).to_csv(out/'calls_all.csv',index=False)
 pd.DataFrame(ctx.baseline_front,columns=ctx.objective_norm_columns).to_csv(out/'frozen_pareto_front.csv',index=False)
 # Matched initialization is a scientific contract, not merely a convenience.
 init_hashes={};init_n=int(cfg['study']['common_initial_calls']);xcols=[f'x01__{f}' for f in features]
 for m in METHODS:
  hh=ctx.history(m)
  if len(hh)>=init_n and all(c in hh.columns for c in xcols):
   arr=np.round(hh.iloc[:init_n][xcols].to_numpy(float),12);init_hashes[m]=__import__('hashlib').sha256(arr.tobytes()).hexdigest()
 init_identity=(len(init_hashes)==len(METHODS) and len(set(init_hashes.values()))==1)
 if int(a.budget)>=init_n and not init_identity:
  errors['COMMON_INIT_IDENTITY']='first 16 logical design points are not identical across all methods'
 manifest={'status':'COMPLETE' if not errors and all(len(ctx.history(m))==int(a.budget) for m in METHODS) else 'PARTIAL_OR_ERROR','job_id':cfg['cases'][case].get('job_id',''),'worker_assignment':cfg['cases'][case].get('worker_assignment',cfg.get('worker','')),'predicted_weight':cfg['cases'][case].get('predicted_weight'),'case':case,'problem_id':comp['problem_id'],'problem_hash':comp['problem_hash'],'species':cfg['cases'][case]['species'],'problem_key':cfg['cases'][case]['problem_key'],'objective_names':comp['objective_names'],'objective_directions':comp['objective_directions'],'stochastic_objective_indices':comp.get('stochastic_objective_indices',[]),'deterministic_objective_indices':comp.get('deterministic_objective_indices',[]),'budget':int(a.budget),'methods':METHODS,'errors':errors,'logical_calls':{m:len(ctx.history(m)) for m in METHODS},'baseline_hv':ctx.baseline_hv,'common_initialization_identity_pass':bool(init_identity),'common_initialization_hashes':init_hashes,'source_provenance':p.get('source_provenance',{})};(out/'run_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
 print('\n'+pd.DataFrame(rows).to_string(index=False));
 if errors:raise SystemExit(f'{case}: method errors: {errors}')
 print(f'CASE COMPLETE: {case} @ {a.budget}')
if __name__=='__main__':main()
