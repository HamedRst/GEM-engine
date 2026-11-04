#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,yaml
from generic_core import hypervolume_exact
from gsmobo_algorithms import BASE_NO50K,BASE_50K
from selected_panel import SELECTED_METHODS,SELECTED_HEURISTICS,METHOD_META,validate_selected_panel
CHECKPOINTS=[32,64,128]

def _bool(s):
 if pd.api.types.is_bool_dtype(s):return s.astype(bool)
 return s.astype(str).str.lower().isin(['true','1','yes','optimal','feasible','optimal_generic_problem'])

def _num_mean(df,col,mask=None):
 if col not in df.columns:return np.nan
 x=pd.to_numeric(df[col],errors='coerce')
 if mask is not None:x=x[mask]
 return float(x.mean()) if x.notna().any() else np.nan

def _validate_history(h,method,expected_budget=128):
 required={'call','exact_feasible','cache_hit','archive_hv_after'}
 miss=sorted(required-set(h.columns))
 if miss:return f'missing required calls.csv columns {miss}'
 c=pd.to_numeric(h['call'],errors='coerce')
 if c.isna().any():return 'non-numeric call index'
 arr=c.to_numpy(float)
 if not np.allclose(arr,np.round(arr),rtol=0,atol=0):return 'non-integer call index'
 ints=arr.astype(int)
 if len(np.unique(ints))!=len(ints):return 'duplicate call indices'
 if len(ints)!=int(expected_budget):return f'logical call rows={len(ints)} expected={expected_budget}'
 if not np.array_equal(np.sort(ints),np.arange(1,int(expected_budget)+1)):return f'call sequence is not exactly 1..{expected_budget}'
 return None

def _self_hv_sequence(g,objcols,ref):
 front=[];vals=[];gains=[];ext=[]
 for _,r in g.sort_values('call').iterrows():
  before=vals[-1] if vals else 0.0
  feas=str(r.get('exact_feasible','')).strip().lower() in {'true','1','yes','optimal','feasible','optimal_generic_problem'} or r.get('exact_feasible') is True
  y=np.array([pd.to_numeric(r.get(c,np.nan),errors='coerce') for c in objcols],float)
  if feas and np.isfinite(y).all():front.append(y)
  after=float(hypervolume_exact(np.asarray(front,float),ref)) if front else 0.0
  vals.append(after);gains.append(max(0.0,after-before));ext.append(bool(after>before+1e-14))
 return np.asarray(vals,float),np.asarray(gains,float),np.asarray(ext,bool)

def row_at(h,cp,landscape_hv,objective_dim,objcols,ref,is_no_prior):
 calls=pd.to_numeric(h['call'],errors='coerce');g=h[calls<=int(cp)].copy().sort_values('call')
 if len(g)!=int(cp) or not np.array_equal(pd.to_numeric(g['call'],errors='coerce').to_numpy(int),np.arange(1,int(cp)+1)):return None
 r=g.iloc[-1];self_seq,self_gains,self_ext=_self_hv_sequence(g,objcols,ref);self_hv=float(self_seq[-1]);union_hv=float(pd.to_numeric(r['archive_hv_after'],errors='coerce'))
 if is_no_prior:
  baseline=0.0;primary=self_hv;delta=self_hv;head=1.0;mode='LIVE_SELF_FRONT'
  gain_seq=self_gains;ext_seq=self_ext;adaptive_score=self_seq[pd.to_numeric(g['call'],errors='coerce').to_numpy()>16]
 else:
  baseline=float(landscape_hv);primary=union_hv;delta=primary-baseline;head=max(1e-12,1.0-baseline);mode='PRIOR_PLUS_LIVE_UNION'
  gain_seq=pd.to_numeric(g.get('exact_hv_gain_this_call',pd.Series(np.zeros(len(g)))),errors='coerce').fillna(0).to_numpy(float);ext_seq=_bool(g.get('exact_front_extension',pd.Series(False,index=g.index))).to_numpy(bool)
  adaptive=g[pd.to_numeric(g['call'],errors='coerce')>16].copy();adaptive_score=pd.to_numeric(adaptive.get('delta_hv_after',pd.Series(dtype=float)),errors='coerce').to_numpy(float)/head
 adaptive=g[pd.to_numeric(g['call'],errors='coerce')>16].copy();guided=_bool(adaptive['guided']) if len(adaptive) and 'guided' in adaptive.columns else pd.Series(False,index=adaptive.index);fallback=adaptive['mode'].astype(str).str.contains('FALLBACK|FEASIBILITY',case=False,regex=True) if len(adaptive) and 'mode' in adaptive.columns else pd.Series(False,index=adaptive.index)
 skipped=_bool(g['exact_skipped']) if 'exact_skipped' in g.columns else pd.Series(False,index=g.index)
 evaluated=~skipped
 denom=int(evaluated.sum())
 feasible_fraction=(float((_bool(g['exact_feasible']) & evaluated).sum())/denom) if denom else np.nan
 return {'checkpoint':int(cp),'calls':int(cp),'evaluation_mode':mode,'baseline_hv':baseline,'landscape_prior_hv_reference':float(landscape_hv),'final_archive_hv':primary,'final_primary_hv':primary,'final_self_hv':self_hv,'final_prior_union_hv':union_hv,'final_delta_hv':delta,'headroom_normalized_score':float(delta/head),'adaptive_auc_headroom_score':float(np.nanmean(adaptive_score)) if len(adaptive_score) else 0.0,'exact_feasible_fraction':feasible_fraction,'exact_skipped':int(skipped.sum()),'exact_successful_calls':denom,'exact_extensions':int(ext_seq.sum()),'physical_exact_calls':int((~_bool(g['cache_hit'])).sum()),'cache_hits':int(_bool(g['cache_hit']).sum()),'guided_fraction_adaptive':float(guided.mean()) if len(guided) else 0.0,'guided_calls_adaptive':int(guided.sum()) if len(guided) else 0,'fallback_fraction_adaptive':float(fallback.mean()) if len(fallback) else 0.0,'mean_guided_candidate_distance':_num_mean(adaptive,'candidate_distance_from_parent',guided),'mean_attempted_candidate_distance':_num_mean(adaptive,'candidate_distance_from_parent'),'mean_guided_minus_parent_acq':_num_mean(adaptive,'guided_minus_parent_acq',guided),'mean_attempted_minus_parent_acq':_num_mean(adaptive,'guided_minus_parent_acq'),'mean_guided_minus_parent_gate':_num_mean(adaptive,'guided_minus_parent_gate',guided),'mean_attempted_minus_parent_gate':_num_mean(adaptive,'guided_minus_parent_gate'),'guided_exact_hv_gain':float(pd.to_numeric(adaptive.loc[guided,'exact_hv_gain_this_call'],errors='coerce').fillna(0).sum()) if (not is_no_prior) and len(adaptive) and 'exact_hv_gain_this_call' in adaptive.columns else np.nan,'unguided_adaptive_exact_hv_gain':float(pd.to_numeric(adaptive.loc[~guided,'exact_hv_gain_this_call'],errors='coerce').fillna(0).sum()) if (not is_no_prior) and len(adaptive) and 'exact_hv_gain_this_call' in adaptive.columns else np.nan,'guided_front_extensions':int(_bool(adaptive.loc[guided,'exact_front_extension']).sum()) if (not is_no_prior) and len(adaptive) and 'exact_front_extension' in adaptive.columns else np.nan,'adaptive_front_extensions':int(ext_seq[pd.to_numeric(g['call'],errors='coerce').to_numpy()>16].sum()),'mean_exact_hv_gain_per_call':float(np.mean(gain_seq)) if len(gain_seq) else 0.0,'heuristic_informative_checkpoint':bool(int(cp)>16),'objective_dim':int(objective_dim),'no_prior_self_front_correction_applied':bool(is_no_prior)}

def safe_relative(diff,baseline,eps=1e-6):
 return float(diff/baseline) if np.isfinite(baseline) and abs(baseline)>eps else np.nan

def main():
 validate_selected_panel();ap=argparse.ArgumentParser();ap.add_argument('--study-config',required=True);a=ap.parse_args();sp=Path(a.study_config).resolve();cfg=yaml.safe_load(sp.read_text()) or {};methods=list(cfg.get('methods_primary') or [])
 if methods!=SELECTED_METHODS:raise SystemExit(f'SELECTED METHOD PANEL CONTRACT FAIL: {methods}')
 worker=str(cfg['worker']);outroot=Path(cfg['paths']['output_root']);rows=[];failures=[]
 for case,c in cfg['cases'].items():
  d=outroot/case/f"seed_{cfg['study']['seed']}";mf=d/'run_manifest.json'
  if not mf.exists():failures.append({'case':case,'reason':'missing run_manifest'});continue
  man=json.loads(mf.read_text());landscape_hv=float(man.get('prior_hv_exact',man.get('runtime_landscape_union_baseline_hv',np.nan)))
  if man.get('status')!='COMPLETE':failures.append({'case':case,'reason':f"run status={man.get('status')} errors={man.get('errors')}"})
  if not bool(man.get('common_initialization_identity_pass',False)):failures.append({'case':case,'reason':'matched common initialization identity FAIL'})
  prep_cfg=yaml.safe_load((Path(cfg['paths']['prepared_root'])/case/'problem_config.yaml').read_text()) or {};cmeta=prep_cfg['problem']['compiled_problem'];objcols=list(cmeta['objective_norm_columns']);ref=tuple(float(x) for x in cmeta.get('hv_reference_normalized',[0.0]*len(objcols)));objective_dim=int(cmeta['objective_dim'])
  summary=pd.read_csv(d/'summary.csv') if (d/'summary.csv').exists() else pd.DataFrame();prior_info=json.loads((d/'representative_prior.json').read_text()) if (d/'representative_prior.json').exists() else {};ledger=pd.read_csv(d/'exact_runtime_ledger.csv') if (d/'exact_runtime_ledger.csv').exists() else pd.DataFrame()
  for m in methods:
   cp=d/m/'calls.csv'
   if not cp.exists():failures.append({'case':case,'method':m,'reason':'missing calls.csv'});continue
   h=pd.read_csv(cp)
   herr=_validate_history(h,m,int(man.get('budget',128) or 128))
   if herr:failures.append({'case':case,'method':m,'reason':herr});continue
   missing_obj=[x for x in objcols if x not in h.columns]
   if missing_obj:failures.append({'case':case,'method':m,'reason':f'missing normalized objective columns {missing_obj}'});continue
   for ck in CHECKPOINTS:
    rr=row_at(h,ck,landscape_hv,objective_dim,objcols,ref,m==BASE_NO50K)
    if rr is None:failures.append({'case':case,'method':m,'checkpoint':int(ck),'reason':'checkpoint history incomplete or noncontiguous'});continue
    typ,heur,lineage,desc=METHOD_META[m];rr.update({'worker':worker,'package_version':str((cfg.get('provenance') or {}).get('package_version',cfg.get('version',''))),'job_id':str(c.get('job_id',man.get('job_id',''))),'experiment_row_id':f"{c.get('job_id',man.get('job_id',''))}::{m}::CP{ck}",'case_key':case,'species':c['species'],'problem_key':c['problem_key'],'problem_id':man.get('problem_id'),'problem_hash':man.get('problem_hash'),'objective_names':'|'.join(map(str,cmeta.get('objective_names',[]))),'objective_directions':'|'.join(map(str,cmeta.get('objective_directions',[]))),'method':m,'method_type':typ,'heuristic':heur,'implementation_lineage':lineage,'implementation_description':desc,'seed':int(cfg['study']['seed']),'run_status':man.get('status',''),'common_initialization_identity_pass':bool(man.get('common_initialization_identity_pass',False)),'representative_prior_rows':int(prior_info.get('rows',0)) if m!=BASE_NO50K else 0,'landscape_prior_rows':int(cfg['study']['landscape_prior_rows']) if m!=BASE_NO50K else 0,'prior_label':str(cfg['study']['prior_label']) if m!=BASE_NO50K else 'NONE','reference_semantics_rows':int(cfg['study'].get('reference_semantics_rows',50000)),'full50k_reference_hv':man.get('full50k_reference_hv'),'prior_hv_exact':landscape_hv if m!=BASE_NO50K else 0.0,'method_display':(m.replace('50K',str(cfg['study']['prior_label'])) if m!=BASE_NO50K else m),'error':str((man.get('errors') or {}).get(m,''))})
    batch_max=int(np.ceil(int(ck)/int(cfg['study']['batch_size'])))
    if len(ledger):
     lm=ledger[(ledger['method'].astype(str)==m)&(pd.to_numeric(ledger['batch'],errors='coerce')<=batch_max)];rr['physical_fba_solves']=int(pd.to_numeric(lm.get('physical_fba_solves',pd.Series(dtype=float)),errors='coerce').fillna(0).sum());rr['physical_exact_wall_seconds']=float(pd.to_numeric(lm.get('physical_exact_wall_seconds',pd.Series(dtype=float)),errors='coerce').fillna(0).sum());rr['exact_bridge_wall_seconds']=float(pd.to_numeric(lm.get('exact_bridge_wall_seconds',pd.Series(dtype=float)),errors='coerce').fillna(0).sum())
    else:rr['physical_fba_solves']=np.nan;rr['physical_exact_wall_seconds']=np.nan;rr['exact_bridge_wall_seconds']=np.nan
    if ck==128 and len(summary) and m in set(summary.get('method',pd.Series(dtype=str)).astype(str)):
     sr=summary.loc[summary.method.astype(str)==m].iloc[0];rr['runtime_seconds']=float(pd.to_numeric(sr.get('runtime_seconds',np.nan),errors='coerce'));rr['optimizer_overhead_seconds']=max(0.0,rr['runtime_seconds']-rr['physical_exact_wall_seconds']) if np.isfinite(rr['runtime_seconds']) and np.isfinite(rr['physical_exact_wall_seconds']) else np.nan
    else:rr['runtime_seconds']=np.nan;rr['optimizer_overhead_seconds']=np.nan
    rows.append(rr)
 df=pd.DataFrame(rows)
 if df.empty:raise SystemExit('No completed result rows found; consolidated CSV not written')
 for col,default in [('no_prior_score',np.nan),('plus_prior_score',np.nan),('knowledge_gain_prior_vs_no_prior',np.nan),('gain_vs_no_prior',np.nan),('gain_vs_plus_prior',np.nan),('relative_gain_vs_no_prior',np.nan),('relative_gain_vs_plus_prior',np.nan),('baseline_relative_reward',np.nan),('baseline_prior_beats_no_prior',False),('beats_no_prior',False),('beats_plus_prior',False),('selected_heuristic_rank',np.nan),('selected_heuristic_winner',False),('baseline_order_status','')]:df[col]=default
 for (case,ck),idx in df.groupby(['case_key','checkpoint']).groups.items():
  sub=df.loc[idx];mp={r.method:float(r.headroom_normalized_score) for r in sub.itertuples()}
  if BASE_NO50K not in mp or BASE_50K not in mp:failures.append({'case':case,'checkpoint':int(ck),'reason':'missing scientific reference baseline'});continue
  s0,sprior=mp[BASE_NO50K],mp[BASE_50K];ranked=sorted([(mp[m],m) for m in SELECTED_HEURISTICS if m in mp],reverse=True);rankmap={m:i+1 for i,(_,m) in enumerate(ranked)}
  for i in idx:
   s=float(df.at[i,'headroom_normalized_score']);m=str(df.at[i,'method']);df.at[i,'no_prior_score']=s0;df.at[i,'plus_prior_score']=sprior;df.at[i,'knowledge_gain_prior_vs_no_prior']=sprior-s0;df.at[i,'baseline_prior_beats_no_prior']=bool(sprior>s0+1e-12);g0=s-s0;df.at[i,'gain_vs_no_prior']=g0;df.at[i,'relative_gain_vs_no_prior']=safe_relative(g0,s0)
   if m in SELECTED_HEURISTICS:
    gp=s-sprior;df.at[i,'gain_vs_plus_prior']=gp;df.at[i,'relative_gain_vs_plus_prior']=safe_relative(gp,sprior);df.at[i,'beats_no_prior']=bool(g0>1e-12);df.at[i,'beats_plus_prior']=bool(gp>1e-12);df.at[i,'baseline_relative_reward']=.60*gp+.20*g0+.20*min(gp,g0);df.at[i,'selected_heuristic_rank']=rankmap[m];df.at[i,'selected_heuristic_winner']=bool(rankmap[m]==1);df.at[i,'baseline_order_status']='FULL_SUCCESS' if gp>1e-12 and sprior>s0+1e-12 else ('HEURISTIC_SUCCESS_BASELINE_FAILURE' if g0>1e-12 and not sprior>s0+1e-12 else ('HEURISTIC_REGRESSION' if gp<-1e-12 else 'TIE_OR_NEUTRAL'))
   elif m==BASE_50K:df.at[i,'baseline_relative_reward']=sprior-s0;df.at[i,'baseline_order_status']='BASELINE_PRIOR_SUCCESS' if sprior>s0+1e-12 else ('BASELINE_PRIOR_REGRESSION' if sprior<s0-1e-12 else 'BASELINE_PRIOR_TIE')
   else:df.at[i,'baseline_order_status']='NO_PRIOR_SELF_FRONT_REFERENCE'
 # Backward-compatible legacy aliases. These refer to the CURRENT prior size, not 50K.
 df['plus50k_score']=df['plus_prior_score'];df['knowledge_gain_50k_vs_no50k']=df['knowledge_gain_prior_vs_no_prior'];df['gain_vs_plus50k']=df['gain_vs_plus_prior'];df['gain_vs_no50k']=df['gain_vs_no_prior'];df['relative_gain_vs_plus50k']=df['relative_gain_vs_plus_prior'];df['relative_gain_vs_no50k']=df['relative_gain_vs_no_prior'];df['baseline_50k_beats_no50k']=df['baseline_prior_beats_no_prior'];df['prior_hv_fraction_of_full50k']=pd.to_numeric(df['landscape_prior_hv_reference'],errors='coerce')/pd.to_numeric(df['full50k_reference_hv'],errors='coerce').replace(0,np.nan)
 expected_rows=len(cfg['cases'])*len(methods)*len(CHECKPOINTS)
 if len(df)!=expected_rows:failures.append({'reason':f'consolidated row count={len(df)} expected={expected_rows}'})
 if df[['case_key','method','checkpoint']].duplicated().any():failures.append({'reason':'duplicate case/method/checkpoint rows in aggregation'})
 df=df.sort_values(['case_key','checkpoint','method']).reset_index(drop=True);prior_tag=str(cfg['study']['prior_tag']);final=Path(__file__).resolve().parent/f'ALL_RESULTS_{worker}_{prior_tag}_SELECTED3.csv';df.to_csv(final,index=False)
 endpoint=df[df.checkpoint==128].copy();heur=endpoint[endpoint.method.isin(SELECTED_HEURISTICS)];win=heur[heur.selected_heuristic_winner.astype(bool)];summary_obj={'worker':worker,'prior_label':str(cfg['study']['prior_label']),'landscape_prior_rows':int(cfg['study']['landscape_prior_rows']),'assigned_cells':len(cfg['cases']),'completed_cells':int(endpoint.case_key.nunique()),'methods_per_cell':len(methods),'scientific_reference_methods':[BASE_NO50K,BASE_50K],'selected_heuristics':SELECTED_HEURISTICS,'no_prior_scoring_contract':'LIVE_SELF_FRONT_ONLY_NO_LANDSCAPE_UNION','selected_heuristic_wins_at_128':{m:int((win.method==m).sum()) for m in SELECTED_HEURISTICS},'heuristic_beats_plus_prior_rows_128':int(heur.beats_plus_prior.astype(bool).sum()),'heuristic_beats_no_prior_rows_128':int(heur.beats_no_prior.astype(bool).sum()),'failures':failures,'csv':str(final)};(Path(__file__).resolve().parent/f'WORKER_SUMMARY_{worker}_{prior_tag}.json').write_text(json.dumps(summary_obj,indent=2),encoding='utf-8');print(json.dumps(summary_obj,indent=2));print(f'CONSOLIDATED CSV: {final}')
 if failures:raise SystemExit('Worker aggregation detected QC/completeness failures; CSV was written for diagnosis but worker must not be treated as PASS')
if __name__=='__main__':main()
