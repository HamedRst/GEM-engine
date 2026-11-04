#!/usr/bin/env python3
from __future__ import annotations

"""Create the one-file worker result requested for systematic development.

One row = method x species x problem x checkpoint x seed.
The CSV repeats matched baselines on every row and derives explicit gains/reward.
No percent effect is reported when the denominator is near zero; absolute
headroom-normalized differences remain the primary cross-cell development score.
"""

import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from nsga_algorithms import (
    BASE_NO50K, BASE_50K, METHODS, HEURISTIC, LINEAGE,
    GHOST_LEG,SPARC_LEG,TRAIL_LEG,GHOST_V111,SPARC_V111,TRAIL_V111,GHOST_A,SPARC_A,TRAIL_A,
)

CHECKPOINTS=[16,32,64,128]
LINEAGE_METHODS={
 'LEGACY_OPERATOR':[GHOST_LEG,SPARC_LEG,TRAIL_LEG],
 'TRUE_NATIVE_V111':[GHOST_V111,SPARC_V111,TRAIL_V111],
 'SYSTEMATIC_A':[GHOST_A,SPARC_A,TRAIL_A],
}
METHOD_META={
 BASE_NO50K:('baseline','no_prior','BASELINE','native live-only NSGA-II'),
 BASE_50K:('baseline','prior_N','BASELINE','exact N-row prior warm-start + native NSGA-II; legacy method ID retained for historical compatibility'),
 GHOST_LEG:('heuristic','ghost','LEGACY_OPERATOR','historical NSGA operator: inherited offspring + structural replacement gate'),
 SPARC_LEG:('heuristic','sparc','LEGACY_OPERATOR','historical NSGA operator: inherited offspring + structural replacement gate'),
 TRAIL_LEG:('heuristic','trail','LEGACY_OPERATOR','historical NSGA operator: inherited offspring + structural replacement gate'),
 GHOST_V111:('heuristic','ghost','TRUE_NATIVE_V111','v1.1.1 NSGA 50% guided-offspring mix without universal post-gate'),
 SPARC_V111:('heuristic','sparc','TRUE_NATIVE_V111','v1.1.1 NSGA 50% guided-offspring mix without universal post-gate'),
 TRAIL_V111:('heuristic','trail','TRUE_NATIVE_V111','v1.1.1 NSGA 50% guided-offspring mix without universal post-gate'),
 GHOST_A:('heuristic','ghost','SYSTEMATIC_A','new parent-protected blind-spot guided offspring'),
 SPARC_A:('heuristic','sparc','SYSTEMATIC_A','new parent-protected productive-subspace guided offspring'),
 TRAIL_A:('heuristic','trail','SYSTEMATIC_A','new parent-protected trajectory-continuation guided offspring'),
}


def safe_relative(diff, baseline, eps=1e-6):
    baseline=float(baseline); diff=float(diff)
    return float(diff/baseline) if np.isfinite(baseline) and abs(baseline)>eps else np.nan


def classify(h,s50,s0,tol=1e-12):
    if h>s50+tol and s50>s0+tol: return 'FULL_SUCCESS'
    if h>s0+tol and not (s50>s0+tol): return 'HEURISTIC_SUCCESS_BASELINE_FAILURE'
    if h<s50-tol and h<=s0+tol and s50<=s0+tol: return 'TOTAL_REGRESSION'
    if h<s50-tol: return 'HEURISTIC_REGRESSION'
    return 'TIE_OR_NEUTRAL'


def _bool_series(s):
    if pd.api.types.is_bool_dtype(s): return s.astype(bool)
    return s.astype(str).str.lower().isin(['true','1','yes'])


def _num_mean(df,col,mask=None):
    if col not in df.columns:return np.nan
    x=pd.to_numeric(df[col],errors='coerce')
    if mask is not None:x=x[mask]
    return float(x.mean()) if x.notna().any() else np.nan


def row_at(h,cp,baseline_hv,objective_dim):
    calls=pd.to_numeric(h['call'],errors='coerce')
    g=h[calls<=int(cp)].copy().sort_values('call')
    if len(g)<int(cp): return None
    r=g.iloc[-1]
    delta=float(pd.to_numeric(r['delta_hv_after'],errors='coerce'))
    # Normalized objectives have reference zero and frozen archive inside the unit
    # hypercube.  1-baselineHV is the remaining frozen-range HV headroom.  Scores
    # may exceed 1 if a live discovery exceeds a frozen normalization maximum.
    head=max(1e-12,1.0-float(baseline_hv))
    score=delta/head
    adaptive=g[pd.to_numeric(g['call'],errors='coerce')>16].copy()
    adaptive_score=pd.to_numeric(adaptive.get('delta_hv_after',pd.Series(dtype=float)),errors='coerce')/head
    guided=_bool_series(adaptive['guided']) if len(adaptive) and 'guided' in adaptive.columns else pd.Series(False,index=adaptive.index)
    accepted=guided
    fallback=adaptive['mode'].astype(str).str.contains('FALLBACK|FEASIBILITY',case=False,regex=True) if len(adaptive) and 'mode' in adaptive.columns else pd.Series(False,index=adaptive.index)
    return {
      'checkpoint':int(cp),'calls':int(cp),'final_delta_hv':delta,'baseline_hv':float(baseline_hv),
      'final_archive_hv':float(pd.to_numeric(r['archive_hv_after'],errors='coerce')),
      'headroom_normalized_score':float(score),
      'adaptive_auc_headroom_score':float(adaptive_score.mean()) if len(adaptive_score) else 0.0,
      'exact_feasible_fraction':float(_bool_series(g['exact_feasible']).mean()),
      'exact_extensions':int(_bool_series(g['exact_front_extension']).sum()),
      'physical_exact_calls':int((~_bool_series(g['cache_hit'])).sum()),
      'cache_hits':int(_bool_series(g['cache_hit']).sum()),
      'guided_fraction_adaptive':float(guided.mean()) if len(guided) else 0.0,
      'guided_calls_adaptive':int(guided.sum()) if len(guided) else 0,
      'fallback_fraction_adaptive':float(fallback.mean()) if len(fallback) else 0.0,
      'mean_guided_candidate_distance':_num_mean(adaptive,'candidate_distance_from_parent',accepted),
      'mean_attempted_candidate_distance':_num_mean(adaptive,'candidate_distance_from_parent'),
      'mean_guided_minus_parent_acq':_num_mean(adaptive,'guided_minus_parent_acq',accepted),
      'mean_attempted_minus_parent_acq':_num_mean(adaptive,'guided_minus_parent_acq'),
      'mean_guided_minus_parent_gate':_num_mean(adaptive,'guided_minus_parent_gate',accepted),
      'mean_attempted_minus_parent_gate':_num_mean(adaptive,'guided_minus_parent_gate'),
      'guided_exact_hv_gain':float(pd.to_numeric(adaptive.loc[guided,'exact_hv_gain_this_call'],errors='coerce').fillna(0).sum()) if len(adaptive) and 'exact_hv_gain_this_call' in adaptive.columns else 0.0,
      'unguided_adaptive_exact_hv_gain':float(pd.to_numeric(adaptive.loc[~guided,'exact_hv_gain_this_call'],errors='coerce').fillna(0).sum()) if len(adaptive) and 'exact_hv_gain_this_call' in adaptive.columns else 0.0,
      'guided_front_extensions':int(_bool_series(adaptive.loc[guided,'exact_front_extension']).sum()) if len(adaptive) and 'exact_front_extension' in adaptive.columns else 0,
      'adaptive_front_extensions':int(_bool_series(adaptive['exact_front_extension']).sum()) if len(adaptive) and 'exact_front_extension' in adaptive.columns else 0,
      'mean_exact_hv_gain_per_call':_num_mean(g,'exact_hv_gain_this_call'),
      'heuristic_informative_checkpoint':bool(int(cp)>16),
      'objective_dim':int(objective_dim),
    }


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study-config',required=True);a=ap.parse_args()
    sp=Path(a.study_config).resolve();cfg=yaml.safe_load(sp.read_text()) or {};worker=str(cfg['worker']);outroot=Path(cfg['paths']['output_root'])
    rows=[];failures=[]
    for case,c in cfg['cases'].items():
        d=outroot/case/f"seed_{cfg['study']['seed']}";mf=d/'run_manifest.json'
        if not mf.exists(): failures.append({'case':case,'reason':'missing run_manifest'});continue
        man=json.loads(mf.read_text());baseline_hv=float(man.get('baseline_hv',np.nan))
        if man.get('status')!='COMPLETE': failures.append({'case':case,'reason':f"run status={man.get('status')} errors={man.get('errors')}"})
        if not bool(man.get('common_initialization_identity_pass',False)): failures.append({'case':case,'reason':'matched common initialization identity FAIL'})
        summary=pd.read_csv(d/'summary.csv') if (d/'summary.csv').exists() else pd.DataFrame()
        objective_dim=None
        if len(summary) and 'objective_dim' in summary.columns:
            vv=pd.to_numeric(summary['objective_dim'],errors='coerce').dropna();objective_dim=int(vv.iloc[0]) if len(vv) else None
        if objective_dim is None:
            prep=Path(cfg['paths']['prepared_root'])/case/'problem_config.yaml';pc=yaml.safe_load(prep.read_text()) or {};objective_dim=int(pc['problem']['compiled_problem']['objective_dim'])
        prior_info={}
        pr=d/'representative_prior.json'
        if pr.exists():prior_info=json.loads(pr.read_text())
        ledger_path=d/'exact_runtime_ledger.csv'; ledger=pd.read_csv(ledger_path) if ledger_path.exists() else pd.DataFrame()
        prep_cfg=yaml.safe_load((Path(cfg['paths']['prepared_root'])/case/'problem_config.yaml').read_text()) or {}
        cmeta=prep_cfg['problem']['compiled_problem']
        objective_names='|'.join(map(str,cmeta.get('objective_names',[])))
        objective_directions='|'.join(map(str,cmeta.get('objective_directions',[])))
        stochastic_indices='|'.join(map(str,cmeta.get('stochastic_objective_indices',[])))
        deterministic_indices='|'.join(map(str,cmeta.get('deterministic_objective_indices',[])))
        for m in METHODS:
            cp=d/m/'calls.csv'
            if not cp.exists(): failures.append({'case':case,'method':m,'reason':'missing calls.csv'});continue
            h=pd.read_csv(cp)
            for ck in CHECKPOINTS:
                rr=row_at(h,ck,baseline_hv,objective_dim)
                if rr is None: continue
                typ,heur,lineage,desc=METHOD_META[m]
                rr.update({
                  'worker':worker,'package_version':str((cfg.get('provenance') or {}).get('package_version',cfg.get('version',''))),'worker_manifest_sha256':str((cfg.get('provenance') or {}).get('worker_manifest_sha256','')),'package_hash_manifest_sha256':str((cfg.get('provenance') or {}).get('package_hash_manifest_sha256','')),'core_source_fingerprint':str((cfg.get('provenance') or {}).get('core_source_fingerprint','')),'job_id':str(c.get('job_id',man.get('job_id',''))),'experiment_row_id':f"{c.get('job_id',man.get('job_id',''))}::{m}::CP{ck}",'case_key':case,'species':c['species'],'problem_key':c['problem_key'],
                  'problem_id':man.get('problem_id'),'problem_hash':man.get('problem_hash'),'objective_names':objective_names,'objective_directions':objective_directions,'stochastic_objective_indices':stochastic_indices,'deterministic_objective_indices':deterministic_indices,'method':m,
                  'method_type':typ,'heuristic':heur,'implementation_lineage':lineage,'implementation_description':desc,
                  'seed':int(cfg['study']['seed']),'run_status':man.get('status',''),
                  'common_initialization_identity_pass':bool(man.get('common_initialization_identity_pass',False)),
                  'representative_prior_rows':int(prior_info.get('rows',0)) if m!=BASE_NO50K else 0,
                  'representative_prior_infeasible_rows':int(prior_info.get('archive_infeasible_rows',0)) if m!=BASE_NO50K else 0,
                  'error':str((man.get('errors') or {}).get(m,'')),
                  'source_benchmark_name':str((man.get('source_provenance') or {}).get('source_benchmark_name','')),
                  'source_case':str((man.get('source_provenance') or {}).get('source_case','')),
                  'source_baseline_sha256':str((man.get('source_provenance') or {}).get('source_baseline_sha256','')),
                  'source_problem_sha256':str((man.get('source_provenance') or {}).get('source_problem_sha256','')),
                  'source_engine_bridge_original_sha256':str((man.get('source_provenance') or {}).get('source_engine_bridge_original_sha256','')),
                  'source_engine_bridge_runtime_sha256':str((man.get('source_provenance') or {}).get('source_engine_bridge_runtime_sha256','')),
                })
                batch_max=int(np.ceil(int(ck)/int(cfg['study']['batch_size'])))
                if len(ledger):
                    lm=ledger[(ledger['method'].astype(str)==m)&(pd.to_numeric(ledger['batch'],errors='coerce')<=batch_max)]
                    rr['physical_fba_solves']=int(pd.to_numeric(lm.get('physical_fba_solves',pd.Series(dtype=float)),errors='coerce').fillna(0).sum())
                    rr['physical_exact_wall_seconds']=float(pd.to_numeric(lm.get('physical_exact_wall_seconds',pd.Series(dtype=float)),errors='coerce').fillna(0).sum())
                    rr['exact_bridge_wall_seconds']=float(pd.to_numeric(lm.get('exact_bridge_wall_seconds',pd.Series(dtype=float)),errors='coerce').fillna(0).sum())
                    rr['cell_cache_hits_ledger']=int(pd.to_numeric(lm.get('cell_cache_hits',pd.Series(dtype=float)),errors='coerce').fillna(0).sum())
                    rr['source_raw_cache_hits']=int(pd.to_numeric(lm.get('source_raw_cache_hits',pd.Series(dtype=float)),errors='coerce').fillna(0).sum())
                else:
                    rr['physical_fba_solves']=np.nan; rr['physical_exact_wall_seconds']=np.nan; rr['exact_bridge_wall_seconds']=np.nan; rr['cell_cache_hits_ledger']=np.nan; rr['source_raw_cache_hits']=np.nan
                if ck==128 and len(summary) and m in set(summary.get('method',pd.Series(dtype=str)).astype(str)):
                    sr=summary.loc[summary.method.astype(str)==m].iloc[0]
                    rr['runtime_seconds']=float(pd.to_numeric(sr.get('runtime_seconds',np.nan),errors='coerce'))
                    rr['optimizer_overhead_seconds']=max(0.0,rr['runtime_seconds']-rr['physical_exact_wall_seconds']) if np.isfinite(rr['runtime_seconds']) and np.isfinite(rr['physical_exact_wall_seconds']) else np.nan
                else:
                    rr['runtime_seconds']=np.nan; rr['optimizer_overhead_seconds']=np.nan
                rows.append(rr)
    df=pd.DataFrame(rows)
    if len(df):
        df['landscape_prior_rows']=int(cfg['study']['landscape_prior_rows'])
        df['reference_semantics_rows']=int(cfg['study']['reference_semantics_rows'])
    if df.empty: raise SystemExit('No completed result rows found; consolidated CSV not written')

    # Repeated baseline-relative fields make every uploaded row self-contained.
    defaults={
      'no50k_score':np.nan,'plus50k_score':np.nan,'knowledge_gain_50k_vs_no50k':np.nan,
      'gain_vs_no50k':np.nan,'gain_vs_plus50k':np.nan,'relative_gain_vs_no50k':np.nan,
      'relative_gain_vs_plus50k':np.nan,'reward_component_vs_plus50k':np.nan,
      'reward_component_vs_no50k':np.nan,'reward_component_consistency':np.nan,
      'baseline_relative_reward':np.nan,'baseline_50k_beats_no50k':False,'baseline_order_status':'',
      'lineage_median_heuristic_score':np.nan,'lineage_median_gain_vs_plus50k':np.nan,
      'lineage_median_reward':np.nan,'lineage_heuristics_beating_plus50k_count':np.nan,
      'lineage_heuristics_beating_no50k_count':np.nan,
      'legacy_operator_median_gain_vs_plus50k':np.nan,'true_native_v111_median_gain_vs_plus50k':np.nan,
      'systematic_a_median_gain_vs_plus50k':np.nan,
      'best_implementation_score_for_heuristic':np.nan,'best_implementation_gain_vs_plus50k':np.nan,
      'best_implementation_lineage_for_heuristic':'',
      'best_per_heuristic_median_score':np.nan,'best_per_heuristic_median_gain_vs_plus50k':np.nan,
    }
    for k,v in defaults.items():df[k]=v

    for (case,ck),idx in df.groupby(['case_key','checkpoint']).groups.items():
        sub=df.loc[idx];mp={r.method:float(r.headroom_normalized_score) for r in sub.itertuples()}
        if BASE_NO50K not in mp or BASE_50K not in mp:continue
        s0,s50=mp[BASE_NO50K],mp[BASE_50K];kg=s50-s0
        lineage_stats={}
        for lin,methods in LINEAGE_METHODS.items():
            vals=[mp[m] for m in methods if m in mp]
            if len(vals)==3:
                med=float(np.median(vals));r=.60*(med-s50)+.20*(med-s0)+.20*min(med-s50,med-s0)
                lineage_stats[lin]={'median':med,'gain50':med-s50,'reward':r,'beat50':int(sum(v>s50+1e-12 for v in vals)),'beat0':int(sum(v>s0+1e-12 for v in vals))}
        best_by_h={}
        for heur in ('ghost','sparc','trail'):
            cand=[]
            for m in METHODS:
                if HEURISTIC.get(m)==heur and m in mp:cand.append((mp[m],LINEAGE.get(m,''),m))
            if cand:
                cand.sort(key=lambda z:(z[0],z[1],z[2]),reverse=True);best_by_h[heur]=cand[0]
        best_vals=[best_by_h[h][0] for h in ('ghost','sparc','trail') if h in best_by_h]
        best_med=float(np.median(best_vals)) if len(best_vals)==3 else np.nan
        for i in idx:
            s=float(df.at[i,'headroom_normalized_score']);m=str(df.at[i,'method']);lin=str(df.at[i,'implementation_lineage']);heur=str(df.at[i,'heuristic'])
            df.at[i,'no50k_score']=s0;df.at[i,'plus50k_score']=s50;df.at[i,'knowledge_gain_50k_vs_no50k']=kg
            df.at[i,'baseline_50k_beats_no50k']=bool(s50>s0+1e-12)
            if 'LEGACY_OPERATOR' in lineage_stats:df.at[i,'legacy_operator_median_gain_vs_plus50k']=lineage_stats['LEGACY_OPERATOR']['gain50']
            if 'TRUE_NATIVE_V111' in lineage_stats:df.at[i,'true_native_v111_median_gain_vs_plus50k']=lineage_stats['TRUE_NATIVE_V111']['gain50']
            if 'SYSTEMATIC_A' in lineage_stats:df.at[i,'systematic_a_median_gain_vs_plus50k']=lineage_stats['SYSTEMATIC_A']['gain50']
            df.at[i,'best_per_heuristic_median_score']=best_med
            df.at[i,'best_per_heuristic_median_gain_vs_plus50k']=best_med-s50 if np.isfinite(best_med) else np.nan
            if lin in lineage_stats:
                st=lineage_stats[lin];df.at[i,'lineage_median_heuristic_score']=st['median'];df.at[i,'lineage_median_gain_vs_plus50k']=st['gain50'];df.at[i,'lineage_median_reward']=st['reward'];df.at[i,'lineage_heuristics_beating_plus50k_count']=st['beat50'];df.at[i,'lineage_heuristics_beating_no50k_count']=st['beat0']
            if heur in best_by_h:
                bs,bl,_=best_by_h[heur];df.at[i,'best_implementation_score_for_heuristic']=bs;df.at[i,'best_implementation_gain_vs_plus50k']=bs-s50;df.at[i,'best_implementation_lineage_for_heuristic']=bl
            g0=s-s0;df.at[i,'gain_vs_no50k']=g0;df.at[i,'relative_gain_vs_no50k']=safe_relative(g0,s0)
            if m in HEURISTIC:
                g50=s-s50;c50=.60*g50;c0=.20*g0;cc=.20*min(g50,g0)
                df.at[i,'gain_vs_plus50k']=g50;df.at[i,'relative_gain_vs_plus50k']=safe_relative(g50,s50)
                df.at[i,'reward_component_vs_plus50k']=c50;df.at[i,'reward_component_vs_no50k']=c0;df.at[i,'reward_component_consistency']=cc
                df.at[i,'baseline_relative_reward']=c50+c0+cc;df.at[i,'baseline_order_status']=classify(s,s50,s0)
            elif m==BASE_50K:
                df.at[i,'baseline_relative_reward']=kg
                df.at[i,'baseline_order_status']='BASELINE_50K_SUCCESS' if s50>s0+1e-12 else ('BASELINE_50K_REGRESSION' if s50<s0-1e-12 else 'BASELINE_50K_TIE')
            else:df.at[i,'baseline_order_status']='NO50K_REFERENCE'


    # Parametric-prior aliases. Legacy *50k* column names are retained only for backward-compatible table joins.
    df['no_prior_score']=df['no50k_score']
    df['prior_N_score']=df['plus50k_score']
    df['knowledge_gain_priorN_vs_no_prior']=df['knowledge_gain_50k_vs_no50k']
    df['gain_vs_no_prior']=df['gain_vs_no50k']
    df['gain_vs_prior_N']=df['gain_vs_plus50k']
    df['relative_gain_vs_no_prior']=df['relative_gain_vs_no50k']
    df['relative_gain_vs_prior_N']=df['relative_gain_vs_plus50k']
    df['baseline_priorN_beats_no_prior']=df['baseline_50k_beats_no50k']
    df['lineage_median_gain_vs_prior_N']=df['lineage_median_gain_vs_plus50k']
    df['best_implementation_gain_vs_prior_N']=df['best_implementation_gain_vs_plus50k']
    df['best_per_heuristic_median_gain_vs_prior_N']=df['best_per_heuristic_median_gain_vs_plus50k']

    # Stable ordering for easy upload/diff across rounds.
    order=['worker','landscape_prior_rows','reference_semantics_rows','job_id','experiment_row_id','package_version','core_source_fingerprint','worker_manifest_sha256','package_hash_manifest_sha256','case_key','species','problem_key','problem_id','problem_hash','objective_names','objective_directions','stochastic_objective_indices','deterministic_objective_indices','source_benchmark_name','source_case','source_baseline_sha256','source_problem_sha256','source_engine_bridge_original_sha256','source_engine_bridge_runtime_sha256','seed','checkpoint','calls','method','method_type','heuristic','implementation_lineage','implementation_description','run_status','error','objective_dim','baseline_hv','final_archive_hv','final_delta_hv','headroom_normalized_score','adaptive_auc_headroom_score','no_prior_score','prior_N_score','knowledge_gain_priorN_vs_no_prior','gain_vs_no_prior','gain_vs_prior_N','relative_gain_vs_no_prior','relative_gain_vs_prior_N','baseline_priorN_beats_no_prior','lineage_median_gain_vs_prior_N','best_implementation_gain_vs_prior_N','best_per_heuristic_median_gain_vs_prior_N','no50k_score','plus50k_score','knowledge_gain_50k_vs_no50k','gain_vs_no50k','gain_vs_plus50k','relative_gain_vs_no50k','relative_gain_vs_plus50k','reward_component_vs_plus50k','reward_component_vs_no50k','reward_component_consistency','baseline_relative_reward','baseline_50k_beats_no50k','baseline_order_status','lineage_median_heuristic_score','lineage_median_gain_vs_plus50k','lineage_median_reward','lineage_heuristics_beating_plus50k_count','lineage_heuristics_beating_no50k_count','legacy_operator_median_gain_vs_plus50k','true_native_v111_median_gain_vs_plus50k','systematic_a_median_gain_vs_plus50k','best_implementation_score_for_heuristic','best_implementation_gain_vs_plus50k','best_implementation_lineage_for_heuristic','best_per_heuristic_median_score','best_per_heuristic_median_gain_vs_plus50k','exact_feasible_fraction','exact_extensions','mean_exact_hv_gain_per_call','physical_exact_calls','cache_hits','guided_fraction_adaptive','guided_calls_adaptive','guided_exact_hv_gain','unguided_adaptive_exact_hv_gain','guided_front_extensions','adaptive_front_extensions','mean_guided_candidate_distance','mean_attempted_candidate_distance','mean_guided_minus_parent_acq','mean_attempted_minus_parent_acq','mean_guided_minus_parent_gate','mean_attempted_minus_parent_gate','fallback_fraction_adaptive','physical_fba_solves','physical_exact_wall_seconds','exact_bridge_wall_seconds','cell_cache_hits_ledger','source_raw_cache_hits','optimizer_overhead_seconds','runtime_seconds','heuristic_informative_checkpoint','common_initialization_identity_pass','representative_prior_rows','representative_prior_infeasible_rows']
    order=[x for x in order if x in df.columns]+[x for x in df.columns if x not in order]
    df=df[order].sort_values(['case_key','checkpoint','method']).reset_index(drop=True)
    final=Path(__file__).resolve().parent/f'ALL_RESULTS_{worker}.csv';df.to_csv(final,index=False)

    endpoint=df[df.checkpoint==128].copy();heur=endpoint[endpoint.method.isin(list(HEURISTIC))]
    cell_first=endpoint.sort_values('method').groupby('case_key',as_index=False).first()
    lineage_summary={}
    for lin,methods in LINEAGE_METHODS.items():
        q=heur[heur.implementation_lineage.astype(str)==lin]
        lineage_summary[lin]={
            'heuristic_rows':int(len(q)),
            'wins_vs_plus50k':int((pd.to_numeric(q.gain_vs_plus50k,errors='coerce')>1e-12).sum()),
            'wins_vs_no50k':int((pd.to_numeric(q.gain_vs_no50k,errors='coerce')>1e-12).sum()),
            'median_gain_vs_plus50k_by_heuristic':{h:float(pd.to_numeric(q.loc[q.heuristic==h,'gain_vs_plus50k'],errors='coerce').median()) for h in ('ghost','sparc','trail') if len(q.loc[q.heuristic==h])},
            'median_reward_by_heuristic':{h:float(pd.to_numeric(q.loc[q.heuristic==h,'baseline_relative_reward'],errors='coerce').median()) for h in ('ghost','sparc','trail') if len(q.loc[q.heuristic==h])},
            'median_lineage_gain_vs_plus50k_across_cells':float(pd.to_numeric(q.drop_duplicates('case_key')['lineage_median_gain_vs_plus50k'],errors='coerce').median()) if len(q) else None,
        }
    summary_obj={
      'landscape_prior_rows':int(cfg['study']['landscape_prior_rows']),'reference_semantics_rows':int(cfg['study']['reference_semantics_rows']),
      'worker':worker,'assigned_cells':len(cfg['cases']),'completed_cells':int(endpoint.case_key.nunique()),'methods_per_cell':len(METHODS),
      'priorN_beats_no_prior_cells':int(cell_first['baseline_50k_beats_no50k'].astype(bool).sum()) if len(cell_first) else 0,
      'plus50k_beats_no50k_cells':int(cell_first['baseline_50k_beats_no50k'].astype(bool).sum()) if len(cell_first) else 0,
      'best_per_heuristic_median_beats_plus50k_cells':int((pd.to_numeric(cell_first['best_per_heuristic_median_gain_vs_plus50k'],errors='coerce')>1e-12).sum()) if len(cell_first) else 0,
      'heuristic_full_success_rows':int((heur.baseline_order_status=='FULL_SUCCESS').sum()),'heuristic_rows':int(len(heur)),
      'lineages':lineage_summary,
      'median_best_per_heuristic_gain_vs_plus50k':float(pd.to_numeric(cell_first['best_per_heuristic_median_gain_vs_plus50k'],errors='coerce').median()) if len(cell_first) else None,
      'failures':failures,'csv':str(final)}
    (Path(__file__).resolve().parent/f'WORKER_SUMMARY_{worker}.json').write_text(json.dumps(summary_obj,indent=2),encoding='utf-8')
    print(json.dumps(summary_obj,indent=2));print(f'CONSOLIDATED CSV: {final}')
    if failures: raise SystemExit('Worker aggregation detected QC/completeness failures; CSV was written for diagnosis but worker must not be treated as PASS')

if __name__=='__main__':main()
