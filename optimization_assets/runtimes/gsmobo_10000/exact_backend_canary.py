#!/usr/bin/env python3
from __future__ import annotations
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,yaml
from runtime import ExactRunner, normalize_design
from generic_problem import feature_bounds


def _validate_standard(case, exp, got):
    if not np.allclose(got, exp, rtol=5e-5, atol=5e-7, equal_nan=False):
        raise SystemExit(f'{case}: exact replay mismatch expected={exp.tolist()} got={got.tolist()}')
    return {
        'replay_contract':'full_objective_exact_replay',
        'compared_objective_indices':list(range(len(exp))),
    }


def _validate_robustness(case, parent_problem, exp, got, result):
    """Validate the robustness source according to its declared prior/live contract.

    The v0.2.0 environmental-robustness source deliberately builds the 50K stress
    channels from a cross-fitted nominal-growth surrogate (ZERO archive stress FBA),
    while every new live candidate is stress-evaluated by exact FBA.  Therefore the
    frozen stress objective is a PRIOR and is not expected to reproduce the live exact
    stress value at the same design.  Requiring equality would be a false canary.

    We still fail closed on the exact-replayable nominal objective and on the live
    oxygen-retention arithmetic/finite-range contract.
    """
    if parent_problem.get('prior_robustness_is_surrogate_derived') is not True:
        raise SystemExit(f'{case}: robustness canary requires source declaration prior_robustness_is_surrogate_derived=True')
    if parent_problem.get('live_robustness_is_exact_fba') is not True:
        raise SystemExit(f'{case}: robustness canary requires source declaration live_robustness_is_exact_fba=True')
    if len(exp) != 2 or len(got) != 2:
        raise SystemExit(f'{case}: robustness canary expected exactly 2 objectives, exp={exp.shape} got={got.shape}')

    # Objective 1 (nominal growth) is exact in both the frozen archive and live replay.
    if not np.allclose(got[0], exp[0], rtol=5e-5, atol=5e-7, equal_nan=False):
        raise SystemExit(
            f'{case}: nominal exact replay mismatch expected={float(exp[0])} got={float(got[0])}'
        )

    raw=result.raw.iloc[0]
    required=['maximum_growth','oxygen_stressed_growth','stress_tolerance']
    missing=[c for c in required if c not in result.raw.columns]
    if missing:
        raise SystemExit(f'{case}: robustness exact output missing required live columns {missing}')
    nominal=float(pd.to_numeric(raw['maximum_growth'],errors='coerce'))
    oxygen=float(pd.to_numeric(raw['oxygen_stressed_growth'],errors='coerce'))
    retention=float(pd.to_numeric(raw['stress_tolerance'],errors='coerce'))
    if not (np.isfinite(nominal) and nominal>1e-12 and np.isfinite(oxygen) and np.isfinite(retention)):
        raise SystemExit(f'{case}: non-finite live robustness arithmetic nominal={nominal} oxygen={oxygen} retention={retention}')
    derived=float(np.clip(oxygen/nominal,0.0,1.0+1e-8))
    if not np.isclose(retention,derived,rtol=1e-10,atol=1e-12):
        raise SystemExit(f'{case}: live oxygen-retention arithmetic mismatch reported={retention} derived={derived}')
    # Generic objective 2 uses fixed [0,1] normalization, so normalized == retention.
    if not np.isclose(float(got[1]),retention,rtol=1e-10,atol=1e-12):
        raise SystemExit(f'{case}: compiled live robustness objective mismatch got={float(got[1])} retention={retention}')
    if not (-1e-12 <= float(got[1]) <= 1.0+2e-8):
        raise SystemExit(f'{case}: live robustness retention outside declared range: {float(got[1])}')

    return {
        'replay_contract':'nominal_exact_plus_surrogate_prior_to_exact_live_robustness',
        'compared_objective_indices':[0],
        'prior_objective_2_normalized':float(exp[1]),
        'live_exact_objective_2_normalized':float(got[1]),
        'objective_2_equality_expected':False,
        'live_oxygen_stressed_growth':oxygen,
        'live_nominal_growth':nominal,
        'live_retention_rederived':derived,
    }


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--study-config',required=True);ap.add_argument('--case',required=True);a=ap.parse_args()
    sp=Path(a.study_config).resolve();cfg=yaml.safe_load(sp.read_text()) or {};case=a.case
    prep=Path(cfg['paths']['prepared_root'])/case
    parent=yaml.safe_load((prep/'problem_config.yaml').read_text()) or {}; p=parent['problem']
    df=pd.read_csv(prep/'baseline_compact.csv.gz');features,bounds=feature_bounds(p)
    good=df['feasible'].astype(bool).to_numpy()&np.isfinite(df[p['compiled_problem']['objective_norm_columns']].to_numpy(float)).all(axis=1);ids=np.where(good)[0]
    if not len(ids):raise SystemExit(f'{case}: no feasible row for exact canary')
    ii=int(ids[len(ids)//2]);x01=normalize_design(df.iloc[[ii]],features,bounds)
    out=Path(cfg['paths']['output_root'])/'_preflight_canary'/case
    er=ExactRunner(cfg,parent,out,{'prepared_dir':str(prep),'study_config_path':str(sp),'case_key':case});pf=er.preflight()
    if pf['status']!='PASS':raise SystemExit(f'{case}: exact runner preflight failed: {pf}')
    r=er.evaluate(x01,'PREFLIGHT_CANARY',1);exp=df.iloc[ii][p['compiled_problem']['objective_norm_columns']].to_numpy(float);got=r.Yn[0]
    if not bool(r.feasible[0]):raise SystemExit(f'{case}: archive-feasible exact canary returned infeasible; status={r.status[0]}')
    source_kind=str((cfg.get('cases',{}).get(case,{}) or {}).get('source_kind',''))
    if source_kind=='robustness':
        contract=_validate_robustness(case,p,exp,got,r)
    else:
        contract=_validate_standard(case,exp,got)
    rec={'status':'PASS','case':case,'archive_row':ii,'expected_normalized':exp.tolist(),'exact_normalized':got.tolist(),'exact_status':r.status[0],'physical_exact_evals':r.physical_exact_evals,**contract}
    (out/'CANARY_PASS.json').write_text(json.dumps(rec,indent=2),encoding='utf-8');print(json.dumps(rec,indent=2))
if __name__=='__main__':main()
