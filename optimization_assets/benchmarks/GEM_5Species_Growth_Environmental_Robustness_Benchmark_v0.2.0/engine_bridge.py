#!/usr/bin/env python3
from __future__ import annotations

"""Portable exact-FBA bridge and frozen-prior builder.

Scientific contract
-------------------
Objective 1: nominal maximum biomass growth.
Objective 2: worst-case maximum biomass growth across four predefined, in-domain
environmental perturbations (carbon, oxygen, nitrogen, phosphate).

The existing 50k landscape is NOT re-simulated under stress. During preparation,
a growth-only ExtraTrees surrogate is fitted to the already-computed exact nominal
growth landscape and queried at stressed versions of the same 50k designs. Those
predictions form the 50k PRIOR used by +50k/GHOST/SPARC/TRAIL methods.

Every NEW live optimizer candidate is evaluated exactly with COBRA/FBA at nominal
conditions and under every configured stress class. Thus optimizer performance is
measured with exact outcomes even though the reusable 50k stress prior is derived
from the frozen nominal landscape.

Legacy compatibility: objective 2 is mirrored into ``product_flux`` because the
frozen v0.3.0 optimizer implementations expect that column name.
"""

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import spearmanr

from stress_robustness import GrowthPriorSurrogate, derive_prior_robustness, validate_stress_groups

GROWTH_CANDIDATES = ("maximum_growth", "exact_maximum_growth", "maximum_growth_exact", "growth_max")
ID_CANDIDATES = ("sample_id", "simulation_id", "condition_id", "row_id", "id")
STATUS_CANDIDATES = ("status", "solver_status", "fba_status", "exact_status")
SUPPORTED_MODEL_SUFFIXES = {".json", ".xml", ".sbml", ".mat", ".yml", ".yaml"}


def _jdump(obj: Any, path: Path) -> None:
    def conv(x):
        if isinstance(x, Path): return str(x)
        if isinstance(x, (np.integer,)): return int(x)
        if isinstance(x, (np.floating,)): return float(x)
        if isinstance(x, np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=conv), encoding="utf-8")


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _detect_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    cols = list(columns)
    for c in candidates:
        if c in cols: return c
    low = {str(c).lower(): c for c in cols}
    for c in candidates:
        if str(c).lower() in low: return low[str(c).lower()]
    for col in cols:
        ll = str(col).lower()
        for c in candidates:
            if str(c).lower() in ll: return col
    return None


def _load_cobra_model(path: Path, solver_name: str | None = None):
    try:
        from cobra import io as cobra_io
    except Exception as exc:
        raise RuntimeError("COBRApy is required to run exact validation") from exc
    s = path.suffix.lower()
    if s == ".json": model = cobra_io.load_json_model(str(path))
    elif s in {".xml", ".sbml"}: model = cobra_io.read_sbml_model(str(path))
    elif s == ".mat": model = cobra_io.load_matlab_model(str(path))
    elif s in {".yml", ".yaml"}:
        loader = getattr(cobra_io, "load_yaml_model", None)
        if loader is None: raise RuntimeError("COBRApy has no load_yaml_model")
        model = loader(str(path))
    else: raise ValueError(f"Unsupported model format: {path}")
    if solver_name:
        model.solver = str(solver_name)
    return model


def _find_design_cache(run_dir: Path) -> Path:
    hits = sorted((run_dir / "design_cache").glob("*.npy"))
    if not hits:
        hits = sorted(run_dir.rglob("*.npy"))
    if not hits:
        raise FileNotFoundError(f"No design-cache .npy found under {run_dir}")
    return hits[0]


def _recover_bounds(run_dir: Path, cfg: dict, model, features: list[str]) -> np.ndarray:
    explicit = cfg.get("sampling", {}).get("bounds", {}) or {}
    if all(r in explicit for r in features):
        out = []
        for rid in features:
            raw = explicit[rid]
            lo, hi = (float(raw["min"]), float(raw["max"])) if isinstance(raw, dict) else (float(raw.min), float(raw.max))
            out.append([lo, hi])
        return np.asarray(out, float)

    # The run's medium-space catalog is tiny and is the preferred source for
    # automatically inferred spaces because it records the exact effective bounds.
    for p in (run_dir / "catalogs" / "medium_space.csv", run_dir / "catalogs" / "medium_space.parquet"):
        if p.exists():
            tab = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
            if {"reaction_id", "effective_min_uptake", "effective_max_uptake"}.issubset(tab.columns):
                tab = tab.set_index(tab["reaction_id"].astype(str))
                if all(r in tab.index for r in features):
                    return np.asarray([[float(tab.loc[r, "effective_min_uptake"]), float(tab.loc[r, "effective_max_uptake"])] for r in features], float)

    # Reconstruct the automatic policy used by GEM Learning Engine v1.0.4.
    # This is a guarded fallback; the exact recovered bounds are written into
    # problem_config.yaml and can be audited before live optimization.
    samp = cfg.get("sampling", {}) or {}
    min_frac = float(samp.get("auto_min_fraction", 0.0))
    max_frac = float(samp.get("auto_max_fraction", 1.0))
    cap = float(samp.get("auto_max_uptake_cap", 20.0))
    medium = {str(k): float(v) for k, v in (getattr(model, "medium", {}) or {}).items()}
    out = []
    for rid in features:
        if rid not in medium:
            raise RuntimeError(
                f"No explicit/catalog bounds and {rid!r} is absent from model.medium. "
                "Provide catalogs/medium_space.csv from the original 50k run."
            )
        baseline = max(0.0, float(medium[rid]))
        hi = min(baseline * max_frac, cap)
        lo = max(0.0, baseline * min_frac)
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise RuntimeError(f"Could not reconstruct valid bounds for {rid}: {lo}, {hi}")
        out.append([lo, hi])
    return np.asarray(out, float)


def _space_description(run_dir: Path, model_path: Path) -> dict:
    cfg = _load_yaml(run_dir / "metadata" / "config_snapshot.yaml")
    pre = _load_json(run_dir / "metadata" / "preflight.json")
    man = _load_json(run_dir / "metadata" / "run_manifest.json")
    model = _load_cobra_model(model_path, cfg.get("solver", {}).get("name"))
    features = [str(x) for x in pre.get("variable_medium_reaction_ids", [])]
    if not features:
        raise RuntimeError("No variable medium reaction IDs in preflight.json")
    bounds = _recover_bounds(run_dir, cfg, model, features)
    biomass = str(cfg.get("model", {}).get("biomass_reaction_id") or cfg.get("model", {}).get("objective_reaction_id") or "")
    if not biomass:
        raise RuntimeError("No biomass reaction in config snapshot")
    model.reactions.get_by_id(biomass)
    names = {}
    for rid in features:
        try: names[rid] = str(model.reactions.get_by_id(rid).name or "")
        except Exception: names[rid] = ""
    return {
        "cfg": cfg, "preflight": pre, "manifest": man, "model": model,
        "features": features, "bounds": bounds, "biomass": biomass, "feature_labels": names,
        "feasible_statuses": {str(x).lower() for x in cfg.get("simulation", {}).get("feasible_statuses", ["optimal"])},
    }


def _decode_design_cache(arr: np.ndarray, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray, str]:
    arr = np.asarray(arr, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    if arr.ndim != 2 or arr.shape[1] != len(bounds):
        raise RuntimeError(f"Design cache shape {arr.shape} != expected dimension {len(bounds)}")
    eps = 1e-7
    unit_ok = bool(np.all((arr >= -eps) & (arr <= 1 + eps)))
    phys_ok = bool(np.all((arr >= lo[None, :] - 1e-7) & (arr <= hi[None, :] + 1e-7)))
    if phys_ok and not unit_ok:
        Xphys = arr
        X01 = (arr - lo) / (hi - lo)
        mode = "physical"
    elif unit_ok and not phys_ok:
        X01 = np.clip(arr, 0, 1)
        Xphys = lo + X01 * (hi - lo)
        mode = "unit_cube"
    elif phys_ok and unit_ok:
        # For tiny physical spaces (CHO) both tests can pass. Compare endpoint fit.
        err_unit = np.nanmedian(np.abs(np.nanmin(arr,0)) + np.abs(np.nanmax(arr,0)-1))
        err_phys = np.nanmedian((np.abs(np.nanmin(arr,0)-lo)+np.abs(np.nanmax(arr,0)-hi))/np.maximum(hi-lo,1e-12))
        if err_phys < err_unit:
            Xphys = arr; X01 = (arr-lo)/(hi-lo); mode="physical"
        else:
            X01=np.clip(arr,0,1); Xphys=lo+X01*(hi-lo); mode="unit_cube"
    else:
        raise RuntimeError("Design cache is neither unit-cube nor within recovered physical bounds")
    return np.asarray(X01,float), np.asarray(Xphys,float), mode


def _header(path: Path) -> list[str]:
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            import pyarrow.parquet as pq
            return list(pq.ParquetFile(path).schema.names)
        if path.suffix.lower() == ".csv":
            return list(pd.read_csv(path, nrows=0).columns)
    except Exception: pass
    return []


def _extract_growth_outcomes(run_dir: Path, expected_n: int) -> tuple[pd.DataFrame, dict]:
    # The portable input installer attaches ``chunks/results`` as a directory
    # symlink to avoid copying the 50k archive.  pathlib.Path.rglob() does not
    # recurse through directory symlinks on macOS/Python, so explicitly scan
    # the canonical compact-results directory first.  Keep the recursive scan
    # as a fallback for copied/nonstandard layouts.
    candidate_paths: list[Path] = []
    seen_paths: set[str] = set()

    compact_dir = run_dir / "chunks" / "results"
    if compact_dir.exists():
        for ext in ("*.parquet", "*.pq", "*.csv"):
            for p in compact_dir.glob(ext):
                key = str(p.resolve())
                if key not in seen_paths:
                    seen_paths.add(key)
                    candidate_paths.append(p)

    for ext in ("*.parquet", "*.pq", "*.csv"):
        for p in run_dir.rglob(ext):
            key = str(p.resolve())
            if key not in seen_paths:
                seen_paths.add(key)
                candidate_paths.append(p)

    files = []
    for p in candidate_paths:
        if any(x in str(p).lower() for x in ("fluxes", "exchange_flux", "reaction_flux", "conditions")):
            continue
        h = _header(p)
        if _detect_column(h, ID_CANDIDATES) and _detect_column(h, GROWTH_CANDIDATES):
            files.append((p,h))
    if not files:
        raise RuntimeError(
            "No compact results table with sample_id + maximum_growth found. "
            f"Scanned canonical results dir={compact_dir} (exists={compact_dir.exists()}) "
            f"and recursive run dir; candidate files seen={len(candidate_paths)}"
        )
    files.sort(key=lambda ph: ("results_chunk" in str(ph[0]).lower(), "summary" in str(ph[0]).lower(), -ph[0].stat().st_size), reverse=True)
    records = {}
    used=[]
    for p,h in files:
        sid=_detect_column(h,ID_CANDIDATES); gcol=_detect_column(h,GROWTH_CANDIDATES); scol=_detect_column(h,STATUS_CANDIDATES)
        cols=[sid,gcol]+([scol] if scol else [])
        try:
            df=pd.read_parquet(p,columns=cols) if p.suffix.lower() in {".parquet",".pq"} else pd.read_csv(p,usecols=cols)
        except Exception:
            continue
        used.append(str(p))
        for _,r in df.iterrows():
            x=pd.to_numeric(r[sid],errors="coerce")
            if pd.isna(x): continue
            i=int(x); gv=float(pd.to_numeric(r[gcol],errors="coerce")); st=str(r[scol]) if scol else ""
            if i not in records: records[i]=(gv,st)
        if len(records)>=expected_n: break
    ids=sorted(records)
    if ids==list(range(expected_n)): offset=0
    elif ids==list(range(1,expected_n+1)): offset=1
    else: raise RuntimeError(f"Recovered {len(ids)} unique result IDs; expected {expected_n} sequential IDs")
    out=pd.DataFrame([{"sample_id":i,"design_index":i-offset,"maximum_growth":records[i][0],"stored_status":records[i][1]} for i in ids])
    return out,{"files_used":used,"id_offset":offset}


def _nominal_feasible(outcomes: pd.DataFrame, feasible_statuses: set[str]) -> np.ndarray:
    g=pd.to_numeric(outcomes["maximum_growth"],errors="coerce").to_numpy(float)
    ok=np.isfinite(g) & (g>=0)
    s=outcomes["stored_status"].astype(str).str.lower().to_numpy()
    if np.any(np.asarray([bool(x) for x in s])):
        # If statuses are blank for some exporter, finite growth still controls them.
        known=np.asarray([x.strip()!="" for x in s])
        status_ok=np.asarray([x in feasible_statuses for x in s])
        ok &= (~known)|status_ok
    return ok


def _base_medium(model, cfg: dict) -> dict[str,float]:
    medium={str(k):float(v) for k,v in (getattr(model,"medium",{}) or {}).items()}
    for rid,val in (cfg.get("sampling",{}).get("fixed_reactions",{}) or {}).items(): medium[str(rid)]=float(val)
    return medium


def _solve_growth(model, desc: dict, xphys: np.ndarray, stress_members: list[str] | None=None, factor: float=1.0) -> dict:
    t=time.perf_counter()
    features=desc["features"]; bounds=desc["bounds"]
    with model:
        medium=_base_medium(model,desc["cfg"])
        for rid,val in zip(features,np.asarray(xphys,float)): medium[rid]=max(0.0,float(val))
        if stress_members:
            bmap={rid:bounds[i] for i,rid in enumerate(features)}
            for rid in stress_members:
                lo=float(bmap[rid][0]); current=float(medium[rid])
                medium[rid]=max(0.0, lo + float(factor)*(current-lo))
        model.medium=medium
        model.objective=model.reactions.get_by_id(desc["biomass"]); model.objective_direction="max"
        sol=model.optimize(); status=str(sol.status); ok=status.lower() in desc["feasible_statuses"]
        growth=float(sol.fluxes[desc["biomass"]]) if ok else np.nan
    return {"status":status,"feasible":bool(ok and np.isfinite(growth)),"growth":growth,"seconds":float(time.perf_counter()-t),"fba_solves":1}


def _replay_qc(desc: dict, outcomes: pd.DataFrame, Xphys: np.ndarray, n: int=5) -> list[dict]:
    g=pd.to_numeric(outcomes["maximum_growth"],errors="coerce").to_numpy(float)
    finite=np.where(np.isfinite(g))[0]
    if not len(finite): return []
    rng=np.random.default_rng(20260822)
    picks=[int(finite[0]),int(finite[len(finite)//2]),int(finite[-1])]
    picks += [int(x) for x in rng.choice(finite,size=min(n,len(finite)),replace=False)]
    seen=[]; rows=[]
    for i in picks:
        if i in seen: continue
        seen.append(i)
        r=_solve_growth(desc["model"],desc,Xphys[i])
        stored=float(g[i]); diff=abs(float(r["growth"])-stored) if r["feasible"] and np.isfinite(stored) else np.nan
        tol=1e-7+1e-6*max(1.0,abs(stored))
        rows.append({"design_index":i,"stored_growth":stored,"replayed_growth":r["growth"],"abs_diff":diff,"tolerance":tol,"pass":bool(np.isfinite(diff) and diff<=tol),"status":r["status"]})
        if len(rows)>=n: break
    return rows


def _prior_growth_qc(X01,growth,feasible,cfg,seed:int) -> dict:
    folds=int(cfg["prior_growth_surrogate"].get("qc_folds",5))
    kf=KFold(n_splits=folds,shuffle=True,random_state=seed+91)
    pred=np.full(len(X01),np.nan,float)
    pinf=np.full(len(X01),np.nan,float)
    for fold,(tr,va) in enumerate(kf.split(X01),1):
        m=GrowthPriorSurrogate.fit(X01[tr],growth[tr],feasible[tr],cfg,seed+1000*fold)
        pred[va],pinf[va]=m.predict_growth_raw(X01[va])
    good=feasible & np.isfinite(growth) & np.isfinite(pred)
    rmse=float(math.sqrt(mean_squared_error(growth[good],pred[good])))
    scale=float(np.nanmax(growth[good])-np.nanmin(growth[good]))
    rho=float(spearmanr(growth[good],pred[good]).statistic) if int(good.sum())>2 else np.nan
    return {"folds":folds,"rows":int(len(X01)),"feasible_rows":int(good.sum()),"r2":float(r2_score(growth[good],pred[good])),"rmse":rmse,"nrmse_range":float(rmse/max(scale,1e-12)),"spearman_rho":rho}


def inspect_run(run_dir:Path,model_path:Path,outdir:Path,study_config:Path,case_key:str,required_rows:int) -> dict:
    t0=time.time(); study=_load_yaml(study_config); case=study["cases"][case_key]; rob=study["robustness"]
    desc=_space_description(run_dir,model_path); features=desc["features"]; bounds=desc["bounds"]
    classes=list(rob["stress_classes"]); groups={k:list(map(str,v)) for k,v in case["stress_groups"].items()}
    validate_stress_groups(features,groups,classes)
    expected=int(desc["preflight"].get("n_simulations",required_rows))
    if expected!=required_rows: raise RuntimeError(f"Archive has {expected} rows, required {required_rows}")
    cache=_find_design_cache(run_dir); raw=np.load(cache,mmap_mode="r")
    if raw.shape[0]!=required_rows: raise RuntimeError(f"Design cache has {raw.shape[0]} rows, required {required_rows}")
    X01,Xphys,cache_mode=_decode_design_cache(np.asarray(raw),bounds)
    outcomes,outdiag=_extract_growth_outcomes(run_dir,required_rows)
    outcomes=outcomes.sort_values("design_index").reset_index(drop=True)
    g=pd.to_numeric(outcomes["maximum_growth"],errors="coerce").to_numpy(float)
    F=_nominal_feasible(outcomes,desc["feasible_statuses"])

    replay=_replay_qc(desc,outcomes,Xphys,n=5)
    replay_pass=bool(replay and all(r["pass"] for r in replay))
    if not replay_pass: raise RuntimeError(f"Nominal exact-replay QC failed: {replay}")

    seed=int(study["study"]["seed"])
    qc=_prior_growth_qc(X01,g,F,study,seed)
    prior=GrowthPriorSurrogate.fit(X01,g,F,study,seed)
    outdir.mkdir(parents=True,exist_ok=True)
    joblib.dump(prior,outdir/"frozen_nominal_growth_surrogate.joblib",compress=3)
    _jdump(qc,outdir/"frozen_nominal_growth_surrogate_qc.json")

    derived=derive_prior_robustness(
        prior,X01,g,F,features,groups,classes,float(rob["fraction_toward_lower_bound"]),
        float(study["prior_growth_surrogate"].get("infeasible_probability_threshold",0.5)),
        bool(rob.get("enforce_monotone_prior",True)),
    )
    baseline=outcomes[["sample_id","design_index","maximum_growth","stored_status"]].copy()
    for j,rid in enumerate(features): baseline[rid]=Xphys[:,j]
    for cls in classes: baseline[f"prior_{cls}_stressed_growth"]=derived["scenario_growth"][cls]
    baseline["prior_worst_stress_class"]=derived["worst_class"]
    baseline["robust_growth"]=derived["robust_growth"]
    baseline["robustness_retention"]=derived["retention"]
    baseline["product_flux"]=baseline["robust_growth"]
    baseline["feasible"]=F & np.isfinite(baseline["robust_growth"].to_numpy(float))
    baseline["status"]=np.where(baseline["feasible"],"nominal_exact_prior_robustness_predicted","nominal_infeasible")
    good=baseline["feasible"].to_numpy(bool)
    growth_scale=float(np.nanmax(baseline.loc[good,"maximum_growth"]))
    robust_scale=float(np.nanmax(baseline.loc[good,"robust_growth"]))
    if growth_scale<=0 or robust_scale<=0: raise RuntimeError(f"Invalid scales: growth={growth_scale}, robust={robust_scale}")
    bpath=outdir/"baseline_compact.csv.gz"; baseline.to_csv(bpath,index=False,compression="gzip")

    pconf={"problem":{
        "case_name":case_key,"case_label":case.get("label",case_key),"run_dir":str(run_dir.resolve()),"model_path":str(model_path.resolve()),
        "model_id":desc["manifest"].get("model_summary",{}).get("model_id"),"model_fingerprint":desc["manifest"].get("model_fingerprint"),
        "config_hash":desc["manifest"].get("config_hash"),"biomass_reaction_id":desc["biomass"],"features":features,"feature_labels":desc["feature_labels"],
        "bounds":{rid:[float(bounds[i,0]),float(bounds[i,1])] for i,rid in enumerate(features)},
        "growth_scale":growth_scale,"product_scale":robust_scale,
        "objective_1_name":"exact_nominal_maximum_growth",
        "objective_2_name":"exact_worst_case_environmental_stressed_growth_live__surrogate_derived_in_50k_prior",
        "objective_2_formula":"min_s growth(T_s(x)); s in {carbon, oxygen, nitrogen, phosphate}",
        "legacy_objective_2_column":"product_flux","raw_robust_growth_column":"robust_growth",
        "stress_classes":classes,"stress_groups":groups,"stress_fraction_toward_lower_bound":float(rob["fraction_toward_lower_bound"]),
        "stress_definition":"in-domain reduction of only variables already represented in the frozen landscape; objective 2 is worst-case growth across the four classes",
        "prior_robustness_is_surrogate_derived":True,"live_robustness_is_exact_fba":True,"hv_reference_normalized":[0.0,0.0],
        "baseline_table":str(bpath.resolve()),"design_cache":str(cache.resolve()),
    }}
    (outdir/"problem_config.yaml").write_text(yaml.safe_dump(pconf,sort_keys=False),encoding="utf-8")
    report={
        "status":"PASS","exact_replay_qc_pass":True,"rows":int(len(baseline)),"feasible":int(good.sum()),"dimension":len(features),
        "growth_scale":growth_scale,"robust_growth_scale":robust_scale,"product_scale":robust_scale,"model_path":str(model_path.resolve()),
        "model_id":pconf["problem"]["model_id"],"model_fingerprint":pconf["problem"]["model_fingerprint"],"config_hash":pconf["problem"]["config_hash"],
        "design_cache_mode":cache_mode,"design_cache":str(cache.resolve()),"outcome_diagnostics":outdiag,"nominal_exact_replay":replay,
        "prior_growth_surrogate_qc":qc,"stress_classes":classes,"stress_groups":groups,"stress_fraction_toward_lower_bound":float(rob["fraction_toward_lower_bound"]),
        "archive_stress_fba_solves":0,"archive_nominal_fba_recomputed":int(len(replay)),"baseline_table":str(bpath.resolve()),"preparation_wall_seconds":float(time.time()-t0),
    }
    _jdump(report,outdir/"landscape_preflight.json")
    return report


def evaluate_proposals(run_dir:Path,model_path:Path,proposal:Path,output:Path,study_config:Path,case_key:str) -> None:
    study=_load_yaml(study_config); case=study["cases"][case_key]; rob=study["robustness"]
    desc=_space_description(run_dir,model_path); features=desc["features"]; classes=list(rob["stress_classes"]); groups=case["stress_groups"]
    validate_stress_groups(features,groups,classes); factor=float(rob["fraction_toward_lower_bound"])
    df=pd.read_csv(proposal); missing=[f for f in features if f not in df.columns]
    if missing: raise RuntimeError(f"Proposal missing features: {missing}")
    rows=[]
    for _,r in df.iterrows():
        x=np.asarray([r[f] for f in features],float)
        nominal=_solve_growth(desc["model"],desc,x)
        scenario={}; statuses={}; total_seconds=float(nominal["seconds"]); solves=int(nominal["fba_solves"])
        if nominal["feasible"]:
            for cls in classes:
                z=_solve_growth(desc["model"],desc,x,list(map(str,groups[cls])),factor)
                total_seconds+=float(z["seconds"]); solves+=int(z["fba_solves"]); statuses[cls]=z["status"]
                scenario[cls]=float(z["growth"]) if z["feasible"] and np.isfinite(z["growth"]) else 0.0
            robust=float(min(scenario.values()))
            worst=min(classes,key=lambda c:scenario[c])
            retention=float(np.clip(robust/max(float(nominal["growth"]),1e-12),0.0,1.0+1e-8))
            feasible=True
        else:
            for cls in classes: scenario[cls]=np.nan; statuses[cls]="not_run_nominal_infeasible"
            robust=np.nan; worst="nominal_infeasible"; retention=np.nan; feasible=False
        rec={
            "proposal_id":r.get("proposal_id",len(rows)),"maximum_growth":nominal["growth"],"robust_growth":robust,"product_flux":robust,
            "robustness_retention":retention,"worst_stress_class":worst,"nominal_status":nominal["status"],"status":"optimal" if feasible else f"nominal={nominal['status']}",
            "feasible":feasible,"fba_solves":solves,"solver_wall_seconds":total_seconds,
        }
        for cls in classes:
            rec[f"{cls}_stressed_growth"]=scenario[cls]; rec[f"{cls}_stress_status"]=statuses[cls]
        rows.append(rec)
    output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(output,index=False)


def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="cmd",required=True)
    a=sp.add_parser("inspect"); a.add_argument("--run-dir",required=True); a.add_argument("--model-path",required=True); a.add_argument("--output-dir",required=True); a.add_argument("--study-config",required=True); a.add_argument("--case-key",required=True); a.add_argument("--required-archive-rows",type=int,default=50000)
    e=sp.add_parser("evaluate"); e.add_argument("--run-dir",required=True); e.add_argument("--model-path",required=True); e.add_argument("--proposal",required=True); e.add_argument("--output",required=True); e.add_argument("--study-config",required=True); e.add_argument("--case-key",required=True)
    args=ap.parse_args()
    if args.cmd=="inspect":
        rep=inspect_run(Path(args.run_dir).resolve(),Path(args.model_path).resolve(),Path(args.output_dir).resolve(),Path(args.study_config).resolve(),args.case_key,int(args.required_archive_rows)); print(json.dumps({k:rep[k] for k in ("status","rows","feasible","dimension","archive_stress_fba_solves")},indent=2))
    else:
        evaluate_proposals(Path(args.run_dir).resolve(),Path(args.model_path).resolve(),Path(args.proposal).resolve(),Path(args.output).resolve(),Path(args.study_config).resolve(),args.case_key)

if __name__=="__main__": main()
