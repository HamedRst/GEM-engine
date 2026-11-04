from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import copy
import json
import math
import os

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from .atlas_discovery import LandscapeRecord
from .data import infer_design_variables, load_landscape


PRODUCT_DISPLAY_LABELS = {
    "yeast": "ethanol production",
    "pputida": "C8:0 mcl-PHA production",
    "cglutamicum": "L-glutamate production",
    "cho": "IgG production",
    "ecoli_antiepex": "antiEpEX-scFv production",
}


EXECUTED_PROBLEM_FILES = [
    f"{i:02d}_{name}.yaml" for i, name in [
        (1,"describe_growth"),
        (2,"global_sensitivity_product"),
        (3,"conditional_sensitivity_product_high_growth"),
        (4,"local_sensitivity_product"),
        (5,"pair_interaction_product"),
        (6,"product_fragility"),
        (7,"growth_robustness"),
        (8,"product_threshold_region"),
        (9,"product_threshold_boundary"),
        (10,"feasibility_drivers"),
        (11,"phenotype_regimes_growth_product"),
        (12,"growth_product_tradeoff"),
        (13,"extract_pareto_growth_product"),
        (14,"compare_high_vs_low_growth_product"),
        (15,"design_space_coverage"),
        (16,"product_outliers"),
        (17,"sampling_saturation"),
        (18,"derived_efficiency_sensitivity"),
        (19,"point_query_TEMPLATE"),
        (20,"response_curve_TEMPLATE"),
        (21,"generic_query"),
        (22,"correlation_matrix_growth_product_inputs"),
        (23,"product_sensitivity_map"),
        (24,"phenotype_dimensionality"),
        (25,"marginal_response_TEMPLATE"),
        (26,"product_surrogate_support"),
        (27,"pareto_sampling_saturation"),
        (28,"product_controller_scan"),
        (29,"product_threshold_switch_scan"),
        (30,"response_surface_TEMPLATE"),
    ]
]


@dataclass
class LandscapeProfile:
    species: str
    label: str
    rows: int
    feasible_rows: int
    design_variables: list[str]
    variable_labels: dict[str, str]
    variable_reaction_ids: dict[str, str]
    primary_variable: str
    secondary_variable: str
    carbon_variable: str
    oxygen_variable: str | None
    balanced_point: dict[str, float]
    midpoint_point: dict[str, float]
    product_quantiles: dict[str, float]
    growth_quantiles: dict[str, float]
    product_range: list[float]
    growth_range: list[float]


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_parquet(path)


def _variable_catalog(run_dir: Path) -> dict[str, str]:
    labels: dict[str, str] = {}
    for p in [run_dir / "catalogs" / "medium_space.csv", run_dir / "catalogs" / "medium_space.parquet"]:
        if not p.exists():
            continue
        try:
            t = _read_table(p)
        except Exception:
            continue
        if "reaction_id" not in t.columns:
            continue
        name_col = next((c for c in ["reaction_name", "name", "metabolite_name"] if c in t.columns), None)
        for _, r in t.iterrows():
            rid = str(r["reaction_id"])
            labels[rid] = str(r[name_col]) if name_col and pd.notna(r[name_col]) else rid
        break
    return labels


def _rid_from_variable(v: str) -> str:
    for prefix in ["requested_medium__", "applied_medium__"]:
        if v.startswith(prefix):
            return v[len(prefix):]
    return v


def _finite_quantiles(s: pd.Series, qs=(0.05,0.2,0.4,0.5,0.6,0.7,0.75,0.8,0.95)) -> dict[str,float]:
    x=pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
    if x.empty:
        raise RuntimeError("Target has no finite values")
    q=x.quantile(list(qs))
    return {f"q{int(round(k*100)):02d}":float(q.loc[k]) for k in qs}


def _range(s: pd.Series) -> list[float]:
    x=pd.to_numeric(s,errors="coerce").replace([np.inf,-np.inf],np.nan).dropna()
    return [float(x.min()),float(x.max())]


def _choose_variables(df: pd.DataFrame, variables: list[str], target: str) -> tuple[str,str]:
    y=pd.to_numeric(df[target],errors="coerce")
    rows=[]
    for v in variables:
        x=pd.to_numeric(df[v],errors="coerce")
        ok=x.notna() & y.notna()
        rho=0.0
        if ok.sum()>=10 and x[ok].nunique()>1 and y[ok].nunique()>1:
            try:
                rr=spearmanr(x[ok],y[ok]).statistic
                rho=float(rr) if np.isfinite(rr) else 0.0
            except Exception:
                rho=0.0
        rows.append((abs(rho),v))
    rows.sort(reverse=True)
    first=rows[0][1] if rows else variables[0]
    second=rows[1][1] if len(rows)>1 else first
    return first,second


def _find_semantic_variable(variables:list[str], labels:dict[str,str], tokens:list[str]) -> str|None:
    ranked=[]
    for v in variables:
        rid=_rid_from_variable(v)
        text=f"{rid} {labels.get(rid,'')}".lower()
        score=sum(len(t) for t in tokens if t.lower() in text)
        if score:
            ranked.append((score,v))
    return max(ranked)[1] if ranked else None


def _balanced_point(df:pd.DataFrame,variables:list[str],growth:str,product:str)->dict[str,float]:
    d=df[variables+[growth,product]].copy()
    for c in d.columns:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d=d.replace([np.inf,-np.inf],np.nan).dropna()
    if d.empty:
        raise RuntimeError("No complete rows for balanced-point selection")
    def norm(c):
        x=d[c].to_numpy(float); lo,hi=np.quantile(x,[0.02,0.98]); den=max(hi-lo,1e-12)
        return np.clip((x-lo)/den,0,1)
    score=(norm(growth)+norm(product))/2
    idx=int(np.argmax(score))
    return {v:float(d.iloc[idx][v]) for v in variables}


def build_profile(record: LandscapeRecord) -> tuple[LandscapeProfile,pd.DataFrame]:
    run_dir=Path(record.path)
    df,_=load_landscape(run_dir,include_fluxes=False)
    if "maximum_growth" not in df.columns or "product_flux" not in df.columns:
        raise RuntimeError(f"{record.species} landscape lacks maximum_growth/product_flux")
    variables=infer_design_variables(df,run_dir)
    if not variables:
        raise RuntimeError(f"{record.species}: no design variables recovered")
    labels=_variable_catalog(run_dir)
    var_labels={v:labels.get(_rid_from_variable(v),_rid_from_variable(v)) for v in variables}
    var_rids={v:_rid_from_variable(v) for v in variables}
    primary,secondary=_choose_variables(df,variables,"product_flux")
    carbon=_find_semantic_variable(variables,labels,["glucose","glc","carbon","d-glucose"]) or primary
    oxygen=_find_semantic_variable(variables,labels,["oxygen","o2"])
    feasible=(df["feasible"].fillna(False).astype(bool) if "feasible" in df.columns else (pd.to_numeric(df["maximum_growth"],errors="coerce").notna() & pd.to_numeric(df["product_flux"],errors="coerce").notna()))
    use=df.loc[feasible].copy()
    if len(use)<50:
        use=df.copy()
    mids={v:float((pd.to_numeric(use[v],errors="coerce").min()+pd.to_numeric(use[v],errors="coerce").max())/2) for v in variables}
    prof=LandscapeProfile(
        species=record.species,label=record.label,rows=int(len(df)),feasible_rows=int(feasible.sum()),
        design_variables=variables,variable_labels=var_labels,variable_reaction_ids=var_rids,
        primary_variable=primary,secondary_variable=secondary,carbon_variable=carbon,oxygen_variable=oxygen,
        balanced_point=_balanced_point(use,variables,"maximum_growth","product_flux"),midpoint_point=mids,
        product_quantiles=_finite_quantiles(use["product_flux"]),growth_quantiles=_finite_quantiles(use["maximum_growth"]),
        product_range=_range(use["product_flux"]),growth_range=_range(use["maximum_growth"]),
    )
    return prof,df


def _load_yaml(path:Path)->dict[str,Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _base_surrogate_config(memory_safe:bool=True)->dict[str,Any]:
    # Shared across all five species to keep the exploration comparison fair.  The
    # persistent cache prevents repeated fitting of the same target/scope.
    return {
        "model":"extra_trees",
        "n_estimators":int(os.environ.get("GEM_ATLAS_TREES", 240 if memory_safe else 320)),
        "cv_folds":5,
        "min_r2":0.50,
        "min_auc":0.65,
        "max_train_rows":50000,
        "n_jobs":int(os.environ.get("GEM_ATLAS_JOBS", 4 if memory_safe else -1)),
    }


def _replace_scope_filter(spec:dict,column_suffix:str,value:float)->None:
    filters=(spec.setdefault("scope",{}).setdefault("filters",[]))
    found=False
    for f in filters:
        if str(f.get("column","")).endswith(column_suffix):
            f["value"]=float(value); found=True
    if not found:
        filters.append({"column":f"result:{column_suffix}","op":">=","value":float(value)})


def instantiate_problem(template:dict[str,Any], template_name:str, record:LandscapeRecord, profile:LandscapeProfile, cache_dir:Path)->dict[str,Any]:
    s=copy.deepcopy(template)
    s["name"]=f"{record.species}__{s.get('name',Path(template_name).stem)}"
    s.setdefault("landscape",{})["path"]=record.path
    # All atlas questions only require compact outcomes + requested media. Loading
    # full 50k reaction-flux matrices on an M2 would be wasteful and unnecessary.
    s["landscape"]["include_fluxes"]=False
    s["surrogate"]=_base_surrogate_config(memory_safe=True) | (s.get("surrogate") or {})
    s["surrogate"]["n_jobs"]=int(os.environ.get("GEM_ATLAS_JOBS", 4))
    s["surrogate"]["n_estimators"]=int(os.environ.get("GEM_ATLAS_TREES", 240))
    s["surrogate"]["max_train_rows"]=50000
    s["surrogate"]["_cache_dir"]=str(cache_dir.resolve())
    s.setdefault("outputs",{})
    s["outputs"].update({"save_figures":True,"publication_formats":["png","pdf","svg"],"dpi":600})
    s["atlas"]={
        "species":record.species,"species_label":record.label,
        "product_reaction_id":record.product_reaction_id,"biomass_reaction_id":record.biomass_reaction_id,
        "primary_variable":profile.primary_variable,"secondary_variable":profile.secondary_variable,
        "primary_variable_label":profile.variable_labels.get(profile.primary_variable,profile.primary_variable),
        "secondary_variable_label":profile.variable_labels.get(profile.secondary_variable,profile.secondary_variable),
        "variable_labels":profile.variable_labels,
        "target_labels":{
            "maximum_growth":"growth rate",
            "product_flux":PRODUCT_DISPLAY_LABELS.get(record.species,"product formation"),
            "product_per_carbon_input":f"{PRODUCT_DISPLAY_LABELS.get(record.species,'product')} per carbon input",
        },
        "source_template":template_name,
    }
    # Semantic display metadata travels inside the resolved problem YAML. The core
    # engine never needs species-specific plotting code; publication labels are data.
    s.setdefault("semantics",{})
    s["semantics"].setdefault("target_labels",s["atlas"]["target_labels"])
    s["semantics"].setdefault("variable_labels",profile.variable_labels)
    s.setdefault("visualization",{"mode":"auto"})
    op=s.setdefault("operator",{}); params=op.setdefault("parameters",{})
    stem=Path(template_name).stem

    if stem.startswith("03_"):
        _replace_scope_filter(s,"maximum_growth",profile.growth_quantiles["q60"])
    elif stem.startswith("04_"):
        params["point"]=profile.balanced_point
    elif stem.startswith("08_"):
        params["threshold"]=profile.product_quantiles["q80"]
    elif stem.startswith("09_"):
        params["threshold"]=profile.product_quantiles["q75"]
    elif stem.startswith("17_"):
        params["sample_sizes"]=[x for x in [500,1000,2500,5000,10000,25000,50000] if x<=profile.rows]
        params["repeats"]=8
        params["evaluation_points"]=min(5000,profile.rows)
    elif stem.startswith("18_"):
        carbon=profile.carbon_variable
        rid=_rid_from_variable(carbon)
        s["name"]=f"{record.species}__product_per_carbon_input_sensitivity"
        s["derived"]=[{"name":"product_per_carbon_input","expression":f'safe_divide(col("result:product_flux"), col("medium:{rid}"))'}]
        s["targets"]=[{"name":"product_per_carbon_input","value":"product_per_carbon_input","direction":"maximize"}]
        _replace_scope_filter(s,"maximum_growth",profile.growth_quantiles["q20"])
    elif stem.startswith("19_"):
        # Query a deliberately unsampled interior location using every design dimension.
        s["variables"]={"mode":"all_design"}
        params["point"]=profile.midpoint_point
        params["neighbors"]=12
        s["outputs"]["save_figures"]=True
    elif stem.startswith("20_"):
        params["variable"]=_rid_from_variable(profile.primary_variable)
        params["point"]=profile.balanced_point
        params["grid_points"]=81
    elif stem.startswith("21_"):
        s.setdefault("scope",{})["filters"]=[
            {"column":"result:maximum_growth","op":">=","value":profile.growth_quantiles["q70"]},
            {"column":"result:product_flux","op":">=","value":profile.product_quantiles["q70"]},
        ]
        params["sort_by"]="product_flux"; params["ascending"]=False; params["max_saved_points"]=10000
        s["outputs"]["save_figures"]=True
    elif stem.startswith("25_"):
        params["variable"]=_rid_from_variable(profile.primary_variable)
        params["grid_points"]=41
        params["background_rows"]=384
    elif stem.startswith("27_"):
        params["sample_sizes"]=[x for x in [500,1000,2500,5000,10000,25000,50000] if x<=profile.rows]
        params["repeats"]=8
    elif stem.startswith("29_"):
        params["threshold"]=profile.product_quantiles["q60"]
        params["anchor_count"]=80; params["levels"]=17
    elif stem.startswith("30_"):
        params["variables"]=[_rid_from_variable(profile.primary_variable),_rid_from_variable(profile.secondary_variable)]
        params["point"]=profile.balanced_point
        params["grid_points"]=35

    # Make the M2 all-problem run tractable without changing the scientific operator.
    if stem.startswith("05_"):
        params["anchor_count"]=96; params["max_pairs"]=min(60,math.comb(len(profile.design_variables),2))
    if stem.startswith("23_"):
        params["anchor_count"]=220
    if stem.startswith("26_"):
        params["evaluation_rows"]=min(8000,profile.rows); params["top_points"]=300
    if stem.startswith("28_"):
        params["anchor_count"]=80; params["levels"]=9

    return s


def generate_resolved_problems(package_root:str|Path, record:LandscapeRecord, output_dir:str|Path, cache_dir:str|Path)->tuple[LandscapeProfile,list[Path]]:
    root=Path(package_root).resolve(); out=Path(output_dir).resolve(); out.mkdir(parents=True,exist_ok=True)
    profile,_=build_profile(record)
    (out/"LANDSCAPE_PROFILE.yaml").write_text(yaml.safe_dump(asdict(profile),sort_keys=False),encoding="utf-8")
    paths=[]
    for name in EXECUTED_PROBLEM_FILES:
        src=root/"problems"/name
        if not src.exists():
            raise FileNotFoundError(src)
        spec=instantiate_problem(_load_yaml(src),name,record,profile,Path(cache_dir))
        dst=out/name.replace("_TEMPLATE","")
        dst.write_text(yaml.safe_dump(spec,sort_keys=False),encoding="utf-8")
        paths.append(dst)
    return profile,paths
