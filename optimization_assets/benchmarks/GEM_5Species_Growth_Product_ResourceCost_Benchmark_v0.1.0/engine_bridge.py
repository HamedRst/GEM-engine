#!/usr/bin/env python3
from __future__ import annotations
"""Portable growth + product + normalized resource-cost bridge.

Archive contract: reuse the completed 50k run's exact maximum growth and
90%-growth-constrained product flux. Resource burden is deterministic from the
same design coordinates, so preparing objective 3 requires ZERO additional FBA.

Live contract: each genuinely new design is evaluated by the original two-stage
FBA formulation (max growth; then max product at >= 90% of that design's max
growth). Resource burden/efficiency is calculated exactly with no FBA solve.
"""
import argparse, json, time
from pathlib import Path
from typing import Any, Iterable
import numpy as np, pandas as pd, yaml

GROWTH_CANDIDATES=("maximum_growth","exact_maximum_growth","maximum_growth_exact","growth_max")
PRODUCT_CANDIDATES=("product_flux","exact_product_flux","product_flux_exact","product","objective_value")
ID_CANDIDATES=("sample_id","simulation_id","condition_id","row_id","id")
STATUS_CANDIDATES=("status","solver_status","fba_status","exact_status")

def _jdump(obj:Any,path:Path):
    def conv(x):
        if isinstance(x,Path): return str(x)
        if isinstance(x,(np.integer,)): return int(x)
        if isinstance(x,(np.floating,)): return float(x)
        if isinstance(x,np.ndarray): return x.tolist()
        raise TypeError(type(x).__name__)
    path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(obj,indent=2,default=conv))
def _load_json(p): return json.loads(Path(p).read_text())
def _load_yaml(p): return yaml.safe_load(Path(p).read_text()) or {}
def _detect(cols,cands):
    cols=list(cols); low={str(c).lower():c for c in cols}
    for c in cands:
        if c in cols:return c
        if c.lower() in low:return low[c.lower()]
    for col in cols:
        for c in cands:
            if c.lower() in str(col).lower(): return col
    return None

def _load_cobra_model(path:Path,solver=None):
    from cobra import io
    s=path.suffix.lower()
    if s==".json": m=io.load_json_model(str(path))
    elif s in {".xml",".sbml"}: m=io.read_sbml_model(str(path))
    elif s==".mat": m=io.load_matlab_model(str(path))
    elif s in {".yml",".yaml"}: m=io.load_yaml_model(str(path))
    else: raise ValueError(f"Unsupported model format: {path}")
    if solver: m.solver=str(solver)
    return m

def _find_design_cache(run_dir):
    hits=sorted((run_dir/"design_cache").glob("*.npy")) or sorted(run_dir.rglob("*.npy"))
    if not hits: raise FileNotFoundError(f"No design-cache .npy under {run_dir}")
    return hits[0]

def _recover_bounds(run_dir,cfg,model,features):
    explicit=cfg.get("sampling",{}).get("bounds",{}) or {}
    if all(r in explicit for r in features):
        out=[]
        for rid in features:
            raw=explicit[rid]; out.append([float(raw["min"]),float(raw["max"])])
        return np.asarray(out,float)
    for p in (run_dir/"catalogs"/"medium_space.csv",run_dir/"catalogs"/"medium_space.parquet"):
        if p.exists():
            t=pd.read_csv(p) if p.suffix==".csv" else pd.read_parquet(p)
            if {"reaction_id","effective_min_uptake","effective_max_uptake"}.issubset(t.columns):
                t=t.set_index(t["reaction_id"].astype(str),drop=False)
                if all(r in t.index for r in features):
                    return np.asarray([[float(t.loc[r,"effective_min_uptake"]),float(t.loc[r,"effective_max_uptake"])] for r in features],float)
    samp=cfg.get("sampling",{}); lo_frac=float(samp.get("auto_min_fraction",0)); hi_frac=float(samp.get("auto_max_fraction",1)); cap=float(samp.get("auto_max_uptake_cap",20))
    med={str(k):float(v) for k,v in (model.medium or {}).items()}; out=[]
    for rid in features:
        if rid not in med: raise RuntimeError(f"Cannot recover bounds for {rid}")
        base=max(0.,med[rid]); lo=max(0.,base*lo_frac); hi=min(base*hi_frac,cap)
        if not hi>lo: raise RuntimeError(f"Degenerate bounds for {rid}: {lo},{hi}")
        out.append([lo,hi])
    return np.asarray(out,float)

def _space_description(run_dir:Path,model_path:Path):
    cfg=_load_yaml(run_dir/"metadata"/"config_snapshot.yaml"); pre=_load_json(run_dir/"metadata"/"preflight.json"); man=_load_json(run_dir/"metadata"/"run_manifest.json")
    model=_load_cobra_model(model_path,cfg.get("solver",{}).get("name")); features=[str(x) for x in pre.get("variable_medium_reaction_ids",[])]
    if not features: raise RuntimeError("No variable_medium_reaction_ids in preflight")
    bounds=_recover_bounds(run_dir,cfg,model,features)
    biomass=str(cfg.get("model",{}).get("biomass_reaction_id") or cfg.get("model",{}).get("objective_reaction_id") or "")
    product=str(cfg.get("model",{}).get("product_reaction_id") or "")
    if not biomass or not product: raise RuntimeError(f"Missing biomass/product IDs: biomass={biomass!r}, product={product!r}")
    model.reactions.get_by_id(biomass); model.reactions.get_by_id(product)
    names={r:str(model.reactions.get_by_id(r).name or "") for r in features}
    frac=float(cfg.get("optimization",{}).get("minimum_growth_fraction",0.9))
    return dict(cfg=cfg,preflight=pre,manifest=man,model=model,features=features,bounds=bounds,biomass=biomass,product=product,minimum_growth_fraction=frac,feature_labels=names,feasible_statuses={str(x).lower() for x in cfg.get("simulation",{}).get("feasible_statuses",["optimal"])})

def _decode(arr,bounds):
    arr=np.asarray(arr,float); lo,hi=bounds[:,0],bounds[:,1]; rng=hi-lo; eps=1e-7
    unit=np.mean((arr < -eps)|(arr>1+eps)); phys=np.mean((arr < lo[None,:]-eps)|(arr>hi[None,:]+eps))
    if phys<1e-6 and unit>1e-6: Xp=arr; mode="physical"; X=(Xp-lo)/rng
    elif unit<1e-6: X=np.clip(arr,0,1); Xp=lo+X*rng; mode="unit_cube"
    elif phys<1e-6: Xp=arr; X=(Xp-lo)/rng; mode="physical"
    else: raise RuntimeError(f"Cannot decode design cache unit_violation={unit}, physical_violation={phys}")
    return np.clip(X,0,1),Xp,mode

def _header(path):
    try:
        if path.suffix.lower() in {".parquet",".pq"}:
            import pyarrow.parquet as pq; return list(pq.ParquetFile(path).schema.names)
        return list(pd.read_csv(path,nrows=0).columns)
    except Exception:return []

def _extract_outcomes(run_dir,expected_n):
    paths=[]; seen=set(); compact=run_dir/"chunks"/"results"
    if compact.exists():
        for ext in ("*.parquet","*.pq","*.csv"):
            for p in compact.glob(ext):
                k=str(p.resolve());
                if k not in seen: seen.add(k); paths.append(p)
    for ext in ("*.parquet","*.pq","*.csv"):
        for p in run_dir.rglob(ext):
            k=str(p.resolve());
            if k not in seen: seen.add(k); paths.append(p)
    files=[]
    for p in paths:
        if any(x in str(p).lower() for x in ("fluxes","exchange_flux","reaction_flux","conditions")): continue
        h=_header(p)
        if _detect(h,ID_CANDIDATES) and _detect(h,GROWTH_CANDIDATES) and _detect(h,PRODUCT_CANDIDATES): files.append((p,h))
    if not files: raise RuntimeError(f"No compact results table with sample_id + growth + product under {run_dir}")
    files.sort(key=lambda z:("results_chunk" in str(z[0]).lower(),-z[0].stat().st_size),reverse=True)
    rec={}; used=[]
    for p,h in files:
        sid=_detect(h,ID_CANDIDATES); gc=_detect(h,GROWTH_CANDIDATES); pc=_detect(h,PRODUCT_CANDIDATES); sc=_detect(h,STATUS_CANDIDATES)
        cols=[sid,gc,pc]+([sc] if sc else [])
        try: df=pd.read_parquet(p,columns=cols) if p.suffix.lower() in {".parquet",".pq"} else pd.read_csv(p,usecols=cols)
        except Exception: continue
        used.append(str(p))
        for _,r in df.iterrows():
            x=pd.to_numeric(r[sid],errors="coerce")
            if pd.isna(x):continue
            i=int(x)
            if i not in rec: rec[i]=(float(pd.to_numeric(r[gc],errors="coerce")),float(pd.to_numeric(r[pc],errors="coerce")),str(r[sc]) if sc else "")
        if len(rec)>=expected_n:break
    ids=sorted(rec)
    if ids==list(range(expected_n)): off=0
    elif ids==list(range(1,expected_n+1)): off=1
    else: raise RuntimeError(f"Recovered {len(ids)} IDs; expected {expected_n} sequential")
    out=pd.DataFrame([dict(sample_id=i,design_index=i-off,maximum_growth=rec[i][0],product_flux=rec[i][1],stored_status=rec[i][2]) for i in ids])
    return out,{"files_used":used,"id_offset":off}

def _feasible(out,feasible_statuses):
    g=pd.to_numeric(out.maximum_growth,errors="coerce").to_numpy(float); p=pd.to_numeric(out.product_flux,errors="coerce").to_numpy(float); ok=np.isfinite(g)&np.isfinite(p)&(g>=0)
    s=out.stored_status.astype(str).str.lower().to_numpy(); known=np.asarray([x.strip()!="" for x in s]); status_ok=np.asarray([x in feasible_statuses for x in s]); return ok & ((~known)|status_ok)

def _base_medium(model,cfg):
    med={str(k):float(v) for k,v in (model.medium or {}).items()}
    for rid,val in (cfg.get("sampling",{}).get("fixed_reactions",{}) or {}).items(): med[str(rid)]=float(val)
    return med

def _reset_glpk_basis(model):
    """Reset only GLPK's simplex basis; do not alter the LP itself."""
    try:
        import swiglpk as glp
    except Exception as e:
        raise RuntimeError("GLPK basis reset requested but swiglpk is unavailable") from e
    glp.glp_std_basis(model.solver.problem)

def _solve_one(desc,xphys):
    t=time.perf_counter(); m=desc["model"]
    with m:
        med=_base_medium(m,desc["cfg"])
        for rid,val in zip(desc["features"],np.asarray(xphys,float)): med[rid]=max(0.,float(val))
        m.medium=med; m.objective=m.reactions.get_by_id(desc["biomass"]); m.objective_direction="max"; s1=m.optimize(); st1=str(s1.status); ok1=st1.lower() in desc["feasible_statuses"]
        if not ok1: return dict(maximum_growth=np.nan,product_flux=np.nan,status=st1,feasible=False,fba_solves=1,solver_wall_seconds=float(time.perf_counter()-t))
        growth=float(s1.fluxes[desc["biomass"]]); biomass_rxn=m.reactions.get_by_id(desc["biomass"]); biomass_rxn.lower_bound=max(float(biomass_rxn.lower_bound),desc["minimum_growth_fraction"]*growth)
        m.objective=m.reactions.get_by_id(desc["product"]); m.objective_direction="max"; solver_iface=str(getattr(m.solver.interface,"__name__","")).lower(); (_reset_glpk_basis(m) if "glpk" in solver_iface else None); s2=m.optimize(); st2=str(s2.status); ok2=st2.lower() in desc["feasible_statuses"]
        product=float(s2.fluxes[desc["product"]]) if ok2 else np.nan
    return dict(maximum_growth=growth,product_flux=product,status=("optimal" if ok2 else f"growth={st1};product={st2}"),feasible=bool(ok2 and np.isfinite(growth) and np.isfinite(product)),fba_solves=2,solver_wall_seconds=float(time.perf_counter()-t))

def _replay(desc,out,Xp,n=5):
    good=np.where(np.isfinite(out.maximum_growth)&np.isfinite(out.product_flux))[0]; rng=np.random.default_rng(20260822); picks=[]
    if len(good): picks=[int(good[0]),int(good[len(good)//2]),int(good[-1])]+[int(x) for x in rng.choice(good,size=min(n,len(good)),replace=False)]
    rows=[]
    for i in dict.fromkeys(picks):
        r=_solve_one(desc,Xp[i]); sg=float(out.iloc[i].maximum_growth); sp=float(out.iloc[i].product_flux); dg=abs(r["maximum_growth"]-sg); dp=abs(r["product_flux"]-sp); tg=1e-7+1e-6*max(1,abs(sg)); tp=1e-7+1e-6*max(1,abs(sp))
        rows.append(dict(design_index=i,stored_growth=sg,replayed_growth=r["maximum_growth"],growth_abs_diff=dg,growth_tolerance=tg,stored_product=sp,replayed_product=r["product_flux"],product_abs_diff=dp,product_tolerance=tp,**{"pass":bool(np.isfinite(dg) and np.isfinite(dp) and dg<=tg and dp<=tp)},status=r["status"],fba_solves=int(r["fba_solves"])))
        if len(rows)>=n:break
    return rows

def inspect_run(run_dir,model_path,outdir,study_config,case_key,required_rows):
    t0=time.time(); study=_load_yaml(study_config); case=study["cases"][case_key]; desc=_space_description(run_dir,model_path)
    expected=int(desc["preflight"].get("n_simulations",required_rows));
    if expected!=required_rows: raise RuntimeError(f"Archive has {expected}, required {required_rows}")
    cache=_find_design_cache(run_dir); raw=np.load(cache,mmap_mode="r")
    if raw.shape[0]!=required_rows: raise RuntimeError(f"Design cache rows={raw.shape[0]}, required={required_rows}")
    X,Xp,mode=_decode(np.asarray(raw),desc["bounds"]); out,diag=_extract_outcomes(run_dir,required_rows); out=out.sort_values("design_index").reset_index(drop=True); F=_feasible(out,desc["feasible_statuses"])
    replay=_replay(desc,out,Xp,n=5)
    if not replay or not all(r["pass"] for r in replay): raise RuntimeError(f"Exact growth/product replay QC failed: {replay}")
    cost=np.mean(X,axis=1); eff=1.0-cost; baseline=out.copy()
    for j,rid in enumerate(desc["features"]): baseline[rid]=Xp[:,j]
    baseline["resource_cost"]=cost; baseline["resource_efficiency"]=eff; baseline["feasible"]=F; baseline["status"]=np.where(F,"archive_exact_growth_product__deterministic_cost","infeasible")
    good=F & np.isfinite(baseline.maximum_growth)&np.isfinite(baseline.product_flux); gs=float(np.nanmax(baseline.loc[good,"maximum_growth"])); ps=float(np.nanmax(baseline.loc[good,"product_flux"]));
    if gs<=0 or ps<=0: raise RuntimeError(f"Invalid scales growth={gs}, product={ps}")
    outdir.mkdir(parents=True,exist_ok=True); bpath=outdir/"baseline_compact.csv.gz"; baseline.to_csv(bpath,index=False,compression="gzip")
    pc={"problem":{"case_name":case_key,"case_label":case.get("label",case_key),"run_dir":str(run_dir.resolve()),"model_path":str(model_path.resolve()),"case_key":case_key,"model_id":desc["manifest"].get("model_summary",{}).get("model_id"),"model_fingerprint":desc["manifest"].get("model_fingerprint"),"config_hash":desc["manifest"].get("config_hash"),"biomass_reaction_id":desc["biomass"],"product_reaction_id":desc["product"],"minimum_growth_fraction":desc["minimum_growth_fraction"],"features":desc["features"],"feature_labels":desc["feature_labels"],"bounds":{r:[float(desc["bounds"][i,0]),float(desc["bounds"][i,1])] for i,r in enumerate(desc["features"])},"growth_scale":gs,"product_scale":ps,"objective_names":["maximum_growth","growth_constrained_product_flux","resource_efficiency"],"objective_directions":["maximize","maximize","maximize"],"reported_cost_objective":"normalized_resource_burden (minimize)","resource_cost_formula":"mean_j((x_j-L_j)/(U_j-L_j))","resource_efficiency_formula":"1 - normalized_resource_burden","archive_additional_fba_solves_for_cost":0,"hv_reference_normalized":[0.0,0.0,0.0],"baseline_table":str(bpath.resolve()),"design_cache":str(cache.resolve())}}
    (outdir/"problem_config.yaml").write_text(yaml.safe_dump(pc,sort_keys=False))
    rep=dict(status="PASS",exact_replay_qc_pass=True,rows=len(baseline),feasible=int(good.sum()),dimension=len(desc["features"]),growth_scale=gs,product_scale=ps,model_path=str(model_path.resolve()),design_cache_mode=mode,design_cache=str(cache.resolve()),outcome_diagnostics=diag,exact_growth_product_replay=replay,archive_additional_fba_solves_for_cost=0,archive_replay_fba_solves=int(sum(r.get("fba_solves",0) for r in replay)),baseline_table=str(bpath.resolve()),preparation_wall_seconds=float(time.time()-t0))
    _jdump(rep,outdir/"landscape_preflight.json"); return rep

def evaluate_proposals(run_dir,model_path,proposal,output,study_config,case_key):
    desc=_space_description(run_dir,model_path); df=pd.read_csv(proposal); miss=[f for f in desc["features"] if f not in df.columns]
    if miss: raise RuntimeError(f"Proposal missing features {miss}")
    lo,hi=desc["bounds"][:,0],desc["bounds"][:,1]; rows=[]
    for _,r in df.iterrows():
        x=np.asarray([r[f] for f in desc["features"]],float); z=np.clip((x-lo)/(hi-lo),0,1); cost=float(np.mean(z)); ev=_solve_one(desc,x); ev.update(proposal_id=r.get("proposal_id",len(rows)),resource_cost=cost,resource_efficiency=1.0-cost); rows.append(ev)
    output.parent.mkdir(parents=True,exist_ok=True); pd.DataFrame(rows).to_csv(output,index=False)

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="cmd",required=True)
    a=sp.add_parser("inspect"); a.add_argument("--run-dir",required=True); a.add_argument("--model-path",required=True); a.add_argument("--output-dir",required=True); a.add_argument("--study-config",required=True); a.add_argument("--case-key",required=True); a.add_argument("--required-archive-rows",type=int,default=50000)
    e=sp.add_parser("evaluate"); e.add_argument("--run-dir",required=True); e.add_argument("--model-path",required=True); e.add_argument("--proposal",required=True); e.add_argument("--output",required=True); e.add_argument("--study-config",required=True); e.add_argument("--case-key",required=True)
    args=ap.parse_args()
    if args.cmd=="inspect":
        rep=inspect_run(Path(args.run_dir).resolve(),Path(args.model_path).resolve(),Path(args.output_dir).resolve(),Path(args.study_config).resolve(),args.case_key,args.required_archive_rows); print(json.dumps({k:rep[k] for k in ("status","rows","feasible","dimension","archive_additional_fba_solves_for_cost")},indent=2))
    else: evaluate_proposals(Path(args.run_dir).resolve(),Path(args.model_path).resolve(),Path(args.proposal).resolve(),Path(args.output).resolve(),Path(args.study_config).resolve(),args.case_key)
if __name__=="__main__": main()
