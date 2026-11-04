from __future__ import annotations

from pathlib import Path
from typing import Any
import textwrap

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .scope import apply_filters
from .visualization_planner import variable_label, target_label, variable_unit, target_unit


# Publication defaults are centralized here so every new operator inherits the same
# output contract. No operator owns styling and no filename/internal identifier is
# used as a scientific title.
PUBLICATION_RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9.5,
    "axes.titlesize": 11.5,
    "axes.labelsize": 9.5,
    "xtick.labelsize": 8.3,
    "ytick.labelsize": 8.3,
    "legend.fontsize": 8.0,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": False,
    "figure.dpi": 120,
    "savefig.transparent": False,
}


def _fmt_num(v: Any) -> str:
    try:
        x = float(v)
    except Exception:
        return str(v)
    if not np.isfinite(x):
        return "–"
    a = abs(x)
    if a != 0 and (a < 1e-3 or a >= 1e4):
        return f"{x:.2e}"
    return f"{x:.3g}"


def _label(name: str, spec: dict) -> str:
    if name in {t.get("name") for t in spec.get("targets", [])}:
        base = target_label(name, spec)
        unit = target_unit(name, spec)
    else:
        base = variable_label(name, spec)
        unit = variable_unit(name, spec)
    return f"{base} ({unit})" if unit else base


def _save(fig: plt.Figure, base: Path, formats: list[str], dpi: int) -> list[str]:
    base.parent.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for ext in formats:
        ext = str(ext).lower().lstrip(".")
        path = base.with_suffix("." + ext)
        fig.savefig(path, dpi=dpi if ext == "png" else None, bbox_inches="tight", facecolor="white")
        paths.append(str(path))
    plt.close(fig)
    return paths


def _title(fig: plt.Figure, ax: plt.Axes, viz: dict, spec: dict) -> int:
    """Place a publication title/subtitle without colliding with the axes."""
    q = str((viz.get("question") or {}).get("title") or "Exploration result")
    q = textwrap.fill(q, 72)
    n_lines = max(1, q.count("\n") + 1)
    species = str((spec.get("atlas") or {}).get("species_label") or "")
    evidence = str((viz.get("evidence") or {}).get("label") or "")
    subtitle = " · ".join(x for x in [species, evidence] if x)
    ax.set_title("")
    fig.suptitle(q, x=0.12, y=0.985, ha="left", va="top", fontsize=11.5, fontweight="bold")
    if subtitle:
        subtitle_y = 0.925 - 0.045 * (n_lines - 1)
        fig.text(0.12, subtitle_y, textwrap.fill(subtitle, 110), ha="left", va="top", fontsize=7.8)
    return n_lines


def _finalize(fig: plt.Figure, ax: plt.Axes, viz: dict, spec: dict) -> None:
    n_lines = _title(fig, ax, viz, spec)
    top = max(0.68, 0.84 - 0.045 * (n_lines - 1))
    fig.subplots_adjust(left=0.16, right=0.97, bottom=0.18, top=top)


def _result_card(result, viz: dict, ax: plt.Axes) -> None:
    ax.axis("off")
    q = (viz.get("question") or {}).get("text", "")
    ax.text(0.0, 0.97, textwrap.fill(str(q), 82), va="top", ha="left", fontsize=11.2, fontweight="bold")
    summary = dict(result.summary or {})
    rows = []
    for k in viz.get("answer_metrics") or []:
        if k not in summary:
            continue
        v = summary.get(k)
        if isinstance(v, (dict, list, tuple)):
            continue
        rows.append((k.replace("_", " ").title(), _fmt_num(v)))
    if not rows:
        for k, v in summary.items():
            if k in {"operator", "surrogate_qc", "point", "anchor", "definition", "guardrail", "evidence"} or isinstance(v, (dict, list, tuple)):
                continue
            rows.append((k.replace("_", " ").title(), _fmt_num(v)))
            if len(rows) >= 6:
                break
    if rows:
        table = ax.table(cellText=rows, colLabels=["Metric", "Value"], cellLoc="left", colLoc="left", bbox=[0.0, 0.20, 0.72, 0.48])
        table.auto_set_font_size(False); table.set_fontsize(8.8)
    purpose = str((viz.get("visualization") or {}).get("purpose") or "")
    if purpose:
        ax.text(0.0, 0.10, textwrap.fill(purpose, 95), va="bottom", ha="left", fontsize=8.2)


def _top_sorted(t: pd.DataFrame, y: str, n: int = 18) -> pd.DataFrame:
    if t.empty:
        return t
    z = t.copy()
    z[y] = pd.to_numeric(z[y], errors="coerce")
    z = z.dropna(subset=[y]).sort_values(y, key=lambda s: s.abs(), ascending=False)
    return z.head(n)


def _barh(ax: plt.Axes, labels: list[str], values: np.ndarray, xlabel: str, zero: bool = False) -> None:
    order = np.argsort(np.abs(values))
    labels = [labels[i] for i in order]
    values = values[order]
    ax.barh(np.arange(len(values)), values)
    ax.set_yticks(np.arange(len(values))); ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)
    if zero:
        ax.axvline(0, linewidth=0.8)
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)


def _distribution(ax: plt.Axes, x: pd.Series, xlabel: str, tail: str | None = None) -> None:
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    ax.hist(x, bins=45)
    if len(x):
        med = float(x.median()); q25 = float(x.quantile(.25)); q75 = float(x.quantile(.75))
        ax.axvline(med, linestyle="--", linewidth=1.2, label=f"Median {_fmt_num(med)}")
        ax.axvline(q25, linestyle=":", linewidth=0.9)
        ax.axvline(q75, linestyle=":", linewidth=0.9, label="IQR")
        if tail == "high":
            q95 = float(x.quantile(.95)); ax.axvline(q95, linewidth=1.0, label=f"q95 {_fmt_num(q95)}")
        elif tail == "low":
            q05 = float(x.quantile(.05)); ax.axvline(q05, linewidth=1.0, label=f"q05 {_fmt_num(q05)}")
        ax.legend(frameon=False)
    ax.set_xlabel(xlabel); ax.set_ylabel("Design count")
    ax.grid(axis="y", alpha=0.16, linewidth=0.5)




def _stability_distribution(ax: plt.Axes, table: pd.DataFrame, column: str, spec: dict, xlabel: str, tail: str) -> None:
    if table.empty or column not in table.columns:
        return _distribution(ax, pd.Series(dtype=float), xlabel, tail=tail)
    if "target" not in table.columns or table["target"].nunique(dropna=True) <= 1:
        return _distribution(ax, table[column], xlabel, tail=tail)
    groups = []
    labels = []
    for target, g in table.groupby("target", sort=False):
        vals = pd.to_numeric(g[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(float)
        if len(vals):
            groups.append(vals)
            labels.append(target_label(str(target), spec))
    if not groups:
        return _distribution(ax, pd.Series(dtype=float), xlabel, tail=tail)
    ax.boxplot(groups, tick_labels=labels, showfliers=False)
    ax.set_ylabel(xlabel)
    ax.set_xlabel("Observable")
    ax.tick_params(axis="x", rotation=20)
    ax.grid(axis="y", alpha=0.16, linewidth=0.5)

def _ranked_importance(result, spec, ax, table: str, y: str, sd: str | None = None, target_filter: str | None = None) -> None:
    t0 = result.tables.get(table, pd.DataFrame()).copy()
    if target_filter is not None and "target" in t0.columns:
        t0 = t0[t0["target"].astype(str) == str(target_filter)].copy()
    t = _top_sorted(t0, y, 18)
    if t.empty:
        return _result_card(result, {}, ax)
    labels = [variable_label(str(v), spec) for v in t["variable"]]
    vals = pd.to_numeric(t[y], errors="coerce").to_numpy(float)
    order = np.argsort(np.abs(vals))
    labels = [labels[i] for i in order]; vals = vals[order]
    yloc = np.arange(len(vals))
    ax.barh(yloc, vals)
    if sd and sd in t.columns:
        err = pd.to_numeric(t[sd], errors="coerce").fillna(0).to_numpy(float)[order]
        ax.errorbar(vals, yloc, xerr=err, fmt="none", capsize=2)
    ax.set_yticks(yloc); ax.set_yticklabels(labels)
    ax.set_xlabel(y.replace("_", " ").title())
    ax.grid(axis="x", alpha=0.18, linewidth=0.6)


def _interaction_matrix(result, spec, ax) -> None:
    t = result.tables.get("interactions", pd.DataFrame()).copy()
    if t.empty:
        return _result_card(result, {}, ax)
    vars_ = list(dict.fromkeys(list(t["variable_1"].astype(str)) + list(t["variable_2"].astype(str))))
    n = len(vars_); mat = np.full((n, n), np.nan, float)
    np.fill_diagonal(mat, 0.0)
    idx = {v:i for i,v in enumerate(vars_)}
    col = "mean_normalized_mixed_difference"
    for _, r in t.iterrows():
        i,j = idx[str(r["variable_1"])], idx[str(r["variable_2"])]
        mat[i,j] = mat[j,i] = float(r[col])
    im = ax.imshow(mat, aspect="equal")
    labels = [variable_label(v, spec) for v in vars_]
    ax.set_xticks(np.arange(n)); ax.set_yticks(np.arange(n))
    ax.set_xticklabels(labels, rotation=45, ha="right"); ax.set_yticklabels(labels)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=.046, pad=.04)
    cbar.set_label("Normalized mixed-difference interaction")


def _multi_stability_comparison(result, spec, ax, metric: str = "robustness") -> None:
    """Compare normalized local stability/fragility across multiple observables."""
    if metric == "fragility":
        tab = result.tables.get("fragility_scores", pd.DataFrame()).copy()
        col = "q95_local_phenotype_change"
        ylabel = "q95 local phenotype change (normalized)"
    else:
        tab = result.tables.get("robustness_scores", pd.DataFrame()).copy()
        col = "landscape_local_stability_score"
        ylabel = "Landscape-local stability score"
    if tab.empty or "target" not in tab.columns or col not in tab.columns:
        return _result_card(result, {}, ax)
    targets = [str(t) for t in result.summary.get("targets", []) if str(t) in set(tab["target"].astype(str))]
    if not targets:
        targets = list(dict.fromkeys(tab["target"].astype(str).tolist()))
    arrays=[]; labels=[]
    for t in targets:
        vals=pd.to_numeric(tab.loc[tab["target"].astype(str)==t, col], errors="coerce").dropna().to_numpy(float)
        if len(vals):
            arrays.append(vals); labels.append(_label(t,spec))
    if not arrays:
        return _result_card(result, {}, ax)
    ax.boxplot(arrays, tick_labels=labels, showfliers=False)
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=.16, linewidth=.5)


def _threshold_boundary(result, scoped_df, spec, ax) -> None:
    t = result.tables.get("boundary_points", pd.DataFrame())
    target = str(result.summary.get("target"))
    threshold = result.summary.get("threshold")
    primary = str((spec.get("atlas") or {}).get("primary_variable") or "")
    if not primary or primary not in scoped_df.columns or target not in scoped_df.columns:
        return _distribution(ax, t.get("normalized_distance_to_threshold", pd.Series(dtype=float)), "Normalized distance to threshold")
    base = scoped_df[[primary,target]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(base) > 20000:
        base = base.sample(20000, random_state=20260822)
    ax.scatter(base[primary], base[target], s=5, alpha=.12, label="Landscape")
    if not t.empty and primary in t.columns:
        ax.scatter(pd.to_numeric(t[primary], errors="coerce"), pd.to_numeric(t[target], errors="coerce"), s=18, label="Threshold-nearest designs")
    ax.axhline(float(threshold), linestyle="--", linewidth=1.2, label=f"Threshold {_fmt_num(threshold)}")
    ax.set_xlabel(_label(primary,spec)); ax.set_ylabel(_label(target,spec)); ax.legend(frameon=False)


def _regime_pair(result, scoped_df, spec, ax, x: str | None = None, y: str | None = None) -> None:
    ass = result.tables.get("cluster_assignments", pd.DataFrame())
    features = list(result.summary.get("features") or [])
    if ass.empty:
        return _result_card(result, {}, ax)
    a = str(x or (features[0] if features else ""))
    b = str(y or (features[1] if len(features) > 1 else ""))
    if not a or not b or a not in scoped_df.columns or b not in scoped_df.columns:
        return _result_card(result, {}, ax)
    joined = scoped_df[[a,b]].copy()
    joined["row_index"] = joined.index
    joined = joined.merge(ass[["row_index","cluster"]], on="row_index", how="inner")
    if len(joined) > 25000:
        joined = joined.sample(25000, random_state=20260822)
    sc = ax.scatter(pd.to_numeric(joined[a], errors="coerce"), pd.to_numeric(joined[b], errors="coerce"), c=pd.to_numeric(joined["cluster"], errors="coerce"), s=7, alpha=.35)
    prof = result.tables.get("cluster_profiles", pd.DataFrame())
    if not prof.empty and a in prof.columns and b in prof.columns:
        ax.scatter(pd.to_numeric(prof[a],errors="coerce"), pd.to_numeric(prof[b],errors="coerce"), s=70, marker="X", label="Regime median")
        ax.legend(frameon=False)
    ax.set_xlabel(_label(a,spec)); ax.set_ylabel(_label(b,spec))
    cb = ax.figure.colorbar(sc, ax=ax, fraction=.046, pad=.04); cb.set_label("Regime")
    sil = result.summary.get("silhouette")
    if sil is not None:
        ax.text(.02,.98,f"Silhouette = {_fmt_num(sil)}", transform=ax.transAxes, va="top", ha="left", fontsize=8)


def _regime_pca(result, spec, ax) -> None:
    t = result.tables.get("regime_pca_projection", pd.DataFrame()).copy()
    if t.empty or not {"PC1","PC2","cluster"}.issubset(t.columns):
        return _result_card(result, {}, ax)
    if len(t) > 25000:
        t = t.sample(25000, random_state=20260822)
    sc = ax.scatter(pd.to_numeric(t["PC1"],errors="coerce"), pd.to_numeric(t["PC2"],errors="coerce"), c=pd.to_numeric(t["cluster"],errors="coerce"), s=7, alpha=.35)
    ev = result.summary.get("pca_variance_explained") or []
    e1 = float(ev[0])*100 if len(ev)>0 else None
    e2 = float(ev[1])*100 if len(ev)>1 else None
    ax.set_xlabel(f"PC1 ({e1:.1f}% variance)" if e1 is not None else "PC1")
    ax.set_ylabel(f"PC2 ({e2:.1f}% variance)" if e2 is not None else "PC2")
    cb=ax.figure.colorbar(sc,ax=ax,fraction=.046,pad=.04); cb.set_label("Regime")
    ax.text(.02,.98,"Projection for visualization; clustering used the full standardized feature space",transform=ax.transAxes,va="top",ha="left",fontsize=7.4)


def _regime_profiles(result, spec, ax) -> None:
    t = result.tables.get("cluster_profiles_standardized", pd.DataFrame()).copy()
    if t.empty or "cluster" not in t.columns:
        return _result_card(result, {}, ax)
    features=[c for c in t.columns if c!="cluster"]
    if not features:
        return _result_card(result, {}, ax)
    # Cap display width by selecting features with greatest between-regime spread.
    if len(features)>18:
        spread=t[features].apply(pd.to_numeric,errors="coerce").max()-t[features].apply(pd.to_numeric,errors="coerce").min()
        features=list(spread.sort_values(ascending=False).head(18).index)
    mat=t[features].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    vmax=max(float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0,1e-12)
    im=ax.imshow(mat,aspect="auto",vmin=-vmax,vmax=vmax)
    ax.set_xticks(np.arange(len(features)));ax.set_xticklabels([_label(str(f),spec) for f in features],rotation=45,ha="right")
    ax.set_yticks(np.arange(len(t)));ax.set_yticklabels([f"Regime {int(c)}" for c in t["cluster"]])
    cb=ax.figure.colorbar(im,ax=ax,fraction=.046,pad=.04);cb.set_label("Standardized median feature value")


def _regime_k_selection(result, ax) -> None:
    t=result.tables.get("k_selection",pd.DataFrame()).copy()
    if t.empty:return _result_card(result,{},ax)
    x=pd.to_numeric(t["k"],errors="coerce"); y=pd.to_numeric(t["silhouette"],errors="coerce")
    ax.plot(x,y,marker="o",linewidth=1.5)
    k=result.summary.get("selected_k")
    if k is not None:
        hit=t[pd.to_numeric(t["k"],errors="coerce")==float(k)]
        if not hit.empty:
            ax.scatter([float(k)],[float(hit.iloc[0]["silhouette"])],s=45,label=f"Selected k={int(k)}")
            ax.legend(frameon=False)
    ax.set_xlabel("Number of regimes (k)");ax.set_ylabel("Silhouette score");ax.grid(alpha=.16,linewidth=.5)


def _regime_pca_loadings(result, spec, ax) -> None:
    t=result.tables.get("regime_pca_loadings",pd.DataFrame()).copy()
    if t.empty:return _result_card(result,{},ax)
    p=t.pivot(index="feature",columns="component",values="loading")
    # Keep features most important to PC1/PC2 if there are many.
    p=p[[c for c in p.columns if int(c)<=2]]
    if len(p)>18:
        score=p.abs().max(axis=1);p=p.loc[score.sort_values(ascending=False).head(18).index]
    mat=p.to_numpy(float)
    vmax=max(float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0,1e-12)
    im=ax.imshow(mat,aspect="auto",vmin=-vmax,vmax=vmax)
    ax.set_xticks(np.arange(len(p.columns)));ax.set_xticklabels([f"PC{int(c)}" for c in p.columns])
    ax.set_yticks(np.arange(len(p.index)));ax.set_yticklabels([_label(str(f),spec) for f in p.index])
    cb=ax.figure.colorbar(im,ax=ax,fraction=.046,pad=.04);cb.set_label("PCA loading")


def _regime_map(result, scoped_df, spec, ax) -> None:
    # Backward-compatible single-view alias.
    features=list(result.summary.get("features") or [])
    if len(features)==2:
        return _regime_pair(result,scoped_df,spec,ax,features[0],features[1])
    return _regime_pca(result,spec,ax)


def _tradeoff(result, scoped_df, spec, ax, pareto_only: bool = False, xcol: str | None = None, ycol: str | None = None) -> None:
    targets = list(result.summary.get("targets") or [])
    if len(targets) < 2:
        targets = [t.get("name") for t in spec.get("targets", []) if t.get("name")]
    if len(targets) < 2:
        return _result_card(result, {}, ax)
    xcol = str(xcol or targets[0]); ycol = str(ycol or targets[1])
    d = scoped_df[[xcol,ycol]].apply(pd.to_numeric, errors="coerce").dropna()
    if len(d) > 25000:
        d = d.sample(25000,random_state=20260822)
    if not pareto_only:
        ax.scatter(d[xcol],d[ycol],s=5,alpha=.13,label="Landscape")
    else:
        ax.scatter(d[xcol],d[ycol],s=4,alpha=.06,label="Landscape context")
    front = result.tables.get("pareto_points", pd.DataFrame())
    if not front.empty:
        fp = front[[xcol,ycol]].apply(pd.to_numeric,errors="coerce").dropna().sort_values(xcol)
        ax.scatter(fp[xcol],fp[ycol],s=22,label="Empirical Pareto front")
        if len(fp) > 1:
            ax.plot(fp[xcol],fp[ycol],linewidth=1.0)
    ax.set_xlabel(_label(xcol,spec)); ax.set_ylabel(_label(ycol,spec)); ax.legend(frameon=False)


def _region_comparison(result, scoped_df, spec, ax) -> None:
    p = (spec.get("operator") or {}).get("parameters") or {}
    target = str(result.summary.get("target"))
    try:
        A = apply_filters(scoped_df, p.get("region_a", [])); B = apply_filters(scoped_df, p.get("region_b", []))
        a = pd.to_numeric(A[target],errors="coerce").dropna(); b = pd.to_numeric(B[target],errors="coerce").dropna()
    except Exception:
        return _result_card(result, {}, ax)
    ax.boxplot([a.to_numpy(float), b.to_numpy(float)], tick_labels=["Region A","Region B"], showfliers=False)
    ax.set_ylabel(_label(target,spec))
    tab = result.tables.get("comparison",pd.DataFrame())
    if not tab.empty:
        r=tab.iloc[0]
        ax.text(.02,.98,f"Median difference (normalized): {_fmt_num(r.get('normalized_median_difference'))}\nMann–Whitney p: {_fmt_num(r.get('p_value'))}",transform=ax.transAxes,ha="left",va="top",fontsize=8)
    ax.grid(axis="y",alpha=.16,linewidth=.5)


def _ecdf(ax, x: np.ndarray, xlabel: str) -> None:
    x = np.asarray(x,float); x=x[np.isfinite(x)]; x=np.sort(x)
    if len(x):
        y=np.arange(1,len(x)+1)/len(x); ax.plot(x,y,linewidth=1.4)
    ax.set_xlabel(xlabel); ax.set_ylabel("Cumulative fraction of designs"); ax.grid(alpha=.16,linewidth=.5)


def _outlier_context(result, scoped_df, spec, ax) -> None:
    scores = result.tables.get("outlier_scores",pd.DataFrame())
    if scores.empty:
        return _result_card(result, {}, ax)
    candidates=[]
    if "maximum_growth" in scoped_df.columns and "product_flux" in scoped_df.columns:
        candidates=["maximum_growth","product_flux"]
    else:
        feats=list(result.summary.get("features") or [])
        candidates=[f for f in feats if f in scoped_df.columns][:2]
    if len(candidates)<2:
        return _distribution(ax,scores["outlier_score"],"Outlier score",tail="high")
    a,b=candidates
    d=scoped_df[[a,b]].copy(); d["row_index"]=d.index
    d=d.merge(scores[["row_index","outlier_score"]],on="row_index",how="inner")
    cutoff=float(d["outlier_score"].quantile(.99)); hit=d["outlier_score"]>=cutoff
    base=d if len(d)<=25000 else d.sample(25000,random_state=20260822)
    ax.scatter(pd.to_numeric(base[a],errors="coerce"),pd.to_numeric(base[b],errors="coerce"),s=5,alpha=.10,label="Landscape")
    h=d.loc[hit]
    ax.scatter(pd.to_numeric(h[a],errors="coerce"),pd.to_numeric(h[b],errors="coerce"),s=24,label="Top 1% outlier score")
    ax.set_xlabel(_label(a,spec)); ax.set_ylabel(_label(b,spec)); ax.legend(frameon=False)


def _point_query(result, scoped_df, spec, ax) -> None:
    target=str(result.summary.get("target")); primary=str((spec.get("atlas") or {}).get("primary_variable") or "")
    near=result.tables.get("nearest_landscape_points",pd.DataFrame())
    if not primary or primary not in scoped_df.columns or target not in scoped_df.columns:
        return _result_card(result, {}, ax)
    d=scoped_df[[primary,target]].apply(pd.to_numeric,errors="coerce").dropna()
    if len(d)>20000:d=d.sample(20000,random_state=20260822)
    ax.scatter(d[primary],d[target],s=5,alpha=.09,label="Landscape")
    if primary in near.columns and target in near.columns:
        ax.scatter(pd.to_numeric(near[primary],errors="coerce"),pd.to_numeric(near[target],errors="coerce"),s=24,label="Nearest exact support")
    point=result.summary.get("point") or {}; qx=point.get(primary); py=result.summary.get("surrogate_prediction")
    if qx is not None and py is not None:
        ax.scatter([float(qx)],[float(py)],s=80,marker="X",label="Query prediction")
    ax.set_xlabel(_label(primary,spec)); ax.set_ylabel(_label(target,spec)); ax.legend(frameon=False)
    ax.text(.02,.98,f"Nearest exact distance: {_fmt_num(result.summary.get('nearest_exact_distance'))}",transform=ax.transAxes,ha="left",va="top",fontsize=8)


def _response_curve(result, spec, ax, marginal: bool=False, target_filter: str | None = None) -> None:
    key="marginal_response" if marginal else "response_curve"
    t=result.tables.get(key,pd.DataFrame()).copy()
    if target_filter is not None and "target" in t.columns:
        t=t[t["target"].astype(str)==str(target_filter)].copy()
    if t.empty:return _result_card(result,{},ax)
    x=pd.to_numeric(t["variable_value"],errors="coerce")
    ycol="mean_prediction" if marginal else "prediction"; y=pd.to_numeric(t[ycol],errors="coerce")
    ax.plot(x,y,linewidth=1.6)
    if marginal and {"q10_prediction","q90_prediction"}.issubset(t.columns):
        lo=pd.to_numeric(t["q10_prediction"],errors="coerce"); hi=pd.to_numeric(t["q90_prediction"],errors="coerce")
        ax.fill_between(x,lo,hi,alpha=.16,label="10–90% across observed backgrounds")
        ax.legend(frameon=False)
    elif (not marginal) and {"exact_q10","exact_q90"}.issubset(t.columns):
        lo=pd.to_numeric(t["exact_q10"],errors="coerce"); hi=pd.to_numeric(t["exact_q90"],errors="coerce")
        ax.fill_between(x,lo,hi,alpha=.16,label="10–90% exact-landscape range within bins")
        ax.scatter(x,y,s=14,label="Exact-landscape bin median")
        ax.legend(frameon=False)
    elif (not marginal) and "ensemble_dispersion_proxy" in t.columns:
        d=pd.to_numeric(t["ensemble_dispersion_proxy"],errors="coerce").fillna(0)
        ax.fill_between(x,y-d,y+d,alpha=.16,label="Tree-ensemble dispersion proxy")
        ax.legend(frameon=False)
    var=str(result.summary.get("variable") or "variable_value"); target=str(target_filter or result.summary.get("target") or "prediction")
    ax.set_xlabel(_label(var,spec)); ax.set_ylabel(_label(target,spec)); ax.grid(alpha=.16,linewidth=.5)


def _query_context(result, scoped_df, full_df, spec, ax) -> None:
    if full_df is None:
        return _result_card(result,{},ax)
    if "maximum_growth" in full_df.columns and "product_flux" in full_df.columns:
        a,b="maximum_growth","product_flux"
    else:
        ts=[t.get("name") for t in spec.get("targets",[]) if t.get("name") and t.get("name") in full_df.columns]
        if len(ts)>=2:a,b=ts[:2]
        else:return _result_card(result,{},ax)
    bg=full_df[[a,b]].apply(pd.to_numeric,errors="coerce").dropna()
    if len(bg)>25000:bg=bg.sample(25000,random_state=20260822)
    sel=scoped_df[[a,b]].apply(pd.to_numeric,errors="coerce").dropna()
    if len(sel)>10000:sel=sel.sample(10000,random_state=20260822)
    ax.scatter(bg[a],bg[b],s=5,alpha=.07,label="Full landscape")
    ax.scatter(sel[a],sel[b],s=8,alpha=.35,label="Query-matching scope")
    ax.set_xlabel(_label(a,spec));ax.set_ylabel(_label(b,spec));ax.legend(frameon=False)
    ax.text(.02,.98,f"Matched {len(scoped_df):,} of {len(full_df):,} designs ({len(scoped_df)/max(len(full_df),1):.1%})",transform=ax.transAxes,ha="left",va="top",fontsize=8)


def _correlation(result,spec,ax) -> None:
    t=result.tables.get("spearman_matrix",pd.DataFrame()).copy()
    if t.empty:return _result_card(result,{},ax)
    idx="feature" if "feature" in t.columns else t.columns[0]
    labels=[_label(str(v),spec) for v in t[idx].astype(str)]
    cols=[c for c in t.columns if c!=idx]; mat=t[cols].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    im=ax.imshow(mat,vmin=-1,vmax=1,aspect="equal")
    ax.set_xticks(np.arange(len(cols)));ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels([_label(str(c),spec) for c in cols],rotation=45,ha="right");ax.set_yticklabels(labels)
    cb=ax.figure.colorbar(im,ax=ax,fraction=.046,pad=.04);cb.set_label("Spearman ρ")


def _sensitivity_heatmap(result,spec,ax) -> None:
    t=result.tables.get("sensitivity_hotspots",pd.DataFrame()).copy()
    if t.empty:return _result_card(result,{},ax)
    sens=[c for c in t.columns if str(c).startswith("sensitivity__")]
    if not sens:return _result_card(result,{},ax)
    t=t.sort_values("gradient_l2_normalized",ascending=False).head(30)
    mat=t[sens].apply(pd.to_numeric,errors="coerce").to_numpy(float)
    vmax=max(float(np.nanmax(np.abs(mat))) if np.isfinite(mat).any() else 1.0,1e-12)
    im=ax.imshow(mat,aspect="auto",vmin=-vmax,vmax=vmax)
    vars_=[c[len("sensitivity__"):] for c in sens]
    ax.set_xticks(np.arange(len(vars_)));ax.set_xticklabels([variable_label(v,spec) for v in vars_],rotation=45,ha="right")
    ax.set_ylabel("High-gradient landscape contexts (ranked)")
    ax.set_yticks([])
    cb=ax.figure.colorbar(im,ax=ax,fraction=.046,pad=.04);cb.set_label("Signed normalized local sensitivity")


def _dimensionality(result,ax) -> None:
    t=result.tables.get("pca_variance",pd.DataFrame())
    if t.empty:return _result_card(result,{},ax)
    x=pd.to_numeric(t["component"],errors="coerce")
    ev=pd.to_numeric(t["explained_variance_ratio"],errors="coerce")
    cum=pd.to_numeric(t["cumulative_explained_variance"],errors="coerce")
    ax.bar(x,ev,label="Per-component variance")
    ax.plot(x,cum,marker="o",linewidth=1.4,label="Cumulative variance")
    th=float(result.summary.get("variance_threshold",.9));ax.axhline(th,linestyle="--",linewidth=1.0,label=f"Target {th:.0%}")
    ax.set_xlabel("Principal component");ax.set_ylabel("Explained variance fraction");ax.set_ylim(0,1.03);ax.legend(frameon=False)


def _surrogate_support(result,scoped_df,spec,ax) -> None:
    t=result.tables.get("surrogate_dispersion_all",pd.DataFrame()).copy()
    primary=str((spec.get("atlas") or {}).get("primary_variable") or ""); secondary=str((spec.get("atlas") or {}).get("secondary_variable") or "")
    if t.empty:return _result_card(result,{},ax)
    if primary in scoped_df.columns and secondary in scoped_df.columns and primary!=secondary:
        d=scoped_df[[primary,secondary]].copy();d["row_index"]=d.index
        d=d.merge(t[["row_index","normalized_ensemble_dispersion"]],on="row_index",how="inner")
        if len(d)>12000:d=d.sample(12000,random_state=20260822)
        sc=ax.scatter(pd.to_numeric(d[primary],errors="coerce"),pd.to_numeric(d[secondary],errors="coerce"),c=pd.to_numeric(d["normalized_ensemble_dispersion"],errors="coerce"),s=8,alpha=.5)
        ax.set_xlabel(_label(primary,spec));ax.set_ylabel(_label(secondary,spec))
        cb=ax.figure.colorbar(sc,ax=ax,fraction=.046,pad=.04);cb.set_label("Normalized ensemble dispersion proxy")
    else:
        ax.scatter(pd.to_numeric(t["observed_target"],errors="coerce"),pd.to_numeric(t["surrogate_prediction"],errors="coerce"),c=pd.to_numeric(t["normalized_ensemble_dispersion"],errors="coerce"),s=8,alpha=.5)
        ax.set_xlabel("Observed target");ax.set_ylabel("Surrogate prediction")


def _saturation(result,ax,kind:str) -> None:
    if kind=="pareto":
        t=result.tables.get("pareto_saturation_summary",pd.DataFrame());xcol="sample_size";ycol="median_igd_plus"
        if t.empty:return _result_card(result,{},ax)
        x=pd.to_numeric(t[xcol],errors="coerce");y=pd.to_numeric(t[ycol],errors="coerce")
        ax.plot(x,y,marker="o",linewidth=1.5,label="Median IGD+")
        if "q90_igd_plus" in t.columns:
            hi=pd.to_numeric(t["q90_igd_plus"],errors="coerce");ax.fill_between(x,y,hi,alpha=.16,label="Median–q90 across replicates")
        ax.set_ylabel("IGD+ to full empirical Pareto front (lower is better)")
    else:
        t=result.tables.get("sampling_saturation_summary",pd.DataFrame());xcol="sample_size";ycol="median_q95_cover_distance"
        if t.empty:return _result_card(result,{},ax)
        x=pd.to_numeric(t[xcol],errors="coerce");y=pd.to_numeric(t[ycol],errors="coerce")
        ax.plot(x,y,marker="o",linewidth=1.5,label="Median q95 cover distance")
        if "sd_q95_cover_distance" in t.columns:
            sd=pd.to_numeric(t["sd_q95_cover_distance"],errors="coerce").fillna(0);ax.fill_between(x,np.maximum(0,y-sd),y+sd,alpha=.16,label="±1 SD across replicates")
        ax.set_ylabel("Normalized q95 nearest-subsample distance (lower is better)")
    ax.set_xscale("log");ax.set_xlabel("Landscape sample size");ax.legend(frameon=False);ax.grid(alpha=.16,linewidth=.5)


def _controller(result,spec,ax,threshold=False) -> None:
    if threshold:
        t=result.tables.get("threshold_controller_ranking",pd.DataFrame()).copy(); y="switch_score"
    else:
        t=result.tables.get("controller_ranking",pd.DataFrame()).copy(); y="median_controller_score"
    t=_top_sorted(t,y,18)
    if t.empty:return _result_card(result,{},ax)
    labels=[variable_label(str(v),spec) for v in t["variable"]];vals=pd.to_numeric(t[y],errors="coerce").to_numpy(float)
    order=np.argsort(vals);labels=[labels[i] for i in order];vals=vals[order]
    yy=np.arange(len(vals));ax.barh(yy,vals);ax.set_yticks(yy);ax.set_yticklabels(labels);ax.set_xlabel("Threshold-switch score" if threshold else "Median controller score")
    if threshold and "crossing_fraction" in t.columns:
        frac=pd.to_numeric(t["crossing_fraction"],errors="coerce").to_numpy(float)[order]
        for yloc,v,f in zip(yy,vals,frac):ax.text(v,yloc,f"  {f:.0%} cross",va="center",fontsize=7)
    elif (not threshold) and "q95_controller_score" in t.columns:
        q95=pd.to_numeric(t["q95_controller_score"],errors="coerce").to_numpy(float)[order];ax.scatter(q95,yy,s=15,label="q95 across backgrounds");ax.legend(frameon=False)
    ax.grid(axis="x",alpha=.16,linewidth=.5)


def _surface(result,spec,ax,target_filter: str | None = None) -> None:
    t=result.tables.get("response_surface",pd.DataFrame()).copy()
    if target_filter is not None and "target" in t.columns:
        t=t[t["target"].astype(str)==str(target_filter)].copy()
    if t.empty or "prediction" not in t.columns:
        return _result_card(result,{},ax)
    auxiliary={"target","prediction","ensemble_dispersion_proxy","exact_q10","exact_q90","exact_n"}
    coord_cols=[str(c) for c in t.columns if str(c) not in auxiliary]
    if len(coord_cols)!=2:
        raise RuntimeError(
            "response_surface visualization expected exactly two coordinate columns; "
            f"found {coord_cols!r} in {list(t.columns)!r}"
        )
    x,y=coord_cols
    pivot=t.pivot(index=y,columns=x,values="prediction")
    xs=np.asarray(pivot.columns,float);ys=np.asarray(pivot.index,float);z=pivot.to_numpy(float)
    im=ax.imshow(z,origin="lower",aspect="auto",extent=[xs.min(),xs.max(),ys.min(),ys.max()])
    try:ax.contour(xs,ys,z,levels=7,linewidths=.6)
    except Exception:pass
    ax.set_xlabel(_label(x,spec));ax.set_ylabel(_label(y,spec));cb=ax.figure.colorbar(im,ax=ax,fraction=.046,pad=.04);cb.set_label(_label(str(target_filter or result.summary.get("target") or "prediction"),spec))


def _view_label(view: dict[str, Any], spec: dict) -> str:
    if view.get("target"):
        return _label(str(view["target"]), spec)
    if view.get("x") and view.get("y"):
        return f"{_label(str(view['x']), spec)} vs {_label(str(view['y']), spec)}"
    fam=str(view.get("family") or "")
    names={
        "regime_pca":"PCA regime projection",
        "regime_profiles":"Regime feature profiles",
        "regime_k_selection":"Regime-count selection",
        "regime_pca_loadings":"PCA loadings",
        "multi_stability_comparison":"Cross-observable stability comparison",
        "pareto_frontier":"Pareto frontier",
        "tradeoff_frontier":"Trade-off frontier",
    }
    return names.get(fam, fam.replace("_"," ").title())


def render(
    result,
    df: pd.DataFrame,
    outdir: Path,
    spec: dict,
    viz_spec: dict | None = None,
    full_df: pd.DataFrame | None = None,
) -> list[str]:
    if not spec.get("outputs", {}).get("save_figures", True):
        return []
    viz = viz_spec or {"visualization":{"family":"summary"},"question":{"title":spec.get("name","Exploration result")},"evidence":{}}
    vcfg = viz.get("visualization") or {}
    default_family = str(vcfg.get("family") or "summary")
    views = list(vcfg.get("views") or [{"key":"main","family":default_family}])
    formats = list(spec.get("outputs", {}).get("publication_formats", ["png", "pdf", "svg"]))
    dpi = int(spec.get("outputs", {}).get("dpi", 600))
    all_paths: list[str] = []

    with plt.rc_context(PUBLICATION_RC):
        for i, view in enumerate(views):
            family = str(view.get("family") or default_family)
            key = str(view.get("key") or ("main" if i == 0 else f"view_{i+1}"))
            fig, ax = plt.subplots(figsize=(7.4, 4.9), constrained_layout=False)
            try:
                if family == "distribution":
                    target = str(view.get("target") or (result.summary.get("targets", [None])[0] if isinstance(result.summary.get("targets"),list) else result.summary.get("target")))
                    if target not in df.columns:
                        target=(spec.get("targets") or [{}])[0].get("name")
                    _distribution(ax,df[target],_label(str(target),spec))
                elif family in {"ranked_effect","ranked_importance"}:
                    tf=view.get("target")
                    if "sensitivity" in result.tables:
                        _ranked_importance(result,spec,ax,"sensitivity","heldout_permutation_r2_drop_mean","heldout_permutation_r2_drop_sd",target_filter=tf)
                    elif "associations" in result.tables:
                        _ranked_importance(result,spec,ax,"associations","spearman_rho",target_filter=tf)
                    else:_result_card(result,viz,ax)
                elif family == "signed_effect":
                    t=_top_sorted(result.tables.get("local_sensitivity",pd.DataFrame()),"normalized_derivative_per_full_range",18)
                    if t.empty:_result_card(result,viz,ax)
                    else:_barh(ax,[variable_label(str(v),spec) for v in t["variable"]],pd.to_numeric(t["normalized_derivative_per_full_range"],errors="coerce").to_numpy(float),"Signed normalized local sensitivity",zero=True)
                elif family == "local_sensitivity_heatmap":_sensitivity_heatmap(result,spec,ax)
                elif family == "interaction_matrix":_interaction_matrix(result,spec,ax)
                elif family == "fragility_distribution":
                    tab=result.tables.get("fragility_scores",pd.DataFrame()).copy()
                    if view.get("target") and "target" in tab.columns: tab=tab[tab["target"].astype(str)==str(view["target"])]
                    _stability_distribution(ax,tab,"q95_local_phenotype_change",spec,"q95 local phenotype change (normalized)",tail="high")
                elif family == "robustness_distribution":
                    tab=result.tables.get("robustness_scores",pd.DataFrame()).copy()
                    if view.get("target") and "target" in tab.columns: tab=tab[tab["target"].astype(str)==str(view["target"])]
                    _stability_distribution(ax,tab,"landscape_local_stability_score",spec,"Landscape-local stability score",tail="low")
                elif family == "multi_stability_comparison":
                    _multi_stability_comparison(result,spec,ax,str(view.get("metric") or "robustness"))
                elif family == "threshold_region_effects":
                    t=_top_sorted(result.tables.get("variable_shifts",pd.DataFrame()),"median_shift",18)
                    if t.empty:_result_card(result,viz,ax)
                    else:_barh(ax,[variable_label(str(v),spec) for v in t["variable"]],pd.to_numeric(t["median_shift"],errors="coerce").to_numpy(float),"Median shift: threshold-meeting minus other designs",zero=True)
                elif family == "threshold_boundary_context":_threshold_boundary(result,df,spec,ax)
                elif family == "feasibility_drivers":
                    t=result.tables.get("feasibility_importance",pd.DataFrame())
                    if t.empty:
                        ax.axis("off");ax.text(.5,.58,"No feasibility boundary exists in this scope",ha="center",va="center",fontsize=12,fontweight="bold");ax.text(.5,.43,f"Feasible fraction: {_fmt_num(result.summary.get('feasible_fraction'))}",ha="center",va="center",fontsize=9)
                    else:_ranked_importance(result,spec,ax,"feasibility_importance","heldout_permutation_auc_drop_mean","heldout_permutation_auc_drop_sd")
                elif family in {"regime_map","regime_pair"}:_regime_pair(result,df,spec,ax,view.get("x"),view.get("y"))
                elif family == "regime_pca":_regime_pca(result,spec,ax)
                elif family == "regime_profiles":_regime_profiles(result,spec,ax)
                elif family == "regime_k_selection":_regime_k_selection(result,ax)
                elif family == "regime_pca_loadings":_regime_pca_loadings(result,spec,ax)
                elif family in {"tradeoff_frontier","pareto_frontier"}:_tradeoff(result,df,spec,ax,pareto_only=(family=="pareto_frontier"),xcol=view.get("x"),ycol=view.get("y"))
                elif family == "region_comparison":_region_comparison(result,df,spec,ax)
                elif family == "coverage_distribution":_ecdf(ax,pd.to_numeric(result.tables.get("nearest_neighbor_distances",pd.DataFrame()).get("nearest_neighbor_distance",pd.Series(dtype=float)),errors="coerce").to_numpy(float),"Normalized nearest-neighbor distance")
                elif family == "outlier_context":_outlier_context(result,df,spec,ax)
                elif family == "point_query_context":_point_query(result,df,spec,ax)
                elif family == "response_curve":_response_curve(result,spec,ax,False,target_filter=view.get("target"))
                elif family == "marginal_response":_response_curve(result,spec,ax,True,target_filter=view.get("target"))
                elif family == "query_context":_query_context(result,df,full_df,spec,ax)
                elif family == "correlation_heatmap":_correlation(result,spec,ax)
                elif family == "dimensionality_curve":_dimensionality(result,ax)
                elif family == "surrogate_support_map":_surrogate_support(result,df,spec,ax)
                elif family == "pareto_saturation":_saturation(result,ax,"pareto")
                elif family == "sampling_saturation":_saturation(result,ax,"sampling")
                elif family == "controller_ranking":_controller(result,spec,ax,False)
                elif family == "threshold_switch_ranking":_controller(result,spec,ax,True)
                elif family == "response_surface":_surface(result,spec,ax,target_filter=view.get("target"))
                else:_result_card(result,viz,ax)

                # Keep the scientific question as the main title and identify the
                # complementary view underneath it when multiple figures are emitted.
                local_viz = dict(viz)
                local_viz["question"] = dict(viz.get("question") or {})
                if len(views) > 1:
                    base_title=str(local_viz["question"].get("title") or "Exploration result")
                    local_viz["question"]["title"] = f"{base_title} — {_view_label(view,spec)}"
                _finalize(fig,ax,local_viz,spec)
                base=outdir/"figures"/("main" if key=="main" else key)
                all_paths.extend(_save(fig,base,formats,dpi))
            except Exception:
                plt.close(fig);raise
    return all_paths

