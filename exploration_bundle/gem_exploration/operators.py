from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu, pearsonr, spearmanr
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.inspection import permutation_importance
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .metrics import igd_plus, normalize_matrix, normalized_oriented_targets, pareto_mask, robust_range, hypervolume_2d
from .scope import apply_filters
from .surrogate import fit_surrogate, finite_difference_gradient


@dataclass
class AnalysisResult:
    summary: dict[str, Any]
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    plot_data: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def _num(df: pd.DataFrame, c: str) -> pd.Series:
    return pd.to_numeric(df[c], errors="coerce")


def _target(spec: dict[str, Any]) -> str | None:
    return spec["targets"][0]["name"] if spec.get("targets") else None


def _numeric_frame(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df[columns].copy()
    for c in columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan).dropna()


def _safe_corr(fn, x: pd.Series, y: pd.Series) -> tuple[float, float]:
    if x.nunique(dropna=True) < 2 or y.nunique(dropna=True) < 2:
        return float("nan"), float("nan")
    try:
        r = fn(x, y)
        return float(r.statistic), float(r.pvalue)
    except Exception:
        return float("nan"), float("nan")


def describe(df, spec, variables, ctx):
    targets = [t["name"] for t in spec.get("targets", [])] or variables
    rows = []
    for c in targets:
        x = _num(df, c)
        q = x.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
        rows.append(
            {
                "target": c,
                "n_finite": int(x.notna().sum()),
                "mean": x.mean(),
                "std": x.std(),
                "min": x.min(),
                "q05": q.loc[0.05],
                "q25": q.loc[0.25],
                "median": q.loc[0.5],
                "q75": q.loc[0.75],
                "q95": q.loc[0.95],
                "max": x.max(),
            }
        )
    tab = pd.DataFrame(rows)
    return AnalysisResult(
        {"operator": "describe", "rows": len(df), "targets": targets},
        {"descriptive_statistics": tab},
        {"kind": "hist", "target": targets[0] if targets else None},
    )


def query(df, spec, variables, ctx):
    p = spec["operator"].get("parameters", {})
    out = apply_filters(df, p.get("filters", []) or [])
    sort_by = p.get("sort_by")
    if sort_by:
        if sort_by not in out.columns:
            raise KeyError(f"query sort_by column {sort_by!r} not found")
        out = out.sort_values(sort_by, ascending=bool(p.get("ascending", False)))
    columns = p.get("columns") or []
    if columns:
        missing = [c for c in columns if c not in out.columns]
        if missing:
            raise KeyError(f"query columns not found: {missing}")
        keep = ["sample_id"] if "sample_id" in out.columns and "sample_id" not in columns else []
        out = out[[*keep, *columns]]
    total = len(out)
    limit = int(p.get("max_saved_points", 10000))
    saved = out.head(limit).copy()
    return AnalysisResult(
        {
            "operator": "query",
            "matched_rows": int(total),
            "fraction_of_scope": float(total / max(len(df), 1)),
            "saved_rows": int(len(saved)),
            "truncated": bool(total > limit),
        },
        {"matched_points": saved},
        {"kind": "none"},
    )


def association(df, spec, variables, ctx):
    targets = [t["name"] for t in spec.get("targets", [])]
    if not targets:
        raise ValueError("association requires at least one target")
    rows = []
    for t in targets:
        y = _num(df, t)
        for v in variables:
            x = _num(df, v)
            ok = x.notna() & y.notna()
            if ok.sum() < 5:
                continue
            pr, pp = _safe_corr(pearsonr, x[ok], y[ok])
            sr, sp = _safe_corr(spearmanr, x[ok], y[ok])
            rows.append(
                {
                    "target": t,
                    "variable": v,
                    "pearson_r": pr,
                    "pearson_p": pp,
                    "spearman_rho": sr,
                    "spearman_p": sp,
                    "n": int(ok.sum()),
                }
            )
    tab = pd.DataFrame(rows)
    if not tab.empty:
        tab = tab.assign(_abs=tab["spearman_rho"].abs()).sort_values(["target", "_abs"], ascending=[True, False]).drop(columns="_abs")
    return AnalysisResult(
        {
            "operator": "association",
            "target": targets[0],
            "targets": targets,
            "n_variables": int(tab["variable"].nunique()) if not tab.empty else 0,
        },
        {"associations": tab},
        {"kind": "bar", "table": "associations", "x": "variable", "y": "spearman_rho"},
    )


def correlation_matrix(df, spec, variables, ctx):
    p = spec["operator"].get("parameters", {})
    requested = p.get("features") or []
    features = list(requested) if requested else list(dict.fromkeys([*variables, *[t["name"] for t in spec.get("targets", [])]]))
    max_features = int(p.get("max_features", 40))
    if len(features) > max_features:
        raise ValueError(
            f"correlation_matrix received {len(features)} features; max_features={max_features}. "
            "Select a scientifically meaningful subset explicitly."
        )
    d = _numeric_frame(df, features)
    if len(d) < 5:
        raise RuntimeError("Insufficient finite rows for correlation matrix")
    pearson = d.corr(method="pearson")
    spearman = d.corr(method="spearman")
    long_rows = []
    for i, a in enumerate(features):
        for b in features[i + 1 :]:
            long_rows.append(
                {
                    "feature_1": a,
                    "feature_2": b,
                    "pearson_r": float(pearson.loc[a, b]),
                    "spearman_rho": float(spearman.loc[a, b]),
                }
            )
    pairwise = pd.DataFrame(long_rows)
    if not pairwise.empty:
        pairwise = pairwise.sort_values("spearman_rho", key=lambda s: s.abs(), ascending=False)
    return AnalysisResult(
        {"operator": "correlation_matrix", "features": features, "n_rows": len(d)},
        {
            "pearson_matrix": pearson.reset_index().rename(columns={"index": "feature"}),
            "spearman_matrix": spearman.reset_index().rename(columns={"index": "feature"}),
            "pairwise_correlations": pairwise,
        },
        {"kind": "heatmap", "table": "spearman_matrix", "index": "feature"},
    )


def _heldout_permutation_regression(df, variables, target, bundle, seed):
    d = _numeric_frame(df, variables + [target])
    if len(d) > 20000:
        d = d.sample(20000, random_state=seed)
    train, test = train_test_split(d, test_size=0.25, random_state=seed)
    model = clone(bundle.model)
    model.fit(train[variables].to_numpy(float), train[target].to_numpy(float))
    imp = permutation_importance(
        model,
        test[variables].to_numpy(float),
        test[target].to_numpy(float),
        n_repeats=10,
        random_state=seed,
        scoring="r2",
        n_jobs=int(bundle.model.n_jobs if getattr(bundle.model, "n_jobs", None) is not None else -1),
    )
    return d, test, imp


def global_sensitivity(df, spec, variables, ctx):
    t = _target(spec)
    cfg = spec["surrogate"]
    bundle = fit_surrogate(df, variables, t, cfg, seed=ctx["seed"])
    d, test, imp = _heldout_permutation_regression(df, variables, t, bundle, ctx["seed"])
    y = d[t]
    rows = []
    for j, v in enumerate(variables):
        rho, _ = _safe_corr(spearmanr, d[v], y)
        rows.append(
            {
                "variable": v,
                "heldout_permutation_r2_drop_mean": float(imp.importances_mean[j]),
                "heldout_permutation_r2_drop_sd": float(imp.importances_std[j]),
                "spearman_rho": rho,
            }
        )
    tab = pd.DataFrame(rows).sort_values("heldout_permutation_r2_drop_mean", ascending=False)
    return AnalysisResult(
        {
            "operator": "global_sensitivity",
            "target": t,
            "surrogate_qc": bundle.metrics,
            "importance_holdout_rows": int(len(test)),
            "metric": "cross-validated surrogate QC + independent held-out permutation importance; not classical Sobol indices",
        },
        {"sensitivity": tab},
        {
            "kind": "bar",
            "table": "sensitivity",
            "x": "variable",
            "y": "heldout_permutation_r2_drop_mean",
        },
    )


def _anchor(spec, df, variables):
    params = spec["operator"].get("parameters", {})
    if "point" in params:
        p = params["point"]
        values = []
        for v in variables:
            raw = p.get(v, p.get(v.replace("requested_medium__", ""), np.nan))
            values.append(float(raw))
        out = np.asarray(values, float)
        if not np.isfinite(out).all():
            raise ValueError("Anchor point must specify every resolved design variable")
        return out
    if "sample_id" in params:
        if "sample_id" not in df.columns:
            raise KeyError("sample_id is unavailable in the landscape")
        hit = df[df.sample_id == int(params["sample_id"])]
        if hit.empty:
            raise KeyError("Requested sample_id not found")
        return hit.iloc[0][variables].to_numpy(float)
    return df[variables].apply(pd.to_numeric, errors="coerce").median().to_numpy(float)


def local_sensitivity(df, spec, variables, ctx):
    t = _target(spec)
    bundle = fit_surrogate(df, variables, t, spec["surrogate"], seed=ctx["seed"])
    p = _anchor(spec, df, variables)
    grad = finite_difference_gradient(bundle, p, float(spec["operator"].get("parameters", {}).get("step_fraction", 0.01)))
    span = bundle.bounds[:, 1] - bundle.bounds[:, 0]
    normalized = grad * span / max(robust_range(_num(df, t).to_numpy(float)), 1e-12)
    tab = pd.DataFrame(
        {
            "variable": variables,
            "derivative": grad,
            "normalized_derivative_per_full_range": normalized,
            "abs_normalized_sensitivity": np.abs(normalized),
        }
    ).sort_values("abs_normalized_sensitivity", ascending=False)
    pred, disp = bundle.predict(p.reshape(1, -1))
    return AnalysisResult(
        {
            "operator": "local_sensitivity",
            "target": t,
            "anchor": dict(zip(variables, p)),
            "surrogate_prediction": float(pred[0]),
            "ensemble_dispersion_proxy": None if disp is None else float(disp[0]),
            "surrogate_qc": bundle.metrics,
        },
        {"local_sensitivity": tab},
        {
            "kind": "bar",
            "table": "local_sensitivity",
            "x": "variable",
            "y": "normalized_derivative_per_full_range",
        },
        ["Ensemble-tree dispersion is a heuristic support/uncertainty proxy, not a calibrated confidence interval."],
    )


def sensitivity_map(df, spec, variables, ctx):
    t = _target(spec)
    p = spec["operator"].get("parameters", {})
    bundle = fit_surrogate(df, variables, t, spec["surrogate"], seed=ctx["seed"])
    d = _numeric_frame(df, variables)
    n = min(int(p.get("anchor_count", 300)), len(d))
    anchors = d.sample(n, random_state=ctx["seed"])
    span = bundle.bounds[:, 1] - bundle.bounds[:, 0]
    yscale = max(robust_range(_num(df, t).to_numpy(float)), 1e-12)
    rows = []
    per_var: dict[str, list[float]] = {v: [] for v in variables}
    step = float(p.get("step_fraction", 0.01))
    for idx, row in anchors.iterrows():
        point = row.to_numpy(float)
        grad = finite_difference_gradient(bundle, point, step)
        norm = grad * span / yscale
        j = int(np.argmax(np.abs(norm)))
        rec = {
            "row_index": int(idx) if isinstance(idx, (int, np.integer)) else str(idx),
            "dominant_variable": variables[j],
            "gradient_l2_normalized": float(np.linalg.norm(norm)),
            "max_abs_normalized_sensitivity": float(np.max(np.abs(norm))),
        }
        if "sample_id" in df.columns and idx in df.index:
            rec["sample_id"] = df.loc[idx, "sample_id"]
        for k, v in enumerate(variables):
            rec[f"sensitivity__{v}"] = float(norm[k])
            per_var[v].append(abs(float(norm[k])))
        rows.append(rec)
    points = pd.DataFrame(rows).sort_values("gradient_l2_normalized", ascending=False)
    var_summary = pd.DataFrame(
        [
            {
                "variable": v,
                "median_abs_local_sensitivity": float(np.median(per_var[v])),
                "q95_abs_local_sensitivity": float(np.quantile(per_var[v], 0.95)),
                "max_abs_local_sensitivity": float(np.max(per_var[v])),
            }
            for v in variables
        ]
    ).sort_values("q95_abs_local_sensitivity", ascending=False)
    return AnalysisResult(
        {
            "operator": "sensitivity_map",
            "target": t,
            "anchors": n,
            "surrogate_qc": bundle.metrics,
            "definition": "local finite-difference surrogate gradients across sampled contexts, normalized by full input range and robust target range",
        },
        {"sensitivity_hotspots": points, "variable_local_sensitivity_summary": var_summary},
        {
            "kind": "bar",
            "table": "variable_local_sensitivity_summary",
            "x": "variable",
            "y": "q95_abs_local_sensitivity",
        },
    )


def interaction(df, spec, variables, ctx):
    t = _target(spec)
    bundle = fit_surrogate(df, variables, t, spec["surrogate"], seed=ctx["seed"])
    params = spec["operator"].get("parameters", {})
    hfrac = float(params.get("step_fraction", 0.03))
    all_pairs = list(combinations(range(len(variables)), 2))
    max_pairs = int(params.get("max_pairs", len(all_pairs)))
    pairs = all_pairs[:max_pairs]
    d = _numeric_frame(df, variables)
    anchors_df = d.sample(min(int(params.get("anchor_count", 128)), len(d)), random_state=ctx["seed"])
    anchors = anchors_df.to_numpy(float)
    span = bundle.bounds[:, 1] - bundle.bounds[:, 0]
    yr = robust_range(_num(df, t).to_numpy(float))
    rows = []
    hotspot_rows = []
    for i, j in pairs:
        vals = []
        hi = max(span[i] * hfrac, 1e-9)
        hj = max(span[j] * hfrac, 1e-9)
        for aidx, point in enumerate(anchors):
            pts = []
            for si, sj in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                q = point.copy()
                q[i] = np.clip(q[i] + si * hi, bundle.bounds[i, 0], bundle.bounds[i, 1])
                q[j] = np.clip(q[j] + sj * hj, bundle.bounds[j, 0], bundle.bounds[j, 1])
                pts.append(q)
            pred, _ = bundle.predict(np.vstack(pts))
            val = abs(pred[0] - pred[1] - pred[2] + pred[3]) / max(yr, 1e-12)
            vals.append(float(val))
        arr = np.asarray(vals, float)
        top = int(np.argmax(arr))
        rows.append(
            {
                "variable_1": variables[i],
                "variable_2": variables[j],
                "mean_normalized_mixed_difference": float(np.mean(arr)),
                "q95_normalized_mixed_difference": float(np.quantile(arr, 0.95)),
                "max_normalized_mixed_difference": float(np.max(arr)),
            }
        )
        hotspot_rows.append(
            {
                "variable_1": variables[i],
                "variable_2": variables[j],
                "anchor_row_index": anchors_df.index[top],
                "normalized_mixed_difference": float(arr[top]),
                **{f"anchor__{v}": float(anchors[top, k]) for k, v in enumerate(variables)},
            }
        )
    tab = pd.DataFrame(rows).sort_values("mean_normalized_mixed_difference", ascending=False)
    hotspots = pd.DataFrame(hotspot_rows).sort_values("normalized_mixed_difference", ascending=False)
    return AnalysisResult(
        {
            "operator": "interaction",
            "target": t,
            "surrogate_qc": bundle.metrics,
            "pairs_evaluated": len(tab),
            "definition": "surrogate mixed finite-difference interaction in normalized phenotype units",
        },
        {"interactions": tab, "interaction_hotspots": hotspots},
        {"kind": "bar_pair", "table": "interactions", "y": "mean_normalized_mixed_difference"},
    )


def _neighbor_change(df, variables, target, k=20):
    X, _, _ = normalize_matrix(df, variables)
    y = _num(df, target).to_numpy(float)
    ok = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    idx = np.where(ok)[0]
    Xg = X[ok]
    yg = y[ok]
    tree = cKDTree(Xg)
    kk = min(k + 1, len(Xg))
    dist, nn = tree.query(Xg, k=kk)
    yr = robust_range(yg)
    rows = []
    for z, orig in enumerate(idx):
        neigh = nn[z, 1:] if nn.ndim > 1 else []
        changes = np.abs(yg[np.asarray(neigh, int)] - yg[z]) / yr if len(neigh) else np.array([np.nan])
        rows.append(
            {
                "row_index": int(orig),
                "sample_id": df.iloc[orig].get("sample_id", orig),
                "q95_local_phenotype_change": float(np.nanquantile(changes, 0.95)),
                "median_neighbor_distance": float(np.nanmedian(dist[z, 1:])) if np.ndim(dist) > 1 else np.nan,
                "n_neighbors": len(neigh),
            }
        )
    return pd.DataFrame(rows)


def fragility(df, spec, variables, ctx):
    targets = [t["name"] for t in spec.get("targets", [])]
    if not targets:
        raise ValueError("fragility requires at least one target")
    k = int(spec["operator"].get("parameters", {}).get("neighbors", 20))
    tabs = []
    for t in targets:
        z = _neighbor_change(df, variables, t, k)
        z.insert(0, "target", t)
        tabs.append(z)
    tab = pd.concat(tabs, ignore_index=True).sort_values("q95_local_phenotype_change", ascending=False)
    return AnalysisResult(
        {
            "operator": "fragility",
            "target": targets[0],
            "targets": targets,
            "neighbors": k,
            "definition": "landscape-local q95 absolute phenotype change normalized by each target's robust range",
        },
        {"fragility_scores": tab},
        {"kind": "hist_table", "table": "fragility_scores", "column": "q95_local_phenotype_change"},
        [
            "This is a sampled-landscape local fragility statistic. It is not equivalent to an exact perturbation robustness experiment unless the neighborhood design supports that interpretation."
        ],
    )


def robustness(df, spec, variables, ctx):
    res = fragility(df, spec, variables, ctx)
    tab = res.tables.pop("fragility_scores").copy()
    tab["landscape_local_stability_score"] = 1.0 - tab["q95_local_phenotype_change"].clip(0, 1)
    tab = tab.sort_values("landscape_local_stability_score", ascending=False)
    res.summary["operator"] = "robustness"
    res.summary["definition"] = "1 - clipped landscape-local fragility; descriptive stability index"
    res.tables = {"robustness_scores": tab}
    res.plot_data = {"kind": "hist_table", "table": "robustness_scores", "column": "landscape_local_stability_score"}
    return res


def threshold_region(df, spec, variables, ctx):
    t = _target(spec)
    p = spec["operator"].get("parameters", {})
    threshold = float(p["threshold"])
    direction = p.get("direction", ">=")
    y = _num(df, t)
    if direction == ">=":
        hit = y >= threshold
    elif direction == "<=":
        hit = y <= threshold
    elif direction == ">":
        hit = y > threshold
    elif direction == "<":
        hit = y < threshold
    else:
        raise ValueError("threshold_region direction must be one of >=, <=, >, <")
    selected = df.loc[hit].copy()
    rows = []
    for v in variables:
        a = _num(df.loc[hit], v)
        b = _num(df.loc[~hit], v)
        rows.append(
            {
                "variable": v,
                "selected_median": a.median(),
                "other_median": b.median(),
                "median_shift": a.median() - b.median(),
            }
        )
    shifts = pd.DataFrame(rows)
    if not shifts.empty:
        shifts = shifts.sort_values("median_shift", key=lambda s: s.abs(), ascending=False)
    return AnalysisResult(
        {
            "operator": "threshold_region",
            "target": t,
            "threshold": threshold,
            "direction": direction,
            "n_selected": int(hit.sum()),
            "fraction_selected": float(hit.mean()),
        },
        {
            "selected_points": selected.head(int(p.get("max_saved_points", 10000))),
            "variable_shifts": shifts,
        },
        {"kind": "bar", "table": "variable_shifts", "x": "variable", "y": "median_shift"},
    )


def threshold_boundary(df, spec, variables, ctx):
    t = _target(spec)
    p = spec["operator"].get("parameters", {})
    threshold = float(p["threshold"])
    y = _num(df, t)
    scale = robust_range(y.to_numpy(float))
    dist = np.abs(y - threshold) / scale
    n = int(p.get("n_points", 200))
    out = df.assign(normalized_distance_to_threshold=dist).sort_values("normalized_distance_to_threshold").head(n)
    return AnalysisResult(
        {
            "operator": "threshold_boundary",
            "target": t,
            "threshold": threshold,
            "n_boundary_points": len(out),
            "median_normalized_distance": float(out.normalized_distance_to_threshold.median()),
        },
        {"boundary_points": out},
        {"kind": "hist_table", "table": "boundary_points", "column": "normalized_distance_to_threshold"},
    )


def feasibility(df, spec, variables, ctx):
    p = spec["operator"].get("parameters", {})
    col = p.get("column", "feasible")
    if col not in df.columns:
        raise KeyError(f"Feasibility column {col!r} not present")
    d = df.copy()
    d["_feasible_numeric"] = d[col].fillna(False).astype(bool).astype(int)
    if d["_feasible_numeric"].nunique() < 2:
        return AnalysisResult(
            {
                "operator": "feasibility",
                "feasible_fraction": float(d["_feasible_numeric"].mean()),
                "surrogate_qc": None,
                "guardrail": "Only one feasibility class is present; driver classification is undefined.",
            },
            {"feasibility_importance": pd.DataFrame(columns=["variable", "heldout_permutation_auc_drop_mean", "heldout_permutation_auc_drop_sd"])},
            {"kind": "none"},
        )
    bundle = fit_surrogate(
        d, variables, "_feasible_numeric", spec["surrogate"], task="classification", seed=ctx["seed"]
    )
    finite = _numeric_frame(d, variables + ["_feasible_numeric"])
    train, test = train_test_split(
        finite,
        test_size=0.25,
        random_state=ctx["seed"],
        stratify=finite["_feasible_numeric"].astype(int),
    )
    model = clone(bundle.model)
    model.fit(train[variables].to_numpy(float), train["_feasible_numeric"].to_numpy(int))
    imp = permutation_importance(
        model,
        test[variables].to_numpy(float),
        test["_feasible_numeric"].to_numpy(int),
        n_repeats=10,
        random_state=ctx["seed"],
        scoring="roc_auc",
        n_jobs=int(spec["surrogate"].get("n_jobs", -1)),
    )
    tab = pd.DataFrame(
        {
            "variable": variables,
            "heldout_permutation_auc_drop_mean": imp.importances_mean,
            "heldout_permutation_auc_drop_sd": imp.importances_std,
        }
    ).sort_values("heldout_permutation_auc_drop_mean", ascending=False)
    return AnalysisResult(
        {
            "operator": "feasibility",
            "feasible_fraction": float(d["_feasible_numeric"].mean()),
            "surrogate_qc": bundle.metrics,
            "importance_holdout_rows": len(test),
        },
        {"feasibility_importance": tab},
        {
            "kind": "bar",
            "table": "feasibility_importance",
            "x": "variable",
            "y": "heldout_permutation_auc_drop_mean",
        },
    )


def regime_discovery(df, spec, variables, ctx):
    p = spec["operator"].get("parameters", {})
    requested = [str(x) for x in (p.get("features") or [])]
    targets = [t["name"] for t in spec.get("targets", [])]
    features = list(requested) if requested else list(dict.fromkeys([*variables, *targets]))
    if len(features) < 2:
        # One observable alone cannot define a regime geometry. Add available design
        # variables rather than silently clustering a single scalar.
        features = list(dict.fromkeys([*features, *variables]))
    if len(features) < 2:
        raise RuntimeError("Regime discovery requires at least two finite features (observables and/or design variables)")
    d = _numeric_frame(df, features)
    if len(d) < 30:
        raise RuntimeError("At least 30 complete rows are required for regime discovery")
    scaler = StandardScaler()
    X = scaler.fit_transform(d)
    max_k = min(int(p.get("max_k", 8)), len(d) - 1, max(2, len(d) // 30))
    rows = []
    best = None
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=ctx["seed"]).fit(X)
        sil = float(
            silhouette_score(X, model.labels_, sample_size=min(5000, len(X)), random_state=ctx["seed"])
        )
        rows.append({"k": k, "silhouette": sil})
        if best is None or sil > best[0]:
            best = (sil, k, model.labels_, model.cluster_centers_)
    if best is None:
        raise RuntimeError("Insufficient rows for regime discovery")
    labels = np.asarray(best[2], int)
    assign = pd.DataFrame({"row_index": d.index, "cluster": labels})
    profiles = d.assign(cluster=labels).groupby("cluster")[features].median().reset_index()

    # Standardized cluster profiles make variables with different biological units
    # comparable without replacing the raw-median profile table.
    zdf = pd.DataFrame(X, index=d.index, columns=features)
    zprofiles = zdf.assign(cluster=labels).groupby("cluster")[features].median().reset_index()

    # A deterministic 2-D PCA projection is generated whenever the regime space is
    # higher-dimensional. This is a visualization aid only; clustering remains in
    # the full standardized feature space above.
    pca = PCA(n_components=min(2, len(features)), random_state=ctx["seed"]).fit(X)
    proj = pca.transform(X)
    projection = pd.DataFrame({"row_index": d.index, "cluster": labels, "PC1": proj[:, 0]})
    if proj.shape[1] >= 2:
        projection["PC2"] = proj[:, 1]
    else:
        projection["PC2"] = 0.0
    loadings = []
    for ci in range(pca.components_.shape[0]):
        for j, feature in enumerate(features):
            loadings.append({"component": ci + 1, "feature": feature, "loading": float(pca.components_[ci, j])})

    sil = float(best[0])
    strength = "weak" if sil < 0.25 else ("moderate" if sil < 0.50 else "stronger")
    warnings = [
        "Clusters are descriptive structure in the sampled landscape and are not automatically distinct biological states.",
    ]
    if sil < 0.25:
        warnings.append("Silhouette is <0.25, indicating weak cluster separation; interpret regimes as a visualization aid rather than discrete states.")

    return AnalysisResult(
        {
            "operator": "regime_discovery",
            "targets": targets,
            "features": features,
            "clustering_space": str(p.get("clustering_space", "resolved_features")),
            "n_features": len(features),
            "selected_k": int(best[1]),
            "silhouette": sil,
            "silhouette_interpretation": strength,
            "pca_variance_explained": [float(x) for x in pca.explained_variance_ratio_],
            "guardrail": "Low silhouette implies continuous/weak regimes; do not interpret clusters as discrete biological regimes without independent evidence.",
        },
        {
            "k_selection": pd.DataFrame(rows),
            "cluster_assignments": assign,
            "cluster_profiles": profiles,
            "cluster_profiles_standardized": zprofiles,
            "regime_pca_projection": projection,
            "regime_pca_loadings": pd.DataFrame(loadings),
        },
        {"kind": "regime_multi", "table": "cluster_assignments"},
        warnings,
    )


def dimensionality(df, spec, variables, ctx):
    p = spec["operator"].get("parameters", {})
    requested = p.get("features") or []
    features = list(requested) if requested else list(dict.fromkeys([*variables, *[t["name"] for t in spec.get("targets", [])]]))
    d = _numeric_frame(df, features)
    if len(d) < max(20, len(features) + 2):
        raise RuntimeError("Insufficient complete rows for PCA dimensionality analysis")
    scaler = StandardScaler()
    X = scaler.fit_transform(d)
    n_components = min(len(features), int(p.get("max_components", len(features))))
    pca = PCA(n_components=n_components, random_state=ctx["seed"]).fit(X)
    ev = pca.explained_variance_ratio_
    cumulative = np.cumsum(ev)
    variance = pd.DataFrame(
        {
            "component": np.arange(1, len(ev) + 1),
            "explained_variance_ratio": ev,
            "cumulative_explained_variance": cumulative,
        }
    )
    loadings = []
    for i in range(len(ev)):
        for j, feature in enumerate(features):
            loadings.append(
                {"component": i + 1, "feature": feature, "loading": float(pca.components_[i, j])}
            )
    threshold = float(p.get("variance_threshold", 0.90))
    effective = int(np.searchsorted(cumulative, threshold) + 1)
    return AnalysisResult(
        {
            "operator": "dimensionality",
            "features": features,
            "variance_threshold": threshold,
            "components_to_threshold": effective,
        },
        {"pca_variance": variance, "pca_loadings": pd.DataFrame(loadings)},
        {"kind": "line", "table": "pca_variance", "x": "component", "y": "cumulative_explained_variance"},
    )


def tradeoff(df, spec, variables, ctx):
    ts = [t["name"] for t in spec["targets"]]
    dirs = [t.get("direction", "maximize") for t in spec["targets"]]
    if len(ts) < 2:
        raise ValueError("tradeoff requires >=2 targets")
    d = df.dropna(subset=ts).copy()
    Y = d[ts].to_numpy(float)
    mask = pareto_mask(Y, dirs)
    front = d.loc[mask].copy()
    corr = d[ts].corr(method="spearman")
    return AnalysisResult(
        {
            "operator": "tradeoff",
            "targets": ts,
            "directions": dirs,
            "n_pareto": len(front),
            "fraction_pareto": len(front) / max(len(d), 1),
        },
        {"pareto_points": front, "spearman_matrix": corr.reset_index().rename(columns={"index": "target"})},
        {"kind": "scatter", "x": ts[0], "y": ts[1], "pareto_table": "pareto_points"},
    )


def pareto(df, spec, variables, ctx):
    res = tradeoff(df, spec, variables, ctx)
    res.summary["operator"] = "pareto"
    return res


def pareto_saturation(df, spec, variables, ctx):
    p = spec["operator"].get("parameters", {})
    targets = [t["name"] for t in spec["targets"]]
    directions = [t.get("direction", "maximize") for t in spec["targets"]]
    if len(targets) < 2:
        raise ValueError("pareto_saturation requires at least two targets")
    d = _numeric_frame(df, targets)
    Z, _, _ = normalized_oriented_targets(d[targets].to_numpy(float), directions)
    full_mask = pareto_mask(Z, ["maximize"] * Z.shape[1])
    full_front = Z[full_mask]
    if len(full_front) == 0:
        raise RuntimeError("No finite full empirical Pareto front")
    sizes = [
        int(x)
        for x in p.get("sample_sizes", [500, 1000, 2500, 5000, 10000, 25000, 50000])
        if 2 <= int(x) <= len(Z)
    ]
    if len(Z) not in sizes:
        sizes.append(len(Z))
    sizes = sorted(set(sizes))
    reps = int(p.get("repeats", 10))
    rng = np.random.default_rng(ctx["seed"])
    full_hv = hypervolume_2d(full_front) if Z.shape[1] == 2 else float("nan")
    rows = []
    for n in sizes:
        nreps = 1 if n == len(Z) else reps
        for r in range(nreps):
            idx = np.arange(len(Z)) if n == len(Z) else rng.choice(len(Z), size=n, replace=False)
            sub = Z[idx]
            sm = pareto_mask(sub, ["maximize"] * sub.shape[1])
            front = sub[sm]
            rec = {
                "sample_size": n,
                "replicate": r,
                "front_size": int(len(front)),
                "igd_plus_to_full_front": float(igd_plus(front, full_front)),
            }
            if Z.shape[1] == 2:
                hv = hypervolume_2d(front)
                rec["normalized_hv_2d"] = float(hv)
                rec["hv_recovery_fraction"] = float(hv / full_hv) if full_hv > 0 else float("nan")
            rows.append(rec)
    tab = pd.DataFrame(rows)
    agg = {
        "median_igd_plus": ("igd_plus_to_full_front", "median"),
        "q90_igd_plus": ("igd_plus_to_full_front", lambda s: float(s.quantile(0.90))),
        "median_front_size": ("front_size", "median"),
    }
    if "hv_recovery_fraction" in tab.columns:
        agg["median_hv_recovery_fraction"] = ("hv_recovery_fraction", "median")
    summ = tab.groupby("sample_size", as_index=False).agg(**agg)
    return AnalysisResult(
        {
            "operator": "pareto_saturation",
            "targets": targets,
            "directions": directions,
            "full_rows": len(Z),
            "full_front_size": len(full_front),
            "full_normalized_hv_2d": None if not np.isfinite(full_hv) else float(full_hv),
            "repeats": reps,
            "definition": "Repeated landscape subsampling versus the full empirical Pareto front using IGD+; 2D problems also report normalized hypervolume recovery.",
        },
        {"pareto_saturation_replicates": tab, "pareto_saturation_summary": summ},
        {"kind": "line", "table": "pareto_saturation_summary", "x": "sample_size", "y": "median_igd_plus"},
    )


def compare_regions(df, spec, variables, ctx):
    p = spec["operator"].get("parameters", {})
    t = _target(spec)
    A = apply_filters(df, p.get("region_a", []))
    B = apply_filters(df, p.get("region_b", []))
    a = _num(A, t).dropna()
    b = _num(B, t).dropna()
    if len(a) < 2 or len(b) < 2:
        raise RuntimeError("Both regions require at least two finite target values.")
    u, pv = mannwhitneyu(a, b, alternative="two-sided")
    pooled = max(robust_range(pd.concat([a, b]).to_numpy()), 1e-12)
    tab = pd.DataFrame(
        [
            {
                "target": t,
                "region_a_n": len(a),
                "region_b_n": len(b),
                "region_a_median": a.median(),
                "region_b_median": b.median(),
                "normalized_median_difference": (a.median() - b.median()) / pooled,
                "mannwhitney_u": u,
                "p_value": pv,
            }
        ]
    )
    return AnalysisResult({"operator": "compare_regions", "target": t}, {"comparison": tab}, {"kind": "none"})


def coverage(df, spec, variables, ctx):
    X, _, _ = normalize_matrix(df, variables)
    X = X[np.all(np.isfinite(X), axis=1)]
    if len(X) < 2:
        raise RuntimeError("Coverage requires at least two finite design points")
    tree = cKDTree(X)
    d, _ = tree.query(X, k=2)
    nn = d[:, 1]
    tab = pd.DataFrame({"nearest_neighbor_distance": nn})
    return AnalysisResult(
        {
            "operator": "coverage",
            "n_points": len(X),
            "dimension": len(variables),
            "median_nn_distance": float(np.median(nn)),
            "q95_nn_distance": float(np.quantile(nn, 0.95)),
            "max_nn_distance": float(np.max(nn)),
        },
        {"nearest_neighbor_distances": tab},
        {"kind": "hist_table", "table": "nearest_neighbor_distances", "column": "nearest_neighbor_distance"},
    )


def outlier(df, spec, variables, ctx):
    p = spec["operator"].get("parameters", {})
    requested = p.get("features") or []
    features = list(requested) if requested else list(dict.fromkeys([*variables, *[t["name"] for t in spec.get("targets", [])]]))
    d = _numeric_frame(df, features)
    model = IsolationForest(
        contamination=float(p.get("contamination", 0.01)), random_state=ctx["seed"], n_jobs=int(spec.get("surrogate", {}).get("n_jobs", -1))
    )
    score = -model.fit(d.to_numpy(float)).score_samples(d.to_numpy(float))
    out = pd.DataFrame({"row_index": d.index, "outlier_score": score}).sort_values("outlier_score", ascending=False)
    return AnalysisResult(
        {"operator": "outlier", "features": features, "n_scored": len(out)},
        {"outlier_scores": out},
        {"kind": "hist_table", "table": "outlier_scores", "column": "outlier_score"},
    )


def point_query(df, spec, variables, ctx):
    t = _target(spec)
    p = spec["operator"].get("parameters", {})
    point = p.get("point") or {}
    values = []
    for v in variables:
        raw = point.get(v, point.get(v.replace("requested_medium__", ""), np.nan))
        values.append(float(raw))
    center = np.asarray(values, float)
    if not np.isfinite(center).all():
        raise ValueError("point_query requires a value for every resolved design variable")
    X, lo, hi = normalize_matrix(df, variables)
    span = np.where(hi > lo, hi - lo, 1.0)
    q = (center - lo) / span
    dist = np.sqrt(np.nansum((X - q) ** 2, axis=1))
    n = int(p.get("neighbors", 10))
    ids = np.argsort(dist)[:n]
    near = df.iloc[ids].copy()
    near.insert(0, "normalized_distance_to_query", dist[ids])
    bundle = fit_surrogate(df, variables, t, spec["surrogate"], seed=ctx["seed"])
    pred, disp = bundle.predict(center.reshape(1, -1))
    return AnalysisResult(
        {
            "operator": "point_query",
            "target": t,
            "point": dict(zip(variables, center)),
            "surrogate_prediction": float(pred[0]),
            "ensemble_dispersion_proxy": None if disp is None else float(disp[0]),
            "surrogate_qc": bundle.metrics,
            "nearest_exact_distance": float(dist[ids[0]]),
            "evidence": "nearest exact landscape support + surrogate interpolation",
        },
        {"nearest_landscape_points": near},
        {"kind": "none"},
        ["Ensemble-tree dispersion is a heuristic support/uncertainty proxy, not a calibrated confidence interval."],
    )



def _surrogate_qc_failure(exc: Exception) -> bool:
    msg = str(exc)
    return "Surrogate regression QC failed" in msg or "Surrogate classifier QC failed" in msg


def _exact_binned_response_curve(df, variable: str, target: str, bins: int = 25) -> pd.DataFrame:
    d = _numeric_frame(df, [variable, target])
    if len(d) < 20:
        raise RuntimeError("Exact-landscape response fallback requires at least 20 finite rows")
    q = min(max(6, int(bins)), max(6, int(np.sqrt(len(d)))))
    try:
        groups = pd.qcut(d[variable], q=q, duplicates="drop")
    except Exception:
        groups = pd.cut(d[variable], bins=q, duplicates="drop")
    z = d.assign(_bin=groups).dropna(subset=["_bin"])
    rows=[]
    for _, g in z.groupby("_bin", observed=True):
        if g.empty: continue
        rows.append({
            "variable_value": float(g[variable].median()),
            "prediction": float(g[target].median()),
            "exact_q10": float(g[target].quantile(0.10)),
            "exact_q90": float(g[target].quantile(0.90)),
            "exact_n": int(len(g)),
        })
    return pd.DataFrame(rows).sort_values("variable_value").reset_index(drop=True)


def _exact_binned_response_surface(df, x: str, y: str, target: str, bins: int = 18) -> pd.DataFrame:
    d = _numeric_frame(df, [x, y, target])
    if len(d) < 50:
        raise RuntimeError("Exact-landscape surface fallback requires at least 50 finite rows")
    nb = min(max(6, int(bins)), 30)
    try:
        xb = pd.qcut(d[x], q=nb, duplicates="drop")
        yb = pd.qcut(d[y], q=nb, duplicates="drop")
    except Exception:
        xb = pd.cut(d[x], bins=nb, duplicates="drop")
        yb = pd.cut(d[y], bins=nb, duplicates="drop")
    z = d.assign(_xb=xb, _yb=yb).dropna(subset=["_xb", "_yb"])
    rows=[]
    for _, g in z.groupby(["_xb", "_yb"], observed=True):
        if len(g) < 2: continue
        rows.append({
            x: float(g[x].median()),
            y: float(g[y].median()),
            "prediction": float(g[target].median()),
            "exact_q10": float(g[target].quantile(0.10)),
            "exact_q90": float(g[target].quantile(0.90)),
            "exact_n": int(len(g)),
        })
    if not rows:
        raise RuntimeError("Exact-landscape surface fallback produced no supported bins")
    return pd.DataFrame(rows)

def response_curve(df, spec, variables, ctx):
    targets = [t["name"] for t in spec.get("targets", [])]
    if not targets:
        raise ValueError("response_curve requires at least one target")
    p = spec["operator"].get("parameters", {})
    var = p.get("variable")
    if not var:
        raise ValueError("response_curve operator.parameters.variable required")
    hits = [v for v in variables if v == var or v.endswith("__" + str(var))]
    if len(hits) != 1:
        raise KeyError(f"Could not uniquely resolve response variable {var!r} from {variables}")
    v = hits[0]
    tabs = []
    qc_by_target = {}
    failures = {}
    modes = {}
    anchors = {}
    for t in targets:
        try:
            bundle = fit_surrogate(df, variables, t, spec["surrogate"], seed=ctx["seed"])
        except RuntimeError as exc:
            if not _surrogate_qc_failure(exc):
                raise
            tab = _exact_binned_response_curve(df, v, t, bins=int(p.get("fallback_bins", 25)))
            tab.insert(0, "target", t)
            tabs.append(tab)
            failures[t] = str(exc)
            modes[t] = "exact_landscape_binned_fallback"
            continue
        anchor = _anchor(spec, df, variables)
        j = variables.index(v)
        grid = np.linspace(bundle.bounds[j, 0], bundle.bounds[j, 1], int(p.get("grid_points", 101)))
        X = np.tile(anchor, (len(grid), 1))
        X[:, j] = grid
        pred, disp = bundle.predict(X)
        tab = pd.DataFrame({"target": t, "variable_value": grid, "prediction": pred})
        if disp is not None:
            tab["ensemble_dispersion_proxy"] = disp
        tabs.append(tab)
        qc_by_target[t] = bundle.metrics
        modes[t] = "validated_surrogate_interpolation"
        anchors[t] = dict(zip(variables, anchor))
    table = pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()
    unique_modes = set(modes.values())
    evidence = next(iter(unique_modes)) if len(unique_modes) == 1 else "mixed_exact_and_validated_surrogate"
    summary = {
        "operator": "response_curve",
        "target": targets[0],
        "targets": targets,
        "variable": v,
        "evidence": evidence,
        "evidence_by_target": modes,
        "surrogate_qc_by_target": qc_by_target,
        "surrogate_qc_failures": failures,
        "anchors_by_target": anchors,
        "definition": "one-variable response; each observable uses validated full-design surrogate interpolation when QC passes and stored exact-landscape binned evidence when QC fails",
    }
    if len(targets) == 1:
        if targets[0] in qc_by_target:
            summary["surrogate_qc"] = qc_by_target[targets[0]]
            summary["anchor"] = anchors.get(targets[0])
        if targets[0] in failures:
            summary["surrogate_qc_failure"] = failures[targets[0]]
    warnings = []
    if failures:
        warnings.append("One or more smooth surrogate response curves failed the pre-specified QC gate; those observables are shown using binned stored exact-landscape evidence instead.")
    return AnalysisResult(
        summary,
        {"response_curve": table},
        {"kind": "line_multi", "table": "response_curve", "x": "variable_value", "y": "prediction"},
        warnings,
    )


def marginal_response(df, spec, variables, ctx):
    targets = [t["name"] for t in spec.get("targets", [])]
    if not targets:
        raise ValueError("marginal_response requires at least one target")
    p = spec["operator"].get("parameters", {})
    var = p.get("variable")
    if not var:
        raise ValueError("marginal_response operator.parameters.variable required")
    hits = [v for v in variables if v == var or v.endswith("__" + str(var))]
    if len(hits) != 1:
        raise KeyError(f"Could not uniquely resolve marginal response variable {var!r}")
    v = hits[0]
    d = _numeric_frame(df, variables)
    backgrounds = d.sample(min(int(p.get("background_rows", 512)), len(d)), random_state=ctx["seed"]).to_numpy(float)
    tabs=[]; qc_by_target={}; failures={}; modes={}
    for t in targets:
        try:
            bundle = fit_surrogate(df, variables, t, spec["surrogate"], seed=ctx["seed"])
        except RuntimeError as exc:
            if not _surrogate_qc_failure(exc):
                raise
            fallback = _exact_binned_response_curve(df, v, t, bins=int(p.get("fallback_bins", 25)))
            fallback = fallback.rename(columns={"prediction":"median_prediction", "exact_q10":"q10_prediction", "exact_q90":"q90_prediction"})
            fallback["mean_prediction"] = fallback["median_prediction"]
            fallback.insert(0, "target", t)
            tabs.append(fallback)
            failures[t]=str(exc); modes[t]="exact_landscape_conditional_fallback"
            continue
        j = variables.index(v)
        grid = np.linspace(bundle.bounds[j, 0], bundle.bounds[j, 1], int(p.get("grid_points", 51)))
        rows = []
        for value in grid:
            X = backgrounds.copy(); X[:, j] = value
            pred, _ = bundle.predict(X)
            rows.append({
                "target": t,
                "variable_value": value,
                "mean_prediction": float(np.mean(pred)),
                "median_prediction": float(np.median(pred)),
                "q10_prediction": float(np.quantile(pred, 0.10)),
                "q90_prediction": float(np.quantile(pred, 0.90)),
            })
        tabs.append(pd.DataFrame(rows)); qc_by_target[t]=bundle.metrics; modes[t]="validated_surrogate_marginal_response"
    table=pd.concat(tabs,ignore_index=True) if tabs else pd.DataFrame()
    unique_modes=set(modes.values())
    evidence=next(iter(unique_modes)) if len(unique_modes)==1 else "mixed_exact_and_validated_surrogate"
    summary={
        "operator":"marginal_response", "target":targets[0], "targets":targets, "variable":v,
        "background_rows":len(backgrounds), "evidence":evidence, "evidence_by_target":modes,
        "surrogate_qc_by_target":qc_by_target, "surrogate_qc_failures":failures,
        "definition":"partial-dependence-style model-averaged response when surrogate QC passes; exact conditional binned response is substituted and explicitly labeled when QC fails",
    }
    if len(targets)==1:
        if targets[0] in qc_by_target: summary["surrogate_qc"]=qc_by_target[targets[0]]
        if targets[0] in failures: summary["surrogate_qc_failure"]=failures[targets[0]]
    warnings=["Marginal response is model-based and descriptive, not causal."]
    if failures:
        warnings.append("For observables whose marginal surrogate failed QC, the figure shows exact conditional landscape bins; this is not claimed to be the same estimand as partial dependence.")
    return AnalysisResult(summary,{"marginal_response":table},{"kind":"line_multi","table":"marginal_response","x":"variable_value","y":"mean_prediction"},warnings)


def surrogate_uncertainty(df, spec, variables, ctx):
    t = _target(spec)
    p = spec["operator"].get("parameters", {})
    bundle = fit_surrogate(df, variables, t, spec["surrogate"], seed=ctx["seed"])
    d = _numeric_frame(df, variables + [t])
    eval_rows = d.sample(min(int(p.get("evaluation_rows", 10000)), len(d)), random_state=ctx["seed"])
    X = eval_rows[variables].to_numpy(float)
    pred, disp = bundle.predict(X)
    if disp is None:
        raise RuntimeError("Selected surrogate does not expose an ensemble dispersion proxy")
    scale = max(robust_range(eval_rows[t].to_numpy(float)), 1e-12)
    tab = pd.DataFrame(
        {
            "row_index": eval_rows.index,
            "observed_target": eval_rows[t].to_numpy(float),
            "surrogate_prediction": pred,
            "absolute_residual": np.abs(pred - eval_rows[t].to_numpy(float)),
            "ensemble_dispersion_proxy": disp,
            "normalized_ensemble_dispersion": disp / scale,
        }
    ).sort_values("normalized_ensemble_dispersion", ascending=False)
    top_n = int(p.get("top_points", 500))
    return AnalysisResult(
        {
            "operator": "surrogate_uncertainty",
            "target": t,
            "surrogate_qc": bundle.metrics,
            "evaluation_rows": len(tab),
            "median_normalized_ensemble_dispersion": float(tab["normalized_ensemble_dispersion"].median()),
            "q95_normalized_ensemble_dispersion": float(tab["normalized_ensemble_dispersion"].quantile(0.95)),
            "guardrail": "Tree-to-tree ensemble dispersion is a heuristic interpolation-support proxy, not calibrated predictive uncertainty.",
        },
        {"highest_dispersion_points": tab.head(top_n), "surrogate_dispersion_all": tab},
        {"kind": "hist_table", "table": "surrogate_dispersion_all", "column": "normalized_ensemble_dispersion"},
    )



def controller_scan(df, spec, variables, ctx):
    """Rank design variables as surrogate-predicted monotonic control knobs."""
    t = _target(spec)
    p = spec["operator"].get("parameters", {})
    bundle = fit_surrogate(df, variables, t, spec["surrogate"], seed=ctx["seed"])
    d = _numeric_frame(df, variables)
    anchors_df = d.sample(min(int(p.get("anchor_count", 96)), len(d)), random_state=ctx["seed"])
    anchors = anchors_df.to_numpy(float)
    levels = int(p.get("levels", 9))
    yscale = max(robust_range(_num(df, t).to_numpy(float)), 1e-12)
    rows = []
    hotspots = []
    for j, var in enumerate(variables):
        grid = np.linspace(bundle.bounds[j, 0], bundle.bounds[j, 1], levels)
        scores = []
        spans = []
        monos = []
        for aidx, anchor in enumerate(anchors):
            X = np.tile(anchor, (levels, 1))
            X[:, j] = grid
            pred, _ = bundle.predict(X)
            span = float((np.max(pred) - np.min(pred)) / yscale)
            rho = spearmanr(grid, pred).statistic if np.ptp(pred) > 1e-15 else 0.0
            monotonicity = float(abs(rho)) if np.isfinite(rho) else 0.0
            score = span * monotonicity
            spans.append(span); monos.append(monotonicity); scores.append(score)
        arr = np.asarray(scores, float)
        top = int(np.argmax(arr))
        rows.append({
            "variable": var,
            "median_normalized_span": float(np.median(spans)),
            "q95_normalized_span": float(np.quantile(spans, 0.95)),
            "median_monotonicity": float(np.median(monos)),
            "median_controller_score": float(np.median(arr)),
            "q95_controller_score": float(np.quantile(arr, 0.95)),
            "max_controller_score": float(np.max(arr)),
        })
        hotspots.append({
            "variable": var,
            "anchor_row_index": anchors_df.index[top],
            "controller_score": float(arr[top]),
            **{f"anchor__{v}": float(anchors[top, k]) for k, v in enumerate(variables)},
        })
    tab = pd.DataFrame(rows).sort_values("median_controller_score", ascending=False)
    hot = pd.DataFrame(hotspots).sort_values("controller_score", ascending=False)
    return AnalysisResult(
        {
            "operator": "controller_scan",
            "target": t,
            "surrogate_qc": bundle.metrics,
            "anchors": len(anchors),
            "levels": levels,
            "definition": "controller score = normalized response span × absolute Spearman monotonicity across a one-variable surrogate sweep, summarized across observed backgrounds",
        },
        {"controller_ranking": tab, "controller_hotspots": hot},
        {"kind": "bar", "table": "controller_ranking", "x": "variable", "y": "median_controller_score"},
        ["Controller scan is surrogate-based and descriptive; it does not establish causal control or exact attainability between sampled points."],
    )


def threshold_switch_scan(df, spec, variables, ctx):
    """Rank variables by their ability to cross a specified target threshold in surrogate sweeps."""
    t = _target(spec)
    p = spec["operator"].get("parameters", {})
    if "threshold" not in p:
        raise ValueError("threshold_switch_scan requires operator.parameters.threshold")
    threshold = float(p["threshold"])
    bundle = fit_surrogate(df, variables, t, spec["surrogate"], seed=ctx["seed"])
    d = _numeric_frame(df, variables)
    anchors_df = d.sample(min(int(p.get("anchor_count", 96)), len(d)), random_state=ctx["seed"])
    anchors = anchors_df.to_numpy(float)
    levels = int(p.get("levels", 21))
    yscale = max(robust_range(_num(df, t).to_numpy(float)), 1e-12)
    rows = []
    hotspot_rows = []
    for j, var in enumerate(variables):
        grid = np.linspace(bundle.bounds[j, 0], bundle.bounds[j, 1], levels)
        input_span = max(bundle.bounds[j, 1] - bundle.bounds[j, 0], 1e-12)
        crossed = []
        steepness = []
        best_score = -np.inf
        best_anchor = None
        best_cross_value = np.nan
        for aidx, anchor in enumerate(anchors):
            X = np.tile(anchor, (levels, 1)); X[:, j] = grid
            pred, _ = bundle.predict(X)
            diff = pred - threshold
            crossings = np.where(diff[:-1] * diff[1:] <= 0)[0]
            if len(crossings) == 0:
                crossed.append(0); steepness.append(0.0); continue
            crossed.append(1)
            local = []
            for k in crossings:
                dx = max(grid[k + 1] - grid[k], 1e-12)
                slope = abs(pred[k + 1] - pred[k]) / yscale / (dx / input_span)
                local.append((float(slope), float((grid[k + 1] + grid[k]) / 2)))
            score, cross_value = max(local, key=lambda z: z[0])
            steepness.append(score)
            if score > best_score:
                best_score = score; best_anchor = aidx; best_cross_value = cross_value
        cross_arr = np.asarray(crossed, float); steep = np.asarray(steepness, float)
        rows.append({
            "variable": var,
            "crossing_fraction": float(np.mean(cross_arr)),
            "median_normalized_crossing_steepness": float(np.median(steep[cross_arr > 0])) if np.any(cross_arr > 0) else 0.0,
            "q95_normalized_crossing_steepness": float(np.quantile(steep[cross_arr > 0], 0.95)) if np.any(cross_arr > 0) else 0.0,
            "switch_score": float(np.mean(cross_arr) * (np.median(steep[cross_arr > 0]) if np.any(cross_arr > 0) else 0.0)),
        })
        if best_anchor is not None:
            hotspot_rows.append({
                "variable": var,
                "threshold": threshold,
                "predicted_crossing_variable_value": best_cross_value,
                "normalized_crossing_steepness": float(best_score),
                "anchor_row_index": anchors_df.index[best_anchor],
                **{f"anchor__{v}": float(anchors[best_anchor, k]) for k, v in enumerate(variables)},
            })
    tab = pd.DataFrame(rows).sort_values("switch_score", ascending=False)
    hot = pd.DataFrame(hotspot_rows)
    if not hot.empty: hot = hot.sort_values("normalized_crossing_steepness", ascending=False)
    return AnalysisResult(
        {
            "operator": "threshold_switch_scan",
            "target": t,
            "threshold": threshold,
            "anchors": len(anchors),
            "levels": levels,
            "surrogate_qc": bundle.metrics,
            "definition": "surrogate one-variable sweeps; switch score combines fraction of backgrounds with a threshold crossing and normalized local crossing steepness",
        },
        {"threshold_controller_ranking": tab, "threshold_switch_hotspots": hot},
        {"kind": "bar", "table": "threshold_controller_ranking", "x": "variable", "y": "switch_score"},
        ["Predicted threshold crossings require exact FBA confirmation before being presented as new exact switch conditions."],
    )


def response_surface(df, spec, variables, ctx):
    targets = [t["name"] for t in spec.get("targets", [])]
    if not targets:
        raise ValueError("response_surface requires at least one target")
    p = spec["operator"].get("parameters", {})
    requested = list(p.get("variables") or [])
    if len(requested) != 2:
        raise ValueError("response_surface requires operator.parameters.variables with exactly two design variables")
    resolved=[]
    for req in requested:
        hits=[v for v in variables if v==req or v.endswith("__"+str(req))]
        if len(hits)!=1:
            raise KeyError(f"Could not uniquely resolve response surface variable {req!r}")
        resolved.append(hits[0])
    tabs=[]; qc_by_target={}; failures={}; modes={}; anchors={}
    for t in targets:
        try:
            bundle=fit_surrogate(df,variables,t,spec["surrogate"],seed=ctx["seed"])
        except RuntimeError as exc:
            if not _surrogate_qc_failure(exc): raise
            tab=_exact_binned_response_surface(df,resolved[0],resolved[1],t,bins=int(p.get("fallback_bins",18)))
            tab.insert(0,"target",t); tabs.append(tab); failures[t]=str(exc); modes[t]="exact_landscape_binned_fallback"
            continue
        anchor=_anchor(spec,df,variables); anchors[t]=dict(zip(variables,anchor))
        j0,j1=variables.index(resolved[0]),variables.index(resolved[1])
        n=int(p.get("grid_points",41))
        gx=np.linspace(bundle.bounds[j0,0],bundle.bounds[j0,1],n); gy=np.linspace(bundle.bounds[j1,0],bundle.bounds[j1,1],n)
        rows=[]
        for yy in gy:
            X=np.tile(anchor,(len(gx),1)); X[:,j0]=gx; X[:,j1]=yy
            pred,disp=bundle.predict(X)
            for ii,xx in enumerate(gx):
                rec={"target":t,resolved[0]:float(xx),resolved[1]:float(yy),"prediction":float(pred[ii])}
                if disp is not None: rec["ensemble_dispersion_proxy"]=float(disp[ii])
                rows.append(rec)
        tabs.append(pd.DataFrame(rows)); qc_by_target[t]=bundle.metrics; modes[t]="validated_surrogate_interpolation"
    table=pd.concat(tabs,ignore_index=True) if tabs else pd.DataFrame()
    unique_modes=set(modes.values()); evidence=next(iter(unique_modes)) if len(unique_modes)==1 else "mixed_exact_and_validated_surrogate"
    summary={
        "operator":"response_surface","target":targets[0],"targets":targets,"variables":resolved,
        "evidence":evidence,"evidence_by_target":modes,"surrogate_qc_by_target":qc_by_target,
        "surrogate_qc_failures":failures,"anchors_by_target":anchors,
        "definition":"two-variable response surface per observable, using full-design validated surrogate interpolation or exact-landscape binned fallback when QC fails",
    }
    if len(targets)==1:
        if targets[0] in qc_by_target: summary["surrogate_qc"]=qc_by_target[targets[0]]
        if targets[0] in failures: summary["surrogate_qc_failure"]=failures[targets[0]]
    warnings=[]
    if failures: warnings.append("One or more smooth response surfaces failed surrogate QC; those observables use exact-landscape binned heatmaps without interpolation claims.")
    return AnalysisResult(summary,{"response_surface":table},{"kind":"surface_multi","table":"response_surface"},warnings)


def sampling_saturation(df, spec, variables, ctx):
    p = spec["operator"].get("parameters", {})
    default = [100, 250, 500, 1000, 2500, 5000, 10000, 25000, 50000]
    sizes = [int(x) for x in p.get("sample_sizes", default) if 2 <= int(x) <= len(df)]
    if len(df) not in sizes:
        sizes.append(len(df))
    sizes = sorted(set(sizes))
    reps = int(p.get("repeats", 10))
    X, _, _ = normalize_matrix(df, variables)
    X = X[np.all(np.isfinite(X), axis=1)]
    rng = np.random.default_rng(ctx["seed"])
    rows = []
    eval_idx = rng.choice(len(X), size=min(int(p.get("evaluation_points", 5000)), len(X)), replace=False)
    Q = X[eval_idx]
    for n in sizes:
        if n > len(X):
            continue
        nreps = 1 if n == len(X) else reps
        for r in range(nreps):
            idx = np.arange(len(X)) if n == len(X) else rng.choice(len(X), size=n, replace=False)
            tree = cKDTree(X[idx])
            d, _ = tree.query(Q, k=1)
            rows.append(
                {
                    "sample_size": n,
                    "replicate": r,
                    "median_cover_distance": float(np.median(d)),
                    "q95_cover_distance": float(np.quantile(d, 0.95)),
                    "max_cover_distance": float(np.max(d)),
                }
            )
    tab = pd.DataFrame(rows)
    if tab.empty:
        raise RuntimeError("No valid sample sizes for saturation analysis")
    summ = tab.groupby("sample_size", as_index=False).agg(
        median_q95_cover_distance=("q95_cover_distance", "median"),
        median_cover_distance=("median_cover_distance", "median"),
        sd_q95_cover_distance=("q95_cover_distance", "std"),
    )
    return AnalysisResult(
        {
            "operator": "sampling_saturation",
            "dimension": len(variables),
            "repeats": reps,
            "definition": "normalized design-space coverage saturation; lower nearest-subsample distance is denser",
        },
        {"sampling_saturation_replicates": tab, "sampling_saturation_summary": summ},
        {"kind": "line", "table": "sampling_saturation_summary", "x": "sample_size", "y": "median_q95_cover_distance"},
    )


REGISTRY = {
    "describe": describe,
    "query": query,
    "association": association,
    "correlation_matrix": correlation_matrix,
    "global_sensitivity": global_sensitivity,
    "local_sensitivity": local_sensitivity,
    "sensitivity_map": sensitivity_map,
    "interaction": interaction,
    "fragility": fragility,
    "robustness": robustness,
    "threshold_region": threshold_region,
    "threshold_boundary": threshold_boundary,
    "feasibility": feasibility,
    "regime_discovery": regime_discovery,
    "dimensionality": dimensionality,
    "tradeoff": tradeoff,
    "pareto": pareto,
    "pareto_saturation": pareto_saturation,
    "compare_regions": compare_regions,
    "coverage": coverage,
    "outlier": outlier,
    "point_query": point_query,
    "response_curve": response_curve,
    "marginal_response": marginal_response,
    "surrogate_uncertainty": surrogate_uncertainty,
    "controller_scan": controller_scan,
    "threshold_switch_scan": threshold_switch_scan,
    "response_surface": response_surface,
    "sampling_saturation": sampling_saturation,
}

# This table is part of the public scientific contract. "surrogate_assisted" means
# exact landscape observations are the training/evidence base, while interpolation or
# a learned property is required to answer the requested question.
OPERATOR_BACKENDS = {
    "describe": "landscape",
    "query": "landscape",
    "association": "landscape",
    "correlation_matrix": "landscape",
    "global_sensitivity": "surrogate_assisted",
    "local_sensitivity": "surrogate_assisted",
    "sensitivity_map": "surrogate_assisted",
    "interaction": "surrogate_assisted",
    "fragility": "landscape",
    "robustness": "landscape",
    "threshold_region": "landscape",
    "threshold_boundary": "landscape",
    "feasibility": "surrogate_assisted",
    "regime_discovery": "landscape",
    "dimensionality": "landscape",
    "tradeoff": "landscape",
    "pareto": "landscape",
    "pareto_saturation": "landscape",
    "compare_regions": "landscape",
    "coverage": "landscape",
    "outlier": "landscape",
    "point_query": "hybrid",
    "response_curve": "surrogate_assisted",
    "marginal_response": "surrogate_assisted",
    "surrogate_uncertainty": "surrogate_assisted",
    "controller_scan": "surrogate_assisted",
    "threshold_switch_scan": "surrogate_assisted",
    "response_surface": "surrogate_assisted",
    "sampling_saturation": "landscape",
}
