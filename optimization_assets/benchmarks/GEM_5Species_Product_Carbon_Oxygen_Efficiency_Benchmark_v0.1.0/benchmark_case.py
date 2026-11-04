from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from core import (
    GlobalSurrogate,
    crossfit_subset_prior_predictions,
    ensure_dir,
    feature_spec,
    normalize_x,
    normalize_y,
    representative_subset,
    save_json,
)
from live_core import ExactRunner, LiveContext, duration
from methods import METHOD_NOTES, RUNNERS
from benchmark_paths import resolve_from_config


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def load_baseline(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path, compression="infer")


def method_qd(ctx: LiveContext, method: str) -> dict:
    h = ctx.history(method)
    if len(h) == 0:
        return {"qd_coverage": np.nan, "qd_score": np.nan, "qd_cells": 0}
    explicit = ctx.parent_cfg.get("problem", {}).get("qd_feature")
    o2_idx = ctx.feature_index(explicit=explicit, keywords=("oxygen", "o2"), required=False)
    if o2_idx is None:
        return {"qd_coverage": np.nan, "qd_score": np.nan, "qd_cells": 0}
    g_edges = np.quantile(ctx.Y0[ctx.good0, 0], [.2, .4, .6, .8])
    o_edges = np.quantile(ctx.X0[:, o2_idx], [.2, .4, .6, .8])
    cells = {}
    for _, r in h.iterrows():
        if not bool(r["exact_feasible"]):
            continue
        g = float(r["exact_product_norm"]); p = float(r["exact_carbon_efficiency_norm"])
        o = float(r[f"x01__{ctx.features[o2_idx]}"])
        if not np.isfinite([g, p, o]).all():
            continue
        key = (int(np.digitize(g, g_edges)), int(np.digitize(o, o_edges)))
        cells[key] = max(cells.get(key, -np.inf), p)
    return {
        "qd_coverage": float(len(cells) / 25),
        "qd_score": float(sum(cells.values())) if cells else 0.0,
        "qd_cells": int(len(cells)),
    }


def _exact_solver_seconds(ctx: LiveContext, method: str) -> float:
    p = ctx.outdir / "exact_runtime_ledger.csv"
    if not p.exists():
        return 0.0
    try:
        d = pd.read_csv(p)
        hit = d[d["method"].astype(str) == str(method)]
        return float(pd.to_numeric(hit.get("physical_exact_wall_seconds"), errors="coerce").fillna(0).sum())
    except Exception:
        return float("nan")


def build_summary(
    ctx: LiveContext, methods: list[str], cumulative_runtimes: dict[str, float],
    session_runtimes: dict[str, float], errors: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for m in methods:
        sm = ctx.summary(
            m,
            runtime_seconds_cumulative=cumulative_runtimes.get(m),
            runtime_seconds_this_session=session_runtimes.get(m),
            exact_solver_wall_seconds=_exact_solver_seconds(ctx, m),
        )
        sm.update(method_qd(ctx, m))
        sm["note"] = METHOD_NOTES.get(m, "")
        sm["error"] = errors.get(m, "")
        rows.append(sm)
    df = pd.DataFrame(rows)
    if "final_delta_hv" in df.columns:
        df = df.sort_values("final_delta_hv", ascending=False, na_position="last").reset_index(drop=True)
    return df


def _update_runtime_ledger(method_dir: Path, *, session_seconds: float, calls_before: int, calls_after: int, budget: int, status: str) -> float:
    path = method_dir / "runtime_sessions.json"
    sessions = []
    if path.exists():
        try:
            obj = json.loads(path.read_text(encoding="utf-8"))
            sessions = list(obj.get("sessions", []))
        except Exception:
            sessions = []
    sessions.append({
        "session_index": len(sessions) + 1,
        "budget_target": int(budget),
        "calls_before": int(calls_before),
        "calls_after": int(calls_after),
        "wall_seconds": float(session_seconds),
        "status": str(status),
        "unix_end_time": float(time.time()),
    })
    cumulative = float(sum(float(x.get("wall_seconds", 0.0)) for x in sessions))
    path.write_text(json.dumps({"cumulative_wall_seconds": cumulative, "sessions": sessions}, indent=2), encoding="utf-8")
    return cumulative


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _save_runtime_provenance(outroot: Path, study_path: Path):
    pkg = Path(__file__).resolve().parent
    snap = outroot / "runtime_provenance"
    snap.mkdir(parents=True, exist_ok=True)
    files = []
    for pat in ("*.py", "*.sh", "*.yaml", "*.yml", "*.toml", "VERSION.txt"):
        files.extend(pkg.glob(pat))
    hashes = {x.name: _sha256(x) for x in sorted(set(files)) if x.is_file()}
    config_hash = _sha256(study_path)

    # A canary→resume run must not silently change source or study config.
    hp = snap / "source_hashes.json"
    ep = snap / "environment.json"
    if hp.exists():
        previous = json.loads(hp.read_text(encoding="utf-8"))
        if previous != hashes:
            raise RuntimeError("Source hashes changed since this run directory was created; refusing mixed-code resume")
    if ep.exists():
        previous_env = json.loads(ep.read_text(encoding="utf-8"))
        if previous_env.get("study_config_sha256") != config_hash:
            raise RuntimeError("study_config changed since this run directory was created; refusing mixed-config resume")

    shutil.copy2(study_path, snap / "study_config_runtime.yaml")
    hp.write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
    ep.write_text(json.dumps({
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "study_config_sha256": config_hash,
    }, indent=2), encoding="utf-8")
    return hashes


def save_fig(fig, fdir: Path, stem: str, dpi: int):
    for ext in ("png", "pdf", "svg"):
        fig.savefig(fdir / f"{stem}.{ext}", dpi=dpi if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def make_figures(ctx: LiveContext, summary: pd.DataFrame, methods: list[str], dpi: int, title: str):
    fdir = ensure_dir(ctx.outdir / "figures")
    vals = pd.to_numeric(summary.get("final_delta_hv"), errors="coerce") if "final_delta_hv" in summary else pd.Series(dtype=float)
    plot = summary[np.isfinite(vals)].copy() if len(summary) else summary.copy()
    if len(plot):
        plot = plot.sort_values("final_delta_hv", ascending=False)
        fig, ax = plt.subplots(figsize=(max(8, 0.62 * len(plot)), 5.2))
        ax.bar(np.arange(len(plot)), plot["final_delta_hv"])
        ax.set_xticks(np.arange(len(plot)), plot["method"], rotation=55, ha="right")
        ax.set_ylabel("Method-only exact normalized 3D HV")
        ax.set_title(title)
        _style_axes(ax); save_fig(fig, fdir, "XS_Fig01_final_delta_hv", dpi)

    fig, ax = plt.subplots(figsize=(8.2, 5.4)); anyline = False
    for m in methods:
        h = ctx.history(m)
        if len(h):
            ax.plot(h["call"], h["delta_hv_after"], marker="o", markersize=2.5, linewidth=1.2, label=m)
            anyline = True
    if anyline:
        ax.set_xlabel("Logical design-evaluation budget")
        ax.set_ylabel("Method-only exact normalized 3D HV")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        _style_axes(ax); save_fig(fig, fdir, "XS_Fig02_hv_vs_calls", dpi)
    else:
        plt.close(fig)

    if len(plot) and "minimum_exact_carbon_demand" in plot:
        fig, ax = plt.subplots(figsize=(max(8, 0.62 * len(plot)), 5.2))
        ax.bar(np.arange(len(plot)), plot["minimum_exact_carbon_demand"])
        ax.set_xticks(np.arange(len(plot)), plot["method"], rotation=55, ha="right")
        ax.set_ylabel("Minimum exact normalized carbon demand")
        _style_axes(ax); save_fig(fig, fdir, "PCO_Fig03_min_carbon_demand", dpi)
    if len(plot) and "minimum_exact_oxygen_demand" in plot:
        fig, ax = plt.subplots(figsize=(max(8, 0.62 * len(plot)), 5.2))
        ax.bar(np.arange(len(plot)), plot["minimum_exact_oxygen_demand"])
        ax.set_xticks(np.arange(len(plot)), plot["method"], rotation=55, ha="right")
        ax.set_ylabel("Minimum exact normalized oxygen demand")
        _style_axes(ax); save_fig(fig, fdir, "PCO_Fig04_min_oxygen_demand", dpi)

    if len(plot) and "exact_feasible_fraction" in plot:
        fig, ax = plt.subplots(figsize=(max(8, 0.62 * len(plot)), 5.2))
        ax.bar(np.arange(len(plot)), plot["exact_feasible_fraction"])
        ax.set_xticks(np.arange(len(plot)), plot["method"], rotation=55, ha="right")
        ax.set_ylabel("Exact feasible fraction")
        ax.set_ylim(0, 1.05)
        _style_axes(ax); save_fig(fig, fdir, "XS_Fig04_feasibility", dpi)

    if len(plot) and "runtime_seconds_cumulative" in plot:
        fig, ax = plt.subplots(figsize=(max(8, 0.62 * len(plot)), 5.2))
        ax.bar(np.arange(len(plot)), pd.to_numeric(plot["runtime_seconds_cumulative"], errors="coerce") / 60)
        ax.set_xticks(np.arange(len(plot)), plot["method"], rotation=55, ha="right")
        ax.set_ylabel("Cumulative optimizer + exact-evaluation wall time (min)")
        _style_axes(ax); save_fig(fig, fdir, "XS_Fig05_runtime", dpi)

    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    P = ctx.baseline_front
    P = P[np.argsort(P[:, 0])] if len(P) else P
    if len(P):
        ax.plot(P[:, 0], P[:, 1], marker="o", markersize=3, linewidth=1.2, label="Exact 50k 3D prior (product-carbon projection)")
    for m in methods:
        h = ctx.history(m)
        if len(h):
            good = h["exact_feasible"].astype(bool).to_numpy()
            Y = h[["exact_product_norm", "exact_carbon_efficiency_norm"]].to_numpy(float)
            good &= np.isfinite(Y).all(axis=1)
            if np.any(good):
                ax.scatter(Y[good, 0], Y[good, 1], s=15, label=m)
    ax.set_xlabel("Normalized exact product flux")
    ax.set_ylabel("Exact carbon efficiency")
    ax.legend(frameon=False, fontsize=6, ncol=2)
    _style_axes(ax); save_fig(fig, fdir, "XS_Fig06_exact_pareto_overlay", dpi)

    # Relative gain is cross-case interpretable while preserving each case's frozen scaling.
    if len(plot) and "relative_hv_gain" in plot:
        fig, ax = plt.subplots(figsize=(max(8, 0.62 * len(plot)), 5.2))
        ax.bar(np.arange(len(plot)), 100 * pd.to_numeric(plot["relative_hv_gain"], errors="coerce"))
        ax.set_xticks(np.arange(len(plot)), plot["method"], rotation=55, ha="right")
        ax.set_ylabel("Method-only exact HV / frozen 50k HV (%)")
        _style_axes(ax); save_fig(fig, fdir, "XS_Fig07_relative_hv_gain", dpi)

    if len(plot) and "exact_extensions" in plot:
        fig, ax = plt.subplots(figsize=(max(8, 0.62 * len(plot)), 5.2))
        ax.bar(np.arange(len(plot)), plot["exact_extensions"])
        ax.set_xticks(np.arange(len(plot)), plot["method"], rotation=55, ha="right")
        ax.set_ylabel("New exact Pareto-front extensions")
        _style_axes(ax); save_fig(fig, fdir, "XS_Fig08_exact_extensions", dpi)

    # Cumulative exact-front extensions as a function of logical budget.
    fig, ax = plt.subplots(figsize=(8.2, 5.4)); anyline = False
    for m in methods:
        h = ctx.history(m)
        if len(h):
            ext = h["exact_front_extension"].astype(bool).astype(int).cumsum()
            ax.step(h["call"], ext, where="post", label=m)
            anyline = True
    if anyline:
        ax.set_xlabel("Logical design-evaluation budget")
        ax.set_ylabel("Cumulative exact front extensions")
        ax.legend(frameon=False, fontsize=7, ncol=2)
        _style_axes(ax); save_fig(fig, fdir, "XS_Fig09_extensions_vs_calls", dpi)
    else:
        plt.close(fig)

    if len(plot) and "delta_hv_per_logical_call" in plot:
        fig, ax = plt.subplots(figsize=(max(8, 0.62 * len(plot)), 5.2))
        ax.bar(np.arange(len(plot)), plot["delta_hv_per_logical_call"])
        ax.set_xticks(np.arange(len(plot)), plot["method"], rotation=55, ha="right")
        ax.set_ylabel("Exact ΔHV per logical exact-FBA call")
        _style_axes(ax); save_fig(fig, fdir, "XS_Fig10_hv_efficiency", dpi)

    if len(plot) and "method_only_hv" in plot:
        fig, ax = plt.subplots(figsize=(max(8, 0.62 * len(plot)), 5.2))
        ax.bar(np.arange(len(plot)), plot["method_only_hv"])
        ax.set_xticks(np.arange(len(plot)), plot["method"], rotation=55, ha="right")
        ax.set_ylabel("Hypervolume of method's new exact points alone")
        _style_axes(ax); save_fig(fig, fdir, "XS_Fig11_method_only_hv", dpi)

    # Per-batch feasibility / discovery trace. This reveals whether gains were early or late.
    fig, ax = plt.subplots(figsize=(8.2, 5.4)); anyline = False
    for m in methods:
        h = ctx.history(m)
        if len(h):
            feas = h["exact_feasible"].astype(bool).astype(int).expanding().mean()
            ax.plot(h["call"], feas, marker="o", markersize=2.2, linewidth=1.0, label=m)
            anyline = True
    if anyline:
        ax.set_xlabel("Logical design-evaluation budget")
        ax.set_ylabel("Cumulative exact feasible fraction")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        _style_axes(ax); save_fig(fig, fdir, "XS_Fig12_feasibility_vs_calls", dpi)
    else:
        plt.close(fig)



def table_text(df: pd.DataFrame) -> str:
    """Report-safe table rendering with no optional `tabulate` dependency."""
    try:
        return df.to_markdown(index=False)
    except Exception:
        return "```text\n" + df.to_string(index=False) + "\n```"

def write_report(ctx: LiveContext, summary: pd.DataFrame, preflight: dict, study_cfg: dict, problem_cfg: dict,
                 methods: list[str], errors: dict[str, str], case_name: str):
    p = problem_cfg["problem"]
    logical = int(summary["calls"].fillna(0).sum()) if "calls" in summary else 0
    physical = int(summary["physical_exact_calls"].fillna(0).sum()) if "physical_exact_calls" in summary else 0
    cols = [c for c in [
        "method", "calls", "final_delta_hv", "relative_hv_gain", "delta_hv_per_logical_call",
        "exact_feasible_fraction", "exact_extensions", "extensions_per_call", "first_positive_delta_hv_call",
        "best_exact_product", "minimum_exact_carbon_demand", "minimum_exact_oxygen_demand", "best_exact_growth", "method_only_hv", "method_only_hv_fraction_of_frozen", "physical_exact_calls", "qd_coverage",
    ] if c in summary.columns]
    lines = [
        f"# Five-species product × carbon efficiency × oxygen efficiency benchmark — {case_name}", "",
        "## Purpose",
        "This matched-budget seven-method benchmark maximizes exact growth-constrained product flux, minimizes normalized carbon demand, and minimizes normalized oxygen demand, while enforcing a frozen high-growth viability constraint. The existing 50k landscape supplies exact maximum growth and product directly; carbon/oxygen efficiencies are deterministic from named design coordinates and require zero archive FBA relabeling.", "",
        "## Frozen problem",
        f"- Model ID: {p.get('model_id')}",
        f"- Model fingerprint: `{p.get('model_fingerprint')}`",
        f"- Config hash: `{p.get('config_hash')}`",
        f"- Biomass reaction: `{p.get('biomass_reaction_id')}`",
        f"- Objectives (maximization space): {p.get('objective_names')}",
        f"- Carbon feature: {p.get('carbon_feature')}",
        f"- Oxygen feature: {p.get('oxygen_feature')}",
        f"- Global growth viability floor: {p.get('growth_viability_floor')}",
        f"- Design dimensions: {len(ctx.features)}",
        f"- Features: {', '.join(ctx.features)}",
        f"- Frozen archive rows: {len(ctx.baseline):,}",
        f"- Frozen feasible rows: {int(ctx.F0.sum()):,}",
        f"- Frozen Pareto-front size: {len(ctx.baseline_front):,}",
        f"- Product normalization scale (frozen viable max): {p.get('product_scale')}",
                f"- Exact 50k three-objective normalized HV: {ctx.baseline_hv:.12f}", "",
    ]
    qc_path = ctx.outdir / "surrogate_qc" / "surrogate_qc_summary.json"
    if qc_path.exists():
        qc = json.loads(qc_path.read_text(encoding="utf-8"))
        lines += [
            "## Product / growth-viability surrogate validation",
            f"- QC status: {qc.get('status')}",
            f"- Validation rows: {qc.get('rows', 0):,}; carbon and oxygen objectives are deterministic and are not learned.",
                        f"- Product OOF R²: {qc.get('product', {}).get('r2')}",
            f"- Product OOF RMSE (normalized): {qc.get('product', {}).get('rmse')}",
            f"- Product OOF Spearman ρ: {qc.get('product', {}).get('spearman_rho')}",
            f"- Deterministic carbon-efficiency max abs error: {qc.get('deterministic_carbon_efficiency_max_abs_error')}",
            f"- Deterministic oxygen-efficiency max abs error: {qc.get('deterministic_oxygen_efficiency_max_abs_error')}",
            "- Full surrogate QC report/plots: `surrogate_qc/`", "",
        ]
    lines += [
        "## Exact bridge QC",
        f"- Preflight: {preflight['status']}",
    ]
    for k, v in preflight["checks"].items():
        lines.append(f"- {k}: {v}")
    lines += ["", "## Exact-call accounting",
              f"- Logical method evaluations: {logical}",
              f"- Physical new exact evaluations: {physical}",
              f"- Cross-method cache reuse: {logical - physical}",
              "- Archive carbon/oxygen-efficiency construction uses 0 FBA solves. The small exact growth/product replay QC is provenance work, not live search, and is reported separately in landscape_preflight.json.", "",
              "## Ranking by exact ΔHV", table_text(summary[cols]), "", "## Method interpretation"]
    for m in methods:
        lines.append(f"- **{m}:** {METHOD_NOTES.get(m, '')}")
    if errors:
        lines += ["", "## Method errors"]
        for m, e in errors.items():
            lines.append(f"- **{m}:** `{e}`")
    lines += [
        "", "## Scientific guardrails",
        "- This is one live seed per organism and is architecture/transfer evidence, not statistical superiority.",
        "- The 50k landscape is a high-coverage sampled exact reference, not a certified global Pareto front.",
        "- Product normalization is frozen from growth-viable exact 50k archive points; carbon and oxygen efficiencies are exact [0,1] coordinate transforms. Scales and the growth threshold are never refit after seeing live outcomes.",
        "- The 50k archive has exact growth/product and deterministic carbon/oxygen efficiencies; there is no surrogate-derived archive objective in the primary three-objective reference.",
        "- The primary cross-species comparison is exact 3D hypervolume and HV-AUC versus gsMOBO+50k at matched logical budgets.",
        "- No organism-specific optimizer hyperparameter tuning was performed by this package; the same study-level settings are reused across cases.",
    ]
    (ctx.outdir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    (ctx.outdir / "report.txt").write_text("\n".join(x.replace("## ", "").replace("# ", "") for x in lines), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-config", default="study_config.yaml")
    ap.add_argument("--case", required=True)
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--methods", default=None, help="comma-separated subset")
    ap.add_argument("--preflight-only", action="store_true")
    args = ap.parse_args()

    study_path = Path(args.study_config).expanduser().resolve()
    cfg = load_yaml(study_path)
    if args.case not in cfg["cases"]:
        raise SystemExit(f"Unknown case {args.case!r}; choices={list(cfg['cases'])}")
    case = dict(cfg["cases"][args.case])
    case["prepared_dir"] = str(resolve_from_config(study_path, cfg["paths"]["prepared_root"]) / args.case)
    case["study_config_path"] = str(study_path)
    case["case_key"] = args.case
    problem_path = Path(case["prepared_dir"]) / "problem_config.yaml"
    if not problem_path.exists():
        raise SystemExit(f"Prepared problem missing: {problem_path}. Run ./preflight_all.sh first.")
    parent_cfg = load_yaml(problem_path)
    baseline_path = Path(parent_cfg["problem"]["baseline_table"])
    baseline = load_baseline(baseline_path)
    features, bounds = feature_spec(parent_cfg)

    seed = int(args.seed if args.seed is not None else cfg["study"]["seed"])
    budget = int(args.budget if args.budget is not None else cfg["study"]["overnight_budget"])
    methods = list(cfg["methods_primary"])
    if args.methods:
        methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    unknown = [m for m in methods if m not in RUNNERS]
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")

    outroot = ensure_dir(resolve_from_config(study_path, cfg["paths"]["output_root"]) / args.case / f"seed_{seed}")
    source_hashes = _save_runtime_provenance(outroot, study_path)
    exact = ExactRunner(cfg, parent_cfg, outroot, features, bounds, case)
    preflight = exact.preflight(); save_json(preflight, outroot / "preflight.json")
    print("=" * 92)
    print(f"GEM 5-SPECIES PRODUCT × CARBON EFFICIENCY × OXYGEN EFFICIENCY — {args.case}")
    print("=" * 92)
    print(f"Seed: {seed} | budget/method: {budget} | methods: {len(methods)}")
    print(f"Frozen archive: {baseline_path}")
    print(f"Exact bridge preflight: {preflight['status']}")
    for k, v in preflight["resolved"].items():
        print(f"  {k}: {v}")
    if preflight["status"] != "PASS":
        raise SystemExit("Preflight failed; no live exact calls were made.")
    if args.preflight_only:
        print("Preflight only: PASS. No live exact calls were made.")
        return

    X0 = normalize_x(baseline[features].to_numpy(float), bounds)
    Y0 = normalize_y(baseline["maximum_growth"], baseline["product_flux"], parent_cfg, carbon_efficiency=baseline["carbon_efficiency"], oxygen_efficiency=baseline["oxygen_efficiency"])
    F0 = baseline["feasible"].astype(bool).to_numpy()

    # v0.2.1 hard sequencing: publication-oriented surrogate QC must complete
    # before the first live optimizer proposal.  Reuse the exact full-data model
    # and all-row out-of-fold predictions produced by that QC stage, avoiding
    # redundant model fitting and guaranteeing that Prior-GP/LPPC calibration
    # uses predictions that excluded the target row's validation fold.
    qc_dir = outroot / "surrogate_qc"
    qc_summary_path = qc_dir / "surrogate_qc_summary.json"
    qc_model_path = qc_dir / "final_global_surrogate.joblib"
    qc_oof_path = qc_dir / "oof_prior_predictions.npz"
    require_qc = bool(cfg.get("surrogate_validation", {}).get("require_before_live", True))
    if require_qc and not (qc_summary_path.exists() and qc_model_path.exists() and qc_oof_path.exists()):
        raise SystemExit(
            f"Surrogate validation is required before live optimization but is missing for {args.case}. "
            f"Run: python surrogate_validation.py --study-config {study_path} --cases {args.case}"
        )
    if qc_summary_path.exists():
        qc_summary = json.loads(qc_summary_path.read_text(encoding="utf-8"))
        if qc_summary.get("status") != "PASS" or int(qc_summary.get("rows", -1)) != len(baseline):
            raise SystemExit(f"Surrogate QC is not a matching PASS for {args.case}: {qc_summary_path}")
    else:
        qc_summary = {}

    if qc_model_path.exists() and bool(cfg.get("surrogate_validation", {}).get("reuse_final_model_for_optimizers", True)):
        print(f"[live] loading QC-validated full-landscape surrogate: {qc_model_path}", flush=True)
        global_model = joblib.load(qc_model_path)
    else:
        print(f"[live] fitting global surrogate on {len(baseline):,} exact rows ({len(features)}D)...", flush=True)
        t = time.time()
        global_model = GlobalSurrogate(cfg, seed).fit(X0, Y0, F0)
        print(f"[live] global surrogate fit complete in {duration(time.time() - t)}", flush=True)

    subset_n = min(int(cfg["study"]["gp_subset_rows"]), len(baseline))
    subset = representative_subset(baseline, X0, Y0, subset_n, seed + 330)
    prior_needed = any(m in {"BASE_PRIOR_GP_PARETO", "NEW_LANDSCAPE_PRIOR_PARETO_CONTINUATION"} for m in methods)
    oof = None
    if prior_needed:
        if qc_oof_path.exists():
            z = np.load(qc_oof_path)
            all_oof = np.asarray(z["oof"], float)
            if len(all_oof) != len(baseline) or all_oof.shape[1] != 3:
                raise RuntimeError(f"QC OOF prediction matrix has wrong shape: {all_oof.shape}")
            oof = all_oof[subset]
            print(f"[live] using QC all-row OOF predictions for {len(subset):,} Prior-GP/LPPC calibration rows", flush=True)
        else:
            shared = ensure_dir(outroot / "_shared")
            cache = shared / "subset_oof_prior.npz"
            if cache.exists():
                z = np.load(cache); oof = z["oof"]; cached_subset = z["subset"].astype(int)
                if not np.array_equal(cached_subset, subset):
                    raise RuntimeError("Cached prior subset does not match current representative subset")
                print("[live] loaded cached cross-fitted prior calibration predictions", flush=True)
            else:
                print(f"[live] computing leak-resistant cross-fitted prior for {len(subset):,} representative rows...", flush=True)
                t = time.time()
                oof = crossfit_subset_prior_predictions(X0, Y0, F0, subset, cfg, seed + 440)
                np.savez_compressed(cache, oof=oof, subset=subset)
                print(f"[live] cross-fit complete in {duration(time.time() - t)}", flush=True)

    ctx = LiveContext(cfg, parent_cfg, baseline, baseline_path, global_model, subset, oof, outroot, exact, seed, budget)
    print(f"[live] exact 50k product-carbon-oxygen prior HV={ctx.baseline_hv:.12f} | 3D prior front points={len(ctx.baseline_front)}")
    save_json({
        "case": args.case,
        "baseline_hv": ctx.baseline_hv,
        "front_size": len(ctx.baseline_front),
        "feasible_rows": int(ctx.F0.sum()),
        "rows": len(ctx.baseline),
        "dimension": len(ctx.features),
        "product_scale": parent_cfg["problem"]["product_scale"],
        "growth_viability_floor": parent_cfg["problem"].get("growth_viability_floor"),
        "carbon_feature": parent_cfg["problem"].get("carbon_feature"),
        "oxygen_feature": parent_cfg["problem"].get("oxygen_feature"),
        "surrogate_qc_status": qc_summary.get("status"),
        "surrogate_product_r2_oof": qc_summary.get("product", {}).get("r2"),
        "surrogate_product_spearman_oof": qc_summary.get("product", {}).get("spearman_rho"),
        "surrogate_3d_exact_front_points": qc_summary.get("pareto_3d", {}).get("exact_front_points"),
        "surrogate_deterministic_carbon_max_abs_error": qc_summary.get("deterministic_carbon_efficiency_max_abs_error"),
        "surrogate_deterministic_oxygen_max_abs_error": qc_summary.get("deterministic_oxygen_efficiency_max_abs_error"),
    }, outroot / "baseline_metrics.json")
    pd.DataFrame(ctx.baseline_front, columns=["product_norm", "carbon_efficiency_norm", "oxygen_efficiency_norm"]).to_csv(outroot / "prior_pareto_front_3d_normalized.csv", index=False)

    cumulative_runtimes, session_runtimes, errors = {}, {}, {}
    for i, m in enumerate(methods, 1):
        print("\n" + "-" * 92)
        print(f"[{args.case}] METHOD {i}/{len(methods)}: {m}")
        print("-" * 92, flush=True)
        calls_before = len(ctx.history(m)); t0 = time.time(); status = "PASS"
        try:
            RUNNERS[m](ctx)
            session_runtimes[m] = time.time() - t0
            (ctx.method_dir(m) / "ERROR.txt").unlink(missing_ok=True)
        except Exception as e:
            session_runtimes[m] = time.time() - t0; status = "ERROR"
            errors[m] = f"{type(e).__name__}: {e}"
            tb = traceback.format_exc()
            (ctx.method_dir(m) / "ERROR.txt").write_text(tb, encoding="utf-8")
            print(f"[{args.case}] {m} ERROR: {errors[m]}", flush=True)
            if not bool(cfg["study"].get("continue_on_method_error", True)):
                raise
        calls_after = len(ctx.history(m))
        cumulative_runtimes[m] = _update_runtime_ledger(
            ctx.method_dir(m), session_seconds=session_runtimes[m],
            calls_before=calls_before, calls_after=calls_after, budget=budget, status=status,
        )
        if status == "PASS":
            print(
                f"[{args.case}] {m} COMPLETE | calls={calls_after}/{budget} | exact ΔHV={ctx.current_delta_hv(m):.9f} "
                f"| session={duration(session_runtimes[m])} | cumulative={duration(cumulative_runtimes[m])}",
                flush=True,
            )
        build_summary(ctx, methods, cumulative_runtimes, session_runtimes, errors).to_csv(outroot / "summary.csv", index=False)

    summary = build_summary(ctx, methods, cumulative_runtimes, session_runtimes, errors)
    summary.to_csv(outroot / "summary.csv", index=False)
    calls = [ctx.history(m) for m in methods if len(ctx.history(m))]
    if calls:
        pd.concat(calls, ignore_index=True).to_csv(outroot / "calls_all.csv", index=False)
    dpi = int(cfg["study"].get("figure_dpi", 600))
    make_figures(ctx, summary, methods, dpi, title=f"{args.case}: product-carbon-oxygen efficiency benchmark")
    write_report(ctx, summary, preflight, cfg, parent_cfg, methods, errors, args.case)
    manifest = {
        "status": "COMPLETE" if not errors else "COMPLETE_WITH_METHOD_ERRORS",
        "case": args.case,
        "seed": seed,
        "budget_per_method": budget,
        "methods": methods,
        "baseline_path": str(baseline_path),
        "baseline_hv": ctx.baseline_hv,
        "surrogate_qc": {
            "status": qc_summary.get("status"),
            "summary_path": str(qc_summary_path),
            "product_r2_oof": qc_summary.get("product", {}).get("r2"),
            "exact_front_recall": qc_summary.get("ranking_fidelity", {}).get("exact_front_recall"),
        },
        "logical_exact_evaluations": int(summary["calls"].fillna(0).sum()),
        "physical_exact_calls": int(summary["physical_exact_calls"].fillna(0).sum()) if "physical_exact_calls" in summary else 0,
        "errors": errors,
        "package_version": str(cfg.get("version", "unknown")),
        "runtime_provenance_dir": str(outroot / "runtime_provenance"),
        "source_hashes": source_hashes,
        "software": {"python": sys.version.split()[0], "python_executable": sys.executable, "platform": platform.platform(), "numpy": np.__version__, "pandas": pd.__version__},
    }
    save_json(manifest, outroot / "run_manifest.json")

    # Matched-budget checkpoint snapshots.  The canonical files above always
    # represent the most recent state; these immutable light-weight snapshots
    # preserve the exact summary/report/manifest at 4/16/32/64/128 (or any
    # user-requested target) without duplicating the large figure set.
    tag = f"{budget:03d}"
    summary.to_csv(outroot / f"summary_budget_{tag}.csv", index=False)
    save_json(manifest, outroot / f"run_manifest_budget_{tag}.json")
    if (outroot / "report.md").exists():
        shutil.copy2(outroot / "report.md", outroot / f"report_budget_{tag}.md")
    if (outroot / "report.txt").exists():
        shutil.copy2(outroot / "report.txt", outroot / f"report_budget_{tag}.txt")

    print("\n" + "=" * 92)
    print(f"{args.case} PRODUCT × CARBON EFFICIENCY × OXYGEN EFFICIENCY BENCHMARK CHECKPOINT FINISHED")
    print("=" * 92)
    cols = [c for c in ["method", "calls", "final_delta_hv", "relative_hv_gain", "delta_hv_per_logical_call",
                         "exact_feasible_fraction", "exact_extensions", "best_exact_product", "minimum_exact_carbon_demand", "minimum_exact_oxygen_demand"] if c in summary.columns]
    print(summary[cols].to_string(index=False))
    print(f"\nResults: {outroot}")


if __name__ == "__main__":
    main()
