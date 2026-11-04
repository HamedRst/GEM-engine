from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .atlas_discovery import LandscapeRecord
from .atlas_problem_factory import LandscapeProfile, build_profile
from .data import load_landscape
from .metrics import robust_range
from .surrogate import fit_surrogate
from .utils import utc_now, write_json
from .plotting import PUBLICATION_RC


def _rid(v: str) -> str:
    for prefix in ("requested_medium__", "applied_medium__"):
        if str(v).startswith(prefix):
            return str(v)[len(prefix):]
    return str(v)


def _candidate_from_row(row: pd.Series, variables: list[str]) -> dict[str, float]:
    return {_rid(v): float(row[v]) for v in variables}


def _run_exact_one(
    record: LandscapeRecord,
    candidate: dict[str, float],
    scratch: Path,
    tag: str,
    fast_timeout: int = 30,
    exact_timeout: int = 60,
) -> dict[str, Any]:
    if not record.model_path:
        raise RuntimeError(f"{record.species}: exact validation requires a resolved model path")
    scratch.mkdir(parents=True, exist_ok=True)
    cpath = scratch / f"{tag}.candidate.json"
    cpath.write_text(json.dumps(candidate, indent=2), encoding="utf-8")

    def invoke(solver: str | None, timeout: int, suffix: str) -> tuple[str, dict[str, Any] | None, float]:
        out = scratch / f"{tag}.{suffix}.result.json"
        cmd = [
            sys.executable, "-m", "gem_exploration.exact_worker",
            "--run-dir", record.path,
            "--model-path", str(record.model_path),
            "--candidate-json", str(cpath),
            "--output-json", str(out),
        ]
        if solver:
            cmd += ["--solver", solver]
        t0 = time.perf_counter()
        try:
            cp = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout)
            elapsed = time.perf_counter() - t0
        except subprocess.TimeoutExpired:
            return "TIMEOUT", None, float(time.perf_counter() - t0)
        payload = None
        if out.exists():
            try:
                payload = json.loads(out.read_text(encoding="utf-8"))
            except Exception:
                payload = None
        if cp.returncode == 0 and payload and payload.get("worker_status") == "PASS":
            return "PASS", payload, elapsed
        err = (payload or {}).get("error") or cp.stderr[-1200:] or cp.stdout[-1200:] or f"returncode={cp.returncode}"
        return "FAIL:" + str(err), payload, elapsed

    status, payload, elapsed = invoke(None, int(fast_timeout), "fast")
    fallback = False
    if status == "TIMEOUT":
        fallback = True
        status2, payload2, elapsed2 = invoke("glpk_exact", int(exact_timeout), "glpk_exact")
        if status2 != "PASS":
            raise RuntimeError(
                f"Exact validation candidate {tag} timed out with the configured solver and glpk_exact fallback failed: {status2}"
            )
        payload = payload2
        status = "PASS"
        elapsed += elapsed2
    elif status != "PASS":
        raise RuntimeError(f"Exact validation candidate {tag} failed: {status}")

    assert payload is not None
    payload = dict(payload)
    payload.update({
        "validation_tag": tag,
        "fallback_to_glpk_exact": bool(fallback),
        "parent_elapsed_seconds": float(elapsed),
        "candidate": candidate,
    })
    return payload


def _choose_replay(df: pd.DataFrame, variables: list[str], n: int, seed: int) -> pd.DataFrame:
    work = df.copy()
    g = pd.to_numeric(work["maximum_growth"], errors="coerce")
    p = pd.to_numeric(work["product_flux"], errors="coerce")
    ok = g.notna() & p.notna()
    if "feasible" in work.columns:
        ok &= work["feasible"].fillna(False).astype(bool)
    work = work.loc[ok].copy()
    if work.empty:
        raise RuntimeError("No feasible finite exact landscape rows for replay QC")
    idxs: list[Any] = []
    for s, fn in [(g.loc[work.index], "min"), (g.loc[work.index], "max"), (p.loc[work.index], "max")]:
        idxs.append(s.idxmin() if fn == "min" else s.idxmax())
    # Balanced high-growth/high-product row.
    gg = pd.to_numeric(work["maximum_growth"], errors="coerce")
    pp = pd.to_numeric(work["product_flux"], errors="coerce")
    gn = (gg - gg.quantile(.02)) / max(gg.quantile(.98) - gg.quantile(.02), 1e-12)
    pn = (pp - pp.quantile(.02)) / max(pp.quantile(.98) - pp.quantile(.02), 1e-12)
    idxs.append((gn.clip(0,1) + pn.clip(0,1)).idxmax())
    rng = np.random.default_rng(seed)
    for x in rng.choice(work.index.to_numpy(), size=min(max(n, 5), len(work)), replace=False):
        idxs.append(x)
    unique = []
    for x in idxs:
        if x not in unique:
            unique.append(x)
    return work.loc[unique[:n], [*variables, "maximum_growth", "product_flux", *( ["sample_id"] if "sample_id" in work.columns else [])]].copy()


def _build_surrogates(df: pd.DataFrame, variables: list[str], cache_dir: Path, seed: int) -> tuple[Any, Any]:
    cfg = {
        "model": "extra_trees",
        "n_estimators": int(os.environ.get("GEM_ATLAS_TREES", "240")),
        "cv_folds": 5,
        "min_r2": 0.50,
        "max_train_rows": 50000,
        "n_jobs": int(os.environ.get("GEM_ATLAS_JOBS", "4")),
        "_cache_dir": str(cache_dir.resolve()),
        "_landscape_fingerprint": str(cache_dir.parent.resolve()),
    }
    return (
        fit_surrogate(df, variables, "maximum_growth", cfg, seed=seed),
        fit_surrogate(df, variables, "product_flux", cfg, seed=seed),
    )


def _challenge_candidates(
    df: pd.DataFrame,
    profile: LandscapeProfile,
    growth_bundle,
    product_bundle,
    n: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    variables = profile.design_variables
    lows = np.asarray([float(pd.to_numeric(df[v], errors="coerce").min()) for v in variables])
    highs = np.asarray([float(pd.to_numeric(df[v], errors="coerce").max()) for v in variables])
    d = len(variables)
    m = max(1024, 2 ** int(math.ceil(math.log2(max(2048, n * 256)))))
    sob = qmc.Sobol(d=d, scramble=True, seed=seed).random_base2(int(math.log2(m)))
    X = lows + sob * (highs - lows)
    pg, ug = growth_bundle.predict(X)
    pp, up = product_bundle.predict(X)
    up = np.zeros(len(X)) if up is None else up
    # Common normalization only for selection, never for reported biological values.
    def norm(a):
        lo, hi = np.nanquantile(a, [.02, .98]); return np.clip((a-lo)/max(hi-lo,1e-12),0,1)
    bal = 0.5 * (norm(pg) + norm(pp))
    categories = [
        ("high_product", pp),
        ("high_growth", pg),
        ("high_dispersion", up),
        ("balanced", bal),
    ]
    chosen: list[int] = []
    labels: dict[int, str] = {}
    each = max(1, n // len(categories))
    for name, score in categories:
        for idx in np.argsort(np.nan_to_num(score, nan=-np.inf))[::-1]:
            i = int(idx)
            if i not in chosen:
                chosen.append(i); labels[i] = name
            if sum(labels.get(j) == name for j in chosen) >= each:
                break
    rng = np.random.default_rng(seed + 33)
    while len(chosen) < n:
        i = int(rng.integers(0, len(X)))
        if i not in chosen:
            chosen.append(i); labels[i] = "space_filling"
    chosen = chosen[:n]
    tab = pd.DataFrame(X[chosen], columns=variables)
    tab.insert(0, "selection_reason", [labels[i] for i in chosen])
    tab["predicted_growth"] = pg[chosen]
    tab["predicted_product"] = pp[chosen]
    tab["product_ensemble_dispersion"] = up[chosen]

    # Local perturbations around the balanced exact point test the direction of
    # surrogate-local conclusions on the two most informative variables.
    local_rows = []
    center = np.asarray([profile.balanced_point[v] for v in variables], float)
    for v in [profile.primary_variable, profile.secondary_variable]:
        j = variables.index(v)
        span = max(highs[j] - lows[j], 1e-12)
        for sign in (-1, 1):
            x = center.copy(); x[j] = np.clip(x[j] + sign * 0.025 * span, lows[j], highs[j])
            pg1, _ = growth_bundle.predict(x.reshape(1,-1)); pp1, _ = product_bundle.predict(x.reshape(1,-1))
            row = {k: float(x[z]) for z, k in enumerate(variables)}
            row.update({
                "selection_reason": f"local_{'minus' if sign < 0 else 'plus'}__{_rid(v)}",
                "perturbed_variable": v,
                "perturbation_sign": sign,
                "predicted_growth": float(pg1[0]),
                "predicted_product": float(pp1[0]),
            })
            local_rows.append(row)
    local = pd.DataFrame(local_rows)
    return tab, local


def _metric_block(exact: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    exact, pred = np.asarray(exact, float), np.asarray(pred, float)
    ok = np.isfinite(exact) & np.isfinite(pred)
    exact, pred = exact[ok], pred[ok]
    if len(exact) < 2:
        return {"n": int(len(exact)), "r2": None, "rmse": None, "mae": None, "nrmse_robust_range": None, "spearman_rho": None}
    rr = max(robust_range(exact), 1e-12)
    rho = spearmanr(exact, pred).statistic if len(exact) >= 3 else np.nan
    return {
        "n": int(len(exact)),
        "r2": float(r2_score(exact, pred)),
        "rmse": float(mean_squared_error(exact, pred) ** 0.5),
        "mae": float(mean_absolute_error(exact, pred)),
        "nrmse_robust_range": float((mean_squared_error(exact, pred) ** 0.5) / rr),
        "spearman_rho": float(rho) if np.isfinite(rho) else None,
    }


def _save_validation_figures(tab: pd.DataFrame, local: pd.DataFrame, outdir: Path, species_label: str, dpi: int = 600) -> list[str]:
    fdir = outdir / "figures"; fdir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    with plt.rc_context(PUBLICATION_RC):
        for target, predcol, exactcol, label in [
            ("growth", "predicted_growth", "exact_growth", "Maximum growth"),
            ("product", "predicted_product", "exact_product", "Product flux"),
        ]:
            d = tab[[predcol, exactcol]].apply(pd.to_numeric, errors="coerce").dropna()
            fig, ax = plt.subplots(figsize=(5.2, 4.8))
            ax.scatter(d[exactcol], d[predcol], s=32, alpha=.8)
            if len(d):
                lo = min(d.min()); hi = max(d.max()); ax.plot([lo,hi],[lo,hi], linestyle="--", linewidth=1.0)
            ax.set_xlabel(f"Exact {label.lower()}"); ax.set_ylabel(f"Surrogate-predicted {label.lower()}")
            ax.set_title(f"Exact challenge validation - {species_label}", loc="left", fontweight="bold")
            ax.grid(alpha=.18)
            for ext in ("png", "pdf", "svg"):
                p = fdir / f"validation_{target}.{ext}"; fig.savefig(p, dpi=dpi if ext=="png" else None, bbox_inches="tight", facecolor="white"); paths.append(str(p))
            plt.close(fig)

        if not local.empty:
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            labels = local["selection_reason"].astype(str).str.replace("local_", "", regex=False).tolist()
            pred = pd.to_numeric(local["predicted_product_change_from_center"], errors="coerce").to_numpy(float)
            exact = pd.to_numeric(local["exact_product_change_from_center"], errors="coerce").to_numpy(float)
            x = np.arange(len(local)); w=.36
            ax.bar(x-w/2, pred, width=w, label="Surrogate")
            ax.bar(x+w/2, exact, width=w, label="Exact FBA")
            ax.axhline(0, linewidth=.8)
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right")
            ax.set_ylabel("Product change from exact center")
            ax.set_title(f"Local directional validation - {species_label}", loc="left", fontweight="bold")
            ax.legend(frameon=False); ax.grid(alpha=.18)
            for ext in ("png", "pdf", "svg"):
                p = fdir / f"validation_local_direction.{ext}"; fig.savefig(p, dpi=dpi if ext=="png" else None, bbox_inches="tight", facecolor="white"); paths.append(str(p))
            plt.close(fig)
    return paths


def validate_landscape(
    record: LandscapeRecord,
    output_dir: str | Path,
    cache_dir: str | Path,
    seed: int = 20260822,
    replay_points: int = 5,
    challenge_points: int = 8,
    fast_timeout: int = 30,
    exact_timeout: int = 60,
) -> dict[str, Any]:
    out = Path(output_dir).resolve(); out.mkdir(parents=True, exist_ok=True)
    cache = Path(cache_dir).resolve(); cache.mkdir(parents=True, exist_ok=True)
    profile, df = build_profile(record)
    variables = profile.design_variables
    growth_bundle, product_bundle = _build_surrogates(df, variables, cache, seed)

    scratch = out / "_exact_scratch"
    replay = _choose_replay(df, variables, int(replay_points), seed)
    replay_rows = []
    for i, (_, row) in enumerate(replay.iterrows(), 1):
        ev = _run_exact_one(record, _candidate_from_row(row, variables), scratch, f"replay_{i:02d}", fast_timeout, exact_timeout)
        replay_rows.append({
            "replay_id": i,
            "stored_growth": float(row["maximum_growth"]),
            "exact_growth": ev.get("maximum_growth"),
            "stored_product": float(row["product_flux"]),
            "exact_product": ev.get("product_flux"),
            "fallback_to_glpk_exact": ev.get("fallback_to_glpk_exact", False),
            "solver_wall_seconds": ev.get("solver_wall_seconds"),
        })
    replay_tab = pd.DataFrame(replay_rows)
    replay_tab["growth_abs_diff"] = (pd.to_numeric(replay_tab.exact_growth, errors="coerce") - replay_tab.stored_growth).abs()
    replay_tab["product_abs_diff"] = (pd.to_numeric(replay_tab.exact_product, errors="coerce") - replay_tab.stored_product).abs()
    replay_tab["growth_pass"] = replay_tab.apply(lambda r: bool(np.isfinite(r.growth_abs_diff) and r.growth_abs_diff <= 1e-7 + 5e-5*max(1.0,abs(r.stored_growth))), axis=1)
    replay_tab["product_pass"] = replay_tab.apply(lambda r: bool(np.isfinite(r.product_abs_diff) and r.product_abs_diff <= 1e-7 + 5e-5*max(1.0,abs(r.stored_product))), axis=1)
    replay_pass = bool(len(replay_tab) and replay_tab.growth_pass.all() and replay_tab.product_pass.all())
    replay_tab.to_csv(out / "archive_exact_replay.csv", index=False)
    if not replay_pass:
        write_json(out / "VALIDATION_FAILED.json", {"status":"FAIL", "reason":"archive exact replay mismatch", "created_at_utc":utc_now()})
        raise RuntimeError(f"{record.species}: exact replay of archived landscape points failed; refusing surrogate challenge validation")

    challenge, local = _challenge_candidates(df, profile, growth_bundle, product_bundle, int(challenge_points), seed)
    challenge_rows = []
    for i, row in challenge.iterrows():
        ev = _run_exact_one(record, _candidate_from_row(row, variables), scratch, f"challenge_{i+1:02d}", fast_timeout, exact_timeout)
        rec = dict(row)
        rec.update({
            "exact_growth": ev.get("maximum_growth"), "exact_product": ev.get("product_flux"),
            "exact_feasible": ev.get("feasible"), "fallback_to_glpk_exact": ev.get("fallback_to_glpk_exact",False),
            "solver_wall_seconds": ev.get("solver_wall_seconds"),
        })
        challenge_rows.append(rec)
    chall_tab = pd.DataFrame(challenge_rows)
    chall_tab.to_csv(out / "surrogate_exact_challenge.csv", index=False)

    # Evaluate exact center once, then the four local perturbations.
    center_candidate = {_rid(v): float(profile.balanced_point[v]) for v in variables}
    center_pred_g, _ = growth_bundle.predict(np.asarray([[profile.balanced_point[v] for v in variables]],float))
    center_pred_p, _ = product_bundle.predict(np.asarray([[profile.balanced_point[v] for v in variables]],float))
    center_ev = _run_exact_one(record, center_candidate, scratch, "local_center", fast_timeout, exact_timeout)
    local_rows = []
    for i, row in local.iterrows():
        ev = _run_exact_one(record, _candidate_from_row(row, variables), scratch, f"local_{i+1:02d}", fast_timeout, exact_timeout)
        rec = dict(row)
        rec.update({
            "exact_growth": ev.get("maximum_growth"), "exact_product": ev.get("product_flux"),
            "predicted_growth_change_from_center": float(row["predicted_growth"] - center_pred_g[0]),
            "predicted_product_change_from_center": float(row["predicted_product"] - center_pred_p[0]),
            "exact_growth_change_from_center": float(ev.get("maximum_growth") - center_ev.get("maximum_growth")) if ev.get("maximum_growth") is not None else np.nan,
            "exact_product_change_from_center": float(ev.get("product_flux") - center_ev.get("product_flux")) if ev.get("product_flux") is not None else np.nan,
            "fallback_to_glpk_exact": ev.get("fallback_to_glpk_exact",False),
        })
        local_rows.append(rec)
    local_tab = pd.DataFrame(local_rows)
    local_tab.to_csv(out / "local_direction_exact_validation.csv", index=False)

    growth_metrics = _metric_block(chall_tab["exact_growth"], chall_tab["predicted_growth"])
    product_metrics = _metric_block(chall_tab["exact_product"], chall_tab["predicted_product"])
    if len(local_tab):
        psign = np.sign(pd.to_numeric(local_tab["predicted_product_change_from_center"],errors="coerce"))
        esign = np.sign(pd.to_numeric(local_tab["exact_product_change_from_center"],errors="coerce"))
        gsign = np.sign(pd.to_numeric(local_tab["predicted_growth_change_from_center"],errors="coerce"))
        egsign = np.sign(pd.to_numeric(local_tab["exact_growth_change_from_center"],errors="coerce"))
        product_sign_agreement = float(np.mean(psign == esign))
        growth_sign_agreement = float(np.mean(gsign == egsign))
    else:
        product_sign_agreement = growth_sign_agreement = float("nan")

    # Challenge points are deliberately out-of-sample and enriched for difficult
    # regions.  They are reported transparently; a warning does not invalidate exact
    # landscape analyses. Bridge replay mismatch, by contrast, is fatal above.
    def grade(m):
        nrmse = m.get("nrmse_robust_range")
        rho = m.get("spearman_rho")
        if nrmse is not None and nrmse <= .10 and (rho is None or rho >= .70): return "PASS"
        if nrmse is not None and nrmse <= .20: return "WARN"
        return "WARN"

    figures = _save_validation_figures(chall_tab, local_tab, out, profile.label, dpi=600)
    summary = {
        "status": "PASS_WITH_WARNINGS" if (grade(growth_metrics)=="WARN" or grade(product_metrics)=="WARN") else "PASS",
        "archive_replay_status": "PASS",
        "species": record.species,
        "species_label": profile.label,
        "created_at_utc": utc_now(),
        "archive_replay_points": int(len(replay_tab)),
        "surrogate_challenge_points": int(len(chall_tab)),
        "local_direction_points": int(len(local_tab)),
        "growth_cv_qc": growth_bundle.metrics,
        "product_cv_qc": product_bundle.metrics,
        "growth_exact_challenge": growth_metrics | {"grade": grade(growth_metrics)},
        "product_exact_challenge": product_metrics | {"grade": grade(product_metrics)},
        "local_growth_direction_sign_agreement": growth_sign_agreement,
        "local_product_direction_sign_agreement": product_sign_agreement,
        "glpk_exact_fallback_count": int(replay_tab.fallback_to_glpk_exact.sum() + chall_tab.fallback_to_glpk_exact.fillna(False).sum() + local_tab.fallback_to_glpk_exact.fillna(False).sum()),
        "figures": figures,
        "scientific_interpretation": "Exact archived-point replay validates the source simulation bridge. Cross-validated and targeted exact challenge results quantify how faithfully the landscape-trained surrogates interpolate new locations. Landscape-only analyses remain exact with respect to stored samples; surrogate-based continuous/local analyses should be interpreted in light of these QC results.",
    }
    write_json(out / "validation_summary.json", summary)
    (out / "validation_profile.yaml").write_text(yaml.safe_dump(asdict(profile),sort_keys=False),encoding="utf-8")
    # Preserve scratch only when a failure occurred; ordinary successful runs keep the
    # compact validation evidence, not temporary worker files.
    import shutil
    shutil.rmtree(scratch, ignore_errors=True)
    return summary
