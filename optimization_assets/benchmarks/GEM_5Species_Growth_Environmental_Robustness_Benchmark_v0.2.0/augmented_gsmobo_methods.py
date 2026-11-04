from __future__ import annotations

"""Four conservative landscape-augmented gsMOBO+50k methods.

The BO backbone is intentionally held fixed:
  * same matched 16-call blind Sobol initialization as BASE_GSMOBO_QPAREGO_50K;
  * same representative exact 50k archive subset + live exact data;
  * same independent-output GP model;
  * same qParEGO random weights and qLogExpectedImprovement acquisition;
  * same exact GEM/FBA closure and matched logical budgets.

The ONLY algorithmic addition is acquisition-optimization initialization from
GEM-landscape structures that are unavailable to ordinary gsMOBO.  Standard
qParEGO candidates are always computed first and remain eligible.  For each
batch slot, the landscape-derived seeds compete with the ordinary candidate
under the *same* qLogEI scalarization.  Replacement is allowed only when the
landscape-seeded solution has a strictly higher acquisition value by a frozen
margin.  One ordinary qParEGO candidate is protected in every four-call batch.

Methods
-------
NEW_GSMOBO50K_GHOST_RESTART
    Cross-fitted Pareto false-negative / blind-spot acquisition restarts.
NEW_GSMOBO50K_SPARC_RESTART
    Exact productive-displacement subspace acquisition restarts.
NEW_GSMOBO50K_TRAIL_RESTART
    Exact monotone archive-trail extrapolation acquisition restarts.
NEW_GSMOBO50K_MOSAIC_RESTART
    Exact near-Pareto metabolic-regime atlas acquisition restarts.
"""

import math
import time
import warnings
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans

from core import (
    candidate_hv_gain,
    continuous_qparego_select,
    fit_gsmobo_gp,
    robust_scale,
)
from live_core import LiveContext, baseline_pareto_row_indices, duration, sobol_points
from novel_methods import (
    _ghost_blindspot_bank,
    _ghost_candidates,
    _sparc_basis,
    _sparc_candidates,
    _trail_lines,
    _trail_candidates,
    _live_extension_centers,
)


AUGMENTED_METHODS = [
    "NEW_GSMOBO50K_GHOST_RESTART",
    "NEW_GSMOBO50K_SPARC_RESTART",
    "NEW_GSMOBO50K_TRAIL_RESTART",
    "NEW_GSMOBO50K_MOSAIC_RESTART",
]

AUGMENTED_METHOD_NOTES = {
    "NEW_GSMOBO50K_GHOST_RESTART": (
        "gsMOBO+50k backbone with acquisition restarts seeded from exact Pareto-quality "
        "archive rows that were cross-fitted surrogate false negatives (GHOST)."
    ),
    "NEW_GSMOBO50K_SPARC_RESTART": (
        "gsMOBO+50k backbone with acquisition restarts drawn from a low-dimensional "
        "productive displacement subspace learned from prepared 50k-prior movements toward elites (SPARC)."
    ),
    "NEW_GSMOBO50K_TRAIL_RESTART": (
        "gsMOBO+50k backbone with acquisition restarts from extrapolated exact monotone "
        "archive improvement trails and live exact extensions (TRAIL)."
    ),
    "NEW_GSMOBO50K_MOSAIC_RESTART": (
        "gsMOBO+50k backbone with acquisition restarts covering distinct exact near-Pareto "
        "metabolic design regimes mined from the persistent archive (MOSAIC)."
    ),
}


def _remaining(ctx: LiveContext, method: str) -> int:
    return max(0, int(ctx.budget) - len(ctx.history(method)))


def _q(ctx: LiveContext, method: str) -> int:
    return min(int(ctx.batch_size), _remaining(ctx, method))


def _progress(ctx: LiveContext, method: str, t0: float, extra: str = "") -> None:
    h = ctx.history(method)
    print(
        f"[{method}] {len(h):3d}/{ctx.budget} | exact ΔHV={ctx.current_delta_hv(method):.9f}"
        f"{extra} | batch {duration(time.time()-t0)}",
        flush=True,
    )


def _cold_blind_pool(ctx: LiveContext, method: str, batch_no: int, rows: int, seed_offset: int = 0) -> np.ndarray:
    """Exact clone of the matched blind Sobol logic used by gsMOBO."""
    n = int(rows)
    seed = int(ctx.seed) + 1009 * int(batch_no) + int(seed_offset)
    X = sobol_points(max(n * 2, n + 64), len(ctx.features), seed)
    X = ctx._filter_new(X, method, exclude_frozen_archive=False)
    if len(X) < n:
        X2 = sobol_points(n * 2, len(ctx.features), seed + 17)
        X = np.vstack([X, ctx._filter_new(X2, method, exclude_frozen_archive=False)])
        X = ctx._filter_new(X, method, exclude_frozen_archive=False)
    return X[:n]


def _history_novel_mask(ctx: LiveContext, method: str, X: np.ndarray, selected: np.ndarray | None = None) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X, float))
    keep = np.isfinite(X).all(axis=1)
    h = ctx.history(method)
    if len(h):
        cols = [f"x01__{f}" for f in ctx.features]
        Xh = h[cols].to_numpy(float)
        keep &= cKDTree(Xh).query(X, k=1)[0] > ctx.min_dwithin
    if selected is not None and len(selected):
        keep &= cKDTree(np.atleast_2d(selected)).query(X, k=1)[0] > ctx.min_dwithin
    return keep


def _fit_50k_gsmobo(ctx: LiveContext, method: str, archive_rows: int):
    hist = ctx.history(method)
    xcols = [f"x01__{f}" for f in ctx.features]
    Xtr = hist[xcols].to_numpy(float)
    Ytr = hist[["exact_growth_norm", "exact_product_norm"]].to_numpy(float)
    Ftr = hist["exact_feasible"].astype(bool).to_numpy()
    Ylive = Ytr.copy()
    bad = (~Ftr) | (~np.isfinite(Ylive).all(axis=1))
    Ylive[bad] = 0.0

    sub = np.asarray(ctx.subset_local[: min(int(archive_rows), len(ctx.baseline))], int)
    Xa = ctx.X0[sub]
    Ya = ctx.Y0[sub].copy()
    Fa = ctx.F0[sub]
    bada = (~Fa) | (~np.isfinite(Ya).all(axis=1))
    Ya[bada] = 0.0

    Xfit = np.vstack([Xa, Xtr])
    Yfit = np.vstack([Ya, Ylive])
    # Preserve later live observations if a blind initialization coincides with an archive row.
    last = {}
    for ii, xx in enumerate(Xfit):
        last[tuple(np.round(xx, 12).tolist())] = ii
    keep_idx = np.asarray(sorted(last.values()), int)
    Xfit = Xfit[keep_idx]
    Yfit = Yfit[keep_idx]
    archive_train_rows = max(0, len(Xfit) - len(Xtr))
    return Xtr, Ytr, Ftr, Xfit, Yfit, archive_train_rows, fit_gsmobo_gp(Xfit, Yfit)


def _make_qparego_acq(model, Xfit: np.ndarray, Yfit: np.ndarray, weight: np.ndarray, seed: int, mc_samples: int, pending: np.ndarray | None = None):
    """Rebuild one qParEGO qLogEI acquisition using a frozen returned weight."""
    from core import require_botorch
    torch, _, _ = require_botorch()
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization

    dtype = torch.double
    tx = torch.tensor(np.asarray(Xfit, float), dtype=dtype)
    ty = torch.tensor(np.asarray(Yfit, float), dtype=dtype)
    model.eval()
    with torch.no_grad():
        scalarization_reference = model.posterior(tx).mean
    tw = torch.tensor(np.asarray(weight, float), dtype=dtype)
    objective = GenericMCObjective(get_chebyshev_scalarization(weights=tw, Y=scalarization_reference))
    with torch.no_grad():
        best_f = objective(ty).max()
    sampler = SobolQMCNormalSampler(torch.Size([int(mc_samples)]), seed=int(seed))
    acq = qLogExpectedImprovement(model=model, best_f=best_f, sampler=sampler, objective=objective)
    if pending is not None and len(pending):
        acq.set_X_pending(torch.tensor(np.asarray(pending, float), dtype=dtype))
    return acq


def _score_acq(acq, X: np.ndarray, batch_size: int = 512) -> np.ndarray:
    from core import require_botorch
    torch, _, _ = require_botorch()
    X = np.atleast_2d(np.asarray(X, float))
    out = []
    with torch.no_grad():
        for s in range(0, len(X), int(batch_size)):
            tx = torch.tensor(X[s:s+batch_size], dtype=torch.double).unsqueeze(1)
            v = acq(tx)
            out.append(np.asarray(v.detach().cpu().numpy(), float).reshape(-1))
    return np.concatenate(out) if out else np.empty(0, float)


def _refine_from_initial_conditions(acq, baseline_x: np.ndarray, proposal_x: np.ndarray, maxiter: int, max_restarts: int):
    """Local acquisition refinement with baseline + informed seeds; safe fallback on any API/numerical issue."""
    from core import require_botorch
    torch, _, _ = require_botorch()
    from botorch.optim.optimize import optimize_acqf

    base = np.asarray(baseline_x, float).reshape(1, -1)
    P = np.atleast_2d(np.asarray(proposal_x, float)) if len(proposal_x) else np.empty((0, base.shape[1]))
    seeds = np.vstack([base, P])[: max(1, int(max_restarts))]
    ics = torch.tensor(seeds, dtype=torch.double).unsqueeze(1)
    d = base.shape[1]
    bounds = torch.stack([torch.zeros(d, dtype=torch.double), torch.ones(d, dtype=torch.double)])
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cand, val = optimize_acqf(
                acq_function=acq,
                bounds=bounds,
                q=1,
                num_restarts=len(seeds),
                raw_samples=None,
                batch_initial_conditions=ics,
                options={"batch_limit": min(10, len(seeds)), "maxiter": int(maxiter)},
                return_best_only=True,
            )
        x = np.asarray(cand.detach().cpu().numpy(), float).reshape(-1, d)[0]
        vv = np.asarray(val.detach().cpu().numpy(), float).reshape(-1)
        score = float(vv[0]) if len(vv) else float(_score_acq(acq, x[None, :])[0])
        return np.clip(x, 0.0, 1.0), score, True
    except Exception:
        # The scientific run must not die because a version-specific BoTorch
        # custom-initial-condition path failed.  Direct seed competition remains valid.
        return base[0], float(_score_acq(acq, base)[0]), False


def _prefilter_by_landscape(ctx: LiveContext, method: str, X: np.ndarray, structural_score: np.ndarray | None, max_rows: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.atleast_2d(np.asarray(X, float))
    if len(X) == 0:
        return X, np.empty(0, float)
    keep = _history_novel_mask(ctx, method, X)
    X = X[keep]
    if structural_score is None:
        structural = np.zeros(len(keep), float)[keep]
    else:
        structural = np.asarray(structural_score, float).reshape(-1)[keep]
    if len(X) == 0:
        return X, structural
    pred, pinf = ctx.global_model.predict(X)
    safe = np.isfinite(pred).all(axis=1) & np.isfinite(pinf) & (pinf <= ctx.safe_threshold)
    hvi = candidate_hv_gain(ctx.current_front(method), pred)
    score = 2.0 * robust_scale(hvi) + 0.65 * robust_scale(np.maximum(structural, 0.0))
    score[~safe] = -np.inf
    ids = np.where(np.isfinite(score))[0]
    if len(ids) > int(max_rows):
        ids = ids[np.argsort(score[ids])[::-1][: int(max_rows)]]
    return X[ids], structural[ids]


# ---------------------------------------------------------------------------
# Landscape seed generators
# ---------------------------------------------------------------------------


def _ghost_seed_pool(ctx: LiveContext, method: str, batch_no: int, hist_n: int, max_rows: int):
    bank = _ghost_blindspot_bank(ctx)
    X, source, blind_prior, _, _ = _ghost_candidates(ctx, method, batch_no, hist_n, bank)
    X, structural = _prefilter_by_landscape(ctx, method, X, blind_prior, max_rows)
    labels = np.asarray(["GHOST_false_negative"] * len(X), object)
    return X, labels, structural


def _sparc_seed_pool(ctx: LiveContext, method: str, batch_no: int, hist_n: int, max_rows: int):
    B, s, energy = _sparc_basis(ctx, method)
    X, labels = _sparc_candidates(ctx, method, batch_no, B)
    # Productive-subspace strength is global for the pool; predicted HVI performs
    # the final landscape prefilter while keeping the subspace mechanism intact.
    structural = np.full(len(X), float(energy), float)
    X, structural = _prefilter_by_landscape(ctx, method, X, structural, max_rows)
    labels = np.asarray([f"SPARC_rank{B.shape[1]}"] * len(X), object)
    return X, labels, structural


def _trail_seed_pool(ctx: LiveContext, method: str, batch_no: int, hist_n: int, max_rows: int):
    lines = _trail_lines(ctx, method)
    if not lines:
        return np.empty((0, len(ctx.features))), np.empty(0, object), np.empty(0, float)
    X, labels, support, anchor_y, normal = _trail_candidates(ctx, method, batch_no, lines)
    X, structural = _prefilter_by_landscape(ctx, method, X, np.asarray(support, float), max_rows)
    labels = np.asarray(["TRAIL_exact_path"] * len(X), object)
    return X, labels, structural


def _mosaic_seed_pool(ctx: LiveContext, method: str, batch_no: int, hist_n: int, max_rows: int):
    """Build restarts from distinct exact near-Pareto design regimes.

    The regime atlas is rebuilt from the prepared 50k-prior objective table; in this benchmark nominal growth is exact and robustness is surrogate-derived.
    Multiple scalarization elites are unioned, clustered in normalized design
    space, and jittered locally.  This gives qParEGO restarts in disconnected or
    narrow metabolic basins that ordinary random raw starts may miss.
    """
    cfg = ctx.cfg.get("augmented_gsmobo", {}).get("mosaic", {})
    good_ids = np.where(ctx.good0)[0]
    Xg = ctx.X0[good_ids]
    Yg = ctx.Y0[good_ids]
    if len(Xg) < 20:
        return np.empty((0, len(ctx.features))), np.empty(0, object), np.empty(0, float)

    nweights = int(cfg.get("weights", 15))
    elite_per_weight = int(cfg.get("elite_per_weight", 48))
    ids = set()
    for w in np.linspace(0.0, 1.0, nweights):
        u = (1.0-w) * Yg[:, 0] + w * Yg[:, 1]
        for j in np.argsort(u)[-min(elite_per_weight, len(u)):]:
            ids.add(int(j))
    pids = baseline_pareto_row_indices(ctx)
    global_to_good = {int(gid): i for i, gid in enumerate(good_ids)}
    for p in pids:
        if int(p) in global_to_good:
            ids.add(global_to_good[int(p)])
    ids = np.asarray(sorted(ids), int)
    Xelite = Xg[ids]
    Yelite = Yg[ids]

    k = int(cfg.get("regimes", 12))
    k = max(2, min(k, len(Xelite)))
    km = KMeans(n_clusters=k, n_init=5, random_state=int(ctx.seed) + 7301).fit(Xelite)
    lab = km.labels_

    # Exact quality envelope independent of the new BO GP.
    qenv = np.maximum(Yelite[:, 0], Yelite[:, 1]) + 0.35 * np.minimum(Yelite[:, 0], Yelite[:, 1])
    centers = []
    center_q = []
    center_lab = []
    reps_per_regime = int(cfg.get("reps_per_regime", 4))
    for c in range(k):
        loc = np.where(lab == c)[0]
        if not len(loc):
            continue
        take = loc[np.argsort(qenv[loc])[::-1][: min(reps_per_regime, len(loc))]]
        for j in take:
            centers.append(Xelite[j]); center_q.append(float(qenv[j])); center_lab.append(int(c))

    Xlive, Ylive = _live_extension_centers(ctx, method)
    for i, x in enumerate(Xlive):
        centers.append(x)
        center_q.append(float(5.0 + Ylive[i].sum()))
        center_lab.append(k + i)

    if not centers:
        return np.empty((0, len(ctx.features))), np.empty(0, object), np.empty(0, float)
    centers = np.asarray(centers, float)
    center_q = np.asarray(center_q, float)
    center_lab = np.asarray(center_lab, int)

    rng = np.random.default_rng(int(ctx.seed) + 740_003 + 1009*batch_no + hist_n)
    rows_per_center = int(cfg.get("rows_per_center", 96))
    scales = np.asarray(cfg.get("jitter_scales", [0.006, 0.015, 0.035, 0.070, 0.120]), float)
    blocks=[]; structural=[]; labels=[]
    for i, c in enumerate(centers):
        n = rows_per_center
        s = rng.choice(scales, size=n)
        Z = np.clip(c + rng.normal(size=(n, len(ctx.features))) * s[:, None], 0.0, 1.0)
        blocks.append(Z)
        structural.extend([center_q[i]] * n)
        labels.extend([f"MOSAIC_regime{center_lab[i]}"] * n)
    X = np.vstack(blocks)
    structural = np.asarray(structural, float)
    X, structural = _prefilter_by_landscape(ctx, method, X, structural, max_rows)
    return X, np.asarray(["MOSAIC_exact_regime"] * len(X), object), structural


GENERATORS = {
    "ghost": _ghost_seed_pool,
    "sparc": _sparc_seed_pool,
    "trail": _trail_seed_pool,
    "mosaic": _mosaic_seed_pool,
}


@dataclass
class _SlotDecision:
    x: np.ndarray
    mode: str
    base_acq: float
    selected_acq: float
    used_landscape: bool
    seed_source: str
    refined: bool
    pool_rows: int


def _choose_augmented_batch(
    ctx: LiveContext,
    method: str,
    model,
    Xfit: np.ndarray,
    Yfit: np.ndarray,
    base_cand: np.ndarray,
    weights: np.ndarray,
    proposal_X: np.ndarray,
    proposal_labels: np.ndarray,
    q: int,
    seed: int,
    mc_samples: int,
    cfg: dict,
) -> list[_SlotDecision]:
    min_adv = float(cfg.get("min_logei_advantage", 0.035))
    topk = int(cfg.get("restart_topk", 12))
    maxiter = int(cfg.get("refine_maxiter", 90))
    protected = max(0, min(q, int(cfg.get("protected_base_slots", 1))))
    score_batch = int(cfg.get("acq_score_batch", 512))

    selected = []
    out: list[_SlotDecision] = []
    P = np.atleast_2d(np.asarray(proposal_X, float)) if len(proposal_X) else np.empty((0, Xfit.shape[1]))
    plabel = np.asarray(proposal_labels, object) if len(proposal_labels) else np.empty(0, object)

    for i in range(q):
        pending = np.asarray(selected, float) if selected else None
        acq = _make_qparego_acq(
            model, Xfit, Yfit, weights[i], seed=int(seed)+1, mc_samples=mc_samples, pending=pending
        )
        bx = np.asarray(base_cand[i], float)
        base_val = float(_score_acq(acq, bx[None, :], batch_size=score_batch)[0])

        # One ordinary qParEGO anchor per batch is deliberately protected.
        if i < protected or len(P) == 0:
            chosen = bx; chosen_val = base_val; use = False; src = "protected_base_qParEGO"; refined=False
        else:
            keep = _history_novel_mask(ctx, method, P, np.asarray(selected, float) if selected else None)
            Pc = P[keep]
            Lc = plabel[keep]
            if len(Pc) == 0:
                chosen = bx; chosen_val = base_val; use=False; src="base_no_novel_aug_seed"; refined=False
            else:
                vals = _score_acq(acq, Pc, batch_size=score_batch)
                order = np.argsort(vals)[::-1]
                take = order[: min(topk, len(order))]
                topP = Pc[take]
                best_seed_val = float(vals[take[0]]) if len(take) else -np.inf
                best_seed = topP[0] if len(topP) else bx
                best_label = str(Lc[take[0]]) if len(take) else "none"

                # Local acquisition refinement is initialized by BOTH the standard
                # qParEGO point and the landscape seeds. If the custom-IC BoTorch
                # path is unavailable, direct seed competition remains active.
                rx, rval, refined = _refine_from_initial_conditions(
                    acq, bx, topP, maxiter=maxiter, max_restarts=1+len(topP)
                )
                # If refinement fell back to base but a raw informed seed has a
                # stronger acquisition, preserve that raw informed option.
                if best_seed_val > rval:
                    rx, rval, refined = best_seed, best_seed_val, False

                novel_ref = bool(_history_novel_mask(
                    ctx, method, rx[None, :], np.asarray(selected, float) if selected else None
                )[0])
                if novel_ref and np.isfinite(rval) and rval >= base_val + min_adv:
                    chosen = rx; chosen_val = float(rval); use=True; src=best_label
                else:
                    chosen = bx; chosen_val = base_val; use=False; src="base_qParEGO_gate"; refined=False

        selected.append(np.asarray(chosen, float))
        out.append(_SlotDecision(
            x=np.asarray(chosen, float),
            mode=("LANDSCAPE_RESTART" if use else "BASE_qParEGO"),
            base_acq=float(base_val),
            selected_acq=float(chosen_val),
            used_landscape=bool(use),
            seed_source=str(src),
            refined=bool(refined),
            pool_rows=int(len(P)),
        ))
    return out


def _run_augmented_gsmobo(ctx: LiveContext, method: str, generator_key: str):
    scfg = ctx.cfg["study"]
    gcfg = ctx.cfg.get("gsmobo", {})
    acfg = ctx.cfg.get("augmented_gsmobo", {})
    initial_calls = int(gcfg.get("initial_calls", scfg.get("cold_start_initial_calls", 16)))
    min_feasible = int(gcfg.get("min_feasible_for_bo", 2))
    mc_samples = int(gcfg.get("mc_samples", 256))
    num_restarts = int(gcfg.get("num_restarts", 5))
    raw_samples = int(gcfg.get("raw_samples", 512))
    maxiter = int(gcfg.get("maxiter", 200))
    fallback_rows = int(gcfg.get("fallback_pool_rows", scfg.get("cold_start_pool_rows", 8192)))
    archive_rows = min(int(gcfg.get("archive_rows", scfg.get("gp_subset_rows", 1024))), len(ctx.baseline))
    proposal_max = int(acfg.get("proposal_max_rows", 768))
    generator = GENERATORS[generator_key]

    while _remaining(ctx, method) > 0:
        t0 = time.time()
        hist = ctx.history(method)
        hist_n = len(hist)
        q = _q(ctx, method)
        bno = hist_n // int(ctx.batch_size) + 1

        # Strictly matched initialization with BASE_GSMOBO_QPAREGO_50K.
        if hist_n < initial_calls:
            q0 = min(q, initial_calls - hist_n)
            X = _cold_blind_pool(ctx, method, bno, max(fallback_rows, q0))[:q0]
            ctx.evaluate_batch(
                method, X, [f"AUG_{generator_key.upper()}_MATCHED_INIT_SOBOL"] * len(X),
                {
                    "selection_uses_frozen_landscape": np.zeros(len(X), dtype=bool),
                    "augmented_gsmobo_phase": ["matched_initialization"] * len(X),
                    "augmented_generator": [generator_key] * len(X),
                    "aug_used_landscape_restart": np.zeros(len(X), dtype=bool),
                    "aug_base_acq": np.full(len(X), np.nan),
                    "aug_selected_acq": np.full(len(X), np.nan),
                    "aug_acq_advantage": np.full(len(X), np.nan),
                    "aug_seed_source": ["matched_blind_sobol"] * len(X),
                    "aug_refined": np.zeros(len(X), dtype=bool),
                    "aug_proposal_pool_rows": np.zeros(len(X), dtype=int),
                },
                exclude_frozen_archive=False,
            )
            _progress(ctx, method, t0, extra=f" | matched init={len(ctx.history(method))}/{initial_calls}")
            continue

        hist = ctx.history(method)
        good = hist["exact_feasible"].astype(bool).to_numpy() & np.isfinite(
            hist[["exact_growth_norm","exact_product_norm"]].to_numpy(float)
        ).all(axis=1)
        if int(good.sum()) < min_feasible:
            X = _cold_blind_pool(ctx, method, bno, max(fallback_rows, q), seed_offset=6000+hist_n)[:q]
            ctx.evaluate_batch(
                method, X, [f"AUG_{generator_key.upper()}_FEASIBILITY_SOBOL"]*len(X),
                {
                    "selection_uses_frozen_landscape": np.zeros(len(X), dtype=bool),
                    "augmented_gsmobo_phase": ["feasibility_extension"]*len(X),
                    "augmented_generator": [generator_key]*len(X),
                    "aug_used_landscape_restart": np.zeros(len(X), dtype=bool),
                },
                exclude_frozen_archive=False,
            )
            _progress(ctx, method, t0, extra=f" | feasible train={int(good.sum())}/{min_feasible}")
            continue

        Xtr, Ytr, Ftr, Xfit, Yfit, archive_train_rows, model = _fit_50k_gsmobo(ctx, method, archive_rows)

        # Unmodified gsMOBO candidate batch first. Same qParEGO randomization as
        # BASE_GSMOBO_QPAREGO_50K at the same live-call count.
        base_cand, base_acq_orig, weights = continuous_qparego_select(
            model, Xfit, Yfit, q=q,
            seed=int(ctx.seed) + 41000 + hist_n,
            mc_samples=mc_samples,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            maxiter=maxiter,
        )
        base_cand = np.clip(np.asarray(base_cand, float), 0.0, 1.0)
        weights = np.asarray(weights, float).reshape(q, 2)

        # GEM-specific acquisition restart pool.  These are NEVER exact-evaluated
        # merely because the landscape likes them; they must beat the ordinary BO
        # candidate under the same qParEGO acquisition gate.
        PX, Plabel, Pstruct = generator(ctx, method, bno, hist_n, proposal_max)

        decisions = _choose_augmented_batch(
            ctx, method, model, Xfit, Yfit, base_cand, weights,
            PX, Plabel, q=q,
            seed=int(ctx.seed) + 41000 + hist_n,
            mc_samples=mc_samples,
            cfg=acfg,
        )

        Xsel = np.vstack([d.x for d in decisions])
        # Defensive uniqueness. If an informed replacement collided after
        # refinement, fall back to the corresponding ordinary qParEGO point.
        selected=[]
        for i, d in enumerate(decisions):
            x = Xsel[i]
            if selected and cKDTree(np.asarray(selected)).query(x, k=1)[0] <= ctx.min_dwithin:
                x = base_cand[i]
                d.x = x; d.mode="BASE_qParEGO_DUPLICATE_GUARD"; d.used_landscape=False; d.seed_source="base_duplicate_guard"
            selected.append(x)
        Xsel = np.asarray(selected, float)

        ctx.evaluate_batch(
            method,
            Xsel,
            [f"{d.mode}|{generator_key.upper()}" for d in decisions],
            {
                "selection_uses_frozen_landscape": np.ones(len(decisions), dtype=bool),
                "gsmobo_uses_50k_archive": np.ones(len(decisions), dtype=bool),
                "augmented_gsmobo_phase": ["qparego_landscape_restart"]*len(decisions),
                "augmented_generator": [generator_key]*len(decisions),
                "gsmobo_train_calls": np.full(len(decisions), len(Xtr), dtype=int),
                "gsmobo_archive_train_rows": np.full(len(decisions), archive_train_rows, dtype=int),
                "gsmobo_feasible_train": np.full(len(decisions), int(good.sum()), dtype=int),
                "gsmobo_weight_growth": weights[:,0],
                "gsmobo_weight_product": weights[:,1],
                "aug_used_landscape_restart": np.asarray([d.used_landscape for d in decisions], bool),
                "aug_base_acq": np.asarray([d.base_acq for d in decisions], float),
                "aug_selected_acq": np.asarray([d.selected_acq for d in decisions], float),
                "aug_acq_advantage": np.asarray([d.selected_acq-d.base_acq for d in decisions], float),
                "aug_seed_source": [d.seed_source for d in decisions],
                "aug_refined": np.asarray([d.refined for d in decisions], bool),
                "aug_proposal_pool_rows": np.asarray([d.pool_rows for d in decisions], int),
            },
            exclude_frozen_archive=False,
        )
        naug = int(sum(d.used_landscape for d in decisions))
        _progress(
            ctx, method, t0,
            extra=f" | gsMOBO50k train={len(Xfit)} | {generator_key} restarts used={naug}/{q} pool={len(PX)}",
        )


def run_gsmobo50k_ghost_restart(ctx: LiveContext):
    return _run_augmented_gsmobo(ctx, "NEW_GSMOBO50K_GHOST_RESTART", "ghost")


def run_gsmobo50k_sparc_restart(ctx: LiveContext):
    return _run_augmented_gsmobo(ctx, "NEW_GSMOBO50K_SPARC_RESTART", "sparc")


def run_gsmobo50k_trail_restart(ctx: LiveContext):
    return _run_augmented_gsmobo(ctx, "NEW_GSMOBO50K_TRAIL_RESTART", "trail")


def run_gsmobo50k_mosaic_restart(ctx: LiveContext):
    return _run_augmented_gsmobo(ctx, "NEW_GSMOBO50K_MOSAIC_RESTART", "mosaic")


AUGMENTED_RUNNERS = {
    "NEW_GSMOBO50K_GHOST_RESTART": run_gsmobo50k_ghost_restart,
    "NEW_GSMOBO50K_SPARC_RESTART": run_gsmobo50k_sparc_restart,
    "NEW_GSMOBO50K_TRAIL_RESTART": run_gsmobo50k_trail_restart,
    "NEW_GSMOBO50K_MOSAIC_RESTART": run_gsmobo50k_mosaic_restart,
}
