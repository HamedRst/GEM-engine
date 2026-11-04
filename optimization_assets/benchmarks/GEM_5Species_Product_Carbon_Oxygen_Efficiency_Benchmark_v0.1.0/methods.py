from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from core import (
    candidate_hv_gain,
    continuous_qparego_select,
    discrete_qlogehvi_select,
    fit_gsmobo_gp,
    fit_prior_gp,
    fit_step4c_gp,
    hypervolume_2d,
    pareto_points,
    posterior_mean_std,
    robust_scale,
    save_json,
)
from live_core import (
    LiveContext,
    baseline_pareto_row_indices,
    candidate_targets_from_front,
    choose_diverse_by_score,
    duration,
    greedy_predicted_hv,
    phenotype_distance_to_front,
    sobol_points,
)


METHOD_NOTES = {
    "BASE_SOBOL_PASSIVE": "Passive scrambled-Sobol control; genuinely new continuous conditions, no surrogate safety filtering.",
    "BASE_COLD_START_BO": "Cold-start qLogEHVI baseline with zero frozen-landscape knowledge for selection: the first 16 logical calls are matched Sobol initialization, then a GP/qLogEHVI model is trained only on this method's own live exact observations. The frozen 50k archive is used only afterward for evaluation/reporting, never for candidate selection.",
    "BASE_GSMOBO_QPAREGO_NO50K": "Matched-budget gsMOBO/qParEGO comparator with NO frozen-landscape knowledge for selection: calls 1–16 use blind Sobol initialization; afterward independent GPs are fit only to this method's own exact observations and qParEGO uses random augmented-Chebyshev scalarizations with qLogExpectedImprovement.",
    "BASE_GSMOBO_QPAREGO_50K": "Matched-budget 50k-informed gsMOBO/qParEGO ablation: it uses the SAME 16 live Sobol initialization and qParEGO acquisition as the NO50K version, but after initialization its independent GPs are warm-started with the established representative subset of the prepared 50k prior (exact product plus deterministic carbon/oxygen efficiencies, with frozen growth-viability labels) (all recoverable Pareto rows first, then Sobol-nearest global design coverage) plus its own live exact observations.",
    "BASE_NSGA2_NO50K": "Matched-budget canonical NSGA-II evolutionary baseline with NO frozen-landscape knowledge for selection: calls 1–16 use the same blind Sobol initialization; afterward exact live observations drive feasibility-aware nondominated sorting, crowding-distance survival, binary tournament selection, simulated-binary crossover (SBX), and polynomial mutation.",
    "BASE_NSGA2_50K": "Matched-budget 50k-informed NSGA-II warm-start ablation: it uses the SAME 16 live Sobol initialization and the same evolutionary operators as NO50K, but after initialization its parent population is warm-started from prepared 50k prior Pareto/representative designs plus the live initialization. All subsequent offspring are new continuous conditions and are exact-FBA validated.",
    "BASE_GLOBAL_STATIC": "Static global-surrogate Pareto search; positive predicted-HV candidates first, with an explicit Pareto-near + design-novelty fallback when predicted HV is flat/zero. No exact-feedback model update.",
    "BASE_GLOBAL_MULTIMODE": "Adaptive UCB allocation across Pareto/product/balanced/constrained/robust/novelty modes.",
    "BASE_STEP4C_BO": "Step4C-style 1,024 representative exact archive GP + discrete qLogEHVI over fresh continuous Sobol candidates.",
    "BASE_PRIOR_GP_PARETO": "Full-landscape global surrogate as GP mean/prior plus online exact correction; qLogEHVI on fresh candidates.",
    "IDEA01_SENSITIVITY_NAV": "PROXY: surrogate finite-difference sensitivity navigation; not the future solver dual/active-set implementation.",
    "IDEA02_PARETO_GAP_TRACE": "Gap-diverse Pareto continuation: rotate across unresolved phenotype gaps, then search each target with the global landscape surrogate instead of allowing the largest gap to monopolize a batch.",
    "IDEA03_OPT_WEIGHTED_SURROGATE": "Optimization-weighted global surrogate emphasizing exact Pareto-near/extreme archive rows.",
    "IDEA04_INVERSE_DESIGN": "Gap-diverse goal-conditioned inverse design using exact phenotype prototypes plus surrogate phenotype targeting; target assignments rotate across accumulated calls for high-budget runs.",
    "IDEA05_ROBUST_ADVERSARIAL": "PARTIAL: candidate selection is surrogate-robust; this common pilot exact-validates nominal centers only.",
    "IDEA06_QUALITY_DIVERSITY": "Quality-diversity repertoire across growth × oxygen niches; oxygen is resolved by reaction ID/name (never a positional fallback).",
    "IDEA07_REGIME_GRAPH": "PROXY: surrogate response-signature graph; true version requires flux/dual/active-set fingerprints.",
    "IDEA08_HIERARCHICAL_SEARCH": "PROXY: dimension-generic coarse-to-fine hierarchy over all medium variables; true pathway/gene/environment hierarchy remains future work.",
    "IDEA09_META_TASK_POLICY": "PROXY: within-landscape synthetic preference/floor meta-task learner; not cross-product/species meta-learning.",
    "IDEA10_PREF_GENERATOR": "Preference-conditioned GMM generator trained on exact high-utility archive designs; generates genuinely new continuous media.",
    "NEW_LANDSCAPE_PRIOR_PARETO_CONTINUATION": "Gap-diverse LPPC synthesis: rotating Pareto targets + landscape-prior GP correction + corrected-HV/uncertainty + design diversity + exact FBA.",
}


def _remaining(ctx: LiveContext, method: str) -> int:
    return max(0, ctx.budget - len(ctx.history(method)))


def _q(ctx: LiveContext, method: str) -> int:
    return min(ctx.batch_size, _remaining(ctx, method))


def _hist_arrays(ctx: LiveContext, method: str):
    h = ctx.history(method)
    if len(h) == 0:
        return np.empty((0, len(ctx.features))), np.empty((0, getattr(ctx, "objective_dim", 3))), np.empty(0, bool)
    X = h[[f"x01__{f}" for f in ctx.features]].to_numpy(float)
    Y = h[ctx.objective_norm_columns].to_numpy(float)
    F = h["exact_feasible"].astype(bool).to_numpy()
    return X, Y, F


def _print_progress(ctx: LiveContext, method: str, t0: float | None = None, extra: str = ""):
    h = ctx.history(method)
    elapsed = f" | batch {duration(time.time()-t0)}" if t0 is not None else ""
    print(
        f"[{method}] {len(h):2d}/{ctx.budget} | exact ΔHV={ctx.current_delta_hv(method):.9f}{extra}{elapsed}",
        flush=True,
    )


def run_base_sobol_passive(ctx: LiveContext):
    method = "BASE_SOBOL_PASSIVE"
    while _remaining(ctx, method) > 0:
        t0 = time.time(); q = _q(ctx, method)
        bno = len(ctx.history(method)) // ctx.batch_size + 1
        pool = ctx.make_pool(method, bno)
        ids = np.arange(min(q, len(pool.X)))
        ctx.evaluate_batch(method, pool.X[ids], ["SOBOL_PASSIVE"] * len(ids))
        _print_progress(ctx, method, t0)



def _cold_blind_pool(ctx: LiveContext, method: str, batch_no: int, rows: int, seed_offset: int = 0) -> np.ndarray:
    """Fresh Sobol candidates without consulting the frozen 50k landscape.

    This deliberately avoids the global surrogate, archive Pareto front,
    archive feasibility labels, and archive-distance filter.  It only removes
    points already proposed by this method and exact duplicates within the new
    pool.  The same batch seed as BASE_SOBOL_PASSIVE is used by default so the
    first cold-start initialization calls are a matched passive design.
    """
    n = int(rows)
    seed = ctx.seed + 1009 * int(batch_no) + int(seed_offset)
    X = sobol_points(max(n * 2, n + 64), len(ctx.features), seed)
    X = ctx._filter_new(X, method, exclude_frozen_archive=False)
    if len(X) < n:
        X2 = sobol_points(n * 2, len(ctx.features), seed + 17)
        X = np.vstack([X, ctx._filter_new(X2, method, exclude_frozen_archive=False)])
        # second pass removes any cross-block duplicates without archive use
        X = ctx._filter_new(X, method, exclude_frozen_archive=False)
    return X[:n]


def run_cold_start_bo(ctx: LiveContext):
    """Cold-start multi-objective BO with no frozen-landscape knowledge.

    Candidate selection sees only variable bounds plus this method's own exact
    observations.  The first ``cold_start_initial_calls`` exact evaluations are
    Sobol initialization and count against the same logical budget.  After
    initialization, a GP/qLogEHVI model is refit from the method's own exact
    history.  Infeasible observations are encoded as the normalized reference
    value (0,0), while the acquisition Pareto set contains feasible observations
    only.  The 50k archive remains available to LiveContext solely for *scoring*
    final ΔHV after an exact call; it is never consulted to choose a candidate.
    """
    method = "BASE_COLD_START_BO"
    bcfg = ctx.cfg["bo"]
    initial_calls = int(ctx.cfg["study"].get("cold_start_initial_calls", 16))
    min_feasible = int(ctx.cfg["study"].get("cold_start_min_feasible_for_bo", 8))
    pool_rows = int(ctx.cfg["study"].get("cold_start_pool_rows", ctx.cfg["study"].get("gp_pool_rows", 8192)))

    while _remaining(ctx, method) > 0:
        t0 = time.time()
        hist = ctx.history(method)
        hist_n = len(hist)
        q = _q(ctx, method)
        bno = hist_n // ctx.batch_size + 1

        # Phase 1: exact Sobol initialization.  These calls are part of the
        # 128-call budget, not free warm-up observations.
        if hist_n < initial_calls:
            q0 = min(q, initial_calls - hist_n)
            X = _cold_blind_pool(ctx, method, bno, max(pool_rows, q0))[:q0]
            ctx.evaluate_batch(
                method,
                X,
                ["COLD_INIT_SOBOL"] * len(X),
                {
                    "selection_uses_frozen_landscape": np.zeros(len(X), dtype=bool),
                    "cold_start_phase": ["initialization"] * len(X),
                    "cold_start_train_calls": np.full(len(X), hist_n, dtype=int),
                },
                exclude_frozen_archive=False,
            )
            _print_progress(ctx, method, t0, extra=f" | cold init={len(ctx.history(method))}/{initial_calls}")
            continue

        # Phase 2: BO trained only on this method's own live exact history.
        hist = ctx.history(method)
        xcols = [f"x01__{f}" for f in ctx.features]
        Xtr = hist[xcols].to_numpy(float)
        Ytr = hist[ctx.objective_norm_columns].to_numpy(float)
        Ftr = hist["exact_feasible"].astype(bool).to_numpy()
        good = Ftr & np.isfinite(Ytr).all(axis=1)

        # If an unusually difficult model has too few feasible initial points,
        # continue blind Sobol exploration until the GP has enough feasible data.
        if int(good.sum()) < min_feasible:
            X = _cold_blind_pool(ctx, method, bno, max(pool_rows, q))[:q]
            ctx.evaluate_batch(
                method,
                X,
                ["COLD_FEASIBILITY_SOBOL"] * len(X),
                {
                    "selection_uses_frozen_landscape": np.zeros(len(X), dtype=bool),
                    "cold_start_phase": ["feasibility_extension"] * len(X),
                    "cold_start_train_calls": np.full(len(X), hist_n, dtype=int),
                    "cold_start_feasible_train": np.full(len(X), int(good.sum()), dtype=int),
                },
                exclude_frozen_archive=False,
            )
            _print_progress(ctx, method, t0, extra=f" | cold feasible train={int(good.sum())}/{min_feasible}")
            continue

        Yfit = Ytr.copy()
        bad = (~Ftr) | (~np.isfinite(Yfit).all(axis=1))
        Yfit[bad] = 0.0
        own_front = pareto_points(Ytr[good])
        fit = fit_step4c_gp(Xtr, Yfit, own_front, float(bcfg["fixed_noise"]))
        choices = _cold_blind_pool(ctx, method, bno, pool_rows)
        if len(choices) < q:
            raise RuntimeError(f"{method}: only {len(choices)} blind candidates available for q={q}")
        cand, acq = discrete_qlogehvi_select(
            fit,
            choices,
            q=q,
            seed=ctx.seed + 30000 + hist_n,
            mc_samples=int(bcfg["mc_samples"]),
            max_batch_size=int(bcfg["discrete_max_batch_size"]),
        )
        acq = np.asarray(acq, float).reshape(-1)
        if len(acq) == 1 and len(cand) > 1:
            acq = np.repeat(acq, len(cand))
        ctx.evaluate_batch(
            method,
            cand,
            ["COLD_qLogEHVI"] * len(cand),
            {
                "selection_uses_frozen_landscape": np.zeros(len(cand), dtype=bool),
                "cold_start_phase": ["adaptive_bo"] * len(cand),
                "cold_start_train_calls": np.full(len(cand), len(Xtr), dtype=int),
                "cold_start_feasible_train": np.full(len(cand), int(good.sum()), dtype=int),
                "cold_start_acq_value": acq[: len(cand)],
            },
            exclude_frozen_archive=False,
        )
        _print_progress(ctx, method, t0, extra=f" | cold GP train={len(Xtr)} feasible={int(good.sum())}")



def _run_gsmobo_qparego(ctx: LiveContext, method: str, *, use_50k_archive: bool):
    """Run one of the two matched-budget gsMOBO/qParEGO ablations.

    Both versions receive exactly the same live-call budget, the same first 16
    blind Sobol exact evaluations, the same qParEGO scalarization/acquisition
    settings, and the same exact GEM/FBA bridge. The only intended difference is
    whether the objective GPs are warm-started with representative prepared-prior points
    from the reusable 50k landscape.

    NO50K
      * candidate selection sees only bounds + this method's live exact history;
      * the frozen archive is used only after exact evaluation for common ΔHV.

    50K
      * calls 1..initial_calls are kept identical to NO50K;
      * adaptive GPs are trained on the established representative exact subset
        of the 50k archive plus this method's own live exact observations;
      * the subset rule is the same Step4C-style rule used elsewhere: all
        recoverable exact Pareto rows first, then global Sobol-nearest coverage;
      * no global surrogate prediction is used as a pseudo-label.

    This cleanly tests the value of prior 50k archive knowledge while holding the
    qParEGO optimizer and live exact-call budget fixed.
    """
    scfg = ctx.cfg["study"]
    gcfg = ctx.cfg.get("gsmobo", {})
    initial_calls = int(gcfg.get("initial_calls", scfg.get("cold_start_initial_calls", 16)))
    min_feasible = int(gcfg.get("min_feasible_for_bo", 2))
    mc_samples = int(gcfg.get("mc_samples", 256))
    num_restarts = int(gcfg.get("num_restarts", 5))
    raw_samples = int(gcfg.get("raw_samples", 512))
    maxiter = int(gcfg.get("maxiter", 200))
    fallback_rows = int(gcfg.get("fallback_pool_rows", scfg.get("cold_start_pool_rows", 8192)))
    archive_rows = min(int(gcfg.get("archive_rows", scfg.get("gp_subset_rows", 1024))), len(ctx.baseline))
    tag = "50K" if use_50k_archive else "NO50K"

    while _remaining(ctx, method) > 0:
        t0 = time.time()
        hist = ctx.history(method)
        hist_n = len(hist)
        q = _q(ctx, method)
        bno = hist_n // ctx.batch_size + 1

        # Strictly matched initialization. Because _cold_blind_pool seeds do not
        # depend on the method name, both variants see identical initial points.
        if hist_n < initial_calls:
            q0 = min(q, initial_calls - hist_n)
            X = _cold_blind_pool(ctx, method, bno, max(fallback_rows, q0))[:q0]
            ctx.evaluate_batch(
                method,
                X,
                [f"GSMOBO_{tag}_INIT_SOBOL"] * len(X),
                {
                    "selection_uses_frozen_landscape": np.zeros(len(X), dtype=bool),
                    "gsmobo_uses_50k_archive": np.full(len(X), bool(use_50k_archive), dtype=bool),
                    "gsmobo_phase": ["initialization"] * len(X),
                    "gsmobo_train_calls": np.full(len(X), hist_n, dtype=int),
                    "gsmobo_archive_train_rows": np.zeros(len(X), dtype=int),
                    "gsmobo_weight_product": np.full(len(X), np.nan),
                    "gsmobo_weight_product": np.full(len(X), np.nan),
                    "gsmobo_acq_value": np.full(len(X), np.nan),
                },
                # Keep initialization and candidate novelty behavior identical.
                # The 50k variant receives archive knowledge only after the matched
                # initialization; raw-row exclusion is not an extra advantage.
                exclude_frozen_archive=False,
            )
            _print_progress(
                ctx, method, t0,
                extra=f" | gsMOBO {tag} init={len(ctx.history(method))}/{initial_calls}",
            )
            continue

        hist = ctx.history(method)
        xcols = [f"x01__{f}" for f in ctx.features]
        Xtr = hist[xcols].to_numpy(float)
        Ytr = hist[ctx.objective_norm_columns].to_numpy(float)
        Ftr = hist["exact_feasible"].astype(bool).to_numpy()
        good = Ftr & np.isfinite(Ytr).all(axis=1)

        if int(good.sum()) < min_feasible:
            X = _cold_blind_pool(ctx, method, bno, max(fallback_rows, q))[:q]
            ctx.evaluate_batch(
                method,
                X,
                [f"GSMOBO_{tag}_FEASIBILITY_SOBOL"] * len(X),
                {
                    "selection_uses_frozen_landscape": np.zeros(len(X), dtype=bool),
                    "gsmobo_uses_50k_archive": np.full(len(X), bool(use_50k_archive), dtype=bool),
                    "gsmobo_phase": ["feasibility_extension"] * len(X),
                    "gsmobo_train_calls": np.full(len(X), hist_n, dtype=int),
                    "gsmobo_archive_train_rows": np.zeros(len(X), dtype=int),
                    "gsmobo_feasible_train": np.full(len(X), int(good.sum()), dtype=int),
                    "gsmobo_weight_product": np.full(len(X), np.nan),
                    "gsmobo_weight_product": np.full(len(X), np.nan),
                    "gsmobo_acq_value": np.full(len(X), np.nan),
                },
                exclude_frozen_archive=False,
            )
            _print_progress(ctx, method, t0, extra=f" | gsMOBO {tag} feasible train={int(good.sum())}/{min_feasible}")
            continue

        # Published gsMOBO convention: unsuccessful exact evaluations contribute
        # zero objective values. The ablation difference is ONLY the training data:
        # live-only vs representative prepared 50k-prior rows + live exact data.
        Ylive = Ytr.copy()
        bad = (~Ftr) | (~np.isfinite(Ylive).all(axis=1))
        Ylive[bad] = 0.0
        archive_train_rows = 0
        if use_50k_archive:
            sub = np.asarray(ctx.subset_local[:archive_rows], int)
            Xa = ctx.X0[sub]
            Ya = ctx.Y0[sub].copy()
            Fa = ctx.F0[sub]
            bada = (~Fa) | (~np.isfinite(Ya).all(axis=1))
            Ya[bada] = 0.0
            Xfit = np.vstack([Xa, Xtr])
            Yfit = np.vstack([Ya, Ylive])
            # Defensive de-duplication: if a blind live Sobol call happens to
            # coincide with an archive row, keep the later live observation.
            last = {}
            for ii, xx in enumerate(Xfit):
                last[tuple(np.round(xx, 12).tolist())] = ii
            keep_idx = np.asarray(sorted(last.values()), int)
            Xfit = Xfit[keep_idx]
            Yfit = Yfit[keep_idx]
            archive_train_rows = max(0, len(Xfit) - len(Xtr))
        else:
            Xfit = Xtr
            Yfit = Ylive
        model = fit_gsmobo_gp(Xfit, Yfit)

        cand, acq, weights = continuous_qparego_select(
            model,
            Xfit,
            Yfit,
            q=q,
            # SAME qParEGO randomization in both variants: same scalarization
            # weights and acquisition initialization at each matched live call.
            seed=ctx.seed + 41000 + hist_n,
            mc_samples=mc_samples,
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            maxiter=maxiter,
        )
        cand = np.clip(np.asarray(cand, float), 0.0, 1.0)
        acq = np.asarray(acq, float).reshape(-1)
        weights = np.asarray(weights, float).reshape(-1, ctx.objective_dim)

        keep = []
        hist_tree = cKDTree(Xtr) if len(Xtr) else None
        for i, x in enumerate(cand):
            if hist_tree is not None and float(hist_tree.query(x, k=1)[0]) <= ctx.min_dwithin:
                continue
            if keep and float(cKDTree(cand[np.asarray(keep, int)]).query(x, k=1)[0]) <= ctx.min_dwithin:
                continue
            keep.append(i)
        keep = np.asarray(keep, int)
        cand2 = cand[keep] if len(keep) else np.empty((0, cand.shape[1]), float)
        acq2 = acq[keep] if len(keep) else np.empty(0, float)
        w2 = weights[keep] if len(keep) else np.empty((0, ctx.objective_dim), float)
        modes = [f"GSMOBO_{tag}_qParEGO_qLogEI"] * len(cand2)

        if len(cand2) < q:
            need = q - len(cand2)
            blind = _cold_blind_pool(ctx, method, bno, max(fallback_rows, need), seed_offset=7000 + hist_n)
            if len(cand2):
                tree = cKDTree(cand2)
                blind = blind[tree.query(blind, k=1)[0] > ctx.min_dwithin]
            blind = blind[:need]
            if len(blind):
                cand2 = np.vstack([cand2, blind]) if len(cand2) else blind
                acq2 = np.concatenate([acq2, np.full(len(blind), np.nan)])
                w2 = np.vstack([w2, np.full((len(blind), ctx.objective_dim), np.nan)]) if len(w2) else np.full((len(blind), ctx.objective_dim), np.nan)
                modes.extend([f"GSMOBO_{tag}_NUMERICAL_FALLBACK_SOBOL"] * len(blind))

        if len(cand2) == 0:
            raise RuntimeError(f"{method}: qParEGO produced no genuinely new candidate")

        nsel = min(q, len(cand2))
        ctx.evaluate_batch(
            method,
            cand2[:q],
            modes[:q],
            {
                "selection_uses_frozen_landscape": np.full(nsel, bool(use_50k_archive), dtype=bool),
                "gsmobo_uses_50k_archive": np.full(nsel, bool(use_50k_archive), dtype=bool),
                "gsmobo_phase": ["adaptive_qparego"] * nsel,
                "gsmobo_train_calls": np.full(nsel, len(Xtr), dtype=int),
                "gsmobo_archive_train_rows": np.full(nsel, archive_train_rows, dtype=int),
                "gsmobo_feasible_train": np.full(nsel, int(good.sum()), dtype=int),
                "gsmobo_weight_product": w2[:q, 0],
                "gsmobo_weight_carbon_efficiency": w2[:q, 1],
                "gsmobo_weight_oxygen_efficiency": w2[:q, 2],
                "gsmobo_acq_value": acq2[:q],
            },
            exclude_frozen_archive=False,
        )
        _print_progress(
            ctx, method, t0,
            extra=f" | gsMOBO {tag} GP train={len(Xfit)} (archive={archive_train_rows}, live={len(Xtr)}) feasible_live={int(good.sum())}",
        )


def run_gsmobo_qparego_no50k(ctx: LiveContext):
    return _run_gsmobo_qparego(ctx, "BASE_GSMOBO_QPAREGO_NO50K", use_50k_archive=False)


def run_gsmobo_qparego_50k(ctx: LiveContext):
    return _run_gsmobo_qparego(ctx, "BASE_GSMOBO_QPAREGO_50K", use_50k_archive=True)




def _nsga2_fast_nondominated_sort(Y: np.ndarray) -> list[list[int]]:
    """Canonical fast nondominated sorting for *maximization* objectives.

    Population sizes in the matched benchmark are deliberately small (16), so
    the transparent O(n^2) implementation is preferable to adding another
    dependency. NaN/inf rows should be removed by the caller.
    """
    Y = np.asarray(Y, float)
    n = len(Y)
    dominates = [[] for _ in range(n)]
    dominated_count = np.zeros(n, dtype=int)
    fronts: list[list[int]] = [[]]
    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            p_dom_q = bool(np.all(Y[p] >= Y[q]) and np.any(Y[p] > Y[q]))
            q_dom_p = bool(np.all(Y[q] >= Y[p]) and np.any(Y[q] > Y[p]))
            if p_dom_q:
                dominates[p].append(q)
            elif q_dom_p:
                dominated_count[p] += 1
        if dominated_count[p] == 0:
            fronts[0].append(p)
    i = 0
    while i < len(fronts) and fronts[i]:
        nxt = []
        for p in fronts[i]:
            for q in dominates[p]:
                dominated_count[q] -= 1
                if dominated_count[q] == 0:
                    nxt.append(q)
        if nxt:
            fronts.append(nxt)
        i += 1
    return fronts


def _nsga2_crowding_distance(Y: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Canonical NSGA-II crowding distance for maximization/minimization alike."""
    ids = np.asarray(ids, int)
    if len(ids) == 0:
        return np.empty(0, float)
    if len(ids) <= 2:
        return np.full(len(ids), np.inf, float)
    vals = np.asarray(Y, float)[ids]
    d = np.zeros(len(ids), float)
    for j in range(vals.shape[1]):
        order = np.argsort(vals[:, j], kind="mergesort")
        d[order[0]] = np.inf
        d[order[-1]] = np.inf
        lo = float(vals[order[0], j]); hi = float(vals[order[-1], j])
        span = hi - lo
        if span <= 1e-15:
            continue
        for k in range(1, len(order) - 1):
            if np.isfinite(d[order[k]]):
                d[order[k]] += (vals[order[k + 1], j] - vals[order[k - 1], j]) / span
    return d


def _nsga2_environmental_select(
    X: np.ndarray, Y: np.ndarray, F: np.ndarray, n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Feasibility-first NSGA-II survival selection.

    Feasible, finite exact observations are ranked by Pareto front and crowding
    distance. Infeasible observations are used only when there are too few
    feasible individuals to fill the population. This keeps feasibility
    handling explicit while preserving canonical NSGA-II mechanics on the
    actual objectives.
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float); F = np.asarray(F, bool)
    n = min(int(n), len(X))
    feasible = F & np.isfinite(Y).all(axis=1)
    good_ids = np.where(feasible)[0]
    selected: list[int] = []
    if len(good_ids):
        fronts = _nsga2_fast_nondominated_sort(Y[good_ids])
        for front_local in fronts:
            front = good_ids[np.asarray(front_local, int)]
            room = n - len(selected)
            if room <= 0:
                break
            if len(front) <= room:
                selected.extend(front.tolist())
            else:
                crowd = _nsga2_crowding_distance(Y, front)
                # stable deterministic tie handling: crowding descending, index ascending
                order = np.lexsort((front, -crowd))
                selected.extend(front[order[:room]].tolist())
                break
    if len(selected) < n:
        bad_ids = np.where(~feasible)[0]
        # If objective values exist despite an infeasible status, prefer the less
        # poor rows; otherwise preserve deterministic source order.
        score = np.full(len(bad_ids), -np.inf, float)
        if len(bad_ids):
            yy = Y[bad_ids]
            finite = np.isfinite(yy).all(axis=1)
            score[finite] = yy[finite].sum(axis=1)
            order = np.lexsort((bad_ids, -score))
            selected.extend(bad_ids[order[: n - len(selected)]].tolist())
    ids = np.asarray(selected[:n], int)
    return X[ids], Y[ids], F[ids]


def _nsga2_rank_and_crowding(Y: np.ndarray, F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return feasibility-aware rank (lower better) and crowding (higher better)."""
    Y = np.asarray(Y, float); F = np.asarray(F, bool)
    rank = np.full(len(Y), 10**6, int)
    crowd = np.zeros(len(Y), float)
    feasible = F & np.isfinite(Y).all(axis=1)
    ids = np.where(feasible)[0]
    if len(ids):
        fronts = _nsga2_fast_nondominated_sort(Y[ids])
        for r, front_local in enumerate(fronts):
            front = ids[np.asarray(front_local, int)]
            rank[front] = r
            crowd[front] = _nsga2_crowding_distance(Y, front)
    # All infeasible rows are deliberately worse than every feasible front.
    bad = np.where(~feasible)[0]
    if len(bad):
        rank[bad] = (int(rank[ids].max()) + 1) if len(ids) else 0
    return rank, crowd


def _nsga2_sbx_pair(p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator, eta_c: float, prob: float):
    """Deb-style simulated binary crossover (SBX) on normalized [0,1] variables."""
    p1 = np.asarray(p1, float); p2 = np.asarray(p2, float)
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > float(prob):
        return c1, c2
    for j in range(len(p1)):
        if rng.random() > 0.5 or abs(p1[j] - p2[j]) <= 1e-14:
            continue
        x1, x2 = sorted((float(p1[j]), float(p2[j])))
        rand = rng.random()
        beta = 1.0 + (2.0 * (x1 - 0.0) / max(x2 - x1, 1e-15))
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / max(2.0 - rand * alpha, 1e-15)) ** (1.0 / (eta_c + 1.0))
        ch1 = 0.5 * ((x1 + x2) - betaq * (x2 - x1))

        beta = 1.0 + (2.0 * (1.0 - x2) / max(x2 - x1, 1e-15))
        alpha = 2.0 - beta ** (-(eta_c + 1.0))
        if rand <= 1.0 / alpha:
            betaq = (rand * alpha) ** (1.0 / (eta_c + 1.0))
        else:
            betaq = (1.0 / max(2.0 - rand * alpha, 1e-15)) ** (1.0 / (eta_c + 1.0))
        ch2 = 0.5 * ((x1 + x2) + betaq * (x2 - x1))
        ch1 = float(np.clip(ch1, 0.0, 1.0)); ch2 = float(np.clip(ch2, 0.0, 1.0))
        if rng.random() <= 0.5:
            c1[j], c2[j] = ch2, ch1
        else:
            c1[j], c2[j] = ch1, ch2
    return c1, c2


def _nsga2_polynomial_mutation(x: np.ndarray, rng: np.random.Generator, eta_m: float, prob: float):
    """Deb-style polynomial mutation on normalized [0,1] variables."""
    x = np.asarray(x, float).copy()
    for j in range(len(x)):
        if rng.random() > float(prob):
            continue
        y = float(np.clip(x[j], 0.0, 1.0))
        delta1 = y
        delta2 = 1.0 - y
        rnd = rng.random()
        mut_pow = 1.0 / (eta_m + 1.0)
        if rnd <= 0.5:
            xy = 1.0 - delta1
            val = 2.0 * rnd + (1.0 - 2.0 * rnd) * (xy ** (eta_m + 1.0))
            deltaq = val ** mut_pow - 1.0
        else:
            xy = 1.0 - delta2
            val = 2.0 * (1.0 - rnd) + 2.0 * (rnd - 0.5) * (xy ** (eta_m + 1.0))
            deltaq = 1.0 - val ** mut_pow
        x[j] = np.clip(y + deltaq, 0.0, 1.0)
    return x


def _nsga2_generate_offspring(
    Xp: np.ndarray, Yp: np.ndarray, Fp: np.ndarray, n: int, seed: int,
    crossover_probability: float, eta_c: float, mutation_probability: float, eta_m: float,
) -> np.ndarray:
    """Generate a deterministic NSGA-II offspring population from exact parents."""
    Xp = np.asarray(Xp, float); Yp = np.asarray(Yp, float); Fp = np.asarray(Fp, bool)
    if len(Xp) < 2:
        raise RuntimeError("NSGA-II requires at least two parents")
    rank, crowd = _nsga2_rank_and_crowding(Yp, Fp)
    rng = np.random.default_rng(int(seed))

    def tournament() -> int:
        a, b = rng.integers(0, len(Xp), size=2)
        a = int(a); b = int(b)
        if rank[a] < rank[b]: return a
        if rank[b] < rank[a]: return b
        if crowd[a] > crowd[b]: return a
        if crowd[b] > crowd[a]: return b
        return a if rng.random() < 0.5 else b

    out = []
    while len(out) < int(n):
        p1 = Xp[tournament()]; p2 = Xp[tournament()]
        c1, c2 = _nsga2_sbx_pair(p1, p2, rng, eta_c, crossover_probability)
        c1 = _nsga2_polynomial_mutation(c1, rng, eta_m, mutation_probability)
        c2 = _nsga2_polynomial_mutation(c2, rng, eta_m, mutation_probability)
        out.append(c1)
        if len(out) < int(n): out.append(c2)
    return np.asarray(out[: int(n)], float)


def _nsga2_archive_seed(ctx: LiveContext, live_X: np.ndarray, live_Y: np.ndarray, live_F: np.ndarray, pop_size: int):
    """Build the 50k-informed parent population after the matched blind initialization."""
    pareto_ids = baseline_pareto_row_indices(ctx).tolist()
    # Add representative exact coverage behind the Pareto rows so sparse fronts
    # (e.g. CHO) still have enough diverse candidate parents.
    for i in np.asarray(ctx.subset_local, int).tolist():
        if i not in pareto_ids:
            pareto_ids.append(i)
    aid = np.asarray(pareto_ids, int)
    Xa, Ya, Fa = ctx.X0[aid], ctx.Y0[aid], ctx.F0[aid]
    X = np.vstack([Xa, live_X])
    Y = np.vstack([Ya, live_Y])
    F = np.concatenate([Fa, live_F])
    return _nsga2_environmental_select(X, Y, F, pop_size)


def _nsga2_replay_parent_population(ctx: LiveContext, method: str, use_50k: bool, pop_size: int):
    """Reconstruct the exact parent population from calls.csv for resumability."""
    h = ctx.history(method)
    xcols = [f"x01__{f}" for f in ctx.features]
    if len(h) < pop_size:
        raise RuntimeError(f"{method}: need {pop_size} initialization calls before evolutionary replay")
    init = h.iloc[:pop_size]
    X = init[xcols].to_numpy(float)
    Y = init[ctx.objective_norm_columns].to_numpy(float)
    F = init["exact_feasible"].astype(bool).to_numpy()
    if use_50k:
        Xp, Yp, Fp = _nsga2_archive_seed(ctx, X, Y, F, pop_size)
    else:
        Xp, Yp, Fp = _nsga2_environmental_select(X, Y, F, pop_size)

    completed_generations = max(0, (len(h) - pop_size) // pop_size)
    for g in range(1, completed_generations + 1):
        a = pop_size + (g - 1) * pop_size
        b = a + pop_size
        off = h.iloc[a:b]
        Xo = off[xcols].to_numpy(float)
        Yo = off[ctx.objective_norm_columns].to_numpy(float)
        Fo = off["exact_feasible"].astype(bool).to_numpy()
        Xp, Yp, Fp = _nsga2_environmental_select(
            np.vstack([Xp, Xo]), np.vstack([Yp, Yo]), np.concatenate([Fp, Fo]), pop_size
        )
    return Xp, Yp, Fp


def _nsga2_static_new_filter(ctx: LiveContext, X: np.ndarray, history_before_generation: pd.DataFrame, use_50k: bool):
    """Filter a deterministically regenerated generation without seeing partial-generation calls."""
    X = np.atleast_2d(np.asarray(X, float))
    keep = np.ones(len(X), bool)
    if use_50k:
        d50 = ctx.baseline_tree.query(X, k=1)[0]
        keep &= d50 > ctx.min_d50k
    if len(history_before_generation):
        xcols = [f"x01__{f}" for f in ctx.features]
        Xh = history_before_generation[xcols].to_numpy(float)
        dh = cKDTree(Xh).query(X, k=1)[0]
        keep &= dh > ctx.min_dwithin
    seen = set()
    for i, x in enumerate(X):
        k = tuple(np.round(x, 12).tolist())
        if k in seen:
            keep[i] = False
        seen.add(k)
    return X[keep]


def _run_nsga2(ctx: LiveContext, method: str, use_50k_archive: bool):
    """Matched-budget NSGA-II baseline/warm-start pair.

    Both variants receive exactly the same blind live calls 1..population_size.
    After that, NO50K evolves only from its own exact history. 50K replaces the
    post-initialization parent population with a feasibility-aware, crowding-
    diverse population selected from prepared 50k-prior Pareto/representative
    designs plus those same live initialization calls. The random evolutionary
    operator stream is matched between the two variants.
    """
    ncfg = ctx.cfg.get("nsga2", {})
    pop_size = int(ncfg.get("population_size", 16))
    initial_calls = int(ncfg.get("initial_calls", pop_size))
    if initial_calls != pop_size:
        raise RuntimeError("NSGA-II matched implementation requires initial_calls == population_size")
    crossover_probability = float(ncfg.get("crossover_probability", 0.90))
    eta_c = float(ncfg.get("eta_c", 20.0))
    eta_m = float(ncfg.get("eta_m", 20.0))
    mutation_probability = float(ncfg.get("mutation_probability", 1.0 / max(1, len(ctx.features))))
    fallback_rows = int(ncfg.get("fallback_pool_rows", 4096))
    tag = "50K" if use_50k_archive else "NO50K"

    while _remaining(ctx, method) > 0:
        t0 = time.time()
        h = ctx.history(method)
        hist_n = len(h)
        q = _q(ctx, method)
        bno = hist_n // ctx.batch_size + 1

        if hist_n < initial_calls:
            q0 = min(q, initial_calls - hist_n)
            X = _cold_blind_pool(ctx, method, bno, max(fallback_rows, q0))[:q0]
            ctx.evaluate_batch(
                method, X, [f"NSGA2_{tag}_INIT_SOBOL"] * len(X),
                {
                    "selection_uses_frozen_landscape": np.zeros(len(X), dtype=bool),
                    "nsga2_uses_50k_archive": np.full(len(X), bool(use_50k_archive), dtype=bool),
                    "nsga2_phase": ["initialization"] * len(X),
                    "nsga2_generation": np.zeros(len(X), dtype=int),
                    "nsga2_population_size": np.full(len(X), pop_size, dtype=int),
                    "nsga2_crossover_probability": np.full(len(X), crossover_probability),
                    "nsga2_eta_c": np.full(len(X), eta_c),
                    "nsga2_eta_m": np.full(len(X), eta_m),
                    "nsga2_mutation_probability": np.full(len(X), mutation_probability),
                },
                exclude_frozen_archive=False,
            )
            _print_progress(ctx, method, t0, extra=f" | NSGA-II {tag} init={len(ctx.history(method))}/{initial_calls}")
            continue

        # Determine the current generation and its offset. A full generation is
        # pop_size exact offspring, evaluated in ordinary benchmark batches of 4.
        adaptive_done = hist_n - initial_calls
        generation = adaptive_done // pop_size + 1
        offset = adaptive_done % pop_size
        generation_start = initial_calls + (generation - 1) * pop_size
        h_before = h.iloc[:generation_start].copy()

        Xp, Yp, Fp = _nsga2_replay_parent_population(ctx, method, use_50k_archive, pop_size)

        # Generate enough deterministic offspring to survive novelty filtering.
        accepted = np.empty((0, len(ctx.features)), float)
        for attempt in range(8):
            raw = _nsga2_generate_offspring(
                Xp, Yp, Fp, max(pop_size * 2, pop_size + 8),
                seed=ctx.seed + 73000 + 1009 * generation + 37 * attempt,
                crossover_probability=crossover_probability,
                eta_c=eta_c,
                mutation_probability=mutation_probability,
                eta_m=eta_m,
            )
            raw = _nsga2_static_new_filter(ctx, raw, h_before, use_50k_archive)
            if len(accepted):
                raw = _nsga2_static_new_filter(ctx, raw, pd.DataFrame(), False)
                # remove exact/near duplicates against already accepted candidates
                if len(raw):
                    d = cKDTree(accepted).query(raw, k=1)[0]
                    raw = raw[d > ctx.min_dwithin]
            if len(raw):
                accepted = np.vstack([accepted, raw]) if len(accepted) else raw
            # stable first-occurrence de-duplication across attempts
            if len(accepted):
                seen = set(); ids = []
                for i, x in enumerate(accepted):
                    k = tuple(np.round(x, 12).tolist())
                    if k not in seen:
                        ids.append(i); seen.add(k)
                accepted = accepted[np.asarray(ids, int)]
            if len(accepted) >= pop_size:
                break
        if len(accepted) < pop_size:
            # Numerical/degeneracy fallback only. 50K may use archive knowledge
            # to reject already-known exact designs; NO50K remains archive-blind.
            blind = _cold_blind_pool(ctx, method, bno, max(fallback_rows, pop_size), seed_offset=19000 + generation)
            blind = _nsga2_static_new_filter(ctx, blind, h_before, use_50k_archive)
            if len(accepted) and len(blind):
                d = cKDTree(accepted).query(blind, k=1)[0]
                blind = blind[d > ctx.min_dwithin]
            if len(blind):
                accepted = np.vstack([accepted, blind]) if len(accepted) else blind
        if len(accepted) < pop_size:
            raise RuntimeError(f"{method}: could not generate a full genuinely new offspring generation")

        full_generation = accepted[:pop_size]
        take = min(q, pop_size - offset)
        Xbatch = full_generation[offset:offset + take]
        if len(Xbatch) == 0:
            raise RuntimeError(f"{method}: empty generation slice at generation={generation}, offset={offset}")
        ctx.evaluate_batch(
            method, Xbatch, [f"NSGA2_{tag}_GEN{generation}"] * len(Xbatch),
            {
                "selection_uses_frozen_landscape": np.full(len(Xbatch), bool(use_50k_archive), dtype=bool),
                "nsga2_uses_50k_archive": np.full(len(Xbatch), bool(use_50k_archive), dtype=bool),
                "nsga2_phase": ["evolution"] * len(Xbatch),
                "nsga2_generation": np.full(len(Xbatch), generation, dtype=int),
                "nsga2_generation_offset": np.arange(offset, offset + len(Xbatch), dtype=int),
                "nsga2_population_size": np.full(len(Xbatch), pop_size, dtype=int),
                "nsga2_parent_feasible": np.full(len(Xbatch), int(Fp.sum()), dtype=int),
                "nsga2_crossover_probability": np.full(len(Xbatch), crossover_probability),
                "nsga2_eta_c": np.full(len(Xbatch), eta_c),
                "nsga2_eta_m": np.full(len(Xbatch), eta_m),
                "nsga2_mutation_probability": np.full(len(Xbatch), mutation_probability),
            },
            # 50K is allowed to avoid exact archive duplicates; NO50K must remain
            # blind to the frozen archive during candidate selection.
            exclude_frozen_archive=bool(use_50k_archive),
        )
        _print_progress(ctx, method, t0, extra=f" | NSGA-II {tag} gen={generation} offset={offset}->{offset+len(Xbatch)}")


def run_nsga2_no50k(ctx: LiveContext):
    return _run_nsga2(ctx, "BASE_NSGA2_NO50K", use_50k_archive=False)


def run_nsga2_50k(ctx: LiveContext):
    return _run_nsga2(ctx, "BASE_NSGA2_50K", use_50k_archive=True)


def run_global_static(ctx: LiveContext):
    method = "BASE_GLOBAL_STATIC"
    # Static means the frozen 50k surrogate/front are never updated from live exact feedback.
    # v0.2.2 fixes v0.2.1's all-zero-HV tie degeneracy: zero predicted gain no
    # longer means "take the first Sobol candidate".  Instead, remaining slots
    # use an explicit Pareto-near + design-novelty fallback.
    tol = float(ctx.cfg["study"].get("predicted_hv_positive_tolerance", 1e-12))
    div_radius = float(ctx.cfg["study"].get("static_pareto_near_diversity_radius", 0.04))
    while _remaining(ctx, method) > 0:
        t0 = time.time(); q = _q(ctx, method); bno = len(ctx.history(method)) // ctx.batch_size + 1
        pool = ctx.make_pool(method, bno, rows=int(ctx.cfg["study"].get("static_pool_rows", 32768)))

        positive = greedy_predicted_hv(
            pool.X, pool.pred, pool.safe, ctx.baseline_front, q, min_gain=tol
        )
        chosen = list(map(int, positive.tolist()))
        modes = ["STATIC_POSITIVE_PRED_HV"] * len(chosen)

        # Explicit fallback only for unfilled slots.  It remains purely static:
        # rank safe candidates by closeness to the frozen predicted Pareto polyline,
        # with a small design-novelty/diversity preference.
        if len(chosen) < q:
            available = pool.safe.copy()
            if chosen:
                available[np.asarray(chosen, int)] = False
            fd = phenotype_distance_to_front(pool.pred, ctx.baseline_front)
            near = 1.0 - robust_scale(fd)
            novelty = robust_scale(pool.novelty)
            score = 0.85 * near + 0.15 * novelty
            if chosen:
                dsel = cKDTree(pool.X[np.asarray(chosen, int)]).query(pool.X, k=1)[0]
                design_div = 1.0 - np.exp(-0.5 * (dsel / max(div_radius, 1e-6)) ** 2)
                score *= 0.35 + 0.65 * design_div
            fill = choose_diverse_by_score(pool.X, score, available, q - len(chosen), div_radius)
            chosen.extend(map(int, fill.tolist()))
            modes.extend(["STATIC_PARETO_NEAR_FALLBACK"] * len(fill))

        chosen = np.asarray(chosen, int)
        if len(chosen) == 0:
            raise RuntimeError(f"{method}: no safe candidates available")

        # Audit metadata: predicted gain is evaluated sequentially using the frozen
        # static planning front, so zero-gain fallback behavior is visible in calls.csv.
        planning = ctx.baseline_front.copy()
        phv = []
        for idx in chosen:
            g = candidate_hv_gain(planning, pool.pred[idx:idx+1])
            gg = float(g[0]) if len(g) and np.isfinite(g[0]) else np.nan
            phv.append(gg)
            planning = pareto_points(np.vstack([planning, pool.pred[idx]]))
        fdsel = phenotype_distance_to_front(pool.pred[chosen], ctx.baseline_front)
        ctx.evaluate_batch(
            method, pool.X[chosen], modes,
            {
                "selection_predicted_hv_gain": np.asarray(phv, float),
                "selection_frozen_front_distance": fdsel,
                "selection_used_fallback": np.asarray([m.endswith("FALLBACK") for m in modes], bool),
            },
        )
        _print_progress(ctx, method, t0, extra=f" | fallback={sum(m.endswith('FALLBACK') for m in modes)}/{len(modes)}")


MULTIMODE=("PARETO","PRODUCT","BALANCED","CONSTRAINED_PRODUCT","ROBUST_VALUE","NOVELTY")


def _multimode_scores(ctx: LiveContext, pool, front, seed: int):
    g,p=pool.pred[:,0],pool.pred[:,1]
    gs,ps=robust_scale(g),robust_scale(p)
    pareto=candidate_hv_gain(front,pool.pred)
    balanced=np.sqrt(np.maximum(gs,0)*np.maximum(ps,0))
    constrained=np.where(g>=ctx.growth_floor,p,-np.inf)
    # Compute fragility only for this pool; still surrogate-derived.
    from core import local_fragility_q95
    frag=local_fragility_q95(pool.X,ctx.global_model,float(ctx.cfg["ideas"].get("fragility_radius",0.05)),seed,batch_centers=2048)
    fs=robust_scale(frag); ns=robust_scale(pool.novelty)
    robust=(0.45*gs+0.55*ps)*(1.0-0.75*fs)
    novelty=ns*(0.20+0.80*balanced)
    return {"PARETO":pareto,"PRODUCT":p.copy(),"BALANCED":balanced,"CONSTRAINED_PRODUCT":constrained,"ROBUST_VALUE":robust,"NOVELTY":novelty}


def _ucb_mode(history: pd.DataFrame, c: float=0.45, reward_scale: float=0.001):
    counts={m:0 for m in MULTIMODE}; rewards={m:0.0 for m in MULTIMODE}
    if len(history):
        for _,r in history.iterrows():
            m=str(r.get("mode",""))
            if m in counts:
                counts[m]+=1; rewards[m]+=float(r.get("exact_hv_gain_this_call",0.0))/reward_scale
    for m in MULTIMODE:
        if counts[m]==0:return m
    total=max(2,sum(counts.values()))
    return max(MULTIMODE,key=lambda m: rewards[m]/counts[m]+c*math.sqrt(math.log(total)/counts[m]))


def run_global_multimode(ctx: LiveContext):
    method="BASE_GLOBAL_MULTIMODE"
    while _remaining(ctx,method)>0:
        t0=time.time(); q=_q(ctx,method); bno=len(ctx.history(method))//ctx.batch_size+1
        front=ctx.current_front(method); pool=ctx.make_pool(method,bno)
        scores=_multimode_scores(ctx,pool,front,ctx.seed+5000+bno)
        available=pool.safe.copy(); chosen=[]; modes=[]; planning=front.copy()
        hist=ctx.history(method)
        for _ in range(q):
            mode=_ucb_mode(pd.concat([hist,pd.DataFrame({"mode":modes,"exact_hv_gain_this_call":[0]*len(modes)})],ignore_index=True) if modes else hist)
            s=np.asarray(scores[mode],float).copy(); s[~available]=-np.inf
            if chosen:
                d=cKDTree(pool.X[np.asarray(chosen,int)]).query(pool.X,k=1)[0]
                div=1-np.exp(-0.5*(d/0.04)**2); good=np.isfinite(s); s[good]*=0.3+0.7*div[good]
            if not np.any(np.isfinite(s)):break
            idx=int(np.nanargmax(s)); chosen.append(idx); modes.append(mode); available[idx]=False
            planning=pareto_points(np.vstack([planning,pool.pred[idx]]))
            scores["PARETO"]=candidate_hv_gain(planning,pool.pred)
        ctx.evaluate_batch(method,pool.X[np.asarray(chosen,int)],modes)
        _print_progress(ctx,method,t0)


def _step4c_training(ctx: LiveContext, method: str):
    sub=ctx.subset_local
    X=ctx.X0[sub]; Y=ctx.Y0[sub].copy(); F=ctx.F0[sub].copy()
    bad=(~F)|(~np.isfinite(Y).all(axis=1)); Y[bad]=0.0
    Xh,Yh,Fh=_hist_arrays(ctx,method)
    if len(Xh):
        Yn=Yh.copy(); badn=(~Fh)|(~np.isfinite(Yn).all(axis=1)); Yn[badn]=0.0
        X=np.vstack([X,Xh]); Y=np.vstack([Y,Yn])
    return X,Y


def run_step4c_bo(ctx: LiveContext):
    method="BASE_STEP4C_BO"; bcfg=ctx.cfg["bo"]
    while _remaining(ctx,method)>0:
        t0=time.time(); q=_q(ctx,method); bno=len(ctx.history(method))//ctx.batch_size+1
        Xtr,Ytr=_step4c_training(ctx,method); front=ctx.current_front(method)
        fit=fit_step4c_gp(Xtr,Ytr,front,float(bcfg["fixed_noise"]))
        pool=ctx.make_pool(method,bno,rows=int(ctx.cfg["study"].get("gp_pool_rows",8192)))
        ids=np.where(pool.safe)[0]
        if len(ids)<q:ids=np.arange(len(pool.X))
        cand,_=discrete_qlogehvi_select(fit,pool.X[ids],q=q,seed=ctx.seed+10000+len(ctx.history(method)),mc_samples=int(bcfg["mc_samples"]),max_batch_size=int(bcfg["discrete_max_batch_size"]))
        ctx.evaluate_batch(method,cand,["qLogEHVI"]*len(cand))
        _print_progress(ctx,method,t0,extra=f" | GP train={len(Xtr)}")


def _prior_training(ctx: LiveContext, method: str):
    sub=ctx.subset_local
    X=ctx.X0[sub]; Y=ctx.Y0[sub].copy(); F=ctx.F0[sub].copy()
    Xh,Yh,Fh=_hist_arrays(ctx,method)
    if len(Xh):
        X=np.vstack([X,Xh]);Y=np.vstack([Y,Yh]);F=np.concatenate([F,Fh])
    if ctx.subset_oof_prior is None:
        raise RuntimeError("Prior-GP requested but cross-fitted subset prior predictions were not prepared")
    override={tuple(np.round(ctx.X0[local],12).tolist()):tuple(ctx.subset_oof_prior[j].tolist()) for j,local in enumerate(sub)}
    return X,Y,F,override


def run_prior_gp_pareto(ctx: LiveContext):
    method="BASE_PRIOR_GP_PARETO";bcfg=ctx.cfg["bo"]
    while _remaining(ctx,method)>0:
        t0=time.time();q=_q(ctx,method);bno=len(ctx.history(method))//ctx.batch_size+1
        Xtr,Ytr,Ftr,override=_prior_training(ctx,method);front=ctx.current_front(method)
        fit=fit_prior_gp(ctx.global_model,Xtr,Ytr,Ftr,front,float(bcfg["fixed_noise"]),override)
        pool=ctx.make_pool(method,bno,rows=int(ctx.cfg["study"].get("gp_pool_rows",8192)))
        ids=np.where(pool.safe)[0]; ids=ids if len(ids)>=q else np.arange(len(pool.X))
        cand,_=discrete_qlogehvi_select(fit,pool.X[ids],q=q,seed=ctx.seed+20000+len(ctx.history(method)),mc_samples=int(bcfg["mc_samples"]),max_batch_size=int(bcfg["discrete_max_batch_size"]))
        ctx.evaluate_batch(method,cand,["PRIOR_qLogEHVI"]*len(cand))
        _print_progress(ctx,method,t0,extra=f" | feasible GP train={int(Ftr.sum())}")


def _gradient_fd(model,x:np.ndarray,eps:float=0.02):
    x=np.asarray(x,float);d=len(x);pts=[];den=[]
    for j in range(d):
        xp=x.copy();xm=x.copy();xp[j]=min(1,xp[j]+eps);xm[j]=max(0,xm[j]-eps);pts += [xp,xm];den.append(max(xp[j]-xm[j],1e-12))
    pred,_=model.predict(np.asarray(pts));J=np.zeros((2,d))
    for j in range(d):J[:,j]=(pred[2*j]-pred[2*j+1])/den[j]
    return J


def run_idea01(ctx: LiveContext):
    method="IDEA01_SENSITIVITY_NAV"; statep=ctx.method_dir(method)/"state.json"
    state=json.loads(statep.read_text()) if statep.exists() else {"trust":0.12};trust=float(state.get("trust",0.12));weights=np.linspace(.1,.9,9);pids=baseline_pareto_row_indices(ctx)
    while _remaining(ctx,method)>0:
        t0=time.time();q=_q(ctx,method);bno=len(ctx.history(method))//ctx.batch_size+1;front=ctx.current_front(method);pool=ctx.make_pool(method,bno)
        anchorX=ctx.X0[pids];anchorY=ctx.Y0[pids]
        Xh,Yh,Fh=_hist_arrays(ctx,method);good=Fh&np.isfinite(Yh).all(axis=1)
        if np.any(good):anchorX=np.vstack([anchorX,Xh[good]]);anchorY=np.vstack([anchorY,Yh[good]])
        chosen=[];modes=[];avail=pool.safe.copy()
        for slot in range(q):
            w=float(weights[(len(ctx.history(method))+slot)%len(weights)]);util=(1-w)*anchorY[:,0]+w*anchorY[:,1];a=int(np.argmax(util));x0=anchorX[a];u0=float(util[a]);J=_gradient_fd(ctx.global_model,x0);grad=(1-w)*J[0]+w*J[1];ng=np.linalg.norm(grad)
            if ng<1e-10:score=candidate_hv_gain(front,pool.pred)
            else:
                target=np.clip(x0+trust*grad/ng,0,1);d=np.linalg.norm(pool.X-target[None,:],axis=1);pu=(1-w)*pool.pred[:,0]+w*pool.pred[:,1];score=(pu-u0)+0.40*robust_scale(candidate_hv_gain(front,pool.pred))-0.80*d
            score[~avail]=-np.inf
            if not np.any(np.isfinite(score)):break
            idx=int(np.nanargmax(score));chosen.append(idx);modes.append(f"wP={w:.2f}|r={trust:.3f}");avail[idx]=False
        before=ctx.current_delta_hv(method);ctx.evaluate_batch(method,pool.X[np.asarray(chosen,int)],modes);after=ctx.current_delta_hv(method)
        trust=min(.30,trust*1.20) if after>before+1e-14 else max(.025,trust*.78);save_json({"trust":trust},statep);_print_progress(ctx,method,t0,extra=f" | trust={trust:.3f}")


def run_idea02(ctx: LiveContext):
    method = "IDEA02_PARETO_GAP_TRACE"
    while _remaining(ctx, method) > 0:
        t0 = time.time(); q = _q(ctx, method); hist_n = len(ctx.history(method)); bno = hist_n // ctx.batch_size + 1
        front = ctx.current_front(method); targets = candidate_targets_from_front(front, .045, 16)
        pool = ctx.make_pool(method, bno); avail = pool.safe.copy(); chosen = []; modes = []
        target_idx = []; target_gap = []; target_dist = []; planning = front.copy()
        # Rotate the target schedule across calls/batches.  This fixes v0.2.1's
        # largest-gap monopolization while preserving the same target geometry.
        start_t = hist_n % max(1, len(targets))
        for slot in range(q):
            ids = np.where(avail)[0]
            if not len(ids):
                break
            ti = (start_t + slot) % len(targets)
            target, gap = targets[ti]
            d = np.linalg.norm(pool.pred[ids] - target[None, :], axis=1)
            hv = candidate_hv_gain(planning, pool.pred[ids])
            score = (gap + 1e-6) / (d + .02) + 1.5 * robust_scale(hv)
            j = int(np.nanargmax(score)); idx = int(ids[j])
            chosen.append(idx); modes.append(f"gap{ti}|target_d={d[j]:.3f}")
            target_idx.append(ti); target_gap.append(float(gap)); target_dist.append(float(d[j]))
            avail[idx] = False
            planning = pareto_points(np.vstack([planning, pool.pred[idx]]))
        if not chosen:
            raise RuntimeError(f"{method}: no safe candidates available")
        ctx.evaluate_batch(
            method, pool.X[np.asarray(chosen, int)], modes,
            {"target_index": np.asarray(target_idx, int), "target_gap": np.asarray(target_gap, float),
             "target_distance": np.asarray(target_dist, float)},
        )
        _print_progress(ctx, method, t0, extra=f" | gaps={','.join(map(str,target_idx))}")


def _fit_weighted_surrogate(ctx: LiveContext):
    ids=np.where(ctx.good0)[0];Y=ctx.Y0[ids];X=ctx.X0[ids];tree=cKDTree(ctx.baseline_front);d=tree.query(Y,k=1)[0];scale=max(float(np.quantile(d,.25)),1e-5);pr=pd.Series(Y[:,1]).rank(pct=True).to_numpy();gr=pd.Series(Y[:,0]).rank(pct=True).to_numpy();w=1+12*np.exp(-(d/scale)**2)+6*pr**6+2*gr**6;scfg=ctx.cfg["surrogate"]
    rg=ExtraTreesRegressor(n_estimators=int(scfg["regressor_trees"]),min_samples_leaf=max(1,int(scfg["min_samples_leaf"])),max_features=1.0,n_jobs=-1,random_state=ctx.seed+3101);rp=ExtraTreesRegressor(n_estimators=int(scfg["regressor_trees"]),min_samples_leaf=max(1,int(scfg["min_samples_leaf"])),max_features=1.0,n_jobs=-1,random_state=ctx.seed+3102);print("[IDEA03_OPT_WEIGHTED_SURROGATE] fitting weighted global regressors...",flush=True);rg.fit(X,Y[:,0],sample_weight=w);rp.fit(X,Y[:,1],sample_weight=w);return rg,rp


def run_idea03(ctx: LiveContext):
    method="IDEA03_OPT_WEIGHTED_SURROGATE";rg,rp=_fit_weighted_surrogate(ctx)
    while _remaining(ctx,method)>0:
        t0=time.time();q=_q(ctx,method);bno=len(ctx.history(method))//ctx.batch_size+1;pool=ctx.make_pool(method,bno);pred=np.column_stack([rg.predict(pool.X),rp.predict(pool.X)]);ids=greedy_predicted_hv(pool.X,pred,pool.safe,ctx.current_front(method),q);ctx.evaluate_batch(method,pool.X[ids],["weighted_surrogate_HV"]*len(ids),{"weighted_pred_growth":pred[ids,0],"weighted_pred_product":pred[ids,1]});_print_progress(ctx,method,t0)


def run_idea04(ctx: LiveContext):
    """Gap-diverse inverse phenotype-to-design search.

    v0.2.3 keeps the successful v0.2.1/v0.2.2 inverse mechanism but makes the
    target schedule budget-scalable: target assignments rotate across *all*
    accumulated logical calls rather than restarting at target 0 in every
    four-call batch.  This prevents a 128-call run from repeatedly spending
    most of its budget on only the first few Pareto gaps.
    """
    method = "IDEA04_INVERSE_DESIGN"
    Xatlas = ctx.X0[ctx.good0]
    Yatlas = ctx.Y0[ctx.good0]
    ytree = cKDTree(Yatlas)
    while _remaining(ctx, method) > 0:
        t0 = time.time(); q = _q(ctx, method); hist_n = len(ctx.history(method))
        bno = hist_n // ctx.batch_size + 1
        front = ctx.current_front(method)
        targets = candidate_targets_from_front(front, .05, 12)
        pool = ctx.make_pool(method, bno)
        avail = pool.safe.copy(); chosen = []; modes = []
        target_idx = []; target_gap = []; target_dist = []
        start_t = hist_n % max(1, len(targets))
        for slot in range(q):
            ids = np.where(avail)[0]
            if not len(ids):
                break
            ti = (start_t + slot) % len(targets)
            target, gap = targets[ti]
            _, loc = ytree.query(target, k=min(64, len(Yatlas)))
            proto = Xatlas[np.atleast_1d(loc)]
            px = cKDTree(proto)
            dx = px.query(pool.X[ids], k=1)[0]
            dy = np.linalg.norm(pool.pred[ids] - target[None, :], axis=1)
            hv = candidate_hv_gain(front, pool.pred[ids])
            score = -(.60 * dx + .75 * dy) + .75 * robust_scale(hv)
            j = int(np.nanargmax(score)); idx = int(ids[j])
            chosen.append(idx)
            modes.append(f"inverse_gap{ti}|gap={gap:.3f}|target_d={dy[j]:.3f}")
            target_idx.append(ti); target_gap.append(float(gap)); target_dist.append(float(dy[j]))
            avail[idx] = False
        if not chosen:
            raise RuntimeError(f"{method}: no safe candidates available")
        ctx.evaluate_batch(
            method, pool.X[np.asarray(chosen, int)], modes,
            {
                "target_index": np.asarray(target_idx, int),
                "target_gap": np.asarray(target_gap, float),
                "target_distance": np.asarray(target_dist, float),
            },
        )
        _print_progress(ctx, method, t0, extra=f" | gaps={','.join(map(str,target_idx))}")


def _robust_profile(model,X,radius,seed,batch=1024):
    X=np.asarray(X,float);n,d=X.shape;rng=np.random.default_rng(seed);dirs=[]
    for j in range(d):
        e=np.zeros(d);e[j]=1;dirs += [e.copy(),-e.copy()]
    rd=rng.normal(size=(8,d));rd/=np.linalg.norm(rd,axis=1,keepdims=True);dirs=np.vstack([np.asarray(dirs),rd,-rd]);q10=np.full((n,2),np.nan);worst=np.full(n,np.nan)
    for s in range(0,n,batch):
        e=min(n,s+batch);neigh=np.clip(X[s:e,None,:]+radius*dirs[None,:,:],0,1);pred,pinf=model.predict(neigh.reshape(-1,d));pred=pred.reshape(e-s,len(dirs),2);pinf=pinf.reshape(e-s,len(dirs));q10[s:e]=np.quantile(pred,.10,axis=1);worst[s:e]=np.max(pinf,axis=1)
    return q10,worst


def run_idea05(ctx: LiveContext):
    method="IDEA05_ROBUST_ADVERSARIAL";radius=float(ctx.cfg["ideas"].get("robust_radius",.035));pids=baseline_pareto_row_indices(ctx);rv,_=_robust_profile(ctx.global_model,ctx.X0[pids],radius,ctx.seed+5101);robust_front=pareto_points(rv)
    while _remaining(ctx,method)>0:
        t0=time.time();q=_q(ctx,method);bno=len(ctx.history(method))//ctx.batch_size+1;pool=ctx.make_pool(method,bno,rows=int(ctx.cfg["study"].get("robust_pool_rows",4096)));rh,rpinf=_robust_profile(ctx.global_model,pool.X,radius,ctx.seed+5100+bno);mask=pool.safe&np.isfinite(rh).all(axis=1)&(rpinf<=.10);mask=mask if mask.sum()>=q else pool.safe&np.isfinite(rh).all(axis=1);ids=greedy_predicted_hv(pool.X,rh,mask,robust_front,q);ctx.evaluate_batch(method,pool.X[ids],[f"robust_q10_r={radius:.3f}"]*len(ids),{"robust_q10_growth":rh[ids,0],"robust_q10_product":rh[ids,1],"robust_worst_pinf":rpinf[ids]});_print_progress(ctx,method,t0)


def run_idea06(ctx: LiveContext):
    method = "IDEA06_QUALITY_DIVERSITY"
    explicit = ctx.parent_cfg.get("problem", {}).get("qd_feature")
    o2_idx = ctx.feature_index(explicit=explicit, keywords=("oxygen", "o2"), required=True)
    g_edges = np.quantile(ctx.Y0[ctx.good0, 0], [.2, .4, .6, .8])
    o_edges = np.quantile(ctx.X0[:, o2_idx], [.2, .4, .6, .8])
    while _remaining(ctx, method) > 0:
        t0 = time.time(); q = _q(ctx, method); bno = len(ctx.history(method)) // ctx.batch_size + 1
        pool = ctx.make_pool(method, bno); cells = {}
        for idx in np.where(pool.safe)[0]:
            key = (
                int(np.digitize(pool.pred[idx, 0], g_edges)),
                int(np.digitize(pool.X[idx, o2_idx], o_edges)),
            )
            cells.setdefault(key, []).append(idx)
        for k in cells:
            cells[k].sort(key=lambda i: pool.pred[i, 1], reverse=True)
        order = sorted(cells, key=lambda k: pool.pred[cells[k][0], 1], reverse=True)
        chosen, modes, depth = [], [], 0
        while len(chosen) < q:
            added = False
            for key in order:
                if depth < len(cells[key]):
                    idx = cells[key][depth]
                    chosen.append(idx)
                    modes.append(f"QD_g{key[0]}_o{key[1]}|axis={ctx.features[o2_idx]}")
                    added = True
                    if len(chosen) >= q:
                        break
            if not added:
                break
            depth += 1
        ctx.evaluate_batch(method, pool.X[np.asarray(chosen, int)], modes)
        _print_progress(ctx, method, t0)


def _gradient_signatures(model,X,eps=.02):
    X=np.asarray(X,float);n,d=X.shape;sig=np.zeros((n,2*d))
    for j in range(d):
        xp=X.copy();xm=X.copy();xp[:,j]=np.minimum(1,xp[:,j]+eps);xm[:,j]=np.maximum(0,xm[:,j]-eps);pp,_=model.predict(xp);pm,_=model.predict(xm);den=(xp[:,j]-xm[:,j])[:,None];grad=(pp-pm)/np.maximum(den,1e-12);sig[:,2*j]=grad[:,0];sig[:,2*j+1]=grad[:,1]
    norm=np.linalg.norm(sig,axis=1,keepdims=True);return sig/np.maximum(norm,1e-12)


def run_idea07(ctx: LiveContext):
    method="IDEA07_REGIME_GRAPH"
    while _remaining(ctx,method)>0:
        t0=time.time();q=_q(ctx,method);bno=len(ctx.history(method))//ctx.batch_size+1;front=ctx.current_front(method);pool=ctx.make_pool(method,bno,rows=int(ctx.cfg["study"].get("graph_pool_rows",8192)));safe_ids=np.where(pool.safe)[0];hv=candidate_hv_gain(front,pool.pred[safe_ids]);balanced=np.sqrt(robust_scale(pool.pred[safe_ids,0])*robust_scale(pool.pred[safe_ids,1]));nov=robust_scale(pool.novelty[safe_ids]);vals=[hv,pool.pred[safe_ids,1],balanced,nov];keep=set();per=int(ctx.cfg["ideas"].get("graph_pool_per_score",500))
        for v in vals:
            take=np.argsort(v)[-min(per,len(v)):];keep.update(safe_ids[take].tolist())
        nodes=np.asarray(sorted(keep),int);Xn=pool.X[nodes];Yn=pool.pred[nodes];sig=_gradient_signatures(ctx.global_model,Xn);nn=NearestNeighbors(n_neighbors=min(11,len(nodes)),n_jobs=-1).fit(Xn);dist,nbr=nn.kneighbors(Xn);rows=[];cols=[];data=[]
        for i in range(len(nodes)):
            for jj in range(1,nbr.shape[1]):
                j=int(nbr[i,jj]);w=float(dist[i,jj])*(1+.55*(1-float(np.dot(sig[i],sig[j]))));rows += [i,j];cols += [j,i];data += [w,w]
        G=coo_matrix((data,(rows,cols)),shape=(len(nodes),len(nodes))).tocsr();pids=baseline_pareto_row_indices(ctx);tree=cKDTree(Xn);starts=np.unique([int(tree.query(x,k=1)[1]) for x in ctx.X0[pids]])[:24];D=dijkstra(G,indices=starts,directed=False);geod=np.min(D,axis=0) if D.ndim==2 else D;val=.60*robust_scale(candidate_hv_gain(front,Yn))+.25*robust_scale(Yn[:,1])+.15*robust_scale(pool.novelty[nodes]);regnov=1-np.max(sig@sig[starts].T,axis=1) if len(starts) else np.zeros(len(nodes));score=val*(1-.45*robust_scale(geod))+.15*robust_scale(regnov);local=choose_diverse_by_score(Xn,score,np.ones(len(nodes),bool),q,.05);ids=nodes[local];ctx.evaluate_batch(method,pool.X[ids],["response_graph"]*len(ids));_print_progress(ctx,method,t0)


def run_idea08(ctx: LiveContext):
    method = "IDEA08_HIERARCHICAL_SEARCH"
    imp = .5 * (ctx.global_model.growth.feature_importances_ + ctx.global_model.product.feature_importances_)
    order = np.argsort(-imp)
    # Freeze a generic hierarchy over *all* dimensions in pairs.  This removes
    # the E. coli-only [:6] truncation that would silently ignore 17/23 CHO
    # variables.  It is still a proxy hierarchy, not a pathway-level hierarchy.
    levels = [order[i:i+2] for i in range(0, len(order), 2)]
    while _remaining(ctx, method) > 0:
        t0 = time.time(); q = _q(ctx, method); bno = len(ctx.history(method)) // ctx.batch_size + 1
        front = ctx.current_front(method); pool = ctx.make_pool(method, bno)
        safe = np.where(pool.safe)[0]
        if len(safe) == 0:
            safe = np.arange(len(pool.X))
        hv = candidate_hv_gain(front, pool.pred)
        bins = np.clip((pool.X * 4).astype(int), 0, 3)
        beam = [()]
        for li, vars_ in enumerate(levels):
            cell = []
            for base in beam:
                mask = np.ones(len(safe), bool)
                for k in range(0, len(base), 2):
                    mask &= bins[safe, base[k]] == base[k+1]
                ids = safe[mask]
                comb = {}
                for idx in ids:
                    key = list(base)
                    for v in vars_:
                        key += [int(v), int(bins[idx, v])]
                    key = tuple(key)
                    score = float(hv[idx] + .15 * pool.pred[idx, 1])
                    if key not in comb or score > comb[key][0]:
                        comb[key] = (score, idx)
                cell += [(v[0], k, v[1]) for k, v in comb.items()]
            cell.sort(reverse=True, key=lambda z: z[0])
            width = min(32, 6 + 4 * li)
            beam = [z[1] for z in cell[:min(width, len(cell))]]
            if not beam:
                break
        cand = []
        for cell in beam:
            mask = np.ones(len(safe), bool)
            for k in range(0, len(cell), 2):
                mask &= bins[safe, cell[k]] == cell[k+1]
            ids = safe[mask]
            ids = ids[np.argsort(hv[ids] + .15 * pool.pred[ids, 1])[::-1]]
            cand.extend(ids[:4].tolist())
        seen = []
        for i in cand:
            if int(i) not in seen:
                seen.append(int(i))
            if len(seen) >= q:
                break
        if len(seen) < q:
            ids = safe[np.argsort(hv[safe] + .15 * pool.pred[safe, 1])[::-1]]
            for i in ids:
                if int(i) not in seen:
                    seen.append(int(i))
                if len(seen) >= q:
                    break
        mode = "hierarchy_all_dims_" + "-".join(map(str, order.tolist()))
        ctx.evaluate_batch(method, pool.X[np.asarray(seen, int)], [mode] * len(seen))
        _print_progress(ctx, method, t0, extra=f" | levels={len(levels)}")


def _fit_meta(ctx: LiveContext):
    rng=np.random.default_rng(ctx.seed+9100);good=np.where(ctx.good0)[0];train_w=np.array([.1,.3,.5,.7,.9]);q50=float(np.quantile(ctx.Y0[good,0],.50));q75=float(np.quantile(ctx.Y0[good,0],.75));train_f=np.array([0,.85*q50,.90*q75]);n=int(ctx.cfg["ideas"].get("meta_pairs",120000));rid=rng.choice(good,size=n,replace=True);w=train_w[rng.integers(0,len(train_w),size=n)];floor=train_f[rng.integers(0,len(train_f),size=n)];Xp=np.column_stack([ctx.X0[rid],w,floor]);Yp=(1-w)*ctx.Y0[rid,0]+w*ctx.Y0[rid,1]-4*np.maximum(floor-ctx.Y0[rid,0],0);m=ExtraTreesRegressor(n_estimators=260,min_samples_leaf=2,max_features=1.0,n_jobs=-1,random_state=ctx.seed+9101);print(f"[IDEA09_META_TASK_POLICY] fitting task-conditioned utility model on {n:,} pairs...",flush=True);m.fit(Xp,Yp);tasks=[(float(w),float(f)) for f in [0,q50,q75] for w in [.2,.4,.6,.8]];return m,tasks


def run_idea09(ctx: LiveContext):
    method="IDEA09_META_TASK_POLICY";model,tasks=_fit_meta(ctx)
    while _remaining(ctx,method)>0:
        t0=time.time();q=_q(ctx,method);bno=len(ctx.history(method))//ctx.batch_size+1;pool=ctx.make_pool(method,bno);rankings=[]
        for w,f in tasks:
            Q=np.column_stack([pool.X,np.full(len(pool.X),w),np.full(len(pool.X),f)]);s=model.predict(Q);s[~pool.safe]=-np.inf;rankings.append(np.argsort(s)[::-1])
        chosen=[];seen=set();depth=0
        while len(chosen)<q:
            added=False
            for rank in rankings:
                pick=None
                for j in rank[:min(len(rank),depth+25)]:
                    if int(j) not in seen:pick=int(j);break
                if pick is not None:chosen.append(pick);seen.add(pick);added=True
                if len(chosen)>=q:break
            if not added:break
            depth+=1
        ctx.evaluate_batch(method,pool.X[np.asarray(chosen,int)],["meta_task"]*len(chosen));_print_progress(ctx,method,t0)


def _task_utility(Y,w,floor):
    Y=np.asarray(Y,float);return (1-w)*Y[:,0]+w*Y[:,1]-4*np.maximum(floor-Y[:,0],0)


def run_idea10(ctx: LiveContext):
    method="IDEA10_PREF_GENERATOR";good=np.where(ctx.good0)[0];weights=np.linspace(.1,.9,9);floors=[0.0,ctx.growth_floor]
    while _remaining(ctx,method)>0:
        t0=time.time();q=_q(ctx,method);bno=len(ctx.history(method))//ctx.batch_size+1;generated=[];tasklabels=[]
        for ti,(floor,w) in enumerate((f,w) for f in floors for w in weights):
            util=_task_utility(ctx.Y0[good],float(w),float(floor));elite_n=max(120,int(.02*len(good)));elite=good[np.argsort(util)[-elite_n:]];comps=min(4,max(1,len(elite)//80));gm=GaussianMixture(n_components=comps,covariance_type="full",reg_covar=1e-5,n_init=2,random_state=ctx.seed+10100+100*bno+ti);gm.fit(ctx.X0[elite]);gen,_=gm.sample(int(ctx.cfg["ideas"].get("generator_samples_per_task",384)));gen=np.clip(gen,0,1);gen=ctx._filter_new(gen,method)
            if len(gen):
                pred,pinf=ctx.global_model.predict(gen);safe=pinf<=ctx.safe_threshold;ids=np.where(safe)[0]
                if len(ids):
                    s=_task_utility(pred[ids],float(w),float(floor));take=ids[np.argsort(s)[::-1][:min(8,len(ids))]];generated.extend(gen[take].tolist());tasklabels.extend([f"GMM_wP={w:.2f}_floor={floor:.3f}"]*len(take))
        if not generated:
            pool=ctx.make_pool(method,bno);ids=greedy_predicted_hv(pool.X,pool.pred,pool.safe,ctx.current_front(method),q);X=pool.X[ids];modes=["GMM_fallback_HV"]*len(ids)
        else:
            X=np.asarray(generated,float);pred,pinf=ctx.global_model.predict(X);mask=pinf<=ctx.safe_threshold;score=candidate_hv_gain(ctx.current_front(method),pred);ids=choose_diverse_by_score(X,score,mask,q,.04);modes=[tasklabels[i] for i in ids];X=X[ids]
        ctx.evaluate_batch(method,X,modes);_print_progress(ctx,method,t0)


def run_new_lppc(ctx: LiveContext):
    """Landscape-Prior Pareto Continuation, gap-diverse v0.2.2 revision.

    The v0.2.1 score globally maximized over targets and could collapse onto the
    largest gap.  v0.2.2 explicitly rotates target assignments across calls,
    while retaining prior-GP correction, corrected HV, uncertainty and design
    diversity.
    """
    method = "NEW_LANDSCAPE_PRIOR_PARETO_CONTINUATION"; bcfg = ctx.cfg["bo"]
    while _remaining(ctx, method) > 0:
        t0 = time.time(); q = _q(ctx, method); hist_n = len(ctx.history(method)); bno = hist_n // ctx.batch_size + 1
        front = ctx.current_front(method); targets = candidate_targets_from_front(front, .045, 16)
        Xtr, Ytr, Ftr, override = _prior_training(ctx, method)
        fit = fit_prior_gp(ctx.global_model, Xtr, Ytr, Ftr, front, float(bcfg["fixed_noise"]), override)
        pool = ctx.make_pool(method, bno, rows=int(ctx.cfg["study"].get("gp_pool_rows", 8192)))
        corr_mean, corr_std = posterior_mean_std(fit.model, pool.X)
        avail = pool.safe & np.isfinite(corr_mean).all(axis=1) & np.isfinite(corr_std).all(axis=1)
        chosen = []; modes = []; target_idx = []; target_gap = []; target_dist = []; planning = front.copy()
        corr_unc = np.linalg.norm(corr_std, axis=1)
        start_t = hist_n % max(1, len(targets))
        for slot in range(q):
            ids = np.where(avail)[0]
            if not len(ids):
                break
            ti = (start_t + slot) % len(targets); target, gap = targets[ti]
            d = np.linalg.norm(corr_mean[ids] - target[None, :], axis=1)
            hv = candidate_hv_gain(planning, corr_mean[ids])
            score = (gap + 1e-6) / (d + .02) + 1.5 * robust_scale(hv) + .20 * robust_scale(corr_unc[ids])
            if chosen:
                ddesign = cKDTree(pool.X[np.asarray(chosen, int)]).query(pool.X[ids], k=1)[0]
                score += .25 * robust_scale(ddesign)
            j = int(np.nanargmax(score)); idx = int(ids[j])
            chosen.append(idx); modes.append(f"LPPC_gap{ti}|target_d={d[j]:.3f}")
            target_idx.append(ti); target_gap.append(float(gap)); target_dist.append(float(d[j]))
            avail[idx] = False
            planning = pareto_points(np.vstack([planning, corr_mean[idx]]))
        if not chosen:
            raise RuntimeError(f"{method}: no safe corrected candidates available")
        chosen = np.asarray(chosen, int)
        ctx.evaluate_batch(
            method, pool.X[chosen], modes,
            {
                "corrected_pred_growth": corr_mean[chosen, 0],
                "corrected_pred_product": corr_mean[chosen, 1],
                "corrected_uncertainty": corr_unc[chosen],
                "target_index": np.asarray(target_idx, int),
                "target_gap": np.asarray(target_gap, float),
                "target_distance": np.asarray(target_dist, float),
            },
        )
        _print_progress(ctx, method, t0, extra=f" | gaps={','.join(map(str,target_idx))} | feasible GP train={int(Ftr.sum())}")


RUNNERS={
    "BASE_SOBOL_PASSIVE":run_base_sobol_passive,
    "BASE_COLD_START_BO":run_cold_start_bo,
    "BASE_GSMOBO_QPAREGO_NO50K":run_gsmobo_qparego_no50k,
    "BASE_GSMOBO_QPAREGO_50K":run_gsmobo_qparego_50k,
    "BASE_NSGA2_NO50K":run_nsga2_no50k,
    "BASE_NSGA2_50K":run_nsga2_50k,
    "BASE_GLOBAL_STATIC":run_global_static,
    "BASE_GLOBAL_MULTIMODE":run_global_multimode,
    "BASE_STEP4C_BO":run_step4c_bo,
    "BASE_PRIOR_GP_PARETO":run_prior_gp_pareto,
    "IDEA01_SENSITIVITY_NAV":run_idea01,
    "IDEA02_PARETO_GAP_TRACE":run_idea02,
    "IDEA03_OPT_WEIGHTED_SURROGATE":run_idea03,
    "IDEA04_INVERSE_DESIGN":run_idea04,
    "IDEA05_ROBUST_ADVERSARIAL":run_idea05,
    "IDEA06_QUALITY_DIVERSITY":run_idea06,
    "IDEA07_REGIME_GRAPH":run_idea07,
    "IDEA08_HIERARCHICAL_SEARCH":run_idea08,
    "IDEA09_META_TASK_POLICY":run_idea09,
    "IDEA10_PREF_GENERATOR":run_idea10,
    "NEW_LANDSCAPE_PRIOR_PARETO_CONTINUATION":run_new_lppc,
}


# v0.2.9 experimental landscape-native methods are isolated in their own
# module so the canonical v0.2.7 comparator implementations above remain
# byte-for-byte reviewable.  novel_methods imports core/live_core only, so
# this late import does not create a methods.py circular dependency.
from novel_methods import NEW_METHOD_NOTES, NEW_RUNNERS
METHOD_NOTES.update(NEW_METHOD_NOTES)
RUNNERS.update(NEW_RUNNERS)

# v0.3.0 conservative landscape-augmented gsMOBO+50k methods.
from augmented_gsmobo_methods import AUGMENTED_METHOD_NOTES, AUGMENTED_RUNNERS
METHOD_NOTES.update(AUGMENTED_METHOD_NOTES)
RUNNERS.update(AUGMENTED_RUNNERS)
