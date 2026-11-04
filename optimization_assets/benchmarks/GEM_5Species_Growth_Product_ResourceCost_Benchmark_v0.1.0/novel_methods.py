from __future__ import annotations

"""Four experimental landscape-native optimizers for GEM Engine v0.2.9.

These methods are deliberately built around information that exists because GEM
Engine maintains a reusable exact 50k design landscape.  They are not intended
as generic wrappers around BO or NSGA-II.

Method IDs
----------
NEW_GHOST_OOF_BLINDSPOT_MINING
NEW_SPARC_PARETO_DISPLACEMENT_SUBSPACE
NEW_TRAIL_MONOTONE_ARCHIVE_EXTRAPOLATION
NEW_PACE_CALIBRATED_CELL_EXPANSION

All methods:
- propose genuinely new continuous designs;
- use the same LiveContext exact-FBA closure/accounting as prior benchmarks;
- resume safely from calls.csv;
- operate under the same 4 -> 16 -> 32 -> 64 -> 128 matched logical budget.
"""

import math
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from core import candidate_hv_gain, pareto_points, robust_scale, simplex_weight_bank, scalarized_improvement_proxy
from live_core import (
    LiveContext,
    baseline_pareto_row_indices,
    candidate_targets_from_front,
    choose_diverse_by_score,
    duration,
    sobol_points,
)


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _remaining(ctx: LiveContext, method: str) -> int:
    return max(0, int(ctx.budget) - len(ctx.history(method)))


def _q(ctx: LiveContext, method: str) -> int:
    return min(int(ctx.batch_size), _remaining(ctx, method))


def _progress(ctx: LiveContext, method: str, t0: float, extra: str = "") -> None:
    h = ctx.history(method)
    print(
        f"[{method}] {len(h):3d}/{ctx.budget} | exact ΔHV={ctx.current_delta_hv(method):.9f}"
        f"{extra} | batch {duration(time.time() - t0)}",
        flush=True,
    )


def _filter_with_metadata(ctx: LiveContext, method: str, X: np.ndarray, *meta_arrays):
    """Apply frozen-archive/history novelty while preserving aligned metadata."""
    X = np.atleast_2d(np.asarray(X, float))
    if len(X) == 0:
        return (X,) + tuple(np.asarray(m)[:0] for m in meta_arrays)

    keep = np.ones(len(X), bool)
    d50, _ = ctx.baseline_tree.query(X, k=1)
    keep &= d50 > ctx.min_d50k

    h = ctx.history(method)
    if len(h):
        cols = [f"x01__{f}" for f in ctx.features]
        Xh = h[cols].to_numpy(float)
        dh, _ = cKDTree(Xh).query(X, k=1)
        keep &= dh > ctx.min_dwithin

    seen = set()
    for i, x in enumerate(X):
        key = tuple(np.round(x, 12).tolist())
        if key in seen:
            keep[i] = False
        seen.add(key)

    ids = np.where(keep)[0]
    out = [X[ids]]
    for m in meta_arrays:
        a = np.asarray(m, dtype=object if isinstance(m, list) and m and isinstance(m[0], str) else None)
        out.append(a[ids])
    return tuple(out)


def _oof_atlas(ctx: LiveContext):
    """Return exact-feasible archive OOF residual atlas (exact - OOF prediction)."""
    path = ctx.outdir / "surrogate_qc" / "oof_prior_predictions.npz"
    if not path.exists():
        raise RuntimeError(
            f"This method requires the mandatory 5-fold OOF artifact: {path}. "
            "Run surrogate_validation.py before live optimization."
        )
    z = np.load(path)
    oof = np.asarray(z["oof"], float)
    if oof.shape != ctx.Y0.shape:
        raise RuntimeError(f"OOF shape {oof.shape} != prepared-prior shape {ctx.Y0.shape}")
    good = ctx.good0 & np.isfinite(oof).all(axis=1)
    ids = np.where(good)[0]
    if len(ids) < 100:
        raise RuntimeError(f"Too few feasible finite OOF rows: {len(ids)}")
    X = ctx.X0[ids]
    Y = ctx.Y0[ids]
    R = Y - oof[ids]
    return ids, X, Y, oof[ids], R, cKDTree(X)


def _pareto_row_normals(ctx: LiveContext):
    """Map frozen Pareto row id -> outward unit normal in M-objective space.

    The original 2-D tangent rotation is preserved exactly.  For 3-D, a local
    tangent plane is estimated by SVD of neighboring Pareto points and its
    least-variance right singular vector is used as the outward normal.
    """
    pids = baseline_pareto_row_indices(ctx)
    if len(pids) == 0:
        return {}, np.empty(0, int)
    Y = ctx.Y0[pids]
    normals = {}
    if Y.shape[1] == 2:
        order=np.argsort(Y[:,0]); pids=pids[order]; Y=Y[order]
        for i,pid in enumerate(pids):
            if len(pids)==1: t=np.array([1.0,-1.0])
            elif i==0: t=Y[1]-Y[0]
            elif i==len(pids)-1: t=Y[-1]-Y[-2]
            else: t=Y[i+1]-Y[i-1]
            if np.linalg.norm(t)<=1e-12: t=np.array([1.0,-1.0])
            n=np.array([-t[1],t[0]],float); n/=max(np.linalg.norm(n),1e-12)
            if np.sum(n)<0:n=-n
            normals[int(pid)]=n
        return normals,pids
    for i,pid in enumerate(pids):
        if len(pids) < 3:
            n=np.ones(Y.shape[1],float); n/=np.linalg.norm(n)
        else:
            dist=np.linalg.norm(Y-Y[i],axis=1); nn=np.argsort(dist)[1:min(len(Y),9)]
            D=Y[nn]-Y[i]
            try:
                _,_,vt=np.linalg.svd(D,full_matrices=True); n=vt[-1].astype(float)
            except np.linalg.LinAlgError:
                n=np.ones(Y.shape[1],float)
            n/=max(np.linalg.norm(n),1e-12)
            if np.sum(n)<0:n=-n
        normals[int(pid)]=n
    return normals,pids

def _candidate_source_pool(ctx: LiveContext, method: str, batch_no: int, rows: int, seed_offset: int = 0):
    pool = ctx.make_pool(method, batch_no, rows=rows, seed_offset=seed_offset)
    return pool.X, pool.pred, pool.pinf, pool.safe, pool.novelty


def _live_extension_centers(ctx: LiveContext, method: str):
    h = ctx.history(method)
    if len(h) == 0 or "exact_front_extension" not in h:
        return np.empty((0, len(ctx.features))), np.empty((0, ctx.objective_dim))
    mask = h["exact_front_extension"].astype(bool).to_numpy()
    if not np.any(mask):
        return np.empty((0, len(ctx.features))), np.empty((0, ctx.objective_dim))
    xcols = [f"x01__{f}" for f in ctx.features]
    X = h.loc[mask, xcols].to_numpy(float)
    Y = h.loc[mask, ctx.objective_norm_columns].to_numpy(float)
    good = np.isfinite(X).all(axis=1) & np.isfinite(Y).all(axis=1)
    return X[good], Y[good]


# ---------------------------------------------------------------------------
# 1) GHOST — OOF false-negative / blind-spot mining
# ---------------------------------------------------------------------------


def _ghost_blindspot_bank(ctx: LiveContext):
    ids, X, Y, OOF, R, tree = _oof_atlas(ctx)
    normals, pids = _pareto_row_normals(ctx)
    id_to_local = {int(pid): i for i, pid in enumerate(ids)}

    # Generic exact-vs-OOF surprise across several trade-off preferences.
    weights = (np.column_stack([1.0-np.linspace(0.0,1.0,11), np.linspace(0.0,1.0,11)])
               if Y.shape[1] == 2 else simplex_weight_bank(Y.shape[1], n=21, seed=ctx.seed+211))
    surprise = np.zeros(len(ids), float)
    exact_quality = np.zeros(len(ids), float)
    for w in weights:
        ue = Y @ np.asarray(w,float)
        up = OOF @ np.asarray(w,float)
        # robust utility scaling avoids one weight dominating solely by scale.
        lo, hi = np.quantile(ue, [0.05, 0.95])
        den = max(float(hi - lo), 1e-8)
        surprise = np.maximum(surprise, (ue - up) / den)
        exact_quality = np.maximum(exact_quality, (ue - lo) / den)

    # Pareto-specific outward underprediction is the most relevant blind spot.
    outward = np.zeros(len(ids), float)
    is_front = np.zeros(len(ids), bool)
    for pid in pids:
        li = id_to_local.get(int(pid))
        if li is None:
            continue
        is_front[li] = True
        outward[li] = max(0.0, float(np.dot(R[li], normals[int(pid)])))

    # Favor exact high-quality rows the OOF model systematically undercalled.
    blind = np.maximum(surprise, 0.0) * np.clip(exact_quality, 0.0, 2.5)
    blind += 2.0 * outward
    if R.shape[1] == 2:
        blind += 0.35 * np.maximum(R[:, 0], 0.0) + 0.65 * np.maximum(R[:, 1], 0.0)
    else:
        blind += np.mean(np.maximum(R, 0.0), axis=1)
    blind[~np.isfinite(blind)] = 0.0

    # Always retain Pareto blind spots, plus strongest non-front high-quality surprises.
    top_n = min(int(ctx.cfg.get("ghost", {}).get("blindspot_centers", 320)), len(ids))
    rank = np.argsort(blind)[::-1]
    chosen = []
    for j in np.where(is_front & (blind > 0))[0]:
        chosen.append(int(j))
    for j in rank:
        if int(j) not in chosen:
            chosen.append(int(j))
        if len(chosen) >= top_n:
            break
    chosen = np.asarray(chosen[:top_n], int)
    return X, Y, R, tree, X[chosen], blind[chosen]


def _ghost_candidates(ctx: LiveContext, method: str, batch_no: int, hist_n: int, bank):
    cfg = ctx.cfg.get("ghost", {})
    atlas_X, atlas_Y, atlas_R, atlas_tree, centers, center_score = bank
    rng = np.random.default_rng(ctx.seed + 210_001 + 1009 * batch_no + 13 * hist_n)
    d = len(ctx.features)

    jitter_rows = int(cfg.get("jitter_rows", 12_288))
    global_rows = int(cfg.get("global_rows", 4096))
    live_rows = int(cfg.get("live_rows", 2048))
    scales = np.asarray(cfg.get("jitter_scales", [0.008, 0.020, 0.050, 0.100]), float)

    blocks = []
    source = []
    prior = []

    if len(centers) and jitter_rows > 0:
        prob = np.maximum(center_score, 0) + 1e-6
        prob = prob / prob.sum()
        ci = rng.choice(len(centers), size=jitter_rows, replace=True, p=prob)
        s = rng.choice(scales, size=jitter_rows, replace=True)
        Xg = np.clip(centers[ci] + rng.normal(size=(jitter_rows, d)) * s[:, None], 0.0, 1.0)
        blocks.append(Xg)
        source.extend(["OOF_blindspot_jitter"] * len(Xg))
        prior.extend(center_score[ci].tolist())

    Xlive, _ = _live_extension_centers(ctx, method)
    if len(Xlive) and live_rows > 0:
        ci = rng.integers(0, len(Xlive), size=live_rows)
        s = rng.choice(np.asarray([0.008, 0.020, 0.050]), size=live_rows)
        Xl = np.clip(Xlive[ci] + rng.normal(size=(live_rows, d)) * s[:, None], 0.0, 1.0)
        blocks.append(Xl)
        source.extend(["live_success_jitter"] * len(Xl))
        prior.extend([float(np.max(center_score)) if len(center_score) else 1.0] * len(Xl))

    if global_rows > 0:
        pool = ctx.make_pool(method, batch_no, rows=global_rows, seed_offset=91)
        blocks.append(pool.X)
        source.extend(["global_sobol"] * len(pool.X))
        prior.extend([0.0] * len(pool.X))

    X = np.vstack(blocks) if blocks else np.empty((0, d))
    X, source, prior = _filter_with_metadata(ctx, method, X, source, np.asarray(prior, float))
    return X, list(source), np.asarray(prior, float), atlas_tree, atlas_R


def run_ghost_oof_blindspot_mining(ctx: LiveContext):
    method = "NEW_GHOST_OOF_BLINDSPOT_MINING"
    bank = _ghost_blindspot_bank(ctx)
    cfg = ctx.cfg.get("ghost", {})
    k = int(cfg.get("residual_neighbors", 48))
    max_pinf = float(cfg.get("max_p_infeasible", max(0.05, ctx.safe_threshold)))
    radius = float(cfg.get("diversity_radius", 0.025))

    while _remaining(ctx, method) > 0:
        t0 = time.time()
        q = _q(ctx, method)
        hist_n = len(ctx.history(method))
        bno = hist_n // ctx.batch_size + 1
        front = ctx.current_front(method)

        X, source, blind_prior, atlas_tree, atlas_R = _ghost_candidates(ctx, method, bno, hist_n, bank)
        if len(X) == 0:
            raise RuntimeError(f"{method}: empty candidate pool")
        pred, pinf = ctx.global_model.predict(X)

        kk = max(4, min(k, len(bank[0])))
        dist, nn = atlas_tree.query(X, k=kk)
        if kk == 1:
            dist = dist[:, None]; nn = nn[:, None]
        r = atlas_R[nn]
        scale = np.maximum(dist[:, -1], 1e-8)
        w = np.exp(-0.5 * (dist / scale[:, None]) ** 2)
        w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        local_bias = np.sum(r * w[:, :, None], axis=1)
        corrected = pred + local_bias

        hvi = candidate_hv_gain(front, corrected)
        novelty = ctx.baseline_tree.query(X, k=1)[0]
        safe = np.isfinite(corrected).all(axis=1) & np.isfinite(pinf) & (pinf <= max_pinf)

        score = 2.2 * robust_scale(hvi)
        score += 0.75 * robust_scale(np.maximum(blind_prior, 0.0))
        score += 0.25 * robust_scale(novelty)
        # Reward regions where the local OOF atlas predicts underestimation of
        # either objective, because those are precisely the false-negative regions.
        score += 0.45 * robust_scale(np.maximum(local_bias[:, 0], 0.0) + np.maximum(local_bias[:, 1], 0.0))

        chosen = choose_diverse_by_score(X, score, safe, q, radius=radius)
        if len(chosen) < q:
            # Relax only the diversity preference, never novelty/exact closure.
            fallback = np.argsort(np.where(safe, score, -np.inf))[::-1]
            extra = [int(i) for i in fallback if int(i) not in set(chosen.tolist())][: q - len(chosen)]
            chosen = np.asarray(list(chosen) + extra, int)
        if len(chosen) == 0:
            raise RuntimeError(f"{method}: no candidate survived safety filtering")

        modes = [str(source[i]) for i in chosen]
        ctx.evaluate_batch(
            method,
            X[chosen],
            modes,
            {
                "ghost_blind_prior": blind_prior[chosen],
                "ghost_local_bias_growth": local_bias[chosen, 0],
                "ghost_local_bias_product": local_bias[chosen, 1],
                "ghost_corrected_growth": corrected[chosen, 0],
                "ghost_corrected_product": corrected[chosen, 1],
                "ghost_corrected_hvi": hvi[chosen],
            },
        )
        _progress(ctx, method, t0, extra=f" | blindspots={len(bank[4])} | pool={len(X)}")


# ---------------------------------------------------------------------------
# 2) SPARC — Pareto-success displacement subspace
# ---------------------------------------------------------------------------


def _sparc_basis(ctx: LiveContext, method: str):
    cfg = ctx.cfg.get("sparc", {})
    good_ids = np.where(ctx.good0)[0]
    X = ctx.X0[good_ids]
    Y = ctx.Y0[good_ids]
    tree = cKDTree(X)
    k = max(8, min(int(cfg.get("neighbor_rows", 64)), len(X)))
    elite_per_weight = int(cfg.get("elite_per_weight", 64))
    nw=int(cfg.get("scalarization_weights",13))
    weights=(np.column_stack([1.0-np.linspace(0.0,1.0,nw),np.linspace(0.0,1.0,nw)])
             if Y.shape[1]==2 else simplex_weight_bank(Y.shape[1], n=max(21,nw), seed=ctx.seed+321))

    vectors = []
    vweights = []
    for w in weights:
        util = Y @ np.asarray(w,float)
        elite_n = min(elite_per_weight, len(util))
        elite = np.argsort(util)[-elite_n:]
        for e in elite:
            dist, nn = tree.query(X[e], k=k)
            nn = np.atleast_1d(nn)
            dist = np.atleast_1d(dist)
            best = None
            best_score = -np.inf
            for dd, j in zip(dist[1:], nn[1:]):
                gain = float(util[e] - util[int(j)])
                if gain <= 1e-8 or dd <= 1e-10:
                    continue
                score = gain / (float(dd) + 0.01)
                if score > best_score:
                    best_score = score
                    best = int(j)
            if best is None:
                continue
            dx = X[e] - X[best]
            nrm = np.linalg.norm(dx)
            if nrm <= 1e-10:
                continue
            vectors.append(dx / nrm)
            vweights.append(max(best_score, 1e-8))

    # Exact live successes become high-value direction evidence.
    h = ctx.history(method)
    if len(h) and "exact_front_extension" in h:
        mask = h["exact_front_extension"].astype(bool).to_numpy()
        if np.any(mask):
            xcols = [f"x01__{f}" for f in ctx.features]
            Xh = h.loc[mask, xcols].to_numpy(float)
            for x in Xh:
                _, j = ctx.baseline_tree.query(x, k=1)
                dx = x - ctx.X0[int(j)]
                nrm = np.linalg.norm(dx)
                if nrm > 1e-10:
                    vectors.append(dx / nrm)
                    vweights.append(4.0 * max(vweights) if vweights else 4.0)

    if len(vectors) < 2:
        B = np.eye(len(ctx.features))[:, : min(2, len(ctx.features))]
        return B, np.ones(B.shape[1]), 1.0

    V = np.asarray(vectors, float)
    ww = np.asarray(vweights, float)
    ww = ww / max(np.median(ww[ww > 0]) if np.any(ww > 0) else 1.0, 1e-12)
    A = V * np.sqrt(np.clip(ww, 1e-4, 100.0))[:, None]
    _, s, vt = np.linalg.svd(A, full_matrices=False)
    energy = s * s
    cum = np.cumsum(energy) / max(float(np.sum(energy)), 1e-12)
    max_rank = min(int(cfg.get("max_rank", 4)), len(ctx.features), len(s))
    min_rank = min(int(cfg.get("min_rank", 2)), max_rank)
    threshold = float(cfg.get("energy_threshold", 0.88))
    r = int(np.searchsorted(cum, threshold) + 1)
    r = max(min_rank, min(max_rank, r))
    B = vt[:r].T
    return B, s[:r], float(cum[r - 1])


def _sparc_candidates(ctx: LiveContext, method: str, batch_no: int, B: np.ndarray):
    cfg = ctx.cfg.get("sparc", {})
    rng = np.random.default_rng(ctx.seed + 320_003 + 1009 * batch_no + len(ctx.history(method)))
    d, r = B.shape
    sub_rows = int(cfg.get("subspace_rows", 14_000))
    global_rows = int(cfg.get("global_rows", 3000))
    scales = np.asarray(cfg.get("step_scales", [0.012, 0.030, 0.060, 0.120, 0.200]), float)

    pids = baseline_pareto_row_indices(ctx)
    anchors = ctx.X0[pids] if len(pids) else ctx.X0[np.where(ctx.good0)[0][:1]]
    Xlive, _ = _live_extension_centers(ctx, method)
    if len(Xlive):
        anchors = np.vstack([anchors, np.repeat(Xlive, 3, axis=0)])

    ai = rng.integers(0, len(anchors), size=sub_rows)
    z = rng.normal(size=(sub_rows, r))
    z /= np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)
    amp = rng.choice(scales, size=sub_rows)
    dx = (z @ B.T) * amp[:, None]
    Xs = np.clip(anchors[ai] + dx, 0.0, 1.0)

    blocks = [Xs]
    labels = ["pareto_displacement_subspace"] * len(Xs)
    if global_rows > 0:
        pool = ctx.make_pool(method, batch_no, rows=global_rows, seed_offset=177)
        blocks.append(pool.X)
        labels.extend(["global_sobol_guard"] * len(pool.X))
    X = np.vstack(blocks)
    X, labels = _filter_with_metadata(ctx, method, X, labels)
    return X, list(labels)


def run_sparc_pareto_displacement_subspace(ctx: LiveContext):
    method = "NEW_SPARC_PARETO_DISPLACEMENT_SUBSPACE"
    cfg = ctx.cfg.get("sparc", {})
    radius = float(cfg.get("diversity_radius", 0.025))

    while _remaining(ctx, method) > 0:
        t0 = time.time(); q = _q(ctx, method)
        hist_n = len(ctx.history(method)); bno = hist_n // ctx.batch_size + 1
        B, sval, energy = _sparc_basis(ctx, method)
        X, labels = _sparc_candidates(ctx, method, bno, B)
        if len(X) == 0:
            raise RuntimeError(f"{method}: empty candidate pool")
        pred, pinf = ctx.global_model.predict(X)
        safe = np.isfinite(pred).all(axis=1) & np.isfinite(pinf) & (pinf <= ctx.safe_threshold)
        front = ctx.current_front(method)
        hvi = candidate_hv_gain(front, pred)
        front_dist = np.min(np.linalg.norm(pred[:, None, :] - pareto_points(front)[None, :, :], axis=2), axis=1)
        novelty = ctx.baseline_tree.query(X, k=1)[0]

        score = 2.4 * robust_scale(hvi) + 0.35 * robust_scale(novelty)
        # If the predicted HVI is flat, favor front-near candidates reached via
        # the learned productive subspace instead of arbitrary Sobol order.
        score += 0.20 * robust_scale(1.0 / (front_dist + 0.02))
        source_bonus = np.asarray([0.25 if x == "pareto_displacement_subspace" else 0.0 for x in labels])
        score += source_bonus

        chosen = choose_diverse_by_score(X, score, safe, q, radius=radius)
        if len(chosen) == 0:
            raise RuntimeError(f"{method}: no safe candidates")
        modes = [labels[i] for i in chosen]
        ctx.evaluate_batch(
            method, X[chosen], modes,
            {
                "sparc_rank": np.full(len(chosen), B.shape[1], int),
                "sparc_energy_captured": np.full(len(chosen), energy, float),
                "sparc_pred_hvi": hvi[chosen],
                "sparc_subspace_singular_1": np.full(len(chosen), float(sval[0]) if len(sval) else np.nan),
            },
        )
        _progress(ctx, method, t0, extra=f" | rank={B.shape[1]} energy={energy:.3f} | pool={len(X)}")


# ---------------------------------------------------------------------------
# 3) TRAIL — monotone exact-archive terminal extrapolation
# ---------------------------------------------------------------------------


@dataclass
class _TrailLine:
    anchor_x: np.ndarray
    anchor_y: np.ndarray
    direction: np.ndarray
    normal: np.ndarray
    support: float
    label: str


def _trail_lines(ctx: LiveContext, method: str):
    cfg = ctx.cfg.get("trail", {})
    good_ids = np.where(ctx.good0)[0]
    Xg = ctx.X0[good_ids]
    Yg = ctx.Y0[good_ids]
    global_to_local = {int(g): i for i, g in enumerate(good_ids)}
    tree = cKDTree(Xg)
    normals, pids = _pareto_row_normals(ctx)
    k1 = max(12, min(int(cfg.get("neighbors", 96)), len(Xg)))
    k2 = max(8, min(int(cfg.get("chain_neighbors", 64)), len(Xg)))
    lines: list[_TrailLine] = []

    for pid in pids:
        li = global_to_local.get(int(pid))
        if li is None:
            continue
        xp, yp, n = Xg[li], Yg[li], normals[int(pid)]
        dist, nn = tree.query(xp, k=k1)
        candidates = []
        for dd, j in zip(np.atleast_1d(dist)[1:], np.atleast_1d(nn)[1:]):
            j = int(j); dd = float(dd)
            if dd <= 1e-8:
                continue
            prog = float(np.dot(yp - Yg[j], n))
            if prog <= 1e-5:
                continue
            v1 = xp - Xg[j]
            v1n = np.linalg.norm(v1)
            if v1n <= 1e-10:
                continue

            # Look one step farther inward. Consistent two-edge chains are much
            # less likely to be accidental local noise than a single secant.
            d2, n2 = tree.query(Xg[j], k=k2)
            best_consistency = 0.0
            for jj in np.atleast_1d(n2)[1:]:
                jj = int(jj)
                prog2 = float(np.dot(Yg[j] - Yg[jj], n))
                if prog2 <= 1e-5:
                    continue
                v2 = Xg[j] - Xg[jj]
                den = np.linalg.norm(v1) * np.linalg.norm(v2)
                if den <= 1e-12:
                    continue
                cos = float(np.dot(v1, v2) / den)
                if cos <= 0:
                    continue
                best_consistency = max(best_consistency, cos * math.sqrt(max(prog2, 0.0)))
            support = (prog / (dd + 0.01)) * (0.35 + best_consistency)
            candidates.append((support, v1 / v1n))

        candidates.sort(key=lambda z: z[0], reverse=True)
        for rank, (support, direction) in enumerate(candidates[: int(cfg.get("lines_per_anchor", 3))]):
            lines.append(_TrailLine(xp.copy(), yp.copy(), direction.copy(), n.copy(), float(support), f"archive_chain_r{rank+1}"))

    # A live exact extension supplies even stronger evidence: the displacement
    # from its nearest frozen archive point has already survived exact FBA.
    h = ctx.history(method)
    if len(h) and "exact_front_extension" in h:
        mask = h["exact_front_extension"].astype(bool).to_numpy()
        xcols = [f"x01__{f}" for f in ctx.features]
        for _, row in h.loc[mask].iterrows():
            x = row[xcols].to_numpy(float)
            y = row[ctx.objective_norm_columns].to_numpy(float)
            if not np.isfinite(x).all() or not np.isfinite(y).all():
                continue
            _, j = ctx.baseline_tree.query(x, k=1)
            v = x - ctx.X0[int(j)]
            nrm = np.linalg.norm(v)
            if nrm <= 1e-10:
                continue
            # Use the local frozen-Pareto outward normal nearest this exact extension.
            if len(pids):
                py=ctx.Y0[pids]
                nearest_pid=int(pids[int(np.argmin(np.linalg.norm(py-y[None,:],axis=1)))])
                n=normals[nearest_pid].copy()
            else:
                n=np.ones(ctx.objective_dim,float)/math.sqrt(float(ctx.objective_dim))
            gain = float(row.get("exact_hv_gain_this_call", 0.0))
            lines.append(_TrailLine(x.copy(), y.copy(), v / nrm, n, 10.0 + 1000.0 * max(gain, 0.0), "live_exact_extension"))

    return lines


def _trail_candidates(ctx: LiveContext, method: str, batch_no: int, lines: list[_TrailLine]):
    cfg = ctx.cfg.get("trail", {})
    rng = np.random.default_rng(ctx.seed + 430_007 + 1009 * batch_no + len(ctx.history(method)))
    gammas = np.asarray(cfg.get("extrapolation_steps", [0.008, 0.018, 0.035, 0.060, 0.100, 0.160]), float)
    repeats = int(cfg.get("jitter_repeats", 3))
    jitter = float(cfg.get("orthogonal_jitter", 0.008))
    global_rows = int(cfg.get("global_rows", 4096))

    Xs = []; supports = []; anchors_y = []; normals = []; labels = []
    for line in lines:
        for step in gammas:
            for rr in range(repeats):
                eps = rng.normal(size=len(ctx.features))
                eps -= np.dot(eps, line.direction) * line.direction
                eps /= max(np.linalg.norm(eps), 1e-12)
                # Keep extrapolation dominant.  Orthogonal exploration grows only
                # sublinearly with step and is capped, avoiding the large 0.16
                # off-trail perturbations that a linear scaling would create at
                # the largest extrapolation radius.
                jitter_amp = min(0.035, jitter * math.sqrt(max(step, gammas[0]) / max(gammas[0], 1e-8)))
                x = np.clip(line.anchor_x + step * line.direction + jitter_amp * eps, 0.0, 1.0)
                Xs.append(x); supports.append(line.support); anchors_y.append(line.anchor_y); normals.append(line.normal); labels.append(line.label)

    if global_rows > 0:
        pool = ctx.make_pool(method, batch_no, rows=global_rows, seed_offset=263)
        for x in pool.X:
            Xs.append(x); supports.append(0.0); anchors_y.append([np.nan] * ctx.objective_dim); normals.append([0.0] * ctx.objective_dim); labels.append("global_sobol_guard")

    X = np.asarray(Xs, float)
    return _filter_with_metadata(
        ctx, method, X, labels, np.asarray(supports, float), np.asarray(anchors_y, float), np.asarray(normals, float)
    )


def run_trail_monotone_archive_extrapolation(ctx: LiveContext):
    method = "NEW_TRAIL_MONOTONE_ARCHIVE_EXTRAPOLATION"
    cfg = ctx.cfg.get("trail", {})
    radius = float(cfg.get("diversity_radius", 0.022))
    while _remaining(ctx, method) > 0:
        t0 = time.time(); q = _q(ctx, method)
        hist_n = len(ctx.history(method)); bno = hist_n // ctx.batch_size + 1
        lines = _trail_lines(ctx, method)
        if len(lines) == 0:
            raise RuntimeError(f"{method}: no monotone archive lines could be constructed")
        X, labels, support, anchor_y, normal = _trail_candidates(ctx, method, bno, lines)
        labels = list(labels); support = np.asarray(support, float); anchor_y = np.asarray(anchor_y, float); normal = np.asarray(normal, float)
        if len(X) == 0:
            raise RuntimeError(f"{method}: empty candidate pool")
        pred, pinf = ctx.global_model.predict(X)
        safe = np.isfinite(pred).all(axis=1) & np.isfinite(pinf) & (pinf <= ctx.safe_threshold)
        front = ctx.current_front(method)
        hvi = candidate_hv_gain(front, pred)
        novelty = ctx.baseline_tree.query(X, k=1)[0]
        prog = np.zeros(len(X), float)
        valid_anchor = np.isfinite(anchor_y).all(axis=1)
        prog[valid_anchor] = np.sum((pred[valid_anchor] - anchor_y[valid_anchor]) * normal[valid_anchor], axis=1)
        prog = np.maximum(prog, 0.0)
        score = 2.0 * robust_scale(hvi) + 0.65 * robust_scale(np.maximum(support, 0.0))
        score += 0.55 * robust_scale(prog) + 0.20 * robust_scale(novelty)
        chosen = choose_diverse_by_score(X, score, safe, q, radius=radius)
        if len(chosen) == 0:
            raise RuntimeError(f"{method}: no safe candidates")
        modes = [labels[i] for i in chosen]
        ctx.evaluate_batch(
            method, X[chosen], modes,
            {
                "trail_support": support[chosen],
                "trail_pred_hvi": hvi[chosen],
                "trail_outward_progress": prog[chosen],
            },
        )
        _progress(ctx, method, t0, extra=f" | lines={len(lines)} | pool={len(X)}")


# ---------------------------------------------------------------------------
# 4) PACE — calibrated missing-cell expansion
# ---------------------------------------------------------------------------


def _pace_targets(front: np.ndarray, max_targets: int = 24):
    all_t = []
    for outward in (0.018, 0.035, 0.060, 0.090):
        for t, gap in candidate_targets_from_front(front, outward=outward, max_targets=max_targets):
            all_t.append((np.asarray(t, float), float(gap), float(outward)))
    # Deduplicate near-identical targets while retaining larger gaps first.
    all_t.sort(key=lambda z: (z[1], z[2]), reverse=True)
    keep = []
    for item in all_t:
        if not keep or min(np.linalg.norm(item[0] - k[0]) for k in keep) > 0.012:
            keep.append(item)
        if len(keep) >= max_targets:
            break
    return keep


def _pace_pool(ctx: LiveContext, method: str, batch_no: int):
    cfg = ctx.cfg.get("pace", {})
    global_rows = int(cfg.get("global_rows", 14_000))
    jitter_rows = int(cfg.get("frontier_jitter_rows", 5000))
    max_candidates = int(cfg.get("max_candidates", 18_000))
    rng = np.random.default_rng(ctx.seed + 540_011 + 1009 * batch_no + len(ctx.history(method)))
    d = len(ctx.features)

    pool = ctx.make_pool(method, batch_no, rows=global_rows, seed_offset=347)
    blocks = [pool.X]
    labels = ["global_sobol"] * len(pool.X)
    pids = baseline_pareto_row_indices(ctx)
    if len(pids) and jitter_rows > 0:
        centers = ctx.X0[rng.choice(pids, size=jitter_rows, replace=True)]
        scales = rng.choice(np.asarray([0.008, 0.020, 0.050, 0.100]), size=jitter_rows)
        Xj = np.clip(centers + rng.normal(size=(jitter_rows, d)) * scales[:, None], 0.0, 1.0)
        blocks.append(Xj); labels.extend(["frontier_jitter"] * len(Xj))
    X = np.vstack(blocks)
    X, labels = _filter_with_metadata(ctx, method, X, labels)
    if len(X) > max_candidates:
        ids = np.linspace(0, len(X)-1, max_candidates, dtype=int)
        X = X[ids]; labels = np.asarray(labels, object)[ids]
    return X, list(labels)


def run_pace_calibrated_cell_expansion(ctx: LiveContext):
    method = "NEW_PACE_CALIBRATED_CELL_EXPANSION"
    cfg = ctx.cfg.get("pace", {})
    ids, atlas_X, atlas_Y, atlas_oof, atlas_R, atlas_tree = _oof_atlas(ctx)
    k = max(8, min(int(cfg.get("residual_neighbors", 48)), len(atlas_X)))
    S = max(8, int(cfg.get("residual_samples", 20)))
    prefilter = int(cfg.get("prefilter_candidates", 5000))
    max_targets = int(cfg.get("max_targets", 24))
    radius = float(cfg.get("design_diversity_radius", 0.020))

    while _remaining(ctx, method) > 0:
        t0 = time.time(); q = _q(ctx, method)
        hist_n = len(ctx.history(method)); bno = hist_n // ctx.batch_size + 1
        front = ctx.current_front(method)
        targets = _pace_targets(front, max_targets=max_targets)
        if not targets:
            raise RuntimeError(f"{method}: no objective-space cells could be constructed")

        X, labels = _pace_pool(ctx, method, bno)
        pred, pinf = ctx.global_model.predict(X)
        safe = np.isfinite(pred).all(axis=1) & np.isfinite(pinf) & (pinf <= ctx.safe_threshold)

        T = np.vstack([t[0] for t in targets])
        gap = np.asarray([t[1] for t in targets], float)
        predicted_hvi = candidate_hv_gain(front, pred)
        # Fast prefilter retains high-HVI points and points close to any desired cell.
        dtarget = np.min(np.linalg.norm(pred[:, None, :] - T[None, :, :], axis=2), axis=1)
        fast = 2.0 * robust_scale(predicted_hvi) + robust_scale(1.0 / (dtarget + 0.025))
        fast[~safe] = -np.inf
        use = np.where(np.isfinite(fast))[0]
        if len(use) > prefilter:
            use = use[np.argsort(fast[use])[::-1][:prefilter]]
        if len(use) == 0:
            raise RuntimeError(f"{method}: no safe prefilter candidates")

        Xu = X[use]; predu = pred[use]; pinfu = pinf[use]
        dist, nn = atlas_tree.query(Xu, k=k)
        if k == 1:
            dist = dist[:, None]; nn = nn[:, None]
        bank = atlas_R[nn]
        # Deterministic empirical residual draws from local neighbors.
        rng = np.random.default_rng(ctx.seed + 540_101 + 1009*bno + hist_n)
        draw = rng.integers(0, k, size=(len(Xu), S))
        rr = bank[np.arange(len(Xu))[:, None], draw]
        samples = predu[:, None, :] + rr
        flat_hvi = candidate_hv_gain(front, samples.reshape(-1, 2)).reshape(len(Xu), S)

        # Cell-specific expected exact gain. Each cell competes for one batch slot.
        cell_score = np.zeros((len(targets), len(Xu)), float)
        cell_reach = np.zeros_like(cell_score)
        for ti, (target, g, outward) in enumerate(targets):
            sigma = max(0.020, 0.35 * float(g) + 0.25 * float(outward))
            ds = np.linalg.norm(samples - target[None, None, :], axis=2)
            kernel = np.exp(-0.5 * (ds / sigma) ** 2)
            # Probability mass landing close to the desired missing cell.
            reach = np.mean(kernel, axis=1)
            expected_cell_hvi = np.mean(flat_hvi * kernel, axis=1)
            ppos = np.mean(flat_hvi > 1e-14, axis=1)
            weight = max(float(g), 0.01)
            cell_score[ti] = weight * (expected_cell_hvi + 0.20 * reach * ppos)
            cell_score[ti] *= np.clip(1.0 - pinfu, 0.0, 1.0)
            cell_reach[ti] = reach

        # Greedy bipartite cell/candidate assignment: one candidate per cell and
        # explicit design diversity across the batch. This prevents a batch from
        # collapsing onto a single high-EHVI neighborhood.
        chosen_local = []
        chosen_target = []
        available_targets = set(range(len(targets)))
        available_candidates = np.ones(len(Xu), bool)
        for _ in range(q):
            best = None; best_val = -np.inf
            for ti in available_targets:
                vals = cell_score[ti].copy()
                vals[~available_candidates] = -np.inf
                if chosen_local:
                    dd = cKDTree(Xu[np.asarray(chosen_local, int)]).query(Xu, k=1)[0]
                    vals[dd < radius] = -np.inf
                j = int(np.nanargmax(vals)) if np.any(np.isfinite(vals)) else -1
                if j >= 0 and vals[j] > best_val:
                    best_val = float(vals[j]); best = (ti, j)
            if best is None:
                break
            ti, j = best
            chosen_target.append(ti); chosen_local.append(j)
            available_targets.remove(ti); available_candidates[j] = False

        if len(chosen_local) < q:
            # Relax cell uniqueness/diversity only enough to fill the exact batch.
            aggregate = np.max(cell_score, axis=0)
            order = np.argsort(aggregate)[::-1]
            for j in order:
                if int(j) not in chosen_local and np.isfinite(aggregate[j]):
                    chosen_local.append(int(j)); chosen_target.append(int(np.argmax(cell_score[:, j])))
                if len(chosen_local) >= q:
                    break
        if not chosen_local:
            raise RuntimeError(f"{method}: cell assignment produced no candidates")

        chosen_local = np.asarray(chosen_local[:q], int)
        chosen_target = np.asarray(chosen_target[:q], int)
        global_ids = use[chosen_local]
        modes = [f"cell_{int(ti)}|{labels[int(gi)]}" for ti, gi in zip(chosen_target, global_ids)]
        ctx.evaluate_batch(
            method,
            X[global_ids],
            modes,
            {
                "pace_target_index": chosen_target,
                "pace_target_growth": T[chosen_target, 0],
                "pace_target_product": T[chosen_target, 1],
                "pace_target_gap": gap[chosen_target],
                "pace_cell_score": np.asarray([cell_score[ti, j] for ti, j in zip(chosen_target, chosen_local)]),
                "pace_cell_reach": np.asarray([cell_reach[ti, j] for ti, j in zip(chosen_target, chosen_local)]),
                "pace_pred_hvi": predicted_hvi[global_ids],
            },
        )
        _progress(ctx, method, t0, extra=f" | cells={len(targets)} | prefilter={len(Xu)}")


NEW_METHOD_NOTES = {
    "NEW_GHOST_OOF_BLINDSPOT_MINING": (
        "GHOST mines cross-fitted Pareto false-negatives in the frozen 50k landscape: high-value prepared-prior rows that the OOF optimizer surrogate under-called. "
        "It generates new continuous designs around those blind-spot archetypes, transports local OOF residual bias, and exact-validates the most promising corrected-HV candidates."
    ),
    "NEW_SPARC_PARETO_DISPLACEMENT_SUBSPACE": (
        "SPARC learns a low-rank decision-space basis from prepared-prior displacement vectors that repeatedly move nearby designs toward high scalarized Pareto utility; "
        "live exact extensions reinforce that basis. New candidates are searched predominantly inside this archive-demonstrated productive subspace, with a global Sobol guard."
    ),
    "NEW_TRAIL_MONOTONE_ARCHIVE_EXTRAPOLATION": (
        "TRAIL extracts locally monotone exact-archive improvement chains that terminate at frozen Pareto designs, then extrapolates the terminal decision-space direction beyond the sampled boundary. "
        "Two-edge directional consistency provides support against isolated surrogate hallucinations, and successful live extensions create new exact-validated trails."
    ),
    "NEW_PACE_CALIBRATED_CELL_EXPANSION": (
        "PACE decomposes unresolved objective space into explicit Pareto-gap cells, uses local OOF residual transport to estimate each candidate's empirical exact-HVI distribution relative to each cell, "
        "and performs one-candidate-per-cell batch assignment to expand multiple missing regions without qEHVI-style batch collapse."
    ),
}

NEW_RUNNERS = {
    "NEW_GHOST_OOF_BLINDSPOT_MINING": run_ghost_oof_blindspot_mining,
    "NEW_SPARC_PARETO_DISPLACEMENT_SUBSPACE": run_sparc_pareto_displacement_subspace,
    "NEW_TRAIL_MONOTONE_ARCHIVE_EXTRAPOLATION": run_trail_monotone_archive_extrapolation,
    "NEW_PACE_CALIBRATED_CELL_EXPANSION": run_pace_calibrated_cell_expansion,
}
