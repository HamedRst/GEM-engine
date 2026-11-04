from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def normalize_matrix(df: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    lo = np.nanmin(X, axis=0)
    hi = np.nanmax(X, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (X - lo) / span, lo, hi


def _orient(Y: np.ndarray, directions: list[str]) -> np.ndarray:
    Z = np.asarray(Y, float).copy()
    for j, d in enumerate(directions):
        if str(d).lower().startswith("min"):
            Z[:, j] *= -1.0
    return Z


def normalized_oriented_targets(Y: np.ndarray, directions: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Orient all objectives as maximize, then min-max normalize using the supplied full set."""
    Z = _orient(Y, directions)
    lo = np.nanmin(Z, axis=0)
    hi = np.nanmax(Z, axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    return (Z - lo) / span, lo, hi


def pareto_mask(Y: np.ndarray, directions: list[str]) -> np.ndarray:
    """Return nondominated rows with all objectives oriented by ``directions``.

    The implementation sorts on objective 0 and maintains only the current skyline.
    Typical complexity is O(n * |front|), which is practical for 50k-row GEM
    landscapes when the Pareto set is much smaller than the full archive.
    Duplicate objective vectors are retained as nondominated.
    """
    Z = _orient(Y, directions)
    good = np.all(np.isfinite(Z), axis=1)
    original = np.where(good)[0]
    P = Z[good]
    out = np.zeros(len(Z), dtype=bool)
    if len(P) == 0:
        return out

    # Descending objective 0; stable sort helps deterministic duplicate handling.
    order = np.argsort(-P[:, 0], kind="mergesort")
    front_idx: list[int] = []
    front_vals = np.empty((0, P.shape[1]), dtype=float)

    for idx in order:
        point = P[idx]
        if len(front_vals):
            dominates_point = np.all(front_vals >= point, axis=1) & np.any(front_vals > point, axis=1)
            if np.any(dominates_point):
                continue
            point_dominates = np.all(point >= front_vals, axis=1) & np.any(point > front_vals, axis=1)
            if np.any(point_dominates):
                keep = ~point_dominates
                front_vals = front_vals[keep]
                front_idx = [front_idx[i] for i in np.where(keep)[0]]
        front_idx.append(int(idx))
        front_vals = np.vstack([front_vals, point])

    out[original[np.asarray(front_idx, dtype=int)]] = True
    return out


def igd_plus(approximation: np.ndarray, reference: np.ndarray) -> float:
    """IGD+ for maximize-oriented, already normalized objective vectors.

    For each reference point r, only objective shortfalls of an approximation
    point a contribute: max(r - a, 0). Lower is better; zero is perfect recovery.
    """
    A = np.asarray(approximation, float)
    R = np.asarray(reference, float)
    A = A[np.all(np.isfinite(A), axis=1)]
    R = R[np.all(np.isfinite(R), axis=1)]
    if len(A) == 0 or len(R) == 0:
        return float("nan")
    # Chunk reference points to keep memory bounded for larger fronts.
    vals = []
    chunk = 512
    for start in range(0, len(R), chunk):
        r = R[start : start + chunk]
        diff = np.maximum(r[:, None, :] - A[None, :, :], 0.0)
        dist = np.sqrt(np.sum(diff * diff, axis=2))
        vals.append(np.min(dist, axis=1))
    return float(np.mean(np.concatenate(vals)))


def hypervolume_2d(front: np.ndarray, reference: tuple[float, float] = (0.0, 0.0)) -> float:
    """Exact 2D hypervolume for maximize-oriented points relative to ``reference``."""
    P = np.asarray(front, float)
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) == 0:
        return 0.0
    ref = np.asarray(reference, float)
    P = P[np.all(P >= ref, axis=1)]
    if len(P) == 0:
        return 0.0
    mask = pareto_mask(P, ["maximize", "maximize"])
    P = P[mask]
    P = P[np.argsort(P[:, 0])]
    hv = 0.0
    prev_x = ref[0]
    for x, y in P:
        if x > prev_x:
            hv += (x - prev_x) * max(y - ref[1], 0.0)
            prev_x = x
    return float(hv)


def robust_range(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if not len(x):
        return 1.0
    q5, q95 = np.quantile(x, [0.05, 0.95])
    return max(float(q95 - q5), 1e-12)
