from __future__ import annotations

"""Deterministic normalized medium-resource burden.

The cost objective is intentionally *not* a monetary claim. Each variable is
first represented in the benchmark's frozen [0,1] design coordinates, where
0 is the variable's lower allowed uptake and 1 its upper allowed uptake. The
resource burden is a non-negative weighted mean across variables, making the
score dimension-independent across species. The optimization engine maximizes
resource efficiency = 1 - burden while reports retain burden as the minimized
cost objective.
"""

from typing import Mapping, Sequence
import numpy as np


def normalized_cost_weights(features: Sequence[str], spec: Mapping | None = None) -> np.ndarray:
    features = [str(f) for f in features]
    spec = dict(spec or {})
    raw = spec.get("feature_weights", {}) or {}
    default = float(spec.get("default_feature_weight", 1.0))
    w = np.asarray([float(raw.get(f, default)) for f in features], float)
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Resource-cost weights must be finite and non-negative")
    if float(w.sum()) <= 0:
        raise ValueError("At least one resource-cost weight must be positive")
    return w / float(w.sum())


def resource_burden(X01: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X01, float))
    if X.ndim != 2:
        raise ValueError(f"X01 must be 2-D, got {X.shape}")
    if weights is None:
        w = np.full(X.shape[1], 1.0 / max(1, X.shape[1]), float)
    else:
        w = np.asarray(weights, float).reshape(-1)
        if len(w) != X.shape[1]:
            raise ValueError(f"weights length {len(w)} != design dimension {X.shape[1]}")
        if not np.isfinite(w).all() or np.any(w < 0) or float(w.sum()) <= 0:
            raise ValueError("Invalid resource-cost weights")
        w = w / float(w.sum())
    # Defensive clipping: all live/archived points should already be in range.
    return np.clip(X, 0.0, 1.0) @ w


def resource_efficiency(X01: np.ndarray, weights: np.ndarray | None = None) -> np.ndarray:
    return 1.0 - resource_burden(X01, weights)
