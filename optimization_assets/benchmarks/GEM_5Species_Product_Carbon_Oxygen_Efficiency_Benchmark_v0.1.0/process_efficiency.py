from __future__ import annotations

"""Deterministic carbon- and oxygen-efficiency objectives.

Each selected exchange variable is represented in the frozen benchmark's
normalized [0,1] design coordinates.  Demand is the normalized uptake-capacity
coordinate and efficiency is one minus that demand.  These objectives therefore
require no FBA relabeling of the completed 50k archive.
"""
from typing import Sequence
import numpy as np


def feature_index(features: Sequence[str], feature: str) -> int:
    f = [str(x) for x in features]
    if str(feature) not in f:
        raise KeyError(f"Feature {feature!r} is not present in design features {f}")
    return f.index(str(feature))


def normalized_demand(X01: np.ndarray, idx: int) -> np.ndarray:
    X = np.atleast_2d(np.asarray(X01, float))
    if X.ndim != 2:
        raise ValueError(f"X01 must be 2-D, got {X.shape}")
    if idx < 0 or idx >= X.shape[1]:
        raise IndexError(f"feature index {idx} outside design dimension {X.shape[1]}")
    return np.clip(X[:, int(idx)], 0.0, 1.0)


def efficiency(X01: np.ndarray, idx: int) -> np.ndarray:
    return 1.0 - normalized_demand(X01, idx)


def carbon_oxygen_metrics(X01: np.ndarray, features: Sequence[str], carbon_feature: str, oxygen_feature: str):
    ci = feature_index(features, carbon_feature)
    oi = feature_index(features, oxygen_feature)
    if ci == oi:
        raise ValueError("Carbon and oxygen features must be distinct")
    c = normalized_demand(X01, ci)
    o = normalized_demand(X01, oi)
    return {
        "carbon_feature_index": ci,
        "oxygen_feature_index": oi,
        "carbon_demand": c,
        "oxygen_demand": o,
        "carbon_efficiency": 1.0 - c,
        "oxygen_efficiency": 1.0 - o,
    }
