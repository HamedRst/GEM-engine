from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor


@dataclass
class GrowthPriorSurrogate:
    regressor: ExtraTreesRegressor
    classifier: ExtraTreesClassifier | None
    constant_infeasible_probability: float | None
    infeasible_col: int | None
    growth_scale: float

    @classmethod
    def fit(cls, X01: np.ndarray, growth_raw: np.ndarray, feasible: np.ndarray, cfg: dict, seed: int):
        X01 = np.asarray(X01, float)
        g = np.asarray(growth_raw, float)
        feasible = np.asarray(feasible, bool)
        good = feasible & np.isfinite(g) & (g >= 0)
        if int(good.sum()) < 100:
            raise RuntimeError(f"Too few feasible nominal-growth rows for prior surrogate: {int(good.sum())}")
        growth_scale = float(np.nanmax(g[good]))
        if not np.isfinite(growth_scale) or growth_scale <= 0:
            raise RuntimeError(f"Invalid nominal growth scale: {growth_scale}")
        scfg = cfg["prior_growth_surrogate"]
        reg = ExtraTreesRegressor(
            n_estimators=int(scfg.get("regressor_trees", 500)),
            min_samples_leaf=int(scfg.get("min_samples_leaf", 2)),
            max_features=1.0,
            n_jobs=-1,
            random_state=int(seed) + 1,
        )
        reg.fit(X01[good], g[good] / growth_scale)

        yin = (~feasible).astype(int)
        classes = np.unique(yin)
        clf = None
        const = None
        col = None
        if len(classes) == 1:
            const = float(classes[0])
        else:
            clf = ExtraTreesClassifier(
                n_estimators=int(scfg.get("classifier_trees", 350)),
                min_samples_leaf=int(scfg.get("min_samples_leaf", 2)),
                max_features=1.0,
                class_weight="balanced",
                n_jobs=-1,
                random_state=int(seed) + 2,
            )
            clf.fit(X01, yin)
            fclasses = np.asarray(clf.classes_)
            col = int(np.where(fclasses == 1)[0][0])
        return cls(reg, clf, const, col, growth_scale)

    def predict_growth_raw(self, X01: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        X01 = np.atleast_2d(np.asarray(X01, float))
        pred = np.asarray(self.regressor.predict(X01), float) * self.growth_scale
        pred = np.maximum(pred, 0.0)
        if self.constant_infeasible_probability is not None:
            pinf = np.full(len(X01), self.constant_infeasible_probability, float)
        else:
            prob = np.asarray(self.classifier.predict_proba(X01), float)
            pinf = prob[:, int(self.infeasible_col)]
        return pred, pinf


def validate_stress_groups(features: Iterable[str], stress_groups: dict[str, list[str]], required_classes: Iterable[str]) -> None:
    features = list(map(str, features))
    missing_classes = [c for c in required_classes if c not in stress_groups]
    if missing_classes:
        raise RuntimeError(f"Stress-group configuration missing classes: {missing_classes}")
    for cls in required_classes:
        members = list(map(str, stress_groups.get(cls, [])))
        if not members:
            raise RuntimeError(f"Stress class {cls!r} has no configured variables")
        missing = [r for r in members if r not in features]
        if missing:
            raise RuntimeError(f"Stress class {cls!r} contains variables absent from the design space: {missing}")


def stress_transform_x01(X01: np.ndarray, features: list[str], members: Iterable[str], fraction_toward_lower_bound: float) -> np.ndarray:
    """Move selected in-domain variables toward their lower bound.

    In normalized coordinates, the lower bound is 0. A fraction of 0.5 therefore
    halves the distance from the lower bound while never leaving [0,1].
    """
    X = np.atleast_2d(np.asarray(X01, float)).copy()
    f = float(fraction_toward_lower_bound)
    if not (0.0 <= f <= 1.0):
        raise ValueError(f"fraction_toward_lower_bound must be in [0,1], got {f}")
    idx = [features.index(str(r)) for r in members]
    X[:, idx] = np.clip(X[:, idx] * f, 0.0, 1.0)
    return X


def derive_prior_robustness(
    model: GrowthPriorSurrogate,
    X01: np.ndarray,
    exact_nominal_growth: np.ndarray,
    exact_nominal_feasible: np.ndarray,
    features: list[str],
    stress_groups: dict[str, list[str]],
    stress_classes: list[str],
    fraction_toward_lower_bound: float,
    pinf_threshold: float,
    enforce_monotone: bool = True,
) -> dict:
    X01 = np.asarray(X01, float)
    nominal = np.asarray(exact_nominal_growth, float)
    nominal_feasible = np.asarray(exact_nominal_feasible, bool)
    validate_stress_groups(features, stress_groups, stress_classes)

    scenario = {}
    scenario_pinf = {}
    for cls in stress_classes:
        Xs = stress_transform_x01(X01, features, stress_groups[cls], fraction_toward_lower_bound)
        pred, pinf = model.predict_growth_raw(Xs)
        bad = (~np.isfinite(pred)) | (~np.isfinite(pinf)) | (pinf > float(pinf_threshold))
        pred[bad] = 0.0
        if enforce_monotone:
            good_nominal = np.isfinite(nominal)
            pred[good_nominal] = np.minimum(pred[good_nominal], np.maximum(nominal[good_nominal], 0.0))
        scenario[cls] = pred
        scenario_pinf[cls] = pinf

    M = np.column_stack([scenario[c] for c in stress_classes])
    robust = np.nanmin(M, axis=1)
    robust[~nominal_feasible] = np.nan
    retention = np.full(len(nominal), np.nan, float)
    ok = nominal_feasible & np.isfinite(nominal) & (nominal > 1e-12) & np.isfinite(robust)
    retention[ok] = np.clip(robust[ok] / nominal[ok], 0.0, 1.0 + 1e-8)
    worst_idx = np.nanargmin(np.where(np.isfinite(M), M, np.inf), axis=1)
    worst_class = np.asarray([stress_classes[i] for i in worst_idx], object)
    worst_class[~nominal_feasible] = "nominal_infeasible"
    return {
        "robust_growth": robust,
        "retention": retention,
        "worst_class": worst_class,
        "scenario_growth": scenario,
        "scenario_pinf": scenario_pinf,
    }
