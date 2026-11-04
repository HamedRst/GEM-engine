from __future__ import annotations

"""Objective-count-agnostic global surrogate with the source benchmark interface."""

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, ExtraTreesClassifier


class GenericGlobalSurrogate:
    def __init__(self, suite_cfg: dict | None = None, seed: int = 0):
        self.cfg = dict(suite_cfg or {})
        self.seed = int(seed)
        self.models = []
        self.classifier = None
        self.constant_pinf = 0.0
        self.objective_dim = None
        self.objective_contracts = []
        self.features = []

    def fit(self, X, Y, feasible):
        X = np.asarray(X, float)
        Y = np.asarray(Y, float)
        feasible = np.asarray(feasible, bool).reshape(-1)
        if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y) or len(X) != len(feasible):
            raise RuntimeError(f"GenericGlobalSurrogate shape mismatch X={X.shape} Y={Y.shape} F={feasible.shape}")
        if Y.shape[1] < 1:
            raise RuntimeError("At least one objective is required")
        self.objective_dim = int(Y.shape[1])
        s = self.cfg.get("surrogate", {}) or {}
        self.objective_contracts = list(s.get("objective_contracts", []) or [])
        self.features = list(s.get("features", []) or [])
        if self.objective_contracts and len(self.objective_contracts) != self.objective_dim:
            raise RuntimeError(f"objective_contracts length {len(self.objective_contracts)} != Y dim {self.objective_dim}")
        if self.objective_contracts and not self.features:
            raise RuntimeError("features are required when objective_contracts are supplied")
        n_estimators = int(s.get("generic_n_estimators", 192))
        min_leaf = int(s.get("generic_min_samples_leaf", 2))
        max_features = s.get("generic_max_features", 1.0)
        n_jobs = int(s.get("generic_n_jobs", -1))

        self.models = []
        if self.objective_contracts:
            from generic_problem import design_objectives_normalized_from_x01
            deterministic = design_objectives_normalized_from_x01(X, self.objective_contracts, self.features)
        else:
            deterministic = np.full_like(Y, np.nan, float)
        for j in range(self.objective_dim):
            contract = self.objective_contracts[j] if self.objective_contracts else {}
            if bool(contract.get("deterministic_from_design", False)):
                pred = deterministic[:, j]
                ok = np.isfinite(pred) & np.isfinite(Y[:, j])
                if not ok.any() or not np.allclose(pred[ok], Y[ok, j], rtol=1e-10, atol=1e-10):
                    raise RuntimeError(f"Objective {j}: deterministic design contract does not reproduce compiled objective values")
                self.models.append(None)
                continue
            good = feasible & np.isfinite(Y[:, j]) & np.isfinite(X).all(axis=1)
            if int(good.sum()) < 8:
                good = np.isfinite(Y[:, j]) & np.isfinite(X).all(axis=1)
            if int(good.sum()) < 2:
                raise RuntimeError(f"Objective {j}: insufficient finite rows for surrogate fit")
            model = ExtraTreesRegressor(
                n_estimators=n_estimators,
                random_state=self.seed + 101 * (j + 1),
                min_samples_leaf=min_leaf,
                max_features=max_features,
                n_jobs=n_jobs,
            )
            model.fit(X[good], Y[good, j])
            self.models.append(model)

        yinf = (~feasible).astype(int)
        if len(np.unique(yinf)) == 2:
            clf = ExtraTreesClassifier(
                n_estimators=max(96, n_estimators // 2),
                random_state=self.seed + 7001,
                min_samples_leaf=max(1, min_leaf),
                max_features=max_features,
                n_jobs=n_jobs,
                class_weight="balanced",
            )
            clf.fit(X, yinf)
            self.classifier = clf
            self.constant_pinf = float(np.mean(yinf))
        else:
            self.classifier = None
            self.constant_pinf = float(yinf[0]) if len(yinf) else 0.0
        return self

    def predict(self, X):
        X = np.atleast_2d(np.asarray(X, float))
        if not self.models or self.objective_dim is None:
            raise RuntimeError("GenericGlobalSurrogate must be fit before predict")
        det = None
        if self.objective_contracts:
            from generic_problem import design_objectives_normalized_from_x01
            det = design_objectives_normalized_from_x01(X, self.objective_contracts, self.features)
        cols=[]
        for j,m in enumerate(self.models):
            if m is None:
                if det is None or not np.isfinite(det[:,j]).all():
                    raise RuntimeError(f"Objective {j}: deterministic prediction unavailable")
                cols.append(det[:,j])
            else:
                cols.append(m.predict(X))
        pred = np.column_stack(cols).astype(float)
        if self.classifier is not None:
            classes = list(self.classifier.classes_)
            probs = self.classifier.predict_proba(X)
            if 1 in classes:
                pinf = probs[:, classes.index(1)]
            else:
                pinf = np.full(len(X), self.constant_pinf, float)
        else:
            pinf = np.full(len(X), self.constant_pinf, float)
        return pred, np.clip(np.asarray(pinf, float), 0.0, 1.0)


# Source compatibility alias.
GlobalSurrogate = GenericGlobalSurrogate
