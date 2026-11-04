from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml
from scipy.spatial import cKDTree
from scipy.stats import qmc
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.model_selection import KFold


GROWTH_CANDIDATES = ("maximum_growth", "exact_maximum_growth", "maximum_growth_exact", "growth_max")
PRODUCT_CANDIDATES = ("product_flux", "exact_product_flux", "product_flux_exact", "product")
STATUS_CANDIDATES = ("status", "solver_status", "fba_status", "exact_status")
INFEASIBLE_WORDS = ("infeasible", "error", "failed", "failure", "unbounded")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def save_json(obj, path: str | Path):
    def conv(x):
        if isinstance(x, Path):
            return str(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, pd.Timestamp):
            return x.isoformat()
        raise TypeError(type(x).__name__)

    Path(path).write_text(json.dumps(obj, indent=2, default=conv), encoding="utf-8")


def detect_column(columns: Sequence[str], candidates: Sequence[str]) -> str | None:
    cols = list(columns)
    for c in candidates:
        if c in cols:
            return c
    low = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    for col in cols:
        ll = col.lower()
        for c in candidates:
            if c.lower() in ll:
                return col
    return None


def locate_parent_config(explicit: str | None = None, start: str | Path | None = None) -> Path:
    if explicit:
        p = Path(explicit).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Parent Step4G config not found: {p}")
        return p

    env = os.getenv("GEM_STEP4G_CONFIG")
    if env:
        p = Path(env).expanduser().resolve()
        if p.exists():
            return p

    here = Path(start or __file__).resolve()
    roots = [here.parent, *here.parents]
    direct_names = [
        Path("config.yaml"),
        Path("../config.yaml"),
        Path("../GEM_Learning_Analysis_Step4G_Optimizer_Research_v0.1.0/config.yaml"),
        Path("../GEM_Learning_Analysis_Step4G_Optimizer_Research_v0.1.1/config.yaml"),
    ]
    for root in roots[:4]:
        for rel in direct_names:
            p = (root / rel).resolve()
            if p.exists() and "Step4G" in str(p.parent):
                return p

    # Broader sibling search under a plausible GEM root.
    seen = set()
    for root in roots[:5]:
        if root in seen or not root.exists():
            continue
        seen.add(root)
        for pat in (
            "GEM_Learning_Analysis_Step4G_Optimizer_Research*/config.yaml",
            "**/GEM_Learning_Analysis_Step4G_Optimizer_Research*/config.yaml",
        ):
            hits = sorted(root.glob(pat))
            if hits:
                # Prefer v0.1.0 because that is the current working project in the study.
                hits.sort(key=lambda p: ("v0.1.0" in str(p), p.stat().st_mtime), reverse=True)
                return hits[0].resolve()

    raise FileNotFoundError(
        "Could not auto-find the Step4G config.yaml. Set GEM_STEP4G_CONFIG or pass --parent-config."
    )


def load_parent_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text())
    out = Path(cfg["paths"]["output_root"])
    if not out.is_absolute():
        cfg["paths"]["output_root"] = str((path.parent / out).resolve())
    cfg["__config_path__"] = str(path)
    return cfg


def feature_spec(cfg: dict):
    """Return ordered feature IDs and a (d, 2) bounds array.

    Backward-compatible with both configuration layouts used by the inherited
    optimizer code and the growth-robustness problem compiler:

    Legacy::
        problem:
          features:
            EX_glc__D_e: [0.0, 20.0]
            EX_o2_e: [0.0, 20.0]

    Compiled robustness problem::
        problem:
          features: [EX_glc__D_e, EX_o2_e]
          bounds:
            EX_glc__D_e: [0.0, 20.0]
            EX_o2_e: [0.0, 20.0]

    The function is intentionally strict about missing/malformed bounds so a
    live benchmark cannot silently run with a different search domain.
    """
    problem = cfg.get("problem", {})
    raw_features = problem.get("features")

    if isinstance(raw_features, dict):
        features = [str(f) for f in raw_features.keys()]
        bounds = np.asarray([raw_features[f] for f in raw_features.keys()], float)
    elif isinstance(raw_features, (list, tuple)):
        features = [str(f) for f in raw_features]
        raw_bounds = problem.get("bounds")
        if isinstance(raw_bounds, dict):
            missing = [f for f in features if f not in raw_bounds]
            if missing:
                raise KeyError(f"Missing bounds for compiled problem features: {missing}")
            bounds = np.asarray([raw_bounds[f] for f in features], float)
        elif isinstance(raw_bounds, (list, tuple, np.ndarray)):
            bounds = np.asarray(raw_bounds, float)
        else:
            raise TypeError(
                "Compiled problem uses a feature list but problem.bounds is not "
                "a mapping or array-like bounds table"
            )
    else:
        raise TypeError(
            "problem.features must be either the legacy {feature: [lo, hi]} "
            "mapping or an ordered feature list"
        )

    if bounds.shape != (len(features), 2):
        raise ValueError(
            f"Feature bounds shape {bounds.shape} does not match "
            f"{len(features)} features (expected {(len(features), 2)})"
        )
    if not np.all(np.isfinite(bounds)):
        raise ValueError("Feature bounds contain non-finite values")
    if np.any(bounds[:, 1] <= bounds[:, 0]):
        bad = [features[i] for i in np.where(bounds[:, 1] <= bounds[:, 0])[0]]
        raise ValueError(f"Feature bounds require hi > lo; invalid features: {bad}")
    return features, bounds


def normalize_x(X: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    lo, hi = bounds[:, 0], bounds[:, 1]
    return (np.asarray(X, float) - lo) / (hi - lo)


def normalize_y(growth, product, cfg: dict) -> np.ndarray:
    return np.column_stack(
        [
            np.asarray(growth, float) / float(cfg["problem"]["growth_scale"]),
            np.asarray(product, float) / float(cfg["problem"]["product_scale"]),
        ]
    )


def infer_feasible(df: pd.DataFrame, growth_col="maximum_growth", product_col="product_flux") -> np.ndarray:
    g = pd.to_numeric(df[growth_col], errors="coerce").to_numpy(float)
    p = pd.to_numeric(df[product_col], errors="coerce").to_numpy(float)
    ok = np.isfinite(g) & np.isfinite(p)
    sc = detect_column(df.columns, STATUS_CANDIDATES)
    if sc is not None:
        s = df[sc].astype(str).str.lower().fillna("")
        bad = np.zeros(len(df), dtype=bool)
        for w in INFEASIBLE_WORDS:
            bad |= s.str.contains(w, regex=False).to_numpy()
        ok &= ~bad
    return ok


def _table_header(path: Path):
    try:
        if path.suffix.lower() == ".csv":
            return list(pd.read_csv(path, nrows=2).columns)
        if path.suffix.lower() in (".parquet", ".pq"):
            import pyarrow.parquet as pq

            return list(pq.ParquetFile(path).schema.names)
    except Exception:
        pass
    return []


def discover_baseline_table(cfg: dict) -> Path:
    features, _ = feature_spec(cfg)
    required = set(features + ["maximum_growth", "product_flux"])
    found = []
    for root_s in cfg.get("search", {}).get("baseline_roots", []):
        root = Path(root_s)
        if not root.exists():
            continue
        for ext in ("*.parquet", "*.pq", "*.csv"):
            for p in root.rglob(ext):
                if required.issubset(set(_table_header(p))):
                    found.append(p)
    if not found:
        raise RuntimeError("Could not discover frozen baseline archive from Step4G search.baseline_roots")
    found.sort(
        key=lambda p: (
            ("baseline" in p.name.lower()) + ("archive" in p.name.lower()),
            p.stat().st_size,
        ),
        reverse=True,
    )
    return found[0]


def load_baseline(cfg: dict) -> tuple[pd.DataFrame, Path]:
    p = discover_baseline_table(cfg)
    df = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_parquet(p)
    g = detect_column(df.columns, GROWTH_CANDIDATES)
    q = detect_column(df.columns, PRODUCT_CANDIDATES)
    if g is None or q is None:
        raise RuntimeError(f"Could not locate growth/product columns in {p}")
    ren = {}
    if g != "maximum_growth":
        ren[g] = "maximum_growth"
    if q != "product_flux":
        ren[q] = "product_flux"
    if ren:
        df = df.rename(columns=ren)
    df["maximum_growth"] = pd.to_numeric(df["maximum_growth"], errors="coerce")
    df["product_flux"] = pd.to_numeric(df["product_flux"], errors="coerce")
    df["feasible"] = infer_feasible(df)
    if "sample_id" not in df.columns:
        df = df.copy()
        df.insert(0, "sample_id", np.arange(len(df), dtype=int))
    return df.reset_index(drop=True), p


def pareto_points(Y: np.ndarray) -> np.ndarray:
    Y = np.asarray(Y, float)
    Y = Y[np.isfinite(Y).all(axis=1)]
    if len(Y) == 0:
        return np.empty((0, 2), float)
    order = np.lexsort((-Y[:, 1], -Y[:, 0]))
    s = Y[order]
    keep = []
    best_y = -np.inf
    for x, y in s:
        if y > best_y + 1e-14:
            keep.append((x, y))
            best_y = y
    return np.asarray(keep, float)


def hypervolume_2d(Y: np.ndarray, ref=(0.0, 0.0)) -> float:
    Y = np.asarray(Y, float)
    ref = np.asarray(ref, float)
    if len(Y) == 0:
        return 0.0
    Y = Y[np.isfinite(Y).all(axis=1)]
    Y = Y[np.all(Y > ref, axis=1)]
    if len(Y) == 0:
        return 0.0
    P = pareto_points(Y)
    P = P[np.argsort(P[:, 0])]
    hv, prev_x = 0.0, ref[0]
    for x, y in P:
        if x > prev_x:
            hv += (x - prev_x) * max(0.0, y - ref[1])
            prev_x = x
    return float(hv)


def candidate_hv_gain(front: np.ndarray, candidates: np.ndarray, ref=(0.0, 0.0)) -> np.ndarray:
    """Vectorized exact 2-D hypervolume gain for adding one candidate."""
    C = np.asarray(candidates, float)
    out = np.zeros(len(C), float)
    valid = np.isfinite(C).all(axis=1)
    if not np.any(valid):
        return out

    ref = np.asarray(ref, float)
    Cv = C[valid].copy()
    Cv[:, 0] = np.maximum(Cv[:, 0], ref[0])
    Cv[:, 1] = np.maximum(Cv[:, 1], ref[1])

    F = pareto_points(front)
    F = F[np.isfinite(F).all(axis=1)]
    if len(F) == 0:
        out[valid] = (
            np.maximum(Cv[:, 0] - ref[0], 0.0)
            * np.maximum(Cv[:, 1] - ref[1], 0.0)
        )
        return out

    F = F[np.argsort(F[:, 0])]
    x_end = F[:, 0]
    y_level = F[:, 1]
    x_start = np.concatenate([[ref[0]], x_end[:-1]])

    cx = Cv[:, 0][:, None]
    cy = Cv[:, 1][:, None]
    overlap = np.maximum(np.minimum(cx, x_end[None, :]) - x_start[None, :], 0.0)
    height_gain = np.maximum(cy - y_level[None, :], 0.0)
    gain = np.sum(overlap * height_gain, axis=1)
    gain += (
        np.maximum(Cv[:, 0] - x_end[-1], 0.0)
        * np.maximum(Cv[:, 1] - ref[1], 0.0)
    )
    out[valid] = np.maximum(gain, 0.0)
    return out


def deterministic_split(n: int, hidden_n: int, seed: int):
    rng = np.random.default_rng(seed)
    hidden = np.sort(rng.choice(n, size=hidden_n, replace=False))
    mask = np.ones(n, dtype=bool)
    mask[hidden] = False
    visible = np.where(mask)[0]
    return visible, hidden


def representative_subset(
    visible_df: pd.DataFrame,
    Xvisible: np.ndarray,
    Yn_visible: np.ndarray,
    n: int,
    seed: int,
) -> np.ndarray:
    """
    Mirrors the Step4C rule: all recoverable exact Pareto points first, then
    Sobol-nearest global input-space coverage.
    Returns local indices into visible_df / Xvisible.
    """
    feasible = visible_df["feasible"].to_numpy(bool) & np.isfinite(Yn_visible).all(axis=1)
    valid_ids = np.where(feasible)[0]
    front = pareto_points(Yn_visible[feasible])

    # Recover archive rows that match front objective coordinates.
    pareto_local = []
    if len(front):
        # Objective count on the front is small; vector comparison is fine.
        for fy in front:
            d = np.max(np.abs(Yn_visible[valid_ids] - fy[None, :]), axis=1)
            hits = valid_ids[d <= 1e-12]
            for h in hits:
                if int(h) not in pareto_local:
                    pareto_local.append(int(h))
                    break

    if len(pareto_local) >= n:
        return np.asarray(pareto_local[:n], dtype=int)

    tree = cKDTree(Xvisible)
    need = n - len(pareto_local)
    sampler = qmc.Sobol(d=Xvisible.shape[1], scramble=True, seed=seed)
    target_n = max(need * 3, 64)
    m = int(np.ceil(np.log2(target_n)))
    targets = sampler.random_base2(m)

    chosen = list(pareto_local)
    chosen_set = set(chosen)
    for target in targets:
        _, idxs = tree.query(target, k=min(8, len(Xvisible)))
        for idx in np.atleast_1d(idxs):
            i = int(idx)
            if i not in chosen_set:
                chosen.append(i)
                chosen_set.add(i)
                break
        if len(chosen) >= n:
            break

    if len(chosen) < n:
        for i in range(len(Xvisible)):
            if i not in chosen_set:
                chosen.append(i)
                chosen_set.add(i)
            if len(chosen) >= n:
                break

    return np.asarray(chosen[:n], dtype=int)


class GlobalSurrogate:
    """Visible-only global landscape model in normalized objective units."""

    def __init__(self, suite_cfg: dict, seed: int):
        scfg = suite_cfg["surrogate"]
        self.seed = int(seed)
        self.growth = ExtraTreesRegressor(
            n_estimators=int(scfg["regressor_trees"]),
            min_samples_leaf=int(scfg["min_samples_leaf"]),
            max_features=1.0,
            n_jobs=-1,
            random_state=seed + 1,
        )
        self.product = ExtraTreesRegressor(
            n_estimators=int(scfg["regressor_trees"]),
            min_samples_leaf=int(scfg["min_samples_leaf"]),
            max_features=1.0,
            n_jobs=-1,
            random_state=seed + 2,
        )
        self.feasibility = ExtraTreesClassifier(
            n_estimators=int(scfg["classifier_trees"]),
            min_samples_leaf=int(scfg["min_samples_leaf"]),
            max_features=1.0,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed + 3,
        )
        self.inf_col = None
        self.constant_infeasible_probability: float | None = None

    def fit(self, X: np.ndarray, Y: np.ndarray, feasible: np.ndarray):
        feasible = np.asarray(feasible, bool)
        good = feasible & np.isfinite(Y).all(axis=1)
        if good.sum() < 100:
            raise RuntimeError("Too few feasible visible rows for global surrogate")
        self.growth.fit(X[good], Y[good, 0])
        self.product.fit(X[good], Y[good, 1])

        # Cross-species guardrail: some completed landscapes (e.g. CHO and
        # C. glutamicum in the current study) are 100% feasible.  A one-class
        # sklearn classifier has a single probability column for class 0; the
        # older E. coli-only code accidentally interpreted that column as
        # P(infeasible)=1.  Treat one-class feasibility exactly instead.
        yin = (~feasible).astype(int)
        classes = np.unique(yin)
        if len(classes) == 1:
            self.constant_infeasible_probability = float(classes[0])
            self.inf_col = None
        else:
            self.constant_infeasible_probability = None
            self.feasibility.fit(X, yin)
            fitted_classes = np.asarray(self.feasibility.classes_)
            self.inf_col = int(np.where(fitted_classes == 1)[0][0])
        return self

    def predict(self, X: np.ndarray):
        X = np.atleast_2d(np.asarray(X, float))
        g = np.asarray(self.growth.predict(X), float)
        p = np.asarray(self.product.predict(X), float)
        if self.constant_infeasible_probability is not None:
            pinf = np.full(len(X), self.constant_infeasible_probability, dtype=float)
        else:
            prob = np.asarray(self.feasibility.predict_proba(X), float)
            pinf = prob[:, int(self.inf_col)]
        return np.column_stack([g, p]), pinf


def crossfit_subset_prior_predictions(
    Xvisible: np.ndarray,
    Yvisible: np.ndarray,
    feasible_visible: np.ndarray,
    subset_local: np.ndarray,
    suite_cfg: dict,
    seed: int,
) -> np.ndarray:
    """
    Honest global-prior predictions for the 1,024 GP calibration rows.
    Each subset point is predicted by a global surrogate that excluded its fold.
    """
    folds = int(suite_cfg["surrogate"].get("crossfit_folds", 5))
    kf = KFold(n_splits=folds, shuffle=True, random_state=seed + 911)
    out = np.full((len(subset_local), 2), np.nan, float)
    pos = {int(v): i for i, v in enumerate(subset_local)}

    for fold_no, (tr, va) in enumerate(kf.split(Xvisible), 1):
        targets = [int(i) for i in va if int(i) in pos]
        if not targets:
            continue
        model = GlobalSurrogate(suite_cfg, seed + 1000 * fold_no).fit(
            Xvisible[tr], Yvisible[tr], feasible_visible[tr]
        )
        pred, _ = model.predict(Xvisible[np.asarray(targets, dtype=int)])
        for j, local_idx in enumerate(targets):
            out[pos[local_idx]] = pred[j]

    if not np.isfinite(out).all():
        missing = int(np.sum(~np.isfinite(out).all(axis=1)))
        raise RuntimeError(f"Crossfit failed to produce prior predictions for {missing} subset rows")
    return out


def local_fragility_q95(
    X: np.ndarray,
    model: GlobalSurrogate,
    radius: float,
    seed: int,
    batch_centers: int = 2048,
) -> np.ndarray:
    X = np.asarray(X, float)
    n, d = X.shape
    center, _ = model.predict(X)

    axis = []
    for j in range(d):
        e = np.zeros(d)
        e[j] = 1.0
        axis.extend([e.copy(), -e.copy()])
    rng = np.random.default_rng(seed)
    rd = rng.normal(size=(6, d))
    rd /= np.linalg.norm(rd, axis=1, keepdims=True)
    rd = np.vstack([rd, -rd])
    dirs = np.vstack([np.asarray(axis), rd])

    q95 = np.full(n, np.nan)
    for start in range(0, n, batch_centers):
        stop = min(n, start + batch_centers)
        neigh = np.clip(
            X[start:stop, None, :] + float(radius) * dirs[None, :, :],
            0.0,
            1.0,
        )
        pred, _ = model.predict(neigh.reshape(-1, d))
        pred = pred.reshape(stop - start, len(dirs), 2)
        disp = np.linalg.norm(pred - center[start:stop, None, :], axis=2)
        q95[start:stop] = np.quantile(disp, 0.95, axis=1)
    return q95


def robust_scale(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, float)
    out = np.zeros_like(v, float)
    good = np.isfinite(v)
    if not np.any(good):
        return out
    lo = float(np.quantile(v[good], 0.05))
    hi = float(np.quantile(v[good], 0.95))
    if hi <= lo + 1e-15:
        out[good] = 0.5
    else:
        out[good] = np.clip((v[good] - lo) / (hi - lo), 0.0, 1.0)
    return out


def duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------- BoTorch / GP helpers ----------


def require_botorch():
    try:
        import torch
        import botorch
        import gpytorch
    except Exception as e:
        raise RuntimeError(
            "BoTorch/GPyTorch are required. Activate the existing gem-step4c environment."
        ) from e
    return torch, botorch, gpytorch


@dataclass
class BOModelBundle:
    model: object
    partitioning: object
    ref_point: list[float]


def fit_step4c_gp(
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    exact_front: np.ndarray,
    fixed_noise: float,
):
    torch, _, _ = require_botorch()
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import SingleTaskGP
    from botorch.models.transforms.outcome import Standardize
    from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
    from gpytorch.mlls import ExactMarginalLogLikelihood

    dtype = torch.double
    tx = torch.tensor(np.asarray(Xtrain, float), dtype=dtype)
    ty = torch.tensor(np.asarray(Ytrain, float), dtype=dtype)
    yvar = torch.full_like(ty, float(fixed_noise))
    model = SingleTaskGP(
        tx,
        ty,
        train_Yvar=yvar,
        outcome_transform=Standardize(m=2),
    )
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    ref = [0.0, 0.0]
    part = FastNondominatedPartitioning(
        ref_point=torch.tensor(ref, dtype=dtype),
        Y=torch.tensor(pareto_points(exact_front), dtype=dtype),
    )
    model.eval()
    return BOModelBundle(model=model, partitioning=part, ref_point=ref)


class _PriorMeanFactory:
    """Namespace to avoid importing torch/gpytorch at module import time."""

    @staticmethod
    def make(global_model: GlobalSurrogate, objective: int, override: dict[tuple[float, ...], float]):
        torch, _, gpytorch = require_botorch()
        Mean = gpytorch.means.Mean

        class SklearnPriorMean(Mean):
            def __init__(self):
                super().__init__()

            def forward(self, x):
                arr = x.detach().cpu().numpy()
                shape = arr.shape[:-1]
                flat = arr.reshape(-1, arr.shape[-1])
                pred, _ = global_model.predict(flat)
                vals = pred[:, objective].copy()
                if override:
                    for i, row in enumerate(flat):
                        key = tuple(np.round(row, 12).tolist())
                        if key in override:
                            vals[i] = override[key]
                return torch.as_tensor(vals, dtype=x.dtype, device=x.device).reshape(shape)

        return SklearnPriorMean()


def fit_prior_gp(
    global_model: GlobalSurrogate,
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    feasible_train: np.ndarray,
    exact_front: np.ndarray,
    fixed_noise: float,
    override_predictions: dict[tuple[float, ...], tuple[float, float]],
):
    torch, _, _ = require_botorch()
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

    good = np.asarray(feasible_train, bool) & np.isfinite(Ytrain).all(axis=1)
    if good.sum() < 16:
        raise RuntimeError("Too few feasible exact observations for prior GP")

    tx = torch.tensor(np.asarray(Xtrain[good], float), dtype=torch.double)
    yg = torch.tensor(np.asarray(Ytrain[good][:, [0]], float), dtype=torch.double)
    yp = torch.tensor(np.asarray(Ytrain[good][:, [1]], float), dtype=torch.double)

    over_g = {k: float(v[0]) for k, v in override_predictions.items()}
    over_p = {k: float(v[1]) for k, v in override_predictions.items()}

    mg = _PriorMeanFactory.make(global_model, 0, over_g)
    mp = _PriorMeanFactory.make(global_model, 1, over_p)

    gp_g = SingleTaskGP(
        tx,
        yg,
        train_Yvar=torch.full_like(yg, float(fixed_noise)),
        mean_module=mg,
        outcome_transform=None,
    )
    gp_p = SingleTaskGP(
        tx,
        yp,
        train_Yvar=torch.full_like(yp, float(fixed_noise)),
        mean_module=mp,
        outcome_transform=None,
    )
    model = ModelListGP(gp_g, gp_p)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()

    ref = [0.0, 0.0]
    part = FastNondominatedPartitioning(
        ref_point=torch.tensor(ref, dtype=torch.double),
        Y=torch.tensor(pareto_points(exact_front), dtype=torch.double),
    )
    return BOModelBundle(model=model, partitioning=part, ref_point=ref)


def qlogehvi_acquisition(bundle: BOModelBundle, seed: int, mc_samples: int):
    torch, _, _ = require_botorch()
    from botorch.acquisition.multi_objective.logei import qLogExpectedHypervolumeImprovement
    from botorch.sampling.normal import SobolQMCNormalSampler

    sampler = SobolQMCNormalSampler(torch.Size([int(mc_samples)]), seed=int(seed))
    return qLogExpectedHypervolumeImprovement(
        model=bundle.model,
        ref_point=bundle.ref_point,
        partitioning=bundle.partitioning,
        sampler=sampler,
    )


def discrete_qlogehvi_select(
    bundle: BOModelBundle,
    choices: np.ndarray,
    q: int,
    seed: int,
    mc_samples: int,
    max_batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    torch, _, _ = require_botorch()
    from botorch.optim.optimize import optimize_acqf_discrete

    torch.manual_seed(int(seed))
    acq = qlogehvi_acquisition(bundle, seed=seed, mc_samples=mc_samples)
    tc = torch.tensor(np.asarray(choices, float), dtype=torch.double)
    # BoTorch API compatibility:
    # v0.17.1 and older do not expose `return_acq_values` on
    # optimize_acqf_discrete, while newer releases do. Both return
    # (candidates, acquisition_values) by default.
    import inspect

    kwargs = dict(
        acq_function=acq,
        q=int(q),
        choices=tc,
        max_batch_size=int(max_batch_size),
        unique=True,
    )
    if "return_acq_values" in inspect.signature(optimize_acqf_discrete).parameters:
        kwargs["return_acq_values"] = True

    cand, vals = optimize_acqf_discrete(**kwargs)
    if vals is None:
        # Defensive fallback for a future API variant. Candidate selection is
        # still valid; compute acquisition values directly if possible.
        with torch.no_grad():
            vals = acq(cand.unsqueeze(0) if cand.ndim == 2 else cand)
    return cand.detach().cpu().numpy(), np.atleast_1d(vals.detach().cpu().numpy())



# ---------- gsMOBO / qParEGO literature-comparator helpers ----------


def fit_gsmobo_gp(Xtrain: np.ndarray, Ytrain: np.ndarray):
    """Fit the independent-output GP model used by the gsMOBO/qParEGO comparator.

    This intentionally does *not* use the frozen 50k surrogate. Inputs are the
    method's own normalized live designs and exact normalized objective values.
    Infeasible values should be encoded by the caller before fitting. The model
    structure mirrors the published gsMOBO implementation: one ``SingleTaskGP``
    per objective combined as a ``ModelListGP``, with observation noise inferred
    by the likelihood rather than fixed from the landscape model.
    """
    torch, _, _ = require_botorch()
    from botorch.fit import fit_gpytorch_mll
    from botorch.models import ModelListGP, SingleTaskGP
    from gpytorch.mlls.sum_marginal_log_likelihood import SumMarginalLogLikelihood

    X = np.asarray(Xtrain, float)
    Y = np.asarray(Ytrain, float)
    if X.ndim != 2 or Y.ndim != 2 or len(X) != len(Y) or Y.shape[1] != 2:
        raise ValueError(f"gsMOBO GP expects X=(n,d), Y=(n,2); got {X.shape}, {Y.shape}")
    if len(X) < 2:
        raise RuntimeError("gsMOBO GP requires at least two live observations")
    if not np.isfinite(X).all() or not np.isfinite(Y).all():
        raise RuntimeError("gsMOBO GP training arrays must be finite")

    tx = torch.tensor(X, dtype=torch.double)
    models = []
    for j in range(2):
        ty = torch.tensor(Y[:, [j]], dtype=torch.double)
        models.append(SingleTaskGP(tx, ty))
    model = ModelListGP(*models)
    mll = SumMarginalLogLikelihood(model.likelihood, model)
    fit_gpytorch_mll(mll)
    model.eval()
    return model



def continuous_qparego_select(
    model,
    Xtrain: np.ndarray,
    Ytrain: np.ndarray,
    q: int,
    seed: int,
    mc_samples: int = 256,
    num_restarts: int = 5,
    raw_samples: int = 512,
    maxiter: int = 200,
):
    """Select a continuous batch with qParEGO + qLogExpectedImprovement.

    The implementation follows the published gsMOBO mechanism: for each member
    of the requested batch, sample a new simplex weight vector, construct an
    augmented Chebyshev scalarization, wrap it in ``qLogExpectedImprovement``,
    and jointly/sequentially optimize the list of acquisition functions over the
    normalized [0,1]^d medium-design box.

    Unlike the frozen-landscape methods, this function receives no archive,
    archive Pareto front, feasibility classifier, or global surrogate.

    Notes on the matched-budget adaptation:
    - the benchmark has exactly two maximization objectives (growth, product);
    - both objectives are already frozen-normalized by GEM Engine;
    - ``best_f`` is evaluated under the same Chebyshev scalarization as the
      acquisition objective (canonical qParEGO/BoTorch behavior);
    - no medium-cost objective is added because the benchmark being compared is
      the existing two-objective growth/product problem.
    """
    torch, _, _ = require_botorch()
    from botorch.acquisition.logei import qLogExpectedImprovement
    from botorch.acquisition.objective import GenericMCObjective
    from botorch.optim.optimize import optimize_acqf_list
    from botorch.sampling.normal import SobolQMCNormalSampler
    from botorch.utils.multi_objective.scalarization import get_chebyshev_scalarization
    from botorch.utils.sampling import sample_simplex

    X = np.asarray(Xtrain, float)
    Y = np.asarray(Ytrain, float)
    q = int(q)
    if q < 1:
        return np.empty((0, X.shape[1]), float), np.empty(0, float), np.empty((0, 2), float)
    if X.ndim != 2 or Y.shape != (len(X), 2):
        raise ValueError(f"qParEGO expects X=(n,d), Y=(n,2); got {X.shape}, {Y.shape}")

    dtype = torch.double
    tx = torch.tensor(X, dtype=dtype)
    ty = torch.tensor(Y, dtype=dtype)
    model.eval()
    torch.manual_seed(int(seed))

    # The released gsMOBO code constructs its Chebyshev transform from posterior
    # means at the observed designs. Preserve that convention while computing
    # best_f on the same scalarized scale for numerical consistency.
    with torch.no_grad():
        scalarization_reference = model.posterior(tx).mean

    sampler = SobolQMCNormalSampler(torch.Size([int(mc_samples)]), seed=int(seed) + 1)
    acq_fun_list = []
    weights_out = []

    for i in range(q):
        wseed = int(seed) + 101 + i
        # BoTorch's sample_simplex signature has changed slightly over releases.
        try:
            weights = sample_simplex(
                d=2, n=1, qmc=False, seed=wseed, device=tx.device, dtype=dtype
            ).squeeze(0)
        except TypeError:
            torch.manual_seed(wseed)
            weights = sample_simplex(2, device=tx.device, dtype=dtype).squeeze()
        objective = GenericMCObjective(
            get_chebyshev_scalarization(weights=weights, Y=scalarization_reference)
        )
        with torch.no_grad():
            best_f = objective(ty).max()
        acq_fun_list.append(
            qLogExpectedImprovement(
                model=model,
                best_f=best_f,
                sampler=sampler,
                objective=objective,
            )
        )
        weights_out.append(weights.detach().cpu().numpy())

    d = X.shape[1]
    bounds = torch.stack([torch.zeros(d, dtype=dtype), torch.ones(d, dtype=dtype)])
    candidates, acq_values = optimize_acqf_list(
        acq_function_list=acq_fun_list,
        bounds=bounds,
        num_restarts=int(num_restarts),
        raw_samples=int(raw_samples),
        options={"batch_limit": 10, "maxiter": int(maxiter)},
    )

    cand = candidates.detach().cpu().numpy()
    vals = (
        np.asarray(acq_values.detach().cpu().numpy(), float).reshape(-1)
        if acq_values is not None
        else np.full(len(cand), np.nan, float)
    )
    weights_np = np.asarray(weights_out, float).reshape(q, 2)
    return cand, vals, weights_np

def posterior_mean_std(model, X: np.ndarray, batch: int = 1024):
    torch, _, _ = require_botorch()
    means, stds = [], []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(X), batch):
            tx = torch.tensor(np.asarray(X[s : s + batch], float), dtype=torch.double)
            post = model.posterior(tx)
            mean = post.mean.detach().cpu().numpy()
            var = post.variance.detach().cpu().numpy()
            mean = np.asarray(mean).reshape(len(tx), -1)
            var = np.asarray(var).reshape(len(tx), -1)
            means.append(mean[:, :2])
            stds.append(np.sqrt(np.maximum(var[:, :2], 0.0)))
    return np.vstack(means), np.vstack(stds)
