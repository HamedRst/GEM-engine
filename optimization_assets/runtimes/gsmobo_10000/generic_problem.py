from __future__ import annotations

"""Generic, frozen-before-live GEM optimization problem compiler.

The optimizer runtime never hard-codes biological objective names.  A
``gemproblem/v0.2`` YAML file defines objective values, directions,
normalization, and constraints.  The compiler turns a frozen archive or live
exact rows into an explicit objective matrix whose internal coordinates are all
maximized.

Supported objective/constraint sources
--------------------------------------
``exact_metric``
    Read a named column from the exact backend/archive.
``design_metric``
    Deterministic function of normalized design variables.  Built-ins:
    ``normalized_mean``, ``normalized_sum``, ``normalized_l2``,
    ``normalized_max``, ``normalized_min``.
``design_feature``
    Deterministic transform of one normalized design variable. The feature may
    be named directly or resolved from a source-problem metadata key, allowing
    one problem definition to work across species with different reaction IDs.
``expression``
    Safe arithmetic expression using exact columns by identifier and helper
    functions ``metric('name')``, ``x('feature')``, ``x01('feature')``,
    ``mean_x01()``, ``sum_x01()``, ``min_x01()``, ``max_x01()``.

Frozen-reference constraints support max/min/mean/median/qXX statistics and are
materialized into the prepared problem config before live search.  Live rows
can never update these thresholds.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
import ast
import hashlib
import json
import math
import re

import numpy as np
import pandas as pd
import yaml

FORMAT = "gemproblem/v0.2"


def load_yaml(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def problem_hash(spec: dict) -> str:
    return hashlib.sha256(canonical_json(spec).encode("utf-8")).hexdigest()


def slugify(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_]+", "_", str(name)).strip("_")
    if not s:
        s = "field"
    if s[0].isdigit():
        s = "v_" + s
    return s.lower()


def as_bool(series: pd.Series) -> np.ndarray:
    if pd.api.types.is_bool_dtype(series):
        return series.to_numpy(bool)
    s = series.astype(str).str.strip().str.lower()
    return s.isin(["true", "1", "yes", "y", "optimal", "feasible"]).to_numpy(bool)


def feature_bounds(problem: dict) -> tuple[list[str], np.ndarray]:
    raw = problem.get("features")
    br = problem.get("bounds")
    if isinstance(raw, dict):
        features = list(map(str, raw.keys()))
        br = raw if br is None else br
    elif isinstance(raw, (list, tuple)):
        features = list(map(str, raw))
    else:
        raise RuntimeError("problem.features must be a list or mapping")
    if not isinstance(br, dict):
        raise RuntimeError("problem.bounds must be a mapping")
    vals = []
    for f in features:
        if f not in br:
            raise RuntimeError(f"problem.bounds missing feature {f!r}")
        b = br[f]
        if isinstance(b, dict):
            lo, hi = float(b["min"]), float(b["max"])
        else:
            lo, hi = float(b[0]), float(b[1])
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise RuntimeError(f"Invalid bounds for {f}: {b}")
        vals.append([lo, hi])
    return features, np.asarray(vals, float)


def normalize_design(df: pd.DataFrame, features: list[str], bounds: np.ndarray) -> np.ndarray:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise RuntimeError(f"Rows are missing design variables: {missing}")
    X = df[features].to_numpy(float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return (X - lo) / (hi - lo)


def source_feasible(df: pd.DataFrame) -> np.ndarray:
    if "source_problem_feasible" in df.columns:
        return as_bool(df["source_problem_feasible"])
    if "feasible" in df.columns:
        return as_bool(df["feasible"])
    status_candidates = [c for c in df.columns if str(c).lower() in {"status", "solver_status", "optimization_status"}]
    if status_candidates:
        s = df[status_candidates[0]].astype(str).str.lower()
        bad = s.str.contains("infeasible|unbounded|failed|error|invalid", regex=True)
        return (~bad).to_numpy(bool)
    return np.ones(len(df), dtype=bool)


# ------------------------------ safe expressions ------------------------------

_ALLOWED_BINOPS = {
    ast.Add: np.add,
    ast.Sub: np.subtract,
    ast.Mult: np.multiply,
    ast.Div: np.divide,
    ast.Pow: np.power,
    ast.Mod: np.mod,
}
_ALLOWED_UNARY = {ast.UAdd: lambda x: x, ast.USub: np.negative}
_ALLOWED_FUNCS = {
    "abs": np.abs,
    "sqrt": np.sqrt,
    "log": np.log,
    "log1p": np.log1p,
    "exp": np.exp,
    "minimum": np.minimum,
    "maximum": np.maximum,
    "clip": np.clip,
}


class _ExpressionEvaluator:
    def __init__(self, df: pd.DataFrame, proposal: pd.DataFrame, features: list[str], bounds: np.ndarray):
        self.df = df
        self.proposal = proposal
        self.features = features
        self.bounds = bounds
        self.X01 = normalize_design(proposal, features, bounds)
        self._name_cols = {slugify(str(c)): str(c) for c in df.columns}

    def metric(self, name: str) -> np.ndarray:
        if name not in self.df.columns:
            raise KeyError(f"Exact metric {name!r} not found; available={list(self.df.columns)[:120]}")
        return pd.to_numeric(self.df[name], errors="coerce").to_numpy(float)

    def x(self, feature: str) -> np.ndarray:
        if feature not in self.features:
            raise KeyError(f"Unknown feature {feature!r}")
        return pd.to_numeric(self.proposal[feature], errors="coerce").to_numpy(float)

    def x01(self, feature: str) -> np.ndarray:
        if feature not in self.features:
            raise KeyError(f"Unknown feature {feature!r}")
        return self.X01[:, self.features.index(feature)]

    def _eval(self, node: ast.AST):
        if isinstance(node, ast.Expression):
            return self._eval(node.body)
        if isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float, str)):
                return node.value
            raise ValueError("Unsupported literal in expression")
        if isinstance(node, ast.Name):
            key = node.id
            if key in {"pi", "e"}:
                return math.pi if key == "pi" else math.e
            if key in self.df.columns:
                return self.metric(key)
            if key in self._name_cols:
                return self.metric(self._name_cols[key])
            raise NameError(f"Unknown expression name {key!r}")
        if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BINOPS:
            return _ALLOWED_BINOPS[type(node.op)](self._eval(node.left), self._eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY:
            return _ALLOWED_UNARY[type(node.op)](self._eval(node.operand))
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only simple function calls are allowed")
            fn = node.func.id
            args = [self._eval(a) for a in node.args]
            if fn in _ALLOWED_FUNCS:
                return _ALLOWED_FUNCS[fn](*args)
            if fn == "metric":
                if len(args) != 1 or not isinstance(args[0], str):
                    raise ValueError("metric() requires one string argument")
                return self.metric(args[0])
            if fn == "x":
                if len(args) != 1 or not isinstance(args[0], str):
                    raise ValueError("x() requires one string argument")
                return self.x(args[0])
            if fn == "x01":
                if len(args) != 1 or not isinstance(args[0], str):
                    raise ValueError("x01() requires one string argument")
                return self.x01(args[0])
            if fn == "mean_x01":
                return np.mean(self.X01, axis=1)
            if fn == "sum_x01":
                return np.sum(self.X01, axis=1)
            if fn == "min_x01":
                return np.min(self.X01, axis=1)
            if fn == "max_x01":
                return np.max(self.X01, axis=1)
            raise NameError(f"Function {fn!r} is not allowed")
        raise ValueError(f"Unsupported expression syntax: {ast.dump(node, include_attributes=False)}")

    def evaluate(self, expr: str) -> np.ndarray:
        tree = ast.parse(str(expr), mode="eval")
        value = self._eval(tree)
        arr = np.asarray(value, float)
        if arr.ndim == 0:
            arr = np.full(len(self.df), float(arr))
        if arr.shape != (len(self.df),):
            raise RuntimeError(f"Expression {expr!r} returned shape {arr.shape}, expected {(len(self.df),)}")
        return arr


def _design_metric(name: str, X01: np.ndarray) -> np.ndarray:
    n = str(name).lower()
    if n in {"normalized_mean", "mean", "mean_x01", "normalized_resource_cost"}:
        return np.mean(X01, axis=1)
    if n in {"normalized_sum", "sum", "sum_x01"}:
        return np.sum(X01, axis=1)
    if n in {"normalized_l2", "l2"}:
        return np.sqrt(np.sum(np.square(X01), axis=1))
    if n in {"normalized_max", "max", "max_x01"}:
        return np.max(X01, axis=1)
    if n in {"normalized_min", "min", "min_x01"}:
        return np.min(X01, axis=1)
    raise KeyError(f"Unsupported design_metric {name!r}")


def resolve_quantity_definition(defn: dict, source_problem: dict, features: list[str]) -> dict:
    """Resolve source-metadata feature aliases without species branches.

    A ``design_feature`` quantity can declare either ``feature`` directly or
    ``feature_from_source_problem`` (for example ``carbon_feature``). The latter
    is resolved once while compiling the frozen problem and materialized into the
    compiled contract, so live optimization never needs organism-specific code.
    """
    out = dict(defn)
    source = str(out.get("source", "exact_metric")).lower()
    if source == "design_feature":
        feature = out.get("feature")
        meta_key = out.get("feature_from_source_problem")
        if feature is None and meta_key:
            feature = source_problem.get(str(meta_key))
        if not str(feature or "").strip():
            raise RuntimeError(f"design_feature {out.get('name')!r} could not resolve a feature")
        feature = str(feature)
        if feature not in features:
            raise RuntimeError(f"design_feature {out.get('name')!r} resolved {feature!r}, not in problem.features")
        out["resolved_feature"] = feature
        out["resolved_feature_index"] = int(features.index(feature))
        out["deterministic_from_design"] = True
    elif source == "design_metric":
        out["deterministic_from_design"] = True
    else:
        out["deterministic_from_design"] = bool(out.get("deterministic_from_design", False))
    return out


def _design_feature(defn: dict, X01: np.ndarray, features: list[str]) -> np.ndarray:
    feature = str(defn.get("resolved_feature") or defn.get("feature") or "")
    if feature not in features:
        raise KeyError(f"Resolved design feature {feature!r} not present")
    z = np.asarray(X01[:, features.index(feature)], float)
    transform = str(defn.get("transform", "normalized")).lower()
    if transform in {"normalized", "identity_normalized", "demand"}:
        return z
    if transform in {"one_minus_normalized", "efficiency", "one_minus_demand"}:
        return 1.0 - z
    raise KeyError(f"Unsupported design_feature transform {transform!r}")


def evaluate_quantity(defn: dict, df: pd.DataFrame, proposal: pd.DataFrame, features: list[str], bounds: np.ndarray) -> np.ndarray:
    source = str(defn.get("source", "exact_metric")).lower()
    ev = _ExpressionEvaluator(df, proposal, features, bounds)
    if source == "exact_metric":
        metric = defn.get("metric") or defn.get("name")
        if not metric:
            raise RuntimeError(f"exact_metric definition needs metric: {defn}")
        return ev.metric(str(metric))
    if source == "design_metric":
        metric = defn.get("metric") or defn.get("name")
        return _design_metric(str(metric), ev.X01)
    if source == "design_feature":
        return _design_feature(defn, ev.X01, features)
    if source == "expression":
        if "expression" not in defn:
            raise RuntimeError(f"expression definition needs expression: {defn}")
        return ev.evaluate(str(defn["expression"]))
    raise RuntimeError(f"Unsupported quantity source {source!r}")


# ------------------------------ schema / stats ------------------------------


def validate_problem_spec(spec: dict) -> dict:
    if spec.get("format") != FORMAT:
        raise RuntimeError(f"Problem format must be {FORMAT!r}, got {spec.get('format')!r}")
    p = spec.get("problem")
    if not isinstance(p, dict):
        raise RuntimeError("Problem YAML needs a mapping at problem:")
    pid = str(p.get("id", "")).strip()
    if not pid:
        raise RuntimeError("problem.id is required")
    objectives = p.get("objectives")
    if not isinstance(objectives, list) or len(objectives) < 1:
        raise RuntimeError("problem.objectives must contain at least one objective")
    names = []
    for i, o in enumerate(objectives):
        if not isinstance(o, dict):
            raise RuntimeError(f"objective {i} must be a mapping")
        name = str(o.get("name", "")).strip()
        if not name:
            raise RuntimeError(f"objective {i} is missing name")
        if name in names:
            raise RuntimeError(f"duplicate objective name {name!r}")
        names.append(name)
        direction = str(o.get("direction", "maximize")).lower()
        if direction not in {"maximize", "minimize"}:
            raise RuntimeError(f"objective {name}: direction must be maximize/minimize")
        source = str(o.get("source", "exact_metric")).lower()
        if source not in {"exact_metric", "design_metric", "design_feature", "expression"}:
            raise RuntimeError(f"objective {name}: unsupported source {source}")
    constraints = p.get("constraints", []) or []
    if not isinstance(constraints, list):
        raise RuntimeError("problem.constraints must be a list")
    for i, c in enumerate(constraints):
        if not isinstance(c, dict):
            raise RuntimeError(f"constraint {i} must be a mapping")
        op = str(c.get("operator", ">=")).strip()
        if op not in {">=", ">", "<=", "<", "==", "!="}:
            raise RuntimeError(f"constraint {i}: unsupported operator {op!r}")
        if "value" not in c and "reference" not in c:
            raise RuntimeError(f"constraint {i}: needs value or reference")
        if "reference" in c:
            r = c["reference"]
            if not isinstance(r, dict) or str(r.get("source", "")) != "frozen_landscape":
                raise RuntimeError(f"constraint {i}: references must use source: frozen_landscape")
            if c.get("freeze_before_live_search", True) is not True:
                raise RuntimeError(f"constraint {i}: frozen references must freeze_before_live_search=true")
    return spec


def _statistic(x: np.ndarray, name: str) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise RuntimeError("Cannot compute frozen statistic from zero finite values")
    s = str(name).lower()
    if s == "max": return float(np.max(x))
    if s == "min": return float(np.min(x))
    if s == "mean": return float(np.mean(x))
    if s == "median": return float(np.median(x))
    m = re.fullmatch(r"q(?:0?\.?)?(\d+(?:\.\d+)?)", s)
    if m:
        q = float(m.group(1))
        if q > 1: q /= 100.0
        if not 0 <= q <= 1: raise RuntimeError(f"Invalid quantile {name!r}")
        return float(np.quantile(x, q))
    raise RuntimeError(f"Unsupported frozen statistic {name!r}")


def _constraint_mask(values: np.ndarray, op: str, threshold: float, atol: float = 1e-12) -> np.ndarray:
    x = np.asarray(values, float)
    finite = np.isfinite(x)
    if op == ">=": return finite & (x >= threshold - atol)
    if op == ">": return finite & (x > threshold)
    if op == "<=": return finite & (x <= threshold + atol)
    if op == "<": return finite & (x < threshold)
    if op == "==": return finite & np.isclose(x, threshold, rtol=0, atol=atol)
    if op == "!=": return finite & ~np.isclose(x, threshold, rtol=0, atol=atol)
    raise RuntimeError(op)


def _fit_normalization(values: np.ndarray, objective: dict) -> dict:
    x = np.asarray(values, float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        raise RuntimeError(f"Objective {objective['name']}: zero finite values for normalization")
    n = objective.get("normalization", {}) or {}
    if isinstance(n, str): n = {"method": n}
    method = str(n.get("method", "frozen_minmax")).lower()
    if method == "frozen_minmax":
        lo, hi = float(np.min(x)), float(np.max(x))
    elif method == "zero_to_frozen_max":
        lo, hi = 0.0, float(np.max(x))
    elif method == "fixed_range":
        lo, hi = float(n["min"]), float(n["max"])
    elif method == "none":
        return {"method": "none"}
    else:
        raise RuntimeError(f"Objective {objective['name']}: unsupported normalization {method!r}")
    if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
        # Constant objectives are legal but useless for dominance.  Give them a
        # harmless unit span so the internal coordinate is finite/constant.
        if np.isfinite(lo) and np.isfinite(hi) and abs(hi - lo) <= 1e-15:
            hi = lo + 1.0
        else:
            raise RuntimeError(f"Objective {objective['name']}: invalid normalization range {lo}, {hi}")
    return {"method": method, "min": lo, "max": hi}


def normalize_objective(values: np.ndarray, direction: str, norm: dict) -> np.ndarray:
    x = np.asarray(values, float)
    method = norm.get("method", "frozen_minmax")
    if method == "none":
        z = x.copy()
        return z if direction == "maximize" else -z
    lo, hi = float(norm["min"]), float(norm["max"])
    z = (x - lo) / (hi - lo)
    return z if direction == "maximize" else 1.0 - z


def design_objectives_normalized_from_x01(X01: np.ndarray, objectives: list[dict], features: list[str]) -> np.ndarray:
    """Evaluate deterministic objective channels directly from normalized X.

    Non-deterministic columns are returned as NaN. This helper is shared by the
    global surrogate and qParEGO acquisition objective so deterministic problem
    channels are never approximated by a GP merely for optimizer convenience.
    """
    X01 = np.atleast_2d(np.asarray(X01, float))
    out = np.full((len(X01), len(objectives)), np.nan, float)
    dummy = pd.DataFrame({f: X01[:,j] for j,f in enumerate(features)})
    # evaluate_quantity expects physical proposal for normalizing design. Instead
    # deterministic built-ins operate on X01, so evaluate them directly here.
    for j,o in enumerate(objectives):
        if not bool(o.get("deterministic_from_design", False)):
            continue
        source = str(o.get("source", "")).lower()
        if source == "design_metric":
            raw = _design_metric(str(o.get("metric") or o.get("name")), X01)
        elif source == "design_feature":
            raw = _design_feature(o, X01, features)
        else:
            raise RuntimeError(f"Deterministic objective source {source!r} is not supported by direct X evaluation")
        out[:,j] = normalize_objective(raw, str(o.get("direction", "maximize")), dict(o["resolved_normalization"]))
    return out


@dataclass
class CompiledProblem:
    spec: dict
    hash: str
    objectives: list[dict]
    constraints: list[dict]
    objective_names: list[str]
    objective_directions: list[str]
    objective_raw_columns: list[str]
    objective_norm_columns: list[str]
    hv_reference_normalized: list[float]

    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "problem_hash": self.hash,
            "objectives": self.objectives,
            "constraints": self.constraints,
            "objective_names": self.objective_names,
            "objective_directions": self.objective_directions,
            "objective_raw_columns": self.objective_raw_columns,
            "objective_norm_columns": self.objective_norm_columns,
            "objective_dim": len(self.objectives),
            "stochastic_objective_indices": [i for i,o in enumerate(self.objectives) if not bool(o.get("deterministic_from_design", False))],
            "deterministic_objective_indices": [i for i,o in enumerate(self.objectives) if bool(o.get("deterministic_from_design", False))],
            "hv_reference_normalized": self.hv_reference_normalized,
        }


def compile_frozen_archive(source_df: pd.DataFrame, source_problem: dict, spec: dict) -> tuple[pd.DataFrame, dict]:
    spec = validate_problem_spec(spec)
    psrc = source_problem.get("problem", source_problem)
    features, bounds = feature_bounds(psrc)
    proposal = source_df[features].copy()
    base_ok = source_feasible(source_df)
    out = source_df.copy().reset_index(drop=True)
    proposal = proposal.reset_index(drop=True)

    # Resolve all frozen constraint thresholds first using source archive values.
    compiled_constraints = []
    constraint_values = []
    constraint_masks = []
    for i, c0 in enumerate(spec["problem"].get("constraints", []) or []):
        c = resolve_quantity_definition(dict(c0), psrc, features)
        name = str(c.get("name", f"constraint_{i+1}"))
        slug = slugify(name)
        values = evaluate_quantity(c, out, proposal, features, bounds)
        if "reference" in c:
            r = dict(c["reference"])
            # A reference may use the same quantity as the constraint or define
            # its own metric/expression.
            ref_def = dict(c)
            if "metric" in r or "expression" in r or "source_type" in r:
                ref_def.update({k: v for k, v in r.items() if k in {"metric", "expression"}})
                if "source_type" in r: ref_def["source"] = r["source_type"]
            ref_values = evaluate_quantity(ref_def, out, proposal, features, bounds)
            ref_values = ref_values[base_ok & np.isfinite(ref_values)]
            stat = _statistic(ref_values, str(r.get("statistic", "max")))
            threshold = stat * float(r.get("multiplier", 1.0)) + float(r.get("offset", 0.0))
            frozen_reference = {
                "source": "frozen_landscape",
                "statistic": str(r.get("statistic", "max")),
                "statistic_value": float(stat),
                "multiplier": float(r.get("multiplier", 1.0)),
                "offset": float(r.get("offset", 0.0)),
            }
        else:
            threshold = float(c["value"])
            frozen_reference = {"source": "constant", "value": threshold}
        mask = _constraint_mask(values, str(c.get("operator", ">=")), threshold, float(c.get("atol", 1e-12)))
        out[f"constraint_value__{slug}"] = values
        out[f"constraint_pass__{slug}"] = mask
        c.update({
            "name": name,
            "slug": slug,
            "resolved_threshold": float(threshold),
            "frozen_reference": frozen_reference,
            "freeze_before_live_search": True,
            "live_discoveries_may_update_threshold": False,
        })
        compiled_constraints.append(c)
        constraint_values.append(values)
        constraint_masks.append(mask)

    feasible = base_ok.copy()
    for mask in constraint_masks:
        feasible &= mask

    # Evaluate raw objectives, then fit normalization only on rows valid for the
    # selected problem.  All statistics are frozen before live search.
    objectives = []
    raw_values = []
    objective_names = []
    objective_directions = []
    objective_raw_columns = []
    objective_norm_columns = []
    for i, o0 in enumerate(spec["problem"]["objectives"]):
        o = resolve_quantity_definition(dict(o0), psrc, features)
        name = str(o["name"]); slug = slugify(name)
        direction = str(o.get("direction", "maximize")).lower()
        raw = evaluate_quantity(o, out, proposal, features, bounds)
        fit_mask = feasible & np.isfinite(raw)
        if not np.any(fit_mask):
            raise RuntimeError(f"Objective {name!r}: no finite problem-feasible frozen values")
        norm = _fit_normalization(raw[fit_mask], o)
        z = normalize_objective(raw, direction, norm)
        raw_col = f"objective_raw__{slug}"
        norm_col = f"objective_norm__{slug}"
        out[raw_col] = raw
        out[norm_col] = z
        o.update({"slug": slug, "resolved_normalization": norm, "raw_column": raw_col, "norm_column": norm_col})
        objectives.append(o); raw_values.append(raw)
        objective_names.append(name); objective_directions.append(direction)
        objective_raw_columns.append(raw_col); objective_norm_columns.append(norm_col)

    finite_objectives = np.isfinite(out[objective_norm_columns].to_numpy(float)).all(axis=1)
    feasible &= finite_objectives
    out["source_problem_feasible"] = base_ok
    out["feasible"] = feasible
    failed_constraints = np.zeros(len(out), dtype=object)
    failed_constraints[:] = ""
    if compiled_constraints:
        for c in compiled_constraints:
            col = f"constraint_pass__{c['slug']}"
            bad = ~as_bool(out[col])
            failed_constraints[bad] = np.where(
                failed_constraints[bad] == "", c["name"], failed_constraints[bad] + ";" + c["name"]
            )
    status = np.where(~base_ok, "source_infeasible", np.where(feasible, "optimal_generic_problem", "generic_constraint_or_objective_invalid"))
    out["status"] = status
    out["failed_constraints"] = failed_constraints
    if "design_index" not in out.columns: out["design_index"] = np.arange(len(out), dtype=int)
    if "sample_id" not in out.columns: out["sample_id"] = out["design_index"].to_numpy()

    h = problem_hash(spec)
    hv_ref = spec["problem"].get("hv_reference_normalized")
    if hv_ref is None:
        hv_ref = [0.0] * len(objectives)
    hv_ref = [float(x) for x in hv_ref]
    if len(hv_ref) != len(objectives):
        raise RuntimeError(f"hv_reference_normalized length {len(hv_ref)} != objective count {len(objectives)}")

    compiled = CompiledProblem(
        spec=spec, hash=h, objectives=objectives, constraints=compiled_constraints,
        objective_names=objective_names, objective_directions=objective_directions,
        objective_raw_columns=objective_raw_columns, objective_norm_columns=objective_norm_columns,
        hv_reference_normalized=hv_ref,
    )
    meta = compiled.to_dict()
    meta.update({
        "problem_id": str(spec["problem"]["id"]),
        "problem_title": str(spec["problem"].get("title", spec["problem"]["id"])),
        "features": features,
        "bounds": bounds.tolist(),
        "rows": int(len(out)),
        "source_feasible_rows": int(base_ok.sum()),
        "problem_feasible_rows": int(feasible.sum()),
        "archive_relabel_fba_solves": 0,
    })
    return out, meta


def compile_live_exact(raw: pd.DataFrame, proposal: pd.DataFrame, prepared_problem: dict, spec: dict) -> pd.DataFrame:
    spec = validate_problem_spec(spec)
    p = prepared_problem.get("problem", prepared_problem)
    compiled = p.get("compiled_problem")
    if not isinstance(compiled, dict):
        raise RuntimeError("Prepared problem is missing compiled_problem")
    expected_hash = str(compiled.get("problem_hash", ""))
    actual_hash = problem_hash(spec)
    if expected_hash != actual_hash:
        raise RuntimeError(f"Problem hash mismatch: prepared={expected_hash} active={actual_hash}")

    features, bounds = feature_bounds(p)
    if len(raw) != len(proposal):
        raise RuntimeError(f"Exact/proposal row mismatch {len(raw)} != {len(proposal)}")
    raw = raw.reset_index(drop=True).copy(); proposal = proposal.reset_index(drop=True).copy()
    base_ok = source_feasible(raw)
    out = raw.copy()

    feasible = base_ok.copy()
    for c in compiled.get("constraints", []):
        values = evaluate_quantity(c, out, proposal, features, bounds)
        threshold = float(c["resolved_threshold"])
        mask = _constraint_mask(values, str(c.get("operator", ">=")), threshold, float(c.get("atol", 1e-12)))
        out[f"constraint_value__{c['slug']}"] = values
        out[f"constraint_pass__{c['slug']}"] = mask
        feasible &= mask

    for o in compiled["objectives"]:
        vals = evaluate_quantity(o, out, proposal, features, bounds)
        z = normalize_objective(vals, str(o["direction"]), dict(o["resolved_normalization"]))
        out[o["raw_column"]] = vals
        out[o["norm_column"]] = z

    norm_cols = list(compiled["objective_norm_columns"])
    feasible &= np.isfinite(out[norm_cols].to_numpy(float)).all(axis=1)
    out["source_problem_feasible"] = base_ok
    out["feasible"] = feasible
    out["status"] = np.where(~base_ok, "source_infeasible", np.where(feasible, "optimal_generic_problem", "generic_constraint_or_objective_invalid"))
    out["problem_hash"] = actual_hash
    return out
