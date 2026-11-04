from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


SUPPORTED_OPERATORS = {
    "describe",
    "query",
    "association",
    "correlation_matrix",
    "global_sensitivity",
    "local_sensitivity",
    "sensitivity_map",
    "interaction",
    "fragility",
    "robustness",
    "threshold_region",
    "threshold_boundary",
    "feasibility",
    "regime_discovery",
    "dimensionality",
    "tradeoff",
    "pareto",
    "pareto_saturation",
    "compare_regions",
    "coverage",
    "outlier",
    "point_query",
    "response_curve",
    "marginal_response",
    "surrogate_uncertainty",
    "controller_scan",
    "threshold_switch_scan",
    "response_surface",
    "sampling_saturation",
}


class SpecError(ValueError):
    pass


def load_problem(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise SpecError("Problem YAML must contain a mapping/object at the top level.")
    data["_problem_path"] = str(p)
    validate_problem(data)
    return data


def _target_list(data: dict[str, Any]) -> list[dict[str, Any]]:
    raw = data.get("targets", data.get("target", []))
    if isinstance(raw, (str, dict)):
        raw = [raw]
    out = []
    for i, target in enumerate(raw or []):
        if isinstance(target, str):
            out.append({"name": target, "value": target, "direction": "maximize"})
        elif isinstance(target, dict):
            q = dict(target)
            q.setdefault(
                "name",
                str(q.get("value", q.get("column", q.get("expression", f"target_{i + 1}")))),
            )
            q.setdefault("direction", "maximize")
            if str(q["direction"]).lower() not in {"maximize", "minimize", "max", "min"}:
                raise SpecError(f"Invalid target direction: {q['direction']!r}")
            out.append(q)
        else:
            raise SpecError(f"Invalid target entry: {target!r}")
    return out


def validate_problem(data: dict[str, Any]) -> None:
    version = str(data.get("spec_version", "1.0"))
    if version != "1.0":
        raise SpecError(f"Unsupported spec_version={version}; expected 1.0")
    if not data.get("name"):
        raise SpecError("Problem requires name.")

    op = data.get("operator", {})
    if isinstance(op, str):
        op = {"name": op}
    name = op.get("name")
    if name not in SUPPORTED_OPERATORS:
        raise SpecError(f"Unsupported operator {name!r}. Supported: {sorted(SUPPORTED_OPERATORS)}")

    targets = _target_list(data)
    no_target = {
        "query",
        "coverage",
        "feasibility",
        "regime_discovery",
        "dimensionality",
        "outlier",
        "sampling_saturation",
        "correlation_matrix",
    }
    if not targets and name not in no_target:
        raise SpecError(f"Operator {name} requires at least one target.")
    if name in {"tradeoff", "pareto", "pareto_saturation"} and len(targets) < 2:
        raise SpecError(f"Operator {name} requires at least two targets.")

    backend = str((data.get("backend") or {}).get("mode", "auto"))
    if backend not in {"auto", "landscape", "surrogate"}:
        raise SpecError("backend.mode must be auto|landscape|surrogate")

    scope = data.get("scope") or {}
    logic = str(scope.get("filter_logic", "all"))
    if logic not in {"all", "any"}:
        raise SpecError("scope.filter_logic must be all|any")
    loc = scope.get("locality") or {}
    if loc and loc.get("mode", "none") not in {"none", "point", "sample"}:
        raise SpecError("scope.locality.mode must be none|point|sample")
    if loc and float(loc.get("radius", 0.10)) <= 0:
        raise SpecError("scope.locality.radius must be > 0")

    vis = data.get("visualization") or {}
    if vis:
        mode = str(vis.get("mode", "auto"))
        if mode not in {"auto", "override"}:
            raise SpecError("visualization.mode must be auto|override")
        if mode == "override" and not vis.get("family"):
            raise SpecError("visualization.mode=override requires visualization.family")

    q = data.get("question") or {}
    if q and not isinstance(q, dict):
        raise SpecError("question must be a mapping/object when provided")
    sem = data.get("semantics") or {}
    if sem and not isinstance(sem, dict):
        raise SpecError("semantics must be a mapping/object when provided")


def normalized_problem(data: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in data.items() if not str(k).startswith("_")}
    out["targets"] = _target_list(data)
    op = out.get("operator", {})
    if isinstance(op, str):
        op = {"name": op}
    op = dict(op)
    op.setdefault("parameters", {})
    out["operator"] = op

    out.setdefault("variables", {"mode": "all_design"})
    if isinstance(out["variables"], list):
        out["variables"] = {"mode": "list", "values": out["variables"]}
    if isinstance(out["variables"], str):
        out["variables"] = {"mode": out["variables"]}

    out.setdefault("scope", {})
    out.setdefault("backend", {"mode": "auto"})
    out.setdefault("surrogate", {})
    out["surrogate"].setdefault("model", "extra_trees")
    out["surrogate"].setdefault("n_estimators", 300)
    out["surrogate"].setdefault("cv_folds", 5)
    out["surrogate"].setdefault("min_r2", 0.50)
    out["surrogate"].setdefault("min_auc", 0.65)
    out["surrogate"].setdefault("max_train_rows", 50000)
    out["surrogate"].setdefault("n_jobs", -1)

    out.setdefault("question", {})
    out.setdefault("semantics", {})
    out.setdefault("visualization", {"mode": "auto"})
    out["visualization"].setdefault("mode", "auto")

    out.setdefault("outputs", {})
    out["outputs"].setdefault("save_figures", True)
    out["outputs"].setdefault("publication_formats", ["png", "pdf", "svg"])
    out["outputs"].setdefault("dpi", 600)
    return out
