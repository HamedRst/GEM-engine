from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json
import re

import yaml

from .catalog import QUESTIONS


@dataclass
class TargetDraft:
    name: str
    value: str
    direction: str = "maximize"
    label: str | None = None
    unit: str | None = None


@dataclass
class FilterDraft:
    column: str
    op: str
    value: Any = None

    def to_dict(self) -> dict[str, Any]:
        d = {"column": self.column, "op": self.op}
        if self.op not in {"finite", "true", "false"}:
            d["value"] = self.value
        return d


@dataclass
class ProblemDraft:
    name: str
    operator: str
    landscape_path: str
    include_fluxes: bool = False
    variable_mode: str = "all_design"
    selected_variables: list[str] = field(default_factory=list)
    targets: list[TargetDraft] = field(default_factory=list)
    derived: list[dict[str, Any]] = field(default_factory=list)
    scope_filters: list[FilterDraft] = field(default_factory=list)
    locality: dict[str, Any] | None = None
    backend_mode: str = "auto"
    surrogate: dict[str, Any] = field(default_factory=lambda: {
        "model": "extra_trees",
        "n_estimators": 300,
        "cv_folds": 5,
        "min_r2": 0.50,
        "min_auc": 0.65,
        "max_train_rows": 50000,
        "n_jobs": -1,
    })
    parameters: dict[str, Any] = field(default_factory=dict)
    question_title: str | None = None
    question_text: str | None = None
    why_it_matters: str | None = None
    variable_labels: dict[str, str] = field(default_factory=dict)
    variable_units: dict[str, str] = field(default_factory=dict)
    target_labels: dict[str, str] = field(default_factory=dict)
    target_units: dict[str, str] = field(default_factory=dict)
    visualization_mode: str = "auto"
    visualization_family: str | None = None
    output_dpi: int = 600


def sanitize_name(text: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", text.strip())
    return s.strip("_.-") or "exploration_question"


def parse_jsonish(value: str | Any, fallback: Any) -> Any:
    if not isinstance(value, str):
        return value
    t = value.strip()
    if not t:
        return deepcopy(fallback)
    try:
        return json.loads(t)
    except Exception:
        try:
            return yaml.safe_load(t)
        except Exception as exc:
            raise ValueError(f"Could not parse structured parameter: {value!r}") from exc


def parse_sample_sizes(value: str | list[int] | tuple[int, ...]) -> list[int]:
    if isinstance(value, (list, tuple)):
        vals = value
    else:
        vals = [x.strip() for x in str(value).replace(";", ",").split(",") if x.strip()]
    out = [int(x) for x in vals]
    if any(x <= 0 for x in out):
        raise ValueError("Sample sizes must be positive integers")
    return out


def build_problem(draft: ProblemDraft) -> dict[str, Any]:
    if draft.operator not in QUESTIONS:
        raise ValueError(f"Unknown operator: {draft.operator}")
    qdef = QUESTIONS[draft.operator]

    if qdef.target_count and len(draft.targets) < qdef.target_count and not qdef.target_optional:
        raise ValueError(f"{qdef.title} requires {qdef.target_count} target(s)")
    # Zero-required-target operators may still carry optional observables for
    # unified formulations such as correlation/dimensionality. The backend
    # ignores them when not needed, while materialization makes them available
    # as named features.
    if qdef.target_count == 0 and not qdef.target_optional:
        targets = list(draft.targets)
    else:
        targets = draft.targets

    if draft.variable_mode == "list" and not draft.selected_variables:
        raise ValueError("Variable mode 'list' requires at least one selected design variable")

    variables: dict[str, Any]
    if draft.variable_mode == "list":
        variables = {"mode": "list", "values": list(draft.selected_variables)}
    else:
        variables = {"mode": "all_design"}

    scope: dict[str, Any] = {}
    if draft.scope_filters:
        scope["filters"] = [f.to_dict() for f in draft.scope_filters]
    if draft.locality:
        scope["locality"] = deepcopy(draft.locality)

    target_dicts = []
    semantics_target_labels = dict(draft.target_labels)
    semantics_target_units = dict(draft.target_units)
    for t in targets:
        target_dicts.append({"name": t.name, "value": t.value, "direction": t.direction})
        if t.label:
            semantics_target_labels[t.name] = t.label
        if t.unit:
            semantics_target_units[t.name] = t.unit

    question = {
        "title": draft.question_title or qdef.title,
        "text": draft.question_text or qdef.question,
        "why_it_matters": draft.why_it_matters or qdef.why,
    }

    problem: dict[str, Any] = {
        "spec_version": "1.0",
        "name": sanitize_name(draft.name),
        "question": question,
        "landscape": {
            "path": str(Path(draft.landscape_path).expanduser()),
            "include_fluxes": bool(draft.include_fluxes),
        },
        "variables": variables,
        "targets": target_dicts,
        "scope": scope,
        "backend": {"mode": draft.backend_mode},
        "surrogate": deepcopy(draft.surrogate),
        "operator": {"name": draft.operator, "parameters": deepcopy(draft.parameters)},
        "semantics": {
            "target_labels": semantics_target_labels,
            "target_units": semantics_target_units,
            "variable_labels": dict(draft.variable_labels),
            "variable_units": dict(draft.variable_units),
        },
        "visualization": {"mode": draft.visualization_mode},
        "outputs": {
            "save_figures": True,
            "publication_formats": ["png", "pdf", "svg"],
            "dpi": int(draft.output_dpi),
        },
    }
    if draft.derived:
        problem["derived"] = deepcopy(draft.derived)
    if draft.visualization_mode == "override":
        if not draft.visualization_family:
            raise ValueError("visualization.mode=override requires visualization_family")
        problem["visualization"]["family"] = draft.visualization_family
    return problem


def dump_problem_yaml(problem: dict[str, Any]) -> str:
    return yaml.safe_dump(problem, sort_keys=False, allow_unicode=True)


def save_problem(problem: dict[str, Any], path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(dump_problem_yaml(problem), encoding="utf-8")
    return p
