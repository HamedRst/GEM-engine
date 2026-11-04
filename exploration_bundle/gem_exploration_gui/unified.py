from __future__ import annotations

from dataclasses import dataclass, field
from copy import deepcopy
from pathlib import Path
from typing import Any
import re

import yaml

from .problem_builder import ProblemDraft, TargetDraft, FilterDraft, build_problem, sanitize_name


@dataclass(frozen=True)
class DetailDefinition:
    key: str
    label: str
    description: str
    internal_operator: str


@dataclass(frozen=True)
class PrimitiveDefinition:
    key: str
    label: str
    prompt: str
    details: tuple[DetailDefinition, ...]


def D(key: str, label: str, description: str, operator: str) -> DetailDefinition:
    return DetailDefinition(key, label, description, operator)


PRIMITIVES: dict[str, PrimitiveDefinition] = {
    "describe": PrimitiveDefinition(
        "describe", "Describe", "Summarize what the selected observable looks like.",
        (D("distribution", "Distribution / summary", "Range, center, spread and tails of the observable.", "describe"),),
    ),
    "relate": PrimitiveDefinition(
        "relate", "Relate", "Ask how an observable is associated with, influenced by, or controlled by other quantities.",
        (
            D("association", "Association", "Model-free Pearson/Spearman association with the exploration variables.", "association"),
            D("correlation", "Correlation structure", "Pairwise correlation structure among selected variables/observables.", "correlation_matrix"),
            D("global_influence", "Global influence", "Cross-validated surrogate permutation importance across the scope.", "global_sensitivity"),
            D("local_influence", "Local influence", "Signed local effect around a resolved local context.", "local_sensitivity"),
            D("influence_map", "Influence across contexts", "How local sensitivity changes across many landscape contexts.", "sensitivity_map"),
            D("interaction", "Pair interaction", "Non-additive effects between pairs of exploration variables.", "interaction"),
            D("controller", "Controller behavior", "Variables acting as consistent monotonic control knobs.", "controller_scan"),
            D("feasibility", "Feasibility relationship", "Variables that distinguish feasible from infeasible designs.", "feasibility"),
        ),
    ),
    "vary": PrimitiveDefinition(
        "vary", "Vary", "Ask how an observable changes while one or more variables are varied.",
        (
            D("local_curve", "One-variable local response", "Vary one variable around a fixed/background design.", "response_curve"),
            D("marginal_curve", "One-variable marginal response", "Average the response across observed backgrounds.", "marginal_response"),
            D("surface", "Two-variable response", "Joint response across two exploration variables.", "response_surface"),
            D("point", "Evidence at a point", "Prediction plus nearest exact landscape support at a specified design.", "point_query"),
            D("threshold_crossing", "Threshold crossing", "Which variable drives the observable across a specified threshold.", "threshold_switch_scan"),
        ),
    ),
    "compare": PrimitiveDefinition(
        "compare", "Compare", "Compare an observable between two explicitly defined conditions or between threshold-meeting and non-meeting designs.",
        (
            D("regions", "Two condition-defined regions", "Compare the observable distribution between Region A and Region B.", "compare_regions"),
            D("threshold_groups", "Threshold-meeting vs other designs", "Compare designs meeting a target threshold with designs that do not.", "threshold_region"),
        ),
    ),
    "perturb": PrimitiveDefinition(
        "perturb", "Change / robustness", "Ask how stable or fragile an observable is as selected design variables change locally.",
        (
            D("robustness", "Robustness / stability", "Landscape-local stability score; lower local change means more robust.", "robustness"),
            D("fragility", "Fragility", "Landscape-local q95 phenotype change; larger change means more fragile.", "fragility"),
        ),
    ),
    "select": PrimitiveDefinition(
        "select", "Select", "Find designs with a specified property or location in the landscape.",
        (
            D("conditions", "Designs satisfying conditions", "Return exact sampled designs satisfying the current conditions.", "query"),
            D("threshold_boundary", "Near a threshold boundary", "Find exact sampled designs closest to a target threshold.", "threshold_boundary"),
            D("outliers", "Unusual designs / outliers", "Identify designs unusual in the selected feature space.", "outlier"),
        ),
    ),
    "structure": PrimitiveDefinition(
        "structure", "Structure", "Ask about latent structure, coverage, sampling sufficiency, or interpolation support.",
        (
            D("regimes", "Phenotype regimes", "Candidate clusters/regimes in selected phenotype space.", "regime_discovery"),
            D("dimensionality", "Effective dimensionality", "How many latent dimensions explain the selected features.", "dimensionality"),
            D("coverage", "Design-space coverage", "How densely the exact landscape covers the selected design variables.", "coverage"),
            D("sampling_saturation", "Sampling saturation", "How coverage improves as landscape sample size increases.", "sampling_saturation"),
            D("surrogate_support", "Surrogate support", "Where interpolation is least supported by the sampled landscape.", "surrogate_uncertainty"),
        ),
    ),
    "pareto": PrimitiveDefinition(
        "pareto", "Multiobjective / Pareto", "Ask about trade-offs and nondominated structure across two or more observables.",
        (
            D("tradeoff", "Trade-off structure", "Show full phenotype trade-off and empirical frontier.", "tradeoff"),
            D("frontier", "Empirical Pareto frontier", "Extract exact sampled nondominated designs.", "pareto"),
            D("saturation", "Pareto sampling saturation", "How much sampling is needed to recover the full empirical frontier.", "pareto_saturation"),
        ),
    ),
}

DETAIL_TO_OPERATOR = {
    (p.key, d.key): d.internal_operator
    for p in PRIMITIVES.values()
    for d in p.details
}

INTERNAL_OPERATOR_TO_FORM = {
    d.internal_operator: (p.key, d.key)
    for p in PRIMITIVES.values()
    for d in p.details
}


def primitive_options() -> dict[str, str]:
    return {k: p.label for k, p in PRIMITIVES.items()}


def detail_options(primitive: str) -> dict[str, str]:
    p = PRIMITIVES[primitive]
    return {d.key: d.label for d in p.details}


def detail_definition(primitive: str, detail: str) -> DetailDefinition:
    for d in PRIMITIVES[primitive].details:
        if d.key == detail:
            return d
    raise ValueError(f"Unknown exploration detail {primitive}/{detail}")


@dataclass
class UnifiedCondition:
    column: str
    op: str
    value: Any = None

    def to_filter(self) -> dict[str, Any]:
        out = {"column": self.column, "op": self.op}
        if self.op not in {"finite", "true", "false"}:
            if self.op in {"between", "in"} and isinstance(self.value, (list, tuple)):
                out["values"] = list(self.value)
            else:
                out["value"] = self.value
        return out


@dataclass
class UnifiedQuestion:
    name: str
    landscape_path: str
    gem_model_path: str | None
    primitive: str
    detail: str
    targets: list[TargetDraft] = field(default_factory=list)
    derived: list[dict[str, Any]] = field(default_factory=list)
    variable_mode: str = "all_design"
    variables: list[str] = field(default_factory=list)
    conditions: list[UnifiedCondition] = field(default_factory=list)
    condition_logic: str = "all"
    region_a: list[UnifiedCondition] = field(default_factory=list)
    region_b: list[UnifiedCondition] = field(default_factory=list)
    locality: dict[str, Any] | None = None
    settings: dict[str, Any] = field(default_factory=dict)
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
    variable_labels: dict[str, str] = field(default_factory=dict)
    variable_units: dict[str, str] = field(default_factory=dict)
    output_dpi: int = 600


_SINGLE_TARGET = {
    "global_sensitivity", "local_sensitivity", "sensitivity_map", "interaction",
    "threshold_region", "threshold_boundary", "compare_regions", "point_query",
    "surrogate_uncertainty", "controller_scan", "threshold_switch_scan",
}
# These operators are scientifically meaningful for one or many observables.
_MULTI_OR_SINGLE_TARGET = {
    "describe", "association", "fragility", "robustness",
    "response_curve", "marginal_response", "response_surface",
}
_NO_TARGET = {"query", "coverage", "feasibility", "sampling_saturation", "correlation_matrix", "dimensionality"}
_MULTI_TARGET = {"tradeoff", "pareto", "pareto_saturation"}
_AT_LEAST_ONE_TARGET = {"regime_discovery"}
_OPTIONAL_TARGET = {"outlier"}


def _target_names(targets: list[TargetDraft]) -> list[str]:
    return [t.name for t in targets]


def _condition_phrase(c: UnifiedCondition) -> str:
    if c.op == "true":
        return f"{c.column} is true"
    if c.op == "false":
        return f"{c.column} is false"
    if c.op == "finite":
        return f"{c.column} is finite"
    if c.op == "top_quantile":
        return f"{c.column} is in the top {float(c.value) * 100:.0f}%"
    if c.op == "bottom_quantile":
        return f"{c.column} is in the bottom {float(c.value) * 100:.0f}%"
    if c.op == "between":
        return f"{c.column} is between {c.value}"
    if c.op == "in":
        return f"{c.column} is in {c.value}"
    return f"{c.column} {c.op} {c.value}"


def _condition_text(q: UnifiedQuestion) -> str:
    if not q.conditions:
        return "the full resolved landscape"
    joiner = " AND " if q.condition_logic == "all" else " OR "
    return joiner.join(_condition_phrase(c) for c in q.conditions)


def _var_text(q: UnifiedQuestion) -> str:
    if q.variable_mode == "all_design":
        return "all resolved design variables"
    if not q.variables:
        return "the selected design variables"
    if len(q.variables) <= 3:
        return ", ".join(q.variable_labels.get(v, v.replace("requested_medium__", "")) for v in q.variables)
    return f"{len(q.variables)} selected design variables"


def _target_text(q: UnifiedQuestion) -> str:
    if not q.targets:
        return "the selected landscape/design quantities"
    labels = [t.label or t.name.replace("_", " ") for t in q.targets]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def build_human_question(q: UnifiedQuestion) -> tuple[str, str, str]:
    target = _target_text(q)
    over = _var_text(q)
    under = _condition_text(q)
    detail = detail_definition(q.primitive, q.detail)

    if q.primitive == "describe":
        text = f"What does {target} look like under {under}?"
        expr = f"describe({target}; under={under})"
    elif q.primitive == "relate":
        text = f"How is {target} related to or controlled by {over} under {under}, using {detail.label.lower()}?"
        expr = f"relate({target}, over={over}, mode={q.detail}, under={under})"
    elif q.primitive == "vary":
        text = f"How does {target} vary over {over} under {under}, using {detail.label.lower()}?"
        expr = f"vary({target}, over={over}, mode={q.detail}, under={under})"
    elif q.primitive == "compare":
        if q.detail == "regions":
            text = f"How does {target} differ between Region A and Region B, within {under}?"
            expr = f"compare({target}, region_A, region_B; under={under})"
        else:
            th = q.settings.get("threshold")
            direction = q.settings.get("direction", ">=")
            text = f"What distinguishes designs where {target} {direction} {th} from designs that do not, under {under}?"
            expr = f"compare({target}, {target}{direction}{th}; under={under})"
    elif q.primitive == "perturb":
        if q.detail == "robustness":
            text = f"How robust is {target} to local changes in {over} under {under}?"
            expr = f"robust({target}, over={over}, under={under})"
        else:
            text = f"How fragile is {target} to local changes in {over} under {under}?"
            expr = f"fragile({target}, over={over}, under={under})"
    elif q.primitive == "select":
        if q.detail == "conditions":
            text = f"Which exact sampled designs satisfy {under}?"
            expr = f"select(where={under})"
        elif q.detail == "threshold_boundary":
            th = q.settings.get("threshold")
            text = f"Which sampled designs lie closest to the {target} threshold {th}, under {under}?"
            expr = f"select(near({target}=={th}), under={under})"
        else:
            text = f"Which designs are unusual in the selected target/variable space under {under}?"
            expr = f"select(outliers({target}, over={over}), under={under})"
    elif q.primitive == "structure":
        text = f"What {detail.label.lower()} is present for {target} over {over} under {under}?"
        expr = f"structure({target}, over={over}, mode={q.detail}, under={under})"
    elif q.primitive == "pareto":
        text = f"What {detail.label.lower()} exists among {target} under {under}?"
        expr = f"pareto({target}, mode={q.detail}, under={under})"
    else:
        raise ValueError(q.primitive)

    title = re.sub(r"\s+", " ", text.rstrip("?")).strip()
    return title, text, expr


def _required_target_check(op: str, targets: list[TargetDraft]) -> list[TargetDraft]:
    if op in _NO_TARGET:
        # Correlation/dimensionality may include selected observables as features;
        # other no-target analyses do not need observables.
        if op in {"correlation_matrix", "dimensionality"}:
            return targets
        return []
    if op in _MULTI_TARGET:
        if len(targets) < 2:
            raise ValueError(f"This exploration requires at least two observables; currently {len(targets)} selected.")
        return targets
    if op in _AT_LEAST_ONE_TARGET:
        if not targets:
            raise ValueError("Regime discovery requires at least one observable. With one observable, design variables provide the additional regime dimensions; with two or more observables, phenotype-space regimes are used by default.")
        return targets
    if op in _MULTI_OR_SINGLE_TARGET:
        if not targets:
            raise ValueError("This exploration requires at least one observable/target.")
        return targets
    if op in _OPTIONAL_TARGET:
        return targets[:1]
    if op in _SINGLE_TARGET:
        if not targets:
            raise ValueError("This exploration requires an observable/target.")
        # robustness/fragility intentionally support multiple observables in the generalized GUI.
        if op in {"robustness", "fragility"}:
            return targets
        return targets[:1]
    return targets


def _selected_variables(q: UnifiedQuestion) -> list[str]:
    return list(q.variables) if q.variable_mode == "list" else []


def _features(q: UnifiedQuestion, targets: list[TargetDraft]) -> list[str]:
    # An empty list in all-design mode tells the backend to combine all resolved
    # design variables with any materialized observables. In selected mode we
    # freeze the exact requested feature set.
    if q.variable_mode == "all_design":
        return []
    features: list[str] = list(q.variables)
    features.extend(t.name for t in targets)
    return list(dict.fromkeys(features))


def _operator_parameters(q: UnifiedQuestion, op: str, targets: list[TargetDraft]) -> dict[str, Any]:
    s = q.settings
    vars_ = _selected_variables(q)
    p: dict[str, Any] = {}

    if op == "query":
        p = {
            "sort_by": s.get("sort_by") or None,
            "ascending": bool(s.get("ascending", False)),
            "max_saved_points": int(s.get("max_saved_points", 10000)),
        }
    elif op == "correlation_matrix":
        features = _features(q, targets)
        p = {"features": features, "max_features": int(s.get("max_features", 40))}
    elif op == "local_sensitivity":
        p = {"step_fraction": float(s.get("step_fraction", 0.01))}
        if s.get("anchor_point"):
            p["point"] = deepcopy(s["anchor_point"])
        elif s.get("anchor_sample_id") is not None:
            p["sample_id"] = int(s["anchor_sample_id"])
    elif op == "sensitivity_map":
        p = {"anchor_count": int(s.get("anchor_count", 300)), "step_fraction": float(s.get("step_fraction", 0.01))}
    elif op == "interaction":
        p = {
            "step_fraction": float(s.get("step_fraction", 0.03)),
            "anchor_count": int(s.get("anchor_count", 128)),
            "max_pairs": int(s.get("max_pairs", 60)),
        }
    elif op in {"fragility", "robustness"}:
        p = {"neighbors": int(s.get("neighbors", 20))}
    elif op == "threshold_region":
        p = {
            "threshold": float(s.get("threshold", 1.0)),
            "direction": str(s.get("direction", ">=")),
            "max_saved_points": int(s.get("max_saved_points", 10000)),
        }
    elif op == "threshold_boundary":
        p = {"threshold": float(s.get("threshold", 1.0)), "n_points": int(s.get("n_points", 200))}
    elif op == "feasibility":
        p = {"column": str(s.get("feasibility_column", "feasible"))}
    elif op == "regime_discovery":
        # Regimes should be discovered in a scientifically meaningful space, not
        # accidentally in the first two design columns.  Default behavior:
        #   * >=2 observables + variables=all -> phenotype regimes (observables only)
        #   * >=2 observables + selected variables -> joint phenotype/design regimes
        #   * 1 observable -> observable + design variables (needs >=2 features)
        target_names = [t.name for t in targets]
        requested_space = str(s.get("regime_space", "auto") or "auto").lower()
        if requested_space not in {"auto", "observations", "joint", "variables"}:
            raise ValueError("regime space must be auto|observations|joint|variables")
        if requested_space == "observations":
            features = target_names
            space = "observations"
        elif requested_space == "variables":
            features = list(q.variables) if q.variable_mode == "list" else []
            space = "variables"
        elif requested_space == "joint":
            features = list(dict.fromkeys([*target_names, *list(q.variables)])) if q.variable_mode == "list" else []
            space = "joint"
        else:
            if len(target_names) >= 2 and q.variable_mode == "all_design":
                features = target_names
                space = "observations"
            elif len(target_names) >= 2 and q.variable_mode == "list":
                features = list(dict.fromkeys([*target_names, *list(q.variables)]))
                space = "joint"
            else:
                features = list(dict.fromkeys([*target_names, *list(q.variables)])) if q.variable_mode == "list" else []
                space = "joint"
        p = {"max_k": int(s.get("max_k", 8)), "features": features, "clustering_space": space}
    elif op == "dimensionality":
        features = _features(q, targets)
        if not features and q.variable_mode == "all_design":
            # empty features intentionally asks the backend to use resolved design variables.
            features = []
        p = {
            "features": features,
            "variance_threshold": float(s.get("variance_threshold", 0.90)),
            "max_components": int(s.get("max_components", 20)),
        }
    elif op == "pareto_saturation":
        p = {"sample_sizes": list(s.get("sample_sizes", [500, 1000, 2500, 5000, 10000, 25000, 50000])), "repeats": int(s.get("repeats", 10))}
    elif op == "compare_regions":
        p = {"region_a": [c.to_filter() for c in q.region_a], "region_b": [c.to_filter() for c in q.region_b]}
        if not p["region_a"] or not p["region_b"]:
            raise ValueError("Region comparison requires at least one condition in both Region A and Region B.")
    elif op == "outlier":
        features = _features(q, targets)
        p = {"contamination": float(s.get("contamination", 0.01))}
        if features:
            p["features"] = features
    elif op == "point_query":
        point = deepcopy(s.get("point") or {})
        if not point:
            raise ValueError("Evidence-at-a-point requires a design-point mapping.")
        p = {"point": point, "neighbors": int(s.get("neighbors", 10))}
    elif op == "response_curve":
        if len(vars_) != 1:
            raise ValueError("One-variable local response requires exactly one selected exploration variable.")
        p = {"variable": vars_[0], "grid_points": int(s.get("grid_points", 101))}
    elif op == "marginal_response":
        if len(vars_) != 1:
            raise ValueError("One-variable marginal response requires exactly one selected exploration variable.")
        p = {"variable": vars_[0], "grid_points": int(s.get("grid_points", 51)), "background_rows": int(s.get("background_rows", 512))}
    elif op == "surrogate_uncertainty":
        p = {"evaluation_rows": int(s.get("evaluation_rows", 10000)), "top_points": int(s.get("top_points", 500))}
    elif op == "controller_scan":
        p = {"anchor_count": int(s.get("anchor_count", 96)), "levels": int(s.get("levels", 9))}
    elif op == "threshold_switch_scan":
        p = {"threshold": float(s.get("threshold", 1.0)), "anchor_count": int(s.get("anchor_count", 96)), "levels": int(s.get("levels", 21))}
    elif op == "response_surface":
        if len(vars_) != 2:
            raise ValueError("Two-variable response requires exactly two selected exploration variables.")
        p = {"variables": vars_, "grid_points": int(s.get("grid_points", 41))}
    elif op == "sampling_saturation":
        p = {
            "sample_sizes": list(s.get("sample_sizes", [500, 1000, 2500, 5000, 10000, 25000, 50000])),
            "repeats": int(s.get("repeats", 10)),
            "evaluation_points": int(s.get("evaluation_points", 5000)),
        }
    return p


def compile_unified_question(q: UnifiedQuestion) -> tuple[dict[str, Any], dict[str, Any]]:
    if q.primitive not in PRIMITIVES:
        raise ValueError(f"Unknown exploration primitive: {q.primitive}")
    op = DETAIL_TO_OPERATOR.get((q.primitive, q.detail))
    if not op:
        raise ValueError(f"Unknown exploration detail: {q.primitive}/{q.detail}")
    if q.condition_logic not in {"all", "any"}:
        raise ValueError("condition_logic must be all|any")
    if q.variable_mode not in {"all_design", "list"}:
        raise ValueError("variable_mode must be all_design|list")
    if q.variable_mode == "list" and not q.variables and op not in {"query", "threshold_boundary"}:
        raise ValueError("Selected-variable mode requires at least one exploration variable.")

    targets = _required_target_check(op, q.targets)
    title, question_text, compact = build_human_question(q)
    params = _operator_parameters(q, op, targets)

    scope_filters = [FilterDraft(c.column, c.op, c.value) for c in q.conditions]

    # A response plot may display only one/two exploration variables, but the
    # surrogate must learn the target from the full design context. The v0.3 GUI
    # accidentally trained on only the displayed variable(s), causing widespread
    # low R2 even when the old Atlas passed on the same landscape.
    execution_variable_mode = q.variable_mode
    execution_selected_variables = list(q.variables)
    if op in {"response_curve", "response_surface", "marginal_response", "point_query"}:
        execution_variable_mode = "all_design"
        execution_selected_variables = []

    draft = ProblemDraft(
        name=q.name,
        operator=op,
        landscape_path=q.landscape_path,
        include_fluxes=False,
        variable_mode=execution_variable_mode,
        selected_variables=execution_selected_variables,
        targets=targets,
        derived=deepcopy(q.derived),
        scope_filters=scope_filters,
        locality=deepcopy(q.locality),
        backend_mode=q.backend_mode,
        surrogate=deepcopy(q.surrogate),
        parameters=params,
        question_title=title,
        question_text=question_text,
        why_it_matters=detail_definition(q.primitive, q.detail).description,
        variable_labels=deepcopy(q.variable_labels),
        variable_units=deepcopy(q.variable_units),
        visualization_mode="auto",
        output_dpi=q.output_dpi,
    )
    problem = build_problem(draft)
    problem["scope"]["filter_logic"] = q.condition_logic
    problem["unified_exploration"] = {
        "spec_version": "0.5",
        "primitive": q.primitive,
        "detail": q.detail,
        "compact_expression": compact,
        "gem_model_path": q.gem_model_path,
    }

    public = {
        "exploration_spec_version": "0.5",
        "name": sanitize_name(q.name),
        "source": {
            "landscape": str(Path(q.landscape_path).expanduser()),
            "gem_model": str(Path(q.gem_model_path).expanduser()) if q.gem_model_path else None,
        },
        "observables": {
            "targets": [
                {k: v for k, v in {
                    "name": t.name,
                    "value": t.value,
                    "direction": t.direction,
                    "label": t.label,
                    "unit": t.unit,
                }.items() if v is not None}
                for t in q.targets
            ],
            "derived": deepcopy(q.derived),
        },
        "exploration": {
            "primitive": q.primitive,
            "detail": q.detail,
            "over": {
                "mode": q.variable_mode,
                "variables": list(q.variables) if q.variable_mode == "list" else [],
            },
            "compact_expression": compact,
        },
        "conditions": {
            "logic": q.condition_logic,
            "filters": [c.to_filter() for c in q.conditions],
        },
        "regions": {
            "A": [c.to_filter() for c in q.region_a],
            "B": [c.to_filter() for c in q.region_b],
        } if (q.region_a or q.region_b) else {},
        "context": {"locality": deepcopy(q.locality)} if q.locality else {},
        "parameters": deepcopy(q.settings),
        "evidence": {
            "backend": q.backend_mode,
            "surrogate_qc": deepcopy(q.surrogate),
            "new_fba_calls": 0,
        },
        "question": {
            "title": title,
            "text": question_text,
        },
        "compiled_execution": {
            "internal_operator": op,
            "note": "Internal operator is an execution detail. The public scientific contract is the unified exploration formulation above.",
        },
    }
    return public, problem


def dump_unified_yaml(data: dict[str, Any]) -> str:
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)



def unified_from_dict(data: dict[str, Any]) -> UnifiedQuestion:
    version = str(data.get("exploration_spec_version", ""))
    if version not in {"0.2", "0.3", "0.4", "0.5"}:
        raise ValueError("exploration_spec_version must be one of 0.2, 0.3, 0.4, 0.5")
    source = data.get("source") or {}
    observables = data.get("observables") or {}
    exp = data.get("exploration") or {}
    over = exp.get("over") or {}
    conds = data.get("conditions") or {}
    regions = data.get("regions") or {}
    evidence = data.get("evidence") or {}

    targets = []
    for t in observables.get("targets") or []:
        targets.append(TargetDraft(
            name=str(t.get("name") or str(t.get("value", "target")).split(":")[-1]),
            value=str(t.get("value")),
            direction=str(t.get("direction", "maximize")),
            label=t.get("label"),
            unit=t.get("unit"),
        ))

    def conditions(items: list[dict[str, Any]]) -> list[UnifiedCondition]:
        out = []
        for f in items or []:
            val = f.get("values", f.get("value"))
            out.append(UnifiedCondition(str(f.get("column")), str(f.get("op", "==")), val))
        return out

    return UnifiedQuestion(
        name=str(data.get("name") or "exploration"),
        landscape_path=str(source.get("landscape") or ""),
        gem_model_path=source.get("gem_model"),
        primitive=str(exp.get("primitive")),
        detail=str(exp.get("detail")),
        targets=targets,
        derived=deepcopy(observables.get("derived") or []),
        variable_mode=str(over.get("mode", "all_design")),
        variables=list(over.get("variables") or []),
        conditions=conditions(conds.get("filters") or []),
        condition_logic=str(conds.get("logic", "all")),
        region_a=conditions(regions.get("A") or []),
        region_b=conditions(regions.get("B") or []),
        locality=deepcopy(((data.get("context") or {}).get("locality"))),
        settings=deepcopy(data.get("parameters") or {}),
        backend_mode=str(evidence.get("backend", "auto")),
        surrogate=deepcopy(evidence.get("surrogate_qc") or {
            "model": "extra_trees", "n_estimators": 300, "cv_folds": 5,
            "min_r2": 0.50, "min_auc": 0.65, "max_train_rows": 50000, "n_jobs": -1,
        }),
    )


def load_unified_yaml(path: str | Path) -> UnifiedQuestion:
    p = Path(path).expanduser().resolve()
    data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("exploration.yaml must contain a mapping/object")
    return unified_from_dict(data)
