from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any
from itertools import combinations

import numpy as np
import pandas as pd


# Scientific visualization is compiled from the resolved problem definition, not
# selected ad hoc by each operator. Operators remain responsible for computation;
# this planner is responsible for communicating the result faithfully.

OPERATOR_VISUAL_FAMILY = {
    "describe": "distribution",
    "query": "query_context",
    "association": "ranked_effect",
    "correlation_matrix": "correlation_heatmap",
    "global_sensitivity": "ranked_importance",
    "local_sensitivity": "signed_effect",
    "sensitivity_map": "local_sensitivity_heatmap",
    "interaction": "interaction_matrix",
    "fragility": "fragility_distribution",
    "robustness": "robustness_distribution",
    "threshold_region": "threshold_region_effects",
    "threshold_boundary": "threshold_boundary_context",
    "feasibility": "feasibility_drivers",
    "regime_discovery": "regime_map",
    "dimensionality": "dimensionality_curve",
    "tradeoff": "tradeoff_frontier",
    "pareto": "pareto_frontier",
    "pareto_saturation": "pareto_saturation",
    "compare_regions": "region_comparison",
    "coverage": "coverage_distribution",
    "outlier": "outlier_context",
    "point_query": "point_query_context",
    "response_curve": "response_curve",
    "marginal_response": "marginal_response",
    "surrogate_uncertainty": "surrogate_support_map",
    "controller_scan": "controller_ranking",
    "threshold_switch_scan": "threshold_switch_ranking",
    "response_surface": "response_surface",
    "sampling_saturation": "sampling_saturation",
}


BACKEND_LABELS = {
    "landscape": "Stored exact-FBA landscape",
    "surrogate_assisted": "Surrogate-assisted interpolation of stored exact-FBA landscape",
    "hybrid": "Hybrid exact-landscape + surrogate-assisted analysis",
}


def _clean(s: Any) -> str:
    return str(s).replace("_", " ").strip()


def _atlas(spec: dict[str, Any]) -> dict[str, Any]:
    return spec.get("atlas") or {}


def _semantics(spec: dict[str, Any]) -> dict[str, Any]:
    return spec.get("semantics") or {}


def variable_label(name: str, spec: dict[str, Any]) -> str:
    atlas = _atlas(spec)
    sem = _semantics(spec)
    labels = {}
    labels.update(atlas.get("variable_labels") or {})
    labels.update(sem.get("variable_labels") or {})
    if name in labels:
        return str(labels[name])
    # Also accept reaction-id keyed labels.
    rid = name
    for p in ("requested_medium__", "applied_medium__"):
        if rid.startswith(p):
            rid = rid[len(p):]
            break
    if rid in labels:
        return str(labels[rid])
    if name == atlas.get("primary_variable"):
        return str(atlas.get("primary_variable_label") or rid)
    if name == atlas.get("secondary_variable"):
        return str(atlas.get("secondary_variable_label") or rid)
    return rid.replace("_", " ")


def target_label(name: str, spec: dict[str, Any]) -> str:
    sem = _semantics(spec)
    atlas = _atlas(spec)
    labels = {}
    labels.update(atlas.get("target_labels") or {})
    labels.update(sem.get("target_labels") or {})
    if name in labels:
        return str(labels[name])
    aliases = {
        "maximum_growth": "growth rate",
        "growth": "growth rate",
        "product_flux": "product formation",
        "product": "product formation",
        "product_per_carbon_input": "product per carbon input",
    }
    return aliases.get(name, _clean(name))


def target_unit(name: str, spec: dict[str, Any]) -> str | None:
    sem = _semantics(spec)
    units = sem.get("target_units") or {}
    return units.get(name)


def variable_unit(name: str, spec: dict[str, Any]) -> str | None:
    sem = _semantics(spec)
    units = sem.get("variable_units") or {}
    if name in units:
        return units[name]
    rid = name
    for p in ("requested_medium__", "applied_medium__"):
        if rid.startswith(p):
            rid = rid[len(p):]
            break
    return units.get(rid)


def _scope_phrase(spec: dict[str, Any], n_scope: int, n_total: int) -> str:
    scope = spec.get("scope") or {}
    filters = scope.get("filters") or []
    if filters:
        return f"the resolved filtered scope ({n_scope:,} of {n_total:,} landscape designs)"
    if n_scope != n_total:
        return f"the resolved analysis scope ({n_scope:,} of {n_total:,} landscape designs)"
    return f"the full resolved landscape ({n_scope:,} designs)"


def _target_phrase(spec: dict[str, Any]) -> str:
    ts = [t.get("name") for t in spec.get("targets", []) if t.get("name")]
    if not ts:
        return "the selected phenotype/design quantity"
    labels = [target_label(str(t), spec) for t in ts]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return f"{labels[0]} and {labels[1]}"
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _variable_phrase(spec: dict[str, Any], variables: list[str]) -> str:
    if not variables:
        return "the resolved design variables"
    if len(variables) <= 3:
        return ", ".join(variable_label(v, spec) for v in variables)
    return f"the {len(variables)} resolved design variables"


def _question_for_operator(op: str, spec: dict[str, Any], variables: list[str], n_scope: int, n_total: int) -> str:
    target = _target_phrase(spec)
    vars_phrase = _variable_phrase(spec, variables)
    scope = _scope_phrase(spec, n_scope, n_total)
    p = (spec.get("operator") or {}).get("parameters") or {}

    q = {
        "describe": f"What is the distribution of {target} across {scope}?",
        "query": f"Which designs satisfy the resolved query conditions within {scope}?",
        "association": f"Which design variables are most strongly associated with {target} across {scope}?",
        "correlation_matrix": f"How are the selected phenotype and design features associated with one another across {scope}?",
        "global_sensitivity": f"Which design variables most strongly influence {target} across {scope}?",
        "local_sensitivity": f"Near the resolved anchor design, which variables most strongly change {target}, and in which direction?",
        "sensitivity_map": f"How does local sensitivity of {target} to the design variables vary across the landscape?",
        "interaction": f"Which pairs of design variables show the strongest non-additive interaction for {target}?",
        "fragility": f"How fragile is {target} to nearby changes in design conditions across {scope}?",
        "robustness": f"How locally stable is {target} across nearby design conditions in {scope}?",
        "threshold_region": f"What distinguishes designs that meet the specified {target} threshold from those that do not?",
        "threshold_boundary": f"Where does the landscape lie closest to the specified {target} threshold boundary?",
        "feasibility": f"Which design variables best distinguish feasible from infeasible designs across {scope}?",
        "regime_discovery": f"Does the selected phenotype space contain reproducible regimes, and where are they located?",
        "dimensionality": f"How many latent dimensions are needed to represent the selected phenotype/design features?",
        "tradeoff": f"What trade-off structure exists between {_target_phrase(spec)}, and where is the empirical Pareto frontier?",
        "pareto": f"Which landscape designs form the empirical Pareto frontier for {_target_phrase(spec)}?",
        "pareto_saturation": f"How much sampling is required to recover the empirical Pareto frontier for {_target_phrase(spec)}?",
        "compare_regions": f"How does {target} differ between the two resolved regions?",
        "coverage": f"How densely and uniformly does the stored landscape cover the resolved design space?",
        "outlier": f"Which designs are unusual relative to the selected phenotype/design features, and where are those outliers located?",
        "point_query": f"What does the landscape support at the specified design point for {target}?",
        "response_curve": f"How is {target} predicted to change as one selected design variable is varied around the resolved anchor?",
        "marginal_response": f"How does the model-averaged response of {target} change across one selected design variable over observed backgrounds?",
        "surrogate_uncertainty": f"Where is surrogate support for {target} weakest or most uncertain within the sampled design space?",
        "controller_scan": f"Which design variables act as the strongest monotonic control knobs for {target} in surrogate sweeps?",
        "threshold_switch_scan": f"Which design variables most consistently drive {target} across the specified threshold in surrogate sweeps?",
        "response_surface": f"How does {target} vary jointly across the two selected design variables around the resolved background?",
        "sampling_saturation": f"How quickly does normalized design-space coverage improve as the number of sampled landscape points increases?",
    }.get(op)
    return q or f"What does the resolved {op} analysis reveal about {target} over {vars_phrase} within {scope}?"


def _purpose_for_family(family: str) -> str:
    return {
        "distribution": "Show the phenotype distribution and its central/spread statistics directly.",
        "ranked_effect": "Rank variables by effect magnitude so the strongest associations are immediately visible.",
        "ranked_importance": "Rank variables by held-out model importance and show uncertainty when available.",
        "signed_effect": "Show both magnitude and direction of the local effect around the explicit anchor.",
        "local_sensitivity_heatmap": "Show how signed local sensitivities vary across multiple landscape contexts.",
        "interaction_matrix": "Show all evaluated pairwise interactions in a symmetric variable-by-variable matrix.",
        "fragility_distribution": "Show the distribution of the local fragility statistic and its high-fragility tail.",
        "robustness_distribution": "Show the distribution of the local stability score and its lower-stability tail.",
        "multi_stability_comparison": "Compare normalized local fragility/stability distributions across multiple observables before inspecting each observable separately.",
        "threshold_region_effects": "Show which variables shift most between designs that meet versus miss the target threshold.",
        "threshold_boundary_context": "Place threshold-nearest designs in phenotype/design context and draw the threshold explicitly.",
        "feasibility_drivers": "Rank variables that distinguish feasibility, or explicitly report when no feasibility boundary exists.",
        "regime_map": "Display discovered regimes in the most interpretable two-dimensional scientific view.",
        "regime_pair": "Show regime assignments directly in a selected observable/feature plane.",
        "regime_pca": "Project the full clustering feature space to two principal components for visualization only; clustering itself remains in the full feature space.",
        "regime_profiles": "Compare standardized median feature profiles across the discovered regimes.",
        "regime_k_selection": "Show silhouette score across candidate cluster counts so the selected number of regimes is auditable.",
        "regime_pca_loadings": "Show which original features contribute most to the first two visualization principal components.",
        "dimensionality_curve": "Show per-component and cumulative explained variance with the requested variance threshold.",
        "tradeoff_frontier": "Show the landscape and empirical Pareto frontier together to expose the trade-off geometry.",
        "pareto_frontier": "Emphasize the empirical Pareto frontier and its relationship to the landscape.",
        "pareto_saturation": "Show frontier-recovery error versus sample size with replicate uncertainty.",
        "region_comparison": "Compare the target distributions for the two defined regions and report effect statistics.",
        "coverage_distribution": "Show nearest-neighbor design-space coverage as an empirical cumulative distribution.",
        "outlier_context": "Highlight identified outliers in a scientifically relevant phenotype/design projection.",
        "point_query_context": "Show the query prediction together with nearest exact landscape support in a contextual projection.",
        "response_curve": "Show the predicted response curve, support/dispersion band when available, and the varied design variable explicitly.",
        "marginal_response": "Show the model-averaged response and between-background spread across the varied variable.",
        "surrogate_support_map": "Map surrogate support/dispersion back onto the sampled design space rather than showing only a histogram.",
        "controller_ranking": "Rank control strength while retaining across-background variability information.",
        "threshold_switch_ranking": "Rank threshold-switch strength and annotate how often a crossing occurs.",
        "response_surface": "Show a two-variable response surface with contours and the target scale.",
        "sampling_saturation": "Show coverage error versus sample size with replicate uncertainty.",
        "query_context": "Show selected/query-matching designs in the most relevant available phenotype plane.",
        "correlation_heatmap": "Show the complete selected correlation matrix with readable semantic labels.",
    }.get(family, "Use the visualization family that most directly represents the resolved scientific answer.")


def _answer_metrics(op: str) -> list[str]:
    return {
        "describe": ["rows", "targets"],
        "query": ["matched_rows", "fraction_of_scope"],
        "global_sensitivity": ["importance_holdout_rows"],
        "local_sensitivity": ["surrogate_prediction", "ensemble_dispersion_proxy"],
        "interaction": ["pairs_evaluated"],
        "threshold_region": ["n_selected", "fraction_selected", "threshold"],
        "threshold_boundary": ["n_boundary_points", "median_normalized_distance", "threshold"],
        "feasibility": ["feasible_fraction", "importance_holdout_rows"],
        "regime_discovery": ["selected_k", "silhouette"],
        "dimensionality": ["components_to_threshold", "variance_threshold"],
        "tradeoff": ["n_pareto", "fraction_pareto"],
        "pareto": ["n_pareto", "fraction_pareto"],
        "pareto_saturation": ["full_front_size", "full_normalized_hv_2d"],
        "coverage": ["median_nn_distance", "q95_nn_distance", "max_nn_distance"],
        "point_query": ["surrogate_prediction", "nearest_exact_distance", "ensemble_dispersion_proxy"],
        "surrogate_uncertainty": ["median_normalized_ensemble_dispersion", "q95_normalized_ensemble_dispersion"],
        "controller_scan": ["anchors", "levels"],
        "threshold_switch_scan": ["threshold", "anchors", "levels"],
        "sampling_saturation": ["dimension", "repeats"],
    }.get(op, [])



def _safe_key(text: str) -> str:
    import re
    z = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_").lower()
    return z[:72] or "view"


def _visualization_views(op: str, family: str, result: Any, spec: dict[str, Any], target_names: list[str]) -> list[dict[str, Any]]:
    """Compile one or more complementary scientific views.

    A single analysis may need several figures when multiple observables/features
    are involved.  The analysis result stays singular; these views are different
    faithful windows onto the same result, not separate statistical tests.
    """
    views: list[dict[str, Any]] = []

    def add(key: str, fam: str, **kwargs: Any) -> None:
        rec = {"key": key, "family": fam, "purpose": _purpose_for_family(fam)}
        rec.update({k: v for k, v in kwargs.items() if v is not None})
        views.append(rec)

    if op == "describe" and target_names:
        for i, t in enumerate(target_names):
            add("main" if i == 0 else f"distribution_{_safe_key(t)}", "distribution", target=t)
        return views

    if op == "association" and target_names:
        for i, t in enumerate(target_names):
            add("main" if i == 0 else f"association_{_safe_key(t)}", "ranked_effect", target=t)
        return views

    if op in {"response_curve", "marginal_response", "response_surface"} and target_names:
        fam = {"response_curve":"response_curve", "marginal_response":"marginal_response", "response_surface":"response_surface"}[op]
        for i, t in enumerate(target_names):
            add("main" if i == 0 else f"{fam}_{_safe_key(t)}", fam, target=t)
        return views

    if op in {"fragility", "robustness"} and len(target_names) > 1:
        add("main", "multi_stability_comparison", metric=op)
        for t in target_names:
            add(f"{family}_{_safe_key(t)}", family, target=t, mode="single_observable")
        return views

    if op == "regime_discovery":
        features = list(result.summary.get("features") or [])
        targets = list(result.summary.get("targets") or target_names)
        # Two features can be shown directly; higher-dimensional clustering needs a
        # projection to avoid pretending that the first two columns define regimes.
        if len(features) == 2:
            add("main", "regime_pair", x=features[0], y=features[1])
        else:
            add("main", "regime_pca")
        add("regime_profiles", "regime_profiles")
        add("regime_k_selection", "regime_k_selection")
        if len(features) > 2:
            add("regime_pca_loadings", "regime_pca_loadings")
        # Direct observable pair views are often the biologically interpretable
        # complement to PCA.  Cap at six to keep output usable for many targets.
        existing = {(views[0].get("x"), views[0].get("y"))} if views else set()
        for a, b in list(combinations(targets, 2))[:6]:
            if (a, b) in existing or (b, a) in existing:
                continue
            add(f"regime_{_safe_key(a)}_vs_{_safe_key(b)}", "regime_pair", x=a, y=b)
        return views

    if op in {"tradeoff", "pareto"} and len(target_names) > 2:
        for i, (a, b) in enumerate(list(combinations(target_names, 2))[:6]):
            add("main" if i == 0 else f"pareto_{_safe_key(a)}_vs_{_safe_key(b)}", family, x=a, y=b)
        return views

    add("main", family)
    return views

def plan_visualization(
    result: Any,
    spec: dict[str, Any],
    variables: list[str],
    scoped_df: pd.DataFrame,
    full_df: pd.DataFrame | None,
    evidence_backend: str,
) -> dict[str, Any]:
    op = str((spec.get("operator") or {}).get("name") or result.summary.get("operator") or "")
    vis_cfg = spec.get("visualization") or {}
    mode = str(vis_cfg.get("mode", "auto"))
    family = str(vis_cfg.get("family") or OPERATOR_VISUAL_FAMILY.get(op, "summary"))
    if mode not in {"auto", "override"}:
        raise ValueError("visualization.mode must be auto|override")
    if mode == "override" and not vis_cfg.get("family"):
        raise ValueError("visualization.mode=override requires visualization.family")

    n_scope = int(len(scoped_df))
    n_total = int(len(full_df)) if full_df is not None else n_scope
    q_cfg = spec.get("question") or {}
    scientific_question = str(q_cfg.get("text") or _question_for_operator(op, spec, variables, n_scope, n_total))
    short_title = str(q_cfg.get("title") or scientific_question.rstrip("?"))

    target_names = [str(t.get("name")) for t in spec.get("targets", []) if t.get("name")]
    variable_labels = {v: variable_label(v, spec) for v in variables}
    target_labels = {t: target_label(t, spec) for t in target_names}

    qc = result.summary.get("surrogate_qc") if isinstance(result.summary, dict) else None
    result_evidence = str(result.summary.get("evidence") or "") if isinstance(result.summary, dict) else ""
    if result_evidence == "exact_landscape_binned_fallback":
        evidence = "Stored exact landscape (binned fallback; surrogate QC failed)"
        qc = None
    elif result_evidence == "validated_surrogate_interpolation":
        evidence = "Validated surrogate interpolation over stored landscape"
        if qc:
            evidence += " (QC passed)"
    elif result_evidence == "mixed_exact_and_validated_surrogate":
        evidence = "Mixed evidence by observable: validated surrogate where QC passed, stored exact-landscape fallback where QC failed"
        qc = None
    else:
        evidence = BACKEND_LABELS.get(evidence_backend, evidence_backend)
        if evidence_backend in {"surrogate_assisted", "hybrid"} and qc:
            evidence += " (surrogate QC passed for this analysis)"

    views = _visualization_views(op, family, result, spec, target_names)

    return {
        "schema": "gem_visualization_spec_v0.5.0",
        "question": {
            "text": scientific_question,
            "title": short_title,
            "why_it_matters": str(q_cfg.get("why_it_matters") or _purpose_for_family(family)),
        },
        "problem": {
            "operator": op,
            "target_names": target_names,
            "target_labels": target_labels,
            "variables": list(variables),
            "variable_labels": variable_labels,
            "rows_in_scope": n_scope,
            "rows_total": n_total,
            "scope": spec.get("scope") or {},
            "parameters": (spec.get("operator") or {}).get("parameters") or {},
        },
        "evidence": {
            "backend": evidence_backend,
            "label": evidence,
            "surrogate_qc": qc,
            "new_fba_calls": 0,
        },
        "visualization": {
            "mode": mode,
            "family": family,
            "purpose": _purpose_for_family(family),
            "views": views,
            "publication_formats": list((spec.get("outputs") or {}).get("publication_formats", ["png", "pdf", "svg"])),
            "dpi": int((spec.get("outputs") or {}).get("dpi", 600)),
        },
        "answer_metrics": _answer_metrics(op),
        "guardrails": list(result.warnings or []),
    }
