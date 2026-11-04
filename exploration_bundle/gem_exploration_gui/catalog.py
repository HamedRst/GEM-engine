from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ParameterField:
    key: str
    label: str
    kind: str = "number"  # number|integer|boolean|choice|variable|variables2|features|sample_sizes|json|column
    default: Any = None
    help: str = ""
    options: tuple[str, ...] = ()
    minimum: float | None = None
    maximum: float | None = None


@dataclass(frozen=True)
class QuestionDefinition:
    operator: str
    title: str
    category: str
    question: str
    why: str
    backend: str
    visualization: str
    target_count: int = 1
    target_optional: bool = False
    variable_policy: str = "all_or_list"
    parameters: tuple[ParameterField, ...] = field(default_factory=tuple)


def P(key, label, kind="number", default=None, help="", options=(), minimum=None, maximum=None):
    return ParameterField(key, label, kind, default, help, tuple(options), minimum, maximum)


QUESTIONS: dict[str, QuestionDefinition] = {
    "describe": QuestionDefinition(
        "describe", "Describe a phenotype", "Describe & query",
        "What does the selected phenotype distribution look like across the chosen landscape scope?",
        "Establishes the basic range, center, spread, and tails before more specific analyses.",
        "landscape", "distribution", 1,
    ),
    "query": QuestionDefinition(
        "query", "Find designs matching conditions", "Describe & query",
        "Which exact sampled designs satisfy the conditions I define?",
        "Turns a biological question into a reproducible exact-landscape filter rather than an optimization problem.",
        "landscape", "query_context", 0,
        parameters=(
            P("sort_by", "Sort result by", "column", None),
            P("ascending", "Ascending sort", "boolean", False),
            P("max_saved_points", "Maximum saved designs", "integer", 10000, minimum=1),
        ),
    ),
    "association": QuestionDefinition(
        "association", "Association with a phenotype", "Drivers & sensitivity",
        "Which design variables co-vary most strongly with the selected phenotype?",
        "Provides model-free Pearson and Spearman associations before using a surrogate-based influence model.",
        "landscape", "ranked_effect", 1,
    ),
    "correlation_matrix": QuestionDefinition(
        "correlation_matrix", "Correlation structure", "Describe & query",
        "How are selected phenotype and design quantities associated with one another?",
        "Reveals coupled variables, redundant features, and phenotype relationships.",
        "landscape", "correlation_heatmap", 0,
        parameters=(
            P("features", "Features", "features", []),
            P("max_features", "Maximum features", "integer", 40, minimum=2),
        ),
    ),
    "global_sensitivity": QuestionDefinition(
        "global_sensitivity", "Global phenotype drivers", "Drivers & sensitivity",
        "Which design variables most strongly influence the selected phenotype across the chosen scope?",
        "Ranks global controllers using cross-validated surrogate permutation importance with fail-closed QC.",
        "surrogate_assisted", "ranked_importance", 1,
    ),
    "local_sensitivity": QuestionDefinition(
        "local_sensitivity", "Local phenotype sensitivity", "Drivers & sensitivity",
        "Near a specific design context, which variables change the phenotype most strongly and in which direction?",
        "Distinguishes local control from global average importance.",
        "surrogate_assisted", "signed_effect", 1,
        parameters=(P("step_fraction", "Perturbation step (fraction of variable range)", "number", 0.01, minimum=0.0001, maximum=0.5),),
    ),
    "sensitivity_map": QuestionDefinition(
        "sensitivity_map", "Sensitivity across contexts", "Drivers & sensitivity",
        "Where in the design landscape is the phenotype locally sensitive, and to which variable?",
        "Shows whether control changes across backgrounds rather than assuming a single global controller.",
        "surrogate_assisted", "local_sensitivity_heatmap", 1,
        parameters=(
            P("anchor_count", "Anchor contexts", "integer", 300, minimum=10),
            P("step_fraction", "Perturbation step", "number", 0.01, minimum=0.0001, maximum=0.5),
        ),
    ),
    "interaction": QuestionDefinition(
        "interaction", "Pairwise variable interactions", "Drivers & sensitivity",
        "Which pairs of design variables show the strongest non-additive effect on the phenotype?",
        "Identifies combinations whose joint effect cannot be explained by adding their individual effects.",
        "surrogate_assisted", "interaction_matrix", 1,
        parameters=(
            P("step_fraction", "Interaction step", "number", 0.03, minimum=0.0001, maximum=0.5),
            P("anchor_count", "Anchor contexts", "integer", 128, minimum=10),
            P("max_pairs", "Maximum pairs", "integer", 60, minimum=1),
        ),
    ),
    "fragility": QuestionDefinition(
        "fragility", "Phenotype fragility", "Stability & boundaries",
        "Where do nearby sampled designs show large changes in the selected phenotype?",
        "Locates locally fragile parts of the stored exact landscape without pretending they are new perturbation experiments.",
        "landscape", "fragility_distribution", 1,
        parameters=(P("neighbors", "Nearest neighbors", "integer", 20, minimum=2),),
    ),
    "robustness": QuestionDefinition(
        "robustness", "Phenotype robustness", "Stability & boundaries",
        "Where is the selected phenotype locally stable across nearby sampled design conditions?",
        "Quantifies descriptive local stability in the stored landscape.",
        "landscape", "robustness_distribution", 1,
        parameters=(P("neighbors", "Nearest neighbors", "integer", 20, minimum=2),),
    ),
    "threshold_region": QuestionDefinition(
        "threshold_region", "Phenotype threshold region", "Stability & boundaries",
        "What distinguishes designs that meet a phenotype threshold from designs that do not?",
        "Turns an engineering requirement into an interpretable region of the exact landscape.",
        "landscape", "threshold_region_effects", 1,
        parameters=(
            P("threshold", "Threshold value", "number", 1.0),
            P("direction", "Threshold direction", "choice", ">=", options=(">=", "<=", ">", "<")),
            P("max_saved_points", "Maximum saved designs", "integer", 10000, minimum=1),
        ),
    ),
    "threshold_boundary": QuestionDefinition(
        "threshold_boundary", "Phenotype threshold boundary", "Stability & boundaries",
        "Which exact sampled designs lie closest to a specified phenotype threshold?",
        "Maps the empirical boundary around a design requirement without extrapolating beyond stored evidence.",
        "landscape", "threshold_boundary_context", 1,
        parameters=(
            P("threshold", "Threshold value", "number", 1.0),
            P("n_points", "Boundary points", "integer", 200, minimum=10),
        ),
    ),
    "feasibility": QuestionDefinition(
        "feasibility", "Feasibility drivers", "Stability & boundaries",
        "Which design variables best distinguish feasible from infeasible sampled designs?",
        "Identifies the variables most associated with feasibility when both classes are actually present.",
        "surrogate_assisted", "feasibility_drivers", 0,
        parameters=(P("column", "Feasibility column", "column", "feasible"),),
    ),
    "regime_discovery": QuestionDefinition(
        "regime_discovery", "Phenotype regime discovery", "Structure & regimes",
        "Does the selected phenotype space contain reproducible candidate regimes, and where are they located?",
        "Screens for descriptive phenotype structure while reporting cluster quality rather than asserting discrete biology.",
        "landscape", "regime_map", 2,
        parameters=(P("max_k", "Maximum number of regimes", "integer", 8, minimum=2),),
    ),
    "dimensionality": QuestionDefinition(
        "dimensionality", "Phenotype/design dimensionality", "Structure & regimes",
        "How many latent dimensions are needed to represent the selected quantities?",
        "Measures whether a high-dimensional landscape is effectively controlled by a smaller number of axes.",
        "landscape", "dimensionality_curve", 0,
        parameters=(
            P("features", "Features", "features", []),
            P("variance_threshold", "Explained-variance threshold", "number", 0.90, minimum=0.5, maximum=0.999),
            P("max_components", "Maximum PCA components", "integer", 20, minimum=1),
        ),
    ),
    "tradeoff": QuestionDefinition(
        "tradeoff", "Multiobjective trade-off", "Trade-offs & Pareto",
        "What trade-off structure exists between the selected objectives, and where is the empirical Pareto frontier?",
        "Connects the full sampled phenotype cloud to its nondominated boundary.",
        "landscape", "tradeoff_frontier", 2,
    ),
    "pareto": QuestionDefinition(
        "pareto", "Empirical Pareto frontier", "Trade-offs & Pareto",
        "Which exact sampled designs form the empirical nondominated set for the selected objectives?",
        "Extracts the exact frontier contained in the stored landscape without claiming the true global optimum.",
        "landscape", "pareto_frontier", 2,
    ),
    "pareto_saturation": QuestionDefinition(
        "pareto_saturation", "Pareto sampling saturation", "Trade-offs & Pareto",
        "How much sampling is required to recover the empirical Pareto structure found in the full stored landscape?",
        "Tests whether the reusable landscape is already saturated for the multiobjective question.",
        "landscape", "pareto_saturation", 2,
        parameters=(
            P("sample_sizes", "Sample sizes", "sample_sizes", [500, 1000, 2500, 5000, 10000, 25000, 50000]),
            P("repeats", "Replicates per size", "integer", 10, minimum=2),
        ),
    ),
    "compare_regions": QuestionDefinition(
        "compare_regions", "Compare two design regions", "Trade-offs & Pareto",
        "How does the selected phenotype differ between two explicitly defined regions?",
        "Provides a direct distributional comparison and effect statistic between biologically meaningful scopes.",
        "landscape", "region_comparison", 1,
        parameters=(
            P("region_a", "Region A filters", "json", [{"column": "result:maximum_growth", "op": "top_quantile", "value": 0.20}]),
            P("region_b", "Region B filters", "json", [{"column": "result:maximum_growth", "op": "bottom_quantile", "value": 0.20}]),
        ),
    ),
    "coverage": QuestionDefinition(
        "coverage", "Design-space coverage", "Sampling & support",
        "How densely and uniformly does the stored landscape cover the selected design variables?",
        "Quantifies the spacing of exact sampled points before interpolating between them.",
        "landscape", "coverage_distribution", 0,
    ),
    "outlier": QuestionDefinition(
        "outlier", "Landscape outliers", "Structure & regimes",
        "Which sampled designs are unusual relative to the selected phenotype/design features, and where are they located?",
        "Highlights atypical exact samples for biological inspection rather than hiding them in aggregate statistics.",
        "landscape", "outlier_context", 1, target_optional=True,
        parameters=(P("contamination", "Expected outlier fraction", "number", 0.01, minimum=0.0001, maximum=0.25),),
    ),
    "point_query": QuestionDefinition(
        "point_query", "Evidence at an arbitrary design", "Continuous response & surrogate",
        "What exact-neighbor and surrogate evidence supports a user-specified design location?",
        "Combines nearest exact support with interpolation while keeping the two evidence types distinct.",
        "hybrid", "point_query_context", 1,
        parameters=(
            P("point", "Design point mapping", "json", {}),
            P("neighbors", "Nearest exact neighbors", "integer", 10, minimum=1),
        ),
    ),
    "response_curve": QuestionDefinition(
        "response_curve", "One-variable response curve", "Continuous response & surrogate",
        "How does the selected phenotype change as one design variable is varied around a fixed context?",
        "Makes a local continuous response explicit while showing it is surrogate-assisted interpolation.",
        "surrogate_assisted", "response_curve", 1,
        parameters=(
            P("variable", "Variable to scan", "variable", None),
            P("grid_points", "Grid points", "integer", 101, minimum=11),
        ),
    ),
    "marginal_response": QuestionDefinition(
        "marginal_response", "Marginal response across backgrounds", "Continuous response & surrogate",
        "How does the phenotype change with one variable on average across observed background designs?",
        "Separates an across-background marginal relationship from a single-anchor local response.",
        "surrogate_assisted", "marginal_response", 1,
        parameters=(
            P("variable", "Variable to scan", "variable", None),
            P("grid_points", "Grid points", "integer", 51, minimum=11),
            P("background_rows", "Background designs", "integer", 512, minimum=20),
        ),
    ),
    "surrogate_uncertainty": QuestionDefinition(
        "surrogate_uncertainty", "Surrogate support map", "Sampling & support",
        "Where is surrogate interpolation least well supported by the stored landscape?",
        "Identifies regions where interpolation should be treated cautiously before making design claims.",
        "surrogate_assisted", "surrogate_support_map", 1,
        parameters=(
            P("evaluation_rows", "Evaluation rows", "integer", 10000, minimum=100),
            P("top_points", "Highest-dispersion points to save", "integer", 500, minimum=10),
        ),
    ),
    "controller_scan": QuestionDefinition(
        "controller_scan", "Controller scan", "Drivers & sensitivity",
        "Which variable behaves most consistently as a monotonic control knob for the selected phenotype across backgrounds?",
        "Looks for practically controllable variables rather than only high global importance.",
        "surrogate_assisted", "controller_ranking", 1,
        parameters=(
            P("anchor_count", "Background anchors", "integer", 96, minimum=10),
            P("levels", "Levels per variable", "integer", 9, minimum=3),
        ),
    ),
    "threshold_switch_scan": QuestionDefinition(
        "threshold_switch_scan", "Threshold-switch controllers", "Stability & boundaries",
        "Which variable most reliably drives the phenotype across a specified threshold?",
        "Identifies candidate control knobs for meeting a quantitative engineering requirement.",
        "surrogate_assisted", "threshold_switch_ranking", 1,
        parameters=(
            P("threshold", "Threshold value", "number", 1.0),
            P("anchor_count", "Background anchors", "integer", 96, minimum=10),
            P("levels", "Levels per variable", "integer", 21, minimum=5),
        ),
    ),
    "response_surface": QuestionDefinition(
        "response_surface", "Two-variable response surface", "Continuous response & surrogate",
        "How does the phenotype vary jointly with two selected design variables around a fixed background?",
        "Visualizes nonlinear joint response in the most direct two-dimensional form.",
        "surrogate_assisted", "response_surface", 1,
        parameters=(
            P("variables", "Two variables", "variables2", []),
            P("grid_points", "Grid points per axis", "integer", 41, minimum=11),
        ),
    ),
    "sampling_saturation": QuestionDefinition(
        "sampling_saturation", "Design-space sampling saturation", "Sampling & support",
        "How does design-space coverage improve as the number of sampled designs increases?",
        "Tests whether additional passive landscape sampling still improves geometric coverage.",
        "landscape", "sampling_saturation", 0,
        parameters=(
            P("sample_sizes", "Sample sizes", "sample_sizes", [500, 1000, 2500, 5000, 10000, 25000, 50000]),
            P("repeats", "Replicates per size", "integer", 10, minimum=2),
            P("evaluation_points", "Coverage evaluation points", "integer", 5000, minimum=100),
        ),
    ),
}


CATEGORIES = [
    "Describe & query",
    "Drivers & sensitivity",
    "Stability & boundaries",
    "Structure & regimes",
    "Trade-offs & Pareto",
    "Sampling & support",
    "Continuous response & surrogate",
]


def questions_by_category() -> dict[str, list[QuestionDefinition]]:
    return {cat: [q for q in QUESTIONS.values() if q.category == cat] for cat in CATEGORIES}
