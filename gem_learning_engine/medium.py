from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from .config import SamplingConfig, SimulationConfig
from .model_io import get_exchange_reactions

LOGGER = logging.getLogger("gem_learning_engine")


@dataclass(frozen=True)
class ExchangeBound:
    reaction_id: str
    requested_min_uptake: float
    requested_max_uptake: float
    min_uptake: float
    max_uptake: float
    original_capacity: float
    baseline_uptake: float
    source: str


@dataclass(frozen=True)
class MediumSpace:
    bounds: tuple[ExchangeBound, ...]
    fixed_uptakes: dict[str, float]
    fixed_reasons: dict[str, str]
    exchange_ids: tuple[str, ...]
    excluded_reactions: dict[str, str]

    @property
    def reaction_ids(self) -> list[str]:
        return [bound.reaction_id for bound in self.bounds]

    @property
    def output_reaction_ids(self) -> list[str]:
        return list(dict.fromkeys([*self.reaction_ids, *self.fixed_uptakes.keys()]))

    def to_frame(self) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        for index, bound in enumerate(self.bounds):
            rows.append(
                {
                    "medium_feature_index": index,
                    "reaction_id": bound.reaction_id,
                    "role": "variable",
                    "requested_min_uptake": bound.requested_min_uptake,
                    "requested_max_uptake": bound.requested_max_uptake,
                    "effective_min_uptake": bound.min_uptake,
                    "effective_max_uptake": bound.max_uptake,
                    "original_capacity": bound.original_capacity,
                    "baseline_uptake": bound.baseline_uptake,
                    "source": bound.source,
                    "reason": None,
                }
            )
        offset = len(rows)
        for position, (reaction_id, value) in enumerate(sorted(self.fixed_uptakes.items())):
            rows.append(
                {
                    "medium_feature_index": offset + position,
                    "reaction_id": reaction_id,
                    "role": "fixed",
                    "requested_min_uptake": value,
                    "requested_max_uptake": value,
                    "effective_min_uptake": value,
                    "effective_max_uptake": value,
                    "original_capacity": None,
                    "baseline_uptake": value,
                    "source": "baseline_or_override",
                    "reason": self.fixed_reasons.get(reaction_id),
                }
            )
        for reaction_id, reason in sorted(self.excluded_reactions.items()):
            rows.append(
                {
                    "medium_feature_index": None,
                    "reaction_id": reaction_id,
                    "role": "excluded",
                    "requested_min_uptake": None,
                    "requested_max_uptake": None,
                    "effective_min_uptake": None,
                    "effective_max_uptake": None,
                    "original_capacity": None,
                    "baseline_uptake": None,
                    "source": "configuration_or_native_bounds",
                    "reason": reason,
                }
            )
        return pd.DataFrame(rows)




_BACKGROUND_IDS = {
    "h", "h2o", "co2", "hco3", "ca2", "cl", "cobalt2", "cu2", "fe2", "fe3",
    "k", "mg2", "mn2", "mobd", "na1", "ni2", "sel", "slnt", "tungs", "zn2",
}
_BACKGROUND_TERMS = (
    "water", "proton", "carbon dioxide", "bicarbonate", "calcium", "chloride",
    "cobalt", "copper", "iron", "potassium", "magnesium", "manganese",
    "molybdate", "sodium", "nickel", "selen", "tungstate", "zinc",
)
_COFACTOR_TERMS = (
    "vitamin", "cobalamin", "biotin", "thiamin", "riboflavin", "folate",
    "pantothen", "pyridox", "nicotinamide", "cofactor",
)

# Inorganic compounds that are legitimate environmental design variables.
# These names are needed for GEMs such as yeast-GEM that use opaque reaction
# and metabolite identifiers (for example r_1992 instead of EX_o2_e).
_VARIABLE_INORGANIC_TERMS = (
    "oxygen",
    "ammonium",
    "ammonia",
    "nitrate",
    "nitrite",
    "phosphate",
    "orthophosphate",
    "sulfate",
    "sulphate",
    "sulfite",
    "sulphite",
)

_VARIABLE_INORGANIC_IDS = {
    "o2", "nh4", "nh3", "no2", "no3", "pi", "po4",
    "so3", "so4",
}

def _exchange_metabolite(reaction: Any) -> Any | None:
    metabolites = list(reaction.metabolites)
    return metabolites[0] if len(metabolites) == 1 else None

def _base_metabolite_id(value: str) -> str:
    lowered = value.lower().replace("[", "_").replace("]", "")
    for suffix in ("_e", "_extracellular", "_external", "_p"):
        if lowered.endswith(suffix):
            return lowered[: -len(suffix)]
    return lowered


def _formula_elements(formula: str) -> set[str]:
    """Return actual element symbols from a molecular formula.

    This distinguishes carbon C from symbols such as Ca, Cu, Cl, Co, etc.
    """
    return set(re.findall(r"[A-Z][a-z]?", str(formula or "")))


def classify_baseline_exchange(model: Any, reaction_id: str) -> tuple[str, str]:
    """Classify a baseline uptake as variable nutrient or fixed background.

    This conservative policy is deliberately auditable and overrideable. It
    avoids treating solvent, charge-balancing ions, trace metals, and vitamins
    as broad continuous optimization variables in automatic runs.
    """
    reaction = model.reactions.get_by_id(reaction_id)
    metabolite = _exchange_metabolite(reaction)
    if metabolite is None:
        return "variable", "multi-metabolite exchange retained as variable"
    base_id = _base_metabolite_id(str(metabolite.id))
    text = f"{base_id} {getattr(metabolite, 'name', '') or ''}".lower()
    formula = str(getattr(metabolite, "formula", "") or "")
    # Background solvent, charge-balancing ions, and trace metals remain fixed.
    if base_id in _BACKGROUND_IDS or any(term in text for term in _BACKGROUND_TERMS):
        return "fixed", "automatic background solvent/ion classification"

    # Vitamins/cofactors remain fixed unless explicitly overridden by the user.
    if any(term in text for term in _COFACTOR_TERMS):
        return "fixed", "automatic vitamin/cofactor classification"

    # Recognize legitimate inorganic nutrients from either semantic names or
    # conventional metabolite IDs. This makes the classifier work with GEMs
    # whose reaction/metabolite identifiers are opaque.
    if (
        base_id in _VARIABLE_INORGANIC_IDS
        or any(term in text for term in _VARIABLE_INORGANIC_TERMS)
    ):
        return "variable", "automatic inorganic nutrient/gas classification"

    # Parse element symbols correctly. The previous character-based test
    # confused Ca, Cu, Cl, Co, etc. with carbon because their formulas contain
    # the character 'C'. Carbon is present only when element symbol C occurs.
    elements = _formula_elements(formula)
    if elements and "C" not in elements:
        return "fixed", "automatic inorganic background classification"

    return "variable", "automatic nutrient/gas classification"


def original_exchange_bounds(model: Any) -> dict[str, tuple[float, float]]:
    return {
        reaction_id: tuple(map(float, model.reactions.get_by_id(reaction_id).bounds))
        for reaction_id in get_exchange_reactions(model)
    }


def _capacity_from_bounds(lower_bound: float) -> float:
    return max(0.0, -float(lower_bound))


def build_medium_space(model: Any, sampling_config: SamplingConfig, simulation_config: SimulationConfig) -> MediumSpace:
    """Discover a safe medium search space from the loaded GEM.

    The default scope is ``model_medium`` because an exchange reaction being
    present does not imply that it is a legitimate culture-medium variable.
    Manual IDs are optional overrides, not a per-species requirement.
    """
    exchange_ids = get_exchange_reactions(model)
    if not exchange_ids:
        raise ValueError("No exchange reactions were discovered.")
    exchange_set = set(exchange_ids)
    baseline = {
        key: float(value)
        for key, value in (getattr(model, "medium", {}) or {}).items()
        if key in exchange_set
    }
    original = original_exchange_bounds(model)

    configured = (
        set(sampling_config.bounds)
        | set(sampling_config.fixed_reactions)
        | set(sampling_config.include_reaction_ids)
        | set(sampling_config.exclude_reaction_ids)
    )
    unknown = configured - exchange_set
    if unknown:
        raise KeyError(f"Configured exchange IDs are not model exchanges: {sorted(unknown)[:10]}")

    fixed_reasons: dict[str, str] = {}
    if sampling_config.discovery_scope == "model_medium":
        candidates = set(baseline)
        if sampling_config.automatic_medium_policy == "nutrients":
            for reaction_id in list(candidates):
                role, reason = classify_baseline_exchange(model, reaction_id)
                if role == "fixed":
                    candidates.remove(reaction_id)
                    fixed_reasons[reaction_id] = reason
    elif sampling_config.discovery_scope == "all_uptake_capable":
        candidates = {reaction_id for reaction_id, (lb, _ub) in original.items() if lb < 0}
    elif sampling_config.discovery_scope == "all_exchanges":
        candidates = set(exchange_ids)
    else:
        raise ValueError(f"Unknown discovery scope: {sampling_config.discovery_scope}")

    candidates.update(sampling_config.include_reaction_ids)
    candidates.update(sampling_config.bounds)
    for reaction_id in set(sampling_config.include_reaction_ids) | set(sampling_config.bounds):
        fixed_reasons.pop(reaction_id, None)
    excluded: dict[str, str] = {reaction_id: "explicitly_excluded" for reaction_id in sampling_config.exclude_reaction_ids}
    candidates.difference_update(sampling_config.exclude_reaction_ids)
    candidates.difference_update(sampling_config.fixed_reactions)

    fixed = dict(baseline)
    for reaction_id in candidates:
        fixed.pop(reaction_id, None)

    for reaction_id, requested_value in sampling_config.fixed_reactions.items():
        fixed_reasons[reaction_id] = "explicit fixed-reaction override"
        requested = float(requested_value)
        native_lb, native_ub = original[reaction_id]
        capacity = _capacity_from_bounds(native_lb)
        forced_minimum = max(0.0, -native_ub) if native_ub < 0 else 0.0
        effective = requested
        if simulation_config.respect_original_exchange_bounds:
            effective = min(max(requested, forced_minimum), capacity)
        if effective != requested:
            LOGGER.warning("Fixed uptake %s adjusted from %.6g to %.6g.", reaction_id, requested, effective)
        fixed[reaction_id] = effective

    bounds: list[ExchangeBound] = []
    for reaction_id in sorted(candidates):
        lower_bound, upper_bound = original[reaction_id]
        capacity = _capacity_from_bounds(lower_bound)
        forced_minimum = max(0.0, -upper_bound) if upper_bound < 0 else 0.0
        baseline_uptake = baseline.get(reaction_id, capacity)

        if reaction_id in sampling_config.bounds:
            specification = sampling_config.bounds[reaction_id]
            requested_minimum = float(specification["min"])
            requested_maximum = float(specification["max"])
            source = "override"
        else:
            reference = baseline_uptake
            if reference <= 0 and (
                sampling_config.discovery_scope == "all_exchanges"
                or reaction_id in sampling_config.include_reaction_ids
            ):
                reference = sampling_config.closed_exchange_max_uptake
            if sampling_config.auto_max_uptake_cap is not None:
                reference = min(reference, float(sampling_config.auto_max_uptake_cap))
            requested_minimum = sampling_config.auto_min_fraction * reference
            requested_maximum = sampling_config.auto_max_fraction * reference
            source = "automatic_capped" if reference < baseline_uptake else "automatic"

        effective_minimum, effective_maximum = requested_minimum, requested_maximum
        if simulation_config.respect_original_exchange_bounds:
            effective_minimum = max(effective_minimum, forced_minimum)
            effective_maximum = min(effective_maximum, capacity)

        if effective_minimum > effective_maximum:
            raise ValueError(
                f"Sampling range for {reaction_id} does not intersect its native uptake range: "
                f"requested=[{requested_minimum}, {requested_maximum}], "
                f"native=[{forced_minimum}, {capacity}]"
            )
        if effective_maximum <= 0 and requested_maximum > 0 and simulation_config.respect_original_exchange_bounds:
            excluded[reaction_id] = "native_model_does_not_permit_uptake"
            LOGGER.warning("Excluding %s because its native bounds do not permit uptake.", reaction_id)
            continue

        bounds.append(
            ExchangeBound(
                reaction_id=reaction_id,
                requested_min_uptake=requested_minimum,
                requested_max_uptake=requested_maximum,
                min_uptake=effective_minimum,
                max_uptake=effective_maximum,
                original_capacity=capacity,
                baseline_uptake=baseline_uptake,
                source=source,
            )
        )

    if not bounds:
        raise ValueError(
            "Automatic discovery found no sampleable uptake reactions. Check model.medium, "
            "discovery_scope, or include_reaction_ids."
        )
    return MediumSpace(
        bounds=tuple(bounds),
        fixed_uptakes=fixed,
        fixed_reasons=fixed_reasons,
        exchange_ids=tuple(exchange_ids),
        excluded_reactions=excluded,
    )


class MediumApplier:
    """Apply sampled uptake magnitudes in-place while reusing one solver model."""

    def __init__(self, model: Any, medium_space: MediumSpace, simulation_config: SimulationConfig):
        self.model = model
        self.medium_space = medium_space
        self.simulation_config = simulation_config
        self.original_bounds = original_exchange_bounds(model)
        self.variable_ids = medium_space.reaction_ids
        self.output_ids = medium_space.output_reaction_ids
        self._prepare_static_medium()

    def _prepare_static_medium(self) -> None:
        for reaction_id, bounds in self.original_bounds.items():
            self.model.reactions.get_by_id(reaction_id).bounds = bounds
        if self.simulation_config.close_unlisted_exchanges:
            active = set(self.variable_ids) | set(self.medium_space.fixed_uptakes)
            for reaction_id in self.medium_space.exchange_ids:
                if reaction_id in active:
                    continue
                reaction = self.model.reactions.get_by_id(reaction_id)
                lower_bound, upper_bound = self.original_bounds[reaction_id]
                if lower_bound <= 0 <= upper_bound:
                    reaction.bounds = (0.0, upper_bound)
                elif upper_bound < 0:
                    LOGGER.warning("Forced exchange %s cannot be closed safely; preserving native bounds.", reaction_id)
        for reaction_id, uptake in self.medium_space.fixed_uptakes.items():
            self._set_uptake(reaction_id, uptake)

    def _set_uptake(self, reaction_id: str, uptake: float) -> float:
        uptake = max(0.0, float(uptake))
        reaction = self.model.reactions.get_by_id(reaction_id)
        original_lower, original_upper = self.original_bounds[reaction_id]
        proposed_lower = -uptake
        if self.simulation_config.respect_original_exchange_bounds:
            proposed_lower = max(proposed_lower, original_lower)
        if proposed_lower > original_upper:
            raise ValueError(
                f"Cannot apply uptake {uptake} to {reaction_id}: lower bound {proposed_lower} "
                f"would exceed upper bound {original_upper}."
            )
        reaction.bounds = (proposed_lower, original_upper)
        return max(0.0, -float(reaction.lower_bound))

    def apply(self, sampled_medium: Mapping[str, float]) -> dict[str, float]:
        for reaction_id in self.variable_ids:
            if reaction_id not in sampled_medium:
                raise KeyError(f"Sample is missing variable exchange {reaction_id}")
            self._set_uptake(reaction_id, sampled_medium[reaction_id])
        return {
            reaction_id: max(0.0, -float(self.model.reactions.get_by_id(reaction_id).lower_bound))
            for reaction_id in self.output_ids
        }
