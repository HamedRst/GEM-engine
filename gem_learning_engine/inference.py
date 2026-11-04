from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

from .config import ModelConfig, OptimizationConfig

_STRONG_PRODUCT_TERMS = (
    "antibody", "scfv", "fab", "igg", "immunoglob", "recombinant",
    "mab", "nanobody", "vhh", "therapeutic", "insulin",
)
_GENERIC_PRODUCT_TERMS = ("protein", "product", "peptide")
_PRODUCT_TERMS = _STRONG_PRODUCT_TERMS + _GENERIC_PRODUCT_TERMS
_BIOMASS_TERMS = ("biomass", "growth", "bio1", "bof")
_TECHNICAL_PRODUCT_EXCLUSIONS = (
    "atpm", "maintenance", "h2o", "proton", "sink needed", "biomass", "growth",
)


@dataclass(frozen=True)
class ReactionCandidate:
    reaction_id: str
    reaction_name: str
    score: float
    reasons: list[str]
    linked_reactions: list[str]


@dataclass(frozen=True)
class RoleSelection:
    reaction_id: str | None
    confidence: str
    score: float
    margin: float
    source: str


@dataclass(frozen=True)
class ModelInferenceReport:
    model_id: str | None
    model_name: str | None
    biomass: RoleSelection
    product: RoleSelection
    embedded_objective: dict[str, float]
    selected_optimization_mode: str
    biomass_candidates: list[ReactionCandidate]
    product_candidates: list[ReactionCandidate]
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        temporary.replace(path)


def _objective_coefficients(model: Any) -> dict[str, float]:
    try:
        from cobra.util.solver import linear_reaction_coefficients

        return {
            reaction.id: float(coefficient)
            for reaction, coefficient in linear_reaction_coefficients(model).items()
            if abs(float(coefficient)) > 0
        }
    except Exception:
        pass
    try:
        return {
            reaction.id: float(coefficient)
            for reaction, coefficient in model.objective.items()
            if abs(float(coefficient)) > 0
        }
    except Exception:
        return {
            reaction.id: float(getattr(reaction, "objective_coefficient", 0.0))
            for reaction in model.reactions
            if abs(float(getattr(reaction, "objective_coefficient", 0.0))) > 0
        }


def _normalized(*values: str | None) -> str:
    return " ".join(str(value or "").lower().replace("_", " ") for value in values)


def _confidence(score: float, margin: float, *, product: bool = False) -> str:
    high = 65.0 if product else 75.0
    medium = 35.0 if product else 45.0
    if score >= high and margin >= 12:
        return "high"
    if score >= medium and margin >= 5:
        return "medium"
    return "low"


def _selection(candidates: list[ReactionCandidate], source: str) -> RoleSelection:
    if not candidates:
        return RoleSelection(None, "low", 0.0, 0.0, source)
    first = candidates[0]
    second = candidates[1].score if len(candidates) > 1 else 0.0
    margin = float(first.score - second)
    return RoleSelection(first.reaction_id, _confidence(first.score, margin), first.score, margin, source)


def _reaction_metabolite_text(reaction: Any) -> str:
    values: list[str] = []
    for metabolite in reaction.metabolites:
        values.extend([metabolite.id, metabolite.name or ""])
    return _normalized(*values)


def rank_biomass_candidates(model: Any) -> list[ReactionCandidate]:
    objective = _objective_coefficients(model)
    candidates: list[ReactionCandidate] = []
    amino_acids = {"ala", "arg", "asn", "asp", "cys", "gln", "glu", "gly", "his", "ile", "leu", "lys", "met", "phe", "pro", "ser", "thr", "trp", "tyr", "val"}
    for reaction in model.reactions:
        text = _normalized(reaction.id, reaction.name, getattr(reaction, "subsystem", None))
        score = 0.0
        reasons: list[str] = []
        metabolite_ids = [metabolite.id.lower() for metabolite in reaction.metabolites]
        aa_count = sum(any(re.search(rf"(^|_){term}(_|$)", mid) for term in amino_acids) for mid in metabolite_ids)
        biomass_named = "biomass" in text or any(term in text for term in _BIOMASS_TERMS)
        biomass_structural = len(metabolite_ids) >= 20 and aa_count >= 10
        if reaction.id in objective:
            if biomass_named or biomass_structural:
                score += 100.0
                reasons.append("embedded biomass-like model objective")
            else:
                score += 5.0
                reasons.append("embedded objective only; weak biomass evidence")
        if "biomass" in text:
            score += 85.0
            reasons.append("biomass term in reaction identifier/name")
        elif any(term in text for term in _BIOMASS_TERMS):
            score += 30.0
            reasons.append("growth-related term")
        if len(metabolite_ids) >= 20:
            score += 12.0
            reasons.append("large biomass-like stoichiometry")
        if aa_count >= 10:
            score += 12.0
            reasons.append("consumes many amino acids")
        if any(mid.startswith("atp_") for mid in metabolite_ids) and any(mid.startswith("adp_") for mid in metabolite_ids):
            score += 5.0
            reasons.append("ATP maintenance component")
        if score > 0:
            candidates.append(ReactionCandidate(reaction.id, reaction.name or "", score, reasons, []))
    return sorted(candidates, key=lambda item: (-item.score, item.reaction_id))


def rank_product_candidates(model: Any) -> list[ReactionCandidate]:
    candidates: list[ReactionCandidate] = []
    for reaction in model.reactions:
        text = _normalized(reaction.id, reaction.name, getattr(reaction, "subsystem", None))
        metabolite_text = _reaction_metabolite_text(reaction)
        combined = f"{text} {metabolite_text}"
        score = 0.0
        reasons: list[str] = []
        linked: list[str] = []
        is_demand = reaction.id.lower().startswith("dm_") or "demand" in text
        is_sink = reaction.id.lower().startswith("sk_") or "sink" in text
        is_exchange = reaction.id.lower().startswith("ex_")
        strong_terms = sorted({term for term in _STRONG_PRODUCT_TERMS if term in combined})
        generic_terms = sorted({term for term in _GENERIC_PRODUCT_TERMS if term in combined})
        matched_terms = strong_terms + generic_terms
        if strong_terms:
            score += 45.0 + min(20.0, 5.0 * len(strong_terms))
            reasons.append("strong product terms: " + ", ".join(strong_terms))
        if generic_terms:
            score += 18.0
            reasons.append("generic product terms: " + ", ".join(generic_terms))
        if is_demand:
            score += 25.0
            reasons.append("demand reaction")
        elif is_sink:
            score += 18.0
            reasons.append("sink reaction")
        elif is_exchange:
            score += 12.0
            reasons.append("exchange reaction")
        if len(reaction.metabolites) == 1:
            metabolite = next(iter(reaction.metabolites))
            producers = []
            for linked_reaction in metabolite.reactions:
                coefficient = float(linked_reaction.metabolites.get(metabolite, 0.0))
                if linked_reaction.id != reaction.id and coefficient > 0:
                    producers.append(linked_reaction)
            linked = sorted(item.id for item in producers)
            producer_text = _normalized(*[item.id for item in producers], *[item.name for item in producers])
            producer_strong = sorted({term for term in _STRONG_PRODUCT_TERMS if term in producer_text})
            producer_generic = sorted({term for term in _GENERIC_PRODUCT_TERMS if term in producer_text})
            if producer_strong:
                score += 35.0
                reasons.append("linked synthesis reaction contains strong product terms")
            elif producer_generic:
                score += 20.0
                reasons.append("linked synthesis reaction contains generic product terms")
            if producers and (is_demand or is_sink or is_exchange):
                score += 8.0
                reasons.append("terminal boundary for a produced metabolite")
        if any(term in combined for term in _TECHNICAL_PRODUCT_EXCLUSIONS):
            score -= 35.0
            reasons.append("technical/biomass boundary penalty")
        if score > 0:
            candidates.append(ReactionCandidate(reaction.id, reaction.name or "", score, reasons, linked[:10]))
    return sorted(candidates, key=lambda item: (-item.score, item.reaction_id))


def infer_model_roles(model: Any, model_config: ModelConfig, optimization: OptimizationConfig) -> ModelInferenceReport:
    objective = _objective_coefficients(model)
    biomass_candidates = rank_biomass_candidates(model)
    product_candidates = rank_product_candidates(model)
    warnings: list[str] = []

    if model_config.biomass_reaction_id:
        biomass = RoleSelection(model_config.biomass_reaction_id, "explicit", 1e9, 1e9, "user override")
    else:
        biomass = _selection(biomass_candidates, "automatic inference")

    if model_config.product_reaction_id:
        product = RoleSelection(model_config.product_reaction_id, "explicit", 1e9, 1e9, "user override")
    else:
        product = _selection(product_candidates, "automatic inference")
        product = replace(product, confidence=_confidence(product.score, product.margin, product=True))

    if biomass.confidence == "low":
        warnings.append("Biomass reaction inference is low confidence; an explicit override is recommended.")
    if product.reaction_id is None or product.confidence == "low":
        warnings.append("No high-confidence product boundary was detected; native-objective mode will be used unless overridden.")

    requested_mode = optimization.mode
    if requested_mode == "auto":
        product_is_acceptable = product.confidence in {"high", "explicit"}
        if optimization.allow_medium_confidence_product and not optimization.require_high_confidence_product:
            product_is_acceptable = product_is_acceptable or product.confidence == "medium"
        if biomass.reaction_id and product.reaction_id and product_is_acceptable:
            selected_mode = "growth_constrained_product"
        else:
            selected_mode = "native_objective"
    else:
        selected_mode = requested_mode

    return ModelInferenceReport(
        model_id=getattr(model, "id", None),
        model_name=getattr(model, "name", None),
        biomass=biomass,
        product=product,
        embedded_objective=objective,
        selected_optimization_mode=selected_mode,
        biomass_candidates=biomass_candidates[:10],
        product_candidates=product_candidates[:10],
        warnings=warnings,
    )


def resolve_inferred_config(
    model: Any,
    model_config: ModelConfig,
    optimization: OptimizationConfig,
) -> tuple[ModelConfig, OptimizationConfig, ModelInferenceReport]:
    report = infer_model_roles(model, model_config, optimization)
    biomass_id = model_config.biomass_reaction_id or report.biomass.reaction_id
    product_id = model_config.product_reaction_id or report.product.reaction_id
    objective_id = model_config.objective_reaction_id
    if objective_id is None and report.selected_optimization_mode in {"growth_constrained_product", "maximize_biomass"}:
        objective_id = biomass_id
    elif objective_id is None and report.selected_optimization_mode == "maximize_product":
        objective_id = product_id

    selected_mode = report.selected_optimization_mode
    if selected_mode in {"growth_constrained_product", "maximize_biomass"} and not biomass_id:
        raise ValueError(f"Optimization mode {selected_mode} requires a biomass reaction, but none was detected.")
    if selected_mode in {"growth_constrained_product", "maximize_product"} and not product_id:
        raise ValueError(f"Optimization mode {selected_mode} requires a product reaction, but none was detected.")

    resolved_model = replace(
        model_config,
        objective_reaction_id=objective_id,
        biomass_reaction_id=biomass_id,
        product_reaction_id=product_id,
    )
    resolved_optimization = replace(optimization, mode=selected_mode)
    return resolved_model, resolved_optimization, report
