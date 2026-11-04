from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

from .config import ModelConfig, SolverConfig

LOGGER = logging.getLogger("gem_learning_engine")


def load_model(model_config: ModelConfig):
    """Load a COBRA model without creating or modifying exchange reactions."""
    import cobra

    model_path = Path(model_config.path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    fmt = model_config.format.lower()
    if fmt == "auto":
        suffix = model_path.suffix.lower()
        if suffix in {".xml", ".sbml"}:
            fmt = "sbml"
        elif suffix == ".json":
            fmt = "json"
        elif suffix == ".mat":
            fmt = "mat"
        else:
            raise ValueError(f"Could not infer model format from extension: {suffix}")

    if fmt == "sbml":
        model = cobra.io.read_sbml_model(str(model_path))
    elif fmt == "json":
        model = cobra.io.load_json_model(str(model_path))
    elif fmt == "mat":
        model = cobra.io.load_matlab_model(str(model_path))
    else:
        raise ValueError(f"Unsupported model format: {fmt}")

    reaction_ids = {reaction.id for reaction in model.reactions}
    if model_config.objective_reaction_id:
        if model_config.objective_reaction_id not in reaction_ids:
            raise KeyError(f"Objective reaction not found: {model_config.objective_reaction_id}")
        model.objective = model.reactions.get_by_id(model_config.objective_reaction_id)
    model.objective_direction = model_config.objective_direction
    return model


def configure_solver(model: Any, solver_config: SolverConfig) -> None:
    if solver_config.name:
        model.solver = solver_config.name
    configuration = getattr(model.solver, "configuration", None)
    if configuration is None:
        return
    if solver_config.timeout_seconds is not None and hasattr(configuration, "timeout"):
        configuration.timeout = solver_config.timeout_seconds
    tolerances = getattr(configuration, "tolerances", None)
    if tolerances is not None:
        if solver_config.feasibility_tolerance is not None and hasattr(tolerances, "feasibility"):
            tolerances.feasibility = solver_config.feasibility_tolerance
        if solver_config.optimality_tolerance is not None and hasattr(tolerances, "optimality"):
            tolerances.optimality = solver_config.optimality_tolerance


def get_exchange_reactions(model: Any) -> list[str]:
    """Discover existing exchange reactions from model structure.

    No species-specific list is needed and no new reaction is generated.
    """
    try:
        exchanges = list(model.exchanges)
        if exchanges:
            return sorted({reaction.id for reaction in exchanges})
    except Exception:
        pass

    detected: list[str] = []
    for reaction in model.reactions:
        is_boundary = bool(getattr(reaction, "boundary", False))
        if is_boundary and len(reaction.metabolites) == 1:
            detected.append(reaction.id)
        elif reaction.id.startswith("EX_"):
            detected.append(reaction.id)
    return sorted(dict.fromkeys(detected))


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
        coefficients: dict[str, float] = {}
        for reaction in model.reactions:
            try:
                coefficient = float(reaction.objective_coefficient)
            except Exception:
                coefficient = 0.0
            if coefficient:
                coefficients[reaction.id] = coefficient
        return coefficients


def model_fingerprint(model: Any, model_path: str | Path | None = None) -> str:
    """Hash the model file and structural identifiers for resume safety."""
    digest = hashlib.sha256()
    if model_path:
        path = Path(model_path)
        if path.exists():
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
    payload = {
        "id": getattr(model, "id", None),
        "reactions": sorted(reaction.id for reaction in model.reactions),
        "metabolites": sorted(metabolite.id for metabolite in model.metabolites),
        "genes": sorted(gene.id for gene in model.genes),
        "objective": _objective_coefficients(model),
        "direction": str(model.objective_direction),
    }
    digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def exchange_catalog(model: Any) -> pd.DataFrame:
    exchange_ids = get_exchange_reactions(model)
    baseline_medium = dict(getattr(model, "medium", {}) or {})
    rows: list[dict[str, Any]] = []
    for index, reaction_id in enumerate(exchange_ids):
        reaction = model.reactions.get_by_id(reaction_id)
        metabolites = list(reaction.metabolites)
        reaction_id_lower = reaction.id.lower()
        name_lower = (reaction.name or "").lower()
        likely_technical = reaction_id_lower.startswith(("dm_", "sk_")) or "demand" in name_lower or "sink" in name_lower
        rows.append(
            {
                "exchange_index": index,
                "reaction_id": reaction.id,
                "reaction_name": reaction.name,
                "metabolite_ids": ";".join(m.id for m in metabolites),
                "metabolite_names": ";".join((m.name or "") for m in metabolites),
                "metabolite_formulas": ";".join((m.formula or "") for m in metabolites),
                "metabolite_compartments": ";".join((m.compartment or "") for m in metabolites),
                "lower_bound": float(reaction.lower_bound),
                "upper_bound": float(reaction.upper_bound),
                "baseline_uptake": float(baseline_medium.get(reaction.id, max(0.0, -float(reaction.lower_bound)))),
                "in_model_medium": reaction.id in baseline_medium,
                "uptake_capable": bool(float(reaction.lower_bound) < 0),
                "secretion_capable": bool(float(reaction.upper_bound) > 0),
                "likely_technical_boundary": likely_technical,
            }
        )
    return pd.DataFrame(rows)


def reaction_catalog(model: Any) -> pd.DataFrame:
    exchange_ids = set(get_exchange_reactions(model))
    objective = _objective_coefficients(model)
    rows: list[dict[str, Any]] = []
    for index, reaction in enumerate(sorted(model.reactions, key=lambda item: item.id)):
        rows.append(
            {
                "reaction_index": index,
                "reaction_id": reaction.id,
                "name": reaction.name,
                "subsystem": getattr(reaction, "subsystem", None),
                "lower_bound": float(reaction.lower_bound),
                "upper_bound": float(reaction.upper_bound),
                "objective_coefficient": float(objective.get(reaction.id, 0.0)),
                "is_exchange": reaction.id in exchange_ids,
                "is_boundary": bool(getattr(reaction, "boundary", False)),
                "gene_reaction_rule": reaction.gene_reaction_rule,
                "reaction": reaction.reaction,
            }
        )
    return pd.DataFrame(rows)


def gene_catalog(model: Any) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for index, gene in enumerate(sorted(model.genes, key=lambda item: item.id)):
        reaction_ids = sorted(reaction.id for reaction in gene.reactions)
        rows.append(
            {
                "gene_index": index,
                "gene_id": gene.id,
                "name": gene.name,
                "reaction_ids": ";".join(reaction_ids),
                "n_reactions": len(reaction_ids),
            }
        )
    return pd.DataFrame(rows)


def metabolite_catalog(model: Any) -> pd.DataFrame:
    exchange_ids = set(get_exchange_reactions(model))
    rows: list[dict[str, Any]] = []
    for index, metabolite in enumerate(sorted(model.metabolites, key=lambda item: item.id)):
        reactions = sorted(reaction.id for reaction in metabolite.reactions)
        rows.append(
            {
                "metabolite_index": index,
                "metabolite_id": metabolite.id,
                "name": metabolite.name,
                "formula": metabolite.formula,
                "charge": metabolite.charge,
                "compartment": metabolite.compartment,
                "n_reactions": len(reactions),
                "in_exchange_reaction": any(reaction_id in exchange_ids for reaction_id in reactions),
                "reaction_ids": ";".join(reactions),
            }
        )
    return pd.DataFrame(rows)


def validate_model_inputs(model: Any, model_config: ModelConfig) -> None:
    reaction_ids = {reaction.id for reaction in model.reactions}
    for label, reaction_id in {
        "objective_reaction_id": model_config.objective_reaction_id,
        "biomass_reaction_id": model_config.biomass_reaction_id,
        "product_reaction_id": model_config.product_reaction_id,
    }.items():
        if reaction_id and reaction_id not in reaction_ids:
            raise KeyError(f"{label} not found in model reactions: {reaction_id}")
    if not get_exchange_reactions(model):
        raise ValueError("No existing exchange reactions could be discovered in the model.")
