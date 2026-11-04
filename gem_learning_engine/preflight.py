from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path
from typing import Any

from .config import PipelineConfig
from .inference import ModelInferenceReport, resolve_inferred_config
from .medium import build_medium_space
from .model_io import configure_solver, load_model, model_fingerprint, validate_model_inputs
from .orchestrator import chunk_ranges
from .sampling import DesignSampler


def resolve_pipeline_config(config: PipelineConfig) -> tuple[PipelineConfig, Any, ModelInferenceReport]:
    model = load_model(config.model)
    configure_solver(model, config.solver)
    model_config, optimization, inference = resolve_inferred_config(model, config.model, config.optimization)
    resolved = replace(config, model=model_config, optimization=optimization)
    if model_config.objective_reaction_id:
        model.objective = model.reactions.get_by_id(model_config.objective_reaction_id)
    model.objective_direction = model_config.objective_direction
    validate_model_inputs(model, model_config)
    return resolved, model, inference


def _production_baseline(model: Any, config: PipelineConfig) -> dict[str, Any]:
    mode = config.optimization.mode
    result: dict[str, Any] = {
        "optimization_mode": mode,
        "baseline_solver_status": None,
        "baseline_objective": None,
        "maximum_growth": None,
        "growth_floor": None,
        "production_status": None,
        "baseline_biomass_flux": None,
        "baseline_product_flux": None,
    }
    if mode != "growth_constrained_product":
        solution = model.optimize()
        result["baseline_solver_status"] = str(solution.status)
        result["baseline_objective"] = float(solution.objective_value) if solution.objective_value is not None else None
        if str(solution.status) == "optimal":
            if config.model.biomass_reaction_id:
                result["baseline_biomass_flux"] = float(solution.fluxes.get(config.model.biomass_reaction_id, float("nan")))
            if config.model.product_reaction_id:
                result["baseline_product_flux"] = float(solution.fluxes.get(config.model.product_reaction_id, float("nan")))
        return result

    with model:
        biomass = model.reactions.get_by_id(str(config.model.biomass_reaction_id))
        product = model.reactions.get_by_id(str(config.model.product_reaction_id))
        model.objective = biomass
        model.objective_direction = "max"
        growth_solution = model.optimize()
        growth_status = str(growth_solution.status)
        result["baseline_solver_status"] = growth_status
        if growth_status != "optimal":
            return result
        maximum_growth = float(growth_solution.fluxes.get(biomass.id, growth_solution.objective_value))
        growth_floor = max(float(biomass.lower_bound), config.optimization.minimum_growth_fraction * maximum_growth)
        biomass.lower_bound = growth_floor
        model.objective = product
        model.objective_direction = "max"
        production = model.optimize()
        result.update(
            {
                "baseline_objective": float(production.objective_value) if production.objective_value is not None else None,
                "maximum_growth": maximum_growth,
                "growth_floor": growth_floor,
                "production_status": str(production.status),
                "baseline_biomass_flux": float(production.fluxes.get(biomass.id, float("nan"))) if str(production.status) == "optimal" else None,
                "baseline_product_flux": float(production.fluxes.get(product.id, float("nan"))) if str(production.status) == "optimal" else None,
            }
        )
        return result


def build_preflight_report(config: PipelineConfig, cache_dir: str | Path) -> tuple[PipelineConfig, dict[str, Any]]:
    resolved, model, inference = resolve_pipeline_config(config)
    space = build_medium_space(model, resolved.sampling, resolved.simulation)
    baseline = _production_baseline(model, resolved)
    sampler = DesignSampler(space, resolved.sampling, resolved.simulation, cache_dir=Path(cache_dir))
    design_plan = sampler.generation_plan()
    n = resolved.simulation.n_simulations
    selected_reactions = (
        len(resolved.outputs.selected_reaction_fluxes)
        if resolved.outputs.store_reaction_fluxes == "selected"
        else len(model.reactions) if resolved.outputs.store_reaction_fluxes == "all" else 0
    )
    selected_exchanges = (
        len(resolved.outputs.selected_exchange_fluxes)
        if resolved.outputs.store_exchange_fluxes == "selected"
        else len(space.exchange_ids) if resolved.outputs.store_exchange_fluxes == "all" else 0
    )
    dense_chunk_mb = resolved.simulation.chunk_size * (selected_reactions + selected_exchanges) * 8 / (1024**2)
    ready = baseline["baseline_solver_status"] == "optimal"
    if resolved.optimization.mode == "growth_constrained_product":
        ready = ready and baseline["production_status"] == "optimal"
    warnings = list(inference.warnings)
    if resolved.sampling.discovery_scope != "model_medium":
        warnings.append("Sampling beyond model.medium requires biological review of included exchanges.")
    if resolved.outputs.store_reaction_fluxes == "all":
        warnings.append("Full reaction-flux storage may dominate memory and disk usage.")
    if resolved.simulation.method == "pfba":
        warnings.append("pFBA is substantially more expensive than FBA for large initial designs.")
    if float(design_plan["estimated_peak_mb"]) > resolved.sampling.max_design_generation_memory_mb:
        ready = False
        warnings.append("Estimated design-generation peak exceeds the configured memory guard.")
    if resolved.optimization.mode == "growth_constrained_product" and baseline["baseline_product_flux"] is not None and abs(float(baseline["baseline_product_flux"])) <= 1e-12:
        warnings.append("The detected product reaction has zero baseline flux at the selected growth fraction.")

    report = {
        "ready": bool(ready),
        **baseline,
        "model_id": getattr(model, "id", None),
        "model_name": getattr(model, "name", None),
        "model_path": resolved.model.path,
        "model_fingerprint": model_fingerprint(model, resolved.model.path),
        "biomass_reaction_id": resolved.model.biomass_reaction_id,
        "product_reaction_id": resolved.model.product_reaction_id,
        "biomass_inference_confidence": inference.biomass.confidence,
        "product_inference_confidence": inference.product.confidence,
        "n_reactions": len(model.reactions),
        "n_metabolites": len(model.metabolites),
        "n_genes": len(model.genes),
        "n_exchanges": len(space.exchange_ids),
        "n_variable_medium_components": len(space.bounds),
        "variable_medium_reaction_ids": space.reaction_ids,
        "n_fixed_medium_components": len(space.fixed_uptakes),
        "fixed_medium_reasons": space.fixed_reasons,
        "n_simulations": n,
        "n_chunks": len(chunk_ranges(n, resolved.simulation.chunk_size)),
        "sampling_strategy": resolved.sampling.strategy,
        "estimated_design_cache_mb": float(design_plan["cache_mb"]),
        "estimated_design_generation_peak_mb": float(design_plan["estimated_peak_mb"]),
        "generated_design_rows": int(design_plan["generated_rows"]),
        "design_generation_memory_guard_mb": resolved.sampling.max_design_generation_memory_mb,
        "estimated_selected_dense_flux_array_mb_per_chunk": dense_chunk_mb,
        "processes": resolved.simulation.processes,
        "estimated_model_copies_in_memory": resolved.simulation.processes if resolved.simulation.processes > 1 else 1,
        "sobol_power_of_two": n > 0 and ((n & (n - 1)) == 0),
        "next_power_of_two": 2 ** math.ceil(math.log2(max(1, n))),
        "inference": inference.to_dict(),
        "resolved_config": resolved.to_dict(),
        "warnings": warnings,
    }
    return resolved, report
