from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from .conditions import ConditionBatch, MediaConditionGenerator
from .config import PipelineConfig, load_config
from .fba import FBAEngine
from .inference import infer_model_roles
from .medium import build_medium_space
from .model_io import (
    configure_solver,
    exchange_catalog,
    gene_catalog,
    load_model,
    metabolite_catalog,
    reaction_catalog,
    validate_model_inputs,
)
from .orchestrator import SimulationOrchestrator
from .sampling import DesignSampler
from .storage import ResultsStore


def build_toy_model():
    import cobra
    from cobra import Metabolite, Model, Reaction

    model = Model("gem_learning_engine_publication_toy")
    glc_e = Metabolite("glc_e", compartment="e", name="Glucose")
    o2_e = Metabolite("o2_e", compartment="e", name="Oxygen")
    glc_c = Metabolite("glc_c", compartment="c", name="Glucose cytosol")
    o2_c = Metabolite("o2_c", compartment="c", name="Oxygen cytosol")
    product_c = Metabolite("product_c", compartment="c", name="Product")
    product_e = Metabolite("product_e", compartment="e", name="Product extracellular")

    ex_glc = Reaction("EX_glc_e", lower_bound=-10, upper_bound=1000)
    ex_glc.add_metabolites({glc_e: -1})
    ex_o2 = Reaction("EX_o2_e", lower_bound=-20, upper_bound=1000)
    ex_o2.add_metabolites({o2_e: -1})
    t_glc = Reaction("T_GLC", lower_bound=0, upper_bound=1000)
    t_glc.add_metabolites({glc_e: -1, glc_c: 1})
    t_glc.gene_reaction_rule = "g_transport"
    t_o2 = Reaction("T_O2", lower_bound=0, upper_bound=1000)
    t_o2.add_metabolites({o2_e: -1, o2_c: 1})
    biomass = Reaction("BIOMASS", lower_bound=0, upper_bound=1000)
    biomass.add_metabolites({glc_c: -1, o2_c: -1})
    biomass.gene_reaction_rule = "g_biomass"
    product = Reaction("PRODUCT", lower_bound=0, upper_bound=1000)
    product.add_metabolites({glc_c: -1, product_c: 1})
    product.gene_reaction_rule = "g_product"
    t_product = Reaction("T_PRODUCT", lower_bound=0, upper_bound=1000)
    t_product.add_metabolites({product_c: -1, product_e: 1})
    ex_product = Reaction("EX_product_e", lower_bound=0, upper_bound=1000)
    ex_product.add_metabolites({product_e: -1})

    model.add_reactions([ex_glc, ex_o2, t_glc, t_o2, biomass, product, t_product, ex_product])
    model.objective = biomass
    return model


def save_toy_model(path: Path) -> None:
    import cobra

    path.parent.mkdir(parents=True, exist_ok=True)
    cobra.io.save_json_model(build_toy_model(), str(path))


def write_validation_config(
    path: Path,
    output_dir: Path,
    model_path: Path,
    processes: int = 1,
    n_simulations: int = 32,
    overwrite: bool = True,
    resume: bool = False,
) -> None:
    data: dict[str, Any] = {
        "project": {
            "name": "validation",
            "output_dir": str(output_dir),
            "overwrite": overwrite,
            "resume": resume,
        },
        "model": {
            "path": str(model_path),
            "format": "json",
            "objective_reaction_id": "BIOMASS",
            "objective_direction": "max",
            "biomass_reaction_id": "BIOMASS",
            "product_reaction_id": "PRODUCT",
        },
        "optimization": {
            "mode": "auto",
            "minimum_growth_fraction": 0.5,
        },
        "experiment": {
            "analysis_type": "random_medium",
            "design_stage": "validation",
            "sampling_batch": 0,
        },
        "simulation": {
            "n_simulations": n_simulations,
            "chunk_size": 8,
            "seed": 7,
            "method": "fba",
            "processes": processes,
            "close_unlisted_exchanges": True,
            "simulation_log_every": 0,
        },
        "sampling": {
            "strategy": "sobol",
            "discovery_scope": "model_medium",
            "exclude_reaction_ids": ["EX_o2_e"],
        },
        "outputs": {
            "store_requested_media": True,
            "store_applied_media": True,
            "store_exchange_fluxes": "selected",
            "selected_exchange_fluxes": ["EX_glc_e", "EX_o2_e", "EX_product_e"],
            "store_reaction_fluxes": "selected",
            "selected_reaction_fluxes": ["BIOMASS", "PRODUCT"],
            "flux_table_format": "wide",
        },
        "diagnostics": {
            "store_shadow_prices": "selected",
            "selected_shadow_price_metabolites": ["glc_c", "o2_c"],
            "store_reduced_costs": "selected",
            "selected_reduced_cost_reactions": ["BIOMASS", "PRODUCT"],
            "store_active_constraints": "exchanges",
        },
        "logging": {"log_to_console": False, "log_to_file": True, "structured_jsonl": True},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def run_all_validations(output_dir: str | Path = "outputs/validation") -> list[dict[str, Any]]:
    root = Path(output_dir)
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    model_path = root / "toy.json"
    config_path = root / "validation.yaml"
    run_dir = root / "run"
    save_toy_model(model_path)
    write_validation_config(config_path, run_dir, model_path)

    results: list[dict[str, Any]] = []
    config = load_config(config_path)
    model = load_model(config.model)
    configure_solver(model, config.solver)
    validate_model_inputs(model, config.model)

    inferred = infer_model_roles(
        model,
        replace(
            config.model,
            objective_reaction_id=None,
            biomass_reaction_id=None,
            product_reaction_id=None,
        ),
        config.optimization,
    )
    results.append(
        {
            "check": "automatic_biomass_product_inference",
            "passed": bool(
                inferred.biomass.reaction_id == "BIOMASS"
                and inferred.product.reaction_id == "EX_product_e"
                and inferred.selected_optimization_mode == "growth_constrained_product"
            ),
        }
    )

    exchanges = exchange_catalog(model)
    reactions = reaction_catalog(model)
    genes = gene_catalog(model)
    metabolites = metabolite_catalog(model)
    results.append(
        {
            "check": "automatic_model_catalogs",
            "passed": (
                {"EX_glc_e", "EX_o2_e", "EX_product_e"}.issubset(set(exchanges["reaction_id"]))
                and {"g_transport", "g_biomass", "g_product"}.issubset(set(genes["gene_id"]))
                and len(reactions) == len(model.reactions)
                and len(metabolites) == len(model.metabolites)
            ),
        }
    )

    medium_space = build_medium_space(model, config.sampling, config.simulation)
    results.append(
        {
            "check": "baseline_medium_preservation",
            "passed": "EX_o2_e" in medium_space.fixed_uptakes and medium_space.fixed_uptakes["EX_o2_e"] > 0,
        }
    )

    generator = MediaConditionGenerator(
        medium_space,
        config.sampling,
        config.simulation,
        config.experiment,
        cache_dir=root / "design_cache",
    )
    generator.prepare()
    whole = generator.generate_chunk(0, 24).requested_media
    split = pd.concat(
        [generator.generate_chunk(0, 7).requested_media, generator.generate_chunk(7, 24).requested_media],
        ignore_index=True,
    )
    results.append(
        {
            "check": "chunk_independent_sobol_sampling",
            "passed": bool(np.allclose(whole.drop(columns="sample_id"), split.drop(columns="sample_id"))),
        }
    )
    variable = medium_space.bounds[0]
    results.append(
        {
            "check": "sobol_bounds",
            "passed": bool(
                whole[variable.reaction_id].between(variable.min_uptake, variable.max_uptake, inclusive="both").all()
            ),
        }
    )

    guarded_sampler = DesignSampler(
        medium_space,
        replace(config.sampling, max_design_generation_memory_mb=1),
        replace(config.simulation, n_simulations=1_000_000),
        cache_dir=root / "guarded_design",
    )
    guard_passed = False
    try:
        guarded_sampler.prepare()
    except MemoryError:
        guard_passed = True
    results.append({"check": "design_generation_memory_guard", "passed": guard_passed})

    engine = FBAEngine(
        model,
        config.model,
        config.optimization,
        config.simulation,
        config.outputs,
        config.diagnostics,
        medium_space,
    )
    base_batch = generator.generate_chunk(0, 4)
    perturbation_batch = ConditionBatch(
        metadata=base_batch.metadata,
        requested_media=base_batch.requested_media,
        gene_perturbations=pd.DataFrame([{"sample_id": 1, "gene_id": "g_biomass", "activity": 0.0}]),
        reaction_perturbations=pd.DataFrame(
            [{"sample_id": 2, "reaction_id": "BIOMASS", "lower_bound": 0.0, "upper_bound": 0.0}]
        ),
        objective_perturbations=pd.DataFrame(
            [
                {
                    "sample_id": 3,
                    "reaction_id": "PRODUCT",
                    "coefficient": 1.0,
                    "objective_direction": "max",
                }
            ]
        ),
    )
    tables = engine.run_chunk(perturbation_batch, chunk_id=0)
    indexed = tables.results.set_index("sample_id")
    results.append(
        {
            "check": "generalized_condition_execution_and_restoration",
            "passed": bool(
                indexed.loc[0, "biomass_flux"] > 0
                and indexed.loc[0, "product_flux"] > 0
                and indexed.loc[0, "optimization_mode"] == "growth_constrained_product"
                and indexed.loc[0, "growth_floor"] > 0
                and abs(indexed.loc[1, "biomass_flux"]) < 1e-9
                and abs(indexed.loc[2, "biomass_flux"]) < 1e-9
                and indexed.loc[3, "product_flux"] > 0
                and model.objective_direction == "max"
            ),
        }
    )
    results.append(
        {
            "check": "diagnostic_extraction",
            "passed": bool(
                not tables.shadow_prices.empty
                and not tables.reduced_costs.empty
                and not tables.active_constraints.empty
            ),
        }
    )

    SimulationOrchestrator(config).run()
    store = ResultsStore(config)
    summary = store.summarize()
    results.append(
        {
            "check": "full_chunked_pipeline",
            "passed": bool(len(summary) == config.simulation.n_simulations and summary["status"].eq("optimal").all()),
        }
    )

    resume_data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    resume_data["project"]["overwrite"] = False
    resume_data["project"]["resume"] = True
    resume_path = root / "resume.yaml"
    resume_path.write_text(yaml.safe_dump(resume_data, sort_keys=False), encoding="utf-8")
    resume_config = load_config(resume_path)
    SimulationOrchestrator(resume_config).run()
    results.append(
        {
            "check": "verified_resume",
            "passed": len(ResultsStore(resume_config).summarize()) == config.simulation.n_simulations,
        }
    )

    result_path = store.table_path("results", 0)
    backup_path = result_path.with_suffix(".bak")
    result_path.replace(backup_path)
    invalidated = not ResultsStore(resume_config).is_chunk_complete(0)
    backup_path.replace(result_path)
    results.append({"check": "manifest_integrity", "passed": invalidated})

    parallel_path = root / "parallel.yaml"
    parallel_dir = root / "parallel_run"
    write_validation_config(parallel_path, parallel_dir, model_path, processes=2, n_simulations=16)
    parallel_config = load_config(parallel_path)
    SimulationOrchestrator(parallel_config).run()
    results.append(
        {
            "check": "persistent_spawn_parallel_workers",
            "passed": len(ResultsStore(parallel_config).summarize()) == 16,
        }
    )

    validation_path = root / "validation_results.json"
    validation_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    failed = [result["check"] for result in results if not result["passed"]]
    if failed:
        raise AssertionError(f"Validation failed: {failed}")
    return results
