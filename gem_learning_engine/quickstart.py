from __future__ import annotations

import math
import os
import re
from pathlib import Path

from .config import (
    DiagnosticsConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    OptimizationConfig,
    OutputConfig,
    PipelineConfig,
    ProjectConfig,
    SamplingConfig,
    SimulationConfig,
    SolverConfig,
)


def safe_slug(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return cleaned or "model"


def recommended_processes() -> int:
    available = max(1, os.cpu_count() or 1)
    return max(1, min(4, available // 2 or 1))


def recommended_chunk_size(n_simulations: int) -> int:
    if n_simulations <= 256:
        return min(64, n_simulations)
    if n_simulations <= 4096:
        return 128
    return 256


def create_quick_config(
    model_path: str | Path,
    n_simulations: int,
    *,
    output_dir: str | Path | None = None,
    processes: int | None = None,
    minimum_growth_fraction: float = 0.5,
    overwrite: bool = False,
    resume: bool = True,
) -> PipelineConfig:
    model_path = Path(model_path).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if n_simulations <= 0:
        raise ValueError("n_simulations must be a positive integer.")
    stem = safe_slug(model_path.stem)
    output = Path(output_dir) if output_dir is not None else Path("outputs") / f"{stem}_{n_simulations}"
    chunk_size = recommended_chunk_size(n_simulations)
    n_chunks = max(1, math.ceil(n_simulations / chunk_size))
    process_count = min(processes or recommended_processes(), n_chunks)
    return PipelineConfig(
        project=ProjectConfig(
            name=f"{stem}_{n_simulations}",
            output_dir=str(output),
            overwrite=overwrite,
            resume=resume,
        ),
        model=ModelConfig(path=str(model_path), format="auto"),
        optimization=OptimizationConfig(
            mode="auto",
            minimum_growth_fraction=minimum_growth_fraction,
        ),
        experiment=ExperimentConfig(
            analysis_type="random_medium",
            design_stage="automatic_initial_space_filling",
            notes="Automatically configured from model path and run count.",
        ),
        solver=SolverConfig(name="glpk", timeout_seconds=60),
        simulation=SimulationConfig(
            n_simulations=n_simulations,
            chunk_size=chunk_size,
            processes=process_count,
            max_in_flight_chunks=max(2, 2 * process_count),
            simulation_log_every=1,
            fail_fast=False,
        ),
        sampling=SamplingConfig(
            strategy="sobol",
            discovery_scope="model_medium",
        ),
        outputs=OutputConfig(
            store_requested_media=True,
            store_applied_media=True,
            # Quick Run keeps the complete final-stage FBA solution for each
            # sampled condition. Long-format Parquet is compact and auditable.
            store_exchange_fluxes="all",
            store_reaction_fluxes="all",
            flux_table_format="long",
        ),
        diagnostics=DiagnosticsConfig(store_active_constraints="exchanges"),
        logging=LoggingConfig(),
    )
