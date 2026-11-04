from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml

ObjectiveDirection = Literal["max", "min"]
SimulationMethod = Literal["fba", "pfba"]
SamplingStrategy = Literal["sobol", "latin_hypercube", "uniform"]
DiscoveryScope = Literal["model_medium", "all_uptake_capable", "all_exchanges"]
AutomaticMediumPolicy = Literal["nutrients", "all_baseline"]
StoreMode = Literal["none", "selected", "nonzero", "all"]
FluxTableFormat = Literal["wide", "long"]
ActiveConstraintMode = Literal["none", "exchanges", "selected", "all"]
OptimizationMode = Literal["auto", "native_objective", "growth_constrained_product", "maximize_biomass", "maximize_product"]


@dataclass(frozen=True)
class ProjectConfig:
    name: str = "gem_learning_engine_run"
    output_dir: str = "outputs/gem_learning_engine_run"
    overwrite: bool = False
    resume: bool = True
    pause_file: str = "PAUSE_REQUESTED"
    verify_chunk_checksums_on_resume: bool = True


@dataclass(frozen=True)
class ModelConfig:
    path: str = "data/models/model.xml"
    format: str = "auto"  # auto, sbml, json, mat
    objective_reaction_id: str | None = None
    objective_direction: ObjectiveDirection = "max"
    biomass_reaction_id: str | None = None
    product_reaction_id: str | None = None


@dataclass(frozen=True)
class OptimizationConfig:
    # auto => infer model roles and use growth-constrained product optimization
    # when a credible biomass/product pair is available; otherwise preserve the
    # model's native objective.
    mode: OptimizationMode = "auto"
    minimum_growth_fraction: float = 0.5
    require_high_confidence_product: bool = False
    allow_medium_confidence_product: bool = True


@dataclass(frozen=True)
class SolverConfig:
    name: str = "glpk"
    timeout_seconds: int | None = 60
    feasibility_tolerance: float | None = None
    optimality_tolerance: float | None = None


@dataclass(frozen=True)
class ExperimentConfig:
    analysis_type: str = "random_medium"
    design_stage: str = "initial_space_filling"
    sampling_batch: int = 0
    parent_query_id: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class SimulationConfig:
    n_simulations: int = 1024
    chunk_size: int = 128
    seed: int = 42
    method: SimulationMethod = "fba"
    processes: int = 1
    # 0 => 2 * processes. Bounds in-flight work and parent memory.
    max_in_flight_chunks: int = 0
    chunk_retry_attempts: int = 1
    close_unlisted_exchanges: bool = False
    respect_original_exchange_bounds: bool = True
    feasible_statuses: list[str] = field(default_factory=lambda: ["optimal"])
    simulation_log_every: int = 1
    fail_fast: bool = False


@dataclass(frozen=True)
class SamplingConfig:
    # Scrambled Sobol is the publication-design default. Uniform and LHS remain
    # available as benchmark baselines.
    strategy: SamplingStrategy = "sobol"
    discovery_scope: DiscoveryScope = "model_medium"
    # In simple mode, vary plausible nutrients/gases while preserving water,
    # protons, trace metals, counterions, and vitamins at baseline. Advanced
    # users can select all_baseline or explicitly include/exclude reactions.
    automatic_medium_policy: AutomaticMediumPolicy = "nutrients"
    auto_min_fraction: float = 0.0
    auto_max_fraction: float = 1.0
    # Prevent unconstrained native bounds (often 1000) from becoming unrealistic
    # automatic design ranges. Explicit sampling.bounds are never capped.
    auto_max_uptake_cap: float | None = 20.0
    closed_exchange_max_uptake: float = 10.0
    bounds: dict[str, dict[str, float]] = field(default_factory=dict)
    fixed_reactions: dict[str, float] = field(default_factory=dict)
    include_reaction_ids: list[str] = field(default_factory=list)
    exclude_reaction_ids: list[str] = field(default_factory=list)
    # Presence/absence experiments should normally be separate from the initial
    # continuous design. Zero by default avoids creating mostly infeasible media.
    include_zero_probability: float = 0.0
    sobol_scramble: bool = True
    sobol_skip: int = 0
    cache_dtype: Literal["float32", "float64"] = "float32"
    # Peak-memory guard for one-time Sobol/LHS design construction. The saved
    # cache is memory-mapped afterward, so this protects only generation.
    max_design_generation_memory_mb: int = 2048


@dataclass(frozen=True)
class OutputConfig:
    compression: str = "zstd"
    store_conditions: bool = True
    store_requested_media: bool = False
    store_applied_media: bool = True
    store_exchange_fluxes: StoreMode = "selected"
    selected_exchange_fluxes: list[str] = field(default_factory=list)
    store_reaction_fluxes: StoreMode = "none"
    selected_reaction_fluxes: list[str] = field(default_factory=list)
    flux_table_format: FluxTableFormat = "wide"
    flux_zero_tolerance: float = 1e-9
    max_dense_flux_memory_mb: int = 512


@dataclass(frozen=True)
class DiagnosticsConfig:
    store_shadow_prices: StoreMode = "none"
    selected_shadow_price_metabolites: list[str] = field(default_factory=list)
    store_reduced_costs: StoreMode = "none"
    selected_reduced_cost_reactions: list[str] = field(default_factory=list)
    store_active_constraints: ActiveConstraintMode = "exchanges"
    selected_active_constraint_reactions: list[str] = field(default_factory=list)
    dual_zero_tolerance: float = 1e-9
    bound_activity_tolerance: float = 1e-7


@dataclass(frozen=True)
class LoggingConfig:
    level: str = "INFO"
    log_to_console: bool = True
    log_to_file: bool = True
    structured_jsonl: bool = True
    max_file_mb: int = 25
    backup_count: int = 5


@dataclass(frozen=True)
class PipelineConfig:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    outputs: OutputConfig = field(default_factory=OutputConfig)
    diagnostics: DiagnosticsConfig = field(default_factory=DiagnosticsConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def data_hash(self) -> str:
        """Hash settings that determine generated conditions and exact results."""
        data = self.to_dict()
        # Runtime location and presentation settings do not define the scientific
        # design. Model identity is protected separately by the structural/file
        # fingerprint, allowing a verified run to be relocated safely.
        data.pop("project", None)
        data.pop("logging", None)
        data["model"]["path"] = None
        payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()[:20]


def _merge(defaults: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    out = dict(defaults)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _positive_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer. Got {value!r}.")


def _nonnegative_int(name: str, value: int) -> None:
    if not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer. Got {value!r}.")


def _probability(name: str, value: float) -> None:
    if not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be between 0 and 1. Got {value!r}.")


def _validate_bounds(bounds: dict[str, dict[str, float]]) -> None:
    for reaction_id, spec in bounds.items():
        if "min" not in spec or "max" not in spec:
            raise ValueError(f"Sampling bound for {reaction_id} must contain min and max.")
        minimum, maximum = float(spec["min"]), float(spec["max"])
        if minimum < 0 or maximum < 0:
            raise ValueError(f"Sampling bounds for {reaction_id} must be non-negative uptake magnitudes.")
        if minimum > maximum:
            raise ValueError(f"Sampling min > max for {reaction_id}: {minimum} > {maximum}")


def validate_config(config: PipelineConfig) -> None:
    _positive_int("simulation.n_simulations", config.simulation.n_simulations)
    _positive_int("simulation.chunk_size", config.simulation.chunk_size)
    _positive_int("simulation.processes", config.simulation.processes)
    _nonnegative_int("simulation.max_in_flight_chunks", config.simulation.max_in_flight_chunks)
    _nonnegative_int("simulation.chunk_retry_attempts", config.simulation.chunk_retry_attempts)
    _nonnegative_int("simulation.simulation_log_every", config.simulation.simulation_log_every)
    _nonnegative_int("sampling.sobol_skip", config.sampling.sobol_skip)
    _positive_int(
        "sampling.max_design_generation_memory_mb",
        config.sampling.max_design_generation_memory_mb,
    )
    _nonnegative_int("experiment.sampling_batch", config.experiment.sampling_batch)
    _probability("sampling.include_zero_probability", config.sampling.include_zero_probability)
    _validate_bounds(config.sampling.bounds)

    if config.sampling.auto_min_fraction < 0 or config.sampling.auto_max_fraction < 0:
        raise ValueError("sampling auto fractions must be non-negative.")
    if config.sampling.auto_min_fraction > config.sampling.auto_max_fraction:
        raise ValueError("sampling.auto_min_fraction cannot exceed sampling.auto_max_fraction.")
    if config.sampling.closed_exchange_max_uptake < 0:
        raise ValueError("sampling.closed_exchange_max_uptake must be non-negative.")
    for reaction_id, value in config.sampling.fixed_reactions.items():
        if float(value) < 0:
            raise ValueError(f"Fixed uptake for {reaction_id} must be non-negative.")

    if config.simulation.method not in {"fba", "pfba"}:
        raise ValueError("simulation.method must be 'fba' or 'pfba'.")
    if config.sampling.strategy not in {"sobol", "latin_hypercube", "uniform"}:
        raise ValueError("sampling.strategy must be sobol, latin_hypercube, or uniform.")
    if config.sampling.discovery_scope not in {"model_medium", "all_uptake_capable", "all_exchanges"}:
        raise ValueError("Invalid sampling.discovery_scope.")
    if config.sampling.automatic_medium_policy not in {"nutrients", "all_baseline"}:
        raise ValueError("sampling.automatic_medium_policy must be nutrients or all_baseline.")
    if config.sampling.auto_max_uptake_cap is not None and float(config.sampling.auto_max_uptake_cap) <= 0:
        raise ValueError("sampling.auto_max_uptake_cap must be positive or null.")
    if config.model.objective_direction not in {"max", "min"}:
        raise ValueError("model.objective_direction must be max or min.")
    if config.optimization.mode not in {"auto", "native_objective", "growth_constrained_product", "maximize_biomass", "maximize_product"}:
        raise ValueError("Invalid optimization.mode.")
    _probability("optimization.minimum_growth_fraction", config.optimization.minimum_growth_fraction)
    if config.outputs.flux_table_format not in {"wide", "long"}:
        raise ValueError("outputs.flux_table_format must be wide or long.")
    if not config.outputs.store_conditions:
        raise ValueError("outputs.store_conditions must remain true; exact condition metadata is mandatory.")
    _positive_int("outputs.max_dense_flux_memory_mb", config.outputs.max_dense_flux_memory_mb)
    _positive_int("logging.max_file_mb", config.logging.max_file_mb)
    _nonnegative_int("logging.backup_count", config.logging.backup_count)

    for field_name in ["store_exchange_fluxes", "store_reaction_fluxes"]:
        value = getattr(config.outputs, field_name)
        if value not in {"none", "selected", "nonzero", "all"}:
            raise ValueError(f"outputs.{field_name} has invalid value: {value}")
    for field_name in ["store_shadow_prices", "store_reduced_costs"]:
        value = getattr(config.diagnostics, field_name)
        if value not in {"none", "selected", "nonzero", "all"}:
            raise ValueError(f"diagnostics.{field_name} has invalid value: {value}")
    if config.diagnostics.store_active_constraints not in {"none", "exchanges", "selected", "all"}:
        raise ValueError("Invalid diagnostics.store_active_constraints.")

    if not config.project.pause_file or "/" in config.project.pause_file or "\\" in config.project.pause_file:
        raise ValueError("project.pause_file must be a simple filename.")


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    data = _merge(PipelineConfig().to_dict(), raw)
    config = PipelineConfig(
        project=ProjectConfig(**data["project"]),
        model=ModelConfig(**data["model"]),
        optimization=OptimizationConfig(**data["optimization"]),
        experiment=ExperimentConfig(**data["experiment"]),
        solver=SolverConfig(**data["solver"]),
        simulation=SimulationConfig(**data["simulation"]),
        sampling=SamplingConfig(**data["sampling"]),
        outputs=OutputConfig(**data["outputs"]),
        diagnostics=DiagnosticsConfig(**data["diagnostics"]),
        logging=LoggingConfig(**data["logging"]),
    )
    validate_config(config)
    return config


def save_config_snapshot(config: PipelineConfig, output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False)
