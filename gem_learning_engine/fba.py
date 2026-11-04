from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from .conditions import ConditionBatch
from .config import DiagnosticsConfig, ModelConfig, OptimizationConfig, OutputConfig, SimulationConfig
from .medium import MediumApplier, MediumSpace
from .model_io import get_exchange_reactions

LOGGER = logging.getLogger("gem_learning_engine")


@dataclass(frozen=True)
class SolveOutcome:
    solution: Any
    optimization_mode: str
    maximum_growth: float | None = None
    growth_floor: float | None = None
    biomass_stage_status: str | None = None


@dataclass(frozen=True)
class SimulationTables:
    condition_metadata: pd.DataFrame
    requested_media: pd.DataFrame
    applied_media: pd.DataFrame
    gene_perturbations: pd.DataFrame
    reaction_perturbations: pd.DataFrame
    objective_perturbations: pd.DataFrame
    results: pd.DataFrame
    simulation_log: pd.DataFrame
    exchange_fluxes: pd.DataFrame
    reaction_fluxes: pd.DataFrame
    shadow_prices: pd.DataFrame
    reduced_costs: pd.DataFrame
    active_constraints: pd.DataFrame
    errors: pd.DataFrame

    def as_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "condition_metadata": self.condition_metadata,
            "requested_media": self.requested_media,
            "applied_media": self.applied_media,
            "gene_perturbations": self.gene_perturbations,
            "reaction_perturbations": self.reaction_perturbations,
            "objective_perturbations": self.objective_perturbations,
            "results": self.results,
            "simulation_log": self.simulation_log,
            "exchange_fluxes": self.exchange_fluxes,
            "reaction_fluxes": self.reaction_fluxes,
            "shadow_prices": self.shadow_prices,
            "reduced_costs": self.reduced_costs,
            "active_constraints": self.active_constraints,
            "errors": self.errors,
        }


class _SeriesAccumulator:
    """Memory-aware accumulator for fluxes, duals, or reduced costs."""

    def __init__(
        self,
        all_ids: list[str],
        mode: str,
        selected_ids: list[str],
        table_format: str,
        zero_tolerance: float,
        chunk_size: int,
        max_dense_memory_mb: int,
        entity_column: str,
        value_column: str,
    ):
        if mode == "selected" and not selected_ids:
            mode = "none"
        self.mode = mode
        self.table_format = table_format
        self.zero_tolerance = zero_tolerance
        self.entity_column = entity_column
        self.value_column = value_column
        if mode == "selected":
            self.ids = list(selected_ids)
        elif mode in {"all", "nonzero"}:
            self.ids = list(all_ids)
        elif mode == "none":
            self.ids = []
        else:
            raise ValueError(f"Invalid storage mode: {mode}")
        self._ids_array = np.asarray(self.ids, dtype=object)
        self._dense_sample_ids: list[int] = []
        self._dense_values: list[np.ndarray] = []
        self._sparse_sample_ids: list[np.ndarray] = []
        self._sparse_entity_ids: list[np.ndarray] = []
        self._sparse_values: list[np.ndarray] = []

        if mode not in {"none", "nonzero"} and table_format == "wide":
            estimate = chunk_size * max(1, len(self.ids)) * 8 * 2.5 / (1024**2)
            if estimate > max_dense_memory_mb:
                raise MemoryError(
                    f"Estimated dense table peak is {estimate:.1f} MB, above "
                    f"outputs.max_dense_flux_memory_mb={max_dense_memory_mb}. Reduce chunk_size, "
                    "select fewer entities, or increase the limit intentionally."
                )

    def add(self, sample_id: int, values: pd.Series | None) -> None:
        if self.mode == "none":
            return
        if values is None or values.empty:
            array = np.full(len(self.ids), np.nan, dtype=np.float64)
        else:
            array = values.reindex(self.ids).to_numpy(dtype=np.float64, copy=True)

        if self.mode == "nonzero":
            mask = np.isfinite(array) & (np.abs(array) > self.zero_tolerance)
            indices = np.flatnonzero(mask)
            if indices.size:
                self._sparse_sample_ids.append(np.full(indices.size, sample_id, dtype=np.int64))
                self._sparse_entity_ids.append(self._ids_array[indices])
                self._sparse_values.append(array[indices])
            return

        if self.table_format == "wide":
            self._dense_sample_ids.append(sample_id)
            self._dense_values.append(array)
        else:
            indices = np.flatnonzero(np.isfinite(array))
            if indices.size:
                self._sparse_sample_ids.append(np.full(indices.size, sample_id, dtype=np.int64))
                self._sparse_entity_ids.append(self._ids_array[indices])
                self._sparse_values.append(array[indices])

    def to_frame(self) -> pd.DataFrame:
        if self.mode == "none":
            return pd.DataFrame()
        if self.mode != "nonzero" and self.table_format == "wide":
            if not self._dense_values:
                return pd.DataFrame(columns=["sample_id", *self.ids])
            matrix = np.vstack(self._dense_values)
            frame = pd.DataFrame(matrix, columns=self.ids)
            frame.insert(0, "sample_id", np.asarray(self._dense_sample_ids, dtype=np.int64))
            return frame
        if not self._sparse_values:
            return pd.DataFrame(columns=["sample_id", self.entity_column, self.value_column])
        return pd.DataFrame(
            {
                "sample_id": np.concatenate(self._sparse_sample_ids),
                self.entity_column: np.concatenate(self._sparse_entity_ids),
                self.value_column: np.concatenate(self._sparse_values),
            }
        )


class _ActiveConstraintAccumulator:
    def __init__(
        self,
        model: Any,
        exchange_ids: list[str],
        reaction_ids: list[str],
        diagnostics: DiagnosticsConfig,
    ):
        mode = diagnostics.store_active_constraints
        if mode == "none":
            ids: list[str] = []
        elif mode == "exchanges":
            ids = exchange_ids
        elif mode == "selected":
            ids = diagnostics.selected_active_constraint_reactions
        elif mode == "all":
            ids = reaction_ids
        else:
            raise ValueError(f"Invalid active-constraint mode: {mode}")
        valid = set(reaction_ids)
        missing = [reaction_id for reaction_id in ids if reaction_id not in valid]
        if missing:
            raise KeyError(f"Active-constraint reactions not found: {missing[:10]}")
        self.model = model
        self.ids = ids
        self.tolerance = diagnostics.bound_activity_tolerance
        self._sample_ids: list[np.ndarray] = []
        self._reaction_ids: list[np.ndarray] = []
        self._bound_types: list[np.ndarray] = []
        self._fluxes: list[np.ndarray] = []
        self._bound_values: list[np.ndarray] = []
        self._slacks: list[np.ndarray] = []

    def add(
        self,
        sample_id: int,
        fluxes: pd.Series | None,
        bound_overrides: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        if not self.ids or fluxes is None or fluxes.empty:
            return
        overrides = bound_overrides or {}
        values = fluxes.reindex(self.ids).to_numpy(dtype=np.float64, copy=False)
        lower = np.asarray(
            [overrides.get(rid, self.model.reactions.get_by_id(rid).bounds)[0] for rid in self.ids],
            dtype=np.float64,
        )
        upper = np.asarray(
            [overrides.get(rid, self.model.reactions.get_by_id(rid).bounds)[1] for rid in self.ids],
            dtype=np.float64,
        )
        ids_array = np.asarray(self.ids, dtype=object)

        lower_slack = values - lower
        lower_active = np.isfinite(values) & (np.abs(lower_slack) <= self.tolerance)
        upper_slack = upper - values
        upper_active = np.isfinite(values) & (np.abs(upper_slack) <= self.tolerance)

        for mask, bound_type, bounds, slacks in [
            (lower_active, "lower", lower, lower_slack),
            (upper_active, "upper", upper, upper_slack),
        ]:
            indices = np.flatnonzero(mask)
            if indices.size:
                self._sample_ids.append(np.full(indices.size, sample_id, dtype=np.int64))
                self._reaction_ids.append(ids_array[indices])
                self._bound_types.append(np.full(indices.size, bound_type, dtype=object))
                self._fluxes.append(values[indices])
                self._bound_values.append(bounds[indices])
                self._slacks.append(slacks[indices])

    def to_frame(self) -> pd.DataFrame:
        if not self._fluxes:
            return pd.DataFrame(columns=["sample_id", "reaction_id", "bound_type", "flux", "bound_value", "slack"])
        return pd.DataFrame(
            {
                "sample_id": np.concatenate(self._sample_ids),
                "reaction_id": np.concatenate(self._reaction_ids),
                "bound_type": np.concatenate(self._bound_types),
                "flux": np.concatenate(self._fluxes),
                "bound_value": np.concatenate(self._bound_values),
                "slack": np.concatenate(self._slacks),
            }
        )


class FBAEngine:
    """Reusable exact FBA/pFBA engine for generalized condition batches."""

    def __init__(
        self,
        model: Any,
        model_config: ModelConfig,
        optimization_config: OptimizationConfig,
        simulation_config: SimulationConfig,
        output_config: OutputConfig,
        diagnostics_config: DiagnosticsConfig,
        medium_space: MediumSpace,
    ):
        self.model = model
        self.model_config = model_config
        self.optimization_config = optimization_config
        self.simulation_config = simulation_config
        self.output_config = output_config
        self.diagnostics_config = diagnostics_config
        self.medium_space = medium_space
        self.exchange_ids = get_exchange_reactions(model)
        self.reaction_ids = [reaction.id for reaction in model.reactions]
        self.metabolite_ids = [metabolite.id for metabolite in model.metabolites]
        self.gene_ids = {gene.id for gene in model.genes}
        self.medium_applier = MediumApplier(model, medium_space, simulation_config)
        self._preflight_dense_memory()

        self._validate_selected(
            output_config.selected_exchange_fluxes,
            set(self.exchange_ids),
            "selected exchange fluxes",
        )
        self._validate_selected(
            output_config.selected_reaction_fluxes,
            set(self.reaction_ids),
            "selected reaction fluxes",
        )
        self._validate_selected(
            diagnostics_config.selected_shadow_price_metabolites,
            set(self.metabolite_ids),
            "selected shadow-price metabolites",
        )
        self._validate_selected(
            diagnostics_config.selected_reduced_cost_reactions,
            set(self.reaction_ids),
            "selected reduced-cost reactions",
        )


    def _preflight_dense_memory(self) -> None:
        if self.output_config.flux_table_format != "wide":
            return
        counts = []
        for mode, selected, all_ids in [
            (self.output_config.store_exchange_fluxes, self.output_config.selected_exchange_fluxes, self.exchange_ids),
            (self.output_config.store_reaction_fluxes, self.output_config.selected_reaction_fluxes, self.reaction_ids),
        ]:
            if mode == "all":
                counts.append(len(all_ids))
            elif mode == "selected":
                counts.append(len(selected))
        estimated = self.simulation_config.chunk_size * sum(counts) * 8 * 2.5 / (1024**2)
        if estimated > self.output_config.max_dense_flux_memory_mb:
            raise MemoryError(
                f"Estimated dense flux table peak is {estimated:.1f} MB per chunk, above "
                f"outputs.max_dense_flux_memory_mb={self.output_config.max_dense_flux_memory_mb}. "
                "Reduce chunk_size, select fewer reactions, use long/nonzero storage, or increase the guard intentionally."
            )

    @staticmethod
    def _validate_selected(ids: list[str], valid: set[str], label: str) -> None:
        missing = [item for item in ids if item not in valid]
        if missing:
            raise KeyError(f"{label} not found in model: {missing[:10]}")

    @staticmethod
    def _group_records(frame: pd.DataFrame) -> dict[int, list[dict[str, Any]]]:
        if frame.empty:
            return {}
        return {
            int(sample_id): group.drop(columns="sample_id").to_dict(orient="records")
            for sample_id, group in frame.groupby("sample_id", sort=False)
        }

    def run_chunk(self, batch: ConditionBatch, chunk_id: int) -> SimulationTables:
        batch.validate()
        result_rows: list[dict[str, Any]] = []
        log_rows: list[dict[str, Any]] = []
        applied_media_rows: list[dict[str, Any]] = []
        error_rows: list[dict[str, Any]] = []

        exchange_fluxes = _SeriesAccumulator(
            self.exchange_ids,
            self.output_config.store_exchange_fluxes,
            self.output_config.selected_exchange_fluxes,
            self.output_config.flux_table_format,
            self.output_config.flux_zero_tolerance,
            len(batch.metadata),
            self.output_config.max_dense_flux_memory_mb,
            "reaction_id",
            "flux",
        )
        reaction_fluxes = _SeriesAccumulator(
            self.reaction_ids,
            self.output_config.store_reaction_fluxes,
            self.output_config.selected_reaction_fluxes,
            self.output_config.flux_table_format,
            self.output_config.flux_zero_tolerance,
            len(batch.metadata),
            self.output_config.max_dense_flux_memory_mb,
            "reaction_id",
            "flux",
        )
        shadow_prices = _SeriesAccumulator(
            self.metabolite_ids,
            self.diagnostics_config.store_shadow_prices,
            self.diagnostics_config.selected_shadow_price_metabolites,
            "long",
            self.diagnostics_config.dual_zero_tolerance,
            len(batch.metadata),
            self.output_config.max_dense_flux_memory_mb,
            "metabolite_id",
            "shadow_price",
        )
        reduced_costs = _SeriesAccumulator(
            self.reaction_ids,
            self.diagnostics_config.store_reduced_costs,
            self.diagnostics_config.selected_reduced_cost_reactions,
            "long",
            self.diagnostics_config.dual_zero_tolerance,
            len(batch.metadata),
            self.output_config.max_dense_flux_memory_mb,
            "reaction_id",
            "reduced_cost",
        )
        active_constraints = _ActiveConstraintAccumulator(
            self.model,
            self.exchange_ids,
            self.reaction_ids,
            self.diagnostics_config,
        )

        requested = batch.requested_media.set_index("sample_id", drop=True)
        gene_groups = self._group_records(batch.gene_perturbations)
        reaction_groups = self._group_records(batch.reaction_perturbations)
        objective_groups = self._group_records(batch.objective_perturbations)
        chunk_start = time.perf_counter()

        for chunk_index, condition in batch.metadata.reset_index(drop=True).iterrows():
            sample_id = int(condition["sample_id"])
            start = time.perf_counter()
            event = {
                "event_time_utc": datetime.now(timezone.utc).isoformat(),
                "sample_id": sample_id,
                "run_number": sample_id + 1,
                "total_runs": int(self.simulation_config.n_simulations),
                "chunk_id": int(chunk_id),
                "chunk_index": int(chunk_index),
                "analysis_type": condition["analysis_type"],
                "design_stage": condition["design_stage"],
                "sampling_strategy": condition["sampling_strategy"],
                "method": self.simulation_config.method,
            }
            genes = gene_groups.get(sample_id, [])
            reactions = reaction_groups.get(sample_id, [])
            objectives = objective_groups.get(sample_id, [])
            sampled_medium = requested.loc[sample_id].to_dict()
            actual_medium: dict[str, float] | None = None

            try:
                self.medium_applier.apply(sampled_medium)
                has_nonmedium_perturbation = bool(genes or reactions or objectives)
                if has_nonmedium_perturbation:
                    with self.model:
                        self._apply_nonmedium_perturbations(genes, reactions, objectives)
                        actual_medium = self._current_medium()
                        outcome = self._solve(use_configured_optimization=not bool(objectives))
                        self._record_solution(
                            sample_id,
                            chunk_id,
                            condition,
                            genes,
                            reactions,
                            objectives,
                            outcome,
                            start,
                            result_rows,
                            event,
                            exchange_fluxes,
                            reaction_fluxes,
                            shadow_prices,
                            reduced_costs,
                            active_constraints,
                        )
                else:
                    actual_medium = self._current_medium()
                    outcome = self._solve(use_configured_optimization=not bool(objectives))
                    self._record_solution(
                        sample_id,
                        chunk_id,
                        condition,
                        genes,
                        reactions,
                        objectives,
                        outcome,
                        start,
                        result_rows,
                        event,
                        exchange_fluxes,
                        reaction_fluxes,
                        shadow_prices,
                        reduced_costs,
                        active_constraints,
                    )
            except Exception as error:
                elapsed = time.perf_counter() - start
                result_rows.append(
                    {
                        "sample_id": sample_id,
                        "run_number": sample_id + 1,
                        "total_runs": int(self.simulation_config.n_simulations),
                        "chunk_id": int(chunk_id),
                        "status": "error",
                        "feasible": False,
                        "objective_value": np.nan,
                        "biomass_flux": np.nan,
                        "product_flux": np.nan,
                        "optimization_mode": self.optimization_config.mode,
                        "maximum_growth": np.nan,
                        "growth_floor": np.nan,
                        "biomass_stage_status": None,
                        "method": self.simulation_config.method,
                        "runtime_seconds": elapsed,
                        "n_gene_perturbations": len(genes),
                        "n_reaction_perturbations": len(reactions),
                        "n_objective_terms": len(objectives),
                    }
                )
                error_rows.append(
                    {
                        "sample_id": sample_id,
                        "chunk_id": int(chunk_id),
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                event.update(
                    {
                        "status": "error",
                        "feasible": False,
                        "runtime_seconds": elapsed,
                        "error_type": type(error).__name__,
                        "error_message": str(error),
                    }
                )
                LOGGER.exception("Exact simulation failed for sample_id=%s", sample_id)
                if actual_medium is None:
                    try:
                        actual_medium = self._current_medium()
                    except Exception:
                        actual_medium = {str(key): float(value) for key, value in sampled_medium.items()}
                if self.simulation_config.fail_fast:
                    raise
            if self.output_config.store_applied_media and actual_medium is not None:
                applied_media_rows.append({"sample_id": sample_id, **actual_medium})
            log_rows.append(event)
            self._maybe_log_progress(chunk_id, chunk_index, len(batch.metadata), event)

        results = pd.DataFrame(result_rows)
        if not results.empty:
            LOGGER.info(
                "Chunk %s solved | samples=%s | feasible=%s | statuses=%s | elapsed=%.3fs | mean_solve=%.5fs",
                chunk_id,
                len(results),
                int(results["feasible"].sum()),
                results["status"].value_counts(dropna=False).to_dict(),
                time.perf_counter() - chunk_start,
                float(results["runtime_seconds"].mean()),
            )

        condition_metadata = batch.metadata.copy()
        condition_metadata.insert(1, "chunk_id", int(chunk_id))
        return SimulationTables(
            condition_metadata=condition_metadata,
            requested_media=batch.requested_media.copy() if self.output_config.store_requested_media else pd.DataFrame(),
            applied_media=pd.DataFrame(applied_media_rows) if self.output_config.store_applied_media else pd.DataFrame(),
            gene_perturbations=batch.gene_perturbations.copy(),
            reaction_perturbations=batch.reaction_perturbations.copy(),
            objective_perturbations=batch.objective_perturbations.copy(),
            results=results,
            simulation_log=pd.DataFrame(log_rows),
            exchange_fluxes=exchange_fluxes.to_frame(),
            reaction_fluxes=reaction_fluxes.to_frame(),
            shadow_prices=shadow_prices.to_frame(),
            reduced_costs=reduced_costs.to_frame(),
            active_constraints=active_constraints.to_frame(),
            errors=pd.DataFrame(error_rows),
        )

    def _apply_nonmedium_perturbations(
        self,
        gene_records: list[dict[str, Any]],
        reaction_records: list[dict[str, Any]],
        objective_records: list[dict[str, Any]],
    ) -> None:
        for record in gene_records:
            gene_id = str(record["gene_id"])
            activity = float(record.get("activity", 0.0))
            if gene_id not in self.gene_ids:
                raise KeyError(f"Gene not found: {gene_id}")
            if activity != 0.0:
                raise NotImplementedError(
                    "The exact first-layer pipeline currently supports gene knockout activity=0 only. "
                    "Continuous gene knockdown requires an explicit GPR-to-reaction mapping policy."
                )
            self.model.genes.get_by_id(gene_id).knock_out()

        for record in reaction_records:
            reaction_id = str(record["reaction_id"])
            if reaction_id not in set(self.reaction_ids):
                raise KeyError(f"Reaction not found: {reaction_id}")
            reaction = self.model.reactions.get_by_id(reaction_id)
            lower = float(record["lower_bound"])
            upper = float(record["upper_bound"])
            if lower > upper:
                raise ValueError(f"Invalid perturbation bounds for {reaction_id}: {lower} > {upper}")
            reaction.bounds = (lower, upper)

        if objective_records:
            terms: dict[Any, float] = {}
            directions = set()
            for record in objective_records:
                reaction_id = str(record["reaction_id"])
                if reaction_id not in set(self.reaction_ids):
                    raise KeyError(f"Objective reaction not found: {reaction_id}")
                terms[self.model.reactions.get_by_id(reaction_id)] = float(record["coefficient"])
                direction = record.get("objective_direction")
                if direction is not None and not pd.isna(direction):
                    directions.add(str(direction))
            if len(directions) > 1:
                raise ValueError(f"Conflicting objective directions: {sorted(directions)}")
            self.model.objective = terms
            if directions:
                direction = next(iter(directions))
                if direction not in {"max", "min"}:
                    raise ValueError(f"Invalid objective direction: {direction}")
                self.model.objective_direction = direction

    def _current_medium(self) -> dict[str, float]:
        return {
            reaction_id: max(0.0, -float(self.model.reactions.get_by_id(reaction_id).lower_bound))
            for reaction_id in self.medium_space.output_reaction_ids
        }

    def _record_solution(
        self,
        sample_id: int,
        chunk_id: int,
        condition: pd.Series,
        genes: list[dict[str, Any]],
        reactions: list[dict[str, Any]],
        objectives: list[dict[str, Any]],
        outcome: SolveOutcome,
        start: float,
        result_rows: list[dict[str, Any]],
        event: dict[str, Any],
        exchange_fluxes: _SeriesAccumulator,
        reaction_fluxes: _SeriesAccumulator,
        shadow_prices: _SeriesAccumulator,
        reduced_costs: _SeriesAccumulator,
        active_constraints: _ActiveConstraintAccumulator,
    ) -> None:
        elapsed = time.perf_counter() - start
        solution = outcome.solution
        status = str(solution.status)
        feasible = status in self.simulation_config.feasible_statuses
        fluxes = solution.fluxes if feasible else pd.Series(dtype=float)
        objective_value = float(solution.objective_value) if solution.objective_value is not None else np.nan
        biomass = self._get_flux(fluxes, self.model_config.biomass_reaction_id)
        product = self._get_flux(fluxes, self.model_config.product_reaction_id)
        result_rows.append(
            {
                "sample_id": sample_id,
                "run_number": sample_id + 1,
                "total_runs": int(self.simulation_config.n_simulations),
                "chunk_id": int(chunk_id),
                "status": status,
                "feasible": bool(feasible),
                "objective_value": objective_value,
                "biomass_flux": biomass,
                "product_flux": product,
                "optimization_mode": outcome.optimization_mode,
                "maximum_growth": outcome.maximum_growth,
                "growth_floor": outcome.growth_floor,
                "biomass_stage_status": outcome.biomass_stage_status,
                "method": self.simulation_config.method,
                "runtime_seconds": elapsed,
                "n_gene_perturbations": len(genes),
                "n_reaction_perturbations": len(reactions),
                "n_objective_terms": len(objectives),
            }
        )
        event.update(
            {
                "status": status,
                "feasible": bool(feasible),
                "objective_value": objective_value,
                "biomass_flux": biomass,
                "product_flux": product,
                "optimization_mode": outcome.optimization_mode,
                "maximum_growth": outcome.maximum_growth,
                "growth_floor": outcome.growth_floor,
                "biomass_stage_status": outcome.biomass_stage_status,
                "runtime_seconds": elapsed,
                "error_type": None,
                "error_message": None,
            }
        )
        if feasible:
            exchange_fluxes.add(sample_id, fluxes)
            reaction_fluxes.add(sample_id, fluxes)
            shadow_prices.add(sample_id, getattr(solution, "shadow_prices", None))
            reduced_costs.add(sample_id, getattr(solution, "reduced_costs", None))
            bound_overrides: dict[str, tuple[float, float]] = {}
            if outcome.growth_floor is not None and self.model_config.biomass_reaction_id:
                biomass_reaction = self.model.reactions.get_by_id(self.model_config.biomass_reaction_id)
                bound_overrides[self.model_config.biomass_reaction_id] = (
                    float(outcome.growth_floor),
                    float(biomass_reaction.upper_bound),
                )
            active_constraints.add(sample_id, fluxes, bound_overrides)
        else:
            exchange_fluxes.add(sample_id, None)
            reaction_fluxes.add(sample_id, None)
            shadow_prices.add(sample_id, None)
            reduced_costs.add(sample_id, None)

    def _single_solve(self):
        if self.simulation_config.method == "fba":
            return self.model.optimize()
        if self.simulation_config.method == "pfba":
            from cobra.flux_analysis import pfba

            return pfba(self.model)
        raise ValueError(f"Unknown simulation method: {self.simulation_config.method}")

    def _solve(self, *, use_configured_optimization: bool = True) -> SolveOutcome:
        mode = self.optimization_config.mode if use_configured_optimization else "native_objective"
        if mode == "auto":
            mode = (
                "growth_constrained_product"
                if self.model_config.biomass_reaction_id and self.model_config.product_reaction_id
                else "native_objective"
            )
        if mode == "native_objective":
            return SolveOutcome(self._single_solve(), mode)

        biomass_id = self.model_config.biomass_reaction_id
        product_id = self.model_config.product_reaction_id
        if mode in {"maximize_biomass", "growth_constrained_product"} and not biomass_id:
            raise ValueError(f"Optimization mode {mode} requires a biomass reaction.")
        if mode in {"maximize_product", "growth_constrained_product"} and not product_id:
            raise ValueError(f"Optimization mode {mode} requires a product reaction.")

        with self.model:
            if mode == "maximize_biomass":
                self.model.objective = self.model.reactions.get_by_id(str(biomass_id))
                self.model.objective_direction = "max"
                return SolveOutcome(self._single_solve(), mode)
            if mode == "maximize_product":
                self.model.objective = self.model.reactions.get_by_id(str(product_id))
                self.model.objective_direction = "max"
                return SolveOutcome(self._single_solve(), mode)

            biomass_reaction = self.model.reactions.get_by_id(str(biomass_id))
            product_reaction = self.model.reactions.get_by_id(str(product_id))
            self.model.objective = biomass_reaction
            self.model.objective_direction = "max"
            biomass_solution = self.model.optimize()
            biomass_status = str(biomass_solution.status)
            if biomass_status not in self.simulation_config.feasible_statuses:
                return SolveOutcome(
                    biomass_solution,
                    mode,
                    maximum_growth=None,
                    growth_floor=None,
                    biomass_stage_status=biomass_status,
                )
            maximum_growth = float(biomass_solution.fluxes.get(str(biomass_id), biomass_solution.objective_value))
            growth_floor = max(float(biomass_reaction.lower_bound), self.optimization_config.minimum_growth_fraction * maximum_growth)
            if growth_floor > float(biomass_reaction.upper_bound):
                raise ValueError(
                    f"Computed growth floor {growth_floor} exceeds upper bound {biomass_reaction.upper_bound}."
                )
            biomass_reaction.lower_bound = growth_floor
            self.model.objective = product_reaction
            self.model.objective_direction = "max"
            product_solution = self._single_solve()
            return SolveOutcome(
                product_solution,
                mode,
                maximum_growth=maximum_growth,
                growth_floor=growth_floor,
                biomass_stage_status=biomass_status,
            )

    @staticmethod
    def _get_flux(fluxes: pd.Series, reaction_id: str | None) -> float:
        if reaction_id is None or fluxes.empty:
            return np.nan
        return float(fluxes.get(reaction_id, np.nan))

    @staticmethod
    def _format_log_value(value: Any, *, precision: int = 6) -> str:
        """Format numeric simulation outputs consistently for human-readable logs."""
        if value is None:
            return "n/a"
        try:
            number = float(value)
        except (TypeError, ValueError):
            return str(value)
        if not np.isfinite(number):
            return "n/a"
        return f"{number:.{precision}g}"

    def _maybe_log_progress(self, chunk_id: int, index: int, total: int, event: dict[str, Any]) -> None:
        every = self.simulation_config.simulation_log_every
        if every <= 0:
            return
        run_number = int(event.get("run_number", int(event.get("sample_id", -1)) + 1))
        total_runs = int(event.get("total_runs", self.simulation_config.n_simulations))
        is_last = index + 1 == total
        if run_number % every == 0 or run_number == 1 or run_number == total_runs or is_last or event.get("status") == "error":
            status = str(event.get("status", "unknown"))
            feasible = "yes" if bool(event.get("feasible", False)) else "no"
            message = (
                "FBA run %s/%s | sample_id=%s | chunk=%s (%s/%s) | "
                "status=%s | feasible=%s | biomass=%s | product=%s | "
                "objective=%s | max_growth=%s | growth_floor=%s | runtime=%ss"
            )
            args = (
                run_number,
                total_runs,
                event.get("sample_id"),
                chunk_id,
                index + 1,
                total,
                status,
                feasible,
                self._format_log_value(event.get("biomass_flux")),
                self._format_log_value(event.get("product_flux")),
                self._format_log_value(event.get("objective_value")),
                self._format_log_value(event.get("maximum_growth")),
                self._format_log_value(event.get("growth_floor")),
                self._format_log_value(event.get("runtime_seconds"), precision=5),
            )
            if status == "error":
                LOGGER.error(
                    message + " | error=%s: %s",
                    *args,
                    event.get("error_type", "Error"),
                    event.get("error_message", "Unknown simulation error"),
                )
            else:
                LOGGER.info(message, *args)

