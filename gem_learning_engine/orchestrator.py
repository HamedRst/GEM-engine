from __future__ import annotations

import logging
import math
import os
import resource
import sys
import multiprocessing as mp
import time
from logging.handlers import QueueHandler, QueueListener
from dataclasses import replace
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Iterator

import psutil
from tqdm import tqdm

from .conditions import MediaConditionGenerator
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
    load_config,
)
from .fba import FBAEngine
from .inference import resolve_inferred_config
from .logging_utils import setup_logging
from .medium import build_medium_space
from .model_io import (
    configure_solver,
    exchange_catalog,
    gene_catalog,
    load_model,
    metabolite_catalog,
    model_fingerprint,
    reaction_catalog,
    validate_model_inputs,
)
from .storage import Checkpoint, ResultsStore

LOGGER = logging.getLogger("gem_learning_engine")
_WORKER_CONTEXT: dict[str, Any] = {}


def chunk_ranges(n_simulations: int, chunk_size: int) -> list[tuple[int, int, int]]:
    return [
        (chunk_id, chunk_id * chunk_size, min(n_simulations, (chunk_id + 1) * chunk_size))
        for chunk_id in range(math.ceil(n_simulations / chunk_size))
    ]


def _rss_mb() -> float:
    try:
        return psutil.Process().memory_info().rss / (1024**2)
    except (psutil.Error, OSError):
        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024**2 if sys.platform == "darwin" else 1024)


class SimulationOrchestrator:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.store = ResultsStore(config)

    def run(self) -> None:
        model = load_model(self.config.model)
        configure_solver(model, self.config.solver)
        resolved_model, resolved_optimization, inference_report = resolve_inferred_config(
            model, self.config.model, self.config.optimization
        )
        self.config = replace(
            self.config, model=resolved_model, optimization=resolved_optimization
        )
        self.store = ResultsStore(self.config)
        if resolved_model.objective_reaction_id:
            model.objective = model.reactions.get_by_id(resolved_model.objective_reaction_id)
        model.objective_direction = resolved_model.objective_direction
        validate_model_inputs(model, resolved_model)
        fingerprint = model_fingerprint(model, resolved_model.path)

        self.store.initialize(fingerprint)
        inference_report.write_json(self.store.paths.metadata_dir / "model_inference.json")
        logger = setup_logging(
            self.config.project.output_dir,
            level=self.config.logging.level,
            log_to_console=self.config.logging.log_to_console,
            log_to_file=self.config.logging.log_to_file,
            structured_jsonl=self.config.logging.structured_jsonl,
            max_file_mb=self.config.logging.max_file_mb,
            backup_count=self.config.logging.backup_count,
        )
        logger.info("Starting project=%s engine_config_hash=%s", self.config.project.name, self.config.data_hash())
        logger.info("Model=%s fingerprint=%s solver=%s", self.config.model.path, fingerprint[:20], self.config.solver.name)
        logger.info(
            "Inferred biomass=%s product=%s optimization=%s",
            self.config.model.biomass_reaction_id,
            self.config.model.product_reaction_id,
            self.config.optimization.mode,
        )

        medium_space = build_medium_space(model, self.config.sampling, self.config.simulation)
        generator = MediaConditionGenerator(
            medium_space,
            self.config.sampling,
            self.config.simulation,
            self.config.experiment,
            cache_dir=self.store.paths.cache_dir,
        )
        design_cache_path = generator.prepare()

        catalogs = {
            "reaction_catalog": reaction_catalog(model),
            "exchange_catalog": exchange_catalog(model),
            "gene_catalog": gene_catalog(model),
            "metabolite_catalog": metabolite_catalog(model),
            "medium_space": medium_space.to_frame(),
        }
        for name, frame in catalogs.items():
            self.store.write_catalog(name, frame)
        model_summary = {
            "model_id": getattr(model, "id", None),
            "n_reactions": len(model.reactions),
            "n_metabolites": len(model.metabolites),
            "n_genes": len(model.genes),
            "n_exchanges": len(medium_space.exchange_ids),
            "n_variable_medium_components": len(medium_space.bounds),
            "n_fixed_medium_components": len(medium_space.fixed_uptakes),
            "biomass_reaction_id": self.config.model.biomass_reaction_id,
            "product_reaction_id": self.config.model.product_reaction_id,
            "optimization_mode": self.config.optimization.mode,
            "minimum_growth_fraction": self.config.optimization.minimum_growth_fraction,
        }
        self.store.write_run_manifest(fingerprint, model_summary)
        self.store.write_design_metadata(
            {
                "strategy": self.config.sampling.strategy,
                "seed": self.config.simulation.seed,
                "n_simulations": self.config.simulation.n_simulations,
                "dimension": len(medium_space.bounds),
                "cache_path": str(design_cache_path) if design_cache_path else None,
                "analysis_type": self.config.experiment.analysis_type,
                "design_stage": self.config.experiment.design_stage,
                "sampling_batch": self.config.experiment.sampling_batch,
                "note": (
                    "Scrambled Sobol is the default initial space-filling design. "
                    "Active-learning and perturbation generators use the same ConditionBatch schema in later layers."
                ),
            }
        )

        ranges = chunk_ranges(self.config.simulation.n_simulations, self.config.simulation.chunk_size)
        checkpoint = self.store.load_checkpoint()
        self._validate_checkpoint(checkpoint, fingerprint)
        todo: list[tuple[int, int, int]] = []
        completed = set(checkpoint.completed_chunks)
        for chunk_id, start, stop in ranges:
            if self.config.project.resume and (chunk_id in completed or self.store.is_chunk_complete(chunk_id)):
                checkpoint.completed_chunks = sorted(set(checkpoint.completed_chunks + [chunk_id]))
                self.store.save_checkpoint(checkpoint)
                logger.info("Skipping verified chunk_id=%s", chunk_id)
            else:
                todo.append((chunk_id, start, stop))

        logger.info(
            "Discovered exchanges=%s | variable=%s | fixed=%s | excluded=%s",
            len(medium_space.exchange_ids),
            len(medium_space.bounds),
            len(medium_space.fixed_uptakes),
            len(medium_space.excluded_reactions),
        )
        logger.info(
            "Chunks total=%s todo=%s chunk_size=%s processes=%s in_flight_limit=%s",
            len(ranges),
            len(todo),
            self.config.simulation.chunk_size,
            self.config.simulation.processes,
            self.config.simulation.max_in_flight_chunks or 2 * self.config.simulation.processes,
        )

        start_time = time.perf_counter()
        if self.store.pause_requested():
            logger.warning("Pause requested before execution. Clear it and rerun to resume.")
        elif self.config.simulation.processes == 1:
            engine = FBAEngine(
                model,
                self.config.model,
                self.config.optimization,
                self.config.simulation,
                self.config.outputs,
                self.config.diagnostics,
                medium_space,
            )
            self._run_sequential(engine, generator, todo, checkpoint)
        else:
            del model
            self._run_parallel(todo, checkpoint, fingerprint, design_cache_path)

        self._write_final_summary(logger, time.perf_counter() - start_time, len(ranges))
        final_status = self.store.status(expected_chunks=len(ranges))
        if not final_status["pause_requested"]:
            incomplete = len(final_status["complete_chunks"]) != len(ranges)
            sample_mismatch = final_status["saved_simulations"] != self.config.simulation.n_simulations
            sample_errors = int(final_status["status_counts"].get("error", 0))
            if final_status["failed_chunks"] or final_status["invalid_chunks"] or incomplete or sample_mismatch or sample_errors:
                raise RuntimeError(
                    "Run did not pass completion verification: "
                    f"failed_chunks={final_status['failed_chunks']}, invalid_chunks={final_status['invalid_chunks']}, "
                    f"complete={len(final_status['complete_chunks'])}/{len(ranges)}, "
                    f"saved={final_status['saved_simulations']}/{self.config.simulation.n_simulations}, errors={sample_errors}."
                )

    def _validate_checkpoint(self, checkpoint: Checkpoint, fingerprint: str) -> None:
        if not self.config.project.resume:
            return
        if checkpoint.config_hash and checkpoint.config_hash != self.config.data_hash():
            raise ValueError(
                "Checkpoint config hash does not match current data settings: "
                f"checkpoint={checkpoint.config_hash}, current={self.config.data_hash()}"
            )
        if checkpoint.model_fingerprint and checkpoint.model_fingerprint != fingerprint:
            raise ValueError("The model file/structure changed; resume is unsafe. Use a new output directory.")

    def _run_sequential(
        self,
        engine: FBAEngine,
        generator: MediaConditionGenerator,
        todo: list[tuple[int, int, int]],
        checkpoint: Checkpoint,
    ) -> None:
        logger = logging.getLogger("gem_learning_engine")
        for chunk_id, start, stop in tqdm(todo, desc="FBA chunks"):
            if self.store.pause_requested():
                logger.warning("Pause requested before chunk_id=%s. Stopping at chunk boundary.", chunk_id)
                break
            attempts = 0
            while True:
                attempts += 1
                rss_before = _rss_mb()
                chunk_start = time.perf_counter()
                try:
                    batch = generator.generate_chunk(start, stop)
                    solve_start = time.perf_counter()
                    tables = engine.run_chunk(batch, chunk_id)
                    solve_elapsed = time.perf_counter() - solve_start
                    performance = {
                        "attempt": attempts,
                        "range_start": start,
                        "range_stop": stop,
                        "solve_seconds": solve_elapsed,
                        "total_seconds_before_write": time.perf_counter() - chunk_start,
                        "rss_mb_before": rss_before,
                        "rss_mb_after_solve": _rss_mb(),
                    }
                    manifest = self.store.write_chunk(chunk_id, tables, performance=performance)
                    checkpoint.completed_chunks = sorted(set(checkpoint.completed_chunks + [chunk_id]))
                    checkpoint.failed_chunks = [item for item in checkpoint.failed_chunks if item != chunk_id]
                    self.store.save_checkpoint(checkpoint)
                    logger.info(
                        "Saved chunk_id=%s rows=%s attempt=%s elapsed=%.3fs rss=%.1fMB",
                        chunk_id,
                        manifest["row_counts"],
                        attempts,
                        time.perf_counter() - chunk_start,
                        _rss_mb(),
                    )
                    break
                except Exception as error:
                    logger.exception("Chunk %s attempt %s failed: %s", chunk_id, attempts, error)
                    if attempts > self.config.simulation.chunk_retry_attempts:
                        checkpoint.failed_chunks = sorted(set(checkpoint.failed_chunks + [chunk_id]))
                        self.store.save_checkpoint(checkpoint)
                        if self.config.simulation.fail_fast:
                            raise
                        break

    def _run_parallel(
        self,
        todo: list[tuple[int, int, int]],
        checkpoint: Checkpoint,
        fingerprint: str,
        design_cache_path: Path | None,
    ) -> None:
        logger = logging.getLogger("gem_learning_engine")
        workers = self.config.simulation.processes
        limit = self.config.simulation.max_in_flight_chunks or max(1, 2 * workers)
        limit = max(limit, workers)
        iterator: Iterator[tuple[int, int, int]] = iter(todo)
        pending: dict[Future, tuple[int, int, int, int]] = {}
        retry_queue: list[tuple[int, int, int, int]] = []
        stop_submitting = False

        def next_job() -> tuple[int, int, int, int] | None:
            if retry_queue:
                return retry_queue.pop(0)
            try:
                chunk_id, start, stop = next(iterator)
                return chunk_id, start, stop, 1
            except StopIteration:
                return None

        def submit_until_full(executor: ProcessPoolExecutor) -> None:
            nonlocal stop_submitting
            while len(pending) < limit and not stop_submitting:
                if self.store.pause_requested():
                    stop_submitting = True
                    logger.warning("Pause requested; no additional chunks will be submitted.")
                    return
                job = next_job()
                if job is None:
                    return
                chunk_id, start, stop, attempt = job
                future = executor.submit(_worker_run_and_save_chunk, chunk_id, start, stop, attempt)
                pending[future] = job

        context = mp.get_context("spawn")
        log_queue = context.Queue()
        listener = QueueListener(log_queue, *logger.handlers, respect_handler_level=True)
        listener.start()
        try:
            with ProcessPoolExecutor(
                max_workers=workers,
                mp_context=context,
                initializer=_worker_init,
                initargs=(
                    self.config.to_dict(),
                    fingerprint,
                    str(design_cache_path) if design_cache_path else None,
                    log_queue,
                ),
            ) as executor, tqdm(total=len(todo), desc="FBA chunks") as progress:
                submit_until_full(executor)
                completed_progress: set[int] = set()
                while pending:
                    done, _ = wait(set(pending), return_when=FIRST_COMPLETED)
                    for future in done:
                        chunk_id, start, stop, attempt = pending.pop(future)
                        try:
                            result = future.result()
                            checkpoint.completed_chunks = sorted(set(checkpoint.completed_chunks + [chunk_id]))
                            checkpoint.failed_chunks = [item for item in checkpoint.failed_chunks if item != chunk_id]
                            logger.info(
                                "Worker saved chunk_id=%s attempt=%s rows=%s elapsed=%.3fs rss_after=%.1fMB",
                                chunk_id,
                                attempt,
                                result["manifest"]["row_counts"],
                                result["elapsed_seconds"],
                                result["rss_mb_after"],
                            )
                            if chunk_id not in completed_progress:
                                progress.update(1)
                                completed_progress.add(chunk_id)
                        except Exception as error:
                            logger.exception("Worker chunk %s attempt %s failed: %s", chunk_id, attempt, error)
                            if attempt <= self.config.simulation.chunk_retry_attempts:
                                retry_queue.append((chunk_id, start, stop, attempt + 1))
                            else:
                                checkpoint.failed_chunks = sorted(set(checkpoint.failed_chunks + [chunk_id]))
                                if chunk_id not in completed_progress:
                                    progress.update(1)
                                    completed_progress.add(chunk_id)
                                if self.config.simulation.fail_fast:
                                    for other in pending:
                                        other.cancel()
                                    raise
                        finally:
                            self.store.save_checkpoint(checkpoint)
                    submit_until_full(executor)
        finally:
            listener.stop()
            log_queue.close()
            log_queue.join_thread()

    def _write_final_summary(self, logger: logging.Logger, elapsed: float, expected_chunks: int) -> None:
        statistics = self.store.aggregate_result_statistics()
        if statistics["saved_simulations"] == 0:
            logger.warning("No exact result metadata were saved.")
        else:
            logger.info(
                "Run complete/paused | saved=%s feasible=%s elapsed=%.2fs mean_runtime=%.5fs",
                statistics["saved_simulations"],
                statistics["feasible_simulations"],
                elapsed,
                statistics["mean_runtime_seconds"],
            )
            logger.info("Solver status counts: %s", statistics["status_counts"])
            self.store.write_json(self.store.paths.metadata_dir / "summary_statistics.json", statistics)
        self.store.write_json(
            self.store.paths.metadata_dir / "run_status.json",
            self.store.status(expected_chunks=expected_chunks),
        )


def _config_from_dict(data: dict[str, Any]) -> PipelineConfig:
    return PipelineConfig(
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


def _worker_init(
    config_dict: dict[str, Any],
    expected_fingerprint: str,
    cache_path: str | None,
    log_queue: Any | None = None,
) -> None:
    config = _config_from_dict(config_dict)
    worker_logger = logging.getLogger("gem_learning_engine")
    worker_logger.handlers.clear()
    worker_logger.setLevel(getattr(logging, config.logging.level.upper(), logging.INFO))
    if log_queue is None:
        worker_logger.addHandler(logging.NullHandler())
    else:
        worker_logger.addHandler(QueueHandler(log_queue))
    worker_logger.propagate = False

    model = load_model(config.model)
    configure_solver(model, config.solver)
    validate_model_inputs(model, config.model)
    fingerprint = model_fingerprint(model, config.model.path)
    if fingerprint != expected_fingerprint:
        raise RuntimeError("Worker model fingerprint differs from parent model fingerprint.")
    medium_space = build_medium_space(model, config.sampling, config.simulation)
    generator = MediaConditionGenerator(
        medium_space,
        config.sampling,
        config.simulation,
        config.experiment,
        cache_dir=None,
    )
    generator.attach_cache(cache_path)
    engine = FBAEngine(
        model,
        config.model,
        config.optimization,
        config.simulation,
        config.outputs,
        config.diagnostics,
        medium_space,
    )
    store = ResultsStore(config)
    _WORKER_CONTEXT.clear()
    _WORKER_CONTEXT.update({"generator": generator, "engine": engine, "store": store})


def _worker_run_and_save_chunk(chunk_id: int, start: int, stop: int, attempt: int) -> dict[str, Any]:
    if not _WORKER_CONTEXT:
        raise RuntimeError("Worker was not initialized.")
    before = _rss_mb()
    started = time.perf_counter()
    generator: MediaConditionGenerator = _WORKER_CONTEXT["generator"]
    engine: FBAEngine = _WORKER_CONTEXT["engine"]
    store: ResultsStore = _WORKER_CONTEXT["store"]
    batch = generator.generate_chunk(start, stop)
    solve_start = time.perf_counter()
    tables = engine.run_chunk(batch, chunk_id)
    solve_elapsed = time.perf_counter() - solve_start
    performance = {
        "attempt": attempt,
        "range_start": start,
        "range_stop": stop,
        "solve_seconds": solve_elapsed,
        "rss_mb_before": before,
        "rss_mb_after_solve": _rss_mb(),
        "worker_pid": os.getpid(),
    }
    manifest = store.write_chunk(chunk_id, tables, performance=performance)
    return {
        "chunk_id": chunk_id,
        "manifest": manifest,
        "elapsed_seconds": time.perf_counter() - started,
        "rss_mb_after": _rss_mb(),
    }


def run_from_config(config_path: str | Path) -> None:
    SimulationOrchestrator(load_config(config_path)).run()
