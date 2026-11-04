from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
import pandas as pd

from .config import ExperimentConfig, SamplingConfig, SimulationConfig
from .medium import MediumSpace
from .sampling import DesignSampler


@dataclass(frozen=True)
class SimulationCondition:
    """Serializable representation of one exact optimization condition.

    The high-throughput media path uses :class:`ConditionBatch` to avoid creating
    thousands of Python objects, but this object is useful for future query and
    active-learning modules.
    """

    sample_id: int
    analysis_type: str
    design_stage: str
    sampling_strategy: str
    sampling_batch: int
    parent_query_id: str | None = None
    acquisition_score: float | None = None
    medium_uptakes: dict[str, float] = field(default_factory=dict)
    gene_knockouts: tuple[str, ...] = ()
    reaction_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)
    objective_terms: dict[str, float] = field(default_factory=dict)
    objective_direction: str | None = None


@dataclass(frozen=True)
class ConditionBatch:
    """Columnar condition batch designed for efficient chunked simulation."""

    metadata: pd.DataFrame
    requested_media: pd.DataFrame
    gene_perturbations: pd.DataFrame
    reaction_perturbations: pd.DataFrame
    objective_perturbations: pd.DataFrame

    @property
    def sample_ids(self) -> np.ndarray:
        return self.metadata["sample_id"].to_numpy(dtype=np.int64, copy=False)

    def validate(self) -> None:
        required = {
            "sample_id",
            "analysis_type",
            "design_stage",
            "sampling_strategy",
            "sampling_batch",
        }
        missing = required - set(self.metadata.columns)
        if missing:
            raise ValueError(f"Condition metadata is missing columns: {sorted(missing)}")
        if self.metadata["sample_id"].duplicated().any():
            raise ValueError("Condition sample_id values must be unique within a batch.")
        expected = set(self.metadata["sample_id"].astype(int))
        for name, frame in {
            "requested_media": self.requested_media,
            "gene_perturbations": self.gene_perturbations,
            "reaction_perturbations": self.reaction_perturbations,
            "objective_perturbations": self.objective_perturbations,
        }.items():
            if not frame.empty:
                if "sample_id" not in frame.columns:
                    raise ValueError(f"{name} must contain sample_id.")
                extra = set(frame["sample_id"].astype(int)) - expected
                if extra:
                    raise ValueError(f"{name} contains sample IDs outside the batch: {sorted(extra)[:10]}")
        if set(self.requested_media["sample_id"].astype(int)) != expected:
            raise ValueError("requested_media must have exactly one row per condition.")


class ConditionGenerator(Protocol):
    def prepare(self): ...
    def generate_chunk(self, start: int, stop: int) -> ConditionBatch: ...


class MediaConditionGenerator:
    """Generate the initial continuous medium design using a cached sampler."""

    def __init__(
        self,
        medium_space: MediumSpace,
        sampling_config: SamplingConfig,
        simulation_config: SimulationConfig,
        experiment_config: ExperimentConfig,
        cache_dir=None,
    ):
        self.medium_space = medium_space
        self.sampling_config = sampling_config
        self.simulation_config = simulation_config
        self.experiment_config = experiment_config
        self.sampler = DesignSampler(
            medium_space=medium_space,
            sampling_config=sampling_config,
            simulation_config=simulation_config,
            cache_dir=cache_dir,
        )

    def prepare(self):
        return self.sampler.prepare()

    def attach_cache(self, path) -> None:
        self.sampler.attach_cache(path)

    def generate_chunk(self, start: int, stop: int) -> ConditionBatch:
        requested_media = self.sampler.sample_chunk(start, stop)
        sample_ids = requested_media["sample_id"].to_numpy(dtype=np.int64, copy=False)
        metadata = pd.DataFrame(
            {
                "sample_id": sample_ids,
                "analysis_type": self.experiment_config.analysis_type,
                "design_stage": self.experiment_config.design_stage,
                "sampling_strategy": self.sampling_config.strategy,
                "sampling_batch": np.full(len(sample_ids), self.experiment_config.sampling_batch, dtype=np.int32),
                "parent_query_id": self.experiment_config.parent_query_id,
                "acquisition_score": np.nan,
                "experiment_notes": self.experiment_config.notes,
            }
        )
        batch = ConditionBatch(
            metadata=metadata,
            requested_media=requested_media,
            gene_perturbations=pd.DataFrame(columns=["sample_id", "gene_id", "activity"]),
            reaction_perturbations=pd.DataFrame(
                columns=["sample_id", "reaction_id", "lower_bound", "upper_bound"]
            ),
            objective_perturbations=pd.DataFrame(
                columns=["sample_id", "reaction_id", "coefficient", "objective_direction"]
            ),
        )
        batch.validate()
        return batch
