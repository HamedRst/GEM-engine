from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import qmc

from .config import SamplingConfig, SimulationConfig
from .medium import MediumSpace

LOGGER = logging.getLogger("gem_learning_engine")

_C1 = np.uint64(0x9E3779B97F4A7C15)
_C2 = np.uint64(0xBF58476D1CE4E5B9)
_C3 = np.uint64(0x94D049BB133111EB)


def _counter_uniform(sample_ids: np.ndarray, dimension: int, seed: int, stream: int = 0) -> np.ndarray:
    """Deterministic random values indexed by sample ID and dimension."""
    rows = np.asarray(sample_ids, dtype=np.uint64)[:, None]
    columns = np.arange(dimension, dtype=np.uint64)[None, :]
    with np.errstate(over="ignore"):
        value = np.uint64(seed) + rows * _C1 + columns * _C2 + np.uint64(stream) * _C3
        value ^= value >> np.uint64(30)
        value *= _C2
        value ^= value >> np.uint64(27)
        value *= _C3
        value ^= value >> np.uint64(31)
    return ((value >> np.uint64(11)).astype(np.float64)) * (1.0 / (1 << 53))


class DesignSampler:
    """Chunk-independent space-filling sampler with memory-mapped design cache.

    Scrambled Sobol is the default. The full low-discrepancy design is generated
    once, saved as a compact .npy file, and memory-mapped by the parent and all
    workers. For non-power-of-two requested sizes, the next power-of-two design
    is generated and the requested prefix is used; this retains extensibility and
    avoids regenerating the design during resume.
    """

    CACHE_VERSION = 4

    def __init__(
        self,
        medium_space: MediumSpace,
        sampling_config: SamplingConfig,
        simulation_config: SimulationConfig,
        cache_dir: str | Path | None = None,
    ):
        self.medium_space = medium_space
        self.sampling_config = sampling_config
        self.simulation_config = simulation_config
        self.reaction_ids = medium_space.reaction_ids
        self.minimums = np.asarray([bound.min_uptake for bound in medium_space.bounds], dtype=np.float64)
        self.maximums = np.asarray([bound.max_uptake for bound in medium_space.bounds], dtype=np.float64)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._array: np.ndarray | None = None
        self._cache_path: Path | None = None
        if not self.reaction_ids:
            raise ValueError("MediumSpace is empty.")

    def _cache_key(self) -> str:
        payload = {
            "strategy": self.sampling_config.strategy,
            "n": self.simulation_config.n_simulations,
            "seed": self.simulation_config.seed,
            "ids": self.reaction_ids,
            "mins": self.minimums.tolist(),
            "maxs": self.maximums.tolist(),
            "dtype": self.sampling_config.cache_dtype,
            "scramble": self.sampling_config.sobol_scramble,
            "skip": self.sampling_config.sobol_skip,
            "version": self.CACHE_VERSION,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:20]


    def generation_plan(self) -> dict[str, int | float]:
        """Estimate one-time design generation size and peak numeric memory."""
        n = self.simulation_config.n_simulations
        dimension = len(self.reaction_ids)
        if self.sampling_config.strategy == "sobol":
            total_needed = n + self.sampling_config.sobol_skip
            exponent = int(math.ceil(math.log2(max(1, total_needed))))
            generated_rows = 2**exponent
        elif self.sampling_config.strategy == "latin_hypercube":
            generated_rows = n
        else:
            generated_rows = 0
        # scipy.stats.qmc returns float64. During persistence, both the generated
        # matrix and target memmap can coexist briefly. Include a 15% safety
        # margin for temporary arrays and allocator overhead.
        cache_itemsize = np.dtype(self.sampling_config.cache_dtype).itemsize
        generated_bytes = generated_rows * dimension * np.dtype(np.float64).itemsize
        cache_bytes = n * dimension * cache_itemsize
        estimated_peak_bytes = int((generated_bytes + cache_bytes) * 1.15)
        return {
            "requested_rows": n,
            "generated_rows": generated_rows,
            "dimension": dimension,
            "generated_float64_mb": generated_bytes / (1024**2),
            "cache_mb": cache_bytes / (1024**2),
            "estimated_peak_mb": estimated_peak_bytes / (1024**2),
        }

    def _enforce_generation_memory_guard(self) -> None:
        if self.sampling_config.strategy == "uniform":
            return
        plan = self.generation_plan()
        limit = float(self.sampling_config.max_design_generation_memory_mb)
        if float(plan["estimated_peak_mb"]) > limit:
            raise MemoryError(
                "Estimated one-time design-generation peak memory "
                f"({plan['estimated_peak_mb']:.1f} MB) exceeds "
                f"sampling.max_design_generation_memory_mb={limit:.1f}. "
                "Reduce the number of variable exchanges, split the design into "
                "publication stages, use float32 cache, or raise the guard only "
                "after confirming server RAM."
            )

    @property
    def cache_path(self) -> Path | None:
        return self._cache_path

    def prepare(self) -> Path | None:
        if self.sampling_config.strategy == "uniform":
            return None
        if self.cache_dir is None:
            raise ValueError("cache_dir is required for cached Sobol/LHS sampling.")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._enforce_generation_memory_guard()
        strategy = self.sampling_config.strategy
        path = self.cache_dir / f"{strategy}_{self._cache_key()}.npy"
        metadata_path = path.with_suffix(".json")
        n = self.simulation_config.n_simulations
        dimension = len(self.reaction_ids)
        dtype = np.dtype(self.sampling_config.cache_dtype)

        valid = False
        if path.exists() and metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                candidate = np.load(path, mmap_mode="r")
                valid = (
                    metadata.get("cache_key") == self._cache_key()
                    and list(candidate.shape) == [n, dimension]
                    and candidate.dtype == dtype
                )
            except Exception:
                valid = False

        if not valid:
            temporary = path.with_suffix(".tmp.npy")
            temporary.unlink(missing_ok=True)
            matrix = self._generate_unit_design(n, dimension)
            # Scale in place to avoid holding a second full float64 design matrix.
            matrix *= (self.maximums - self.minimums)
            matrix += self.minimums
            memmap = np.lib.format.open_memmap(temporary, mode="w+", dtype=dtype, shape=matrix.shape)
            memmap[:] = matrix.astype(dtype, copy=False)
            memmap.flush()
            del memmap, matrix
            temporary.replace(path)
            metadata_path.write_text(
                json.dumps(
                    {
                        "cache_key": self._cache_key(),
                        "strategy": strategy,
                        "shape": [n, dimension],
                        "dtype": dtype.name,
                        "seed": self.simulation_config.seed,
                        "reaction_ids": self.reaction_ids,
                        "sobol_scramble": self.sampling_config.sobol_scramble,
                        "sobol_skip": self.sampling_config.sobol_skip,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )

        self._cache_path = path
        self._array = np.load(path, mmap_mode="r")
        return path

    def _generate_unit_design(self, n: int, dimension: int) -> np.ndarray:
        if self.sampling_config.strategy == "sobol":
            skip = self.sampling_config.sobol_skip
            total_needed = n + skip
            exponent = int(math.ceil(math.log2(max(1, total_needed))))
            generated_count = 2**exponent
            if n & (n - 1):
                LOGGER.info(
                    "Sobol sample count %s is not a power of two; generating %s points and using the first %s after skip.",
                    n,
                    generated_count,
                    n,
                )
            engine = qmc.Sobol(
                d=dimension,
                scramble=self.sampling_config.sobol_scramble,
                seed=self.simulation_config.seed,
            )
            unit = engine.random_base2(exponent)
            return unit[skip : skip + n]
        if self.sampling_config.strategy == "latin_hypercube":
            engine = qmc.LatinHypercube(d=dimension, seed=self.simulation_config.seed)
            return engine.random(n)
        raise ValueError(f"Unsupported cached strategy: {self.sampling_config.strategy}")

    def attach_cache(self, path: str | Path | None) -> None:
        if path is None:
            return
        self._cache_path = Path(path)
        self._array = np.load(self._cache_path, mmap_mode="r")

    def sample_chunk(self, start: int, stop: int) -> pd.DataFrame:
        if start < 0 or stop <= start or stop > self.simulation_config.n_simulations:
            raise ValueError(f"Invalid chunk range: start={start}, stop={stop}")
        sample_ids = np.arange(start, stop, dtype=np.int64)
        if self.sampling_config.strategy == "uniform":
            unit = _counter_uniform(sample_ids, len(self.reaction_ids), self.simulation_config.seed)
            values = self.minimums + unit * (self.maximums - self.minimums)
        else:
            if self._array is None:
                self.prepare()
            assert self._array is not None
            values = np.asarray(self._array[start:stop], dtype=np.float64)

        probability = self.sampling_config.include_zero_probability
        if probability > 0:
            zero_draws = _counter_uniform(sample_ids, len(self.reaction_ids), self.simulation_config.seed, stream=999983)
            values = np.where(zero_draws < probability, 0.0, values)

        frame = pd.DataFrame(values, columns=self.reaction_ids)
        frame.insert(0, "sample_id", sample_ids)
        return frame
