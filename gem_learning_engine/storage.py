from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from . import __version__
from .config import PipelineConfig, save_config_snapshot
from .fba import SimulationTables


@dataclass(frozen=True)
class OutputPaths:
    output_dir: Path
    catalogs_dir: Path
    metadata_dir: Path
    cache_dir: Path
    chunks_dir: Path
    manifests_dir: Path
    checkpoints_dir: Path
    logs_dir: Path


@dataclass
class Checkpoint:
    config_hash: str
    model_fingerprint: str = ""
    completed_chunks: list[int] = field(default_factory=list)
    failed_chunks: list[int] = field(default_factory=list)
    started_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


_TABLE_SUBDIRS = {
    "condition_metadata": "conditions",
    "requested_media": "conditions",
    "applied_media": "conditions",
    "gene_perturbations": "perturbations",
    "reaction_perturbations": "perturbations",
    "objective_perturbations": "perturbations",
    "results": "results",
    "simulation_log": "logs",
    "exchange_fluxes": "fluxes",
    "reaction_fluxes": "fluxes",
    "shadow_prices": "diagnostics",
    "reduced_costs": "diagnostics",
    "active_constraints": "diagnostics",
    "errors": "errors",
}


class ResultsStore:
    def __init__(self, config: PipelineConfig):
        self.config = config
        output_dir = Path(config.project.output_dir)
        self.paths = OutputPaths(
            output_dir=output_dir,
            catalogs_dir=output_dir / "catalogs",
            metadata_dir=output_dir / "metadata",
            cache_dir=output_dir / "design_cache",
            chunks_dir=output_dir / "chunks",
            manifests_dir=output_dir / "manifests",
            checkpoints_dir=output_dir / "checkpoints",
            logs_dir=output_dir / "logs",
        )
        self.pause_path = self.paths.checkpoints_dir / config.project.pause_file
        self.checkpoint_path = self.paths.checkpoints_dir / "checkpoint.json"

    def initialize(self, model_fingerprint: str) -> None:
        output = self.paths.output_dir
        if output.exists() and any(output.iterdir()):
            if self.config.project.overwrite:
                shutil.rmtree(output)
            elif not self.config.project.resume:
                raise FileExistsError(
                    f"Output directory is not empty and neither overwrite nor resume is enabled: {output}"
                )

        for path in asdict(self.paths).values():
            Path(path).mkdir(parents=True, exist_ok=True)
        for subdir in sorted(set(_TABLE_SUBDIRS.values())):
            (self.paths.chunks_dir / subdir).mkdir(parents=True, exist_ok=True)

        snapshot = self.paths.metadata_dir / "config_snapshot.yaml"
        if not snapshot.exists() or self.config.project.overwrite:
            save_config_snapshot(self.config, snapshot)

        if not self.checkpoint_path.exists():
            self.save_checkpoint(
                Checkpoint(config_hash=self.config.data_hash(), model_fingerprint=model_fingerprint)
            )

    def write_run_manifest(self, model_fingerprint: str, model_summary: dict[str, Any]) -> None:
        packages: dict[str, str] = {}
        for package_name in ["cobra", "numpy", "pandas", "pyarrow", "scipy", "optlang"]:
            try:
                module = __import__(package_name)
                packages[package_name] = str(getattr(module, "__version__", "unknown"))
            except Exception:
                packages[package_name] = "unavailable"
        manifest = {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "engine_version": __version__,
            "config_hash": self.config.data_hash(),
            "model_fingerprint": model_fingerprint,
            "model_summary": model_summary,
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": packages,
        }
        self.write_json(self.paths.metadata_dir / "run_manifest.json", manifest)

    def table_path(self, table_name: str, chunk_id: int) -> Path:
        if table_name not in _TABLE_SUBDIRS:
            raise KeyError(f"Unknown table: {table_name}")
        return self.paths.chunks_dir / _TABLE_SUBDIRS[table_name] / f"{table_name}_chunk_{chunk_id:06d}.parquet"

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    def _atomic_parquet(self, frame: pd.DataFrame, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        frame.to_parquet(temporary, index=False, compression=self.config.outputs.compression)
        temporary.replace(path)

    def write_chunk(
        self,
        chunk_id: int,
        tables: SimulationTables,
        performance: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        files: dict[str, dict[str, Any]] = {}
        row_counts: dict[str, int] = {}
        for table_name, frame in tables.as_dict().items():
            row_counts[table_name] = int(len(frame))
            # Empty optional outputs are represented in the manifest but are not
            # written as separate Parquet files, reducing inode and metadata load.
            if frame.empty:
                files[table_name] = {"path": None, "rows": 0, "sha256": None, "size_bytes": 0}
                continue
            path = self.table_path(table_name, chunk_id)
            self._atomic_parquet(frame, path)
            files[table_name] = {
                "path": str(path.relative_to(self.paths.output_dir)),
                "rows": int(len(frame)),
                "sha256": self._sha256(path),
                "size_bytes": path.stat().st_size,
            }

        result_summary: dict[str, Any] = {
            "saved_simulations": int(len(tables.results)),
            "feasible_simulations": int(tables.results["feasible"].sum())
            if not tables.results.empty and "feasible" in tables.results
            else 0,
            "status_counts": tables.results["status"].value_counts(dropna=False).to_dict()
            if not tables.results.empty and "status" in tables.results
            else {},
            "runtime_seconds_sum": float(tables.results["runtime_seconds"].sum())
            if not tables.results.empty and "runtime_seconds" in tables.results
            else 0.0,
        }
        manifest = {
            "chunk_id": int(chunk_id),
            "config_hash": self.config.data_hash(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "row_counts": row_counts,
            "summary": result_summary,
            "files": files,
            "performance": performance or {},
        }
        path = self.paths.manifests_dir / f"chunk_{chunk_id:06d}.json"
        self.write_json_atomic(path, manifest)
        return manifest

    def _accepted_config_hashes(self) -> set[str]:
        """Return hashes that validly describe this output directory.

        Automatic role inference resolves an input configuration before execution.
        Callers may later construct a store from either the original automatic
        configuration or the resolved snapshot. Accepting the persisted snapshot
        hash keeps status/resume inspection reliable without weakening chunk
        integrity: every accepted hash is tied to this run directory.
        """
        hashes = {self.config.data_hash()}
        snapshot = self.paths.metadata_dir / "config_snapshot.yaml"
        if snapshot.exists():
            try:
                from .config import load_config

                hashes.add(load_config(snapshot).data_hash())
            except Exception:
                # A malformed snapshot is handled by the normal manifest mismatch.
                pass
        return hashes

    def is_chunk_complete(self, chunk_id: int, verify_checksums: bool | None = None) -> bool:
        manifest_path = self.paths.manifests_dir / f"chunk_{chunk_id:06d}.json"
        if not manifest_path.exists():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("config_hash") not in self._accepted_config_hashes():
                return False
            files = manifest.get("files", {})
            # Results and condition metadata are mandatory for every complete chunk.
            for required in ["condition_metadata", "results"]:
                if required not in files or not files[required].get("path"):
                    return False
            for record in files.values():
                relative = record.get("path")
                if not relative:
                    if int(record.get("rows", 0)) != 0:
                        return False
                    continue
                path = self.paths.output_dir / relative
                if not path.exists() or path.stat().st_size != int(record.get("size_bytes", -1)):
                    return False
                should_verify = (
                    self.config.project.verify_chunk_checksums_on_resume
                    if verify_checksums is None
                    else bool(verify_checksums)
                )
                if should_verify and self._sha256(path) != record.get("sha256"):
                    return False
            return True
        except Exception:
            return False

    def load_checkpoint(self) -> Checkpoint:
        if not self.checkpoint_path.exists():
            return Checkpoint(config_hash=self.config.data_hash())
        data = json.loads(self.checkpoint_path.read_text(encoding="utf-8"))
        return Checkpoint(**data)

    def save_checkpoint(self, checkpoint: Checkpoint) -> None:
        checkpoint.updated_at_utc = datetime.now(timezone.utc).isoformat()
        self.write_json_atomic(self.checkpoint_path, asdict(checkpoint))

    def request_pause(self, reason: str) -> None:
        self.write_json_atomic(
            self.pause_path,
            {"requested_at_utc": datetime.now(timezone.utc).isoformat(), "reason": reason},
        )

    def clear_pause(self) -> None:
        self.pause_path.unlink(missing_ok=True)

    def pause_requested(self) -> bool:
        return self.pause_path.exists()

    def write_catalog(self, name: str, frame: pd.DataFrame) -> None:
        parquet = self.paths.catalogs_dir / f"{name}.parquet"
        csv = self.paths.catalogs_dir / f"{name}.csv"
        self._atomic_parquet(frame, parquet)
        temporary = csv.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(csv)

    def write_design_metadata(self, metadata: dict[str, Any]) -> None:
        self.write_json(self.paths.metadata_dir / "design_metadata.json", metadata)

    def summarize(self) -> pd.DataFrame:
        paths = sorted((self.paths.chunks_dir / "results").glob("results_chunk_*.parquet"))
        if not paths:
            return pd.DataFrame()
        frames = [pd.read_parquet(path) for path in paths]
        summary = pd.concat(frames, ignore_index=True)
        if "sample_id" in summary:
            summary = summary.sort_values("sample_id").drop_duplicates("sample_id", keep="last").reset_index(drop=True)
        return summary

    def status(self, expected_chunks: int) -> dict[str, Any]:
        """Return run progress without concatenating result tables into memory.

        Chunk manifests contain the counters needed by the CLI and GUI. Older
        manifests without summaries are counted by row count and reported via
        ``summary_missing_chunks`` rather than repeatedly reading all Parquet
        result files during status polling.
        """
        manifests = sorted(self.paths.manifests_dir.glob("chunk_*.json"))
        complete: list[int] = []
        invalid: list[int] = []
        summary_missing: list[int] = []
        saved_simulations = 0
        feasible_simulations = 0
        runtime_seconds_sum = 0.0
        status_counts: dict[str, int] = {}

        for path in manifests:
            chunk_id = int(path.stem.split("_")[-1])
            if not self.is_chunk_complete(chunk_id, verify_checksums=False):
                invalid.append(chunk_id)
                continue
            complete.append(chunk_id)
            try:
                manifest = json.loads(path.read_text(encoding="utf-8"))
                summary = manifest.get("summary") or {}
                if summary:
                    saved_simulations += int(summary.get("saved_simulations", 0))
                    feasible_simulations += int(summary.get("feasible_simulations", 0))
                    runtime_seconds_sum += float(summary.get("runtime_seconds_sum", 0.0))
                    for key, value in (summary.get("status_counts") or {}).items():
                        status_counts[str(key)] = status_counts.get(str(key), 0) + int(value)
                else:
                    summary_missing.append(chunk_id)
                    saved_simulations += int((manifest.get("row_counts") or {}).get("results", 0))
            except Exception:
                summary_missing.append(chunk_id)

        checkpoint = self.load_checkpoint()
        accepted_hashes = self._accepted_config_hashes()
        effective_hash = checkpoint.config_hash if checkpoint.config_hash in accepted_hashes else self.config.data_hash()
        return {
            "project": self.config.project.name,
            "output_dir": str(self.paths.output_dir),
            "config_hash": effective_hash,
            "expected_chunks": expected_chunks,
            "complete_chunks": complete,
            "invalid_chunks": invalid,
            "summary_missing_chunks": summary_missing,
            "checkpoint_completed_chunks": checkpoint.completed_chunks,
            "failed_chunks": checkpoint.failed_chunks,
            "saved_simulations": int(saved_simulations),
            "feasible_simulations": int(feasible_simulations),
            "status_counts": status_counts,
            "runtime_seconds_sum": float(runtime_seconds_sum),
            "pause_requested": self.pause_requested(),
        }

    def aggregate_result_statistics(self) -> dict[str, Any]:
        """Stream result chunks and compute bounded-memory run statistics."""
        import pyarrow.parquet as pq

        total = 0
        feasible = 0
        runtime_count = 0
        runtime_sum = 0.0
        runtime_sq_sum = 0.0
        status_counts: dict[str, int] = {}
        numeric = {
            "objective_value": {"count": 0, "sum": 0.0, "min": None, "max": None},
            "biomass_flux": {"count": 0, "sum": 0.0, "min": None, "max": None},
            "product_flux": {"count": 0, "sum": 0.0, "min": None, "max": None},
            "maximum_growth": {"count": 0, "sum": 0.0, "min": None, "max": None},
            "growth_floor": {"count": 0, "sum": 0.0, "min": None, "max": None},
        }

        for path in sorted((self.paths.chunks_dir / "results").glob("results_chunk_*.parquet")):
            available = set(pq.ParquetFile(path).schema.names)
            columns = [
                name
                for name in ["status", "feasible", "runtime_seconds", *numeric.keys()]
                if name in available
            ]
            frame = pd.read_parquet(path, columns=columns)
            total += len(frame)
            if "feasible" in frame:
                feasible += int(frame["feasible"].sum())
            if "status" in frame:
                for key, value in frame["status"].value_counts(dropna=False).to_dict().items():
                    status_counts[str(key)] = status_counts.get(str(key), 0) + int(value)
            if "runtime_seconds" in frame:
                values = pd.to_numeric(frame["runtime_seconds"], errors="coerce").dropna()
                runtime_count += len(values)
                runtime_sum += float(values.sum())
                runtime_sq_sum += float((values * values).sum())
            for name, stats in numeric.items():
                if name not in frame:
                    continue
                values = pd.to_numeric(frame[name], errors="coerce").dropna()
                if values.empty:
                    continue
                stats["count"] += int(len(values))
                stats["sum"] += float(values.sum())
                current_min = float(values.min())
                current_max = float(values.max())
                stats["min"] = current_min if stats["min"] is None else min(float(stats["min"]), current_min)
                stats["max"] = current_max if stats["max"] is None else max(float(stats["max"]), current_max)

        mean_runtime = runtime_sum / runtime_count if runtime_count else 0.0
        runtime_variance = (
            max(0.0, runtime_sq_sum / runtime_count - mean_runtime * mean_runtime)
            if runtime_count
            else 0.0
        )

        described: dict[str, dict[str, float | int | None]] = {}
        for name, stats in numeric.items():
            count = int(stats["count"])
            described[name] = {
                "count": count,
                "min": stats["min"],
                "max": stats["max"],
                "mean": float(stats["sum"]) / count if count else None,
            }

        return {
            "saved_simulations": int(total),
            "feasible_simulations": int(feasible),
            "status_counts": status_counts,
            "runtime_seconds_sum": float(runtime_sum),
            "mean_runtime_seconds": float(mean_runtime),
            "runtime_std_seconds": float(runtime_variance ** 0.5),
            **described,
        }

    @staticmethod
    def write_json(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    @staticmethod
    def write_json_atomic(path: Path, data: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
        temporary.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        temporary.replace(path)
