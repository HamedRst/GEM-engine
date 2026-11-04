from __future__ import annotations

import hashlib
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .config import PipelineConfig
from .orchestrator import chunk_ranges
from .storage import ResultsStore


class CSVExportError(RuntimeError):
    """Raised when a complete run cannot be exported safely."""


def default_csv_path(config: PipelineConfig) -> Path:
    """Return the standard complete-dataset CSV path for a run."""
    return Path(config.project.output_dir) / "exports" / "whole_dataset.csv"


def _schema_names(path: Path) -> list[str]:
    import pyarrow.parquet as pq

    return list(pq.ParquetFile(path).schema.names)


def _table_files(store: ResultsStore, table_name: str) -> list[Path]:
    subdir = store.table_path(table_name, 0).parent
    return sorted(subdir.glob(f"{table_name}_chunk_*.parquet"))


def _chunk_id(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[-1])


def _union_schema(files: Iterable[Path]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for path in files:
        for name in _schema_names(path):
            if name not in seen:
                ordered.append(name)
                seen.add(name)
    return ordered


def _catalog_ids(store: ResultsStore, catalog_name: str, id_column: str) -> list[str]:
    path = store.paths.catalogs_dir / f"{catalog_name}.parquet"
    if not path.exists():
        return []
    frame = pd.read_parquet(path, columns=[id_column])
    return [str(value) for value in frame[id_column].dropna().tolist()]


def _observed_ids(files: Iterable[Path], id_column: str) -> list[str]:
    values: set[str] = set()
    for path in files:
        names = set(_schema_names(path))
        if id_column not in names:
            continue
        frame = pd.read_parquet(path, columns=[id_column])
        values.update(str(value) for value in frame[id_column].dropna().unique())
    return sorted(values)


def _series_ids(
    store: ResultsStore,
    files: list[Path],
    *,
    mode: str,
    selected_ids: list[str],
    catalog_name: str,
    id_column: str,
) -> list[str]:
    if mode == "none" or not files:
        return []
    if mode == "selected":
        return list(selected_ids)
    if mode == "all":
        catalog = _catalog_ids(store, catalog_name, id_column)
        if catalog:
            return catalog
        # In all-mode every feasible row contains the same full entity set.
        # Read only the first non-empty file instead of scanning a large run.
        for path in files:
            names = _schema_names(path)
            if id_column in names:
                frame = pd.read_parquet(path, columns=[id_column])
                observed = sorted(str(value) for value in frame[id_column].dropna().unique())
                if observed:
                    return observed
            else:
                observed = [str(name) for name in names if name != "sample_id"]
                if observed:
                    return observed
        return []
    return _observed_ids(files, id_column)


def _read_chunk(store: ResultsStore, table_name: str, chunk_id: int) -> pd.DataFrame:
    path = store.table_path(table_name, chunk_id)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _prefix_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    if frame.empty or "sample_id" not in frame.columns:
        return pd.DataFrame()
    renamed = {
        column: f"{prefix}{column}"
        for column in frame.columns
        if column != "sample_id"
    }
    return frame.rename(columns=renamed)


def _merge_one_to_one(base: pd.DataFrame, addition: pd.DataFrame, label: str) -> pd.DataFrame:
    if addition.empty:
        return base
    if "sample_id" not in addition.columns:
        raise CSVExportError(f"{label} is missing sample_id.")
    if addition["sample_id"].duplicated().any():
        duplicates = addition.loc[addition["sample_id"].duplicated(), "sample_id"].head(5).tolist()
        raise CSVExportError(f"{label} has duplicate sample IDs: {duplicates}")
    return base.merge(addition, on="sample_id", how="left", validate="one_to_one")


def _pivot_series(
    frame: pd.DataFrame,
    *,
    entity_column: str,
    value_column: str,
    prefix: str,
    entity_ids: list[str],
) -> pd.DataFrame:
    if frame.empty or "sample_id" not in frame.columns:
        return pd.DataFrame()

    if entity_column in frame.columns and value_column in frame.columns:
        duplicate = frame.duplicated(["sample_id", entity_column])
        if duplicate.any():
            raise CSVExportError(
                f"Series table contains duplicate sample/entity pairs for {entity_column}."
            )
        wide = frame.pivot(index="sample_id", columns=entity_column, values=value_column)
        wide.columns = wide.columns.astype(str)
        if entity_ids:
            wide = wide.reindex(columns=entity_ids)
        wide = wide.reset_index()
    else:
        wide = frame.copy()
        available = [column for column in wide.columns if column != "sample_id"]
        if entity_ids:
            for entity_id in entity_ids:
                if entity_id not in wide.columns:
                    wide[entity_id] = pd.NA
            wide = wide[["sample_id", *entity_ids]]
        elif not available:
            return pd.DataFrame()

    return wide.rename(
        columns={column: f"{prefix}{column}" for column in wide.columns if column != "sample_id"}
    )


def _records_json(frame: pd.DataFrame, column_name: str) -> pd.DataFrame:
    if frame.empty or "sample_id" not in frame.columns:
        return pd.DataFrame()
    payload_columns = [column for column in frame.columns if column != "sample_id"]
    rows: list[dict[str, Any]] = []
    for sample_id, group in frame.groupby("sample_id", sort=False):
        records = group[payload_columns].where(pd.notna(group[payload_columns]), None).to_dict(orient="records")
        rows.append(
            {
                "sample_id": int(sample_id),
                column_name: json.dumps(records, separators=(",", ":"), default=str),
            }
        )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _build_column_plan(store: ResultsStore, config: PipelineConfig) -> dict[str, Any]:
    files = {name: _table_files(store, name) for name in [
        "results",
        "condition_metadata",
        "requested_media",
        "applied_media",
        "simulation_log",
        "errors",
        "gene_perturbations",
        "reaction_perturbations",
        "objective_perturbations",
        "reaction_fluxes",
        "exchange_fluxes",
        "shadow_prices",
        "reduced_costs",
        "active_constraints",
    ]}

    reaction_ids = _series_ids(
        store,
        files["reaction_fluxes"],
        mode=config.outputs.store_reaction_fluxes,
        selected_ids=config.outputs.selected_reaction_fluxes,
        catalog_name="reaction_catalog",
        id_column="reaction_id",
    )
    exchange_ids = _series_ids(
        store,
        files["exchange_fluxes"],
        mode=config.outputs.store_exchange_fluxes,
        selected_ids=config.outputs.selected_exchange_fluxes,
        catalog_name="exchange_catalog",
        id_column="reaction_id",
    )
    shadow_ids = _series_ids(
        store,
        files["shadow_prices"],
        mode=config.diagnostics.store_shadow_prices,
        selected_ids=config.diagnostics.selected_shadow_price_metabolites,
        catalog_name="metabolite_catalog",
        id_column="metabolite_id",
    )
    reduced_ids = _series_ids(
        store,
        files["reduced_costs"],
        mode=config.diagnostics.store_reduced_costs,
        selected_ids=config.diagnostics.selected_reduced_cost_reactions,
        catalog_name="reaction_catalog",
        id_column="reaction_id",
    )

    result_columns = [c for c in _union_schema(files["results"]) if c != "sample_id"]
    condition_columns = [
        c for c in _union_schema(files["condition_metadata"])
        if c not in {"sample_id", "chunk_id"}
    ]
    requested_columns = [c for c in _union_schema(files["requested_media"]) if c != "sample_id"]
    applied_columns = [c for c in _union_schema(files["applied_media"]) if c != "sample_id"]
    log_columns = [c for c in _union_schema(files["simulation_log"]) if c != "sample_id"]
    error_columns = [c for c in _union_schema(files["errors"]) if c != "sample_id"]

    columns = ["sample_id", *result_columns]
    columns += [f"condition__{c}" for c in condition_columns]
    columns += [f"requested_medium__{c}" for c in requested_columns]
    columns += [f"applied_medium__{c}" for c in applied_columns]
    columns += [f"reaction_flux__{c}" for c in reaction_ids]
    columns += [f"exchange_flux__{c}" for c in exchange_ids]
    columns += [f"shadow_price__{c}" for c in shadow_ids]
    columns += [f"reduced_cost__{c}" for c in reduced_ids]
    columns += [f"log__{c}" for c in log_columns]
    columns += [f"error__{c}" for c in error_columns]

    optional_json = {
        "gene_perturbations": "gene_perturbations_json",
        "reaction_perturbations": "reaction_perturbations_json",
        "objective_perturbations": "objective_perturbations_json",
        "active_constraints": "active_constraints_json",
    }
    for table_name, column_name in optional_json.items():
        if files[table_name]:
            columns.append(column_name)

    counts = Counter(columns)
    duplicates = sorted(column for column, count in counts.items() if count > 1)
    if duplicates:
        raise CSVExportError(f"CSV column names are not unique: {duplicates[:10]}")

    return {
        "files": files,
        "reaction_ids": reaction_ids,
        "exchange_ids": exchange_ids,
        "shadow_ids": shadow_ids,
        "reduced_ids": reduced_ids,
        "columns": columns,
        "optional_json": optional_json,
    }


def _build_chunk_frame(
    store: ResultsStore,
    chunk_id: int,
    plan: dict[str, Any],
) -> pd.DataFrame:
    results = _read_chunk(store, "results", chunk_id)
    if results.empty:
        raise CSVExportError(f"Chunk {chunk_id} has no result table.")
    if "sample_id" not in results.columns:
        raise CSVExportError(f"Chunk {chunk_id} result table is missing sample_id.")
    if results["sample_id"].duplicated().any():
        raise CSVExportError(f"Chunk {chunk_id} result table has duplicate sample IDs.")

    frame = results.copy()

    condition = _read_chunk(store, "condition_metadata", chunk_id)
    if not condition.empty:
        condition = condition.drop(columns=["chunk_id"], errors="ignore")
        frame = _merge_one_to_one(frame, _prefix_columns(condition, "condition__"), "condition metadata")

    for table_name, prefix in [
        ("requested_media", "requested_medium__"),
        ("applied_media", "applied_medium__"),
    ]:
        table = _read_chunk(store, table_name, chunk_id)
        frame = _merge_one_to_one(frame, _prefix_columns(table, prefix), table_name)

    reaction = _pivot_series(
        _read_chunk(store, "reaction_fluxes", chunk_id),
        entity_column="reaction_id",
        value_column="flux",
        prefix="reaction_flux__",
        entity_ids=plan["reaction_ids"],
    )
    frame = _merge_one_to_one(frame, reaction, "reaction fluxes")

    exchange = _pivot_series(
        _read_chunk(store, "exchange_fluxes", chunk_id),
        entity_column="reaction_id",
        value_column="flux",
        prefix="exchange_flux__",
        entity_ids=plan["exchange_ids"],
    )
    frame = _merge_one_to_one(frame, exchange, "exchange fluxes")

    shadow = _pivot_series(
        _read_chunk(store, "shadow_prices", chunk_id),
        entity_column="metabolite_id",
        value_column="shadow_price",
        prefix="shadow_price__",
        entity_ids=plan["shadow_ids"],
    )
    frame = _merge_one_to_one(frame, shadow, "shadow prices")

    reduced = _pivot_series(
        _read_chunk(store, "reduced_costs", chunk_id),
        entity_column="reaction_id",
        value_column="reduced_cost",
        prefix="reduced_cost__",
        entity_ids=plan["reduced_ids"],
    )
    frame = _merge_one_to_one(frame, reduced, "reduced costs")

    for table_name, prefix in [
        ("simulation_log", "log__"),
        ("errors", "error__"),
    ]:
        table = _read_chunk(store, table_name, chunk_id)
        frame = _merge_one_to_one(frame, _prefix_columns(table, prefix), table_name)

    for table_name, column_name in plan["optional_json"].items():
        table = _read_chunk(store, table_name, chunk_id)
        frame = _merge_one_to_one(frame, _records_json(table, column_name), table_name)

    frame = frame.reindex(columns=plan["columns"])
    return frame.sort_values("sample_id").reset_index(drop=True)


def export_whole_dataset_csv(
    config: PipelineConfig,
    destination: Path | None = None,
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Create one wide CSV row per exact FBA simulation using bounded memory.

    The exporter processes one completed Parquet chunk at a time. Summary values,
    exact conditions, requested/applied media, complete final-stage reaction and
    exchange fluxes, optional dual diagnostics, logs, and perturbations are joined
    by ``sample_id``. The output is atomically replaced only after all chunks have
    been written successfully.
    """

    store = ResultsStore(config)
    expected = len(chunk_ranges(config.simulation.n_simulations, config.simulation.chunk_size))
    status = store.status(expected)
    complete = sorted(set(int(value) for value in status["complete_chunks"]))
    missing = sorted(set(range(expected)) - set(complete))

    if status["invalid_chunks"]:
        raise CSVExportError(f"Invalid chunks prevent export: {status['invalid_chunks']}")
    if not allow_partial and missing:
        raise CSVExportError(
            f"Run is incomplete: {len(complete)}/{expected} chunks are complete. Missing: {missing[:20]}"
        )
    if not allow_partial and int(status["saved_simulations"]) != int(config.simulation.n_simulations):
        raise CSVExportError(
            "Run is incomplete: "
            f"saved {status['saved_simulations']} of {config.simulation.n_simulations} simulations."
        )
    if not complete:
        raise CSVExportError("No completed simulation chunks were found.")

    output = Path(destination) if destination is not None else default_csv_path(config)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + f".{os.getpid()}.tmp")
    temporary.unlink(missing_ok=True)

    plan = _build_column_plan(store, config)
    rows_written = 0
    header = True
    sample_ids_seen: set[int] = set()

    try:
        for chunk_id in complete:
            frame = _build_chunk_frame(store, chunk_id, plan)
            ids = set(int(value) for value in frame["sample_id"].tolist())
            overlap = sample_ids_seen.intersection(ids)
            if overlap:
                raise CSVExportError(f"Duplicate sample IDs across chunks: {sorted(overlap)[:10]}")
            sample_ids_seen.update(ids)
            frame.to_csv(
                temporary,
                mode="w" if header else "a",
                header=header,
                index=False,
                lineterminator="\n",
            )
            header = False
            rows_written += len(frame)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "csv_path": str(output),
        "source_output_dir": str(store.paths.output_dir),
        "project": config.project.name,
        "config_hash": config.data_hash(),
        "complete_run": not missing and rows_written == config.simulation.n_simulations,
        "rows": int(rows_written),
        "columns": int(len(plan["columns"])),
        "size_bytes": int(output.stat().st_size),
        "sha256": _sha256(output),
        "included_chunks": complete,
        "missing_chunks": missing,
        "column_groups": {
            "reaction_fluxes": len(plan["reaction_ids"]),
            "exchange_fluxes": len(plan["exchange_ids"]),
            "shadow_prices": len(plan["shadow_ids"]),
            "reduced_costs": len(plan["reduced_ids"]),
        },
    }
    manifest_path = output.with_name("whole_dataset_manifest.json")
    store.write_json_atomic(manifest_path, manifest)
    manifest["manifest_path"] = str(manifest_path)
    return manifest
