from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Optional

import typer

from . import __version__
from .config import PipelineConfig, load_config, save_config_snapshot
from .inference import resolve_inferred_config
from .export_csv import export_whole_dataset_csv
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
from .orchestrator import SimulationOrchestrator, chunk_ranges
from .preflight import build_preflight_report
from .quickstart import create_quick_config
from .storage import ResultsStore
from .validation import run_all_validations

app = typer.Typer(help="GEM Learning Engine exact simulation pipeline")


def _config_from_inputs(
    config: Optional[Path],
    model: Optional[Path],
    runs: Optional[int],
    output_dir: Optional[Path],
    processes: Optional[int],
    growth_fraction: float,
    overwrite: bool,
) -> PipelineConfig:
    if config is not None:
        if model is not None or runs is not None:
            raise typer.BadParameter("Use either --config or the simple --model/--runs workflow, not both.")
        return load_config(config)
    if model is None or runs is None:
        raise typer.BadParameter("Provide --model and --runs, or provide --config.")
    return create_quick_config(
        model,
        runs,
        output_dir=output_dir,
        processes=processes,
        minimum_growth_fraction=growth_fraction,
        overwrite=overwrite,
        resume=not overwrite,
    )


@app.command()
def run(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Advanced YAML configuration."),
    model: Optional[Path] = typer.Option(None, "--model", "-m", help="COBRA JSON, SBML/XML, or MATLAB model."),
    runs: Optional[int] = typer.Option(None, "--runs", "-n", min=1, help="Number of medium simulations."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", "-o", help="Output directory for simple mode."),
    processes: Optional[int] = typer.Option(None, "--processes", "-p", min=1, help="Worker processes; auto-selected when omitted."),
    growth_fraction: float = typer.Option(0.5, "--growth-fraction", min=0.0, max=1.0),
    overwrite: bool = typer.Option(False, "--overwrite", help="Replace an existing output directory."),
    skip_preflight: bool = typer.Option(False, "--skip-preflight", help="Advanced: skip automatic preflight."),
) -> None:
    """Run using only a model and run count, or use an advanced YAML config."""
    cfg = _config_from_inputs(config, model, runs, output_dir, processes, growth_fraction, overwrite)
    if not skip_preflight:
        output_root = Path(cfg.project.output_dir)
        if cfg.project.overwrite and output_root.exists():
            shutil.rmtree(output_root)
            cfg = replace(cfg, project=replace(cfg.project, overwrite=False, resume=True))
        resolved, report = build_preflight_report(cfg, output_root / "_preflight_design")
        preflight_path = output_root / "metadata" / "preflight.json"
        preflight_path.parent.mkdir(parents=True, exist_ok=True)
        preflight_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        typer.echo(json.dumps({
            "ready": report["ready"],
            "model_id": report["model_id"],
            "biomass_reaction_id": report["biomass_reaction_id"],
            "product_reaction_id": report["product_reaction_id"],
            "optimization_mode": report["optimization_mode"],
            "n_simulations": report["n_simulations"],
            "warnings": report["warnings"],
        }, indent=2))
        if not report["ready"]:
            raise typer.Exit(code=2)
        cfg = resolved
    SimulationOrchestrator(cfg).run()


@app.command()
def validate(output_dir: Path = typer.Option(Path("outputs/validation"), "--output-dir", "-o")) -> None:
    """Run model, inference, production, storage, resume, parallel, and GUI checks."""
    typer.echo(json.dumps(run_all_validations(output_dir), indent=2))


@app.command("inspect-model")
def inspect_model(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    model_path: Optional[Path] = typer.Option(None, "--model", "-m"),
    output_dir: Path = typer.Option(Path("outputs/model_inspection"), "--output-dir", "-o"),
) -> None:
    """Export model catalogs and automatic biomass/product inference."""
    if config is None and model_path is None:
        raise typer.BadParameter("Provide --model or --config.")
    cfg = load_config(config) if config else create_quick_config(model_path, 1, output_dir=output_dir / "unused", processes=1)
    model = load_model(cfg.model)
    configure_solver(model, cfg.solver)
    resolved_model, resolved_optimization, inference = resolve_inferred_config(model, cfg.model, cfg.optimization)
    if resolved_model.objective_reaction_id:
        model.objective = model.reactions.get_by_id(resolved_model.objective_reaction_id)
    validate_model_inputs(model, resolved_model)
    medium_space = build_medium_space(model, cfg.sampling, cfg.simulation)
    output_dir.mkdir(parents=True, exist_ok=True)
    catalogs = {
        "reaction_catalog": reaction_catalog(model),
        "exchange_catalog": exchange_catalog(model),
        "gene_catalog": gene_catalog(model),
        "metabolite_catalog": metabolite_catalog(model),
        "medium_space": medium_space.to_frame(),
    }
    for name, frame in catalogs.items():
        frame.to_parquet(output_dir / f"{name}.parquet", index=False, compression=cfg.outputs.compression)
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    inference.write_json(output_dir / "model_inference.json")
    typer.echo(json.dumps({
        "model_fingerprint": model_fingerprint(model, cfg.model.path),
        "model_id": getattr(model, "id", None),
        "biomass_reaction_id": resolved_model.biomass_reaction_id,
        "product_reaction_id": resolved_model.product_reaction_id,
        "optimization_mode": resolved_optimization.mode,
        "biomass_confidence": inference.biomass.confidence,
        "product_confidence": inference.product.confidence,
        "reactions": len(model.reactions),
        "exchanges": len(medium_space.exchange_ids),
        "output_dir": str(output_dir),
    }, indent=2))


@app.command()
def preflight(
    config: Optional[Path] = typer.Option(None, "--config", "-c"),
    model: Optional[Path] = typer.Option(None, "--model", "-m"),
    runs: Optional[int] = typer.Option(None, "--runs", "-n", min=1),
    output: Path = typer.Option(Path("outputs/preflight.json"), "--output", "-o"),
    processes: Optional[int] = typer.Option(None, "--processes", "-p", min=1),
    growth_fraction: float = typer.Option(0.5, "--growth-fraction", min=0.0, max=1.0),
) -> None:
    """Resolve IDs, test baseline growth/product, and estimate run resources."""
    cfg = _config_from_inputs(config, model, runs, output.parent, processes, growth_fraction, False)
    resolved, report = build_preflight_report(cfg, output.parent / "_preflight_design")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    save_config_snapshot(resolved, output.with_name("resolved_config.yaml"))
    typer.echo(json.dumps(report, indent=2))
    if not report["ready"]:
        raise typer.Exit(code=2)


@app.command()
def status(config: Path = typer.Option(..., "--config", "-c")) -> None:
    cfg = load_config(config)
    snapshot = Path(cfg.project.output_dir) / "metadata" / "config_snapshot.yaml"
    if snapshot.exists():
        cfg = load_config(snapshot)
    store = ResultsStore(cfg)
    expected = len(chunk_ranges(cfg.simulation.n_simulations, cfg.simulation.chunk_size))
    typer.echo(json.dumps(store.status(expected), indent=2))


@app.command("pause")
def pause_run(
    config: Path = typer.Option(..., "--config", "-c"),
    reason: str = typer.Option("User requested pause", "--reason"),
) -> None:
    store = ResultsStore(load_config(config))
    store.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
    store.request_pause(reason)
    typer.echo(f"Pause requested at the next chunk boundary: {store.pause_path}")


@app.command("clear-pause")
def clear_pause(config: Path = typer.Option(..., "--config", "-c")) -> None:
    store = ResultsStore(load_config(config))
    store.clear_pause()
    typer.echo("Pause request cleared. Rerun the same configuration to resume.")


@app.command("export-csv")
def export_csv(
    config: Path = typer.Option(..., "--config", "-c", help="Configuration for the completed run."),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Destination CSV; defaults to <run>/exports/whole_dataset.csv."
    ),
    allow_partial: bool = typer.Option(
        False, "--allow-partial", help="Export completed chunks even when the run is not finished."
    ),
) -> None:
    """Generate one complete, wide CSV row for every exact FBA analysis."""
    cfg = load_config(config)
    snapshot = Path(cfg.project.output_dir) / "metadata" / "config_snapshot.yaml"
    if snapshot.exists():
        cfg = load_config(snapshot)
    report = export_whole_dataset_csv(cfg, destination=output, allow_partial=allow_partial)
    typer.echo(json.dumps(report, indent=2))


@app.command("version")
def version() -> None:
    typer.echo(__version__)


if __name__ == "__main__":
    app()
