from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import psutil
import yaml
from fastapi import Body, Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

from . import __version__
from .config import PipelineConfig, load_config, save_config_snapshot
from .export_csv import default_csv_path
from .inference import infer_model_roles
from .orchestrator import chunk_ranges
from .quickstart import create_quick_config, safe_slug
from .storage import ResultsStore
from .medium import build_medium_space
from .model_io import configure_solver, get_exchange_reactions, load_model


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class JobRecord:
    job_id: str
    kind: str
    config_path: str | None
    command: list[str]
    pid: int
    status: str
    started_at_utc: str
    updated_at_utc: str
    log_path: str
    exit_code: int | None = None
    message: str | None = None


class Workspace:
    """Restrict GUI-created and GUI-edited paths to one server workspace."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def resolve(self, value: str | Path, *, must_exist: bool = False) -> Path:
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = self.root / path
        path = path.resolve()
        try:
            path.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"Path must be inside GUI workspace {self.root}: {path}") from error
        if must_exist and not path.exists():
            raise FileNotFoundError(path)
        return path

    def display(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            return str(path.resolve())


class JobManager:
    """Launch long tasks outside the web server and persist lightweight state."""

    def __init__(self, workspace: Workspace):
        self.workspace = workspace
        self.state_dir = workspace.root / ".gem_gui"
        self.logs_dir = self.state_dir / "logs"
        self.state_path = self.state_dir / "jobs.json"
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._jobs: dict[str, JobRecord] = {}
        self._processes: dict[str, subprocess.Popen[Any]] = {}
        self._load()
        self.refresh()

    def _load(self) -> None:
        if not self.state_path.exists():
            return
        try:
            payload = json.loads(self.state_path.read_text(encoding="utf-8"))
            self._jobs = {item["job_id"]: JobRecord(**item) for item in payload}
        except Exception:
            self._jobs = {}

    def _save(self) -> None:
        temporary = self.state_path.with_suffix(".json.tmp")
        temporary.write_text(
            json.dumps([asdict(item) for item in self._jobs.values()], indent=2),
            encoding="utf-8",
        )
        temporary.replace(self.state_path)

    @staticmethod
    def _pid_active(record: JobRecord) -> bool:
        try:
            process = psutil.Process(record.pid)
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return False
            command = " ".join(process.cmdline())
            expected = " ".join(record.command[-4:])
            return not expected or all(token in command for token in record.command[-2:] if token)
        except (psutil.Error, OSError):
            return False

    def refresh(self) -> None:
        with self._lock:
            changed = False
            for record in self._jobs.values():
                if record.status not in {"running", "terminating"}:
                    continue
                process = self._processes.get(record.job_id)
                if process is not None:
                    code = process.poll()
                    if code is None:
                        continue
                    record.exit_code = int(code)
                    record.status = "completed" if code == 0 else "failed"
                    record.updated_at_utc = _utc_now()
                    self._processes.pop(record.job_id, None)
                    changed = True
                elif not self._pid_active(record):
                    record.status = "finished"
                    record.updated_at_utc = _utc_now()
                    record.message = record.message or "Process ended while the GUI was not monitoring it."
                    changed = True
            if changed:
                self._save()

    def active_for_config(self, config_path: Path, kind: str = "run") -> JobRecord | None:
        self.refresh()
        resolved = str(config_path.resolve())
        candidates = [
            item
            for item in self._jobs.values()
            if item.kind == kind and item.config_path == resolved and item.status in {"running", "terminating"}
        ]
        return max(candidates, key=lambda item: item.started_at_utc) if candidates else None

    def start(
        self,
        *,
        kind: str,
        command: list[str],
        config_path: Path | None = None,
        unique_for_config: bool = False,
    ) -> JobRecord:
        with self._lock:
            if unique_for_config and config_path is not None:
                active = self.active_for_config(config_path, kind=kind)
                if active is not None:
                    raise RuntimeError(f"A {kind} job is already active for this configuration (PID {active.pid}).")

            job_id = uuid.uuid4().hex[:12]
            log_path = self.logs_dir / f"{kind}_{job_id}.log"
            with log_path.open("a", encoding="utf-8") as log:
                log.write(f"[{_utc_now()}] command: {' '.join(command)}\n")
                log.flush()
                kwargs: dict[str, Any] = {
                    "cwd": str(self.workspace.root),
                    "stdout": log,
                    "stderr": subprocess.STDOUT,
                    "env": {**os.environ, "PYTHONUNBUFFERED": "1"},
                }
                if os.name == "nt":
                    kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
                else:
                    kwargs["start_new_session"] = True
                process = subprocess.Popen(command, **kwargs)

            record = JobRecord(
                job_id=job_id,
                kind=kind,
                config_path=str(config_path.resolve()) if config_path else None,
                command=command,
                pid=int(process.pid),
                status="running",
                started_at_utc=_utc_now(),
                updated_at_utc=_utc_now(),
                log_path=str(log_path),
            )
            self._jobs[job_id] = record
            self._processes[job_id] = process
            self._save()
            threading.Thread(target=self._monitor, args=(job_id, process), daemon=True).start()
            return record

    def _monitor(self, job_id: str, process: subprocess.Popen[Any]) -> None:
        code = process.wait()
        with self._lock:
            record = self._jobs.get(job_id)
            if record is None:
                return
            record.exit_code = int(code)
            record.status = "completed" if code == 0 else "failed"
            record.updated_at_utc = _utc_now()
            self._processes.pop(job_id, None)
            self._save()

    def terminate(self, job_id: str) -> JobRecord:
        with self._lock:
            self.refresh()
            record = self._jobs.get(job_id)
            if record is None:
                raise KeyError(job_id)
            if record.status not in {"running", "terminating"}:
                return record
            record.status = "terminating"
            record.updated_at_utc = _utc_now()
            self._save()
            try:
                process = psutil.Process(record.pid)
                children = process.children(recursive=True)
                for child in children:
                    child.terminate()
                process.terminate()
                _, alive = psutil.wait_procs([*children, process], timeout=8)
                for item in alive:
                    item.kill()
            except psutil.NoSuchProcess:
                pass
            except psutil.Error as error:
                record.message = str(error)
            self.refresh()
            return record

    def list(self, config_path: Path | None = None, limit: int = 30) -> list[JobRecord]:
        self.refresh()
        items = list(self._jobs.values())
        if config_path is not None:
            resolved = str(config_path.resolve())
            items = [item for item in items if item.config_path == resolved]
        return sorted(items, key=lambda item: item.started_at_utc, reverse=True)[:limit]

    def get(self, job_id: str) -> JobRecord:
        self.refresh()
        if job_id not in self._jobs:
            raise KeyError(job_id)
        return self._jobs[job_id]


def _tail(path: Path, max_bytes: int = 180_000) -> str:
    if not path.exists() or not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        data = handle.read()
    return data.decode("utf-8", errors="replace")


def _config_summary(config: PipelineConfig) -> dict[str, Any]:
    return {
        "project": config.project.name,
        "model_path": config.model.path,
        "output_dir": config.project.output_dir,
        "n_simulations": config.simulation.n_simulations,
        "chunk_size": config.simulation.chunk_size,
        "processes": config.simulation.processes,
        "method": config.simulation.method,
        "sampling_strategy": config.sampling.strategy,
        "discovery_scope": config.sampling.discovery_scope,
        "biomass_reaction_id": config.model.biomass_reaction_id,
        "product_reaction_id": config.model.product_reaction_id,
        "optimization_mode": config.optimization.mode,
        "minimum_growth_fraction": config.optimization.minimum_growth_fraction,
        "solver": config.solver.name,
        "config_hash": config.data_hash(),
    }


def _safe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", date_format="iso"))


class ConfigPathRequest(BaseModel):
    path: str


class SaveConfigRequest(BaseModel):
    path: str
    content: str


class UpdateConfigRequest(BaseModel):
    path: str
    updates: dict[str, Any] = Field(default_factory=dict)


class InspectRequest(BaseModel):
    path: str
    output_dir: str | None = None


class TerminateRequest(BaseModel):
    job_id: str


class PauseRequest(BaseModel):
    path: str
    reason: str = "Requested from web GUI"


class QuickConfigRequest(BaseModel):
    model_path: str
    runs: int = Field(ge=1, le=100_000_000)
    output_dir: str | None = None
    processes: int | None = Field(default=None, ge=1)
    minimum_growth_fraction: float = Field(default=0.5, ge=0.0, le=1.0)



class LandscapeSaveRequest(BaseModel):
    path: str
    biomass_reaction_id: str | None = None
    product_reaction_id: str | None = None
    optimization_mode: str = "growth_constrained_product"
    minimum_growth_fraction: float = Field(default=0.5, ge=0.0, le=1.0)
    variable_bounds: dict[str, dict[str, float]] = Field(default_factory=dict)
    fixed_reactions: dict[str, float] = Field(default_factory=dict)
    exclude_reaction_ids: list[str] = Field(default_factory=list)


def create_app(workspace_root: str | Path = ".", token: str | None = None) -> FastAPI:
    workspace = Workspace(workspace_root)
    jobs = JobManager(workspace)
    app = FastAPI(title="GEM Learning Engine GUI", version=__version__)
    app.state.workspace = workspace
    app.state.jobs = jobs
    app.state.token = token

    def authorize(x_gle_token: str | None = Header(default=None)) -> None:
        if token and x_gle_token != token:
            raise HTTPException(status_code=401, detail="A valid GUI token is required.")

    auth = Depends(authorize)

    def config_path(value: str, must_exist: bool = True) -> Path:
        try:
            return workspace.resolve(value, must_exist=must_exist)
        except (ValueError, FileNotFoundError) as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    def runtime_config(path: Path) -> PipelineConfig:
        """Load a config using the GUI workspace as the runtime working directory."""
        config = load_config(path)
        output = Path(config.project.output_dir).expanduser()
        if not output.is_absolute():
            output = workspace.root / output
        output = output.resolve()
        snapshot = output / "metadata" / "config_snapshot.yaml"
        if snapshot.exists():
            config = load_config(snapshot)
        return replace(config, project=replace(config.project, output_dir=str(output)))

    @app.get("/", response_class=HTMLResponse)
    def index() -> HTMLResponse:
        html_path = Path(__file__).with_name("gui_assets") / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))


    @app.get("/api/health", dependencies=[auth])
    def health() -> dict[str, Any]:
        return {"ok": True, "engine_version": __version__, "workspace": str(workspace.root)}

    @app.post("/api/model/upload", dependencies=[auth])
    async def upload_model(request: Request, filename: str = Query(...)) -> dict[str, Any]:
        clean_name = Path(filename).name
        if not clean_name.lower().endswith((".json", ".xml", ".sbml", ".mat")):
            raise HTTPException(status_code=400, detail="Supported model types: JSON, XML/SBML, and MAT.")
        data = await request.body()
        if not data:
            raise HTTPException(status_code=400, detail="The uploaded model is empty.")
        if len(data) > 500 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Model upload exceeds the 500 MB GUI limit.")
        uploads = workspace.resolve("data/models/uploads", must_exist=False)
        uploads.mkdir(parents=True, exist_ok=True)
        destination = uploads / clean_name
        if destination.exists():
            destination = uploads / f"{destination.stem}_{uuid.uuid4().hex[:8]}{destination.suffix}"
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_bytes(data)
        temporary.replace(destination)
        return {"model_path": workspace.display(destination), "size_bytes": len(data)}

    @app.post("/api/quick/configure", dependencies=[auth])
    def quick_configure(request: QuickConfigRequest) -> dict[str, Any]:
        model_path = config_path(request.model_path)

        # Every Quick Run is isolated in its own immutable run directory.
        # Re-running the same model and run count therefore never mixes data.
        # Resume continues to use the generated configuration for that exact run.
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            + "_"
            + uuid.uuid4().hex[:6]
        )
        output_group = request.output_dir or f"outputs/{safe_slug(model_path.stem)}_{request.runs}"
        output_dir = workspace.resolve(output_group, must_exist=False) / f"run_{run_id}"
        generated_name = f"{safe_slug(model_path.stem)}_{request.runs}_{run_id}.yaml"

        try:
            config = create_quick_config(
                model_path,
                request.runs,
                output_dir=output_dir,
                processes=request.processes,
                minimum_growth_fraction=request.minimum_growth_fraction,
            )
            config = replace(
                config,
                project=replace(
                    config.project,
                    name=f"{safe_slug(model_path.stem)}_{request.runs}_{run_id}",
                ),
            )
            generated = workspace.resolve(
                f"configs/generated/{generated_name}", must_exist=False
            )
            save_config_snapshot(config, generated)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "path": workspace.display(generated),
            "run_id": run_id,
            "output_dir": workspace.display(output_dir),
            "config": config.to_dict(),
            "summary": _config_summary(config),
        }

    @app.get("/api/configs", dependencies=[auth])
    def configs() -> dict[str, Any]:
        found: list[str] = []
        for pattern in ["configs/**/*.yaml", "configs/**/*.yml", "*.yaml", "*.yml"]:
            for path in workspace.root.glob(pattern):
                if path.is_file():
                    found.append(workspace.display(path))
        return {"configs": sorted(set(found))[:300]}

    @app.post("/api/config/load", dependencies=[auth])
    def load_config_api(request: ConfigPathRequest) -> dict[str, Any]:
        path = config_path(request.path)
        try:
            config = load_config(path)
            raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
        return {
            "path": workspace.display(path),
            "content": path.read_text(encoding="utf-8"),
            "config": config.to_dict(),
            "raw": raw,
            "summary": _config_summary(config),
        }

    @app.post("/api/config/save", dependencies=[auth])
    def save_config_api(request: SaveConfigRequest) -> dict[str, Any]:
        path = config_path(request.path, must_exist=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            yaml.safe_load(request.content)
            temporary.write_text(request.content, encoding="utf-8")
            load_config(temporary)
            temporary.replace(path)
            config = load_config(path)
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Configuration was not saved: {error}") from error
        return {"saved": True, "path": workspace.display(path), "summary": _config_summary(config)}

    @app.post("/api/config/update", dependencies=[auth])
    def update_config_api(request: UpdateConfigRequest) -> dict[str, Any]:
        path = config_path(request.path)
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        for dotted, value in request.updates.items():
            target = raw
            parts = dotted.split(".")
            for part in parts[:-1]:
                target = target.setdefault(part, {})
            target[parts[-1]] = value
        content = yaml.safe_dump(raw, sort_keys=False)
        return save_config_api(SaveConfigRequest(path=request.path, content=content))

    def _landscape_payload(config_file: Path) -> dict[str, Any]:
        cfg = load_config(config_file)
        model = load_model(cfg.model)
        configure_solver(model, cfg.solver)
        space = build_medium_space(model, cfg.sampling, cfg.simulation)
        inference = infer_model_roles(model, cfg.model, cfg.optimization)

        rows = _safe_records(space.to_frame())
        baseline = {
            str(key): float(value)
            for key, value in (getattr(model, "medium", {}) or {}).items()
        }

        all_exchange_ids = [str(item) for item in get_exchange_reactions(model)]
        exchange_rows: list[dict[str, Any]] = []
        exchange_by_id: dict[str, dict[str, Any]] = {}

        for rid in all_exchange_ids:
            reaction = model.reactions.get_by_id(rid)
            lower_bound, upper_bound = map(float, reaction.bounds)
            metabolites = list(reaction.metabolites)
            metabolite = metabolites[0] if len(metabolites) == 1 else None
            component_name = ""
            if metabolite is not None:
                component_name = str(
                    getattr(metabolite, "name", None)
                    or getattr(metabolite, "id", "")
                    or ""
                )
            item = {
                "reaction_id": rid,
                "reaction_name": str(reaction.name or ""),
                "component_name": component_name or str(reaction.name or rid),
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
                "native_uptake_capacity": max(0.0, -lower_bound),
                "baseline_uptake": float(baseline.get(rid, 0.0)),
            }
            exchange_rows.append(item)
            exchange_by_id[rid] = item

        for row in rows:
            rid = str(row["reaction_id"])
            info = exchange_by_id.get(rid, {})
            row["reaction_name"] = info.get("reaction_name", "")
            row["component_name"] = info.get("component_name", row["reaction_name"])
            if row.get("original_capacity") is None:
                row["original_capacity"] = info.get("native_uptake_capacity")
            if row.get("baseline_uptake") is None:
                row["baseline_uptake"] = info.get("baseline_uptake", 0.0)

        reactions = [
            {
                "reaction_id": str(reaction.id),
                "reaction_name": str(reaction.name or ""),
            }
            for reaction in model.reactions
        ]

        return {
            "path": workspace.display(config_file),
            "model": {
                "path": cfg.model.path,
                "biomass_reaction_id": cfg.model.biomass_reaction_id,
                "product_reaction_id": cfg.model.product_reaction_id,
            },
            "optimization": {
                "mode": cfg.optimization.mode,
                "minimum_growth_fraction": cfg.optimization.minimum_growth_fraction,
            },
            "inference": {
                "biomass": {
                    "reaction_id": inference.biomass.reaction_id,
                    "confidence": inference.biomass.confidence,
                    "source": inference.biomass.source,
                },
                "product": {
                    "reaction_id": inference.product.reaction_id,
                    "confidence": inference.product.confidence,
                    "source": inference.product.source,
                },
                "selected_optimization_mode": inference.selected_optimization_mode,
                "warnings": list(inference.warnings),
            },
            "sampling": {
                "discovery_scope": cfg.sampling.discovery_scope,
                "automatic_medium_policy": cfg.sampling.automatic_medium_policy,
                "bounds": cfg.sampling.bounds,
                "fixed_reactions": cfg.sampling.fixed_reactions,
                "include_reaction_ids": cfg.sampling.include_reaction_ids,
                "exclude_reaction_ids": cfg.sampling.exclude_reaction_ids,
                "auto_max_uptake_cap": cfg.sampling.auto_max_uptake_cap,
            },
            "medium_space": rows,
            "exchanges": exchange_rows,
            "reactions": reactions,
        }

    @app.get("/api/landscape/setup", dependencies=[auth])
    def landscape_setup(path: str = Query(...)) -> dict[str, Any]:
        config_file = config_path(path)
        try:
            return _landscape_payload(config_file)
        except Exception as error:
            raise HTTPException(status_code=400, detail=str(error)) from error

    @app.post("/api/landscape/setup", dependencies=[auth])
    def save_landscape_setup(request: LandscapeSaveRequest) -> dict[str, Any]:
        config_file = config_path(request.path)
        current_cfg = load_config(config_file)

        allowed_modes = {
            "auto",
            "native_objective",
            "growth_constrained_product",
            "maximize_biomass",
            "maximize_product",
        }
        if request.optimization_mode not in allowed_modes:
            raise HTTPException(status_code=400, detail="Invalid optimization mode.")

        model = load_model(current_cfg.model)
        configure_solver(model, current_cfg.solver)
        all_reaction_ids = {str(reaction.id) for reaction in model.reactions}
        all_exchange_ids = {str(item) for item in get_exchange_reactions(model)}

        for rid, label in (
            (request.biomass_reaction_id, "biomass"),
            (request.product_reaction_id, "product"),
        ):
            if rid and rid not in all_reaction_ids:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown {label} reaction ID: {rid}",
                )

        if (
            request.optimization_mode
            in {"growth_constrained_product", "maximize_biomass"}
            and not request.biomass_reaction_id
        ):
            raise HTTPException(
                status_code=400,
                detail=f"{request.optimization_mode} requires a biomass reaction.",
            )
        if (
            request.optimization_mode
            in {"growth_constrained_product", "maximize_product"}
            and not request.product_reaction_id
        ):
            raise HTTPException(
                status_code=400,
                detail=f"{request.optimization_mode} requires a product reaction.",
            )

        variable_ids = set(request.variable_bounds)
        fixed_ids = set(request.fixed_reactions)
        excluded_ids = set(request.exclude_reaction_ids)

        overlap = (
            (variable_ids & fixed_ids)
            | (variable_ids & excluded_ids)
            | (fixed_ids & excluded_ids)
        )
        if overlap:
            raise HTTPException(
                status_code=400,
                detail=f"Exchange roles overlap: {sorted(overlap)}",
            )

        unknown = (variable_ids | fixed_ids | excluded_ids) - all_exchange_ids
        if unknown:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown exchange reaction IDs: {sorted(unknown)[:20]}",
            )

        for rid, specification in request.variable_bounds.items():
            if "min" not in specification or "max" not in specification:
                raise HTTPException(
                    status_code=400,
                    detail=f"{rid} requires both min and max uptake.",
                )
            minimum = float(specification["min"])
            maximum = float(specification["max"])
            if minimum < 0 or maximum < 0 or minimum > maximum:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid uptake bounds for {rid}: [{minimum}, {maximum}]",
                )

        for rid, value in request.fixed_reactions.items():
            if float(value) < 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Fixed uptake must be non-negative for {rid}.",
                )

        raw = yaml.safe_load(config_file.read_text(encoding="utf-8")) or {}
        raw.setdefault("model", {})
        raw.setdefault("optimization", {})
        raw.setdefault("sampling", {})

        raw["model"]["objective_reaction_id"] = None
        raw["model"]["biomass_reaction_id"] = request.biomass_reaction_id or None
        raw["model"]["product_reaction_id"] = request.product_reaction_id or None
        raw["optimization"]["mode"] = request.optimization_mode
        raw["optimization"]["minimum_growth_fraction"] = float(
            request.minimum_growth_fraction
        )

        raw["sampling"]["bounds"] = {
            rid: {
                "min": float(specification["min"]),
                "max": float(specification["max"]),
            }
            for rid, specification in sorted(request.variable_bounds.items())
        }
        raw["sampling"]["include_reaction_ids"] = sorted(variable_ids)
        raw["sampling"]["fixed_reactions"] = {
            rid: float(value)
            for rid, value in sorted(request.fixed_reactions.items())
        }
        raw["sampling"]["exclude_reaction_ids"] = sorted(excluded_ids)

        content = yaml.safe_dump(raw, sort_keys=False)

        temporary = config_file.with_name(
            config_file.name + ".landscape_validation.yaml"
        )
        try:
            temporary.write_text(content, encoding="utf-8")
            candidate = load_config(temporary)
            candidate_model = load_model(candidate.model)
            configure_solver(candidate_model, candidate.solver)
            build_medium_space(
                candidate_model,
                candidate.sampling,
                candidate.simulation,
            )
            candidate_inference = infer_model_roles(
                candidate_model,
                candidate.model,
                candidate.optimization,
            )
            if (
                candidate.optimization.mode == "growth_constrained_product"
                and (
                    not candidate.model.biomass_reaction_id
                    or not candidate.model.product_reaction_id
                )
            ):
                raise ValueError(
                    "growth_constrained_product requires explicit biomass and product reactions."
                )
        except Exception as error:
            temporary.unlink(missing_ok=True)
            raise HTTPException(
                status_code=400,
                detail=(
                    "Objective/design-space configuration is invalid and was not saved: "
                    f"{error}"
                ),
            ) from error
        temporary.unlink(missing_ok=True)

        saved = save_config_api(
            SaveConfigRequest(path=request.path, content=content)
        )
        return {
            **saved,
            "landscape": _landscape_payload(config_file),
        }

    def start_cli_job(kind: str, path: Path | None, arguments: list[str], unique: bool = False) -> JobRecord:
        command = [sys.executable, "-m", "gem_learning_engine.cli", *arguments]
        try:
            return jobs.start(
                kind=kind,
                command=command,
                config_path=path,
                unique_for_config=unique,
            )
        except RuntimeError as error:
            raise HTTPException(status_code=409, detail=str(error)) from error

    @app.post("/api/run/start", dependencies=[auth])
    def run_start(request: ConfigPathRequest) -> dict[str, Any]:
        path = config_path(request.path)
        load_config(path)
        record = start_cli_job("run", path, ["run", "--config", str(path)], unique=True)
        return asdict(record)

    @app.post("/api/run/pause", dependencies=[auth])
    def run_pause(request: PauseRequest) -> dict[str, Any]:
        path = config_path(request.path)
        config = runtime_config(path)
        store = ResultsStore(config)
        store.paths.checkpoints_dir.mkdir(parents=True, exist_ok=True)
        store.request_pause(request.reason)
        return {"pause_requested": True, "pause_file": str(store.pause_path)}

    @app.post("/api/run/resume", dependencies=[auth])
    def run_resume(request: ConfigPathRequest) -> dict[str, Any]:
        path = config_path(request.path)
        config = runtime_config(path)
        store = ResultsStore(config)
        store.clear_pause()
        record = start_cli_job("run", path, ["run", "--config", str(path)], unique=True)
        return asdict(record)

    @app.post("/api/job/terminate", dependencies=[auth])
    def terminate(request: TerminateRequest) -> dict[str, Any]:
        try:
            return asdict(jobs.terminate(request.job_id))
        except KeyError as error:
            raise HTTPException(status_code=404, detail="Job not found") from error

    @app.post("/api/inspect", dependencies=[auth])
    def inspect(request: InspectRequest) -> dict[str, Any]:
        path = config_path(request.path)
        config = load_config(path)
        default = Path(config.project.output_dir).with_name(Path(config.project.output_dir).name + "_inspection")
        output = workspace.resolve(request.output_dir or default, must_exist=False)
        record = start_cli_job(
            "inspect",
            path,
            ["inspect-model", "--config", str(path), "--output-dir", str(output)],
        )
        return {**asdict(record), "output_dir": workspace.display(output)}

    @app.post("/api/preflight", dependencies=[auth])
    def preflight(request: ConfigPathRequest) -> dict[str, Any]:
        path = config_path(request.path)
        config = load_config(path)
        output = workspace.resolve(Path(config.project.output_dir) / "metadata" / "preflight.json", must_exist=False)
        record = start_cli_job(
            "preflight",
            path,
            ["preflight", "--config", str(path), "--output", str(output)],
        )
        return {**asdict(record), "output": workspace.display(output)}

    @app.post("/api/validate", dependencies=[auth])
    def validate_api(output_dir: str = Body(default="outputs/gui_validation", embed=True)) -> dict[str, Any]:
        output = workspace.resolve(output_dir, must_exist=False)
        record = start_cli_job("validate", None, ["validate", "--output-dir", str(output)])
        return {**asdict(record), "output_dir": workspace.display(output)}

    @app.get("/api/jobs", dependencies=[auth])
    def list_jobs(path: str | None = Query(default=None)) -> dict[str, Any]:
        resolved = config_path(path) if path else None
        return {"jobs": [asdict(item) for item in jobs.list(resolved)]}

    @app.get("/api/status", dependencies=[auth])
    def status(path: str = Query(...)) -> dict[str, Any]:
        config_file = config_path(path)
        config = runtime_config(config_file)
        store = ResultsStore(config)
        expected = len(chunk_ranges(config.simulation.n_simulations, config.simulation.chunk_size))
        pipeline = store.status(expected)
        active = jobs.active_for_config(config_file)
        latest = jobs.list(config_file, limit=1)
        saved = int(pipeline["saved_simulations"])
        total = int(config.simulation.n_simulations)
        pipeline["progress_fraction"] = min(1.0, saved / total) if total else 0.0
        pipeline["progress_percent"] = round(100.0 * pipeline["progress_fraction"], 2)
        pipeline["active_job"] = asdict(active) if active else None
        pipeline["latest_job"] = asdict(latest[0]) if latest else None
        return pipeline

    @app.get("/api/logs", dependencies=[auth])
    def logs(path: str = Query(...), job_id: str | None = Query(default=None)) -> dict[str, Any]:
        config_file = config_path(path)
        config = runtime_config(config_file)
        store = ResultsStore(config)
        pipeline_log = _tail(store.paths.logs_dir / "pipeline.log")
        job_log = ""
        selected: JobRecord | None = None
        if job_id:
            try:
                selected = jobs.get(job_id)
            except KeyError:
                selected = None
        else:
            candidates = jobs.list(config_file, limit=1)
            selected = candidates[0] if candidates else None
        if selected:
            job_log = _tail(Path(selected.log_path))
        return {
            "pipeline_log": pipeline_log,
            "job_log": job_log,
            "job": asdict(selected) if selected else None,
        }

    @app.get("/api/results", dependencies=[auth])
    def results(path: str = Query(...), limit: int = Query(default=100, ge=1, le=500)) -> dict[str, Any]:
        config_file = config_path(path)
        config = runtime_config(config_file)
        store = ResultsStore(config)
        frames: list[pd.DataFrame] = []
        remaining = limit
        for result_path in sorted(
            (store.paths.chunks_dir / "results").glob("results_chunk_*.parquet"), reverse=True
        ):
            if remaining <= 0:
                break
            frame = pd.read_parquet(result_path)
            if len(frame) > remaining:
                frame = frame.tail(remaining)
            frames.append(frame)
            remaining -= len(frame)
        if not frames:
            return {"columns": [], "rows": []}
        frame = pd.concat(reversed(frames), ignore_index=True)
        preferred = [
            "sample_id", "status", "feasible", "objective_value", "biomass_flux", "product_flux",
            "runtime_seconds", "method", "chunk_id",
        ]
        columns = [item for item in preferred if item in frame.columns]
        columns += [item for item in frame.columns if item not in columns]
        frame = frame[columns]
        return {"columns": columns, "rows": _safe_records(frame)}

    @app.post("/api/export/csv", dependencies=[auth])
    def export_csv_start(request: ConfigPathRequest) -> dict[str, Any]:
        config_file = config_path(request.path)
        config = runtime_config(config_file)
        store = ResultsStore(config)
        expected = len(chunk_ranges(config.simulation.n_simulations, config.simulation.chunk_size))
        pipeline = store.status(expected)
        active_run = jobs.active_for_config(config_file, kind="run")
        if active_run is not None:
            raise HTTPException(status_code=409, detail="Wait for the FBA run to finish before generating the complete CSV.")
        if pipeline["invalid_chunks"]:
            raise HTTPException(
                status_code=409,
                detail=f"Invalid chunks prevent CSV export: {pipeline['invalid_chunks']}",
            )
        if len(pipeline["complete_chunks"]) != expected or int(pipeline["saved_simulations"]) != int(config.simulation.n_simulations):
            raise HTTPException(
                status_code=409,
                detail=(
                    "The run is not complete. "
                    f"Saved {pipeline['saved_simulations']} of {config.simulation.n_simulations} simulations "
                    f"and {len(pipeline['complete_chunks'])} of {expected} chunks."
                ),
            )
        record = start_cli_job(
            "export_csv",
            config_file,
            ["export-csv", "--config", str(config_file)],
            unique=True,
        )
        output = default_csv_path(config)
        return {**asdict(record), "csv_path": workspace.display(output)}

    @app.get("/api/export/csv/status", dependencies=[auth])
    def export_csv_status(path: str = Query(...)) -> dict[str, Any]:
        config_file = config_path(path)
        config = runtime_config(config_file)
        output = default_csv_path(config)
        manifest = output.with_name("whole_dataset_manifest.json")
        active = jobs.active_for_config(config_file, kind="export_csv")
        export_jobs = [item for item in jobs.list(config_file, limit=30) if item.kind == "export_csv"]
        latest = export_jobs[0] if export_jobs else None
        payload: dict[str, Any] = {
            "available": output.exists(),
            "csv_path": workspace.display(output),
            "active_job": asdict(active) if active else None,
            "latest_job": asdict(latest) if latest else None,
        }
        if output.exists():
            safe_output = workspace.resolve(output, must_exist=True)
            payload.update(
                {
                    "size_bytes": safe_output.stat().st_size,
                    "modified_at_utc": datetime.fromtimestamp(
                        safe_output.stat().st_mtime, tz=timezone.utc
                    ).isoformat(),
                }
            )
        if manifest.exists():
            safe_manifest = workspace.resolve(manifest, must_exist=True)
            try:
                payload["manifest"] = json.loads(safe_manifest.read_text(encoding="utf-8"))
            except Exception:
                payload["manifest"] = None
        return payload

    @app.get("/api/export/csv/download", dependencies=[auth])
    def export_csv_download(path: str = Query(...)) -> FileResponse:
        config_file = config_path(path)
        config = runtime_config(config_file)
        output = workspace.resolve(default_csv_path(config), must_exist=True)
        filename = f"{safe_slug(config.project.name)}_whole_dataset.csv"
        return FileResponse(output, media_type="text/csv", filename=filename)

    @app.get("/api/catalogs", dependencies=[auth])
    def catalogs(path: str = Query(...)) -> dict[str, Any]:
        config_file = config_path(path)
        store = ResultsStore(runtime_config(config_file))
        items = [item.stem for item in sorted(store.paths.catalogs_dir.glob("*.parquet"))]
        return {"catalogs": items}

    @app.get("/api/catalog", dependencies=[auth])
    def catalog(
        path: str = Query(...),
        name: str = Query(...),
        limit: int = Query(default=100, ge=1, le=500),
    ) -> dict[str, Any]:
        if not name.replace("_", "").isalnum():
            raise HTTPException(status_code=400, detail="Invalid catalog name")
        config_file = config_path(path)
        store = ResultsStore(runtime_config(config_file))
        parquet = store.paths.catalogs_dir / f"{name}.parquet"
        if not parquet.exists():
            raise HTTPException(status_code=404, detail="Catalog not found")
        frame = pd.read_parquet(parquet).head(limit)
        return {"columns": list(frame.columns), "rows": _safe_records(frame)}

    @app.get("/api/preflight/report", dependencies=[auth])
    def preflight_report(path: str = Query(...)) -> dict[str, Any]:
        config_file = config_path(path)
        config = runtime_config(config_file)
        report = Path(config.project.output_dir) / "metadata" / "preflight.json"
        if not report.exists():
            return {"available": False}
        return {"available": True, "report": json.loads(report.read_text(encoding="utf-8"))}

    from .add_general_api import attach_general_uploads
    attach_general_uploads(app,workspace,auth)
    from .optimization.service import attach
    attach(app, workspace, auth)
    from .exploration_integration import attach_exploration
    attach_exploration(app, workspace, auth, token=token)

    from .job_browser import list_all as combined_jobs, get_one as job_details, active as active_jobs
    from .session_lifecycle import attach as attach_session

    def all_jobs():
        return combined_jobs(workspace, jobs, getattr(app.state, 'optimization_queue', None))

    @app.get('/api/unified/jobs', dependencies=[auth])
    def unified_jobs() -> dict[str, Any]:
        return {'jobs': all_jobs()[:300]}

    @app.get('/api/unified/job/{key:path}', dependencies=[auth])
    def unified_job(key: str) -> dict[str, Any]:
        try:
            return job_details(workspace, jobs, getattr(app.state, 'optimization_queue', None), key)
        except KeyError as exc:
            raise HTTPException(404, detail=str(exc)) from exc

    def has_active_computation() -> bool:
        # A status heartbeat must not scan every trajectory/CSV every few seconds.
        if any(j.status in ('running', 'terminating') for j in jobs.list(limit=1000)):
            return True
        queue = getattr(app.state, 'optimization_queue', None)
        if queue is not None:
            with queue.lock:
                if queue.state.get('enabled') and any(j.get('status') == 'queued' for j in queue.state.get('jobs', [])):
                    return True
                for j in queue.state.get('jobs', []):
                    if j.get('status') in ('running', 'cancelling') and queue.worker_active(queue.root / j['id']):
                        return True
        records = workspace.root / 'exploration_workspace' / 'run_records'
        if records.is_dir():
            for f in records.glob('exp-*.json'):
                try:
                    d = json.loads(f.read_text(encoding='utf-8'))
                    if d.get('status') == 'running' and d.get('pid') == os.getpid():
                        return True
                except (OSError, ValueError):
                    continue
        return False

    attach_session(app, auth, has_active_computation)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the GEM Learning Engine browser GUI using uvicorn.")
    parser.add_argument("--workspace", default=".", help="Workspace root visible to the GUI.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--token", default=os.environ.get("GEM_GUI_TOKEN"))
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Allow non-loopback binding without an API token. Not recommended.",
    )
    parser.add_argument("--reload", action="store_true", help="Development only.")
    args = parser.parse_args()

    if args.host not in {"127.0.0.1", "localhost", "::1"} and not args.token and not args.insecure:
        parser.error("Binding beyond localhost requires --token or explicit --insecure.")

    if os.environ.get("GEM_ENABLE_EXPLORATION") == "1" and args.host not in {"127.0.0.1", "localhost", "::1"}:
        parser.error("Integrated Exploration is localhost-only; use a local Mac browser or an authenticated SSH port-forward.")

    import uvicorn

    app = create_app(args.workspace, token=args.token)
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
