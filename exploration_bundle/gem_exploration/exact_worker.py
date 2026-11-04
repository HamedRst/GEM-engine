from __future__ import annotations

"""Single-candidate exact FBA worker used by the Exploration Atlas validator.

A new process is intentionally used for each candidate so a pathological LP solve
cannot stall the complete five-landscape exploration campaign.  The parent process
owns timeout/fallback policy and may retry the *same* candidate with glpk_exact.
"""

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def _load_model(path: Path, solver: str | None = None):
    try:
        from cobra import io as cobra_io
    except Exception as exc:
        raise RuntimeError("COBRApy is required for exact FBA validation") from exc
    s = path.suffix.lower()
    if s == ".json":
        model = cobra_io.load_json_model(str(path))
    elif s in {".xml", ".sbml"}:
        model = cobra_io.read_sbml_model(str(path))
    elif s == ".mat":
        model = cobra_io.load_matlab_model(str(path))
    elif s in {".yaml", ".yml"}:
        loader = getattr(cobra_io, "load_yaml_model", None)
        if loader is None:
            raise RuntimeError("This COBRApy installation has no YAML model loader")
        model = loader(str(path))
    else:
        raise ValueError(f"Unsupported model format: {path}")
    if solver:
        model.solver = str(solver)
    return model


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _evaluate(run_dir: Path, model_path: Path, candidate: dict[str, float], solver_override: str | None) -> dict[str, Any]:
    cfg = _read_yaml(run_dir / "metadata" / "config_snapshot.yaml")
    model_cfg = cfg.get("model") or {}
    sampling = cfg.get("sampling") or {}
    optimization = cfg.get("optimization") or {}
    simulation = cfg.get("simulation") or {}
    solver = solver_override or (cfg.get("solver") or {}).get("name")
    model = _load_model(model_path, solver)

    biomass = str(model_cfg.get("biomass_reaction_id") or model_cfg.get("objective_reaction_id") or "")
    product = str(model_cfg.get("product_reaction_id") or "")
    if not biomass or not product:
        raise RuntimeError(f"Missing biomass/product reaction in frozen config: biomass={biomass!r}, product={product!r}")
    model.reactions.get_by_id(biomass)
    model.reactions.get_by_id(product)

    feasible_statuses = {str(x).lower() for x in simulation.get("feasible_statuses", ["optimal"])}
    fixed = sampling.get("fixed_reactions") or {}
    base_medium = {str(k): float(v) for k, v in (getattr(model, "medium", {}) or {}).items()}
    for rid, value in fixed.items():
        base_medium[str(rid)] = max(0.0, float(value))
    for rid, value in candidate.items():
        if rid not in base_medium:
            # COBRApy accepts only exchange reactions supported by model.medium.
            # Adding a valid exchange explicitly is still safe when it was closed in
            # the model's default medium.
            try:
                model.reactions.get_by_id(str(rid))
            except Exception as exc:
                raise RuntimeError(f"Candidate reaction {rid!r} not found in model") from exc
        base_medium[str(rid)] = max(0.0, float(value))

    t0 = time.perf_counter()
    with model:
        model.medium = base_medium
        model.objective = model.reactions.get_by_id(biomass)
        model.objective_direction = "max"
        s1 = model.optimize()
        st1 = str(s1.status)
        ok1 = st1.lower() in feasible_statuses
        growth = float(s1.fluxes[biomass]) if ok1 else np.nan
        if not ok1 or not np.isfinite(growth):
            return {
                "status": f"growth={st1}", "growth_status": st1, "product_status": "not_run",
                "feasible": False, "maximum_growth": None, "product_flux": None,
                "growth_floor": None, "fba_solves": 1, "solver": str(model.solver.interface.__name__),
                "solver_wall_seconds": float(time.perf_counter() - t0),
            }

        frac = float(optimization.get("minimum_growth_fraction", 0.9))
        floor = max(0.0, frac * growth)
        biomass_rxn = model.reactions.get_by_id(biomass)
        old_lb = float(biomass_rxn.lower_bound)
        biomass_rxn.lower_bound = max(old_lb, floor)
        model.objective = model.reactions.get_by_id(product)
        model.objective_direction = "max"
        s2 = model.optimize()
        st2 = str(s2.status)
        ok2 = st2.lower() in feasible_statuses
        prod = float(s2.fluxes[product]) if ok2 else np.nan
        feasible = bool(ok2 and np.isfinite(prod))
        return {
            "status": "optimal" if feasible else f"growth={st1};product={st2}",
            "growth_status": st1, "product_status": st2, "feasible": feasible,
            "maximum_growth": float(growth), "product_flux": float(prod) if np.isfinite(prod) else None,
            "growth_floor": float(floor), "fba_solves": 2,
            "solver": str(model.solver.interface.__name__),
            "solver_wall_seconds": float(time.perf_counter() - t0),
        }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--candidate-json", required=True)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--solver", default=None)
    args = ap.parse_args()
    candidate = json.loads(Path(args.candidate_json).read_text(encoding="utf-8"))
    try:
        result = _evaluate(
            Path(args.run_dir).expanduser().resolve(),
            Path(args.model_path).expanduser().resolve(),
            {str(k): float(v) for k, v in candidate.items()},
            args.solver,
        )
        result["worker_status"] = "PASS"
    except Exception as exc:
        result = {"worker_status": "FAIL", "error": f"{type(exc).__name__}: {exc}"}
    out = Path(args.output_json).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    raise SystemExit(0 if result.get("worker_status") == "PASS" else 2)


if __name__ == "__main__":
    main()
