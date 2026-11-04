from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
from scipy import sparse
from scipy.optimize import linprog
from .types import ModelData

@dataclass
class FluxSolution:
    status: str
    objective_value: float
    fluxes: Dict[str, float]
    method: str = "FBA"
    message: str = ""


def model_summary(model: ModelData, solution: Optional[FluxSolution] = None) -> dict:
    exchange = 0
    transport = 0
    zero_width = 0
    gene_associated = 0
    subsystem_counts = {}
    for r in model.reactions.values():
        if len(r.stoichiometry) == 1 or r.id.upper().startswith(("EX_", "DM_", "SK_")):
            exchange += 1
        comps = {model.metabolites[mid].compartment for mid in r.stoichiometry if mid in model.metabolites}
        if len(comps) > 1:
            transport += 1
        if abs(r.upper_bound - r.lower_bound) < 1e-15:
            zero_width += 1
        if r.gene_rule:
            gene_associated += 1
        if r.subsystem:
            subsystem_counts[r.subsystem] = subsystem_counts.get(r.subsystem, 0) + 1
    out = {
        "model_id": model.id,
        "model_name": model.name,
        "reactions": len(model.reactions),
        "metabolites": len(model.metabolites),
        "genes": model.gene_count,
        "compartments": len(model.compartments),
        "exchange_boundary_reactions": exchange,
        "transport_reactions": transport,
        "fixed_flux_reactions": zero_width,
        "gene_associated_reactions": gene_associated,
        "objective_reactions": model.objective_reaction_ids(),
        "objective_direction": model.objective_direction,
        "subsystems": subsystem_counts,
    }
    if solution is not None:
        active = [v for v in solution.fluxes.values() if abs(v) > 1e-9]
        out.update({
            "solution_status": solution.status,
            "objective_value": solution.objective_value,
            "active_reactions": len(active),
        })
    return out


def _matrix(model: ModelData):
    rids = list(model.reactions)
    mids = list(model.metabolites)
    ri = {r: i for i, r in enumerate(rids)}
    mi = {m: i for i, m in enumerate(mids)}
    rows, cols, data = [], [], []
    for rid, r in model.reactions.items():
        j = ri[rid]
        for mid, coeff in r.stoichiometry.items():
            if mid in mi and coeff and not model.metabolites[mid].boundary_condition:
                rows.append(mi[mid]); cols.append(j); data.append(coeff)
    S = sparse.csr_matrix((data, (rows, cols)), shape=(len(mids), len(rids)), dtype=float)
    return S, rids, mids


def run_fba(model: ModelData, objective_reaction: str | None = None,
            objective_direction: str = "max", bounds_overrides: Optional[Dict[str, tuple[float, float]]] = None) -> FluxSolution:
    S, rids, _ = _matrix(model)
    n = len(rids)
    c = np.zeros(n, dtype=float)
    if objective_reaction:
        if objective_reaction not in model.reactions:
            return FluxSolution("error", math.nan, {}, "FBA", f"Unknown objective reaction: {objective_reaction}")
        c[rids.index(objective_reaction)] = 1.0
    elif model.objectives:
        for rid, coef in model.objectives.items():
            if rid in model.reactions:
                c[rids.index(rid)] = coef
    else:
        return FluxSolution("error", math.nan, {}, "FBA", "Model has no objective; choose one explicitly.")
    maximize = objective_direction.lower().startswith("max")
    bounds = []
    overrides = bounds_overrides or {}
    for rid in rids:
        r = model.reactions[rid]
        lb, ub = overrides.get(rid, (r.lower_bound, r.upper_bound))
        bounds.append((lb, ub))
    res = linprog(-c if maximize else c, A_eq=S, b_eq=np.zeros(S.shape[0]), bounds=bounds, method="highs", options={"time_limit":120.0,"primal_feasibility_tolerance":1e-9,"dual_feasibility_tolerance":1e-9})
    if not res.success:
        return FluxSolution({1:"limit",2:"infeasible",3:"unbounded"}.get(res.status,"error"), math.nan, {}, "FBA", res.message)
    residual=float(np.max(np.abs(S @ res.x),initial=0))
    violation=max((max(lb-res.x[i],res.x[i]-ub,0) for i,(lb,ub) in enumerate(bounds)),default=0)
    if residual>1e-7 or violation>1e-7:
        return FluxSolution("numerical_error", math.nan, {}, "FBA", f"Solution failed validation: residual {residual:.3g}, bound violation {violation:.3g}")
    obj = float(c @ res.x)
    return FluxSolution("optimal", obj, {rid: float(res.x[i]) for i, rid in enumerate(rids)}, "FBA", res.message)


def top_exchange_fluxes(model: ModelData, solution: FluxSolution, n: int = 15):
    rows = []
    for rid, v in solution.fluxes.items():
        r = model.reactions.get(rid)
        if not r:
            continue
        if len(r.stoichiometry) == 1 or rid.upper().startswith(("EX_", "DM_", "SK_")):
            rows.append((rid, r.name, v))
    return sorted(rows, key=lambda x: abs(x[2]), reverse=True)[:n]
