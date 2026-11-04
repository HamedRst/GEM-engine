from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
from .types import ModelData

@dataclass
class ExchangeInput:
    reaction_id: str
    reaction_name: str
    metabolite_id: str
    metabolite_name: str
    compartment: str
    coefficient: float
    lower_bound: float
    upper_bound: float
    uptake_capacity: float


def _is_external(model: ModelData, compartment: str) -> bool:
    cid=(compartment or '').lower()
    cname=(model.compartments.get(compartment, '') or '').lower()
    return cid in {'e','extracellular','extra'} or 'extracellular' in cname or 'external' in cname


def external_exchange_reactions(model: ModelData) -> list[ExchangeInput]:
    """Return one-metabolite extracellular boundary reactions with their current uptake capacity.

    Uptake direction is inferred from the stoichiometric sign. For the common EX convention
    (coefficient < 0), negative flux is uptake. For coefficient > 0, positive flux is uptake.
    """
    rows=[]
    for rid,r in model.reactions.items():
        if len(r.stoichiometry) != 1:
            continue
        mid,coef=next(iter(r.stoichiometry.items()))
        m=model.metabolites.get(mid)
        if not m or not _is_external(model,m.compartment):
            continue
        if coef < 0:
            uptake=max(0.0, -float(r.lower_bound))
        elif coef > 0:
            uptake=max(0.0, float(r.upper_bound))
        else:
            continue
        rows.append(ExchangeInput(
            reaction_id=rid,reaction_name=r.name or rid,
            metabolite_id=mid,metabolite_name=m.name or mid,compartment=m.compartment,
            coefficient=float(coef),lower_bound=float(r.lower_bound),upper_bound=float(r.upper_bound),
            uptake_capacity=uptake,
        ))
    rows.sort(key=lambda x:(x.metabolite_name.lower(),x.reaction_id))
    return rows


def encoded_inputs(model: ModelData) -> list[ExchangeInput]:
    return [x for x in external_exchange_reactions(model) if x.uptake_capacity > 1e-12]


def custom_medium_bounds(model: ModelData, selected: Dict[str,float], close_unselected: bool=True) -> Dict[str,tuple[float,float]]:
    """Build reaction-bound overrides for a custom medium.

    selected maps exchange reaction id -> positive maximum uptake magnitude.
    If close_unselected=True, uptake through all other extracellular one-metabolite exchange
    reactions is closed while secretion remains allowed in its original direction.
    """
    all_ex={x.reaction_id:x for x in external_exchange_reactions(model)}
    out:Dict[str,tuple[float,float]]={}
    if close_unselected:
        for rid,x in all_ex.items():
            lb,ub=x.lower_bound,x.upper_bound
            if x.coefficient < 0:  # negative v = uptake
                lb=max(0.0,lb)
            else:                  # positive v = uptake
                ub=min(0.0,ub)
            out[rid]=(lb,ub)
    for rid,cap in selected.items():
        if rid not in all_ex:
            raise KeyError(f'Not an extracellular exchange reaction: {rid}')
        cap=max(0.0,float(cap))
        x=all_ex[rid]
        lb,ub=out.get(rid,(x.lower_bound,x.upper_bound))
        if x.coefficient < 0:
            lb=-cap
            ub=max(ub,0.0)
        else:
            ub=cap
            lb=min(lb,0.0)
        out[rid]=(lb,ub)
    return out
