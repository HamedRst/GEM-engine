from __future__ import annotations
from collections import deque
from typing import Optional
import networkx as nx
from .types import ModelData
from .analysis import FluxSolution

CURRENCY_TOKENS = {
    "h", "h2o", "atp", "adp", "amp", "pi", "ppi", "nad", "nadh", "nadp", "nadph",
    "coa", "accoa", "o2", "co2", "nh4", "na1", "k", "cl", "hco3", "fadh2", "fad"
}

def _base_id(mid: str) -> str:
    x = mid.lower()
    for prefix in ("m_",):
        if x.startswith(prefix): x = x[len(prefix):]
    x = x.replace("__91__", "[").replace("__93__", "]")
    for sep in ("[", "_"):
        if sep in x:
            x = x.split(sep)[0]
    return x


def is_currency(mid: str, name: str = "", metabolite=None) -> bool:
    if metabolite is not None:
        from .pathways import identify
        role,_=identify(metabolite)
        if role in CURRENCY_TOKENS:return True
    b = _base_id(mid)
    n = name.lower().strip()
    return b in CURRENCY_TOKENS or n in {
        "water", "proton", "hydrogen", "phosphate", "diphosphate", "oxygen", "carbon dioxide"
    }


def _new_graph():
    return nx.DiGraph()


def _add_reaction_flow(G, model: ModelData, rid: str, flux: Optional[float], reverse: bool = False,
                       exclude_currency: bool = True, target: Optional[str] = None):
    r = model.reactions[rid]
    rxn_node = f"R::{rid}"
    G.add_node(rxn_node, kind="reaction", id=rid, label=r.name or rid, subsystem=r.subsystem,
               flux=flux, reversible=r.reversible)
    for mid, coeff in r.stoichiometry.items():
        m = model.metabolites.get(mid)
        if not m: continue
        if exclude_currency and is_currency(mid, m.name, m) and mid != target:
            continue
        met_node = f"M::{mid}"
        G.add_node(met_node, kind="metabolite", id=mid, label=m.name or mid,
                   compartment=m.compartment, target=(mid == target))
        c = -coeff if reverse else coeff
        if c < 0:
            G.add_edge(met_node, rxn_node, stoich=abs(coeff), flux=abs(flux) if flux is not None else None)
        elif c > 0:
            G.add_edge(rxn_node, met_node, stoich=abs(coeff), flux=abs(flux) if flux is not None else None)


def active_subnetwork(model: ModelData, solution: FluxSolution, threshold: float = 1e-8,
                      max_reactions: int = 80, exclude_currency: bool = True):
    active = [(rid, v) for rid, v in solution.fluxes.items() if abs(v) >= threshold and rid in model.reactions]
    active.sort(key=lambda x: abs(x[1]), reverse=True)
    total_active = len(active)
    kept = active[:max_reactions] if max_reactions else active
    G = _new_graph()
    for rid, v in kept:
        _add_reaction_flow(G, model, rid, v, reverse=(v < 0), exclude_currency=exclude_currency)
    G.graph.update(view="active", total_active_reactions=total_active, shown_reactions=len(kept),
                   truncated=(len(kept) < total_active), threshold=threshold)
    return G


def structural_graph(model: ModelData, exclude_currency: bool = True, target: Optional[str] = None):
    G = _new_graph()
    for rid, r in model.reactions.items():
        _add_reaction_flow(G, model, rid, None, r.upper_bound <= 0 and r.lower_bound < 0, exclude_currency, target)
        G.nodes[f"R::{rid}"]["bounds_blocked"] = r.lower_bound == r.upper_bound == 0
        if r.reversible:
            _add_reaction_flow(G, model, rid, None, True, exclude_currency, target)
    return G


def flux_graph(model: ModelData, solution: FluxSolution, threshold: float = 1e-8,
               exclude_currency: bool = True, target: Optional[str] = None):
    G = _new_graph()
    for rid, v in solution.fluxes.items():
        if rid in model.reactions and abs(v) >= threshold:
            _add_reaction_flow(G, model, rid, v, v < 0, exclude_currency, target)
    return G


def _bfs_nodes(G: nx.DiGraph, source: str, depth: int, reverse: bool = False):
    H = G.reverse(copy=False) if reverse else G
    seen = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        d = seen[u]
        if d >= depth * 2:  # metabolite -> reaction -> metabolite counts as one biochemical step
            continue
        for v in H.successors(u):
            if v not in seen:
                seen[v] = d + 1
                q.append(v)
    return seen


def target_subnetwork(model: ModelData, target_metabolite: str, direction: str = "upstream",
                      depth: int = 3, solution: Optional[FluxSolution] = None,
                      active_only: bool = False, threshold: float = 1e-8,
                      max_nodes: int = 120, exclude_currency: bool = True):
    if target_metabolite not in model.metabolites:
        raise KeyError(f"Unknown metabolite: {target_metabolite}")
    base = flux_graph(model, solution, threshold, exclude_currency, target_metabolite) if (active_only and solution) \
        else structural_graph(model, exclude_currency, target_metabolite)
    source = f"M::{target_metabolite}"
    if source not in base:
        H = nx.DiGraph(); m=model.metabolites[target_metabolite]
        H.add_node(source, kind="metabolite", id=target_metabolite, label=m.name, compartment=m.compartment, target=True)
        H.graph.update(view="target", target=target_metabolite, direction=direction, depth=depth)
        return H
    selected = {source: 0}
    if direction in {"upstream", "both"}:
        selected.update(_bfs_nodes(base, source, depth, reverse=True))
    if direction in {"downstream", "both"}:
        for k,v in _bfs_nodes(base, source, depth, reverse=False).items():
            selected[k] = min(selected.get(k, 10**9), v)
    nodes = sorted(selected, key=lambda n: (selected[n], 0 if base.nodes[n].get("kind") == "metabolite" else 1, n))
    truncated = len(nodes) > max_nodes
    nodes = nodes[:max_nodes]
    H = base.subgraph(nodes).copy()
    H.graph.update(view="target", target=target_metabolite, direction=direction, depth=depth,
                   truncated=truncated, active_only=active_only)
    return H


def neighborhood_subnetwork(model: ModelData, entity_id: str, entity_kind: str = "metabolite", depth: int = 2,
                            max_nodes: int = 100, exclude_currency: bool = True):
    base = structural_graph(model, exclude_currency, entity_id if entity_kind=="metabolite" else None)
    source = ("M::" if entity_kind == "metabolite" else "R::") + entity_id
    if source not in base:
        raise KeyError(entity_id)
    U = base.to_undirected()
    dists = nx.single_source_shortest_path_length(U, source, cutoff=depth*2)
    nodes = sorted(dists, key=lambda x: (dists[x], x))[:max_nodes]
    H = base.subgraph(nodes).copy()
    H.graph.update(view="neighborhood", entity_id=entity_id, entity_kind=entity_kind, depth=depth,
                   truncated=len(dists)>max_nodes)
    return H


def compare_flux_network(model: ModelData, a: FluxSolution, b: FluxSolution, threshold: float = 1e-8,
                         delta_threshold: float = 1e-6, max_reactions: int = 80, exclude_currency: bool = True):
    rows = []
    for rid in model.reactions:
        va, vb = a.fluxes.get(rid,0.0), b.fluxes.get(rid,0.0)
        d = vb-va
        if max(abs(va), abs(vb)) >= threshold and abs(d) >= delta_threshold:
            rows.append((rid, va, vb, d))
    rows.sort(key=lambda x: abs(x[3]), reverse=True)
    total = len(rows); rows=rows[:max_reactions]
    G=_new_graph()
    for rid,va,vb,d in rows:
        _add_reaction_flow(G, model, rid, vb, vb<0, exclude_currency)
        n=G.nodes.get(f"R::{rid}")
        if n is not None:
            n["flux_a"]=va; n["flux_b"]=vb; n["delta_flux"]=d
    G.graph.update(view="comparison", total_changed_reactions=total, shown_reactions=len(rows), truncated=len(rows)<total)
    return G

def subsystem_subnetwork(model: ModelData, subsystem: str, solution: Optional[FluxSolution] = None,
                         active_only: bool = False, threshold: float = 1e-8,
                         max_reactions: int = 100, exclude_currency: bool = True):
    matches=[]
    for rid,r in model.reactions.items():
        if (r.subsystem or "") == subsystem:
            v=solution.fluxes.get(rid,0.0) if solution else None
            if active_only and (solution is None or abs(v) < threshold):
                continue
            matches.append((rid,v))
    if solution:
        matches.sort(key=lambda x: abs(x[1] or 0.0), reverse=True)
    else:
        matches.sort(key=lambda x: x[0])
    total=len(matches); kept=matches[:max_reactions] if max_reactions else matches
    G=_new_graph()
    for rid,v in kept:
        rev=(v is not None and v<0)
        _add_reaction_flow(G,model,rid,v,rev,exclude_currency)
        if v is None and model.reactions[rid].reversible:
            _add_reaction_flow(G,model,rid,None,True,exclude_currency)
    G.graph.update(view="subsystem",subsystem=subsystem,total_reactions=total,
                   shown_reactions=len(kept),truncated=len(kept)<total,active_only=active_only)
    return G


def publication_target_subnetwork(model: ModelData, target_metabolite: str, direction: str = 'upstream',
                                  depth: int = 3, solution: Optional[FluxSolution] = None,
                                  flux_weighted: bool = False, threshold: float = 1e-8,
                                  max_reactions: int = 28, exclude_currency: bool = True):
    """Build a compact target-centered graph intended for manuscript figures.

    Structural mode (flux_weighted=False) requires no FBA solution. Flux-weighted mode restricts
    the graph to active reactions and ranks candidate reactions by |flux| within each distance layer.
    The final graph is forced to the weakly connected component containing the target.
    """
    if target_metabolite not in model.metabolites:
        raise KeyError(f'Unknown metabolite: {target_metabolite}')
    if flux_weighted and (solution is None or solution.status != 'optimal'):
        raise ValueError('Flux-weighted publication view requires an optimal standard FBA solution.')
    base = flux_graph(model, solution, threshold, exclude_currency, target_metabolite) if flux_weighted \
        else structural_graph(model, exclude_currency, target_metabolite)
    source=f'M::{target_metabolite}'
    if source not in base:
        H=nx.DiGraph()
        m=model.metabolites[target_metabolite]
        H.add_node(source,kind='metabolite',id=target_metabolite,label=m.name,compartment=m.compartment,target=True)
        H.graph.update(view='publication_target',target=target_metabolite,direction=direction,depth=depth,
                       flux_weighted=flux_weighted,truncated=False)
        return H

    dist={source:0}
    if direction in {'upstream','both'}:
        for n,d in _bfs_nodes(base,source,depth,reverse=True).items():
            dist[n]=min(dist.get(n,10**9),d)
    if direction in {'downstream','both'}:
        for n,d in _bfs_nodes(base,source,depth,reverse=False).items():
            dist[n]=min(dist.get(n,10**9),d)
    candidates=set(dist)
    rxns=[n for n in candidates if base.nodes[n].get('kind')=='reaction']

    def rxn_rank(n):
        d=dist.get(n,999)
        rid=base.nodes[n].get('id','')
        if flux_weighted:
            f=abs(float(base.nodes[n].get('flux') or 0.0))
            return (d,-f,rid)
        # Prefer reactions close to target and with lower non-currency branching degree.
        deg=base.in_degree(n)+base.out_degree(n)
        return (d,deg,rid)

    keep_rxns=set(sorted(rxns,key=rxn_rank)[:max_reactions])
    keep={source}
    for r in keep_rxns:
        keep.add(r)
        keep.update(base.predecessors(r))
        keep.update(base.successors(r))
    keep &= candidates
    H=base.subgraph(keep).copy()
    if source in H:
        comp=next((c for c in nx.weakly_connected_components(H) if source in c),{source})
        H=H.subgraph(comp).copy()
    # Remove metabolites that became isolated after pruning, except target.
    H.remove_nodes_from([n for n in list(H.nodes) if n != source and H.degree(n)==0])
    H.graph.update(view='publication_target',target=target_metabolite,direction=direction,depth=depth,
                   flux_weighted=flux_weighted,threshold=threshold,max_reactions=max_reactions,
                   candidate_reactions=len(rxns),shown_reactions=sum(1 for _,d in H.nodes(data=True) if d.get('kind')=='reaction'),
                   truncated=len(rxns)>max_reactions)
    return H
