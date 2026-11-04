from __future__ import annotations
import csv, json
from pathlib import Path
import networkx as nx
from .types import ModelData
from .analysis import FluxSolution, model_summary


def export_graph_tables(G: nx.DiGraph, output_dir: str|Path, stem: str="network"):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    nodes=out/f"{stem}_nodes.csv"; edges=out/f"{stem}_edges.csv"; meta=out/f"{stem}_metadata.json"
    node_keys=sorted({k for _,d in G.nodes(data=True) for k in d})
    with nodes.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["node"]+node_keys); w.writeheader()
        for n,d in G.nodes(data=True): w.writerow({"node":n,**d})
    edge_keys=sorted({k for *_,d in G.edges(data=True) for k in d})
    with edges.open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=["source","target"]+edge_keys); w.writeheader()
        for u,v,d in G.edges(data=True): w.writerow({"source":u,"target":v,**d})
    meta.write_text(json.dumps(G.graph,indent=2,default=str),encoding="utf-8")
    return {"nodes":nodes,"edges":edges,"metadata":meta}


def export_fluxes(solution: FluxSolution, output: str|Path):
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True)
    with p.open("w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["reaction_id","flux"])
        for rid,v in sorted(solution.fluxes.items(), key=lambda kv:abs(kv[1]), reverse=True): w.writerow([rid,v])
    return p


def export_summary(model: ModelData, solution: FluxSolution|None, output: str|Path):
    p=Path(output); p.parent.mkdir(parents=True,exist_ok=True)
    import math
    data=model_summary(model,solution)
    data={k:(None if isinstance(v,float) and not math.isfinite(v) else v) for k,v in data.items()}
    p.write_text(json.dumps(data,indent=2,default=str,allow_nan=False),encoding="utf-8")
    return p
