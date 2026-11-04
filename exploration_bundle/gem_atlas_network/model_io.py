from __future__ import annotations
import json
from pathlib import Path
from typing import Dict
from lxml import etree
from .types import MetaboliteData, ReactionData, ModelData

SBML = "http://www.sbml.org/sbml/level3/version1/core"
FBC = "http://www.sbml.org/sbml/level3/version1/fbc/version2"
GROUPS = "http://www.sbml.org/sbml/level3/version1/groups/version1"


def _num(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        if v is None:return float(default)
        raise ValueError(f"Invalid numeric model value: {v}")


def _annotations(node):
    out={}
    for e in node.findall(".//{http://www.w3.org/1999/02/22-rdf-syntax-ns#}li"):
        v=e.get("{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource", "")
        if v:
            bits=v.rstrip('/').split('/')
            key=bits[-2] if len(bits)>1 else 'uri'
            out.setdefault(key, []).append(bits[-1])
    return out


def load_model(path: str | Path) -> ModelData:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".json":
        m = _load_cobra_json(path)
        return _validate(m, path)
    if suffix in {".xml", ".sbml"}:
        m = _load_sbml(path)
        return _validate(m, path)
    raise ValueError(f"Unsupported model format: {path.suffix}. Use .xml, .sbml, or .json")


def _load_cobra_json(path: Path) -> ModelData:
    d = json.loads(path.read_text(encoding="utf-8"))
    metabolites = {}
    for m in d.get("metabolites", []):
        metabolites[m["id"]] = MetaboliteData(
            id=m["id"], name=m.get("name") or m["id"],
            compartment=m.get("compartment", ""), formula=m.get("formula") or "",
            charge=_num(m.get("charge"), 0.0) if m.get("charge") is not None else None,
            annotations=m.get("annotation", {}) or {},
        )
    reactions = {}
    for r in d.get("reactions", []):
        if "lower_bound" not in r or "upper_bound" not in r:raise ValueError(f"Reaction {r['id']} lacks explicit COBRA bounds")
        reactions[r["id"]] = ReactionData(
            id=r["id"], name=r.get("name") or r["id"],
            lower_bound=_num(r.get("lower_bound"), -1000),
            upper_bound=_num(r.get("upper_bound"), 1000),
            stoichiometry={k: _num(v) for k, v in r.get("metabolites", {}).items()},
            subsystem=r.get("subsystem") or "", gene_rule=r.get("gene_reaction_rule") or "",
            annotations=r.get("annotation", {}) or {},
        )
    compartments = d.get("compartments", {}) or {}
    objectives: Dict[str, float] = {}
    objective = d.get("objective")
    if isinstance(objective, str):
        objectives[objective] = 1.0
    elif isinstance(objective, dict):
        for k, v in objective.items():
            objectives[k] = _num(v)
    if not objectives:
        for r in d.get("reactions", []):
            c = _num(r.get("objective_coefficient"), 0.0)
            if c:
                objectives[r["id"]] = c
    genes = d.get("genes", []) or []
    return ModelData(
        id=d.get("id") or path.stem, name=d.get("name") or d.get("id") or path.stem,
        metabolites=metabolites, reactions=reactions, compartments=compartments,
        objectives=objectives, objective_direction=d.get("objective_direction", "max"), gene_count=len(genes), source_path=path,
        genes={g["id"]:{"name":g.get("name") or g["id"], "annotations":g.get("annotation", {})} for g in genes},
    )


def _load_sbml(path: Path) -> ModelData:
    parser = etree.XMLParser(huge_tree=True, recover=False, resolve_entities=False, no_network=True, load_dtd=False)
    tree = etree.parse(str(path), parser)
    if tree.docinfo.doctype:
        raise ValueError("DTD declarations are not allowed")
    root = tree.getroot()
    SBML = etree.QName(root).namespace
    FBC = next((v for v in root.nsmap.values() if v and "/fbc/version" in v), "http://www.sbml.org/sbml/level3/version1/fbc/version2")
    model = root.find(f"{{{SBML}}}model")
    if model is None:
        raise ValueError("No SBML model element found")

    parameters = {}
    for p in model.findall(f".//{{{SBML}}}listOfParameters/{{{SBML}}}parameter"):
        parameters[p.get("id")] = _num(p.get("value"), 0.0)

    compartments = {}
    for c in model.findall(f".//{{{SBML}}}listOfCompartments/{{{SBML}}}compartment"):
        compartments[c.get("id")] = c.get("name") or c.get("id")

    genes = {}
    gene_refs = {}
    for g in model.findall(f".//{{{FBC}}}geneProduct"):
        ref = g.get(f"{{{FBC}}}id")
        gid = ref[2:] if ref.startswith("G_") else ref
        import re
        gid = re.sub(r"__(\d+)__", lambda mt: chr(int(mt.group(1))), gid)
        if gid in genes: raise ValueError(f"Duplicate gene: {gid}")
        genes[gid] = {"name": g.get(f"{{{FBC}}}name") or g.get(f"{{{FBC}}}label") or gid, "annotations": _annotations(g), "sbml_id": ref}
        gene_refs[ref] = gid
    def gpr(node):
        tag = etree.QName(node).localname
        if tag == "geneProductRef":
            ref=node.get(f"{{{FBC}}}geneProduct")
            if ref not in gene_refs: raise ValueError(f"Unknown SBML gene reference {ref}")
            return gene_refs[ref]
        if tag in {"and", "or"}:
            return "(" + (" " + tag + " ").join(gpr(c) for c in node) + ")"
        if tag == "geneProductAssociation" and len(node) == 1: return gpr(node[0])
        raise ValueError(f"Invalid SBML GPR element {tag}")

    metabolites = {}
    for s in model.findall(f".//{{{SBML}}}listOfSpecies/{{{SBML}}}species"):
        sid = s.get("id")
        charge = s.get(f"{{{FBC}}}charge")
        formula = s.get(f"{{{FBC}}}chemicalFormula") or ""
        metabolites[sid] = MetaboliteData(
            id=sid, name=s.get("name") or sid,
            compartment=s.get("compartment") or "", formula=formula,
            charge=_num(charge) if charge is not None else None,
            annotations=_annotations(s),boundary_condition=s.get("boundaryCondition")=="true",
        )

    subsystem_by_rxn = {}
    for grp in model.findall(f".//{{{GROUPS}}}group"):
        gname = grp.get(f"{{{GROUPS}}}name") or grp.get(f"{{{GROUPS}}}id") or ""
        for mem in grp.findall(f".//{{{GROUPS}}}member"):
            rid = mem.get(f"{{{GROUPS}}}idRef")
            if rid and rid not in subsystem_by_rxn:
                subsystem_by_rxn[rid] = gname

    def bound_value(r, attr, default):
        key = r.get(f"{{{FBC}}}{attr}")
        if key is None:
            raise ValueError(f"Reaction {r.get('id')} lacks FBC {attr}; explicit bounds are required")
        if key in parameters:
            return parameters[key]
        try:return float(key)
        except ValueError:raise ValueError(f"Unresolved FBC bound parameter: {key}")

    reactions = {}
    for r in model.findall(f".//{{{SBML}}}listOfReactions/{{{SBML}}}reaction"):
        rid = r.get("id")
        sto = {}
        for sr in r.findall(f"{{{SBML}}}listOfReactants/{{{SBML}}}speciesReference"):
            sto[sr.get("species")] = sto.get(sr.get("species"), 0.0) - _num(sr.get("stoichiometry"), 1.0)
        for sr in r.findall(f"{{{SBML}}}listOfProducts/{{{SBML}}}speciesReference"):
            sto[sr.get("species")] = sto.get(sr.get("species"), 0.0) + _num(sr.get("stoichiometry"), 1.0)
        lb = bound_value(r, "lowerFluxBound", -1000.0 if r.get("reversible") == "true" else 0.0)
        ub = bound_value(r, "upperFluxBound", 1000.0)
        reactions[rid] = ReactionData(
            id=rid, name=r.get("name") or rid, lower_bound=lb, upper_bound=ub,
            stoichiometry=sto, subsystem=subsystem_by_rxn.get(rid, ""),
            gene_rule=gpr(r.find(f"{{{FBC}}}geneProductAssociation")) if r.find(f"{{{FBC}}}geneProductAssociation") is not None else "",
            annotations=_annotations(r),
        )

    objectives = {}
    direction = "max"
    lo = model.find(f"{{{FBC}}}listOfObjectives")
    if lo is not None:
        active = lo.get(f"{{{FBC}}}activeObjective")
        objs = lo.findall(f"{{{FBC}}}objective")
        obj = next((o for o in objs if o.get(f"{{{FBC}}}id") == active), objs[0] if objs else None)
        if obj is not None:
            direction = "min" if (obj.get(f"{{{FBC}}}type") or "maximize").lower().startswith("min") else "max"
            for fo in obj.findall(f".//{{{FBC}}}fluxObjective"):
                rid = fo.get(f"{{{FBC}}}reaction")
                objectives[rid] = _num(fo.get(f"{{{FBC}}}coefficient"), 1.0)

    gene_count = len(model.findall(f".//{{{FBC}}}geneProduct"))
    return ModelData(
        id=model.get("id") or path.stem, name=model.get("name") or model.get("id") or path.stem,
        metabolites=metabolites, reactions=reactions, compartments=compartments,
        objectives=objectives, objective_direction=direction, gene_count=gene_count, source_path=path, genes=genes,
    )


def _validate(m, path):
    import math
    from .genes import parse_rule, rule_genes
    # Reject duplicate IDs before dict construction can conceal them.
    if path.suffix.lower()=='.json':
        raw=json.loads(path.read_text())
        for k in ('metabolites','reactions','genes'):
            ids=[x['id'] for x in raw.get(k,[])]
            if len(ids)!=len(set(ids)): raise ValueError(f'Duplicate {k} IDs')
    else:
        tree=etree.parse(str(path), etree.XMLParser(resolve_entities=False,no_network=True))
        SBML=etree.QName(tree.getroot()).namespace
        for tag in ('species','reaction'):
            ids=[x.get('id') for x in tree.findall(f'.//{{{SBML}}}{tag}')]
            if len(ids)!=len(set(ids)): raise ValueError(f'Duplicate {tag} IDs')
    for r in m.reactions.values():
        if not math.isfinite(r.lower_bound) or not math.isfinite(r.upper_bound) or r.lower_bound>r.upper_bound:
            raise ValueError(f'Invalid bounds: {r.id}')
        for mid,c in r.stoichiometry.items():
            if mid not in m.metabolites or not math.isfinite(c): raise ValueError(f'Invalid stoichiometry: {r.id}/{mid}')
        if r.gene_rule:
            for gid in rule_genes(parse_rule(r.gene_rule)):
                if gid not in m.genes:
                    m.genes[gid]={'name':gid,'annotations':{},'inferred_from_gpr':True}
    m.gene_count=len(m.genes)
    if any(not r.gene_rule for r in m.reactions.values()):
        m.warnings.append('Reactions without GPR have unknown genetic control.')
    if any(rid not in m.reactions for rid in m.objectives): raise ValueError('Unknown model objective reaction')
    return m
