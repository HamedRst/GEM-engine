from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json

import pandas as pd

from gem_exploration.data import infer_design_variables, load_landscape


@dataclass
class LandscapeInspection:
    path: str
    rows: int
    columns: int
    source_mode: str
    feasible_rows: int | None
    design_variables: list[str]
    target_options: list[dict[str, str]]
    numeric_columns: list[str]
    meta: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ModelInspection:
    path: str
    model_id: str
    reactions: int
    metabolites: int
    genes: int
    reaction_labels: dict[str, str]
    reaction_bounds: dict[str, tuple[float, float]]
    gene_labels: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _alias_for_column(c: str) -> str:
    if c.startswith("requested_medium__"):
        return f"requested_medium:{c[len('requested_medium__') :]}"
    if c.startswith("applied_medium__"):
        return f"applied_medium:{c[len('applied_medium__') :]}"
    if c.startswith("exchange_flux__"):
        return f"exchange:{c[len('exchange_flux__') :]}"
    if c.startswith("reaction_flux__"):
        return f"reaction:{c[len('reaction_flux__') :]}"
    if c.startswith("shadow_price__"):
        return f"shadow:{c[len('shadow_price__') :]}"
    if c.startswith("reduced_cost__"):
        return f"reduced_cost:{c[len('reduced_cost__') :]}"
    return f"result:{c}"


def _kind_for_column(c: str) -> str:
    if c.startswith("requested_medium__"):
        return "design variable"
    if c.startswith("applied_medium__"):
        return "applied medium"
    if c.startswith("exchange_flux__"):
        return "stored exchange flux"
    if c.startswith("reaction_flux__"):
        return "stored reaction flux"
    if c.startswith("shadow_price__"):
        return "stored shadow price"
    if c.startswith("reduced_cost__"):
        return "stored reduced cost"
    return "result / stored quantity"


def _files(root: Path, table: str) -> list[Path]:
    hits: list[Path] = []
    for pat in (f"{table}_chunk_*.parquet", f"{table}_chunk_*.csv"):
        hits.extend(root.rglob(pat))
    return sorted(set(hits))


def _read_unique_ids(path: Path, entity: str, max_ids: int = 20000) -> list[str]:
    """Read only the identity column from one representative long-table chunk.

    GEM Engine long-table chunks repeat the same reaction/metabolite catalog for
    each sample. Reading one chunk is enough to discover the available identities
    without materializing the full 50k × reaction matrix.
    """
    try:
        if path.suffix.lower() in {".parquet", ".pq"}:
            s = pd.read_parquet(path, columns=[entity])[entity]
        else:
            s = pd.read_csv(path, usecols=[entity])[entity]
    except Exception:
        return []
    vals = pd.Series(s).dropna().astype(str).drop_duplicates().tolist()
    return vals[:max_ids]


def discover_stored_long_catalog(root: Path) -> list[dict[str, str]]:
    specs = [
        ("exchange_fluxes", "reaction_id", "exchange", "exchange_flux__", "stored exchange flux"),
        ("reaction_fluxes", "reaction_id", "reaction", "reaction_flux__", "stored reaction flux"),
        ("shadow_prices", "metabolite_id", "shadow", "shadow_price__", "stored shadow price"),
        ("reduced_costs", "reaction_id", "reduced_cost", "reduced_cost__", "stored reduced cost"),
    ]
    out: list[dict[str, str]] = []
    for table, entity, alias_prefix, col_prefix, kind in specs:
        fs = _files(root, table)
        if not fs:
            continue
        for rid in _read_unique_ids(fs[0], entity):
            out.append(
                {
                    "column": f"{col_prefix}{rid}",
                    "alias": f"{alias_prefix}:{rid}",
                    "kind": kind,
                    "source_table": table,
                    "entity_id": rid,
                }
            )
    return out


def _read_case_metadata(root: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {}
    for rel in ("metadata/preflight.json", "metadata/model_inference.json", "metadata/run_manifest.json"):
        p = root / rel
        if not p.exists():
            continue
        try:
            meta[rel.replace("/", "_")] = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            pass
    return meta


def inspect_landscape(path: str, include_fluxes: bool = False, discover_flux_catalog: bool = True) -> tuple[LandscapeInspection, pd.DataFrame]:
    p = Path(path).expanduser().resolve()
    # Initial GUI inspection intentionally loads the compact base landscape only.
    # Stored long-table fluxes are catalogued lazily and materialized only if a
    # question actually asks for one of them.
    df, meta = load_landscape(p, include_fluxes=include_fluxes)
    root = p if p.is_dir() else None
    variables = infer_design_variables(df, root)
    numeric = [str(c) for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

    target_options: list[dict[str, str]] = []
    technical = {"sample_id", "run_number", "total_runs", "chunk_id", "fba_solves", "solver_wall_seconds", "runtime_seconds"}
    for c in numeric:
        if c in technical:
            continue
        alias = _alias_for_column(c)
        target_options.append({"column": c, "alias": alias, "kind": _kind_for_column(c)})

    flux_catalog: list[dict[str, str]] = []
    if root is not None and discover_flux_catalog:
        flux_catalog = discover_stored_long_catalog(root)

    feasible_rows = None
    if "feasible" in df.columns:
        try:
            feasible_rows = int(pd.Series(df["feasible"]).fillna(False).astype(bool).sum())
        except Exception:
            feasible_rows = None

    clean_meta = {k: v for k, v in meta.items() if k not in {"source_records"}}
    clean_meta["columns"] = [str(c) for c in df.columns]
    clean_meta["stored_long_catalog"] = flux_catalog
    if root is not None:
        clean_meta["case_metadata"] = _read_case_metadata(root)

    info = LandscapeInspection(
        path=str(p),
        rows=int(len(df)),
        columns=int(len(df.columns)),
        source_mode=str(meta.get("source_mode", "unknown")),
        feasible_rows=feasible_rows,
        design_variables=list(variables),
        target_options=target_options,
        numeric_columns=numeric,
        meta=clean_meta,
    )
    return info, df


def load_cobra_model(path: str):
    p = Path(path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"GEM model file does not exist: {p}")
    try:
        from cobra.io import load_json_model, load_matlab_model, read_sbml_model
    except Exception as exc:
        raise RuntimeError("COBRApy is required to inspect GEM model files") from exc

    suffix = p.suffix.lower()
    if suffix == ".json":
        return load_json_model(str(p))
    if suffix == ".mat":
        return load_matlab_model(str(p))
    if suffix in {".xml", ".sbml"}:
        return read_sbml_model(str(p))
    raise ValueError("Supported GEM model formats are SBML (.xml/.sbml), COBRA JSON (.json), and MATLAB (.mat)")


def inspect_model(path: str) -> tuple[ModelInspection, Any]:
    p = Path(path).expanduser().resolve()
    model = load_cobra_model(str(p))
    labels: dict[str, str] = {}
    bounds: dict[str, tuple[float, float]] = {}
    for r in model.reactions:
        label = str(getattr(r, "name", "") or r.id)
        labels[str(r.id)] = label
        bounds[str(r.id)] = (float(r.lower_bound), float(r.upper_bound))
    genes = {str(g.id): str(getattr(g, "name", "") or g.id) for g in model.genes}
    info = ModelInspection(
        path=str(p),
        model_id=str(getattr(model, "id", "") or p.stem),
        reactions=int(len(model.reactions)),
        metabolites=int(len(model.metabolites)),
        genes=int(len(model.genes)),
        reaction_labels=labels,
        reaction_bounds=bounds,
        gene_labels=genes,
    )
    return info, model


def reaction_id_from_design_variable(name: str) -> str:
    for prefix in ("requested_medium__", "applied_medium__"):
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def semantic_variable_labels(variables: list[str], model_info: ModelInspection | None) -> dict[str, str]:
    if model_info is None:
        return {}
    out: dict[str, str] = {}
    for v in variables:
        rid = reaction_id_from_design_variable(v)
        if rid in model_info.reaction_labels:
            out[v] = model_info.reaction_labels[rid]
            out[rid] = model_info.reaction_labels[rid]
    return out


def model_coverage(variables: list[str], model_info: ModelInspection | None) -> tuple[int, int]:
    if model_info is None:
        return 0, len(variables)
    matched = 0
    for v in variables:
        if reaction_id_from_design_variable(v) in model_info.reaction_labels:
            matched += 1
    return matched, len(variables)


def infer_matching_model_path(landscape_path: str) -> str | None:
    """Find the matching model next to a canonical benchmark landscape.

    This intentionally ignores stale absolute paths saved on another machine and
    prefers a same-package `inputs/models/<case_key>.*` file.
    """
    p = Path(landscape_path).expanduser().resolve()
    if not p.exists():
        return None
    case_key = p.name if p.is_dir() else p.stem
    supported = {".xml", ".sbml", ".json", ".mat"}

    # Canonical benchmark layout: <package>/inputs/runs/<case> and <package>/inputs/models/<case>.*
    for parent in [p, *p.parents[:6]]:
        models = parent / "inputs" / "models"
        if not models.is_dir():
            continue
        hits = [x for x in models.glob(case_key + ".*") if x.suffix.lower() in supported]
        if len(hits) == 1:
            return str(hits[0].resolve())

    # If metadata records a stale source path, use only its basename to search locally.
    if p.is_dir():
        pre = p / "metadata" / "preflight.json"
        if pre.exists():
            try:
                old = str(json.loads(pre.read_text(encoding="utf-8")).get("model_path") or "")
            except Exception:
                old = ""
            base = Path(old).name if old else ""
            if base:
                for parent in [p, *p.parents[:6]]:
                    hits = [x for x in parent.rglob(base) if x.is_file()]
                    if len(hits) == 1:
                        return str(hits[0].resolve())
    return None
