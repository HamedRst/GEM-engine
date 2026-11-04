from __future__ import annotations
from pathlib import Path
from typing import Any
import json
import hashlib
import yaml
import numpy as np
import pandas as pd
from .expressions import resolve_column

LONG_TABLES={
 'exchange_fluxes':('reaction_id','flux','exchange_flux__'),
 'reaction_fluxes':('reaction_id','flux','reaction_flux__'),
 'shadow_prices':('metabolite_id','shadow_price','shadow_price__'),
 'reduced_costs':('reaction_id','reduced_cost','reduced_cost__'),
}
PREFIX_TABLES={
 'requested_media':'requested_medium__',
 'applied_media':'applied_medium__',
 'condition_metadata':'condition__',
}

# Process-local cache used by the atlas runner. The source landscape is treated as
# immutable/read-only, so repeated exploration questions can safely share the loaded
# compact frame instead of rereading 50k chunk tables dozens of times.
_LANDSCAPE_CACHE: dict[tuple[str,str,bool,tuple[str,...]], tuple[pd.DataFrame,dict[str,Any]]] = {}


def _manifest_provenance(root: Path, source_files: list[Path]) -> tuple[list[dict[str, Any]], str | None]:
    """Reuse GEM Engine chunk-manifest hashes when available.

    This avoids rereading potentially multi-GB flux tables merely to hash them again
    for every exploration question. Files absent from a manifest are marked for
    direct hashing by the run layer.
    """
    by_path: dict[Path, dict[str, Any]] = {}
    manifest_files: list[Path] = []
    for mp in sorted(root.rglob("chunk_*.json")):
        try:
            payload = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        files = payload.get("files")
        if not isinstance(files, dict):
            continue
        manifest_files.append(mp)
        for info in files.values():
            if not isinstance(info, dict) or not info.get("path"):
                continue
            fp = (root / str(info["path"])).resolve()
            by_path[fp] = {
                "path": str(fp),
                "size_bytes": info.get("size_bytes"),
                "sha256": info.get("sha256"),
                "hash_source": str(mp),
            }
    records = []
    for sp in source_files:
        rp = sp.resolve()
        rec = by_path.get(rp)
        if rec is None:
            rec = {"path": str(rp), "size_bytes": rp.stat().st_size if rp.exists() else None, "sha256": None, "hash_source": None}
        records.append(rec)
    composite = None
    hashed = sorted((r["path"], r.get("sha256")) for r in records if r.get("sha256"))
    if hashed:
        h = hashlib.sha256()
        for path, digest in hashed:
            h.update(path.encode("utf-8")); h.update(b"\0"); h.update(str(digest).encode("ascii")); h.update(b"\n")
        composite = h.hexdigest()
    return records, composite

def _read(path: Path) -> pd.DataFrame:
    s=path.suffix.lower()
    if s=='.csv': return pd.read_csv(path)
    if s in {'.parquet','.pq'}: return pd.read_parquet(path)
    raise ValueError(f'Unsupported table type: {path}')

def _files(root: Path, table: str) -> list[Path]:
    pats=[f'{table}_chunk_*.parquet',f'{table}_chunk_*.csv']
    hits=[]
    for p in pats: hits += list(root.rglob(p))
    return sorted(set(hits))

def _concat(files:list[Path]) -> pd.DataFrame:
    if not files: return pd.DataFrame()
    return pd.concat([_read(p) for p in files],ignore_index=True,sort=False)

def _concat_long_selected(files:list[Path], entity:str, value:str, selected_ids:set[str]) -> pd.DataFrame:
    """Read only requested identities from long chunk tables.

    This prevents a single flux question from expanding a 50k landscape into the
    complete reaction-by-sample matrix. CSV chunks are streamed; parquet reads only
    the three required columns before filtering.
    """
    frames=[]
    usecols=['sample_id',entity,value]
    for fp in files:
        if fp.suffix.lower() in {'.parquet','.pq'}:
            t=pd.read_parquet(fp,columns=usecols)
            t=t[t[entity].astype(str).isin(selected_ids)]
            if not t.empty: frames.append(t)
        else:
            for ch in pd.read_csv(fp,usecols=usecols,chunksize=500000):
                z=ch[ch[entity].astype(str).isin(selected_ids)]
                if not z.empty: frames.append(z)
    if not frames: return pd.DataFrame(columns=usecols)
    return pd.concat(frames,ignore_index=True,sort=False)

def _wide_long(frame:pd.DataFrame, entity:str, value:str, prefix:str)->pd.DataFrame:
    if frame.empty or 'sample_id' not in frame.columns: return pd.DataFrame()
    if entity in frame.columns and value in frame.columns:
        x=frame.pivot_table(index='sample_id',columns=entity,values=value,aggfunc='last').reset_index()
    else: x=frame.copy()
    return x.rename(columns={c:f'{prefix}{c}' for c in x.columns if c!='sample_id'})


def _canonical_prepared_problem_config(root: Path) -> Path | None:
    """Return sibling prepared/<case>/problem_config.yaml for canonical benchmark runs."""
    try:
        if root.parent.name == "runs" and root.parent.parent.name == "inputs":
            benchmark_root = root.parent.parent.parent
            p = benchmark_root / "prepared" / root.name / "problem_config.yaml"
            if p.exists():
                return p.resolve()
    except Exception:
        pass
    return None


def _enrich_deterministic_resource_quantities(root: Path, base: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    """Add deterministic resource-cost quantities when the source benchmark defines them.

    This is zero-FBA enrichment. It is intentionally fail-closed: quantities are
    added only when a canonical sibling problem_config.yaml explicitly provides
    the feature set, bounds, and the recognized normalized-resource-cost formula.
    """
    if {"resource_cost", "resource_efficiency"}.issubset(base.columns):
        return base, {}, []

    p = _canonical_prepared_problem_config(root)
    if p is None:
        return base, {}, []
    try:
        raw = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        problem = raw.get("problem", raw)
    except Exception:
        return base, {}, []

    formula = str(
        problem.get("resource_cost_formula")
        or problem.get("normalized_resource_cost_formula")
        or ""
    ).strip().lower()
    efficiency_formula = str(problem.get("resource_efficiency_formula") or "").strip().lower()
    recognized_cost = (
        "mean_j" in formula
        and "x_j" in formula
        and "l_j" in formula
        and "u_j" in formula
    ) or ("weighted mean_j" in formula and "x_j" in formula)
    recognized_eff = ("1 -" in efficiency_formula or "1-" in efficiency_formula or not efficiency_formula)
    if not (recognized_cost and recognized_eff):
        return base, {}, []

    features = [str(x) for x in (problem.get("features") or [])]
    bounds = problem.get("bounds") or {}
    if not features or not isinstance(bounds, dict):
        return base, {}, []

    normalized = []
    resolved_columns: list[str] = []
    for rid in features:
        col = f"requested_medium__{rid}" if f"requested_medium__{rid}" in base.columns else (rid if rid in base.columns else None)
        if col is None or rid not in bounds:
            return base, {}, []
        try:
            lo, hi = [float(x) for x in bounds[rid]]
        except Exception:
            return base, {}, []
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return base, {}, []
        x = pd.to_numeric(base[col], errors="coerce").to_numpy(float)
        z = (x - lo) / (hi - lo)
        normalized.append(z)
        resolved_columns.append(col)

    Z = np.column_stack(normalized)
    cost = np.nanmean(Z, axis=1)
    out = base.copy()
    out["resource_cost"] = cost
    out["normalized_resource_cost"] = cost
    out["resource_efficiency"] = 1.0 - cost
    meta = {
        "deterministic_resource_enrichment": {
            "status": "applied",
            "source_problem_config": str(p),
            "resource_cost_formula": problem.get("resource_cost_formula") or problem.get("normalized_resource_cost_formula"),
            "resource_efficiency_formula": problem.get("resource_efficiency_formula") or "1 - resource_cost",
            "features": features,
            "resolved_columns": resolved_columns,
            "new_fba_calls": 0,
        }
    }
    return out, meta, [p]


def load_gem_engine_run(root:Path, include_fluxes:bool=True, selected_long_columns:list[str]|None=None) -> tuple[pd.DataFrame,dict[str,Any]]:
    result_files=_files(root,'results')
    if not result_files:
        # exported whole dataset produced by v1.0.4 exporter
        for candidate in [root/'whole_dataset.csv', root/'whole_dataset.parquet']:
            if candidate.exists():
                df=_read(candidate); return df, {'source_mode':'whole_dataset','source_files':[str(candidate)]}
        raise FileNotFoundError(f'No GEM Engine results chunks or whole_dataset found under {root}')
    base=_concat(result_files)
    if 'sample_id' not in base.columns: raise RuntimeError('Results table lacks sample_id.')
    base=base.drop_duplicates('sample_id',keep='last')
    source_files=list(result_files)
    for table,prefix in PREFIX_TABLES.items():
        fs=_files(root,table)
        if not fs: continue
        t=_concat(fs); source_files += fs
        if 'sample_id' not in t.columns: continue
        t=t.drop(columns=[c for c in ['chunk_id'] if c in t.columns],errors='ignore')
        t=t.drop_duplicates('sample_id',keep='last')
        t=t.rename(columns={c:f'{prefix}{c}' for c in t.columns if c!='sample_id'})
        base=base.merge(t,on='sample_id',how='left',validate='one_to_one')

    # DESIGN-CACHE INPUT RECOVERY PATCH v1
    # Some canonical GEM Engine 50k archives keep the exact sampled design matrix
    # in design_cache/*.npy rather than duplicating requested-media values in chunk
    # tables.  The Atlas must recover those inputs before any exploration question
    # can identify its design variables.  Recovery is fail-closed: feature IDs,
    # matrix dimensions, row count, and sample-id ordering must all agree.
    if not any(str(c).startswith('requested_medium__') for c in base.columns):
        pre_path = root/'metadata'/'preflight.json'
        cache_hits = sorted((root/'design_cache').glob('*.npy')) if (root/'design_cache').exists() else []
        if pre_path.exists() and cache_hits:
            try:
                pre = json.loads(pre_path.read_text(encoding='utf-8'))
            except Exception as exc:
                raise RuntimeError(f'Could not parse preflight metadata for design-cache recovery: {pre_path}: {exc}') from exc
            feature_ids = [str(x) for x in (pre.get('variable_medium_reaction_ids') or [])]
            if not feature_ids:
                raise RuntimeError(
                    f'No requested-media chunks and preflight has no variable_medium_reaction_ids: {pre_path}'
                )
            # Prefer the one matrix whose shape matches the full run and feature count.
            matches=[]
            for cp in cache_hits:
                try:
                    arr=np.load(cp,mmap_mode='r')
                except Exception:
                    continue
                if arr.ndim==2 and arr.shape[0]==len(base) and arr.shape[1]==len(feature_ids):
                    matches.append((cp,arr))
            if len(matches)!=1:
                shapes=[]
                for cp in cache_hits[:20]:
                    try:
                        shapes.append(f'{cp.name}:{tuple(np.load(cp,mmap_mode="r").shape)}')
                    except Exception:
                        shapes.append(f'{cp.name}:UNREADABLE')
                raise RuntimeError(
                    'Design-cache recovery requires exactly one matrix matching '
                    f'rows={len(base)} and variables={len(feature_ids)}; found {len(matches)}. '
                    f'Candidates: {shapes}'
                )
            cache_path, X = matches[0]
            # GEM Engine design-cache row order is the simulation/sample order.  Only
            # accept the common contiguous integer sample-id contracts so alignment is
            # independently checkable rather than assumed.
            sid=pd.to_numeric(base['sample_id'],errors='coerce')
            if sid.isna().any() or sid.nunique()!=len(base):
                raise RuntimeError('Cannot prove design-cache alignment: sample_id is nonnumeric or nonunique.')
            sid_int=sid.astype(np.int64).to_numpy()
            n=len(base)
            order=np.argsort(sid_int)
            sorted_ids=sid_int[order]
            if np.array_equal(sorted_ids,np.arange(n,dtype=np.int64)):
                row_index=sorted_ids
            elif np.array_equal(sorted_ids,np.arange(1,n+1,dtype=np.int64)):
                row_index=sorted_ids-1
            else:
                raise RuntimeError(
                    'Cannot prove design-cache alignment: sample_id is not contiguous 0..N-1 or 1..N.'
                )
            # Reorder base by sample_id, then attach X rows in the corresponding cache order.
            base=base.iloc[order].reset_index(drop=True)
            for j,rid in enumerate(feature_ids):
                base[f'requested_medium__{rid}']=np.asarray(X[row_index,j],dtype=float)
            source_files += [cache_path, pre_path]

    base, derived_meta, derived_files = _enrich_deterministic_resource_quantities(root, base)
    if derived_files:
        source_files += derived_files

    if include_fluxes:
        requested=set(str(x) for x in (selected_long_columns or []))
        for table,(entity,value,prefix) in LONG_TABLES.items():
            fs=_files(root,table)
            if not fs: continue
            ids={x[len(prefix):] for x in requested if x.startswith(prefix)}
            if selected_long_columns is not None and not ids:
                continue
            source_files += fs
            long_frame=_concat_long_selected(fs,entity,value,ids) if selected_long_columns is not None else _concat(fs)
            t=_wide_long(long_frame,entity,value,prefix)
            if not t.empty: base=base.merge(t,on='sample_id',how='left',validate='one_to_one')
    source_records, fingerprint = _manifest_provenance(root, source_files)
    meta={'source_mode':'gem_engine_run','source_files':[str(p) for p in source_files],
          'source_records':source_records,'landscape_fingerprint':fingerprint}
    if derived_meta:
        meta.update(derived_meta)
    for p in [root/'metadata'/'run_manifest.json',root/'metadata'/'config_snapshot.yaml',root/'catalogs'/'medium_space.csv']:
        if p.exists(): meta[p.name]=str(p)
    return base.sort_values('sample_id').reset_index(drop=True),meta

def load_landscape(path:str|Path, mode:str='auto', include_fluxes:bool=True, selected_long_columns:list[str]|None=None) -> tuple[pd.DataFrame,dict[str,Any]]:
    p=Path(path).expanduser().resolve()
    if not p.exists(): raise FileNotFoundError(f'Landscape path does not exist: {p}')
    selected_key=tuple(sorted(str(x) for x in (selected_long_columns or [])))
    key=(str(p),str(mode),bool(include_fluxes),selected_key)
    if key in _LANDSCAPE_CACHE:
        return _LANDSCAPE_CACHE[key]
    if p.is_file():
        out=(_read(p), {'source_mode':'table','source_files':[str(p)], 'source_records':[{'path':str(p),'size_bytes':p.stat().st_size,'sha256':None,'hash_source':None}]})
    else:
        out=load_gem_engine_run(p,include_fluxes=include_fluxes,selected_long_columns=selected_long_columns)
    _LANDSCAPE_CACHE[key]=out
    return out

def clear_landscape_cache() -> None:
    _LANDSCAPE_CACHE.clear()

def infer_design_variables(df:pd.DataFrame, root:Path|None=None) -> list[str]:
    cols=[c for c in df.columns if str(c).startswith('requested_medium__')]
    if cols: return cols
    if root is not None:
        for p in [root/'catalogs'/'medium_space.csv',root/'catalogs'/'medium_space.parquet']:
            if p.exists():
                t=_read(p)
                if 'reaction_id' in t.columns:
                    ids=t['reaction_id'].astype(str).tolist(); out=[]
                    for rid in ids:
                        for c in [rid,f'requested_medium__{rid}']:
                            if c in df.columns: out.append(c); break
                    if out: return out
    exclude_prefix=('exchange_flux__','reaction_flux__','shadow_price__','reduced_cost__','condition__','applied_medium__')
    exclude={'sample_id','run_number','total_runs','chunk_id','feasible','runtime_seconds','objective_value','maximum_growth','biomass_flux','product_flux','growth_floor'}
    out=[]
    for c in df.columns:
        if c in exclude or str(c).startswith(exclude_prefix): continue
        if pd.api.types.is_numeric_dtype(df[c]) and df[c].nunique(dropna=True)>2: out.append(str(c))
    return out

def resolve_variables(df:pd.DataFrame, varspec:dict[str,Any], inferred:list[str]) -> list[str]:
    mode=str(varspec.get('mode','all_design'))
    if mode in {'all','all_design'}: vals=list(inferred)
    elif mode=='list': vals=[resolve_column(df,v) for v in varspec.get('values',[])]
    else: raise ValueError(f'Unsupported variables.mode={mode}')
    inc=varspec.get('include',[]) or []; exc=set(varspec.get('exclude',[]) or [])
    for v in inc:
        c=resolve_column(df,v)
        if c not in vals: vals.append(c)
    exc_res=set()
    for v in exc:
        try: exc_res.add(resolve_column(df,v))
        except Exception: exc_res.add(v)
    vals=[v for v in vals if v not in exc_res]
    if not vals: raise RuntimeError('No design variables resolved.')
    return vals
