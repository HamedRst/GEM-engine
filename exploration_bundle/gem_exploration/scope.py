from __future__ import annotations
from typing import Any
import numpy as np
import pandas as pd
from .expressions import resolve_column


def _one(df: pd.DataFrame, f: dict[str, Any]) -> np.ndarray:
    col = resolve_column(df, str(f['column']))
    s = df[col]
    op = str(f.get('op', '=='))
    val = f.get('value')
    if op == '>=':
        return pd.to_numeric(s, errors='coerce').to_numpy(float) >= float(val)
    if op == '<=':
        return pd.to_numeric(s, errors='coerce').to_numpy(float) <= float(val)
    if op == '>':
        return pd.to_numeric(s, errors='coerce').to_numpy(float) > float(val)
    if op == '<':
        return pd.to_numeric(s, errors='coerce').to_numpy(float) < float(val)
    if op == '==':
        return s.astype(str).to_numpy() == str(val)
    if op == '!=':
        return s.astype(str).to_numpy() != str(val)
    if op == 'between':
        vals = f.get('values', val)
        if not isinstance(vals, (list, tuple)) or len(vals) != 2:
            raise ValueError("between filter requires exactly two values")
        lo, hi = vals
        x = pd.to_numeric(s, errors='coerce').to_numpy(float)
        return (x >= float(lo)) & (x <= float(hi))
    if op == 'in':
        vals = f.get('values', val)
        if not isinstance(vals, (list, tuple, set)):
            vals = [vals]
        return s.isin(vals).to_numpy()
    if op == 'finite':
        return np.isfinite(pd.to_numeric(s, errors='coerce').to_numpy(float))
    if op == 'true':
        return s.fillna(False).astype(bool).to_numpy()
    if op == 'false':
        return ~s.fillna(False).astype(bool).to_numpy()
    if op in {'top_quantile', 'bottom_quantile'}:
        x = pd.to_numeric(s, errors='coerce')
        q = float(val)
        if not (0 < q <= 1):
            raise ValueError(f"{op} value must be in (0,1]")
        th = float(x.quantile(q if op == 'bottom_quantile' else 1 - q))
        return (x <= th if op == 'bottom_quantile' else x >= th).to_numpy()
    raise ValueError(f'Unsupported filter op={op}')


def apply_filters(df: pd.DataFrame, filters: list[dict[str, Any]], logic: str = 'all') -> pd.DataFrame:
    filters = list(filters or [])
    if not filters:
        return df.copy()
    logic = str(logic or 'all').lower()
    if logic not in {'all', 'any'}:
        raise ValueError("filter logic must be all|any")
    masks = [_one(df, f) for f in filters]
    mask = masks[0].copy()
    for m in masks[1:]:
        mask = (mask & m) if logic == 'all' else (mask | m)
    return df.loc[mask].copy()


def apply_scope(df: pd.DataFrame, scope: dict[str, Any], variables: list[str]) -> tuple[pd.DataFrame, dict[str, Any]]:
    logic = str(scope.get('filter_logic', 'all') or 'all').lower()
    out = apply_filters(df, scope.get('filters', []) or [], logic=logic)
    info = {'after_filters': len(out), 'filter_logic': logic}
    loc = scope.get('locality') or {}
    mode = loc.get('mode', 'none')
    if mode == 'none':
        return out, info
    if len(out) == 0:
        return out, info
    X = out[variables].apply(pd.to_numeric, errors='coerce').to_numpy(float)
    lo = np.nanmin(df[variables].to_numpy(float), axis=0)
    hi = np.nanmax(df[variables].to_numpy(float), axis=0)
    span = np.where(hi > lo, hi - lo, 1.0)
    if mode == 'sample':
        sid = int(loc['sample_id'])
        hit = df[df['sample_id'] == sid]
        if hit.empty:
            raise KeyError(f'locality sample_id={sid} not found')
        center = hit.iloc[0][variables].to_numpy(float)
    else:
        point = loc.get('point') or {}
        center = np.array([float(point.get(v, point.get(v.replace('requested_medium__', ''), np.nan))) for v in variables])
        if not np.isfinite(center).all():
            raise ValueError('Point locality must define every resolved design variable.')
    dist = np.sqrt(np.nansum(((X - center) / span) ** 2, axis=1))
    radius = float(loc.get('radius', 0.10))
    out = out.loc[dist <= radius].copy()
    info.update({'locality_mode': mode, 'radius': radius, 'after_locality': len(out)})
    return out, info
