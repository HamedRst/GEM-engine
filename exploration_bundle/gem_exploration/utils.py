from __future__ import annotations
import hashlib, json, math, os, platform, sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_file(path: Path, block_size: int = 1024*1024) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(block_size), b''):
            h.update(b)
    return h.hexdigest()

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def jsonable(x: Any) -> Any:
    if isinstance(x, Path): return str(x)
    if isinstance(x, (np.floating, np.integer)): return x.item()
    if isinstance(x, np.ndarray): return x.tolist()
    if isinstance(x, pd.Series): return {str(k): jsonable(v) for k,v in x.to_dict().items()}
    if isinstance(x, pd.DataFrame): return x.to_dict(orient='records')
    if isinstance(x, dict): return {str(k): jsonable(v) for k,v in x.items()}
    if isinstance(x, (list,tuple)): return [jsonable(v) for v in x]
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)): return None
    return x

def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, sort_keys=True), encoding='utf-8')

def package_versions() -> dict[str,str]:
    out={'python':sys.version.split()[0], 'platform':platform.platform()}
    for name in ['numpy','pandas','scipy','sklearn','matplotlib','yaml','pyarrow']:
        try:
            m=__import__(name)
            out[name]=str(getattr(m,'__version__','unknown'))
        except Exception:
            out[name]='unavailable'
    return out

def ensure_clean_dir(path: Path, overwrite: bool=False) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise FileExistsError(f'Output directory is not empty: {path}. Use --overwrite or choose another path.')
        import shutil; shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
