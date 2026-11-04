from __future__ import annotations
from pathlib import Path

SUPPORTED_MODEL_SUFFIXES = {".json", ".xml", ".sbml", ".mat", ".yml", ".yaml"}

def resolve_from_config(config_path, value):
    cfg = Path(config_path).expanduser().resolve()
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = cfg.parent / p
    return p.resolve()

def resolve_model_path(config_path, cfg, case_key, case):
    cfg_path = Path(config_path).expanduser().resolve()

    explicit = case.get("model_path")
    if explicit:
        p = resolve_from_config(cfg_path, explicit)
        if not p.is_file():
            raise FileNotFoundError(f"{case_key}: configured model_path does not exist: {p}")
        if p.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
            raise ValueError(f"{case_key}: unsupported model format: {p}")
        return p

    model_glob = str(case.get("model_glob", "")).strip()
    if not model_glob:
        raise KeyError(f"{case_key}: neither model_path nor model_glob is configured")

    model_root = resolve_from_config(
        cfg_path,
        cfg.get("paths", {}).get("input_models_root", "./inputs/models"),
    )

    hits = sorted(
        p.resolve()
        for p in model_root.glob(model_glob)
        if p.is_file() and p.suffix.lower() in SUPPORTED_MODEL_SUFFIXES
    )

    if not hits:
        raise FileNotFoundError(
            f"{case_key}: no supported model matched {model_glob!r} under {model_root}"
        )
    if len(hits) != 1:
        raise RuntimeError(
            f"{case_key}: expected exactly one model matching {model_glob!r}; found {len(hits)}: "
            + ", ".join(str(p) for p in hits)
        )

    return hits[0]
