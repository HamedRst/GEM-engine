from __future__ import annotations

from pathlib import Path
from typing import Any

SUPPORTED_MODEL_SUFFIXES = {".json", ".xml", ".sbml", ".mat", ".yml", ".yaml"}


def resolve_from_config(config_path: str | Path, value: str | Path) -> Path:
    cfg_path = Path(config_path).expanduser().resolve()
    p = Path(value).expanduser()
    if not p.is_absolute():
        p = cfg_path.parent / p
    return p.resolve()


def resolve_model_path(config_path: str | Path, cfg: dict[str, Any], case_key: str, case: dict[str, Any]) -> Path:
    """Resolve a portable server-side model path.

    Resolution order:
      1. explicit case.model_path
      2. case.model_glob inside paths.input_models_root
      3. original basename from the run's config_snapshot.yaml if present in input_models_root
    """
    if case.get("model_path"):
        p = resolve_from_config(config_path, case["model_path"])
        if p.exists():
            return p
        raise FileNotFoundError(f"Configured model_path does not exist for {case_key}: {p}")

    models_root = resolve_from_config(config_path, cfg["paths"]["input_models_root"])
    models_root.mkdir(parents=True, exist_ok=True)
    pattern = case.get("model_glob", f"{case_key}.*")
    hits = [p for p in sorted(models_root.glob(pattern)) if p.is_file() and p.suffix.lower() in SUPPORTED_MODEL_SUFFIXES]
    if len(hits) == 1:
        return hits[0].resolve()
    if len(hits) > 1:
        raise RuntimeError(f"Multiple model files match {pattern!r} for {case_key}: {[p.name for p in hits]}")

    run_dir = resolve_from_config(config_path, case["run_dir"])
    snap = run_dir / "metadata" / "config_snapshot.yaml"
    if snap.exists():
        try:
            import yaml
            raw = yaml.safe_load(snap.read_text(encoding="utf-8")) or {}
            original = Path(str(raw.get("model", {}).get("path", ""))).name
            if original:
                q = models_root / original
                if q.exists():
                    return q.resolve()
        except Exception:
            pass

    raise FileNotFoundError(
        f"No model file found for {case_key}. Put exactly one supported model file in {models_root} "
        f"matching {pattern!r}, or set cases.{case_key}.model_path in the study config."
    )
