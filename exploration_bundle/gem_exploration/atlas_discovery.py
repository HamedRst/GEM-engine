from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import json
import re

import yaml


SPECIES_ORDER = [
    "yeast",
    "pputida",
    "cglutamicum",
    "cho",
    "ecoli_antiepex",
]

SPECIES_LABELS = {
    "yeast": "S. cerevisiae (Yeast)",
    "pputida": "P. putida",
    "cglutamicum": "C. glutamicum",
    "cho": "CHO",
    "ecoli_antiepex": "E. coli antiEpEX-scFv",
}

SPECIES_TOKENS = {
    "yeast": ["yeast", "saccharomyces", "cerevisiae", "yeast-gem"],
    "pputida": ["pputida", "p_putida", "p.putida", "putida", "ijn1463"],
    "cglutamicum": ["cglutamicum", "c_glutamicum", "c.glutamicum", "glutamicum", "icgb21fr"],
    "cho": ["cho", "icho2441", "chinese hamster"],
    "ecoli_antiepex": ["ecoli", "e_coli", "e.coli", "ijo1366", "antiepex", "scfv"],
}

SUPPORTED_MODEL_SUFFIXES = {".json", ".xml", ".sbml", ".mat", ".yml", ".yaml"}


@dataclass
class LandscapeRecord:
    species: str
    label: str
    path: str
    model_path: str | None
    rows: int | None
    optimization_mode: str | None
    biomass_reaction_id: str | None
    product_reaction_id: str | None
    model_id: str | None
    model_fingerprint: str | None
    config_hash: str | None
    score: float
    discovery_notes: list[str]


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _n_simulations(cfg: dict, pre: dict, man: dict) -> int | None:
    candidates = [
        pre.get("n_simulations"),
        (cfg.get("simulation") or {}).get("n_simulations"),
        man.get("n_simulations"),
        (man.get("simulation") or {}).get("n_simulations"),
        (man.get("run_summary") or {}).get("n_simulations"),
    ]
    for x in candidates:
        try:
            n = int(x)
            if n > 0:
                return n
        except Exception:
            pass
    return None


def _species_match(text: str) -> tuple[str | None, dict[str, int]]:
    s = text.lower()
    scores: dict[str, int] = {}
    for species, toks in SPECIES_TOKENS.items():
        score = 0
        for tok in toks:
            if tok.lower() in s:
                score += max(1, len(tok))
        scores[species] = score
    best = max(scores, key=scores.get) if scores else None
    if best is None or scores[best] <= 0:
        return None, scores
    ordered = sorted(scores.values(), reverse=True)
    if len(ordered) > 1 and ordered[0] == ordered[1]:
        return None, scores
    return best, scores


def _path_from_cfg(run_dir: Path, cfg: dict) -> Path | None:
    model_cfg = cfg.get("model") or {}
    raw = (
        model_cfg.get("path")
        or model_cfg.get("model_path")
        or model_cfg.get("file")
        or model_cfg.get("model_file")
    )
    if not raw:
        return None
    p = Path(str(raw)).expanduser()
    if p.is_absolute() and p.exists():
        return p.resolve()
    for base in [run_dir, run_dir.parent, run_dir.parent.parent]:
        q = (base / p).resolve()
        if q.exists():
            return q
    return p if p.is_absolute() else None


def _candidate_model_basenames(cfg: dict, man: dict) -> list[str]:
    vals = []
    model_cfg = cfg.get("model") or {}
    for raw in [
        model_cfg.get("path"), model_cfg.get("model_path"), model_cfg.get("file"), model_cfg.get("model_file"),
        man.get("model_path"), (man.get("model_summary") or {}).get("path"),
    ]:
        if raw:
            vals.append(Path(str(raw)).name)
    return list(dict.fromkeys(vals))


def resolve_model_path(run_dir: Path, workspace_root: Path, cfg: dict, man: dict) -> tuple[Path | None, list[str]]:
    notes: list[str] = []
    direct = _path_from_cfg(run_dir, cfg)
    if direct is not None and direct.exists():
        return direct.resolve(), ["model path recovered directly from config"]

    basenames = _candidate_model_basenames(cfg, man)
    search_roots = [run_dir, run_dir.parent, run_dir.parent.parent, workspace_root]
    hits: list[Path] = []
    seen = set()
    for root in search_roots:
        if not root.exists():
            continue
        # First prefer expected model folders to avoid a huge full-tree search.
        for sub in [root / "models", root / "inputs" / "models", root / "model", root]:
            if not sub.exists():
                continue
            for base in basenames:
                q = sub / base
                if q.exists() and q.is_file():
                    rp = q.resolve()
                    if str(rp) not in seen:
                        seen.add(str(rp)); hits.append(rp)
        if hits:
            break

    if not hits and basenames and workspace_root.exists():
        for base in basenames:
            for q in workspace_root.rglob(base):
                if q.is_file() and q.suffix.lower() in SUPPORTED_MODEL_SUFFIXES:
                    rp = q.resolve()
                    if str(rp) not in seen:
                        seen.add(str(rp)); hits.append(rp)

    if len(hits) == 1:
        notes.append("model path relocated by basename search")
        return hits[0], notes
    if len(hits) > 1:
        notes.append("multiple model files matched; exact validation will fail closed until one is selected")
        return None, notes + [str(p) for p in hits[:10]]

    # Last resort: scan the run neighborhood for a model that contains the species/model id in its name.
    mid = str((man.get("model_summary") or {}).get("model_id") or man.get("model_id") or "").lower()
    local_hits = []
    for root in [run_dir.parent, run_dir.parent.parent]:
        if not root.exists():
            continue
        for q in root.rglob("*"):
            if not q.is_file() or q.suffix.lower() not in SUPPORTED_MODEL_SUFFIXES:
                continue
            name = q.name.lower()
            if mid and mid in name:
                local_hits.append(q.resolve())
    local_hits = list(dict.fromkeys(map(str, local_hits)))
    if len(local_hits) == 1:
        return Path(local_hits[0]), ["model path recovered by model-id neighborhood search"]
    notes.append("model file could not be resolved automatically")
    return None, notes


def _has_landscape_tables(run_dir: Path) -> bool:
    candidates = [
        run_dir / "whole_dataset.csv",
        run_dir / "whole_dataset.parquet",
        run_dir / "chunks" / "results",
    ]
    if any(p.exists() for p in candidates):
        return True
    for pat in ["results_chunk_*.csv", "results_chunk_*.parquet"]:
        if next(run_dir.rglob(pat), None) is not None:
            return True
    return False


def _candidate_from_run(run_dir: Path, workspace_root: Path) -> LandscapeRecord | None:
    meta = run_dir / "metadata"
    cfg_path = meta / "config_snapshot.yaml"
    pre_path = meta / "preflight.json"
    man_path = meta / "run_manifest.json"
    if not (cfg_path.exists() and pre_path.exists() and man_path.exists()):
        return None
    cfg, pre, man = _load_yaml(cfg_path), _load_json(pre_path), _load_json(man_path)
    rows = _n_simulations(cfg, pre, man)
    model_cfg = cfg.get("model") or {}
    opt = cfg.get("optimization") or {}
    biomass = model_cfg.get("biomass_reaction_id") or model_cfg.get("objective_reaction_id")
    product = model_cfg.get("product_reaction_id")
    mode = opt.get("mode")
    model_id = (man.get("model_summary") or {}).get("model_id") or man.get("model_id")
    text = " ".join([
        str(run_dir), str(model_id or ""), str(product or ""), str(model_cfg.get("path") or ""),
        str((man.get("model_summary") or {}).get("name") or ""),
    ])
    species, match_scores = _species_match(text)
    if species is None:
        return None

    notes: list[str] = []
    score = float(match_scores.get(species, 0))
    if rows == 50000:
        score += 200
        notes.append("exactly 50,000 simulations")
    elif rows is not None:
        score += min(100, rows / 500)
        notes.append(f"reported simulations={rows}")
    if str(mode).lower() == "growth_constrained_product":
        score += 50; notes.append("growth_constrained_product simulation policy")
    if biomass:
        score += 15
    if product:
        score += 25
    if _has_landscape_tables(run_dir):
        score += 50; notes.append("landscape result tables detected")
    if next((run_dir / "design_cache").glob("*.npy"), None):
        score += 20; notes.append("design cache detected")

    model_path, model_notes = resolve_model_path(run_dir, workspace_root, cfg, man)
    notes.extend(model_notes)
    if model_path is not None:
        score += 25

    return LandscapeRecord(
        species=species,
        label=SPECIES_LABELS[species],
        path=str(run_dir.resolve()),
        model_path=str(model_path.resolve()) if model_path is not None else None,
        rows=rows,
        optimization_mode=str(mode) if mode is not None else None,
        biomass_reaction_id=str(biomass) if biomass else None,
        product_reaction_id=str(product) if product else None,
        model_id=str(model_id) if model_id else None,
        model_fingerprint=str(man.get("model_fingerprint")) if man.get("model_fingerprint") else None,
        config_hash=str(man.get("config_hash")) if man.get("config_hash") else None,
        score=score,
        discovery_notes=notes,
    )


def scan_candidates(workspace_root: str | Path) -> list[LandscapeRecord]:
    root = Path(workspace_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    records: list[LandscapeRecord] = []
    seen = set()
    # Scanning config snapshots is dramatically cheaper and safer than looking for
    # arbitrary result files. Each valid GEM Engine run has this metadata contract.
    for cfg_path in root.rglob("config_snapshot.yaml"):
        if cfg_path.parent.name != "metadata":
            continue
        run_dir = cfg_path.parent.parent
        key = str(run_dir.resolve())
        if key in seen:
            continue
        seen.add(key)
        rec = _candidate_from_run(run_dir, root)
        if rec is not None:
            records.append(rec)
    return records


def _manual_record(species: str, entry: dict[str, Any], workspace_root: Path) -> LandscapeRecord:
    run_dir = Path(str(entry["path"])).expanduser()
    if not run_dir.is_absolute():
        run_dir = (workspace_root / run_dir).resolve()
    else:
        run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Configured landscape for {species} does not exist: {run_dir}")
    rec = _candidate_from_run(run_dir, workspace_root)
    if rec is None:
        raise RuntimeError(f"Configured path does not look like a supported GEM Engine landscape: {run_dir}")
    if rec.species != species:
        rec.discovery_notes.append(f"manual species override: inferred {rec.species}, configured {species}")
        rec.species = species; rec.label = SPECIES_LABELS[species]
    raw_model = entry.get("model_path")
    if raw_model and str(raw_model).upper() != "AUTO":
        mp = Path(str(raw_model)).expanduser()
        if not mp.is_absolute():
            mp = (workspace_root / mp).resolve()
        if not mp.exists():
            raise FileNotFoundError(f"Configured model for {species} does not exist: {mp}")
        rec.model_path = str(mp.resolve())
        rec.discovery_notes.append("manual model path override")
    return rec


def resolve_five_landscapes(workspace_root: str | Path, config_path: str | Path | None = None) -> dict[str, LandscapeRecord]:
    root = Path(workspace_root).expanduser().resolve()
    manual: dict[str, Any] = {}
    if config_path is not None:
        p = Path(config_path).expanduser().resolve()
        if p.exists():
            raw = _load_yaml(p)
            manual = raw.get("landscapes") or {}

    all_candidates: list[LandscapeRecord] | None = None
    resolved: dict[str, LandscapeRecord] = {}
    errors = []
    for species in SPECIES_ORDER:
        entry = manual.get(species) or {}
        raw_path = entry.get("path") if isinstance(entry, dict) else None
        if raw_path and str(raw_path).upper() != "AUTO":
            try:
                resolved[species] = _manual_record(species, entry, root)
            except Exception as exc:
                errors.append(f"{species}: {exc}")
            continue

        if all_candidates is None:
            all_candidates = scan_candidates(root)
        candidates = [r for r in all_candidates if r.species == species]
        candidates.sort(key=lambda r: r.score, reverse=True)
        if not candidates:
            errors.append(f"{species}: no compatible 50k landscape discovered under {root}")
            continue
        best = candidates[0]
        # Fail closed when two runs look equally authoritative. This avoids silently
        # choosing an old pilot/copy simply because it happened to be scanned first.
        if len(candidates) > 1 and abs(best.score - candidates[1].score) < 1e-9 and best.path != candidates[1].path:
            errors.append(
                f"{species}: ambiguous landscape discovery. Top candidates have equal score: "
                f"{best.path} ; {candidates[1].path}. Set study/landscapes.yaml explicitly."
            )
            continue
        resolved[species] = best

    if errors:
        detail = "\n".join(f"  - {e}" for e in errors)
        raise RuntimeError(f"Could not resolve the required five landscapes:\n{detail}")
    return resolved


def write_resolved(records: dict[str, LandscapeRecord], path: str | Path) -> Path:
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "gem_exploration_atlas_landscapes_v0.2.0",
        "landscapes": {s: asdict(records[s]) for s in SPECIES_ORDER},
    }
    p.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return p


def print_candidate_table(records: list[LandscapeRecord]) -> str:
    lines = ["species\tscore\trows\tmode\tmodel\tpath"]
    for r in sorted(records, key=lambda x: (SPECIES_ORDER.index(x.species), -x.score)):
        lines.append(f"{r.species}\t{r.score:.1f}\t{r.rows}\t{r.optimization_mode}\t{r.model_path or 'UNRESOLVED'}\t{r.path}")
    return "\n".join(lines)
