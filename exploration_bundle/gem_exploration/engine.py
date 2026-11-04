from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from .config import normalized_problem
from .data import infer_design_variables, load_landscape, resolve_variables
from .expressions import materialize_targets_and_derived
from .operators import OPERATOR_BACKENDS, REGISTRY
from .plotting import render
from .visualization_planner import plan_visualization
from .scope import apply_scope
from .utils import ensure_clean_dir, package_versions, sha256_file, sha256_text, utc_now, write_json


VERSION = "0.5.0"


def _enforce_backend(spec: dict[str, Any]) -> str:
    op = spec["operator"]["name"]
    required = OPERATOR_BACKENDS[op]
    requested = str((spec.get("backend") or {}).get("mode", "auto"))
    if requested == "auto":
        return required
    if requested == "landscape" and required in {"surrogate_assisted", "hybrid"}:
        raise RuntimeError(
            f"Operator {op!r} requires surrogate assistance ({required}) but backend.mode=landscape was forced. "
            "Use backend.mode=auto or surrogate."
        )
    if requested == "surrogate" and required == "landscape":
        raise RuntimeError(
            f"Operator {op!r} is defined as a direct landscape analysis and does not use a surrogate backend. "
            "Use backend.mode=auto or landscape."
        )
    return required


def prepare_problem(
    raw: dict[str, Any],
    landscape_override: str | None = None,
    seed: int = 20260822,
) -> dict[str, Any]:
    """Resolve a problem against a landscape without executing the operator."""
    spec = normalized_problem(raw)
    ppath = Path(raw["_problem_path"]).resolve()
    lcfg = spec.get("landscape", {}) or {}
    lpath = landscape_override or lcfg.get("path")
    if not lpath:
        raise ValueError("Landscape path is required either in YAML landscape.path or via --landscape.")

    df, meta = load_landscape(
        lpath,
        mode=lcfg.get("mode", "auto"),
        include_fluxes=bool(lcfg.get("include_fluxes", True)),
        selected_long_columns=list(lcfg.get("selected_long_columns") or []) if lcfg.get("selected_long_columns") is not None else None,
    )
    # Pass a stable source identity into the surrogate cache.  This is internal
    # metadata only; it never changes the scientific problem definition.
    spec.setdefault("surrogate", {})["_landscape_fingerprint"] = (
        meta.get("landscape_fingerprint") or str(Path(lpath).expanduser().resolve())
    )
    df, created = materialize_targets_and_derived(df, spec)
    root = Path(lpath).resolve() if Path(lpath).is_dir() else None
    inferred = infer_design_variables(df, root)
    variables = resolve_variables(df, spec["variables"], inferred)
    scoped, scope_info = apply_scope(df, spec.get("scope", {}), variables)

    # A generic query can be meaningful with a small number of matching rows. All
    # statistical/model-based operators require a safer minimum support gate.
    op = spec["operator"]["name"]
    minimum = 1 if op == "query" else max(20, 2 * len(variables))
    if len(scoped) < minimum:
        raise RuntimeError(
            f"Scope leaves too few rows for {op}: {len(scoped)} < required minimum {minimum}. "
            "Broaden the scope or generate/refine the landscape."
        )
    actual_backend = _enforce_backend(spec)
    return {
        "spec": spec,
        "problem_path": ppath,
        "landscape_path": Path(lpath).expanduser().resolve(),
        "df": df,
        "scoped": scoped,
        "meta": meta,
        "variables": variables,
        "created_columns": created,
        "scope_info": scope_info,
        "operator": op,
        "actual_backend": actual_backend,
        "seed": seed,
    }


def plan_problem(raw: dict[str, Any], landscape_override: str | None = None, seed: int = 20260822) -> dict[str, Any]:
    prepared = prepare_problem(raw, landscape_override, seed)
    spec = prepared["spec"]
    return {
        "status": "PASS",
        "problem": spec["name"],
        "operator": prepared["operator"],
        "evidence_backend": prepared["actual_backend"],
        "landscape": str(prepared["landscape_path"]),
        "rows_total": len(prepared["df"]),
        "rows_in_scope": len(prepared["scoped"]),
        "resolved_variables": prepared["variables"],
        "targets": spec.get("targets", []),
        "created_columns": prepared["created_columns"],
        "scope": prepared["scope_info"],
        "new_fba_calls": 0,
        "note": "This exploration package is read-only with respect to the source landscape. Surrogate-assisted operators interpolate learned landscape quantities; they do not create exact FBA evidence.",
    }


def _source_records(meta: dict[str, Any]) -> list[dict[str, Any]]:
    existing = meta.get("source_records") or []
    by_path = {str(Path(r["path"]).resolve()): dict(r) for r in existing if r.get("path")}
    records = []
    for source in meta.get("source_files", []):
        sp = Path(source).resolve()
        rec = by_path.get(str(sp), {"path": str(sp)})
        try:
            rec.setdefault("size_bytes", sp.stat().st_size)
            # Reuse authoritative chunk-manifest hashes when available. Only hash
            # directly when the source has no recorded digest.
            if not rec.get("sha256"):
                rec["sha256"] = sha256_file(sp)
                rec["hash_source"] = "direct"
        except Exception as exc:
            rec["error"] = str(exc)
        records.append(rec)
    return records


def run_problem(
    raw: dict[str, Any],
    landscape_override: str | None = None,
    output_override: str | None = None,
    overwrite: bool = False,
    seed: int = 20260822,
) -> Path:
    prepared = prepare_problem(raw, landscape_override, seed)
    spec = prepared["spec"]
    ppath = prepared["problem_path"]
    df = prepared["df"]
    scoped = prepared["scoped"]
    meta = prepared["meta"]
    variables = prepared["variables"]
    created = prepared["created_columns"]
    scope_info = prepared["scope_info"]
    op = prepared["operator"]
    actual_backend = prepared["actual_backend"]
    lpath = prepared["landscape_path"]

    out = Path(output_override or spec.get("output_dir") or f'results/{spec["name"]}').expanduser().resolve()
    ensure_clean_dir(out, overwrite)

    ctx = {
        "seed": seed,
        "all_rows": len(df),
        "scope_info": scope_info,
        "source_meta": meta,
        "evidence_backend": actual_backend,
    }
    result = REGISTRY[op](scoped, spec, variables, ctx)

    # Compile the scientific question and publication visualization from the resolved
    # problem definition. This is deliberately separate from the operator so the same
    # analysis can be communicated consistently across species and future operators.
    viz_spec = plan_visualization(result, spec, variables, scoped, df, actual_backend)

    for name, table in result.tables.items():
        table.to_csv(out / f"{name}.csv", index=False)
    import yaml
    (out / "visualization_spec.yaml").write_text(yaml.safe_dump(viz_spec, sort_keys=False), encoding="utf-8")
    write_json(out / "visualization_spec.json", viz_spec)
    figs = render(result, scoped, out, spec, viz_spec=viz_spec, full_df=df)

    problem_text = ppath.read_text(encoding="utf-8")
    shutil.copy2(ppath, out / "problem.yaml")
    records = _source_records(meta)
    pd.DataFrame(records).to_csv(out / "source_files.csv", index=False)

    summary = dict(result.summary)
    summary.update(
        {
            "name": spec["name"],
            "operator": op,
            "evidence_backend": actual_backend,
            "created_at_utc": utc_now(),
            "landscape_path": str(lpath),
            "rows_total": len(df),
            "rows_in_scope": len(scoped),
            "resolved_design_variables": variables,
            "created_columns": created,
            "scope": scope_info,
            "figures": figs,
            "warnings": result.warnings,
            "scientific_question": viz_spec.get("question", {}).get("text"),
            "visualization_family": viz_spec.get("visualization", {}).get("family"),
            "visualization_views": (viz_spec.get("visualization", {}).get("views") or []),
            "visualization_spec": str(out / "visualization_spec.yaml"),
            "new_fba_calls": 0,
        }
    )
    summary.setdefault("variables", variables)
    write_json(out / "summary.json", summary)

    manifest = {
        "engine": "GEM Generalized Exploration Engine",
        "version": VERSION,
        "created_at_utc": utc_now(),
        "problem_sha256": sha256_text(problem_text),
        "problem_path": str(ppath),
        "source": meta,
        "source_file_records": records,
        "resolved_variables": variables,
        "rows_total": len(df),
        "rows_in_scope": len(scoped),
        "evidence_backend": actual_backend,
        "packages": package_versions(),
        "seed": seed,
        "new_fba_calls": 0,
        "scientific_guardrails": [
            "surrogate outputs are predictions/interpolations, not exact FBA",
            "tree ensemble dispersion is not a calibrated confidence interval",
            "global_sensitivity uses held-out model-based permutation importance, not classical Sobol indices",
            "sampled-landscape fragility/robustness is not automatically equivalent to an exact perturbation experiment",
            "clusters are descriptive unless quality and independent biological evidence support discrete regimes",
            "all filters/targets/variables are frozen in copied problem.yaml",
        ],
    }
    write_json(out / "run_manifest.json", manifest)

    report = [
        "# GEM Exploration Report",
        "",
        f'**Problem:** {spec["name"]}',
        f"**Operator:** `{op}`",
        f"**Evidence backend:** `{actual_backend}`",
        f"**Landscape:** `{lpath}`",
        f"**Rows:** {len(scoped):,} in scope / {len(df):,} total",
        f"**Design variables:** {len(variables)}",
        "**New FBA calls:** 0",
        "",
        "## Scientific question",
        str(viz_spec.get("question", {}).get("text", "")),
        "",
        "## Formal problem / evidence",
        f"**Visualization family:** `{viz_spec.get('visualization', {}).get('family', '-')}`",
        f"**Evidence:** {viz_spec.get('evidence', {}).get('label', actual_backend)}",
        f"**Resolved scope:** {len(scoped):,} / {len(df):,} designs",
        "",
        "## Result summary",
        "```json",
        json.dumps(summary, indent=2, default=str),
        "```",
        "",
        "## Scientific interpretation guardrails",
        "- Landscape analyses are conditional on the simulation policy used to generate the source landscape.",
        "- Surrogate-assisted quantities are explicitly prediction-based and are not reported as new exact FBA solutions.",
        "- Model-based permutation importance is not labeled as a classical Sobol sensitivity index.",
        "- Tree-ensemble dispersion is a heuristic support proxy, not a calibrated uncertainty interval.",
        "- The exact YAML problem definition and source hashes are saved with the run.",
    ]
    if result.warnings:
        report += ["", "## Operator-specific warnings", *[f"- {w}" for w in result.warnings]]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return out
