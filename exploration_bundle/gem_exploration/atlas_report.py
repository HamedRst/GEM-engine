from __future__ import annotations

import json
import re
import textwrap
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from .atlas_discovery import SPECIES_ORDER, SPECIES_LABELS
from .atlas_problem_factory import EXECUTED_PROBLEM_FILES
from .plotting import PUBLICATION_RC
from .utils import utc_now


def _load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _short(v: Any) -> str:
    if isinstance(v, float):
        if not np.isfinite(v): return "–"
        if v != 0 and (abs(v) < 1e-3 or abs(v) >= 1e4): return f"{v:.2e}"
        return f"{v:.4g}"
    if isinstance(v, (np.integer, int)):
        return f"{int(v):,}"
    if isinstance(v, bool):
        return "Yes" if v else "No"
    if v is None:
        return "–"
    if isinstance(v, (dict, list, tuple)):
        return ""
    s = str(v)
    return s if len(s) <= 52 else s[:49] + "..."


def _add_header(fig: plt.Figure, title: str, subtitle: str | None = None) -> None:
    fig.text(0.055, 0.972, title, ha="left", va="top", fontsize=15.5, fontweight="bold")
    if subtitle:
        fig.text(0.055, 0.944, subtitle, ha="left", va="top", fontsize=8.6)
    fig.text(0.945, 0.972, "GEM Exploration Atlas", ha="right", va="top", fontsize=8.3)


def _problem_key(problem_file: str) -> str:
    return problem_file.replace("_TEMPLATE", "").replace(".yaml", "")


def _resolved_problem(results_root: Path, species: str, problem_file: str) -> dict[str, Any]:
    filename = problem_file.replace("_TEMPLATE", "")
    return _load_yaml(results_root / species / "resolved_problems" / filename)


def _viz_spec(results_root: Path, species: str, problem_file: str) -> dict[str, Any]:
    key = _problem_key(problem_file)
    return _load_yaml(results_root / species / "problems" / key / "visualization_spec.yaml")


def _best_question_spec(results_root: Path, problem_file: str) -> tuple[dict[str, Any], dict[str, Any]]:
    for s in SPECIES_ORDER:
        v = _viz_spec(results_root, s, problem_file)
        p = _resolved_problem(results_root, s, problem_file)
        if v:
            return v, p
    for s in SPECIES_ORDER:
        p = _resolved_problem(results_root, s, problem_file)
        if p:
            return {}, p
    return {}, {}


def _question_number(problem_file: str) -> str:
    m = re.match(r"(\d+)_", Path(problem_file).name)
    return m.group(1) if m else ""


def _formal_definition(viz: dict, problem: dict) -> list[tuple[str, str]]:
    p = viz.get("problem") or {}
    e = viz.get("evidence") or {}
    vis = viz.get("visualization") or {}
    targets = list((p.get("target_labels") or {}).values())
    if not targets:
        targets = [str(t.get("name")) for t in problem.get("targets", []) if t.get("name")]
    variables = p.get("variables") or []
    rows = p.get("rows_in_scope")
    total = p.get("rows_total")
    out = [
        ("Analysis", str(p.get("operator") or (problem.get("operator") or {}).get("name") or "–")),
        ("Target", ", ".join(targets) if targets else "No single target"),
        ("Design variables", f"{len(variables)} resolved variables" if variables else str((problem.get("variables") or {}).get("mode", "resolved from landscape"))),
        ("Scope", f"{int(rows):,} / {int(total):,} landscape designs" if rows is not None and total is not None else "Resolved from problem YAML"),
        ("Evidence", str(e.get("label") or (problem.get("backend") or {}).get("mode") or "auto")),
        ("Primary visualization", str(vis.get("family") or "auto")),
    ]
    return out


def _cover(pdf: PdfPages, run_summary: dict[str, Any]) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.text(0.08, 0.83, "GEM Exploration Atlas", fontsize=26, fontweight="bold")
    fig.text(0.08, 0.78, "Question-driven generalized landscape exploration", fontsize=14)
    fig.text(0.08, 0.71, "Scientific question → resolved evidence → publication visualization", fontsize=10.8)
    stats = [
        f"Species completed: {run_summary.get('landscapes_completed', 0)} / 5",
        f"Exploration analyses supported: {run_summary.get('problems_passed', 0)} / {run_summary.get('problems_total', 150)}",
        f"Unsupported / failed analyses: {run_summary.get('problems_failed', 0)}",
        f"New FBA calls during exploration: {run_summary.get('new_fba_calls_during_exploration', 0)}",
        f"Created: {run_summary.get('created_at_utc', utc_now())}",
    ]
    fig.text(0.08, 0.59, "\n".join(stats), fontsize=10.2, linespacing=1.55)
    guard = (
        "Interpretation rule: direct-landscape analyses describe stored exact simulation outcomes. "
        "Surrogate-assisted analyses interpolate those exact outcomes and are shown only after their configured QC gate passes. "
        "Unsupported surrogate inference is displayed explicitly rather than replaced by a misleading figure."
    )
    fig.text(0.08, 0.37, textwrap.fill(guard, 88), fontsize=9.3, linespacing=1.5)
    fig.text(0.08, 0.18, "Every supported question exports a standalone 600-dpi PNG and editable PDF/SVG plus its visualization_spec.yaml, resolved problem.yaml, numeric tables, summary, and provenance.", fontsize=8.7, wrap=True)
    plt.axis("off"); pdf.savefig(fig, bbox_inches="tight"); plt.close(fig)


def _methods_page(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11)); _add_header(fig, "Question and visualization contract")
    rows = [
        ("1. Problem YAML", "Defines target(s), design variables, scope, operator, backend, thresholds/anchors and optional semantic labels."),
        ("2. Analysis operator", "Produces structured numerical results and evidence tables. The operator does not choose its own default chart."),
        ("3. Visualization planner", "Compiles the scientific question and answer geometry from the resolved YAML + result, then selects a reusable scientific visualization family."),
        ("4. Publication renderer", "Uses one universal style contract: semantic labels, readable typography, units when provided, uncertainty/support when available, vector PDF/SVG and 600-dpi PNG."),
        ("5. Fail-closed QC", "If a surrogate-dependent question fails its QC threshold, no surrogate conclusion is plotted. The report explains why that inference is unsupported."),
    ]
    y=.86
    for title,body in rows:
        fig.text(.08,y,title,fontsize=11,fontweight="bold"); y-=.034
        fig.text(.08,y,textwrap.fill(body,105),fontsize=9.2,linespacing=1.4); y-=.13
    fig.text(.08,.12,"The plotting layer is therefore part of the scientific method contract rather than a post-processing decoration.",fontsize=9.0,wrap=True)
    plt.axis("off");pdf.savefig(fig,bbox_inches="tight");plt.close(fig)


def _landscape_page(pdf: PdfPages, results_root: Path) -> None:
    fig=plt.figure(figsize=(8.5,11));_add_header(fig,"Landscape overview")
    rows=[]
    for s in SPECIES_ORDER:
        p=_load_yaml(results_root/s/"resolved_problems"/"LANDSCAPE_PROFILE.yaml")
        if not p:rows.append([SPECIES_LABELS[s],"–","–","–","–"])
        else:
            pr=p.get("product_range") or [None,None]
            rows.append([p.get("label",SPECIES_LABELS[s]),f"{int(p.get('rows',0)):,}",f"{int(p.get('feasible_rows',0)):,}",str(len(p.get("design_variables") or [])),f"{_short(pr[0])} to {_short(pr[1])}"])
    ax=fig.add_axes([.06,.40,.88,.40]);ax.axis("off")
    tab=ax.table(cellText=rows,colLabels=["Landscape","Rows","Feasible","Design dim.","Product range"],loc="center",cellLoc="left",colLoc="left")
    tab.auto_set_font_size(False);tab.set_fontsize(8.2);tab.scale(1,1.55)
    fig.text(.07,.30,textwrap.fill("All landscapes are analyzed by the same generalized operators. Species-specific differences are represented as resolved YAML semantics, thresholds, anchors and design dimensionality rather than custom plotting code.",105),fontsize=9.0)
    plt.axis("off");pdf.savefig(fig,bbox_inches="tight");plt.close(fig)


def _failure_message(error: str) -> tuple[str,str]:
    s=str(error or "Unsupported analysis")
    m=re.search(r"R2=([0-9.eE+-]+)\s*<\s*([0-9.eE+-]+)",s)
    if m:
        return "Unsupported surrogate inference", f"Surrogate QC failed: R² = {m.group(1)} below required {m.group(2)}. No surrogate-based scientific conclusion is shown."
    return "Analysis unsupported / failed", s


def _metric_lines(summary: dict[str,Any], viz: dict, max_items: int=4) -> list[str]:
    keys=list(viz.get("answer_metrics") or [])
    if not keys:
        keys=["n_selected","fraction_selected","feasible_fraction","selected_k","silhouette","n_pareto","fraction_pareto","matched_rows","surrogate_prediction","nearest_exact_distance","q95_nn_distance"]
    out=[]
    for k in keys:
        if k not in summary:continue
        txt=_short(summary.get(k))
        if txt and txt!="–":out.append(f"{k.replace('_',' ')}: {txt}")
        if len(out)>=max_items:break
    return out


def _species_panel(fig: plt.Figure, results_root: Path, species: str, problem_file: str, box: tuple[float,float,float,float]) -> None:
    key=_problem_key(problem_file);qdir=results_root/species/"problems"/key
    status=_load_json(qdir/"ATLAS_STATUS.json");summary=_load_json(qdir/"summary.json");viz=_load_yaml(qdir/"visualization_spec.yaml")
    x,y,w,h=box
    fig.text(x,y+h+.012,SPECIES_LABELS[species],fontsize=10.2,fontweight="bold")
    ax=fig.add_axes([x,y,w,h]);ax.axis("off")
    png=qdir/"figures"/"main.png"
    if status.get("status")=="PASS" and png.exists():
        try:ax.imshow(plt.imread(png));ax.axis("off")
        except Exception:ax.text(.5,.5,"Figure unreadable",ha="center",va="center")
    else:
        head,body=_failure_message(status.get("error","No supported result"))
        ax.text(.5,.60,head,ha="center",va="center",fontsize=11,fontweight="bold")
        ax.text(.5,.43,textwrap.fill(body,62),ha="center",va="center",fontsize=8.2)
        ax.text(.5,.24,"Fail-closed: no replacement inference was generated.",ha="center",va="center",fontsize=7.8)
    lines=_metric_lines(summary,viz,4)
    if lines:
        fig.text(x,y-.012,"  ·  ".join(lines),fontsize=7.2,va="top",wrap=True)


def _question_pages(pdf: PdfPages, results_root: Path, problem_file: str) -> None:
    viz,problem=_best_question_spec(results_root,problem_file)
    qnum=_question_number(problem_file)
    q=(viz.get("question") or {}).get("text") or str(problem.get("name") or _problem_key(problem_file)).replace("_"," ").title()
    title=(viz.get("question") or {}).get("title") or q.rstrip("?")
    formal=_formal_definition(viz,problem)
    groups=[SPECIES_ORDER[0:2],SPECIES_ORDER[2:4],SPECIES_ORDER[4:5]]
    for pi,group in enumerate(groups,1):
        fig=plt.figure(figsize=(8.5,11))
        head=f"Question {qnum}: {textwrap.shorten(str(title), width=88, placeholder='…')}" if qnum else str(title)
        _add_header(fig,head,f"Part {pi}/{len(groups)} · same resolved question across species")
        fig.text(.07,.895,"Scientific question",fontsize=9.2,fontweight="bold")
        fig.text(.07,.865,textwrap.fill(str(q),105),fontsize=10.3,linespacing=1.35)
        y=.79
        cols=[(.07,.54),(.55,.94)]
        for (label,value),(x0,x1) in zip(formal[:3],cols*2):
            pass
        # Compact formal definition in two columns.
        left=formal[:3];right=formal[3:]
        for i,(lab,val) in enumerate(left):
            fig.text(.07,y-i*.037,lab+":",fontsize=7.8,fontweight="bold");fig.text(.20,y-i*.037,textwrap.shorten(str(val),55,placeholder="…"),fontsize=7.8)
        for i,(lab,val) in enumerate(right):
            fig.text(.52,y-i*.037,lab+":",fontsize=7.8,fontweight="bold");fig.text(.67,y-i*.037,textwrap.shorten(str(val),48,placeholder="…"),fontsize=7.8)
        purpose=str((viz.get("visualization") or {}).get("purpose") or "")
        if purpose:fig.text(.07,.675,textwrap.fill("Figure intent: "+purpose,110),fontsize=7.8)

        if len(group)==2:
            _species_panel(fig,results_root,group[0],problem_file,(.07,.365,.86,.265))
            _species_panel(fig,results_root,group[1],problem_file,(.07,.055,.86,.265))
        else:
            _species_panel(fig,results_root,group[0],problem_file,(.07,.29,.86,.34))
            fig.text(.07,.17,"Interpretation rule",fontsize=8.5,fontweight="bold")
            ev=str((viz.get("evidence") or {}).get("label") or "Resolved evidence backend")
            fig.text(.07,.135,textwrap.fill(f"Evidence: {ev}. Read the figure together with the saved visualization_spec.yaml and result tables. Surrogate-assisted figures represent validated interpolation, not new exact FBA evidence.",108),fontsize=8.0)
        pdf.savefig(fig,bbox_inches="tight");plt.close(fig)


def _validation_pages(pdf: PdfPages, results_root: Path) -> None:
    available=[];skipped=[]
    for s in SPECIES_ORDER:
        summ=_load_json(results_root/s/"exact_validation"/"validation_summary.json")
        if summ and summ.get("status") not in {None,"NOT RUN","SKIPPED"}:available.append((s,summ))
        else:skipped.append(s)
    if not available:
        fig=plt.figure(figsize=(8.5,11));_add_header(fig,"Exact-FBA validation")
        fig.text(.08,.76,"Validation was not run in this campaign",fontsize=15,fontweight="bold")
        fig.text(.08,.69,textwrap.fill("The exploration run was intentionally executed without the exact-validation stage. Direct landscape analyses remain descriptions of stored exact-FBA points; surrogate-assisted analyses remain subject to their internal QC gates. A later run with model files can add archived replay and targeted new exact-FBA challenges without redefining the 30 exploration questions.",95),fontsize=10,linespacing=1.5)
        plt.axis("off");pdf.savefig(fig,bbox_inches="tight");plt.close(fig);return
    for species,summ in available:
        vdir=results_root/species/"exact_validation"
        fig=plt.figure(figsize=(11,8.5));_add_header(fig,f"Exact-FBA validation — {SPECIES_LABELS[species]}",f"Status: {summ.get('status','–')}")
        imgs=[vdir/"figures"/"validation_growth.png",vdir/"figures"/"validation_product.png",vdir/"figures"/"validation_local_direction.png"]
        positions=[(.04,.42,.29,.41),(.355,.42,.29,.41),(.67,.42,.29,.41)]
        for p,pos in zip(imgs,positions):
            ax=fig.add_axes(pos);ax.axis("off")
            if p.exists():ax.imshow(plt.imread(p));ax.axis("off")
            else:ax.text(.5,.5,"Not available",ha="center",va="center")
        g=summ.get("growth_exact_challenge") or {};pr=summ.get("product_exact_challenge") or {}
        lines=[f"Archive replay points: {summ.get('archive_replay_points','–')}",f"New surrogate challenge points: {summ.get('surrogate_challenge_points','–')}",f"Growth nRMSE: {_short(g.get('nrmse_robust_range'))}",f"Product nRMSE: {_short(pr.get('nrmse_robust_range'))}",f"Local product sign agreement: {_short(summ.get('local_product_direction_sign_agreement'))}"]
        fig.text(.06,.28,"\n".join(lines),fontsize=9.0,linespacing=1.45)
        pdf.savefig(fig,bbox_inches="tight");plt.close(fig)


def build_combined_report(results_root: str | Path, package_root: str | Path, output_pdf: str | Path, run_summary: dict[str,Any] | None = None) -> Path:
    root=Path(results_root).resolve();out=Path(output_pdf).resolve();out.parent.mkdir(parents=True,exist_ok=True)
    run_summary=run_summary or _load_json(root/"RUN_SUMMARY.json")
    with plt.rc_context(PUBLICATION_RC),PdfPages(out) as pdf:
        _cover(pdf,run_summary);_methods_page(pdf);_landscape_page(pdf,root)
        for pf in EXECUTED_PROBLEM_FILES:_question_pages(pdf,root,pf)
        _validation_pages(pdf,root)
        fig=plt.figure(figsize=(8.5,11));_add_header(fig,"Reproducibility and output index")
        txt=("Every supported species/question directory contains the resolved problem.yaml, visualization_spec.yaml/json, structured result CSVs, summary.json, run_manifest.json, source provenance and the primary publication figure in PNG/PDF/SVG. The visualization spec records the scientific question, evidence backend, selected visual family, semantic labels and guardrails so the figure can be regenerated deterministically from the problem/result contract.")
        fig.text(.08,.80,textwrap.fill(txt,105),fontsize=10,linespacing=1.5)
        fig.text(.08,.62,"The combined report is an automatically generated question book; the standalone vector figures remain the preferred manuscript assets.",fontsize=9.0,wrap=True)
        plt.axis("off");pdf.savefig(fig,bbox_inches="tight");plt.close(fig)
    return out
