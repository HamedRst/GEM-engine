from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from nicegui import run, ui

from gem_exploration.config import load_problem
from gem_exploration.engine import plan_problem, run_problem

from .calculator import (
    EXPLORE_EXAMPLES,
    PUBLIC_EXPLORE_HELP,
    CalculatorCompilation,
    CalculatorRequest,
    SemanticRegistry,
    active_fragment,
    autocomplete_entries,
    build_calculator_line,
    build_registry,
    compile_calculator,
    insert_suggestion,
    parse_calculator_line,
)
from .inspectors import LandscapeInspection, ModelInspection, inspect_landscape, inspect_model, model_coverage, infer_matching_model_path
from .problem_builder import sanitize_name

APP_NAME = "GEM Exploration Calculator"
VERSION = "0.5.0"
DEFAULT_WORKSPACE = str(Path.home() / "Desktop" / "GEM" / "GEM_Exploration_Calculator_Workspace")


@dataclass
class State:
    landscape_path: str = ""
    model_path: str = ""
    workspace: str = DEFAULT_WORKSPACE
    landscape_info: LandscapeInspection | None = None
    landscape_df: pd.DataFrame | None = None
    model_info: ModelInspection | None = None
    model: Any = None
    registry: SemanticRegistry | None = None

    name: str = "exploration"
    observe: str = "product"
    explore: str = "auto"
    variables: str = "glucose, oxygen"
    conditions: str = "none"
    calculator_line: str = ""
    active_field: str = "variables"

    min_r2: float = 0.50
    min_auc: float = 0.65
    n_estimators: int = 300
    cv_folds: int = 5

    compilation: CalculatorCompilation | None = None
    plan: dict[str, Any] | None = None
    last_output: str | None = None
    last_error: str | None = None


state = State()
state.calculator_line = build_calculator_line(state.observe, state.explore, state.variables, state.conditions)


def _png_data_uri(path: Path) -> str:
    b = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b}"


def _open_folder(path: str) -> None:
    try:
        subprocess.Popen(["open", path])
    except Exception:
        pass


def _osascript(script: str) -> str | None:
    if sys.platform != "darwin":
        return None
    p = subprocess.run(["osascript", "-e", script], capture_output=True, text=True)
    if p.returncode != 0:
        return None
    out = p.stdout.strip()
    return out or None


def _choose_folder_native(prompt: str) -> str | None:
    safe = str(prompt).replace('"', '\"')
    return _osascript(f'POSIX path of (choose folder with prompt "{safe}")')


def _choose_file_native(prompt: str) -> str | None:
    safe = str(prompt).replace('"', '\"')
    return _osascript(f'POSIX path of (choose file with prompt "{safe}")')


def _set_field(field: str, value: Any) -> None:
    setattr(state, field, "" if value is None else str(value))
    state.calculator_line = build_calculator_line(state.observe, state.explore, state.variables, state.conditions)
    state.compilation = None
    state.plan = None
    state.last_error = None
    calculator_line_panel.refresh()
    suggestion_panel.refresh()
    interpretation_panel.refresh()
    advanced_preview.refresh()


def _set_active(field: str) -> None:
    state.active_field = field
    suggestion_panel.refresh()


def _apply_suggestion(field: str, value: str) -> None:
    current = getattr(state, field)
    updated = insert_suggestion(current, value, field)
    setattr(state, field, updated)
    state.active_field = field
    state.calculator_line = build_calculator_line(state.observe, state.explore, state.variables, state.conditions)
    calculator_fields.refresh()
    calculator_line_panel.refresh()
    suggestion_panel.refresh()
    interpretation_panel.refresh()


def _apply_explore_suggestion(value: str) -> None:
    state.explore = value
    state.active_field = "explore"
    state.calculator_line = build_calculator_line(state.observe, state.explore, state.variables, state.conditions)
    calculator_fields.refresh()
    calculator_line_panel.refresh()
    suggestion_panel.refresh()
    interpretation_panel.refresh()


def _parse_line_action() -> None:
    try:
        data = parse_calculator_line(state.calculator_line)
        state.observe = data["observe"]
        state.explore = data["explore"]
        state.variables = data["variables"]
        state.conditions = data["conditions"]
        state.compilation = None
        state.plan = None
        state.last_error = None
        calculator_fields.refresh()
        suggestion_panel.refresh()
        interpretation_panel.refresh()
        advanced_preview.refresh()
        ui.notify("Calculator line parsed", type="positive")
    except Exception as exc:
        ui.notify(str(exc), type="negative", multi_line=True, timeout=8)


def _compile_current() -> CalculatorCompilation:
    if not state.landscape_info or not state.registry:
        raise RuntimeError("Load a landscape first")
    req = CalculatorRequest(
        name=state.name or "exploration",
        landscape_path=state.landscape_info.path,
        gem_model_path=state.model_info.path if state.model_info else None,
        observe=state.observe,
        explore=state.explore,
        variables=state.variables,
        conditions=state.conditions,
        min_r2=state.min_r2,
        min_auc=state.min_auc,
        n_estimators=state.n_estimators,
        cv_folds=state.cv_folds,
    )
    comp = compile_calculator(req, state.registry)
    state.compilation = comp
    state.last_error = None
    return comp


def _save_compilation(comp: CalculatorCompilation, suffix: str) -> tuple[Path, Path]:
    root = Path(state.workspace).expanduser().resolve() / "definitions"
    root.mkdir(parents=True, exist_ok=True)
    name = sanitize_name(comp.public_yaml.get("name") or state.name or "exploration")
    ep = root / f"{name}{suffix}_exploration.yaml"
    pp = root / f"{name}{suffix}_problem.yaml"
    ep.write_text(yaml.safe_dump(comp.public_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    pp.write_text(yaml.safe_dump(comp.internal_problem, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return ep, pp


@ui.refreshable
def source_status() -> None:
    with ui.card().classes("w-full p-4"):
        with ui.row().classes("w-full items-center justify-between gap-3"):
            ui.label("Evidence source").classes("text-h6")
            ui.badge("Read-only source", color="positive")
        ui.label("Choose the landscape folder. The calculator will try to find and load the matching GEM automatically.").classes("text-caption text-grey-7")
        with ui.row().classes("w-full gap-2 items-center mt-2 flex-wrap"):
            ui.button("Choose landscape…", icon="folder_open", on_click=_choose_landscape_action).props("color=primary")
            ui.button("Choose GEM…", icon="science", on_click=_choose_model_action).props("outline")
            if state.landscape_info:
                ui.badge(Path(state.landscape_info.path).name)
            if state.model_info:
                ui.badge(state.model_info.model_id, color="positive")
        with ui.expansion("Advanced: paths / table file", icon="terminal").classes("w-full mt-2"):
            with ui.row().classes("w-full gap-3 items-end"):
                ui.input(
                    "Landscape folder / CSV / Parquet",
                    value=state.landscape_path,
                    on_change=lambda e: setattr(state, "landscape_path", e.value),
                ).props("outlined dense clearable").classes("grow")
                ui.button("Load", on_click=_load_landscape_action).props("outline")
            with ui.row().classes("w-full gap-3 items-end mt-2"):
                ui.input(
                    "GEM model (.xml/.sbml/.json/.mat)",
                    value=state.model_path,
                    on_change=lambda e: setattr(state, "model_path", e.value),
                ).props("outlined dense clearable").classes("grow")
                ui.button("Load", on_click=_load_model_action).props("outline")
        if state.landscape_info:
            flux_catalog = state.landscape_info.meta.get("stored_long_catalog", []) or []
            with ui.row().classes("gap-2 mt-3 flex-wrap"):
                ui.badge(f"{state.landscape_info.rows:,} rows")
                ui.badge(f"{len(state.landscape_info.design_variables)} design variables")
                ui.badge(f"{len([e for e in (state.registry.entries if state.registry else []) if e.kind == 'observable'])} discoverable observables")
                if flux_catalog:
                    ui.badge(f"{len(flux_catalog)} stored flux/dual quantities", color="secondary")
                if state.landscape_info.feasible_rows is not None:
                    ui.badge(f"{state.landscape_info.feasible_rows:,} feasible")
                if state.model_info:
                    matched, total = model_coverage(state.landscape_info.design_variables, state.model_info)
                    ui.badge(f"GEM labels {matched}/{total}", color="positive" if matched == total else "warning")
                    ui.badge(f"{state.model_info.genes:,} GEM genes")
            if state.registry:
                with ui.expansion("What can this landscape analyze?", icon="inventory_2").classes("w-full mt-2"):
                    direct_obs=[e for e in state.registry.entries if e.kind == "observable" and not e.column.startswith(("exchange_flux__","reaction_flux__","shadow_price__","reduced_cost__"))]
                    vars_=[e for e in state.registry.entries if e.kind == "variable"]
                    ui.label("Stored/derived-ready observables").classes("text-subtitle2")
                    with ui.row().classes("gap-1 flex-wrap"):
                        for e in direct_obs[:24]:
                            ui.badge(e.display)
                    if len(direct_obs)>24:
                        ui.label(f"+ {len(direct_obs)-24} more; type a name in OBSERVE to search.").classes("text-caption text-grey-7")
                    ui.label("Design variables / interventions represented in this landscape").classes("text-subtitle2 mt-2")
                    with ui.row().classes("gap-1 flex-wrap"):
                        for e in vars_[:24]:
                            ui.badge(e.display, color="secondary")
                    if flux_catalog:
                        ui.label(f"Stored long-table evidence: {len(flux_catalog):,} exchange/reaction/dual quantities are searchable in OBSERVE and loaded only when requested.").classes("text-caption text-grey-7 mt-2")
                    ui.label("You can also create zero-FBA derived observables directly in OBSERVE, e.g. product/glucose, product-growth, log1p(product), or resource cost/oxygen, whenever the required source quantities exist in the landscape.").classes("text-caption text-grey-7 mt-2")
            if state.model_info:
                ui.label("Genes/reactions that are present only in the GEM but were not stored or varied in this landscape are not silently treated as evidence. Gene perturbation requires a perturbation-capable landscape or new exact FBA evaluation.").classes("text-caption text-grey-7 mt-2")
        else:
            ui.label("Nothing is loaded yet.").classes("text-caption text-grey-7 mt-2")


@ui.refreshable
def calculator_line_panel() -> None:
    with ui.card().classes("w-full p-4"):
        ui.label("One-line exploration calculator").classes("text-h6")
        ui.label("Type the science in any field order. EXPLORE is optional: if omitted, the calculator infers a conservative default. Pipes are optional.").classes("text-caption text-grey-7")
        with ui.row().classes("w-full items-end gap-2"):
            ui.input(
                "Calculator line",
                value=state.calculator_line,
                on_change=lambda e: setattr(state, "calculator_line", e.value),
                placeholder="observe:(product), variables:(glucose, oxygen), conditions:(growth > 0.2)",
            ).props("outlined dense clearable").classes("grow font-mono")
            ui.button("Parse", icon="calculate", on_click=_parse_line_action).props("color=primary")


@ui.refreshable
def calculator_fields() -> None:
    with ui.card().classes("w-full p-4"):
        ui.label("Scientific calculator").classes("text-h6")
        ui.label("For most questions you only need OBSERVE, VARIABLES and CONDITIONS. The calculator infers EXPLORE automatically; open the optional section only for a specific analysis such as Pareto, regimes, robustness or interactions.").classes("text-caption text-grey-7")
        with ui.grid(columns=3).classes("w-full gap-3 mt-2"):
            observe = ui.input("OBSERVE", value=state.observe, on_change=lambda e: _set_field("observe", e.value), placeholder="product, ATP flux, resource cost, product/glucose...").props("outlined dense clearable")
            observe.on("focus", lambda: _set_active("observe"))
            variables = ui.input("VARIABLES", value=state.variables, on_change=lambda e: _set_field("variables", e.value), placeholder="glucose, oxygen or all").props("outlined dense clearable")
            variables.on("focus", lambda: _set_active("variables"))
            conditions = ui.input("CONDITIONS", value=state.conditions, on_change=lambda e: _set_field("conditions", e.value), placeholder="growth > 0.2, oxygen > 0.3 or growth >= q80").props("outlined dense clearable")
            conditions.on("focus", lambda: _set_active("conditions"))
        with ui.expansion("Optional: choose a specific exploration type", icon="tune").classes("w-full mt-2"):
            explore = ui.input("EXPLORE", value=state.explore, on_change=lambda e: _set_field("explore", e.value), placeholder="auto, pareto, structure(regimes), robust, relate(interaction)...").props("outlined dense clearable").classes("w-full")
            explore.on("focus", lambda: _set_active("explore"))
            ui.label("Leave this as auto for ordinary describe/relate/vary questions. Specialized functions remain available without exposing the 29 internal operators.").classes("text-caption text-grey-7")


@ui.refreshable
def suggestion_panel() -> None:
    with ui.card().classes("w-full p-3"):
        field = state.active_field
        ui.label(f"Suggestions · {field}").classes("text-subtitle2")
        if field == "explore":
            current = state.explore.strip().lower()
            hits = [x for x in EXPLORE_EXAMPLES if not current or current in x.lower()]
            if not hits:
                hits = EXPLORE_EXAMPLES[:12]
            with ui.row().classes("gap-1 flex-wrap"):
                for x in hits[:12]:
                    ui.button(x, on_click=lambda x=x: _apply_explore_suggestion(x)).props("flat dense no-caps")
            ui.label("Public functions are intentionally small: " + ", ".join(PUBLIC_EXPLORE_HELP)).classes("text-caption text-grey-7")
            return
        if not state.registry:
            ui.label("Load the landscape to activate semantic autocomplete.").classes("text-caption text-grey-7")
            return
        text = getattr(state, field)
        frag = active_fragment(text, field)
        hits = autocomplete_entries(state.registry, frag, field, limit=8)
        if not hits:
            ui.label("No semantic match yet. Continue typing, or use the canonical ID.").classes("text-caption text-grey-7")
            return
        with ui.column().classes("w-full gap-0"):
            for h in hits:
                label = h["label"]
                value = h["display"]
                with ui.row().classes("w-full items-center justify-between"):
                    ui.button(label, on_click=lambda value=value, field=field: _apply_suggestion(field, value)).props("flat dense no-caps").classes("text-left")
                    if h["reaction_id"]:
                        ui.label(h["canonical"]).classes("text-caption text-grey-6")


@ui.refreshable
def interpretation_panel() -> None:
    with ui.card().classes("w-full p-4"):
        with ui.row().classes("w-full items-center justify-between"):
            ui.label("What the calculator understands").classes("text-h6")
            ui.input("Exploration name", value=state.name, on_change=lambda e: setattr(state, "name", e.value)).props("dense outlined").classes("w-64")
        if not state.landscape_info:
            ui.label("Load a landscape first.").classes("text-grey-7")
            return
        try:
            comp = _compile_current()
            q = comp.public_yaml.get("question") or {}
            ui.label(q.get("text") or q.get("title") or "").classes("text-subtitle1")
            s = comp.resolved_summary
            with ui.row().classes("gap-2 flex-wrap mt-2"):
                ui.badge("Observe: " + ", ".join(s["observe"]))
                ui.badge("Explore: " + s["explore_function"])
                ui.badge("Variables: " + ", ".join(s["variables"][:4]) + ("…" if len(s["variables"]) > 4 else ""))
                ui.badge("Internal: " + str(s["internal_operator"]), color="grey")
                ui.badge("New FBA: 0", color="positive")
            ui.label("Visualization will be compiled automatically from this resolved question and its result.").classes("text-caption text-grey-7 mt-2")
        except Exception as exc:
            state.compilation = None
            state.last_error = str(exc)
            ui.label(str(exc)).classes("text-warning")


@ui.refreshable
def advanced_preview() -> None:
    with ui.expansion("Advanced details · YAML, internal plan, QC", icon="settings").classes("w-full"):
        if state.compilation:
            ui.label("Public exploration contract").classes("text-subtitle2")
            ui.code(yaml.safe_dump(state.compilation.public_yaml, sort_keys=False, allow_unicode=True), language="yaml").classes("w-full max-h-[28rem] overflow-auto")
            ui.label("Compiled internal problem").classes("text-subtitle2 mt-3")
            ui.code(yaml.safe_dump(state.compilation.internal_problem, sort_keys=False, allow_unicode=True), language="yaml").classes("w-full max-h-[28rem] overflow-auto")
        else:
            ui.label("Compile or preflight a valid formulation to see advanced details.").classes("text-caption text-grey-7")
        if state.plan:
            ui.separator().classes("my-3")
            ui.label("Resolved preflight").classes("text-subtitle2")
            rows = [
                {"field": "Status", "value": state.plan.get("status")},
                {"field": "Evidence backend", "value": state.plan.get("evidence_backend")},
                {"field": "Rows in scope", "value": state.plan.get("rows_in_scope")},
                {"field": "Rows total", "value": state.plan.get("rows_total")},
                {"field": "Resolved variables", "value": len(state.plan.get("resolved_variables") or [])},
                {"field": "New FBA calls", "value": state.plan.get("new_fba_calls")},
            ]
            ui.table(columns=[{"name":"field","label":"Field","field":"field"},{"name":"value","label":"Value","field":"value"}], rows=rows, row_key="field").classes("w-full")


@ui.refreshable
def result_preview() -> None:
    with ui.card().classes("w-full p-4"):
        ui.label("Result").classes("text-h6")
        if not state.last_output:
            ui.label("Run a question to view its results. You can open the YAML under Advanced details.").classes("text-grey-7")
            return
        out = Path(state.last_output)
        if not out.exists():
            ui.label("The latest output folder no longer exists.").classes("text-warning")
            return
        summary_path = out / "summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text())
            except Exception:
                summary = {}
        question = summary.get("scientific_question") or ((state.compilation.public_yaml.get("question") or {}).get("text") if state.compilation else "")
        if question:
            ui.label(question).classes("text-subtitle1")
        with ui.row().classes("gap-2 flex-wrap"):
            ui.badge(f'Rows in scope: {summary.get("rows_in_scope", "-")}')
            ui.badge(f'Evidence: {summary.get("evidence", summary.get("evidence_backend", "-"))}')
            ui.badge(f'Visualization: {summary.get("visualization_family", "-")}')
            ui.badge(f'New FBA: {summary.get("new_fba_calls", 0)}', color="positive")
            qc = summary.get("surrogate_qc")
            if isinstance(qc, dict):
                if qc.get("r2") is not None:
                    ui.badge(f'R²: {float(qc.get("r2")):.3f}')
                if qc.get("auc") is not None:
                    ui.badge(f'AUC: {float(qc.get("auc")):.3f}')
        fig_dir = out / "figures"
        pngs = sorted(fig_dir.glob("*.png"), key=lambda p: (0 if p.name == "main.png" else 1, p.name)) if fig_dir.exists() else []
        view_labels: dict[str, str] = {}
        vsp = out / "visualization_spec.json"
        if vsp.exists():
            try:
                vraw = json.loads(vsp.read_text())
                for v in ((vraw.get("visualization") or {}).get("views") or []):
                    key = str(v.get("key") or "")
                    bits = []
                    if v.get("target"): bits.append(str(v.get("target")))
                    if v.get("x") and v.get("y"): bits.append(f"{v.get('x')} vs {v.get('y')}")
                    view_labels[key] = " · ".join(bits) or str(v.get("family") or key).replace("_", " ").title()
            except Exception:
                view_labels = {}
        if pngs:
            if len(pngs) > 1:
                ui.label(f"This question generated {len(pngs)} complementary scientific views because the selected observables/features cannot be represented faithfully in one plot.").classes("text-caption text-grey-7 mt-2")
            for fp in pngs:
                key = fp.stem
                if len(pngs) > 1:
                    ui.label(view_labels.get(key, key.replace("_", " ").title())).classes("text-subtitle2 mt-3")
                ui.image(_png_data_uri(fp)).classes("w-full max-w-5xl mx-auto mt-2")
        else:
            ui.label("No plot is available for this result. See the numerical results below.").classes("text-caption text-grey-7")
        csvs = sorted(out.glob("*.csv"))
        if csvs:
            with ui.expansion("Numerical result tables", icon="table_view").classes("w-full mt-3"):
                for p in csvs[:5]:
                    try:
                        d = pd.read_csv(p).head(12)
                        cols = list(d.columns[:10])
                        rows = d[cols].where(pd.notna(d[cols]), None).to_dict("records")
                        ui.label(p.name).classes("text-caption")
                        ui.table(columns=[{"name":str(c),"label":str(c),"field":str(c)} for c in cols], rows=rows, pagination=6).classes("w-full")
                    except Exception:
                        continue
        ui.button("Open output folder", icon="folder_open", on_click=lambda: _open_folder(str(out))).props("outline")


async def _choose_landscape_action() -> None:
    if sys.platform != "darwin":
        ui.notify("Native chooser is available on macOS; use Advanced paths on other systems.", type="warning")
        return
    path = await run.io_bound(_choose_folder_native, "Choose a GEM landscape run folder")
    if not path:
        return
    state.landscape_path = path.rstrip("/")
    await _load_landscape_action()


async def _choose_model_action() -> None:
    if sys.platform != "darwin":
        ui.notify("Native chooser is available on macOS; use Advanced paths on other systems.", type="warning")
        return
    path = await run.io_bound(_choose_file_native, "Choose the matching GEM model")
    if not path:
        return
    state.model_path = path
    await _load_model_action()


async def _load_landscape_action() -> None:
    path = state.landscape_path.strip()
    if not path:
        ui.notify("Choose a landscape first", type="warning")
        return
    try:
        info, df = await run.io_bound(inspect_landscape, path, False, True)
        state.landscape_info = info
        state.landscape_df = df
        state.landscape_path = info.path

        # Never keep a stale model from a previously loaded landscape.
        state.model_info = None
        state.model = None
        state.model_path = ""
        candidate = await run.io_bound(infer_matching_model_path, info.path)
        if candidate:
            try:
                minfo, model = await run.io_bound(inspect_model, candidate)
                state.model_info = minfo
                state.model = model
                state.model_path = minfo.path
            except Exception:
                # Landscape analysis remains available even if semantic model loading fails.
                pass

        state.registry = build_registry(info, state.model_info)
        state.compilation = None
        state.plan = None
        if state.model_info:
            ui.notify(f"Landscape loaded: {info.rows:,} rows · matching GEM auto-loaded: {state.model_info.model_id}", type="positive")
        else:
            ui.notify(f"Landscape loaded: {info.rows:,} rows · choose the GEM if you want biological names", type="positive")
        source_status.refresh(); calculator_fields.refresh(); suggestion_panel.refresh(); interpretation_panel.refresh(); advanced_preview.refresh()
    except Exception as exc:
        ui.notify(f"Landscape load failed: {exc}", type="negative", multi_line=True, timeout=8)


async def _load_model_action() -> None:
    path = state.model_path.strip()
    if not path:
        ui.notify("Enter a GEM model path", type="warning")
        return
    try:
        info, model = await run.io_bound(inspect_model, path)
        state.model_info = info
        state.model = model
        state.model_path = info.path
        if state.landscape_info:
            state.registry = build_registry(state.landscape_info, info)
        ui.notify(f"GEM loaded: {info.model_id}. No optimization/FBA was run.", type="positive")
        source_status.refresh(); calculator_fields.refresh(); suggestion_panel.refresh(); interpretation_panel.refresh()
    except Exception as exc:
        ui.notify(f"GEM load failed: {exc}", type="negative", multi_line=True, timeout=8)


def _compile_action() -> None:
    try:
        comp = _compile_current()
        _save_compilation(comp, "_draft")
        ui.notify("Formulation compiled", type="positive")
        interpretation_panel.refresh(); advanced_preview.refresh()
    except Exception as exc:
        state.last_error = str(exc)
        ui.notify(str(exc), type="negative", multi_line=True, timeout=8)
        interpretation_panel.refresh(); advanced_preview.refresh()


async def _preflight_action() -> None:
    try:
        comp = _compile_current()
        _, pp = _save_compilation(comp, "_draft")
        raw = load_problem(pp)
        state.plan = await run.io_bound(plan_problem, raw, state.landscape_info.path if state.landscape_info else None, 20260822)
        ui.notify(f'Preflight PASS · {state.plan.get("rows_in_scope", 0):,} rows in scope · new FBA 0', type="positive")
        advanced_preview.refresh(); interpretation_panel.refresh()
    except Exception as exc:
        state.plan = None
        state.last_error = str(exc)
        ui.notify(f"Preflight failed: {exc}", type="negative", multi_line=True, timeout=8)
        advanced_preview.refresh()


async def _run_action() -> None:
    if not state.landscape_info:
        ui.notify("Load a landscape first", type="warning")
        return
    try:
        comp = _compile_current()
        stamp = time.strftime("%Y%m%d_%H%M%S")
        ep, pp = _save_compilation(comp, f"_{stamp}")
        raw = load_problem(pp)
        # Always preflight before execution so an invalid scope never starts analysis.
        state.plan = await run.io_bound(plan_problem, raw, state.landscape_info.path, 20260822)
        root = Path(state.workspace).expanduser().resolve() / "results"
        root.mkdir(parents=True, exist_ok=True)
        out = root / f'{sanitize_name(comp.public_yaml.get("name") or state.name)}_{stamp}'
        result_path = await run.io_bound(run_problem, raw, state.landscape_info.path, str(out), False, 20260822)
        result_path = Path(result_path)
        shutil.copy2(ep, result_path / "exploration.yaml")
        shutil.copy2(pp, result_path / "problem.yaml")
        state.last_output = str(result_path)
        state.last_error = None
        ui.notify("Analysis complete · source landscape unchanged · new FBA calls 0", type="positive")
        advanced_preview.refresh(); result_preview.refresh(); interpretation_panel.refresh()
    except Exception as exc:
        state.last_error = str(exc)
        ui.notify(f"Analysis failed closed: {exc}", type="negative", multi_line=True, timeout=8)
        advanced_preview.refresh(); result_preview.refresh()


@ui.refreshable
def examples_panel() -> None:
    with ui.expansion("Calculator examples", icon="lightbulb").classes("w-full"):
        examples = [
            ("2-D response", "product", "vary", "glucose, oxygen", "growth >= q80, feasible = true"),
            ("Exact associations", "product", "relate", "all", "feasible = true"),
            ("Resource cost", "resource cost", "describe", "all", "feasible = true"),
            ("Resource efficiency vs oxygen", "resource efficiency", "vary", "oxygen", "feasible = true"),
            ("Interactions", "product", "relate(interaction)", "all", "growth >= q50"),
            ("Robustness", "product", "robust", "oxygen", "growth > 0.2"),
            ("Exact selection", "product", "select", "all", "growth > 0.2, oxygen > 0.3"),
            ("Pareto frontier", "growth, product", "pareto", "all", "feasible = true"),
            ("Regimes", "growth, product", "structure(regimes)", "all", "feasible = true"),
            ("Joint regimes", "growth, product, resource efficiency", "structure(regimes)", "glucose, oxygen", "feasible = true"),
            ("Multi-observable response", "growth, product", "vary", "oxygen", "feasible = true"),
            ("Derived efficiency", "product/glucose", "describe", "all", "feasible = true"),
            ("Sampling coverage", "growth", "structure(coverage)", "all", "none"),
        ]
        for label, observe, explore, variables, conditions in examples:
            with ui.row().classes("w-full items-center justify-between"):
                ui.label(f"{label}: observe={observe} · explore={explore} · variables={variables} · conditions={conditions}").classes("text-caption")
                ui.button("Use", on_click=lambda observe=observe, explore=explore, variables=variables, conditions=conditions: _use_example(observe, explore, variables, conditions)).props("flat dense")


def _use_example(observe: str, explore: str, variables: str, conditions: str) -> None:
    state.observe = observe; state.explore = explore; state.variables = variables; state.conditions = conditions
    state.calculator_line = build_calculator_line(observe, explore, variables, conditions)
    state.compilation = None; state.plan = None; state.last_error = None
    calculator_fields.refresh(); calculator_line_panel.refresh(); suggestion_panel.refresh(); interpretation_panel.refresh(); advanced_preview.refresh()


def _build_page(*, embedded: bool = False, on_save_to_pool=None) -> None:
    if not embedded:
        ui.page_title(f"{APP_NAME} v{VERSION}")
    with (ui.column() if embedded else ui.header()).classes("items-center justify-between"):
        with ui.column().classes("gap-0"):
            ui.label(f"{APP_NAME} v{VERSION}").classes("text-h5")
            ui.label("OBSERVE · EXPLORE · VARIABLES · CONDITIONS").classes("text-caption")
        ui.badge("General calculator · dynamic quantities · multi-view visualization", color="positive")

    with ui.column().classes("w-full max-w-[1250px] mx-auto p-4 gap-4"):
        ui.label("Load a reusable landscape and GEM, then build and save a question to Saved Questions. Stored and derived quantities are discovered dynamically.").classes("text-body1")
        source_status()
        calculator_line_panel()
        calculator_fields()
        suggestion_panel()
        examples_panel()
        interpretation_panel()
        with ui.card().classes("w-full p-4"):
            with ui.row().classes("w-full items-center justify-between gap-2 flex-wrap"):
                with ui.row().classes("gap-2"):
                    ui.button("Preflight", icon="fact_check", on_click=_preflight_action).props("outline color=primary")
                    if on_save_to_pool is not None:
                        def save_into_pool():
                            try:
                                c = _compile_current()
                                on_save_to_pool(c.public_yaml, c.internal_problem)
                                ui.notify("Saved calculator question to shared YAML pool", type="positive")
                            except Exception as exc:
                                ui.notify(str(exc), type="negative")
                        ui.button("Save to Query Pool", icon="playlist_add", on_click=save_into_pool).props("outline")
                    if on_save_to_pool is None:
                        ui.button("Run", icon="play_arrow", on_click=_run_action).props("color=positive")
                ui.label("Preflight is automatic before every run. Invalid or undersupported scopes fail closed before analysis.").classes("text-caption text-grey-7")
        result_preview()
        advanced_preview()
        with ui.expansion("Workspace & evidence policy", icon="verified").classes("w-full"):
            ui.input("Workspace", value=state.workspace, on_change=lambda e: setattr(state, "workspace", e.value)).props("outlined dense").classes("w-full")
            ui.markdown(
                """
- The source landscape is read-only.
- Direct questions use the stored landscape first and perform **zero new FBA calls**.
- The GEM file is used for reaction semantics/autocomplete and compatibility metadata.
- Stored exchange/reaction fluxes are discovered and loaded only when referenced.
- Surrogate response questions train on the full design context; if surrogate QC fails, 1-D/2-D vary questions fall back to exact-landscape binned evidence rather than inventing a smooth surface.
- A question may generate multiple plots when multiple observables/features require more than one faithful view (for example regime discovery and >2-objective Pareto analysis).
- Every completed run saves `exploration.yaml`, compiled `problem.yaml`, result tables, `visualization_spec.yaml`, and PNG/PDF/SVG exports.
"""
            )


def main() -> None:
    _build_page()
    ui.run(title=f"{APP_NAME} v{VERSION}", reload=False, show=True, port=8092, storage_secret="gem-exploration-calculator-v050-local")


if __name__ in {"__main__", "__mp_main__"}:
    main()
