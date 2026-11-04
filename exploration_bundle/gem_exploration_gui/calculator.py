from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable
import math
import re
import ast

from .inspectors import LandscapeInspection, ModelInspection, reaction_id_from_design_variable, semantic_variable_labels
from .problem_builder import TargetDraft, sanitize_name
from .unified import UnifiedCondition, UnifiedQuestion, compile_unified_question


PUBLIC_EXPLORE_HELP: dict[str, str] = {
    "describe": "Summarize the observable distribution.",
    "relate": "Ask which variables are associated with or control the observable.",
    "vary": "Ask how the observable changes over one or more variables.",
    "compare": "Compare the observable between two regions or across a threshold.",
    "robust": "Ask where the observable is locally stable; use robust(fragility) for fragility.",
    "select": "Select exact sampled designs matching conditions, boundaries, or outliers.",
    "structure": "Explore regimes, dimensionality, coverage, saturation, or surrogate support.",
    "pareto": "Explore multiobjective trade-offs, frontiers, or sampling saturation.",
}

EXPLORE_EXAMPLES = [
    "auto",
    "describe",
    "relate",
    "relate(association)",
    "relate(correlation)",
    "relate(local)",
    "relate(map)",
    "relate(interaction)",
    "relate(controller)",
    "relate(feasibility)",
    "vary",
    "vary(local)",
    "vary(marginal)",
    "vary(point)",
    "vary(threshold=1.0)",
    "compare(threshold=1.0)",
    "compare(regions)",
    "robust",
    "robust(fragility)",
    "select",
    "select(boundary=1.0)",
    "select(outliers)",
    "structure(regimes)",
    "structure(dimensions)",
    "structure(coverage)",
    "structure(saturation)",
    "structure(support)",
    "pareto",
    "pareto(tradeoff)",
    "pareto(saturation)",
]


@dataclass(frozen=True)
class SemanticEntry:
    canonical: str
    display: str
    kind: str
    column: str
    aliases: tuple[str, ...] = ()
    reaction_id: str | None = None
    unit: str | None = None

    def search_blob(self) -> str:
        return " ".join([self.canonical, self.display, self.column, self.reaction_id or "", *self.aliases]).lower()


@dataclass
class SemanticRegistry:
    entries: list[SemanticEntry] = field(default_factory=list)

    def by_canonical(self) -> dict[str, SemanticEntry]:
        return {e.canonical: e for e in self.entries}

    def _norm(self, text: str) -> str:
        s = str(text or "").strip().lower()
        s = re.sub(r"[_:\-/]+", " ", s)
        s = re.sub(r"\b(exchange|uptake|reaction|flux|requested|medium|result)\b", " ", s)
        return re.sub(r"\s+", " ", s).strip()

    def search(self, text: str, kinds: set[str] | None = None, limit: int = 8) -> list[SemanticEntry]:
        q = str(text or "").strip()
        if not q:
            rows = [e for e in self.entries if kinds is None or e.kind in kinds]
            return rows[:limit]
        ql = q.lower()
        qn = self._norm(q)
        scored: list[tuple[float, SemanticEntry]] = []
        for e in self.entries:
            if kinds is not None and e.kind not in kinds:
                continue
            fields = [e.canonical, e.display, e.column, e.reaction_id or "", *e.aliases]
            fields_l = [str(x).lower() for x in fields if x]
            fields_n = [self._norm(x) for x in fields_l]
            score = 0.0
            if any(ql == x for x in fields_l):
                score = 1000.0
            elif any(qn and qn == x for x in fields_n):
                score = 950.0
            elif any(ql in x for x in fields_l):
                score = 800.0 - min(len(x) for x in fields_l if ql in x) * 0.01
            elif any(qn and qn in x for x in fields_n):
                score = 760.0
            else:
                ratio = max([SequenceMatcher(None, qn, x).ratio() for x in fields_n if x] or [0.0])
                if ratio >= 0.32:
                    score = ratio * 500.0
            if score:
                scored.append((score, e))
        scored.sort(key=lambda x: (-x[0], x[1].display.lower(), x[1].canonical.lower()))
        out: list[SemanticEntry] = []
        seen: set[str] = set()
        for _, e in scored:
            if e.canonical in seen:
                continue
            seen.add(e.canonical)
            out.append(e)
            if len(out) >= limit:
                break
        return out

    def resolve(self, text: str, kinds: set[str] | None = None) -> SemanticEntry:
        q = str(text or "").strip()
        if not q:
            raise ValueError("Empty quantity/variable name")
        # strict aliases first
        ql = q.lower()
        exact = []
        for e in self.entries:
            if kinds is not None and e.kind not in kinds:
                continue
            fields = [e.canonical, e.display, e.column, e.reaction_id or "", *e.aliases]
            if any(ql == str(x).lower() for x in fields if x):
                exact.append(e)
        if len(exact) == 1:
            return exact[0]
        if len(exact) > 1:
            labels = ", ".join(f"{x.display} [{x.canonical}]" for x in exact[:6])
            raise ValueError(f"Ambiguous name {q!r}: {labels}")
        hits = self.search(q, kinds=kinds, limit=6)
        if not hits:
            raise ValueError(f"Unknown quantity/variable {q!r}")
        # If fuzzy best is substantially better, accept; otherwise fail with suggestions.
        if len(hits) == 1:
            return hits[0]
        top = hits[0]
        qn = self._norm(q)
        top_ratio = max(SequenceMatcher(None, qn, self._norm(x)).ratio() for x in [top.display, top.canonical, top.column, *top.aliases] if x)
        second = hits[1]
        second_ratio = max(SequenceMatcher(None, qn, self._norm(x)).ratio() for x in [second.display, second.canonical, second.column, *second.aliases] if x)
        if top_ratio >= 0.78 and top_ratio - second_ratio >= 0.10:
            return top
        labels = ", ".join(f"{x.display} [{x.canonical}]" for x in hits[:5])
        raise ValueError(f"Could not uniquely resolve {q!r}. Suggestions: {labels}")


_SPECIAL_ALIASES: dict[str, tuple[str, ...]] = {
    "maximum_growth": ("growth", "growth rate", "max growth", "maximum growth", "biomass", "biomass growth"),
    "product_flux": ("product", "production", "product flux", "product formation", "product production"),
    "resource_cost": ("cost", "resource cost", "metabolic cost"),
    "resource_efficiency": ("efficiency", "resource efficiency"),
    "feasible": ("feasible", "feasibility", "valid"),
    "status": ("status",),
}




def _common_reaction_aliases(rid: str) -> list[str]:
    r = str(rid or "").lower()
    out: list[str] = []
    mapping = [
        (("glc", "glucose"), "glucose"),
        (("o2", "oxygen"), "oxygen"),
        (("nh4", "ammon", "ammonium"), "ammonium"),
        (("pi", "phosphate"), "phosphate"),
        (("so4", "sulfate"), "sulfate"),
        (("co2",), "carbon dioxide"),
        (("etoh", "ethanol"), "ethanol"),
    ]
    for needles, label in mapping:
        if any(n in r for n in needles):
            out.extend([label, f"{label} exchange", f"{label} uptake"])
    return out

def _humanize_column(column: str) -> str:
    c = column
    for prefix in ("requested_medium__", "applied_medium__", "exchange_flux__", "reaction_flux__", "shadow_price__", "reduced_cost__"):
        if c.startswith(prefix):
            c = c[len(prefix):]
    return c.replace("__", " ").replace("_", " ").strip().title()


def build_registry(landscape: LandscapeInspection, model: ModelInspection | None) -> SemanticRegistry:
    entries: list[SemanticEntry] = []
    sem = semantic_variable_labels(landscape.design_variables, model)

    # Stored quantities / result aliases.
    seen: set[str] = set()
    for item in landscape.target_options:
        alias = str(item["alias"])
        col = str(item["column"])
        kind = "observable" if not col.startswith(("requested_medium__", "applied_medium__")) else "variable"
        display = _humanize_column(col)
        aliases: list[str] = []
        aliases.extend(_SPECIAL_ALIASES.get(col, ()))
        if alias.startswith("result:"):
            aliases.append(alias.split(":", 1)[1])
        rid = None
        if col.startswith(("requested_medium__", "applied_medium__")):
            rid = reaction_id_from_design_variable(col)
            display = sem.get(col) or (model.reaction_labels.get(rid) if model else None) or _humanize_column(col)
            aliases.extend([rid, display, display.replace(" exchange", ""), display.replace(" Exchange", ""), *_common_reaction_aliases(rid)])
        if alias not in seen:
            entries.append(SemanticEntry(alias if kind == "observable" else col, display, kind, col, tuple(dict.fromkeys(a for a in aliases if a)), rid))
            seen.add(alias)

    # Design variables may not all appear among numeric target options in some sources.
    for v in landscape.design_variables:
        if any(e.column == v for e in entries):
            continue
        rid = reaction_id_from_design_variable(v)
        display = sem.get(v) or (model.reaction_labels.get(rid) if model else None) or _humanize_column(v)
        aliases = [rid, display, display.replace(" exchange", ""), display.replace(" Exchange", ""), *_common_reaction_aliases(rid)]
        entries.append(SemanticEntry(v, display, "variable", v, tuple(dict.fromkeys(a for a in aliases if a)), rid))

    # Stored long-table quantities are discovered without loading the full flux matrix.
    # They become selectable observables and are materialized lazily only when a
    # question actually references them.
    for item in landscape.meta.get("stored_long_catalog", []) or []:
        alias = str(item.get("alias") or "")
        col = str(item.get("column") or "")
        if not alias or not col or any(e.canonical == alias for e in entries):
            continue
        rid = str(item.get("entity_id") or "") or None
        display = (model.reaction_labels.get(rid) if (model and rid and rid in model.reaction_labels) else None) or _humanize_column(col)
        aliases = [rid or "", display, *_common_reaction_aliases(rid or "")]
        entries.append(SemanticEntry(alias, display, "observable", col, tuple(dict.fromkeys(a for a in aliases if a)), rid))

    # Boolean/non-numeric columns can still be valid conditions.
    for c in ("feasible", "status"):
        if c in landscape.meta.get("columns", []) and not any(e.column == c for e in entries):
            entries.append(SemanticEntry(f"result:{c}", c.title(), "observable", c, _SPECIAL_ALIASES.get(c, ())))

    # The common columns are important enough to add if present in the loaded dataframe schema metadata.
    cols = set(landscape.numeric_columns)
    for c in ("maximum_growth", "product_flux", "resource_cost", "resource_efficiency", "feasible"):
        if c in cols and not any(e.column == c for e in entries):
            entries.append(SemanticEntry(f"result:{c}", _humanize_column(c), "observable", c, _SPECIAL_ALIASES.get(c, ())))

    # Prefer observables first, then design variables alphabetically.
    entries.sort(key=lambda e: (0 if e.kind == "observable" else 1, e.display.lower(), e.canonical.lower()))
    return SemanticRegistry(entries)


def split_top_level(text: str, separators: tuple[str, ...] = (",",)) -> list[str]:
    s = str(text or "")
    out: list[str] = []
    start = 0
    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "([{" :
            depth += 1
        elif ch in ")]}":
            depth = max(0, depth - 1)
        if depth == 0:
            matched = None
            for sep in separators:
                if s[i:i+len(sep)].lower() == sep.lower():
                    matched = sep
                    break
            if matched:
                part = s[start:i].strip()
                if part:
                    out.append(part)
                i += len(matched)
                start = i
                continue
        i += 1
    part = s[start:].strip()
    if part:
        out.append(part)
    return out


def _parse_bool_or_number(raw: str) -> Any:
    t = str(raw).strip()
    if not t:
        return None
    if (t.startswith("'") and t.endswith("'")) or (t.startswith('"') and t.endswith('"')):
        return t[1:-1]
    lo = t.lower()
    if lo in {"true", "yes"}:
        return True
    if lo in {"false", "no"}:
        return False
    if lo in {"none", "null"}:
        return None
    try:
        return float(t)
    except Exception:
        return t


def _condition_from_tokens(lhs: str, op: str, rhs: str, registry: SemanticRegistry) -> UnifiedCondition:
    entry = registry.resolve(lhs, kinds={"observable", "variable"})
    canonical = entry.canonical
    val_text = str(rhs).strip()
    q = re.fullmatch(r"q\s*(\d{1,3})", val_text, flags=re.I)
    if q:
        percentile = int(q.group(1)) / 100.0
        if not (0 < percentile < 1):
            raise ValueError(f"Quantile must be between q1 and q99, got {val_text}")
        if op in {">", ">="}:
            return UnifiedCondition(canonical, "top_quantile", 1.0 - percentile)
        if op in {"<", "<="}:
            return UnifiedCondition(canonical, "bottom_quantile", percentile)
        raise ValueError("Quantile conditions require >, >=, <, or <=")
    top = re.fullmatch(r"top\s*(\d+(?:\.\d+)?)\s*%?", val_text, flags=re.I)
    if top:
        frac = float(top.group(1)) / 100.0
        return UnifiedCondition(canonical, "top_quantile", frac)
    bottom = re.fullmatch(r"bottom\s*(\d+(?:\.\d+)?)\s*%?", val_text, flags=re.I)
    if bottom:
        frac = float(bottom.group(1)) / 100.0
        return UnifiedCondition(canonical, "bottom_quantile", frac)
    val = _parse_bool_or_number(val_text)
    if isinstance(val, bool) and op in {"=", "=="}:
        return UnifiedCondition(canonical, "true" if val else "false", None)
    return UnifiedCondition(canonical, "==" if op == "=" else op, val)


def parse_conditions(text: str, registry: SemanticRegistry) -> tuple[list[UnifiedCondition], str, list[UnifiedCondition], list[UnifiedCondition]]:
    raw = str(text or "").strip()
    if not raw or raw.lower() in {"none", "all", "full", "no conditions"}:
        return [], "all", [], []

    # Region comparison syntax: A[growth >= q80]; B[growth <= q20]
    region_a: list[UnifiedCondition] = []
    region_b: list[UnifiedCondition] = []
    region_match = re.findall(r"\b([AB])\s*\[(.*?)\]", raw, flags=re.I)
    if region_match:
        for label, body in region_match:
            conds, logic, _, _ = parse_conditions(body, registry)
            if logic != "all":
                raise ValueError("Region A/B conditions currently use AND/comma logic inside each region")
            if label.upper() == "A":
                region_a.extend(conds)
            else:
                region_b.extend(conds)
        leftover = re.sub(r"\b[AB]\s*\[.*?\]", "", raw, flags=re.I).replace(";", " ").strip()
        if leftover:
            raise ValueError(f"Could not parse extra condition text outside A[] / B[]: {leftover!r}")
        return [], "all", region_a, region_b

    has_or = bool(re.search(r"\s+OR\s+", raw, flags=re.I))
    has_and = bool(re.search(r"\s+AND\s+", raw, flags=re.I))
    if has_or and has_and:
        raise ValueError("Mixed AND/OR in one calculator condition line is intentionally fail-closed. Use only commas/AND or only OR.")
    logic = "any" if has_or else "all"
    seps = (" OR ",) if has_or else (" AND ", ",")
    parts = split_top_level(raw, separators=seps)
    out: list[UnifiedCondition] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # shorthand: feasible / not feasible
        if part.lower() in {"feasible", "is feasible"}:
            e = registry.resolve("feasible", kinds={"observable", "variable"})
            out.append(UnifiedCondition(e.canonical, "true", None))
            continue
        if part.lower() in {"not feasible", "infeasible"}:
            e = registry.resolve("feasible", kinds={"observable", "variable"})
            out.append(UnifiedCondition(e.canonical, "false", None))
            continue
        m = re.match(r"^(.+?)\s*(>=|<=|!=|==|=|>|<)\s*(.+)$", part)
        if not m:
            raise ValueError(f"Could not parse condition {part!r}. Example: growth > 0.2")
        out.append(_condition_from_tokens(m.group(1).strip(), m.group(2), m.group(3).strip(), registry))
    return out, logic, [], []


def _safe_derived_name(expr: str) -> str:
    return sanitize_name(re.sub(r"\s+", "_", expr.lower()))[:80]


_DERIVED_FUNCS = {"abs", "sqrt", "log", "log1p", "exp", "clip", "minimum", "maximum", "safe_divide"}


def _replace_semantic_phrases(expr: str, registry: SemanticRegistry) -> tuple[str, dict[str, SemanticEntry]]:
    """Replace human-readable quantity phrases with safe temporary identifiers.

    This lets users write expressions such as ``resource cost / glucose`` even
    though ``resource cost`` is not a valid Python identifier. Longest aliases
    are replaced first so a specific biological name wins over a short substring.
    """
    work = str(expr or "").strip()
    placeholders: dict[str, SemanticEntry] = {}
    candidates: list[tuple[str, SemanticEntry]] = []
    for e in registry.entries:
        names = [e.display, e.canonical, e.column, e.reaction_id or "", *e.aliases]
        if e.canonical.startswith("result:"):
            names.append(e.canonical.split(":", 1)[1])
        for name in names:
            n = str(name or "").strip()
            if n and len(n) >= 2:
                candidates.append((n, e))
    # Prefer longer phrases and exact biological labels over short aliases.
    candidates.sort(key=lambda x: (-len(x[0]), x[0].lower()))
    occupied: set[str] = set()
    counter = 0
    for name, entry in candidates:
        # Avoid replacing allowed function names.
        if name.lower() in _DERIVED_FUNCS:
            continue
        pattern = re.compile(rf"(?<![A-Za-z0-9_:]){re.escape(name)}(?![A-Za-z0-9_:])", re.I)
        if not pattern.search(work):
            continue
        key = f"__q{counter}__"; counter += 1
        work = pattern.sub(key, work)
        placeholders[key] = entry
    return work, placeholders


class _DerivedTransformer(ast.NodeTransformer):
    def __init__(self, placeholders: dict[str, SemanticEntry], registry: SemanticRegistry):
        self.placeholders = placeholders
        self.registry = registry

    def _col_call(self, canonical: str) -> ast.Call:
        return ast.Call(func=ast.Name(id="col", ctx=ast.Load()), args=[ast.Constant(value=canonical)], keywords=[])

    def visit_Name(self, node: ast.Name):
        if node.id in self.placeholders:
            return ast.copy_location(self._col_call(self.placeholders[node.id].canonical), node)
        if node.id in _DERIVED_FUNCS:
            return node
        # A short alias such as product/glucose may survive phrase replacement if
        # it was not unique enough to substitute. Resolve it semantically here.
        try:
            e = self.registry.resolve(node.id, kinds={"observable", "variable"})
        except Exception as exc:
            raise ValueError(f"Unknown quantity {node.id!r} in derived expression") from exc
        return ast.copy_location(self._col_call(e.canonical), node)

    def visit_Call(self, node: ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _DERIVED_FUNCS:
            raise ValueError(f"Unsupported derived-expression function: {ast.unparse(node.func) if hasattr(ast, 'unparse') else 'call'}")
        return self.generic_visit(node)

    def visit_BinOp(self, node: ast.BinOp):
        node = self.generic_visit(node)
        if isinstance(node.op, ast.Div):
            return ast.copy_location(ast.Call(func=ast.Name(id="safe_divide", ctx=ast.Load()), args=[node.left, node.right], keywords=[]), node)
        return node


def _expression_to_safe(expr: str, registry: SemanticRegistry) -> tuple[str, str, str]:
    """Compile a user-facing derived quantity into the engine's safe expression language.

    Supported operators: +, -, *, /, **, %, parentheses and numeric constants.
    Supported functions: abs, sqrt, log, log1p, exp, clip, minimum, maximum.
    Quantity names may be biological display names, aliases, reaction IDs or
    canonical landscape columns. Division is rewritten to safe_divide().
    """
    raw = str(expr or "").strip()
    if not raw:
        raise ValueError("Derived observable expression is empty")
    prepared, placeholders = _replace_semantic_phrases(raw, registry)
    try:
        tree = ast.parse(prepared, mode="eval")
    except SyntaxError as exc:
        raise ValueError(
            f"Could not parse derived observable {raw!r}. Examples: product/glucose, product - growth, log1p(product)."
        ) from exc
    allowed_nodes = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Load, ast.Constant,
        ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Pow, ast.Mod, ast.USub, ast.UAdd, ast.keyword,
    )
    for node in ast.walk(tree):
        if not isinstance(node, allowed_nodes):
            raise ValueError(f"Unsupported syntax in derived observable: {type(node).__name__}")
    transformed = _DerivedTransformer(placeholders, registry).visit(tree)
    ast.fix_missing_locations(transformed)
    try:
        safe = ast.unparse(transformed.body)
    except Exception as exc:
        raise ValueError(f"Could not compile derived observable {raw!r}") from exc
    name = _safe_derived_name(raw)
    return name, safe, raw


def parse_observe(text: str, registry: SemanticRegistry) -> tuple[list[TargetDraft], list[dict[str, Any]]]:
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("observe is required. Example: product")
    targets: list[TargetDraft] = []
    derived: list[dict[str, Any]] = []
    for token in split_top_level(raw, separators=(",",)):
        direction = "maximize"
        t = token.strip()
        mm = re.match(r"^(min|max|minimize|maximize)\s*\((.*)\)$", t, flags=re.I)
        if mm:
            direction = "minimize" if mm.group(1).lower().startswith("min") else "maximize"
            t = mm.group(2).strip()
        try:
            e = registry.resolve(t, kinds={"observable", "variable"})
            name = sanitize_name(e.canonical.split(":")[-1].replace("requested_medium__", ""))
            targets.append(TargetDraft(name=name, value=e.canonical, direction=direction, label=e.display, unit=e.unit))
        except ValueError:
            name, safe_expr, label = _expression_to_safe(t, registry)
            derived.append({"name": name, "expression": safe_expr})
            targets.append(TargetDraft(name=name, value=name, direction=direction, label=label))
    return targets, derived


def parse_variables(text: str, registry: SemanticRegistry) -> tuple[str, list[str], dict[str, float]]:
    raw = str(text or "").strip()
    if not raw or raw.lower() in {"all", "all variables", "all design", "all design variables"}:
        return "all_design", [], {}
    vars_: list[str] = []
    assignments: dict[str, float] = {}
    for token in split_top_level(raw, separators=(",",)):
        if "=" in token:
            lhs, rhs = token.split("=", 1)
            e = registry.resolve(lhs.strip(), kinds={"variable"})
            assignments[e.canonical] = float(rhs.strip())
            vars_.append(e.canonical)
        else:
            e = registry.resolve(token.strip(), kinds={"variable"})
            vars_.append(e.canonical)
    vars_ = list(dict.fromkeys(vars_))
    return "list", vars_, assignments


def _parse_explore_call(text: str) -> tuple[str, dict[str, Any], list[str]]:
    raw = str(text or "").strip().lower()
    if not raw:
        raise ValueError("explore is required. Example: vary")
    aliases = {
        "summarize": "describe", "summary": "describe", "distribution": "describe",
        "relation": "relate", "relationship": "relate", "sensitivity": "relate",
        "response": "vary", "change": "vary",
        "stability": "robust", "robustness": "robust", "fragility": "robust",
        "filter": "select", "search": "select", "find": "select",
        "cluster": "structure", "clustering": "structure", "dimension": "structure", "dimensionality": "structure",
        "tradeoff": "pareto", "trade-off": "pareto", "frontier": "pareto",
    }
    m = re.match(r"^([a-zA-Z_-]+)\s*(?:\((.*)\))?$", raw)
    if not m:
        raise ValueError(f"Could not parse explore={text!r}. Example: vary or relate(interaction)")
    base = aliases.get(m.group(1), m.group(1))
    if base not in PUBLIC_EXPLORE_HELP:
        raise ValueError(f"Unknown explore function {base!r}. Use: {', '.join(PUBLIC_EXPLORE_HELP)}")
    args: dict[str, Any] = {}
    flags: list[str] = []
    body = (m.group(2) or "").strip()
    if body:
        for item in split_top_level(body, separators=(",",)):
            item = item.strip()
            if not item:
                continue
            if "=" in item:
                k, v = item.split("=", 1)
                args[k.strip()] = _parse_bool_or_number(v.strip().rstrip("%"))
                if v.strip().endswith("%"):
                    args[k.strip()] = float(str(args[k.strip()])) / 100.0
            elif re.fullmatch(r"\d+(?:\.\d+)?%", item):
                args["fraction"] = float(item.rstrip("%")) / 100.0
            else:
                flags.append(item.lower())
    # aliases that imply flags
    if m.group(1) == "fragility":
        flags.append("fragility")
    if m.group(1) in {"tradeoff", "trade-off"}:
        flags.append("tradeoff")
    if m.group(1) == "frontier":
        flags.append("frontier")
    if m.group(1) in {"cluster", "clustering"}:
        flags.append("regimes")
    if m.group(1) in {"dimension", "dimensionality"}:
        flags.append("dimensions")
    if m.group(1) == "sensitivity":
        flags.append("sensitivity")
    return base, args, flags


def _detail_for(base: str, args: dict[str, Any], flags: list[str], target_count: int, variable_mode: str, variable_count: int, region_a: list[UnifiedCondition], region_b: list[UnifiedCondition]) -> tuple[str, dict[str, Any]]:
    f = set(flags)
    settings: dict[str, Any] = {}
    if base == "describe":
        return "distribution", settings
    if base == "relate":
        if "correlation" in f or "matrix" in f:
            return "correlation", settings
        if "association" in f:
            return "association", settings
        if "local" in f:
            if "step" in args:
                settings["step_fraction"] = float(args["step"])
            return "local_influence", settings
        if "map" in f or "context" in f:
            return "influence_map", settings
        if "interaction" in f or "interactions" in f:
            return "interaction", settings
        if "controller" in f or "control" in f:
            return "controller", settings
        if "feasibility" in f or "feasible" in f:
            return "feasibility", settings
        # Plain `relate` is exact-landscape-first. Model-based global influence
        # remains available explicitly as relate(influence) / relate(sensitivity).
        if "influence" in f or "sensitivity" in f or "importance" in f:
            return "global_influence", settings
        return "association", settings
    if base == "vary":
        if "point" in f:
            return "point", settings
        if "marginal" in f or "average" in f:
            return "marginal_curve", settings
        if "threshold" in args:
            settings["threshold"] = float(args["threshold"])
            return "threshold_crossing", settings
        if "threshold" in f:
            raise ValueError("vary(threshold=VALUE) requires a threshold value")
        if variable_mode == "all_design" or variable_count > 2:
            return "influence_map", settings  # implemented through relate primitive below by caller remap
        if variable_count == 2:
            return "surface", settings
        if variable_count == 1:
            return "local_curve", settings
        raise ValueError("vary needs one or two selected variables, or variables: all")
    if base == "compare":
        if "threshold" in args:
            settings["threshold"] = float(args["threshold"])
            settings["direction"] = str(args.get("direction", ">="))
            return "threshold_groups", settings
        if region_a and region_b or "regions" in f:
            return "regions", settings
        raise ValueError("compare needs either compare(threshold=VALUE) or conditions: A[...] ; B[...]")
    if base == "robust":
        if "fraction" in args:
            raise ValueError(
                "robust(10%) requests a defined perturbation experiment. The current read-only landscape backend cannot silently reinterpret that as nearest-neighbor robustness. "
                "Use `robust` for sampled-landscape local stability, or enable a future exact-FBA perturbation/refinement backend."
            )
        if "fragility" in f or "fragile" in f:
            return "fragility", settings
        if "neighbors" in args:
            settings["neighbors"] = int(args["neighbors"])
        return "robustness", settings
    if base == "select":
        if "boundary" in args:
            settings["threshold"] = float(args["boundary"])
            return "threshold_boundary", settings
        if "boundary" in f:
            raise ValueError("select(boundary=VALUE) requires a threshold value")
        if "outlier" in f or "outliers" in f:
            if "fraction" in args:
                settings["contamination"] = float(args["fraction"])
            return "outliers", settings
        return "conditions", settings
    if base == "structure":
        if "regime" in f or "regimes" in f or "cluster" in f:
            if "space" in args:
                settings["regime_space"] = str(args["space"]).lower()
            return "regimes", settings
        if "dimension" in f or "dimensions" in f or "pca" in f:
            return "dimensionality", settings
        if "coverage" in f:
            return "coverage", settings
        if "saturation" in f:
            return "sampling_saturation", settings
        if "support" in f or "uncertainty" in f:
            return "surrogate_support", settings
        # infer: multi-target -> regimes, no target -> coverage, otherwise support
        if target_count >= 2:
            return "regimes", settings
        return "coverage", settings
    if base == "pareto":
        if "tradeoff" in f or "trade-off" in f:
            return "tradeoff", settings
        if "saturation" in f:
            return "saturation", settings
        return "frontier", settings
    raise ValueError(base)


@dataclass
class CalculatorRequest:
    name: str
    landscape_path: str
    gem_model_path: str | None
    observe: str
    explore: str
    variables: str
    conditions: str
    backend_mode: str = "auto"
    min_r2: float = 0.50
    min_auc: float = 0.65
    n_estimators: int = 300
    cv_folds: int = 5
    output_dpi: int = 600


@dataclass
class CalculatorCompilation:
    request: CalculatorRequest
    unified_question: UnifiedQuestion
    public_yaml: dict[str, Any]
    internal_problem: dict[str, Any]
    resolved_summary: dict[str, Any]


def _infer_explore_text(observe_text: str, variables_text: str) -> str:
    """Conservative calculator inference when EXPLORE is omitted or set to auto.

    One/two explicit variables imply a response/vary question. `all` implies a
    direct association question rather than an unnecessary surrogate sensitivity
    model. With no usable variable expression, default to describe.
    """
    v = str(variables_text or "").strip().lower()
    if not v or v in {"none", "no variables"}:
        return "describe"
    if v in {"all", "all variables", "all design", "all design variables"}:
        return "relate(association)"
    # Explicit variables are the most natural calculator-style response question.
    return "vary"


def compile_calculator(req: CalculatorRequest, registry: SemanticRegistry) -> CalculatorCompilation:
    targets, derived = parse_observe(req.observe, registry)
    variable_mode, variables, assignments = parse_variables(req.variables, registry)
    conditions, logic, region_a, region_b = parse_conditions(req.conditions, registry)
    explore_text = str(req.explore or "").strip()
    if not explore_text or explore_text.lower() in {"auto", "infer", "automatic"}:
        explore_text = _infer_explore_text(req.observe, req.variables)
    base, args, flags = _parse_explore_call(explore_text)
    detail, inferred_settings = _detail_for(base, args, flags, len(targets), variable_mode, len(variables), region_a, region_b)

    primitive = "perturb" if base == "robust" else base
    # vary(all) is semantically a broad response/influence question; execute through the tested
    # sensitivity-map implementation while keeping public calculator wording as vary.
    if base == "vary" and detail == "influence_map":
        primitive = "relate"
        detail = "influence_map"

    settings = dict(inferred_settings)
    # Generic arguments common to existing tested implementations.
    if "grid" in args:
        settings["grid_points"] = int(args["grid"])
    if "neighbors" in args:
        settings["neighbors"] = int(args["neighbors"])
    if "repeats" in args:
        settings["repeats"] = int(args["repeats"])
    if "k" in args:
        settings["max_k"] = int(args["k"])

    # Point-query convenience: variables field contains assignments for every design variable.
    if base == "vary" and detail == "point":
        if not assignments:
            raise ValueError("vary(point) requires assignments in variables, e.g. glucose=5, oxygen=10, ...")
        settings["point"] = assignments
        # point query must resolve all design vars in the backend; all-design mode is correct here.
        variable_mode = "all_design"
        variables = []

    # Correlation and dimensionality treat selected observables/variables as features.
    variable_labels = {e.canonical: e.display for e in registry.entries if e.kind == "variable"}
    q = UnifiedQuestion(
        name=req.name or sanitize_name(f"{base}_{detail}"),
        landscape_path=req.landscape_path,
        gem_model_path=req.gem_model_path,
        primitive=primitive,
        detail=detail,
        targets=targets,
        derived=derived,
        variable_mode=variable_mode,
        variables=variables,
        conditions=conditions,
        condition_logic=logic,
        region_a=region_a,
        region_b=region_b,
        settings=settings,
        backend_mode=req.backend_mode,
        surrogate={
            "model": "extra_trees",
            "n_estimators": int(req.n_estimators),
            "cv_folds": int(req.cv_folds),
            "min_r2": float(req.min_r2),
            "min_auc": float(req.min_auc),
            "max_train_rows": 50000,
            "n_jobs": -1,
        },
        variable_labels=variable_labels,
        output_dpi=int(req.output_dpi),
    )
    public, problem = compile_unified_question(q)

    # Materialize stored long-table flux evidence only when the formulation asks
    # for it. This keeps ordinary questions lightweight while allowing any stored
    # exchange/reaction flux discovered in the landscape to be an observable.
    canon = registry.by_canonical()
    required_long_columns: set[str] = set()
    long_prefixes = ("exchange_flux__", "reaction_flux__", "shadow_price__", "reduced_cost__")
    referenced = [t.value for t in targets] + [c.column for c in conditions] + [c.column for c in region_a] + [c.column for c in region_b]
    for token in referenced:
        e = canon.get(str(token))
        if e and e.column.startswith(long_prefixes):
            required_long_columns.add(e.column)
    for d in derived:
        expr = str(d.get("expression") or "")
        for e in registry.entries:
            if e.column.startswith(long_prefixes) and e.canonical in expr:
                required_long_columns.add(e.column)
    problem.setdefault("landscape", {})["include_fluxes"] = bool(required_long_columns)
    if required_long_columns:
        problem["landscape"]["selected_long_columns"] = sorted(required_long_columns)
        public.setdefault("evidence", {})["stored_long_columns_loaded_on_demand"] = sorted(required_long_columns)

    # Human-facing calculator question: use biological/display names rather than raw column aliases.
    target_text = ", ".join(t.label or t.name for t in targets) if targets else "selected designs"
    if variable_mode == "all_design":
        variable_text = "all design variables"
    else:
        variable_text = ", ".join(canon.get(v).display if canon.get(v) else v for v in variables)
    cond_bits = []
    for c in conditions:
        label = canon.get(c.column).display if canon.get(c.column) else c.column
        if c.op == "true":
            cond_bits.append(f"{label} is true")
        elif c.op == "false":
            cond_bits.append(f"{label} is false")
        elif c.op == "top_quantile":
            cond_bits.append(f"{label} is in the top {float(c.value)*100:.0f}%")
        elif c.op == "bottom_quantile":
            cond_bits.append(f"{label} is in the bottom {float(c.value)*100:.0f}%")
        else:
            cond_bits.append(f"{label} {c.op} {c.value}")
    condition_text = (" AND " if logic == "all" else " OR ").join(cond_bits) if cond_bits else "the full landscape"
    if base == "describe":
        calc_question = f"What does {target_text} look like under {condition_text}?"
    elif base == "relate":
        calc_question = f"How is {target_text} related to or controlled by {variable_text} under {condition_text}?"
    elif base == "vary":
        calc_question = f"How does {target_text} vary over {variable_text} under {condition_text}?"
    elif base == "compare":
        calc_question = f"How does {target_text} differ across the defined comparison under {condition_text}?"
    elif base == "robust":
        calc_question = f"How robust is {target_text} to local changes in {variable_text} under {condition_text}?"
    elif base == "select":
        calc_question = f"Which exact sampled designs satisfy or define the requested selection under {condition_text}?"
    elif base == "structure":
        calc_question = f"What structure is present in {target_text} over {variable_text} under {condition_text}?"
    else:
        calc_question = f"What multiobjective/Pareto structure exists among {target_text} under {condition_text}?"
    public["question"]["text"] = calc_question
    public["question"]["title"] = calc_question.rstrip("?")
    problem["question"]["text"] = calc_question
    problem["question"]["title"] = calc_question.rstrip("?")

    # Restore the calculator language as the public contract, even when an internal primitive was remapped.
    public["exploration_spec_version"] = "0.4"
    public["calculator"] = {
        "observe": req.observe,
        "explore": req.explore,
        "variables": req.variables,
        "conditions": req.conditions,
    }
    public["exploration"]["calculator_function"] = base
    public["compiled_execution"]["note"] = (
        "The internal operator is an execution detail. The public scientific contract is the four-field exploration calculator."
    )
    problem["calculator_exploration"] = {
        "spec_version": "0.5",
        "observe": req.observe,
        "explore": req.explore,
        "variables": req.variables,
        "conditions": req.conditions,
    }

    summary = {
        "observe": [t.label or t.name for t in targets],
        "explore_function": base,
        "compiled_public_primitive": primitive,
        "compiled_detail": detail,
        "internal_operator": (public.get("compiled_execution") or {}).get("internal_operator"),
        "variables": [variable_labels.get(v, v) for v in variables] if variables else ["all design variables"],
        "conditions": req.conditions.strip() or "none",
        "new_fba_calls": 0,
    }
    return CalculatorCompilation(req, q, public, problem, summary)


def autocomplete_entries(registry: SemanticRegistry, fragment: str, field: str, limit: int = 8) -> list[dict[str, str]]:
    kinds = {"variable"} if field == "variables" else {"observable", "variable"}
    hits = registry.search(fragment, kinds=kinds, limit=limit)
    return [
        {
            "canonical": e.canonical,
            "display": e.display,
            "kind": e.kind,
            "reaction_id": e.reaction_id or "",
            "label": f"{e.display}  [{e.reaction_id or e.canonical}]",
        }
        for e in hits
    ]


def active_fragment(text: str, field: str | None = None) -> str:
    """Return the semantic token currently being typed for autocomplete.

    OBSERVE may be a derived expression (for example ``product/glu`` or
    ``log1p(prod)``), so suggestions must follow the last arithmetic/function
    boundary rather than replacing the whole expression. CONDITIONS suggest the
    quantity on the left of the active comparator.
    """
    s = str(text or "")
    tail = re.split(r"[,;]|\bAND\b|\bOR\b", s, flags=re.I)[-1].strip()
    if field == "conditions":
        comp = re.search(r">=|<=|!=|==|=|>|<", tail)
        if comp:
            return tail[:comp.start()].strip()
        return tail
    if field == "observe":
        # Keep only the token after the most recent expression boundary.  Spaces
        # inside a biological display name are preserved because they are not a
        # boundary by themselves.
        m = list(re.finditer(r"[+*/%(),]", tail))
        if m:
            tail = tail[m[-1].end():]
        else:
            # A binary minus is an expression boundary; do not treat a leading
            # minus as one.
            minus = list(re.finditer(r"(?<=\S)\s*-\s*", tail))
            if minus:
                tail = tail[minus[-1].end():]
        return tail.strip()
    return tail


def _replace_expression_fragment(current: str, suggestion: str) -> str:
    """Replace only the active semantic token in an OBSERVE expression."""
    cur = str(current or "")
    # Work within the last top-level comma-separated observable expression.
    parts = split_top_level(cur, separators=(",",))
    if cur.rstrip().endswith(","):
        return cur.rstrip() + " " + suggestion
    token = parts[-1] if parts else cur
    token_start = cur.rfind(token) if token and token in cur else 0
    rel = token
    boundaries = list(re.finditer(r"[+*/%()]", rel))
    start = boundaries[-1].end() if boundaries else 0
    if not boundaries:
        minus = list(re.finditer(r"(?<=\S)\s*-\s*", rel))
        if minus:
            start = minus[-1].end()
    # Preserve whitespace immediately after an operator/function delimiter.
    while start < len(rel) and rel[start].isspace():
        start += 1
    return cur[: token_start + start] + suggestion


def insert_suggestion(current: str, suggestion: str, field: str) -> str:
    cur = str(current or "")
    if field == "observe":
        return _replace_expression_fragment(cur, suggestion)
    if field == "variables":
        parts = split_top_level(cur, separators=(",",))
        if not parts:
            return suggestion
        # If current ends in comma, append; otherwise replace final token.
        if cur.rstrip().endswith(","):
            return cur.rstrip() + " " + suggestion
        prefix = cur[: cur.rfind(parts[-1])] if parts[-1] in cur else ""
        return prefix + suggestion
    if field == "conditions":
        matches = list(re.finditer(r"[,;]|\bAND\b|\bOR\b", cur, flags=re.I))
        start = matches[-1].end() if matches else 0
        segment = cur[start:]
        comp = re.search(r">=|<=|!=|==|=|>|<", segment)
        if comp:
            lhs_end = start + comp.start()
            return cur[:start] + " " + suggestion + cur[start + comp.start():]
        return cur[:start] + (" " if start else "") + suggestion
    return suggestion


def build_calculator_line(observe: str, explore: str, variables: str, conditions: str) -> str:
    return f"observe: {observe.strip()} | explore: {explore.strip()} | variables: {variables.strip()} | conditions: {(conditions.strip() or 'none')}"


def _unwrap_field_value(value: str) -> str:
    """Remove one true outer wrapper without damaging calls like robust(10%)."""
    v = str(value or "").strip().strip("|").strip().strip(",;").strip()
    if len(v) >= 2 and v[0] in "([{" and v[-1] in ")]}":
        pairs = {"(": ")", "[": "]", "{": "}"}
        if pairs.get(v[0]) == v[-1]:
            depth = 0
            wraps_all = True
            for i, ch in enumerate(v):
                if ch == v[0]:
                    depth += 1
                elif ch == v[-1]:
                    depth -= 1
                    if depth == 0 and i != len(v) - 1:
                        wraps_all = False
                        break
            if wraps_all and depth == 0:
                return v[1:-1].strip()
    return v


def parse_calculator_line(text: str) -> dict[str, str]:
    """Parse the calculator line in any field order.

    Accepted examples include both the pipe form and the user's compact form:
      observe: product | explore: vary | variables: glucose, oxygen | conditions: growth>0.2
      observe:(product), conditions:(growth>0.2, oxygen>0.3), variables:(glucose, oxygen)

    EXPLORE is optional and defaults to `auto`; VARIABLES defaults to `all`.
    """
    raw = str(text or "").strip()
    if not raw:
        raise ValueError("Calculator line is empty")

    aliases = {
        "observe": "observe", "observation": "observe", "observations": "observe", "obs": "observe", "target": "observe", "targets": "observe",
        "explore": "explore", "analysis": "explore", "question": "explore",
        "variable": "variables", "variables": "variables", "over": "variables",
        "condition": "conditions", "conditions": "conditions", "under": "conditions",
    }
    # Detect named sections globally rather than splitting on commas, because commas
    # are valid inside variables and conditions.
    keys_alt = "|".join(sorted((re.escape(k) for k in aliases), key=len, reverse=True))
    pattern = re.compile(rf"(?i)(?<![A-Za-z0-9_])({keys_alt})\s*:\s*")
    matches = list(pattern.finditer(raw))
    if not matches:
        raise ValueError("Use at least observe:, e.g. observe: product, variables: glucose, oxygen")

    out: dict[str, str] = {}
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw)
        key = aliases[m.group(1).lower()]
        value = _unwrap_field_value(raw[start:end])
        if key in out:
            raise ValueError(f"Calculator field {key!r} was provided more than once")
        out[key] = value

    if not out.get("observe"):
        raise ValueError("Missing calculator field: observe")
    out.setdefault("explore", "auto")
    out.setdefault("variables", "all")
    out.setdefault("conditions", "none")
    return out

