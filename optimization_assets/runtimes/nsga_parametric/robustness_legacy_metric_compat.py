#!/usr/bin/env python3
from __future__ import annotations

import py_compile
import re
from pathlib import Path

MARKER = "GSMOBO25_OXYGEN_RETENTION_LEGACY_METRIC_COMPAT_V071"


def patch_generic_problem(path: Path) -> dict:
    """Add the historical GSMOBO `stress_tolerance` lookup adapter.

    The authoritative GSMOBO problem YAML is NOT changed.  The historical
    robustness benchmark already stores the oxygen-stressed growth explicitly:
      archive: prior_oxygen_stressed_growth
      live:    oxygen_stressed_growth
    The public objective is oxygen_limitation_growth_retention, so the legacy
    metric key is resolved as oxygen_stressed_growth / nominal maximum_growth.

    This patch is applied only to COPIES of the NSGA runtime; never the completed
    source package.  It is fail-closed if the expected KeyError anchor differs.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if MARKER in text:
        py_compile.compile(str(path), doraise=True)
        return {"status": "already_patched", "path": str(path), "marker": MARKER}

    # Anchor immediately before the exact metric KeyError observed in the
    # canonical generic_problem.py lineage.  Capture indentation defensively.
    pattern = re.compile(
        r'(?m)^(?P<indent>[ \t]*)raise KeyError\(f["\']Exact metric \{name!r\} not found; available=\{list\(self\.df\.columns\)\[:120\]\}["\']\)\s*$'
    )
    m = pattern.search(text)
    if not m:
        raise RuntimeError(
            f"FAIL-CLOSED: generic_problem.py compatibility anchor not found: {path}; "
            "completed NSGA source was not modified"
        )
    ind = m.group("indent")
    block = (
        f"{ind}# {MARKER}\n"
        f"{ind}# Authoritative YAML legacy metric key -> oxygen-limitation growth retention.\n"
        f"{ind}if str(name) == 'stress_tolerance':\n"
        f"{ind}    _oxygen_col = None\n"
        f"{ind}    for _candidate in ('prior_oxygen_stressed_growth', 'oxygen_stressed_growth'):\n"
        f"{ind}        if _candidate in self.df.columns:\n"
        f"{ind}            _oxygen_col = _candidate\n"
        f"{ind}            break\n"
        f"{ind}    if _oxygen_col is not None and 'maximum_growth' in self.df.columns:\n"
        f"{ind}        _nom = self.df['maximum_growth'].astype(float)\n"
        f"{ind}        _oxy = self.df[_oxygen_col].astype(float)\n"
        f"{ind}        _den = _nom.where(_nom.abs() > 1.0e-12)\n"
        f"{ind}        return (_oxy / _den).clip(lower=0.0, upper=1.00000001)\n"
    )
    text = text[:m.start()] + block + text[m.start():]
    path.write_text(text, encoding="utf-8")
    py_compile.compile(str(path), doraise=True)
    if MARKER not in path.read_text(encoding="utf-8"):
        raise RuntimeError(f"FAIL-CLOSED: compatibility marker missing after patch: {path}")
    return {"status": "patched", "path": str(path), "marker": MARKER}


def verify_generic_problem(path: Path) -> None:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    required = [
        MARKER,
        "prior_oxygen_stressed_growth",
        "oxygen_stressed_growth",
        "str(name) == 'stress_tolerance'",
        "maximum_growth",
    ]
    missing = [x for x in required if x not in text]
    if missing:
        raise RuntimeError(f"FAIL-CLOSED: legacy metric compatibility incomplete in {path}: {missing}")
    py_compile.compile(str(path), doraise=True)
