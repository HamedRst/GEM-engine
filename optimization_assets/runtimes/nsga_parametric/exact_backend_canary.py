#!/usr/bin/env python3
from __future__ import annotations

"""Contract-aware exact-backend canary for GEM NSGA parametric-prior instances.

For the four problem families whose frozen archive values are exact outcomes,
all normalized objectives must replay under the live exact backend.

For ``growth_robustness`` the historical source benchmark has an intentionally
asymmetric contract:
  * frozen landscape objective-2 information is surrogate-derived prior knowledge;
  * new/live candidates are evaluated by exact FBA under stress.
Therefore objective 2 is NOT required to equal its frozen-prior value.  Instead:
  1) nominal growth (objective 0) must exact-replay;
  2) the frozen prior oxygen-retention value is independently re-derived from
     ``prior_oxygen_stressed_growth / maximum_growth``;
  3) the live exact oxygen-retention value is independently re-derived from
     ``oxygen_stressed_growth / maximum_growth`` and must equal the generalized
     bridge's exact normalized objective 2.

This matches the already-used GEM robustness canary contract
``nominal_exact_plus_surrogate_prior_to_exact_live_robustness``.
"""

import argparse
import hashlib
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

MARKER = "GEM_GSMOBO25_ROBUSTNESS_CANARY_CONTRACT_V073"
REPLAY_CONTRACT_ROB = "nominal_exact_plus_surrogate_prior_to_exact_live_robustness"
REPLAY_CONTRACT_FULL = "full_objective_exact_replay"
ROB_BENCH = "GEM_5Species_Growth_Environmental_Robustness_Benchmark_v0.2.0"
ROB_ENGINE_SHA = "573fedab092da2df4b99968b5c6adf452307a183fd4ccd73d5edb9228f631be3"
# Solver replay differences observed in the canonical exact canaries are well below
# these values; these are canary tolerances only and never modify scientific data.
REPLAY_ATOL = 2.0e-6
REPLAY_RTOL = 2.0e-6
RAW_RETENTION_ATOL = 2.0e-8
ARCHIVE_RETENTION_ATOL = 2.0e-9

HERE = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def core_table(prep: Path) -> Path:
    for name in ("baseline_compact.csv.gz", "baseline.csv.gz", "baseline_compact.csv", "baseline.csv"):
        q = prep / name
        if q.is_file():
            return q
    hits = sorted(prep.glob("*baseline*.csv*"))
    if len(hits) == 1:
        return hits[0]
    raise RuntimeError(f"Canary cannot identify one prepared baseline table in {prep}")


def bool_mask(s: pd.Series) -> np.ndarray:
    if s.dtype == bool:
        return s.fillna(False).to_numpy(bool)
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "optimal"]).to_numpy(bool)


def objective_columns(df: pd.DataFrame, objective_names: list[str]) -> list[str]:
    cols = [f"objective_norm__{n}" for n in objective_names]
    if all(c in df.columns for c in cols):
        return cols
    # Fail closed rather than silently guessing objective ordering.
    raise RuntimeError(
        "Canary expected normalized objective columns are absent. "
        f"Expected={cols}; available={list(df.columns)[:160]}"
    )


def choose_archive_row(df: pd.DataFrame, objective_names: list[str]) -> tuple[int, pd.Series]:
    cols = objective_columns(df, objective_names)
    Y = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    valid = np.all(np.isfinite(Y), axis=1)
    for c in ("exact_feasible", "feasible", "solver_feasible"):
        if c in df.columns:
            valid &= bool_mask(df[c])
            break
    ids = np.flatnonzero(valid)
    if not len(ids):
        raise RuntimeError("Canary found no feasible row with finite normalized objectives")
    # Match the canonical canary's deterministic middle-feasible-row selection.
    # This also lets a repaired target reuse any exact canary cache already created.
    idx = int(ids[len(ids)//2])
    return idx, df.iloc[idx]


def exact_status(row: pd.Series) -> str:
    for c in ("exact_status", "status", "solver_status", "source_status", "nominal_status"):
        if c in row.index and pd.notna(row[c]):
            return str(row[c])
    return "unknown"


def physical_exact_evals(row: pd.Series) -> int | None:
    for c in ("physical_exact_evals", "physical_source_evals", "new_physical_source_evals"):
        if c in row.index:
            try:
                return int(float(row[c]))
            except Exception:
                pass
    for c in ("source_cache_hit", "exact_cache_hit", "cache_hit"):
        if c in row.index:
            try:
                hit = str(row[c]).strip().lower() in {"true", "1", "yes", "y"}
                return 0 if hit else 1
            except Exception:
                pass
    return None


def _finite_float(x, name: str) -> float:
    try:
        v = float(x)
    except Exception as e:
        raise RuntimeError(f"Canary {name} is not numeric: {x!r}") from e
    if not math.isfinite(v):
        raise RuntimeError(f"Canary {name} is not finite: {v!r}")
    return v


def _retention(oxygen_growth: float, nominal_growth: float) -> float:
    n = _finite_float(nominal_growth, "nominal_growth")
    o = _finite_float(oxygen_growth, "oxygen_stressed_growth")
    if n <= 1.0e-12:
        raise RuntimeError(f"Canary nominal growth too small to define retention: {n}")
    return float(np.clip(o / n, 0.0, 1.00000001))


def verify_robustness_source_identity(case: str) -> tuple[Path, str]:
    species = case.split("__", 1)[0]
    source_case = species + "_growth_robustness"
    runtime_root = HERE / "runtime_exact_sources" / ROB_BENCH
    engine = runtime_root / "engine_bridge.py"
    if not engine.is_file() or sha256(engine) != ROB_ENGINE_SHA:
        raise RuntimeError("Canary exact historical robustness engine SHA mismatch")
    pconf = runtime_root / "prepared" / source_case / "problem_config.yaml"
    baseline = runtime_root / "prepared" / source_case / "baseline_compact.csv.gz"
    if not pconf.is_file() or not baseline.is_file():
        raise RuntimeError(f"Canary runtime historical robustness source incomplete for {source_case}")
    return runtime_root, source_case


def archive_prior_retention(case: str, design_index: int, expected_obj2: float) -> float:
    runtime_root, source_case = verify_robustness_source_identity(case)
    src = read_table(runtime_root / "prepared" / source_case / "baseline_compact.csv.gz")
    if "design_index" not in src.columns:
        raise RuntimeError(f"Historical robustness baseline lacks design_index: {source_case}")
    hit = src.loc[pd.to_numeric(src["design_index"], errors="coerce") == int(design_index)]
    if len(hit) != 1:
        raise RuntimeError(f"Historical robustness baseline design_index {design_index} lookup returned {len(hit)} rows: {source_case}")
    r = hit.iloc[0]
    for c in ("maximum_growth", "prior_oxygen_stressed_growth"):
        if c not in r.index:
            raise RuntimeError(f"Historical robustness baseline missing {c}: {source_case}")
    ret = _retention(r["prior_oxygen_stressed_growth"], r["maximum_growth"])
    if not math.isclose(ret, float(expected_obj2), rel_tol=0.0, abs_tol=ARCHIVE_RETENTION_ATOL):
        raise RuntimeError(
            f"{case}: frozen prior oxygen-retention semantic mismatch; "
            f"rederived={ret:.16g} prepared_objective2={float(expected_obj2):.16g}"
        )
    return ret


def direct_historical_robustness_live(case: str, proposal: pd.DataFrame) -> pd.Series:
    """Fallback/independent exact robustness evaluation using the SHA-locked historical engine."""
    runtime_root, source_case = verify_robustness_source_identity(case)
    pconf = yaml.safe_load((runtime_root / "prepared" / source_case / "problem_config.yaml").read_text()) or {}
    problem = pconf.get("problem") or {}
    run_dir = Path(str(problem.get("run_dir", ""))).expanduser()
    model_path = Path(str(problem.get("model_path", ""))).expanduser()
    study_config = runtime_root / "study_config.yaml"
    if not run_dir.is_dir():
        raise RuntimeError(f"{case}: historical runtime run_dir missing for direct exact robustness canary: {run_dir}")
    if not model_path.is_file():
        raise RuntimeError(f"{case}: historical runtime model_path missing for direct exact robustness canary: {model_path}")
    if not study_config.is_file():
        raise RuntimeError(f"{case}: historical runtime study_config missing for direct exact robustness canary: {study_config}")
    engine = runtime_root / "engine_bridge.py"
    with tempfile.TemporaryDirectory(prefix="gem_robustness_direct_canary_") as td:
        td = Path(td)
        prop = td / "proposal.csv"
        out = td / "exact_source.csv"
        proposal.to_csv(prop, index=False)
        cmd = [
            sys.executable, str(engine), "evaluate",
            "--run-dir", str(run_dir),
            "--model-path", str(model_path),
            "--proposal", str(prop),
            "--output", str(out),
            "--study-config", str(study_config),
            "--case-key", source_case,
        ]
        cp = subprocess.run(cmd, cwd=str(runtime_root))
        if cp.returncode != 0:
            raise RuntimeError(f"{case}: direct historical robustness engine returned {cp.returncode}")
        if not out.is_file():
            raise RuntimeError(f"{case}: direct historical robustness engine did not create output")
        df = pd.read_csv(out)
        if len(df) != 1:
            raise RuntimeError(f"{case}: direct historical robustness engine returned {len(df)} rows; expected 1")
        return df.iloc[0].copy()


def run_generic_exact(study_config: Path, case: str, proposal: pd.DataFrame) -> pd.Series:
    bridge = HERE / "multi_problem_bridge.py"
    if not bridge.is_file():
        raise RuntimeError(f"Canary missing generic bridge: {bridge}")
    with tempfile.TemporaryDirectory(prefix="gem_gsmobo25_canary_") as td:
        td = Path(td)
        prop = td / "proposal.csv"
        out = td / "exact.csv"
        proposal.to_csv(prop, index=False)
        cmd = [
            sys.executable, str(bridge), "evaluate",
            "--study-config", str(study_config),
            "--case-key", case,
            "--proposal", str(prop),
            "--output", str(out),
        ]
        cp = subprocess.run(cmd, cwd=str(HERE))
        if cp.returncode != 0:
            raise RuntimeError(f"{case}: generic exact bridge canary returned {cp.returncode}")
        if not out.is_file():
            raise RuntimeError(f"{case}: generic exact bridge did not create {out}")
        df = pd.read_csv(out)
        if len(df) != 1:
            raise RuntimeError(f"{case}: exact bridge returned {len(df)} rows; expected 1")
        return df.iloc[0].copy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study-config", required=True)
    ap.add_argument("--case", required=True)
    args = ap.parse_args()
    cfgp = Path(args.study_config).expanduser().resolve()
    case = str(args.case)
    if not cfgp.is_file():
        raise SystemExit(f"Canary study_config missing: {cfgp}")

    authp = HERE / "AUTHORITATIVE_GSMOBO25_CONTRACT.json"
    if not authp.is_file():
        raise SystemExit(f"Canary authoritative contract missing: {authp}")
    auth = json.loads(authp.read_text(encoding="utf-8"))
    if case not in auth.get("case_keys", []):
        raise SystemExit(f"Canary case is not in authoritative 25: {case}")
    problem_name = case.split("__", 1)[1]
    spec = auth["problems"][problem_name]
    objective_names = list(spec["objective_names"])

    cfg = yaml.safe_load(cfgp.read_text(encoding="utf-8")) or {}
    prepared_root = Path((cfg.get("paths") or {}).get("prepared_root", cfgp.parent / "prepared")).expanduser().resolve()
    prep = prepared_root / case
    if not prep.is_dir():
        raise SystemExit(f"Canary prepared case missing: {prep}")
    parentp = prep / "problem_config.yaml"
    if not parentp.is_file():
        raise SystemExit(f"Canary prepared problem_config missing: {parentp}")
    parent = yaml.safe_load(parentp.read_text(encoding="utf-8")) or {}
    p = parent.get("problem") or {}
    features = list(p.get("features") or [])
    if not features:
        raise SystemExit(f"Canary prepared problem config has no features: {case}")

    baseline = read_table(core_table(prep))
    idx, archive = choose_archive_row(baseline, objective_names)
    # Stable replay-only canary selection. CHO PCO design_index 2579 is inside every
    # supported N>=5000 prefix and has exact-replayed in the historical 5K/50K controls.
    # This changes only which archived row is used for the preflight canary; it never
    # changes landscape values, normalization, optimization, or replay tolerances.
    _override = {"cho__product_carbon_oxygen": 2579}.get(case)
    if _override is not None:
        if "design_index" not in baseline.columns:
            raise SystemExit(f"{case}: stable canary override requires design_index")
        _di = pd.to_numeric(baseline["design_index"], errors="coerce")
        _hit = baseline.loc[_di == int(_override)]
        if len(_hit) != 1:
            raise SystemExit(f"{case}: stable canary design_index {_override} lookup returned {len(_hit)} rows")
        _row = _hit.iloc[0]
        _cols0 = objective_columns(baseline, objective_names)
        if not np.all(np.isfinite([_finite_float(_row[c], c) for c in _cols0])):
            raise SystemExit(f"{case}: stable canary row {_override} has nonfinite objectives")
        if "feasible" in _row.index and not bool_mask(pd.Series([_row["feasible"]]))[0]:
            raise SystemExit(f"{case}: stable canary row {_override} is not feasible")
        idx = int(_hit.index[0]); archive = _row
        print(f"[canary-row-override] {case}: design_index={_override}", flush=True)
    cols = objective_columns(baseline, objective_names)
    expected = np.asarray([_finite_float(archive[c], c) for c in cols], float)
    design_index = int(float(archive["design_index"])) if "design_index" in archive.index else idx

    missing = [f for f in features if f not in archive.index]
    if missing:
        raise SystemExit(f"Canary archive row missing proposal features for {case}: {missing}")
    proposal = pd.DataFrame([{**{"proposal_id": "PREFLIGHT_CANARY"}, **{f: float(archive[f]) for f in features}}])
    exact = run_generic_exact(cfgp, case, proposal)

    # Require the same normalized objective columns emitted by the live bridge.
    exact_vals = []
    for c in cols:
        if c not in exact.index:
            raise SystemExit(f"{case}: exact bridge missing normalized objective column {c}; available={list(exact.index)[:180]}")
        exact_vals.append(_finite_float(exact[c], c))
    got = np.asarray(exact_vals, float)
    status = exact_status(exact)
    if not (status.lower().startswith("optimal") or str(exact.get("feasible", "")).lower() in {"true", "1", "yes"}):
        raise SystemExit(f"{case}: exact canary status is not feasible/optimal: {status!r}")

    robustness = problem_name == "growth_robustness"
    compare_idx = [0] if robustness else list(range(len(expected)))
    if not np.allclose(expected[compare_idx], got[compare_idx], rtol=REPLAY_RTOL, atol=REPLAY_ATOL, equal_nan=False):
        raise SystemExit(
            f"{case}: exact replay mismatch on required objective indices {compare_idx}; "
            f"expected={expected.tolist()} got={got.tolist()}"
        )

    report = {
        "status": "PASS",
        "case": case,
        "archive_row": design_index,
        "expected_normalized": expected.tolist(),
        "exact_normalized": got.tolist(),
        "exact_status": status,
        "physical_exact_evals": physical_exact_evals(exact),
        "replay_contract": REPLAY_CONTRACT_ROB if robustness else REPLAY_CONTRACT_FULL,
        "compared_objective_indices": compare_idx,
    }

    if robustness:
        if len(expected) != 2 or len(got) != 2:
            raise SystemExit(f"{case}: robustness objective dimension must be 2; expected={len(expected)} exact={len(got)}")
        prior_ret = archive_prior_retention(case, design_index, float(expected[1]))
        live_source = exact
        live_source_mode = "generic_bridge_raw_exact_columns"
        if not all(c in exact.index for c in ("oxygen_stressed_growth", "maximum_growth")):
            # Some generalized bridges can strip raw source columns after compiling objectives.
            # In that case, independently run the SHA-locked historical exact engine rather
            # than weakening the semantic check.
            live_source = direct_historical_robustness_live(case, proposal)
            live_source_mode = "direct_sha_locked_historical_engine_fallback"
        for c in ("oxygen_stressed_growth", "maximum_growth"):
            if c not in live_source.index:
                raise SystemExit(
                    f"{case}: exact robustness source missing {c}; "
                    f"cannot validate oxygen-retention contract. available={list(live_source.index)[:180]}"
                )
        live_oxygen = _finite_float(live_source["oxygen_stressed_growth"], "live oxygen_stressed_growth")
        live_nominal = _finite_float(live_source["maximum_growth"], "live maximum_growth")
        live_ret = _retention(live_oxygen, live_nominal)
        if not math.isclose(live_ret, float(got[1]), rel_tol=0.0, abs_tol=RAW_RETENTION_ATOL):
            raise SystemExit(
                f"{case}: live exact oxygen-retention semantic mismatch; "
                f"rederived={live_ret:.16g} bridge_objective2={float(got[1]):.16g}"
            )
        # If present, historical source must show the explicit asymmetric contract.
        runtime_root, source_case = verify_robustness_source_identity(case)
        sp = yaml.safe_load((runtime_root / "prepared" / source_case / "problem_config.yaml").read_text()) or {}
        spr = sp.get("problem") or {}
        if spr.get("prior_robustness_is_surrogate_derived") is not True or spr.get("live_robustness_is_exact_fba") is not True:
            raise SystemExit(f"{case}: historical robustness source does not certify surrogate-prior/exact-live asymmetry")
        report.update({
            "prior_objective_2_normalized": float(expected[1]),
            "live_exact_objective_2_normalized": float(got[1]),
            "objective_2_equality_expected": False,
            "archive_prior_retention_rederived": prior_ret,
            "live_oxygen_stressed_growth": live_oxygen,
            "live_nominal_growth": live_nominal,
            "live_retention_rederived": live_ret,
            "historical_robustness_engine_sha256": ROB_ENGINE_SHA,
            "historical_prior_mode": "surrogate_derived_oxygen_stress_prior",
            "live_mode": "exact_fba_oxygen_stress",
            "live_source_validation_mode": live_source_mode,
        })

    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
