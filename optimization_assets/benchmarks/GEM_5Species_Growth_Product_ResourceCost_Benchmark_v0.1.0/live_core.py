from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import qmc

from core import (
    GROWTH_CANDIDATES,
    PRODUCT_CANDIDATES,
    STATUS_CANDIDATES,
    GlobalSurrogate,
    candidate_hv_gain,
    detect_column,
    ensure_dir,
    feature_spec,
    hypervolume_2d,
    hypervolume_nd,
    infer_feasible,
    normalize_x,
    normalize_y,
    pareto_points,
    robust_scale,
)


def duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def unnormalize_x(X01: np.ndarray, bounds: np.ndarray) -> np.ndarray:
    X01 = np.asarray(X01, float)
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + X01 * (hi - lo)


def stable_method_offset(name: str) -> int:
    return sum((i + 1) * ord(c) for i, c in enumerate(name)) % 1_000_003


def sobol_points(n: int, d: int, seed: int) -> np.ndarray:
    n = int(n)
    if n <= 0:
        return np.empty((0, d), float)
    m = int(math.ceil(math.log2(n)))
    eng = qmc.Sobol(d=d, scramble=True, seed=int(seed))
    return np.asarray(eng.random_base2(m)[:n], float)


def key_for_x(x: np.ndarray, decimals: int = 12) -> tuple[float, ...]:
    return tuple(np.round(np.asarray(x, float), int(decimals)).tolist())


def _first_existing(paths):
    for p in paths:
        q = Path(p).expanduser()
        if q.exists():
            return q.resolve()
    return None


@dataclass
class ExactBatchResult:
    X01: np.ndarray
    Xphys: np.ndarray
    Yraw: np.ndarray
    Yn: np.ndarray
    feasible: np.ndarray
    status: list[str]
    cache_hit: np.ndarray
    raw: pd.DataFrame
    physical_exact_evals: int = 0
    physical_exact_wall_seconds: float = 0.0


class ExactRunner:
    """Portable exact evaluator for growth × product × resource efficiency."""

    def __init__(self, cfg: dict, parent_cfg: dict, outroot: Path, features: list[str], bounds: np.ndarray, case: dict):
        self.cfg = cfg
        self.parent_cfg = parent_cfg
        self.outroot = ensure_dir(outroot)
        self.features = list(features)
        self.bounds = np.asarray(bounds, float)
        self.case = dict(case)
        self.engine_run = Path(parent_cfg["problem"]["run_dir"]).expanduser().resolve()
        self.model_path = Path(parent_cfg["problem"]["model_path"]).expanduser().resolve()
        self.exact_python = Path(sys.executable).resolve()
        self.bridge = Path(__file__).with_name("engine_bridge.py").resolve()
        self.study_config = Path(case["study_config_path"]).expanduser().resolve()
        self.case_key = str(parent_cfg["problem"].get("case_key", case.get("case_key", "")))
        self.decimals = int(cfg["study"].get("exact_cache_round_decimals", 12))
        self.cache_path = self.outroot / "exact_cache.csv"
        self.cache = self._load_cache()

    def _load_cache(self):
        out = {}
        if not self.cache_path.exists():
            return out
        df = pd.read_csv(self.cache_path)
        xcols = [f"x01__{f}" for f in self.features]
        if not all(c in df.columns for c in xcols):
            return out
        for _, r in df.iterrows():
            x = np.asarray([r[c] for c in xcols], float)
            out[key_for_x(x, self.decimals)] = r.to_dict()
        return out

    def preflight(self) -> dict:
        problem_path = Path(self.case["prepared_dir"]) / "problem_config.yaml"
        landscape_pf = Path(self.case["prepared_dir"]) / "landscape_preflight.json"
        probe_ok = False
        probe_message = "not run"
        try:
            probe = subprocess.run(
                [str(self.exact_python), "-c", "import cobra; print('cobra-import-ok')"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=60,
            )
            probe_ok = probe.returncode == 0 and "cobra-import-ok" in probe.stdout
            probe_message = probe.stdout.strip()[-1000:]
        except Exception as exc:
            probe_message = f"{type(exc).__name__}: {exc}"
        checks = {
            "archive_run_exists": self.engine_run.exists(),
            "model_file_exists": self.model_path.exists(),
            "python_exists": self.exact_python.exists(),
            "python_can_import_cobra": probe_ok,
            "exact_bridge_exists": self.bridge.exists(),
            "prepared_problem_exists": problem_path.exists(),
            "landscape_preflight_exists": landscape_pf.exists(),
        }
        if landscape_pf.exists():
            try:
                lp = json.loads(landscape_pf.read_text())
                checks["landscape_preflight_pass"] = (
                    lp.get("status") == "PASS"
                    and bool(lp.get("exact_replay_qc_pass"))
                    and int(lp.get("rows", -1)) == int(self.cfg["resource_cost"].get("required_archive_rows", 50000))
                    and int(lp.get("archive_additional_fba_solves_for_cost", -1)) == 0
                )
            except Exception:
                checks["landscape_preflight_pass"] = False
        return {
            "status": "PASS" if all(checks.values()) else "FAIL",
            "checks": checks,
            "import_probe": probe_message,
            "resolved": {
                "archive_run": str(self.engine_run),
                "model_path": str(self.model_path),
                "python": str(self.exact_python),
                "exact_bridge": str(self.bridge),
                "prepared_problem": str(problem_path),
                "case_key": self.case_key,
                "objectives": self.parent_cfg["problem"].get("objective_names", []),
                "resource_cost_formula": self.parent_cfg["problem"].get("resource_cost_formula"),
            },
        }

    def _stream_command(self, cmd: list[str], log_path: Path):
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        with log_path.open("a", encoding="utf-8") as log:
            log.write("COMMAND: " + " ".join(cmd) + "\n")
            p = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            assert p.stdout is not None
            for line in p.stdout:
                print("[exact] " + line.rstrip(), flush=True)
                log.write(line)
            rc = p.wait()
        if rc != 0:
            raise RuntimeError(f"Exact growth/product/cost bridge failed with exit code {rc}. See {log_path}")

    def evaluate(self, X01: np.ndarray, method: str, batch_no: int) -> ExactBatchResult:
        X01 = np.atleast_2d(np.asarray(X01, float))
        Xphys = unnormalize_x(X01, self.bounds)
        rows = [None] * len(X01)
        cache_hit = np.zeros(len(X01), dtype=bool)
        new_local = []
        physical_exact_wall_seconds = 0.0
        exact = pd.DataFrame()
        for i, x in enumerate(X01):
            k = key_for_x(x, self.decimals)
            if k in self.cache:
                rows[i] = dict(self.cache[k]); cache_hit[i] = True
            else:
                new_local.append(i)

        if new_local:
            batch_root = ensure_dir(self.outroot / "_exact_runs" / method / f"batch_{batch_no:03d}")
            attempt = 1
            while (batch_root / f"attempt_{attempt:02d}").exists(): attempt += 1
            run_dir = ensure_dir(batch_root / f"attempt_{attempt:02d}")
            prop = pd.DataFrame(Xphys[np.asarray(new_local, int)], columns=self.features)
            prop.insert(0, "proposal_id", [f"{method}__b{batch_no:03d}__q{i+1:03d}" for i in range(len(prop))])
            proposal_path = run_dir / "proposal.csv"; result_path = run_dir / "exact_results.csv"; prop.to_csv(proposal_path, index=False)
            cmd = [
                str(self.exact_python), str(self.bridge), "evaluate",
                "--run-dir", str(self.engine_run), "--model-path", str(self.model_path),
                "--proposal", str(proposal_path), "--output", str(result_path),
                "--study-config", str(self.study_config), "--case-key", self.case_key,
            ]
            t0 = time.time(); self._stream_command(cmd, run_dir / "exact_bridge.log"); physical_exact_wall_seconds = float(time.time() - t0)
            exact = pd.read_csv(result_path)
            if len(exact) != len(new_local): raise RuntimeError(f"Exact bridge returned {len(exact)} rows for {len(new_local)} proposals")
            feasible = infer_feasible(exact, "maximum_growth", "product_flux")
            status_col = detect_column(exact.columns, STATUS_CANDIDATES)
            for j, local_i in enumerate(new_local):
                rec = {
                    "maximum_growth": float(pd.to_numeric(exact.iloc[j]["maximum_growth"], errors="coerce")),
                    "product_flux": float(pd.to_numeric(exact.iloc[j]["product_flux"], errors="coerce")),
                    "resource_cost": float(pd.to_numeric(exact.iloc[j].get("resource_cost", np.nan), errors="coerce")),
                    "resource_efficiency": float(pd.to_numeric(exact.iloc[j].get("resource_efficiency", np.nan), errors="coerce")),
                    "status": str(exact.iloc[j][status_col]) if status_col is not None else ("optimal" if feasible[j] else "infeasible"),
                    "feasible": bool(feasible[j]), "source_method": method, "source_batch": int(batch_no),
                    "fba_solves": int(pd.to_numeric(exact.iloc[j].get("fba_solves", 0), errors="coerce") or 0),
                    "solver_wall_seconds": float(pd.to_numeric(exact.iloc[j].get("solver_wall_seconds", np.nan), errors="coerce")),
                }
                for kk, f in enumerate(self.features):
                    rec[f"x01__{f}"] = float(X01[local_i, kk]); rec[f] = float(Xphys[local_i, kk])
                rows[local_i] = rec; self.cache[key_for_x(X01[local_i], self.decimals)] = dict(rec)
            pd.DataFrame(list(self.cache.values())).to_csv(self.cache_path, index=False)
            print(f"[exact] {method} batch {batch_no}: {len(new_local)} physical candidates, {int(pd.to_numeric(exact.get('fba_solves', pd.Series(dtype=float)), errors='coerce').fillna(0).sum())} FBA solves in {duration(physical_exact_wall_seconds)}", flush=True)

        out = pd.DataFrame(rows)
        g = pd.to_numeric(out["maximum_growth"], errors="coerce").to_numpy(float)
        p = pd.to_numeric(out["product_flux"], errors="coerce").to_numpy(float)
        feasible = out["feasible"].astype(bool).to_numpy()
        reff = pd.to_numeric(out["resource_efficiency"], errors="coerce").to_numpy(float)
        Yraw = np.column_stack([g, p, reff]); Yn = normalize_y(g, p, self.parent_cfg, resource_efficiency=reff); Yn[~feasible] = np.nan

        ledger_path = self.outroot / "exact_runtime_ledger.csv"
        physical_fba_solves = int(pd.to_numeric(exact.get("fba_solves", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()) if len(exact) else 0
        ledger_row = pd.DataFrame([{
            "method": method, "batch": int(batch_no), "logical_evals": int(len(X01)),
            "physical_exact_evals": int(len(new_local)), "physical_fba_solves": physical_fba_solves,
            "cache_hits": int(cache_hit.sum()), "physical_exact_wall_seconds": float(physical_exact_wall_seconds),
        }])
        ledger = pd.concat([pd.read_csv(ledger_path), ledger_row], ignore_index=True) if ledger_path.exists() else ledger_row
        ledger.to_csv(ledger_path, index=False)
        return ExactBatchResult(X01=X01, Xphys=Xphys, Yraw=Yraw, Yn=Yn, feasible=feasible,
                                status=out["status"].astype(str).tolist(), cache_hit=cache_hit, raw=out,
                                physical_exact_evals=int(len(new_local)), physical_exact_wall_seconds=float(physical_exact_wall_seconds))

@dataclass
class Pool:
    X: np.ndarray
    pred: np.ndarray
    pinf: np.ndarray
    safe: np.ndarray
    novelty: np.ndarray


class LiveContext:
    def __init__(
        self,
        cfg: dict,
        parent_cfg: dict,
        baseline: pd.DataFrame,
        baseline_path: Path,
        global_model: GlobalSurrogate,
        subset_local: np.ndarray,
        subset_oof_prior: np.ndarray | None,
        outdir: Path,
        exact_runner: ExactRunner,
        seed: int,
        budget: int,
    ):
        self.cfg = cfg
        self.parent_cfg = parent_cfg
        self.baseline = baseline.reset_index(drop=True)
        self.baseline_path = Path(baseline_path)
        self.global_model = global_model
        self.subset_local = np.asarray(subset_local, int)
        self.subset_oof_prior = subset_oof_prior
        self.outdir = ensure_dir(outdir)
        self.exact_runner = exact_runner
        self.seed = int(seed)
        self.budget = int(budget)
        self.batch_size = int(cfg["study"]["batch_size"])
        self.features, self.bounds = feature_spec(parent_cfg)
        self.feature_labels = dict(parent_cfg.get("problem", {}).get("feature_labels", {}))
        self.X0 = normalize_x(self.baseline[self.features].to_numpy(float), self.bounds)
        self.Y0 = normalize_y(self.baseline["maximum_growth"], self.baseline["product_flux"], parent_cfg, resource_efficiency=self.baseline["resource_efficiency"])
        self.objective_norm_columns = ["exact_growth_norm", "exact_product_norm", "exact_resource_efficiency_norm"]
        self.objective_dim = 3
        self.F0 = self.baseline["feasible"].astype(bool).to_numpy()
        self.good0 = self.F0 & np.isfinite(self.Y0).all(axis=1)
        self.baseline_front = pareto_points(self.Y0[self.good0])
        ref = tuple(parent_cfg.get("problem", {}).get("hv_reference_normalized", [0.0, 0.0, 0.0]))
        self.hv_ref = ref
        if len(ref) != self.objective_dim or any(abs(float(x)) > 1e-15 for x in ref):
            raise RuntimeError(f"Expected normalized HV reference [0,0,0], got {ref}")
        self.baseline_hv = hypervolume_nd(self.baseline_front, ref=np.asarray(ref, float))
        self.baseline_tree = cKDTree(self.X0)
        self.growth_floor = float(np.quantile(self.Y0[self.good0, 0], 0.75))
        self.safe_threshold = float(cfg["surrogate"]["infeasible_probability_threshold"])
        self.min_d50k = float(cfg["study"].get("min_distance_to_frozen_50k", 1e-8))
        self.min_dwithin = float(cfg["study"].get("min_distance_within_method", 1e-8))

    def feature_index(self, explicit: str | None = None, keywords=("oxygen", "o2"), required: bool = False) -> int | None:
        """Resolve a biologically named design variable without positional fallbacks."""
        if explicit:
            if explicit in self.features:
                return self.features.index(explicit)
            raise KeyError(f"Requested feature {explicit!r} is not in this problem: {self.features}")
        for i, f in enumerate(self.features):
            hay = (str(f) + " " + str(self.feature_labels.get(f, ""))).lower()
            if any(str(k).lower() in hay for k in keywords):
                return i
        if required:
            raise RuntimeError(
                f"Could not identify a feature matching {keywords}. Configure qd_feature explicitly; "
                f"features={[(f, self.feature_labels.get(f, '')) for f in self.features]}"
            )
        return None

    def method_dir(self, method: str) -> Path:
        return ensure_dir(self.outdir / method)

    def history(self, method: str) -> pd.DataFrame:
        p = self.method_dir(method) / "calls.csv"
        if p.exists():
            return pd.read_csv(p)
        return pd.DataFrame()

    def save_history(self, method: str, df: pd.DataFrame):
        df.to_csv(self.method_dir(method) / "calls.csv", index=False)

    def current_front(self, method: str) -> np.ndarray:
        h = self.history(method)
        front = self.baseline_front.copy()
        if len(h):
            good = h["exact_feasible"].astype(bool).to_numpy()
            Y = h[self.objective_norm_columns].to_numpy(float)
            good &= np.isfinite(Y).all(axis=1)
            if np.any(good):
                front = pareto_points(np.vstack([front, Y[good]]))
        return front

    def method_only_front(self, method: str) -> np.ndarray:
        h=self.history(method)
        if not len(h): return np.empty((0,self.objective_dim),float)
        good=h["exact_feasible"].astype(bool).to_numpy()
        Y=h[self.objective_norm_columns].to_numpy(float); good &= np.isfinite(Y).all(axis=1)
        return pareto_points(Y[good]) if np.any(good) else np.empty((0,self.objective_dim),float)

    def current_delta_hv(self, method: str) -> float:
        """Primary live metric: method-only exact 3-D HV at the current budget."""
        F=self.method_only_front(method)
        return float(hypervolume_nd(F, ref=np.asarray(self.hv_ref,float))) if len(F) else 0.0

    def _filter_new(self, X: np.ndarray, method: str, *, exclude_frozen_archive: bool = True) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        keep = np.ones(len(X), dtype=bool)
        if exclude_frozen_archive:
            d50, _ = self.baseline_tree.query(X, k=1)
            keep &= d50 > self.min_d50k
        h = self.history(method)
        if len(h):
            cols = [f"x01__{f}" for f in self.features]
            Xh = h[cols].to_numpy(float)
            tree = cKDTree(Xh)
            dh, _ = tree.query(X, k=1)
            keep &= dh > self.min_dwithin
        # Remove within-pool exact duplicates after rounding.
        seen = set()
        for i, x in enumerate(X):
            k = key_for_x(x, 12)
            if k in seen:
                keep[i] = False
            seen.add(k)
        return X[keep]

    def make_pool(self, method: str, batch_no: int, rows: int | None = None, seed_offset: int = 0) -> Pool:
        n = int(rows or self.cfg["study"]["pool_rows"])
        # Shared Sobol opportunity pool by batch; method-specific offset is only
        # added when explicitly requested. This reduces candidate-pool confounding.
        seed = self.seed + 1009 * int(batch_no) + int(seed_offset)
        X = sobol_points(max(n * 2, n + 64), len(self.features), seed)
        X = self._filter_new(X, method)
        if len(X) < n:
            X2 = sobol_points(n * 2, len(self.features), seed + 17)
            X = np.vstack([X, self._filter_new(X2, method)])
        X = X[:n]
        pred, pinf = self.global_model.predict(X)
        safe = np.isfinite(pred).all(axis=1) & np.isfinite(pinf) & (pinf <= self.safe_threshold)
        novelty = self.baseline_tree.query(X, k=1)[0]
        return Pool(X=X, pred=pred, pinf=pinf, safe=safe, novelty=novelty)

    def evaluate_batch(
        self,
        method: str,
        X01: np.ndarray,
        modes: list[str] | None = None,
        metadata: dict[str, np.ndarray | list | float | str] | None = None,
        *,
        exclude_frozen_archive: bool = True,
    ) -> pd.DataFrame:
        h = self.history(method)
        start_call = len(h)
        batch_no = start_call // self.batch_size + 1
        X01 = np.atleast_2d(np.asarray(X01, float))
        if len(X01) == 0:
            raise RuntimeError(f"{method}: candidate generator returned an empty batch")
        X01 = self._filter_new(X01, method, exclude_frozen_archive=exclude_frozen_archive)
        if len(X01) == 0:
            raise RuntimeError(f"{method}: proposed batch had no genuinely new points after archive/history filtering")
        if start_call + len(X01) > self.budget:
            X01 = X01[: self.budget - start_call]
        modes = list(modes or [method] * len(X01))[: len(X01)]

        pred, pinf = self.global_model.predict(X01)
        result = self.exact_runner.evaluate(X01, method, batch_no)
        archive_front = self.current_front(method)
        live_front = self.method_only_front(method)
        live_hv = hypervolume_nd(live_front, ref=np.asarray(self.hv_ref,float)) if len(live_front) else 0.0
        rows = []
        for i in range(len(X01)):
            before = float(live_hv)
            y = result.Yn[i]
            feas = bool(result.feasible[i] and np.isfinite(y).all())
            extends_archive = False
            if feas:
                # Positive contribution against the frozen+live front can be
                # decided by Pareto dominance without recomputing 50k HV.
                dominated = bool(len(archive_front) and np.any(np.all(archive_front >= y[None,:]-1e-14,axis=1)))
                extends_archive = not dominated
                archive_front = pareto_points(np.vstack([archive_front, y]))
                live_front = pareto_points(np.vstack([live_front, y])) if len(live_front) else y[None,:]
                live_hv = hypervolume_nd(live_front, ref=np.asarray(self.hv_ref,float))
            after = float(live_hv)
            rec = {
                "method": method,
                "call": start_call + i + 1,
                "batch": batch_no,
                "mode": modes[i] if i < len(modes) else method,
                "exact_feasible": feas,
                "status": result.status[i],
                "cache_hit": bool(result.cache_hit[i]),
                "pred_growth_norm": float(pred[i, 0]),
                "pred_product_norm": float(pred[i, 1]),
                "pred_resource_efficiency_norm": float(pred[i, 2]),
                "pred_p_infeasible": float(pinf[i]),
                "exact_growth": float(result.Yraw[i, 0]) if np.isfinite(result.Yraw[i, 0]) else np.nan,
                "exact_product": float(result.Yraw[i, 1]) if np.isfinite(result.Yraw[i, 1]) else np.nan,
                "exact_resource_efficiency": float(result.Yraw[i, 2]) if np.isfinite(result.Yraw[i, 2]) else np.nan,
                "exact_resource_cost": float(pd.to_numeric(result.raw.iloc[i].get("resource_cost", np.nan), errors="coerce")) if i < len(result.raw) else np.nan,
                "exact_growth_norm": float(y[0]) if np.isfinite(y[0]) else np.nan,
                "exact_product_norm": float(y[1]) if np.isfinite(y[1]) else np.nan,
                "exact_resource_efficiency_norm": float(y[2]) if np.isfinite(y[2]) else np.nan,
                "exact_hv_gain_this_call": float(max(0.0, after - before)),
                "exact_front_extension": bool(extends_archive),
                "method_only_hv_after": float(after),
                "archive_hv_after": np.nan,
                "delta_hv_after": float(after),
            }
            for j, f in enumerate(self.features):
                rec[f"x01__{f}"] = float(X01[i, j])
                rec[f] = float(result.Xphys[i, j])
            if metadata:
                for k, v in metadata.items():
                    if isinstance(v, (np.ndarray, list, tuple)) and len(v) == len(X01):
                        vv = v[i]
                        if np.isscalar(vv):
                            rec[k] = float(vv) if isinstance(vv, (float, np.floating)) else vv
                    elif np.isscalar(v):
                        rec[k] = v
            rows.append(rec)

        new = pd.DataFrame(rows)
        allh = pd.concat([h, new], ignore_index=True) if len(h) else new
        self.save_history(method, allh)
        return allh

    def summary(
        self, method: str, runtime_seconds_cumulative: float | None = None,
        runtime_seconds_this_session: float | None = None,
        exact_solver_wall_seconds: float | None = None,
    ) -> dict:
        h = self.history(method)
        if len(h) == 0:
            return {"method": method, "calls": 0, "status": "NO_CALLS"}
        feasible = h["exact_feasible"].astype(bool)
        final_hv = float(pd.to_numeric(h.iloc[-1].get("method_only_hv_after", h.iloc[-1].get("delta_hv_after", np.nan)), errors="coerce"))
        calls = int(len(h))
        extensions = int(h["exact_front_extension"].astype(bool).sum())
        delta_hv = float(final_hv)

        method_y = h.loc[feasible, self.objective_norm_columns].to_numpy(float) if feasible.any() else np.empty((0, self.objective_dim), float)
        method_y = method_y[np.isfinite(method_y).all(axis=1)] if len(method_y) else method_y
        method_only_hv = hypervolume_nd(method_y, ref=np.asarray(self.hv_ref, float)) if len(method_y) else 0.0
        delta_series = pd.to_numeric(h["delta_hv_after"], errors="coerce").to_numpy(float)
        ext_series = h["exact_front_extension"].astype(bool).to_numpy()
        first_positive = int(h.iloc[int(np.where(delta_series > 1e-15)[0][0])]["call"]) if np.any(delta_series > 1e-15) else np.nan
        first_extension = int(h.iloc[int(np.where(ext_series)[0][0])]["call"]) if np.any(ext_series) else np.nan

        out = {
            "method": method,
            "calls": calls,
            "exact_feasible_fraction": float(feasible.mean()),
            "exact_extensions": extensions,
            "extensions_per_call": float(extensions / calls) if calls else np.nan,
            "first_exact_extension_call": first_extension,
            "first_positive_delta_hv_call": first_positive,
            "final_method_only_exact_hv": final_hv,
            "final_archive_hv": np.nan,
            "final_delta_hv": delta_hv,
            "relative_hv_gain": np.nan,
            "delta_hv_per_logical_call": float(delta_hv / calls) if calls else np.nan,
            "method_only_hv": float(method_only_hv),
            "method_only_hv_fraction_of_frozen": float(method_only_hv / self.baseline_hv) if self.baseline_hv > 0 else np.nan,
            "method_only_hv_regret_to_frozen": float(self.baseline_hv - method_only_hv) if self.baseline_hv > 0 else np.nan,
            "best_exact_product": float(pd.to_numeric(h.loc[feasible, "exact_product"], errors="coerce").max()) if feasible.any() else np.nan,
            "minimum_exact_resource_cost": float(pd.to_numeric(h.loc[feasible, "exact_resource_cost"], errors="coerce").min()) if feasible.any() else np.nan,
            "best_exact_resource_efficiency": float(pd.to_numeric(h.loc[feasible, "exact_resource_efficiency"], errors="coerce").max()) if feasible.any() else np.nan,
            "best_exact_growth": float(pd.to_numeric(h.loc[feasible, "exact_growth"], errors="coerce").max()) if feasible.any() else np.nan,
            "best_exact_product_norm": float(pd.to_numeric(h.loc[feasible, "exact_product_norm"], errors="coerce").max()) if feasible.any() else np.nan,
            "best_exact_growth_norm": float(pd.to_numeric(h.loc[feasible, "exact_growth_norm"], errors="coerce").max()) if feasible.any() else np.nan,
            "best_exact_resource_efficiency_norm": float(pd.to_numeric(h.loc[feasible, "exact_resource_efficiency_norm"], errors="coerce").max()) if feasible.any() else np.nan,
            "physical_exact_calls": int((~h["cache_hit"].astype(bool)).sum()),
            "cache_hits": int(h["cache_hit"].astype(bool).sum()),
        }
        ledger_path = self.outdir / "exact_runtime_ledger.csv"
        if ledger_path.exists():
            ledger = pd.read_csv(ledger_path)
            mm = ledger[ledger["method"].astype(str) == str(method)]
            out["physical_fba_solves"] = int(pd.to_numeric(mm.get("physical_fba_solves", pd.Series(dtype=float)), errors="coerce").fillna(0).sum())
        else:
            out["physical_fba_solves"] = 0
        if runtime_seconds_cumulative is not None:
            out["runtime_seconds_cumulative"] = float(runtime_seconds_cumulative)
        if runtime_seconds_this_session is not None:
            out["runtime_seconds_this_session"] = float(runtime_seconds_this_session)
        if exact_solver_wall_seconds is not None:
            out["exact_solver_wall_seconds_cumulative"] = float(exact_solver_wall_seconds)
            if runtime_seconds_cumulative is not None:
                out["optimizer_overhead_seconds_cumulative"] = float(max(0.0, runtime_seconds_cumulative - exact_solver_wall_seconds))
        return out


def candidate_targets_from_front(front: np.ndarray, outward: float = 0.045, max_targets: int = 16):
    F = pareto_points(front)
    F = F[np.argsort(F[:, 0])]
    if len(F) < 2:
        return [(F[0] if len(F) else np.array([0.5, 0.5]), 1.0)]
    targets = []
    for a, b in zip(F[:-1], F[1:]):
        t = b - a
        gap = float(np.linalg.norm(t))
        if gap <= 1e-12:
            continue
        n = np.array([-t[1], t[0]], float)
        n /= max(np.linalg.norm(n), 1e-12)
        if np.sum(n) < 0:
            n = -n
        targets.append((0.5 * (a + b) + outward * n, gap))
    targets.sort(key=lambda z: z[1], reverse=True)
    if len(F) >= 3:
        for endpoint, neighbor in ((F[0], F[1]), (F[-1], F[-2])):
            t = endpoint - neighbor
            t /= max(np.linalg.norm(t), 1e-12)
            n = np.array([-t[1], t[0]])
            if np.sum(n) < 0:
                n = -n
            targets.append((endpoint + 0.04 * t + outward * n, float(np.linalg.norm(endpoint - neighbor))))
    return targets[:max_targets]


def greedy_predicted_hv(
    X: np.ndarray, pred: np.ndarray, safe: np.ndarray, front: np.ndarray, q: int,
    min_gain: float | None = None,
) -> np.ndarray:
    """Greedy predicted-HV selection.

    When ``min_gain`` is provided, selection stops rather than silently
    choosing zero-gain ties.  This is critical for static surrogate search:
    an all-zero acquisition surface must be handled by an explicit documented
    fallback, not by Sobol pool ordering.
    """
    available = np.asarray(safe, bool).copy()
    chosen = []
    planning = np.asarray(front, float).copy()
    for _ in range(int(q)):
        ids = np.where(available)[0]
        if not len(ids):
            break
        gain = candidate_hv_gain(planning, pred[ids])
        if not np.any(np.isfinite(gain)):
            break
        j = int(np.nanargmax(gain))
        best_gain = float(gain[j])
        if min_gain is not None and (not np.isfinite(best_gain) or best_gain <= float(min_gain)):
            break
        idx = int(ids[j])
        chosen.append(idx)
        available[idx] = False
        planning = pareto_points(np.vstack([planning, pred[idx]]))
    return np.asarray(chosen, int)


def phenotype_distance_to_front(pred: np.ndarray, front: np.ndarray) -> np.ndarray:
    """Euclidean distance in normalized phenotype space to the 2-D Pareto polyline.

    Used only as an explicit static fallback when the surrogate predicts no
    positive hypervolume gain.  Smaller values mean 'Pareto-near'.
    """
    Y = np.atleast_2d(np.asarray(pred, float))
    F = pareto_points(np.asarray(front, float))
    if len(F) == 0:
        return np.full(len(Y), np.inf)
    if len(F) == 1:
        return np.linalg.norm(Y - F[0][None, :], axis=1)
    F = F[np.argsort(F[:, 0])]
    out = np.full(len(Y), np.inf, float)
    for a, b in zip(F[:-1], F[1:]):
        ab = b - a
        den = float(np.dot(ab, ab))
        if den <= 1e-18:
            d = np.linalg.norm(Y - a[None, :], axis=1)
        else:
            t = np.clip(((Y - a[None, :]) @ ab) / den, 0.0, 1.0)
            proj = a[None, :] + t[:, None] * ab[None, :]
            d = np.linalg.norm(Y - proj, axis=1)
        out = np.minimum(out, d)
    return out


def choose_diverse_by_score(X: np.ndarray, score: np.ndarray, mask: np.ndarray, q: int, radius: float = 0.04) -> np.ndarray:
    score = np.asarray(score, float)
    available = np.asarray(mask, bool).copy()
    out = []
    for _ in range(int(q)):
        s = score.copy()
        s[~available] = -np.inf
        if out:
            d = cKDTree(X[np.asarray(out, int)]).query(X, k=1)[0]
            div = 1.0 - np.exp(-0.5 * (d / max(radius, 1e-6)) ** 2)
            good = np.isfinite(s)
            s[good] *= 0.30 + 0.70 * div[good]
        if not np.any(np.isfinite(s)):
            break
        idx = int(np.nanargmax(s))
        out.append(idx)
        available[idx] = False
    return np.asarray(out, int)


def exact_front_from_history(ctx: LiveContext, method: str) -> np.ndarray:
    return ctx.current_front(method)


def baseline_pareto_row_indices(ctx: LiveContext) -> np.ndarray:
    ids = np.where(ctx.good0)[0]
    out = []
    for y in ctx.baseline_front:
        d = np.max(np.abs(ctx.Y0[ids] - y[None, :]), axis=1)
        j = int(ids[int(np.argmin(d))])
        if j not in out:
            out.append(j)
    return np.asarray(out, int)
