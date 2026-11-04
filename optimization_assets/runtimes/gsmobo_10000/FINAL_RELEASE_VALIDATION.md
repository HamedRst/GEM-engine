# v0.4.2 10K release validation

Validated before release:

- exactly 25 original cells: the original five species × the original five problems;
- canonical `gsmobo_algorithms.py` SHA256 `df02dd309c8712525c7e9c7f915ce8a43d4b34a4e7cfeeb0fb271e855e9665b7`, byte-identical to the completed working 5K package;
- all five problem YAML files, `MANIFEST_ALL25.csv`, `problem_catalog.yaml`, generic problem/core/surrogate code, surrogate QC, and exact-backend canary are byte-identical to the completed working 5K package;
- `multi_problem_bridge.py` preserves the working contiguous-`design_index` Sobol-prefix logic and verifies the selected feature rows against the authoritative `.npy` design cache;
- `runtime.py` differs only in NO-PRIOR display/summary scoring: the live self-front is reported instead of the old landscape-union HV. `current_delta_hv()` is used only for progress printing in the canonical algorithm source, so proposal generation is unchanged;
- selected heuristic implementations are exactly GHOST Legacy VB, SPARC A, and TRAIL True Native v1.1.1; NO-PRIOR and current-prior qParEGO remain scientific references;
- authoritative Sobol-prefix selection was self-tested at the actual prior size N=10,000 against a 50,000-row source and its `.npy` design cache;
- full-50K problem normalization, constraints, viability gates, and HV coordinate semantics remain frozen;
- corrected NO-PRIOR aggregate/audit contract is `LIVE_SELF_FRONT_ONLY_NO_LANDSCAPE_UNION`;
- a synthetic 128-call audit separated a live self-front from an intentionally inflated old union-HV diagnostic, confirming the correction does not use a landscape Pareto front;
- every final surrogate is reloaded in a **fresh Python process** and used to predict real prepared prior rows before live optimization, explicitly testing joblib/module handoff;
- aggregation rejects missing, duplicate, noninteger, or noncontiguous call histories and requires all 128 logical calls for every selected method;
- the independent audit requires NO-PRIOR calls exactly 1..128 for all 25 cases;
- `POSTRUN_VERIFY.py` requires the exact 25 × 5 × 3 = 375-row result matrix and agreement between consolidated NO-PRIOR HV and the independent audit at calls 32, 64, and 128;
- the top-level merge requires both campaigns to have `COMPLETE` run status and PASS post-run verification before producing combined outputs.

The real source-benchmark preparation, all-row OOF surrogate QC, BoTorch/GPyTorch smoke test, and exact-backend canaries require the user's local `gem-step4c` environment and original source benchmark folders. The top-level `RUN_BOTH.command` performs those deep checks for **both** 10K and 25K before either live 128-call campaign begins.
