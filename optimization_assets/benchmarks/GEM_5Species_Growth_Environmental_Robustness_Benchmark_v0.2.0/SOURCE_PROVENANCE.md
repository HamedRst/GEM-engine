# Source provenance

## Inherited exact optimizer implementations

The benchmark code was based on the exact user-supplied package:

`GEM_CrossSpecies_Four_GSMOBO_Augmentations_v0.3.0_development_split_isolated(1).zip`

The four literature/comparator implementations are consistent with the supplied:

`GEM_CrossSpecies_Live_Optimizer_v0.2.7_four_comparators_isolated(1).zip`

The inherited method IDs used here are:

- BASE_GSMOBO_QPAREGO_NO50K
- BASE_GSMOBO_QPAREGO_50K
- BASE_NSGA2_NO50K
- BASE_NSGA2_50K
- NEW_GSMOBO50K_GHOST_RESTART
- NEW_GSMOBO50K_SPARC_RESTART
- NEW_GSMOBO50K_TRAIL_RESTART

Core inherited files include `methods.py`, `augmented_gsmobo_methods.py`, `novel_methods.py`, `core.py`, and the resumable benchmark infrastructure.

## New benchmark-specific code

Added/modified for the five-species Growth × Environmental Robustness query:

- `stress_robustness.py`
- `engine_bridge.py`
- `prepare_cases.py`
- portable exact-run integration in `live_core.py`
- portable/renamed reporting in `benchmark_case.py`
- `aggregate_growth_robustness.py`
- `install_server_inputs.py`
- `verify_inputs.py`
- `server_preflight.py`
- server shell controllers and documentation

## Frozen landscape principle

No 50k archive is re-simulated under stress. The previous exact 50k nominal-growth landscape is reused, with a frozen nominal-growth surrogate queried at in-domain stress transforms to construct the robustness prior. All new live optimizer outcomes are exact FBA evaluations.
