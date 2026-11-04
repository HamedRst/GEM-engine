# Source provenance

This package is an isolated derivative of `GEM_5Species_Growth_Product_ResourceCost_Benchmark_v0.1.0`, which generalized the previously frozen gsMOBO/NSGA-II/GHOST/SPARC/TRAIL implementation to exact three-objective search.

No original GEM Learning Engine source file or completed 50k output is modified.

Changes in this package are limited to the new biological problem formulation and the deterministic-objective plumbing required for:

`[product, carbon efficiency, oxygen efficiency]`

with a frozen global high-growth feasibility constraint.

The carbon/oxygen reaction mappings are carried forward from the previously audited five-species environmental-robustness protocol:

- Yeast: `r_1714`, `r_1992`
- P. putida: `EX_glc__D_e`, `EX_o2_e`
- C. glutamicum: `EX_glc__D_e`, `EX_o2_e`
- CHO: `EX_glc(e)`, `EX_o2(e)`
- E. coli antiEpEX-scFv: `EX_glc__D_e`, `EX_o2_e`

The seven-method comparison panel, seed, batch size, matched checkpoints, and primary gsMOBO+50k reference are preserved.
