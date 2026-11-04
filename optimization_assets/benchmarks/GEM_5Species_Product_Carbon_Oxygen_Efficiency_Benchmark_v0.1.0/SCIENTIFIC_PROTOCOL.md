# Scientific protocol — Product × Carbon Efficiency × Oxygen Efficiency

## Prespecified question

For each of five species, optimize a three-objective bioprocess-efficiency problem over the original frozen medium/design space:

- maximize growth-constrained product flux;
- minimize carbon-input demand;
- minimize oxygen-input demand;

subject to an absolute growth-viability constraint.

## Growth viability

Let `mu_max,50k` be the maximum exact maximum-growth value in the frozen 50k archive for a species. Before any live optimization:

`mu_floor = 0.90 * mu_max,50k`.

A design is benchmark-feasible only if its exact maximum growth is at least `mu_floor` and its exact product solve succeeds. This threshold is frozen and may not be changed after observing optimizer results.

The archive product label is the original GEM Learning Engine growth-constrained product flux, obtained by maximizing product after constraining biomass to at least 90% of that design's own maximum growth.

## Carbon / oxygen definitions

The two resource objectives use only variables already in the frozen design space. For the named carbon/oxygen exchange coordinate:

`demand_j(x) = clip((x_j-L_j)/(U_j-L_j), 0, 1)`

`efficiency_j(x) = 1-demand_j(x)`.

The optimizer therefore maximizes:

`Y = [product/product_scale, carbon_efficiency, oxygen_efficiency]`.

The normalized hypervolume reference is `[0,0,0]`.

## Species mappings

| Species | Carbon | Oxygen |
|---|---|---|
| S. cerevisiae / yeast-GEM | `r_1714` | `r_1992` |
| P. putida / iJN1463 | `EX_glc__D_e` | `EX_o2_e` |
| C. glutamicum / iCGB21FR | `EX_glc__D_e` | `EX_o2_e` |
| CHO / iCHO2441 | `EX_glc(e)` | `EX_o2(e)` |
| E. coli / iJO1366 antiEpEX-scFv | `EX_glc__D_e` | `EX_o2_e` |

These mappings were frozen from the previously audited environmental-stress mappings; they are not inferred positionally.

## Archive reuse

No new 50k landscape is generated. The existing exact 50k maximum-growth and product labels are reused. Carbon and oxygen objectives are deterministic transforms of the original coordinates. Therefore archive objective construction requires **0 additional FBA solves**.

Small exact replay QC is allowed and recorded separately as provenance; it is not optimizer budget.

## Methods and budgets

Seven methods are compared under matched logical budgets:

- gsMOBO
- gsMOBO+50k
- NSGA-II
- NSGA-II+50k
- GHOST
- SPARC
- TRAIL

Checkpoints: 4, 16, 32, 64, 128 logical candidate evaluations per method/species.

All technically valid methods continue through the same checkpoints. Scientific underperformance never causes selective early stopping.

## Surrogate policy

Only the expensive product objective is regressed. Carbon and oxygen efficiencies are evaluated exactly from candidate coordinates. A separate classifier estimates benchmark infeasibility, including the global growth threshold.

For gsMOBO/qParEGO, the product objective is GP-modeled while carbon and oxygen are inserted analytically into the three-objective scalarization. NSGA-II and GHOST/SPARC/TRAIL operate on the same exact normalized three-objective vectors.

## Primary outcomes

Primary:

- method-only exact normalized 3D hypervolume at each checkpoint;
- exact-HV AUC across logical calls;
- paired endpoint/AUC change vs gsMOBO+50k.

Secondary:

- exact feasible fraction;
- exact Pareto/front extensions;
- best exact product;
- minimum normalized carbon demand;
- minimum normalized oxygen demand;
- physical FBA solves;
- runtime.

## Interpretation guardrails

This benchmark tests performance on a specific Pareto geometry: production versus carbon and oxygen supply demand under high-growth viability. A method that wins here is not automatically universal. Results must be interpreted together with the previous growth/product/resource-cost and robustness problems.
