# Frozen Scientific Protocol — Growth × Environmental Robustness

## Purpose

Test whether GHOST, SPARC, and TRAIL improve sample-efficient optimization relative to `BASE_GSMOBO_QPAREGO_50K` when the **same previously generated 50k GEM landscape** is reused for a different biological query.

## Robustness definition

For candidate design `x`, the four stress transformations act only on input variables already represented in the original design/surrogate space. In normalized coordinates, selected stress variables are transformed as:

`x'_j = 0.50 * x_j`

Because normalized lower bound is zero, this is equivalent in physical units to:

`x'_j = L_j + 0.50 * (x_j - L_j)`

where `L_j` is the original sampled lower bound. No transformed point leaves the learned domain.

For each new live candidate:

- objective 1 = exact nominal maximum biomass growth;
- exact carbon-stressed growth is solved;
- exact oxygen-stressed growth is solved;
- exact nitrogen-stressed growth is solved;
- exact phosphate-stressed growth is solved;
- objective 2 = minimum of those four exact stressed-growth values.

A nominally feasible candidate with an infeasible stress scenario receives zero growth for that stress scenario, rather than being removed from the nominal feasible set.

Secondary retention metric:

`worst_case_stressed_growth / nominal_growth`

Retention is descriptive and is not the optimized robustness objective.

## Stress classes by species

| Species | Carbon | Oxygen | Nitrogen | Phosphate |
|---|---|---|---|---|
| Yeast | `r_1714` | `r_1992` | `r_1654` | `r_2005` |
| P. putida | `EX_glc__D_e` | `EX_o2_e` | `EX_nh4_e` | `EX_pi_e` |
| C. glutamicum | `EX_glc__D_e` | `EX_o2_e` | `EX_nh4_e`, `EX_urea_e` | `EX_pi_e` |
| CHO | `EX_glc(e)` | `EX_o2(e)` | all 17 sampled amino-acid nutrient inputs | `EX_pi(e)` |
| E. coli antiEpEX-scFv | `EX_glc__D_e` | `EX_o2_e` | `EX_nh4_e` | `EX_pi_e` |

These are matched **conceptual environmental stress classes**. They are not asserted to be chemically identical perturbations across organisms.

## Frozen 50k prior reuse

The original nominal-growth labels remain exact. Robustness values for the 50k prior are **not** new exact FBA labels. They are derived by querying a frozen growth-only surrogate at the four in-domain stress transformations and taking the worst predicted stressed growth.

No 50k stress-FBA relabeling is permitted in this benchmark.

This makes the experiment directly test the GEM Engine reuse concept: one expensive exact landscape supports a new downstream biological query.

## Live exact oracle

Every new optimizer candidate is evaluated by exact FBA. Therefore the primary optimizer-performance metrics are based only on exact live results.

## Primary comparison

Reference method: `BASE_GSMOBO_QPAREGO_50K`.

Primary outcomes:

1. method-only exact normalized hypervolume at each matched call budget;
2. call-averaged method-only exact-HV AUC over calls 1..128;
3. percentage gain of GHOST/SPARC/TRAIL versus gsMOBO+50k within each species;
4. cross-species win count.

Prior-relative delta-HV is secondary because objective 2 in the frozen prior is surrogate-derived.

## Locked budget

- seed: 20260822
- batch size: 4
- milestones: 4, 16, 32, 64, 128
- seven methods
- five species
- no scientific-performance-based early stopping
- technical failure can stop a case, with the failure recorded

## Interpretation boundary

This benchmark measures **robustness to nutrient/environmental limitation represented in the original surrogate inputs**. It does not measure generic cellular stress tolerance and must not be described as robustness to pH, temperature, ROS, antibiotics, osmotic stress, or other unrepresented dimensions.
