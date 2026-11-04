# Scientific protocol — Growth × Product × Resource Efficiency

## Prespecified question

Does adding GEM-landscape structural acquisition restarts (GHOST, SPARC, TRAIL) improve sample efficiency over the strong `gsMOBO+50k` reference when the optimization problem is expanded from two to three objectives?

## Objectives

For a design vector x within the frozen original medium bounds:

- maximize `maximum_growth(x)` by exact FBA;
- maximize `product_flux(x)` by exact product optimization subject to biomass >= 0.9 × `maximum_growth(x)`;
- minimize `resource_cost(x) = mean_j((x_j-L_j)/(U_j-L_j))`.

For optimization/hypervolume, objective 3 is represented as `resource_efficiency = 1-resource_cost`.

Equal normalized weights are used across medium variables. This is a resource-use burden metric, not a claim about monetary cost.

## Frozen 50k reuse

The original 50k growth/product outcomes are reused exactly. Objective 3 is deterministic from the archived input vectors. No 50k cost relabeling FBA is permitted.

A small exact replay sample verifies that the current model/config reproduces both stored growth and stored growth-constrained product values before any live search.

## Surrogate contract

Only growth and product are learned by the global ExtraTrees surrogate and by gsMOBO GPs. Resource efficiency is known exactly at proposal time and is inserted deterministically into predictions and qParEGO acquisition scalarization.

Five-fold OOF QC is required before live optimization. OOF objective 3 must agree with the analytic formula to machine precision.

## Method panel and budget

Seven methods; matched 4/16/32/64/128 logical evaluations per organism. All methods passing technical gates continue through the same checkpoints regardless of scientific performance.

Primary comparator for GHOST/SPARC/TRAIL: `BASE_GSMOBO_QPAREGO_50K`.

## Primary endpoints

1. Method-only exact normalized 3-D hypervolume at 128 calls.
2. Call-averaged method-only exact 3-D HV AUC over calls 1…128.
3. Wins/losses/ties against gsMOBO+50k across the five organisms.

Secondary endpoints include checkpoint HV, exact archive-front extensions, best growth, best product, minimum resource burden among discovered feasible designs, feasible fraction, runtime, logical evaluations, and physical FBA solves.

## Interpretation guardrails

- One seed is architecture/generalization evidence, not a statistical superiority claim.
- The 50k archive is a high-coverage sampled reference, not a certified global Pareto front.
- No organism-specific optimizer hyperparameter tuning is allowed after seeing outcomes.
- Do not convert normalized resource burden into dollars without an explicit, sourced price model.
- Preserve negative results. Do not modify methods after seeing one species and continue calling the comparison prespecified.
