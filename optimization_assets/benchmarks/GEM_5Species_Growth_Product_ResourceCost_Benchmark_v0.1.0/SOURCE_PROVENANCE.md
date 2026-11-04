# Source provenance

This benchmark combines the portable 5-species execution/QC scaffolding developed for the completed growth–environmental-robustness benchmark with the exact growth-constrained-product formulation and four-comparator / GHOST-SPARC-TRAIL implementations from the prior cross-species optimizer packages.

Scientific source lineage:

- exact growth/product formulation: original GEM Learning Engine 50k run configuration and the v0.3.0 cross-species augmentation package;
- gsMOBO / gsMOBO+50k / NSGA-II / NSGA-II+50k: four-comparator cross-species implementation;
- GHOST / SPARC / TRAIL: `GEM_CrossSpecies_Four_GSMOBO_Augmentations_v0.3.0_development_split_isolated`;
- portable Mac input/replay/QC orchestration: the completed five-species robustness benchmark, with robustness-specific physics removed;
- new v0.1.0 work: true 3-objective generalization, exact 3-D Pareto/HV, deterministic normalized resource-burden objective, 3-D qParEGO scalarization, and five-species problem wiring including E. coli antiEpEX-scFv.

The frozen `GEM_Learning_Engine_v1.0.4` files are treated as read-only scientific inputs.
