# Activity atlas — science and UI contract (v0.6.2)

## Scope
An additive section within the original Network Visualization tab. It does NOT replace or modify the native GEM Atlas visual style, Network YAML, query-pool actions, Calculator, adaptive QC, or solvers.

## Built-in dataset
The packaged demo is engineered *E. coli* antiEpEX-scFv, original seed-42 50K product-conditioned landscape. Region membership and population counts are computed using archived 49,236 feasible growth and product coordinates; activity summaries derive from a pre-quantized 12,000-environment sample embedded in the viewer, NOT from independently recomputed all-condition raw fluxes. Other GEMs are not dynamically loaded into this viewer.

## Two independently configurable segmentation methods
- ANGULAR: normalize growth and product by their own positive observed maxima, then angle θ=atan2(product_norm, growth_norm). Divide [0,90°] into `n_angular_regions` equal angles (2–12; default 6). Always include low-magnitude points: sectors indicate orientation only, not magnitude or within-medium physiological tradeoffs. Region 1: growth-biased ratio; region n: product-biased ratio.
- CIRCULAR: r=sqrt(growth_norm²+product_norm²). Divide r by landscape empirical quantiles into `n_circular_regions` rings (2–12; default 6). Region 1 low combined output magnitude; region n high magnitude. NOT distance to the Pareto frontier, and not guaranteed nondominated.

## Molecular interpretation
Gene layer = modeled GPR-linked reaction-use proxy, not transcription, enzyme abundance, or knockout necessity; some genes map to multiple reactions and have ambiguity status. Metabolite layer = flux-derived turnover/material processing, not concentration. Values are descriptive of optimizer-selected flux states. Sorting within each heatmap is the signed first-to-last standardized change for that segmentation; genes and metabolites are independently sorted for angular and circular heatmaps. An all-feature searchable row browser with click-to-landscape color overlay is provided.

## Manuscript limitations
The current static interface is an illustrative E. coli dataset. For publication comparisons, reaggregate all feasible archived flux rows using verified original GEM and explicit multi-reaction gene definitions; then test objective robustness/FVA and environment-matched comparisons. Neither visual pattern similarity nor color maps alone establishes causality or novelty.

## Technical details
NiceGUI startup: `gem_exploration_hub.app` registers static assets at `/gem-exploration-activity-assets` and displays `render_activity_section()` within the original Network Visualization tab under a collapsed expansion. The assets work offline in modern browsers with DecompressionStream; the large sample data is bundled locally. Dynamic region count 2–12 changes segmentation and regional summaries, without new FBA. Figures, six-region CSVs, and builders included at `examples/activity_atlas_ecoli/`.
