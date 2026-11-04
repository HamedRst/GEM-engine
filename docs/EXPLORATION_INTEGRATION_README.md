# Exploration integration

The main GEM Engine interface includes Exploration immediately after Optimization.
The Exploration page provides a shared YAML question workflow for the Calculator,
Network Atlas, and gene/reaction/metabolite Activity Atlas. Figures and results
are generated only when a question is run.

Exploration runs under the same local server as GEM Engine. Results are stored
outside the application under the user-selected workspace.

The optimization core and frozen inputs are preserved from the current M5 build.
The supported release method panel contains five gsMOBO and five NSGA-II methods.
