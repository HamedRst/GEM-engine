# Exact-solver timeout/fallback hotfix

This technical hotfix does **not** change gsMOBO, GHOST, SPARC, TRAIL, seeds, manifests, objectives, bounds, or candidate proposals.

Uniform exact-backend policy for every method/cell:

1. Evaluate the source exact batch with the source benchmark's default solver.
2. If the whole batch exceeds 30 s, terminate it and retry the exact same candidates individually.
3. If an individual candidate exceeds 30 s under the default solver, terminate that attempt and retry the **same candidate / same LP** with `glpk_exact` in a fresh process.
4. `glpk_exact` has a 60 s hard timeout. If it also fails/times out, execution fails closed. No candidate is skipped, replaced, or marked infeasible merely because a solver timed out.
5. Output records include `source_solver_route`, `source_solver_fallback_used`, `source_solver_isolated_retry`, `source_primary_timeout_seconds`, `source_fallback_solver`, and `source_exact_attempt_wall_seconds`.

The policy was introduced after a reproducible Yeast Product×Cost point stalled standard GLPK for hours while `glpk_exact` solved the same point successfully.
