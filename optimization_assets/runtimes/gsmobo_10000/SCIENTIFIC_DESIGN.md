# Scientific contract — 10K selected-heuristic landscape-size ablation

Only prior archive size changes relative to the working five-species 5K experiment.

- Prior rows: 10000. Verified authoritative Sobol prefix of the same 50K archive.
- Reference semantics rows: 50000. Normalization, qXX/frozen constraints, viability thresholds and HV reference are compiled from the full 50K source first and then applied unchanged to the 10K prefix.
- Same five species and same five problem definitions (25 cells).
- Same seed 20260822, 16 common initialization calls, batch size 4 and endpoint 128.
- Same exact FBA backend and cache policy.
- Same algorithm source; selected heuristic implementations are exactly GHOST Legacy VB, SPARC A and TRAIL True Native v1.1.1.
- NO-PRIOR candidate generation remains live-history-only. NO-PRIOR reported HV is recomputed from live exact calls only and never unioned with the 10K or 50K Pareto front.
- Current-prior qParEGO remains a reference comparator.
