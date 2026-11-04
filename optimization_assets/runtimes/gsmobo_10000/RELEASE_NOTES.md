# v0.4.2 10K hardened release notes

- Built from the completed original five-species 5K gsMOBO source.
- Only prior size changes to 10K; the canonical `gsmobo_algorithms.py` remains byte-identical (SHA256 `df02dd309c8712525c7e9c7f915ce8a43d4b34a4e7cfeeb0fb271e855e9665b7`).
- Executed heuristic implementations remain exactly GHOST Legacy VB, SPARC Systematic A, TRAIL True Native v1.1.1; NO-PRIOR and current-prior qParEGO remain scientific references.
- NO-PRIOR optimization proposals are unchanged; primary HV is live self-front only. Terminal/summary reporting is also corrected to self-front.
- Authoritative source `design_index` is now required to be exactly zero-based Sobol order (`0..49999`) before a 10K/25K prefix is accepted.
- Audit failure now propagates to final run failure.
- Aggregation now rejects missing/duplicate/noncontiguous call histories and missing checkpoints.
- Added strict `POSTRUN_VERIFY.py` checking the complete 25×5×3 result matrix and agreement between NO-PRIOR aggregate and audit.
- Top-level bundle preflights both 10K and 25K completely before beginning either live campaign.
- Added a fresh-process surrogate serialization/handoff canary: every saved `final_global_surrogate.joblib` is reloaded and used for real prepared-row predictions before live optimization, catching missing-module/joblib relocation failures.
