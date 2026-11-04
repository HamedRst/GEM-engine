# SI-Landscape V3: landscape acquisition and computational cost

Input: original GEM Learning Engine landscape-generation runs only, seed 42/43/44 ×
C. glutamicum, CHO, E. coli antiEpEX, P. putida, and yeast. No optimizer, no new FBA.
Native source: `outputs_multiseed_50k/seedXX/<species>_50k/chunks/results/results_chunk_*.parquet`.
This is a reconstruction from original **stored evaluations**, not a new simulation.

The 196 native Parquet results chunks per run are read *without copying* and
sorted by unique, verified, 50,000 contiguous native `sample_id` values (0-based
or 1-based). IDs indicate planned sample allocation order; they need not denote
physical completion time in concurrent execution. Missing/duplicate IDs cause
failure rather than guessed ordering. `feasible` plus finite objectives determines
whether a sample contributes to hypervolume. Infeasible samples still count
in sample-budget size.

Hypervolume uses exactly two native output columns, `biomass_flux` and
`product_flux`, both maximized, normalized by feasible 50K min and max **within
that same species/seed**. Fixed reference point is (0,0) in normalized coordinates.
For milestone n, HV(n)/HV(50,000)*100. No re-fitting bounds or selecting a new
reference at any smaller budget. 50K is 100% by definition. All subsets are
nested prefixes of the native 50K sample-ID order, not separate independent
5K, 10K, 25K campaigns. Over species, first aggregate within each seed, then
mean ± sample SD across three seed trajectories.

Time: original `runtime_seconds` is per native sample. The right panel's default
`worker` mode displays exact cumulative **sum of recorded per-sample durations**
(in worker-minutes), including any infeasible samples whose runtime was measured.
It is **not parallel elapsed wall-clock time**. Missing per-point durations make
that milestone missing. `performance_macos_time.txt` provides separately recorded
50K wall-clock endpoint durations for some runs (found for 43/44, not 42).
The generator **never** divides worker-minutes by an assumed number of cores,
interpolates measured wall-clock times, or infers a start time from the earliest
completion timestamp. A full wall-clock trajectory is permitted only when a
verified run start UTC is manually entered in SOURCE_MAP and all 50K native
completion timestamps are available. In that case `--time-mode wall` plots real
measured times; otherwise automatic plot mode uses worker runtime trajectories.

The archived source directory contains `_CANDIDATE_UNVERIFIED`. File-name evidence
and 50K structural checks cannot establish full scientific seed/model provenance.
Thus results remain **PROVISIONAL** until original run lineage, model identities,
configuration hashes, and timestamp provenance are independently certified.
The status is retained in the source audit and metadata but does not appear as an on-figure banner. No files are changed in Results/Landscape_Generation.
