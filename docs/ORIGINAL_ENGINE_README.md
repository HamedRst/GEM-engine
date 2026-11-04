# GEM Learning Engine 1.0.4

A ready-to-run exact FBA pipeline for generating reproducible medium–growth–production datasets from genome-scale metabolic models.

## Standard workflow

You only provide:

1. a COBRA-compatible model (`.json`, `.xml`, `.sbml`, or `.mat`), and
2. the number of simulations.

The engine then:

- detects the model format;
- discovers exchange reactions and the existing medium;
- infers biomass and product reactions with confidence scores;
- chooses native-objective or growth-constrained product optimization;
- classifies plausible nutrients/gases as variables while fixing water, ions, trace metals, and vitamins by default;
- builds a deterministic scrambled-Sobol design;
- runs exact FBA in resumable chunks;
- saves requested and applied media, growth, product, complete final-stage reaction/exchange fluxes, solver status, provenance, logs, manifests, and checksums;
- refuses to report successful completion when chunks, samples, or sample-level errors are missing.

## Browser interface

After installation:

```bash
./scripts/run_gui.sh
```

On Windows:

```powershell
.\scripts\run_gui.ps1
```

Open `http://127.0.0.1:8080` and use **Quick Run**:

1. choose the model file;
2. enter the run count;
3. select **Upload, validate, and start**.

Manual reaction IDs are optional advanced overrides.

After a run reaches 100%, open **Results** and select **Generate CSV**. The exporter processes one completed Parquet chunk at a time, so it does not load the complete run into memory. When finished, use **Download CSV**. The file remains inside that run's isolated `exports/` directory.

## One-command CLI

```bash
gem-learning-engine run \
  --model "model.json" \
  --runs 1024
```

For the bundled anti-EpEX scFv model:

```bash
gem-learning-engine run \
  --model data/models/examples/iJO1366_antiEpEX_scFv.json \
  --runs 1024
```

A preflight report is generated automatically before execution. The inferred IDs, confidence, medium variables, baseline feasibility, and resource estimates are saved under the output directory.

## Production objective

When a credible biomass/product pair is detected, the engine uses a two-stage formulation for each condition:

1. maximize biomass;
2. constrain biomass to a configurable fraction of that condition's maximum growth;
3. maximize product flux.

The default growth fraction is `0.5`. Override it with:

```bash
gem-learning-engine run --model model.json --runs 1024 --growth-fraction 0.75
```

## Automatic medium policy

Simple mode uses a conservative `nutrients` policy:

- likely nutrient and gas exchanges are variable;
- solvent, protons, background ions, trace metals, and vitamins/cofactors remain fixed at the model baseline;
- automatically inferred ranges are capped at 20 uptake units unless the native capacity is lower;
- explicit YAML bounds and include/exclude lists override the heuristic.

Every classification is written to `catalogs/medium_space.csv` and the preflight report.

## Output structure

```text
outputs/<run>/
├── catalogs/                 model and medium catalogs (CSV + Parquet)
├── checkpoints/              resume state and pause request
├── chunks/                   atomic result tables
├── design_cache/             deterministic design cache
├── logs/                     rotating text and JSONL logs
├── manifests/                per-chunk checksums and row counts
├── exports/
│   ├── whole_dataset.csv     one row per FBA analysis, generated on request
│   └── whole_dataset_manifest.json
└── metadata/
    ├── config_snapshot.yaml  resolved configuration
    ├── model_inference.json  candidate scores and selected IDs
    ├── preflight.json
    ├── run_manifest.json
    ├── run_status.json
    └── summary_statistics.json
```

## Complete CSV export

From the GUI, use **Results → Generate CSV** after completion. From the CLI:

```bash
gem-learning-engine export-csv --config configs/generated/<run-config>.yaml
```

The output is `exports/whole_dataset.csv` inside the selected run folder. It contains one row per exact FBA analysis and can be very large when all reaction fluxes are stored. Parquet chunk files remain the authoritative, efficient source dataset.

## Advanced configuration

```bash
gem-learning-engine run --config configs/anti_epex_reference.yaml
```

The YAML interface exposes solver settings, process count, chunk size, sampling ranges, diagnostics, storage policies, reaction overrides, and objective mode.

## Pause and resume

Use the GUI pause control, or:

```bash
gem-learning-engine pause --config configs/anti_epex_reference.yaml
gem-learning-engine clear-pause --config configs/anti_epex_reference.yaml
gem-learning-engine run --config configs/anti_epex_reference.yaml
```

The engine resumes only checksum-valid chunks from the same model fingerprint and data-generating configuration.

## Validation

```bash
./scripts/validate_installation.sh
```

This runs the unit/integration suite, toy pipeline validation, preflight checks, and the independent SciPy/HiGHS anti-EpEX reference validation. See `docs/VALIDATION_REPORT.md`.

## Scope

This release is the exact-simulation/data-generation layer. Surrogate learning, active learning, sensitivity analysis, media optimization, and gene knockout studies should consume these versioned outputs rather than altering the exact first-layer records.
