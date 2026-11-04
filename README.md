# GEM Engine

**A reusable landscape-guided framework for exploration and optimization of genome-scale metabolic design spaces**

[![Public release](https://img.shields.io/badge/public_release-v1.0.0-2563eb)](https://github.com/HamedRst/GEM-engine/releases)
![Python](https://img.shields.io/badge/Python-3.12-3776ab)
![macOS application](https://img.shields.io/badge/app-macOS-555555)
[![Research data](https://img.shields.io/badge/data-Zenodo%20DOI-168b8b)](https://doi.org/10.5281/zenodo.23921829)

GEM Engine is a research software framework for **genome-scale metabolic modeling, reusable simulation landscapes, multi-objective optimization, and interactive exploration**. It brings exact flux balance analysis (FBA), surrogate modeling, and a graphical interface together so that previously simulated metabolic design spaces can be reused to investigate new scientific questions instead of repeatedly recomputing the same landscape.

**Get started:** [macOS app](#option-1-macos-application) · [Run from source](#option-2-install-and-run-from-source) · [Command-line workflows](#using-the-scripts-and-command-line-tools) · [Publication data](#data-and-figures)

---

## What GEM Engine does

The workflow separates **building a reusable FBA landscape** from **asking questions of that landscape**:

```text
Genome-scale metabolic model
           │
           ▼
  Exact FBA simulations
           │
           ▼
 Reusable design landscape
           │
           ├──► Surrogate-assisted multi-objective optimization
           │       (gsMOBO, NSGA-II; BASE, SPARC, GHOST, TRAIL)
           │
           └──► Interactive exploration
                   (Activity · Calculator · Network · Saved)
```

The accompanying study benchmarks five optimization question types over five original model systems. **Those five questions are examples, not a fixed limit of the framework.** Users can formulate additional objectives and constraints through the supported GUI workflows or YAML problem definitions, provided the required model, landscape, metrics, and exact-evaluation configuration are available.

GEM Engine also includes a **surrogate-quality-based automatic selector** for choosing between the supported SPARC optimization families in the studied setting.

## Choose how to use it

| | **macOS application** | **Source code and scripts** |
|---|---|---|
| Best for | Exploring models and running workflows through a GUI | Reproducible pipelines, scripting, and method inspection |
| Entry point | `GEM Engine.app` | `gem-learning-engine-gui` and `gem-learning-engine` |
| Setup | First-launch setup inside the app; internet required initially | Python 3.12 and `uv` |
| Inputs | Compatible models, landscapes, and problem definitions | Same scientific inputs, plus YAML configurations |

## Requirements

- **macOS app:** macOS; the provided application is the Mac distribution of GEM Engine. First launch may download a Python environment and the native Graphviz renderer and therefore requires internet access. Compatibility beyond the tested Mac environment should be checked before broad deployment.
- **Source installation:** Python **3.12**, [`uv`](https://docs.astral.sh/uv/getting-started/installation/), and sufficient disk space for the models and landscapes you intend to use.
- **Scientific modeling:** COBRApy and a supported linear-programming solver are installed with the project dependencies; GLPK is used in the bundled example configuration.
- **Network visualizations:** the native Graphviz `dot` executable is required. The macOS launcher can bootstrap it automatically; for a manual source installation, install Graphviz separately and check `dot -V`.
- **Large computations:** memory, CPU, and storage requirements depend on model size, landscape size, and optimization settings. Full research datasets are **not** bundled into this repository.

## Option 1: macOS application

1. Open [**GitHub Releases**](https://github.com/HamedRst/GEM-engine/releases) and download the GEM Engine **macOS application ZIP** for this public release. *(The app must be added to the release assets by the maintainers; it is not part of the source-code checkout.)*
2. Extract the ZIP and locate **`GEM Engine.app`**. Move it to `Applications` if desired, or run it from the extracted folder.
3. Double-click **`GEM Engine.app`**. Follow the initial setup prompts. The app installs its required Python environment in your user profile, checks its packaged scientific inputs, and starts the interface in your default browser.
4. Use the **Optimization** and **Exploration** sections to load or prepare a compatible design landscape, define a question, run an analysis, and inspect or save results.

The interface runs locally in your browser; it is **not** a hosted online service. If macOS blocks opening an app you trust, use Finder's **Control-click → Open** and follow macOS's security prompts. The macOS app may report its historical internal build identifier `2.1.0`; **`v1.0.0` refers to the first public repository release**, not to a change in the scientific algorithms.

**App troubleshooting:** first-launch setup and server logs are written to `~/Library/Logs/GEM Engine/` (`setup.log` and `server.log`). Keep the extracted app bundle intact; moving individual files out of it may prevent startup.

## Option 2: Install and run from source

The following steps use the **Python 3.12 installation route tested with this source package**. Run them from Terminal after cloning or downloading the repository.

```bash
# Clone your personal fork/repository, or download its source ZIP from GitHub.
# Then change into the folder containing pyproject.toml:
cd /path/to/GEM-engine

# Create an isolated Python 3.12 environment.
uv venv --python 3.12 .venv

# Install the optimization and exploration dependencies.
uv pip install --python .venv/bin/python -e '.[optimization,exploration]'

# Verify the frozen scientific assets and GUI files.
.venv/bin/python scripts/validate_gui_only.py
```

> **Why not `uv sync` here?** The preserved `pyproject.toml` advertises Python `>=3.10`, while the pinned BoTorch dependency requires Python `>=3.11`. On some `uv` versions, universal resolution fails even when Python 3.12 is selected. The environment-specific `uv pip install` command above avoids that resolver issue without changing the research source.

### Launch the integrated GUI

```bash
mkdir -p "$HOME/GEM-Engine-Workspace"

GEM_ENABLE_EXPLORATION=1 GEM_EXPLORATION_EMBEDDED=1 \
  .venv/bin/gem-learning-engine-gui \
  --workspace "$HOME/GEM-Engine-Workspace" \
  --host 127.0.0.1 --port 8089
```

Then open **http://127.0.0.1:8089** in a browser. Choose another free local port if `8089` is already in use. The separate workspace keeps new results outside the source tree.

## Using the scripts and command-line tools

The source distribution includes the exact-FBA simulation CLI and the exploration engine. These examples use the bundled **demonstration model**; they do not reproduce the full publication benchmarks by themselves.

### Inspect a metabolic model

```bash
.venv/bin/gem-learning-engine inspect-model \
  --model data/models/demo_model.json \
  --output-dir "$HOME/GEM-Engine-Workspace/model_inspection"
```

### Generate a small exact-FBA landscape

```bash
.venv/bin/gem-learning-engine run \
  --model data/models/demo_model.json \
  --runs 64 --processes 1 \
  --output-dir "$HOME/GEM-Engine-Workspace/demo_simulation"
```

For configured runs, use `gem-learning-engine run --config /path/to/config.yaml`. Available commands include `preflight`, `run`, `status`, `pause`, `clear-pause`, `inspect-model`, and `export-csv`. Run `.venv/bin/gem-learning-engine --help` for the full command reference.

### Explore a landscape from the command line

```bash
PYTHONPATH=exploration_bundle \
  .venv/bin/python -m gem_exploration list-operators

PYTHONPATH=exploration_bundle \
  .venv/bin/python -m gem_exploration inspect \
  --landscape /path/to/your/landscape.csv
```

The exploration CLI also supports `validate`, `plan`, and `run` for compatible YAML question definitions. Use `PYTHONPATH=exploration_bundle .venv/bin/python -m gem_exploration --help` for usage.

**Custom questions:** objectives, constraints, and mathematical expressions are supported within the available design variables and measurements. A new objective is not automatically computable for every metabolic model; a compatible landscape and supported evaluation protocol are required.

## Repository contents

| Path | Contents |
|---|---|
| `gem_learning_engine/` | Core exact-FBA engine, CLI, integrated GUI, optimization orchestration, and automatic selector |
| `optimization_assets/` | Frozen benchmark inputs, reference landscapes, optimizers, and integrity manifests |
| `exploration_bundle/` | Activity, Calculator, Network, and general landscape-exploration modules |
| `configs/`, `examples/`, `data/` | Configuration examples and small sample models |
| `scripts/` | Runtime support and source integrity validation |
| `Figures_and_Tables/` | Manuscript figures, tables, and supporting materials included with the repository |
| `Application/` | Place for the macOS distribution pointer; large app archives are distributed via **GitHub Releases** |

Some scientific-asset directories retain historical names such as `GEM_5Species_*`. Those names reflect the original benchmark lineage and are retained because the current software refers to these frozen inputs; they are **not separate application versions**.

## Data and figures

The simulation landscapes, optimization benchmarks, exploration results, and publication figure/table materials are archived separately on Zenodo:

**[GEM Engine publication dataset — DOI: 10.5281/zenodo.23921829](https://doi.org/10.5281/zenodo.23921829)**

The dataset deposit contains `Landscape_Generation.zip`, `Optimization.zip`, `Exploration.zip`, and `Figures_and_Tables.zip`. Large research outputs are not duplicated in this code repository. Trained surrogate-model checkpoints are not included in the primary dataset deposit; the packaged automatic-selection configuration is retained in the source.

## Versioning, reproducibility, and citation

- **`v1.0.0`** denotes the **first public GEM Engine distribution**. The preserved scientific Python package identifies itself internally as **`gem-learning-engine 1.0.4`**; the Mac launcher may contain the earlier internal GUI build identifier `2.1.0`.
- The publication source retains the scientific methods and frozen computational assets from the patched research implementation. Development-only materials were excluded from the public-facing copy; retained benchmark integrity manifests were reconciled with those exclusions.
- To reproduce a study result, use the corresponding archived dataset and the **specific GitHub release/commit**, along with the model, landscape, objective configuration, and random seed. Merely launching the GUI does not reproduce every reported benchmark.
- **Data citation:** [https://doi.org/10.5281/zenodo.23921829](https://doi.org/10.5281/zenodo.23921829). Please also cite the associated GEM Engine research article when available.

## License and attribution

A software license has not yet been declared in this repository. **Public availability alone does not grant permission to redistribute or modify the source code.** Licensing should be finalized with the authors and institution before an open-source release. Third-party genome-scale metabolic models may have their own licenses and attribution requirements.

---

**Oliveira Lab**
