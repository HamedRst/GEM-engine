# GEM Exploration Calculator GUI v0.5.0

A local Mac-first scientific calculator for exploring reusable GEM landscapes. The normal interface is deliberately small: **OBSERVE · VARIABLES · CONDITIONS**. A specialized **EXPLORE** function is optional and lives under an advanced expansion.

## Normal workflow

1. Click **Choose landscape…** and select a landscape run folder.
2. The calculator tries to find the matching GEM in the canonical neighboring `inputs/models/` folder. If it cannot, click **Choose GEM…**.
3. Type a calculator question, for example:

```text
observe:(product), variables:(glucose, oxygen), conditions:(growth >= q80, feasible=true)
```

or a derived quantity:

```text
observe:(log1p(product/glucose)), variables:(oxygen), conditions:(feasible=true)
```

`EXPLORE` can stay `auto` for ordinary description/association/1-D or 2-D variation. Use the optional field only when you want a specific scientific operation such as `pareto`, `structure(regimes)`, `robust`, `relate(interaction)`, or `structure(dimensions)`.

## Scientific evidence policy

The calculator asks this in order:

1. **Can the stored exact-FBA landscape answer the question directly?** Use it.
2. **Can the requested quantity be deterministically derived from stored evidence?** Derive it with 0 new FBA calls.
3. **Is interpolation/model-based inference genuinely required?** Fit a surrogate and enforce QC.
4. For ordinary 1-D/2-D response questions, if surrogate QC fails, show a clearly labeled **exact-landscape binned fallback** instead of inventing a smooth response.
5. If the requested biological state is not represented by the landscape, fail closed rather than pretending the GEM itself is evidence.

v0.5.0 is read-only with respect to the source landscape and intentionally performs **0 new FBA calls**. A general targeted exact-FBA rescue engine is not claimed in this release.

## Dynamic quantities

The observable vocabulary is discovered from the loaded landscape and matching GEM metadata; it is not fixed to growth/product. The registry can include stored outputs, stored reaction/exchange fluxes, design variables, and deterministic benchmark-derived quantities such as resource cost/efficiency when their formula and bounds are available.

Simple derived expressions are compiled safely from existing evidence, including:

```text
product/glucose
product - growth
log1p(product)
log1p(product/glucose)
resource cost / oxygen
sqrt(product)
```

Supported expression operations include `+ - * / ** %`, parentheses, and safe functions such as `abs`, `sqrt`, `log`, `log1p`, `exp`, `clip`, `minimum`, and `maximum`. Division is compiled through guarded `safe_divide` logic.

Autocomplete works inside expressions: typing `product/glu` suggests the glucose quantity without replacing `product/`.

## Multi-observable and multi-view results

A question is not forced into one plot when one plot would hide important structure.

- Multiple observations in `describe`, `relate`, 1-D response, marginal response, and 2-D response produce one faithful view per observable.
- Multi-observable robustness/fragility produces an overview comparison plus per-observable distributions.
- Pareto analysis with >2 objectives produces pairwise objective projections (capped at six) while the Pareto front itself is computed in the full objective space.
- **Regime discovery is explicitly multi-view.** With two clustering features it shows the direct scientific plane. With >2 features it clusters in the full standardized feature space and shows: PCA projection for visualization, standardized regime profiles, silhouette-vs-k selection, PCA loadings, and direct observable-pair views when useful. It never implies that the first two arbitrary columns defined the clusters.

All plots are compiled from the resolved scientific question and result geometry. The user does not choose from a plot library.

## Important biological boundary

A gene or reaction existing in the GEM does not mean the loaded landscape contains perturbation evidence for it. Gene knockout/knockdown or a new FBA objective requires either a landscape that contains that intervention/state or a future general exact-FBA perturbation backend. The GUI should expose the model semantics but mark unsupported perturbation evidence as unavailable.

## Install and run on the MacBook Air

Put the ZIP under `~/Desktop/GEM/`, then:

```bash
cd "$HOME/Desktop/GEM"
unzip -q "GEM_Exploration_Calculator_GUI_v0.5.0.zip"
cd "$HOME/Desktop/GEM/GEM_Exploration_Calculator_GUI_v0.5.0"
chmod +x *.command
./SETUP.command
./RUN_GUI.command
```

The GUI uses local port **8092** so it can coexist with earlier versions.

## Validation

`./SETUP.command` and `./VALIDATE.command` exercise the public calculator compiler, compact parser, 29 internal operator mappings, core operator/visualization self-tests, execution smoke tests, dynamic observables, canonical resource-cost reconstruction, stored-flux loading, full-design surrogate context, exact-landscape response fallback, general derived expressions, multi-observable analyses, and multi-view regime/robustness/Pareto rendering.
