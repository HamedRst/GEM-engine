# Calculator language v0.5.0

## Minimal public form

```text
observe:(<quantity or expression>), variables:(<variable list or all>), conditions:(<filters or none>)
```

Optional:

```text
explore:(<public scientific function>)
```

Fields may be reordered and pipes are optional.

## Examples

```text
observe:(product), variables:(glucose, oxygen), conditions:(growth >= q80, feasible=true)
observe:(resource efficiency), variables:(oxygen), conditions:(feasible=true)
observe:(log1p(product/glucose)), variables:(oxygen), conditions:(feasible=true)
observations:(growth, product), explore:(pareto), variables:(all), conditions:(feasible=true)
```

## Public exploration vocabulary

`describe`, `relate`, `vary`, `compare`, `robust`, `select`, `structure`, `pareto`.

Specialized modifiers remain available, e.g. `relate(interaction)`, `relate(local)`, `structure(regimes)`, `structure(dimensions)`, `structure(coverage)`, `select(outliers)`, `compare(regions)`, `pareto(saturation)`.

## Conditions

Support comparisons (`> >= < <= = !=`), quantiles such as `q80`, comma/AND conjunction, OR logic, and named A/B regions for region comparison.

## Derived expressions

Supported operators: `+ - * / ** %`.
Supported functions: `abs`, `sqrt`, `log`, `log1p`, `exp`, `clip`, `minimum`, `maximum`.
Names inside expressions are resolved through the dynamic semantic registry; division uses safe denominator handling.

## Evidence rule

A quantity may be analyzed if it is stored or faithfully derivable from stored landscape evidence. A model entity that was not stored/varied is semantic metadata, not perturbation evidence. An explicit perturbation such as `robust(10%)` is therefore not silently translated to nearest-neighbor robustness.
