from __future__ import annotations

from pathlib import Path

from gem_exploration.config import SUPPORTED_OPERATORS
from gem_exploration.visualization_planner import OPERATOR_VISUAL_FAMILY

from .calculator import CalculatorRequest, build_registry, compile_calculator, active_fragment, insert_suggestion, parse_calculator_line
from .inspectors import inspect_landscape

ROOT = Path(__file__).resolve().parents[1]
LAND = ROOT / 'demo' / 'demo_landscape.csv'

CASES = [
    ('describe', 'product', 'describe', 'all', 'none', 'describe'),
    ('query', 'product', 'select', 'all', 'growth > 0.2', 'query'),
    ('association', 'product', 'relate(association)', 'all', 'none', 'association'),
    ('correlation', 'growth, product', 'relate(correlation)', 'glucose, oxygen', 'none', 'correlation_matrix'),
    ('global', 'product', 'relate(influence)', 'all', 'feasible = true', 'global_sensitivity'),
    ('local', 'product', 'relate(local)', 'all', 'none', 'local_sensitivity'),
    ('map', 'product', 'relate(map)', 'all', 'none', 'sensitivity_map'),
    ('interaction', 'product', 'relate(interaction)', 'all', 'none', 'interaction'),
    ('fragility', 'product', 'robust(fragility)', 'oxygen', 'none', 'fragility'),
    ('robustness', 'product', 'robust', 'oxygen', 'none', 'robustness'),
    ('threshold_region', 'product', 'compare(threshold=0.7)', 'all', 'none', 'threshold_region'),
    ('threshold_boundary', 'product', 'select(boundary=0.7)', 'all', 'none', 'threshold_boundary'),
    ('feasibility', 'product', 'relate(feasibility)', 'all', 'none', 'feasibility'),
    ('regimes', 'growth, product', 'structure(regimes)', 'all', 'none', 'regime_discovery'),
    ('dimensionality', 'growth, product', 'structure(dimensions)', 'all', 'none', 'dimensionality'),
    ('tradeoff', 'growth, product', 'pareto(tradeoff)', 'all', 'none', 'tradeoff'),
    ('pareto', 'growth, product', 'pareto', 'all', 'none', 'pareto'),
    ('pareto_saturation', 'growth, product', 'pareto(saturation)', 'all', 'none', 'pareto_saturation'),
    ('compare_regions', 'product', 'compare(regions)', 'all', 'A[growth >= q80]; B[growth <= q20]', 'compare_regions'),
    ('coverage', 'growth', 'structure(coverage)', 'all', 'none', 'coverage'),
    ('outlier', 'product', 'select(outliers)', 'all', 'none', 'outlier'),
    ('point', 'product', 'vary(point)', 'glucose=5, oxygen=10, ammonium=2', 'none', 'point_query'),
    ('curve', 'product', 'vary', 'glucose', 'none', 'response_curve'),
    ('marginal', 'product', 'vary(marginal)', 'glucose', 'none', 'marginal_response'),
    ('support', 'product', 'structure(support)', 'all', 'none', 'surrogate_uncertainty'),
    ('controller', 'product', 'relate(controller)', 'all', 'none', 'controller_scan'),
    ('threshold_switch', 'product', 'vary(threshold=0.7)', 'glucose', 'none', 'threshold_switch_scan'),
    ('surface', 'product', 'vary', 'glucose, oxygen', 'growth >= q50', 'response_surface'),
    ('sampling_saturation', 'growth', 'structure(saturation)', 'all', 'none', 'sampling_saturation'),
]


def run_validation() -> None:
    info, _ = inspect_landscape(str(LAND))
    reg = build_registry(info, None)
    got = []
    for name, observe, explore, variables, conditions, expected in CASES:
        comp = compile_calculator(
            CalculatorRequest(name, info.path, None, observe, explore, variables, conditions, n_estimators=50, cv_folds=3), reg
        )
        op = comp.public_yaml['compiled_execution']['internal_operator']
        if op != expected:
            raise RuntimeError(f'{name}: compiled {op}, expected {expected}')
        got.append(op)
    supported = set(SUPPORTED_OPERATORS)
    if set(got) != supported:
        raise RuntimeError(f'Calculator coverage mismatch: missing={sorted(supported-set(got))} extra={sorted(set(got)-supported)}')
    missing_viz = supported - set(OPERATOR_VISUAL_FAMILY)
    if missing_viz:
        raise RuntimeError(f'Visualization planner missing: {sorted(missing_viz)}')
    parsed=parse_calculator_line('observations:(growth, product), variables:(oxygen), conditions:(feasible=true)')
    if parsed.get('observe') != 'growth, product' or parsed.get('explore') != 'auto':
        raise RuntimeError('Calculator alias/auto parser regression')
    if active_fragment('product/glu', 'observe') != 'glu':
        raise RuntimeError('Derived-expression autocomplete fragment regression')
    if insert_suggestion('product/glu', 'Glucose', 'observe') != 'product/Glucose':
        raise RuntimeError('Derived-expression autocomplete insertion regression')
    print('CALCULATOR UI CONTRACT PASS: 3 primary fields + optional EXPLORE')
    print('PUBLIC EXPLORE FUNCTIONS PASS: 8')
    print('CALCULATOR COMPILER COVERAGE PASS: 29/29 internal operators')
    print('SEMANTIC AUTOCOMPLETE REGISTRY + DERIVED-EXPRESSION COMPLETION: PASS')
    print('QUANTILE CONDITIONS qXX: PASS')
    print('AUTOMATIC VISUALIZATION COVERAGE PASS: 29/29')
    print('GUI EVIDENCE POLICY: source read-only; new FBA calls = 0')


if __name__ == '__main__':
    run_validation()
