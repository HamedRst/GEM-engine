#!/usr/bin/env python3
"""Release installation checks for the unchanged GEM Engine scientific runtime."""
from pathlib import Path
from gem_learning_engine.optimization.catalog import (
    FINAL_SELECTED_METHODS, canonical_cases, methods, verify_sources,
)
from gem_learning_engine.optimization.problems import template_bytes, validate_content

ROOT = Path(__file__).resolve().parents[1]
assert verify_sources()['status'] == 'PASS'
cases = canonical_cases()
assert len(cases) == 25, f'Expected 25 canonical cases, found {len(cases)}'
assert len({c['species'] for c in cases}) == 5
for case in cases:
    validate_content(template_bytes(case['problem_key']).decode('utf-8'))
for family in ('gsmobo', 'nsga'):
    assert len(FINAL_SELECTED_METHODS[family]) == 5
    for prior in (5000, 10000, 25000, 50000):
        available = {m['id'] for m in methods(family, prior, generalized=True)}
        assert available == set(FINAL_SELECTED_METHODS[family]), (family, prior)
for rel in (
    'gem_learning_engine/exploration_integration.py',
    'exploration_bundle/gem_exploration_hub/app.py',
    'exploration_bundle/gem_exploration_tab/assets/index.html',
    'gem_learning_engine/optimization/auto_selector_model.json',
):
    assert (ROOT / rel).is_file(), f'Missing application asset: {rel}'
print('PASS: frozen optimizer and benchmark hashes, 25 cases, 10 selected methods, and GUI assets')
