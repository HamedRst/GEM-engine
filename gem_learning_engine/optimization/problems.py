"""Problem-file authoring around the unchanged benchmark compiler."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml

from .vendor.generic_problem import validate_problem_spec, problem_hash

TEMPLATES = Path(__file__).with_name('templates')
LABELS = {
    'growth_product': 'Growth × Product',
    'product_cost': 'Product × Resource Cost',
    'product_carbon_oxygen': 'Product × Carbon/Oxygen Efficiency',
    'growth_robustness': 'Growth × Oxygen-Limitation Robustness',
    'constrained_product': 'Product in the Top-Decile Growth Region',
}


def template_bytes(key: str) -> bytes:
    if key not in LABELS:
        raise ValueError('Unknown benchmark template')
    return (TEMPLATES / f'{key}.gemproblem.yaml').read_bytes()


def validate_content(content: str) -> dict[str, Any]:
    obj = yaml.safe_load(content)
    if not isinstance(obj, dict):
        raise ValueError('A problem file must be a YAML mapping')
    validated = validate_problem_spec(obj)
    # The frozen compiler's validator permits an omitted direction, but its
    # live evaluator expects it. Normalize only the in-memory custom schema.
    for objective in validated['problem']['objectives']:
        objective.setdefault('direction','maximize')
    # JSON rejects NaN/Infinity before a problem is assigned to a job.
    json.dumps(validated, allow_nan=False)
    return validated


def describe(content: str) -> dict[str, Any]:
    spec = validate_content(content)
    p = spec['problem']
    matched = [key for key in LABELS if yaml.safe_load(template_bytes(key)) == yaml.safe_load(content)]
    return {
        'id': p['id'], 'title': p.get('title', p['id']),
        'objectives': p['objectives'], 'constraints': p.get('constraints', []),
        'problem_hash': problem_hash(spec),
        'sha256': hashlib.sha256(content.encode('utf-8')).hexdigest(),
        'benchmark_template': matched[0] if matched else None,
    }


def generate_template(key: str, destination: Path) -> dict[str, Any]:
    """Preserve the canonical file byte for byte, including its schema and values."""
    data = template_bytes(key)
    validate_content(data.decode('utf-8'))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an existing problem definition with different content.
    if destination.exists() and destination.read_bytes() != data:
        raise FileExistsError('A different problem exists at this path; choose another filename')
    if not destination.exists():
        with destination.open('xb') as f:
            f.write(data)
    return {'path': str(destination), **describe(data.decode('utf-8')),
            'canonical_byte_identity': destination.read_bytes() == data}


def create_custom(problem_id: str, title: str, objectives: list[dict], constraints: list[dict]) -> str:
    """Simple form rows use exactly the frozen compiler's existing schema."""
    if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_\-]*', problem_id):
        raise ValueError('Use letters, numbers, underscores or hyphens for the problem name')
    spec = {'format': 'gemproblem/v0.2', 'problem': {
        'id': problem_id, 'title': title or problem_id,
        'objectives': objectives, 'constraints': constraints,
    }}
    # Semantic identity with a canonical benchmark preset must yield its exact
    # original bytes, even when the problem was authored through form controls.
    for key in LABELS:
        canonical=template_bytes(key).decode('utf-8')
        if yaml.safe_load(canonical)==spec:
            validate_content(canonical)
            return canonical
    content = yaml.safe_dump(spec, sort_keys=False, allow_unicode=True)
    validate_content(content)
    return content
