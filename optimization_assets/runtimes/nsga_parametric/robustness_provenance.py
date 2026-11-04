#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import yaml

HERE = Path(__file__).resolve().parent
CONTRACT_PATH = HERE / 'ROBUSTNESS_RUNTIME_SEMANTIC_CONTRACT.json'
CONTRACT = json.loads(CONTRACT_PATH.read_text())
BENCHMARK_NAME = CONTRACT['benchmark_name']
ROOT_TOKEN = '${ROBUSTNESS_BENCHMARK_ROOT}'


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def _normalize_obj(obj):
    if isinstance(obj, dict):
        return {k: _normalize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_obj(v) for v in obj]
    if isinstance(obj, str):
        marker = '/' + BENCHMARK_NAME
        i = obj.find(marker)
        if i >= 0:
            return ROOT_TOKEN + obj[i + len(marker):]
        return obj
    return obj


def semantic_problem_fingerprint(path: Path) -> str:
    doc = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    payload = json.dumps(_normalize_obj(doc), sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def _problem_doc(path: Path) -> dict:
    doc = yaml.safe_load(Path(path).read_text(encoding='utf-8')) or {}
    p = doc.get('problem', doc)
    if not isinstance(p, dict):
        raise RuntimeError(f'robustness source problem config has no mapping problem block: {path}')
    return p


def _declared_path(p: dict, key: str, fallback: Path) -> Path:
    val = p.get(key)
    if val is None:
        return Path(fallback)
    q = Path(str(val)).expanduser()
    return q if q.is_absolute() else (Path(fallback).parent / q).resolve()


def _assert_under_benchmark(path: Path, bench: Path, label: str):
    path = Path(path).expanduser().resolve()
    bench = Path(bench).expanduser().resolve()
    try:
        path.relative_to(bench)
    except Exception:
        raise RuntimeError(f'robustness {label} escapes certified benchmark root: {path} not under {bench}')


def verify_robustness_source(bench: Path, source_case: str, *, require_paths_exist: bool = True) -> dict:
    """Verify scientific identity of a relocatable robustness source.

    The problem YAML is intentionally path-relocatable, so its raw byte SHA is not a scientific
    invariant. We normalize only the absolute benchmark root and hash the remaining parsed YAML.
    Baseline, design cache, model, and engine remain strict byte-SHA invariants.
    """
    bench = Path(bench).expanduser().resolve()
    if source_case not in CONTRACT['cases']:
        raise RuntimeError(f'robustness source case absent from semantic contract: {source_case}')
    exp = CONTRACT['cases'][source_case]
    pc = bench / 'prepared' / source_case / 'problem_config.yaml'
    eng = bench / 'engine_bridge.py'
    if not pc.is_file():
        raise RuntimeError(f'missing robustness problem_config: {pc}')
    if not eng.is_file():
        raise RuntimeError(f'missing robustness engine_bridge: {eng}')

    sem = semantic_problem_fingerprint(pc)
    if sem != exp['semantic_problem_fingerprint_sha256']:
        raise RuntimeError(
            f'robustness semantic problem fingerprint mismatch for {source_case}; '
            f'expected={exp["semantic_problem_fingerprint_sha256"]} got={sem} path={pc}'
        )
    eng_sha = sha256_file(eng)
    if eng_sha != exp['engine_bridge_sha256']:
        raise RuntimeError(
            f'robustness engine SHA mismatch for {source_case}; '
            f'expected={exp["engine_bridge_sha256"]} got={eng_sha} path={eng}'
        )

    p = _problem_doc(pc)
    for key in ('case_name', 'case_key'):
        if str(p.get(key)) != str(exp[key]):
            raise RuntimeError(f'robustness {key} mismatch for {source_case}: {p.get(key)!r} != {exp[key]!r}')
    for key in ('model_id', 'model_fingerprint', 'config_hash'):
        if exp.get(key) is not None and p.get(key) != exp.get(key):
            raise RuntimeError(f'robustness {key} mismatch for {source_case}')

    baseline = _declared_path(p, 'baseline_table', bench / 'prepared' / source_case / 'baseline_compact.csv.gz')
    design = _declared_path(p, 'design_cache', bench / 'inputs' / 'runs' / source_case / 'design_cache' / exp['design_cache_filename'])
    model = _declared_path(p, 'model_path', bench / 'inputs' / 'models' / exp['model_filename'])
    run_dir = _declared_path(p, 'run_dir', bench / 'inputs' / 'runs' / source_case)

    # A relocated source must be internally self-contained under its own benchmark root.
    for q, label in ((baseline, 'baseline_table'), (design, 'design_cache'), (model, 'model_path'), (run_dir, 'run_dir')):
        _assert_under_benchmark(q, bench, label)

    if Path(model).name != exp['model_filename']:
        raise RuntimeError(f'robustness model filename mismatch for {source_case}: {Path(model).name}')
    if Path(design).name != exp['design_cache_filename']:
        raise RuntimeError(f'robustness design-cache filename mismatch for {source_case}: {Path(design).name}')

    if require_paths_exist:
        for q, label in ((baseline, 'baseline'), (design, 'design cache'), (model, 'model'), (run_dir, 'run_dir')):
            if not Path(q).exists():
                raise RuntimeError(f'missing robustness {label} for {source_case}: {q}')
        checks = (
            (baseline, exp['baseline_sha256'], 'baseline'),
            (design, exp['design_cache_sha256'], 'design cache'),
            (model, exp['model_sha256'], 'model'),
        )
        for q, expected, label in checks:
            got = sha256_file(q)
            if got != expected:
                raise RuntimeError(
                    f'robustness {label} SHA mismatch for {source_case}; '
                    f'expected={expected} got={got} path={q}'
                )

    return {
        'mode': 'relocation_invariant_semantic_problem_plus_strict_scientific_assets',
        'benchmark_root': str(bench),
        'source_case': source_case,
        'problem_config_path': str(pc),
        'problem_config_raw_sha256': sha256_file(pc),
        'problem_config_semantic_sha256': sem,
        'engine_sha256': eng_sha,
        'baseline_path': str(baseline),
        'baseline_sha256': sha256_file(baseline) if Path(baseline).is_file() else None,
        'design_cache_path': str(design),
        'design_cache_sha256': sha256_file(design) if Path(design).is_file() else None,
        'model_path': str(model),
        'model_sha256': sha256_file(model) if Path(model).is_file() else None,
        'run_dir': str(run_dir),
    }
