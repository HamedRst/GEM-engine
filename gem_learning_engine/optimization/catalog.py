"""Portable catalog and immutable optimizer package identities."""
from pathlib import Path
import ast
import hashlib
import json
import yaml

ASSETS = Path(__file__).resolve().parents[2] / 'optimization_assets'
SPECIES = {'yeast': 'Yeast', 'pputida': 'P. putida', 'cglutamicum': 'C. glutamicum',
           'cho': 'CHO', 'ecoli_antiepex': 'E. coli anti-EpEX scFv'}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

# Final selected implementation identities from the 10K/25K gsMOBO v0.4.2
# package and the final published-method comparison for NSGA-II, backed by
# the parametric v1.0.3 source. Only final algorithm IDs and their required numerical helpers ship.
FINAL_SELECTED_METHODS = {
    'gsmobo': (
        'BASE_GSMOBO_QPAREGO_NO50K',
        'BASE_GSMOBO_QPAREGO_50K',
        'DEV_GSMOBO50K_GHOST_LEGACY_VB',
        'DEV_GSMOBO50K_SPARC_A',
        'DEV_GSMOBO50K_TRAIL_TRUE_NATIVE_V111',
    ),
    'nsga': (
        'BASE_NSGA2_NO50K',
        'BASE_NSGA2_50K',
        'DEV_NSGA50K_GHOST_A',
        'DEV_NSGA50K_SPARC_LEGACY_OPERATOR',
        'DEV_NSGA50K_TRAIL_LEGACY_OPERATOR',
    ),
}

def runtime_key(family, size, generalized=False):
    if not 4 <= int(size) <= 50000:
        raise ValueError('Prior size must be 4..50000 within the loaded landscape')
    if family == 'gsmobo': return 'gsmobo_10000'
    if family == 'nsga': return 'nsga_parametric'
    raise ValueError('Unknown optimizer family')

def methods(family, size, generalized=False):
    key = runtime_key(family, size, generalized=generalized)
    path = ASSETS / 'runtimes' / key
    # Parse constants without importing heavy libraries into the GUI process.
    tree = ast.parse((path / ('gsmobo_algorithms.py' if family == 'gsmobo' else 'nsga_algorithms.py')).read_text())
    env = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            try:
                value = eval(compile(ast.Expression(node.value), '<catalog>', 'eval'), {'__builtins__': {}}, env)
                for target in node.targets:
                    if isinstance(target, ast.Name): env[target.id] = value
            except Exception: pass
    cfg = yaml.safe_load((path / 'original_study_config.yaml').read_text())
    if generalized:
        ids = list(FINAL_SELECTED_METHODS[family])
        absent = set(ids) - set(env['METHODS'])
        if absent:
            raise RuntimeError(f'Final {family} selection absent from {key}: {sorted(absent)}')
        selected = ids
    else:
        ids = env['METHODS']
        selected = cfg.get('methods_primary', ids)
    rows = []
    for mid in ids:
        heuristic = env.get('HEURISTIC', {}).get(mid)
        lineage = env.get('LINEAGE', {}).get(mid, 'BASELINE')
        prior = 'NO50K' not in mid
        title = ('gsMOBO / qParEGO' if family == 'gsmobo' else 'NSGA-II')
        title += f' + {heuristic.upper()}' if heuristic and generalized else (f' + {heuristic.upper()} ({lineage})' if heuristic else (' with prior' if prior else ' without prior'))
        rows.append({'id': mid, 'label': title, 'family': family, 'prior': prior,
                     'lineage': lineage, 'selected_campaign_panel': mid in selected,
                     'runtime_key': key, 'algorithm_sha256': sha(path / ('gsmobo_algorithms.py' if family == 'gsmobo' else 'nsga_algorithms.py')),
                     'campaign_reference_available': family == 'gsmobo' and size in (5000, 10000, 25000),
                     'configuration_origin': cfg.get('provenance', {}), 'final_selected': bool(generalized), 'source_package': cfg.get('provenance',{}).get('package_version',cfg.get('version'))})
    return rows

def automatic_selection_method():
    """Meta-selector: pre-run QC dispatch, not a new physical optimizer algorithm."""
    from .automatic_selection import AUTO_ID, MODEL, hash_file
    if not MODEL.is_file(): raise RuntimeError('Automatic selection model missing')
    return {'id':AUTO_ID,'label':'Automatic selection (Surrogate QC → SPARC)',
        'family':'automatic','prior':True,'lineage':'QC_ONLY_RIDGE',
        'selected_campaign_panel':False,'runtime_key':'auto_preflight',
        'algorithm_sha256':hash_file(MODEL),'campaign_reference_available':False,
        'final_selected':True,'source_package':'QC_ONLY_RIDGE_DEPLOYMENT_297_V1'}

def canonical_cases():
    cfg = yaml.safe_load((ASSETS / 'runtimes/gsmobo_10000/original_study_config.yaml').read_text())
    result = []
    for case, c in cfg['cases'].items():
        source = ASSETS / 'benchmarks' / Path(c['source_benchmark']).name
        pc = yaml.safe_load((source / 'prepared' / c['source_case'] / 'problem_config.yaml').read_text())['problem']
        model = source / 'inputs/models' / Path(pc.get('model_path', '')).name
        if not model.is_file():
            candidates = list((source / 'inputs/models').glob(c['species'] + '*'))
            model = candidates[0] if len(candidates) == 1 else model
        result.append({'case': case, **c, 'source_benchmark': str(source), 'model_path': str(model),
                       'model_sha256': sha(model) if model.is_file() else None,
                       'landscape_path': str(source / 'prepared' / c['source_case'] / 'baseline_compact.csv.gz')})
    return result

def verify_sources():
    manifest = json.loads((ASSETS / 'SOURCE_HASHES.json').read_text())
    failures = []
    for key, info in manifest.items():
        for name, expected in info['files'].items():
            path = ASSETS / 'runtimes' / key / name
            if not path.is_file() or sha(path) != expected: failures.append(f'{key}/{name}')
    data_manifest=ASSETS/'SOURCE_DATA_HASHES.json'
    if data_manifest.exists():
        for name, expected in json.loads(data_manifest.read_text()).items():
            path=ASSETS/name
            if not path.is_file() or sha(path)!=expected:failures.append(name)
    if failures: raise ValueError('Optimizer source integrity failed: ' + ', '.join(failures))
    return {'status': 'PASS', 'packages': len(manifest)}
