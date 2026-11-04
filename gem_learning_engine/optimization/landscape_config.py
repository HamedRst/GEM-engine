"""Read the landscape GUI's saved, authoritative configuration for Optimization.

No guesses from sampled minimum/maximum values, exchange-column naming, or
species-specific constants. Works without COBRApy or any new FBA evaluations.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..config import load_config


def _json_if_present(path: Path) -> dict:
    if path.is_file():
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    return {}


def read_existing_landscape_config(workspace, config_path: str) -> dict:
    """Return recorded scientific settings from a previous landscape run.

    Config paths are checked against the configured workspace; output file
    paths are never allowed to escape it. A saved catalog is authoritative
    for *effective* bounds, even when sampling.bounds is empty/automatic.
    """
    selected = workspace.resolve(config_path, must_exist=True)
    if selected.suffix.lower() not in ('.yaml', '.yml'):
        raise ValueError('Choose the earlier landscape-generation configuration YAML.')
    preliminary = load_config(selected)
    run = workspace.resolve(preliminary.project.output_dir)
    if selected.name == 'config_snapshot.yaml' and selected.parent.name == 'metadata':
        run = selected.parent.parent
    snapshot = run / 'metadata' / 'config_snapshot.yaml'
    config_file = snapshot if snapshot.exists() else selected
    cfg = load_config(config_file)
    # Run directory recorded in the saved snapshot may be an old Mac path;
    # resolve from the original chosen config instead when accessible.
    try:
        model = workspace.resolve(cfg.model.path, must_exist=True)
        model_path = workspace.display(model)
        model_available = True
    except (ValueError, FileNotFoundError) as exc:
        model_path = str(cfg.model.path)
        model_available = False
        model_warning = str(exc)

    catalog = run / 'catalogs' / 'medium_space.csv'
    catalog_parquet = run / 'catalogs' / 'medium_space.parquet'
    source = ''
    if catalog.exists():
        frame = pd.read_csv(catalog)
        source = workspace.display(catalog)
    elif catalog_parquet.exists():
        frame = pd.read_parquet(catalog_parquet)
        source = workspace.display(catalog_parquet)
    else:
        frame = None

    features: list[str] = []
    bounds: dict[str, list[float]] = {}
    fixed_medium: dict[str, float] = {}
    warnings: list[str] = []
    if not model_available:
        warnings.append('Saved model path is not available in this GUI workspace: ' + model_warning)
    if frame is not None:
        required = {'role', 'reaction_id', 'medium_feature_index', 'effective_min_uptake', 'effective_max_uptake'}
        if not required.issubset(frame.columns):
            raise ValueError('Saved medium_space catalog lacks essential design metadata: ' + ', '.join(sorted(required-set(frame.columns))))
        variable = frame.loc[frame['role'].astype(str) == 'variable'].copy()
        variable['medium_feature_index'] = pd.to_numeric(variable['medium_feature_index'], errors='raise')
        variable = variable.sort_values('medium_feature_index')
        if not variable['medium_feature_index'].tolist() == list(range(len(variable))):
            raise ValueError('Saved design-variable ordering is incomplete or ambiguous.')
        for _, row in variable.iterrows():
            rid = str(row['reaction_id'])
            if rid in bounds:
                raise ValueError(f'Duplicate reaction ID in saved design metadata: {rid}')
            lo, hi = float(row['effective_min_uptake']), float(row['effective_max_uptake'])
            if not (0 <= lo <= hi):
                raise ValueError(f'Invalid effective uptake bounds recorded for {rid}')
            features.append(rid)
            bounds[rid] = [lo, hi]
        fixed = frame.loc[frame['role'].astype(str) == 'fixed']
        if 'baseline_uptake' in fixed:
            fixed_medium = {str(row['reaction_id']): float(row['baseline_uptake']) for _, row in fixed.iterrows()}
    elif cfg.sampling.include_reaction_ids and all(rid in cfg.sampling.bounds for rid in cfg.sampling.include_reaction_ids):
        features = list(cfg.sampling.include_reaction_ids)
        bounds = {rid: [float(cfg.sampling.bounds[rid]['min']), float(cfg.sampling.bounds[rid]['max'])] for rid in features}
        source = 'explicit sampling bounds in configuration'
    else:
        warnings.append('Saved medium-space catalog missing: cannot determine automatic effective bounds without regenerating the original design space. No guess was made.')

    inference = _json_if_present(run / 'metadata' / 'model_inference.json')
    manifest = _json_if_present(run / 'metadata' / 'run_manifest.json')
    run_status = _json_if_present(run / 'metadata' / 'run_status.json')
    mode = cfg.optimization.mode
    if mode == 'auto':
        mode = inference.get('selected_optimization_mode') or 'auto'
    # Same mode resolution as the landscape generation engine: auto uses
    # growth-constrained product only when *both* configured reaction IDs exist.
    if mode == 'auto':
        # If the completed run did not record an effective mode, only the
        # explicit configuration can establish what 'auto' meant.
        # Do not use inferred reaction IDs to speculate about the run's FBA.
        mode = ('growth_constrained_product' if cfg.model.biomass_reaction_id and cfg.model.product_reaction_id else 'native_objective')
        if not cfg.model.biomass_reaction_id and not cfg.model.product_reaction_id and inference.get('biomass') and inference.get('product'):
            mode = 'unresolved_auto'
            warnings.append('FBA auto mode is ambiguous without its recorded effective mode; inspect the original run preflight before optimization.')
    from .general import FBA_MODES
    adapter_mode = mode if mode in FBA_MODES else None
    if adapter_mode is None:
        warnings.append(f'No supported exact adapter for saved FBA mode {mode!r}; a different FBA procedure needs a new landscape.')

    biomass = cfg.model.biomass_reaction_id or (inference.get('biomass') or {}).get('reaction_id') or ''
    product = cfg.model.product_reaction_id or (inference.get('product') or {}).get('reaction_id') or ''
    if not product and mode in ('growth_constrained_product','maximize_product'):
        warnings.append('Saved FBA mode requires a product reaction but it is absent from landscape configuration.')
    status_saved = run_status.get('saved_simulations')
    completed = status_saved is not None and int(status_saved) == int(cfg.simulation.n_simulations) and not run_status.get('failed_chunks') and not run_status.get('invalid_chunks')

    return {
        'config_path': workspace.display(config_file),
        'selected_config_path': workspace.display(selected),
        'run_dir': workspace.display(run),
        'model_path': model_path,
        'model_available': model_available,
        'source_label': str(cfg.project.name),
        'features': features,
        'bounds': bounds,
        'medium_catalog': source,
        'fixed_medium': fixed_medium,
        'biomass_reaction_id': biomass,
        'product_reaction_id': product,
        'objective_reaction_id': '', 'objective_direction': 'max',
        'fba_mode': adapter_mode,
        'recorded_fba_mode': mode,
        'minimum_growth_fraction': float(cfg.optimization.minimum_growth_fraction),
        'feature_mode': 'medium_uptake',
        'landscape_seed': int(cfg.simulation.seed),
        'landscape_rows': int(cfg.simulation.n_simulations),
        'saved_rows': int(status_saved) if status_saved is not None else None,
        'completed': completed,
        'solver': str(cfg.solver.name),
        'feasible_statuses': list(cfg.simulation.feasible_statuses),
        'sampling_strategy': cfg.sampling.strategy,
        'sampling_close_unlisted_exchanges': bool(cfg.simulation.close_unlisted_exchanges),
        'sampling_respect_original_exchange_bounds': bool(cfg.simulation.respect_original_exchange_bounds),
        'selected_exchange_fluxes': list(cfg.outputs.selected_exchange_fluxes),
        'selected_reaction_fluxes': list(cfg.outputs.selected_reaction_fluxes),
        'store_exchange_fluxes': cfg.outputs.store_exchange_fluxes,
        'store_reaction_fluxes': cfg.outputs.store_reaction_fluxes,
        'model_fingerprint': manifest.get('model_fingerprint'),
        'warnings': warnings,
        'automatic_source': 'saved landscape configuration + effective medium-space catalog',
    }
