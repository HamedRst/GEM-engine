"""General, explicit model+landscape binding. Frozen benchmark sources remain untouched.

A landscape is historical observations; a GEM is its exact evaluator. We never infer
that an arbitrary CSV and model have the same FBA contract, nor silently equate
stored reaction fluxes to an unsupported live calculation.
"""
from __future__ import annotations
import ast
import hashlib
import json
import re
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from .catalog import sha
from .problems import validate_content
from .vendor import generic_problem as gp

ALLOWED_ARCHIVES={'.csv','.gz','.parquet'}
ALLOWED_MODELS={'.xml','.sbml','.json','.mat'}
STANDARD_METRICS={'maximum_growth','product_flux','biomass_flux','objective_value','resource_cost','resource_efficiency'}
FBA_MODES={'native_objective','maximize_biomass','maximize_product','growth_constrained_product','maximize_reaction','growth_constrained_reaction'}


def read_landscape(path, rows=None):
    path=Path(path)
    if path.suffix=='.parquet':return pd.read_parquet(path).head(rows) if rows else pd.read_parquet(path)
    if path.suffix in ('.csv','.gz'):
        return pd.read_csv(path,nrows=rows) if rows else pd.read_csv(path)
    raise ValueError('Landscape must be CSV, CSV.GZ, or Parquet')


def inspect(workspace, model_path, landscape_path):
    model=workspace.resolve(model_path,must_exist=True)
    landscape=workspace.resolve(landscape_path,must_exist=True)
    if model.suffix.lower() not in ALLOWED_MODELS:raise ValueError('Model must be SBML/XML, COBRA JSON, or MATLAB .mat')
    if landscape.suffix.lower() not in ALLOWED_ARCHIVES:raise ValueError('Landscape must be CSV, CSV.GZ, or Parquet')
    sample=read_landscape(landscape,rows=256)
    columns=list(map(str,sample.columns))
    suggestions=[c for c in columns if c.startswith('EX_') or c.startswith('R_EX_')]
    meta={'model_path':str(model),'landscape_path':str(landscape),'columns':columns,
          'suggested_features':suggestions,'preview_rows':len(sample),
          'feature_ranges':{c:[float(pd.to_numeric(sample[c],errors='coerce').min()),float(pd.to_numeric(sample[c],errors='coerce').max())] for c in suggestions[:40] if pd.to_numeric(sample[c],errors='coerce').notna().any()}}
    try:
        import cobra
        m=load_model(model)
        meta['model_id']=m.id
        meta['reaction_count']=len(m.reactions)
        meta['exchange_reactions']=[r.id for r in m.exchanges][:250]
        meta['objective_reactions']=[r.id for r in m.reactions if r.objective_coefficient!=0][:20]
        meta['cobra_model_inspected']=True
    except ImportError:
        meta['cobra_model_inspected']=False
        meta['warning']='COBRApy unavailable in this environment; model reactions cannot be checked yet'
    return meta


def load_model(path):
    import cobra
    path=Path(path); ext=path.suffix.lower()
    if ext in ('.xml','.sbml'):return cobra.io.read_sbml_model(str(path))
    if ext=='.json':return cobra.io.load_json_model(str(path))
    if ext=='.mat':return cobra.io.load_matlab_model(str(path))
    raise ValueError('Unsupported GEM format')


def referenced_metrics(spec):
    """Extract every exact metric / expression column dependency safely."""
    out=set()
    for item in list(spec['problem']['objectives'])+list(spec['problem'].get('constraints',[]) or []):
        source=item.get('source','exact_metric')
        if source=='exact_metric':out.add(str(item.get('metric') or item.get('name')))
        if source=='expression':
            tree=ast.parse(item.get('expression',''),mode='eval')
            for n in ast.walk(tree):
                if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='metric':
                    if n.args and isinstance(n.args[0],ast.Constant):out.add(str(n.args[0].value))
                if isinstance(n,ast.Name) and n.id not in {'metric','x','x01','mean_x01','sum_x01','min_x01','max_x01','abs','sqrt','log','log1p','exp','minimum','maximum','clip','pi','e'}:
                    out.add(n.id)
    return out


def validate_contract(binding,spec):
    # The frozen archive must contain values for each requested physical metric.
    # A reaction ID may be used directly as metric("RXN_ID"): no alias mapping.
    df=read_landscape(binding['landscape_path'], rows=1)
    requested=referenced_metrics(spec)
    allowed=STANDARD_METRICS|set(binding.get('metric_reactions',{}))|set(binding['features'])
    # Derived live metrics are valid ONLY with a declared, reproducible FBA
    # recipe.  Do not trust an archive column on its own as proof that the
    # generalized exact evaluator can calculate that metric for new designs.
    recipes=binding.get('derived_metric_recipes') or {}
    if not isinstance(recipes,dict):
        raise ValueError('derived_metric_recipes must be a mapping')
    for name,recipe in recipes.items():
        if not isinstance(recipe,dict):
            raise ValueError('Derived exact metric recipe must be a mapping: '+str(name))
        kind=recipe.get('type')
        if kind not in ('stressed_growth_retention','environmental_stress_panel'):
            raise ValueError('Unsupported derived exact metric recipe: '+str(kind))
        if binding.get('fba_mode')!='maximize_biomass' or not binding.get('biomass_reaction_id') or binding.get('feature_mode')!='medium_uptake':
            raise ValueError('Stressed growth requires a biomass-maximization medium-uptake FBA contract')
        groups=(recipe.get('groups') if kind=='environmental_stress_panel'
                else {'oxygen':recipe.get('reaction_ids')})
        if not isinstance(groups,dict) or not groups or any(not isinstance(v,(list,tuple)) or not v for v in groups.values()):
            raise ValueError('Derived stress metric needs nonempty stress groups: '+str(name))
        if any(not isinstance(r,str) or r not in binding['features'] for ids in groups.values() for r in ids):
            raise ValueError('Stress-group reaction IDs must belong to the saved design space: '+str(name))
        if recipe.get('primary_stress_class','oxygen') not in groups:
            raise ValueError('Primary stress class missing from the recorded groups: '+str(name))
        fraction=float(recipe.get('fraction_toward_lower_bound',float('nan')))
        if not np.isfinite(fraction) or not 0<=fraction<=1:
            raise ValueError('Stress reduction fraction must lie in [0,1]: '+str(name))
        if not isinstance(name,str) or not name:
            raise ValueError('Derived metric name must be non-empty')
        allowed.add(name)
    if binding.get('biomass_reaction_id'):
        allowed.add('biomass_flux')
    if not binding.get('product_reaction_id'):
        allowed.discard('product_flux')
    if binding.get('fba_mode') not in ('maximize_biomass','growth_constrained_product','growth_constrained_reaction'):
        allowed.discard('maximum_growth')
    model_ids=None
    try:
        model_ids={r.id for r in load_model(binding['model_path']).reactions}
    except ImportError:
        pass  # Runtime preflight will enforce reaction identity when cobra is available.
    direct={x for x in requested if x in df.columns and (model_ids is None or x in model_ids)}
    allowed|=direct
    unsupported=sorted(requested-allowed)
    if unsupported:
        raise ValueError('YAML requires metrics unavailable from the recorded FBA procedure: '+', '.join(unsupported)+'. For reaction fluxes use the exact model reaction ID as the metric and record that flux in the landscape. A different underlying FBA objective requires a newly generated landscape and its own config.')
    missing=sorted((requested-{'resource_cost','resource_efficiency'}-set(binding['features']))-set(df.columns))
    # Some aliases can be derived from recorded native columns; never invent values.
    if missing:
        raise ValueError('Selected landscape lacks recorded columns for: '+', '.join(missing)+'. Generate a compatible landscape that records the requested fluxes; do not reuse values from a different FBA experiment.')
    evaluation=spec['problem'].get('evaluation')
    if evaluation is not None:
        if not isinstance(evaluation,dict):raise ValueError('problem.evaluation must be a mapping')
        contract={k:binding.get(k) for k in ('fba_mode','biomass_reaction_id','product_reaction_id','objective_reaction_id','objective_direction','minimum_growth_fraction')}
        mismatch={k:(v,contract[k]) for k,v in evaluation.items() if k in contract and v!=contract[k]}
        if mismatch:raise ValueError('Problem requests a different underlying FBA experiment: '+str(mismatch)+'. Generate a new landscape with its own saved configuration.')


def register(workspace,payload):
    payload=dict(payload)
    saved=None
    if str(payload.get('landscape_config_path') or '').strip():
        # Never accept a user override that silently changes the experiment
        # recorded by the previous landscape-generation GUI.
        from .landscape_config import read_existing_landscape_config
        saved=read_existing_landscape_config(workspace,payload['landscape_config_path'])
        inherited={'model_path':saved['model_path'],
                   'features':saved['features'],'bounds':saved['bounds'],
                   'biomass_reaction_id':saved['biomass_reaction_id'],
                   'product_reaction_id':saved['product_reaction_id'],
                   'fba_mode':saved['fba_mode'],
                   'minimum_growth_fraction':saved['minimum_growth_fraction'],
                   'feature_mode':saved['feature_mode'],
                   'landscape_seed':saved['landscape_seed']}
        if not saved.get('model_available') or not saved.get('features') or not saved.get('fba_mode'):
            raise ValueError('Saved landscape configuration is incomplete; cannot safely reconstruct the FBA contract')
        for key,value in inherited.items():
            original=payload.get(key)
            if original not in (None,'',[],{}):
                if key=='features' and isinstance(original,str):original=[v.strip() for v in original.split(',') if v.strip()]
                if key=='bounds' and isinstance(original,str):original=json.loads(original)
                if key=='landscape_seed':original=int(original)
                if key=='minimum_growth_fraction':original=float(original)
                if key=='bounds':
                    norm=lambda x:{k:[float(a) for a in v] for k,v in x.items()}
                    if norm(original)!=norm(value):raise ValueError('Design bounds differ from saved landscape configuration; regenerate the landscape with a new config')
                elif str(original)!=str(value):
                    raise ValueError(f'{key} differs from saved landscape configuration; regenerate the landscape with its own configuration')
            payload[key]=value
    m=workspace.resolve(payload['model_path'],must_exist=True)
    l=workspace.resolve(payload['landscape_path'],must_exist=True)
    if m.suffix.lower() not in ALLOWED_MODELS or l.suffix.lower() not in ALLOWED_ARCHIVES:
        raise ValueError('Unsupported model or landscape format')
    features=payload.get('features',[])
    if isinstance(features,str):features=[x.strip() for x in features.split(',') if x.strip()]
    if not 1<=len(features)<=100 or len(features)!=len(set(features)):raise ValueError('Provide 1–100 distinct design-variable/reaction IDs')
    bounds=payload.get('bounds',{})
    if isinstance(bounds,str):bounds=json.loads(bounds)
    if set(features)!=set(bounds):raise ValueError('Bounds JSON must define min/max for every design variable, and no extra variables')
    pp={'features':features,'bounds':bounds}
    gf,gb=gp.feature_bounds(pp)
    df=read_landscape(l)
    if not len(df):raise ValueError('Empty landscape')
    if any(f not in df.columns for f in features):raise ValueError('Landscape missing one or more design-variable columns')
    X=df[features].apply(pd.to_numeric,errors='coerce').to_numpy(float)
    if not np.isfinite(X).all():raise ValueError('Nonfinite landscape design coordinates')
    if np.any(X<gb[:,0]-1e-6) or np.any(X>gb[:,1]+1e-6):raise ValueError('Some landscape designs fall outside the supplied bounds')
    if 'design_index' in df and not np.array_equal(pd.to_numeric(df.design_index).to_numpy(),np.arange(len(df))):
        raise ValueError('Landscape design_index must equal 0..N-1 in source order; reorder explicitly before registration')
    if 'feasible' not in df and not any(c in df for c in ('status','solver_status')):
        raise ValueError('Landscape needs a feasible or solver-status column; feasibility cannot be inferred from objective values')
    biomass=str(payload.get('biomass_reaction_id','')).strip()
    product=str(payload.get('product_reaction_id','')).strip()
    mode=str(payload.get('feature_mode','medium_uptake'))
    if mode not in ('medium_uptake','reaction_lower_bound','reaction_upper_bound'):raise ValueError('Unsupported design control mode')
    solver_mode=str(payload.get('fba_mode','growth_constrained_product'))
    if solver_mode not in FBA_MODES:
        raise ValueError('Unsupported FBA objective mode; use the mode saved with your landscape configuration')
    objective_reaction=str(payload.get('objective_reaction_id','') or '').strip()
    direction=str(payload.get('objective_direction','max') or 'max').strip().lower()
    if direction not in ('max','min'):raise ValueError('Objective direction must be max or min')
    if solver_mode in ('maximize_biomass','growth_constrained_product','growth_constrained_reaction') and not biomass:
        raise ValueError('This recorded FBA procedure needs a biomass reaction ID; select its landscape configuration')
    if solver_mode in ('maximize_product','growth_constrained_product') and not product:
        raise ValueError('This recorded FBA procedure needs a product reaction ID; select its landscape configuration')
    if solver_mode in ('maximize_reaction','growth_constrained_reaction') and not objective_reaction:
        raise ValueError('This recorded FBA procedure needs the reaction ID for its FBA objective')
    metric_reactions=payload.get('metric_reactions',{}) or {}
    if isinstance(metric_reactions,str):metric_reactions=json.loads(metric_reactions)
    if not isinstance(metric_reactions,dict):raise ValueError('Metric reaction map must be a JSON object')
    if not all(re.fullmatch('[A-Za-z][A-Za-z0-9_]*',str(k)) for k in metric_reactions):raise ValueError('Metric aliases must be simple alphanumeric/underscore names')
    if any(k in STANDARD_METRICS for k in metric_reactions):raise ValueError('Do not override standard metric aliases')
    if 'cobra_model_inspected' not in payload or payload.get('cobra_model_inspected') is not False:
        try:
            mm=load_model(m)
            missing=[r for r in set(features+[r for r in (biomass,product,objective_reaction) if r]+list(metric_reactions.values())) if not mm.reactions.has_id(str(r))]
            if missing:raise ValueError('Model missing reactions: '+', '.join(missing))
            if mode=='medium_uptake':
                exchanges={r.id for r in mm.exchanges}
                if set(features)-exchanges:raise ValueError('For medium uptake, all design variables must be model exchange reactions')
        except ImportError:pass  # Installation/runtime preflight will enforce COBRApy.
    fraction=float(payload.get('minimum_growth_fraction',0.9))
    if not 0<=fraction<=1:raise ValueError('Minimum growth fraction must be between 0 and 1')
    slug=str(payload.get('label','custom')).strip() or 'custom'
    slug=re.sub('[^a-zA-Z0-9_-]+','_',slug).strip('_')[:64]
    token=uuid.uuid4().hex[:12]
    b={'id':token,'case_prefix':'user_'+token,'label':slug,'model_path':str(m),'landscape_path':str(l),'model_sha256':sha(m),'landscape_sha256':sha(l),
       'features':features,'bounds':bounds,'biomass_reaction_id':biomass,'product_reaction_id':product,
       'feature_mode':mode,'fba_mode':solver_mode,'minimum_growth_fraction':fraction,
       'objective_reaction_id':objective_reaction,'objective_direction':direction,
       'metric_reactions':metric_reactions,'rows':len(df),'landscape_seed':payload.get('landscape_seed'),
       'landscape_config_path':str(payload.get('landscape_config_path') or ''),
       'landscape_config_sha256':sha(workspace.resolve(payload['landscape_config_path'],must_exist=True)) if saved else None,
       'fixed_medium':saved.get('fixed_medium',{}) if saved else {},
       'close_unlisted_exchanges':saved.get('sampling_close_unlisted_exchanges',False) if saved else False,
       'solver_from_landscape':saved.get('solver') if saved else None,
       'created_at':time.time(),'source_type':'general'}
    directory=workspace.root/'optimization_bindings'/'general'/token
    directory.mkdir(parents=True,exist_ok=False)
    (directory/'binding.json').write_text(json.dumps(b,indent=2))
    return b


def list_general(workspace):
    out=[]
    for p in sorted((workspace.root/'optimization_bindings'/'general').glob('*/binding.json')):
        try:out.append(json.loads(p.read_text()))
        except (ValueError,OSError):continue
    return out


def preview_problem(workspace,binding,spec_text,prior_rows=5000):
    # Verify immutable bound inputs *before* parsing the model in validate_contract.
    # A replaced/corrupted file should report provenance change, not an incidental
    # COBRA JSON/SBML parse error (and it must never be evaluated).
    if sha(binding['model_path'])!=binding['model_sha256'] or sha(binding['landscape_path'])!=binding['landscape_sha256']:
        raise ValueError('Bound input changed after registration; re-register new file to establish provenance')
    if binding.get('landscape_config_sha256') and sha(binding['landscape_config_path'])!=binding['landscape_config_sha256']:
        raise ValueError('Landscape-generation configuration changed after registration; re-register the source')
    spec=validate_content(spec_text)
    validate_contract(binding,spec)
    df=read_landscape(binding['landscape_path'])
    pp={'features':binding['features'],'bounds':binding['bounds']}
    full,comp=gp.compile_frozen_archive(df,pp,spec)
    n=int(prior_rows)
    if n<1 or n>len(df):raise ValueError(f'Prior size must be between 1 and {len(df)}')
    baseline=gp.compile_live_exact(df.head(n),df.head(n)[binding['features']],{'problem':{**pp,'compiled_problem':comp}},spec)
    return {'problem_hash':comp['problem_hash'],'objectives':comp['objective_names'],'reference_rows':len(df),
            'prior_rows':n,'prior_feasible':int(gp.as_bool(baseline['feasible']).sum()),
            'total_feasible':int(gp.as_bool(full['feasible']).sum()),'fba_solves':0}


def sample_sources(workspace,kind):
    import os
    kinds={'model':ALLOWED_MODELS,'landscape':ALLOWED_ARCHIVES,'problem':{'.yaml','.yml'}}
    if kind not in kinds:raise ValueError('Unknown file type')
    results=[];root=workspace.root.resolve()
    skip={'optimization_assets','optimization_runs','.venv','.git','node_modules','_source_bindings','_exact_bridge_caches'}
    for directory,dirs,files in os.walk(root):
        dirs[:]=[d for d in dirs if d not in skip and not d.startswith('.')]
        for f in files:
            p=Path(directory)/f
            if p.suffix.lower() in kinds[kind]:results.append(str(p.relative_to(root)))
            if len(results)>=250:return sorted(results)
    return sorted(results)
