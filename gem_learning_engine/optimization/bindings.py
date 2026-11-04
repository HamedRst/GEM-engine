"""Bind an existing native 50K growth/product run without regenerating its landscape."""
from pathlib import Path
import importlib.util
import json
import shutil
import uuid
import numpy as np
import pandas as pd
import yaml
from .catalog import ASSETS, sha
from .worker import atomic_json

def list_bindings(workspace):
    records=[]
    for path in (workspace.root/'optimization_bindings').glob('*/binding.json'):
        binding=json.loads(path.read_text())
        for key in ['growth_product','product_cost','constrained_product']:
            records.append({**binding,'case':binding['species']+'__'+key,'problem_key':key})
    return records

def bind_native(workspace, config_path):
    from ..config import load_config
    from ..model_io import model_fingerprint
    config_path=workspace.resolve(config_path,must_exist=True);cfg=load_config(config_path)
    run=workspace.resolve(cfg.project.output_dir,must_exist=True)
    snapshot=run/'metadata/config_snapshot.yaml'
    if not snapshot.exists():raise ValueError('Run has no saved configuration snapshot')
    raw=yaml.safe_load(snapshot.read_text());cfg=load_config(snapshot)
    model_path=Path(cfg.model.path).expanduser()
    if not model_path.is_absolute():model_path=workspace.resolve(model_path,must_exist=True)
    else:model_path=workspace.resolve(model_path,must_exist=True)
    if cfg.simulation.n_simulations!=50000:raise ValueError('Binding currently requires a completed 50K reference landscape')
    if cfg.optimization.mode!='growth_constrained_product':raise ValueError('Select a landscape generated with Growth-Constrained Product to preserve the original exact source contract')
    status=json.loads((run/'metadata/run_status.json').read_text()) if (run/'metadata/run_status.json').exists() else {}
    if status.get('saved_simulations')!=50000 or status.get('invalid_chunks') or status.get('failed_chunks'):
        raise ValueError('Landscape completion/integrity checks failed')
    token=uuid.uuid4().hex[:12];case='custom_'+token+'__growth_product';root=workspace.root/'optimization_bindings'/token
    view=root/'native_run_view';(view/'metadata').mkdir(parents=True)
    for name in ['config_snapshot.yaml','run_manifest.json','design_metadata.json']:
        shutil.copy2(run/'metadata'/name,view/'metadata'/name)
    shutil.copytree(run/'catalogs',view/'catalogs');shutil.copytree(run/'design_cache',view/'design_cache')
    medium=pd.read_parquet(run/'catalogs/medium_space.parquet')
    variables=medium[medium['role']=='variable'].sort_values('medium_feature_index')['reaction_id'].tolist()
    atomic_json(view/'metadata/preflight.json',{'n_simulations':50000,'variable_medium_reaction_ids':variables,
                'origin':'saved medium-space catalog; no FBA calls'})
    bridge_path=ASSETS/'benchmarks/GEM_5Species_Growth_Product_ResourceCost_Benchmark_v0.1.0/engine_bridge.py'
    spec=importlib.util.spec_from_file_location('_native_source',bridge_path);bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    desc=bridge._space_description(view,model_path)
    if desc['manifest'].get('model_fingerprint') != model_fingerprint(desc['model'],model_path):raise ValueError('Model fingerprint differs from the generated landscape')
    cache=bridge._find_design_cache(view);arr=np.load(cache)
    if len(arr)!=50000:raise ValueError('Reference design cache is incomplete')
    X,Xp,mode=bridge._decode(arr,desc['bounds']);table,diag=bridge._extract_outcomes(run,50000)
    table=table.sort_values('design_index').reset_index(drop=True)
    F=bridge._feasible(table,desc['feasible_statuses'])
    for j,rid in enumerate(desc['features']):table[rid]=Xp[:,j]
    table['resource_cost']=np.mean(X,axis=1);table['resource_efficiency']=1-table['resource_cost'];table['feasible']=F
    table['status']=np.where(F,'archive_exact_growth_product__deterministic_cost','infeasible')
    prep=root/'prepared'/case;prep.mkdir(parents=True)
    bp=prep/'baseline_compact.csv.gz';table.to_csv(bp,index=False,compression='gzip')
    pc={'problem':{'case_name':case,'run_dir':str(view),'model_path':str(model_path),'features':desc['features'],
                  'bounds':{rid:list(map(float,desc['bounds'][j])) for j,rid in enumerate(desc['features'])},
                  'feature_labels':desc['feature_labels'],'biomass_reaction_id':desc['biomass'],'product_reaction_id':desc['product'],
                  'minimum_growth_fraction':desc['minimum_growth_fraction'],'baseline_table':str(bp),'design_cache':str(cache)}}
    (prep/'problem_config.yaml').write_text(yaml.safe_dump(pc,sort_keys=False));shutil.copy2(bridge_path,root/'engine_bridge.py')
    source_cfg={'paths':{'input_models_root':str(model_path.parent)},'cases':{case:{'run_dir':str(view),'model_glob':model_path.name}}}
    (root/'study_config.yaml').write_text(yaml.safe_dump(source_cfg))
    binding={'case':case,'species':'custom_'+token,'label':cfg.project.name,'problem_key':'growth_product','source_benchmark':str(root),
             'source_kind':'growth_product','source_case':case,'model_path':str(model_path),'model_sha256':sha(model_path),
             'landscape_path':str(bp),'config_path':str(config_path),'landscape_seed':cfg.simulation.seed,
             'reference_rows':50000,'binding_fba_solves':0,'design_cache_sha256':sha(cache),'native_config_sha256':sha(snapshot)}
    atomic_json(root/'binding.json',binding);return binding
