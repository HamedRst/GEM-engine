"""Strict, isolated exact-FBA acquisition for compatible CSV landscape archives.

Never guesses historical evaluator definitions. A source needs a verified manifest.
Unknown formats and incompatible states fail closed, leaving source bytes unchanged.
"""
from __future__ import annotations
import hashlib,json,math,os,re,shutil,tempfile
from pathlib import Path
from datetime import datetime,timezone
import pandas as pd
import numpy as np
from scipy.stats import qmc
from scipy.spatial import cKDTree
from gem_atlas_network.model_io import load_model
from gem_atlas_network.analysis import run_fba


def digest(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()


def _jsonhash(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def evaluator_identity(model_path,spec):
    """Identity intentionally includes medium, all design controls, solver and objectives."""
    model_path=Path(model_path).expanduser().resolve()
    variables=spec.get('variables') or []
    if not variables or len(variables)>64:raise ValueError('Define 1-64 design variables')
    names=[]
    for v in variables:
        if set(v)-{'name','reaction_id','column','mode','min','max'}:raise ValueError('Unknown design-variable field')
        if v.get('mode','uptake_limit') not in {'uptake_limit','upper_bound','fixed_flux'}:raise ValueError('Unsupported variable mode')
        if not isinstance(v.get('name'),str) or not isinstance(v.get('reaction_id'),str):raise ValueError('Variable name and reaction ID required')
        lo,hi=v.get('min'),v.get('max')
        if type(lo) not in (float,int) or type(hi) not in (float,int) or not math.isfinite(lo) or not math.isfinite(hi) or lo>=hi:raise ValueError('Invalid variable bounds')
        if 'column' not in v or not isinstance(v['column'],str) or not v['column'].startswith('requested_medium__'):
            raise ValueError('Explicit requested_medium__<reaction_id> output column required for native Calculator compatibility')
        names.append(v['name'])
    if len(set(names))!=len(names):raise ValueError('Variable names must be unique')
    if len(set(v['column'] for v in variables))!=len(names):raise ValueError('Variable columns must be unique')
    biomass=spec.get('biomass_reaction');product=spec.get('product_reaction')
    if not biomass or not product:raise ValueError('Exact biomass/product reaction IDs required')
    frac=spec.get('min_growth_fraction',0.0)
    if type(frac) not in (int,float) or not 0<=frac<=1:raise ValueError('Invalid viability fraction')
    medium=spec.get('medium_bounds') or {}
    for rid,limits in medium.items():
        if not isinstance(limits,list) or len(limits)!=2 or not all(isinstance(x,(float,int)) and math.isfinite(x) for x in limits) or limits[0]>limits[1]:
            raise ValueError('Medium bounds must be finite [lower,upper] pairs')
    ident={'model_sha256':digest(model_path),'model_filename':model_path.name,'variables':variables,
           'biomass_reaction':biomass,'product_reaction':product,'min_growth_fraction':frac,
           'medium_bounds':medium,'solver':'gem_atlas_network.SciPy_HiGHS_v0.3.8',
           'objective_policy':'growth_max_then_product_max_with_growth_floor'}
    return _jsonhash(ident),ident


def _input_spec(model,spec):
    for k in ('biomass_reaction','product_reaction'):
        if spec[k] not in model.reactions:raise ValueError(f'Unknown {k}: {spec[k]}')
    for v in spec['variables']:
        if v['reaction_id'] not in model.reactions:raise ValueError('Unknown variable reaction '+v['reaction_id'])
    for rid in (spec.get('medium_bounds') or {}):
        if rid not in model.reactions:raise ValueError('Unknown medium reaction '+rid)


def fba_point(model,spec,values):
    """Two exact steady-state optimizations, never mutates original GEM."""
    overrides={rid:tuple(map(float,b)) for rid,b in (spec.get('medium_bounds') or {}).items()}
    for v in spec['variables']:
        reaction=model.reactions[v['reaction_id']]
        low,high=overrides.get(v['reaction_id'],(reaction.lower_bound,reaction.upper_bound))
        value=float(values[v['name']]);mode=v.get('mode','uptake_limit')
        if mode=='uptake_limit':low=max(low,-value)
        elif mode=='upper_bound':high=min(high,value)
        else:low=max(low,value);high=min(high,value)
        if low>high:raise ValueError('Incompatible design/medium bounds')
        overrides[v['reaction_id']]=(low,high)
    growth=run_fba(model,spec['biomass_reaction'],'max',overrides)
    if growth.status!='optimal':
        if growth.status!='infeasible':
            raise RuntimeError('Exact FBA growth solve failed: '+str(growth.status)+'; not safe to append')
        return {'growth':float('nan'),'product':float('nan'),'feasible':False,'solver_status':growth.status,'fba_calls':1}
    g=growth.fluxes[spec['biomass_reaction']]
    biomass=model.reactions[spec['biomass_reaction']]
    lower,upper=overrides.get(spec['biomass_reaction'],(biomass.lower_bound,biomass.upper_bound))
    fraction=float(spec.get('min_growth_fraction',0))
    overrides[spec['biomass_reaction']]=(max(lower, fraction*max(0,g)),upper)
    if overrides[spec['biomass_reaction']][0]>upper:
        return {'growth':g,'product':float('nan'),'feasible':False,'solver_status':'growth_constraint_incompatible','fba_calls':1}
    product=run_fba(model,spec['product_reaction'],'max',overrides)
    if product.status not in ('optimal','infeasible'):
        raise RuntimeError('Exact FBA product solve failed: '+str(product.status)+'; not safe to append')
    return {'growth':float(g),'product':float(product.fluxes.get(spec['product_reaction'],float('nan'))) if product.status=='optimal' else float('nan'),
            'feasible':bool(product.status=='optimal'),'solver_status':product.status,'fba_calls':2}


def make_source_manifest(source_csv,model_path,spec):
    """For NEW, independently produced, verified archives; never attach to unverified historical data."""
    source=Path(source_csv).resolve()
    identity,spec_identity=evaluator_identity(model_path,spec)
    return {'schema_version':'1.0','source_sha256':digest(source),'evaluator_sha256':identity,'evaluator':spec_identity,
            'rows':len(pd.read_csv(source)),'provenance_required':'External numerical validation before use with historical data'}


def extend_archive(source_csv,source_manifest,model_path,spec,workspace,group,min_exact_rows,max_new_fba,seed=42,candidate_designs=None):
    """Copy verified source and append unique exact-FBA designs; never edit source."""
    if not isinstance(group,str) or not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_-]{0,79}',group):raise ValueError('Invalid group')
    if type(min_exact_rows)!=int or type(max_new_fba)!=int or not 0<=min_exact_rows<=100000 or not 0<=max_new_fba<=10000:raise ValueError('Invalid acquisition budgets')
    source=Path(source_csv).expanduser().resolve();manifest=json.loads(Path(source_manifest).read_text())
    identity,provenance=evaluator_identity(model_path,spec)
    if manifest.get('schema_version')!='1.0' or manifest.get('source_sha256')!=digest(source) or manifest.get('evaluator_sha256')!=identity:
        raise ValueError('Source landscape identity/evaluator mismatch: do not mix exact evidence')
    model=load_model(model_path);_input_spec(model,spec)
    dest=Path(workspace).expanduser().resolve()/'landscapes'/group
    # Verify before creating/changing a workspace.
    orig=pd.read_csv(source);names=[v['column'] for v in spec['variables']]
    needed=set(names)|{'growth','product','feasible'}
    if not needed.issubset(set(orig.columns)):raise ValueError('Historical landscape columns are not compatible with evaluator schema')
    work=dest/'extended_landscape.csv';record=dest/'working_manifest.json'
    if work.exists()!=record.exists():raise ValueError('Incomplete prior working copy')
    if work.exists():
        wm=json.loads(record.read_text())
        if wm.get('source_sha256')!=manifest['source_sha256'] or wm.get('evaluator_sha256')!=identity or wm.get('working_sha256')!=digest(work):
            raise ValueError('Working landscape mismatched or modified')
        df=pd.read_csv(work)
    else:
        if len(orig)>=min_exact_rows:
            return {'working_landscape':str(source),'source_sha256':manifest['source_sha256'],
                    'working_sha256':manifest['source_sha256'],'source_rows':len(orig),
                    'working_rows':len(orig),'new_designs':0,'new_fba_solves':0,
                    'sufficient_exact_rows':True,'evaluator_sha256':identity,'source_reused_without_copy':True}
        df=orig.copy()
    existing=df[names].apply(pd.to_numeric,errors='raise').to_numpy(dtype=float)
    bounds=np.array([[v['min'],v['max']] for v in spec['variables']],float)
    span=bounds[:,1]-bounds[:,0]
    tree=cKDTree((existing-bounds[:,0])/span) if len(existing) else None
    need=min(max(0,min_exact_rows-len(df)),max_new_fba)
    added=[];calls=0
    if need:
        if candidate_designs is not None:
            if not isinstance(candidate_designs,list):raise ValueError('Candidate designs must be a list')
            candidates=[]
            for item in candidate_designs:
                if not isinstance(item,dict) or set(item)!={v['name'] for v in spec['variables']}:
                    raise ValueError('Candidate keys must match evaluator variables')
                vec=np.array([item[v['name']] for v in spec['variables']],float)
                if not np.isfinite(vec).all() or np.any(vec<bounds[:,0]-1e-9) or np.any(vec>bounds[:,1]+1e-9):
                    raise ValueError('Candidate outside verified evaluator design bounds')
                candidates.append(np.clip((vec-bounds[:,0])/span,0,1))
        else:
            candidates=qmc.Sobol(d=len(names),scramble=True,seed=seed).random_base2(m=int(math.ceil(math.log2(max(64,need*16)))))
        for cand in candidates:
            if len(added)>=need:break
            if tree is not None and tree.query(cand,k=1)[0]<=1e-9:continue
            if any(np.linalg.norm(cand-x)<1e-9 for x in [a['_normalized'] for a in added]):continue
            values={v['name']:float(v['min']+cand[i]*span[i]) for i,v in enumerate(spec['variables'])}
            outputs=fba_point(model,spec,values);calls+=outputs.pop('fba_calls')
            row={col:float('nan') for col in df.columns}
            row.update({v['column']:values[v['name']] for v in spec['variables']});row.update(outputs)
            added.append({'_normalized':cand,'row':row})
    if not work.exists() and not added:
        return {'working_landscape':str(source),'source_sha256':manifest['source_sha256'],
                'working_sha256':manifest['source_sha256'],'source_rows':len(orig),
                'working_rows':len(orig),'new_designs':0,'new_fba_solves':0,
                'sufficient_exact_rows':len(orig)>=min_exact_rows,
                'evaluator_sha256':identity,'source_reused_without_copy':True}
    if not work.exists() or added:
        dest.mkdir(parents=True,exist_ok=True)
        if added:df=pd.concat([df,pd.DataFrame([a['row'] for a in added])],ignore_index=True)
        # Atomic write for the archive; preserve exact original source.
        tmp=dest/'extended_landscape.partial.csv'
        df.to_csv(tmp,index=False);os.replace(tmp,work)
        prior_events=json.loads(record.read_text()).get('events',[]) if record.exists() else []
        wm={'schema_version':'1.0','source_path':str(source),'source_sha256':manifest['source_sha256'],
            'evaluator_sha256':identity,'working_sha256':digest(work),'rows':len(df),'events':prior_events+[
               {'timestamp':datetime.now(timezone.utc).isoformat(),'new_designs':len(added),'new_fba_solves':calls,'seed':seed}]}
        partial_record=dest/'working_manifest.partial.json'
        partial_record.write_text(json.dumps(wm,indent=2))
        os.replace(partial_record,record)
    return {'working_landscape':str(work),'source_sha256':manifest['source_sha256'],'working_sha256':digest(work),
            'source_rows':len(orig),'working_rows':len(df),'new_designs':len(added),'new_fba_solves':calls,
            'sufficient_exact_rows':len(df)>=min_exact_rows,'evaluator_sha256':identity}
