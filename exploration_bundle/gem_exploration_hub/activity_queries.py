"""Portable YAML-driven, no-FBA activity exploration.

The real GEM/landscape pair or an explicitly named bundled demo supplies the
scientific data. Query YAML selects ONE biological view and does not modify the
native Calculator / Network Atlas engines.
"""
from __future__ import annotations
import json, math, re, shutil, hashlib
from pathlib import Path
import numpy as np

FEATURE_TYPES={'gene','reaction','metabolite'}
REGION_METHODS={'angular','circular'}
VIEWS={'overview','feature','regions'}

def validate_payload(payload):
    if not isinstance(payload,dict): raise ValueError('Activity query requires mapping payload')
    source=payload.get('source')
    if source not in {'matched_npz','matched_table'}:raise ValueError('Choose a matched NPZ, CSV or Parquet dataset')
    if source in {'matched_npz','matched_table'}:
        if not isinstance(payload.get('model_path'),str) or not payload['model_path'].strip():raise ValueError('GEM model path required')
        if not isinstance(payload.get('matrix_path'),str) or not payload['matrix_path'].strip():raise ValueError('Matched NPZ flux matrix path required')
    if source=='matched_table':
        for col in ('growth_column','product_column'):
            if not isinstance(payload.get(col,'growth' if col=='growth_column' else 'product'),str) or not payload.get(col,'growth' if col=='growth_column' else 'product').strip():
                raise ValueError('Valid '+col+' is required')
    view=payload.get('view')
    if view not in VIEWS:raise ValueError('Supported views: regions, overview, feature')
    method=payload.get('region_method')
    if method not in REGION_METHODS:raise ValueError('Supported region methods: angular, circular')
    count=payload.get('n_regions')
    if type(count) is not int or count not in range(2,13):raise ValueError('n_regions must be an integer 2–12')
    feature_type=payload.get('feature_type')
    if view in {'overview','feature'} and feature_type not in FEATURE_TYPES:raise ValueError('Choose gene, reaction, or metabolite')
    if view=='feature':
        fid=payload.get('feature_id')
        if not isinstance(fid,str) or not fid.strip() or len(fid)>250:raise ValueError('Feature ID required (case-sensitive)')
    if payload.get('new_fba_policy','never')!='never':raise ValueError('Activity replot must not acquire new FBA')
    if 'model_sha256' in payload and (not isinstance(payload['model_sha256'],str) or not re.fullmatch(r'[0-9a-fA-F]{64}',payload['model_sha256'])):
        raise ValueError('model_sha256 must be a SHA256 hex digest')
    for key in ('growth_label','product_label'):
        if key in payload and (not isinstance(payload[key],str) or not 1<=len(payload[key])<=120):
            raise ValueError(key+' must be a short nonempty axis label')
    if 'sample_size' in payload and (type(payload['sample_size']) is not int or not 100<=payload['sample_size']<=100000):
        raise ValueError('sample_size must be an integer from 100 to 100000')
    return payload

def _bundle_features(js):
    # Read known schema, not arbitrary executable content: extract JSON assigned to const.
    prefix='window.GEM_BUNDLE='
    st=js.find(prefix)
    if st<0:raise ValueError('Expected GEM_BUNDLE schema not found')
    # source bundle writes JSON terminated by ;, with embedded semicolons only inside strings.
    from json import JSONDecoder
    desc,_=JSONDecoder().raw_decode(js[st+len(prefix):].lstrip())
    return desc.get('feature',[])

def run_activity(payload, output, root):
    validate_payload(payload)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    root=Path(root)
    assets=root/'gem_exploration_tab'/'assets'
    target=output/'viewer';target.mkdir(exist_ok=False)
    template=(assets/'index.html').read_text(encoding='utf-8')
    if payload['source']=='bundled_demo':
        bundle=(assets/'bundle.js').read_bytes()
        source_info={'source':'explicit_demo_only','example':'E. coli antiEpEX-scFv','preview':True,
                     'warning':'Bundled 12,000-condition activity sample is not full publication flux coverage.'}
    else:
        from gem_exploration_tab.build_bundle import export
        from tempfile import TemporaryDirectory
        model_file=Path(payload['model_path']).expanduser().resolve()
        matrix_file=Path(payload['matrix_path']).expanduser().resolve()
        if not model_file.is_file() or model_file.suffix.lower() not in {'.json','.xml','.sbml'}:
            raise ValueError('Valid COBRA JSON/SBML model file required')
        allowed={'.npz'} if payload['source']=='matched_npz' else {'.csv','.parquet'}
        if not matrix_file.is_file() or matrix_file.suffix.lower() not in allowed:
            raise ValueError('Matching '+('/'.join(sorted(allowed)))+' flux matrix required')
        source_model_hash=hashlib.sha256(model_file.read_bytes()).hexdigest()
        if payload.get('model_sha256') and payload['model_sha256'].lower()!=source_model_hash:
            raise ValueError('Model file SHA256 differs from the fingerprint saved in YAML')
        if model_file.suffix.lower()=='.json':model=json.loads(model_file.read_text(encoding='utf-8'))
        else:
            from cobra.io import read_sbml_model,save_json_model
            with TemporaryDirectory() as temp:
                converted=Path(temp)/'converted.json';save_json_model(read_sbml_model(str(model_file)),str(converted))
                model=json.loads(converted.read_text(encoding='utf-8'))
        sampled=int(payload.get('sample_size',12000))
        if payload['source']=='matched_npz':
            with np.load(matrix_file,allow_pickle=False) as matrix:
                required={'growth','product','flux','reaction_ids'}
                if not required.issubset(matrix.files):raise ValueError('Missing NPZ inputs: '+str(sorted(required-set(matrix.files))))
                if 'model_sha256' in matrix.files:
                    embedded=str(np.asarray(matrix['model_sha256']).item())
                    if embedded.lower()!=source_model_hash:raise ValueError('NPZ embedded model SHA256 differs from supplied GEM; refusing mismatched archive')
                export(model,matrix['growth'],matrix['product'],matrix['flux'],target/'bundle.js',
                    name=(model.get('name') or model.get('id') or 'GEM'),
                    sample_size=sampled,reaction_ids=matrix['reaction_ids'],
                    growth_label=payload.get('growth_label','Growth (model objective units)'),
                    product_label=payload.get('product_label','Product (model flux units)'))
        else:
            # Generic wide CSV/Parquet with one column per reaction; no biological IDs hard-coded.
            # Every reaction flux is required; a phenotype-only landscape cannot infer it.
            import pandas as pd
            if matrix_file.suffix.lower()=='.csv':
                columns=list(pd.read_csv(matrix_file,nrows=0).columns)
            else:
                # Read only Parquet schema; a 50K-by-2500 matrix may be gigabytes.
                import pyarrow.parquet as pq
                columns=list(pq.read_schema(matrix_file).names)
            grow=payload.get('growth_column','growth');prod=payload.get('product_column','product')
            if grow==prod:raise ValueError('Growth and product columns must be distinct')
            ids=[r['id'] for r in model['reactions']]
            if len(set(ids))!=len(ids):raise ValueError('Duplicate reaction IDs in GEM')
            if all(i in columns and i not in {grow,prod} for i in ids):rx_cols=ids
            elif all('flux__'+i in columns for i in ids):rx_cols=['flux__'+i for i in ids]
            else:
                missing=[i for i in ids if i not in columns and 'flux__'+i not in columns]
                raise ValueError(f'Landscape lacks complete internal flux vectors; missing {len(missing)} reactions (first: {missing[:5]}). A phenotype-only landscape is insufficient; acquire a matching flux archive, not inferred activities.')
            if grow not in columns or prod not in columns:raise ValueError('Missing growth/product columns in landscape')
            needed=[grow,prod]+[c for c in rx_cols if c not in {grow,prod}]
            if matrix_file.suffix.lower()=='.csv':
                table=pd.read_csv(matrix_file,usecols=needed,dtype={c:'float32' for c in needed},low_memory=False)
            else:
                table=pd.read_parquet(matrix_file,columns=needed)
            if not len(table):raise ValueError('Empty landscape table')
            export(model,table[grow].to_numpy(),table[prod].to_numpy(),table[rx_cols].to_numpy(dtype=np.float32),
                target/'bundle.js',name=(model.get('name') or model.get('id') or 'GEM'),sample_size=sampled,
                reaction_ids=np.asarray(ids),
                growth_label=payload.get('growth_label','Growth (model objective units)'),
                product_label=payload.get('product_label','Product (model flux units)'))
        bundle=(target/'bundle.js').read_bytes()
        source_info={'source':payload['source'],'model_path':str(model_file),'matrix_path':str(matrix_file),
                'model_sha256':source_model_hash,
                'alignment_check':'complete ordered reaction IDs and optional model SHA256; reaction IDs alone cannot prove same stoichiometry',
                'matrix_sha256':hashlib.sha256(matrix_file.read_bytes()).hexdigest()}
    if payload['source']=='bundled_demo': (target/'bundle.js').write_bytes(bundle)
    features=_bundle_features(bundle.decode('utf-8'))
    if payload['view']=='feature':
        wanted=(payload['feature_type'],payload['feature_id'])
        if not any((f.get('type'),f.get('id'))==wanted for f in features):
            raise ValueError('Selected feature ID not present in matching model/data: '+str(wanted))
    from json import JSONDecoder
    raw=bundle.decode('utf-8');src=raw.index('window.GEM_BUNDLE=')+len('window.GEM_BUNDLE=')
    b,_=JSONDecoder().raw_decode(raw[src:])
    coverage={'model_name':b.get('name'),'total_model_features':{'genes':b['model_genes'],'reactions':b['model_reactions'],'metabolites':b['model_metabolites']},
              'available_activities':{'genes':b['ng'],'reactions':b['nr'],'metabolites':b['nm']},
              'excluded_genes':b['model_genes']-b['ng'],
              'eligible_gene_ids':[f['id'] for f in features if f['type']=='gene'],
              'unmapped_gene_ids':[],
              'objective_rows_valid':b['nall'],'activity_sample_rows':b['n'],
              'source_objective_excluded':b.get('source_objective_excluded','not archived for demo'),
              'interpretation':{'gene':'Model-predicted, GPR-associated flux utilization; not expression or individual gene causal activity',
                                'reaction':'Stored FBA reaction flux; can be signed',
                                'metabolite':'Stoichiometry-weighted absolute flux processing; not concentration or pool size'},
              'uncertainty':'Biochemical alternatives and FBA degeneracy remain; flux outputs are not validated biological causality'}
    if payload['source']!='bundled_demo':
        known={f['id'] for f in features if f['type']=='gene'}
        coverage['unmapped_gene_ids']=[g['id'] for g in model['genes'] if g['id'] not in known]
    (output/'FEATURE_COVERAGE.json').write_text(json.dumps(coverage,indent=2),encoding='utf-8')
    options={k:payload.get(k) for k in ('view','feature_type','feature_id','region_method','n_regions')}
    # Inject only a validated JSON config, while leaving the original scientific renderer intact.
    marker='<script src="bundle.js"></script>'
    if marker not in template:raise RuntimeError('Activity viewer template changed: missing known script tag')
    template=template.replace(marker,'<script>window.GEM_QUERY_CONFIG='+json.dumps(options,separators=(',',':')).replace('<','\\u003c')+';</script>'+marker,1)
    (target/'index.html').write_text(template,encoding='utf-8')
    (output/'query_result.json').write_text(json.dumps({'query':options,'source':source_info,'view':'viewer/index.html'},indent=2))
    return {'problem_type':'exploration.activity','result_directory':str(output),'view':'viewer/index.html',
            'source':source_info,'query':options,'feature_coverage':'FEATURE_COVERAGE.json','new_fba_calls':0}
