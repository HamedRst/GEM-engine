#!/usr/bin/env python3
"""Model-neutral archive-to-single-page exporter. Requires numpy; JSON COBRA model.
Example: python build_bundle.py --model my_model.json --matrix my_landscape.npz --output my_bundle.js
NPZ arrays: growth[n], product[n], flux[n,reactions] and optional reaction_ids[reactions].
No model-specific identifiers, objectives, species, paths, or reaction names are hard-coded.
"""
import argparse,base64,gzip,json,re
from pathlib import Path
import numpy as np

def pack(a):return base64.b64encode(gzip.compress(np.ascontiguousarray(a).tobytes(),compresslevel=6)).decode()
def gpr_genes(rule, valid_ids):
    """Conservative COBRA GPR leaf IDs, including numeric and punctuated IDs."""
    return set(x.strip() for x in re.split(r"(?i)\b(?:and|or)\b|[()]",rule or "") if x.strip() in valid_ids)

def model_links(model):
    rxlist=model['reactions']; geneids=set(z['id'] for z in model['genes']);
    gene_to_rx={k:[] for k in geneids}; met_to_rx={z['id']:[] for z in model['metabolites']};
    rx_to_mets={};rx_to_genes={}
    for r in rxlist:
        rid=r['id'];names=gpr_genes(r.get('gene_reaction_rule',''),geneids)
        rx_to_genes[rid]=sorted(names)
        for gene in names:gene_to_rx[gene].append(rid)
        metids=list(r.get('metabolites',{}))
        rx_to_mets[rid]=metids
        for mid in metids:
            if mid in met_to_rx:met_to_rx[mid].append(rid)
    return {'gene_to_reaction':gene_to_rx,'reaction_to_metabolite':rx_to_mets,'reaction_to_gene':rx_to_genes,'metabolite_to_reaction':met_to_rx}

def export(model, growth, product, flux, output, *, name='Uploaded GEM / landscape', sample_size=12000, reaction_ids=None, growth_label='Growth (model objective units)', product_label='Product (model flux units)'):
    reaction=model['reactions'];genes=model['genes']; mets=model['metabolites']
    ids=[r['id'] for r in reaction]
    if reaction_ids is None:raise ValueError('Reaction IDs are required to verify flux-column/model correspondence. Supply matrix reaction_ids.')
    if list(map(str,reaction_ids))!=ids:raise ValueError('Reaction order mismatch: do not silently assume flux and model reaction order match.')
    g=np.asarray(growth,dtype=np.float64);p=np.asarray(product,dtype=np.float64);V=np.asarray(flux,dtype=np.float32)
    if V.ndim!=2 or g.ndim!=1 or p.ndim!=1 or V.shape!=(len(g),len(ids)) or p.shape!=g.shape:raise ValueError('Mismatching matrix dimensions')
    if len(set(ids))!=len(ids):raise ValueError('Duplicate reaction IDs in model')
    if not len(g):raise ValueError('No landscape conditions')
    if not np.isfinite(V).all():
        raise ValueError('Flux matrix contains non-finite values; missing FBA fluxes cannot be interpreted as zero. Supply complete valid vectors or preprocess with documented exclusions.')
    good=np.isfinite(g)&np.isfinite(p)&(g>=0)&(p>=0)
    if not np.any(good):raise ValueError('No valid points')
    g=g[good];p=p[good];V=V[good]
    gmax=float(g.max()); pmax=float(p.max())
    if gmax<=0 or pmax<=0:raise ValueError('Growth and product maxima must be positive')
    rng=np.random.default_rng(42)
    index=np.sort(rng.choice(len(g),min(len(g),sample_size),replace=False))
    xy=np.column_stack([np.rint(g/gmax*65535),np.rint(p/pmax*65535)]).clip(0,65535).astype('<u2')
    # Feature summary normalized reaction use: log1p(|v|/q90) averaged over unambiguous GPRs.
    chosen=V[index]; sc=np.maximum(np.quantile(np.abs(chosen),.9,axis=0),.05)
    rx_use=np.log1p(np.abs(chosen)/sc).astype(np.float32)
    bygene={z['id']:[] for z in genes}
    for j,r in enumerate(reaction):
        rule=r.get('gene_reaction_rule','')
        if not re.search(r'\bor\b',rule,flags=re.I):
            for tok in gpr_genes(rule,set(bygene)):
                if tok in bygene:bygene[tok].append(j)
    feature=[];mat=[]
    # keep model-wide inventory; absent GPR is unavailable, not zero.
    for gene in genes:
        linked=bygene[gene['id']]
        if not linked:continue
        value=rx_use[:,linked].mean(axis=1)
        mx=max(np.percentile(value,99),1e-9)
        mat.append(np.rint(np.clip(value/mx,0,1)*255).astype('uint8'))
        feature.append(dict(type='gene',id=gene['id'],name=gene.get('name') or gene['id'],linked=len(linked),scale=float(mx), status='GPR-linked use (OR-ambiguous rules excluded)'))
    geneCount=len(feature)
    mi={met['id']:k for k,met in enumerate(mets)}
    # avoid huge dense stoichiometric matrix
    turn=np.zeros((len(index),len(mets)),dtype=np.float32)
    for j,r in enumerate(reaction):
        av=np.abs(chosen[:,j]);
        for mid,c in r.get('metabolites',{}).items():
            if mid in mi:turn[:,mi[mid]]+=.5*abs(float(c))*av
    for k,met in enumerate(mets):
        value=np.log1p(turn[:,k]);mx=max(float(np.percentile(value,99)),1e-9)
        mat.append(np.rint(np.clip(value/mx,0,1)*255).astype('uint8'))
        feature.append(dict(type='metabolite',id=met['id'],name=met.get('name') or met['id'],linked=None,scale=float(mx),status='Flux-sum processing; not concentration'))
    metabCount=len(mets)
    q99=np.maximum(np.percentile(np.abs(chosen),99,axis=0),1e-8)
    signed=np.rint(np.clip(chosen/q99,-1,1)*127).astype('int8')
    # -128 = missing/sentinel; signed entries within -127..127
    for j,r in enumerate(reaction):
        feature.append(dict(type='reaction',id=r['id'],name=r.get('name') or r['id'],linked=None,scale=float(q99[j]),status='Signed reaction flux; clipped at per-reaction 99th percentile'))
    matrix=np.asarray(mat,dtype=np.uint8)
    if matrix.shape!=(geneCount+metabCount,len(index)):raise AssertionError('Feature shape')
    data=dict(name=name,x_label=str(growth_label),y_label=str(product_label),source_objective_valid=int(len(g)),source_objective_excluded=int(len(good)-len(g)),model_reactions=len(ids),model_genes=len(genes),model_metabolites=len(mets),feature=feature,
              gene_exclusions={'total':len(genes)-geneCount,'reason':'No unambiguous GPR-linked reaction flux; OR isozyme attribution excluded'},
              flux_policy='Complete finite, single saved flux state per landscape condition; no combination of independent FBA optima',
              gmax=gmax,pmax=pmax,nall=len(g),n=len(index),ng=geneCount,nm=metabCount,nr=len(ids),
              allxy=pack(xy),samplexy=pack(xy[index]),activity=pack(matrix),signed=pack(signed.T),links=model_links(model))
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text('window.GEM_BUNDLE='+json.dumps(data,separators=(',',':'))+';\n')
    return data

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--model',required=True);a.add_argument('--matrix',required=True,help='NPZ arrays growth,product,flux,reaction_ids')
    a.add_argument('--output',required=True);a.add_argument('--name',default='Uploaded GEM / landscape');a.add_argument('--sample-size',type=int,default=12000)
    o=a.parse_args();model=json.loads(Path(o.model).read_text());z=np.load(o.matrix,allow_pickle=False)
    d=export(model,z['growth'],z['product'],z['flux'],o.output,name=o.name,sample_size=o.sample_size,reaction_ids=z['reaction_ids'])
    print(f"Exported {d['name']}: {d['nall']} conditions, {d['ng']} genes, {d['nr']} reactions, {d['nm']} metabolites")
if __name__=='__main__':main()
