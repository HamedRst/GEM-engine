"""Conservative full-model chemical matching and exact intermediate continuity.

Provisional templates expose computational evidence but cannot emit an unqualified
complete-pathway claim until independent expert curation is recorded in the library.
"""
import json, hashlib, re
from pathlib import Path
from itertools import product
LIBRARY=Path(__file__).with_name('pathway_library')
LIBRARY_ID='atlas_canonical'
LIBRARY_VERSION='1.0.0'
TEMPLATES={p.stem:json.loads(p.read_text()) for p in sorted(LIBRARY.glob('*.json')) if p.stem!='metabolites'}
IDENTITIES=json.loads((LIBRARY/'metabolites.json').read_text())
LIBRARY_SHA256=hashlib.sha256(b''.join(p.read_bytes() for p in sorted(LIBRARY.glob('*.json')))).hexdigest()
CARRIERS={'h','h2o','nad','nadh','nadp','nadph','atp','adp','gdp','gtp','pi','coa','co2'}

def normalized_id(mid, compartment):
    x=mid.removeprefix('M_').replace('__91__','[').replace('__93__',']')
    if '[' in x: x=x.split('[')[0]
    if compartment and x.endswith('_'+compartment): x=x[:-(len(compartment)+1)]
    # COBRA encodes double underscore stereochemistry as single underscore in some models.
    if x.endswith(('_L','_D')) and not x.endswith(('__L','__D')):x=x[:-2]+'__'+x[-1]
    return x

def identify(m):
    candidates={}; evidence={}
    for role,d in IDENTITIES.items():
        basis=[]
        # Exact references, validated against formula when present. No fuzzy name matching.
        for key,vals in d['annotations'].items():
            actual=m.annotations.get(key,[]); actual=[actual] if isinstance(actual,str) else actual
            if set(actual)&set(vals):basis.append(key)
        bigg=m.annotations.get('bigg.metabolite',[]);bigg=[bigg] if isinstance(bigg,str) else bigg
        if role in [normalized_id(x,m.compartment) for x in bigg]:basis.append('bigg.metabolite')
        if normalized_id(m.id,m.compartment)==role:basis.append('BiGG-compatible ID')
        exact_name=m.name.casefold() in [x.casefold() for x in d['names']]
        formula_ok=bool(m.formula) and m.formula in d['formulas']
        if basis and (not m.formula or formula_ok):
            candidates[role]=True;evidence[role]={'basis':basis,'formula':m.formula,'formula_verified':formula_ok}
        elif exact_name and formula_ok:
            candidates[role]=True;evidence[role]={'basis':['exact curated chemical name + formula'],'formula':m.formula,'formula_verified':True}
    if len(candidates)!=1:return None, {'ambiguous_roles':sorted(candidates)}
    role=next(iter(candidates));return role,evidence[role]

def _matches(model, steps):
    identity={mid:identify(m) for mid,m in model.metabolites.items()}
    found={k:[] for k in steps}
    unresolved=set()
    for rid,r in model.reactions.items():
        eq={}; bindings={}; evid={}; comps=set(); bad=False
        for mid,c in r.stoichiometry.items():
            if not c:continue
            role,ev=identity[mid]
            if role is None:bad=True;break
            if role=='h':continue
            if role in bindings and bindings[role]!=mid:bad=True;break
            bindings[role]=mid;eq[role]=eq.get(role,0)+c;evid[mid]=ev
            comps.add(model.metabolites[mid].compartment)
        if bad:
            unresolved.add(rid);continue
        if len(comps)!=1 or not eq:continue
        comp=next(iter(comps))
        for k,st in steps.items():
            for alt,expected in enumerate([st['stoichiometry']]+st['alternatives']):
                if set(eq)!=set(expected):continue
                scale=eq[next(iter(expected))]/expected[next(iter(expected))]
                if not scale or any(abs(eq[x]-scale*expected[x])>1e-7*max(1,abs(eq[x])) for x in eq):continue
                sign=1 if scale>0 else -1
                allowed=r.upper_bound>0 if sign==1 else r.lower_bound<0
                found[k].append({'reaction_id':rid,'compartment':comp,'direction_sign':sign,'direction_allowed':allowed,'alternative_index':alt,'bindings':bindings.copy(),'identity_evidence':evid.copy(),'equation':eq,'gpr':r.gene_rule,'bounds':[r.lower_bound,r.upper_bound]})
    return found, unresolved

def group_routes(group):
    if isinstance(group,str):return [[group]]
    return [[group[0]],group[1]]

def recognize(model, template_id, off=()):
    from .genes import reaction_states
    t=TEMPLATES[template_id]; found,unresolved=_matches(model,t['steps'])
    states=reaction_states(model,off)
    # Enumerate small defined OR choices, then choose a globally consistent assignment.
    best=[];bestmissing=[];bestcomp=None;visited=0
    for route in product(*(group_routes(g) for g in t['required_step_groups'])):
        names=sum(route,[])
        # Most constrained biochemical transformations first; carriers do not establish continuity.
        names=sorted(names,key=lambda k:(len(found[k]),k))
        for comp in sorted({x['compartment'] for rows in found.values() for x in rows}):
            def dfs(i,bindings,selected,missing):
                nonlocal best,bestmissing,bestcomp,visited
                visited+=1
                if visited>100000:raise ValueError('Pathway mapping exceeded search limit; unresolved')
                if len(selected)+len(names)-i<len(best):return
                if i==len(names):
                    if len(selected)>len(best):best=selected[:];bestmissing=missing[:];bestcomp=comp
                    return
                k=names[i]
                for hit in sorted(found[k],key=lambda h:(not h['direction_allowed'],h['reaction_id'])):
                    if hit['compartment']!=comp:continue
                    core={role:mid for role,mid in hit['bindings'].items() if role not in CARRIERS}
                    if any(role in bindings and bindings[role]!=mid for role,mid in core.items()):continue
                    dfs(i+1,{**bindings,**core},selected+[{'step_id':k,**hit}],missing)
                dfs(i+1,bindings,selected,missing+[k])
            dfs(0,{},[],[])
    if not best:
        bestmissing=[g if isinstance(g,str) else g[0] for g in t['required_step_groups']]
    fully_represented=bool(best) and not bestmissing
    blocked=[x['step_id'] for x in best if not x['direction_allowed']]
    # Structural representation and allowed biochemical direction are separately recorded.
    qualified=fully_represented and not blocked
    if qualified and t['complete_claims_enabled']:status='Complete in model'
    elif qualified:status='Unresolved'
    elif best:status='Partially represented'
    elif unresolved:status='Unresolved'
    else:status='Not identified in model'
    # Include all chemically equivalent isoenzymes consistent with the selected backbone.
    selectedbindings={role:mid for h in best for role,mid in h['bindings'].items() if role not in CARRIERS}
    step_evidence=[]
    for hit in best:
        alternatives=[h for h in found[hit['step_id']] if h['compartment']==bestcomp and all(role not in selectedbindings or selectedbindings[role]==mid for role,mid in h['bindings'].items() if role not in CARRIERS)]
        allowed=[h for h in alternatives if h['direction_allowed']]
        statuses=[states[h['reaction_id']] for h in allowed]
        gs='supported' if 'supported' in statuses else 'unknown' if 'unknown' in statuses else 'disabled' if statuses else 'bounds_blocked'
        step_evidence.append({'step_id':hit['step_id'],'matches':alternatives,'gene_status':gs})
    gs='disrupted' if any(e['gene_status']=='disabled' for e in step_evidence) else 'unknown' if any(e['gene_status']=='unknown' for e in step_evidence) else 'supported' if step_evidence else 'unknown'
    return {'template_id':template_id,'name':t['name'],'scope':t['scope'],'template_version':t['release_version'],'curation_status':t['curation_status'],'status':status,'representation':'all required connected transformations matched' if fully_represented else 'partial or unresolved','all_steps_matched':fully_represented,'direction_compatible':qualified,'missing_steps':bestmissing,'bounds_blocked_steps':blocked,'compartment':bestcomp,'gene_status':gs,'steps':step_evidence,'unresolved_reaction_count':len(unresolved),'reference_urls':t['reference_urls'],'note':'Provisional definition: independent expert review pending. Model representation is not in-vivo activity. No biological absence inferred.'}

def discover(model, off=()):return {k:recognize(model,k,off) for k in TEMPLATES}
