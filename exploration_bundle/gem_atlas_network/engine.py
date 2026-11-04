"""Single scientific entry point for GUI, YAML and batch. Models remain immutable."""
from pathlib import Path
from dataclasses import asdict
from datetime import datetime,timezone
import json,time,copy,hashlib,platform,importlib.metadata,subprocess,shutil
import networkx as nx
from . import analysis
from .problems import Problem,resolve_model,sha256,config_hash,dump_yaml
from .model_io import load_model
from .genes import reaction_states,rule_genes,parse_rule,gene_details
from .pathways import recognize,discover,TEMPLATES,LIBRARY_VERSION,LIBRARY_SHA256
from .medium import custom_medium_bounds
from .graphing import structural_graph,publication_target_subnetwork,_add_reaction_flow,subsystem_subnetwork,is_currency
from .render import render_publication_figure
from .export import export_graph_tables,export_fluxes,export_summary
from .routes import route_search
RELEASE='0.3.8'

def publication_model_name(model):
    # The bundled SBML header transposes B/G; published reconstruction is iCGB21FR.
    # Preserve the original model bytes, raw model.id and all numerical results.
    return 'iCGB21FR' if model.id == 'iCBG21FR' else model.name

def validate_model_problem(p,m):
    for gid in set(p.analysis.knockout_gene_ids+p.subject.gene_ids):
        if gid not in m.genes:raise ValueError('Unknown model gene: '+gid)
    if p.subject.kind=='route_search':
        q=p.subject.route
        for mid in ([q.substrate,q.product]+q.required_metabolites+q.excluded_metabolites):
            if mid and mid not in m.metabolites:raise ValueError('Unknown route metabolite: '+mid)
        for rid in (q.required_reactions+q.excluded_reactions+q.allowed_reactions):
            if rid not in m.reactions:raise ValueError('Unknown route reaction: '+rid)
    if p.subject.kind=='target_neighborhood':
        entities=m.metabolites if p.subject.entity_kind=='metabolite' else m.reactions
        if p.subject.target_id not in entities:raise ValueError('Unknown target: '+p.subject.target_id)
    if p.subject.subsystem and p.subject.subsystem not in {r.subsystem for r in m.reactions.values()}:raise ValueError('Unknown subsystem')
    if p.fba:
        obj=p.fba.objective
        if obj.source=='model' and not m.objectives:raise ValueError('Model has no objective; select an objective reaction')
        if obj.source=='reaction' and obj.reaction_id not in m.reactions:raise ValueError('Unknown objective reaction')
        if p.fba.medium.source=='custom':custom_medium_bounds(m,p.fba.medium.uptake_bounds,p.fba.medium.close_unselected)
        if set(p.fba.bounds_overrides)-set(m.reactions):raise ValueError('Unknown reaction in bound overrides')
    return True

def prepare(p,base_dir,allowed_root=None,supplied=None):
    if not isinstance(p,Problem):p=Problem.model_validate(p)
    path=resolve_model(p,base_dir,allowed_root,supplied);m=load_model(path)
    validate_model_problem(p,m)
    return p,m,path

def selection(p,m,pathway=None):
    maxr=p.render.max_display_reactions
    s=p.subject
    if s.kind=='recognized_pathway':
        selected=set()
        for step in pathway['steps']:
            if s.branch_only and step['step_id'] not in {'PDC','ADH','LLDH'}:continue
            selected.update(x['reaction_id'] for x in step['matches'])
        return sorted(selected),len(selected)>maxr
    if s.kind=='genes':
        genes=set(s.gene_ids)
        selected=sorted(rid for rid,r in m.reactions.items() if genes & rule_genes(parse_rule(r.gene_rule)))
        return selected,len(selected)>maxr
    if s.kind=='target_neighborhood' and s.entity_kind=='metabolite':
        g=publication_target_subnetwork(m,s.target_id,p.render.direction,p.render.depth,max_reactions=maxr,exclude_currency=p.render.hide_currency_metabolites)
        return [d['id'] for _,d in g.nodes(data=True) if d.get('kind')=='reaction'],g.graph.get('truncated',False)
    if s.kind=='target_neighborhood':
        base=structural_graph(m,p.render.hide_currency_metabolites)
        nodes=nx.single_source_shortest_path_length(base.to_undirected(),'R::'+s.target_id,cutoff=p.render.depth*2)
        selected=sorted((n for n in nodes if n.startswith('R::')),key=lambda n:(nodes[n],n))
        return [n[3:] for n in selected],len(selected)>maxr
    if s.subsystem:
        selected=sorted(rid for rid,r in m.reactions.items() if r.subsystem==s.subsystem)
    else:
        # A compact connected preview centered on a highly connected non-currency metabolite.
        base=structural_graph(m,True)
        mets=[n for n,d in base.nodes(data=True) if d.get('kind')=='metabolite']
        if not mets:return [],False
        source=max(mets,key=lambda n:(base.degree(n),n))
        distances=nx.single_source_shortest_path_length(base.to_undirected(),source)
        selected=[n[3:] for n in sorted(distances,key=lambda n:(distances[n],n)) if n.startswith('R::')]
    return selected,len(selected)>maxr

def graph_for(p,m,rids,states,solution=None,pathway=None):
    G=nx.DiGraph();hide=p.render.hide_currency_metabolites
    for rid in rids:
        r=m.reactions[rid];v=solution.fluxes.get(rid,0.0) if solution else None
        rev=r.upper_bound<=0 and r.lower_bound<0
        _add_reaction_flow(G,m,rid,v,rev,hide,p.subject.target_id)
        if r.reversible:_add_reaction_flow(G,m,rid,v,True,hide,p.subject.target_id)
        d=G.nodes['R::'+rid];d.update(gpr_state=states[rid],bounds_blocked=r.lower_bound==r.upper_bound==0,signed_flux=v)
        # Only the edge orientation in the selected signed solution carries flux.
        if solution:
            for mid,c in r.stoichiometry.items():
                met='M::'+mid;rn='R::'+rid
                forward=(met,rn) if c<0 else (rn,met)
                backward=forward[::-1]
                for edge,sign in ((forward,1),(backward,-1)):
                    if G.has_edge(*edge):G.edges[edge]['flux']=abs(v) if v*sign>p.fba.flux_threshold else 0.0
        for u,w in list(G.in_edges('R::'+rid))+list(G.out_edges('R::'+rid)):
            G.edges[u,w]['disabled']=states[rid]=='disabled' or d['bounds_blocked']
    if pathway:
        for key in pathway['missing_steps']:
            if p.subject.branch_only and key not in {'PDC','ADH','LLDH'}:continue
            G.add_node('missing::'+key,kind='reaction',id=key,label='Missing / unresolved\n'+key,missing=True)
    G.graph.update(view=p.subject.kind,flux_weighted=solution is not None,target=p.subject.target_id or p.subject.pathway_id or '',direction=p.render.direction,depth=p.render.depth,zero_fba=solution is None)
    return G


def gene_context_reactions(model, focus_ids, depth=1, max_reactions=16):
    """Reproducible bounded structural neighborhood around gene-associated reactions.

    Uses non-currency shared metabolites and a mild degree penalty to avoid
    choosing ATP/water/hub detours; does NOT predict changed flux or rerouting.
    """
    focus=set(focus_ids)
    selected=set(focus)
    if not focus or depth<=0:return sorted(selected)[:max_reactions],len(selected)>max_reactions
    met_to_reactions={}
    for rid,r in model.reactions.items():
        for mid in r.stoichiometry:
            if mid not in model.metabolites:continue
            mt=model.metabolites[mid]
            if is_currency(mid,mt.name,mt):continue
            met_to_reactions.setdefault(mid,set()).add(rid)
    frontier=set(focus)
    for _ in range(depth):
        neighbors={}
        for rid in sorted(frontier):
            if rid not in model.reactions:continue
            for mid in model.reactions[rid].stoichiometry:
                for nrid in met_to_reactions.get(mid,()):
                    if nrid in selected:continue
                    # Prefer reactions sharing specific metabolites over generic hubs.
                    specificity=1.0/max(1,len(met_to_reactions[mid]))
                    same_subsystem=(model.reactions[nrid].subsystem and model.reactions[nrid].subsystem==model.reactions[rid].subsystem)
                    neighbors[nrid]=neighbors.get(nrid,0)+specificity+(0.025 if same_subsystem else 0)
        ordered=sorted(neighbors,key=lambda rid:(-neighbors[rid],rid))
        room=max(0,max_reactions-len(selected))
        added=set(ordered[:room]);selected.update(added);frontier=added
        if not frontier:break
    return sorted(selected),len(selected)>=max_reactions

def _package_version(name):
    try:return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:return 'not-installed'

def _clean(value):
    import math
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {k:_clean(v) for k,v in value.items()}
    if isinstance(value,list):return [_clean(v) for v in value]
    return value

def execute(problem,base_dir,output_root,allowed_root=None,supplied=None):
    started=time.perf_counter();p,m,path=prepare(problem,base_dir,allowed_root,supplied)
    output_root=Path(output_root).resolve();out=output_root/p.problem_id/(p.output.subdirectory or 'result')
    if not out.resolve().is_relative_to(output_root):raise ValueError('Output path escapes run root')
    out.mkdir(parents=True,exist_ok=True)
    if p.subject.kind=='route_search':
        return execute_route_query(p,m,path,output_root,started)
    initial=[(r.id,r.lower_bound,r.upper_bound) for r in m.reactions.values()]
    off=p.analysis.knockout_gene_ids;ko=reaction_states(m,off);wt=reaction_states(m)
    pathway=recognize(m,p.subject.pathway_id,off) if p.subject.kind=='recognized_pathway' else None
    if pathway and p.subject.branch_only:
        pathway['display_scope']='Terminal pyruvate-to-ethanol branch (PDC/ADH)' if p.subject.pathway_id=='ETHANOL_FERMENTATION_EMP_PDC_ADH' else 'Terminal pyruvate-to-L-lactate branch'
    ids,truncated=selection(p,m,pathway)
    solutions={};calls=0
    scenarios=['wild_type','knockout'] if off and p.analysis.compare_to_wild_type else ['knockout' if off else 'wild_type']
    if p.analysis.mode=='fba':
        cfg=p.fba
        medium=custom_medium_bounds(m,cfg.medium.uptake_bounds,cfg.medium.close_unselected) if cfg.medium.source=='custom' else {}
        medium.update(cfg.bounds_overrides)
        for name in scenarios:
            overrides=medium.copy();states=ko if name=='knockout' else wt
            overrides.update({rid:(0,0) for rid,state in states.items() if state=='disabled'})
            calls+=1
            sol=analysis.run_fba(m,cfg.objective.reaction_id if cfg.objective.source=='reaction' else None,cfg.direction,overrides)
            solutions[name]=sol
            export_fluxes(sol,out/(name+'_fluxes.csv'))
        # Rank general-network FBA by magnitude; paired diagrams share one reaction list.
        if p.subject.kind=='network' and not p.subject.subsystem:
            ids=sorted(ids,key=lambda rid:(-max((abs(s.fluxes.get(rid,0)) for s in solutions.values()),default=0),rid))
        if p.render.active_only:
            ids=[rid for rid in ids if any(abs(s.fluxes.get(rid,0))>=cfg.flux_threshold for s in solutions.values())]
    ids=ids[:p.render.max_display_reactions]
    figures={}
    # Additional context view never replaces the original compact GPR panel.
    context_ids=[];context_truncated=False
    if p.subject.kind=='genes' and p.render.gene_context_depth>0 and ids:
        context_ids,context_truncated=gene_context_reactions(m,ids,p.render.gene_context_depth,p.render.gene_context_max_reactions)
    for name in scenarios:
        states=ko if name=='knockout' else wt
        sol=solutions.get(name)
        if sol and sol.status!='optimal':continue
        G=graph_for(p,m,ids,states,sol,pathway);G.graph['truncated']=truncated
        title=pathway.get('display_scope',pathway['name']) if pathway else (publication_model_name(m)+' · '+p.subject.kind.replace('_',' '))
        subtitle=name.replace('_',' ').title()+' · '+('Standard FBA; selected solution; dotted edges: zero flux' if sol else 'Structural; zero FBA calls')
        paths=render_publication_figure(G,out/name,title,subtitle,label_mode=p.render.label_mode,show_legend=p.render.include_legend,show_flux_values=p.render.show_flux_values,dpi=p.render.dpi)
        export_graph_tables(G,out,name)
        figures[name]={k:str(v.relative_to(output_root)) for k,v in paths.items() if k in p.render.export_formats or k=='dot'}
        if context_ids:
            cg=graph_for(p,m,context_ids,states,sol,pathway)
            for rid in ids:
                if 'R::'+rid in cg:cg.nodes['R::'+rid].update(target=True,gene_focus=True)
            cg.graph.update(target='GPR focus: '+', '.join(ids[:4]),depth=p.render.gene_context_depth,truncated=context_truncated)
            context_title=publication_model_name(m)+' · gene knockout neighborhood'
            context_subtitle=name.replace('_',' ').title()+' · focus reactions highlighted; context shows model connectivity, not predicted rerouting'
            cpaths=render_publication_figure(cg,out/('context_'+name),context_title,context_subtitle,
                label_mode=p.render.label_mode,show_legend=p.render.include_legend,
                show_flux_values=p.render.show_flux_values,dpi=p.render.dpi)
            export_graph_tables(cg,out,'context_'+name)
            figures['context_'+name]={k:str(v.relative_to(output_root)) for k,v in cpaths.items() if k in p.render.export_formats or k=='dot'}
    assert initial==[(r.id,r.lower_bound,r.upper_bound) for r in m.reactions.values()]
    (out/'problem.yaml').write_text(dump_yaml(p))
    export_summary(m,next(iter(solutions.values()),None),out/'model_summary.json')
    # No flux caching: each explicit execution uses current biological inputs.
    result={'release':RELEASE,'schema_version':'1.0','problem_type':p.problem_type,'problem_id':p.problem_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'configuration_sha256':config_hash(p),'model_sha256':sha256(path),'model_id':m.id,'pathway_library_version':LIBRARY_VERSION,'pathway_library_sha256':LIBRARY_SHA256,'solver':{'name':'SciPy HiGHS','scipy_version':importlib.metadata.version('scipy'),'method':'standard FBA','time_limit_seconds':120},'solver_calls':calls,'analysis_mode':p.analysis.mode,'disabled_reactions':[rid for rid,state in ko.items() if state=='disabled'],'unknown_gpr_reactions':[rid for rid,state in ko.items() if state=='unknown'],'gene_details':gene_details(m,p.subject.gene_ids or off),'pathway':pathway,'pathway_scenario_evidence':discover(m,off) if off else None,'solutions':{k:asdict(v) for k,v in solutions.items()},'figures':figures,'shown_reactions':ids,'display_truncated':truncated,'gene_context':{'focus_reactions':ids,'context_reactions':context_ids,'depth':p.render.gene_context_depth,'max_reactions':p.render.gene_context_max_reactions,'bounded':context_truncated} if context_ids else None,'duration_seconds':time.perf_counter()-started,'warnings':m.warnings+(['FBA solutions may be non-unique. Flux changes are predictions, not experimentally verified rerouting.'] if calls else []),'environment':{'python':platform.python_version(),'platform':platform.platform(),'packages':{k:_package_version(k) for k in ['nicegui','numpy','scipy','networkx','pydot','matplotlib','lxml','PyYAML']},'graphviz':subprocess.run(['dot','-V'],capture_output=True,text=True,timeout=5).stderr.strip() if shutil.which('dot') else 'unavailable; Matplotlib fallback'},'output_directory':str(out.relative_to(output_root))}
    result=_clean(result)
    (out/'result.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (out/'evidence.md').write_text(evidence_report(result))
    return result

def evidence_report(r):
    lines=['# GEM Atlas scientific evidence',f"Problem: {r['problem_id']}",f"Model SHA256: {r['model_sha256']}",f"Configuration SHA256: {r['configuration_sha256']}",f"Mode: {r['analysis_mode']} · solver calls: {r['solver_calls']}",'Gene ON restores normal model availability; OFF evaluates Boolean GPR. No expression or overexpression is inferred.','Models contain metabolic gene associations, not a full genome inventory.','Missing GPR means unknown genetic control.','Structural recognition is not a flux feasibility or in-vivo activity measurement.','All seven source-checked templates are provisional pending independent human expert review. Unqualified complete-pathway claims are disabled.']
    if r['pathway']:
        p=r['pathway'];lines += [f"## {p['name']}",p['scope'],p.get('display_scope','Full template displayed'),f"Status: {p['status']} · {p['representation']}",f"Missing/unresolved steps: {', '.join(p['missing_steps']) or 'none'}",f"Direction-blocked steps: {', '.join(p['bounds_blocked_steps']) or 'none'}",f"GPR scenario: {p['gene_status']}",'| Step | Model reactions | GPR |','| --- | --- | --- |']
        lines += [f"| {e['step_id']} | {', '.join(h['reaction_id'] for h in e['matches'])} | {e['gene_status']} |" for e in p['steps']]
        lines += ['## References']+[f'- {u}' for u in p['reference_urls']]
    for name,s in r['solutions'].items():lines += [f"{name}: {s['status']}; objective = {s['objective_value']}"]
    lines += r['warnings']
    return '\n\n'.join(lines)+'\n'


def _route_image(p,m,route,out,output_root):
    """Render exactly the requested reaction trace, not a generic neighborhood."""
    G=nx.DiGraph()
    # Keep context metabolites for reaction interpretation while highlighting
    # route-defining metabolites as targets in the original publication style.
    for step in route['reaction_trace']:
        rid=step['reaction_id'];rev=step['direction']=='reverse'
        _add_reaction_flow(G,m,rid,None,rev,p.render.hide_currency_metabolites,None)
    for mid in route['metabolite_trace']:
        node='M::'+mid
        if node in G and mid==route['product_id']:G.nodes[node]['target']=True
    G.graph.update(view='route_search',target=route['substrate_id']+' → '+route['product_id'],
                   direction='reaction bounds',depth=len(route['reaction_trace']),
                   zero_fba=True,truncated=False)
    prefix=out/('route_'+str(route['index']))
    title='Route '+str(route['index'])+' · '+m.metabolites[route['substrate_id']].name+' → '+m.metabolites[route['product_id']].name
    subtitle=route['status'].replace('_',' ')+' · projected connectivity; no atom provenance'
    paths=render_publication_figure(G,prefix,title,subtitle,label_mode=p.render.label_mode,
       show_legend=p.render.include_legend,show_flux_values=False,dpi=p.render.dpi)
    # Keep complete all-participant Graphviz map as route_<n>_full_network,
    # and show an SI-readable wrapped step spine as the main route figure.
    if len(route['reaction_trace']) >= 8 and p.metadata.get('case_kind') != 'exploratory':
        from .route_overview import draw_route_overview
        import shutil as _shutil
        for ext in ('png','svg','pdf'):
            _shutil.copy2(prefix.with_suffix('.'+ext), out/('route_'+str(route['index'])+'_full_network.'+ext))
        draw_route_overview(m,route,prefix,p.render.dpi)
    export_graph_tables(G,out,'route_'+str(route['index']))
    return {k:str(v.relative_to(output_root)) for k,v in paths.items() if k in p.render.export_formats or k=='dot'}


def render_selected_route(p,base_dir,output_root,result,number,allowed_root=None,supplied=None):
    """Lazy figure rendering for selecting one of multiple discovered routes."""
    if number<1 or number>len(result['route_search']['routes']):raise ValueError('Unknown route number')
    _,m,_=prepare(p,base_dir,allowed_root,supplied)
    out=Path(output_root)/result['output_directory'];out.mkdir(parents=True,exist_ok=True)
    route=result['route_search']['routes'][number-1]
    paths=_route_image(p,m,route,out,Path(output_root))
    return paths


def execute_route_query(p,m,model_path,output_root,started):
    out=Path(output_root)/p.problem_id/(p.output.subdirectory or 'result')
    out.mkdir(parents=True,exist_ok=True)
    search=route_search(m,p.subject.route)
    figures={}
    if search['routes']:
        figures['route_1']=_route_image(p,m,search['routes'][0],out,Path(output_root))
    (out/'problem.yaml').write_text(dump_yaml(p))
    export_summary(m,None,out/'model_summary.json')
    result={'release':RELEASE,'schema_version':p.schema_version,'problem_type':p.problem_type,
      'problem_id':p.problem_id,'timestamp_utc':datetime.now(timezone.utc).isoformat(),
      'configuration_sha256':config_hash(p),'model_sha256':sha256(model_path),
      'model_id':m.id,'analysis_mode':'route_search','solver_calls':0,
      'route_lp_checks':search.get('lp_checks_performed',0) if p.subject.route.feasibility=='steady_state' else 0,
      'route_search':search,'disabled_reactions':[],'unknown_gpr_reactions':[],
      'gene_details':[],'pathway':None,'pathway_scenario_evidence':None,
      'solutions':{},'figures':figures,'shown_reactions':[x['reaction_id'] for x in search['routes'][0]['reaction_trace']] if search['routes'] else [],
      'display_truncated':not search['search_complete_within_bounds'],
      'duration_seconds':time.perf_counter()-started,'warnings':search['warnings']+m.warnings,
      'environment':{'python':platform.python_version(),'platform':platform.platform(),'packages':{k:_package_version(k) for k in ['numpy','scipy','networkx','PyYAML']}},
      'output_directory':str(out.relative_to(output_root))}
    (out/'result.json').write_text(json.dumps(_clean(result),indent=2,allow_nan=False))
    lines=['# GEM Atlas bounded route-search evidence',f"Problem: {p.problem_id}",
      f"Model SHA256: {sha256(model_path)}",f"Configuration SHA256: {config_hash(p)}",
      f"Found: {search['found_count']} distinct directed traces (bounded, nonexhaustive unless complete)",
      f"Search complete within configured limits: {search['search_complete_within_bounds']}",
      f"Stop reason: {search['stop_reason']}",
      f"LP-feasible routes: {search['steady_state_supported_count']} (tested {search.get('lp_checks_performed', 0)} candidates)",
      f"LP rejected infeasible: {search.get('lp_rejected_infeasible',0)}; LP undetermined: {search.get('lp_rejected_undetermined',0)}",
      '## Candidate reaction traces']
    for r in search['routes']:
        lines.append(f"### Route {r['index']}: {r['substrate_id']} → {r['product_id']}")
        lines.append(' → '.join(x['reaction_id']+'('+x['direction']+')' for x in r['reaction_trace']))
        lines.append('Status: '+r['status'])
        if r.get('support_check'):lines.append('LP check: '+json.dumps(r['support_check']))
    lines+=['## Scientific limitations']+search['warnings']
    (out/'evidence.md').write_text('\n\n'.join(lines)+'\n')
    return result
