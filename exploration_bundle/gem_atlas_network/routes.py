"""Bounded, reproducible reaction-trace discovery. NOT atom mapping or an EFM finder.

Node projection is deliberately labeled topological: a metabolite co-occurrence
in a reaction is not proof that the source atoms become product atoms.
"""
from __future__ import annotations
from collections import deque
from time import monotonic
import math
import networkx as nx
from .graphing import is_currency

BOUNDARY_PREFIXES=('EX_', 'DM_', 'SK_', 'SINK_', 'EXCHANGE_', 'R_EX_')


def boundary(r):
    # Boundary/sink reactions have just one mass-balanced model participant.
    return len([c for c in r.stoichiometry.values() if c]) < 2 or r.id.upper().startswith(BOUNDARY_PREFIXES)


def directional_network(model, query):
    """Metabolite -> (reaction,direction) -> metabolite; preserve alternatives."""
    graph=nx.DiGraph()
    exclude=set(query.excluded_metabolites)
    allow=set(filter(None,[query.substrate,query.product]))|set(query.required_metabolites)
    for rid,r in sorted(model.reactions.items()):
        if rid in query.excluded_reactions or (query.allowed_reactions and rid not in query.allowed_reactions) or boundary(r):continue
        if any(mid in exclude for mid,c in r.stoichiometry.items() if c):continue
        for direction,enabled in ((1,r.upper_bound>0),(-1,r.lower_bound<0)):
            if not enabled:continue
            reactants=[];products=[]
            for mid,c in r.stoichiometry.items():
                if mid not in model.metabolites or abs(c)<1e-12:continue
                met=model.metabolites[mid]
                if query.hide_currency and mid not in allow and is_currency(mid,met.name,met):continue
                if c*direction<0:reactants.append(mid)
                if c*direction>0:products.append(mid)
            if not reactants or not products:continue
            rn=('r',rid,direction)
            for mid in reactants:graph.add_edge(('m',mid),rn)
            for mid in products:graph.add_edge(rn,('m',mid))
    return graph


def append_checked_candidate(model,q,route,found,lp_audit):
    """Evaluate before accepting a candidate when LP-only reporting is requested.

    Does not turn a virtual-feed steady-state LP into biochemical verification.
    """
    if q.feasibility=='steady_state' and q.only_lp_feasible:
        ok,info=steady_state_support(model,q,route)
        lp_audit['tested']+=1
        if ok:
            route['status']='steady_state_supported'
            route['support_check']=info
        else:
            st=info.get('solver_status')
            key='infeasible' if st==2 or info.get('reason')=='incompatible_reaction_bounds' else 'undetermined'
            lp_audit[key]+=1
            if len(lp_audit['rejected_samples'])<30:
                lp_audit['rejected_samples'].append({'reaction_trace':route['reaction_trace'], 'reaction_count':route['reaction_count'],'reason':info['reason'],'solver_status':st})
            return False
    found.append(route)
    return True


def route_search(model,q):
    """Enumerate distinct simple metabolite-reaction traces, shortest first.

    If an endpoint is missing, search a bounded deterministic set of eligible
    model metabolites, not every biochemical pathway in the organism.
    """
    from .problems import RouteQuery
    if not isinstance(q,RouteQuery):q=RouteQuery.model_validate(q)
    g=directional_network(model,q)
    if q.substrate and q.product and q.required_metabolites and q.ordered_intermediates:
        return ordered_waypoint_search(model,q,g)
    start=monotonic()
    if q.substrate:
        origins=[('m',q.substrate)];reverse=False
    else:
        origins=[('m',q.product)];reverse=True
    H=g.reverse(copy=False) if reverse else g
    seen=set();found=[];inspected=0;frontier=deque()
    lp_audit={'tested':0,'infeasible':0,'undetermined':0,'rejected_samples':[]}
    for origin in origins:
        if origin in H:frontier.append((origin,(origin,),frozenset((origin,))))
    expansions=0;stopped=None
    while frontier:
        if monotonic()-start>q.timeout_seconds:stopped='time_limit';break
        if inspected>=q.max_candidates:stopped='candidate_limit';break
        if len(found)>=q.max_results:stopped='result_limit';break
        if len(frontier)>q.max_frontier:stopped='frontier_limit';break
        node,trace,used=frontier.popleft();expansions+=1
        nsteps=sum(x[0]=='r' for x in trace)
        if node[0]=='m' and len(trace)>1:
            forward=tuple(reversed(trace)) if reverse else trace
            mets=[x[1] for x in forward if x[0]=='m']
            rxns=[{'reaction_id':x[1],'direction':'forward' if x[2]==1 else 'reverse'} for x in forward if x[0]=='r']
            src=mets[0];dst=mets[-1]
            endpoint_ok=((q.substrate is None or src==q.substrate) and
                         (q.product is None or dst==q.product) and src!=dst and
                         nsteps>=q.min_reaction_steps)
            if endpoint_ok:
                inspected+=1
                rx_ids=[x['reaction_id'] for x in rxns]
                if set(q.required_metabolites).issubset(mets) and set(q.required_reactions).issubset(rx_ids):
                    if not q.ordered_intermediates or [mets.index(x) for x in q.required_metabolites]==sorted(mets.index(x) for x in q.required_metabolites):
                        signature=(src,dst,tuple((r['reaction_id'],r['direction']) for r in rxns))
                        if signature not in seen:
                            seen.add(signature)
                            route={'index':len(found)+1,'substrate_id':src,'product_id':dst,
                                          'reaction_trace':rxns,'metabolite_trace':mets,'reaction_count':len(rxns),
                                          'status':'topological_candidate'}
                            append_checked_candidate(model,q,route,found,lp_audit)
        if nsteps>=q.max_reaction_steps and node[0]=='m':continue
        for nxt in sorted(H.successors(node)):
            if nxt in used:continue
            # Enforce nonrepeated reactions in either direction.
            if nxt[0]=='r' and any(x[0]=='r' and x[1]==nxt[1] for x in trace):continue
            frontier.append((nxt,trace+(nxt,),used|{nxt}))
    count=len(found)
    if q.feasibility=='steady_state' and found and not q.only_lp_feasible:
        for route in found:
            ok,info=steady_state_support(model,q,route)
            route['status']='steady_state_supported' if ok else 'steady_state_infeasible' if info.get('reason') in ('infeasible','incompatible_reaction_bounds') else 'steady_state_undetermined'
            route['support_check']=info
            lp_audit['tested']+=1
            if not ok:lp_audit['infeasible' if info.get('solver_status')==2 or info.get('reason')=='incompatible_reaction_bounds' else 'undetermined']+=1
    warnings=[
        'Traces are directed metabolite/reaction connectivity, not proof of carbon-atom transfer, biochemical pathway identity, or flux in vivo.',
        'A reaction may consume other co-substrates; connectivity alone does not establish a self-sufficient conversion.',
        'A steady-state-supported trace may rely on other model reactions/exchanges; source atoms are NOT tracked.',
        'The result count is a bounded count of distinct linear traces, not all pathways, elementary flux modes, or pathways in an organism.',
    ]
    if q.substrate is None or q.product is None:warnings.append('Unspecified endpoint search is bounded by trace depth and result/candidate/time limits; it does not list every possible endpoint.')
    if stopped:warnings.append('Enumeration stopped at '+stopped+'; more results may exist.')
    return {'search_mode':'directed_reaction_traces','searched_from':'target_backward' if reverse else 'source_forward',
            'query':q.model_dump(mode='json'),'found_count':count,
            'steady_state_supported_count':sum(x['status']=='steady_state_supported' for x in found),
            'candidate_traces_examined':inspected,'lp_checks_performed':lp_audit['tested'],
            'lp_rejected_infeasible':lp_audit['infeasible'],'lp_rejected_undetermined':lp_audit['undetermined'],
            'lp_rejected_samples':lp_audit['rejected_samples'],
            'search_complete_within_bounds':stopped is None,
            'stop_reason':stopped or 'all_reachable_simple_traces_within_depth_explored',
            'graph_nodes':g.number_of_nodes(),'graph_edges':g.number_of_edges(),
            'elapsed_seconds':round(monotonic()-start,4),'routes':found,'warnings':warnings}


def steady_state_support(model,q,route):
    """Full-model LP with virtual source feed and target drain.

    This is a *necessary consistency check*, not evidence of atom tracking,
    actual physiological flux, or a self-contained pathway. Existing model
    exchange bounds remain available (reported as a limitation).
    """
    from scipy.optimize import linprog
    from scipy.sparse import coo_matrix,hstack
    import numpy as np
    mids=list(model.metabolites);rxns=list(model.reactions)
    midix={m:i for i,m in enumerate(mids)};cols={r:i for i,r in enumerate(rxns)}
    rows=[];colidx=[];values=[];bounds=[]
    exclusions=set(q.excluded_reactions)
    signed={x['reaction_id']:(1 if x['direction']=='forward' else -1) for x in route['reaction_trace']}
    for rid,j in cols.items():
        r=model.reactions[rid];lb,ub=float(r.lower_bound),float(r.upper_bound)
        if rid in exclusions:lb=ub=0
        if rid in signed:
            if signed[rid]>0:lb=max(lb,q.min_flux)
            else:ub=min(ub,-q.min_flux)
        if lb>ub:return False,{'reason':'incompatible_reaction_bounds'}
        bounds.append((lb,ub))
        for mid,c in r.stoichiometry.items():
            if mid in midix and abs(c)>1e-12:
                rows.append(midix[mid]);colidx.append(j);values.append(float(c))
    A=coo_matrix((values,(rows,colidx)),shape=(len(mids),len(rxns))).tocsr()
    feed=coo_matrix(([1.0],([midix[route['substrate_id']]],[0])),shape=(len(mids),1)).tocsr()
    sink=coo_matrix(([-1.0],([midix[route['product_id']]],[0])),shape=(len(mids),1)).tocsr()
    A=hstack([A,feed,sink],format='csr')
    bounds.extend([(q.min_flux,q.min_flux),(q.min_flux,1000)])
    try:
        sol=linprog(np.zeros(A.shape[1]),A_eq=A,b_eq=np.zeros(A.shape[0]),bounds=bounds,method='highs',options={'time_limit':q.lp_time_limit_seconds})
    except Exception as exc:return False,{'reason':'solver_error','detail':str(exc)[:200]}
    lp_reason=('supported' if sol.success else 'infeasible' if sol.status==2 else 'time_or_iteration_limit' if sol.status==1 else 'solver_status_'+str(sol.status))
    info={'reason':lp_reason,'solver_status':int(sol.status),
          'lp_time_limit_seconds':q.lp_time_limit_seconds,'min_required_flux':q.min_flux,
          'scope':'Full GEM with native exchange bounds + virtual source feed/product drain; no atom attribution'}
    if sol.success:
        info['source_feed_flux']=float(sol.x[-2]);info['target_drain_flux']=float(sol.x[-1]);
        info['active_support_reaction_count']=sum(abs(float(x))>1e-8 for x in sol.x[:-2])
    return bool(sol.success),info


def ordered_waypoint_search(model,q,g):
    """Bounded multi-segment route search for known biochemical landmarks.

    Keeps each directional reaction distinct; never fuses repeated reactions or
    metabolites. Search may be truncated, which is reported explicitly.
    """
    from itertools import product
    nodes=[q.substrate,*q.required_metabolites,q.product]
    start=monotonic();segments=[];incomplete=False;status=None
    for a,b in zip(nodes,nodes[1:]):
        src=('m',a);dst=('m',b)
        if src not in g or dst not in g:
            return {'search_mode':'ordered_waypoint_traces','query':q.model_dump(mode='json'),
               'found_count':0,'steady_state_supported_count':0,'candidate_traces_examined':0,
               'search_complete_within_bounds':True,'stop_reason':'required_segment_disconnected',
               'graph_nodes':g.number_of_nodes(),'graph_edges':g.number_of_edges(),
               'elapsed_seconds':round(monotonic()-start,4),'routes':[],
               'warnings':['A required intermediate segment is not connected in the selected directed projection; this is not evidence of biological absence.']}
        part=[]
        try:
            generator=nx.shortest_simple_paths(g,src,dst)
            for sequence in generator:
                if monotonic()-start>q.timeout_seconds:
                    incomplete=True;status='time_limit';break
                steps=sum(n[0]=='r' for n in sequence)
                if steps>q.max_reaction_steps:break
                part.append(tuple(sequence))
                if len(part)>=min(12,q.max_candidates):
                    incomplete=True;status='segment_alternative_limit';break
        except nx.NetworkXNoPath:pass
        if not part:
            return {'search_mode':'ordered_waypoint_traces','query':q.model_dump(mode='json'),
               'found_count':0,'steady_state_supported_count':0,'candidate_traces_examined':0,
               'search_complete_within_bounds':not incomplete,'stop_reason':status or 'required_segment_disconnected',
               'graph_nodes':g.number_of_nodes(),'graph_edges':g.number_of_edges(),
               'elapsed_seconds':round(monotonic()-start,4),'routes':[],
               'warnings':['No bounded route satisfies a required consecutive landmark segment.']}
        segments.append(part)
        if status=='time_limit':break
    found=[];seen=set();inspected=0
    lp_audit={'tested':0,'infeasible':0,'undetermined':0,'rejected_samples':[]}
    if len(segments)==len(nodes)-1:
        for combo in product(*segments):
            if monotonic()-start>q.timeout_seconds:incomplete=True;status='time_limit';break
            if inspected>=q.max_candidates:incomplete=True;status='candidate_limit';break
            if len(found)>=q.max_results:incomplete=True;status='result_limit';break
            trace=tuple(n for i,segment in enumerate(combo) for n in (segment if i==0 else segment[1:]))
            rnodes=[n for n in trace if n[0]=='r'];mids=[n[1] for n in trace if n[0]=='m']
            if len(rnodes)>q.max_reaction_steps or len(rnodes)<q.min_reaction_steps:continue
            if len(set(mids))!=len(mids) or len({r[1] for r in rnodes})!=len(rnodes):continue
            inspected+=1
            if not set(q.required_reactions).issubset({r[1] for r in rnodes}):continue
            sig=tuple((n[1],n[2]) for n in rnodes)
            if sig in seen:continue
            seen.add(sig)
            rxns=[{'reaction_id':n[1],'direction':'forward' if n[2]==1 else 'reverse'} for n in rnodes]
            route={'index':len(found)+1,'substrate_id':q.substrate,'product_id':q.product,
                          'reaction_trace':rxns,'metabolite_trace':mids,'reaction_count':len(rxns),
                          'status':'topological_candidate'}
            append_checked_candidate(model,q,route,found,lp_audit)
    for route in found:
        if q.feasibility=='steady_state' and not q.only_lp_feasible:
            ok,info=steady_state_support(model,q,route)
            route.update(status='steady_state_supported' if ok else 'steady_state_infeasible' if info.get('reason') in ('infeasible','incompatible_reaction_bounds') else 'steady_state_undetermined',support_check=info)
            lp_audit['tested']+=1
            if not ok:lp_audit['infeasible' if info.get('solver_status')==2 or info.get('reason')=='incompatible_reaction_bounds' else 'undetermined']+=1
    return {'search_mode':'ordered_waypoint_traces','query':q.model_dump(mode='json'),'found_count':len(found),
            'steady_state_supported_count':sum(x['status']=='steady_state_supported' for x in found),
            'candidate_traces_examined':inspected,'lp_checks_performed':lp_audit['tested'],
            'lp_rejected_infeasible':lp_audit['infeasible'],'lp_rejected_undetermined':lp_audit['undetermined'],
            'lp_rejected_samples':lp_audit['rejected_samples'],
            'search_complete_within_bounds':not incomplete,
            'stop_reason':status or 'all_segment_combinations_examined_within_limits',
            'graph_nodes':g.number_of_nodes(),'graph_edges':g.number_of_edges(),
            'elapsed_seconds':round(monotonic()-start,4),'routes':found,
            'warnings':['Routes pass through specified landmarks in sequence but atom provenance and independent biochemical pathway identity are not established.',
                        'Each co-substrate of every reaction must be separately available for an actual conversion.',
                        'Count refers only to bounded simple directed reaction traces, not all biological pathways.',
                        'Search truncated: more routes may exist.' if incomplete else 'Search complete only within provided landmark segment and depth bounds.']}
