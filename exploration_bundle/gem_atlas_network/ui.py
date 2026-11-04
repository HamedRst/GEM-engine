"""One-page, per-client NiceGUI interface. All science goes through engine.execute."""
import asyncio,copy,json,uuid
from pathlib import Path
from nicegui import ui,run,background_tasks,app
from .problems import Problem,REGISTRY,ROOT,parse_yaml,dump_yaml,sha256,resolve_model,export_bundle,import_bundle
from .model_io import load_model
from .pathways import discover,TEMPLATES
from .genes import gene_details
from .medium import encoded_inputs,external_exchange_reactions
from .engine import execute,validate_model_problem,render_selected_route
from .batch import BatchRunner
from .ui_helpers import new_run_root
from .result_archive import export_all_results

LABELS={'yeast_growth_product_cost':'Yeast','pputida_growth_product_cost':'Pseudomonas putida','cglutamicum_growth_product_cost':'Corynebacterium glutamicum','cho_growth_product_cost':'CHO','ecoli_antiepex_growth_product_cost':'E. coli antibody-expression model'}

def build_network_visualization_tab(outputs_dir,examples_dir=None,on_save_to_pool=None,shared_model_path_getter=None):
    ui.colors(primary='#157A6E',secondary='#667085',accent='#B98B3B')
    try:
        session_id=app.storage.user.setdefault('atlas_network_session',uuid.uuid4().hex)
    except (RuntimeError,AssertionError):session_id=uuid.uuid4().hex
    if not isinstance(session_id,str) or len(session_id)!=32 or any(c not in '0123456789abcdef' for c in session_id):session_id=uuid.uuid4().hex
    session=Path(outputs_dir)/('session_'+session_id);session.mkdir(parents=True,exist_ok=True)
    assets=session/'assets';assets.mkdir(exist_ok=True);runner=BatchRunner(session/'batch')
    state={'model':None,'path':None,'base':assets,'root':assets,'problem':None,'pathways':{},'queue':[],'result':None,'busy':False,'restoring':False,'progress':[],'revision':0,'exporting':False}
    controls={}
    queue_file=session/'queue.json'
    if queue_file.exists():
        try:
            jobs=json.loads(queue_file.read_text())
            for job in jobs:
                p=Problem.model_validate(job['problem'])
                base=(Path(outputs_dir)/job['base']).resolve();allowed=(Path(outputs_dir)/job['root']).resolve()
                if not base.is_relative_to(Path(outputs_dir).resolve()) or not allowed.is_relative_to(Path(outputs_dir).resolve()):raise ValueError('Invalid persisted queue path')
                supplied=(Path(outputs_dir)/job['supplied']).resolve() if job.get('supplied') else None
                if supplied and not supplied.is_relative_to(Path(outputs_dir).resolve()):raise ValueError('Invalid persisted model path')
                state['queue'].append((p,base,allowed,supplied))
        except Exception as ex:state['queue']=[];ui.notify('Saved queue needs attention: '+str(ex),type='warning')
    def persist_queue():
        root=Path(outputs_dir).resolve();jobs=[]
        for p,base,allowed,supplied in state['queue']:
            source=Path(supplied).resolve() if supplied else None
            jobs.append({'problem':p.model_dump(mode='json',exclude_none=True),'base':str(Path(base).resolve().relative_to(root)),'root':str(Path(allowed).resolve().relative_to(root)),'supplied':str(source.relative_to(root)) if source and source.is_relative_to(root) else None})
        temp=queue_file.with_suffix('.tmp');temp.write_text(json.dumps(jobs,indent=2));temp.replace(queue_file)
    def changed():
        if not state['restoring']:
            state['revision']+=1;state['result']=None;result_area.clear();status.set_text('Configuration changed. Click Run Now to generate current results.')
    def value(key):return controls[key].value
    def collect():
        if not state['model']:raise ValueError('Browse a GEM file or select a bundled example first')
        d=state['problem'].model_dump(mode='json',exclude_none=True) if state['problem'] else {}
        d.update(schema_version='1.0',problem_type='exploration.network_visualization',problem_id=value('id'))
        d['model']={'registry_id':value('model'),'sha256':sha256(state['path'])} if value('model') in REGISTRY else {'path':state['path'].name,'sha256':sha256(state['path'])}
        s={'kind':value('subject')}
        if value('subject')=='network' and value('target'):s.update(kind='target_neighborhood',target_id=value('target'),entity_kind=value('entity'))
        if value('subject')=='network' and value('subsystem'):s['subsystem']=value('subsystem')
        if value('subject')=='recognized_pathway':s.update(pathway_id=value('pathway'),branch_only=value('branch'))
        if value('subject')=='genes':s['gene_ids']=value('genes') or []
        if value('subject')=='route_search':
            s['route']={'substrate':value('route_start'),'product':value('route_end'),
              'required_metabolites':value('route_required') or [],
              'excluded_metabolites':value('route_forbidden') or [],
              'required_reactions':value('route_required_rxn') or [],
              'allowed_reactions':value('route_allowed_rxn') or [],
              'excluded_reactions':value('route_forbidden_rxn') or [],
              'ordered_intermediates':value('route_ordered'),
              'hide_currency':value('route_hide_currency'),
              'feasibility':value('route_feasibility'),
              'only_lp_feasible':bool(value('route_only_lp')),
              'min_reaction_steps':int(value('route_min_steps')),
              'max_reaction_steps':int(value('route_max_steps')),
              'max_results':int(value('route_max_results')),
              'max_candidates':int(value('route_max_candidates')),
              'timeout_seconds':float(value('route_timeout'))}
            # Do not silently equate a route LP with ordinary FBA visualization.
            d['fba']=None
        d['subject']=s
        d['analysis']={'mode':'structural' if value('subject')=='route_search' else value('mode'),'knockout_gene_ids':[] if value('subject')=='route_search' else (value('off') or []),'compare_to_wild_type':False if value('subject')=='route_search' else value('compare')}
        d['render']={**d.get('render',{}),'max_display_reactions':int(value('maxr')),'depth':int(value('depth')),'direction':value('direction'),'hide_currency_metabolites':value('currency'),'label_mode':value('labels'),'show_flux_values':value('fluxlabels'),'active_only':value('active'),'gene_context_depth':1 if value('gene_context') else 0,'gene_context_max_reactions':int(value('gene_context_max')),'include_legend':value('legend'),'export_formats':value('formats') or ['svg','pdf','png']}
        if value('mode')=='fba':
            selected={rid:float(controls['uptake_'+rid].value) for rid in (value('nutrients') or [])}
            d['fba']={'objective':{'source':'model'} if value('objective')=='__model__' else {'source':'reaction','reaction_id':value('objective')},'direction':value('objdirection'),'medium':{'source':value('medium'),'uptake_bounds':selected if value('medium')=='custom' else {},'close_unselected':value('close')},'flux_threshold':float(value('threshold')),'bounds_overrides':json.loads(value('bounds') or '{}')}
        else:d.pop('fba',None)
        p=Problem.model_validate(d);validate_model_problem(p,state['model']);return p
    async def use_model(path,registry=None,loaded=None,base=None,root=None):
        state['load_generation']=state.get('load_generation',0)+1
        generation=state['load_generation']
        state['restoring']=True;state['revision']+=1;state['model']=None;state['result']=None;result_area.clear()
        controls['model'].disable();visualize_button.disable();status.set_text('Loading GEM and inspecting its chemistry…')
        try:
            def prepare():
                m=load_model(path);return m,discover(m)
            m,pws=await run.io_bound(prepare)
            if generation!=state['load_generation']:return
            if loaded:validate_model_problem(loaded,m)
            state.update(model=m,path=Path(path),base=Path(base or assets),root=Path(root or assets),pathways=pws,result=None)
            p=loaded or Problem(problem_id='atlas_'+uuid.uuid4().hex[:10],model={'registry_id':registry} if registry else {'path':Path(path).name,'sha256':sha256(path)})
            state['problem']=p
            controls['model'].set_options({**LABELS,**({'uploaded':'Browsed GEM · '+Path(path).name} if not registry else {})});controls['model'].set_value(registry or 'uploaded')
            controls['id'].set_value(p.problem_id)
            controls['subject'].set_value('network' if p.subject.kind=='target_neighborhood' else p.subject.kind)
            controls['entity'].set_value(p.subject.entity_kind)
            refresh_targets(p.subject.entity_kind)
            controls['target'].set_value(p.subject.target_id)
            controls['subsystem'].set_options({s:s for s in sorted({r.subsystem for r in m.reactions.values() if r.subsystem})});controls['subsystem'].set_value(p.subject.subsystem)
            controls['genes'].set_options({g:g+' · '+v['name'] if v.get('name') and v['name']!=g else g for g,v in m.genes.items()});controls['genes'].set_value(p.subject.gene_ids)
            controls['off'].set_options(controls['genes'].options);controls['off'].set_value(p.analysis.knockout_gene_ids)
            controls['pathway'].set_options({k:v['name']+' · '+ ('all steps matched; provisional' if v['direction_compatible'] else v['status'].lower()) for k,v in pws.items()});controls['pathway'].set_value(p.subject.pathway_id or 'EMP_GLYCOLYSIS')
            controls['branch'].set_value(p.subject.branch_only)
            route=p.subject.route
            for field,attr in [('route_start','substrate'),('route_end','product'),('route_required','required_metabolites'),('route_forbidden','excluded_metabolites')]:
                controls[field].set_options({k:v.name+' · '+k for k,v in m.metabolites.items()})
                controls[field].set_value(getattr(route,attr) if route else ([] if field in ('route_required','route_forbidden') else None))
            for field,attr in [('route_required_rxn','required_reactions'),('route_forbidden_rxn','excluded_reactions'),('route_allowed_rxn','allowed_reactions')]:
                controls[field].set_options({k:v.name+' · '+k for k,v in m.reactions.items()})
                controls[field].set_value(getattr(route,attr) if route else [])
            for field,attr,default in [('route_ordered','ordered_intermediates',True),('route_hide_currency','hide_currency',True),('route_feasibility','feasibility','topology'),('route_only_lp','only_lp_feasible',False),('route_min_steps','min_reaction_steps',1),('route_max_steps','max_reaction_steps',8),('route_max_results','max_results',10),('route_max_candidates','max_candidates',600),('route_timeout','timeout_seconds',12.0)]:
                controls[field].set_value(getattr(route,attr) if route else default)
            controls['mode'].set_value(p.analysis.mode);controls['compare'].set_value(p.analysis.compare_to_wild_type)
            controls['gene_context'].set_value(p.render.gene_context_depth>0)
            controls['gene_context_max'].set_value(p.render.gene_context_max_reactions)
            obj={'__model__':'Encoded model objective'} if m.objectives else {}
            obj.update({r:r+' · '+v.name for r,v in m.reactions.items()})
            controls['objective'].set_options(obj);controls['objective'].set_value(p.fba.objective.reaction_id if p.fba and p.fba.objective.source=='reaction' else '__model__' if m.objectives else None)
            controls['objdirection'].set_value(p.fba.direction if p.fba else m.objective_direction)
            controls['medium'].set_value(p.fba.medium.source if p.fba else 'model_encoded')
            controls['close'].set_value(p.fba.medium.close_unselected if p.fba else True)
            ex=external_exchange_reactions(m)
            controls['nutrients'].set_options({x.reaction_id:x.metabolite_name+' · '+x.reaction_id for x in ex})
            initial=p.fba.medium.uptake_bounds if p.fba else {x.reaction_id:x.uptake_capacity for x in encoded_inputs(m)}
            state['initial_uptake']=initial;controls['nutrients'].set_value(list(initial));refresh_uptakes()
            controls['threshold'].set_value(p.fba.flux_threshold if p.fba else 1e-8)
            controls['bounds'].set_value(json.dumps(p.fba.bounds_overrides if p.fba else {}))
            for k,v in [('maxr',p.render.max_display_reactions),('depth',p.render.depth),('direction',p.render.direction),('currency',p.render.hide_currency_metabolites),('labels',p.render.label_mode),('fluxlabels',p.render.show_flux_values),('active',p.render.active_only),('legend',p.render.include_legend),('formats',p.render.export_formats)]:controls[k].set_value(v)
            summary.set_text(f"{('iCGB21FR' if m.id=='iCBG21FR' else m.name)} · {len(m.reactions):,} reactions · {len(m.metabolites):,} metabolites · {len(m.genes):,} model-associated genes")
            encoded.set_text(f'Model-encoded medium: {len(encoded_inputs(m))} open extracellular uptake exchanges. Original bounds are used.')
            update_visibility();details.refresh();gene_switches.refresh();result_area.clear();status.set_text('Ready. Structural mode uses zero FBA calls.')
        finally:
            if generation==state['load_generation']:
                state['restoring']=False;controls['model'].enable();
                if not state['busy']:visualize_button.enable()
                details.refresh();gene_switches.refresh()
    def choose(e):
        # Check at synchronous dispatch time: programmatic restoration must never
        # enqueue a later asynchronous model change that discards the loaded YAML.
        if state['restoring'] or e.value not in REGISTRY:return
        selected=e.value
        async def load_selected():
            try:await use_model(REGISTRY[selected],selected)
            except Exception as ex:ui.notify(str(ex),type='negative')
        background_tasks.create(load_selected())
    def refresh_targets(kind=None):
        m=state['model']
        if not m:return
        kind=kind or value('entity')
        things=m.metabolites if kind=='metabolite' else m.reactions
        controls['target'].set_options({k:v.name+' · '+k for k,v in things.items()})
    def refresh_uptakes():
        if not state['restoring']:
            state.setdefault('initial_uptake',{}).update({k[7:]:c.value for k,c in controls.items() if k.startswith('uptake_')})
        for k in list(controls):
            if k.startswith('uptake_'):controls.pop(k)
        uptake_area.clear()
        with uptake_area:
            for rid in value('nutrients') or []:
                controls['uptake_'+rid]=ui.number(rid+' · maximum uptake',value=state.get('initial_uptake',{}).get(rid,10.0),min=0,on_change=lambda e:changed()).classes('w-full')
    def update_visibility():
        network_area.set_visibility(value('subject')=='network');pathway_area.set_visibility(value('subject')=='recognized_pathway');genes_area.set_visibility(value('subject')=='genes');route_area.set_visibility(value('subject')=='route_search');fba_area.set_visibility(value('mode')=='fba' and value('subject')!='route_search');controls['mode'].set_visibility(value('subject')!='route_search');gene_comparison_area.set_visibility(value('subject')!='route_search');medium_area.set_visibility(value('medium')=='custom')
        controls['branch'].set_visibility(value('pathway') in {'ETHANOL_FERMENTATION_EMP_PDC_ADH','HOMOLACTIC_L_FERMENTATION_EMP_LDH'})
    def mode_changed():
        if value('subject')=='route_search' and value('mode')!='structural':
            controls['mode'].set_value('structural')
        changed();update_visibility();details.refresh()
    def gene_change():
        changed();details.refresh();gene_switches.refresh()
    @ui.refreshable
    def gene_switches():
        if state['restoring']:return
        for gid in value('genes') or []:
            def toggle_gene(e,g=gid):
                off=set(value('off') or [])
                if e.value:off.discard(g)
                else:off.add(g)
                controls['off'].set_value(sorted(off));changed()
            ui.switch(gid+' · '+('OFF' if gid in (value('off') or []) else 'ON'),value=gid not in (value('off') or []),on_change=toggle_gene)
        if value('genes'):ui.label('Switch OFF to simulate knockout. Compare with wild type below.').classes('text-sm')
    @ui.refreshable
    def details():
        if not state['model'] or state['restoring']:return
        with ui.expansion('Scientific evidence and gene details',icon='science').classes('w-full'):
            ui.label('Genes listed here are the metabolic genes represented in this GEM, not a full genomic inventory. ON restores normal modeled availability; it does not simulate overexpression.').classes('text-sm')
            if value('subject')=='recognized_pathway' and value('pathway'):
                p=state['pathways'][value('pathway')];ui.label(p['scope']);ui.label(p['note']);ui.label('Missing/unresolved: '+(', '.join(p['missing_steps']) or 'none'));ui.label('Direction blocked: '+(', '.join(p['bounds_blocked_steps']) or 'none'))
                ui.table(columns=[{'name':k,'label':v,'field':k} for k,v in [('step','Step'),('reactions','Mapped reactions'),('gpr','GPR')]],rows=[{'step':e['step_id'],'reactions':', '.join(x['reaction_id'] for x in e['matches']),'gpr':e['gene_status']} for e in p['steps']]).classes('w-full')
                for url in p['reference_urls']:ui.link(url,url,new_tab=True).classes('text-xs')
            for row in gene_details(state['model'],value('genes') or []):
                ui.label(row['id']+' · '+row['name']);ui.code(json.dumps(row,indent=2),language='json').classes('w-full')
    def cached_model_for(p):
        """Resolve uploaded models by exact checksum, never silently substitute by name."""
        try:return resolve_model(p,assets,assets)
        except (FileNotFoundError,ValueError):
            if p.model.sha256:
                for candidate in REGISTRY.values():
                    if sha256(candidate)==p.model.sha256:return candidate
                for candidate in (assets/'models').glob('*/*'):
                    if candidate.is_file() and candidate.suffix.lower() in {'.xml','.sbml','.json'} and sha256(candidate)==p.model.sha256:
                        return candidate
            raise FileNotFoundError('No model matched the YAML file checksum. Browse the exact referenced GEM to restore this problem.')
    async def new_problem():
        if state['busy']:
            ui.notify('Wait until the current run finishes.',type='warning');return
        # Allow the user to abandon a YAML awaiting a missing GEM.
        state.pop('pending',None)
        if not state['path']:
            ui.notify('Browse a GEM first.',type='warning');return
        registry=value('model') if value('model') in REGISTRY else None
        try:await use_model(state['path'],registry=registry,base=state['base'],root=state['root'])
        except Exception as ex:ui.notify(str(ex),type='negative')
    async def upload(e):
        try:
            data=e.content.read()
            if len(data)>150*1024*1024:raise ValueError('Upload exceeds 150 MiB')
            name=Path(e.name).name
            if name!=e.name:raise ValueError('Invalid upload filename')
            path=assets/name
            if path.suffix.lower() in {'.xml','.sbml','.json'}:
                # Keep successive browsed GEMs separate, even with identical filenames.
                model_dir=assets/'models'/uuid.uuid4().hex;model_dir.mkdir(parents=True)
                path=model_dir/name;path.write_bytes(data)
                pending=state.pop('pending',None)
                if pending:
                    p,b,r=pending
                    if p.model.sha256 and sha256(path)!=p.model.sha256:state['pending']=pending;raise ValueError('Model checksum mismatch; upload the referenced model')
                    await use_model(path,loaded=p,base=model_dir,root=assets)
                else:await use_model(path,base=model_dir,root=assets)
            elif path.suffix.lower() in {'.yaml','.yml'}:
                path.write_bytes(data);p=parse_yaml(data)
                try:mp=cached_model_for(p)
                except FileNotFoundError:
                    state['pending']=(p,assets,assets);status.set_text('YAML loaded. Browse the referenced GEM to finish restoring this problem.');return
                reg=next((k for k,v in REGISTRY.items() if v==mp),None)
                await use_model(mp,reg,p,mp.parent if reg is None else assets,assets)
            elif path.suffix.lower()=='.zip':
                path.write_bytes(data);bundle=assets/('bundle_'+uuid.uuid4().hex)
                entries=import_bundle(path,bundle)
                for yp,p in entries:
                    mp=resolve_model(p,yp.parent,bundle)
                    state['queue'].append((p,yp.parent,bundle,None))
                yp,p=entries[0];mp=resolve_model(p,yp.parent,bundle)
                await use_model(mp,loaded=p,base=yp.parent,root=bundle)
                persist_queue();queue_view.refresh()
            else:raise ValueError('Upload SBML, COBRA JSON, YAML, or a portable ZIP')
            ui.notify('Loaded. Click Run Now when ready.')
        except Exception as ex:ui.notify(str(ex),type='negative');status.set_text(str(ex))
    def save():
        try:
            p=collect();ui.download(dump_yaml(p).encode(),filename=p.problem_id+'.yaml');status.set_text('Problem saved; no FBA was run.')
        except Exception as ex:ui.notify(str(ex),type='negative')
    def bundle():
        try:
            p=collect();path=export_bundle(p,state['path'],assets/(p.problem_id+'.zip'));ui.download(path)
        except Exception as ex:ui.notify(str(ex),type='negative')
    async def visualize():
        if state['busy']:return
        try:
            p=collect();revision=state['revision'];state['busy']=True;visualize_button.disable()
            status.set_text('Rendering structural figure…' if p.analysis.mode=='structural' else 'Running standard FBA and rendering…')
            # Each Run Now has immutable outputs. Reusing a problem ID cannot serve stale images
            # from the browser cache and previous scientific results remain reproducible.
            run_root=new_run_root(session,p.problem_id)
            result=await run.io_bound(execute,p,state['base'],run_root,state['root'],state['path'])
            if revision!=state['revision']:
                status.set_text('Previous run finished; settings have changed. Click Run Now for current configuration.')
                return
            state['result']=result;result_area.clear()
            result_folder=run_root/result['output_directory']
            async def export_everything():
                if state['exporting']:
                    ui.notify('An export is already running.',type='warning');return
                state['exporting']=True
                export_button.disable()
                status.set_text('Exporting every discovered route and bundling this problem’s results…')
                try:
                    archive=await run.io_bound(export_all_results,p,state['base'],run_root,result,state['root'],state['path'])
                    ui.download(archive)
                    status.set_text('All results archived for '+p.problem_id+' · '+str(archive))
                except Exception as ex:
                    status.set_text('Export failed: '+str(ex))
                    ui.notify('Export failed; no incomplete ZIP was downloaded: '+str(ex),type='negative')
                finally:
                    state['exporting']=False
                    export_button.enable()
            with result_area:
                ui.label('Saved results folder: '+str(result_folder)).classes('text-xs text-gray-600 break-all')
                with ui.row().classes('items-center w-full flex-wrap'):
                    export_button=ui.button('Export All Results (ZIP)',icon='archive',on_click=export_everything)
                    ui.label('Routes: all SVG/PNG/PDF images, reaction tables, YAML, JSON and evidence.').classes('text-xs text-gray-600')

                if result.get('route_search') is not None:
                    rr=result['route_search'];q=p.subject.route
                    ui.label('Metabolic Route Explorer · '+p.problem_id).classes('text-xl font-semibold')
                    ui.label(f"{rr['found_count']} distinct traces found · {rr['candidate_traces_examined']} candidates inspected · "+
                      ('Search complete within declared depth' if rr['search_complete_within_bounds'] else 'Search truncated: '+str(rr['stop_reason'])))
                    if q.feasibility=='steady_state':
                        if q.only_lp_feasible:
                            ui.label(f"LP-feasible routes: {rr['steady_state_supported_count']} · candidates evaluated: {rr.get('lp_checks_performed', 0)} · rejected: {rr.get('lp_rejected_infeasible', 0)} infeasible, {rr.get('lp_rejected_undetermined', 0)} undetermined.")
                        else:
                            ui.label(f"{rr['steady_state_supported_count']} of {rr['found_count']} candidates pass the optional steady-state support check.")
                    ui.label('These are bounded directed traces, not verified biological pathways or atom-transfer routes.').classes('text-sm text-orange-700')
                    if rr['routes']:
                        options={r['index']:f"Route {r['index']} · {r['substrate_id']} → {r['product_id']} · {r['reaction_count']} reactions · {r['status']}" for r in rr['routes']}
                        selected=ui.select(options,value=1,label='Select a route to visualize').classes('w-full')
                        route_preview=ui.column().classes('w-full')
                        def show_figure(files):
                            route_preview.clear()
                            with route_preview:
                                img='svg' if 'svg' in files else 'png' if 'png' in files else None
                                if img:
                                    file=run_root/files[img];url='/atlas-output/'+str(file.relative_to(Path(outputs_dir)))+'?v='+str(uuid.uuid4().hex)
                                    ui.image(url).props('fit=contain').classes('w-full').style('height:650px')
                                    ui.link('Open full route figure',url,new_tab=True)
                                for ext,name in files.items():
                                    if ext!='dot':ui.button('Export '+ext.upper(),on_click=lambda q=run_root/name:ui.download(q)).props('flat dense')
                        show_figure(result['figures'].get('route_1',{}))
                        async def select_route():
                            if state['exporting']:
                                ui.notify('Wait for Export All Results to finish.',type='warning');return
                            idx=int(selected.value)
                            if state['revision']!=revision:return
                            files=result['figures'].get('route_'+str(idx))
                            if files is None:
                                try:
                                    files=await run.io_bound(render_selected_route,p,state['base'],run_root,result,idx,state['root'],state['path'])
                                    result['figures']['route_'+str(idx)]=files
                                    (run_root/result['output_directory']/'result.json').write_text(json.dumps(result,indent=2))
                                except Exception as ex:
                                    ui.notify('Unable to render route: '+str(ex),type='negative');return
                            show_figure(files)
                        ui.button('Visualize selected route',on_click=select_route)
                        ui.table(columns=[{'name':k,'label':k.replace('_',' ').title(),'field':k} for k in ['index','substrate_id','product_id','reaction_count','status']],
                            rows=[{k:r[k] for k in ['index','substrate_id','product_id','reaction_count','status']} for r in rr['routes']]).classes('w-full')
                    for warning in rr['warnings']:ui.label('• '+warning).classes('text-xs text-gray-600')
                    out=run_root/result['output_directory']
                    ui.button('Export scientific evidence',on_click=lambda q=out/'evidence.md':ui.download(q)).props('flat')
                    ui.button('Export all route data (JSON)',on_click=lambda q=out/'result.json':ui.download(q)).props('flat')
                    status.set_text(f"Route search complete: {rr['found_count']} bounded candidates.")
                    return
                ui.label('Result · '+p.problem_id).classes('text-xl font-semibold')
                ui.label('New run · '+result['configuration_sha256'][:12]).classes('text-xs text-gray-500')
                ui.label(f"{result['analysis_mode'].title()} · {result['solver_calls']} FBA calls · {len(result['disabled_reactions'])} GPR-disabled reactions")
                if result['pathway']:ui.label(result['pathway']['representation']+' · provisional definition; independent review pending')
                if result['display_truncated']:ui.label('Figure limited for readability. Recognition and FBA used the full model.').classes('text-sm')
                with ui.row().classes('w-full flex-wrap'):
                    for scenario,files in result['figures'].items():
                        with ui.card().classes('flex-1 min-w-80'):
                            ui.label(scenario.replace('_',' ').title()).classes('font-semibold')
                            img='svg' if 'svg' in files else 'png' if 'png' in files else None
                            if img:
                                file=run_root/files[img];url='/atlas-output/'+str(file.relative_to(Path(outputs_dir)))
                                ui.image(url).props('fit=contain').classes('w-full').style('height:600px')
                                ui.link('Open full figure',url,new_tab=True).classes('text-sm')
                            for ext,path in files.items():
                                if ext=='dot':continue
                                ui.button('Export '+ext.upper(),on_click=lambda q=run_root/path:ui.download(q)).props('flat dense')
                for scenario,sol in result['solutions'].items():ui.label(f"{scenario}: {sol['status']} · objective {sol['objective_value']}")
                if result['solver_calls']:ui.label('FBA flux distributions may be non-unique. These are modeled solutions, not measured activity.').classes('text-sm')
                ui.button('Export scientific report',on_click=lambda q=run_root/result['output_directory']/'evidence.md':ui.download(q)).props('flat')
                ui.button('Export result data',on_click=lambda q=run_root/result['output_directory']/'result.json':ui.download(q)).props('flat')
            status.set_text('Visualization complete. '+p.problem_id+'.' if all(s['status']=='optimal' for s in result['solutions'].values()) else 'FBA has a non-optimal scenario. See solver status; available valid figures are shown.')
        except Exception as ex:ui.notify(str(ex),type='negative');status.set_text(str(ex))
        finally:
            state['busy']=False
            if not state['restoring']:visualize_button.enable()
    def add_queue():
        try:
            p=collect()
            if len(state['queue'])>=200:raise ValueError('Queue is limited to 200 problems')
            if any(j[0].problem_id==p.problem_id for j in state['queue']):raise ValueError('This problem ID is already queued; enter a new ID for another problem')
            state['queue'].append((p,state['base'],state['root'],state['path']));persist_queue();queue_view.refresh();ui.notify('Added to batch; no analysis run.')
        except Exception as ex:ui.notify(str(ex),type='negative')
    async def queue_upload(e):
        try:
            p=parse_yaml(e.content.read());state['queue'].append((p,assets,assets,None));persist_queue();queue_view.refresh()
        except Exception as ex:ui.notify(str(ex),type='negative')
    @ui.refreshable
    def queue_view():
        rows=runner.validate(state['queue']) if state['queue'] else []
        for row in rows:row['status']=runner.records.get(row['problem_id'],{}).get('status','queued')
        ui.table(columns=[{'name':k,'label':k.replace('_',' ').title(),'field':k} for k in ['problem_id','mode','subject','ready','status','error']],rows=rows).classes('w-full')
        for row in rows:
            record=runner.records.get(row['problem_id'],{})
            if record.get('output_directory'):
                folder=runner.root/record['output_directory'];rp=folder/'result.json'
                batch_run_root=runner.root/record.get('run_root','.')
                if rp.is_file():
                    result=json.loads(rp.read_text())
                    with ui.expansion('Result · '+row['problem_id']).classes('w-full'):
                        ui.label('Saved results folder: '+str(folder)).classes('text-xs break-all')
                        async def export_batch_results(problem_id=row['problem_id'],root=batch_run_root,data=result):
                            job=next((j for j in state['queue'] if j[0].problem_id==problem_id),None)
                            if job is None:
                                ui.notify('Original batch job unavailable; reload its YAML to export all route figures.',type='negative');return
                            try:
                                path=await run.io_bound(export_all_results,job[0],job[1],root,data,job[2],job[3])
                                ui.download(path)
                            except Exception as exc:ui.notify('Export All Results failed: '+str(exc),type='negative')
                        ui.button('Export All Results (ZIP)',icon='archive',on_click=export_batch_results).props('outline')
                        ui.label(f"{result['analysis_mode']} · {result['solver_calls']} FBA calls")
                        for name,files in result['figures'].items():
                            ui.label(name.replace('_',' ').title())
                            if 'svg' in files:
                                file=batch_run_root/files['svg'];url='/atlas-output/'+str(file.relative_to(Path(outputs_dir)))+'?v='+str(file.stat().st_mtime_ns)
                                ui.image(url).props('fit=contain').classes('w-full').style('height:450px')
                            for ext,path in files.items():
                                if ext=='dot':continue
                                ui.button('Export '+ext.upper(),on_click=lambda q=batch_run_root/path:ui.download(q)).props('flat dense')
                        ui.button('Export result data',on_click=lambda q=rp:ui.download(q)).props('flat')
                        ui.button('Export evidence report',on_click=lambda q=folder/'evidence.md':ui.download(q)).props('flat')

    async def run_all():
        if runner.running:return
        try:
            state['progress']=[];batch_status.set_text('Running batch…')
            jobs=copy.deepcopy(state['queue'])
            await run.io_bound(runner.run,jobs,True,lambda i,n,row:state['progress'].append((i,n,dict(row))))
            batch_status.set_text('Paused after current job.' if runner.pause_requested else 'Batch finished.');queue_view.refresh()
        except Exception as ex:ui.notify(str(ex),type='negative')
    def poll_progress():
        if runner.running and state['progress']:
            i,n,row=state['progress'][-1];batch_status.set_text(f"{i}/{n} · {row['problem_id']} · {row['status']}")
    async def clear_queue():
        if runner.running:ui.notify('Pause the batch first.');return
        with ui.dialog() as dialog,ui.card():
            ui.label('Clear the queue? Saved results remain available.')
            ui.button('Keep queue',on_click=lambda:dialog.submit(False));ui.button('Clear queue',on_click=lambda:dialog.submit(True))
        if await dialog:state['queue']=[];persist_queue();queue_view.refresh()
    with ui.card().classes('w-full'):
        ui.label('1 · Browse a GEM').classes('text-lg font-semibold')
        ui.label('Browse any compatible SBML (.xml/.sbml) or COBRA JSON (.json) file. The five included models are only optional examples.').classes('text-sm text-gray-600')
        ui.upload(label='Browse GEM file',on_upload=upload,auto_upload=True).props('accept=.xml,.sbml,.json').classes('w-full')
        if shared_model_path_getter is not None:
            async def use_shared_exploration_model():
                supplied=shared_model_path_getter()
                if not supplied:
                    ui.notify('Load a compatible GEM in the shared Exploration source panel first.', type='warning')
                    return
                candidate=Path(supplied).expanduser().resolve()
                if not candidate.is_file() or candidate.suffix.lower() not in {'.json','.sbml','.xml'}:
                    ui.notify('The shared GEM path is not an existing compatible model file.', type='negative')
                    return
                try:
                    await use_model(candidate,base=candidate.parent,root=assets)
                    ui.notify('Loaded the shared Exploration GEM in Network Atlas.',type='positive')
                except Exception as exc:
                    ui.notify('Cannot load shared GEM: '+str(exc),type='negative',multi_line=True)
            ui.button('Use the same GEM as Landscape Activity',icon='sync',on_click=use_shared_exploration_model).props('outline color=primary')

        with ui.row().classes('w-full items-center flex-wrap'):
            controls['model']=ui.select(LABELS,label='Or select an included example GEM',on_change=choose).classes('flex-1 min-w-64').props('use-input')
            ui.upload(label='Load saved problem (YAML or bundle ZIP)',on_upload=upload,auto_upload=True).props('accept=.yaml,.yml,.zip').classes('flex-1 min-w-64')
        summary=ui.label('No GEM loaded yet. Browse a model to begin.').classes('text-sm text-gray-600')
    with ui.card().classes('w-full'):
        ui.label('2 · Choose what to visualize').classes('text-lg font-semibold')
        controls['subject']=ui.toggle({'route_search':'Find Routes','network':'Network','genes':'Genes','recognized_pathway':'Legacy Templates'},value='network',on_change=lambda e:mode_changed())
        with ui.column().classes('w-full') as network_area:
            with ui.row().classes('w-full'):
                controls['entity']=ui.select({'metabolite':'Metabolite','reaction':'Reaction'},value='metabolite',label='Target type',on_change=lambda e:(refresh_targets(),changed())).classes('w-40')
                controls['target']=ui.select({},label='Optional target · search name or ID',clearable=True,on_change=lambda e:changed()).props('use-input').classes('flex-1')
        with ui.column().classes('w-full') as route_area:
            ui.label('Choose a start and/or product. Leaving one blank discovers alternatives within a bounded search.').classes('text-sm')
            with ui.row().classes('w-full flex-wrap'):
                controls['route_start']=ui.select({},label='Start metabolite (optional)',clearable=True,on_change=lambda e:changed()).props('use-input').classes('flex-1 min-w-64')
                controls['route_end']=ui.select({},label='Product metabolite (optional)',clearable=True,on_change=lambda e:changed()).props('use-input').classes('flex-1 min-w-64')
            controls['route_required']=ui.select({},multiple=True,label='Must pass through (optional intermediates)',on_change=lambda e:changed()).props('use-input use-chips').classes('w-full')
            controls['route_feasibility']=ui.select({'topology':'Directed connectivity (fast; no solver)','steady_state':'Check steady-state support (LP; experimental)'},value='topology',label='Evidence level',on_change=lambda e:changed()).classes('w-full')
            with ui.expansion('Advanced route constraints and search limits').classes('w-full'):
                controls['route_forbidden']=ui.select({},multiple=True,label='Excluded metabolites',on_change=lambda e:changed()).props('use-input use-chips').classes('w-full')
                controls['route_required_rxn']=ui.select({},multiple=True,label='Required reactions',on_change=lambda e:changed()).props('use-input use-chips').classes('w-full')
                controls['route_allowed_rxn']=ui.select({},multiple=True,label='Only allow these reactions (optional curated route set)',on_change=lambda e:changed()).props('use-input use-chips').classes('w-full')
                controls['route_forbidden_rxn']=ui.select({},multiple=True,label='Excluded reactions',on_change=lambda e:changed()).props('use-input use-chips').classes('w-full')
                controls['route_ordered']=ui.checkbox('Intermediates must appear in listed order',value=True,on_change=lambda e:changed())
                controls['route_hide_currency']=ui.checkbox('Exclude currency-metabolite shortcuts (recommended)',value=True,on_change=lambda e:changed())
                controls['route_only_lp']=ui.checkbox('Report only LP-feasible routes',value=False,on_change=lambda e:(controls['route_feasibility'].set_value('steady_state') if e.value else None,changed()))
                controls['route_min_steps']=ui.number('Minimum reactions in a trace',value=1,min=1,max=15,on_change=lambda e:changed())
                controls['route_max_steps']=ui.number('Maximum reactions in a trace',value=8,min=1,max=15,on_change=lambda e:changed())
                controls['route_max_results']=ui.number('Maximum results',value=10,min=1,max=30,on_change=lambda e:changed())
                controls['route_max_candidates']=ui.number('Maximum candidate traces inspected',value=600,min=10,max=5000,on_change=lambda e:changed())
                controls['route_timeout']=ui.number('Search time limit (seconds)',value=12,min=1,max=90,on_change=lambda e:changed())
            ui.label('Route results are candidate reaction traces, not claimed canonical pathways. Use model IDs, not generic biological names.').classes('text-xs text-gray-600')
        with ui.column().classes('w-full') as pathway_area:
            controls['pathway']=ui.select({},label='Recognized pathway',on_change=lambda e:mode_changed()).props('use-input').classes('w-full')
            ui.label('Small canonical catalog. Source-checked definitions remain provisional until independent expert review.').classes('text-sm text-gray-600')
            controls['branch']=ui.checkbox('Show terminal fermentation branch only',value=False,on_change=lambda e:changed())
        with ui.column().classes('w-full') as genes_area:
            controls['genes']=ui.select({},multiple=True,label='Model-associated genes',on_change=lambda e:gene_change()).props('use-input use-chips').classes('w-full')
            ui.label('Choose genes to see their associated reactions.').classes('text-sm')
            controls['gene_context']=ui.checkbox('Also export expanded knockout neighborhood',value=True,on_change=lambda e:changed())
            controls['gene_context_max']=ui.number('Expanded view: max reactions',value=16,min=2,max=60,on_change=lambda e:changed())
            gene_switches()
        with ui.expansion('Gene ON / OFF comparison',icon='toggle_off').classes('w-full') as gene_comparison_area:
            controls['off']=ui.select({},multiple=True,label='Genes OFF · all others stay ON',on_change=lambda e:(changed(),gene_switches.refresh())).props('use-input use-chips').classes('w-full')
            controls['compare']=ui.checkbox('Compare with wild type',value=True,on_change=lambda e:changed())
            ui.button('Restore wild type',on_click=lambda:(controls['off'].set_value([]),changed(),gene_switches.refresh())).props('flat')
            ui.label('OFF uses AND/OR GPR logic. ON restores normal model availability.').classes('text-sm')
        controls['mode']=ui.toggle({'structural':'Structural — No FBA','fba':'FBA — Standard FBA'},value='structural',on_change=lambda e:mode_changed())
        with ui.column().classes('w-full') as fba_area:
            controls['objective']=ui.select({},label='Objective',on_change=lambda e:changed()).props('use-input').classes('w-full')
            controls['objdirection']=ui.select({'max':'Maximize','min':'Minimize'},value='max',label='Objective direction',on_change=lambda e:changed())
            controls['medium']=ui.select({'model_encoded':'Model-encoded medium','custom':'Custom medium'},value='model_encoded',label='Medium',on_change=lambda e:mode_changed())
            encoded=ui.label().classes('text-sm')
            with ui.column().classes('w-full') as medium_area:
                controls['nutrients']=ui.select({},multiple=True,label='Nutrients',on_change=lambda e:(refresh_uptakes(),changed())).props('use-input use-chips').classes('w-full')
                controls['close']=ui.checkbox('Close uptake of unselected nutrients',value=True,on_change=lambda e:changed())
                uptake_area=ui.column().classes('w-full')
            ui.label('Only Visualize or Run All executes FBA. Flux distributions can be non-unique.').classes('text-sm')
        with ui.expansion('Advanced figure and FBA settings',icon='tune').classes('w-full'):
            with ui.row():
                controls['depth']=ui.number('Depth',value=3,min=1,max=8,on_change=lambda e:changed())
                controls['direction']=ui.select(['upstream','downstream','both'],value='upstream',label='Direction',on_change=lambda e:changed())
                controls['maxr']=ui.number('Display reaction limit',value=28,min=1,max=200,on_change=lambda e:changed())
                controls['labels']=ui.select(['names','ids'],value='names',label='Labels',on_change=lambda e:changed())
            controls['subsystem']=ui.select({},label='Optional subsystem',clearable=True,on_change=lambda e:changed()).props('use-input').classes('w-full')
            controls['currency']=ui.checkbox('Hide currency metabolites in figure',value=True,on_change=lambda e:changed())
            controls['legend']=ui.checkbox('Include legend',value=True,on_change=lambda e:changed())
            controls['fluxlabels']=ui.checkbox('Show flux values on edges',value=False,on_change=lambda e:changed())
            controls['active']=ui.checkbox('FBA: show only reactions with flux',value=False,on_change=lambda e:changed())
            controls['threshold']=ui.number('FBA flux threshold',value=1e-8,min=1e-12,on_change=lambda e:changed())
            controls['bounds']=ui.textarea('Optional FBA bound overrides (JSON: reaction ID → [lower, upper])',value='{}',on_change=lambda e:changed()).classes('w-full')
            controls['formats']=ui.select(['svg','pdf','png'],multiple=True,value=['svg','pdf','png'],label='Export formats',on_change=lambda e:changed())
        ui.separator()
        ui.label('3 · Name, save, or run this problem').classes('text-lg font-semibold')
        controls['id']=ui.input('Problem name / ID',value='atlas_'+uuid.uuid4().hex[:10],on_change=lambda e:changed()).classes('w-full')
        ui.label('Save this question to Saved Questions. Execution starts only when you run a saved question.' if on_save_to_pool else 'Save YAML without running. Run Now shows the current settings, with a fresh result every time.').classes('text-sm text-gray-600')
        with ui.row().classes('items-center w-full flex-wrap'):
            ui.button('New Problem',icon='add',on_click=new_problem).props('outline')
            native_save_button=ui.button('Save YAML',icon='save',on_click=save).props('outline')
            if on_save_to_pool is not None:native_save_button.set_visibility(False)
            if on_save_to_pool is not None:
                def save_shared_network():
                    try:
                        p=collect()
                        on_save_to_pool(p.model_dump(mode='json',exclude_none=True))
                        ui.notify('Saved network problem to shared YAML pool',type='positive')
                    except Exception as ex:
                        ui.notify(str(ex),type='negative')
                ui.button('Save to Query Pool',icon='playlist_add',on_click=save_shared_network).props('outline')
            visualize_button=ui.button('Run Now',icon='play_arrow',on_click=visualize)
            if on_save_to_pool is not None:visualize_button.set_visibility(False)
            native_batch_button=ui.button('Add to Batch',icon='playlist_add',on_click=add_queue).props('outline')
            if on_save_to_pool is not None:native_batch_button.set_visibility(False)
        status=ui.label('Browse a GEM to begin.').classes('text-sm')
        details()
    result_area=ui.column().classes('w-full')
    with ui.expansion('Advanced · Portable bundle and batch execution',icon='playlist_play').classes('w-full') as legacy_batch_panel:
        if on_save_to_pool is not None:legacy_batch_panel.set_visibility(False)
        ui.button('Export Problem Bundle (includes GEM)',on_click=bundle).props('outline')
        ui.upload(label='Load several YAML problems into batch',multiple=True,auto_upload=True,on_upload=queue_upload).props('accept=.yaml,.yml').classes('w-full')
        queue_view()
        with ui.row():
            ui.button('Run All / Resume',on_click=run_all)
            ui.button('Pause after current job',on_click=lambda:runner.pause()).props('outline')
            ui.button('Validate Queue',on_click=lambda:queue_view.refresh()).props('flat')
            ui.button('Clear Queue',on_click=clear_queue).props('flat')
        batch_status=ui.label('Queue is empty.')
        ui.button('Export batch manifest',on_click=lambda:ui.download(runner.path) if runner.path.exists() else ui.notify('Run a batch first.')).props('flat')
        ui.button('Export batch results CSV',on_click=lambda:ui.download(runner.root/'batch_results.csv') if runner.path.exists() else ui.notify('Run a batch first.')).props('flat')
    ui.timer(.5,poll_progress)
    update_visibility()
