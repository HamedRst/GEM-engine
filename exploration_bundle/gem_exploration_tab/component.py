
"""Unified Exploration query builder and result viewer."""
from __future__ import annotations
import hashlib, json, os, shutil, uuid
from pathlib import Path
from nicegui import ui, app, run
from .build_bundle import export

_ROOT=Path(__file__).resolve().parent
_DEFAULT_ACTIVITY=_ROOT/'assets'
_OUTPUTS=Path(os.environ.get('GEM_EXPLORATION_OUTPUTS',str(Path.home()/'Documents'/'GEM Engine Workspace'/'exploration_workspace'/'atlas_outputs')))/'single_tab_activity'
_OUTPUTS.mkdir(exist_ok=True,parents=True)
_ROUTES_REGISTERED=False

def register_exploration_routes():
    """Call once at application startup, before NiceGUI serves requests."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:return
    app.add_static_files('/exploration-atlas',str(_DEFAULT_ACTIVITY))
    app.add_static_files('/exploration-datasets',str(_OUTPUTS))
    _ROUTES_REGISTERED=True

def _check_paths(model_path: str, matrix_path: str):
    model=Path(model_path).expanduser().resolve()
    matrix=Path(matrix_path).expanduser().resolve()
    if not model.is_file() or model.suffix.lower() not in {'.json','.sbml','.xml'}:
        raise ValueError('GEM source must be a valid COBRA JSON or SBML (.xml/.sbml) file.')
    if not matrix.is_file() or matrix.suffix.lower()!='.npz':
        raise ValueError('Supply an existing NPZ flux matrix with growth, product, flux and reaction_ids.')
    return model,matrix

def _build_dataset(model_path,matrix_path,sample_size):
    import numpy as np
    model_file,matrix_file=_check_paths(model_path,matrix_path)
    if model_file.suffix.lower()=='.json':
        model=json.loads(model_file.read_text(encoding='utf-8'))
    else:
        # Convert format only. This does not solve FBA or change the source GEM.
        try:
            from cobra.io import read_sbml_model, save_json_model
        except ImportError as exc:
            raise RuntimeError('COBRApy is required for SBML model conversion') from exc
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as tmp:
            converted=Path(tmp)/'converted_model.json'
            save_json_model(read_sbml_model(str(model_file)),str(converted))
            model=json.loads(converted.read_text(encoding='utf-8'))
    for k in ('reactions','genes','metabolites'):
        if k not in model or not isinstance(model[k],list):raise ValueError('Invalid COBRA JSON: missing '+k)
    with np.load(matrix_file,allow_pickle=False) as matrix:
        required={'growth','product','flux','reaction_ids'}
        if not required.issubset(matrix.files):raise ValueError('Missing NPZ fields: '+', '.join(sorted(required-set(matrix.files))))
        # In exporter, the column order is verified against every reaction ID.
        # FBA is NOT run and the supplied source files are never modified.
        token=uuid.uuid4().hex[:16]
        dirname=_OUTPUTS/token
        dirname.mkdir(parents=True,exist_ok=False)
        try:
            shutil.copyfile(_DEFAULT_ACTIVITY/'index.html',dirname/'index.html')
            data=export(model,matrix['growth'],matrix['product'],matrix['flux'],dirname/'bundle.js',
                name=(model.get('name') or model.get('id') or 'User-supplied GEM'),
                sample_size=int(sample_size),reaction_ids=matrix['reaction_ids'])
            (dirname/'SOURCE.json').write_text(json.dumps({
                'model_sha256':hashlib.sha256(model_file.read_bytes()).hexdigest(),
                'flux_matrix_sha256':hashlib.sha256(matrix_file.read_bytes()).hexdigest(),
                'model_id':model.get('id',''),
                'num_conditions':data['nall'],'preview_conditions':data['n'],
                'feature_counts':{'genes':data['ng'],'reactions':data['nr'],'metabolites':data['nm']},
                'data_policy':'Objective-conditioned saved FBA flux, not gene expression or metabolite concentration.'
                },indent=2),encoding='utf-8')
        except Exception:
            shutil.rmtree(dirname,ignore_errors=True)
            raise
    return token,data

def render_exploration_tab(*, outputs_dir, example_models_dir=None, calculator_render=None,
                           query_pool_render=None, on_save_network=None,
                           on_save_activity=None, on_run_activity=None):
    """A single Exploration page. Visual outputs are created ONLY by running YAML.

    Native Calculator and native Network Atlas remain available, unchanged.
    """
    from gem_atlas_network.ui import build_network_visualization_tab
    import re
    import yaml
    example_models_dir=example_models_dir or (Path(__file__).resolve().parents[1]/'examples'/'models')
    shared={'model_path':None}
    ui.add_head_html("""<style>.gem-exploration-nav-active{background:rgba(98,217,238,.19)!important;color:#fff!important;border-bottom:2px solid #62d9ee!important}</style><script>
    window.addEventListener('message',function(event){
      if(event.origin!==window.location.origin || !event.data || event.data.type!=='gem-exploration-iframe-height')return;
      document.querySelectorAll('iframe.gem-result-frame').forEach(function(frame){
        if(frame.contentWindow===event.source){
          let h=Math.max(650,Math.min(24000,Number(event.data.height)||1000));
          frame.style.height=(h+20)+'px';
        }
      });
    });</script>""")
    with ui.column().classes('w-full max-w-[1560px] mx-auto px-4 py-3 gap-4'):
        ui.label('GEM Engine · Exploration').classes('text-3xl font-bold text-[#17384a]')
        ui.label('Build a question, save it, and run it when ready.').classes('text-gray-600')
        # All source paths remain in the persistent shared workspace, not in page-local uploads.
        panels={}
        nav_buttons={}
        source_settings=Path(outputs_dir).resolve()/'selected_sources.json'
        previous={}
        if source_settings.exists():
            try:
                previous=json.loads(source_settings.read_text(encoding='utf-8'))
            except (ValueError,OSError):previous={}
        shared.update({name:previous.get(name) if previous.get(name) and Path(previous[name]).expanduser().exists() else None
                       for name in ('model_path','landscape_path','config_path')})
        navigation=ui.row().classes('w-full items-center gap-2 flex-wrap').style('border-bottom:1px solid rgba(164,207,226,.2);padding-bottom:10px;')
        def choose(name):
            for key,widget in panels.items():widget.set_visibility(key==name)
            for key,button in nav_buttons.items():button.classes(add='gem-exploration-nav-active' if key==name else '',remove='' if key==name else 'gem-exploration-nav-active')
            if name=='landscape':
                if shared.get('model_path'):model.set_value(shared['model_path'])
                if shared.get('landscape_path'):matrix.set_value(shared['landscape_path'])
            elif name=='calculator':
                # Pre-fill original Calculator source controls without altering its solver.
                from gem_exploration_gui.app import state as calc_state, source_status
                if shared.get('landscape_path'):calc_state.landscape_path=shared['landscape_path']
                if shared.get('model_path'):calc_state.model_path=shared['model_path']
                source_status.refresh()
        with navigation:
            for key,label in [('setup','Setup'),('landscape','Landscape View'),('calculator','Calculator'),
                              ('network','Network Visualization'),('questions','Saved Questions')]:
                nav_buttons[key]=ui.button(label,on_click=lambda n=key:choose(n)).props('outline no-caps').classes('gem-exploration-nav')
        with ui.column().classes('w-full') as setup_panel:
            ui.label('Input files').classes('text-xl font-semibold')
            ui.label('Choose the model, landscape and optional configuration once. Saved questions can reuse these sources.').classes('text-sm text-gray-600')
            model_path=ui.input('GEM model',value=shared.get('model_path') or '',placeholder='Choose a file below or enter an existing local path').classes('w-full')
            landscape_path=ui.input('Landscape (CSV / Parquet / NPZ or its containing folder)',value=shared.get('landscape_path') or '',placeholder='Saved landscape path').classes('w-full')
            config_path=ui.input('Landscape configuration YAML (optional)',value=shared.get('config_path') or '',placeholder='Simulation configuration').classes('w-full')
            setup_status=ui.label('No sources selected.').classes('text-sm text-gray-600')
            def use_sources():
                from pathlib import Path
                import json
                values=[str(x.value or '').strip() for x in (model_path,landscape_path,config_path)]
                for val in values:
                    if val and not Path(val).expanduser().is_file() and not Path(val).expanduser().is_dir():
                        raise ValueError('Source does not exist on this computer: '+val)
                if not values[0] and not values[1]:raise ValueError('Select a model and/or a landscape')
                shared.update(model_path=values[0] or None,landscape_path=values[1] or None,config_path=values[2] or None)
                source_settings.parent.mkdir(parents=True,exist_ok=True)
                tmp=source_settings.with_suffix('.tmp')
                tmp.write_text(json.dumps(shared,indent=2),encoding='utf-8')
                tmp.replace(source_settings)
                setup_status.set_text('Shared sources saved for this session. Your question YAML is created separately.')
                ui.notify('Sources ready. Choose a question builder.',type='positive')
            def save_sources():
                try:use_sources()
                except Exception as exc:ui.notify(str(exc),type='negative')
            with ui.row():
                ui.button('Use selected sources',on_click=save_sources).props('color=primary')
                def load_config_fields():
                    try:
                        source=Path(config_path.value or '').expanduser().resolve()
                        if source.suffix.lower() not in ('.yaml','.yml') or not source.is_file():
                            raise ValueError('Choose an existing YAML configuration first')
                        cfg=yaml.safe_load(source.read_text(encoding='utf-8'))
                        if not isinstance(cfg,dict):raise ValueError('Configuration must contain a YAML mapping')
                        model_cfg=cfg.get('model',{}) or {};project_cfg=cfg.get('project',{}) or {}
                        model_candidate=model_cfg.get('path') or model_cfg.get('file')
                        if model_candidate:
                            path=Path(model_candidate).expanduser()
                            if not path.is_absolute():path=source.parent/path
                            if path.is_file():model_path.set_value(str(path.resolve()))
                        landscape_cfg=cfg.get('landscape',{}) or {}
                        landscape_candidate=landscape_cfg.get('path') or landscape_cfg.get('file')
                        if landscape_candidate:
                            path=Path(landscape_candidate).expanduser()
                            if not path.is_absolute():path=source.parent/path
                            if path.exists():landscape_path.set_value(str(path.resolve()))
                        ui.notify('Configuration inspected. Verify the source paths, then save.',type='positive')
                    except Exception as exc:ui.notify(str(exc),type='negative')
                ui.button('Read configuration',on_click=load_config_fields).props('outline')
            def upload_source(field):
                async def take(e):
                    from pathlib import Path
                    name=Path(e.name).name
                    if name in ('','.','..'):return
                    destination=Path(outputs_dir).resolve()/'user_sources';destination.mkdir(parents=True,exist_ok=True)
                    target=destination/name
                    if target.exists():target=destination/(target.stem+'_'+__import__('uuid').uuid4().hex[:8]+target.suffix)
                    target.write_bytes(e.content.read())
                    field.set_value(str(target))
                    ui.notify('Added '+name,type='positive')
                return take
            with ui.row().classes('w-full flex-wrap gap-4'):
                ui.upload(label='Upload GEM',on_upload=upload_source(model_path),auto_upload=True).props('accept=.xml,.sbml,.json,.mat').classes('max-w-sm')
                ui.upload(label='Upload landscape',on_upload=upload_source(landscape_path),auto_upload=True).props('accept=.csv,.parquet,.npz,.gz').classes('max-w-sm')
                ui.upload(label='Upload configuration',on_upload=upload_source(config_path),auto_upload=True).props('accept=.yaml,.yml').classes('max-w-sm')
            ui.label('Files uploaded here stay outside the application bundle. Existing absolute paths may also be used.').classes('text-xs text-gray-600')
        panels['setup']=setup_panel
        with ui.column().classes('w-full') as landscape_panel:
            with ui.card().classes('w-full p-4').props('id=activity-question'):
                ui.label('Landscape View · new question').classes('text-xl font-semibold')
                ui.label('Choose a region map, an activity heatmap, or one feature.').classes('text-sm text-gray-600')
                with ui.row().classes('items-end gap-3 flex-wrap w-full'):
                    question_id=ui.input('Question ID',value='compare_gene_activity').classes('w-64')
                    view=ui.select({'overview':'Full-feature heatmap','feature':'One feature on landscape','regions':'Growth/product region map'},value='overview',label='What to visualize?').classes('w-64')
                    layer=ui.select({'gene':'Genes','reaction':'Reactions','metabolite':'Metabolites'},value='gene',label='Biological layer').classes('w-52')
                    fid=ui.input('Exact feature ID (single-feature view)',placeholder='e.g. b0432 or CYTBO3_4pp').classes('w-72')
                with ui.row().classes('items-end gap-3 flex-wrap w-full'):
                    method=ui.select({'angular':'Angular · trade-off direction','circular':'Circular · combined capability'},value='angular',label='Region method').classes('w-72')
                    angular_count=ui.number('Angular sectors',value=6,min=2,max=12,step=1).classes('w-36')
                    circular_count=ui.number('Circular rings',value=6,min=2,max=12,step=1).classes('w-36')
                    source=ui.select({'matched_npz':'GEM + matched flux NPZ','matched_table':'GEM + CSV/Parquet flux table'},value='matched_npz',label='Dataset source').classes('w-80')
                    sample=ui.number('Max preview conditions',value=12000,min=100,max=100000,step=100).classes('w-44')
                with ui.row().classes('items-end w-full gap-3 flex-wrap'):
                    model=ui.input('COBRA JSON / SBML model path',placeholder='/path/to/model.json').classes('grow min-w-[280px]')
                    matrix=ui.input('Saved landscape NPZ path',placeholder='/path/to/landscape.npz').classes('grow min-w-[280px]')
                with ui.row().classes('gap-3 flex-wrap'):
                    growth_col=ui.input('CSV/Parquet growth column',value='growth').classes('w-60')
                    product_col=ui.input('CSV/Parquet product column',value='product').classes('w-60')
                with ui.expansion('Advanced · axis labels and provenance',value=False).classes('w-full'):
                    with ui.row().classes('w-full gap-3'):
                        growth_label=ui.input('Growth axis label',value='Growth (model objective units)').classes('w-80')
                        product_label=ui.input('Product axis label',value='Product (model flux units)').classes('w-80')
                ui.label('Activity analysis requires matching reaction fluxes for the selected GEM. Phenotype-only landscapes are insufficient.').classes('text-sm text-gray-600')
                preview=ui.textarea('Generated YAML',value='',placeholder='Click Preview YAML or Save YAML').props('readonly autogrow outlined').classes('w-full')
                status=ui.label('Choose a question and save its YAML before running it.').classes('text-sm text-gray-600')
                def values():
                    payload={'source':str(source.value),'view':str(view.value),'feature_type':str(layer.value),
                             'feature_id':str(fid.value or '').strip(),'region_method':str(method.value),
                             'n_regions':int(angular_count.value if method.value=='angular' else circular_count.value),'new_fba_policy':'never',
                             'growth_label':str(growth_label.value or '').strip(),'product_label':str(product_label.value or '').strip()}
                    if payload['source'] in {'matched_npz','matched_table'}:
                        payload.update(model_path=str(model.value or '').strip(),matrix_path=str(matrix.value or '').strip(),sample_size=int(sample.value))
                        if payload['source']=='matched_table':
                            payload.update(growth_column=str(growth_col.value or '').strip(),product_column=str(product_col.value or '').strip())
                    from gem_exploration_hub.activity_queries import validate_payload
                    validate_payload(payload)
                    name=str(question_id.value or '').strip()
                    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',name):
                        raise ValueError('Question ID must use 1–80 letters/numbers/underscores/hyphens')
                    return name,payload
                def show_yaml():
                    try:
                        name,payload=values()
                        value={'schema_version':'1.0','problem_id':name,'problem_type':'exploration.activity',
                               'payload':payload,'execution':{'new_fba_policy':'never'}}
                        preview.set_value(yaml.safe_dump(value,sort_keys=False))
                        status.set_text('YAML validated. Save it, then run it from Saved Questions. No plots generated yet.')
                    except Exception as exc:ui.notify(str(exc),type='negative')
                def save_activity():
                    name,payload=values()
                    if on_save_activity is None:raise RuntimeError('This standalone view needs a query-pool callback')
                    path=on_save_activity(name,payload)
                    if payload['source'] in {'matched_npz','matched_table'}:shared['model_path']=str(payload['model_path'])
                    show_yaml()
                    status.set_text('Saved YAML: '+str(path))
                    return name
                result_display=ui.column().classes('w-full')
                async def run_now():
                    try:
                        name=save_activity()
                        if on_run_activity is not None:
                            status.set_text('Running '+name+'…')
                            summary=await on_run_activity(name)
                            result_display.clear()
                            with result_display:
                                from urllib.parse import quote
                                result=Path(summary['result_directory']).resolve()
                                ui.label('Rendered result · '+name).classes('text-xl font-semibold')
                                url='/exploration-results/'+quote(result.name,safe='')+'/viewer/index.html'
                                ui.element('iframe').props('src="'+url+'"').classes('gem-result-frame').style('width:100%;min-height:680px;height:1200px;border:1px solid #dde4eb;border-radius:9px')
                            status.set_text('Run complete: '+name+'; requested result shown below.')
                        else:status.set_text('YAML saved. Run it from Shared YAML query pool.')
                    except Exception as exc:
                        status.set_text('Run failed: '+str(exc));ui.notify(str(exc),type='negative',multi_line=True)
                with ui.row().classes('gap-2'):
                    ui.button('Preview YAML',icon='code',on_click=show_yaml).props('outline')
                    def save_click():
                        try:
                            save_activity();ui.notify('YAML saved to shared query pool',type='positive')
                        except Exception as exc:ui.notify(str(exc),type='negative')
                    ui.button('Save YAML',icon='save',on_click=save_click).props('outline')
        panels['landscape']=landscape_panel
        with ui.column().classes('w-full') as calculator_panel:
            with ui.column().classes('w-full').props('id=original-calculator'):
                ui.label('Build and save Calculator questions.').classes('text-sm text-gray-600')
                if calculator_render is not None: calculator_render()
        panels['calculator']=calculator_panel
        with ui.column().classes('w-full') as network_panel:
            with ui.column().classes('w-full').props('id=native-network-atlas'):
                ui.label('Explore networks, pathways, FBA, and gene knockouts.').classes('text-sm text-gray-600')
                build_network_visualization_tab(outputs_dir,example_models_dir,on_save_to_pool=on_save_network,shared_model_path_getter=lambda:shared['model_path'])
        panels['network']=network_panel
        with ui.column().classes('w-full') as questions_panel:
            with ui.card().classes('w-full p-4').props('id=shared-queries'):
                ui.label('Saved Questions').classes('text-xl font-semibold')
                if query_pool_render is not None:query_pool_render()
        panels['questions']=questions_panel
        choose('setup')
