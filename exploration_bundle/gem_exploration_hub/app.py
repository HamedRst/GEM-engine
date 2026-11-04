"""Single Exploration page with native Calculator/Atlas and run-to-render YAML results."""
from __future__ import annotations
import os
from pathlib import Path
import json
import secrets
from nicegui import ui,app,run
from .pool import QueryPool
from .run_registry import execute as run_one
from gem_exploration_gui.app import _build_page as calculator_page
from gem_atlas_network.ui import build_network_visualization_tab
from gem_exploration_tab import register_exploration_routes, render_exploration_tab

ROOT=Path(__file__).resolve().parents[1]
OUTPUTS=Path(os.environ.get('GEM_EXPLORATION_OUTPUTS',str(ROOT/'outputs'))).expanduser().resolve();OUTPUTS.mkdir(parents=True,exist_ok=True)
POOL=QueryPool(Path(os.environ.get('GEM_EXPLORATION_WORKSPACE',str(Path.home()/'Desktop'/'GEM_Exploration_Workspace'))))
# Legacy bundled demo questions are not shown or executed in the public question list.
# Their files are left untouched in workspaces created by earlier versions.
_LEGACY_DEMOS={
  'DEMO_01_describe_growth','DEMO_02_sensitivity_product','DEMO_03_interactions',
  'DEMO_04_robustness','DEMO_05_response','DEMO_06_tradeoff','DEMO_07_pareto',
  'NETWORK_G6P_BRANCH','NETWORK_PGK_KNOCKOUT','NETWORK_PPP_PATHWAY',
}
def visible_question_names():
    return [name for name in POOL.names() if name not in _LEGACY_DEMOS]

app.add_static_files('/exploration-results',str(POOL.root/'results'))
app.add_static_files('/atlas-output',str(OUTPUTS))
# Additive read-only Exploration atlas: no solver/FBA/YAML modifications.
register_exploration_routes()
ui.colors(primary='#157A6E',secondary='#667085',accent='#B98B3B')


def query_pool_panel(on_refresh_ready=None):
    ui.label('Saved Questions').classes('text-xl font-semibold')
    ui.label('Import or save questions from any builder, then run one or all.').classes('text-sm text-gray-600')
    status=ui.label('No query selected.')
    result_area=ui.column().classes('w-full')
    def render_result(summary):
        result_area.clear()
        from urllib.parse import quote
        from pathlib import Path
        result=summary.get('result_directory')
        with result_area:
            ui.label('Run results · '+str(summary.get('problem_id',''))).classes('text-lg font-semibold')
            if summary.get('problem_type')=='exploration.activity' and result:
                # Pool runner always creates a separate output directory; the root is fixed.
                resolved=Path(result).resolve().relative_to((POOL.root/'results').resolve())
                url='/exploration-results/'+quote(resolved.as_posix(),safe='/')+'/viewer/index.html'
                ui.label('Requested biological view · generated only after running the YAML').classes('text-sm text-gray-600')
                ui.link('Feature coverage and interpretation audit', '/exploration-results/'+quote(resolved.as_posix(),safe='/')+'/FEATURE_COVERAGE.json')
                ui.element('iframe').props('src="'+url+'"').classes('gem-result-frame').style('width:100%;min-height:680px;height:1280px;border:1px solid #dae4ea;border-radius:10px;')
            elif result:
                directory=Path(result).resolve()
                if not directory.is_relative_to((POOL.root/'results').resolve()):return
                pics=list(directory.rglob('*.png'))[:18]
                if pics:
                    for fig in pics:
                        relative=fig.relative_to((POOL.root/'results').resolve()).as_posix()
                        with ui.card().classes('w-full'):
                            ui.label(fig.name)
                            ui.image('/exploration-results/'+quote(relative,safe='/')).classes('w-full max-w-4xl')
                else:
                    ui.label('Saved results directory: '+str(directory))
                docs=list(directory.rglob('*.md'))[:10]+list(directory.rglob('*.pdf'))[:12]+list(directory.rglob('*.svg'))[:12]
                docs+=list(directory.rglob('FEATURE_COVERAGE.json'))
                for doc in docs:
                    rel=doc.relative_to((POOL.root/'results').resolve()).as_posix()
                    ui.link('Open '+doc.name,'/exploration-results/'+quote(rel,safe='/'))
            else:
                ui.label(str(summary))

    @ui.refreshable
    def listing():
        names=visible_question_names()
        if not names:ui.label('Save questions from either subtab to start.');return
        for name in names:
            with ui.row().classes('w-full items-center'):
                ui.label(name).classes('flex-1')
                ui.button('YAML',on_click=lambda n=name:ui.download(POOL.root/'problems'/f'{n}.yaml')).props('flat dense')
                async def go(n=name):
                    status.set_text('Running '+n+'…')
                    try:
                        result=await run.io_bound(run_one,POOL.root,n)
                        if result.get('acquisition'):
                            acquired=result['acquisition']
                            note=f" · Exact FBA: {result.get('new_fba_calls',0)} · New designs: {result.get('new_designs',0)} · QC: {acquired.get('status','unknown')}"
                        else:note=''
                        status.set_text('Complete: '+n+note+' · output saved under '+str(POOL.root/'results'))
                        render_result(result)
                        ui.notify('Completed '+n,type='positive')
                    except Exception as e:
                        status.set_text('Failed closed: '+str(e));ui.notify(str(e),type='negative',multi_line=True)
                ui.button('Run',on_click=go).props('outline dense')
    with ui.row():
        ui.button('Refresh saved queries',on_click=listing.refresh).props('outline')
        async def run_all():
            for name in visible_question_names():
                status.set_text('Running '+name+'…')
                try:
                    result=await run.io_bound(run_one,POOL.root,name)
                    render_result(result)
                except Exception as ex:
                    ui.notify(name+': '+str(ex),type='negative')
                    status.set_text('Stopped on '+name+'; see failed output')
                    return
            status.set_text('Finished all saved queries')
        ui.button('Run All',on_click=run_all)
    listing()
    if on_refresh_ready is not None:on_refresh_ready(listing.refresh)
    ui.label('Import saved question YAML (Activity, Calculator or Network)').classes('font-medium mt-3')
    async def uploaded(e):
        try:
            data=e.content.read();name=Path(e.name).name
            temp=POOL.root/'uploads';temp.mkdir(exist_ok=True)
            dest=temp/name;dest.write_bytes(data)
            POOL.import_native(dest);listing.refresh()
            ui.notify('Added '+name,type='positive')
        except Exception as ex:ui.notify(str(ex),type='negative')
    ui.upload(on_upload=uploaded,auto_upload=True).props('accept=.yaml,.yml').classes('max-w-lg')
    with ui.expansion('Advanced · adaptive FBA when surrogate QC is insufficient',icon='science').classes('w-full'):
        ui.label('Optional. Runs exact FBA only after failed original Calculator QC. Requires a verified source manifest and matching model. Sources are never edited.').classes('text-sm text-gray-600')
        selected=ui.select(options=visible_question_names(),label='Saved Calculator question').classes('w-full max-w-md')
        ui.button('Refresh questions',on_click=lambda:selected.set_options(visible_question_names())).props('flat dense')
        manifest_input=ui.input('Verified manifest path (leave blank for source.csv.manifest.json)').classes('w-full')
        model_input=ui.input('GEM model path (leave blank to use YAML model)').classes('w-full')
        with ui.row().classes('items-end'):
            budget=ui.number('Max new exact designs (group total)',value=16,min=0,max=10000,step=1)
            batch=ui.number('Designs per QC attempt',value=4,min=1,max=1000,step=1)
            rounds=ui.number('Max QC attempts',value=4,min=1,max=100,step=1)
        def enable_adaptive():
            try:
                from .acquisition import evaluator_identity,digest
                if not selected.value:raise ValueError('Choose a saved Calculator question')
                q=POOL.load(selected.value)
                if q['problem_type']!='exploration.calculator':raise ValueError('Adaptive mode is for Calculator questions')
                source=q['payload']['internal_problem'].get('landscape',{}).get('path')
                if not source or not Path(source).is_file():raise ValueError('Exact source CSV path in the question is unavailable')
                source=Path(source).expanduser().resolve()
                if source.suffix.lower()!='.csv':raise ValueError('This verified acquisition backend currently requires CSV')
                model=model_input.value or (q['payload']['internal_problem'].get('unified_exploration') or {}).get('gem_model_path')
                if not model:raise ValueError('Provide the verified GEM model path')
                manifest=Path(manifest_input.value).expanduser().resolve() if manifest_input.value else Path(str(source)+'.manifest.json')
                m=json.loads(manifest.read_text())
                evalu=m.get('evaluator')
                if not isinstance(evalu,dict):raise ValueError('Manifest needs a verified evaluator definition')
                sha,_=evaluator_identity(model,evalu)
                if m.get('source_sha256')!=digest(source) or m.get('evaluator_sha256')!=sha:
                    raise ValueError('Source and model/evaluator do not match verified manifest')
                values=(int(budget.value),int(batch.value),int(rounds.value))
                if not 0<=values[0]<=10000 or not 1<=values[1]<=1000 or not 1<=values[2]<=100:
                    raise ValueError('Invalid budgets')
                q['execution']={'new_fba_policy':'adaptive_qc','landscape_copy':'when_exact_fba_needed',
                    'acquisition':{'source_csv':str(source),'source_manifest':str(manifest),'model_path':str(Path(model).expanduser().resolve()),
                        'evaluator':evalu,'query_group':q['problem_id'],'max_new_designs_total':values[0],
                        'batch_size':values[1],'max_rounds':values[2],'seed':42}}
                POOL.save(q,overwrite=True)
                ui.notify('Adaptive QC enabled for '+q['problem_id']+'; exact FBA will run only if necessary.',type='positive')
            except Exception as ex:ui.notify(str(ex),type='negative',multi_line=True)
        ui.button('Validate and enable adaptive QC',on_click=enable_adaptive)
        ui.label('New FBA requires explicit opt-in per YAML. You can switch back by editing new_fba_policy: never in the saved YAML.').classes('text-caption text-grey-7')



@ui.page('/')
def page():
    ui.page_title('GEM Engine | Exploration · Single Tab')
    # A direct Exploration browser tab is a GUI client too. Its heartbeat is
    # independent from the main GEM page and keeps the local server alive.
    ui.add_head_html("""<script>
    (function(){
      const sid='explore-'+(globalThis.crypto?.randomUUID?.()||String(Date.now())+Math.random());
      const send=(event)=>fetch('/api/session/'+event,{method:'POST',
        headers:{'Content-Type':'application/json'},body:JSON.stringify({session_id:sid}),keepalive:true}).catch(()=>{});
      send('open');setInterval(()=>send('heartbeat'),4000);
      window.addEventListener('pagehide',()=>{
        try{navigator.sendBeacon('/api/session/close',new Blob([JSON.stringify({session_id:sid})],{type:'application/json'}))}catch(e){}
      });
    })();
    </script>""")
    if os.environ.get('GEM_EXPLORATION_EMBEDDED') == '1':
        ui.dark_mode().enable()
        ui.colors(primary='#62d9ee', secondary='#91a9b6', accent='#75e0a7')
        ui.add_head_html("""<style>
        :root { --gem-bg:#071019; --gem-panel:#0e1d2b; --gem-input:#081823; --gem-text:#eef8fb; --gem-muted:#91a9b6; --gem-border:rgba(164,207,226,.16); }
        html, body, #app, .nicegui-content {background:var(--gem-bg)!important;color:var(--gem-text)!important;font-family:Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif!important;}
        .nicegui-content {padding:0!important;}
        .q-header {background:#0b1a27!important;color:var(--gem-text)!important;border-bottom:1px solid var(--gem-border)!important;}
        .q-card, .q-expansion-item, .q-list {background:var(--gem-panel)!important;color:var(--gem-text)!important;border:1px solid var(--gem-border);border-radius:15px;}
        .q-field__control, .q-field__native, .q-select .q-field__control {color:var(--gem-text)!important;}
        .q-field--outlined .q-field__control:before {border-color:rgba(164,207,226,.30)!important;}
        .q-field__label,.q-field__bottom,.q-placeholder,.text-gray-600,.text-grey-7,.text-grey-6 {color:var(--gem-muted)!important;}
        [class*="text-[#17384a]"] {color:var(--gem-text)!important;}
        .q-btn--outline:before {border-color:rgba(98,217,238,.48)!important;}
        .q-btn {border-radius:10px!important;}
        .q-table, .q-menu, .q-dialog__inner>.q-card {background:var(--gem-panel)!important;color:var(--gem-text)!important;}
        a {color:#62d9ee;}
        .gem-result-frame {background:#fff!important;}
        </style>""")
    with ui.header().classes('items-center'):
        ui.label('GEM Engine · Exploration').classes('text-xl font-bold')
        ui.space();ui.label('Questions run only when requested').classes('text-sm')
    async def run_from_builder(name):
        result=await run.io_bound(run_one,POOL.root,name)
        # The saved-pool listing is refreshed when the user clicks Refresh.
        ui.notify('Question completed: '+name,type='positive')
        return result
    pool_refresh={}
    def update_pool_listing():
        if 'refresh' in pool_refresh:pool_refresh['refresh']()
    def pool_render():
        query_pool_panel(on_refresh_ready=lambda fn:pool_refresh.update(refresh=fn))
    def calculator_save(public,internal):
        saved=POOL.save_calculator(public,internal)
        update_pool_listing()
        return saved
    def network_save(native):
        saved=POOL.save_network(native)
        update_pool_listing()
        return saved
    def save_activity_idempotent(name,payload):
        if name in visible_question_names():
            previous=POOL.load(name)
            if previous['problem_type']!='exploration.activity' or previous['payload']!=payload:
                raise FileExistsError('Question ID already belongs to different YAML; choose a different ID')
            update_pool_listing()
            return POOL.root/'problems'/f'{name}.yaml'
        saved=POOL.save_activity(name,payload)
        update_pool_listing()
        return saved
    render_exploration_tab(
        outputs_dir=OUTPUTS,
        example_models_dir=ROOT/'examples'/'models',
        calculator_render=lambda:calculator_page(embedded=True,on_save_to_pool=calculator_save),
        query_pool_render=pool_render,
        on_save_network=network_save,
        on_save_activity=save_activity_idempotent,
        on_run_activity=run_from_builder,
    )


def main():
    secret=ROOT/'.exploration_storage_secret'
    if not secret.exists():secret.write_text(secrets.token_hex(32));secret.chmod(0o600)
    ui.run(title='GEM Engine | Exploration',host='127.0.0.1',port=int(os.environ.get('GEM_EXPLORATION_PORT','8092')),
           reload=False,show=os.environ.get('GEM_EXPLORATION_HEADLESS')!='1',storage_secret=secret.read_text().strip())

if __name__=='__main__':main()
