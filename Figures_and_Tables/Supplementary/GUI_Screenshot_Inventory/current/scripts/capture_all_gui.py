#!/usr/bin/env python3
"""Read-only capture inventory of actual GEM Engine interface and existing results.

No scientific jobs, FBA, result generation, image retouching, or composites.
Snapshots are raw Playwright viewport/element captures from the running GUI.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, time, subprocess, uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError, HTTPError
from urllib.request import urlopen

from PIL import Image
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
MAIN='GUI_Screenshot_Inventory'
W,H=1900,1200

def url_json(url, timeout=3):
    with urlopen(url, timeout=timeout) as response:
        return json.load(response)


def app_candidates(home=None):
    """Only known GEM Engine app paths; never traverse the entire user disk."""
    home=Path(home or Path.home())
    candidates=[Path('/Applications/GEM Engine.app'), home/'Applications'/'GEM Engine.app',
                home/'Desktop'/'GEM Engine.app', home/'Downloads'/'GEM Engine.app']
    for root in (home/'Downloads', home/'Desktop'):
        if root.is_dir():
            candidates.extend(sorted(root.glob('GEM_Engine_v2.1_Unified_Optimization_Exploration_Mac_QA*/GEM Engine.app')))
    out=[];seen=set()
    for app in candidates:
        if not app.is_dir():continue
        resolved=app.resolve()
        if resolved in seen:continue
        seen.add(resolved)
        launcher=resolved/'Contents'/'MacOS'/'GEM Engine'
        if launcher.is_file():out.append(resolved)
    return out


def probe(base):
    try:
        health=url_json(base+'/api/health', timeout=1.0)
        exp=url_json(base+'/api/exploration/status', timeout=1.0)
        if not isinstance(health, dict) or not isinstance(exp,dict):return None
        return {'health':health,'exploration':exp}
    except (OSError, ValueError, TimeoutError, HTTPError, URLError):
        return None


def port_from_file(home=None):
    home=Path(home or Path.home())
    port_path=home/'Library/Application Support/GEM Engine/server.port'
    try:
        raw=port_path.read_text(encoding='utf-8').strip()
        value=int(raw)
        if 1<=value<=65535:return value
    except (ValueError,OSError):pass
    return None


def local_server_candidates(home=None):
    """Port file may be stale. Probe only GEM Engine's documented loopback range."""
    ports=[]
    reported=port_from_file(home)
    if reported:ports.append(reported)
    ports.extend(p for p in range(8081,8110) if p!=reported)
    return ['http://127.0.0.1:'+str(p) for p in ports]


def live_gui_url(home=None):
    for base in local_server_candidates(home):
        response=probe(base)
        if response and response['exploration'].get('available') is True:
            return base
    return None


def diagnostics(home=None):
    home=Path(home or Path.home())
    logs=home/'Library'/'Logs'/'GEM Engine'
    pieces=[]
    for name in ('setup.log','server.log'):
        file=logs/name
        if file.is_file():
            try:
                lines=file.read_text(encoding='utf-8',errors='replace').splitlines()
                pieces.append(str(file)+':\n'+'\n'.join(lines[-15:]))
            except OSError:pass
    return '\n'.join(pieces)


def ensure_gui_server(*, base=None,app=None,wait_seconds=240, launch=True, home=None):
    """Launch installed Mac app if needed; never start a simulation or worker job."""
    home=Path(home or Path.home())
    if base:
        base=base.rstrip('/')
        state=probe(base)
        if not state:
            raise RuntimeError('Cannot connect to --url '+base+'. Start GEM Engine.app or specify its active loopback URL.')
        if not state['exploration'].get('available'):
            raise RuntimeError('Exploration is not ready at '+base+': '+str(state['exploration']))
        return base, 'explicit_url'
    live=live_gui_url(home)
    if live:return live,'already_running'
    if not launch:
        raise RuntimeError('No live integrated GEM Engine service found. Open GEM Engine.app and retry.')
    if sys.platform!='darwin':
        raise RuntimeError('Automatic launch requires macOS. Start GEM Engine.app manually, then use --url.')
    if app:
        chosen=Path(app).expanduser().resolve()
        if not (chosen/'Contents'/'MacOS'/'GEM Engine').is_file():
            raise RuntimeError('Not an extracted GEM Engine.app: '+str(chosen))
    else:
        options=app_candidates(home)
        if not options:
            raise RuntimeError('No extracted GEM Engine.app found in Downloads, Desktop, or Applications. Use --app "/path/to/GEM Engine.app".')
        # A patch-installed Downloads release is prioritized over an older /Applications copy.
        preferred=[x for x in options if 'GEM_Engine_v2.1_Unified_Optimization_Exploration_Mac_QA' in str(x)]
        chosen=(preferred or options)[0]
    print('No running GUI found; launching installed GEM Engine.app: '+str(chosen),flush=True)
    completed=subprocess.run(['open','-a',str(chosen)],capture_output=True,text=True)
    if completed.returncode!=0:
        raise RuntimeError('macOS could not launch GEM Engine.app: '+(completed.stderr or completed.stdout).strip())
    print('Waiting for the original GUI AND integrated Exploration to become ready (up to '+str(wait_seconds)+' seconds).',flush=True)
    start=time.monotonic()
    while time.monotonic()-start < wait_seconds:
        live=live_gui_url(home)
        if live:
            print('Connected to live GEM Engine + Exploration: '+live,flush=True)
            return live,'launched_by_capture'
        time.sleep(2)
    details=diagnostics(home)
    raise RuntimeError('GEM Engine did not become ready after '+str(wait_seconds)+' seconds. '
         'If a macOS setup/permissions dialog opened, complete it and retry. '
         'Logs: ~/Library/Logs/GEM Engine/setup.log and server.log.\n'+details[-4500:])

def browser(pw):
    candidates=['/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
                '/Applications/Chromium.app/Contents/MacOS/Chromium',
                '/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge']
    for candidate in candidates:
        if Path(candidate).is_file():
            return pw.chromium.launch(headless=True,executable_path=candidate,args=['--allow-file-access-from-files'])
    return pw.chromium.launch(headless=True,args=['--allow-file-access-from-files'])

def snap(page, path, *, full=False):
    page.screenshot(path=str(path),full_page=full,animations='disabled')
    if not path.is_file() or path.stat().st_size<3000:raise RuntimeError('Screenshot failed: '+str(path))
    return str(path)



# This is a capture inventory, not a figure compositor. No screenshots are altered.
SECTIONS = {
    'landscape': 'Landscape Generation and common application navigation',
    'optimization': 'Optimization configuration, question definition, and execution workflow',
    'exploration': 'Exploration: setup, activity, calculator, network, questions',
    'results': 'Read-only result browsing and previously generated native viewers',
}

class CaptureInventory:
    def __init__(self, root, url, health, exploration):
        self.root = Path(root)
        self.url = url
        self.entries=[]
        self.errors=[]
        self.health=health
        self.exploration=exploration
        self.expected=[]
        self.shots=self.root/'screenshots'
        for name in SECTIONS:
            (self.shots/name).mkdir(parents=True,exist_ok=True)

    def record(self, category, stem, desc, page, *, target=None, required=False, full_page=False, note=None):
        relative=Path('screenshots')/category/(stem+'.png')
        target_path=self.root/relative
        entry={'section':category,'id':stem,'description':desc,'file':str(relative),
               'source':'previously saved native GUI output' if stem.startswith('native_') else 'running GUI',
               'required':required, 'capture_kind':'element' if target else 'browser viewport',
               'result':'pending'}
        if note:entry['note']=note
        self.expected.append(entry)
        try:
            if target is not None:
                loc=page.locator(target) if isinstance(target,str) else target
                if not loc.count():raise RuntimeError('element missing: '+str(target))
                loc=loc.first
                loc.scroll_into_view_if_needed(timeout=8000)
                page.wait_for_timeout(320)
                loc.screenshot(path=str(target_path),animations='disabled',timeout=20000)
            else:
                page.wait_for_timeout(250)
                page.screenshot(path=str(target_path),full_page=full_page,animations='disabled',timeout=30000)
            if target_path.stat().st_size<3500:raise RuntimeError('unexpectedly small screenshot')
            entry['result']='CAPTURED'
            entry['bytes']=target_path.stat().st_size
            entry['sha256']=sha256(target_path)
            from PIL import Image
            with Image.open(target_path) as im:entry['pixels']=[im.width,im.height]
            self.entries.append(entry)
            print('CAPTURED:',relative,flush=True)
            return True
        except Exception as exc:
            entry['result']='SKIPPED'
            entry['reason']=f'{type(exc).__name__}: {exc}'[:700]
            if target_path.is_file():target_path.unlink()
            self.entries.append(entry)
            self.errors.append(entry)
            print('SKIPPED:',relative,'—',entry['reason'][:150],flush=True)
            return False

    def manifest(self, connection):
        counts={k:{'captured':sum(1 for e in self.entries if e['section']==k and e['result']=='CAPTURED'),
                   'skipped':sum(1 for e in self.entries if e['section']==k and e['result']=='SKIPPED')}
                for k in SECTIONS}
        return {'status':'PASS' if all(counts[k]['captured']>0 for k in ('landscape','optimization','exploration')) else 'INCOMPLETE',
            'type':'unedited native browser screenshot inventory; no final SI figures',
            'captured_at_utc':datetime.now(timezone.utc).isoformat(),
            'gui_server_url':self.url,'connection_method':connection,
            'health':self.health,'exploration_service':self.exploration,
            'no_scientific_jobs_started':True,'no_post_processing':True,
            'screenshots':self.entries,'counts':counts,'suggested_sections':SECTIONS,
            'limitations':['Native result screenshots appear only when existing viewer files can be found.',
              'Unpopulated GUI controls are documented as such; no data or settings are fabricated.',
              'Some advanced GUI options are expanded only; Execute/Run/Save buttons are never used.']}


def sha256(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for c in iter(lambda:f.read(1024*1024),b''):digest.update(c)
    return digest.hexdigest()


def safely_scroll(page, target):
    try:
        loc=page.locator(target).first
        if loc.count():
            loc.scroll_into_view_if_needed(timeout=7000)
            page.wait_for_timeout(300)
            return True
    except Exception:pass
    return False


def visible_text_click(scope, label):
    # Only activates view toggles, never requests analysis, simulation or export.
    queries=[scope.get_by_role('button',name=label,exact=True),
             scope.get_by_text(label,exact=True)]
    for loc in queries:
        try:
            if loc.count():
                loc.first.click(timeout=4000)
                return True
        except Exception:continue
    return False


def capture_main_sections(page, inv, config=None):
    page.goto(inv.url+'/',wait_until='domcontentloaded',timeout=60000)
    page.locator('#serverStatus').wait_for(state='visible',timeout=30000)
    page.wait_for_timeout(800)
    if config:
        # Read an existing config only. Never call Apply/Save/Run.
        page.locator('#configPath').fill(config)
        page.evaluate('(path) => loadConfig(path)',config)
        page.wait_for_timeout(800)
    inv.record('landscape','00_navigation_and_quick_run',
               'Main application navigation, quick-run preparation and status',page,required=True)
    def nav(name):
        page.locator(f'button[data-page="{name}"]').click(timeout=8000)
        page.evaluate('window.scrollTo(0,0)')
        page.wait_for_timeout(750)
    def near(sel):safely_scroll(page,sel)
    nav('dashboard')
    inv.record('landscape','01_quick_run_model_and_sampling',
               'Model upload, simulation count, growth fraction, and run preparation',page,target='#dashboard article.card:first-child')
    near('#dashboard .grid.cards')
    inv.record('landscape','02_progress_controls_and_jobs',
               'Live simulation progress, pause/resume, and active job list',page)
    nav('configuration')
    inv.record('landscape','03_guided_configuration',
               'Editable model, objectives, sampling, solver, and simulation configuration',page,required=True)
    inv.record('landscape','04_guided_settings_detail',
               'Full guided settings panel for landscape generation',page,target='#configuration .grid.two article.card:first-child')
    inv.record('landscape','05_raw_yaml_editor',
               'Native YAML configuration editor and validation controls',page,target='#configuration .grid.two article.card:nth-child(2)')
    near('#landscapeNotice')
    inv.record('landscape','06_medium_design_space',
               'Landscape medium/exchange design table with variable, fixed, and excluded choices',page,target='#configuration > article.card')
    nav('operations')
    inv.record('landscape','07_preflight_and_logs',
               'Preflight, validation, operation status, and live job log viewer',page)
    inv.record('landscape','08_live_log_detail',
               'Recent server/workflow log panel',page,target='#operations article.card:nth-child(2)')
    nav('catalogs')
    inv.record('landscape','09_model_catalogs',
               'Model reaction/exchange catalog browser',page)
    nav('results')
    inv.record('results','01_job_result_browser',
               'Read-only native result browser and selected job view',page)
    # Browsing an existing result only, do not trigger Generate CSV or Start.
    try:
        sel=page.locator('#resultJob')
        vals=sel.locator('option').evaluate_all('ops=>ops.map(x=>({value:x.value,text:x.textContent})).filter(x=>x.value)')
        if vals:
            sel.select_option(vals[0]['value']);page.wait_for_timeout(900)
            inv.record('results','02_existing_job_preview',
                       'Existing job details and exported result thumbnails',page)
    except Exception as exc:print('Existing result could not be selected:',str(exc)[:120])
    near('#csvNotice')
    inv.record('results','03_dataset_export_controls',
               'Full dataset CSV export controls and explanation (not executed)',page)
    near('#resultsBody')
    inv.record('results','04_exact_result_table',
               'Recent exact-FBA result table',page)
    nav('optimization')
    inv.record('optimization','00_three_step_workflow',
               'Optimization workflow navigation with its three stages',page,required=True)
    for stage in ('setup','build','run'):
        page.locator(f'button[data-opt-step="{stage}"]').click(timeout=6000)
        page.evaluate('window.scrollTo(0,0)')
        page.wait_for_timeout(600)
        inv.record('optimization',f'0{1 if stage=="setup" else 3 if stage=="build" else 6}_{stage}_overview',
                   f'Optimization {stage} stage with native controls',page,required=True)
        if stage=='setup':
            inv.record('optimization','02_setup_model_landscape_details',
                       'Model, landscape, YAML configuration and physical design bounds',page,target='#optGeneralPanel')
        elif stage=='build':
            near('#optYaml')
            inv.record('optimization','04_question_yaml_builder',
                       'Optimization question builder, exact objectives/constraints and YAML',page)
            near('#optObjectiveList')
            inv.record('optimization','05_objectives_constraints',
                       'Objective and constraint selection tools',page)
        else:
            near('#optSavedYamlFiles')
            inv.record('optimization','07_saved_questions_selector',
                       'Saved YAML questions and native load controls',page)
            near('#optQueue')
            inv.record('optimization','08_queue_and_algorithms',
                       'Optimizer run queue and algorithm configuration',page)
            near('#optSummary')
            inv.record('optimization','09_optimization_results_panel',
                       'Optimization summary/results display (when existing results are available)',page)
    return page


def capture_exploration(page, inv):
    page.locator('button[data-page="exploration"]').click(timeout=8000)
    page.wait_for_function("() => document.getElementById('gemExplorationFrame').dataset.loaded==='yes'",timeout=50000)
    inner=page.frame_locator('#gemExplorationFrame')
    inner.get_by_text('GEM Engine · Exploration',exact=False).first.wait_for(timeout=45000)
    page.wait_for_timeout(1000)
    def switch(label):
        if not visible_text_click(inner,label):
            raise RuntimeError('Cannot navigate integrated Exploration tab: '+label)
        page.wait_for_timeout(650)
        # Place top of iframe near viewport; actual capture remains the unedited browser image.
        try:page.locator('#gemExplorationFrame').scroll_into_view_if_needed(timeout=8000)
        except Exception:pass
    for label,stem,description in [
      ('Setup','01_setup','Shared source setup: matching GEM model, landscape and configuration'),
      ('Landscape View','02_landscape_activity_builder','Landscape activity question builder for genes, reactions and metabolites'),
      ('Calculator','04_calculator','Native calculator question language, conditions, and visualization configuration'),
      ('Network Visualization','06_network_overview','Network and pathway visualization builder'),
      ('Saved Questions','11_saved_queries','Shared native YAML question pool and existing run results')]:
        switch(label)
        inv.record('exploration',stem,description,page,required=True)
        if label=='Setup':
            inv.record('exploration','01b_source_file_controls',
                       'Existing GEM, landscape and config inputs used by Exploration',page,target='#gemExplorationFrame')
        if label=='Landscape View':
            try:
                inner.get_by_text('Generated YAML',exact=False).first.scroll_into_view_if_needed(timeout=7000)
            except Exception:pass
            inv.record('exploration','03_activity_yaml_and_region_settings',
                       'Angular/circular region count, biological-layer choice and YAML preview',page)
        if label=='Calculator':
            try:inner.get_by_text('Calculator line',exact=False).first.scroll_into_view_if_needed(timeout=7000)
            except Exception:pass
            inv.record('exploration','05_calculator_expression_and_preflight',
                       'Calculator expression editor, evidence controls and preflight (no execution)',page)
            try:
                if visible_text_click(inner,'Calculator examples'):
                    inv.record('exploration','05b_calculator_examples',
                               'Native Calculator language example questions (not executed)',page)
            except Exception:pass
            try:
                if visible_text_click(inner,'Workspace & evidence policy'):
                    inv.record('exploration','05c_calculator_evidence_policy',
                               'Calculator reproducibility, evidence and workspace policies',page)
            except Exception:pass
        if label=='Network Visualization':
            net=inner.locator('#native-network-atlas')
            for txt,stem2,desc in [
              ('Find Routes','07_pathway_route_search','Pathway/route discovery with native evidence-level options'),
              ('Genes','08_gene_knockout_controls','Gene-associated network and knockout controls'),
              ('Network','09_metabolic_network_controls','Reaction/metabolite network visualization controls')]:
                if visible_text_click(net,txt):
                    page.wait_for_timeout(350)
                    inv.record('exploration',stem2,desc,page)
            # Switching to FBA only reveals controls; never click Run Now.
            if visible_text_click(net,'FBA — Standard FBA') or visible_text_click(net,'FBA'):
                inv.record('exploration','10_fba_network_configuration',
                           'Native FBA network settings and objective (not executed)',page)
            try:
                if visible_text_click(net,'Advanced figure and FBA settings'):
                    inv.record('exploration','10b_network_advanced_figure_settings',
                               'Native network graph depth, labeling, flux threshold, and export formats',page)
            except Exception:pass
        if label=='Saved Questions':
            try:
                opt=inner.get_by_text('Advanced · adaptive FBA',exact=False).first
                opt.scroll_into_view_if_needed(timeout=6000)
                opt.click(timeout=3000)
            except Exception:pass
            inv.record('exploration','12_adaptive_qc_and_run_controls',
                       'Saved question run controls, optional adaptive exact-FBA QC policy (not executed)',page)


def existing_viewers(archive):
    """Discover actual completed native GUI viewer pages; never run/rebuild them."""
    for typ in ('Gene','Reaction','Metabolite'):
        root=archive/'Results'/'Exploration'/(typ+'_Activity')/'Seed42_50K'/'question_pool'/'results'
        if not root.is_dir():continue
        candidates=[]
        for q in sorted(root.glob('*/query.yaml')):
            if not (q.parent/'viewer'/'index.html').is_file():continue
            text=q.read_text(encoding='utf-8',errors='replace')
            if ('feature_type: '+typ.lower()) not in text:continue
            import re
            match=re.search(r'^\s*view:\s*(\w+)',text,re.M)
            view=match.group(1) if match else None
            mode=re.search(r'^\s*region_method:\s*(\w+)',text,re.M)
            region=mode.group(1) if mode else ''
            if view in ('overview','regions','feature'):
                candidates.append((view,region,q.parent/'viewer'/'index.html'))
        for view in ('overview','regions','feature'):
            matches=[x for x in candidates if x[0]==view]
            if not matches:continue
            angular=[x for x in matches if x[1]=='angular']
            pick=(angular or matches)[0]
            yield (typ.lower(),view,pick[2])


def capture_existing_viewers(browser_handle,inv,archive):
    count=0
    for typ,view,path in existing_viewers(archive):
        page=browser_handle.new_page(viewport={'width':1520,'height':1120},device_scale_factor=2)
        try:
            page.goto(path.resolve().as_uri(),wait_until='load',timeout=45000)
            try:page.wait_for_function('() => ready === true',timeout=35000)
            except Exception:pass
            page.wait_for_timeout(800)
            label=f'{typ}_{view}'
            # Native saved GUI viewer screenshot, not copy/edit of a PNG plot.
            inv.record('results','native_'+label,
                  f'Previously executed {typ} native Exploration {view} viewer',page,
                  note='Existing run viewer opened read-only. No analysis or FBA started.')
            count+=1
        except Exception as exc:
            print('EXISTING VIEWER SKIPPED:',str(path),'—',str(exc)[:180],flush=True)
        finally:page.close()
    return count


def capture(archive, base, stage, config, connection, include_viewers=True):
    health=url_json(base+'/api/health');exploration=url_json(base+'/api/exploration/status')
    if not exploration.get('available'):
        raise RuntimeError('Integrated Exploration is not ready: '+repr(exploration))
    inv=CaptureInventory(stage,base,health,exploration)
    with sync_playwright() as pw:
        br=browser(pw)
        try:
            page=br.new_page(viewport={'width':W,'height':H},device_scale_factor=2)
            capture_main_sections(page,inv,config)
            capture_exploration(page,inv)
            if include_viewers:capture_existing_viewers(br,inv,archive)
        finally:br.close()
    report=inv.manifest(connection)
    (stage/'CAPTURE_MANIFEST.json').write_text(json.dumps(report,indent=2,default=str)+'\n',encoding='utf-8')
    lines=['# Native GEM Engine GUI screenshot index','','These are unedited native screenshots captured from the actual GUI. No supplementary figures have been assembled yet.','','Capture status: **'+report['status']+'**','']
    for sec,desc in SECTIONS.items():
        lines += ['## '+sec.title(),'_'+desc+'_','']
        for entry in inv.entries:
            if entry['section']!=sec:continue
            lines.append('- '+('**CAPTURED**' if entry['result']=='CAPTURED' else 'SKIPPED')+' — '+entry['description']+' (`'+entry['file']+'`)'+(' — '+entry['reason'] if entry['result']=='SKIPPED' else ''))
        lines.append('')
    (stage/'SCREENSHOT_INDEX.md').write_text('\n'.join(lines),encoding='utf-8')
    if report['status']!='PASS':
        raise RuntimeError('Mandatory interface categories could not all be captured. See CAPTURE_MANIFEST.json in failed stage.')
    return report


def main():
    ap=argparse.ArgumentParser(description='Read-only comprehensive GUI screenshot inventory; no simulations or SI composites')
    ap.add_argument('--archive',type=Path,default=Path.home()/'Desktop/GEM_Publication_Archive')
    ap.add_argument('--url',help='Explicit running GEM Engine URL; otherwise find or launch installed app')
    ap.add_argument('--app',type=Path,help='Exact installed app to launch if no GUI server is running')
    ap.add_argument('--no-launch',action='store_true',help='Require an already running GUI service')
    ap.add_argument('--wait',type=int,default=300)
    ap.add_argument('--config',help='Optional existing GUI YAML configuration path (read-only)')
    ap.add_argument('--no-existing-viewers',action='store_true',help='Skip read-only snapshots of existing completed Exploration outputs')
    args=ap.parse_args()
    archive=args.archive.expanduser().resolve()
    if not archive.is_dir():raise RuntimeError('Archive not found: '+str(archive))
    if not 10<=args.wait<=1800:raise RuntimeError('--wait must be 10–1800 seconds')
    base,connection=ensure_gui_server(base=args.url,app=args.app,wait_seconds=args.wait,launch=not args.no_launch)
    parent=archive/'Figures_and_Tables'/'Supplementary'
    inventory_root=parent/MAIN
    if inventory_root.is_symlink():raise RuntimeError('Unsafe symlinked GUI screenshot inventory root')
    inventory_root.mkdir(parents=True,exist_ok=True)
    target=inventory_root/'current'
    staged=inventory_root/'_STAGING';staged.mkdir(exist_ok=True)
    stage=staged/('capture_'+uuid.uuid4().hex[:8]);stage.mkdir()
    try:
        script_dir=stage/'scripts';script_dir.mkdir()
        for name in ('capture_all_gui.py','CAPTURE_ALL_GUI.command','README.md'):
            shutil.copy2(HERE/name,script_dir/name)
        report=capture(archive,base,stage,args.config,connection,include_viewers=not args.no_existing_viewers)
        if target.exists() and (target.is_symlink() or not target.is_dir()):
            raise RuntimeError('Unsafe existing screenshot inventory directory')
        if target.exists():
            backups=inventory_root/'_BACKUPS';backups.mkdir(exist_ok=True)
            moved=backups/(MAIN+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:6]);target.rename(moved)
        try:stage.rename(target)
        except Exception:
            if 'moved' in locals() and not target.exists():moved.rename(target)
            raise
        print('GUI SCREENSHOT INVENTORY COMPLETE:',target)
        print('Captured:',sum(v['captured'] for v in report['counts'].values()),'images')
        print('INDEX:',target/'SCREENSHOT_INDEX.md')
    except Exception:
        if stage.exists():
            failed=inventory_root/'_FAILED';failed.mkdir(exist_ok=True)
            stage.rename(failed/(stage.name+'_'+uuid.uuid4().hex[:6]))
        raise

if __name__=='__main__':
    try:main()
    except Exception as exc:
        print('FAIL:',type(exc).__name__,str(exc),file=sys.stderr)
        sys.exit(2)
