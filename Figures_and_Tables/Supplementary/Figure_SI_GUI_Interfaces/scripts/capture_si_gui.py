#!/usr/bin/env python3
"""Capture actual running GEM Engine browser UI, never execute scientific jobs.

All outputs are confined to one archive Supplementary figure folder. Screenshots
are real Playwright browser captures, composites contain only screenshot pixels
plus panel labels; no figure contents, plots, or data are generated separately.
"""
from __future__ import annotations
import argparse, hashlib, json, os, shutil, sys, time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import sync_playwright

HERE=Path(__file__).resolve().parent
MAIN='Figure_SI_GUI_Interfaces'
W,H=1800,1120

def url_json(url):
    with urlopen(url,timeout=15) as r: return json.load(r)

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

def composition(pairs,out):
    # Pure screenshot assembly with panel labels in a separate title band.
    imgs=[]
    for letter,path in pairs:
        im=Image.open(path).convert('RGB');w,h=im.size
        # Crop only empty vertical extent; use native screenshot pixels unchanged otherwise.
        crop_h=min(h,2100)
        im=im.crop((0,0,w,crop_h))
        imgs.append((letter,im))
    target_w=2500;label_h=73;margin=28;gap=30;scaled=[]
    for letter,im in imgs:
        new_h=round(im.height*target_w/im.width)
        scaled.append((letter,im.resize((target_w,new_h),resample=Image.Resampling.LANCZOS)))
    total_h=margin*2+sum(im.height+label_h for _,im in scaled)+gap*(len(scaled)-1)
    final=Image.new('RGB',(target_w+2*margin,total_h),'white');draw=ImageDraw.Draw(final)
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial Bold.ttf',35)
    except OSError:
        try:font=ImageFont.truetype('/Library/Fonts/Arial.ttf',35)
        except OSError:font=ImageFont.load_default()
    y=margin
    for letter,im in scaled:
        draw.text((margin+8,y+13),letter,fill='#1b3246',font=font)
        y+=label_h;final.paste(im,(margin,y));y+=im.height+gap
    final.save(out,optimize=True)

def capture(archive,url,out,config,exploration_required=True):
    source=url_json(url+'/api/health')
    exp=url_json(url+'/api/exploration/status')
    if exploration_required and not exp.get('available'):
        raise RuntimeError('Integrated Exploration not available: '+str(exp)+'. Start the actual GEM Engine.app first.')
    shots=out/'screenshots';shots.mkdir(parents=True,exist_ok=True)
    manifest={'captured_at_utc':datetime.now(timezone.utc).isoformat(),
       'gui_source':url,'server_health':source,'exploration_status':exp,
       'mode':'REAL LOCAL GEM ENGINE GUI SCREENSHOTS; no scientific jobs started',
       'screenshots':{}}
    with sync_playwright() as pw:
        b=browser(pw)
        try:
            page=b.new_page(viewport={'width':W,'height':H},device_scale_factor=2)
            page.goto(url+'/',wait_until='domcontentloaded',timeout=60000)
            page.locator('#serverStatus').wait_for(state='visible',timeout=30000)
            page.wait_for_timeout(1400)
            if config:
                page.locator('#configPath').fill(config)
                page.evaluate('(path) => loadConfig(path)',config)
                page.wait_for_timeout(900)
            def save(key,file):
                manifest['screenshots'][key]=str(Path('screenshots')/file)
                return snap(page,shots/file)
            page.locator('button[data-page="dashboard"]').click()
            page.wait_for_timeout(280)
            save('landscape_quick','landscape_01_quick_run.png')
            page.locator('button[data-page="configuration"]').click()
            page.wait_for_timeout(1000)
            save('landscape_configuration','landscape_02_configuration.png')
            # Save a separate scroll/canvas view of the native medium setup; no modifications.
            target=page.locator('#landscapeBody')
            if target.count() and target.locator('tr').count()>0:
                page.locator('#landscapeNotice').scroll_into_view_if_needed(timeout=7000)
                page.wait_for_timeout(300)
                save('landscape_medium','landscape_03_medium_design.png')
            page.locator('button[data-page="optimization"]').click()
            page.wait_for_timeout(800)
            for stage,filename in [('setup','optimization_01_setup.png'),('build','optimization_02_question.png'),('run','optimization_03_run.png')]:
                page.locator(f'[data-opt-step="{stage}"]').click()
                page.wait_for_timeout(700)
                save('optimization_'+stage,filename)
            page.locator('button[data-page="exploration"]').click()
            page.wait_for_function("() => document.getElementById('gemExplorationFrame').dataset.loaded==='yes'",timeout=45000)
            frame=page.frame_locator('#gemExplorationFrame')
            frame.get_by_text('GEM Engine · Exploration',exact=False).first.wait_for(timeout=45000)
            page.wait_for_timeout(1400)
            for name,key,filename in [('Setup','setup','exploration_01_setup.png'),
                                       ('Landscape View','landscape','exploration_02_landscape_view.png'),
                                       ('Calculator','calculator','exploration_03_calculator.png'),
                                       ('Network Visualization','network','exploration_04_network.png'),
                                       ('Saved Questions','questions','exploration_05_saved_questions.png')]:
                # Click the actual integrated Exploration control, never synthesize its data.
                frame.get_by_role('button',name=name,exact=True).first.click(timeout=20000)
                page.wait_for_timeout(650)
                save('exploration_'+key,filename)
            # Three SI figures: each is an assembly of native captures.
            figures=out/'figures';figures.mkdir(parents=True,exist_ok=True)
            composition([('A · Quick run',shots/'landscape_01_quick_run.png'),
                         ('B · Model and landscape configuration',shots/'landscape_02_configuration.png')],
                        figures/'SI_GUI_01_Landscape_Generation.png')
            composition([('A · Setup',shots/'optimization_01_setup.png'),
                         ('B · Build problem',shots/'optimization_02_question.png'),
                         ('C · Saved questions and run',shots/'optimization_03_run.png')],
                        figures/'SI_GUI_02_Optimization.png')
            composition([('A · Landscape View',shots/'exploration_02_landscape_view.png'),
                         ('B · Calculator',shots/'exploration_03_calculator.png'),
                         ('C · Network Visualization',shots/'exploration_04_network.png')],
                        figures/'SI_GUI_03_Exploration.png')
            manifest['figures']=[f'figures/{p.name}' for p in sorted(figures.glob('*.png'))]
        finally:b.close()
    (out/'CAPTURE_MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print('PASS: 3 SI interface figures created, source screenshots untouched:')
    for name in manifest.get('figures',[]):print(' ',out/name)
    return manifest

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,default=Path.home()/'Desktop/GEM_Publication_Archive')
    ap.add_argument('--url',help='Running GEM Engine base URL, default derived from local Mac app port file')
    ap.add_argument('--config',help='Optional existing GUI config path, e.g. configs/my_model.yaml')
    args=ap.parse_args()
    archive=args.archive.expanduser().resolve()
    if not archive.is_dir():raise RuntimeError('Archive not found: '+str(archive))
    port_path=Path.home()/'Library/Application Support/GEM Engine/server.port'
    if args.url:base=args.url.rstrip('/')
    elif port_path.is_file():base='http://127.0.0.1:'+port_path.read_text().strip()
    else:raise RuntimeError('Start GEM Engine.app first, or provide --url http://127.0.0.1:PORT')
    out=archive/'Figures_and_Tables'/'Supplementary'/MAIN
    out.mkdir(parents=True,exist_ok=True)
    scripts=out/'scripts';scripts.mkdir(exist_ok=True)
    shutil.copy2(HERE/'capture_si_gui.py',scripts/'capture_si_gui.py') if (HERE/'capture_si_gui.py').resolve()!=(scripts/'capture_si_gui.py').resolve() else None
    capture(archive,base,out,args.config)

if __name__=='__main__':
    try: main()
    except Exception as e:
        print('FAIL:',type(e).__name__,str(e),file=sys.stderr)
        sys.exit(2)
