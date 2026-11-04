#!/usr/bin/env python3
"""Stage → verify → atomically replace only GEM Figure 4.

The former active folder is MOVED to a timestamped backup outside Main/Figure4.
Figure 2/3 and all source data are read-only. No duplicates within active Figure4.
"""
from __future__ import annotations
import argparse, json, shutil, subprocess, sys, uuid
from pathlib import Path
from datetime import datetime, timezone
SRC=Path(__file__).resolve().parent
DEST='Figure_04_Heuristic_5K_Trajectories'
PROBLEMS=('constrained_product','growth_product','growth_robustness','product_carbon_oxygen','product_cost')
OWNER='GEM_FIGURE4_PANEL_B_FINAL_RUNTIME_V5'

def archive_find():return Path.home()/'Desktop/GEM_Publication_Archive'

def plot_python(arc):
    py=arc/'Figures_and_Tables/Main/Figure_03_ThreeSeed_Optimization_Aggregate/scripts/.plot_venv/bin/python'
    if not py.is_file():py=Path(sys.executable)
    subprocess.run([str(py),'-c','import matplotlib,numpy; print("Plot environment:",matplotlib.__version__)'],check=True)
    return py

def install(arc,script_dir=SRC,plot_seed=42):
    arc=Path(arc).expanduser().resolve();script_dir=Path(script_dir).resolve()
    main=arc/'Figures_and_Tables/Main'
    if not main.is_dir() or not (arc/'Results/Optimizations').is_dir():
        raise RuntimeError('Expected GEM publication archive missing: '+str(arc))
    dest=main/DEST
    backdir=arc/'Figures_and_Tables/_REPLACED_FIGURE_04_BACKUPS'
    py=plot_python(arc)
    stage=main/('.'+DEST+'.staging_'+uuid.uuid4().hex[:9]);stage.mkdir()
    moved=None
    try:
        scripts=stage/'scripts';scripts.mkdir()
        for fn in ('plot_panel_b.py','install.py','RUN_PANEL_B.command','README.md'):
            shutil.copy2(script_dir/fn,scripts/fn)
        (stage/'FIGURE4_GENERATOR_OWNERSHIP.json').write_text(json.dumps({'owner':OWNER,'plotted_seed':plot_seed})+'\n')
        cmd=[str(py),str(scripts/'plot_panel_b.py'),'--archive',str(arc),
             '--out',str(stage/'outputs'),'--plot-seed',str(plot_seed)]
        with (stage/'GENERATION_LOG.txt').open('w') as log:
            proc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,text=True)
        if proc.returncode!=0:
            fail=backdir/('FAILED_FIGURE4_STAGE_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:5]);backdir.mkdir(parents=True,exist_ok=True)
            stage.rename(fail)
            raise RuntimeError('Figure4 generation failed; existing folder untouched. Log: '+str(fail/'GENERATION_LOG.txt'))
        outputs=stage/'outputs'
        report=json.loads((outputs/'RUN_REPORT.json').read_text())
        if (report.get('status')!='PASS' or report.get('n_figures')!=5 or
            report.get('n_input_5k_runs')!=600 or report.get('plotted_seed')!=plot_seed or
            report.get('timing_coverage',{}).get('MISSING')!=0):
            raise RuntimeError('Stage coverage/method/runtime validation failed')
        for pr in PROBLEMS:
            for ext in ('pdf','png','svg'):
                f=outputs/f'Figure_4B_{pr}_5K_seed{plot_seed}.{ext}'
                if not f.is_file() or f.stat().st_size<500:
                    raise RuntimeError('Missing or empty figure: '+str(f))
        backdir.mkdir(parents=True,exist_ok=True)
        if dest.exists() or dest.is_symlink():
            if not dest.is_dir() or dest.is_symlink():
                raise RuntimeError('Unsafe existing Figure4 destination: '+str(dest))
            moved=backdir/(DEST+'_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'_'+uuid.uuid4().hex[:7])
            dest.rename(moved)
        try:
            stage.rename(dest)
        except BaseException:
            if moved is not None and moved.exists() and not dest.exists():moved.rename(dest)
            raise
        print('PASS: Figure 4 replaced with five final-runtime-bar figures')
        print('ACTIVE FIGURE 4:',dest)
        print('OLD FIGURE 4 BACKUP:',moved if moved else 'none')
        print('PLOTTED SEED:',plot_seed)
        print('SOURCE RESULTS, FIGURES 2 AND 3: untouched')
        return dest
    finally:
        if stage.is_dir() and stage.name.startswith('.'+DEST+'.staging_'):
            shutil.rmtree(stage)

def main():
    a=argparse.ArgumentParser()
    a.add_argument('--archive',type=Path,default=archive_find())
    a.add_argument('--plot-seed',type=int,choices=(42,43,44),default=42)
    ns=a.parse_args();install(ns.archive,plot_seed=ns.plot_seed)
if __name__=='__main__':main()
