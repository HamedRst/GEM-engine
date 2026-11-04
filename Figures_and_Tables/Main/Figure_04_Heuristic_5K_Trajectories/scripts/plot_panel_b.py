#!/usr/bin/env python3
"""GEM Figure 4 Panel B: 5K landscape, one displayed seed, 5-problem exact trajectories.

Read-only scientific input. No optimizer/FBA rerun. Panel B consists of five
20-subplot (5 species x 4 columns) figures, one for each problem.

Per-call exact HV = saved calls.csv/archive_hv_after - SAME case/seed 5K frozen
landscape HV, checked against calls.csv/delta_hv_after. Main panels plot seed42
only, intentionally with NO variance ribbons. All 3 seeds in supporting CSVs.

Runtime columns compare measured final 128-call optimizer wall time. Runtime is
NOT extrapolated from checkpoint timestamps, and excludes offline landscape
creation and pre-optimization QC when not included in the saved worker runtime.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, shutil, statistics, sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SEEDS=(42,43,44)
PRIOR=5000
NCALLS=128
SPECIES=('yeast','pputida','cglutamicum','cho','ecoli_antiepex')
SP_LABELS={'yeast':'S. cerevisiae','pputida':'P. putida','cglutamicum':'C. glutamicum','cho':'CHO','ecoli_antiepex':'E. coli'}
PROBLEMS=('constrained_product','growth_product','growth_robustness','product_carbon_oxygen','product_cost')
PROBLEM_LABELS={'constrained_product':'Constrained product', 'growth_product':'Growth × product',
 'growth_robustness':'Growth × robustness', 'product_carbon_oxygen':'Product–carbon–oxygen',
 'product_cost':'Product × cost'}
FAMILIES=('gsMOBO','NSGA-II')
METHODS=('Base','GHOST','SPARC','TRAIL')
COLORS={'Base':'#252525','GHOST':'#0072B2','SPARC':'#009E73','TRAIL':'#E69F00'}
# Only fields representing ACTUAL measured cumulative processing duration; never budgets/limits.
CUM_SEC=('runtime_seconds_cumulative','cumulative_runtime_seconds','cumulative_wall_seconds',
    'worker_wall_seconds_cumulative','wall_seconds_cumulative','elapsed_seconds_cumulative',
    'elapsed_wall_seconds_cumulative',
    'worker_elapsed_seconds_cumulative','cumulative_elapsed_seconds','runtime_cumulative_seconds')
CUM_MIN=('runtime_minutes_cumulative','cumulative_runtime_minutes','cumulative_wall_minutes',
    'elapsed_minutes_cumulative','worker_wall_minutes_cumulative')
TIMING_NAMES=('runtime_checkpoints.csv','runtime_trajectory.csv','elapsed_by_call.csv',
    'time_by_call.csv','timing_checkpoints.csv','runtime_history.csv','runtime_checkpoints.json',
    'runtime_trajectory.json')
HISTORY_COLS=('archive_hv_after','delta_hv_after')

class PanelBError(RuntimeError):pass

def num(v):
    try:
        n=float(v)
        if math.isfinite(n):return n
    except (TypeError,ValueError):pass
    return None

def archive_root(script,override=None):
    if override:
        p=Path(override).expanduser().resolve()
        if not (p/'Results'/'Optimizations').is_dir():raise PanelBError('Missing archive Results/Optimizations at '+str(p))
        return p
    for p in (script.parent,*script.parents):
        if (p/'Results'/'Optimizations').is_dir() and (p/'Figures_and_Tables').is_dir():return p
    raise PanelBError('Could not find GEM_Publication_Archive: use --archive PATH')

def frozen_reader(archive):
    p=archive/'Figures_and_Tables/Main/Figure_03_ThreeSeed_Optimization_Aggregate/scripts/figure_3_seeds.py'
    if not p.is_file() or p.is_symlink():raise PanelBError('Required frozen scientific reader missing: '+str(p))
    spec=importlib.util.spec_from_file_location('gem_publication_frozen_af_panelb',p)
    obj=importlib.util.module_from_spec(spec);spec.loader.exec_module(obj)
    if tuple(obj.SEEDS)!=SEEDS or NCALLS!=obj.CALLS or PRIOR not in obj.SIZES:
        raise PanelBError('Frozen scientific reader version incompatible with this analysis')
    for fam in FAMILIES:
        if set(METHODS)-set(obj.ALGORITHMS[fam]):raise PanelBError('Missing family methods in frozen scientific reader')
    return obj,p

def filehash(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()

def unique_file(job, basename):
    f=job/basename
    if f.is_file() and not f.is_symlink():return f
    science=job/'scientific_results'
    if science.is_dir():
        matches=[p for p in science.rglob(basename) if p.is_file() and not p.is_symlink() and 'surrogate_qc' not in p.parts]
        if len(matches)==1:return matches[0]
        if len(matches)>1:raise PanelBError('Ambiguous '+basename+' for '+str(job))
    return None

def read_csv(path,maxbytes=5000000):
    if not path or not path.is_file() or path.is_symlink() or path.stat().st_size>maxbytes:
        raise PanelBError('CSV missing, unsafe, or oversized: '+str(path))
    with path.open(newline='',encoding='utf-8-sig') as f:
        rd=csv.DictReader(f)
        if not rd.fieldnames:raise PanelBError('No CSV columns '+str(path))
        return list(rd),tuple(rd.fieldnames)

def parse_history(rows, cols, prior_hv, final_hv, path):
    if 'call' not in cols or not ('archive_hv_after' in cols or 'delta_hv_after' in cols):
        raise PanelBError('Exact per-call HV history fields missing in '+str(path))
    pts={0:0.0}
    for r in rows:
        k=num(r.get('call'))
        if k is None or int(k)!=k or int(k)<1 or int(k)>NCALLS:raise PanelBError('Bad call index in '+str(path))
        i=int(k)
        if i in pts:raise PanelBError('Repeated call index '+str(i)+' in '+str(path))
        archive=num(r.get('archive_hv_after')); stored=num(r.get('delta_hv_after'))
        if archive is None and stored is None:raise PanelBError('Missing HV at call '+str(i))
        val=(archive-prior_hv) if archive is not None else stored
        if stored is not None and archive is not None and abs(val-stored)>max(2e-7,1e-6*max(abs(val),abs(stored))):
            raise PanelBError('HV delta disagreement in '+str(path)+' call='+str(i))
        if val < -max(1e-7,prior_hv*1e-6):raise PanelBError('Cumulative landscape union HV below baseline in '+str(path))
        val=max(0.,val)
        pts[i]=val
    if set(pts)!=set(range(0,NCALLS+1)):raise PanelBError('Incomplete 128-call HV history: '+str(path))
    for k in range(1,NCALLS+1):
        if pts[k]+1e-7<pts[k-1]:raise PanelBError('Cumulative ΔHV decreases in '+str(path)+' at '+str(k))
    if abs(pts[NCALLS]+prior_hv-final_hv)>max(2e-6,2e-5*max(final_hv,prior_hv)):
        raise PanelBError('History endpoint != official final HV at '+str(path))
    return pts

def measured_times_from_rows(rows,cols,source):
    """Return 128 complete measured points OR sparse actual checkpoint time points.
    Require an explicit cumulative unit-bearing column; never infer timing.
    """
    cname=next((s for s in CUM_SEC if s in cols),None)
    if cname:factor=1/60
    else:
        cname=next((s for s in CUM_MIN if s in cols),None)
        factor=1
    if not cname:return {},''
    callname=next((s for s in ('call','calls','logical_calls','logical_evaluations','call_index','step') if s in cols),None)
    if not callname:return {},''
    pts={}
    for r in rows:
        k=num(r.get(callname));v=num(r.get(cname))
        if k is None or int(k)!=k or int(k)<1 or int(k)>NCALLS or v is None or v<0:continue
        k=int(k)
        if k in pts:raise PanelBError('Duplicate time checkpoint in '+str(source))
        pts[k]=v*factor
    if len(pts)<2:return {},''
    sortedkeys=sorted(pts)
    if any(pts[b]+1e-7<pts[a] for a,b in zip(sortedkeys,sortedkeys[1:])):
        raise PanelBError('Measured cumulative runtime decreases in '+str(source))
    return pts,cname

def measured_time_points(job,history_rows,history_cols,arch):
    pts,key=measured_times_from_rows(history_rows,history_cols,job/'calls.csv')
    if pts:return pts,'calls.csv',key
    # limited dedicated time logs, exact cumulative fields only
    for name in TIMING_NAMES:
        f=unique_file(job,name)
        if not f:continue
        if f.suffix=='.csv':
            rows,cols=read_csv(f)
        else:
            try: o=json.loads(f.read_text(encoding='utf-8'))
            except (ValueError,OSError):continue
            if isinstance(o,dict):o=o.get('checkpoints',o.get('rows',o.get('history',[])))
            if not isinstance(o,list) or not o or any(not isinstance(x,dict) for x in o):continue
            rows=o;cols=tuple({x for r in rows for x in r})
        pts,key=measured_times_from_rows(rows,cols,f)
        if pts:return pts,f.relative_to(arch).as_posix(),key
    return {},'',''

def checked_runtime(pts,endpoint,job):
    if not pts:return
    if 128 in pts and endpoint is not None:
        if abs(pts[128]-endpoint)>max(.1,endpoint*.07):
            raise PanelBError('Per-call measured runtime vs worker endpoint mismatch: '+str(job))

def read_original(archive, reader):
    camp=reader.campaign_root(archive,42,PRIOR)
    cases=sorted(p.name for p in (camp/'jobs').iterdir() if p.is_dir() and '__' in p.name)
    expected={f'{sp}__{problem}' for sp in SPECIES for problem in PROBLEMS}
    if set(cases)!=expected:raise PanelBError('Archive problem universe unexpected; missing '+repr(expected-set(cases)))
    rows=[];times=[];sources=[];warnings=[]
    for seed in SEEDS:
        camp=reader.campaign_root(archive,seed,PRIOR)
        if not (camp/'CAMPAIGN_COMPLETION.json').is_file():raise PanelBError('Campaign completion missing '+str(camp))
        for case in cases:
            jobland=camp/'jobs'/case/'LANDSCAPE_ONLY'
            base=reader.analyze_job(jobland,'Landscape',archive)
            landscape_hv=base['hv']
            if landscape_hv<=0:raise PanelBError('Nonpositive 5K frozen landscape HV '+str(jobland))
            sp,pr=case.split('__',1)
            for family in FAMILIES:
                for method in METHODS:
                    algo=reader.ALGORITHMS[family][method]
                    job=camp/'jobs'/case/algo
                    if not job.is_dir():raise PanelBError('Archived method missing '+str(job))
                    fin=reader.analyze_job(job,method,archive)
                    cpath=unique_file(job,'calls.csv')
                    hist,columns=read_csv(cpath)
                    hv=parse_history(hist,columns,landscape_hv,fin['hv'],cpath)
                    tpoints,tsource,tcol=measured_time_points(job,hist,columns,archive)
                    runtime=fin['runtime_minutes']
                    checked_runtime(tpoints,runtime,job)
                    tstatus='MEASURED_CUMULATIVE' if tpoints else 'ENDPOINT_ONLY' if runtime is not None else 'MISSING'
                    if tstatus=='MISSING':warnings.append('No timing measurement: '+str(job.relative_to(archive)))
                    if tpoints and len(tpoints)!=128:warnings.append('Sparse measured timing checkpoints: '+str(job.relative_to(archive)))
                    for i in range(NCALLS+1):
                        rows.append(dict(landscape_seed=seed,prior_points=PRIOR,case=case,species=sp,problem=pr,
                            family=family,method=method,algorithm=algo,logical_call=i,exact_delta_hv=hv[i],
                            frozen_5k_hv=landscape_hv,official_final_hv=fin['hv']))
                    for i,t in tpoints.items():
                        times.append(dict(landscape_seed=seed,case=case,species=sp,problem=pr,family=family,method=method,
                            logical_call=i,worker_cumulative_minutes=t,source_kind='MEASURED_CUMULATIVE'))
                    if runtime is not None and 128 not in tpoints:
                        times.append(dict(landscape_seed=seed,case=case,species=sp,problem=pr,family=family,method=method,
                            logical_call=128,worker_cumulative_minutes=runtime,source_kind='MEASURED_ENDPOINT'))
                    sources.append(dict(landscape_seed=seed,case=case,family=family,method=method,algorithm=algo,
                        calls_file=cpath.relative_to(archive).as_posix(),calls_sha256=filehash(cpath),
                        landscape_hv=landscape_hv,final_hv=fin['hv'],final_hv_source=fin['hv_source'],
                        runtime_status=tstatus,runtime_endpoint_minutes=runtime,
                        runtime_endpoint_source=fin['runtime_source'],runtime_call_source=tsource,
                        runtime_call_field=tcol,n_measured_runtime_points=len(tpoints)))
        print(f'Validated source landscape seed {seed}: 25 cases × 8 knowledge-guided algorithms + landscapes',flush=True)
    if len(rows)!=3*25*8*(NCALLS+1):raise PanelBError('Incomplete exact-call data')
    return rows,times,sources,warnings

def seed_statistics(rows,times):
    grouped=defaultdict(dict)
    for r in rows:
        key=(r['species'],r['problem'],r['family'],r['method'],r['logical_call'])
        if r['landscape_seed'] in grouped[key]:raise PanelBError('Duplicate seed values')
        grouped[key][r['landscape_seed']]=r['exact_delta_hv']
    agg=[]
    for k,values in grouped.items():
        if set(values)!=set(SEEDS):raise PanelBError('Missing ΔHV seed '+str(k))
        vv=[values[s] for s in SEEDS]
        agg.append(dict(zip(('species','problem','family','method','logical_call'),k),
            mean_exact_delta_hv=statistics.mean(vv),sd_across_3_seeds=statistics.stdev(vv),n_seeds=3))
    timed=defaultdict(dict)
    for r in times:
        key=(r['species'],r['problem'],r['family'],r['method'],r['logical_call'])
        if r['landscape_seed'] in timed[key]:raise PanelBError('Duplicate runtime for seed '+str(key))
        timed[key][r['landscape_seed']]=r['worker_cumulative_minutes']
    runtime=[]
    for k,values in timed.items():
        # ONLY plot intermediate means when all three seeds have timestamps at SAME call.
        if set(values)!=set(SEEDS):continue
        vv=[values[s] for s in SEEDS]
        runtime.append(dict(zip(('species','problem','family','method','logical_call'),k),
            mean_worker_minutes=statistics.mean(vv),sd_across_3_seeds=statistics.stdev(vv),n_seeds=3,
            runtime_plot_kind='MEASURED_CURVE' if k[-1]!=128 else 'MEASURED_ENDPOINT'))
    return agg,runtime

def matplotlib_lib():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    return plt,Line2D

def method_label(family, method):
    """Family-explicit algorithm name; never label the native parent as Base."""
    if method == 'Base':
        return f'{family} + landscape'
    return f'{family} + {method}'


def plot_five(raw, times, outdir, plot_seed=42):
    """Five 5×4 Panel-B figures; one predeclared seed, no seed ribbons.

    Exact per-call ΔHV on cols 1/3; measured FULL 128-call worker time on
    cols 2/4. Runtime bars are not checkpoint reconstructions.
    """
    if plot_seed not in SEEDS:
        raise PanelBError('Selected landscape seed must be 42, 43 or 44')
    plt, Line2D = matplotlib_lib()
    from matplotlib.patches import Patch
    hv={(r['species'],r['problem'],r['family'],r['method'],r['logical_call']):r['exact_delta_hv']
        for r in raw if r['landscape_seed']==plot_seed}
    tm={(r['species'],r['problem'],r['family'],r['method']):r['worker_cumulative_minutes']
        for r in times if r['landscape_seed']==plot_seed and r['logical_call']==NCALLS}
    if len(hv)!=25*8*(NCALLS+1):
        raise PanelBError('Selected seed has incomplete 128-call exact HV trajectories')
    if len(tm)!=25*8 or any(not (math.isfinite(v) and v > 0) for v in tm.values()):
        raise PanelBError('Selected seed must have 200 positive, measured 128-call runtimes')
    saved=[]
    for problem in PROBLEMS:
        fig,axes=plt.subplots(5,4,figsize=(15.3,11.0),sharex=False)
        fig.patch.set_facecolor('white')
        fig.text(.020,.973,'B',fontsize=16,weight='bold',ha='left',va='top',color='#151515')
        fig.suptitle(f'5K landscape  |  {PROBLEM_LABELS[problem]}  |  seed {plot_seed}',
            x=.530,y=.981,fontsize=12,weight='semibold',color='#222222')
        heads=('gsMOBO family · exact ΔHV','gsMOBO family · total runtime',
               'NSGA-II family · exact ΔHV','NSGA-II family · total runtime')
        for j,h in enumerate(heads):
            axes[0,j].set_title(h,fontsize=10,weight='semibold',pad=9,color='#222222')
        for i,sp in enumerate(SPECIES):
            for j,(fam,runtime_panel) in enumerate((('gsMOBO',False),('gsMOBO',True),('NSGA-II',False),('NSGA-II',True))):
                ax=axes[i,j]
                ax.set_facecolor('white')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                for edge in ('left','bottom'):
                    ax.spines[edge].set_color('#626262')
                    ax.spines[edge].set_linewidth(.65)
                ax.grid(False)
                ax.tick_params(labelsize=7.1,direction='out',length=2,colors='#363636',pad=2)
                if j==0:
                    ax.annotate(SP_LABELS[sp],xy=(0,.5),xycoords='axes fraction',
                        xytext=(-49,0),textcoords='offset points',rotation=90,
                        ha='center',va='center',fontsize=9.0,weight='semibold',color='#292929')
                if not runtime_panel:
                    for method in METHODS:
                        y=[hv[(sp,problem,fam,method,k)] for k in range(NCALLS+1)]
                        ax.step(range(NCALLS+1),y,where='post',color=COLORS[method],
                            lw=1.45 if method=='Base' else 1.55,
                            label=method_label(fam,method),zorder=2)
                    ax.set_xlim(0,NCALLS)
                    ax.set_xticks((0,32,64,96,128))
                    ymax=max(hv[(sp,problem,fam,m,128)] for m in METHODS)
                    ax.set_ylim(0,max(1e-6,ymax*1.14))
                    if i==len(SPECIES)-1:
                        ax.set_xlabel('Logical exact FBA calls',fontsize=7.7)
                    else:
                        ax.tick_params(labelbottom=False)
                    if j in (0,2):ax.set_ylabel('Exact ΔHV',fontsize=7.7)
                    handles=[Line2D([0],[0],color=COLORS[m],lw=1.7,label=method_label(fam,m)) for m in METHODS]
                else:
                    values=[tm[(sp,problem,fam,m)] for m in METHODS]
                    ax.bar(range(4),values,width=.68,
                        color=[COLORS[m] for m in METHODS],edgecolor='#4a4a4a',
                        linewidth=.30,zorder=2)
                    ax.set_xlim(-.55,3.55)
                    # Reserve upper area for a compact in-plot legend.
                    ax.set_ylim(0,max(values)*1.48)
                    ax.set_xticks(range(4))
                    if i==len(SPECIES)-1:
                        ax.set_xticklabels(('Landscape','GHOST','SPARC','TRAIL'),fontsize=6.3,
                            rotation=28,ha='right')
                        ax.set_xlabel('Algorithm',fontsize=7.7)
                    else:ax.set_xticklabels([])
                    if j in (1,3):ax.set_ylabel('Runtime (min)',fontsize=7.7)
                    handles=[Patch(facecolor=COLORS[m],edgecolor='none',label=method_label(fam,m)) for m in METHODS]
                # Repeated inside the *actual* individual axes as requested.
                lg=ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(.014,.992),
                    fontsize=6.0,ncol=1,frameon=True,framealpha=.91,facecolor='white',
                    edgecolor='none',handlelength=1.5,handletextpad=.35,labelspacing=.15,
                    borderpad=.25,columnspacing=.4)
                lg.set_zorder(6)
        # Compact grid; no large inter-subplot whitespace or repeated external legend.
        fig.subplots_adjust(left=.090,right=.987,top=.914,bottom=.093,wspace=.22,hspace=.25)
        stem=f'Figure_4B_{problem}_5K_seed{plot_seed}'
        for ext in ('pdf','svg','png'):
            kwargs={'dpi':300} if ext=='png' else {}
            fig.savefig(outdir/(stem+'.'+ext),bbox_inches='tight',facecolor='white',**kwargs)
        plt.close(fig)
        saved.append(stem)
    return saved


def write_csv(p,rows):
    if not rows:
        p.write_text('');return
    p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(rows[0]);
    with p.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def generate(archive,out,plot_seed=42):
    reader,reader_path=frozen_reader(archive)
    raw, times, sources, warnings=read_original(archive,reader)
    agg,rt=seed_statistics(raw,times)
    out.mkdir(parents=True,exist_ok=False)
    images=plot_five(raw,times,out,plot_seed)
    data=out/'plot_data';data.mkdir()
    write_csv(data/'CASE_SEED_CALL_EXACT_DELTA_HV.csv',raw)
    write_csv(data/'THREE_SEED_HV_MEAN_SD.csv',sorted(agg,key=lambda x:(x['problem'],x['species'],x['family'],x['method'],x['logical_call'])))
    write_csv(data/'MEASURED_RUNTIME_CHECKPOINTS.csv',times)
    write_csv(data/'THREE_SEED_MEASURED_RUNTIME_MEAN_SD.csv',rt)
    write_csv(data/'PROVENANCE_AND_TIMING_AUDIT.csv',sources)
    report={'status':'PASS' if all(x['runtime_status']!='MISSING' for x in sources) else 'FAIL_MISSING_RUNTIME',
        'n_input_5k_runs':len(sources),'seed_values':SEEDS,'plotted_seed':plot_seed,'n_cases':25,'n_figures':len(images),
        'n_HV_call_samples':len(raw),'n_runtime_checkpoint_samples':len(times),
        'timing_coverage':{name:sum(x['runtime_status']==name for x in sources) for name in ('MEASURED_CUMULATIVE','ENDPOINT_ONLY','MISSING')},
        'scientific_reader':reader_path.relative_to(archive).as_posix(),
        'scientific_reader_sha256':filehash(reader_path),
        'delta_hv_definition':'official archive_hv_after at each call - frozen same-seed 5K landscape HV',
        'variance_definition':'3-seed mean and SD exported to CSV only; main figure shows selected seed without SD bands',
        'runtime_definition':'measured optimizer worker wall time at 128 logical exact-FBA calls; bar height in minutes; excludes any unrecorded separate QC/landscape preparation',
        'algorithm_definitions':{fam:{m:reader.ALGORITHMS[fam][m] for m in METHODS} for fam in FAMILIES},
        'plot_files':images,'warnings':warnings[:25]}
    (out/'RUN_REPORT.json').write_text(json.dumps(report,indent=2,default=list)+'\n')
    (out/'README_METRICS.txt').write_text(
        f'Figure 4 Panel B | five 20-subpanel versions, single predeclared landscape seed {plot_seed}.\n'
        'Row = species; columns = gsMOBO exact delta HV, gsMOBO worker runtime, '
        'NSGA-II exact delta HV, NSGA-II worker runtime.\n'
        'ΔHV = saved archive_hv_after at each exact call minus SAME seed/5K frozen landscape HV.\n'
        'Runtime bars = saved measured optimizer worker wall minutes at logical call 128; no checkpoint mtime inference.\n'
        'No runtime trajectory fabricated when call-associated timestamps are absent.\n'
        'No-prior algorithms excluded; baseline WITH 5K knowledge + GHOST/SPARC/TRAIL.\n'
        'Three-seed HV mean/SD and timing audits preserved as SUPPORTING CSV ONLY.\n'
        'Main figure selects seed 42 before inspecting results; change with --plot-seed, '
        'but do not cherry-pick publication seed.\n')
    return report

def main(argv=None):
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--archive',type=Path,default=None)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--plot-seed',type=int,default=42,choices=SEEDS)
    args=ap.parse_args(argv)
    arc=archive_root(Path(__file__).resolve(),args.archive)
    if args.out.exists():raise PanelBError('Refusing to overwrite an existing output: '+str(args.out))
    report=generate(arc,args.out,args.plot_seed)
    print('PASS:',args.out,flush=True)
    print('Figure versions:',report['n_figures'],'| Source jobs:',report['n_input_5k_runs'],flush=True)
    print('Runtime coverage:',report['timing_coverage'],flush=True)
    print('Runtime columns: measured final 128-call optimizer wall-time bars (no checkpoint reconstruction).',flush=True)
    return 0
if __name__=='__main__':
    try:sys.exit(main())
    except Exception as exc:
        print('ERROR:',str(exc),file=sys.stderr);sys.exit(2)
