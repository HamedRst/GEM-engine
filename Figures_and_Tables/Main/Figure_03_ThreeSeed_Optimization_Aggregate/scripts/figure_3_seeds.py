#!/usr/bin/env python3
"""GEM paper Figure 3 (three seeds). Read-only, archive-relative, exact 128-call endpoints.

Adapted to the established GEM publication figure v31 visual convention:
  (A,D) stacked fractional wins, 25 cases per prior/seed;
  (B,E) 100 * official endpoint HV / frozen LANDSCAPE_ONLY HV;
  (C,F) worker-wall runtime in minutes, replacing frontier-extension bars.

Three seeds: winner bars = arithmetic mean of three 25-case win allocations; NO errors.
HV/runtime bars = mean of three per-seed case means, errors = SD of three seed means.
Fail-closed on missing/colliding 128-call data. No FBA, algorithm, or model runs.
"""
from __future__ import annotations
import argparse, csv, json, math, os, re, statistics, sys, traceback
from pathlib import Path
from collections import defaultdict

SIZES=(5000,10000,25000,50000)
SEEDS=(42,43,44)
CASES=25
CALLS=128
TIE_TOL=1e-12
METHODS=('Landscape','No prior','Base','GHOST','SPARC','TRAIL')
OPT_METHODS=METHODS[1:]
RUNTIME_METHODS=OPT_METHODS
ALGORITHMS={
'gsMOBO':{'No prior':'BASE_GSMOBO_QPAREGO_NO50K','Base':'BASE_GSMOBO_QPAREGO_50K',
'GHOST':'DEV_GSMOBO50K_GHOST_LEGACY_VB','SPARC':'DEV_GSMOBO50K_SPARC_A',
'TRAIL':'DEV_GSMOBO50K_TRAIL_TRUE_NATIVE_V111'},
'NSGA-II':{'No prior':'BASE_NSGA2_NO50K','Base':'BASE_NSGA2_50K',
'GHOST':'DEV_NSGA50K_GHOST_A','SPARC':'DEV_NSGA50K_SPARC_LEGACY_OPERATOR',
'TRAIL':'DEV_NSGA50K_TRAIL_LEGACY_OPERATOR'}}
FAMILIES=('gsMOBO','NSGA-II')
EXPECTED_ALGOS={'LANDSCAPE_ONLY',*(x for d in ALGORITHMS.values() for x in d.values())}
COLORS={'Landscape':'#BDBDBD','No prior':'#6A3D9A','Base':'#222222',
'GHOST':'#0072B2','SPARC':'#E69F00','TRAIL':'#009E73'}

class FigureInputError(RuntimeError):pass

def archive_from_script(script:Path):
    """Do not resolve input paths outside the movable archive tree."""
    for p in (script.parent,*script.parents):
        if (p/'Results'/'Optimizations').is_dir() and (p/'Figures_and_Tables').is_dir():
            return p
    raise FigureInputError('Move this installed folder inside an archive containing Results/Optimizations and Figures_and_Tables')

def campaign_name(seed,size):
    k=size//1000
    if seed==42:return f'GEM_ALL25_FINAL10_PRIOR{k}K_128'
    return f'GEM_ALL25_FINAL10_PRIOR{k}K_LS{seed}_AS{ {43:20260823,44:20260824}[seed] }_128'  # 43->20260823, 44->20260824

def campaign_root(archive,seed,size):
    expected=campaign_name(seed,size)
    p=archive/'Results'/'Optimizations'/f'landscape_seed_{seed}'/f'prior_{size}'/expected
    if not p.is_dir() or p.is_symlink():
        raise FigureInputError('Missing archived canonical campaign: '+p.relative_to(archive).as_posix())
    return p

def safe_json(path):
    if not path.is_file() or path.is_symlink() or path.stat().st_size>15_000_000:return None
    try:return json.loads(path.read_text(encoding='utf8'))
    except (UnicodeError,ValueError,OSError):return None

def walk_obj(obj, stem='', depth=0):
    if depth>8:return
    if isinstance(obj,dict):
        for k,v in obj.items():
            yield from walk_obj(v, (stem+'.'+str(k)).strip('.'), depth+1)
    elif isinstance(obj,list):
        if len(obj)<10:
            for i,v in enumerate(obj):yield from walk_obj(v,stem+f'[{i}]',depth+1)
    else:yield (stem,obj)

def as_num(x):
    if isinstance(x,bool) or x is None:return None
    try:
        n=float(x)
        return n if math.isfinite(n) else None
    except (TypeError,ValueError):return None

def key_last(path):
    return path.split('.')[-1].lower().replace('-','_')

def collect_json(job):
    """Only completed scientific_result payloads; job.json (configuration) is never a score."""
    science=job/'scientific_results'
    candidates=[]
    if science.is_dir():
        for base,dirs,files in os.walk(science,followlinks=False):
            dirs[:]=[d for d in dirs if not (Path(base)/d).is_symlink() and d not in {'surrogate_qc','images','plots','models'}]
            for f in files:
                p=Path(base)/f
                if p.suffix.lower()=='.json' and not p.is_symlink() and p.stat().st_size<15_000_000:
                    candidates.append(p)
    # Some runners keep authoritative final/result/performance at job root.
    for f in ('result.json','results.json','summary.json','final_summary.json','metrics.json',
              'run_metadata.json','performance.json','runtime.json','job_status.json','status.json'):
        p=job/f
        if p.is_file() and not p.is_symlink():candidates.append(p)
    # Re-read only tiny candidates. Never read giant CSV trajectories.
    records=[]
    for p in sorted(set(candidates),key=lambda z:str(z)):
        data=safe_json(p)
        if data is not None: records.append((p,data))
    return records

def metadata_value(records,names,allowed_prefix=None):
    hits=[]
    for path,ob in records:
        for key,val in walk_obj(ob):
            if key_last(key) in names and (n:=as_num(val)) is not None:
                if allowed_prefix and not allowed_prefix(path,key):continue
                hits.append((path,key,n))
    return hits

def authoritative_hv(records,method):
    """Prefer official final_hv; accept explicitly tracked no-prior self-front override."""
    def good(p,k):
        k=k.lower(); pname=p.name.lower()
        if 'history' in k or '[' in k or 'trajectory' in k or 'candidate' in k:return False
        if ('run' in k and 'summary' not in k) and 'baseline' in k:return False
        if 'calls' in pname or 'trajectory' in pname or 'checkpoint' in pname:return False
        return True
    pri=[]
    if method=='No prior':pri=[{'final_self_hv'}, {'final_hv'}, {'final_archive_hv'}]
    elif method=='Landscape':pri=[{'final_hv'}, {'landscape_hv'},{'final_archive_hv'}]
    else:pri=[{'final_hv'}, {'final_archive_hv'}]
    for j,keys in enumerate(pri):
        hits=metadata_value(records,keys,good)
        if hits:
            vals={round(v,12) for _,_,v in hits}
            if len(vals)>1:
                raise FigureInputError('Conflicting '+str(keys)+' endpoint values: '+', '.join(f'{p.name}:{k}={v}' for p,k,v in hits[:7]))
            return hits[0][2],hits[0][0],hits[0][1],('NO_PRIOR_FALLBACK_UNVERIFIED' if method=='No prior' and j>0 else 'PRIMARY')
    return None,None,None,'MISSING'

def calls_value(records,method):
    # Prefer top-level job summary calls or the official endpoint fields.
    hits=[]
    for p,ob in records:
        if not isinstance(ob,dict):continue
        for key in ('calls','calls_completed','n_calls','optimizer_calls','calls_used','evaluations'):
            if key in ob and as_num(ob[key]) is not None:hits.append((p,key,as_num(ob[key])))
    if hits:
        vals={int(v) for _,_,v in hits}
        if len(vals)>1:raise FigureInputError('Conflicting call counts: '+str(hits[:5]))
        return int(hits[0][2])
    return None

def collect_status(job,records):
    # Markers are advisory; require final summary + 128 calls separately.
    statuses=[]
    for p,ob in records:
        if not isinstance(ob,dict):continue
        for k in ('status','state','job_status'):
            if k in ob and isinstance(ob[k],str):
                statuses.append((p.name,ob[k].upper().strip()))
    for p in (job/'status.json',job/'job_status.json'):
        x=safe_json(p)
        if isinstance(x,dict):
            for k in ('status','state','job_status'):
                if isinstance(x.get(k),str):statuses.append((p.name,x[k].upper().strip()))
    return statuses

def official_runtime(records,job):
    """Only measured worker wall-clock. Reject configuration timeouts, naive mtimes, and generation timing."""
    keys={'worker_wall_seconds','worker_wall_sec','worker_wall_s','worker_elapsed_seconds',
          'worker_elapsed_sec','wall_time_seconds','elapsed_wall_seconds','elapsed_wall_sec',
          'runtime_seconds','runtime_sec','total_runtime_sec','wall_seconds', 'wall_time_sec',
          'wall_time_s','elapsed_seconds','elapsed_sec','wall_runtime_sec',
          'worker_wall_time_seconds','worker_wall_time_sec','wall_clock_seconds','wall_clock_sec',
          'walltime_seconds','walltime_sec','seconds_elapsed','elapsed_wall_time_sec'}
    key_min={'worker_wall_minutes','wall_minutes','runtime_minutes','elapsed_wall_minutes'}
    exclude={'timeout','limit','estimated','generation','budget','prediction','per_call','mean','average','prior'}
    candidates=[]
    for p,ob in records:
        for k,v in walk_obj(ob):
            last=key_last(k)
            if last not in keys|key_min:continue
            if '[' in k or any(t in k.lower() for t in exclude):continue
            if (n:=as_num(v)) is not None and n>=0:
                # Favor explicit worker metrics; reject multiple contradictory similarly named timings.
                priority=(4 if last.startswith('worker_') else 3 if 'wall' in last else 2)
                candidates.append((priority,p,k,n*(60 if last in key_min else 1)))
    # Plain text from measured run metadata allowed only on exact key names.
    for base in (job,job/'scientific_results'):
        for fn in ('run_metadata.txt','performance.txt','runtime.txt','worker_time.txt'):
            p=base/fn
            if p.is_file() and not p.is_symlink() and p.stat().st_size<200000:
                content=p.read_text(encoding='utf8',errors='replace')
                for line in content.splitlines():
                    m=re.fullmatch(r'\s*([\w.]+)\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)\s*',line)
                    if m and m.group(1).lower() in keys|key_min:
                        key=m.group(1).lower(); n=float(m.group(2))*(60 if key in key_min else 1)
                        candidates.append((4,p,key,n))
    if not candidates:return None,None,None
    best=max(priority for priority,*_ in candidates)
    chosen=[(p,k,v) for priority,p,k,v in candidates if priority==best]
    vals=[v for _,_,v in chosen]
    if max(vals)-min(vals)>max(0.05,0.01*max(vals)):
        raise FigureInputError('Conflicting measured runtime fields: '+str([(p.name,k,v) for p,k,v in chosen[:7]]))
    p,k,v=chosen[0]
    return v/60,p,k

def analyze_job(job,method,archive):
    rec=collect_json(job)
    if not rec:raise FigureInputError('No readable scientific-result summary JSON under '+job.relative_to(archive).as_posix())
    hv,p,key,note=authoritative_hv(rec,method)
    if hv is None:raise FigureInputError('No official final HV summary: '+job.relative_to(archive).as_posix())
    if hv<0:raise FigureInputError('Negative HV in '+str(job))
    calls=calls_value(rec,method)
    expected=0 if method=='Landscape' else CALLS
    # Some M5 summaries omit top-level call count; use known completion marker + final calls.csv audit.
    if calls is None:
        csvp=job/'scientific_results'/'calls.csv'
        if not csvp.is_file():
            csvps=list((job/'scientific_results').rglob('calls.csv')) if (job/'scientific_results').is_dir() else []
            csvp=csvps[0] if len(csvps)==1 else csvp
        if csvp.is_file() and not csvp.is_symlink() and method!='Landscape':
            with csvp.open(newline='',encoding='utf8') as fp:
                rd=csv.DictReader(fp)
                possible=[r for r in rd if any(as_num(r.get(k))==128 for k in ('call','calls','call_index','step','eval'))]
            if possible:calls=128
    # Some LANDSCAPE_ONLY rows have no calls field; zero-call established by exact algorithm identity.
    if method=='Landscape' and calls is None:calls=0
    if calls != expected:
        raise FigureInputError(f'Expected {expected} calls, found {calls!r} at '+job.relative_to(archive).as_posix())
    statuses=collect_status(job,rec)
    blocking=[(src,st) for src,st in statuses if st in {'FAILED','RUNNING','QUEUED','PENDING','ERROR','CANCELLED','TIMED_OUT','INCOMPLETE'}]
    if blocking:raise FigureInputError(f'Nonfinal status {blocking}: '+str(job))
    runtime,rpath,rkey=(None,None,None) if method=='Landscape' else official_runtime(rec,job)
    return {'hv':hv,'runtime_minutes':runtime,'hv_source':p.relative_to(archive).as_posix(),
            'hv_key':key,'hv_quality':note,'runtime_source':rpath.relative_to(archive).as_posix() if rpath else '',
            'runtime_key':rkey or '', 'calls':calls}

def load_data(archive):
    data={}; origin=[]; problems=[]; cases_by_campaign={}; runtime_unavailable=[]
    for seed in SEEDS:
        for size in SIZES:
            camp=campaign_root(archive,seed,size)
            marker=camp/'CAMPAIGN_COMPLETION.json'
            if not marker.is_file() or safe_json(marker) is None:
                raise FigureInputError('Missing valid CAMPAIGN_COMPLETION.json: '+camp.relative_to(archive).as_posix())
            jobs=camp/'jobs'
            if not jobs.is_dir():raise FigureInputError('Missing jobs/ directory: '+str(jobs))
            case_dirs=sorted(p for p in jobs.iterdir() if p.is_dir() and not p.is_symlink() and '__' in p.name)
            if len(case_dirs)!=CASES:raise FigureInputError(f'{seed}/{size}: expected 25 case folders, found {len(case_dirs)}')
            cases_by_campaign[(seed,size)]={p.name for p in case_dirs}
            for case in case_dirs:
                expected=EXPECTED_ALGOS
                actual={p.name for p in case.iterdir() if p.is_dir() and not p.is_symlink()}
                if not expected.issubset(actual):
                    raise FigureInputError(f'{seed}/{size} {case.name}: missing algorithm jobs {sorted(expected-actual)}')
                for algo in sorted(expected):
                    method='Landscape' if algo=='LANDSCAPE_ONLY' else next(m for fam,d in ALGORITHMS.items() for m,a in d.items() if a==algo)
                    fam='Landscape' if algo=='LANDSCAPE_ONLY' else next(fam for fam,d in ALGORITHMS.items() if algo in d.values())
                    key=(seed,size,case.name,fam,method)
                    job=case/algo
                    datum=analyze_job(job,method,archive)
                    data[key]=datum
                    origin.append({'landscape_seed':seed,'prior_points':size,'case':case.name,'family':fam,
                                   'method':method,'algorithm':algo,**datum})
                    if datum['runtime_minutes'] is None and method!='Landscape':
                        runtime_unavailable.append((seed,size,case.name,fam,method))
            print(f'Validated seed {seed}, prior {size//1000}K: 25 cases, 275 final jobs',flush=True)
    # Fail closed if any campaign uses a different 25-case universe.
    standard=next(iter(cases_by_campaign.values()))
    for s,kases in cases_by_campaign.items():
        if kases!=standard:raise FigureInputError(f'Nonidentical case sets across campaigns: {s} missing={standard-kases} extra={kases-standard}')
    return data,origin,sorted(standard),runtime_unavailable

def summaries(data,cases):
    win_rows=[]; metric_rows=[]; audit=[]
    for seed in SEEDS:
        for size in SIZES:
            for fam in FAMILIES:
                wins={m:0.0 for m in METHODS}; values={m:[] for m in METHODS}
                runtimes={m:{} for m in OPT_METHODS}
                for case in cases:
                    base=data[(seed,size,case,'Landscape','Landscape')]['hv']
                    if base<=0:raise FigureInputError(f'Nonpositive frozen reference {seed}/{size}/{case}: {base}')
                    scores={'Landscape':base}
                    for m in OPT_METHODS:
                        datum=data[(seed,size,case,fam,m)]; scores[m]=datum['hv']
                        runtimes[m][case]=datum['runtime_minutes']
                    best=max(scores.values()); ties=[m for m,v in scores.items() if abs(v-best)<=TIE_TOL]
                    for m in ties:wins[m]+=1/len(ties)
                    audit.append(dict(landscape_seed=seed,prior_points=size,family=fam,case=case,
                                      winning_methods='|'.join(ties),share_each=1/len(ties),winner_hv=best))
                    for m,v in scores.items():values[m].append(100*v/base)
                if not math.isclose(sum(wins.values()),25,abs_tol=1e-8):raise FigureInputError('Win-sum !=25')
                for m in METHODS:
                    win_rows.append(dict(seed=seed,prior_points=size,family=fam,method=m,wins=wins[m]))
                    metric_rows.append(dict(seed=seed,prior_points=size,family=fam,method=m,
                                            normalized_hv_pct=statistics.mean(values[m]),
                                            runtime_minutes=None,runtime_n=0))
    # Cross-panel identical runtime cohort: cases with reliable runtime for EVERY method,
    # BOTH families, all sizes and seeds. Avoid accidental unmatched comparisons.
    common_runtime=[case for case in cases if all(data[(s,p,case,fam,m)]['runtime_minutes'] is not None
                 for s in SEEDS for p in SIZES for fam in FAMILIES for m in OPT_METHODS)]
    if len(common_runtime)<20:
        raise FigureInputError(f'Only {len(common_runtime)}/25 cases have fully matched measured worker wall runtimes across all seeds, sizes, families, and methods. No runtime means invented. Check input manifest.')
    index={(r['seed'],r['prior_points'],r['family'],r['method']):r for r in metric_rows}
    for s in SEEDS:
        for p in SIZES:
            for fam in FAMILIES:
                for m in OPT_METHODS:
                    vals=[data[(s,p,case,fam,m)]['runtime_minutes'] for case in common_runtime]
                    r=index[(s,p,fam,m)];r['runtime_minutes']=statistics.mean(vals);r['runtime_n']=len(vals)
    aggregated=[]
    for p in SIZES:
        for fam in FAMILIES:
            for m in METHODS:
                ws=[r['wins'] for r in win_rows if r['prior_points']==p and r['family']==fam and r['method']==m]
                ms=[r for r in metric_rows if r['prior_points']==p and r['family']==fam and r['method']==m]
                if len(ws)!=3 or len(ms)!=3:raise FigureInputError('Missing seed component')
                hv=[r['normalized_hv_pct'] for r in ms]
                times=[r['runtime_minutes'] for r in ms]
                aggregated.append(dict(prior_points=p,family=fam,method=m,
                    avg_wins=statistics.mean(ws), # NO WIN SD in figure
                    hv_mean=statistics.mean(hv),hv_sd=statistics.stdev(hv),
                    runtime_mean=statistics.mean(times) if m!='Landscape' else None,
                    runtime_sd=statistics.stdev(times) if m!='Landscape' else None,
                    n_seeds=3,win_case_n=25,runtime_case_n=len(common_runtime)))
            win_total=sum(r['avg_wins'] for r in aggregated if r['prior_points']==p and r['family']==fam)
            if not math.isclose(win_total,25,abs_tol=1e-9):raise FigureInputError(f'Average win total = {win_total}')
    return win_rows,metric_rows,aggregated,audit,common_runtime

def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def plot(agg,out):
    try:
        import numpy as np
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:raise FigureInputError('Python needs numpy and matplotlib in your environment: '+str(exc))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10.5,
      'axes.labelsize':9.5,'axes.linewidth':0.8,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
      'legend.fontsize':7.6,'legend.frameon':False,'figure.dpi':180,'savefig.dpi':600,
      'pdf.fonttype':42,'ps.fonttype':42})
    def get(fam,p,m):return next(r for r in agg if r['family']==fam and r['prior_points']==p and r['method']==m)
    def clean_axis(ax):
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out',length=3,width=0.7)
        ax.grid(axis='y',color='#E7E7E7',linewidth=.6,zorder=0)
    def label(ax,txt):ax.text(-.13,1.08,txt,transform=ax.transAxes,ha='left',va='bottom',fontsize=11.5,fontweight='bold',clip_on=False)
    fig,axes=plt.subplots(2,3,figsize=(10.9,6.4))
    x=np.arange(len(SIZES),dtype=float)
    all_hv=[(r['hv_mean']-r['hv_sd'],r['hv_mean']+r['hv_sd']) for r in agg]
    hmin=min(a for a,_ in all_hv);hmax=max(b for _,b in all_hv);hspan=max(1,hmax-hmin)
    hvlim=(min(hmin,100)-.06*hspan,max(hmax,100)+.06*hspan)
    rmax=max(r['runtime_mean']+r['runtime_sd'] for r in agg if r['method']!='Landscape')
    for row,fam in enumerate(FAMILIES):
        ax=axes[row,0];bottom=np.zeros(len(SIZES))
        for m in METHODS:
            heights=np.array([get(fam,p,m)['avg_wins'] for p in SIZES]);
            ax.bar(x,heights,bottom=bottom,width=.66,color=COLORS[m],edgecolor='white',linewidth=.35,zorder=3)
            for xi,yi,bi in zip(x,heights,bottom):
                if yi>=1:
                    txt=str(round(float(yi))) if math.isclose(float(yi),round(float(yi)),abs_tol=1e-9) else f'{yi:.1f}'
                    ax.text(xi,bi+yi/2,txt,ha='center',va='center',fontsize=7,fontweight='bold',color='black' if m=='Landscape' else 'white')
            bottom+=heights
        for xi,b in zip(x,bottom):ax.text(xi,b+.35,f'{b:.0f}',ha='center',va='bottom',fontsize=7.2,fontweight='bold')
        ax.set_ylim(0,27);ax.set_ylabel('Wins (of 25)');ax.set_title('Case wins',fontweight='bold',pad=6)
        ax.set_xticks(x);ax.set_xticklabels([f'{p//1000}K' for p in SIZES]);clean_axis(ax);label(ax,'A' if row==0 else 'D')
        ax=axes[row,1];width=.125;offsets=(np.arange(len(METHODS))-(len(METHODS)-1)/2)*width
        for m,o in zip(METHODS,offsets):
            heights=np.array([get(fam,p,m)['hv_mean'] for p in SIZES]);sd=np.array([get(fam,p,m)['hv_sd'] for p in SIZES])
            ax.bar(x+o,heights,width=width*.92,color=COLORS[m],zorder=3)
            ax.errorbar(x+o,heights,yerr=sd,fmt='none',color='#333333',elinewidth=.65,capsize=1.4,capthick=.65,zorder=4)
        ax.axhline(100,color='#888888',linestyle='--',linewidth=.8,zorder=1)
        ax.set_xticks(x);ax.set_xticklabels([f'{p//1000}K' for p in SIZES]);ax.set_ylabel('Final HV (% of own landscape)')
        ax.set_title('Normalized final HV',fontweight='bold',pad=6);ax.set_ylim(*hvlim);clean_axis(ax);label(ax,'B' if row==0 else 'E')
        ax=axes[row,2];width=.145;offsets=(np.arange(len(RUNTIME_METHODS))-(len(RUNTIME_METHODS)-1)/2)*width
        for m,o in zip(RUNTIME_METHODS,offsets):
            heights=np.array([get(fam,p,m)['runtime_mean'] for p in SIZES]);sd=np.array([get(fam,p,m)['runtime_sd'] for p in SIZES]);
            ax.bar(x+o,heights,width=width*.92,color=COLORS[m],zorder=3)
            ax.errorbar(x+o,heights,yerr=sd,fmt='none',color='#333333',elinewidth=.65,capsize=1.4,capthick=.65,zorder=4)
        ax.set_xticks(x);ax.set_xticklabels([f'{p//1000}K' for p in SIZES]);ax.set_ylabel('Worker wall time (min)')
        ax.set_title('Optimization runtime',fontweight='bold',pad=6);ax.set_ylim(0,rmax*1.10 if rmax>0 else 1)
        clean_axis(ax);label(ax,'C' if row==0 else 'F')
        axes[row,1].text(.5,1.14,fam,transform=axes[row,1].transAxes,ha='center',va='bottom',fontsize=13,fontweight='bold')
    legend_labels={'Landscape':'Landscape','No prior':'Base optimizer','Base':'Base optimizer + landscape knowledge','GHOST':'GHOST','SPARC':'SPARC','TRAIL':'TRAIL'}
    fig.legend(handles=[Patch(facecolor=COLORS[m],edgecolor='none',label=legend_labels[m]) for m in METHODS],
       loc='lower center',bbox_to_anchor=(.5,.018),ncol=6,frameon=False,columnspacing=1.05,handlelength=1.4)
    fig.subplots_adjust(left=.07,right=.99,top=.92,bottom=.11,hspace=.42,wspace=.34)
    for ext in ('png','pdf','svg'):
        fig.savefig(out/f'Figure_3_combined_families_3seeds.{ext}',bbox_inches='tight',**({'dpi':600} if ext=='png' else {}))
    plt.close(fig)

def run(archive,figure_dir,allow_quiet=False):
    data,origin,cases,missing=load_data(archive)
    wins,seedstats,aggregate,audit,cohort=summaries(data,cases)
    output=figure_dir/'outputs';plots=output/'plot_data';plots.mkdir(parents=True,exist_ok=True)
    write_csv(plots/'INPUT_SOURCE_AND_METRIC_AUDIT.csv',origin)
    write_csv(plots/'SEED_LEVEL_WINS.csv',wins)
    write_csv(plots/'SEED_LEVEL_HV_AND_RUNTIME.csv',seedstats)
    write_csv(plots/'AGGREGATE_THREE_SEEDS.csv',aggregate)
    write_csv(plots/'WINNER_TIE_AUDIT.csv',audit)
    report=['GEM PUBLICATION FIGURE 3 — THREE SEEDS','='*68,
      'Source: 12 canonical, archived M5 optimization campaigns (relative to archive root)',
      'All seeds 42,43,44; sizes 5K/10K/25K/50K; 25 biological cases; 10 methods + LANDSCAPE_ONLY',
      'Endpoint: final official hypervolume at 128 calls for optimization; 0-call LANDSCAPE_ONLY baseline',
      'Winner: exact tie tolerance 1e-12, fractional allocation within six methods, per seed + size + case + family',
      'Winner panel: MEAN of the three per-seed 25-case winner counts, NO error bars',
      'HV panel: mean and SD of the three SEED-LEVEL means of 100 x (method_HV / own frozen landscape_HV)',
      'Runtime panel: mean and SD of SEED-LEVEL mean measured worker wall-minutes on a matched cohort.',
      'Landscape excluded from runtime because it is a zero-call baseline, not an optimizer.',
      f'Matched runtime case cohort n={len(cohort)}/25; excluded: {sorted(set(cases)-set(cohort))}',
      f'Missing individual runtime values in input: {len(missing)}',
      'Relative reproducibility: all scientific inputs resolved from Results/Optimizations UNDER this archive.',
      'WARN: no-prior final_hv fallback, if present, is marked explicitly in INPUT_SOURCE_AND_METRIC_AUDIT.csv.',
      'Validation: 12 campaigns x 275 complete = 3300 jobs, expected 3000 optimization jobs at 128 calls.',
      'NOTE: no source results altered; graphics exported PNG/PDF/SVG.',
      'Runtime values are NOT inferred from campaign completion dates or landscape-generation timing.',
      'Software: Python '+sys.version.split()[0]]
    try:
        import matplotlib,numpy
        report.append('Software: matplotlib '+matplotlib.__version__+'; numpy '+numpy.__version__)
    except ImportError:pass
    (output/'RUN_REPORT.txt').write_text('\n'.join(report)+'\n',encoding='utf8')
    plot(aggregate,output)
    print('\n'.join(report))
    print('PASS: '+str((output/'Figure_3_combined_families_3seeds.pdf').relative_to(archive)))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive',type=Path,default=None,help='Optional root of moved/copied archive; defaults to script-relative autodetection')
    ap.add_argument('--figure-dir',type=Path,default=None,help='Optional installed Figure folder (must be within archive)')
    args=ap.parse_args()
    archive=args.archive.resolve() if args.archive else archive_from_script(Path(__file__).resolve())
    figure_dir=args.figure_dir.resolve() if args.figure_dir else Path(__file__).resolve().parent.parent
    if archive not in figure_dir.parents:raise FigureInputError('Figure output must remain within selected archive')
    try:run(archive,figure_dir)
    except (FigureInputError,OSError,ValueError,KeyError,TypeError) as exc:
        print('FIGURE BUILD BLOCKED SAFELY: '+str(exc),file=sys.stderr)
        # Safe diagnosis in figure folder; never produce partial misleading plot.
        path=figure_dir/'FIGURE_BUILD_DIAGNOSTIC.txt';path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(str(exc)+'\n',encoding='utf8');sys.exit(2)
if __name__=='__main__':main()
