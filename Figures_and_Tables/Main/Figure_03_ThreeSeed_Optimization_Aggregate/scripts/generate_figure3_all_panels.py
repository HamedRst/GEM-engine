#!/usr/bin/env python3
"""Unified Figure 3 A-I: same science as frozen A-F + QC-alone 5-method G-I.

One entry point; external original scripts act only as frozen scientific readers.
No exact FBA, model training on held-out cases, QC regeneration, optimizer reruns,
or modification to prior figures, sources, or existing figures.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import statistics
import sys
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

SEEDS=(42,43,44)
SIZES=(5000,10000,25000,50000)
FIVE=('QC-only selector','gsMOBO + SPARC','NSGA-II + SPARC',
      'gsMOBO (with knowledge)','NSGA-II (with knowledge)')
FIVE_COLORS={
    'QC-only selector':'#6A3D9A',
    'gsMOBO + SPARC':'#E69F00',
    'NSGA-II + SPARC':'#009E73',
    'gsMOBO (with knowledge)':'#0072B2',
    'NSGA-II (with knowledge)':'#222222',
}
TIE_TOL=1e-12
NAME='Figure_3_All_Nine_Panels_ThreeSeeds_V4'  # Stable filename: replace existing figure instead of duplicating

class FigureError(RuntimeError):pass

def find_archive(path):
    for p in (path.parent,*path.parents):
        if (p/'Results/Optimizations').is_dir() and (p/'Figures_and_Tables').is_dir(): return p
    raise FigureError('Cannot locate movable GEM_Publication_Archive containing Results/Optimizations')

def load_source(path,module_name):
    if not path.is_file() or path.is_symlink(): raise FigureError('Missing frozen scientific module: '+str(path))
    spec=importlib.util.spec_from_file_location(module_name,path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def modules(archive):
    root=archive/'Figures_and_Tables'/'Main'
    old_af=root/'Figure_03_ThreeSeed_Optimization_Aggregate'/'scripts'/'figure_3_seeds.py'
    old_ghi=root/'Figure_03_GHI_QC_Selector'/'scripts'/'selector_ghi.py'
    af=load_source(old_af,'gem_frozen_figure3_AF')
    qc=load_source(old_ghi,'gem_frozen_figure3_QC_ALONE')
    for key,expected in [('ALPHA',6.0),('TARGET_CLIP',20.0),('SWITCH_MARGIN',0.5)]:
        if getattr(qc,key,None)!=expected: raise FigureError(f'Frozen QC model changed {key}')
    if tuple(af.SEEDS)!=SEEDS or tuple(af.SIZES)!=SIZES or tuple(qc.SEEDS)!=SEEDS or tuple(qc.SIZES)!=SIZES:
        raise FigureError('Frozen source seeds/sizes inconsistent with published protocol')
    if getattr(af,'CALLS',None)!=128: raise FigureError('Frozen optimization calls !=128')
    if getattr(af,'TIE_TOL',None)!=TIE_TOL: raise FigureError('Frozen A-F win tolerance changed')
    required_colors=set(FIVE_COLORS.values())
    if not required_colors.issubset(set(af.COLORS.values())):
        raise FigureError('Five-method colors do not match original A-F palette')
    return af,qc

def measured_qc_overhead(path,archive,expected):
    """Only accept original, source-hash-verified pre-run timing evidence.

    Missing timing => optimizer-only runtime in Panel I; never invent a QC cost.
    """
    if not path.is_file():return None,'QC computation overhead not measured; I reports measured optimizer runtime only.'
    required={'landscape_seed','prior_points','case','qc_preprocessing_seconds',
              'selector_decision_seconds','source_file','source_sha256','timing_kind'}
    timings={};provenance={}
    with path.open(newline='',encoding='utf8') as f:
        reader=csv.DictReader(f)
        if not required.issubset(set(reader.fieldnames or [])):
            raise FigureError('Measured QC timing table lacks required columns')
        for line,r in enumerate(reader,2):
            try:
                key=int(r['landscape_seed']),int(r['prior_points']),r['case'].strip()
                qc=float(r['qc_preprocessing_seconds']);decision=float(r['selector_decision_seconds'])
            except (ValueError,KeyError,TypeError) as e:
                raise FigureError(f'Malformed QC timing line {line}: {e}')
            if key in timings or not all(math.isfinite(x) and x>=0 for x in (qc,decision)):
                raise FigureError(f'Duplicate/negative/nonfinite QC overhead line {line}')
            if r['timing_kind']!='MEASURED_QC_COMPUTATION':
                raise FigureError('Timing source must be original measured QC computation')
            rel=r['source_file'].strip();sha=r['source_sha256'].strip().lower()
            if not rel or Path(rel).is_absolute() or '..' in Path(rel).parts or len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha):
                raise FigureError('Invalid QC timing source path/SHA256')
            timings[key]=(qc+decision)/60
            provenance[rel]=sha
    if set(timings)!=expected:
        raise FigureError('Measured QC overhead missing expected 300 rows: '+str(sorted(expected-set(timings))[:5]))
    for rel,sha in provenance.items():
        p=archive/rel
        if not p.is_file() or p.is_symlink():raise FigureError('Timing evidence file missing '+rel)
        hasher=hashlib.sha256()
        with p.open('rb') as h:
            for chunk in iter(lambda:h.read(1<<20),b''):hasher.update(chunk)
        if hasher.hexdigest()!=sha:raise FigureError('Timing evidence SHA mismatch '+rel)
    return timings,'Preoptimization QC computation and selector decision overhead measured and included in Panel I.'

def quintet_for(r,data):
    key=r['landscape_seed'],r['prior_points'],r['case']
    lookup={
        'gsMOBO + SPARC':('gsMOBO','SPARC'),
        'NSGA-II + SPARC':('NSGA-II','SPARC'),
        'gsMOBO (with knowledge)':('gsMOBO','Base'),
        'NSGA-II (with knowledge)':('NSGA-II','Base'),
    }
    vals={'QC-only selector': {'hv':r['selected_hv'],'runtime':r['selected_runtime_min']}}
    for name,(fam,variant) in lookup.items():
        datum=data[(*key,fam,variant)]
        vals[name]={'hv':datum['hv'],'runtime':datum['runtime_minutes']}
    for m,v in vals.items():
        if not math.isfinite(v['hv']) or v['hv']<0:
            raise FigureError(f'Invalid five-method HV: {key}/{m}')
    chosen=r['selected_method']
    if chosen not in ('gsMOBO + SPARC','NSGA-II + SPARC'):
        raise FigureError('QC selector chose unsupported method: '+str(chosen))
    if not math.isclose(vals['QC-only selector']['hv'],vals[chosen]['hv'],abs_tol=1e-12,rel_tol=0):
        raise FigureError('QC selector HV does not match chosen SPARC outcome')
    return vals

def five_method_statistics(predictions,data,overhead=None):
    """G = POOLED top-result percentage; exact ties get FULL credit, never 1/n.

    H = 100*(final HV - frozen landscape HV) / frozen landscape HV.
    I = source-measured worker wall minutes + only measured QC overhead, if available.
    """
    grouped=defaultdict(list)
    all_casewise=[]
    for r in predictions:
        k=(r['landscape_seed'],r['prior_points']);grouped[k].append(r)
        v=quintet_for(r,data)
        base=r['landscape_hv']
        if not math.isfinite(base) or base<=0:raise FigureError('Invalid frozen landscape HV')
        best=max(x['hv'] for x in v.values())
        leaders=[name for name in FIVE if abs(v[name]['hv']-best)<=TIE_TOL]
        for name in FIVE:
            item=v[name]
            key=(r['landscape_seed'],r['prior_points'],r['case'])
            add=overhead[key] if overhead is not None and name=='QC-only selector' else 0.0
            all_casewise.append({'landscape_seed':r['landscape_seed'],'prior_points':r['prior_points'],
                'case':r['case'],'method':name,'method_hv':item['hv'],'frozen_landscape_hv':base,
                'delta_hv':item['hv']-base,'delta_hv_pct':100*(item['hv']-base)/base,
                'optimizer_runtime_min':item['runtime'],'measured_qc_selector_overhead_min':add if (overhead is not None or name!='QC-only selector') else '',
                'top_result':int(name in leaders),'co_winners':'|'.join(leaders)})
    # Exact biologic coverage for every seed and size; optional 3 outliers excluded.
    expected_len=297 if len(predictions)==297 else 300 if len(predictions)==300 else None
    if expected_len is None:raise FigureError(f'Unexpected primary/sensitivity sample count: {len(predictions)}')
    for seed in SEEDS:
        for size in SIZES:
            expected=24 if expected_len==297 and size==5000 else 25
            cases=[r['case'] for r in grouped[seed,size]]
            if len(cases)!=expected or len(set(cases))!=expected:
                raise FigureError(f'Unexpected cohort at {seed}/{size}: {len(cases)}')
    runtime_by_case=defaultdict(dict)
    for r in all_casewise:
        runtime_by_case[(r['landscape_seed'],r['prior_points'],r['case'])][r['method']]=r['optimizer_runtime_min']
    distinct_cases={r['case'] for r in predictions}
    # Same fully matched cases across 3 seeds, 4 sizes, and every five-method runtime.
    common=[]
    for case in sorted(distinct_cases):
        if all((vals:=runtime_by_case.get((seed,size,case))) is not None
               and len(vals)==len(FIVE)
               and all(v is not None and math.isfinite(v) and v>=0 for v in vals.values())
               for seed in SEEDS for size in SIZES):common.append(case)
    if len(common)<20:raise FigureError(f'Only {len(common)} fully matched five-method runtime cases')
    runtimes=defaultdict(list)
    gains=defaultdict(list)
    full_win_counts=defaultdict(int)
    n_by_size=defaultdict(int)
    for r in all_casewise:
        seed,size,case,method=r['landscape_seed'],r['prior_points'],r['case'],r['method']
        k=(seed,size,method)
        gains[k].append(r['delta_hv_pct'])
        full_win_counts[(size,method)]+=r['top_result']
        if case in common:
            extra=overhead[(seed,size,case)] if overhead is not None and method=='QC-only selector' else 0.0
            runtimes[k].append(r['optimizer_runtime_min']+extra)
    for seed in SEEDS:
        for size in SIZES:n_by_size[size]+=len(grouped[seed,size])
    seed_summary=[]
    for size in SIZES:
        for seed in SEEDS:
            for method in FIVE:
                key=(seed,size,method)
                g=gains[key];t=runtimes[key]
                if not g or len(t)!=len(common):raise FigureError('Empty gains or unmatched runtime cohort')
                seed_summary.append({'landscape_seed':seed,'prior_points':size,'method':method,
                   'n_hv':len(g),'n_runtime':len(t),'delta_hv_mean_pct':statistics.mean(g),
                   'runtime_mean_min':statistics.mean(t),
                   'top_result_count':sum(r['top_result'] for r in all_casewise if r['landscape_seed']==seed and r['prior_points']==size and r['method']==method)})
    grouped_seed={(r['landscape_seed'],r['prior_points'],r['method']):r for r in seed_summary}
    summary=[]
    for size in SIZES:
        for method in FIVE:
            rec=[grouped_seed[(seed,size,method)] for seed in SEEDS]
            n=n_by_size[size]
            top_count=full_win_counts[(size,method)]
            if not 0<=top_count<=n:raise FigureError('Top-case count exceeds number of observations')
            summary.append({'prior_points':size,'method':method,'n_pooled_cases':n,
               'top_case_count':top_count,'top_result_cases_pct':100*top_count/n,
               'delta_hv_mean_pct':statistics.mean(r['delta_hv_mean_pct'] for r in rec),
               'delta_hv_sd_pct':statistics.stdev(r['delta_hv_mean_pct'] for r in rec),
               'runtime_mean_min':statistics.mean(r['runtime_mean_min'] for r in rec),
               'runtime_sd_min':statistics.stdev(r['runtime_mean_min'] for r in rec)})
    return all_casewise,seed_summary,summary,common

def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def draw(af,sel,af_agg,ghi_agg,destination,overhead_verified):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10.5,
        'axes.labelsize':9,'axes.linewidth':0.8,'xtick.labelsize':8.2,'ytick.labelsize':8.2,
        'legend.fontsize':8,'legend.frameon':False,'figure.dpi':180,
        'savefig.dpi':600,'pdf.fonttype':42,'ps.fonttype':42})
    fig=plt.figure(figsize=(13.35,11.4))
    grid=fig.add_gridspec(5,3,left=.069,right=.989,top=.967,bottom=.034,
                          height_ratios=[1,1,.16,1,.16],hspace=.54,wspace=.38)
    axes=[[fig.add_subplot(grid[row,j]) for j in range(3)] for row in (0,1,3)]
    top_legend_ax=fig.add_subplot(grid[2,:]);bottom_legend_ax=fig.add_subplot(grid[4,:])
    top_legend_ax.axis('off');bottom_legend_ax.axis('off')
    x=np.arange(4,dtype=float)
    def style(ax,label):
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.tick_params(direction='out',length=3,width=.7)
        ax.grid(axis='y',color='#E7E7E7',linewidth=.6,zorder=0)
        ax.set_xticks(x);ax.set_xticklabels(['5K','10K','25K','50K'])
        ax.text(-.12,1.075,label,transform=ax.transAxes,fontsize=11.5,
                fontweight='bold',ha='left',va='bottom',clip_on=False)
    # Presentation-only palette for panels A-F. Do not mutate the frozen source
    # module or the separate five-method G-I palette.
    af_plot_colors=dict(af.COLORS)
    af_plot_colors.update({'SPARC':'#009E73', 'TRAIL':'#F0B400'})
    fams=('gsMOBO','NSGA-II')
    lookup={(r['family'],r['prior_points'],r['method']):r for r in af_agg}
    lookup5={(r['prior_points'],r['method']):r for r in ghi_agg}
    all_hv=[(r['hv_mean']-r['hv_sd'],r['hv_mean']+r['hv_sd']) for r in af_agg]
    hv_low=min(a for a,_ in all_hv);hv_high=max(b for _,b in all_hv)
    hv_span=max(1,hv_high-hv_low)
    hv_ylim=(min(100,hv_low)-.06*hv_span,max(100,hv_high)+.06*hv_span)
    runtime_ceiling=max([r['runtime_mean']+r['runtime_sd'] for r in af_agg if r['method']!='Landscape'] +
                         [r['runtime_mean_min']+r['runtime_sd_min'] for r in ghi_agg])
    runtime_ylim=max(1,runtime_ceiling*1.105)
    for ri,fam in enumerate(fams):
        # A/D: each 25-case stack is sorted independently, bottom -> top.
        # Fractional win counts, ties, and scientific source values remain unchanged.
        ax=axes[ri][0];bottom=np.zeros(4)
        for j,size in enumerate(SIZES):
            vals={m:float(lookup[(fam,size,m)]['avg_wins']) for m in af.METHODS}
            ordered=sorted(af.METHODS,key=lambda m:(vals[m],af.METHODS.index(m)))
            height_bottom=0.0
            for m in ordered:
                height=vals[m]
                ax.bar(x[j],height,bottom=height_bottom,width=.66,
                       color=af_plot_colors[m],edgecolor='white',linewidth=.35,zorder=3)
                if height>=1:
                    label=f'{height:.0f}' if math.isclose(height,round(height),abs_tol=1e-9) else f'{height:.1f}'
                    ax.text(x[j],height_bottom+height/2,label,fontsize=6.8,ha='center',va='center',
                            fontweight='bold',color='black' if m in ('Landscape','TRAIL') else 'white')
                height_bottom+=height
            bottom[j]=height_bottom
        for xi,b in zip(x,bottom):ax.text(xi,b+.3,f'{b:.0f}',ha='center',va='bottom',fontsize=7)
        ax.set_ylim(0,27);ax.set_ylabel('Wins (of 25)')
        ax.set_title('Case wins',fontweight='bold',pad=7);style(ax,'A' if ri==0 else 'D')
        # B/E: sort by mean final HV low -> high *within each size group*,
        # preserving error bar association with the correct method.
        ax=axes[ri][1];w=.125
        for j,size in enumerate(SIZES):
            ordered=sorted(af.METHODS,
                key=lambda m:(float(lookup[(fam,size,m)]['hv_mean']),af.METHODS.index(m)))
            for i,m in enumerate(ordered):
                rec=lookup[(fam,size,m)]
                height=float(rec['hv_mean']);err=float(rec['hv_sd'])
                xx=x[j]+(i-(len(ordered)-1)/2)*w
                ax.bar(xx,height,width=w*.92,color=af_plot_colors[m],zorder=3)
                ax.errorbar(xx,height,yerr=err,fmt='none',color='#333333',
                            elinewidth=.65,capsize=1.4,capthick=.65,zorder=4)
        ax.axhline(100,color='#888888',linestyle='--',linewidth=.8,zorder=1)
        ax.set_ylabel('Final HV (% of own landscape)')
        ax.set_title('Normalized final HV',fontweight='bold',pad=7);ax.set_ylim(*hv_ylim)
        style(ax,'B' if ri==0 else 'E')
        # C/F: sort by mean runtime low -> high within each size group.
        ax=axes[ri][2];w=.145
        for j,size in enumerate(SIZES):
            ordered=sorted(af.RUNTIME_METHODS,
                key=lambda m:(float(lookup[(fam,size,m)]['runtime_mean']),af.RUNTIME_METHODS.index(m)))
            for i,m in enumerate(ordered):
                rec=lookup[(fam,size,m)]
                height=float(rec['runtime_mean']);err=float(rec['runtime_sd'])
                xx=x[j]+(i-(len(ordered)-1)/2)*w
                ax.bar(xx,height,width=w*.92,color=af_plot_colors[m],zorder=3)
                ax.errorbar(xx,height,yerr=err,fmt='none',color='#333333',
                            elinewidth=.65,capsize=1.4,capthick=.65,zorder=4)
        ax.set_ylim(0,runtime_ylim);ax.set_ylabel('Runtime (min)')
        ax.set_title('Runtime',fontweight='bold',pad=7);style(ax,'C' if ri==0 else 'F')
        axes[ri][1].text(.5,1.19,fam,transform=axes[ri][1].transAxes,
                         ha='center',va='bottom',fontsize=11.5,fontweight='bold')
    # G-I: sort the five comparison methods low -> high left to right
    # independently for each size, retaining the original five-method colors.
    ax=axes[2][0];w=.161
    for j,size in enumerate(SIZES):
        ordered=sorted(FIVE,key=lambda m:(float(lookup5[(size,m)]['top_result_cases_pct']),FIVE.index(m)))
        for i,m in enumerate(ordered):
            height=float(lookup5[(size,m)]['top_result_cases_pct'])
            ax.bar(x[j]+(i-2)*w,height,width=w*.94,color=FIVE_COLORS[m],zorder=3)
    ax.set_ylim(0,100);ax.set_yticks([0,25,50,75,100]);ax.set_ylabel('Top-result cases (%)')
    ax.set_title('Frequency of top result',fontweight='bold',pad=7);style(ax,'G')

    ax=axes[2][1]
    for j,size in enumerate(SIZES):
        ordered=sorted(FIVE,key=lambda m:(float(lookup5[(size,m)]['delta_hv_mean_pct']),FIVE.index(m)))
        for i,m in enumerate(ordered):
            rec=lookup5[(size,m)]
            height=float(rec['delta_hv_mean_pct']);err=float(rec['delta_hv_sd_pct'])
            xx=x[j]+(i-2)*w
            ax.bar(xx,height,width=w*.94,color=FIVE_COLORS[m],zorder=3)
            ax.errorbar(xx,height,yerr=err,fmt='none',color='#333333',
                        elinewidth=.65,capsize=1.4,capthick=.65,zorder=4)
    ax.axhline(0,color='#888888',lw=.8)
    ax.set_ylabel('Landscape-normalized ΔHV (%)')
    ax.set_title('Improvement beyond frozen landscape',fontweight='bold',pad=7);style(ax,'H')

    ax=axes[2][2]
    for j,size in enumerate(SIZES):
        ordered=sorted(FIVE,key=lambda m:(float(lookup5[(size,m)]['runtime_mean_min']),FIVE.index(m)))
        for i,m in enumerate(ordered):
            rec=lookup5[(size,m)]
            height=float(rec['runtime_mean_min']);err=float(rec['runtime_sd_min'])
            xx=x[j]+(i-2)*w
            ax.bar(xx,height,width=w*.94,color=FIVE_COLORS[m],zorder=3)
            ax.errorbar(xx,height,yerr=err,fmt='none',color='#333333',
                        elinewidth=.65,capsize=1.4,capthick=.65,zorder=4)
    ax.set_ylim(0,runtime_ylim);ax.set_ylabel('Runtime (min)')
    ax.set_title('Runtime',fontweight='bold',pad=7);style(ax,'I')
    # Two legends, separate families and selector comparison in one unified sheet.
    original_labels={'Landscape':'Landscape','No prior':'Base optimizer','Base':'Base optimizer + landscape knowledge',
        'GHOST':'GHOST','SPARC':'SPARC','TRAIL':'TRAIL'}
    top_legend_ax.legend(handles=[Patch(facecolor=af_plot_colors[m],edgecolor='none',label=original_labels[m]) for m in af.METHODS],
        loc='center',bbox_to_anchor=(.5,.5),ncol=6,columnspacing=1.25,handlelength=1.35,fontsize=7.6,frameon=False)
    bottom_legend_ax.legend(handles=[Patch(facecolor=FIVE_COLORS[m],edgecolor='none',label=m) for m in FIVE],
        loc='center',bbox_to_anchor=(.5,.5),ncol=5,columnspacing=1.10,handlelength=1.30,fontsize=7.8,frameon=False)
    destination.mkdir(parents=True,exist_ok=True)
    for extension in ('pdf','png','svg'):
        fig.savefig(destination/f'{NAME}.{extension}',bbox_inches='tight',
                    **({'dpi':600} if extension=='png' else {}))
    plt.close(fig)

def compute(archive,figure_root,skip_source_hash=False):
    af,qc=modules(archive)
    # Read 3300 archived scientific results ONCE through the original authoritative reader.
    data,origin,cases,missing=af.load_data(archive)
    winrows,afseed,afagg,tieaudit,af_runtime_cohort=af.summaries(data,cases)
    qc_root=archive/'Figures_and_Tables'/'Main'/'Figure_03_GHI_QC_Selector'
    qctable=qc_root/'inputs'/'PREOPT_QC_PER_SEED.csv'
    qcs,qc_sources=qc.read_qc(qctable)
    if not skip_source_hash: qc.validate_qc_repro_sources(archive,qc_sources)
    examples=qc.make_examples(data,qcs)
    main=qc.select_cross_species_problem(examples,drop_outlier=True)
    sensitivity=qc.select_cross_species_problem(examples,drop_outlier=False)
    expected={(seed,size,f'{sp}__{problem}') for seed in SEEDS for size in SIZES
              for sp in qc.SPECIES for problem in qc.PROBLEMS}
    overhead,overhead_status=measured_qc_overhead(qc_root/'inputs'/'QC_AND_SELECTOR_MEASURED_TIMINGS.csv',archive,expected)
    pcs,ghiseed,ghiagg,shared=five_method_statistics(main,data,overhead)
    pcs_full,seed_full,agg_full,shared_full=five_method_statistics(sensitivity,data,overhead)
    if len(main)!=297 or len(sensitivity)!=300:raise FigureError('Unexpected selector analysis sizes')
    # Edit the currently installed V4 output IN PLACE. No new timestamp run,
    # renamed figure, or duplicate Figure 3 folder is ever created.
    outbase=figure_root/'outputs'/'FIGURE3_ALL_NINE_PANELS_V4'
    outbase.mkdir(parents=True,exist_ok=True)
    # Legacy V4 stores each generated figure inside a timestamped run_* folder.
    # Choose the most recently created existing run, and overwrite its exact files.
    existing=[p for p in outbase.iterdir() if p.is_dir() and p.name.startswith('run_')
              and (p/(NAME+'.pdf')).is_file() and not p.is_symlink()]
    if existing:
        dest=max(existing,key=lambda p:(p.stat().st_mtime,p.name))
    else:
        dest=outbase/'CURRENT'
    draw(af,qc,afagg,ghiagg,dest,overhead is not None)
    plots=dest/'plot_data'
    write_csv(plots/'ORIGINAL_AF_3SEED_AGGREGATE.csv',afagg)
    write_csv(plots/'ORIGINAL_AF_SEED_LEVEL.csv',afseed)
    write_csv(plots/'ORIGINAL_AF_WIN_COUNTS_BY_SEED.csv',winrows)
    write_csv(plots/'ORIGINAL_AF_TIE_AUDIT.csv',tieaudit)
    write_csv(plots/'ORIGINAL_AF_SCIENTIFIC_SOURCE_AUDIT.csv',origin)
    write_csv(plots/'GHI_CASEWISE_MAIN_297.csv',pcs)
    write_csv(plots/'GHI_SEED_LEVEL_MAIN_297.csv',ghiseed)
    write_csv(plots/'GHI_POOLED_TOP_PCT_MAIN_297.csv',ghiagg)
    write_csv(plots/'GHI_POOLED_TOP_PCT_FULL_300_SENSITIVITY.csv',agg_full)
    write_csv(plots/'GHI_CASEWISE_FULL_300_SENSITIVITY.csv',pcs_full)
    write_csv(plots/'GHI_SEED_LEVEL_FULL_300_SENSITIVITY.csv',seed_full)
    write_csv(plots/'QC_SELECTOR_PRIMARY_PREDICTIONS.csv',main)
    write_csv(plots/'QC_SELECTOR_FULL_300_PREDICTIONS.csv',sensitivity)
    report={
      'generator':'Unified single-command Figure 3 A-I V4 with V5 presentation changes',
      'source':'frozen original A-F figure_3_seeds.py + frozen one-feature selector_ghi.py',
      'source_qc_sha256_verified':not skip_source_hash,
      'seeds':list(SEEDS),'landscape_sizes':list(SIZES),'endpoint_exact_calls':128,
      'figure_file':f'{NAME}.pdf',
      'C_F_I_identical_ylabel':'Runtime (min)',
      'C_F_I_identical_title':'Runtime',
      'C_F_I_identical_yaxis_max':True,
      'A_D_win_policy':'Original fractional wins, averaged across three seeds; stacked per-size in ascending height order',
      'B_E_HV_policy':'100*(official final method HV/own frozen landscape HV), mean ± SD of three seed means',
      'G_top_cases_policy':'100*(number of seed+case observations tied for best official final HV among five methods)/(number of eligible pooled seed+case observations), full credit per tied method, NO per-seed averaging and NO fractional winner share',
      'G_tie_policy':'full_credit_all_tied_methods','G_main_denominators_by_size':{str(s):72 if s==5000 else 75 for s in SIZES},
      'G_methods':list(FIVE),
      'H_delta_policy':'100*(final HV - same seed+case+size frozen landscape HV)/that same frozen landscape HV',
      'H_I_aggregate_policy':'mean±sample SD of the three seed-level case means',
      'I_runtime_status':overhead_status,
      'I_end_to_end_QC_overhead_measured':overhead is not None,
      'I_runtime_cohort_matched_cases':len(shared),
      'AF_runtime_cohort_matched_cases':len(af_runtime_cohort),
      'primary_selector_observations':len(main),
      'full_untouched_selector_observations':len(sensitivity),
      'CHO_5K_exclusion':'Excluded from primary in each of 3 seeds based on historical sensitivity; full-data separate sensitivity output remains',
      'method_selection':'QC alone only; Ridge alpha6; train label clipping +/-20 pp; positive 0.5 pp gsMOBO threshold',
      'prior_figures_sources_modified':False,
      'visual_palette_A_to_F':{'SPARC':'#009E73','TRAIL':'#F0B400'},
      'visual_order':'Every group sorted ascending per landscape size; stacked A/D bottom to top; grouped B/C/E/F/G/H/I left to right',
      'output_policy':'Replaces same-named V4 figure and plot data in most recent existing V4 run folder; no new run directories',
      'caution':'If no measured source-audited QC training timings, C/F/I present optimizer runtime only; figure itself has no inaccurate end-to-end claim.'}
    (dest/'FIGURE_3_UNIFIED_BUILD_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    (dest/'README_RESULTS.txt').write_text(
      'Unified Figure 3 A-I. Scientific calculations are unchanged; sorting and colors updated.\n'
      'G: pooled top-result percentage with full credit for each tied method.\n'
      'H: normalized final hypervolume improvement over the same frozen landscape.\n'
      'C, F, I y-axis: Runtime (min), shared limits.\n'
      'I overhead: '+overhead_status+'\n'
      'Primary: 297 rows. Complete-data sensitivity: 300 rows.\n'
      'Case-wise sources and summary CSVs are in plot_data/.\n',encoding='utf8')
    return dest,report


# Replot source-data-only path. This does not import the frozen QC selector or
# recompute any underlying scientific metric, and requires the archived V4
# numerical summary tables to be present in the same existing output folder.

def _finite_numeric(value, context, allow_blank=False):
    if value in ('',None) and allow_blank: return None
    try: x=float(value)
    except (ValueError,TypeError): raise FigureError(f'Invalid numeric field {context}: {value!r}')
    if not math.isfinite(x): raise FigureError(f'Nonfinite numeric field {context}')
    return x

def _saved_csv(path, needed):
    if not path.is_file() or path.is_symlink(): raise FigureError('Expected original V4 data table not found: '+str(path))
    with path.open('r',newline='',encoding='utf-8-sig') as f:
        rdr=csv.DictReader(f)
        if not needed.issubset(rdr.fieldnames or []):
            raise FigureError('V4 summary lacks expected fields '+str(sorted(needed-set(rdr.fieldnames or [])))+' in '+str(path))
        return list(rdr)

def _validate_saved_aggregates(afrows,ghirows,af):
    expected_af={(fam,n,m) for fam in ('gsMOBO','NSGA-II') for n in SIZES for m in af.METHODS}
    found_af=set()
    for r in afrows:
        key=(r['family'],int(r['prior_points']),r['method'])
        if key in found_af: raise FigureError('Duplicate original A-F summary key: '+str(key))
        found_af.add(key)
        for f in ('avg_wins','hv_mean','hv_sd'):
            if _finite_numeric(r[f],str(key)+' '+f)<0: raise FigureError('Negative A-F summary: '+str(key)+' '+f)
        if r['method']!='Landscape':
            for f in ('runtime_mean','runtime_sd'):
                if _finite_numeric(r[f],str(key)+' '+f)<0: raise FigureError('Negative A-F runtime')
        # Historical null Landscape runtime is expected, not an error.
        if r.get('n_seeds') not in (None,'','3'):
            raise FigureError('Original V4 A-F is not a three-seed summary: '+str(key))
    if found_af!=expected_af:raise FigureError('A-F saved V4 summary has missing/extra keys: missing='+str(sorted(expected_af-found_af)[:6])+ ' extra='+str(sorted(found_af-expected_af)[:6]))
    for fam in ('gsMOBO','NSGA-II'):
        for n in SIZES:
            w=sum(float(r['avg_wins']) for r in afrows if r['family']==fam and int(r['prior_points'])==n)
            if not math.isclose(w,25,abs_tol=1e-6,rel_tol=0):
                raise FigureError(f'A-F original win totals invalid: {fam}/{n} got {w}')
    expected_ghi={(n,m) for n in SIZES for m in FIVE}
    found_ghi=set()
    for r in ghirows:
        key=(int(r['prior_points']),r['method'])
        if key in found_ghi:raise FigureError('Duplicate original G-I summary key: '+str(key))
        found_ghi.add(key)
        for f in ('top_result_cases_pct','delta_hv_mean_pct','delta_hv_sd_pct','runtime_mean_min','runtime_sd_min'):
            v=_finite_numeric(r[f],str(key)+' '+f)
            if f!='delta_hv_mean_pct' and v<0:raise FigureError('Negative G-I summary metric: '+str(key)+' '+f)
        p=float(r['top_result_cases_pct'])
        if p>100+1e-6:raise FigureError('Top-result percentage exceeds 100: '+str(key))
        if r.get('n_pooled_cases') not in (None,''):
            expected_n=72 if int(r['prior_points'])==5000 else 75
            if int(r['n_pooled_cases'])!=expected_n:
                raise FigureError('Wrong G-I pooled primary denominator: '+str(key))
    if found_ghi!=expected_ghi:
        raise FigureError('G-I saved V4 summary has missing/extra keys: missing='+str(sorted(expected_ghi-found_ghi)[:6])+' extra='+str(sorted(found_ghi-expected_ghi)[:6]))

def _existing_v4_runs(figure_root):
    outbase=figure_root/'outputs'/'FIGURE3_ALL_NINE_PANELS_V4'
    if not outbase.is_dir():return []
    folders=[p for p in outbase.iterdir() if p.is_dir() and not p.is_symlink() and (p.name.startswith('run_') or p.name=='CURRENT')]
    return sorted([p for p in folders if (p/(NAME+'.pdf')).is_file() and (p/'plot_data'/'ORIGINAL_AF_3SEED_AGGREGATE.csv').is_file()
        and (p/'plot_data'/'GHI_POOLED_TOP_PCT_MAIN_297.csv').is_file()], key=lambda p:(p.stat().st_mtime,p.name),reverse=True)

def replot_saved(archive,figure_root):
    """Pure re-render. Uses the two frozen, already-computed V4 numeric tables."""
    af_path=figure_root/'scripts'/'figure_3_seeds.py'
    af=load_source(af_path,'original_figure3_palette_and_methods')
    # Importing this module runs no archived scientific jobs. We use only
    # immutable figure method names/order and the source palette definition.
    if tuple(af.SEEDS)!=SEEDS or tuple(af.SIZES)!=SIZES or tuple(af.METHODS)!=('Landscape','No prior','Base','GHOST','SPARC','TRAIL'):
        raise FigureError('Original frozen A-F definitions changed; refusing to alter science')
    candidates=_existing_v4_runs(figure_root)
    if not candidates:
        raise FigureError('No complete original V4 plot-data tables found under '+
           str(figure_root/'outputs'/'FIGURE3_ALL_NINE_PANELS_V4')+
           '/run_*/plot_data. Figure left unchanged. Please supply the V4 output run folder, including plot_data/')
    dest=candidates[0]
    source_a=dest/'plot_data'/'ORIGINAL_AF_3SEED_AGGREGATE.csv'
    source_g=dest/'plot_data'/'GHI_POOLED_TOP_PCT_MAIN_297.csv'
    afrows=_saved_csv(source_a,{'family','prior_points','method','avg_wins','hv_mean','hv_sd','runtime_mean','runtime_sd'})
    ghirows=_saved_csv(source_g,{'prior_points','method','top_result_cases_pct','delta_hv_mean_pct','delta_hv_sd_pct','runtime_mean_min','runtime_sd_min'})
    _validate_saved_aggregates(afrows,ghirows,af)
    # CSV inputs are strings; after strict validation, recover numeric types
    # used by the original V4 plotting code, without changing any value.
    for r in afrows:
        r['prior_points']=int(r['prior_points'])
        for field in ('avg_wins','hv_mean','hv_sd'):
            r[field]=float(r[field])
        for field in ('runtime_mean','runtime_sd'):
            r[field]=None if r[field] in ('',None) else float(r[field])
    for r in ghirows:
        r['prior_points']=int(r['prior_points'])
        for field in ('top_result_cases_pct','delta_hv_mean_pct','delta_hv_sd_pct','runtime_mean_min','runtime_sd_min'):
            r[field]=float(r[field])
    # Do not fabricate/re-evaluate the frozen selector. Use V4's exact source
    # summary numbers and re-render to a temporary folder first.
    with tempfile.TemporaryDirectory(prefix='.figure3_style_stage_',dir=dest) as temp:
        stage=Path(temp)
        draw(af,None,afrows,ghirows,stage,False)
        for ext in ('pdf','svg','png'):
            new=stage/(NAME+'.'+ext)
            if not new.is_file() or new.stat().st_size<2000:
                raise FigureError('Failed to render V6 '+ext+'; prior figure left untouched')
        # Update exactly the old figure files, never make a run_* directory.
        for ext in ('pdf','svg','png'):
            os.replace(stage/(NAME+'.'+ext),dest/(NAME+'.'+ext))
    digest=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    audit={
        'generator':'Figure 3 V6 presentation-only repair',
        'based_on':'original V4 saved numerical plot_data/ tables',
        'science_recalculated':False,
        'qc_selector_retrained':False,
        'optimizers_rerun':False,
        'source_A_F_csv':str(source_a.relative_to(archive)),
        'source_A_F_csv_sha256':digest(source_a),
        'source_G_I_csv':str(source_g.relative_to(archive)),
        'source_G_I_csv_sha256':digest(source_g),
        'palette_A_F':{'SPARC':'#009E73','TRAIL':'#F0B400'},
        'sort':'ascending within each size; A/D bottom-to-top, grouped B/C/E/F/G/H/I left-to-right',
        'output':'overwrites exact existing V4 PDF PNG SVG in its current run directory; no new run',
        'run_folder':str(dest.relative_to(archive)),
        'qc_dependency':'Not required for pure presentation redraw',
    }
    (dest/'FIGURE_3_STYLE_ONLY_UPDATE.json').write_text(json.dumps(audit,indent=2)+'\n',encoding='utf-8')
    return dest,audit

def main(argv=None):
    ap=argparse.ArgumentParser(description='Style-only GEM Figure 3 replot from existing official V4 data tables')
    ap.add_argument('--archive',type=Path,help='GEM_Publication_Archive root, autodetected when omitted')
    ap.add_argument('--figure-dir',type=Path,help='Canonical original Figure 3 directory')
    args=ap.parse_args(argv)
    script=Path(__file__).resolve()
    archive=args.archive.resolve() if args.archive else find_archive(script)
    figure=args.figure_dir.resolve() if args.figure_dir else archive/'Figures_and_Tables/Main/Figure_03_ThreeSeed_Optimization_Aggregate'
    if figure!=archive/'Figures_and_Tables/Main/Figure_03_ThreeSeed_Optimization_Aggregate':
        print('BLOCKED: Output must be within canonical original Figure 3 directory',file=sys.stderr)
        return 2
    try:
        dest,audit=replot_saved(archive,figure)
    except (FigureError,ValueError,TypeError,OSError,KeyError,RuntimeError) as exc:
        print('FIGURE 3 STYLE-ONLY: SAFELY BLOCKED:',exc,file=sys.stderr)
        return 2
    print('PASS:',dest/(NAME+'.pdf'))
    print('Colors and sorted bars updated in place. Original V4 aggregate data reused unchanged.')
    print('NO frozen selector, scientific recomputation, new optimization, or new run directory.')
    return 0

if __name__=='__main__':sys.exit(main())
