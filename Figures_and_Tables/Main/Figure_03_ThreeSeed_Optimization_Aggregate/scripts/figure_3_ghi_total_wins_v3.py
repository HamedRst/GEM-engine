#!/usr/bin/env python3
"""GEM Figure 3 G-I V3: three-seed TOTAL win credits and landscape-referenced gains.

Additive release into the existing Figure 3 A-F directory. Frozen QC-only selector
source imported unchanged from Figure_03_GHI_QC_Selector. QC computation overhead
is included only if source-verified timing exists; otherwise graph is explicitly
optimizer-runtime-only. All previous results, source files, and A-F are retained.
"""
from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,math,statistics,sys,time,traceback
from datetime import datetime, timezone
from collections import defaultdict
from pathlib import Path

SIZES=(5000,10000,25000,50000)
SEEDS=(42,43,44)
METHODS=('QC-only selector','gsMOBO + SPARC','NSGA-II + SPARC','gsMOBO + landscape','NSGA-II + landscape')
COLORS={
 'QC-only selector':'#6A3D9A',
 'gsMOBO + SPARC':'#E69F00',
 'NSGA-II + SPARC':'#009E73',
 'gsMOBO + landscape':'#0072B2',
 'NSGA-II + landscape':'#222222'
}
TIE_TOL=1e-12
class FigureError(RuntimeError): pass

def frozen_module(figure_dir):
    f=figure_dir/'scripts'/'selector_ghi.py'
    if not f.is_file() or f.is_symlink():raise FigureError('Frozen QC-only V1 source not present: '+str(f))
    spec=importlib.util.spec_from_file_location('frozen_gem_qc_only',f)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    for k,v in {'ALPHA':6.0,'TARGET_CLIP':20.0,'SWITCH_MARGIN':0.5}.items():
        if getattr(m,k,None)!=v:raise FigureError('Frozen QC-only model setting unexpectedly changed: '+k)
    return m

def get_method_data(row,data):
    seed,size,case=row['landscape_seed'],row['prior_points'],row['case']
    lookup={
      'gsMOBO + SPARC':('gsMOBO','SPARC'),
      'NSGA-II + SPARC':('NSGA-II','SPARC'),
      'gsMOBO + landscape':('gsMOBO','Base'),
      'NSGA-II + landscape':('NSGA-II','Base'),
    }
    out={}
    for name in METHODS:
        if name=='QC-only selector':
            out[name]={'hv':row['selected_hv'],'runtime_minutes':row['selected_runtime_min']}
        else:
            fam,method=lookup[name]
            d=data[(seed,size,case,fam,method)]
            out[name]={'hv':d['hv'],'runtime_minutes':d['runtime_minutes']}
        if not math.isfinite(out[name]['hv']) or out[name]['hv']<0:
            raise FigureError('Invalid method HV: '+str((seed,size,case,name)))
    return out

def enrich(predictions,data):
    enriched=[]
    for row in predictions:
        d=get_method_data(row,data)
        copy=dict(row)
        for method,m in d.items():
            copy[method]={'hv':m['hv'], 'runtime_min':m['runtime_minutes'],
                'landscape_delta_hv':m['hv']-row['landscape_hv'],
                'landscape_delta_pct':100*(m['hv']-row['landscape_hv'])/row['landscape_hv']}
        enriched.append(copy)
    return enriched

def timing_data(path,root,expected):
    """A source-audited timing report from ORIGINAL QC computation, not an invented proxy.
    Each row = one seed+prior+case with actual elapsed QC computation and decision time.
    Historical archive currently has NO such time fields, so absence => preview.
    """
    if not path.is_file():return None,'No QC-computation timing table found. Optimizer-only preview; NOT end-to-end runtime.'
    rows={};sources=set()
    with path.open(newline='',encoding='utf8') as f:
        rd=csv.DictReader(f)
        keys={'landscape_seed','prior_points','case','qc_preprocessing_seconds','selector_decision_seconds',
              'source_file','source_sha256','timing_kind'}
        if not keys.issubset(rd.fieldnames or []):raise FigureError('QC timing table is incomplete. Required: '+','.join(sorted(keys)))
        for i,r in enumerate(rd,2):
            try:
                key=int(r['landscape_seed']),int(r['prior_points']),r['case'].strip()
                qc=float(r['qc_preprocessing_seconds']);inference=float(r['selector_decision_seconds'])
            except (TypeError,ValueError,KeyError) as e:raise FigureError(f'QC timing line {i} invalid: {e}')
            if key in rows or not math.isfinite(qc) or not math.isfinite(inference) or min(qc,inference)<0:
                raise FigureError(f'QC timing line {i} duplicates/negative/nonfinite')
            if r['timing_kind']!='MEASURED_QC_COMPUTATION':
                raise FigureError('QC timing must be original measured QC computation, not file-read time or estimate')
            src=r['source_file'];sha=r['source_sha256'].lower()
            if (not src or Path(src).is_absolute() or '..' in Path(src).parts or
                    len(sha)!=64 or any(c not in '0123456789abcdef' for c in sha)):
                raise FigureError('Invalid relative timing source or SHA256')
            rows[key]=(qc+inference)/60
            sources.add((src,sha))
    if set(rows)!=expected:raise FigureError(f'QC overhead records {len(rows)}/300; missing {sorted(expected-set(rows))[:4]}')
    for relative,sha in sources:
        source=root/relative
        if not source.is_file() or source.is_symlink():raise FigureError('QC timing source absent '+relative)
        digest=hashlib.sha256(source.read_bytes()).hexdigest()
        if digest!=sha:raise FigureError('QC timing source checksum mismatch '+relative)
    return rows,'Complete verified original QC fit plus selector decision timing: end-to-end runtime.'

def score(rows,runtime_overhead=None):
    """For each seed/size/case, allocate wins fractionally across *five* rows.
    Because QC selector clones one SPARC endpoint, its wins are exact ties. This
    preserves the A/D tie convention and prevents inflated winner counts.
    """
    cases=defaultdict(list)
    for r in rows:cases[(r['landscape_seed'],r['prior_points'])].append(r)
    per_case=[];per_seed=[]
    # Same cases across all methods, sizes, and seeds for runtime (drop CHO 5K
    # from the all-size matched cohort in the outlier-excluded analysis).
    shared={r['case'] for r in rows if all(
        any(q['case']==r['case'] and q['landscape_seed']==s and q['prior_points']==size and
            all(q[m]['runtime_min'] is not None and math.isfinite(q[m]['runtime_min']) for m in METHODS)
            for q in rows) for s in SEEDS for size in SIZES)}
    if len(shared)<20:raise FigureError('Insufficient matched case runtime cohort')
    for seed in SEEDS:
        for size in SIZES:
            cohort=cases[seed,size]
            expected_n=25 if len(rows)==300 else 24 if size==5000 else 25
            if len(cohort)!=expected_n:raise FigureError(f'Incorrect cohort {seed}/{size}: {len(cohort)}')
            wins={m:0.0 for m in METHODS}
            for r in cohort:
                best=max(r[m]['hv'] for m in METHODS)
                ties=[m for m in METHODS if abs(r[m]['hv']-best)<=TIE_TOL]
                for m in ties:wins[m]+=1/len(ties)
                key=(seed,size,r['case'])
                for m in METHODS:
                    per_case.append({'landscape_seed':seed,'prior_points':size,'case':r['case'],'method':m,
                     'method_hv':r[m]['hv'],'frozen_landscape_hv':r['landscape_hv'],
                     'delta_hv_absolute':r[m]['landscape_delta_hv'],
                     'delta_hv_percent_of_landscape':r[m]['landscape_delta_pct'],
                     'selected_optimizer_only_minutes':r[m]['runtime_min'],
                     'qc_plus_selection_overhead_minutes':(runtime_overhead[key] if runtime_overhead is not None and m=='QC-only selector' else 0.0 if m!='QC-only selector' else ''),
                     'fractional_win_share':1/len(ties) if m in ties else 0,
                     'winner_ties':'|'.join(ties)})
            if abs(sum(wins.values())-expected_n)>1e-8:raise FigureError('Winner fractions do not sum to case count')
            for m in METHODS:
                rv=[r[m]['runtime_min'] + (runtime_overhead[(seed,size,r['case'])] if runtime_overhead and m=='QC-only selector' else 0)
                     for r in cohort if r['case'] in shared]
                if any(x is None or x<0 or not math.isfinite(x) for x in rv):raise FigureError('Invalid measured worker runtime')
                delta=[r[m]['landscape_delta_pct'] for r in cohort]
                per_seed.append({'landscape_seed':seed,'prior_points':size,'method':m,
                  'wins':wins[m], 'delta_mean_percent':statistics.mean(delta),
                  'runtime_mean_minutes':statistics.mean(rv),'n_hv':len(delta), 'n_runtime':len(rv)})
    aggr=[]
    for size in SIZES:
        for m in METHODS:
            by=[x for x in per_seed if x['prior_points']==size and x['method']==m]
            if len(by)!=3:raise FigureError('Missing exactly 3 seed-level summaries')
            aggr.append({'prior_points':size,'method':m,'n_seeds':3,'n_cases':by[0]['n_hv'],
               'total_win_credits':sum(q['wins'] for q in by),
               'delta_mean_percent':statistics.mean(q['delta_mean_percent'] for q in by),
               'delta_sd_percent':statistics.stdev(q['delta_mean_percent'] for q in by),
               'runtime_mean_minutes':statistics.mean(q['runtime_mean_minutes'] for q in by),
               'runtime_sd_minutes':statistics.stdev(q['runtime_mean_minutes'] for q in by)})
    for size in SIZES:
        expected_total=72 if len(rows)==297 and size==5000 else 75
        observed=sum(z['total_win_credits'] for z in aggr if z['prior_points']==size)
        if not math.isclose(observed, expected_total, rel_tol=0, abs_tol=1e-8):
            raise FigureError(f'Total win credits mismatch size={size}: {observed} != {expected_total}')
    return per_case,per_seed,aggr,sorted(shared)

def plot(aggr,root,full_runtime):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10.5,
        'axes.labelsize':9.0,'axes.linewidth':0.8,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'legend.fontsize':7.4,'legend.frameon':False,'figure.dpi':180,'savefig.dpi':600,
        'pdf.fonttype':42,'ps.fonttype':42})
    fig,axs=plt.subplots(1,3,figsize=(11.5,3.8))
    xs=np.arange(4,dtype=float);bar=.16
    for ax,letter,title,key,ylabel in zip(axs,['G','H','I'],
       ['Case wins across three seeds','Improvement beyond frozen landscape',
        'Runtime including QC overhead' if full_runtime else 'Optimization runtime'],
       ['total_win_credits','delta_mean_percent','runtime_mean_minutes'],
       ['Win count','Landscape-normalized ΔHV (%)','Wall time (min)']):
        for i,m in enumerate(METHODS):
            vals=[next(z for z in aggr if z['prior_points']==size and z['method']==m) for size in SIZES]
            xx=xs+(i-2)*bar
            heights=np.array([v[key] for v in vals])
            ax.bar(xx,heights,width=bar*.95,color=COLORS[m],zorder=3)
            if key!='total_win_credits':
                err=np.array([v['delta_sd_percent' if key=='delta_mean_percent' else 'runtime_sd_minutes'] for v in vals])
                ax.errorbar(xx,heights,yerr=err,fmt='none',color='#333333',elinewidth=.65,capsize=1.4,capthick=.65,zorder=4)
        if key=='delta_mean_percent':ax.axhline(0,color='#777777',lw=.75)
        else:ax.set_ylim(bottom=0)
        ax.set_xticks(xs);ax.set_xticklabels(['5K','10K','25K','50K']);ax.set_ylabel(ylabel);ax.set_title(title,weight='bold',pad=6)
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.grid(axis='y',color='#E7E7E7',lw=.6,zorder=0)
        ax.text(-.13,1.10,letter,transform=ax.transAxes,ha='left',va='bottom',fontsize=11.5,fontweight='bold',clip_on=False)
    display_names={'gsMOBO + landscape':'gsMOBO (with knowledge)',
                   'NSGA-II + landscape':'NSGA-II (with knowledge)'}
    fig.legend(handles=[Patch(facecolor=COLORS[m],label=display_names.get(m,m)) for m in METHODS],loc='lower center',
               bbox_to_anchor=(.5,.015),ncol=5,columnspacing=.9,handlelength=1.2,frameon=False)
    fig.subplots_adjust(left=.07,right=.99,top=.79,bottom=.24,wspace=.36)
    basename='Figure_3_GHI_TotalWinCounts_LandscapeDeltaHV_3seeds_V3'
    root.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png','svg'):
        fig.savefig(root/f'{basename}.{ext}',bbox_inches='tight',**({'dpi':600} if ext=='png' else {}))
    plt.close(fig)
    return basename

def write_rows(p,rs):
    p.parent.mkdir(parents=True,exist_ok=True)
    if not rs:return
    with p.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rs[0]),extrasaction='ignore');w.writeheader();w.writerows(rs)

def run(archive,figure):
    selector=frozen_module(figure)
    qc,sources=selector.read_qc(figure/'inputs'/'PREOPT_QC_PER_SEED.csv')
    selector.validate_qc_repro_sources(archive,sources)
    data,origin=selector.load_fig3_science(archive)
    source=selector.make_examples(data,qc)
    pred=selector.select_cross_species_problem(source,drop_outlier=True)
    full=selector.select_cross_species_problem(source,drop_outlier=False)
    complete_keys={(s,size,f'{sp}__{p}') for s in SEEDS for size in SIZES for sp in selector.SPECIES for p in selector.PROBLEMS}
    timing,status=timing_data(figure/'inputs'/'QC_AND_SELECTOR_MEASURED_TIMINGS.csv',archive,complete_keys)
    enriched=enrich(pred,data)
    # New versioned run directory INSIDE the original Figure 3 A-F figure folder.
    # Never modify its existing scripts, PDFs, CSVs, or earlier G-I outputs.
    outbase=archive/'Figures_and_Tables'/'Main'/'Figure_03_ThreeSeed_Optimization_Aggregate'/'outputs'/'GHI_FIVE_METHODS_TOTAL_WINS_V3'
    outbase.mkdir(parents=True,exist_ok=True)
    base=datetime.now(timezone.utc).strftime('run_%Y%m%d_%H%M%S_UTC')
    idx=1
    while (outbase/(base if idx==1 else f'{base}_{idx}')).exists(): idx+=1
    out=outbase/(base if idx==1 else f'{base}_{idx}')
    out.mkdir()
    pd=out/'plot_data';pd.mkdir(parents=True,exist_ok=True)
    full_enriched=enrich(full,data)
    percase,perseed,aggr,shared=score(enriched,timing)
    _,fs,fa,fc=score(full_enriched,timing)
    write_rows(pd/'FIVE_METHOD_CASEWISE_MAIN_297.csv',percase)
    write_rows(pd/'FIVE_METHOD_PER_SEED_MAIN_297.csv',perseed)
    write_rows(pd/'FIVE_METHOD_3SEED_TOTAL_WINCOUNTS_MAIN_297.csv',aggr)
    write_rows(pd/'FIVE_METHOD_3SEED_TOTAL_WINCOUNTS_FULL_300_SENSITIVITY.csv',fa)
    write_rows(pd/'FIVE_METHOD_PER_SEED_FULL_300_SENSITIVITY.csv',fs)
    name=plot(aggr,out,timing is not None)
    report={'selector_model':'UNCHANGED: surrogate QC alone Ridge, alpha=6, clipped y ±20pp, switch margin +0.5pp',
      'methods':list(METHODS),'panel_G':'Five-way fractional win credits per case, SUMMED across seeds 42+43+44 (not averaged); total sum=72 at 5K,=75 otherwise, with three-seed excluded outlier',
      'panel_H':'100*(method_final_HV - same_case_same_seed_same_size_frozen_landscape_HV)/frozen_landscape_HV',
      'panel_I':status,
      'color_palette':'Exact five hex colors reused from Figure 3 A-F palette',
      'win_total_not_mean':True,
      'qc_overhead_absence_stated_in_plot':False,'exclusion':'CHO constrained product 5K excluded across 42/43/44 from main; full 300 sensitivity saved',
      'primary_rows':len(enriched),'full_rows':len(full_enriched),'matched_runtime_cases':shared,
      'winner_ties_note':'Selector replicates one SPARC endpoint, so its wins tie with that method; splits are mandatory',
      'runtime_publication_ready':timing is not None,'figure_file':name+'.pdf',
      'original_campaigns_or_fig3_AF_modified':False}
    (out/'FIVE_METHOD_BUILD_REPORT.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print('PASS: '+str(out/(name+'.pdf')))
    print('Main:',len(enriched),'Full sensitivity:',len(full_enriched),'Matched runtime:',len(shared))
    if timing is None:
        print('NOTE: panel I contains optimizer-only worker runtime; measured pre-run QC time is absent.')
        print('Details appear ONLY in metadata/README, not printed on the figure.')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path);ap.add_argument('--figure-dir',type=Path)
    a=ap.parse_args()
    # Running script is in Figure_03_ThreeSeed_Optimization_Aggregate/scripts.
    archive=a.archive.resolve() if a.archive else (Path(__file__).resolve().parents[4])
    figure=a.figure_dir.resolve() if a.figure_dir else (archive/'Figures_and_Tables/Main/Figure_03_GHI_QC_Selector')
    if not (archive/'Results/Optimizations').is_dir():
        raise FigureError('Missing GEM_Publication_Archive/Results/Optimizations')
    selector=frozen_module(figure)
    try:run(archive,figure)
    except Exception as e:
        print('FIGURE GHI FIVE: BLOCKED: '+str(e),file=sys.stderr)
        return 2
    return 0
if __name__=='__main__':sys.exit(main())
