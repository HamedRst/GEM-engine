#!/usr/bin/env python3
"""Landscape-knowledge six-panel figure, three frozen landscape seeds (42-44).

Provenance: adapts the original GEM_Landscape_Knowledge_Figure_v1.5.1
  GEM_landscape_knowledge_figure_v1.py (historical run log, Oct 2026),
using frozen 128-call scientific job reader from published A-F figure 3 archive.
Does not fabricate or rerun any FBA/model/optimizer.
"""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, statistics, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

SEEDS=(42,43,44)
SIZES=(5000,10000,25000,50000)
ORDER=('yeast','pputida','cglutamicum','cho','ecoli_antiepex')
DISPLAY={'yeast':'Yeast','pputida':'P. putida','cglutamicum':'C. glutamicum',
         'cho':'CHO','ecoli_antiepex':'E. coli'}
METHODS=('Landscape','gsMOBO (no prior)','gsMOBO + landscape','NSGA-II (no prior)','NSGA-II + landscape')
COLORS={'Landscape':'#BDBDBD','gsMOBO (no prior)':'#6A3D9A',
        'gsMOBO + landscape':'#222222','NSGA-II (no prior)':'#0072B2',
        'NSGA-II + landscape':'#009E73'}
FILE_STEM='Figure_Landscape_Knowledge_Three_Seeds_Trends_First'

class FigureError(RuntimeError): pass

def archive_root(script:Path, override:Path|None=None)->Path:
    if override:
        p=override.expanduser().resolve()
        if not (p/'Results'/'Optimizations').is_dir():raise FigureError('Missing Results/Optimizations in '+str(p))
        return p
    for p in (script.parent,*script.parents):
        if (p/'Results'/'Optimizations').is_dir() and (p/'Figures_and_Tables').is_dir():return p
    raise FigureError('Could not find GEM_Publication_Archive; use --archive PATH')

def load_frozen_reader(archive:Path):
    p=archive/'Figures_and_Tables/Main/Figure_03_ThreeSeed_Optimization_Aggregate/scripts/figure_3_seeds.py'
    if not p.is_file() or p.is_symlink():raise FigureError('Frozen A-F scientific reader missing: '+str(p))
    spec=importlib.util.spec_from_file_location('frozen_gem_af_reader_for_knowledge_figure',p)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    if tuple(obj.SEEDS)!=SEEDS or tuple(obj.SIZES)!=SIZES or obj.CALLS!=128:
        raise FigureError('Frozen scientific reader is not the validated 3-seed / 128-call version')
    return obj,p

def finite_hv(data,key):
    try: h=float(data[key]['hv'])
    except (ValueError,TypeError,KeyError) as e:raise FigureError(f'Missing official final HV: {key}: {e}')
    if not math.isfinite(h) or h<0:raise FigureError('Nonfinite/negative official HV: '+repr(key))
    return h

def canonical_control_hvs(frozen,archive,data,cases):
    """Recheck legacy endpoint field choices before defining fixed no-prior baselines.

    This matters because the shared A-F reader uses an overall no-prior endpoint
    priority, whereas the historical plot specifically used gsMOBO self-front and
    NSGA-II archive-front. Missing desired fields are not fabricated: documented
    source-reader values remain fallbacks and are explicitly audited.
    """
    controls={}; audit=[]
    for seed in SEEDS:
        for case in cases:
            for fam,size,field in (('gsMOBO',10000,'final_self_hv'),('NSGA-II',5000,'final_archive_hv')):
                record=data[(seed,size,case,fam,'No prior')]
                previous=finite_hv(data,(seed,size,case,fam,'No prior'))
                source_key=record.get('hv_key','')
                if source_key.split('.')[-1]==field:
                    value=previous; status='FROZEN_READER_ALREADY_EXACT_FIELD'; located=record.get('hv_source','')
                else:
                    algorithm=frozen.ALGORITHMS[fam]['No prior']
                    camp=frozen.campaign_root(archive,seed,size)
                    rec=frozen.collect_json(camp/'jobs'/case/algorithm)
                    hits=frozen.metadata_value(rec,{field},lambda p,k:'history' not in k.lower() and '[' not in k and 'trajectory' not in k.lower())
                    unique={round(v,12) for _,_,v in hits}
                    if len(unique)>1:raise FigureError('Conflicting canonical '+field+' from '+str((seed,case,fam)))
                    if hits:
                        value=hits[0][2];located=str(hits[0][0].relative_to(archive))
                        status='EXPLICIT_HISTORICAL_FIELD_RECOVERED'
                    else:
                        value=previous;located=record.get('hv_source','')
                        status='HISTORICAL_FIELD_MISSING_FALLBACK_TO_FROZEN_READER'
                if not math.isfinite(value) or value<0:raise FigureError('Bad canonical baseline for '+str((seed,case,fam)))
                controls[(seed,case,fam)]=value
                audit.append(dict(landscape_seed=seed,case=case,family=fam,
                    canonical_run_prior_points=size,historical_field=field,
                    selected_canonical_hv=value,frozen_reader_hv=previous,
                    frozen_reader_hv_key=source_key,historical_field_status=status,
                    selected_source=located))
    if len(controls)!=150:raise FigureError('Incomplete canonical baseline universe')
    return controls,audit

def build_casewise(data,cases,canonical_override=None):
    """Historical six-panel semantics: fixed no-prior per seed/case, each 50K ref per seed/case.
    gsMOBO canonical no-prior taken from 10K; NSGA-II canonical no-prior from 5K.
    Notes: saved no-prior variants for other prior sizes are preserved in an audit file.
    """
    result=[];no_prior_audit=[]
    if len(cases)!=25 or len(set(cases))!=25:raise FigureError('Expected 25 distinct biological problems')
    counts=Counter(c.split('__',1)[0] for c in cases)
    if set(counts)!=set(ORDER) or any(n!=5 for n in counts.values()):
        raise FigureError('Expected 5 optimization problems for each of five original species: '+repr(counts))
    for seed in SEEDS:
        for case in cases:
            species=case.split('__',1)[0]
            problem=case.split('__',1)[1]
            ref50=finite_hv(data,(seed,50000,case,'Landscape','Landscape'))
            if ref50<=0:raise FigureError('Nonpositive 50K landscape reference: '+str((seed,case)))
            gs_plain=(canonical_override[(seed,case,'gsMOBO')] if canonical_override is not None else
                finite_hv(data,(seed,10000,case,'gsMOBO','No prior')))
            ns_plain=(canonical_override[(seed,case,'NSGA-II')] if canonical_override is not None else
                finite_hv(data,(seed,5000,case,'NSGA-II','No prior')))
            for size in SIZES:
                ref=finite_hv(data,(seed,size,case,'Landscape','Landscape'))
                if ref<=0:raise FigureError('Nonpositive own-size landscape reference '+str((seed,size,case)))
                raw={
                    'Landscape':ref,
                    'gsMOBO (no prior)':gs_plain,
                    'gsMOBO + landscape':finite_hv(data,(seed,size,case,'gsMOBO','Base')),
                    'NSGA-II (no prior)':ns_plain,
                    'NSGA-II + landscape':finite_hv(data,(seed,size,case,'NSGA-II','Base')),
                }
                for family,canonical in [('gsMOBO',gs_plain),('NSGA-II',ns_plain)]:
                    actual=finite_hv(data,(seed,size,case,family,'No prior'))
                    no_prior_audit.append(dict(landscape_seed=seed,prior_points=size,case=case,family=family,
                        per_campaign_no_prior_hv=actual,canonical_no_prior_hv=canonical,
                        difference=actual-canonical,canonical_source_prior_points=10000 if family=='gsMOBO' else 5000))
                for method in METHODS:
                    h=raw[method]
                    result.append(dict(landscape_seed=seed,prior_points=size,species=species,problem=problem,
                        case=case,method=method,final_hv=h,own_size_landscape_hv=ref,landscape_50k_hv=ref50,
                        pct_own_landscape=100*h/ref,pct_50k_landscape=100*h/ref50,
                        method_source='LANDSCAPE_ONLY' if method=='Landscape' else (
                            'CANONICAL_NO_PRIOR_10K' if method=='gsMOBO (no prior)' else
                            'CANONICAL_NO_PRIOR_5K' if method=='NSGA-II (no prior)' else 'SAME_CAMPAIGN_BASE_WITH_LANDSCAPE')))
    if len(result)!=3*4*25*5:raise FigureError('Case data not complete')
    return result,no_prior_audit

def groups(rows,keys,field):
    grouped=defaultdict(list)
    for r in rows:grouped[tuple(r[k] for k in keys)].append(float(r[field]))
    return grouped

def build_stats(case_rows):
    """Two-stage mean: within-seed average across biological cases, then 3-seed mean.
    Equal replicate numbers mean pooling yields same point estimate; seed SD is reported.
    """
    seed_rows=[];agg_rows=[]
    for panel,subset_keys,metric in (
        ('trends',('prior_points','method'),'pct_50k_landscape'),
        ('species',('prior_points','species','method'),'pct_own_landscape')):
        by_seed=groups(case_rows,('landscape_seed',)+subset_keys,metric)
        grouped=defaultdict(list)
        expected_n=25 if panel=='trends' else 5
        for key,vals in sorted(by_seed.items(), key=lambda x:repr(x[0])):
            seed,*rest=key
            if len(vals)!=expected_n:raise FigureError(f'Wrong case count for {key}: {len(vals)}, expected {expected_n}')
            mean=statistics.mean(vals)
            d=dict(zip(subset_keys,rest))
            seed_rows.append(dict(panel=panel,landscape_seed=seed,**d,mean_hv_pct=mean,n_cases=len(vals)))
            grouped[tuple(rest)].append((seed,mean))
        for rest,components in grouped.items():
            if set(seed for seed,_ in components)!={42,43,44}:raise FigureError('Missing one or more seeds for '+str(rest))
            means=[m for _,m in sorted(components)]
            agg_rows.append(dict(panel=panel,**dict(zip(subset_keys,rest)),mean_hv_pct=statistics.mean(means),
                sd_across_3_seed_means=statistics.stdev(means),n_seeds=3,
                n_cases_per_seed=expected_n,n_total_cases=3*expected_n))
    # Pairwise gs/ns knowledge gains in percentage POINTS relative to same-seed 50K reference.
    by={(r['prior_points'],r['method']):r for r in agg_rows if r['panel']=='trends'}
    gain=[]
    for size in SIZES:
        for family in ('gsMOBO','NSGA-II'):
            base=by[(size,f'{family} (no prior)')]['mean_hv_pct']
            with_=by[(size,f'{family} + landscape')]['mean_hv_pct']
            gain.append(dict(prior_points=size,family=family,
                             mean_delta_pp_vs_canonical_no_prior=with_-base,
                             denominator='same-seed/case 50K landscape HV',n_total_cases=75))
    return seed_rows,agg_rows,gain

def as_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True,exist_ok=True)
    keys=list(rows[0]);keys+=sorted(set().union(*(set(r) for r in rows))-set(keys))
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def draw(agg,gains,dest):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9.3,'axes.titlesize':10.7,
       'axes.labelsize':9,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
       'axes.linewidth':.75,'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':500})
    look={(r['panel'],r['prior_points'],r.get('species',''),r['method']):r for r in agg}
    def v(panel,size,method,species=''):
        return look[(panel,size,species,method)]['mean_hv_pct']
    def sd(panel,size,method,species=''):
        return look[(panel,size,species,method)]['sd_across_3_seed_means']
    fig,axs=plt.subplots(2,3,figsize=(15.3,8.0))
    axs=axs.flatten()
    def styling(ax,letter,title,foot=None):
        ax.spines['top'].set_visible(False);ax.spines['right'].set_visible(False)
        ax.grid(axis='y',alpha=.9,color='#E6E6E6',linewidth=.7,zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(direction='out',width=.6,length=3)
        ax.set_title(title,fontweight='bold',pad=14)
        ax.text(-.12,1.10,letter,transform=ax.transAxes,fontweight='bold',fontsize=13,va='top',ha='left')
        if foot:ax.text(.5,1.005,foot,transform=ax.transAxes,ha='center',va='bottom',fontsize=8,color='#444444')
    def trend(ax,family,letter):
        x=np.arange(4)
        names=('Landscape',f'{family} (no prior)',f'{family} + landscape')
        for name in names:
            y=np.array([v('trends',s,name) for s in SIZES])
            yerr=np.array([sd('trends',s,name) for s in SIZES])
            ax.errorbar(x,y,yerr=yerr,color=COLORS[name],marker='o',linewidth=2.0,
                        markersize=4.6,elinewidth=0.85,capsize=2.5,capthick=0.85,zorder=4)
        ax.axhline(100,linestyle='--',color='#8D8D8D',linewidth=.9,zorder=1)
        gs=[g for g in gains if g['family']==family]
        lowvals=[v('trends',s,m)-sd('trends',s,m) for s in SIZES for m in names]
        highvals=[v('trends',s,m)+sd('trends',s,m) for s in SIZES for m in names]
        ymin=min(lowvals);ymax=max(highvals);spread=max(15,ymax-ymin)
        ax.set_ylim(min(60,ymin-.12*spread),max(112,ymax+.12*spread))
        ax.set_xticks(x,[f'{s//1000}K' for s in SIZES]);ax.set_xlabel('Landscape size')
        ax.set_ylabel('Mean final HV (% of 50K landscape)')
        styling(ax,letter,f'{family}: landscape-size effect','Mean: 25 cases × 3 seeds')
        # Gain labels kept above curve but offset to avoid collisions.
        for j,g in enumerate(gs):
            y=v('trends',g['prior_points'],f'{family} + landscape') + sd('trends',g['prior_points'],f'{family} + landscape')
            xoff=7 if j==0 else -7 if j==3 else 0
            align='left' if j==0 else 'right' if j==3 else 'center'
            ax.annotate(f"{g['mean_delta_pp_vs_canonical_no_prior']:+.1f} pp",(j,y),
                  xytext=(xoff,10),textcoords='offset points',ha=align,fontsize=7.6,
                  color=COLORS[f'{family} + landscape'],fontweight='bold')
    trend(axs[0],'gsMOBO','A')
    trend(axs[1],'NSGA-II','B')
    order=[('C',5000,axs[2]),('D',10000,axs[3]),('E',25000,axs[4]),('F',50000,axs[5])]
    allbars=[r['mean_hv_pct']+r['sd_across_3_seed_means'] for r in agg if r['panel']=='species']
    upper=max(112,max(allbars)*1.08)
    for letter,size,ax in order:
        x=np.arange(5);width=.155
        for j,m in enumerate(METHODS):
            y=[v('species',size,m,sp) for sp in ORDER]
            yerr=[sd('species',size,m,sp) for sp in ORDER]
            ax.bar(x+(j-2)*width,y,width=width*.96,color=COLORS[m],zorder=3,
                   yerr=yerr,error_kw={'ecolor':'#494949','elinewidth':.65,
                                      'capsize':1.6,'capthick':.65})
        ax.axhline(100,color='#999999',linewidth=.8,linestyle='--',zorder=1)
        ax.set_xticks(x,[DISPLAY[sp] for sp in ORDER],rotation=0)
        ax.set_ylabel('Mean final HV (% of own-size landscape)')
        ax.set_ylim(0,upper)
        styling(ax,letter,f'{size//1000}K landscape','Mean: 5 problems × 3 seeds')
    handles=[Patch(facecolor=COLORS[m],label=m) for m in METHODS]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.013),
               ncol=5,frameon=False,fontsize=9,handlelength=1.3,columnspacing=1.5)
    fig.text(.5,.09,'Error bars: SD across the three seed-level means',ha='center',fontsize=8,color='#555555')
    fig.subplots_adjust(left=.065,right=.99,top=.91,bottom=.145,hspace=.49,wspace=.33)
    for ext in ('pdf','svg','png'):
        fig.savefig(dest/f'{FILE_STEM}.{ext}',bbox_inches='tight',facecolor='white',
             **({'dpi':500} if ext=='png' else {}))
    plt.close(fig)

def run(archive:Path,figure:Path,simulate_data=None):
    if simulate_data is None:
        frozen,readerpath=load_frozen_reader(archive)
        data,origin,cases,missing_runtime=frozen.load_data(archive)
        controls,control_audit=canonical_control_hvs(frozen,archive,data,cases)
    else:
        data,cases=simulate_data
        controls=None;control_audit=[]
        readerpath=Path('SYNTHETIC_TEST_READER')
    rows,audit=build_casewise(data,cases,canonical_override=controls)
    seedstat,agg,gains=build_stats(rows)
    dst=figure/'outputs'
    dst.mkdir(parents=True,exist_ok=True)
    plots=dst/'plot_data'
    as_csv(plots/'CASEWISE_1500_METHOD_OBSERVATIONS.csv',rows)
    as_csv(plots/'PER_SEED_AGGREGATES.csv',seedstat)
    as_csv(plots/'THREE_SEED_AGGREGATES.csv',agg)
    as_csv(plots/'KNOWLEDGE_GAIN_TRENDS_50K_REFERENCE.csv',gains)
    as_csv(plots/'NO_PRIOR_CANONICAL_VS_CAMPAIGN_AUDIT.csv',audit)
    as_csv(plots/'CANONICAL_NO_PRIOR_EXACT_FIELD_AUDIT.csv',control_audit)
    draw(agg,gains,dst)
    differing=[a for a in audit if abs(a['difference'])>1e-9]
    fallbacks=[a for a in control_audit if 'FALLBACK' in a['historical_field_status']]
    rep={'status':'PASS','code_lineage':'GEM Landscape-Knowledge Figure v1.5.1 (historical) adapted to archived 3-seed scientific reader',
      'reader_path':str(readerpath),'reader_sha256':(hashlib.sha256(readerpath.read_bytes()).hexdigest() if readerpath.is_file() else 'SYNTHETIC_TEST'),
      'historical_source_provenance':'V1 identified GEM_Landscape_Knowledge_Figure_v1.5.1; exact original source not included in generator',
      'seeds':list(SEEDS),'sizes':list(SIZES),'biological_case_count':25,'total_case_seed_size_comparisons':300,
      'method_case_observations':len(rows),'endpoint_calls':128,'species_order':list(ORDER),
      'new_order':'A gsMOBO trend; B NSGA-II trend; C 5K; D 10K; E 25K; F 50K',
      'trend_denominator':'own seed-case 50K frozen landscape HV',
      'species_denominator':'own seed-case same-size frozen landscape HV',
      'aggregation_trend':'25 species-problem cases per seed, then average 3 seed means (75 case-seed units per size)',
      'aggregation_species':'5 problems per species per seed, then average 3 seed means (15 case-seed units per species/size)',
      'no_prior_semantics':'gsMOBO from same seed/case 10K run final_self_hv (fallback audited); NSGA-II from same seed/case 5K; repeated across sizes',
      'no_prior_saved_differences_above_1e_9':len(differing),
      'historical_exact_baseline_field_fallbacks':len(fallbacks),
      'canonical_field_audit':'plot_data/CANONICAL_NO_PRIOR_EXACT_FIELD_AUDIT.csv',
      'note_on_differences':'Independent no-prior job endpoints from other campaign sizes are exported in audit, not silently substituted.',
      'sd_across_seeds':'Plotted error bars: sample SD across 3 independent landscape seed-level means, NOT SEM or pooled-case SD',
      'fba_rerun':False,'original_archive_results_modified':False,
      'files':{'figure':FILE_STEM+'.pdf','casewise_csv':'plot_data/CASEWISE_1500_METHOD_OBSERVATIONS.csv'}}
    (dst/'RUN_REPORT.json').write_text(json.dumps(rep,indent=2)+'\n')
    (dst/'README_RESULTS.txt').write_text('Six-panel landscape-knowledge figure using seeds 42, 43, 44.\n'
      'A/B: 50K-reference mean over 25 cases x 3 seeds. C-F: same-size-reference species mean over 5 problems x 3 seeds.\n'
      'The canonical no-prior endpoint is landscape-size-independent; per-size saved alternatives audited.\n'
      'No optimizer or FBA rerun. See RUN_REPORT.json for code provenance.\n'
      'Error bars: SD of three seed-level means. Results are a single set, no timestamped run duplicates.\n'
      'Original archive results/other figures unaffected.\n')
    return dst,rep

def main(argv=None):
    ap=argparse.ArgumentParser(description='Generate original landscape-knowledge plot using 3 archived seeds')
    ap.add_argument('--archive',type=Path)
    ap.add_argument('--figure-dir',type=Path,help='Explicit output Figure 2 folder (used by safe staged installer)')
    args=ap.parse_args(argv)
    try:
        archive=archive_root(Path(__file__).resolve(),args.archive)
        figure=args.figure_dir.resolve() if args.figure_dir else archive/'Figures_and_Tables/Main/Figure_02_Landscape_Knowledge_ThreeSeeds'
        if not figure.is_dir():raise FigureError('Expected installed figure directory missing: '+str(figure))
        dst,rep=run(archive,figure)
    except (FigureError,OSError,KeyError,ValueError,RuntimeError) as exc:
        print('SAFELY BLOCKED:',exc,file=sys.stderr)
        return 2
    print('PASS: '+str(dst/(FILE_STEM+'.pdf')))
    print('Error bars: sample SD across 3 seed-level means; 5 problems/3 seeds per species.')
    print('Validated 300 case-seed-size comparisons; 5 methods each; no scientific source modified.')
    print('One clean output set: PDF, SVG, PNG, casewise/per-seed CSVs and provenance audits.')
    return 0

if __name__=='__main__':sys.exit(main())
