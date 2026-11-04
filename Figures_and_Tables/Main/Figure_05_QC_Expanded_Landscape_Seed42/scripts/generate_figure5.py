#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, importlib.util, json, math, os, re, sys, shutil, tempfile
from collections import defaultdict
from datetime import datetime, timezone
from zipfile import ZipFile, ZIP_DEFLATED
from pathlib import Path

SEED=42
PRIOR_POINTS=50000
CALLS=128
FIGURE_NAME='Figure_5_QC_Expanded_Landscape_Seed42'
SPECIES_ORDER=('yeast','pputida','cglutamicum','cho','ecoli_antiepex')
DISPLAY={
    'yeast': ('S. cerevisiae','Ethanol'),
    'pputida': ('P. putida','C8:0 mcl-PHA'),
    'cglutamicum': ('C. glutamicum','L-glutamate'),
    'cho': ('CHO','IgG'),
    'ecoli_antiepex': ('E. coli antiEpEX-scFv','antiEpEX-scFv'),
}
PRODUCT_KEYWORDS={
    'yeast': ('ethanol',),
    'pputida': ('c8', 'pha', 'mcl'),
    'cglutamicum': ('glutamate',),
    'cho': ('igg',),
    'ecoli_antiepex': ('antiepex','scfv'),
}
CASE_ALIASES={
    'yeast':'yeast','s_cerevisiae':'yeast','scerevisiae':'yeast','s.cerevisiae':'yeast',
    'pputida':'pputida','p_putida':'pputida','p.putida':'pputida',
    'cglutamicum':'cglutamicum','c_glutamicum':'cglutamicum','c.glutamicum':'cglutamicum',
    'cho':'cho',
    'ecoli_antiepex':'ecoli_antiepex','e_coli_antiepex':'ecoli_antiepex','ecoliantiepex':'ecoli_antiepex','e.coli_antiepex':'ecoli_antiepex',
}
SELECTED_METHOD_TO_ALGO={
    'gsMOBO + SPARC':'DEV_GSMOBO50K_SPARC_A',
    'NSGA-II + SPARC':'DEV_NSGA50K_SPARC_LEGACY_OPERATOR',
}
LANDSCAPE_SPECIES_DIR={
    'yeast':'yeast_50k','pputida':'pputida_50k','cglutamicum':'cglutamicum_50k','cho':'cho_50k','ecoli_antiepex':'ecoli_antiepex_50k'
}

class FigureError(RuntimeError):
    pass


def read_csv(path: Path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def load_module(path: Path, name: str):
    if not path.is_file() or path.is_symlink():
        raise FigureError('Missing frozen module: '+str(path))
    spec=importlib.util.spec_from_file_location(name, path)
    mod=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_case_species(case: str) -> str:
    key=(case.split('__')[0].strip().lower().replace('-','_').replace(' ','_'))
    if key in CASE_ALIASES:
        return CASE_ALIASES[key]
    for alias,species in CASE_ALIASES.items():
        if alias in key:
            return species
    raise FigureError('Could not map case to species: '+case)


def species_case_score(species: str, case: str) -> tuple[int,int,str]:
    c=case.lower().replace('-', '_')
    kw=PRODUCT_KEYWORDS[species]
    hits=sum(int(k in c) for k in kw)
    exact=int(species in c)
    # higher hits better, shorter names tie-break, then lexicographic
    return (hits, exact, -len(case))


def is_growth_product_case(case: str, job: Path | None = None) -> bool:
    """Confirm *problem class*, never choose a case based on product-name keywords alone."""
    part=case.split('__', 1)[1] if '__' in case else case
    norm=normalize_name(part)
    if norm in {'gp','gxp','growth_product','growth_vs_product','growth_x_product',
                'growth_and_product','growth_product_pareto','biomass_product',
                'biomass_vs_product','biomass_x_product'}:
        return True
    if re.search(r'(^|_)(growth|biomass)_(vs_|x_|and_)?product($|_)', norm):
        return True
    if job:
        for file in [job/'job.json',job/'config.json',job/'run_metadata.json',job/'scientific_results'/'problem_definition.json']:
            if not file.is_file() or file.is_symlink() or file.stat().st_size>2_000_000:
                continue
            try:
                obj=json.loads(file.read_text(encoding='utf-8'))
            except (ValueError,UnicodeError):
                continue
            for key in ('problem_type','problem_name','problem_id','optimization_problem','objective_pair','optimization_task'):
                val=obj.get(key) if isinstance(obj,dict) else None
                if isinstance(val,str) and re.search(r'(growth|biomass)',val,re.I) and re.search(r'product',val,re.I):
                    return True
            # Objective pairs count only if explicit and exactly two named growth/product objectives.
            for key in ('objective_names','objectives'):
                val=obj.get(key) if isinstance(obj,dict) else None
                if isinstance(val,list) and len(val)==2:
                    names=' '.join(str(x.get('name',x.get('id','')) if isinstance(x,dict) else x) for x in val).lower()
                    if re.search(r'growth|biomass',names) and 'product' in names:
                        return True
    return False


def choose_representative_cases(selector_rows, campaign: Path, audit_output: Path | None = None):
    by_species=defaultdict(list)
    case_audit=[]
    for r in selector_rows:
        species=parse_case_species(r['case'])
        method=r['selected_method']
        algo=SELECTED_METHOD_TO_ALGO.get(method)
        job=campaign/'jobs'/r['case']/(algo or 'MISSING_SELECTED_ALGORITHM')
        matched=is_growth_product_case(r['case'],job)
        case_audit.append({
            'species':species,'case':r['case'],'seed':SEED,'prior_points':PRIOR_POINTS,
            'selected_method':method,'selected_algorithm':algo or '',
            'growth_product_evidence':'YES' if matched else 'NO',
            'selected_job_relpath':str(job)
        })
        if matched:by_species[species].append(r)
    if audit_output:
        write_csv(audit_output,case_audit)
    chosen={}
    problems=[]
    for sp in SPECIES_ORDER:
        rows=by_species.get(sp, [])
        if len(rows)!=1:
            problems.append(f'{sp}: {len(rows)} verified Growth × Product cases (expected 1)')
        else:
            chosen[sp]=rows[0]
    if problems:
        raise FigureError('Cannot uniquely identify Growth × Product cases. '+ '; '.join(problems)+
                          '. Case evidence written to CASE_SELECTION_AUDIT.csv. No figure generated.')
    return chosen

def qc_selector_predictions(archive: Path):
    """Read the SAVED frozen QC-only predictions from the already-generated Figure 3.

    Never import selector_ghi.py, retrain the model, or choose an algorithm independently.
    """
    output=archive/'Figures_and_Tables'/'Main'/'Figure_03_ThreeSeed_Optimization_Aggregate'/'outputs'/'FIGURE3_ALL_NINE_PANELS_V4'
    candidates=list(output.glob('run_*/plot_data/QC_SELECTOR_PRIMARY_PREDICTIONS.csv'))
    if not candidates:
        candidates=list(output.glob('*/plot_data/QC_SELECTOR_PRIMARY_PREDICTIONS.csv'))
    if not candidates:
        raise FigureError('Missing saved Figure 3 QC_SELECTOR_PRIMARY_PREDICTIONS.csv. '
                          'Expected under Figure_03_ThreeSeed_Optimization_Aggregate/outputs/FIGURE3_ALL_NINE_PANELS_V4/<run>/plot_data/. '
                          'The previous module is not required.')
    # Prefer existing run containing the current Figure 3 PDF, newest modification first.
    candidates.sort(key=lambda p:(not (p.parent.parent/'Figure_3_All_Nine_Panels_ThreeSeeds_V4.pdf').exists(),
                                  -p.stat().st_mtime, str(p)))
    chosen=candidates[0]
    rows=list(read_csv(chosen))
    required={'case','landscape_seed','prior_points','selected_method','selected_hv','landscape_hv'}
    if not rows or not required.issubset(rows[0]):
        raise FigureError('Saved QC selector prediction file missing required fields: '+str(sorted(required-set(rows[0]) if rows else required)))
    selected=[r for r in rows if int(r['landscape_seed'])==SEED and int(r['prior_points'])==PRIOR_POINTS]
    if len(selected)!=25 or len({r['case'] for r in selected})!=25:
        raise FigureError(f'Saved Figure 3 must have 25 distinct cases for seed42/50K, found {len(selected)}')
    for r in selected:
        if r['selected_method'] not in SELECTED_METHOD_TO_ALGO:
            raise FigureError('QC-only selection not supported: '+r['selected_method'])
        base=safe_float(r['landscape_hv']); target=safe_float(r['selected_hv'])
        if base is None or target is None or base<=0 or target<0:
            raise FigureError('Invalid selected/landscape HV for '+r['case'])
    return selected,chosen

def campaign_name(seed, size):
    k=size//1000
    if seed==42:
        return f'GEM_ALL25_FINAL10_PRIOR{k}K_128'
    return f'GEM_ALL25_FINAL10_PRIOR{k}K_LS{seed}_AS{{43:20260823,44:20260824}}[seed]_128'


def campaign_root(archive: Path, seed: int, size: int) -> Path:
    p=archive/'Results'/'Optimizations'/f'landscape_seed_{seed}'/f'prior_{size}'/campaign_name(seed,size)
    if not p.is_dir() or p.is_symlink():
        raise FigureError('Missing canonical optimization campaign: '+str(p.relative_to(archive)))
    if not (p/'CAMPAIGN_COMPLETION.json').is_file():
        raise FigureError('Missing canonical campaign completion marker: '+str(p.relative_to(archive)))
    return p


def discover_landscape_root(archive: Path, species: str) -> Path:
    base=archive/'Results'/'Landscape_Generation'
    opts=[]
    for p in base.rglob('outputs_multiseed_50k'):
        root=p/f'seed{SEED}'/LANDSCAPE_SPECIES_DIR[species]
        if (root/'chunks'/'results').is_dir():
            opts.append(root)
    if not opts:
        raise FigureError(f'No native 50K landscape found for seed{SEED} {species}')
    opts=sorted(opts, key=lambda p: ('GEM_Learning_Engine' not in str(p), len(str(p)), str(p)))
    return opts[0]


def load_landscape(root: Path):
    import pyarrow.parquet as pq
    files=sorted((root/'chunks'/'results').glob('results_chunk_*.parquet'))
    x=[]; y=[]; feas=[]
    for p in files:
        table=pq.read_table(p, columns=['biomass_flux','product_flux','feasible'])
        d=table.to_pydict(); n=len(d['biomass_flux'])
        for i in range(n):
            a=d['biomass_flux'][i]; b=d['product_flux'][i]; f=d['feasible'][i]
            try:
                a=float(a); b=float(b)
            except (TypeError,ValueError):
                a=b=math.nan
            ff=(f is True) or str(f).strip().lower() in {'true','1','yes','t'}
            x.append(a); y.append(b); feas.append(ff and math.isfinite(a) and math.isfinite(b))
    if len(x)!=50000:
        raise FigureError(f'Expected 50,000 landscape points in {root.name}, got {len(x)}')
    return {'x':x,'y':y,'feasible':feas}


def safe_float(x):
    try:
        y=float(x)
        return y if math.isfinite(y) else None
    except (TypeError,ValueError):
        return None


def normalize_name(name: str) -> str:
    return re.sub(r'[^a-z0-9]+','_', str(name).strip().lower()).strip('_')


def find_xy_columns(columns):
    """Accept measured PHYSICAL flux coordinates only, not arbitrary normalized objectives."""
    rev=defaultdict(list)
    for c in columns:rev[normalize_name(c)].append(c)
    for a,b in [
        ('biomass_flux','product_flux'),('growth_rate','product_flux'),
        ('growth_flux','product_flux'),('biomass_reaction_flux','product_flux'),
        ('biomass_flux','target_product_flux'),('growth_rate','target_product_flux'),
        ('biomass_flux','product_reaction_flux'),('growth_rate','product_reaction_flux'),
    ]:
        if a in rev and b in rev:
            return rev[a][0], rev[b][0], 'PHYSICAL_FLUX_PAIR'
    return None

def candidate_score(path: Path, row_count: int, col_mode: str, coverage: float, range_score: float):
    # Prioritize completed exact evaluated calls over suggested/surrogate candidates.
    n=path.name.lower()
    s=100 if n=='calls.csv' else 80 if 'evaluated' in n else 75 if 'calls' in n else 70 if 'pareto' in n else 60 if 'archive' in n else 10
    if 120<=row_count<=150: s+=30
    elif 10<=row_count<=1000:s+=10
    s+=int(10*coverage)+int(10*range_score)
    return s

def row_is_feasible(row):
    fields={normalize_name(k):v for k,v in row.items()}
    for key in ('feasible','is_feasible','fba_feasible','valid_fba'):
        if key in fields and fields[key] is not None:
            val=fields[key]
            if isinstance(val,bool):
                return val
            return str(val).strip().lower() in {'true','1','yes','t'}
    return True  # no recorded feasibility field; require finite physical flux coordinates


def read_csv_candidate(path: Path, xcol: str, ycol: str):
    pts=[]
    with path.open('r',encoding='utf-8-sig',newline='') as f:
        rd=csv.DictReader(f)
        for row in rd:
            if not row_is_feasible(row):continue
            x=safe_float(row.get(xcol)); y=safe_float(row.get(ycol))
            if x is not None and y is not None:
                pts.append((x,y))
    return pts

def read_parquet_candidate(path: Path, xcol: str, ycol: str):
    import pyarrow.parquet as pq
    cols=list(pq.read_schema(path).names)
    feasible_cols=[c for c in cols if normalize_name(c) in {'feasible','is_feasible','fba_feasible','valid_fba'}]
    selected=[xcol,ycol]+feasible_cols
    table=pq.read_table(path, columns=selected)
    d=table.to_pydict(); pts=[]
    n=len(d[xcol])
    for i in range(n):
        if not row_is_feasible({c:d[c][i] for c in feasible_cols}):continue
        x=safe_float(d[xcol][i]); y=safe_float(d[ycol][i])
        if x is not None and y is not None:pts.append((x,y))
    return pts

def extract_points_from_json_obj(obj, path='root'):
    found=[]
    if isinstance(obj,list):
        if obj and all(isinstance(v,dict) for v in obj):
            cols=list({k for row in obj for k in row})
            pair=find_xy_columns(cols)
            if pair:
                a,b,mode=pair; pts=[]
                for row in obj:
                    if not row_is_feasible(row):continue
                    x=safe_float(row.get(a));y=safe_float(row.get(b))
                    if x is not None and y is not None:pts.append((x,y))
                if pts:found.append((path,pts,mode))
        for i,v in enumerate(obj[:50]):
            found.extend(extract_points_from_json_obj(v, f'{path}[{i}]'))
    elif isinstance(obj,dict):
        pair=find_xy_columns(obj.keys())
        if pair:
            a,b,mode=pair
            xx=obj.get(a); yy=obj.get(b)
            if isinstance(xx,list) and isinstance(yy,list) and len(xx)==len(yy):
                pts=[]
                feas_key=next((k for k in obj if normalize_name(k) in {'feasible','is_feasible','fba_feasible','valid_fba'}),None)
                for i,(xv,yv) in enumerate(zip(xx,yy)):
                    if feas_key and isinstance(obj[feas_key],list) and not row_is_feasible({feas_key:obj[feas_key][i]}):continue
                    x=safe_float(xv);y=safe_float(yv)
                    if x is not None and y is not None:pts.append((x,y))
                if pts:found.append((path,pts,mode))
        for k,v in list(obj.items())[:50]:
            found.extend(extract_points_from_json_obj(v,f'{path}.{k}'))
    return found

def extract_selected_points(job: Path, x_bounds, y_bounds, audit_path: Path | None = None):
    """Find *explicitly physical* native optimizer objective coordinates.

    Excludes surrogate predictions, normalized generic objective columns, and unrelated
    training/landscape tables. If uncertain, save an audit and fail closed.
    """
    x_min,x_max=x_bounds; y_min,y_max=y_bounds
    candidates=[]; audit=[]
    allowed_suffix={'.csv','.parquet','.json'}
    for root in (job/'scientific_results', job):
        if not root.is_dir():continue
        for p in root.rglob('*'):
            if not p.is_file() or p.is_symlink() or p.suffix.lower() not in allowed_suffix:
                continue
            # Reject files likely to contain surrogate predictions rather than evaluated fluxes.
            lower=str(p.relative_to(job)).lower()
            if any(t in lower for t in ('surrogate','training','predicted','proposal','candidate',
                                        'uncertainty','model_fit','crossval','preopt','landscape')):
                continue
            if p.stat().st_size>40_000_000:continue
            mode=''; pts=[]; cols=[]
            try:
                if p.suffix.lower()=='.csv':
                    with p.open('r',encoding='utf-8-sig',newline='') as f:
                        rd=csv.reader(f); cols=next(rd, [])
                    pair=find_xy_columns(cols)
                    if pair:
                        pts=read_csv_candidate(p,pair[0],pair[1]); mode=pair[2]
                elif p.suffix.lower()=='.parquet':
                    import pyarrow.parquet as pq
                    cols=list(pq.read_schema(p).names)
                    pair=find_xy_columns(cols)
                    if pair:
                        pts=read_parquet_candidate(p,pair[0],pair[1]); mode=pair[2]
                else:
                    ob=json.loads(p.read_text(encoding='utf8'))
                    cols=list(ob.keys()) if isinstance(ob,dict) else []
                    found=extract_points_from_json_obj(ob,p.name)
                    if found:
                        _,pts,mode=max(found,key=lambda t:len(t[1]))
            except Exception as e:
                audit.append({'file':str(p.relative_to(job)),'columns':str(cols[:20]),
                              'valid_points':0,'status':'READ_ERROR','note':str(e)[:180]})
                continue
            status=('ACCEPTED_PHYSICAL_COLUMNS' if pts else 'NO_PHYSICAL_FLUX_PAIR')
            audit.append({'file':str(p.relative_to(job)),'columns':str(cols[:20]),
                          'valid_points':len(pts),'status':status,'note':mode})
            if len(pts)<10: continue
            unique={ (round(x,12),round(y,12)):(x,y) for x,y in pts }
            points=list(unique.values())
            inside=sum((x_min-0.2*abs(x_max-x_min)<=x<=x_max+0.5*abs(x_max-x_min)) and
                       (y_min-0.2*abs(y_max-y_min)<=y<=y_max+0.5*abs(y_max-y_min)) for x,y in points)
            fraction=inside/max(1,len(points))
            score=candidate_score(p,len(pts),mode,1.0,fraction)
            candidates.append((score,p,points,len(pts),fraction))
    if audit_path:
        write_csv(audit_path,audit or [{'file':'','columns':'','valid_points':0,'status':'NO_CANDIDATES','note':''}])
    if not candidates:
        raise FigureError('No verified physical growth/product flux point columns in selected job: '+str(job)+
                          '. See SOURCE_CANDIDATES_*.csv; generic objectives/normalized scores cannot be substituted.')
    candidates.sort(key=lambda x:(-x[0],-x[3],str(x[1])))
    best=candidates[0]
    # If two similarly qualified highest-quality source files disagree, do not silently pick one.
    if len(candidates)>1 and candidates[1][0]>=best[0]-2:
        set0={(round(x,9),round(y,9)) for x,y in best[2]}
        set1={(round(x,9),round(y,9)) for x,y in candidates[1][2]}
        if set0 != set1:
            raise FigureError('Ambiguous equally plausible evaluated-point tables: '+
                              str(best[1])+' versus '+str(candidates[1][1])+'. See source audit.')
    return {'path':best[1], 'points':best[2], 'score':best[0],
            'mode':'PHYSICAL_FLUX_PAIR', 'n_points':len(best[2])}

def pareto_frontier(points):
    buckets={}
    for x,y in points:
        if x in buckets:
            if y > buckets[x]: buckets[x]=y
        else:
            buckets[x]=y
    pts=sorted((x,y) for x,y in buckets.items())
    keep=[]; best_y=-float('inf')
    for x,y in reversed(pts):
        if y > best_y + 1e-12:
            keep.append((x,y)); best_y=y
    keep.reverse()
    return keep


def build_panels(archive: Path, audit_dir: Path):
    selector_rows,prediction_source=qc_selector_predictions(archive)
    camp=campaign_root(archive, SEED, PRIOR_POINTS)
    representatives=choose_representative_cases(selector_rows,camp,audit_dir/'CASE_SELECTION_AUDIT.csv')
    panels=[]; plot_rows=[]; panel_audit=[]
    for species in SPECIES_ORDER:
        row=representatives[species]
        selected_method=row['selected_method'];algo=SELECTED_METHOD_TO_ALGO[selected_method]
        case=row['case'];job=camp/'jobs'/case/algo
        if not job.is_dir() or job.is_symlink():
            raise FigureError('Missing selected optimizer job: '+str(job.relative_to(archive)))
        land_root=discover_landscape_root(archive,species)
        land=load_landscape(land_root)
        feas=[(x,y) for x,y,f in zip(land['x'],land['y'],land['feasible']) if f]
        if len(feas)<10:raise FigureError('No feasible native landscape for '+species)
        xs=[x for x,y in feas];ys=[y for x,y in feas]
        opt=extract_selected_points(job,(min(xs),max(xs)),(min(ys),max(ys)),
                                    audit_dir/f'SOURCE_CANDIDATES_{species}.csv')
        # Use actual evaluated fluxes to compute a union frontier. Never deform coordinates.
        baseline=pareto_frontier(feas)
        union=feas+opt['points']
        front=pareto_frontier(union)
        native_base= {(round(x,10),round(y,10)) for x,y in baseline}
        native_opt= {(round(x,10),round(y,10)) for x,y in opt['points']}
        frontier_opt_points=sum((round(x,10),round(y,10)) in native_opt and
                                 (round(x,10),round(y,10)) not in native_base for x,y in front)
        delta=100*(float(row['selected_hv'])/float(row['landscape_hv'])-1)
        if delta>0.001 and frontier_opt_points<1:
            raise FigureError(f'{species} native optimizer table does not expand the growth/product frontier, '
                              f'but saved selected HV claims +{delta:.3f}%. Check raw-coordinate provenance; refusing synthetic or mismatched figure.')
        record={
            'species':species,'display':DISPLAY[species][0], 'product':DISPLAY[species][1],
            'landscape':land,'frontier':front,'baseline_frontier':baseline,
            'selected_points':opt['points'],'selected_method':selected_method,
            'selected_algorithm':algo,'selected_case':case,
            'selected_source':str(opt['path'].relative_to(archive)),
            'prediction_source':str(prediction_source.relative_to(archive)),
            'landscape_root':str(land_root.relative_to(archive)),
            'selected_delta_hv_pct':delta,
            'selected_frontier_new_points':frontier_opt_points,
        }
        panels.append(record)
        panel_audit.append({k:record[k] for k in (
            'species','selected_case','selected_method','selected_algorithm','selected_source',
            'prediction_source','landscape_root','selected_delta_hv_pct','selected_frontier_new_points')})
        for series,points in [('empirical_pareto_front',front),('original_pareto_front',baseline),('selected_optimizer_points',opt['points'])]:
            for i,(x,y) in enumerate(points,start=1):
                plot_rows.append({'species':species,'series':series,'point_index':i,
                                  'growth_rate':x,'product_flux':y,'selected_method':selected_method,
                                  'selected_algorithm':algo,'case':case,'source_table':str(opt['path'].relative_to(archive))})
        print(f'PASS {species}: {case}, QC→{selected_method}, {len(opt["points"])} real optimizer points; '
              f'{frontier_opt_points} new frontier points',flush=True)
    write_csv(audit_dir/'SELECTED_JOB_AND_COORDINATE_AUDIT.csv',panel_audit)
    return panels,plot_rows

def draw(panels, destination: Path):
    """Render verified native points in the existing 5-panel Figure 5 layout.

    Styling only: more distinct scatter/frontier palette and an entirely
    contained legend. No selection, objective, or Pareto calculations change.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10.5,
        'axes.labelsize':9.0,'axes.linewidth':0.8,'xtick.labelsize':8.5,'ytick.labelsize':8.5,
        'legend.fontsize':7.5,'figure.dpi':180,'savefig.dpi':300,'pdf.fonttype':42,'ps.fonttype':42})
    # Sharper than V3 while maintaining the established blue/orange visual language.
    landscape_color = '#238DCA'
    frontier_line_color = '#1265AD'
    frontier_marker_color = '#F06A00'

    fig=plt.figure(figsize=(13.8,8.0), facecolor='white')
    gs=fig.add_gridspec(2,3, left=0.05, right=0.98, top=0.96, bottom=0.08,
                        wspace=0.28, hspace=0.30)
    axes=[fig.add_subplot(gs[0,0]), fig.add_subplot(gs[0,1]), fig.add_subplot(gs[0,2]),
          fig.add_subplot(gs[1,0]), fig.add_subplot(gs[1,1])]
    legend_ax=fig.add_subplot(gs[1,2])
    letters=list('ABCDE')
    for ax,panel,letter in zip(axes, panels, letters):
        land=panel['landscape']
        xf=[x for x,f in zip(land['x'], land['feasible']) if f]
        yf=[y for y,f in zip(land['y'], land['feasible']) if f]
        ax.set_facecolor('white')
        ax.scatter(xf, yf, s=3.8, color=landscape_color, alpha=0.44,
                   edgecolors='none', rasterized=True, zorder=2)
        fx=[p[0] for p in panel['frontier']]
        fy=[p[1] for p in panel['frontier']]
        ax.plot(fx, fy, color=frontier_line_color, linewidth=1.4, zorder=4)
        ax.scatter(fx, fy, s=15, color=frontier_marker_color, edgecolors='none', zorder=5)
        ax.text(-0.17, 1.06, letter, transform=ax.transAxes, fontsize=13,
                fontweight='bold', ha='left', va='bottom')
        ax.text(-0.04, 1.04, panel['display'], transform=ax.transAxes,
                fontsize=13, fontweight='bold', ha='left', va='bottom')
        ax.set_xlabel(r'Growth rate ($h^{-1}$)', fontsize=11, fontweight='bold')
        ax.set_ylabel(r'Product flux (mmol gDW$^{-1}$ h$^{-1}$)',
                      fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#666666')
        ax.spines['bottom'].set_color('#666666')
        ax.tick_params(color='#666666', labelsize=8)

    legend_ax.set_axis_off()
    # A real bounded panel, with the text wrapping fitted to this cell.
    border=patches.Rectangle((0.012,0.012),0.976,0.976,transform=legend_ax.transAxes,
                             facecolor='white',edgecolor='#BBC5CF',linewidth=1.0,
                             zorder=0)
    legend_ax.add_patch(border)
    legend_text=[]
    def add(text,x,y,size=9.4,bold=False,ha='left'):
        obj=legend_ax.text(x,y,text,fontsize=size,weight='bold' if bold else 'normal',
                           ha=ha,va='center',transform=legend_ax.transAxes,zorder=3)
        legend_text.append(obj)
        return obj
    add('Legend',0.055,0.940,13.6,True)
    legend_ax.scatter([0.094],[0.856],s=42,color=landscape_color,
                      transform=legend_ax.transAxes,zorder=3)
    add('Landscape (50,000 exact-FBA designs)',0.158,0.856,9.4,True)
    legend_ax.scatter([0.094],[0.783],s=42,color=frontier_marker_color,
                      transform=legend_ax.transAxes,zorder=3)
    add('Empirical Pareto front',0.158,0.783,9.4,True)

    add('Species and product',0.055,0.691,11.8,True)
    names=[('S. cerevisiae','Ethanol'),('P. putida','C8:0 mcl-PHA'),
           ('C. glutamicum','L-glutamate'),('CHO','IgG'),
           ('E. coli antiEpEX-scFv','antiEpEX-scFv')]
    for idx,(label,product) in enumerate(names):
        yy=0.619-0.069*idx
        name_size=8.75 if idx==4 else 9.3
        add(label+':',0.07,yy,name_size,True)
        add(product,0.69,yy,9.0)

    add('Notes',0.055,0.252,11.8,True)
    add('All five panels use the same axes:',0.070,0.194,9.1,True)
    # Two compact lines instead of V3's long overflowing axis-description line.
    add(r'Growth rate ($h^{-1}$) vs product flux',0.078,0.141,8.8)
    add(r'(mmol gDW$^{-1}$ h$^{-1}$)',0.078,0.088,8.8)
    add('Higher values are better for both objectives.',0.078,0.036,8.3)

    # Check every legend string really fits inside the visible rectangle.
    fig.canvas.draw()
    renderer=fig.canvas.get_renderer()
    abox=legend_ax.get_window_extent(renderer)
    left=abox.x0+0.025*abox.width
    right=abox.x0+0.975*abox.width
    bottom=abox.y0+0.018*abox.height
    top=abox.y0+0.982*abox.height
    for item in legend_text:
        bounds=item.get_window_extent(renderer)
        if bounds.x0 < left - 2 or bounds.x1 > right + 2 or bounds.y0 < bottom - 2 or bounds.y1 > top + 2:
            raise FigureError('Legend text exceeds its boundary: '+item.get_text())

    destination.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png','svg'):
        fig.savefig(destination/f'{FIGURE_NAME}.{ext}',bbox_inches='tight',pad_inches=0.06,
                    facecolor=fig.get_facecolor(),**({'dpi':300} if ext=='png' else {'dpi':200}))
    plt.close(fig)


def compute(archive: Path, figure_root: Path):
    # Figure 5 output path is canonical. Keep previously generated files safe
    # until all new PDFs/SVGs/PNGs have rendered and passed layout checks.
    out=figure_root/'outputs'
    audit_dir=out/'plot_data'
    audit_dir.mkdir(parents=True,exist_ok=True)
    try:
        panels,plot_rows=build_panels(archive,audit_dir)
        staging=Path(tempfile.mkdtemp(prefix='.fig5_render_',dir=out))
        try:
            draw(panels,staging)
            for ext in ('pdf','png','svg'):
                final=out/f'{FIGURE_NAME}.{ext}'
                os.replace(staging/f'{FIGURE_NAME}.{ext}',final)
        finally:
            shutil.rmtree(staging,ignore_errors=True)
        write_csv(audit_dir/'FIGURE5_EMPIRICAL_EXPANDED_FRONTIERS.csv',plot_rows)
        metadata={
            'figure':'Figure 5','seed':SEED,'prior_points':PRIOR_POINTS,
            'scientific_basis':'Native seed42 50K landscape union with actual evaluated raw physical flux coordinates from the optimizer job chosen by existing saved Figure3 QC-only predictions, for the growth×product case per species',
            'no_synthetic_points':True,'source_folders_modified':False,
            'generated_utc':datetime.now(timezone.utc).isoformat(),
            'panels':[{k:p[k] for k in ('species','selected_case','selected_method','selected_algorithm','selected_source','prediction_source','landscape_root','selected_delta_hv_pct','selected_frontier_new_points')} for p in panels]
        }
        (audit_dir/'FIGURE5_METADATA.json').write_text(json.dumps(metadata,indent=2)+'\n')
        (out/'FIGURE5_DIAGNOSTIC.txt').unlink(missing_ok=True)
        print('SUCCESS:',out/f'{FIGURE_NAME}.pdf')
        return out
    except Exception as e:
        diagnostic=out/'FIGURE5_DIAGNOSTIC.txt'
        diagnostic.write_text('FIGURE 5 NOT EXPORTED (empirical evidence incomplete): '+str(e)+'\n')
        bundle=out/'FIGURE5_SOURCE_AUDIT.zip'
        with ZipFile(bundle,'w',ZIP_DEFLATED) as z:
            z.write(diagnostic,diagnostic.name)
            for f in sorted(audit_dir.glob('*.csv')):
                if f.stat().st_size<2_000_000:
                    z.write(f,'plot_data/'+f.name)
        print('FIGURE 5 BLOCKED:', e, file=sys.stderr)
        print('DIAGNOSTIC PACKAGE:',bundle, file=sys.stderr)
        raise

def main(argv=None):
    ap=argparse.ArgumentParser(description='Generate Figure 5 using exact QC-selected optimizer points (seed42)')
    ap.add_argument('--archive', type=Path, default=Path.home()/'Desktop'/'GEM_Publication_Archive')
    ap.add_argument('--figure-root', type=Path, default=None)
    args=ap.parse_args(argv)
    archive=args.archive.expanduser().resolve()
    figure_root=(args.figure_root.expanduser().resolve() if args.figure_root else archive/'Figures_and_Tables'/'Main'/'Figure_05_QC_Expanded_Landscape_Seed42')
    if not (archive/'Figures_and_Tables').is_dir():
        raise FigureError('Archive does not look valid: '+str(archive))
    compute(archive, figure_root)

if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        sys.exit('ERROR: '+str(exc))
