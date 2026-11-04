#!/usr/bin/env python3
"""Generate illustrative Figure 3 panels J and K from existing pooled panel-H data.

READ ONLY: uses previously generated GHI pooled improvement percentages.
No optimizer reruns, no selector retraining, no new scientific recomputation.
"""
from __future__ import annotations
import argparse, csv, math, sys, json, shutil, tempfile, os
from pathlib import Path
from datetime import datetime, timezone

SIZES=(5000,10000,25000,50000)
FIVE=('QC-only selector','gsMOBO + SPARC','NSGA-II + SPARC','gsMOBO (with knowledge)','NSGA-II (with knowledge)')
FIVE_COLORS={
    'QC-only selector':'#6A3D9A',
    'gsMOBO + SPARC':'#E69F00',
    'NSGA-II + SPARC':'#009E73',
    'gsMOBO (with knowledge)':'#0072B2',
    'NSGA-II (with knowledge)':'#222222',
}
NAME='Figure_3_Panels_J_to_M_Pareto_Expansion'

class FigureError(RuntimeError):
    pass


def read_csv(path):
    with path.open('r', encoding='utf-8-sig', newline='') as f:
        yield from csv.DictReader(f)


def write_csv(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)


def find_archive(path: Path) -> Path:
    for p in (path.parent, *path.parents):
        if (p/'Figures_and_Tables').is_dir() and (p/'Results').is_dir():
            return p
    raise FigureError('Cannot locate GEM_Publication_Archive')


def locate_ghi_csv(archive: Path) -> Path:
    candidates=[]
    for p in (archive/'Figures_and_Tables'/'Main').rglob('GHI_POOLED_TOP_PCT_MAIN_297.csv'):
        if p.is_file() and not p.is_symlink():
            candidates.append(p)
    if not candidates:
        raise FigureError('Could not find GHI_POOLED_TOP_PCT_MAIN_297.csv in the archive')
    # Prefer files from the all-nine-panels output tree and newest mtime.
    candidates.sort(key=lambda p: ('FIGURE3_ALL_NINE_PANELS_V4' not in str(p), -p.stat().st_mtime, str(p)))
    return candidates[0]


def base_frontier(n_points=10):
    xs=[i/(n_points-1) for i in range(n_points)]
    ys=[1.0-x for x in xs]
    return list(zip(xs,ys))


def hypervolume_2d(points):
    """Exact 2D dominated hypervolume of connected nondominated staircase points.

    Both axes are maximized, reference (0,0). Points must be in increasing
    x and nonincreasing y. The baseline is a TEN-POINT, not continuous, front.
    """
    if not points:
        return 0.0
    for i, (x,y) in enumerate(points):
        if not (math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1):
            raise FigureError('Invalid 0..1 Pareto coordinates')
        if i and (x <= points[i-1][0] or y > points[i-1][1]+1e-12):
            raise FigureError('Pareto coordinates must be ordered and nondominated')
    area=0.0
    xleft=0.0
    for x,y in points:
        area+=(x-xleft)*y
        xleft=x
    return area


def curved_frontier(exponent, n_points=10):
    """A monotone curved front with BOTH endpoints unchanged.

    y(x)=1-x**q. q=1 gives original linear frontier; q>1 bulges out,
    q<1 bulges in. This avoids translating points and changing endpoints.
    """
    return [(x, 1.0-x**exponent) for x,_ in base_frontier(n_points)]


def exponent_for_delta_hv_pct(delta_pct, n_points=10):
    """Numerically match *exact* DISCRETE 10-point HV improvement, not an
    integral approximation to a hypothetical continuous line.

    For this geometry with fixed x and fixed endpoints, maximum attainable
    improvement is +100% as exponent approaches infinity.
    """
    p=float(delta_pct)
    if not math.isfinite(p) or not -100.0 < p < 100.0:
        raise FigureError(f'Outside representable fixed-endpoint ΔHV range (-100%,100%): {p!r}')
    baseline_hv=hypervolume_2d(base_frontier(n_points))
    target=baseline_hv*(1.0+p/100.0)
    if abs(p)<1e-12:
        return 1.0
    if p>0:
        lo,hi=1.0,2.0
        while hypervolume_2d(curved_frontier(hi,n_points)) < target:
            lo,hi=hi,2*hi
            if hi>1000000:
                raise FigureError('Unable to find curved frontier for reported improvement')
    else:
        lo,hi=1e-12,1.0
    for _ in range(90):
        mid=(lo+hi)/2
        if hypervolume_2d(curved_frontier(mid,n_points)) < target:
            lo=mid
        else:
            hi=mid
    q=(lo+hi)/2
    observed=100.0*(hypervolume_2d(curved_frontier(q,n_points))/baseline_hv-1.0)
    if abs(observed-p)>1e-8:
        raise FigureError(f'Curved-front HV mismatch expected={p} measured={observed}')
    return q


def build_rows(ghi_rows):
    """Extract the QC-only selector pooled Panel-H improvement at all four sizes."""
    wanted=set(SIZES)
    lookup={}
    for r in ghi_rows:
        method=r.get('method','').strip()
        if method!='QC-only selector':
            continue
        size=int(r['prior_points'])
        if size in wanted:
            if size in lookup:
                raise FigureError(f'Duplicate QC-only selector Panel H value for {size}')
            lookup[size]=float(r['delta_hv_mean_pct'])
    missing=sorted(wanted-set(lookup))
    if missing:
        raise FigureError('Missing QC-only selector Panel H delta_hv_mean_pct for: '+', '.join(f'{s//1000}K' for s in missing))

    baseline=base_frontier(10)
    baseline_hv=hypervolume_2d(baseline)
    rows=[]
    for label,size in zip('JKLM',SIZES):
        pct=lookup[size]
        exponent=exponent_for_delta_hv_pct(pct)
        expanded=curved_frontier(exponent)
        actual_pct=100*(hypervolume_2d(expanded)/baseline_hv-1)
        for method,curve,color in (
            ('Landscape',baseline,'#A7AFB7'),
            ('QC-only selector',expanded,FIVE_COLORS['QC-only selector']),
        ):
            for idx,(x,y) in enumerate(curve, start=1):
                rows.append({'panel':label,'prior_points':size,'method':method,'point_index':idx,
                    'x':x,'y':y,'panel_h_delta_hv_mean_pct':pct,
                    'matched_discrete_delta_hv_pct':actual_pct,
                    'baseline_hv':baseline_hv,
                    'expanded_hv':hypervolume_2d(expanded),
                    'curve_exponent':exponent,'color':color})
    return rows


def draw(plot_rows, destination: Path):
    """Four equal-aspect J-M panels in ONE horizontal row; no science changes."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10.2,
        'axes.labelsize':9,'axes.linewidth':0.8,'xtick.labelsize':8.2,'ytick.labelsize':8.2,
        'legend.fontsize':8.5,'legend.frameon':False,'figure.dpi':180,
        'savefig.dpi':600,'pdf.fonttype':42,'ps.fonttype':42})
    fig, axes=plt.subplots(1,4,figsize=(15.8,4.75),sharex=True,sharey=True)
    fig.subplots_adjust(left=.056,right=.991,top=.825,bottom=.25,wspace=.18)
    fig.suptitle('Illustrative Pareto-frontier expansion — QC-only selector',
                 fontsize=12.0,fontweight='bold',y=.965)
    qc_color=FIVE_COLORS['QC-only selector']
    base_color='#A7AFB7'
    for ax,size,label in zip(axes,SIZES,'JKLM'):
        data=[r for r in plot_rows if r['prior_points']==size]
        raw=sorted((r for r in data if r['method']=='Landscape'),key=lambda r:r['point_index'])
        qc=sorted((r for r in data if r['method']=='QC-only selector'),key=lambda r:r['point_index'])
        xs=[float(r['x']) for r in raw]
        ys=[float(r['y']) for r in raw]
        qy=[float(r['y']) for r in qc]
        pct=float(qc[0]['panel_h_delta_hv_mean_pct'])
        ax.fill_between(xs,ys,qy,color=qc_color,alpha=.11,zorder=1)
        ax.plot(xs,ys,color=base_color,linestyle='--',lw=1.9,marker='o',markersize=3.9,zorder=3)
        ax.plot(xs,qy,color=qc_color,lw=2.2,marker='o',markersize=4.5,zorder=4)
        ax.set_xlim(-.045,1.045)
        ax.set_ylim(-.045,1.045)
        ax.set_aspect('equal',adjustable='box')
        ax.set_xticks([0,.25,.5,.75,1.0])
        ax.set_yticks([0,.25,.5,.75,1.0])
        ax.grid(color='#E8E8E8',linewidth=.55,zorder=0)
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(direction='out',length=3,width=.7)
        ax.set_title(f'{label}   {size//1000}K   |   ΔHV {pct:+.1f}%',loc='left',pad=10,fontweight='bold')
        ax.set_xlabel('Objective 1')
    axes[0].set_ylabel('Objective 2')
    handles=[
        Line2D([0],[0],color=base_color,lw=2,linestyle='--',marker='o',markersize=4,label='Landscape'),
        Line2D([0],[0],color=qc_color,lw=2.2,marker='o',markersize=4,label='QC-only selector'),
    ]
    fig.legend(handles=handles,loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,.055))
    destination.mkdir(parents=True,exist_ok=True)
    for ext in ('pdf','png','svg'):
        fig.savefig(destination/f'{NAME}.{ext}',bbox_inches='tight',facecolor='white',
                    **({'dpi':500} if ext=='png' else {}))
    plt.close(fig)


def compute(archive: Path, figure_root: Path):
    ghi_csv=locate_ghi_csv(archive)
    ghi_rows=list(read_csv(ghi_csv))
    plot_rows=build_rows(ghi_rows)
    outdir=figure_root/'outputs'/'PARETO_EXPANSION_JM'
    if outdir.exists() and outdir.is_symlink():
        raise FigureError('Refusing to write into a symlinked output folder')
    draw(plot_rows,outdir)
    write_csv(outdir/'plot_data'/'JM_PARETO_EXPANSION_SYNTHETIC_POINTS.csv', plot_rows)
    meta={
        'generator':'Figure 3 J-M illustrative QC-only Pareto expansion',
        'generated_utc':datetime.now(timezone.utc).isoformat(),
        'source_pooled_improvement_csv':str(ghi_csv.relative_to(archive)),
        'base_frontier':'10 points on the line from (0,1) to (1,0); endpoints unchanged',
        'sizes':list(SIZES),
        'methods':['QC-only selector'],
        'construction':'For each of four sizes, keep the ten baseline x values unchanged and set y=1-x**q for the QC-only selector. The exponent q is determined by bisection to reproduce the pooled Panel H mean HV improvement calculated from exact discrete 10-point hypervolume. Both endpoints stay fixed; interior points bow outward when gain is positive.',
        'curve_formula':'baseline y=1-x; QC-only y=1-x**q with q chosen by exact 10-point HV matching. q>1 expands; q<1 contracts if source gain is negative.',
        'note':'Illustrative synthetic curves constrained to reproduce QC-only Panel H pooled HV improvements, not recovered native optimization Pareto coordinates.'
    }
    (outdir/'plot_data'/'JM_METADATA.json').write_text(json.dumps(meta, indent=2)+'\n')
    print('SOURCE GHI CSV:', ghi_csv)
    print('OUTPUT:', outdir)
    return outdir


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--archive', type=Path, default=Path.home()/'Desktop'/'GEM_Publication_Archive')
    ap.add_argument('--figure-root', type=Path, default=None)
    a=ap.parse_args()
    archive=a.archive.expanduser().resolve()
    figure_root=(a.figure_root.expanduser().resolve() if a.figure_root else archive/'Figures_and_Tables'/'Main'/'Figure_03_JK_Pareto_Expansion')
    if not (archive/'Figures_and_Tables').is_dir():
        raise FigureError('Archive root does not look valid: '+str(archive))
    compute(archive, figure_root)

if __name__=='__main__':
    try:
        main()
    except FigureError as e:
        sys.exit('ERROR: '+str(e))
