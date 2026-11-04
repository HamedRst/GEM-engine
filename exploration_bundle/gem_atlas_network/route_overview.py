"""Readable wrapped overview for long directed traces.

This is a reaction-spine overview, *not* a self-contained stoichiometric
pathway diagram: full_graph and reaction tables retain all cosubstrates/products.
"""
from __future__ import annotations
from pathlib import Path
from math import ceil
from textwrap import wrap
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def _label(txt, width=17):
    value=' '.join(str(txt).split())
    lines=wrap(value,width=width,break_long_words=False,break_on_hyphens=True)
    if len(lines)>2:
        lines=[lines[0],lines[1][:max(5,width-1)]+'…']
    return '\n'.join(lines) or '—'


def draw_route_overview(model,route,output_base, dpi=220):
    """Create SI-readable 3-step-per-row figures, preserving individual reaction order."""
    step_count=len(route['reaction_trace'])
    if step_count<8:return {}
    out=Path(output_base)
    names=[model.metabolites[mid].name for mid in route['metabolite_trace']]
    steps=route['reaction_trace']
    if len(names)!=len(steps)+1:
        raise ValueError('Cannot create route overview: reaction/metabolite spine differs')
    per_row=3
    rows=ceil(step_count/per_row)
    fig,ax=plt.subplots(figsize=(15.8,2.85*rows+1.0))
    fig.patch.set_facecolor('white');ax.set_xlim(0,1);ax.set_ylim(0,1);ax.axis('off')
    ax.text(.035,.975,f"Route {route['index']} · {step_count}-reaction directed trace",ha='left',va='top',fontsize=18,fontweight='bold',color='#101828',transform=ax.transAxes)
    ax.text(.035,.936,f"{names[0]} → {names[-1]}   |   {route['status'].replace('_',' ')}",ha='left',va='top',fontsize=11,color='#344054',transform=ax.transAxes)
    top=.83; bottom=.12
    ys=[top-(top-bottom)*(i/(rows-1) if rows>1 else 0.5) for i in range(rows)]
    xs=[.09,.35,.61,.87]
    for row,y in enumerate(ys):
        a=row*per_row;b=min(step_count,(row+1)*per_row)
        for j in range(b-a+1):
            pos=a+j
            mid=route['metabolite_trace'][pos]
            selected=pos==0 or pos==step_count
            fc='#D8F3EC' if selected else '#F3F6FA'
            ec='#0F766E' if selected else '#52657A'
            rect=FancyBboxPatch((xs[j]-.075,y-.037),.15,.074,boxstyle='round,pad=0.01,rounding_size=0.012',
                                linewidth=1.5,edgecolor=ec,facecolor=fc,transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(xs[j],y,_label(names[pos],19),ha='center',va='center',fontsize=9.5,fontweight='semibold',color='#101828',transform=ax.transAxes)
        for j in range(b-a):
            step=steps[a+j];r=model.reactions[step['reaction_id']]
            xa=xs[j]+.086;xb=xs[j+1]-.087
            ax.add_patch(FancyArrowPatch((xa,y),(xb,y),arrowstyle='-|>',mutation_scale=15,
                                        linewidth=1.8,color='#455468',transform=ax.transAxes))
            label=f"{a+j+1}. {_label(r.name,19).replace(chr(10),' ')}\n{step['reaction_id']} ({'rev' if step['direction']=='reverse' else 'fwd'})"
            ax.text((xa+xb)/2,y+.046,label,ha='center',va='bottom',fontsize=8.0,color='#253349',transform=ax.transAxes)
        if row<rows-1:
            # Explicit continuation across row boundaries; not another biochemical reaction.
            ax.annotate('continued ↓',xy=(.94,y-.075),xycoords=ax.transAxes,
                        ha='center',va='top',fontsize=9,color='#64748b')
    ax.text(.035,.035,'Path spine only. See reaction stoichiometry audit for all cofactors and coproducts; detailed graph visibility follows currency filters.\n'
            'Directed connectivity and whole-model LP support do not prove carbon-atom provenance or a self-contained fermentation flux.',
            transform=ax.transAxes,ha='left',va='bottom',fontsize=9,color='#475467')
    paths={}
    for ext in ('svg','pdf','png'):
        filename=out.with_suffix('.'+ext)
        fig.savefig(filename,format=ext,dpi=dpi,facecolor='white',bbox_inches='tight')
        paths[ext]=filename
    plt.close(fig)
    return paths
