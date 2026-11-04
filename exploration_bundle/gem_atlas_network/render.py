from __future__ import annotations
from pathlib import Path
import html, math, shutil, textwrap
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Sharpened publication palette for stronger contrast in exported pathway figures.
# The geometry/layout are unchanged; only color/typography contrast is increased.
MET_FILL = '#F8FAFC'
MET_BORDER = '#344054'
EXT_MET_FILL = '#FCEFCF'
EXT_MET_BORDER = '#B7791F'
RXN_FILL = '#E7EEF6'
RXN_BORDER = '#475467'
SPECIAL_RXN_FILL = '#D8F3EC'
SPECIAL_RXN_BORDER = '#0E9384'
TARGET_FILL = '#C7EDE4'
TARGET_BORDER = '#0F766E'
EDGE_COLOR = '#475467'
TEXT_COLOR = '#101828'
MUTED = '#344054'
BOX_BORDER = '#98A2B3'


def _short(text: str, width: int = 28) -> str:
    text=' '.join(str(text or '').split())
    if len(text) <= width:
        return text
    return textwrap.shorten(text,width=width,placeholder='…')


def _wrapped(text: str, width: int = 22, max_lines: int = 2) -> str:
    text=' '.join(str(text or '').split())
    lines=textwrap.wrap(text,width=width) or ['']
    if len(lines)>max_lines:
        lines=lines[:max_lines]
        lines[-1]=_short(lines[-1],max(5,width-1))
    return '\n'.join(lines)


def _is_special_reaction(d: dict) -> bool:
    rid=str(d.get('id') or '').lower()
    name=str(d.get('label') or '').lower()
    return rid.startswith(('r_ex_','ex_')) or 'exchange' in name or 'transport' in name


def _is_external_metabolite(d: dict) -> bool:
    comp=str(d.get('compartment') or '').lower()
    return comp in {'e','extracellular','extracellular space'}


def _summary_html(G: nx.DiGraph, title: str, subtitle: str, show_legend: bool=True) -> str:
    rxn_count=sum(1 for _,d in G.nodes(data=True) if d.get('kind')=='reaction')
    met_count=sum(1 for _,d in G.nodes(data=True) if d.get('kind')=='metabolite')
    t=next((d for _,d in G.nodes(data=True) if d.get('target')),None)
    target=(t.get('label') or t.get('id')) if t else G.graph.get('target','')
    mode='Flux-weighted' if G.graph.get('flux_weighted') else 'Structural'
    direction=G.graph.get('direction','')
    depth=G.graph.get('depth','')
    parts=[]
    parts.append(f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="17"><B>{html.escape(title)}</B></FONT></TD></TR>')
    if subtitle:
        parts.append(f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="10" COLOR="{MUTED}">{html.escape(subtitle)}</FONT></TD></TR>')
    info=(f'<TABLE BORDER="1" COLOR="{BOX_BORDER}" CELLBORDER="0" CELLSPACING="0" CELLPADDING="4">'
          f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9"><B>Target:</B> {html.escape(str(target))}</FONT></TD></TR>'
          f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9"><B>Mode:</B> {mode} &nbsp;&nbsp; <B>Direction:</B> {html.escape(str(direction))} &nbsp;&nbsp; <B>Depth:</B> {html.escape(str(depth))}</FONT></TD></TR>'
          f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9"><B>Shown:</B> {rxn_count} reactions &nbsp;&nbsp; {met_count} metabolites</FONT></TD></TR>'
          + (f'<TR><TD ALIGN="LEFT"><FONT POINT-SIZE="9" COLOR="{MUTED}">Filtered for publication readability</FONT></TD></TR>' if G.graph.get('truncated') else '')
          + '</TABLE>')
    parts.append(f'<TR><TD ALIGN="LEFT">{info}</TD></TR>')
    if show_legend:
        flux = any(d.get('flux') is not None for _,_,d in G.edges(data=True))
        has_disabled = any(d.get('gpr_state')=='disabled' or d.get('bounds_blocked') for _,d in G.nodes(data=True)) or any(d.get('disabled') for _,_,d in G.edges(data=True))
        has_missing = any(d.get('missing') for _,d in G.nodes(data=True))
        has_focus = any(d.get('gene_focus') for _,d in G.nodes(data=True))
        legend=(
            f'<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="5" CELLPADDING="1"><TR>'
            f'<TD><FONT POINT-SIZE="9"><B>Legend:</B></FONT></TD>'
            f'<TD BGCOLOR="{MET_FILL}" BORDER="1" COLOR="{MET_BORDER}">&nbsp;&nbsp;</TD><TD><FONT POINT-SIZE="9">metabolite</FONT></TD>'
            f'<TD BGCOLOR="{EXT_MET_FILL}" BORDER="1" COLOR="{EXT_MET_BORDER}">&nbsp;&nbsp;</TD><TD><FONT POINT-SIZE="9">extracellular</FONT></TD>'
            f'<TD BGCOLOR="{TARGET_FILL}" BORDER="1" COLOR="{TARGET_BORDER}">&nbsp;&nbsp;</TD><TD><FONT POINT-SIZE="9">target</FONT></TD>'
            f'<TD BGCOLOR="{RXN_FILL}" BORDER="1" COLOR="{RXN_BORDER}">&nbsp;&nbsp;</TD><TD><FONT POINT-SIZE="9">reaction</FONT></TD>'
            f'<TD BGCOLOR="{SPECIAL_RXN_FILL}" BORDER="1" COLOR="{SPECIAL_RXN_BORDER}">&nbsp;&nbsp;</TD><TD><FONT POINT-SIZE="9">exchange / transport</FONT></TD>'
            + (f'<TD><FONT POINT-SIZE="9">edge width ∝ |flux|</FONT></TD>' if flux else '')
            + '</TR>'
            + ('<TR><TD><FONT POINT-SIZE="9"><B>States:</B></FONT></TD>'
               '<TD BGCOLOR="#C7EDE4" BORDER="2" COLOR="#0F766E">&nbsp;&nbsp;</TD>'
               '<TD><FONT POINT-SIZE="9">gene-linked focus</FONT></TD></TR>' if has_focus else '')
            + ('<TR><TD></TD><TD BGCOLOR="#FEE4E2" BORDER="2" COLOR="#B42318" STYLE="dashed">&nbsp;&nbsp;</TD>'
               '<TD><FONT POINT-SIZE="9" COLOR="#B42318">knockout-disabled reaction (red dashed box)</FONT></TD>'
               '<TD COLSPAN="2"><FONT POINT-SIZE="9" COLOR="#B42318">- - - &#8594; disabled connection</FONT></TD></TR>' if has_disabled else '')
            + ('<TR><TD></TD><TD BGCOLOR="#FFF4E5" BORDER="2" COLOR="#B54708" STYLE="dashed">&nbsp;&nbsp;</TD>'
               '<TD COLSPAN="4"><FONT POINT-SIZE="9" COLOR="#B54708">missing / unresolved (orange dashed)</FONT></TD></TR>' if has_missing else '')
            + '</TABLE>'
        )
        parts.append(f'<TR><TD ALIGN="LEFT">{legend}</TD></TR>')
    return '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0" CELLPADDING="2">' + ''.join(parts) + '</TABLE>>'


def _pydot_render(G: nx.DiGraph, output_base: Path, title: str, subtitle: str,
                  label_mode: str='names', show_legend: bool=True, show_flux_values: bool=False, dpi: int=300):
    if shutil.which('dot') is None:
        return None
    try:
        import pydot
    except Exception:
        return None

    graph=pydot.Dot('G',graph_type='digraph',rankdir='LR',bgcolor='white',
                    splines='spline',overlap='false',nodesep='0.34',ranksep='0.66',dpi='72',
                    pad='0.20',margin='0.05',fontname='Helvetica-Bold')
    graph.set_label(_summary_html(G,title,subtitle,show_legend))
    graph.set_labelloc('t'); graph.set_labeljust('l')

    fluxvals=[abs(float(d.get('flux') or 0.0)) for _,_,d in G.edges(data=True) if d.get('flux') is not None]
    maxf=max(fluxvals,default=0.0)
    safe={n:f'n{i}' for i,n in enumerate(G.nodes())}

    for n,d in G.nodes(data=True):
        kind=d.get('kind')
        raw=d.get('id') if label_mode=='ids' else (d.get('label') or d.get('id') or n)
        label=_wrapped(raw,23 if kind=='metabolite' else 20,2)
        if kind=='metabolite':
            target=bool(d.get('target'))
            external=_is_external_metabolite(d)
            fill = TARGET_FILL if target else (EXT_MET_FILL if external else MET_FILL)
            border = TARGET_BORDER if target else (EXT_MET_BORDER if external else MET_BORDER)
            node=pydot.Node(safe[n],label=label,shape='ellipse',style='filled',
                            fillcolor=fill,color=border,
                            penwidth='2.5' if target else '1.5',fontname='Helvetica-Bold',fontsize='11.0',fontcolor=TEXT_COLOR,
                            margin='0.10,0.06')
        else:
            special=_is_special_reaction(d)
            node=pydot.Node(safe[n],label=label,shape='box',style='rounded,filled',
                            fillcolor=SPECIAL_RXN_FILL if special else RXN_FILL,
                            color=SPECIAL_RXN_BORDER if special else RXN_BORDER,
                            penwidth='1.2',fontname='Helvetica-Bold',fontsize='9.4',fontcolor=TEXT_COLOR,
                            margin='0.10,0.05')
        if d.get('gene_focus') and d.get('kind')=='reaction':
            node.set_color(TARGET_BORDER);node.set_fillcolor(TARGET_FILL);node.set_penwidth('2.5')
        if d.get('missing'):
            node.set_style('rounded,dashed'); node.set_color('#B54708');node.set_fillcolor('#FFF4E5')
        elif d.get('gpr_state')=='disabled' or d.get('bounds_blocked'):
            node.set_style('rounded,filled,dashed');node.set_fillcolor('#FEE4E2');node.set_color('#B42318')
        if d.get('signed_flux') is not None:
            node.set_tooltip('Signed model flux: '+format(d['signed_flux'],'.6g'))
        graph.add_node(node)

    for u,v,d in G.edges(data=True):
        f=d.get('flux')
        width=1.45 if f is None or maxf<=0 else 1.15 + 3.45*math.sqrt(abs(float(f))/maxf)
        attrs=dict(color=EDGE_COLOR,penwidth=f'{width:.2f}',arrowsize='0.72')
        if show_flux_values and f is not None:
            attrs.update(label=f'{float(f):.3g}',fontsize='7',fontcolor=MUTED,fontname='Arial')
        if d.get('disabled'):
            attrs.update(style='dashed',color='#B42318',penwidth='1.2')
        elif f is not None and abs(float(f))<1e-12:
            attrs.update(style='dotted',color='#98A2B3',penwidth='0.7')
        graph.add_edge(pydot.Edge(safe[u],safe[v],**attrs))

    dot_path=output_base.with_suffix('.dot')
    graph.write_raw(str(dot_path))
    paths={}
    for ext in ('png','svg','pdf'):
        p=output_base.with_suffix('.'+ext)
        try:
            # Graphviz 2.x scales SVG transforms inconsistently at raster DPI.
            # High resolution applies to PNG only; SVG/PDF retain vector units.
            graph.set_dpi(str(dpi if ext=='png' else 72))
            graph.write(str(p),format=ext,prog='dot')
        except Exception:
            return None
        paths[ext]=p
    paths['dot']=dot_path
    return paths


def _layout_fallback(G: nx.DiGraph):
    if not G.nodes:
        return {}
    try:
        from networkx.drawing.nx_pydot import graphviz_layout
        if shutil.which('dot'):
            return graphviz_layout(G,prog='dot')
    except Exception:
        pass
    try:
        H=G.copy()
        if nx.is_directed_acyclic_graph(H):
            pos={}
            gens=list(nx.topological_generations(H))
            for x,gen in enumerate(gens):
                ordered=sorted(gen)
                for i,n in enumerate(ordered):
                    pos[n]=(x,(len(ordered)-1)/2-i)
            return pos
    except Exception:
        pass
    return nx.spring_layout(G.to_undirected(),seed=42,k=max(.25,1.7/math.sqrt(max(1,len(G)))))


def _matplotlib_render(G: nx.DiGraph, output_base: Path, title: str, subtitle: str,
                       label_mode: str='names', show_legend: bool=True, show_flux_values: bool=False,
                       width_in: float=13, height_in: float=7.5, dpi: int=300):
    # Fallback only; primary publication output is Graphviz to preserve the preferred prior style.
    pos=_layout_fallback(G)
    fig,ax=plt.subplots(figsize=(width_in,height_in)); ax.set_axis_off()
    if not G.nodes:
        ax.text(.5,.5,'No nodes matched the selected filters',ha='center',va='center',fontsize=13)
    else:
        mets=[n for n,d in G.nodes(data=True) if d.get('kind')=='metabolite' and not d.get('target') and not _is_external_metabolite(d)]
        ext=[n for n,d in G.nodes(data=True) if d.get('kind')=='metabolite' and not d.get('target') and _is_external_metabolite(d)]
        targets=[n for n,d in G.nodes(data=True) if d.get('kind')=='metabolite' and d.get('target')]
        rxns=[n for n,d in G.nodes(data=True) if d.get('kind')=='reaction' and not _is_special_reaction(d) and not d.get('gene_focus')]
        special=[n for n,d in G.nodes(data=True) if d.get('kind')=='reaction' and _is_special_reaction(d)]
        if mets: nx.draw_networkx_nodes(G,pos,nodelist=mets,node_shape='o',node_size=980,node_color=MET_FILL,edgecolors=MET_BORDER,linewidths=1.4,ax=ax)
        if ext: nx.draw_networkx_nodes(G,pos,nodelist=ext,node_shape='o',node_size=980,node_color=EXT_MET_FILL,edgecolors=EXT_MET_BORDER,linewidths=1.4,ax=ax)
        if rxns: nx.draw_networkx_nodes(G,pos,nodelist=rxns,node_shape='s',node_size=760,node_color=RXN_FILL,edgecolors=RXN_BORDER,linewidths=1.2,ax=ax)
        focus=[n for n,d in G.nodes(data=True) if d.get('kind')=='reaction' and d.get('gene_focus')]
        if focus:nx.draw_networkx_nodes(G,pos,nodelist=focus,node_shape='s',node_size=950,node_color=TARGET_FILL,edgecolors=TARGET_BORDER,linewidths=2.5,ax=ax)
        if special: nx.draw_networkx_nodes(G,pos,nodelist=special,node_shape='s',node_size=760,node_color=SPECIAL_RXN_FILL,edgecolors=SPECIAL_RXN_BORDER,linewidths=1.2,ax=ax)
        if targets: nx.draw_networkx_nodes(G,pos,nodelist=targets,node_shape='o',node_size=1320,node_color=TARGET_FILL,edgecolors=TARGET_BORDER,linewidths=2.5,ax=ax)
        fluxes=[abs(float(d.get('flux') or 0.0)) for _,_,d in G.edges(data=True) if d.get('flux') is not None]
        maxf=max(fluxes,default=0.0)
        widths=[1.25 if d.get('flux') is None or maxf<=0 else 1.0+3.2*math.sqrt(abs(float(d['flux']))/maxf) for _,_,d in G.edges(data=True)]
        enabled_edges=[(u,v) for u,v,d in G.edges(data=True) if not d.get('disabled')]
        disabled_edges=[(u,v) for u,v,d in G.edges(data=True) if d.get('disabled')]
        width_by_edge={e:w for e,w in zip(G.edges(),widths)}
        if enabled_edges:
            nx.draw_networkx_edges(G,pos,edgelist=enabled_edges,width=[width_by_edge[e] for e in enabled_edges],
                                   edge_color=EDGE_COLOR,alpha=.90,arrows=True,arrowsize=16,ax=ax)
        if disabled_edges:
            nx.draw_networkx_edges(G,pos,edgelist=disabled_edges,width=1.7,style='dashed',
                                   edge_color='#B42318',arrows=True,arrowsize=16,ax=ax)
        disabled_nodes=[n for n,d in G.nodes(data=True) if d.get('kind')=='reaction' and
                        (d.get('gpr_state')=='disabled' or d.get('bounds_blocked'))]
        if disabled_nodes:
            nx.draw_networkx_nodes(G,pos,nodelist=disabled_nodes,node_shape='s',node_size=950,
                                   node_color='#FEE4E2',edgecolors='#B42318',linewidths=2.2,ax=ax)
        labels={}
        for n,d in G.nodes(data=True):
            raw=d.get('id') if label_mode=='ids' else d.get('label') or d.get('id')
            labels[n]=_wrapped(raw,18,2)
        nx.draw_networkx_labels(G,pos,labels=labels,font_size=8.8,font_family='DejaVu Sans',font_weight='bold',font_color=TEXT_COLOR,ax=ax)
    ax.set_title(title,fontsize=15,fontweight='bold',loc='left',pad=18,color=TEXT_COLOR)
    if subtitle: ax.text(0,1.01,subtitle,transform=ax.transAxes,ha='left',va='bottom',fontsize=10,color=MUTED)
    if show_legend:
        handles=[
            Line2D([0],[0],marker='o',color='w',markerfacecolor=MET_FILL,markeredgecolor=MET_BORDER,markersize=8,label='Metabolite'),
            Line2D([0],[0],marker='o',color='w',markerfacecolor=EXT_MET_FILL,markeredgecolor=EXT_MET_BORDER,markersize=8,label='Extracellular metabolite'),
            Line2D([0],[0],marker='o',color='w',markerfacecolor=TARGET_FILL,markeredgecolor=TARGET_BORDER,markersize=9,label='Target metabolite'),
            Line2D([0],[0],marker='s',color='w',markerfacecolor=RXN_FILL,markeredgecolor=RXN_BORDER,markersize=8,label='Reaction'),
            Line2D([0],[0],marker='s',color='w',markerfacecolor=SPECIAL_RXN_FILL,markeredgecolor=SPECIAL_RXN_BORDER,markersize=8,label='Exchange / transport'),
        ]
        if any(d.get('gene_focus') for _,d in G.nodes(data=True)):
            handles.append(Line2D([0],[0],marker='s',color='w',markerfacecolor=TARGET_FILL,
                       markeredgecolor=TARGET_BORDER,markeredgewidth=2,markersize=9,label='Gene-linked focus'))
        if any(d.get('gpr_state')=='disabled' or d.get('bounds_blocked') for _,d in G.nodes(data=True)):
            handles.extend([
                Line2D([0],[0],marker='s',linestyle='None',markerfacecolor='#FEE4E2',
                       markeredgecolor='#B42318',markersize=9,label='Knockout-disabled reaction (red dashed box)'),
                Line2D([0],[0],color='#B42318',linestyle='--',label='Disabled connection (red dashed)')])
        if any(d.get('missing') for _,d in G.nodes(data=True)):
            handles.append(Line2D([0],[0],color='#B54708',linestyle='--',label='Missing / unresolved'))
        if any(d.get('flux') is not None for _,_,d in G.edges(data=True)):
            handles.append(Line2D([0],[0],color='#98A2B3',linestyle=':',label='Zero flux (dotted)'))
        ax.legend(handles=handles,loc='upper right',frameon=False,fontsize=8.8)
    fig.tight_layout()
    paths={}
    for extn in ('png','svg','pdf'):
        p=output_base.with_suffix('.'+extn); fig.savefig(p,dpi=dpi,bbox_inches='tight',facecolor='white'); paths[extn]=p
    plt.close(fig)
    return paths


def render_publication_figure(G: nx.DiGraph, output_base: str|Path, title: str='Metabolic pathway',
                              subtitle: str='', layout: str='publication', label_mode: str='names',
                              show_legend: bool=True, show_flux_values: bool=False,
                              width_in: float=13, height_in: float=7.5, dpi: int=300,
                              compartment_colors: bool=False):
    output_base=Path(output_base); output_base.parent.mkdir(parents=True,exist_ok=True)
    paths=_pydot_render(G,output_base,title,subtitle,label_mode,show_legend,show_flux_values,dpi)
    if paths is not None:
        return paths
    return _matplotlib_render(G,output_base,title,subtitle,label_mode,show_legend,show_flux_values,width_in,height_in,dpi)
