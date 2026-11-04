import shutil,tempfile
from pathlib import Path
from .model_io import load_model
from .problems import REGISTRY
from .pathways import TEMPLATES
from .graphing import publication_target_subnetwork
from .render import render_publication_figure

def main():
    if not shutil.which('dot'):raise SystemExit('Graphviz dot is missing. Run INSTALL.command after installing Graphviz.')
    for path in REGISTRY.values():load_model(path)
    assert len(TEMPLATES)==7
    m=load_model(REGISTRY['cglutamicum_growth_product_cost'])
    g=publication_target_subnetwork(m,'M_glu__L_e',max_reactions=8)
    with tempfile.TemporaryDirectory() as td:
        files=render_publication_figure(g,Path(td)/'smoke','Installation smoke test','Structural · zero FBA')
        if 'dot' not in files:raise SystemExit('Graphviz could not export all formats. Check your Graphviz installation.')
        assert all(files[x].stat().st_size>100 for x in ['svg','pdf','png'])
    print('PASS: five GEMs, seven offline templates, SVG/PDF/PNG, structural preflight (zero FBA).')
if __name__=='__main__':main()
