import os,secrets
from pathlib import Path
from nicegui import ui,app
from .ui import build_network_visualization_tab
ROOT=Path(__file__).resolve().parents[1]
OUTPUTS=ROOT/'outputs';OUTPUTS.mkdir(exist_ok=True)
app.add_static_files('/atlas-output',str(OUTPUTS))
SECRET_FILE=ROOT/'.atlas_storage_secret'
if not SECRET_FILE.exists():
    SECRET_FILE.write_text(secrets.token_hex(32));SECRET_FILE.chmod(0o600)
STORAGE_SECRET=SECRET_FILE.read_text().strip()
ui.colors(primary='#157A6E',secondary='#667085',accent='#B98B3B')
ui.add_head_html('<style>body{background:#f7f8fa}.q-card{border-radius:12px;box-shadow:0 1px 3px rgba(16,24,40,.08)}.nicegui-content{max-width:1280px;margin:auto;width:100%}</style>')
@ui.page('/')
def page():
    with ui.header().classes('items-center'):
        ui.label('GEM Atlas').classes('text-lg font-bold');ui.space();ui.label('Network Visualization v0.3.4').classes('text-sm')
    ui.label('Explore metabolic routes, networks and genes').classes('text-3xl font-semibold')
    ui.label('Browse a GEM, choose Find Routes, Network or Genes, then Run Now.').classes('text-gray-600')
    build_network_visualization_tab(OUTPUTS,ROOT/'examples/models')
ui.run(title='GEM Atlas — Network Visualization',host='127.0.0.1',port=int(os.environ.get('ATLAS_PORT','8080')),reload=False,show=os.environ.get('ATLAS_HEADLESS')!='1',storage_secret=STORAGE_SECRET)
