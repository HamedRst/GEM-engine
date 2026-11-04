"""Future Exploration integration hooks; no other solvers are implemented here.

Mount build_network_visualization_tab(outputs_dir, examples_dir) on one page.
Route problem_type == 'exploration.network_visualization' to engine.execute.
Ensure app.add_static_files('/atlas-output', str(outputs_dir)) is registered.
Loading/saving problems never solves FBA; only explicit execution does.
"""
from .engine import execute
from .ui import build_network_visualization_tab
PROBLEM_TYPE='exploration.network_visualization'
