"""Mount the Exploration application on the GEM Engine GUI."""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path

from fastapi import FastAPI
from starlette.staticfiles import StaticFiles


def attach_exploration(app: FastAPI, workspace, auth, *, token: str | None = None) -> None:
    enabled = os.environ.get('GEM_ENABLE_EXPLORATION') == '1'
    state = {'available': False,
             'message': 'Open GEM Engine.app to enable Exploration.'}

    @app.get('/api/exploration/status', dependencies=[auth])
    def exploration_status() -> dict:
        return dict(state)

    if not enabled:
        return

    # A token on the API does not authenticate websocket and static-file requests
    # within an iframe. Never silently expose the add-on under a token-enabled host.
    # The launcher serves only loopback, and remote access needs explicit forwarding.
    if token is not None:
        raise RuntimeError('Embedded Exploration does not support token-based remote exposure. '
                           'Start the integrated GUI locally without GEM_GUI_TOKEN.')

    bundle = Path(__file__).resolve().parent.parent / 'exploration_bundle'
    if not (bundle / 'gem_exploration_hub' / 'app.py').is_file():
        raise FileNotFoundError('Bundled Explorer source missing: ' + str(bundle))
    workspace_root = Path(workspace.root)
    pool_root = Path(os.environ.get('GEM_EXPLORATION_WORKSPACE',
                                   str(workspace_root / 'exploration_workspace'))).expanduser().resolve()
    outputs_root = Path(os.environ.get('GEM_EXPLORATION_OUTPUTS',
                                      str(workspace_root / 'exploration_workspace' / 'atlas_outputs'))).expanduser().resolve()
    for folder in (pool_root / 'results', outputs_root, outputs_root / 'single_tab_activity'):
        folder.mkdir(parents=True, exist_ok=True)

    # Hub resolves imports relative to its original standalone installation.
    # Keep scientific packages intact under the bundle, avoiding overwriting the
    # GEM engine's own scientific modules, global config, and optimization assets.
    if str(bundle) not in sys.path:
        sys.path.insert(0, str(bundle))
    os.environ['GEM_EXPLORATION_WORKSPACE'] = str(pool_root)
    os.environ['GEM_EXPLORATION_OUTPUTS'] = str(outputs_root)
    os.environ['GEM_EXPLORATION_EMBEDDED'] = '1'

    try:
        from nicegui import ui
    except ImportError as exc:
        raise RuntimeError('Exploration dependency NiceGUI is missing. '
                           'Run INSTALL_GEM_ENGINE.command.') from exc

    # Registers its @ui.page and original Atlas/Calculator/Activity routes.
    import gem_exploration_hub.app  # noqa: F401

    # Historical viewer/export links use root-absolute paths. Serve the exact
    # original result trees as READ-ONLY static mounts under new route names.
    app.mount('/exploration-results', StaticFiles(directory=str(pool_root / 'results')), name='exploration-results')
    app.mount('/atlas-output', StaticFiles(directory=str(outputs_root)), name='atlas-output')
    app.mount('/exploration-atlas', StaticFiles(directory=str(bundle / 'gem_exploration_tab' / 'assets')),
              name='exploration-atlas')
    app.mount('/exploration-datasets', StaticFiles(directory=str(outputs_root / 'single_tab_activity')),
              name='exploration-datasets')

    # Persist NiceGUI's storage secret outside the original source tree.
    private = workspace_root / '.gem_gui' / 'exploration_storage_secret'
    private.parent.mkdir(parents=True, exist_ok=True)
    if not private.exists():
        fd = os.open(private, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            f.write(secrets.token_hex(32))
    ui.run_with(app, mount_path='/exploration', storage_secret=private.read_text(encoding='utf-8').strip())
    state.update(available=True, message='Ready: one Exploration tab, YAML-driven and on-demand',
                 url='/exploration/')
