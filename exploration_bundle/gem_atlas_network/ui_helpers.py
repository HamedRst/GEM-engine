"""Filesystem-only helpers for separated, non-overwriting per-problem results."""
from pathlib import Path
from datetime import datetime,timezone
import re,uuid


def new_run_root(base,problem_id=None):
    """Allocate immutable results: <base>/results/<problem_id>/runs/<run_id>.

    ``problem_id=None`` retains the old helper layout for older integrations.
    Never derive a filesystem directory from an unchecked identifier.
    """
    base=Path(base).resolve()
    if problem_id is not None:
        if not isinstance(problem_id,str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',problem_id):
            raise ValueError('Invalid problem ID for results directory')
        base=base/'results'/problem_id/'runs'
    else:
        base=base/'runs'
    base.mkdir(parents=True,exist_ok=True)
    stamp=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    root=base/('run_'+stamp+'_'+uuid.uuid4().hex[:8])
    root.mkdir(exist_ok=False)
    return root
