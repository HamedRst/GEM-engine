"""File intake for user-defined GEM optimization inputs.

This is an interface-only extension. The existing scientific optimization
engines and their source files are not modified.
"""
from __future__ import annotations

import hashlib
import os
import re
import uuid
from pathlib import Path

from fastapi import HTTPException, Query, Request

KINDS = {
    'landscape': ('.csv', '.csv.gz', '.parquet'),
    'configuration': ('.yaml', '.yml'),
}
MAX_BYTES = 2 * 1024 * 1024 * 1024


def attach_general_uploads(app, workspace, auth):
    @app.post('/api/optimization/general/upload/{kind}', dependencies=[auth])
    async def upload_general_input(kind: str, request: Request, filename: str = Query(...)):
        if kind not in KINDS:
            raise HTTPException(400, detail='Unsupported upload type')
        name = Path(filename).name
        if name != filename or not name or name in ('.', '..') or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._ -]{0,180}', name):
            raise HTTPException(400, detail='Use a regular filename containing letters, numbers, dots, spaces, dashes, or underscores')
        if not any(name.lower().endswith(suffix) for suffix in KINDS[kind]):
            raise HTTPException(400, detail='Unsupported file extension for '+kind)
        outdir = workspace.resolve(f'optimization_inputs/{kind}')
        outdir.mkdir(parents=True, exist_ok=True)
        destination = outdir / f'{uuid.uuid4().hex[:12]}_{name.replace(" ", "_")}'
        temporary = destination.with_name(destination.name + '.uploading')
        n = 0
        digest = hashlib.sha256()
        try:
            with temporary.open('xb') as fp:
                async for chunk in request.stream():
                    n += len(chunk)
                    if n > MAX_BYTES:
                        raise HTTPException(413, detail='Upload exceeds 2 GiB limit; copy the file into the workspace and enter its relative path')
                    fp.write(chunk)
                    digest.update(chunk)
                fp.flush()
                os.fsync(fp.fileno())
            if n == 0:
                raise HTTPException(400, detail='Empty file')
            temporary.replace(destination)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        return {'kind':kind,'path':workspace.display(destination),'size_bytes':n,'sha256':digest.hexdigest()}
