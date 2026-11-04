"""Portable all-results archive with immutable per-problem provenance.

For route searches, renders every reported candidate (not just the first or the
currently selected route) before archiving. Archive is built atomically and
includes checksums, scientific qualifications, query YAML and route tables.
"""
from __future__ import annotations
from pathlib import Path
import hashlib,json,zipfile
from datetime import datetime,timezone
from .engine import render_selected_route


def _result_folder(run_root, result):
    run_root=Path(run_root).resolve()
    folder=(run_root/result['output_directory']).resolve()
    if not folder.is_relative_to(run_root) or not folder.is_dir():
        raise ValueError('Result directory is missing or outside the run folder')
    return folder


def export_all_results(p,base_dir,run_root,result,allowed_root=None,supplied=None):
    """Return ZIP path. Never claim partial route export as success.

    All figures (SVG/PDF/PNG) are generated for every discovered route in a
    route_search result, regardless of how many routes were previewed or which
    export formats were chosen for manual figure viewing.
    """
    root=Path(run_root).resolve()
    folder=_result_folder(root,result)
    if result.get('problem_id')!=p.problem_id:
        raise ValueError('Result and configured problem do not match')
    if result.get('route_search') is not None:
        total=len(result['route_search']['routes'])
        # Export All means all figure formats, even if the preview was SVG-only.
        full_p=p.model_copy(deep=True)
        full_p.render.export_formats=['svg','pdf','png']
        for index in range(1,total+1):
            files=result['figures'].get(f'route_{index}',{})
            required={'svg','png','pdf'}
            if any(ext not in files or not (root/files[ext]).is_file() for ext in required):
                files=render_selected_route(full_p,base_dir,root,result,index,allowed_root,supplied)
                if not required.issubset(files):
                    raise RuntimeError(f'Route {index} could not be exported in every format')
                result['figures'][f'route_{index}']=files
        if len([k for k in result['figures'] if k.startswith('route_')])!=total:
            raise RuntimeError('Route export is incomplete')
        # Preserve the actual analysis configuration and route-search result.
        result_file=folder/'result.json'
        tmp=result_file.with_suffix('.json.tmp')
        tmp.write_text(json.dumps(result,indent=2,allow_nan=False))
        tmp.replace(result_file)
    files=sorted(f for f in folder.rglob('*') if f.is_file() and not f.is_symlink())
    if not files:raise RuntimeError('No result files found')
    prefix=Path(p.problem_id)/root.name
    entries=[]
    for f in files:
        if not f.resolve().is_relative_to(folder):raise ValueError('Result file escapes result directory')
        relative=f.relative_to(folder)
        entries.append({'path':str(prefix/relative),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),
                        'size_bytes':f.stat().st_size})
    manifest={'format':'GEM Atlas complete results archive v1',
              'problem_id':p.problem_id,'run_id':root.name,
              'created_utc':datetime.now(timezone.utc).isoformat(),
              'model_sha256':result['model_sha256'],
              'configuration_sha256':result['configuration_sha256'],
              'route_count':len(result['route_search']['routes']) if result.get('route_search') is not None else None,
              'route_count_is_bounded':bool(result.get('route_search') is not None),
              'claims':'Candidate directed traces are not proven biochemical pathways, atom-mapped routes, or in-vivo fluxes.',
              'files':entries}
    zip_path=root/(p.problem_id+'__'+root.name+'__ALL_RESULTS.zip')
    partial=zip_path.with_suffix('.zip.tmp')
    try:
        with zipfile.ZipFile(partial,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
            for entry,f in zip(entries,files):z.write(f,arcname=entry['path'])
            z.writestr(str(prefix/'MANIFEST.json'),json.dumps(manifest,indent=2)+'\n')
        # Fail closed: inspect archive before exposing download.
        with zipfile.ZipFile(partial) as z:
            if z.testzip() is not None:raise RuntimeError('Archive CRC verification failed')
        partial.replace(zip_path)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    return zip_path
