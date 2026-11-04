#!/usr/bin/env python3
"""Idempotent, owned and relocatable plotting-only Python virtual environment.

No FBA, COBRApy, optimizer, project-specific installation, or data writes.
Works with an unpacked ZIP or the installed location inside an archive.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

OWNER = 'GEM_FIGURE3_PLOTTING_ONLY'
MARKER = '.gem_figure3_env.json'
EXPECTED = {'numpy': '2.2.6', 'matplotlib': '3.10.7'}

class EnvironmentError(RuntimeError):
    pass

def python_in_env(env: Path) -> Path:
    return env / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')

def probe_interpreter(exe: str):
    try:
        out = subprocess.run([exe, '-c', 'import sys,json;print(json.dumps({"version":list(sys.version_info[:3]),"executable":sys.executable}))'],
                             text=True,capture_output=True,timeout=12,check=True)
        info=json.loads(out.stdout.strip().splitlines()[-1]); version=info['version']
        if version[0] == 3 and 10 <= version[1] <= 13:
            return info['executable'],version
    except (OSError,subprocess.CalledProcessError,subprocess.TimeoutExpired,ValueError,IndexError,KeyError):
        return None
    return None

def select_python():
    override=os.environ.get('GEM_PLOT_PYTHON','').strip()
    if override:
        result=probe_interpreter(override)
        if not result:
            raise EnvironmentError('GEM_PLOT_PYTHON must be an existing CPython 3.10-3.13 executable: '+override)
        return result[0]
    choices=('python3.13','python3.12','python3.11','python3.10','python3','python')
    for candidate in choices:
        resolved=shutil.which(candidate)
        if not resolved:continue
        result=probe_interpreter(resolved)
        if result:return result[0]
    raise EnvironmentError('Python 3.10-3.13 was not found. Install a supported Python 3 and rerun.')

def inspect_env_python(py:Path):
    if not py.exists(): return None
    code='''import json,sys,importlib.metadata as m
try:
 import numpy,matplotlib
 print(json.dumps({"numpy":m.version("numpy"),"matplotlib":m.version("matplotlib"),"prefix":sys.prefix,"version":list(sys.version_info[:3])}))
except Exception as e:
 print(json.dumps({"error":str(e)}))'''
    try:
        p=subprocess.run([str(py),'-c',code],text=True,capture_output=True,timeout=25,check=True)
        info=json.loads(p.stdout.strip().splitlines()[-1])
        return info
    except (OSError,ValueError,IndexError,subprocess.CalledProcessError,subprocess.TimeoutExpired):return None

def marker_read(path:Path):
    try:
        obj=json.loads(path.read_text())
        return obj if isinstance(obj,dict) and obj.get('owner')==OWNER else None
    except (OSError,ValueError):return None

def marker_write(path:Path,obj):
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n',encoding='utf8')
    temp.replace(path)

def ensure_env(venv:Path,requirements:Path,dry_run=False):
    venv=venv.expanduser().absolute()
    requirements=requirements.expanduser().absolute()
    if not requirements.is_file() or requirements.is_symlink():
        raise EnvironmentError('Missing or unsafe requirements.txt: '+str(requirements))
    if venv.is_symlink() or requirements.parent.is_symlink():
        raise EnvironmentError('Refusing symlink environment or script directory')
    if venv.name != '.plot_venv' or venv.parent != requirements.parent:
        raise EnvironmentError('Environment must be scripts/.plot_venv (never outside figure scripts)')
    sha=hashlib.sha256(requirements.read_bytes()).hexdigest()
    base=select_python()
    old=marker_read(venv/MARKER)
    if venv.exists() and not old:
        raise EnvironmentError('Refusing to overwrite an unowned plotting directory: '+str(venv))
    desired={
        'owner':OWNER,'schema':1,'venv_absolute_path':str(venv),
        'requirements_sha256':sha,'created_using_python':base,
    }
    relocated=bool(old and old.get('venv_absolute_path') != str(venv))
    new_reqs=bool(old and old.get('requirements_sha256') != sha)
    old_python=bool(old and old.get('created_using_python') != base)
    if not any((relocated,new_reqs,old_python)) and old and old.get('status')=='ready':
        actual=inspect_env_python(python_in_env(venv))
        if (actual and all(actual.get(k)==v for k,v in EXPECTED.items())
            and Path(actual.get('prefix','')).absolute()==venv):
            print('PLOT ENV READY: using existing isolated .plot_venv (no installation needed)')
            print('Python: '+str(python_in_env(venv)))
            return
    if dry_run:
        print('PLOT ENV PLAN: '+('rebuild on moved archive/changed environment' if old else 'create plotting-only environment'))
        print('Python: '+base)
        print('Packages: '+', '.join(f'{k}=={v}' for k,v in EXPECTED.items()))
        return
    if any((relocated,new_reqs,old_python)) or (old and not python_in_env(venv).exists()):
        # Only files within our explicitly owned, private virtual environment are disposable.
        if old and old.get('owner')==OWNER and venv.is_dir() and not venv.is_symlink():
            print('Rebuilding disposable plotting environment (archive moved or Python changed).')
            shutil.rmtree(venv)
    if not venv.exists():
        print('Creating plotting-only virtualenv: '+str(venv))
        try:subprocess.run([base,'-m','venv',str(venv)],check=True)
        except subprocess.CalledProcessError as e:raise EnvironmentError('Python venv creation failed: '+str(e)) from e
    marker_write(venv/MARKER,{**desired,'status':'building'})
    py=python_in_env(venv)
    # Even if numpy/matplotlib are globally installed, use only the local virtualenv.
    print('Installing pinned plotting libraries from requirements.txt (network may be needed once).')
    try:
        subprocess.run([str(py),'-m','pip','install','--disable-pip-version-check',
                        '--only-binary=:all:','-r',str(requirements)],check=True)
    except subprocess.CalledProcessError as e:
        raise EnvironmentError('Plotting library installation failed (possibly no internet or missing wheels). '
                               'Original GEM data and optimizer environments are unchanged. '
                               'Reconnect to PyPI and rerun RUN_FIGURE.command. '+str(e)) from e
    actual=inspect_env_python(py)
    if not actual or any(actual.get(k)!=v for k,v in EXPECTED.items()) or Path(actual.get('prefix','')).absolute()!=venv:
        raise EnvironmentError('Installed plotting environment did not pass exact package/prefix check: '+str(actual))
    marker_write(venv/MARKER,{**desired,'status':'ready','installed':EXPECTED,
                               'resolved_versions':actual['version']})
    # Record exact transitive dependency versions for reproducibility; machine-specific only.
    freeze=subprocess.run([str(py),'-m','pip','freeze','--disable-pip-version-check'],capture_output=True,text=True)
    if freeze.returncode==0:(venv/'RESOLVED_PLOT_ENV.txt').write_text(freeze.stdout,encoding='utf8')
    print('PLOT ENV READY: '+str(py))
    print('Installed: '+', '.join(f'{k}=={v}' for k,v in EXPECTED.items()))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--venv',type=Path,default=Path(__file__).absolute().parent/'.plot_venv')
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    try:ensure_env(args.venv,Path(__file__).absolute().parent/'requirements.txt',args.dry_run)
    except EnvironmentError as e:
        print('PLOT ENV BLOCKED: '+str(e),file=sys.stderr);sys.exit(2)
if __name__=='__main__':main()
