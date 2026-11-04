#!/usr/bin/env python3
"""Stop the local GUI after the last browser tab closes; never stop active jobs."""
from __future__ import annotations
import json
import os
import signal
import sys
import time
from pathlib import Path
from urllib.request import urlopen


def watch(pid,port,pid_file,*,interval=3,grace=24,startup_grace=110):
    started=time.monotonic();idle_since=None
    while True:
        try:os.kill(pid,0)
        except ProcessLookupError:return 'server already stopped'
        except PermissionError:return 'not owned by this user'
        if pid_file.exists() and pid_file.read_text().strip()!=str(pid):return 'another process owns the PID file'
        try:
            with urlopen('http://127.0.0.1:'+str(port)+'/api/session/status',timeout=3) as response:
                d=json.load(response)
        except (OSError,ValueError):
            idle_since=None;time.sleep(interval);continue
        has_client=bool(d.get('browser_sessions'))
        active=bool(d.get('active_jobs'))
        opened=bool(d.get('ever_open'))
        if has_client or active or (not opened and time.monotonic()-started<startup_grace):
            idle_since=None
        else:
            if idle_since is None:idle_since=time.monotonic()
            elif time.monotonic()-idle_since>=grace:
                # Recheck immediately before sending SIGTERM, guarding against a newly opened tab.
                try:
                    with urlopen('http://127.0.0.1:'+str(port)+'/api/session/status',timeout=3) as response:
                        latest=json.load(response)
                    if latest.get('browser_sessions') or latest.get('active_jobs'):
                        idle_since=None;time.sleep(interval);continue
                except (OSError,ValueError):time.sleep(interval);continue
                try:
                    import psutil
                    proc=psutil.Process(pid)
                    if '-m gem_learning_engine.gui' not in ' '.join(proc.cmdline()):return 'PID no longer belongs to GEM Engine GUI'
                except ImportError:pass
                except Exception:return 'process ownership could not be validated'
                try:os.kill(pid,signal.SIGTERM)
                except ProcessLookupError:pass
                return 'browser closed and jobs idle; requested GUI shutdown'
        time.sleep(interval)

if __name__=='__main__':
    if len(sys.argv)!=4:raise SystemExit('expected PID PORT PID_FILE')
    pid=int(sys.argv[1]);port=int(sys.argv[2]);pid_file=Path(sys.argv[3])
    print(watch(pid,port,pid_file),flush=True)
