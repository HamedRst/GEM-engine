#!/usr/bin/env python3
"""Fail closed if another *actual* GEM optimization process is active.

Read-only process inspection. Never kills or signals a process.

macOS note:
`caffeinate -i python -u run_worker.py ...` places `run_worker.py` in the
*caffeinate wrapper's* command line. Depending on process timing/topology,
that caffeinate process is not guaranteed to appear in the guard's ancestor
chain. A wrapper is not itself an optimizer, so wrapper executables are
ignored explicitly. The child Python GEM process is still detected normally.
"""
from __future__ import annotations
import argparse, os, shlex, subprocess

PATTERNS=(
    'run_generic_problem.py',
    'benchmark_case.py',
    'run_case.py',
    'run_worker.py',
    'RUN_THIS_VARIANT.sh',
    'benchmark_step4c.py',
    'run_benchmark.py',
)

# These are launch/scheduling wrappers, not optimization workers themselves.
# Ignoring them is safe because the actual Python/script child remains visible
# and is what this guard is intended to detect.
WRAPPER_BASENAMES={
    'caffeinate','bash','zsh','sh','dash','env','nice','nohup','timeout'
}

def parse_ps(text):
    rows=[]
    for line in text.splitlines():
        z=line.strip()
        if not z: continue
        parts=z.split(None,2)
        if len(parts)<3: continue
        try: rows.append((int(parts[0]),int(parts[1]),parts[2]))
        except ValueError: continue
    return rows

def ancestor_pids(rows,pid):
    parent={p:pp for p,pp,_ in rows};out=set();cur=int(pid)
    while cur and cur not in out:
        out.add(cur);cur=parent.get(cur,0)
    return out

def executable_basename(cmd):
    try:
        toks=shlex.split(cmd,posix=True)
    except ValueError:
        toks=cmd.split()
    if not toks: return ''
    return os.path.basename(toks[0])

def find_hits(rows,current_pid):
    ignore=ancestor_pids(rows,current_pid);hits=[]
    for pid,ppid,cmd in rows:
        if pid in ignore: continue
        if 'process_guard.py' in cmd or 'ps -axo' in cmd: continue
        # Critical macOS fix: never classify caffeinate/shell/nice wrappers as
        # the optimizer merely because their command line mentions run_worker.
        if executable_basename(cmd) in WRAPPER_BASENAMES: continue
        if any(p in cmd for p in PATTERNS): hits.append((pid,cmd))
    return hits

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--worker',default='');a=ap.parse_args()
    text=subprocess.run(['ps','-axo','pid=,ppid=,command='],capture_output=True,text=True,check=True).stdout
    rows=parse_ps(text);hits=find_hits(rows,os.getpid())
    if hits:
        print('GEM PROCESS GUARD: REFUSING TO START')
        print('Another known GEM optimization process appears active on this computer:')
        for pid,cmd in hits[:20]: print(f'  PID {pid}: {cmd}')
        print('No process was stopped or modified. Finish/stop the other campaign intentionally, then run this launcher again.')
        raise SystemExit(3)
    print(f'GEM PROCESS GUARD: PASS{(" | worker="+a.worker) if a.worker else ""}')
if __name__=='__main__': main()
