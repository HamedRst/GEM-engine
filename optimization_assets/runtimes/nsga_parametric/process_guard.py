#!/usr/bin/env python3
"""Fail closed if another known GEM optimization campaign is active.

Read-only process inspection. It never kills or signals a process.
"""
from __future__ import annotations
import argparse, os, subprocess, sys

PATTERNS=(
    'run_generic_problem.py',
    'benchmark_case.py',
    'run_case.py',
    'run_worker.py',
    'RUN_THIS_VARIANT.sh',
    'benchmark_step4c.py',
    'run_benchmark.py',
)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--worker',default=''); a=ap.parse_args()
    out=subprocess.run(['ps','-axo','pid=,ppid=,command='],capture_output=True,text=True,check=True).stdout
    own={os.getpid(),os.getppid()}; hits=[]
    for line in out.splitlines():
        z=line.strip()
        if not z: continue
        parts=z.split(None,2)
        if len(parts)<3: continue
        try: pid=int(parts[0]); ppid=int(parts[1])
        except ValueError: continue
        cmd=parts[2]
        if pid in own or ppid in own: continue
        # Ignore process-list / guard commands themselves.
        if 'process_guard.py' in cmd or 'ps -axo' in cmd: continue
        if any(p in cmd for p in PATTERNS): hits.append((pid,cmd))
    if hits:
        print('GEM PROCESS GUARD: REFUSING TO START')
        print('Another known GEM optimization process appears active on this computer:')
        for pid,cmd in hits[:20]: print(f'  PID {pid}: {cmd}')
        print('No process was stopped or modified. Finish/stop the other campaign intentionally, then run this launcher again.')
        raise SystemExit(3)
    print(f'GEM PROCESS GUARD: PASS{(" | worker="+a.worker) if a.worker else ""}')

if __name__=='__main__': main()
