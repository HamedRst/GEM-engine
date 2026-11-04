"""Persistent lightweight GUI job receipts for saved Exploration questions."""
from __future__ import annotations
import json
import os
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from .runner import run_one

LOCK=RLock()

def now():
    return datetime.now(timezone.utc).isoformat()

def receipt_root(pool_root):
    root=Path(pool_root).expanduser().resolve()/'run_records'
    root.mkdir(parents=True,exist_ok=True)
    return root

def write_receipt(path,data):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,default=str),encoding='utf-8')
    os.replace(tmp,path)

def execute(pool_root, name):
    root=receipt_root(pool_root)
    jid='exp-'+uuid.uuid4().hex[:12]
    path=root/(jid+'.json')
    log=root/(jid+'.log')
    record={'id':jid,'name':name,'kind':'exploration','status':'running','created_at':now(),
            'started_at':now(),'updated_at':now(),'log_path':str(log),'result_directory':None,
            'pid':os.getpid(),'error':None}
    with LOCK: write_receipt(path,record)
    with log.open('a',encoding='utf-8') as f:
        f.write(f'[{now()}] Started saved Exploration question: {name}\n');f.flush()
    try:
        result=run_one(pool_root,name)
        record.update(status='completed',result_directory=result.get('result_directory'),
                      problem_type=result.get('problem_type'),completed_at=now(),updated_at=now())
        with log.open('a',encoding='utf-8') as f:f.write(f'[{now()}] Completed: {name}\n')
        return result
    except BaseException as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}',updated_at=now(),completed_at=now())
        with log.open('a',encoding='utf-8') as f:f.write(f'[{now()}] Failed: {record["error"]}\n')
        raise
    finally:
        with LOCK:write_receipt(path,record)
