"""Read-only unified job browsing across landscape, Optimization and Exploration."""
from __future__ import annotations
import json
import os
from datetime import datetime,timezone
from pathlib import Path
from urllib.parse import quote

ACTIVE={'running','cancelling','terminating'}

def _read(path):
    try:return json.loads(Path(path).read_text(encoding='utf-8'))
    except (ValueError,OSError,TypeError):return None

def _safe_log(path):
    if not path:return ''
    try:
        p=Path(path)
        with p.open('rb') as f:
            f.seek(0,2);f.seek(max(f.tell()-80000,0));return f.read().decode('utf-8',errors='replace')
    except OSError:return ''

def list_all(workspace,jobs,opt_queue):
    out=[]
    for j in jobs.list(limit=150):
        d=j.__dict__.copy()
        out.append({'key':'landscape:'+j.job_id,'id':j.job_id,'category':'landscape','kind':j.kind,
                    'label':f'{j.kind} · {Path(j.config_path).stem if j.config_path else j.job_id}',
                    'status':j.status,'started_at':j.started_at_utc,'config_path':j.config_path,
                    'log_path':j.log_path,'pid':j.pid})
    if opt_queue is not None:
        for j in opt_queue.refresh().get('jobs',[]):
            jid=j.get('id','')
            out.append({'key':'optimization:'+jid,'id':jid,'category':'optimization','kind':j.get('family','optimization'),
                        'label':f'{j.get("case","Optimization")} · {j.get("method","method")}',
                        'status':j.get('status','unknown'),'started_at':j.get('created_at',0),
                        'log_path':str(opt_queue.root/jid/'run.log'),'pid':j.get('pid'),
                        'calls':j.get('calls',0),'budget':j.get('budget',0)})
    for f in sorted((workspace.root/'exploration_workspace'/'run_records').glob('exp-*.json'))[-500:]:
        d=_read(f)
        if not isinstance(d,dict):continue
        out.append({'key':'exploration:'+d['id'],'id':d['id'],'category':'exploration',
                    'kind':d.get('problem_type') or 'exploration','label':d.get('name',d['id']),
                    'status':d.get('status','unknown'),'started_at':d.get('started_at',''),
                    'log_path':d.get('log_path'),'pid':d.get('pid')})
    def order(j):
        v=j.get('started_at',0)
        return str(v) if isinstance(v,str) else (datetime.fromtimestamp(v,timezone.utc).isoformat() if v else '')
    return sorted(out,key=order,reverse=True)

def get_one(workspace,jobs,opt_queue,key):
    index={j['key']:j for j in list_all(workspace,jobs,opt_queue)}
    if key not in index:raise KeyError('Job not found')
    j=dict(index[key]);category=j['category'];jid=j['id']
    artifacts=[];details={};log=_safe_log(j.get('log_path'))
    if category=='landscape':
        config=j.get('config_path')
        if config:
            try:
                from .config import load_config
                conf=load_config(config)
                output=Path(conf.project.output_dir).expanduser()
                if not output.is_absolute():output=workspace.root/output
                output=output.resolve()
                meta=output/'metadata'/'config_snapshot.yaml'
                details['output_directory']=str(output)
                details['configuration']=str(meta if meta.is_file() else config)
                for file in (output/'exports').glob('*.csv') if (output/'exports').is_dir() else []:
                    artifacts.append({'name':file.name,'kind':'csv','path':str(file)})
            except Exception as exc:details['output_error']=str(exc)
    elif category=='optimization':
        base=opt_queue.root/jid
        for fn in ('summary.json','provenance.json','worker_state.json'):
            data=_read(base/fn)
            if data is not None:details[fn[:-5]]=data
        for fn in ('hv_trajectory.csv','pareto_front.csv'):
            p=base/fn
            if p.is_file():
                import pandas as pd
                try:details[fn[:-4]]=json.loads(pd.read_csv(p).head(1500).to_json(orient='records'))
                except (ValueError,pd.errors.EmptyDataError):details[fn[:-4]]=[]
        details['output_directory']=str(base)
        details['export_url']='/api/optimization/job/'+jid+'/export' if j['status'] not in ACTIVE else None
    else:
        record=_read(workspace.root/'exploration_workspace'/'run_records'/(jid+'.json')) or {}
        if record.get('error'):details['error']=record['error']
        output=record.get('result_directory')
        if output:
            base=(workspace.root/'exploration_workspace'/'results').resolve()
            p=Path(output).resolve()
            if p.is_dir() and p.is_relative_to(base):
                details['output_directory']=str(p)
                s=_read(p/'run_summary.json')
                if s:details['summary']=s
                for file in sorted(p.rglob('*')):
                    if not file.is_file() or len(artifacts)>=50:break
                    if file.suffix.lower() not in {'.png','.pdf','.svg','.md','.json','.csv','.html'}:continue
                    rel=file.relative_to(base).as_posix()
                    artifacts.append({'name':file.name,'kind':file.suffix.lstrip('.'),
                                      'url':'/exploration-results/'+quote(rel,safe='/')})
    j['details']=details;j['artifacts']=artifacts;j['log']=log
    return j

def active(jobs_list):
    return any(j.get('status') in ACTIVE for j in jobs_list)
