"""Sequential resumable jobs, atomic manifests and failure isolation."""
from pathlib import Path
import json,hashlib,time,csv
from datetime import datetime,timezone
from .engine import execute,prepare,RELEASE
from .problems import config_hash,sha256,parse_yaml
from .pathways import LIBRARY_SHA256
from .ui_helpers import new_run_root

class BatchRunner:
    def __init__(self,output_root):
        self.root=Path(output_root).resolve();self.root.mkdir(parents=True,exist_ok=True)
        self.path=self.root/'batch_manifest.json';self.pause_requested=False;self.running=False
        self.records=json.loads(self.path.read_text()).get('jobs',{}) if self.path.exists() else {}
    def pause(self):self.pause_requested=True
    def save(self):
        temp=self.path.with_suffix('.tmp');temp.write_text(json.dumps({'release':RELEASE,'timestamp_utc':datetime.now(timezone.utc).isoformat(),'jobs':self.records},indent=2));temp.replace(self.path)
        with (self.root/'batch_results.csv').open('w',newline='') as f:
            keys=['problem_id','status','configuration_sha256','model_sha256','duration_seconds','solver_calls','error','output_directory']
            w=csv.DictWriter(f,fieldnames=keys,extrasaction='ignore');w.writeheader();w.writerows(self.records.values())
    def validate(self,jobs):
        if len(jobs)>200:raise ValueError('Queue is limited to 200 jobs')
        ids=[job[0].problem_id for job in jobs]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate problem IDs in batch; choose unique names')
        rows=[]
        for p,base,allowed,supplied in jobs:
            try:prepare(p,base,allowed,supplied);error=''
            except Exception as e:error=str(e)
            rows.append({'problem_id':p.problem_id,'mode':p.analysis.mode,'subject':p.subject.kind,'ready':not error,'error':error})
        return rows
    def run(self,jobs,resume=True,progress=None):
        if self.running:raise ValueError('Batch is already running')
        self.validate(jobs);self.running=True;self.pause_requested=False
        try:
            for i,(p,base,allowed,supplied) in enumerate(jobs):
                if self.pause_requested:break
                conf=config_hash(p);row={'problem_id':p.problem_id,'configuration_sha256':conf,'status':'validating'}
                try:
                    _,_,path=prepare(p,base,allowed,supplied);digest=sha256(path)
                    signature=hashlib.sha256((conf+digest+LIBRARY_SHA256+RELEASE).encode()).hexdigest()
                    old=self.records.get(p.problem_id,{})
                    result_path=self.root/old.get('output_directory','__missing__')/'result.json'
                    if resume and old.get('status')=='complete' and old.get('signature')==signature and result_path.is_file():
                        if progress:progress(i+1,len(jobs),{**old,'resume_action':'skipped identical completed job'})
                        continue
                    row.update(model_sha256=digest,signature=signature,status='running');self.records[p.problem_id]=row;self.save()
                    if progress:progress(i,len(jobs),row)
                    job_root=new_run_root(self.root,p.problem_id)
                    r=execute(p,base,job_root,allowed,supplied)
                    good=all(v['status']=='optimal' for v in r['solutions'].values())
                    row.update(status='complete' if good else 'failed',solver_calls=r['solver_calls'],duration_seconds=r['duration_seconds'],run_root=str(job_root.relative_to(self.root)),output_directory=str((job_root/r['output_directory']).relative_to(self.root)),error='' if good else 'FBA did not return optimal; inspect result.json')
                except Exception as e:row.update(status='failed',error=str(e))
                self.records[p.problem_id]=row;self.save()
                if progress:progress(i+1,len(jobs),row)
        finally:self.running=False;self.save()
        return self.records
