"""Persistent sequential queue and authenticated GUI endpoints."""
from __future__ import annotations
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import uuid
import zipfile
import yaml
from fastapi import Body, HTTPException
from fastapi.responses import FileResponse
from .catalog import ASSETS, SPECIES, FINAL_SELECTED_METHODS, canonical_cases, methods, sha, verify_sources, automatic_selection_method
from .automatic_selection import AUTO_ID, DECISION_NAME
from .problems import LABELS, template_bytes, describe, validate_content, create_custom
from .worker import atomic_json
from .bindings import bind_native, list_bindings
from .general import inspect as inspect_general, register as register_general, list_general, preview_problem, sample_sources
from .landscape_config import read_existing_landscape_config

class Queue:
    def __init__(self, workspace):
        self.workspace = workspace
        self.root = workspace.root / 'optimization_runs'; self.root.mkdir(exist_ok=True)
        self.path = self.root / 'queue.json'; self.lock = threading.RLock()
        self.process = None; self.thread = None
        self._closed = False
        self.state = json.loads(self.path.read_text()) if self.path.exists() else {'enabled': False, 'jobs': []}
        self.refresh()
        if self.state['enabled']: self.start()

    def save(self): atomic_json(self.path, self.state)
    def jobdir(self, jid):
        if not any(j['id']==jid for j in self.state['jobs']): raise ValueError('Unknown queue job')
        return self.root / jid
    def refresh(self):
        with self.lock:
            for j in self.state['jobs']:
                out = self.root / j['id']; receipt = out / 'worker_state.json'
                if j['status'] in ('running','cancelling'):
                    r = json.loads(receipt.read_text()) if receipt.exists() else {}
                    if r.get('status') in ('completed','failed','cancelled'):
                        j.update(status=r['status'], error=r.get('error'), runtime_seconds=r.get('runtime_seconds'))
                    elif not self.worker_active(out) and time.time()-j.get('launched_at',0)>3:
                        j.update(status='interrupted', error='Worker ended without a completion receipt; resume saved history')
                record = out / DECISION_NAME
                actual_method = j['method']
                if j['method']==AUTO_ID and record.is_file():
                    decision=json.loads(record.read_text())
                    actual_method=decision['selected_method']
                    j['selected_method']=actual_method
                    j['selected_family']=decision['selected_family']
                scientific = out / 'scientific_results' / j['case'] / f"seed_{j['seed']}" / actual_method / 'calls.csv' 
                if scientific.exists():
                    # Count complete CSV records through pandas, not raw line count.
                    try:
                        import pandas as pd
                        j['calls'] = len(pd.read_csv(scientific, usecols=['call']))
                    except Exception: pass
            self.save()
            return json.loads(json.dumps(self.state))

    @staticmethod
    def worker_active(out):
        path = out / 'worker.lock'
        if not path.exists(): return False
        with path.open('a') as lock:
            try: fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError: return True
        return False

    def enqueue(self, request):
        if request.get('general_binding_id'):
            return self.enqueue_general(request)
        case = next((c for c in canonical_cases()+list_bindings(self.workspace) if c['case']==request['case']), None)
        if case is None: raise ValueError('Select a supported source case')
        n = int(request.get('prior_rows',5000)); seed = int(request.get('seed',20260822)); budget=int(request.get('budget',128))
        if n<5000 or n>50000: raise ValueError('Prior size must be 5,000–50,000')
        if budget<4 or budget>4096 or budget%4: raise ValueError('Budget must be a multiple of 4, from 4 through 4096')
        content=request.get('problem_yaml') or template_bytes(case['problem_key']).decode()
        solver=request.get('exact_solver','glpk')
        if solver not in ('glpk','glpk_presolve'):raise ValueError('Unsupported exact solver')
        info=describe(content)
        requested=request.get('methods') or ['LANDSCAPE_ONLY']
        catalog=[]
        for family in ('gsmobo','nsga'):
            try: catalog.extend(methods(family,n))
            except ValueError: pass
        if requested==['ALL']: requested=['LANDSCAPE_ONLY']+[m['id'] for m in catalog if m['id'] in FINAL_SELECTED_METHODS[m['family']]]
        catalog.append(automatic_selection_method())
        if len(requested)!=len(set(requested)): raise ValueError('Duplicate methods in assignment')
        if any(mid!='LANDSCAPE_ONLY' and mid not in {m['id'] for m in catalog} for mid in requested):
            raise ValueError('Method not supported for this prior size')
        created=[]
        with self.lock:
            for mid in requested:
                meta=next((m for m in catalog if m['id']==mid),None)
                if mid!='LANDSCAPE_ONLY' and meta is None: raise ValueError('Method not supported for this prior size')
                family=('nsga' if mid==AUTO_ID else meta['family']) if meta else ('gsmobo' if n in (5000,10000,25000,50000) else 'nsga')
                model=request.get('model_path') or case['model_path']; landscape=request.get('landscape_path') or case['landscape_path']
                # Files from the built-in catalog are immutable; custom paths must be in workspace.
                if model!=case['model_path']: model=str(self.workspace.resolve(model,must_exist=True))
                if landscape!=case['landscape_path']: landscape=str(self.workspace.resolve(landscape,must_exist=True))
                if sha(model)!=case['model_sha256']: raise ValueError('Model/landscape identity mismatch')
                if sha(landscape)!=sha(case['landscape_path']): raise ValueError('Landscape/source identity mismatch')
                jid=uuid.uuid4().hex[:16]
                job={'id':jid,'case':case['case'],'family':family,'method':mid,'prior_rows':n,'seed':seed,'budget':budget,
                     'problem_yaml':content,'problem_hash':info['problem_hash'],'model_path':model,'landscape_path':landscape,
                     'status':'queued','created_at':time.time(),'calls':0,'meta':meta}
                job['exact_solver']=solver
                if case['species'].startswith('custom_'):job['binding']=case
                out=self.root/jid;out.mkdir();atomic_json(out/'job.json',job);self.state['jobs'].append(job);created.append(job)
            self.save()
        if self.state['enabled']: self.start()
        return created

    def enqueue_general(self, request):
        registered=next((b for b in list_general(self.workspace) if b['id']==request['general_binding_id']),None)
        if registered is None:raise ValueError('Select a registered model + landscape; presets are optional')
        n=int(request.get('prior_rows',min(5000,registered['rows'])))
        budget=int(request.get('budget',128));seed=int(request.get('seed',20260822))
        if n<4 or n>registered['rows']:raise ValueError(f'Prior size must be 4–{registered["rows"]} rows')
        if budget<4 or budget>4096 or budget%4:raise ValueError('Budget must be a multiple of 4, from 4 to 4096')
        content=request.get('problem_yaml')
        if not content:raise ValueError('Create or load a valid YAML problem first')
        info=describe(content)
        preview=preview_problem(self.workspace,registered,content,n)  # fail closed before queueing final selected methods
        if preview['prior_feasible']<4:
            raise ValueError('Fewer than four viable prior designs; choose another landscape, problem or prefix size')
        solver=request.get('exact_solver','glpk')
        if solver not in ('glpk','glpk_presolve'):raise ValueError('Unsupported exact solver')
        catalog=sum([methods(f,n,generalized=True) for f in ('gsmobo','nsga')],[])
        requested=request.get('methods') or ['LANDSCAPE_ONLY']
        if requested==['ALL']:requested=['LANDSCAPE_ONLY']+[m['id'] for m in catalog]
        catalog.append(automatic_selection_method())
        if len(requested)!=len(set(requested)):raise ValueError('Duplicate methods in assignment')
        if any(m!='LANDSCAPE_ONLY' and m not in {c['id'] for c in catalog} for m in requested):
            raise ValueError('Unknown optimizer method')
        case=registered['case_prefix']+'__'+info['id']
        created=[]
        with self.lock:
            for mid in requested:
                meta=next((m for m in catalog if m['id']==mid),None)
                family=('nsga' if mid==AUTO_ID else meta['family']) if meta else 'gsmobo'
                jid=uuid.uuid4().hex[:16]
                job={'id':jid,'case':case,'family':family,'method':mid,'prior_rows':n,'seed':seed,
                     'budget':budget,'problem_yaml':content,'problem_hash':info['problem_hash'],
                     'model_path':registered['model_path'],'landscape_path':registered['landscape_path'],
                     'general_binding':registered,'general_binding_id':registered['id'],
                     'meta':meta,'exact_solver':solver,'status':'queued',
                     'created_at':time.time(),'calls':0}
                out=self.root/jid;out.mkdir();atomic_json(out/'job.json',job)
                self.state['jobs'].append(job);created.append(job)
            self.save()
        if self.state['enabled']:self.start()
        return created

    def start(self):
        with self.lock:
            self.state['enabled']=True;self.save()
            if self.thread and self.thread.is_alive():return
            self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()

    def loop(self):
        # A server instance owns the queue runner lock for the entire dispatch loop.
        with (self.root/'queue.lock').open('a') as controller:
            try:fcntl.flock(controller,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:return
            while not self._closed:
                with self.lock:
                    self.refresh()
                    if not self.state['enabled']:return
                    if any(j['status'] in ('running','cancelling') for j in self.state['jobs']):job=None
                    else:job=next((j for j in self.state['jobs'] if j['status']=='queued'),None)
                    if job:
                        out=self.root/job['id'];receipt=out/'worker_state.json'
                        if receipt.exists():receipt.unlink()
                        job.update(status='running',launched_at=time.time());self.save()
                        env=os.environ.copy();env['PYTHONPATH']=str(Path(__file__).resolve().parents[2])+os.pathsep+env.get('PYTHONPATH','')
                        env.setdefault('OMP_NUM_THREADS','1');env.setdefault('OPENBLAS_NUM_THREADS','1')
                        with (out/'run.log').open('a') as log:
                            self.process=subprocess.Popen([sys.executable,'-m','gem_learning_engine.optimization.worker','--job',str(out/'job.json')],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
                        job['pid']=self.process.pid;self.save()
                if self.process:
                    self.process.wait();self.process=None
                else:time.sleep(.25)

    def action(self, jid, action):
        with self.lock:
            job=next((j for j in self.state['jobs'] if j['id']==jid),None)
            if job is None:raise ValueError('Unknown job')
            if action=='cancel':
                if job['status']=='queued':job['status']='cancelled'
                elif job['status']=='running':
                    if self.worker_active(self.root/jid):
                        try:os.killpg(job['pid'],signal.SIGTERM)
                        except ProcessLookupError:pass
                    job['status']='cancelling'
            elif action=='resume':
                if job['status'] not in ('cancelled','interrupted','failed'):raise ValueError('Only stopped jobs can resume')
                job.update(status='queued',error=None)
            else:raise ValueError('Unknown job action')
            self.save()
        if self.state['enabled']:self.start()
        return job

def attach(app,workspace,auth):
    queue=Queue(workspace);app.state.optimization_queue=queue
    def safe(fn):
        try:return fn()
        except (ValueError,RuntimeError,KeyError,FileNotFoundError,yaml.YAMLError) as e:raise HTTPException(400,detail=str(e)) from e
    @app.get('/api/optimization/catalog',dependencies=[auth])
    def catalog(prior_rows:int=5000):
        bindings=list_bindings(workspace)
        return safe(lambda:{'species':{**SPECIES,**{b['species']:b['label'] for b in bindings}},'problems':LABELS,'cases':canonical_cases()+bindings,
                            'methods':sum([methods(f,prior_rows,generalized=True) for f in ('gsmobo','nsga')],[])+[automatic_selection_method()], 'general_bindings':list_general(workspace),
                            'integrity':verify_sources()})
    @app.get('/api/optimization/files',dependencies=[auth])
    def files(kind:str='model'):return safe(lambda:{'files':sample_sources(workspace,kind)})
    @app.post('/api/optimization/general/from-landscape-config',dependencies=[auth])
    def general_from_landscape_config(payload:dict=Body(...)):
        return safe(lambda:read_existing_landscape_config(workspace,payload['config_path']))
    @app.post('/api/optimization/general/inspect',dependencies=[auth])
    def general_inspect(payload:dict=Body(...)):
        return safe(lambda:inspect_general(workspace,payload['model_path'],payload['landscape_path']))
    @app.post('/api/optimization/general/register',dependencies=[auth])
    def general_register(payload:dict=Body(...)):return safe(lambda:register_general(workspace,payload))
    @app.post('/api/optimization/general/preview',dependencies=[auth])
    def general_preview(payload:dict=Body(...)):
        def compute():
            b=next((x for x in list_general(workspace) if x['id']==payload['general_binding_id']),None)
            if b is None:raise ValueError('Unknown general binding')
            return preview_problem(workspace,b,payload['problem_yaml'],payload.get('prior_rows',min(b['rows'],5000)))
        return safe(compute)
    @app.post('/api/optimization/general/import-native',dependencies=[auth])
    def import_native(payload:dict=Body(...)):
        def perform():
            from .general import register as register_source
            native=bind_native(workspace,payload['config_path'])
            meta=yaml.safe_load((Path(native['source_benchmark'])/'prepared'/native['source_case']/'problem_config.yaml').read_text())['problem']
            new_source=register_source(workspace,{'label':native['label'],
                'model_path':native['model_path'],'landscape_path':native['landscape_path'],
                'features':meta['features'],'bounds':meta['bounds'],
                'biomass_reaction_id':meta['biomass_reaction_id'],
                'product_reaction_id':meta['product_reaction_id'],
                'minimum_growth_fraction':meta['minimum_growth_fraction'],
                'feature_mode':'medium_uptake','fba_mode':'growth_constrained_product',
                'landscape_seed':native['landscape_seed']})
            return {'general_binding':new_source,'native_source':native,
                    'warning':'New exact FBA calculations use registered generic adapter; historical solver equivalence requires a separate replay check'}
        return safe(perform)
    @app.post('/api/optimization/bind-native',dependencies=[auth])
    def bind(payload:dict=Body(...)):return safe(lambda:bind_native(workspace,payload['config_path']))
    @app.get('/api/optimization/template',dependencies=[auth])
    def template(key:str):return safe(lambda:{'yaml':template_bytes(key).decode(),**describe(template_bytes(key).decode())})
    @app.post('/api/optimization/problem/validate',dependencies=[auth])
    def validate(payload:dict=Body(...)):return safe(lambda:describe(payload['yaml']))
    @app.post('/api/optimization/problem/custom',dependencies=[auth])
    def custom(payload:dict=Body(...)):
        return safe(lambda:{'yaml':create_custom(payload['id'],payload.get('title',''),payload['objectives'],payload.get('constraints',[]))})
    @app.post('/api/optimization/problem/save',dependencies=[auth])
    def save_problem(payload:dict=Body(...)):
        def save():
            info=describe(payload['yaml']);path=workspace.resolve(payload['path'])
            if path.exists() and not payload.get('overwrite',False):raise ValueError('File exists; select overwrite to edit it')
            path.parent.mkdir(parents=True,exist_ok=True);path.write_text(payload['yaml']);return {'path':workspace.display(path),**info}
        return safe(save)
    @app.get('/api/optimization/problem/load',dependencies=[auth])
    def load_problem(path:str):return safe(lambda:{'yaml':workspace.resolve(path,must_exist=True).read_text()})
    @app.post('/api/optimization/queue',dependencies=[auth])
    def enqueue(payload:dict=Body(...)):return safe(lambda:{'jobs':queue.enqueue(payload)})
    @app.get('/api/optimization/queue',dependencies=[auth])
    def state():return queue.refresh()
    @app.post('/api/optimization/queue/start',dependencies=[auth])
    def start():queue.start();return queue.refresh()
    @app.post('/api/optimization/queue/pause',dependencies=[auth])
    def pause():
        with queue.lock:queue.state['enabled']=False;queue.save()
        return queue.refresh()
    @app.post('/api/optimization/job/{jid}/{action}',dependencies=[auth])
    def action(jid:str,action:str):return safe(lambda:queue.action(jid,action))
    @app.get('/api/optimization/job/{jid}',dependencies=[auth])
    def job(jid:str):
        def view():
            root=queue.jobdir(jid);data={}
            for name in ['summary','provenance','worker_state','environment']:
                p=root/(name+'.json');data[name]=json.loads(p.read_text()) if p.exists() else None
            p=root/'run.log';data['log']=p.read_text(errors='replace')[-40000:] if p.exists() else ''
            import pandas as pd
            for name in ['hv_trajectory','pareto_front']:
                p=root/(name+'.csv')
                try:data[name]=json.loads(pd.read_csv(p).to_json(orient='records')) if p.exists() else []
                except pd.errors.EmptyDataError:data[name]=[]
            return data
        return safe(view)
    @app.get('/api/optimization/job/{jid}/export',dependencies=[auth])
    def export(jid:str):
        def make():
            root=queue.jobdir(jid)
            if queue.worker_active(root):raise ValueError('Pause or finish the worker before exporting its complete files')
            path=queue.root/(jid+'.zip')
            with zipfile.ZipFile(path,'w',zipfile.ZIP_DEFLATED) as z:
                for f in root.rglob('*'):
                    if f.is_file() and 'sources' not in f.relative_to(root).parts and f.name!='worker.lock':z.write(f,f.relative_to(root))
            return FileResponse(path,filename=f'GEM_optimization_{jid}.zip',media_type='application/zip')
        return safe(make)
