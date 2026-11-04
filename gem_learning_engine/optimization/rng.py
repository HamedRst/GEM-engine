"""Reproducible execution state around unchanged scientific optimizer modules."""
from pathlib import Path
import hashlib,json,random
import numpy as np

def identity(job):
    keys=('family','case','method','seed')
    return hashlib.sha256(json.dumps({k:job[k] for k in keys},sort_keys=True).encode()).hexdigest()

def seed(job):
    import torch
    value=int(identity(job)[:8],16)
    random.seed(value);np.random.seed(value);torch.manual_seed(value)
    return value

def snapshot(job,calls):
    import torch
    state=np.random.get_state()
    return {'policy':'explicit_rng_v1','identity':identity(job),'calls':int(calls),
            'torch_cpu':torch.get_rng_state().tolist(),'numpy':[state[0],state[1].tolist(),state[2],state[3],state[4]],'python':random.getstate()}

def restore(state):
    import torch
    def tuples(obj):return tuple(tuples(x) for x in obj) if isinstance(obj,list) else obj
    torch.set_rng_state(torch.tensor(state['torch_cpu'],dtype=torch.uint8))
    n=state['numpy'];np.random.set_state((n[0],np.asarray(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    random.setstate(tuples(state['python']))

def begin(job,out,calls):
    path=Path(out)/'rng_checkpoints'/f'{int(calls):06d}.json'
    if calls:
        if not path.exists():
            raise RuntimeError('This history predates reproducible RNG checkpoints. Keep it as evidence and assign a fresh job; exact continuation cannot be certified.')
        state=json.loads(path.read_text())
        if state['identity']!=identity(job) or state['calls']!=calls:raise ValueError('RNG checkpoint does not match this job history')
        restore(state);return {'policy':'explicit_rng_v1','resumed_from_calls':calls}
    value=seed(job)
    return {'policy':'explicit_rng_v1','execution_seed':value,'resumed_from_calls':0}
