"""One versioned, audited YAML pool; all native solver configuration is preserved."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
import yaml

VERSION='1.0'
ID=re.compile(r'^[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}$')
TYPES={'exploration.calculator','exploration.network_visualization','exploration.activity'}


def _yaml(mapping):
    return yaml.safe_dump(mapping,sort_keys=False,allow_unicode=True)


def validate(item):
    if not isinstance(item,dict) or item.get('schema_version')!=VERSION:raise ValueError('Invalid query envelope version')
    pid=item.get('problem_id')
    if not isinstance(pid,str) or not ID.fullmatch(pid):raise ValueError('Invalid problem identifier')
    kind=item.get('problem_type')
    if kind not in TYPES:raise ValueError('Invalid problem kind')
    payload=item.get('payload')
    if not isinstance(payload,dict):raise ValueError('Native payload required')
    if kind=='exploration.calculator':
        from gem_exploration.config import validate_problem
        internal=payload.get('internal_problem')
        if not isinstance(internal,dict):raise ValueError('Compiled native calculator problem required')
        validate_problem(internal)
        public=payload.get('public_question')
        if not isinstance(public,dict):raise ValueError('Readable native calculator question required')
    elif kind=='exploration.activity':
        from .activity_queries import validate_payload
        validate_payload(payload)
    else:
        from gem_atlas_network.problems import Problem
        Problem.model_validate(payload)
        if payload.get('problem_id')!=pid:raise ValueError('Network identity mismatch')
    return item


class QueryPool:
    def __init__(self,root):
        self.root=Path(root).expanduser().resolve();self.root.mkdir(parents=True,exist_ok=True)
        (self.root/'problems').mkdir(exist_ok=True)
        (self.root/'results').mkdir(exist_ok=True)

    def save(self,item,overwrite=False):
        validate(item)
        pid=item['problem_id'];path=self.root/'problems'/f'{pid}.yaml'
        if path.exists() and not overwrite:raise FileExistsError(f'{pid} already exists. Use a new ID, or deliberately overwrite.')
        data=_yaml(item)
        fd,temporary=tempfile.mkstemp(dir=path.parent,prefix='.query_',suffix='.yaml')
        try:
            with os.fdopen(fd,'w',encoding='utf-8') as f:f.write(data);f.flush();os.fsync(f.fileno())
            os.replace(temporary,path)
        finally:
            if os.path.exists(temporary):os.unlink(temporary)
        return path

    def save_calculator(self,public,internal,name=None):
        name=name or public.get('name') or internal.get('name')
        pid=re.sub(r'[^a-zA-Z0-9_-]+','_',str(name or 'calculation')).strip('_')[:80]
        if not pid:raise ValueError('Question needs a name')
        item={'schema_version':VERSION,'problem_id':pid,'problem_type':'exploration.calculator',
              'payload':{'public_question':public,'internal_problem':internal},
              'execution':{'new_fba_policy':'never','landscape_copy':'only_if_new_fba'}}
        return self.save(item)

    def save_activity(self,problem_id,payload):
        return self.save({'schema_version':VERSION,'problem_id':problem_id,'problem_type':'exploration.activity',
                          'payload':payload,'execution':{'new_fba_policy':'never'}})

    def save_network(self,native):
        return self.save({'schema_version':VERSION,'problem_id':native['problem_id'],
              'problem_type':'exploration.network_visualization','payload':native,
              'execution':{'new_fba_policy':'native_per_query','landscape_copy':'not_applicable'}})

    def load(self,pid):
        if not isinstance(pid,str) or not ID.fullmatch(pid):raise ValueError('Invalid problem identifier')
        path=self.root/'problems'/f'{pid}.yaml'
        if path.stat().st_size>2_000_000:raise ValueError('Query exceeds 2 MB')
        return validate(yaml.safe_load(path.read_text(encoding='utf-8')))

    def names(self):return [f.stem for f in sorted((self.root/'problems').glob('*.yaml')) if ID.fullmatch(f.stem)]

    def import_native(self,path):
        data=yaml.safe_load(Path(path).read_text(encoding='utf-8'))
        if not isinstance(data,dict):raise ValueError('Expected YAML mapping')
        if data.get('payload') is not None:return self.save(data)
        if data.get('problem_type')=='exploration.network_visualization':return self.save_network(data)
        raise ValueError('Calculator imports need a compiled native problem plus readable question; save via the Calculator section.')
