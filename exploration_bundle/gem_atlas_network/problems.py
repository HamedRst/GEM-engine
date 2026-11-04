"""Versioned safe configuration contract and portable bundles. Never solves FBA."""
from pathlib import Path, PurePosixPath
import json,hashlib,re,zipfile,stat,copy
import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing import Literal
from .pathways import LIBRARY_ID, LIBRARY_VERSION, TEMPLATES
ROOT=Path(__file__).resolve().parents[1]
REGISTRY={p.stem:p for p in (ROOT/'examples/models').glob('*')}
MAX_YAML=2*1024*1024
MAX_BUNDLE=150*1024*1024

def sha256(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
class Strict(BaseModel):
    model_config=ConfigDict(extra='forbid', allow_inf_nan=False)
class ModelRef(Strict):
    path:str|None=None
    registry_id:str|None=None
    sha256:str|None=None
    @model_validator(mode='after')
    def valid(self):
        if not self.path and not self.registry_id:raise ValueError('A model path or registry ID is required')
        if self.path: relative(self.path,allow_parent=True)
        if self.sha256 and not re.fullmatch('[a-f0-9]{64}',self.sha256):raise ValueError('Invalid model SHA256')
        return self
class RouteQuery(Strict):
    substrate:str|None=None
    product:str|None=None
    required_metabolites:list[str]=Field(default_factory=list,max_length=12)
    required_reactions:list[str]=Field(default_factory=list,max_length=20)
    allowed_reactions:list[str]=Field(default_factory=list,max_length=200)
    excluded_metabolites:list[str]=Field(default_factory=list,max_length=40)
    excluded_reactions:list[str]=Field(default_factory=list,max_length=40)
    ordered_intermediates:bool=True
    hide_currency:bool=True
    feasibility:Literal['topology','steady_state']='topology'
    only_lp_feasible:bool=False
    max_reaction_steps:int=Field(default=8,ge=1,le=15)
    min_reaction_steps:int=Field(default=1,ge=1,le=15)
    max_results:int=Field(default=10,ge=1,le=30)
    max_candidates:int=Field(default=600,ge=10,le=5000)
    max_frontier:int=Field(default=12000,ge=100,le=50000)
    timeout_seconds:float=Field(default=12.0,gt=0,le=90)
    lp_time_limit_seconds:float=Field(default=4.0,gt=0,le=30)
    min_flux:float=Field(default=0.001,gt=0,le=10)
    @model_validator(mode='after')
    def valid(self):
        if not (self.substrate or self.product):raise ValueError('Specify a starting metabolite, a product, or both')
        if self.substrate and self.substrate==self.product:raise ValueError('Start and product must differ')
        if self.min_reaction_steps>self.max_reaction_steps:raise ValueError('Minimum route steps exceed maximum')
        if self.only_lp_feasible and self.feasibility!='steady_state':raise ValueError('Only LP-feasible routes requires steady_state LP evidence level')
        if set(self.required_metabolites)&set(self.excluded_metabolites):raise ValueError('Required and excluded metabolites conflict')
        if set(self.required_reactions)&set(self.excluded_reactions):raise ValueError('Required and excluded reactions conflict')
        if self.allowed_reactions and not set(self.required_reactions).issubset(self.allowed_reactions):raise ValueError('Required reactions must also be allowed')
        if set(self.allowed_reactions)&set(self.excluded_reactions):raise ValueError('Allowed and excluded reactions conflict')
        if any(m in self.excluded_metabolites for m in [self.substrate,self.product] if m):raise ValueError('Endpoint cannot be excluded')
        return self

class Subject(Strict):
    kind:Literal['network','target_neighborhood','recognized_pathway','genes','route_search']='network'
    target_id:str|None=None
    entity_kind:Literal['metabolite','reaction']='metabolite'
    pathway_id:str|None=None
    variant_id:Literal['canonical_v1']='canonical_v1'
    branch_only:bool=False
    gene_ids:list[str]=Field(default_factory=list,max_length=1000)
    route:RouteQuery|None=None
    subsystem:str|None=None
class Analysis(Strict):
    mode:Literal['structural','fba']='structural'
    knockout_gene_ids:list[str]=Field(default_factory=list,max_length=1000)
    compare_to_wild_type:bool=False
class Objective(Strict):
    source:Literal['model','reaction']='model'
    reaction_id:str|None=None
class Medium(Strict):
    source:Literal['model_encoded','custom']='model_encoded'
    uptake_bounds:dict[str,float]=Field(default_factory=dict)
    close_unselected:bool=True
class FBA(Strict):
    objective:Objective=Field(default_factory=Objective)
    direction:Literal['max','min']='max'
    medium:Medium=Field(default_factory=Medium)
    flux_threshold:float=Field(default=1e-8,gt=0)
    bounds_overrides:dict[str,tuple[float,float]]=Field(default_factory=dict)
class Library(Strict):
    id:Literal['atlas_canonical']='atlas_canonical'
    version:Literal['1.0.0']='1.0.0'
class Render(Strict):
    style_preset:Literal['atlas_v023_publication']='atlas_v023_publication'
    hide_currency_metabolites:bool=True
    max_display_reactions:int=Field(default=28,ge=1,le=200)
    depth:int=Field(default=3,ge=1,le=8)
    direction:Literal['upstream','downstream','both']='upstream'
    label_mode:Literal['names','ids']='names'
    include_legend:bool=True
    show_flux_values:bool=False
    active_only:bool=False
    gene_context_depth:int=Field(default=1,ge=0,le=3)
    gene_context_max_reactions:int=Field(default=16,ge=2,le=60)
    dpi:int=Field(default=300,ge=72,le=600)
    export_formats:list[Literal['svg','pdf','png']]=Field(default_factory=lambda:['svg','pdf','png'],min_length=1)
class Output(Strict):
    subdirectory:str|None=None
    @model_validator(mode='after')
    def valid(self):
        if self.subdirectory:relative(self.subdirectory)
        return self
class Problem(Strict):
    schema_version:Literal['1.0']='1.0'
    problem_type:Literal['exploration.network_visualization']='exploration.network_visualization'
    problem_id:str
    description:str=''
    model:ModelRef
    subject:Subject=Field(default_factory=Subject)
    analysis:Analysis=Field(default_factory=Analysis)
    fba:FBA|None=None
    pathway_library:Library=Field(default_factory=Library)
    render:Render=Field(default_factory=Render)
    output:Output=Field(default_factory=Output)
    metadata:dict=Field(default_factory=dict)
    @model_validator(mode='after')
    def valid(self):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,79}',self.problem_id):raise ValueError('Problem ID must use 1–80 letters, digits, underscores or hyphens')
        if self.analysis.mode=='fba' and self.fba is None:raise ValueError('FBA configuration is required in FBA mode')
        if self.analysis.mode=='structural' and self.fba is not None:raise ValueError('Structural problems cannot include FBA settings')
        if self.subject.kind=='route_search' and self.subject.route is None:raise ValueError('Route search needs route query settings')
        if self.subject.kind!='route_search' and self.subject.route is not None:raise ValueError('Route query only applies to Route Explorer')
        if self.subject.kind=='route_search' and self.analysis.mode!='structural':raise ValueError('Route Explorer uses its own optional LP check, not the FBA visualization mode')
        if self.subject.kind=='recognized_pathway' and self.subject.pathway_id not in TEMPLATES:raise ValueError('Unknown pathway template')
        if self.subject.kind=='target_neighborhood' and not self.subject.target_id:raise ValueError('Choose a target')
        if self.subject.kind=='genes' and not self.subject.gene_ids:raise ValueError('Choose at least one model gene')
        if self.subject.branch_only and self.subject.pathway_id not in {'ETHANOL_FERMENTATION_EMP_PDC_ADH','HOMOLACTIC_L_FERMENTATION_EMP_LDH'}:raise ValueError('Terminal branch only applies to fermentation templates')
        if self.fba:
            if self.fba.objective.source=='reaction' and not self.fba.objective.reaction_id:raise ValueError('Choose an objective reaction')
            if self.fba.medium.source=='model_encoded' and self.fba.medium.uptake_bounds:raise ValueError('Custom uptake bounds require custom medium')
            if any(v<0 for v in self.fba.medium.uptake_bounds.values()):raise ValueError('Uptake capacities must be nonnegative')
            if any(lb>ub for lb,ub in self.fba.bounds_overrides.values()):raise ValueError('Lower bound exceeds upper bound')
        return self

def relative(value,allow_parent=False):
    if '\\' in value or ':' in value or value.startswith(('/', '~')):raise ValueError('Use a portable relative path')
    p=PurePosixPath(value)
    if not allow_parent and '..' in p.parts:raise ValueError('Path traversal is not allowed')
    if not p.parts:raise ValueError('Empty path')
    return p

class UniqueSafeLoader(yaml.SafeLoader):pass

def mapping(loader,node,deep=False):
    pairs=loader.construct_pairs(node,deep=deep);out={}
    for k,v in pairs:
        if k in out:raise ValueError('Duplicate YAML key: '+str(k))
        out[k]=v
    return out
UniqueSafeLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,mapping)

def parse_yaml(data):
    if isinstance(data,bytes):data=data.decode('utf-8')
    if len(data.encode())>MAX_YAML:raise ValueError('YAML exceeds 2 MiB')
    # Aliases/anchors are not needed for generated problems; reject expansion attacks.
    for token in yaml.scan(data):
        if isinstance(token,(yaml.tokens.AliasToken,yaml.tokens.AnchorToken)):raise ValueError('YAML anchors and aliases are not supported')
    d=yaml.load(data,Loader=UniqueSafeLoader)
    if not isinstance(d,dict):raise ValueError('A YAML problem must be a mapping')
    for k in ('schema_version','problem_type','problem_id','model'): 
        if k not in d:raise ValueError('Missing required field: '+k)
    return Problem.model_validate(d)

def dump_yaml(p):return yaml.safe_dump(p.model_dump(mode='json',exclude_none=True),sort_keys=False,allow_unicode=True)

def config_hash(p):
    d=p.model_dump(mode='json');d.pop('metadata',None);d.pop('description',None)
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def resolve_model(p, base_dir, allowed_root=None, supplied=None):
    candidates=[];base_dir=Path(base_dir).resolve()
    allowed_root=Path(allowed_root or base_dir).resolve()
    if supplied:candidates.append(Path(supplied).resolve())
    if p.model.path:
        path=(base_dir/p.model.path).resolve()
        if not path.is_relative_to(allowed_root):raise ValueError('Model path leaves the declared model root')
        candidates.append(path)
    if p.model.registry_id:
        if p.model.registry_id not in REGISTRY:raise ValueError('Unknown bundled model registry ID')
        candidates.append(REGISTRY[p.model.registry_id])
    if p.model.sha256:candidates.extend(x for x in REGISTRY.values() if sha256(x)==p.model.sha256)
    mismatched=False
    for x in candidates:
        if not x.is_file():continue
        if p.model.sha256 and sha256(x)!=p.model.sha256:raise ValueError('Model checksum mismatch; upload the exact referenced GEM')
        return x
    if mismatched:raise ValueError('Model checksum mismatch; upload the exact referenced GEM')
    raise FileNotFoundError('Referenced model is missing; upload its SBML or COBRA JSON file')

def export_bundle(p,model_path,output):
    d=p.model_dump(mode='json',exclude_none=True);d['model']={'path':'models/'+Path(model_path).name,'sha256':sha256(model_path)}
    portable=Problem.model_validate(d)
    data=dump_yaml(portable).encode();model=Path(model_path).read_bytes()
    manifest={'schema_version':'1.0','files':{'problem.yaml':hashlib.sha256(data).hexdigest(),d['model']['path']:hashlib.sha256(model).hexdigest()}}
    with zipfile.ZipFile(output,'w',zipfile.ZIP_DEFLATED) as z:
        z.writestr('problem.yaml',data);z.writestr(d['model']['path'],model);z.writestr('manifest.json',json.dumps(manifest,indent=2))
    return Path(output)

def import_bundle(path,destination):
    dest=Path(destination).resolve();dest.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(path) as z:
        infos=z.infolist()
        if len(infos)>500 or sum(x.file_size for x in infos)>MAX_BUNDLE:raise ValueError('Bundle exceeds size or file-count limit')
        names=set()
        for item in infos:
            relative(item.filename)
            if item.filename in names:raise ValueError('Duplicate bundle member')
            names.add(item.filename)
            if stat.S_ISLNK(item.external_attr>>16):raise ValueError('Bundle symlinks are not allowed')
            target=(dest/item.filename).resolve()
            if not target.is_relative_to(dest):raise ValueError('Unsafe bundle member')
        if 'manifest.json' not in names:raise ValueError('Portable bundle requires manifest.json')
        manifest=json.loads(z.read('manifest.json'))
        if manifest.get('schema_version')!='1.0':raise ValueError('Unsupported bundle schema')
        files=manifest.get('files',{})
        if not isinstance(files,dict) or set(files)!={x.filename for x in infos if not x.is_dir() and x.filename!='manifest.json'}:raise ValueError('Bundle manifest does not cover every file')
        for name,digest in files.items():
            if hashlib.sha256(z.read(name)).hexdigest()!=digest:raise ValueError('Bundle checksum mismatch: '+name)
        problems=[]
        for name in files:
            if name.endswith(('.yaml','.yml')):problems.append((name,parse_yaml(z.read(name))))
        if not problems:raise ValueError('Bundle contains no problems')
        for name,p in problems:
            if p.model.path:
                target=(dest/Path(name).parent/p.model.path).resolve()
                if not target.is_relative_to(dest):raise ValueError('Model path escapes bundle')
        # Write only after all path, hash and YAML checks have passed.
        for item in infos:
            target=dest/item.filename
            if item.is_dir():target.mkdir(parents=True,exist_ok=True)
            else:target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(item))
    return [(dest/name,p) for name,p in problems]
