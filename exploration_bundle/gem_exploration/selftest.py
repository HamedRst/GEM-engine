from __future__ import annotations
from pathlib import Path
import tempfile, yaml
import numpy as np
import pandas as pd
from .config import load_problem
from .engine import run_problem

OP_SPECS={
'describe': {'targets':['product']},
'query': {'targets':[],'operator_params':{'filters':[{'column':'product','op':'>=','value':0.5}],'max_saved_points':50}},
'association': {'targets':['product']},
'correlation_matrix': {'targets':['growth','product'],'operator_params':{'features':['x1','x2','growth','product']}},
'global_sensitivity': {'targets':['product']},
'local_sensitivity': {'targets':['product']},
'sensitivity_map': {'targets':['product'],'operator_params':{'anchor_count':40,'step_fraction':0.02}},
'interaction': {'targets':['product'],'operator_params':{'anchor_count':40,'max_pairs':6}},
'fragility': {'targets':['product'],'operator_params':{'neighbors':12}},
'robustness': {'targets':['product'],'operator_params':{'neighbors':12}},
'threshold_region': {'targets':['product'],'operator_params':{'threshold':0.8,'direction':'>='}},
'threshold_boundary': {'targets':['product'],'operator_params':{'threshold':0.8,'n_points':50}},
'feasibility': {'targets':[],'operator_params':{'column':'feasible'}},
'regime_discovery': {'targets':['growth','product'],'operator_params':{'max_k':4}},
'dimensionality': {'targets':[],'operator_params':{'features':['x1','x2','x3','growth','product'],'variance_threshold':0.9}},
'tradeoff': {'targets':[{'name':'growth','value':'growth','direction':'maximize'},{'name':'product','value':'product','direction':'maximize'}]},
'pareto': {'targets':[{'name':'growth','value':'growth','direction':'maximize'},{'name':'product','value':'product','direction':'maximize'}]},
'pareto_saturation': {'targets':[{'name':'growth','value':'growth','direction':'maximize'},{'name':'product','value':'product','direction':'maximize'}],'operator_params':{'sample_sizes':[100,250,500,900],'repeats':2}},
'compare_regions': {'targets':['product'],'operator_params':{'region_a':[{'column':'x1','op':'>=','value':0.5}],'region_b':[{'column':'x1','op':'<','value':0.5}]}},
'coverage': {'targets':[]},
'outlier': {'targets':['product']},
'point_query': {'targets':['product'],'operator_params':{'point':{'x1':0.5,'x2':0.5,'x3':0.5},'neighbors':5}},
'response_curve': {'targets':['product'],'operator_params':{'variable':'x1','grid_points':25}},
'marginal_response': {'targets':['product'],'operator_params':{'variable':'x1','grid_points':15,'background_rows':80}},
'surrogate_uncertainty': {'targets':['product'],'operator_params':{'evaluation_rows':200,'top_points':20}},
'controller_scan': {'targets':['product'],'operator_params':{'anchor_count':30,'levels':7}},
'threshold_switch_scan': {'targets':['product'],'operator_params':{'threshold':0.7,'anchor_count':30,'levels':11}},
'response_surface': {'targets':['product'],'operator_params':{'variables':['x1','x2'],'grid_points':12}},
'sampling_saturation': {'targets':[],'operator_params':{'sample_sizes':[50,100,250,500],'repeats':2,'evaluation_points':200}},
}

def synthetic(n=900,seed=7):
    rng=np.random.default_rng(seed); X=rng.random((n,3));
    growth=0.2+0.6*X[:,0]-0.15*(X[:,1]-.4)**2+0.05*np.sin(6*X[:,2])
    product=0.1+0.8*X[:,1]+0.25*X[:,0]*X[:,2]+0.08*np.sin(5*X[:,0])
    feasible=(X[:,0]+X[:,1]>.22)
    return pd.DataFrame({'sample_id':np.arange(n),'x1':X[:,0],'x2':X[:,1],'x3':X[:,2],'growth':growth,'product':product,'feasible':feasible})

def run_self_test():
    with tempfile.TemporaryDirectory(prefix='gem_explore_selftest_') as td:
        root=Path(td); data=root/'synthetic.csv'; synthetic().to_csv(data,index=False)
        passes=[]
        for op,info in OP_SPECS.items():
            targets=info.get('targets',[]); norm=[]
            for t in targets:
                norm.append(t if isinstance(t,dict) else {'name':t,'value':t,'direction':'maximize'})
            spec={'spec_version':'1.0','name':f'selftest_{op}','landscape':{'path':str(data)},'variables':{'mode':'list','values':['x1','x2','x3']},'targets':norm,'scope':{},'backend':{'mode':'auto'},'surrogate':{'model':'extra_trees','n_estimators':60,'cv_folds':3,'min_r2':0.25,'min_auc':0.6,'max_train_rows':900,'n_jobs':1},'operator':{'name':op,'parameters':info.get('operator_params',{})},'outputs':{'save_figures':False}}
            p=root/f'{op}.yaml'; p.write_text(yaml.safe_dump(spec,sort_keys=False),encoding='utf-8')
            out=run_problem(load_problem(p),output_override=str(root/f'out_{op}'),overwrite=True,seed=123)
            if not (out/'summary.json').exists(): raise AssertionError(f'{op} did not create summary')
            passes.append(op)
        print(f'SELF-TEST PASS: {len(passes)}/{len(OP_SPECS)} operators')
        print(', '.join(passes))
