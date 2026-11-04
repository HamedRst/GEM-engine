"""Portable sample query pool, created locally so paths point to the user's installation."""
from pathlib import Path
import yaml
from .pool import QueryPool
from gem_exploration_gui.inspectors import inspect_landscape
from gem_exploration_gui.calculator import build_registry,CalculatorRequest,compile_calculator


def install_examples(workspace,only_if_empty=True):
    pool=QueryPool(workspace)
    if only_if_empty and pool.names():return []
    root=Path(__file__).resolve().parents[1]
    landscape=root/'demo/demo_landscape.csv'
    info,_=inspect_landscape(str(landscape))
    reg=build_registry(info,None)
    cases=[
        ('DEMO_01_describe_growth','growth','describe','all','none'),
        ('DEMO_02_sensitivity_product','product','relate(influence)','all','feasible=true'),
        ('DEMO_03_interactions','product','relate(interaction)','glucose, oxygen','feasible=true'),
        ('DEMO_04_robustness','product','robust','oxygen','feasible=true'),
        ('DEMO_05_response','product','vary','glucose','feasible=true'),
        ('DEMO_06_tradeoff','growth, product','pareto(tradeoff)','all','feasible=true'),
        ('DEMO_07_pareto','growth, product','pareto','all','feasible=true'),
    ]
    paths=[]
    for name,observe,explore,variables,conditions in cases:
        if name in pool.names():continue
        comp=compile_calculator(CalculatorRequest(name,str(landscape),None,observe,explore,variables,conditions,n_estimators=70,cv_folds=3),reg)
        paths.append(pool.save_calculator(comp.public_yaml,comp.internal_problem))
    for name in ['NETWORK_G6P_BRANCH','NETWORK_PGK_KNOCKOUT','NETWORK_PPP_PATHWAY']:
        if name in pool.names():continue
        source=root/'examples/native_network_queries'/f'{name}.yaml'
        paths.append(pool.save_network(yaml.safe_load(source.read_text(encoding='utf-8'))))
    return paths

if __name__=='__main__':
    import sys
    print('\n'.join(map(str,install_examples(sys.argv[1] if len(sys.argv)>1 else str(Path.home()/'Desktop/GEM_Exploration_Workspace')))))

