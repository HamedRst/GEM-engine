"""Headless integration smoke; this does not substitute for browser GUI testing."""
from pathlib import Path
from tempfile import TemporaryDirectory
import yaml
from .pool import QueryPool
from .runner import run_one
from gem_exploration_gui.inspectors import inspect_landscape
from gem_exploration_gui.calculator import CalculatorRequest,compile_calculator,build_registry


def main():
    root=Path(__file__).resolve().parents[1]
    with TemporaryDirectory() as td:
        pool=QueryPool(td)
        land=root/'demo/demo_landscape.csv'
        info,_=inspect_landscape(str(land))
        comp=compile_calculator(CalculatorRequest('demo_verify',info.path,None,'product','describe','all','none',n_estimators=40,cv_folds=3),build_registry(info,None))
        pool.save_calculator(comp.public_yaml,comp.internal_problem)
        result=run_one(td,'demo_verify')
        assert Path(result['result_directory']).exists()
        print('ORIGINAL CALCULATOR THROUGH SHARED POOL: PASS')
        native=yaml.safe_load((root/'examples/native_network_queries/SI_N01_G6P_branch.yaml').read_text())
        pool.save_network(native)
        net=run_one(td,native['problem_id'])
        assert net['result']['solver_calls']==0
        print('ORIGINAL ATLAS STRUCTURAL RENDER THROUGH SHARED POOL: PASS')
    print('Shared YAML pool integration checks: PASS')
if __name__=='__main__':main()
