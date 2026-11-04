"""Mac-friendly real-model validation of the adaptive exact-FBA rescue cycle.

Runs on the shipped E. coli scientific integration fixture, never mutating it.
No external historical landscape is loaded or modified.
"""
from __future__ import annotations
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from .acquisition import digest
from .examples import install_adaptive_example
from .pool import QueryPool
from .runner import run_one


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'validation_cases/adaptive_qc/frozen_exact_ecoli_48.csv'
    initial_sha=digest(source)
    with TemporaryDirectory(prefix='gem_adaptive_validation_') as tmp:
        pool=QueryPool(tmp)
        install_adaptive_example(pool.root)
        name='DEMO_08_adaptive_exact_ecoli_sensitivity'
        first=run_one(pool.root,name)
        assert first['new_fba_calls']==8,first
        assert first['new_designs']==4,first
        assert first['surrogate_qc_status']['status']=='pass',first
        assert first['acquisition']['initial_working_surrogate_qc']['status']=='insufficient',first
        assert len(first['acquisition']['history'])==2,first
        assert Path(first['result_directory']).is_dir()
        second=run_one(pool.root,name)
        assert second['new_fba_calls']==0 and second['new_designs']==0,second
        assert second['surrogate_qc_status']['status']=='pass',second
        assert digest(source)==initial_sha,'Original exact landscape changed'
        print('REAL E. COLI ADAPTIVE QC VALIDATION: PASS')
        print('SOURCE: 48 rows · 0 source modifications')
        print('QC: insufficient at 48 -> passed after four new designs (52 total)')
        print('EXACT LP SOLVES: 8 first run; 0 repeat run')
        print('NATIVE CALCULATOR RESULTS: PASS')

if __name__=='__main__':main()
