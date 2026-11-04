#!/usr/bin/env python3
from __future__ import annotations
import hashlib,py_compile,tempfile
from pathlib import Path
import numpy as np,pandas as pd,yaml
from scipy.stats import qmc
HERE=Path(__file__).resolve().parent
PRIOR_ROWS=10000; WORKER='MACSTUDIO'; EXPECTED_ALGO_SHA='df02dd309c8712525c7e9c7f915ce8a43d4b34a4e7cfeeb0fb271e855e9665b7'
def ok(x): print('PASS ',x)
def test_compile():
 files=[]
 for p in HERE.rglob('*.py'):
  if any(x.startswith('worker_state') or x.startswith('results_') or x=='__pycache__' for x in p.parts): continue
  py_compile.compile(str(p),doraise=True);files.append(p)
 assert len(files)>=15;ok(f'py_compile ({len(files)} files)')
def test_manifest():
 m=pd.read_csv(HERE/'MANIFEST_ALL25.csv');assert len(m)==25 and m.case_key.nunique()==25 and m.job_id.nunique()==25 and set(m.worker.astype(str))=={WORKER};assert set(m.species)=={'yeast','pputida','cglutamicum','cho','ecoli_antiepex'};assert set(m.problem)=={'growth_product','product_carbon_oxygen','growth_robustness','product_cost','constrained_product'};ok(f'ALL25 single-worker manifest ({WORKER})')
def test_algorithm_identity():
 h=hashlib.sha256((HERE/'gsmobo_algorithms.py').read_bytes()).hexdigest();assert h==EXPECTED_ALGO_SHA,(h,EXPECTED_ALGO_SHA);txt=(HERE/'gsmobo_algorithms.py').read_text();expected=['BASE_GSMOBO_QPAREGO_NO50K','BASE_GSMOBO_QPAREGO_50K','DEV_GSMOBO50K_GHOST_LEGACY_VB','DEV_GSMOBO50K_SPARC_LEGACY_VB','DEV_GSMOBO50K_TRAIL_LEGACY_VB','DEV_GSMOBO50K_GHOST_TRUE_NATIVE_V111','DEV_GSMOBO50K_SPARC_TRUE_NATIVE_V111','DEV_GSMOBO50K_TRAIL_TRUE_NATIVE_V111','DEV_GSMOBO50K_GHOST_A','DEV_GSMOBO50K_SPARC_A','DEV_GSMOBO50K_TRAIL_A'];assert all(x in txt for x in expected);ok('canonical v0.3.0 gsMOBO algorithm SHA + 11-method panel')
def test_problem_semantics():
 from generic_problem import compile_frozen_archive,compile_live_exact
 feats=['x0','x1','x2','x3','x4'];sp={'problem':{'features':feats,'bounds':{f:[0.,1.] for f in feats},'carbon_feature':'x1','oxygen_feature':'x2'}};X=qmc.Sobol(5,scramble=True,seed=20260822).random_base2(12);df=pd.DataFrame(X,columns=feats);df['maximum_growth']=.1+.8*X[:,0];df['product_flux']=.1+.6*X[:,1];df['carbon_efficiency']=1-X[:,1];df['oxygen_efficiency']=1-X[:,2];df['stress_tolerance']=.3+.6*X[:,3];df['resource_efficiency']=1-X.mean(1);df['feasible']=True;df['status']='optimal';df['design_index']=np.arange(len(df))
 for fn in ['growth_product.gemproblem.yaml','product_carbon_oxygen.gemproblem.yaml','growth_robustness.gemproblem.yaml','product_cost.gemproblem.yaml','constrained_product.gemproblem.yaml']:
  spec=yaml.safe_load((HERE/'problems'/fn).read_text());full,meta=compile_frozen_archive(df,sp,spec);prepared={'problem':{**sp['problem'],'compiled_problem':meta}};prefix=compile_live_exact(df.iloc[:1024].copy(),df.iloc[:1024][feats].copy(),prepared,spec);assert len(prefix)==1024;assert prepared['problem']['compiled_problem']['constraints']==meta['constraints'];
 ok('all five problem definitions reuse frozen full-reference semantics on prefix')
def test_sobol_prefix():
 from multi_problem_bridge import _select_sobol_prefix
 with tempfile.TemporaryDirectory() as td:
  td=Path(td);src=td/'Source';dc=src/'inputs'/'runs'/'fake'/'design_cache';dc.mkdir(parents=True)
  A=qmc.Sobol(5,scramble=True,seed=42).random_base2(16)[:50000]
  np.save(dc/'sobol_test.npy',A)
  df=pd.DataFrame(A,columns=[f'x{i}' for i in range(5)]);df['design_index']=np.arange(len(df))
  sp={'problem':{'features':[f'x{i}' for i in range(5)],'bounds':{f:[0.,1.] for f in [f'x{i}' for i in range(5)]}}}
  prefix,man=_select_sobol_prefix(df,src,'fake',sp,PRIOR_ROWS)
  assert len(prefix)==PRIOR_ROWS
  assert man['landscape_prior_rows']==PRIOR_ROWS and man['source_archive_rows']==50000
  assert man['design_index_start']==0 and man['design_index_end']==PRIOR_ROWS-1
  assert man['sobol_prefix_max_abs_error']<1e-12
  one=df.copy();one['design_index']=one['design_index']+1
  prefix1,man1=_select_sobol_prefix(one,src,'fake',sp,PRIOR_ROWS)
  assert len(prefix1)==PRIOR_ROWS and man1['design_index_start']==1 and man1['sobol_prefix_max_abs_error']<1e-12
  bad=df.copy();bad.loc[100,'design_index']=bad.loc[100,'design_index']+5
  try:_select_sobol_prefix(bad,src,'fake',sp,PRIOR_ROWS)
  except RuntimeError as e:assert 'not a contiguous Sobol sequence' in str(e) or 'duplicate design_index' in str(e)
  else:raise AssertionError('noncontiguous design_index sequence was not rejected')
 ok(f'authoritative contiguous Sobol-prefix identity verification at actual prior size N={PRIOR_ROWS}, preserving working indexing convention')
def test_selected_panel_and_no_prior_contract():
 from selected_panel import SELECTED_METHODS,SELECTED_HEURISTICS,validate_selected_panel
 validate_selected_panel();assert len(SELECTED_METHODS)==5 and len(SELECTED_HEURISTICS)==3
 txt=(HERE/'aggregate_worker.py').read_text();assert 'LIVE_SELF_FRONT' in txt and 'final_self_hv' in txt and 'final_prior_union_hv' in txt and 'no_prior_self_front_correction_applied' in txt
 at=(HERE/'AUDIT_NO_PRIOR_SELF_FRONT.py').read_text();assert 'NO CURRENT-PRIOR OR 50K LANDSCAPE FRONTIER IS UNIONED' in at
 ok('selected 3 heuristic implementations + 2 references; NO-PRIOR self-front reporting contract')


def test_process_guard_ancestor_logic():
 from process_guard import parse_ps,ancestor_pids,find_hits
 # Normal ancestry case.
 rows=parse_ps("100 1 /bin/bash RUN_BOTH.command\n101 100 /usr/bin/caffeinate -i python run_worker.py --mode live\n102 101 python run_worker.py --mode live\n103 102 python process_guard.py\n777 1 python other/run_worker.py --worker X\n")
 anc=ancestor_pids(rows,103)
 assert {100,101,102,103}.issubset(anc) and 777 not in anc
 hits=find_hits(rows,103)
 assert [pid for pid,_ in hits]==[777],hits
 # Exact macOS failure reproduced from v0.4.1: caffeinate may be a sibling
 # rather than an ancestor, but its command line still mentions run_worker.py.
 # The wrapper must be ignored while a real unrelated Python runner is blocked.
 rows2=parse_ps("200 1 /bin/zsh ./RUN_BOTH.command\n201 200 /usr/bin/caffeinate -i /env/python -u run_worker.py --worker MACSTUDIO --mode preflight\n202 200 /env/python -u run_worker.py --worker MACSTUDIO --mode preflight\n203 202 /env/python process_guard.py --worker MACSTUDIO\n888 1 /env/python -u /other/run_worker.py --worker MACSTUDIO --mode live\n")
 hits2=find_hits(rows2,203)
 assert [pid for pid,_ in hits2]==[888],hits2
 ok('process guard ignores own/sibling caffeinate wrappers but still blocks unrelated Python GEM runners')

def test_corrected_no_prior_row_at():
 from aggregate_worker import row_at
 cols=['o1','o2'];rows=[]
 for i in range(1,129):
  rows.append({'call':i,'exact_feasible':True,'cache_hit':False,'archive_hv_after':0.9,'delta_hv_after':0.6,'exact_hv_gain_this_call':0.0,'exact_front_extension':False,'guided':False,'mode':'x','o1':0.10+0.003*i,'o2':0.12+0.002*i})
 h=pd.DataFrame(rows);r=row_at(h,128,0.3,2,cols,(0.0,0.0),True)
 assert r['evaluation_mode']=='LIVE_SELF_FRONT' and r['baseline_hv']==0.0 and r['no_prior_self_front_correction_applied'] is True
 assert r['final_primary_hv']==r['final_self_hv'] and r['final_primary_hv']<0.9 and abs(r['final_prior_union_hv']-0.9)<1e-12
 ok('NO-PRIOR aggregator ignores deliberately inflated landscape-union HV')

def test_immutable_science_hashes():
 import hashlib
 assert hashlib.sha256((HERE/'gsmobo_algorithms.py').read_bytes()).hexdigest()==EXPECTED_ALGO_SHA
 expected={
  'constrained_product.gemproblem.yaml':'9ea765c69f12a8fae1abc09167c672d9592e9e9a339a6d6dfbe85e5024145511',
  'growth_product.gemproblem.yaml':'e9eb90d783b3b80f201d3d677201e58989e03011b3140409333d2bd2ef6d2a17',
  'growth_robustness.gemproblem.yaml':'68edc9866b8ebad99cb6570fcd988385f6264fe81215f3df97b500340e11ed99',
  'product_carbon_oxygen.gemproblem.yaml':'930199697fc58951e23edc7a1896b0c730b63772b80738d0b2f5ddb9f1f5ac26',
  'product_cost.gemproblem.yaml':'97ce67994b54046781c96277b0a83212e1b26434bfaf5ec3b2a68dba356db1d2',
 }
 for fn,h in expected.items(): assert hashlib.sha256((HERE/'problems'/fn).read_bytes()).hexdigest()==h,(fn,h)
 ok('canonical algorithm + five problem YAML hashes frozen to working 5K package')

def main():
 print('='*96);print(f'GEM gsMOBO PRIOR-SIZE ABLATION SELFTEST | worker={WORKER} | prior_rows={PRIOR_ROWS} | ZERO FBA');print('='*96);test_compile();test_manifest();test_algorithm_identity();test_immutable_science_hashes();test_problem_semantics();test_sobol_prefix();test_process_guard_ancestor_logic();test_corrected_no_prior_row_at();test_selected_panel_and_no_prior_contract();print('='*96);print('PACKAGE SELFTEST PASS | ZERO biological FBA calls');print('='*96)
if __name__=='__main__':main()
