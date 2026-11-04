"""Unit/integration tests: simulate PyArrow for synthetic native chunks only.

Tested scientific code and plotting without connecting to any original GEM data.
A real Mac run will use genuine PyArrow 23.0.1.
"""
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'si_landscape.py'
sys.path.insert(0,str(SCRIPT.parent))
import si_landscape as sl

class DummyTable:
    def __init__(self,v):self.v=v
    def to_pydict(self):return self.v

def mock_read_table(path,columns=None):
    with open(path) as f:d=json.load(f)
    return DummyTable({c:d[c] for c in columns})

class NativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pq=types.ModuleType('pyarrow.parquet');cls.pq.read_table=mock_read_table
        cls.pa=types.ModuleType('pyarrow');cls.pa.parquet=cls.pq
        cls.saved={x:sys.modules.get(x) for x in ('pyarrow','pyarrow.parquet')}
        sys.modules['pyarrow']=cls.pa;sys.modules['pyarrow.parquet']=cls.pq
    @classmethod
    def tearDownClass(cls):
        for k,v in cls.saved.items():
            if v is None:sys.modules.pop(k,None)
            else:sys.modules[k]=v
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.archive=Path(self.tmp.name)/'archive'
        self.base=self.archive/'Results'/'Landscape_Generation'/'_CANDIDATE_UNVERIFIED'/'GEM_Learning_Engine_v1.0.4'
        self.root=self.base/'outputs_multiseed_50k'/'seed43'/'yeast_50k'
        d=self.root/'chunks'/'results';d.mkdir(parents=True)
        for chunk in range(2):
            start=chunk*25000;ids=list(range(start,start+25000))
            v={"sample_id":ids,
               'biomass_flux':[0.02+float(i)/50000 for i in ids],
               'product_flux':[1.2-float(i)/50000 for i in ids],
               'feasible':[True]*len(ids),
               'runtime_seconds':[.5]*len(ids),
               'total_runs':[50000]*len(ids)}
            (d/f'results_chunk_{chunk:06}.parquet').write_text(json.dumps(v))
        perf=self.base/'performance_multiseed_50k'/'yeast_50k_seed43'/'performance_macos_time.txt'
        perf.parent.mkdir(parents=True)
        perf.write_text('600.000 real 200.0 user 3.0 sys\n')
        self.mapping={'seed':'43','species':'yeast','run_relpath':str(self.root.relative_to(self.archive)),
            'performance_relpath':str(perf.relative_to(self.archive)), 'start_time_utc':''}
    def test_discovery(self):
        p=sl.discover(self.archive,Path(self.tmp.name)/'figure')
        rec=list(sl.read_csv(p));self.assertEqual(len(rec),15)
        active=[x for x in rec if x['run_relpath']]
        self.assertEqual(len(active),1)
        self.assertEqual(active[0]['result_chunk_count'],'2')
    def test_source_and_metrics(self):
        data=sl.load_run(self.mapping,self.archive)
        self.assertEqual(len(data['ids']),50000)
        self.assertEqual(len(data['files']),2)
        rows,audit=sl.compute_run(data,self.archive)
        self.assertEqual(len(rows),len(sl.SIZES))
        self.assertAlmostEqual(rows[-1]['knowledge_percent'],100.)
        self.assertTrue(all(a['knowledge_percent']<=b['knowledge_percent']+1e-7 for a,b in zip(rows,rows[1:])))
        self.assertAlmostEqual(rows[-1]['cumulative_evaluation_minutes'],50000*.5/60)
        self.assertAlmostEqual(rows[-1]['generation_wall_minutes'],10.)
        self.assertEqual(rows[0]['generation_wall_minutes'],'')
        self.assertEqual(audit['status'],'CONTENT_PASS_50000')
        self.assertEqual(audit['provenance'],'PROVISIONAL_CANDIDATE_DIRECTORY')
        rec=sl.aggregate(rows)
        self.assertEqual(len(rec),len(sl.SIZES))
        dest=Path(self.tmp.name)/'plot';dest.mkdir()
        self.assertEqual(sl.figure(rec,dest,True,'worker'),'worker')
        self.assertTrue((dest/'SI-Landscape.pdf').is_file())
        self.assertTrue((dest/'SI-Landscape.svg').is_file())
    def test_duplicate_ids_blocked(self):
        p=self.root/'chunks'/'results'/'results_chunk_000001.parquet'
        d=json.loads(p.read_text());d['sample_id'][0]=0;p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'Duplicate native sample_id'):
            sl.load_run(self.mapping,self.archive)
    def test_missing_ids_blocked(self):
        p=self.root/'chunks'/'results'/'results_chunk_000001.parquet'
        d=json.loads(p.read_text());d['sample_id'][-1]=60000;p.write_text(json.dumps(d))
        with self.assertRaisesRegex(ValueError,'not a contiguous'):
            sl.load_run(self.mapping,self.archive)
    def test_start_parser(self):
        p=Path(self.tmp.name)/'run_metadata.txt'
        p.write_text('run_start_utc: 2026-09-29T10:11:12Z\n')
        self.assertEqual(sl.parse_verified_start(p),'2026-09-29T10:11:12Z')
        p.write_text('start_time: 2026-09-29T10:11:12\n')
        self.assertEqual(sl.parse_verified_start(p),'')
    def test_hypervolume(self):
        self.assertAlmostEqual(sl.exact_hv2d([(1,.5),(.5,1)]),.75)
        self.assertAlmostEqual(sl.exact_hv2d([(0,1),(1,0)]),0)
        self.assertEqual(sl.parse_sample_id('sample_000023'),23)

if __name__=='__main__':unittest.main()
