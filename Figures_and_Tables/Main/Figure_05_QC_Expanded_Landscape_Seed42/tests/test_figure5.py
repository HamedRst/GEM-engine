from __future__ import annotations
import csv, json, sys, tempfile, types, unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import generate_figure5 as f5


def write_csv(path, rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',newline='') as fp:
        writer=csv.DictWriter(fp,list(rows[0]))
        writer.writeheader();writer.writerows(rows)

class TestFigure5(unittest.TestCase):
    def test_exact_growth_product_selection_and_colors(self):
        self.assertTrue(f5.is_growth_product_case('yeast__Growth_Product'))
        self.assertTrue(f5.is_growth_product_case('cho__GP'))
        self.assertFalse(f5.is_growth_product_case('cho__P2'))
        self.assertFalse(f5.is_growth_product_case('pputida__carbon_efficiency'))
        self.assertEqual(f5.SELECTED_METHOD_TO_ALGO['gsMOBO + SPARC'],'DEV_GSMOBO50K_SPARC_A')

    def test_accept_physical_columns_only(self):
        self.assertEqual(f5.find_xy_columns(['biomass_flux','product_flux'])[:2],('biomass_flux','product_flux'))
        self.assertEqual(f5.find_xy_columns(['growth_rate','product_flux'])[:2],('growth_rate','product_flux'))
        for cols in (['objective_1','objective_2'],['y1','y2'],['f1','f2'],['normalized_growth','normalized_product']):
            self.assertIsNone(f5.find_xy_columns(cols))

    def test_saved_selector_no_missing_module(self):
        with tempfile.TemporaryDirectory() as tmp:
            arc=Path(tmp)
            dir=arc/'Figures_and_Tables'/'Main'/'Figure_03_ThreeSeed_Optimization_Aggregate'/'outputs'/'FIGURE3_ALL_NINE_PANELS_V4'/'run_20261019_134818_UTC'/'plot_data'
            species=['yeast','pputida','cglutamicum','cho','ecoli_antiepex']
            rows=[]
            for sp in species:
                for problem in ('Growth_Product','P2','P3','P4','P5'):
                    rows.append(dict(case=f'{sp}__{problem}',landscape_seed='42',prior_points='50000',
                                     selected_method='gsMOBO + SPARC',selected_hv='1.1',landscape_hv='1.0'))
            write_csv(dir/'QC_SELECTOR_PRIMARY_PREDICTIONS.csv',rows)
            found,path=f5.qc_selector_predictions(arc)
            self.assertEqual(len(found),25)
            self.assertEqual(path.name,'QC_SELECTOR_PRIMARY_PREDICTIONS.csv')
            campaign=arc/'Results'/'Optimizations'/'landscape_seed_42'/'prior_50000'/'GEM_ALL25_FINAL10_PRIOR50K_128'
            chosen=f5.choose_representative_cases(found,campaign,arc/'AUDIT.csv')
            self.assertEqual(len(chosen),5)
            for sp in species:self.assertEqual(chosen[sp]['case'],sp+'__Growth_Product')
            self.assertTrue((arc/'AUDIT.csv').is_file())

    def test_native_exact_csv_and_no_generic(self):
        with tempfile.TemporaryDirectory() as tmp:
            job=Path(tmp)/'job'; src=job/'scientific_results'
            pts=[{'biomass_flux':i/127,'product_flux':1-i/127+0.12} for i in range(128)]
            write_csv(src/'calls.csv',pts)
            write_csv(src/'generic_obj.csv',[{'objective_1':i/127,'objective_2':1-i/127} for i in range(128)])
            a=f5.extract_selected_points(job,(0,1),(0,1),Path(tmp)/'fileaudit.csv')
            self.assertEqual(a['n_points'],128)
            self.assertEqual(a['path'].name,'calls.csv')
            self.assertTrue((Path(tmp)/'fileaudit.csv').is_file())
            (src/'calls.csv').unlink()
            with self.assertRaisesRegex(f5.FigureError,'No verified physical growth/product'):
                f5.extract_selected_points(job,(0,1),(0,1))

    def test_pareto_nondominated_union(self):
        points=[(0,1),(0.4,0.6),(0.5,0.1),(0.6,0.65),(1,0)]
        front=f5.pareto_frontier(points)
        self.assertIn((0.6,0.65),front)
        self.assertNotIn((0.4,0.6),front)
        self.assertNotIn((0.5,0.1),front)

    def test_full_five_species_no_retraining(self):
        class MockTable:
            def __init__(self,n):self.n=n
            def to_pydict(self):
                from math import pi, sin
                n=self.n
                xx=[(i%101)/100 for i in range(n)]
                yy=[1.0-x-0.25*sin(pi*x)-0.05*((i//101)%13) for i,x in enumerate(xx)]
                return {'biomass_flux':xx,'product_flux':yy,'feasible':[True]*n}
        def read_mock(path,columns=None):
            return MockTable(50000)
        saved_pa=sys.modules.get('pyarrow');saved_pq=sys.modules.get('pyarrow.parquet')
        pa=types.ModuleType('pyarrow');pq=types.ModuleType('pyarrow.parquet');pq.read_table=read_mock
        pa.parquet=pq;sys.modules['pyarrow']=pa;sys.modules['pyarrow.parquet']=pq
        try:
            with tempfile.TemporaryDirectory() as tmp:
                arc=Path(tmp);(arc/'Figures_and_Tables'/'Main').mkdir(parents=True)
                pred=arc/'Figures_and_Tables'/'Main'/'Figure_03_ThreeSeed_Optimization_Aggregate'/'outputs'/'FIGURE3_ALL_NINE_PANELS_V4'/'run_001'/'plot_data'/'QC_SELECTOR_PRIMARY_PREDICTIONS.csv'
                rows=[]
                camp=arc/'Results'/'Optimizations'/'landscape_seed_42'/'prior_50000'/'GEM_ALL25_FINAL10_PRIOR50K_128'
                camp.mkdir(parents=True);(camp/'CAMPAIGN_COMPLETION.json').write_text('{}')
                for i,sp in enumerate(f5.SPECIES_ORDER):
                    for problem in ('Growth_Product','P2','P3','P4','P5'):
                        rows.append(dict(case=f'{sp}__{problem}',landscape_seed='42',prior_points='50000',
                             selected_method='gsMOBO + SPARC' if i%2==0 else 'NSGA-II + SPARC',selected_hv='1.25',landscape_hv='1.0'))
                    alg='DEV_GSMOBO50K_SPARC_A' if i%2==0 else 'DEV_NSGA50K_SPARC_LEGACY_OPERATOR'
                    job=camp/'jobs'/f'{sp}__Growth_Product'/alg/'scientific_results'/'calls.csv'
                    import math
                    pts=[dict(biomass_flux=j/127,product_flux=1-j/127+0.22*math.sin(math.pi*j/127)) for j in range(128)]
                    write_csv(job,pts)
                    land=arc/'Results'/'Landscape_Generation'/'orig'/'outputs_multiseed_50k'/'seed42'/f'{sp}_50k'/'chunks'/'results'
                    land.mkdir(parents=True)
                    (land/'results_chunk_000000.parquet').write_text('mock')
                write_csv(pred,rows)
                dest=arc/'Figures_and_Tables'/'Main'/'Figure_05_QC_Expanded_Landscape_Seed42'
                out=f5.compute(arc,dest)
                self.assertTrue((out/f'{f5.FIGURE_NAME}.pdf').is_file())
                self.assertTrue((out/f'{f5.FIGURE_NAME}.svg').is_file())
                m=json.loads((out/'plot_data'/'FIGURE5_METADATA.json').read_text())
                self.assertTrue(m['no_synthetic_points'])
                self.assertEqual(len(m['panels']),5)
                for panel in m['panels']:
                    self.assertGreater(panel['selected_frontier_new_points'],0)
        finally:
            if saved_pa is None:sys.modules.pop('pyarrow',None)
            else:sys.modules['pyarrow']=saved_pa
            if saved_pq is None:sys.modules.pop('pyarrow.parquet',None)
            else:sys.modules['pyarrow.parquet']=saved_pq

if __name__=='__main__':unittest.main()
