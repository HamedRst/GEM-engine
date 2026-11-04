"""Scientific correctness, native GHI mapping, and layout tests."""
import csv, tempfile, unittest, sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import generate_figure3_jk as jm


def fixture(gains=(5.0,10.0,20.0,35.0)):
    return [{'prior_points':str(size),'method':'QC-only selector','delta_hv_mean_pct':str(gain)}
            for size,gain in zip(jm.SIZES,gains)]

class FourPanelsTest(unittest.TestCase):
    def test_pareto_endpoints_and_exact_hv_match(self):
        rows=jm.build_rows(fixture())
        self.assertEqual(len(rows),80)
        self.assertEqual({r['panel'] for r in rows},set('JKLM'))
        self.assertEqual({r['method'] for r in rows},{'Landscape','QC-only selector'})
        baseline=jm.base_frontier()
        self.assertAlmostEqual(jm.hypervolume_2d(baseline),4/9,places=12)
        for size,gain in zip(jm.SIZES,(5,10,20,35)):
            qc=sorted([r for r in rows if r['prior_points']==size and r['method']=='QC-only selector'],key=lambda r:r['point_index'])
            vals=[(r['x'],r['y']) for r in qc]
            self.assertEqual(vals[0],(0.0,1.0))
            self.assertEqual(vals[-1],(1.0,0.0))
            self.assertAlmostEqual(100*(jm.hypervolume_2d(vals)/(4/9)-1),gain,places=9)
            self.assertGreater(vals[4][1],baseline[4][1])
            self.assertGreater(vals[5][1],baseline[5][1])
            self.assertEqual(len(qc),10)
    def test_no_fabricated_upward_gain_for_negative(self):
        rows=jm.build_rows(fixture((-10,0,10,20)))
        qc=[r for r in rows if r['prior_points']==5000 and r['method']=='QC-only selector']
        baseline=jm.base_frontier()
        self.assertLess(qc[4]['y'],baseline[3][1])
        self.assertAlmostEqual(qc[0]['matched_discrete_delta_hv_pct'],-10,places=8)
    def test_missing_improvement_rejected(self):
        with self.assertRaisesRegex(jm.FigureError,'Missing QC-only selector'):
            jm.build_rows(fixture()[:-1])
    def test_excessive_gain_rejected(self):
        with self.assertRaisesRegex(jm.FigureError,'Outside representable'):
            jm.build_rows(fixture((5,10,20,101)))
    def test_draw_four_panel_pdf(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)
            jm.draw(jm.build_rows(fixture()),out)
            self.assertTrue((out/(jm.NAME+'.pdf')).stat().st_size>1000)
            self.assertTrue((out/(jm.NAME+'.svg')).stat().st_size>1000)
            self.assertTrue((out/(jm.NAME+'.png')).stat().st_size>1000)
            # The output is a four-across, landscape canvas.
            from PIL import Image
            with Image.open(out/(jm.NAME+'.png')) as image:
                self.assertGreater(image.width/image.height,2.5)
    def test_locate_fig3_source(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)/'Archive'; main=root/'Figures_and_Tables'/'Main'/'Figure_03_ThreeSeed_Optimization_Aggregate'/'outputs'/'FIGURE3_ALL_NINE_PANELS_V4'/'run_20261019'/'plot_data'
            main.mkdir(parents=True)
            p=main/'GHI_POOLED_TOP_PCT_MAIN_297.csv'
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['prior_points','method','delta_hv_mean_pct']);w.writeheader();w.writerows(fixture())
            self.assertEqual(jm.locate_ghi_csv(root),p)
            data=jm.build_rows(list(jm.read_csv(p)))
            self.assertEqual(len(data),80)

if __name__=='__main__':unittest.main()
