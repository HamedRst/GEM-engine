import sys
import tempfile
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from install import perform_install

class ReplaceTest(unittest.TestCase):
    def test_single_target_atomic_replace(self):
        with tempfile.TemporaryDirectory() as d:
            ar=Path(d)/'Archive'
            (ar/'Results'/'Landscape_Generation').mkdir(parents=True)
            root=ar/'Figures_and_Tables'/'Supplementary'/'SI-Landscape'
            (root/'outputs').mkdir(parents=True)
            (root/'outputs'/'SI-Landscape.pdf').write_text('STALE')
            perform_install(ar)
            self.assertFalse((root/'outputs'/'SI-Landscape.pdf').exists())
            self.assertTrue((root/'scripts'/'si_landscape.py').exists())
            self.assertEqual(len(list(root.parent.glob('*SI-Landscape*'))),1)
            (root/'inputs'/'SOURCE_MAP.csv').write_text('seed,species,run_relpath\n42,yeast,Results/native\n')
            perform_install(ar)
            self.assertFalse((root/'inputs'/'SOURCE_MAP.csv').exists(), 'Seed42-only V5 map must be rebuilt for V3 15-run coverage')
            self.assertEqual(len(list(root.parent.glob('*SI-Landscape*'))),1)

if __name__=='__main__':unittest.main()
