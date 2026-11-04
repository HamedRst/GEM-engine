import json
import unittest
from pathlib import Path
class UploadedInventoryTest(unittest.TestCase):
    def test_native_layout_from_uploaded_source_inventory(self):
        fixture=Path(__file__).resolve().parent/'fixtures'/'uploaded_inventory_summary.json'
        report=json.loads(fixture.read_text())
        self.assertEqual(report['total_candidate_rows'],5884)
        self.assertEqual(len(report['native_chunks_per_case']),30)
        self.assertEqual({a['n_files'] for a in report['native_chunks_per_case']},{196})
        self.assertEqual(report['whole_dataset_csv_count'],4)
if __name__=='__main__':unittest.main()
