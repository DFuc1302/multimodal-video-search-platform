import json
import tempfile
import unittest
from pathlib import Path
from colab_summary_ms import covered_until, choose_window, save


def segment(start, end):
    return dict(moc_thoi_gian=dict(bat_dau=start, ket_thuc=end), mo_ta_hinh_anh='Cảnh thử')


class ResumeBudgetTests(unittest.TestCase):
    def test_legacy_partial_resumes_exact_timestamp(self):
        old = [segment(i*30, (i+1)*30) for i in range(81)]
        offset = covered_until(old, 2450)
        self.assertEqual(offset, 2430)
        updated = old + [segment(offset, min(offset+120, 2450))]
        self.assertEqual(covered_until(updated, 2450), 2450)
        self.assertEqual(updated[:81], old)

    def test_resume_mixed_windows(self):
        self.assertEqual(covered_until([segment(0, 30), segment(30, 150), segment(150, 450)], 500), 450)

    def test_reject_gap_overlap_empty_or_out_of_bounds(self):
        for bad in [[segment(0, 30), segment(31, 60)],
                    [segment(0, 30), segment(20, 60)], [segment(0, 1001)],
                    [dict(moc_thoi_gian=dict(bat_dau=0, ket_thuc=30), mo_ta_hinh_anh='')]]:
            with self.assertRaises(ValueError):
                covered_until(bad, 1000)

    def test_budget_adjustment_and_maximum(self):
        self.assertEqual(choose_window(120, 300, 537900, 10800, None), 120)
        self.assertEqual(choose_window(120, 300, 537900, 10800, 1.98), 150)
        self.assertEqual(choose_window(120, 300, 537900, 10, 1.98), 300)

    def test_save_roundtrip_preserves_vietnamese(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'checkpoints'/'M03_V028.json'
            payload = dict(segments=[segment(0, 30)])
            save(path, payload)
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')), payload)
            self.assertFalse(path.with_suffix('.tmp').exists())

    def test_generated_notebook_and_replacement_cell(self):
        root = Path(__file__).resolve().parents[1]
        notebook = json.loads((root/'output/colab_summary/AIC_Summary_M_S_Colab.ipynb').read_text(encoding='utf-8'))
        for cell in notebook['cells']:
            if cell['cell_type'] == 'code':
                compile(''.join(cell['source']), '<cell>', 'exec')
        replacement = (root/'output/colab_summary/Resume_M_3h.py').read_text(encoding='utf-8')
        compile(replacement, '<replacement>', 'exec')
        self.assertIn("FAMILIES = ['M']", replacement)
        self.assertIn('subprocess.Popen', replacement)


if __name__ == '__main__':
    unittest.main()
