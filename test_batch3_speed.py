import ast
from contextlib import nullcontext
import hashlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from typing import Optional

from batch3_speed import environment
from colab_batch3 import copy_with_progress, recipe_matches
from colab_batch3_progress_compat import PRE_SPEED_CODE


class SpeedTests(unittest.TestCase):
    def test_copy_hashes_same_bytes_in_single_pass(self):
        data = b'video bytes' * 100
        output = io.BytesIO()
        checksum = copy_with_progress(io.BytesIO(data), output, len(data), 'test')
        self.assertEqual(output.getvalue(), data)
        self.assertEqual(checksum, hashlib.sha256(data).hexdigest())

    def test_completed_full_resolution_preserved_only_with_same_core_recipe(self):
        old = {'version': 5, 'interval': 1, 'clip_model': 'large', 'code': PRE_SPEED_CODE}
        new = {**old, 'version': 6, 'speed_profile': 'a100-fast', 'code': {'new': 'code'}}
        self.assertTrue(recipe_matches(old, new))
        self.assertFalse(recipe_matches(old, {**new, 'interval': 2}))
        self.assertFalse(recipe_matches(old, {**new, 'clip_model': 'base'}))
        self.assertFalse(recipe_matches({**old, 'code': {'unknown': 'code'}}, new))

    def test_fast_and_legacy_environments_are_explicit(self):
        args = SimpleNamespace(ocr_batch_size=32, speed_profile='a100-fast')
        self.assertEqual(environment(args)['OCR_MAX_SIDE'], '1280')
        self.assertEqual(environment(args)['OCR_DETECT_BATCH_SIZE'], '8')
        args.speed_profile = 'legacy'
        self.assertEqual(environment(args)['OCR_MAX_SIDE'], '0')
        self.assertEqual(environment(args)['SKIP_DETECTION_BACKFILL'], '0')

    def test_detector_microbatches_preserve_order_and_recognition_batch(self):
        self.exercise_ocr(False)

    def test_detector_failure_splits_without_losing_frames(self):
        self.exercise_ocr(True)

    def exercise_ocr(self, fail_large):
        tree = ast.parse(Path(__file__).with_name('ocr_index.py').read_text(encoding='utf-8'))
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_easyocr_read_group')
        calls = []
        class OCR:
            def readtext_batched(self, images, **kwargs):
                calls.append((len(images), kwargs['batch_size']))
                if fail_large and len(images) > 2:
                    raise RuntimeError('simulated detector OOM')
                return [[x] for x in images]
            def readtext(self, image, **kwargs):
                calls.append((1, kwargs['batch_size']))
                return [image]
        ns = {'Optional': Optional, '_ocr_lock': nullcontext(), 'OCR_BATCH_SIZE': 32,
              'OCR_DETECT_BATCH_SIZE': 8, 'OCR_RECOGNITION_BATCH_SIZE': 64,
              '_parse_easyocr_items': lambda result, w, h: result,
              'log': SimpleNamespace(warning=lambda *args: None)}
        exec(compile(ast.Module(body=[func], type_ignores=[]), '<ocr-test>', 'exec'), ns)
        items = [{'source': i, 'w': 1280, 'h': 720} for i in range(19)]
        output = ns['_easyocr_read_group'](OCR(), items)
        self.assertEqual(output, [([i], None) for i in range(19)])
        self.assertTrue(all(size <= 8 and recognition == 64 for size, recognition in calls))


if __name__ == '__main__':
    unittest.main()
