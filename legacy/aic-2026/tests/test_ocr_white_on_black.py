"""Offline filter regressions: python -m unittest test_ocr_white_on_black."""
import ast
from pathlib import Path
import unittest

import numpy as np

from ocr_quality import filter_white_on_black


class WhiteOnBlackTests(unittest.TestCase):
    def reading(self, text="road", box=None):
        return dict(text=text, conf=.9, box=box or [0, 0, 1, 1])

    def panel(self, background=20, foreground=(235, 235, 235)):
        image = np.full((40, 200, 3), background, dtype=np.uint8)
        for x in range(5, 190, 10):
            image[8:32, x:x+3] = foreground
        return image

    def test_white_text_and_timestamp_kept_with_original_boxes(self):
        texts = [self.reading(), self.reading("10.Jun 2026 18:59:56")]
        self.assertEqual(filter_white_on_black(self.panel(), texts), texts)

    def test_colored_text_light_background_and_no_text_rejected(self):
        for image in (self.panel(foreground=(0, 220, 255)),
                      self.panel(background=180),
                      self.panel(foreground=(20, 20, 20))):
            self.assertEqual(filter_white_on_black(image, [self.reading()]), [])

    def test_scenery_in_one_corner_rejected(self):
        image = self.panel()
        image[:20, :100] = (110, 140, 180)
        self.assertEqual(filter_white_on_black(image, [self.reading()]), [])

    def test_bad_boxes_rejected(self):
        for box in ([1, 1, 0, 0], [2, 2, 3, 3], [float('nan'), 0, 1, 1]):
            self.assertEqual(filter_white_on_black(self.panel(), [self.reading(box=box)]), [])

    def test_refinement_is_filtered_using_original_pixels(self):
        # Load only the finishing function; no model or production DB imports.
        tree = ast.parse((Path(__file__).parent / 'pipeline' / 'ocr' / 'index.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_finish_ocr')
        texts = [self.reading()]
        scope = dict(OCR_ENHANCE=True, OCR_WHITE_ON_BLACK_ONLY=True,
                     filter_white_on_black=filter_white_on_black,
                     _refine_ocr=lambda image, readings: texts)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'ocr_index.py', 'exec'), scope)
        finish = scope['_finish_ocr']
        self.assertEqual(finish(self.panel(background=180), []), [])
        self.assertEqual(finish(self.panel(), []), texts)
        # Internal enhanced tiles must not be color-filtered.
        self.assertEqual(finish(self.panel(background=180), texts, enhance=False), texts)
        scope['OCR_ENHANCE'] = False
        self.assertEqual(finish(self.panel(background=180), texts), [])
        self.assertEqual(finish(self.panel(), texts), texts)
        scope['OCR_WHITE_ON_BLACK_ONLY'] = False
        self.assertEqual(finish(self.panel(background=180), texts), texts)

    def test_fast_decode_keeps_original_colors_for_filter(self):
        tree = ast.parse((Path(__file__).parent / 'pipeline' / 'ocr' / 'index.py').read_text(encoding='utf-8'))
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_read_prepared_ocr_image')
        image = self.panel()
        from typing import Optional
        scope = dict(Path=Path, Optional=Optional, np=np, OCR_ENHANCE=False,
                     OCR_WHITE_ON_BLACK_ONLY=True, _read_image=lambda path: image,
                     _prepare_ocr_image=lambda original: original[::2, ::2])
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'ocr_index.py', 'exec'), scope)
        self.assertIs(scope['_read_prepared_ocr_image'](Path('example.jpg')), image)
        scope['OCR_WHITE_ON_BLACK_ONLY'] = False
        self.assertEqual(scope['_read_prepared_ocr_image'](Path('example.jpg')).shape, (20, 100, 3))


if __name__ == '__main__':
    unittest.main()
