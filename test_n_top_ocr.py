import unittest
import numpy as np
from rerun_n_top_ocr import white_readings


class TopWhiteTests(unittest.TestCase):
    def test_white_on_colored_background_and_coordinate_mapping(self):
        image = np.full((100,200,3), (120,90,30), dtype=np.uint8)
        image[20:25,20:70] = 245
        item = dict(text='ROAD',conf=.9,box=[.05,.1,.4,.4])
        result = white_readings(image,[item],.25)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['box'],[.05,.025,.4,.1])

    def test_colored_text_without_white_is_excluded(self):
        image = np.zeros((100,200,3),dtype=np.uint8)
        image[20:25,20:70]=(0,0,255)
        self.assertEqual(white_readings(image,[dict(text='RED',conf=.9,box=[.05,.1,.4,.4])],.25),[])


if __name__ == '__main__':
    unittest.main()
