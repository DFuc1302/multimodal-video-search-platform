import unittest

from align_official_keyframe_maps import align


class OfficialMapAlignmentTests(unittest.TestCase):
    def test_aligns_by_time_not_image_number_and_preserves_decoded_index(self):
        official = [{'n': '1', 'pts_time': '0', 'fps': '24', 'frame_idx': '0'},
                    {'n': '2', 'pts_time': '10', 'fps': '24', 'frame_idx': '240'}]
        local = [{'n': '2', 'pts_time': '1', 'fps': '25', 'frame_idx': '25'},
                 {'n': '11', 'pts_time': '10', 'fps': '25', 'frame_idx': '250'}]
        result, stats = align(local, official)
        self.assertEqual([r['frame_idx'] for r in result], ['24', '240'])
        self.assertEqual([r['n'] for r in result], ['2', '11'])
        self.assertEqual([r['pts_time'] for r in result], ['1', '10'])
        self.assertEqual([r['source_frame_idx'] for r in result], ['25', '250'])
        self.assertEqual(stats['exact_timestamp_anchors'], 1)
        self.assertEqual(align(result, official)[0], result)
        self.assertEqual(align(result, official)[1]['idx_changed'], 0)

    def test_matches_btc_float_truncation(self):
        row = {'n': '4', 'pts_time': '16.08', 'fps': '25.0', 'frame_idx': '401'}
        result, _ = align([dict(row, frame_idx='402')], [row])
        self.assertEqual(result[0]['frame_idx'], '401')

    def test_rejects_incompatible_official_formula(self):
        row = {'n': '1', 'pts_time': '1', 'fps': '25', 'frame_idx': '26'}
        with self.assertRaisesRegex(ValueError, 'convention'):
            align([row], [row])

    def test_rejects_collisions_instead_of_inventing_an_index(self):
        local = [{'n': '1', 'pts_time': '173.486', 'fps': '25', 'frame_idx': '3753'},
                 {'n': '2', 'pts_time': '173.527', 'fps': '25', 'frame_idx': '3754'}]
        official = [{'n': '1', 'pts_time': '0', 'fps': '21.12', 'frame_idx': '0'}]
        with self.assertRaisesRegex(ValueError, 'collision'):
            align(local, official)


if __name__ == '__main__':
    unittest.main()
