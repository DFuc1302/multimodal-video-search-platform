import tempfile
import unittest
import zipfile
from pathlib import Path

from dres_frames import FrameError, FrameResolver, build_answer
from official_submission_maps import OfficialMaps


class OfficialMapTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.local = self.root/'M01_V001.csv'
        self.local.write_text('n,pts_time,fps,frame_idx\n1,0,25,0\n2,2.1,25,53\n3,4.1,25,103\n4,2.2,25,55\n5,3,25,75\n6,9,25,225\n')
        self.archive = self.root/'official.zip'
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr('map-keyframes/M01-V001.csv', 'n,pts_time,fps,frame_idx\r\r\n1,0,25,0\r\r\n2,2,25,50\r\r\n3,4,25,100\r\r\n')
        self.resolver = FrameResolver(lambda _: self.local, OfficialMaps(self.archive))

    def test_official_payload_and_local_evidence(self):
        result = build_answer(self.resolver, 'trake', 'M01_V001', [53,103])
        self.assertEqual(result['payload']['answerSets'][0]['answers'][0]['text'], 'TR-M01-V001-50,100')
        self.assertEqual(result['mapping']['frames'][0]['frame_idx'], 53)
        self.assertEqual(result['mapping']['frames'][0]['delta_ms'], -100)
        self.assertTrue(result['warnings'])
        by_n = build_answer(self.resolver, 'trake', 'M01_V001', [2,3], frame_mode='n')
        self.assertEqual(result['payload'], by_n['payload'])

    def test_ambiguous_distant_and_collapsed_events_blocked(self):
        for frames in ([75], [225], [53,55]):
            with self.subTest(frames=frames), self.assertRaises(FrameError):
                build_answer(self.resolver, 'trake', 'M01_V001', frames)

    def test_missing_map_does_not_fallback_to_local(self):
        self.resolver.official_maps = OfficialMaps(self.root/'missing')
        with self.assertRaisesRegex(FrameError, 'map BTC'):
            build_answer(self.resolver, 'trake', 'M01_V001', [53])
        kis = build_answer(self.resolver, 'textual_kis', 'M01_V001', [53])
        self.assertEqual(kis['payload']['answerSets'][0]['answers'][0]['start'], 2100)

    def test_conflicting_official_aliases_blocked(self):
        with zipfile.ZipFile(self.archive, 'a') as z:
            z.writestr('other/M01_V001.csv', self.local.read_text())
        with self.assertRaisesRegex(FrameError, '2 file'):
            build_answer(self.resolver, 'trake', 'M01_V001', [53])

    def test_duplicate_official_rows_blocked(self):
        with zipfile.ZipFile(self.archive, 'w') as z:
            z.writestr('M01_V001.csv', 'n,pts_time,fps,frame_idx\n1,0,25,0\n1,2,25,50\n')
        with self.assertRaisesRegex(FrameError, 'không hợp lệ'):
            build_answer(self.resolver, 'trake', 'M01_V001', [53])


if __name__ == '__main__':
    unittest.main()
