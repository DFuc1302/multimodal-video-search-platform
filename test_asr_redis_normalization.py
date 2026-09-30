import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from migrate_to_redis import import_asr
from shared import normalize_ascii_text


class AsrNormalizationTest(unittest.TestCase):
    def test_import_preserves_transcript_and_normalizes_search_field(self):
        segment = {'text': 'Đài truyền hình Thành phố Hồ Chí Minh',
                   'start': 2.0, 'end': 4.0, 'cache_version': 2}
        client = MagicMock()
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'asr_cache.json').write_text(json.dumps({
                'video-name-v1::M01_V001.mp4': [segment],
            }), encoding='utf-8')
            self.assertEqual(import_asr(client, 'test:', root), (1, 1))
        mapping = client.pipeline.return_value.hset.call_args.kwargs['mapping']
        self.assertEqual(mapping['text'], 'dai truyen hinh thanh pho ho chi minh')
        self.assertEqual(mapping['text'], normalize_ascii_text(segment['text']))
        payload = json.loads(mapping['payload'])
        for field, value in segment.items():
            self.assertEqual(payload[field], value)


if __name__ == '__main__':
    unittest.main()
