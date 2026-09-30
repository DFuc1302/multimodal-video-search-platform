import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from migrate_to_redis import FrameRows, import_frames


class MigrationTests(unittest.TestCase):
    def test_rows_are_repeatable_and_not_loaded_as_a_list(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'meta.db'
            with sqlite3.connect(path) as conn:
                conn.execute('CREATE TABLE frames(file TEXT)')
                conn.executemany('INSERT INTO frames VALUES (?)', [('b',), ('a',)])
            rows = FrameRows(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(list(rows), [{'file': 'a'}, {'file': 'b'}])
            self.assertEqual(list(rows), list(rows))

    def test_hybrid_import_uses_no_image_vectors(self):
        with tempfile.TemporaryDirectory() as temp, patch('migrate_to_redis.faiss.read_index') as read:
            client = MagicMock()
            rows = [{'file': 'N001-V001/001.jpg', 'ocr_joined': 'hello', 'ocr_status': 'done'}]
            files, timelines, statuses = import_frames(client, 'test:', Path(temp), rows,
                                                       Path(temp), text_only=True)
            read.assert_not_called()
            payload = client.pipeline.return_value.hset.call_args.kwargs['mapping']
            self.assertNotIn('vector', payload)
            self.assertEqual(payload['text'], 'hello')
            self.assertEqual(statuses, {'done': 1})
            self.assertEqual(files, ['N001-V001/001.jpg'])


if __name__ == '__main__':
    unittest.main()
