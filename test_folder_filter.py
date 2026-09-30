import ast
from contextlib import nullcontext
from pathlib import Path
import sqlite3
import unittest

from folder_filter import matches_folder


class FolderFilterTests(unittest.TestCase):
    def test_folder_boundaries(self):
        self.assertTrue(matches_folder('topic/keyframes/L26_V001/001.jpg', 'L26'))
        self.assertTrue(matches_folder('topic/keyframes/L26_V002/999.jpg', 'l26'))
        self.assertTrue(matches_folder(r'topic\keyframes\L26_V001\001.jpg', 'L26_V001'))
        self.assertFalse(matches_folder('L260_V001/001.jpg', 'L26'))
        self.assertFalse(matches_folder('L26_V0010/001.jpg', 'L26_V001'))
        self.assertFalse(matches_folder('L27_V001/L26.jpg', 'L26'))
        self.assertFalse(matches_folder('L26XV001/001.jpg', 'L26_V001'))

    def test_sql_filter_includes_all_frames_without_video_facets(self):
        # Exercise the real query builder without importing GPU/model dependencies.
        source = ast.parse((Path(__file__).parent / 'app' / 'shared.py').read_text(encoding='utf-8'))
        function = next(n for n in source.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'query_frame_files')
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        conn.execute('CREATE TABLE frames(file TEXT)')
        files = ['topic/L26_V001/001.jpg', 'topic/L26_V001/002.jpg',
                 'topic/L26_V002/001.jpg', 'topic/L27_V001/001.jpg',
                 'topic/L26_V0010/001.jpg']
        conn.executemany('INSERT INTO frames VALUES (?)', [(f,) for f in files])
        scope = dict(TOP_K_MAX=100, REDIS_ENABLED=False, sqlite3=sqlite3,
                     matches_folder=matches_folder, get_db=lambda: nullcontext(conn))
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'shared.py', 'exec'), scope)
        query = scope['query_frame_files']
        self.assertEqual(query(video='L26_V001'), files[:2])
        self.assertEqual(set(query(video='L26')), set(files) - {files[3]})
        self.assertEqual(set(query(video='L26_V001', exclude_video=True)), set(files[2:]))
        self.assertEqual(query(video='L26', exclude_video=True), [files[3]])


if __name__ == '__main__':
    unittest.main()
