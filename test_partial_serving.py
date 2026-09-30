"""Partial datasets must not prevent independent search features from working."""
import json
import os
import shutil
import subprocess
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
import main
import shared
import server_data_health as health
import run_new_dataset_pipeline as pipeline


class PartialServingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.db = self.root / 'meta.db'
        with sqlite3.connect(self.db) as conn:
            conn.execute('''CREATE TABLE frames (file TEXT PRIMARY KEY, mtime REAL,
                width INTEGER, height INTEGER, objects TEXT, categories TEXT, colors TEXT,
                has_text INTEGER, ocr_joined TEXT, ocr_ascii TEXT, detections TEXT,
                ocr_texts TEXT, ocr_status TEXT)''')
            conn.executemany('INSERT INTO frames VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)', [
                ('N001-V001/001.jpg', 1, 100, 100, '["car"]', '["Vehicles"]', '["white"]',
                 1, 'hello road', 'hello road', '[]', '[broken', 'done'),
                ('N001-V001/002.jpg', 1, 100, 100, '[]', '[]', '[]',
                 0, '', '', '[]', '[]', 'pending')])
        def get_db():
            conn = sqlite3.connect(self.db.resolve().as_uri() + '?mode=ro', uri=True)
            conn.row_factory = sqlite3.Row
            return conn
        for obj, name, value in [
            (health, 'LOG_PATH', self.root / 'sever_log.json'), (health, '_issues', {}),
            (health, '_scan', {'status': 'not_started'}),
            (main, 'ALLOW_PARTIAL_DATA', True), (main, 'REDIS_ENABLED', False),
            (main, 'REDIS_TEXT_ENABLED', False), (main, 'get_db', get_db),
            (main, 'DATA_ROOT', self.root), (main, 'KEYFRAMES_DIR', self.root / 'keyframes'),
            (main, 'VIDEOS_DIR', self.root / 'videos'), (main, 'META_DB', self.db),
            (main, '_federated_runtime', lambda: None),
            (main, '_single_get_collection', lambda: health.EmptyCollection()),
            (main, 'scan_dataset', lambda *a: None),
            (main, '_single_load_asr_cache', lambda: {}),
            (main, '_readiness', dict(main._readiness)),
            (shared, 'get_db', get_db), (shared, 'META_CACHE_MODE', 'lazy'),
            (shared, 'FTS5_AVAILABLE', False), (shared, '_meta_cache', {})]:
            p = patch.object(obj, name, value)
            p.start()
            self.addCleanup(p.stop)
        self.client = TestClient(main.app)

    def test_startup_ocr_and_filters_survive_missing_vectors_fts_and_bad_json(self):
        with self.client as client:
            self.assertEqual(client.get('/ready').status_code, 200)
            result = client.get('/search', params={'ocr_query': 'hello'}).json()
            self.assertEqual([r['file'] for r in result['results']], ['N001-V001/001.jpg'])
            self.assertEqual(result['results'][0]['ocr_texts'], [])
            result = client.get('/search', params={'objects': 'car'}).json()
            self.assertEqual(result['total'], 1)
            result = client.get('/search', params={'query': 'hello', 'search_mode': 'hybrid'}).json()
            self.assertEqual(result['total'], 1)
            self.assertTrue(result['partial'])

    def test_missing_asr_and_vectors_return_empty_results_and_keep_ui_alive(self):
        with self.client as client:
            self.assertEqual(client.post('/asr/search', json={'phrase': 'hello'}).json()['total'], 0)
            self.assertEqual(client.get('/similar', params={'file': 'N001-V001/001.jpg'}).json()['total'], 0)
            result = client.get('/video-match', params={'query': 'road'}).json()
            self.assertEqual(result['videos'], [])
            result = client.get('/video-match/frames', params={'query': 'road', 'page': 1}).json()
            self.assertEqual(result['videos'], [])
            self.assertEqual(client.get('/').status_code, 200)

    def test_missing_media_is_logged_without_breaking_subsequent_search(self):
        with self.client as client:
            self.assertEqual(client.get('/image/N001-V001/001.jpg').status_code, 404)
            self.assertEqual(client.get('/search', params={'ocr_query': 'hello'}).json()['total'], 1)
        payload = json.loads(health.LOG_PATH.read_text())
        self.assertTrue(any(i['component'] == 'keyframe' for i in payload['issues']))

    def test_asr_processing_disabled(self):
        with patch.dict('os.environ', {'ASR_PROCESSING_ENABLED': '0'}):
            self.assertEqual(main.asr_transcribe_all({}).status_code, 403)
        self.assertNotIn('asr', pipeline.STAGES)

    def test_inventory_reports_pending_missing_and_unindexed_frames(self):
        folder = self.root / 'keyframes' / 'N001-V001'
        folder.mkdir(parents=True)
        (folder / '001.jpg').touch()
        (folder / '003.jpg').touch()
        health.scan_dataset(self.root)
        payload = json.loads(health.LOG_PATH.read_text())
        components = {i['component'] for i in payload['issues']}
        self.assertEqual(payload['scan']['status'], 'complete')
        self.assertTrue({'inventory.ocr.pending', 'inventory.keyframe_missing',
                         'inventory.metadata_missing', 'inventory.map_missing',
                         'inventory.asr_external_pending'} <= components)

    def test_repeated_issue_does_not_grow_json(self):
        for _ in range(10):
            health.record_issue('image', 'Missing', path='one.jpg')
        self.assertEqual(len(json.loads(health.LOG_PATH.read_text())['issues']), 1)

    def test_pipeline_audit_and_resume_need_no_asr_argument_or_output(self):
        args = pipeline.build_parser().parse_args(['--stage', 'audit'])
        args.work_dir = self.root / 'work'
        args.work_dir.mkdir()
        args.old_index = self.root / 'old'
        args.old_index.mkdir()
        (args.old_index / 'serve_manifest.json').write_text(json.dumps({'clip_model': 'test', 'frame_count': 0}))
        args.merged_index = self.root / 'merged'
        args.extracted_dir = self.root / 'extracted'
        args.videos_dir = self.root / 'videos'
        with patch.object(pipeline, 'inventory', return_value=[]), \
             patch.object(pipeline, 'old_video_ids', return_value=set()), \
             patch.object(pipeline, 'run') as run:
            pipeline.execute(args)
            pipeline.execute(args)
        run.assert_not_called()
        state = json.loads((args.work_dir / 'state.json').read_text())
        self.assertEqual(state['recipe']['whisper_model'], 'external')
        self.assertFalse((args.work_dir / 'new_index/asr_cache.json').exists())

    def test_malformed_external_asr_does_not_block_packaging(self):
        import merge_processed_data
        import prepare_serve_bundle
        path = self.root / 'asr_cache.json'
        path.write_text('{unfinished')
        with patch.dict('os.environ', {'ASR_EXTERNAL': '1'}):
            self.assertEqual(merge_processed_data._load_asr_entries(path), {})
            with sqlite3.connect(':memory:') as conn:
                result = prepare_serve_bundle._build_asr_fts(conn, path, batch_size=10, optimize=False)
            self.assertEqual(result, {'rows': 0, 'videos': 0})
        self.assertEqual(path.read_text(), '{unfinished')

    @unittest.skipUnless(os.name == 'posix', 'WSL launchers require bash')
    def test_redis_launchers_fall_back_without_stopping_existing_services(self):
        activate = self.root / 'venv/bin/activate'
        activate.parent.mkdir(parents=True)
        activate.write_text('true\n')
        (self.root / 'start_redis_wsl.sh').write_text('exit 1\n')
        (self.root / 'run_server_wsl.sh').write_text(
            'echo "FALLBACK:$BACKEND_MODE:$STORAGE_BACKEND:$SERVER_REDIS_FALLBACK"\n')
        env = dict(os.environ, TEAM_VENV=str(self.root / 'venv'),
                   KEYFRAMES_DIR=str(self.root), DATA_ROOT=str(self.root), ALLOW_PARTIAL_DATA='1')
        for name in ('run_hybrid_server_wsl.sh', 'run_redis_server_wsl.sh'):
            target = self.root / name
            shutil.copyfile(Path(__file__).with_name(name), target)
            result = subprocess.run(['bash', str(target)], env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('FALLBACK:sqlite:sqlite:1', result.stdout)
            strict = subprocess.run(['bash', str(target)], env={**env, 'ALLOW_PARTIAL_DATA': '0'},
                                    capture_output=True, text=True)
            self.assertNotEqual(strict.returncode, 0)
            self.assertNotIn('FALLBACK:', strict.stdout)


if __name__ == '__main__':
    unittest.main()
