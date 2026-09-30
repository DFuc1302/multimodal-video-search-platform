"""Regression checks using temporary data; no model downloads or production writes."""
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile
import numpy as np

os.environ.setdefault('SEARCH_LOG_ENABLED', '0')

import main
import kisc
import search_pages
from search_telemetry import SearchLogger
import asr_index
import import_colab_batch2 as batch_import
from federated_runtime import FederatedRuntime
from redis_store import RedisStore


class SearchTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(file=f'L26_V001/{n:03}.jpg', width=10, height=10,
                          objects=objects, categories=[], colors=[], detections=[],
                          has_text=True, ocr_joined='exit airport', ocr_ascii='exit airport')
                     for n, objects in [(1, ['person']), (2, ['car']), (3, ['person', 'car'])]]
        collection = SimpleNamespace(count=lambda: len(self.rows))
        for name, value in {
            'get_collection': lambda: collection,
            'get_meta_by_files': lambda files: [r for r in self.rows if r['file'] in files],
            'query_frame_files': lambda **kw: [r['file'] for r in self.rows],
            '_fetch_frame_rows': lambda files: [],
            '_ocr_match_files': lambda text, limit: ([r['file'] for r in self.rows], True),
            '_clip_and_candidates': lambda *a: ([r['file'] for r in self.rows],
                                               {r['file']: 1-i*.1 for i, r in enumerate(self.rows)}),
            '_clip_or_candidates': lambda *a: ([r['file'] for r in self.rows],
                                              {r['file']: 1-i*.1 for i, r in enumerate(self.rows)}),
        }.items():
            p = patch.object(main, name, value)
            p.start()
            self.addCleanup(p.stop)

    def search(self, **kwargs):
        args = dict(query=None, ocr_query=None, objects=None, categories=None, colors=None,
                    video=None, video_mode='include', min_conf=0., top_k=10,
                    query_operator='and', ocr_operator='and', search_mode='clip',
                    rerank=False, timeline=False, page=1, page_size=1000, domain='')
        args.update(kwargs)
        return main.search(**args)

    def test_ocr_clauses_with_video_filter(self):
        result = self.search(ocr_query='exit, airport', video='L26_V001')
        self.assertEqual(result['total'], 3)

    def test_explicit_object_remains_required(self):
        result = self.search(query='person, street', query_operator='or', objects='car')
        self.assertTrue(result['results'])
        self.assertTrue(all('car' in row['objects'] for row in result['results']))

    def test_pagination_preserves_reranking(self):
        reranker = SimpleNamespace(rerank=lambda q, pool, context: (list(reversed(pool)), {'status': 'applied'}))
        with patch.object(main, '_vlm_reranker', reranker):
            result = self.search(query='scene, street', rerank=True)
        self.assertEqual(result['results'][0]['file'], self.rows[-1]['file'])

    def test_first_page_does_not_request_the_complete_index(self):
        requested = []

        def candidates(_collection, _terms, n_results):
            requested.append(n_results)
            return ([r['file'] for r in self.rows],
                    {r['file']: 1-i*.1 for i, r in enumerate(self.rows)})

        large_collection = SimpleNamespace(count=lambda: 177321)
        with patch.object(main, 'get_collection', lambda: large_collection), \
             patch.object(main, '_clip_and_candidates', candidates):
            result = self.search(query='person, street', page=1, page_size=200)

        self.assertTrue(result['results'])
        self.assertEqual(requested, [1000])
        self.assertLess(requested[0], large_collection.count())

    def test_domain_only_without_pagination(self):
        self.rows[0]['file'] = '01_topic/L26_V001/001.jpg'
        with patch.object(main, 'get_all_meta', lambda: self.rows), \
             patch.object(main, 'query_frame_files', lambda **kw: []):
            result = self.search(domain='01', page=0)
        self.assertEqual(result['total'], 1)

    def test_ocr_fallback_respects_comma_operator(self):
        with patch.object(main, '_ocr_match_files', lambda *a: ([], False)):
            result = self.search(ocr_query='exit, airport', video='L26_V001')
        self.assertEqual(result['total'], 3)

    def test_ocr_and_no_hit_stays_empty(self):
        with patch.object(main, '_ocr_match_files', lambda term, limit: ([], True)):
            result = self.search(ocr_query='exit, absent', video='L26_V001')
        self.assertEqual(result['total'], 0)


class RuntimeTests(unittest.TestCase):
    def test_missing_image_is_not_server_error(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(main, 'DATA_ROOT', Path(temp)):
            self.assertIsNone(main._find_asset_in_candidate_roots('keyframes', 'missing/001.jpg'))

    def test_image_cannot_escape_media_root(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'media').mkdir()
            (root/'private.jpg').write_bytes(b'private')
            with patch.object(main, 'KEYFRAMES_DIR', root/'media'):
                for name in ('../private.jpg', str(root/'private.jpg'), '..\\private.jpg'):
                    self.assertIsNone(main._keyframe_path(name))

    def test_summary_missing_model_returns_503(self):
        with patch.object(main, 'REDIS_TEXT_ENABLED', True), patch.object(main, '_summary_engine', None), \
             patch('redis_serving.RedisSummarySearch', side_effect=OSError('missing model')):
            response = main.search_summaries(query='scene', top_k=2, category='', video='', mode='summary', page=0, page_size=1000)
        self.assertEqual(response.status_code, 503)

    def test_backup_includes_committed_wal_data(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            db = root/'meta.db'
            conn = sqlite3.connect(db)
            self.addCleanup(conn.close)
            conn.execute('PRAGMA journal_mode=WAL')
            conn.execute('CREATE TABLE example(value TEXT)')
            conn.execute("INSERT INTO example VALUES ('committed')")
            conn.commit()
            previous = Path.cwd()
            try:
                os.chdir(root)
                with patch.multiple(main, ALLOW_HEAVY_RUNTIME_JOBS=True, META_DB=db,
                                    FAISS_DIR=root/'absent', ASR_CACHE_FILE=root/'absent.json'):
                    result = main.create_backup()
                with zipfile.ZipFile(result['file']) as archive:
                    archive.extract('meta.db', root/'restored')
                with sqlite3.connect(root/'restored/meta.db') as restored:
                    self.assertEqual(restored.execute('SELECT value FROM example').fetchone()[0], 'committed')
            finally:
                os.chdir(previous)
                conn.close()

    def test_federated_folder_boundaries_without_facets(self):
        conn = sqlite3.connect(':memory:')
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        conn.execute('CREATE TABLE frames(file TEXT)')
        files = ['L26_V001/001.jpg', 'L26_V0010/001.jpg', 'L27_V001/001.jpg']
        conn.executemany('INSERT INTO frames VALUES (?)', [(f,) for f in files])
        reader = SimpleNamespace(connect=lambda: conn)
        runtime = FederatedRuntime.__new__(FederatedRuntime)
        runtime.readers = [reader]
        runtime.owner = lambda file: reader
        self.assertEqual(runtime.query_frame_files(video='L26_V001'), files[:1])
        self.assertEqual(runtime.query_frame_files(video='L26_V001', exclude_video=True), files[1:])

    def test_redis_facet_filters_normalize_case(self):
        store = RedisStore.__new__(RedisStore)
        store.all_rows = lambda: [dict(file='L26/1.jpg', colors=['Red'])]
        self.assertEqual(store.query_files(colors={'Red'}), ['L26/1.jpg'])

    def test_invalid_asr_page_does_not_build_search(self):
        with patch.object(search_pages, 'snapshot', side_effect=AssertionError('must validate before searching')):
            response = search_pages.asr_page({'phrase': 'hello', 'page': 'bad'})
        self.assertEqual(response.status_code, 400)

    def test_asr_fallback_ignores_old_cache_and_deduplicates_intervals(self):
        current = dict(video_id='L26_V001', video_name='L26_V001.mp4', current=True, segments=[])
        old = dict(current, video_id='L26_V002', current=False)
        match = dict(start=0, end=2, text='hello world')
        with patch.object(search_pages, 'REDIS_TEXT_ENABLED', False), \
             patch.object(search_pages, 'snapshot', lambda key, build: build()), \
             patch.object(main, '_search_asr_fts', lambda *a, **k: None), \
             patch.object(main, 'load_asr_cache', lambda: {}), \
             patch.object(main, '_asr_cache_files', lambda: []), \
             patch.object(main, '_asr_cache_inventory', lambda c: {'entries': [current, old]}), \
             patch.object(main, 'find_phrase_segments', lambda *a: [match]), \
             patch.object(main, '_load_asr_keyframe_timeline', lambda v: ([], {})):
            result = search_pages.asr_page({'phrase': 'hello, world'})
        self.assertEqual(result['matched_intervals'], 1)
        self.assertEqual(result['videos_scanned'], 1)


class ConversationTests(unittest.TestCase):
    def test_turn_uses_selected_storage_backend(self):
        session = kisc.KISSession('selected')
        file = 'L26_V001/001.jpg'
        vector = np.array([1., 0.], dtype=np.float32)
        collection = SimpleNamespace(count=lambda: 1,
            query=lambda **kw: {'metadatas': [[{'file': file}]], 'distances': [[0.]]},
            reconstruct_by_files=lambda files: {file: vector})
        loader = lambda files: [dict(file=file, objects=[], categories=[], colors=[])]
        with patch.object(kisc, 'get_session', lambda sid: session), \
             patch.object(kisc, 'get_collection', side_effect=AssertionError('wrong backend')), \
             patch.object(kisc, '_combined_query_vector', lambda q: vector), \
             patch.object(kisc, 'REDIS_ENABLED', False):
            result = kisc.process_turn('selected', 'scene', collection=collection, metadata_loader=loader)
        self.assertEqual(result['results'][0]['file'], file)

    def test_woman_is_not_man(self):
        self.assertEqual(kisc._extract_slot_hints('a woman outdoors')['gender'], 'woman')
        self.assertEqual(kisc._extract_slot_hints('female person')['gender'], 'woman')

    def test_partial_words_do_not_apply_filters(self):
        self.assertEqual(kisc._extract_object_hints('education in a busy town'), set())
        self.assertEqual(kisc._extract_color_hints('camera recording'), set())
        self.assertEqual(kisc._extract_object_hints('a hot dog and a cell phone'), {'hot dog', 'cell phone'})

    def test_expired_session_is_unavailable_without_starting_another(self):
        session = kisc.KISSession('expired')
        session.updated_at = 0
        with patch.object(kisc, 'REDIS_ENABLED', False), patch.object(kisc, '_sessions', {'expired': session}):
            self.assertIsNone(kisc.get_session('expired'))


class PipelineTests(unittest.TestCase):
    def test_invalid_asr_timestamps_are_rejected_without_crashing(self):
        for start, end in [(float('nan'), 2), (0, float('nan')), (float('inf'), float('inf')), (-2, 0)]:
            segment = dict(cache_version=2, text='hello', search_text='hello', start=start, end=end)
            self.assertFalse(asr_index.is_current_asr_segments([segment]))

    def test_asr_conflict_does_not_publish_imported_media(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source, destination = root/'source', root/'destination'
            source.mkdir()
            destination.mkdir()
            frames = source/'L26_V001'
            frames.mkdir()
            (frames/'001.jpg').write_bytes(b'test')
            mapping = source/'L26_V001.csv'
            mapping.write_text('n,frame_idx\n1,1\n')
            (destination/'asr_cache.json').write_text(json.dumps({'video-name-v1::L26_V001.mp4': ['old']}))
            with patch.object(batch_import, 'validate_source', lambda p: [('L26_V001', frames, mapping, 1)]), \
                 patch.object(batch_import, 'merge_asr_shards', lambda p: {'video-name-v1::L26_V001.mp4': ['new']}), \
                 patch.object(batch_import, 'validate_frame_output', lambda *a, **k: None):
                with self.assertRaises(batch_import.ImportError):
                    batch_import.import_outputs(source=source, destination=destination, existing_roots=[], apply=True, merge_asr=True)
            self.assertFalse((destination/'keyframes/L26_V001').exists())


class TelemetryTests(unittest.TestCase):
    def test_frame_pagination_records_ranks_and_nested_frames(self):
        with tempfile.TemporaryDirectory() as temp:
            logger = SearchLogger(temp)
            events = []
            with patch.object(logger, '_append', events.append):
                cases = [('/search', {'results': [{'file': 'L26/1.jpg'}]}),
                         ('/asr/frames', {'results': [{'frames': [{'file': 'L26/1.jpg'}]}]}),
                         ('/video-match/frames', {'videos': [{'frames': [{'file': 'L26/1.jpg'}]}]})]
                for endpoint, payload in cases:
                    logger.retrieval(endpoint, {'query': 'hello'}, dict(payload, page=2, total=2000), 200, 10, 'a'*32)
            self.assertEqual(len(events), 3)
            for event in events:
                self.assertEqual(event['ranking'], ['L26/1.jpg'])
                self.assertEqual(event['rank_offset'], 1000)

    def test_different_summary_filters_have_different_query_ids(self):
        with tempfile.TemporaryDirectory() as temp:
            logger = SearchLogger(temp)
            events = []
            with patch.object(logger, '_append', events.append):
                for category in ['01', '02']:
                    logger.retrieval('/summary/search', {'query': 'scene', 'category': category, 'mode': 'summary'}, {}, 200, 1, 'a'*32)
            self.assertNotEqual(events[0]['query_id'], events[1]['query_id'])


if __name__ == '__main__':
    unittest.main()
