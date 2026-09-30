"""Prove real OCR/ASR routes stay responsive while CLIP is blocked or fails."""
import sqlite3
import threading
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
import main
import test_partial_serving as fixtures
from startup_access import request_available


class StagedStartupTests(unittest.TestCase):
    def setUp(self):
        fixtures.PartialServingTests.setUp(self)
        with sqlite3.connect(self.db) as db:
            for name in ['frame_facets', 'frame_locator']:
                db.execute(f'CREATE TABLE {name} (file TEXT)')
            db.execute('CREATE VIRTUAL TABLE frames_fts USING fts5(file UNINDEXED, text)')
            db.execute("INSERT INTO frames_fts VALUES ('N001-V001/001.jpg', 'hello road')")
            db.execute('CREATE TABLE serve_meta (key TEXT, value TEXT)')
            db.executemany('INSERT INTO serve_meta VALUES (?,?)', [('schema_version','2'),('frame_count','2')])
        p=patch.object(main._shared,'FTS5_AVAILABLE',True);p.start();self.addCleanup(p.stop)
        col = MagicMock()
        col.count.return_value = 2
        col.dimension.return_value = 2
        col.query.return_value = {'metadatas': [[{'file':'N001-V001/001.jpg'}]]}
        for name, value in [('ALLOW_PARTIAL_DATA',False), ('init_db',lambda:None),
                            ('warm_meta_cache',lambda:None), ('_load_serve_manifest',lambda:{}),
                            ('_single_get_collection',lambda:col)]:
            p=patch.object(main,name,value);p.start();self.addCleanup(p.stop)

    def exercise(self, fail):
        entered, release = threading.Event(), threading.Event()
        collection_loads = []
        collection = main._single_get_collection()
        def get_collection():
            self.assertTrue(main._readiness.get('text_ready'), 'FAISS opened before OCR became ready')
            collection_loads.append(True)
            return collection
        def clip(*args):
            entered.set()
            if not release.wait(10): raise RuntimeError('Test CLIP timeout')
            if fail: raise RuntimeError('Test CLIP failure')
            return np.array([1.,0.])
        with patch.object(main,'embed_texts',side_effect=clip), patch.object(main, 'get_collection', side_effect=get_collection), patch.dict('os.environ',{'PRELOAD_CLIP_ON_STARTUP':'1','AUTO_INDEX_ON_STARTUP':'0'}):
            with self.client as client:
                try:
                    self.assertTrue(entered.wait(5))
                    state=client.get('/ready').json()
                    self.assertTrue(state['text_ready'])
                    self.assertFalse(state['ready'])
                    ocr=client.get('/search',params={'ocr_query':'hello'})
                    self.assertEqual(ocr.status_code,200,ocr.text)
                    self.assertEqual(ocr.json()['total'],1)
                    self.assertEqual(client.post('/asr/search',json={'phrase':'hello'}).status_code,200)
                    self.assertEqual(client.post('/asr/frames',json={'phrase':'hello'}).status_code,200)
                    self.assertEqual(client.get('/search',params={'query':'road'}).status_code,503)
                    self.assertEqual(client.get('/video-match/frames',params={'query':'road'}).status_code,503)
                    self.assertEqual(client.get('/search',params={'query':'nohit','search_mode':'ocr'}).json()['total'],0)
                finally:
                    release.set()
                    main.app.state.startup_thread.join(5)
                state=client.get('/ready').json()
                self.assertEqual(state['ready'],not fail)
                self.assertTrue(state['text_ready'])
                self.assertEqual(client.get('/search',params={'ocr_query':'hello'}).json()['total'],1)
                self.assertTrue(collection_loads)

    def test_text_search_before_clip_completes(self):
        self.exercise(False)

    def test_clip_failure_preserves_text_search(self):
        self.exercise(True)

    def test_gate_never_admits_visual_or_mutation_while_loading(self):
        state={'ready':False,'text_ready':True}
        for path,params,method in [('/search',{'query':'road'},'GET'),('/similar',{},'GET'),
                                  ('/asr/transcribe',{},'POST'),('/search',{'rerank':'True'},'GET')]:
            self.assertFalse(request_available(state,path,params,method))


if __name__ == '__main__': unittest.main()
