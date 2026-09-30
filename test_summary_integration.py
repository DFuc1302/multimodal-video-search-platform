"""Regression tests for multi-source summaries and safe publication."""
from contextlib import closing
import json
import os
from pathlib import Path
import sqlite3
import tempfile
import unittest
from unittest.mock import patch
import update_summary_index as build
import summary_catalog as catalog

class SummaryIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.old = self.root/'old'
        self.new = self.root/'classified'
        self.old.mkdir(); self.new.mkdir()
        self.addCleanup(catalog.active_categories.cache_clear)

    def summary(self, root, group, name, text='Một người đi xe đạp'):
        p=root/group/'video_summaries_json'/f'{name}.json'
        p.parent.mkdir(parents=True,exist_ok=True)
        build.dump(p,{'phan_doan_thoi_gian':[{'moc_thoi_gian':{'bat_dau':0,'ket_thuc':10},'mo_ta_hinh_anh':text}]})
        return p

    def fixture(self):
        self.summary(self.old,'09_news','L01_V001')
        self.summary(self.new,'11_cycle','S01-V001')
        self.summary(self.new,'12_security','N001_V001')
        self.summary(self.old,'01_food','L02_V001')
        rows=build.discover([self.old,self.new])
        source=self.root/'assets'; source.mkdir()
        maps=self.root/'maps'; maps.mkdir()
        files=['09_news/keyframes/L01_V001/001.jpg','S01_V001/frame_000001.jpg','N001-V001/0001.jpg','01_food/keyframes/L02_V001/001.jpg']
        with closing(sqlite3.connect(self.root/'meta.db')) as db, db:
            db.execute('CREATE TABLE frames(file TEXT)')
            db.executemany('INSERT INTO frames VALUES(?)',[(f,) for f in files])
        for file in files:
            p=source/file; p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'image')
            (maps/(Path(file).parent.name+'.csv')).write_text('n,pts_time,frame_idx\n1,5,125\n')
        build.dump(self.root/'external_storage.json',{'sources':[{'id':'s','data_root':str(source),'keyframes_dir':str(source),'map_dir':str(maps)}], 'video_sources':{Path(f).parent.name:'s' for f in files}})
        return rows,files

    def test_rules_and_reviewed_reassignment(self):
        self.assertEqual(catalog.topic('M01-V002','01'),'10')
        self.assertEqual(catalog.topic('S01_V002','05'),'11')
        self.assertEqual(catalog.topic('N001-V002',''),'12')
        self.assertEqual(catalog.topic('L21_V002','09_old'),'10')
        self.assertEqual(catalog.topic('L21_V002','07_old'),'07')
        self.summary(self.new,'10_reviewed','L20_V001')
        self.assertEqual(build.discover([self.new])[0]['category_code'],'10')

    def test_aliases_preserve_actual_paths_and_missing_segments(self):
        rows,files=self.fixture()
        docs,frames,issues=build.collect(rows,self.root)
        self.assertEqual(issues,[])
        self.assertEqual({f[1] for f in frames},set(files))
        self.assertEqual({d['category'] for d in docs},{'01','10','11','12'})
        docs.append(dict(video='S01_V001',category='11',start=20,end=30,text='empty interval',kind='segment'))
        missing=build.write_database(self.root/'summary.db',docs,frames)
        self.assertEqual(missing,[dict(video='S01_V001',start=20,end=30)])

    def test_conflicting_aliases_rejected(self):
        self.summary(self.old,'11_cycle','S01-V001','first')
        self.summary(self.new,'11_cycle','S01_V001','different')
        with self.assertRaisesRegex(ValueError,'Conflicting summaries'):
            build.discover([self.old,self.new])

    def test_bad_json_reported_without_losing_other_videos(self):
        rows,_=self.fixture()
        Path(rows[0]['summary_json']).write_text('{broken')
        docs,frames,issues=build.collect(rows,self.root)
        self.assertEqual(len({d['video'] for d in docs}),3)
        self.assertEqual(len(issues),1)

    def test_failed_validation_preserves_pointer(self):
        pointer=self.root/'active_summary.json'
        pointer.write_text('{"release":"summary_releases/old"}')
        original=pointer.read_bytes()
        with patch.object(build,'validate_release',side_effect=ValueError('bad vector count')):
            with self.assertRaises(ValueError):
                build.publish(self.root,self.root/'summary_releases/new')
        self.assertEqual(pointer.read_bytes(),original)

    def test_real_faiss_sqlite_publication_and_search(self):
        import faiss
        import numpy as np
        import summary_search
        rows,files=self.fixture()
        docs,frames,issues=build.collect(rows,self.root)
        release=self.root/'summary_releases/test';release.mkdir(parents=True)
        build.write_database(release/'summaries.db',docs,frames)
        build.dump(release/'manifest.json',rows)
        index=faiss.IndexFlatIP(4);index.add(np.eye(4,dtype='float32'))
        faiss.write_index(index,str(release/'summaries.faiss'))
        build.publish(self.root,release)
        with patch.dict(os.environ,{'DATA_ROOT':str(self.root)}):
            self.assertEqual(catalog.active_index(),release)
            self.assertEqual(catalog.file_topic(files[0]),'10')
            with patch('sentence_transformers.SentenceTransformer') as model:
                model.return_value.encode.return_value=np.array([[1,0,0,0]],dtype='float32')
                engine=summary_search.SummarySearch()
                result=engine.search('xe đạp',category='11',video='S01-V001')
                self.assertEqual(len(result['results']),1)
                self.assertEqual(result['results'][0]['file'],files[1])
                self.assertEqual(result['results'][0]['frame_idx'],125)
                self.assertEqual(result['results'][0]['category_code'],'11')
                build.dump(self.root/'active_summary.json',{'release':'summary_releases/future'})
                self.assertEqual(engine.search('xe đạp',category='11')['total'],1)

    def test_video_filter_aliases(self):
        from folder_filter import matches_folder
        self.assertTrue(matches_folder('S01-V001/frame_000001.jpg', 'S01_V001'))
        self.assertTrue(matches_folder('12_security/keyframes/N001_V001/001.jpg', 'N001-V001'))
        self.assertFalse(matches_folder('N001_V002/001.jpg', 'N001-V001'))

    def test_combined_api_keeps_reclassified_old_frame(self):
        import ast
        import threading
        from types import SimpleNamespace
        from fastapi import Query
        tree = ast.parse((Path(__file__).parent/'app'/'main.py').read_text(encoding='utf-8'))
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='search_summaries')
        function.decorator_list=[]
        old_file='09_news/keyframes/L01_V001/001.jpg'
        engine=SimpleNamespace(search=lambda *a: dict(results=[dict(file=old_file,category_code='10',score=1)],total=1))
        namespace=dict(Query=Query, _summary_engine=engine, _summary_engine_lock=threading.Lock(),
                       file_topic=lambda f:'10', search=lambda **kw:dict(results=[dict(file=old_file,score=1)]),
                       get_collection=lambda:SimpleNamespace(count=lambda:1))
        exec(compile(ast.Module(body=[function],type_ignores=[]),'<api-test>','exec'),namespace)
        result=namespace['search_summaries']('news',top_k=10,category='10',video='',mode='combined',page=0,page_size=100)
        self.assertEqual(result['total'],1)
        self.assertEqual(result['results'][0]['file'],old_file)

    def test_pointer_cannot_escape_release_root(self):
        build.dump(self.root/'active_summary.json',{'release':'../bad'})
        with self.assertRaises(ValueError):
            catalog.active_index(self.root)

if __name__=='__main__':
    unittest.main()
