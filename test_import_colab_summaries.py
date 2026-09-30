import csv
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from import_colab_summaries import collect, segment_text


class ImportTests(unittest.TestCase):
    def test_search_text_excludes_model_metadata(self):
        self.assertEqual(segment_text(dict(mo_ta_hinh_anh='Cầu', am_thanh_loi_thoai='Tin tức',
            generation_profile=dict(model='Qwen', max_pixels=112896))), 'Cầu\nTin tức')

    def test_exact_frame_mapping_and_missing_frame_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); source=root/'summaries'; source.mkdir()
            maps=root/'maps'; maps.mkdir()
            (root/'external_storage.json').write_text(json.dumps(dict(sources=[dict(id='M',
                data_root=str(root), map_dir=str(maps))], video_sources={'m01_v001':'M'})))
            with sqlite3.connect(root/'meta.db') as db:
                db.execute('CREATE TABLE frame_locator(video TEXT, n INTEGER, file TEXT)')
                db.execute("INSERT INTO frame_locator VALUES('m01_v001', 7, 'M01_V001/0007.jpg')")
            db.close()
            (maps/'M01_V001.csv').write_text('n,pts_time,frame_idx\n7,20,500\n8,40,1000\n')
            data=dict(video_metadata=dict(video_id='M01_V001'), phan_doan_thoi_gian=[
                dict(moc_thoi_gian=dict(bat_dau=0,ket_thuc=30),mo_ta_hinh_anh='Cầu'),
                dict(moc_thoi_gian=dict(bat_dau=30,ket_thuc=60),mo_ta_hinh_anh='Phố')])
            file=source/'M01_V001.json'; file.write_text(json.dumps(data), encoding='utf-8')
            docs, frames, missing, hashes=collect(source, root)
            self.assertEqual(len(docs),2)
            self.assertEqual(frames,[('M01_V001','M01_V001/0007.jpg',20.0,500)])
            self.assertEqual(missing,[dict(video='M01_V001',start=30.0,end=60.0)])
            self.assertEqual(len(hashes),1)
            data['phan_doan_thoi_gian'][1]['moc_thoi_gian']['bat_dau']=31
            file.write_text(json.dumps(data), encoding='utf-8')
            with self.assertRaises(ValueError):
                collect(source,root)


if __name__=='__main__':
    unittest.main()
