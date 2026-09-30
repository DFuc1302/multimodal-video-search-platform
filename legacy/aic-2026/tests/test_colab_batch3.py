import ast
import csv
from fractions import Fraction
import io
import json
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import tempfile
import types
import unittest
from unittest.mock import patch
import zipfile

import colab_batch3 as runner
from import_colab_batch3 import unpack


class Batch3Tests(unittest.TestCase):
    def test_progress_does_not_count_failed_work(self):
        class Bar:
            def __init__(self): self.done = 0
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def update(self, count): self.done += count
        bar = Bar()
        with patch.object(runner, 'progress_bar', return_value=bar):
            items = runner.progress_items([1, 2, 3], 'test')
            with self.assertRaisesRegex(RuntimeError, 'failed'):
                for item in items:
                    if item == 2:
                        raise RuntimeError('failed')
            items.close()
        self.assertEqual(bar.done, 1)

    def test_resume_accepts_only_known_display_change(self):
        from colab_batch3_progress_compat import PRE_PROGRESS_CODE
        old = {'version':5, 'interval':1, 'code':PRE_PROGRESS_CODE}
        current = {'version':5, 'interval':1, 'code':{'new':'source'}}
        self.assertTrue(runner.recipe_matches(old, current))
        self.assertFalse(runner.recipe_matches(old, {**current, 'interval':0.5}))
        self.assertFalse(runner.recipe_matches({**old, 'code':{'unknown':'source'}}, current))

    def test_grouping_and_duplicate_ids(self):
        self.assertEqual(runner.group_for_video("m10-V001"), "Videos_M10.zip")
        self.assertEqual(runner.group_for_video("N040_V001"), "Video_N031-N040.zip")
        self.assertIsNone(runner.group_for_video("N041_V001"))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "M01_V001.mp4").touch()
            (root / "nested").mkdir()
            (root / "nested" / "m01_v001.mkv").touch()
            with self.assertRaisesRegex(RuntimeError, "Duplicate"):
                runner.loose_videos(root, "Videos_M01.zip")

    def test_unsafe_and_duplicate_zip_members(self):
        for names in [("../M01_V001.mp4",), ("a/M01_V001.mp4", "b/m01_v001.mkv")]:
            stream = io.BytesIO()
            with zipfile.ZipFile(stream, "w") as z:
                for name in names:
                    z.writestr(name, b"video")
            with zipfile.ZipFile(stream) as z, self.assertRaises(RuntimeError):
                runner.video_members(z)

    def test_export_import_and_corrupt_download(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "source"
            root.mkdir()
            (root / "keyframes" / "M01_V001").mkdir(parents=True)
            (root / "keyframes" / "M01_V001" / "0001.jpg").write_bytes(b"image")
            (root / "videos").mkdir()
            (root / "videos" / "M01_V001.mp4").write_bytes(b"must not export")
            (root / "meta.db").write_bytes(b"metadata")
            archive = Path(temp) / "processed.zip"
            runner.package(root, archive)
            with zipfile.ZipFile(archive) as z:
                self.assertNotIn("videos/M01_V001.mp4", z.namelist())
                self.assertIn("keyframes/M01_V001/0001.jpg", z.namelist())
            destination = Path(temp) / "imported"
            unpack(archive, destination)
            unpack(archive, destination)  # idempotent matching receipt
            self.assertEqual((destination / "meta.db").read_bytes(), b"metadata")
            with archive.open("ab") as f:
                f.write(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                unpack(archive, destination)

    def test_import_rejects_traversal_even_with_matching_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            archive = Path(temp) / "bad.zip"
            with zipfile.ZipFile(archive, "w") as z:
                z.writestr("../escape.txt", "bad")
            runner.write_json(archive.with_suffix('.sha256.json'), {"sha256": runner.digest(archive)})
            with self.assertRaisesRegex(RuntimeError, "Unsafe"):
                unpack(archive, Path(temp) / "target")
            self.assertFalse((Path(temp) / "escape.txt").exists())

    def test_dedup_keeps_original_frame_numbers_and_adjacent_chain(self):
        from PIL import Image
        # A-B and B-C are close, A-C is not: only A remains in the chain.
        fake_hash = types.SimpleNamespace(phash=lambda im: im.getpixel((0, 0)))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            folder = root / "keyframes" / "M01_V001"
            folder.mkdir(parents=True)
            (root / "map-keyframes").mkdir()
            rows = []
            for n, value in enumerate([0, 3, 6, 100], 1):
                Image.new("L", (8, 8), value).save(folder / f"{n:04d}.jpg")
                rows.append(dict(n=n, pts_time=(n-1)/2, fps=30, frame_idx=(n-1)*15))
            # int subtraction is signed, so use a tiny absolute-distance hash.
            class Hash:
                def __init__(self, value): self.value = value
                def __sub__(self, other): return abs(self.value-other.value)
            fake_hash.phash = lambda im: Hash(im.getpixel((0, 0)))
            with patch.dict('sys.modules', {'imagehash': fake_hash}):
                self.assertEqual(runner.dedup_frames(root, 'M01_V001', rows, 4), 2)
            self.assertEqual(sorted(p.name for p in folder.iterdir()), ['0001.jpg', '0004.jpg'])
            with (root / 'map-keyframes' / 'M01_V001.csv').open() as f:
                kept = list(csv.DictReader(f))
            self.assertEqual([r['frame_idx'] for r in kept], ['0', '45'])

    @unittest.skipUnless(shutil.which('ffmpeg') and shutil.which('ffprobe'), 'requires FFmpeg')
    def test_real_video_frame_indices_and_timestamps(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / 'M01_V001.mp4'
            subprocess.run(['ffmpeg', '-v', 'error', '-f', 'lavfi', '-i',
                'testsrc2=size=160x90:rate=30000/1001', '-t', '2', '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p', str(video)], check=True)
            rows = runner.extract_frames(video, root / 'processed')
            self.assertEqual([r['frame_idx'] for r in rows], [0, 14, 29, 44, 59])
            for row in rows:
                self.assertAlmostEqual(float(row['pts_time']), row['frame_idx'] * 1001/30000, places=4)
            speed_rows = runner.extract_frames(video, root / 'speed', interval=1)
            self.assertEqual([r['frame_idx'] for r in speed_rows], [0, 29, 59])
            for row in speed_rows:
                self.assertAlmostEqual(float(row['pts_time']), row['frame_idx'] * 1001/30000, places=4)

    def test_visual_pipeline_never_invokes_asr_and_keeps_n_filter(self):
        for stem, group, white_only in [('M01_V001', 'Videos_M01.zip', '0'),
                                        ('N031_V001', 'Video_N031-N040.zip', '1')]:
            with self.subTest(stem=stem), tempfile.TemporaryDirectory() as temp:
                batch = Path(temp)
                (batch / 'videos').mkdir()
                (batch / 'videos' / (stem + '.mp4')).write_bytes(b'source video')
                args = types.SimpleNamespace(batch_root=batch, input_mode='videos',
                    work_root=batch / 'work', clip_model='test-model', clip_batch_size=64,
                    ocr_batch_size=32, ocr_enhance=0, interval=1.0, threshold=4)
                calls = []
                def fake_extract(video, root, interval):
                    folder = root / 'keyframes' / video.stem
                    folder.mkdir(parents=True)
                    (folder / '0001.jpg').write_bytes(b'frame')
                    (root / 'map-keyframes').mkdir()
                    (root / 'map-keyframes' / (video.stem + '.csv')).write_text(
                        'n,pts_time,fps,frame_idx\n1,0,30,0\n')
                    return [{'n':1}]
                def fake_run(script, env, *options):
                    calls.append(script)
                    self.assertEqual(env['OCR_WHITE_ON_BLACK_ONLY'], white_only)
                    self.assertEqual(env['OCR_ENHANCE'], '0')
                    self.assertEqual(env['OCR_ALL_FRAMES'], '1')
                    sqlite3.connect(Path(env['DATA_ROOT']) / 'meta.db').close()
                fake_verify = types.SimpleNamespace(verify_bundle=lambda *a: None)
                with patch.dict(os.environ), patch.dict('sys.modules', {'run_new_dataset_pipeline':fake_verify}), \
                     patch.object(runner, 'extract_frames', fake_extract), \
                     patch.object(runner, 'dedup_frames', return_value=1), patch.object(runner, 'run', fake_run):
                    self.assertEqual(runner.process_archive(args, group, {'version':'test'}), [stem])
                self.assertEqual(calls, ['ocr_index.py', 'prepare_serve_bundle.py'])
                with zipfile.ZipFile(batch / 'processed_data' / (Path(group).stem + '_processed.zip')) as z:
                    self.assertEqual(json.loads(z.read('asr_cache.json')), {})
                    self.assertEqual(json.loads(z.read('batch3_receipt.json'))['asr_status'],
                                     'delegated_not_processed')

    def test_batch_yolo_preserves_order_boxes_and_classes(self):
        # Exercise the actual function in isolation from unavailable ML runtimes.
        tree = ast.parse(Path(__file__).with_name('shared.py').read_text(encoding='utf-8'))
        func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'detect_objects_batch')
        class Array:
            def __init__(self, value): self.value = value
            def tolist(self): return self.value
        image = types.SimpleNamespace(shape=(100, 200, 3))
        result = types.SimpleNamespace(names={0:'person'}, boxes=types.SimpleNamespace(
            xyxy=Array([[20, 10, 100, 50]]), cls=Array([0]), conf=Array([0.91234])))
        calls = []
        def predict(images, **kwargs):
            calls.append(len(images))
            return [result] * len(images)
        namespace = {'ImageLike': object, 'load_image': lambda x: (None, image),
            'get_yolo': lambda: predict, 'YOLO_IMGSZ': 640,
            'class_to_category': lambda x: 'human', '_looks_like_text_region': lambda x: True}
        exec(compile(ast.Module(body=[func], type_ignores=[]), '<batch_yolo>', 'exec'), namespace)
        output = namespace['detect_objects_batch'](['a', 'b'])
        self.assertEqual(calls, [2])
        self.assertEqual(len(output), 2)
        self.assertEqual(output[0][0][0]['box'], [0.1, 0.1, 0.5, 0.5])
        self.assertEqual(output[0][0][0]['conf'], 0.912)
        self.assertTrue(output[1][1])


if __name__ == '__main__':
    unittest.main()
