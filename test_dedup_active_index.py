"""Small real SQLite/FAISS fixtures; never opens the user's dataset or Redis."""
import json
import os
from pathlib import Path
import socket
import sqlite3
import subprocess
import tempfile
import time
import unittest

import faiss
import numpy as np
from PIL import Image

import dedup_active_index as job


class DedupTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("AIC_TEST_REDIS_BIN"), "Set AIC_TEST_REDIS_BIN to test a private Redis Stack process")
    def test_redis_prunes_and_preserves_other_data(self):
        import redis
        from migrate_to_redis import create_index, key, wait_index
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", 0))
                port = sock.getsockname()[1]
            binary = Path(os.environ["AIC_TEST_REDIS_BIN"])
            module = binary.parent.parent / "lib" / "redis" / "modules" / "redisearch.so"
            process = subprocess.Popen([str(binary), "--bind", "127.0.0.1", "--port", str(port),
                                        "--dir", str(root), "--save", "", "--appendonly", "no",
                                        "--loadmodule", str(module)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            client = redis.Redis(host="127.0.0.1", port=port, socket_timeout=5, protocol=2)
            try:
                for _ in range(100):
                    try:
                        client.ping()
                        break
                    except redis.ConnectionError:
                        time.sleep(.05)
                client.ping()
                prefix = "test:dedup:"
                for kind in ("frame", "asr", "summary"):
                    create_index(client, prefix, kind)
                    client.set(prefix + kind + "_count", 0)
                files = [f"N001-V001/{i:03}.jpg" for i in range(1, 4)]
                for filename in files:
                    client.hset(key(prefix, "frame", filename), mapping={"file": filename, "text": "example", "payload": "{}"})
                client.set(prefix + "frame_count", 3)
                wait_index(client, prefix, "frame", 3)
                client.set("unrelated:key", "keep")
                client.set(prefix + "asr_cache", '{"keep":true}')
                target = root / "filtered"
                (target / "map-keyframes").mkdir(parents=True)
                (target / "map-keyframes" / "N001-V001.csv").write_text("n,pts_time,frame_idx\n1,0,0\n3,2,50\n")
                job.write_json(target / "serve_stats.json", {"frames": 2})
                with sqlite3.connect(target / "meta.db") as db:
                    db.execute("CREATE TABLE frames(file TEXT,ocr_joined TEXT,ocr_status TEXT)")
                    db.executemany("INSERT INTO frames VALUES(?,'example','done')", [(files[0],), (files[2],)])
                manifest = {"source_index": str(root / "original"), "frame_vectors": False, "ready": True}
                selected = [["id2", files[1], 2]]
                job.refresh_redis(client, prefix, manifest, target, selected)
                # A second run models interruption before the phase checkpoint.
                job.refresh_redis(client, prefix, manifest, target, selected)
                self.assertFalse(client.exists(key(prefix, "frame", files[1])))
                self.assertEqual(client.lrange(prefix + "files", 0, -1), [files[0].encode(), files[2].encode()])
                timeline = json.loads(client.get(prefix + "timeline:N001-V001"))
                self.assertEqual([f["n"] for f in timeline], [1, 3])
                self.assertEqual(client.get("unrelated:key"), b"keep")
                self.assertEqual(client.get(prefix + "asr_cache"), b'{"keep":true}')
                self.assertEqual(json.loads(client.get(prefix + "manifest"))["source_index"], str(target))
                self.assertTrue((root / "dump.rdb").is_file())
            finally:
                process.terminate()
                process.wait(timeout=10)

    def test_adjacent_chain(self):
        self.assertEqual(job.removal_indices([0, 3, 6, None, 9, 9], 3), [1, 2, 5])

    def test_plan_resume_and_changed_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "keyframes"
            folder = root / "N001-V001"
            folder.mkdir(parents=True)
            work = Path(tmp) / "work"
            work.mkdir()
            for n in range(1, 4):
                Image.new("RGB", (32, 32), "red").save(folder / f"{n:03}.jpg")
            plans = job.make_plan(root, work, "cpu", 1, 8, 4)
            self.assertEqual(plans[0]["remove"], ["002.jpg", "003.jpg"])
            self.assertEqual(len(list(folder.glob("*.jpg"))), 3)
            (folder / "002.jpg").unlink()
            job.validate_images(root, plans, allow_deleted=True)
            with self.assertRaises(RuntimeError):
                job.validate_images(root, plans)
            Image.new("RGB", (32, 32), "blue").save(folder / "001.jpg")
            with self.assertRaises(RuntimeError):
                job.validate_images(root, plans, allow_deleted=True)

    def test_filter_real_hnsw_sqlite_maps_and_other_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            source, target, root = base / "source", base / "filtered", base / "extracted" / "keyframes"
            source.mkdir()
            root.mkdir(parents=True)
            (source / "keyframes").mkdir()
            (source / "clip_text").mkdir()
            maps = base / "extracted" / "map-keyframes"
            maps.mkdir()
            maps.joinpath("N001-V001.csv").write_text("n,pts_time,frame_idx\n1,0,0\n2,1,25\n3,2,50\n")
            other_maps = base / "other_maps"
            other_maps.mkdir()
            other_maps.joinpath("N002-V001.csv").write_text("n,pts_time,frame_idx\n2,1,25\n")
            job.write_json(source / "external_storage.json", {
                "sources": [{"id": "new", "data_root": str(root.parent), "keyframes_dir": str(root), "map_dir": str(maps)},
                            {"id": "old", "data_root": str(base / "other"), "map_dir": str(other_maps)}],
                "video_sources": {"n001-v001": "new", "n002-v001": "old"}})
            job.write_json(source / "serve_manifest.json", {"clip_model": "fixture"})
            job.write_json(source / "asr_cache.json", {})
            files = ["N001-V001/001.jpg", "N001-V001/002.jpg", "N001-V001/003.jpg", "N002-V001/002.jpg"]
            (source / "summary_index").mkdir()
            with sqlite3.connect(source / "summary_index" / "summaries.db") as db:
                db.execute("CREATE TABLE frames(file TEXT)")
                db.executemany("INSERT INTO frames VALUES(?)", [(f,) for f in files])
                db.execute("CREATE TABLE docs(id INTEGER, text TEXT)")
                db.execute("INSERT INTO docs VALUES(1, 'keep video summary')")
            with sqlite3.connect(source / "meta.db") as db:
                db.execute("CREATE TABLE frames(id TEXT PRIMARY KEY, file TEXT, mtime REAL, objects TEXT DEFAULT '[]', categories TEXT DEFAULT '[]', colors TEXT DEFAULT '[]', has_text INTEGER DEFAULT 1, ocr_joined TEXT, ocr_ascii TEXT, faiss_id INTEGER)")
                db.executemany("INSERT INTO frames(id,file,mtime,ocr_joined,ocr_ascii,faiss_id) VALUES(?,?,0,?,?,?)",
                               [(str(i), f, "deletedword" if i == 1 else "keptword", "deletedword" if i == 1 else "keptword", i + 10) for i, f in enumerate(files)])
            vectors = np.eye(4, dtype="float32")
            index = faiss.IndexIDMap2(faiss.IndexHNSWFlat(4, 8, faiss.METRIC_INNER_PRODUCT))
            index.add_with_ids(vectors, np.arange(10, 14, dtype="int64"))
            (source / "faiss_index").mkdir()
            faiss.write_index(index, str(source / "faiss_index" / "frames.faiss"))
            job.write_json(source / "faiss_index" / "frames_meta.json", {"dim": 4, "fid_to_meta": {
                str(i): {"file": f, "faiss_id": i + 10, "mtime": 0} for i, f in enumerate(files)}})
            plans = [{"folder": "N001-V001", "remove": ["002.jpg"], "files": {}}]
            selected = job.selected_files(source, root, plans)
            self.assertEqual(selected, [["1", files[1], 11]])
            job.build_filtered(source, target, root, plans, selected)
            # A retry replaces the staged DB/index safely, without deleting more rows.
            job.build_filtered(source, target, root, plans, selected)
            self.assertEqual(job.read_json(target / "serve_manifest.json")["clip_text_artifact"], "clip_text")
            with sqlite3.connect(target / "meta.db") as db:
                self.assertEqual(db.execute("select count(*) from frames").fetchone()[0], 3)
                self.assertEqual(db.execute("select count(*) from frames_fts where frames_fts match 'deletedword'").fetchone()[0], 0)
                self.assertEqual(db.execute("select count(*) from frame_locator where file=?", (files[1],)).fetchone()[0], 0)
            rebuilt = faiss.read_index(str(target / "faiss_index" / "frames.faiss"))
            self.assertEqual(set(faiss.vector_to_array(rebuilt.id_map)), {10, 12, 13})
            np.testing.assert_array_equal(rebuilt.reconstruct(12), vectors[2])
            self.assertNotIn("2,1,25", (target / "map-keyframes" / "source_0" / "N001-V001.csv").read_text())
            self.assertIn("2,1,25", (target / "map-keyframes" / "source_1" / "N002-V001.csv").read_text())
            self.assertIn("2,1,25", maps.joinpath("N001-V001.csv").read_text())
            with sqlite3.connect(target / "summary_index" / "summaries.db") as db:
                self.assertEqual(db.execute("SELECT count(*) FROM frames").fetchone()[0], 3)
                self.assertEqual(db.execute("SELECT text FROM docs").fetchone()[0], "keep video summary")
            with sqlite3.connect(source / "meta.db") as db:
                self.assertEqual(db.execute("select count(*) from frames").fetchone()[0], 4)
            job.build_filtered(source, target, root, plans, selected, faiss_mode="flat")
            flat = faiss.read_index(str(target / "faiss_index" / "frames.faiss"))
            self.assertIsInstance(faiss.downcast_index(flat.index), faiss.IndexFlat)
            np.testing.assert_array_equal(flat.reconstruct(12), vectors[2])
            scores, found = flat.search(vectors[2:3], 3)
            self.assertEqual(found[0, 0], 12)
            self.assertNotIn(11, found[0])
            # Existing-only mode reads names, prunes missing references (including
            # the summary's separate frame table), and never removes real images.
            video_folder = root / "N001-V001"
            video_folder.mkdir()
            for name in ["001.jpg", "003.jpg"]:
                (video_folder / name).write_bytes(b"not an image: must never be decoded")
            inventory = job.existing_plan(root)
            job.validate_images(root, inventory)
            missing = job.selected_files(source, root, inventory)
            self.assertEqual(missing, selected)
            job.build_filtered(source, target, root, inventory, missing, faiss_mode="flat")
            self.assertEqual({p.name for p in video_folder.iterdir()}, {"001.jpg", "003.jpg"})
            self.assertNotIn("2,1,25", (target / "map-keyframes" / "source_0" / "N001-V001.csv").read_text())
            self.assertIn("2,1,25", maps.joinpath("N001-V001.csv").read_text())
            with sqlite3.connect(target / "summary_index" / "summaries.db") as db:
                self.assertEqual(db.execute("SELECT count(*) FROM frames").fetchone()[0], 3)
            (video_folder / "004.jpg").write_bytes(b"new")
            with self.assertRaises(RuntimeError):
                job.validate_images(root, inventory)

    def test_existing_only_refuses_empty_or_absent_storage(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(RuntimeError):
                job.existing_plan(Path(tmp))
            with self.assertRaises(OSError):
                job.existing_plan(Path(tmp) / "not-mounted")

    def test_delete_resumes_without_renumbering(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            root, maps, work = base / "keyframes", base / "map-keyframes", base / "work"
            (root / "N001-V001").mkdir(parents=True)
            maps.mkdir()
            for name in ["001.jpg", "002.jpg", "003.jpg"]:
                (root / "N001-V001" / name).write_bytes(b"fixture")
            path = maps / "N001-V001.csv"
            path.write_text("n,pts_time,frame_idx\n1,0,0\n2,1,25\n3,2,50\n")
            plans = [{"folder": "N001-V001", "remove": ["002.jpg"]}]
            job.delete_sources(root, maps, plans, work)
            job.delete_sources(root, maps, plans, work)
            self.assertEqual({p.name for p in (root / "N001-V001").iterdir()}, {"001.jpg", "003.jpg"})
            self.assertIn("3,2,50", path.read_text())
            self.assertIn("2,1,25", (work / "maps_before" / path.name).read_text())


if __name__ == "__main__":
    unittest.main()
