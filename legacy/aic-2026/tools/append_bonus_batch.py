#!/usr/bin/env python3
"""
append_bonus_batch.py — Incremental merge of bonus_batch1 into an existing merged dataset
without rebuilding FAISS from scratch.
"""

import argparse
import json
import sqlite3
from pathlib import Path
import numpy as np

def append_shard(target_dir: Path, bonus_dir: Path, backup_index: bool = True):
    target_dir = target_dir.resolve()
    bonus_dir = bonus_dir.resolve()

    if not target_dir.is_dir():
        raise FileNotFoundError(f"Target directory does not exist: {target_dir}")
    if not bonus_dir.is_dir():
        raise FileNotFoundError(f"Bonus directory does not exist: {bonus_dir}")

    print(f"=== Đang gộp bổ sung từ {bonus_dir.name} vào {target_dir} ===")

    # 1. Symlink keyframes và map-keyframes
    bonus_kf = bonus_dir / "keyframes"
    target_kf = target_dir / "keyframes"
    if bonus_kf.is_dir():
        target_kf.mkdir(parents=True, exist_ok=True)
        count_kf = 0
        for item in bonus_kf.iterdir():
            dest = target_kf / item.name
            if not dest.exists():
                try:
                    dest.symlink_to(item.resolve())
                    count_kf += 1
                except OSError as e:
                    print(f"Warning symlink {dest}: {e}")
        print(f"  [1/5] Đã liên kết {count_kf} thư mục keyframes mới.")

    bonus_maps = bonus_dir / "map-keyframes"
    target_maps = target_dir / "map-keyframes"
    if bonus_maps.is_dir():
        target_maps.mkdir(parents=True, exist_ok=True)
        count_maps = 0
        for item in bonus_maps.glob("*.csv"):
            dest = target_maps / item.name
            if not dest.exists():
                try:
                    dest.symlink_to(item.resolve())
                    count_maps += 1
                except OSError as e:
                    print(f"Warning symlink {dest}: {e}")
        print(f"  [2/5] Đã liên kết {count_maps} file map-keyframes CSV mới.")

    # 2. Gộp SQLite metadata
    target_db = target_dir / "meta.db"
    bonus_db = bonus_dir / "meta.db"
    if bonus_db.is_file() and target_db.is_file():
        conn = sqlite3.connect(target_db)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(frames)").fetchall()]
        cols_str = ", ".join(cols)

        cursor.execute("ATTACH DATABASE ? AS bonus_db", (str(bonus_db.resolve()),))
        cursor.execute(f"""
            INSERT OR IGNORE INTO frames ({cols_str})
            SELECT {cols_str} FROM bonus_db.frames
        """)
        changes = conn.total_changes
        conn.commit()
        cursor.execute("DETACH DATABASE bonus_db")
        conn.close()
        print(f"  [3/5] Đã nạp metadata SQLite (thêm mới/cập nhật: {changes} dòng).")

    # 3. Gộp trực tiếp FAISS vectors (Add incrementally)
    import faiss
    target_index_path = target_dir / "faiss_index" / "frames.faiss"
    bonus_index_path = bonus_dir / "faiss_index" / "frames.faiss"
    target_meta_path = target_dir / "faiss_index" / "frames_meta.json"
    bonus_meta_path = bonus_dir / "faiss_index" / "frames_meta.json"

    if bonus_index_path.is_file() and target_index_path.is_file():
        if backup_index:
            bak_path = target_dir / "faiss_index" / "frames.faiss.bak"
            if not bak_path.exists():
                import shutil
                shutil.copy2(target_index_path, bak_path)

        target_index = faiss.read_index(str(target_index_path))
        bonus_index = faiss.read_index(str(bonus_index_path))

        bonus_meta = json.loads(bonus_meta_path.read_text(encoding="utf-8"))
        target_meta = json.loads(target_meta_path.read_text(encoding="utf-8"))

        new_items = []
        for fid, meta in bonus_meta.get("fid_to_meta", {}).items():
            if fid not in target_meta.get("fid_to_meta", {}):
                new_items.append((fid, meta))

        if new_items:
            print(f"  [4/5] Đang nạp {len(new_items)} vector mới vào FAISS index...")
            vectors = []
            faiss_ids = []
            for fid, meta in new_items:
                fid_int = int(meta["faiss_id"])
                try:
                    vec = bonus_index.reconstruct(fid_int)
                except Exception:
                    id_array = faiss.vector_to_array(bonus_index.id_map)
                    pos = np.where(id_array == fid_int)[0]
                    vec = bonus_index.index.reconstruct(int(pos[0]))
                vectors.append(vec)
                faiss_ids.append(fid_int)

            vecs_np = np.asarray(vectors, dtype=np.float32)
            ids_np = np.asarray(faiss_ids, dtype=np.int64)
            target_index.add_with_ids(vecs_np, ids_np)
            faiss.write_index(target_index, str(target_index_path))

            target_meta["fid_to_meta"].update({fid: meta for fid, meta in new_items})
            target_meta_path.write_text(json.dumps(target_meta, ensure_ascii=False), encoding="utf-8")
            print(f"  ✓ Đã cập nhật FAISS (Tổng số vector hiện tại: {target_index.ntotal}).")
        else:
            print("  [4/5] Các vector trong bonus_batch1 đã có sẵn trong FAISS.")

    # 4. Gộp ASR cache nếu có
    bonus_asr = bonus_dir / "asr_cache.json"
    target_asr = target_dir / "asr_cache.json"
    if bonus_asr.is_file():
        b_data = json.loads(bonus_asr.read_text(encoding="utf-8"))
        t_data = json.loads(target_asr.read_text(encoding="utf-8")) if target_asr.is_file() else {}
        t_data.update(b_data)
        target_asr.write_text(json.dumps(t_data, ensure_ascii=False), encoding="utf-8")
        print("  ✓ Đã cập nhật file asr_cache.json.")

    # 5. Tái tạo Manifest và FTS5
    print("  [5/5] Đang cập nhật serve_manifest, FTS5 và thống kê...")
    from prepare_serve_bundle import prepare_dataset
    prepare_dataset(target_dir, strict_ocr=False)
    print("\n🎉 HOÀN TẤT: Đã gộp thành công bonus_batch1 vào dataset sẵn sàng thi đấu!")

def main():
    parser = argparse.ArgumentParser(description="Gộp bổ sung shard mới vào bản merge có sẵn mà không cần build lại FAISS từ đầu.")
    parser.add_argument(
        "--target",
        default="data/search_index",
        help="Đường dẫn thư mục bản merge hiện có (ví dụ: data/search_index hoặc data/merged_data)",
    )
    parser.add_argument(
        "--bonus",
        default="/mnt/d/AIC/data/processed/bonus_batch1",
        help="Đường dẫn thư mục shard cần gộp thêm",
    )
    args = parser.parse_args()
    append_shard(Path(args.target), Path(args.bonus))

if __name__ == "__main__":
    main()
