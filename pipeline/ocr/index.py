"""
ocr_index.py — Image / OCR indexing
====================================
Scans every image in keyframes/, runs YOLO detection, CLIP embedding, color
extraction, and PaddleOCR text extraction, and stores everything (including
each image's OCR text) into meta.db (SQLite) and faiss_index/ (FAISS).

This is the exact same indexing logic that used to live in app.py — only
moved here unchanged. main.py imports run_ocr / build_index / etc. from this
file so the on-demand routes (/frame/ocr, /index/run) keep working.

Run standalone to (re)index everything in keyframes/:
    python ocr_index.py
"""

import hashlib
import json
from collections import OrderedDict
from contextlib import contextmanager
import os
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Optional, Union

import numpy as np
from tqdm import tqdm

try:
    import cv2
except ImportError:  # Keep the module importable even before requirements are installed.
    cv2 = None

from shared import (
    DATA_ROOT, KEYFRAMES_DIR, META_DB, BATCH_SIZE, OCR_ENGINE, OCR_ALL_FRAMES, OCR_BATCH_SIZE,
    OCR_PIPELINE_VERSION, OCR_ENHANCE, OCR_REGION_LIMIT,
    log, get_yolo, get_ocr, _ocr_lock,
    unload_clip, unload_yolo, unload_ocr, YOLO_IMGSZ,
    get_collection, get_db, get_existing_mtimes, upsert_frame,
    detect_objects, extract_dominant_colors,
    embed_image_batch,
    is_valid_image,
    cache_update_ocr, join_ocr_texts,
    flush_vector_store,
    _index_progress, _index_lock,
)

from ocr_quality import clean_ocr_text, refine_regions

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
OCR_COMMIT_EVERY = 100
SQLITE_BUSY_TIMEOUT_MS = 30_000
VALIDATE_IMAGES = os.getenv("VALIDATE_IMAGES", "0").lower() in {"1", "true", "yes"}
VERIFY_FAISS = os.getenv(
    "VERIFY_FAISS", os.getenv("VERIFY_CHROMA", "0")
).lower() in {"1", "true", "yes"}
VECTOR_CHECK_BATCH = 4096
OCR_MAX_SIDE = max(0, int(os.getenv("OCR_MAX_SIDE", "0")))
OCR_PREFETCH_WORKERS = max(1, int(os.getenv("OCR_PREFETCH_WORKERS", "4")))
OCR_RETRY_ERRORS = os.getenv("OCR_RETRY_ERRORS", "1").lower() in {"1", "true", "yes"}
OCR_MAX_ATTEMPTS = max(1, int(os.getenv("OCR_MAX_ATTEMPTS", "3")))
OCR_DEDUP_CACHE_SIZE = max(0, int(os.getenv("OCR_DEDUP_CACHE_SIZE", "4096")))

# ══════════════════════════════════════════════════════════════════════════════
# OCR
# ══════════════════════════════════════════════════════════════════════════════

def _frame_id(rel: str) -> str:
    return rel.replace("\\", "/").replace("/", "__").replace(".", "_")


def _parse_easyocr_items(result, w: int, h: int) -> list:
    texts = []
    for item in result or []:
        try:
            box_pts, text, conf = item
            xs = [p[0] for p in box_pts]
            ys = [p[1] for p in box_pts]
            clean = clean_ocr_text(text)
            if not clean:
                continue
            texts.append({
                "text": clean,
                "conf": round(float(conf), 3),
                "box": [round(min(xs)/w,4), round(min(ys)/h,4), round(max(xs)/w,4), round(max(ys)/h,4)],
            })
        except Exception:
            continue
    return texts


def _vector_existing_ids(col, ids: list[str]) -> set[str]:
    """
    Return the subset of ids already present in the vector store. This makes
    the CLIP pass resumable after shutdowns between FAISS and SQLite writes.
    """
    found: set[str] = set()
    for start in range(0, len(ids), VECTOR_CHECK_BATCH):
        chunk = ids[start:start + VECTOR_CHECK_BATCH]
        if not chunk:
            continue
        try:
            got = col.get(ids=chunk)
            found.update(got.get("ids") or [])
        except Exception as e:
            log.warning(f"Could not check vector ids at {start}: {e}")
    return found

def _read_image(path: Path) -> Optional[np.ndarray]:
    """
    Read one image with OpenCV. PaddleOCR accepts OpenCV BGR arrays directly,
    which avoids letting PaddleOCR open/decode the same file again.
    """
    if cv2 is None:
        return None
    try:
        return cv2.imread(str(path), cv2.IMREAD_COLOR)
    except Exception as e:
        log.warning(f"OpenCV could not read {path.name}: {e}")
        return None


def _prepare_ocr_image(image: np.ndarray) -> np.ndarray:
    if OCR_MAX_SIDE <= 0:
        return image
    h, w = image.shape[:2]
    longest = max(h, w)
    if longest <= OCR_MAX_SIDE:
        return image
    scale = OCR_MAX_SIDE / float(longest)
    return cv2.resize(
        image,
        (max(1, round(w * scale)), max(1, round(h * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _ocr_content_hash(image: np.ndarray) -> str:
    """Exact prepared-pixel hash; reuse is lossless, never perceptual/approximate."""
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.blake2b(digest_size=16)
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(memoryview(contiguous))
    return digest.hexdigest()


def _read_prepared_ocr_image(path: Path) -> Optional[np.ndarray]:
    """Decode ahead; retain original pixels only when region refinement needs them."""
    image = _read_image(path)
    return image if OCR_ENHANCE or image is None else _prepare_ocr_image(image)


def _resolve_ocr_paths(rels):
    # Resolve the central catalog once per batch, not once per frame.
    from external_assets import ExternalAssetResolver
    resolver = None
    catalog_loaded = False
    paths = []
    for rel in rels:
        path = KEYFRAMES_DIR / rel
        if not path.is_file():
            if not catalog_loaded:
                resolver = ExternalAssetResolver.load(DATA_ROOT)
                catalog_loaded = True
            path = (resolver.resolve_image(rel) if resolver else None) or path
        paths.append(path)
    return paths


def _submit_image_reads(
    executor: ThreadPoolExecutor, paths: list[Path]
) -> list:
    return [executor.submit(_read_prepared_ocr_image, path) for path in paths]


@contextmanager
def _ocr_process_lock():
    """Prevent two processes from claiming and OCR'ing the same database rows."""
    lock_path = META_DB.with_suffix(META_DB.suffix + ".ocr.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+", encoding="utf-8")
    try:
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(
                f"Another OCR process is already using {META_DB}"
            ) from exc
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        yield
    finally:
        try:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        except Exception:
            pass
        handle.close()


def _easyocr_read_group(ocr, items: list[dict]) -> list[tuple[list, Optional[str]]]:
    """Run a same-shape group; recursively isolate failures instead of losing a batch."""
    if not items:
        return []
    if len(items) == 1:
        item = items[0]
        try:
            with _ocr_lock:
                result = ocr.readtext(item["source"], detail=1, paragraph=False)
            return [(_parse_easyocr_items(result, item["w"], item["h"]), None)]
        except Exception as exc:
            return [([], str(exc))]

    try:
        with _ocr_lock:
            results = ocr.readtext_batched(
                [item["source"] for item in items],
                detail=1,
                paragraph=False,
                batch_size=min(OCR_BATCH_SIZE, len(items)),
            )
        if len(results) != len(items):
            raise RuntimeError("EasyOCR returned a mismatched batch size")
        return [
            (_parse_easyocr_items(result, item["w"], item["h"]), None)
            for result, item in zip(results, items)
        ]
    except Exception as exc:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        middle = len(items) // 2
        log.warning(
            "EasyOCR batch of %d failed (%s); splitting into %d + %d",
            len(items), exc, middle, len(items) - middle,
        )
        return (
            _easyocr_read_group(ocr, items[:middle])
            + _easyocr_read_group(ocr, items[middle:])
        )


def _image_size(path: Path, image: Optional[np.ndarray] = None) -> tuple[int, int]:
    """Return (width, height), preferring an already decoded image."""
    if image is not None:
        h, w = image.shape[:2]
        return int(w), int(h)

    image = _read_image(path)
    if image is not None:
        h, w = image.shape[:2]
        return int(w), int(h)

    # Last-resort fallback for environments without cv2.
    from PIL import Image
    with Image.open(path) as img:
        return img.size


def _configure_conn(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Small SQLite pragmas that help large indexing batches stay responsive."""
    try:
        conn.execute(f"PRAGMA busy_timeout={SQLITE_BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA temp_store=MEMORY")
        conn.execute("PRAGMA cache_size=-100000")
    except Exception:
        pass
    return conn


def _commit_with_reopen(conn: sqlite3.Connection, context: str) -> sqlite3.Connection:
    try:
        conn.commit()
        return conn
    except sqlite3.DatabaseError as e:
        log.error(f"DB commit failed during {context}: {e} — reopening")
        try:
            conn.close()
        except Exception:
            pass
        return _configure_conn(get_db())


def run_ocr(
    source: Union[Path, np.ndarray],
    w: int,
    h: int,
    *,
    raise_errors: bool = False,
    enhance: bool = True,
) -> list:
    texts = []
    if OCR_ENGINE in {"none", "off", "0", "false", "no"}:
        return texts
    try:
        ocr = get_ocr()
        source_name = source.name if isinstance(source, Path) else "image"
        image = _read_image(source) if isinstance(source, Path) else source
        if image is None:
            raise RuntimeError(f"Cannot decode OCR source: {source_name}")
        ocr_source = _prepare_ocr_image(image)
        h, w = ocr_source.shape[:2]

        if OCR_ENGINE == "easyocr":
            with _ocr_lock:
                result = ocr.readtext(ocr_source, detail=1, paragraph=False)
            texts = _parse_easyocr_items(result, w, h)
            return _refine_ocr(image, texts) if enhance and OCR_ENHANCE else texts

        with _ocr_lock:
            try:
                result = ocr.ocr(ocr_source, cls=True)
            except TypeError:
                result = ocr.ocr(ocr_source)
            except Exception as e:
                if "Unknown argument" in str(e) or "unexpected keyword" in str(e):
                    result = ocr.ocr(ocr_source)
                else:
                    raise

        # PaddleOCR can return None, [[]], [None], or a valid list depending
        # on the image — guard every possible failure mode before iterating.
        if result is None:
            return texts
        if not isinstance(result, list) or len(result) == 0:
            return texts
        if result[0] is None or not isinstance(result[0], list) or len(result[0]) == 0:
            return texts

        for line in result[0]:
            try:
                if line is None:
                    continue
                box_pts, (text, conf) = line
                xs = [p[0] for p in box_pts]
                ys = [p[1] for p in box_pts]
                texts.append({
                    "text": clean_ocr_text(text),
                    "conf": round(float(conf), 3),
                    "box":  [round(min(xs)/w,4), round(min(ys)/h,4), round(max(xs)/w,4), round(max(ys)/h,4)],
                })
            except Exception:
                continue

    except Exception as e:
        source_name = source.name if isinstance(source, Path) else "image"
        if raise_errors:
            raise
        log.warning(f"OCR skipped {source_name}: {e}")
        return texts

    return _refine_ocr(image, texts) if enhance and OCR_ENHANCE else texts

# ══════════════════════════════════════════════════════════════════════════════
# Indexing
# ══════════════════════════════════════════════════════════════════════════════

def _refine_ocr(image, texts):
    return refine_regions(image, texts, lambda crop: run_ocr(
        crop, crop.shape[1], crop.shape[0], raise_errors=True, enhance=False
    ), max_regions=OCR_REGION_LIMIT)


def _write_ocr_outcome(
    conn: sqlite3.Connection,
    rel: str,
    *,
    texts: Optional[list] = None,
    error: Optional[str] = None,
    content_hash: str = "",
    reused_from: Optional[str] = None,
) -> None:
    now = time.time()
    if error is not None:
        conn.execute(
            """
            UPDATE frames SET
                ocr_status='error',
                ocr_version=?,
                ocr_content_hash=?,
                ocr_error=?,
                ocr_updated_at=?
            WHERE file=?
            """,
            (OCR_PIPELINE_VERSION, content_hash, str(error)[:1000], now, rel),
        )
        return

    texts = texts or []
    joined, ascii_text = join_ocr_texts(texts)
    status = "reused" if reused_from else ("done" if texts else "no_text")
    conn.execute(
        """
        UPDATE frames SET
            has_text=?,
            ocr_texts=?,
            ocr_joined=?,
            ocr_ascii=?,
            ocr_status=?,
            ocr_version=?,
            ocr_source_file=?,
            ocr_content_hash=?,
            ocr_error='',
            ocr_updated_at=?
        WHERE file=?
        """,
        (
            int(bool(texts)),
            json.dumps(texts),
            joined,
            ascii_text,
            status,
            OCR_PIPELINE_VERSION,
            reused_from or rel,
            content_hash,
            now,
            rel,
        ),
    )
    cache_update_ocr(rel, texts, joined, ascii_text)


def backfill_ocr():
    with _ocr_process_lock():
        return _backfill_ocr_locked()


def _backfill_ocr_locked():
    """Fill explicit pending OCR states; safe to resume and merge across machines."""
    if OCR_ENGINE in {"none", "off", "0", "false", "no"}:
        log.info("OCR pre-warm skipped: OCR_ENGINE=none.")
        return

    with get_db() as db:
        db.execute(
            "UPDATE frames SET ocr_status='pending' "
            "WHERE ocr_status='processing'"
        )
        statuses = ["pending"]
        if OCR_ALL_FRAMES:
            statuses.append("skipped")
        if OCR_RETRY_ERRORS:
            statuses.append("error")
        placeholders = ",".join("?" for _ in statuses)
        rows = db.execute(
            f"""
            SELECT file, width, height
            FROM frames
            WHERE ocr_status IN ({placeholders})
              AND COALESCE(ocr_attempts,0) < ?
            ORDER BY file
            """,
            (*statuses, OCR_MAX_ATTEMPTS),
        ).fetchall()

    if not rows:
        log.info("OCR pre-warm: nothing to do, all selected frames are terminal.")
        return

    log.info(
        "Running OCR on %d frame(s), batch=%d, max_side=%s, workers=%d...",
        len(rows),
        OCR_BATCH_SIZE,
        OCR_MAX_SIDE or "original",
        OCR_PREFETCH_WORKERS,
    )
    ocr = get_ocr()
    with _index_lock:
        _index_progress.update(
            stage="ocr_prewarm",
            running=True,
            done=0,
            total=len(rows),
            started_at=time.time(),
        )

    conn = _configure_conn(get_db())
    bar = tqdm(
        total=len(rows),
        desc=f"OCR all frames (batch={OCR_BATCH_SIZE})",
        unit="frame",
    )
    written_since_commit = 0
    executor = ThreadPoolExecutor(
        max_workers=OCR_PREFETCH_WORKERS,
        thread_name_prefix="ocr-decode",
    )
    pending_reads = None
    recent_hashes: OrderedDict[str, tuple[list, str]] = OrderedDict()
    try:
        for start in range(0, len(rows), OCR_BATCH_SIZE):
            batch_rows = rows[start:start + OCR_BATCH_SIZE]
            rels = [str(row["file"]) for row in batch_rows]
            paths = _resolve_ocr_paths(rels)
            conn.executemany(
                """
                UPDATE frames SET
                    ocr_status='processing',
                    ocr_attempts=COALESCE(ocr_attempts,0)+1,
                    ocr_error=''
                WHERE file=?
                """,
                [(rel,) for rel in rels],
            )

            if pending_reads is None:
                pending_reads = _submit_image_reads(executor, paths)
            images = [future.result() for future in pending_reads]
            next_start = start + OCR_BATCH_SIZE
            if next_start < len(rows):
                next_rows = rows[next_start:next_start + OCR_BATCH_SIZE]
                next_rels = [str(row["file"]) for row in next_rows]
                next_paths = _resolve_ocr_paths(next_rels)
                next_reads = _submit_image_reads(executor, next_paths)
            else:
                next_reads = None
            outcomes: dict[str, tuple[list, Optional[str]]] = {}
            content_hashes: dict[str, str] = {}
            reused_from: dict[str, str] = {}
            representatives: dict[str, str] = {}
            duplicates: dict[str, str] = {}
            groups: dict[tuple[int, int], list[dict]] = {}
            for rel, path, image in zip(rels, paths, images):
                if not path.exists():
                    outcomes[rel] = ([], "source image is missing")
                    continue
                if image is None:
                    outcomes[rel] = ([], "OpenCV could not decode source image")
                    continue
                prepared = _prepare_ocr_image(image)
                content_hash = _ocr_content_hash(image)
                content_hashes[rel] = content_hash
                cached = (
                    recent_hashes.get(content_hash)
                    if OCR_DEDUP_CACHE_SIZE else None
                )
                if cached is not None:
                    cached_texts, source_rel = cached
                    outcomes[rel] = (cached_texts, None)
                    reused_from[rel] = source_rel
                    recent_hashes.move_to_end(content_hash)
                    continue
                representative = (
                    representatives.get(content_hash)
                    if OCR_DEDUP_CACHE_SIZE else None
                )
                if representative is not None:
                    duplicates[rel] = representative
                    continue
                if OCR_DEDUP_CACHE_SIZE:
                    representatives[content_hash] = rel
                h, w = prepared.shape[:2]
                groups.setdefault((int(h), int(w)), []).append(
                    {
                        "rel": rel,
                        "source": prepared,
                        "original": image,
                        "w": int(w),
                        "h": int(h),
                        "content_hash": content_hash,
                    }
                )

            for items in groups.values():
                if OCR_ENGINE == "easyocr" and OCR_BATCH_SIZE > 1:
                    results = _easyocr_read_group(ocr, items)
                    if OCR_ENHANCE:
                        refined = []
                        for item, (texts, error) in zip(items, results):
                            try:
                                refined.append((_refine_ocr(item["original"], texts), None)
                                               if error is None else (texts, error))
                            except Exception as exc:
                                refined.append(([], str(exc)))
                        results = refined
                else:
                    results = []
                    for item in items:
                        try:
                            texts = run_ocr(
                                item["original"],
                                item["w"],
                                item["h"],
                                raise_errors=True,
                            )
                            results.append((texts, None))
                        except Exception as exc:
                            results.append(([], str(exc)))
                for item, result in zip(items, results):
                    outcomes[item["rel"]] = result

            for rel, representative in duplicates.items():
                texts, error = outcomes.get(
                    representative, ([], "OCR representative produced no outcome")
                )
                outcomes[rel] = (texts, error)
                if error is None:
                    reused_from[rel] = representative

            if OCR_DEDUP_CACHE_SIZE:
                for content_hash, representative in representatives.items():
                    texts, error = outcomes.get(
                        representative, ([], "OCR representative produced no outcome")
                    )
                    if error is None:
                        recent_hashes[content_hash] = (texts, representative)
                        recent_hashes.move_to_end(content_hash)
                while len(recent_hashes) > OCR_DEDUP_CACHE_SIZE:
                    recent_hashes.popitem(last=False)

            for rel in rels:
                texts, error = outcomes.get(rel, ([], "OCR produced no outcome"))
                try:
                    _write_ocr_outcome(
                        conn,
                        rel,
                        texts=texts,
                        error=error,
                        content_hash=content_hashes.get(rel, ""),
                        reused_from=reused_from.get(rel),
                    )
                except sqlite3.DatabaseError as exc:
                    log.error("DB error on %s: %s; reopening", rel, exc)
                    conn = _commit_with_reopen(conn, "OCR outcome")
                    _write_ocr_outcome(
                        conn,
                        rel,
                        texts=texts,
                        error=error,
                        content_hash=content_hashes.get(rel, ""),
                        reused_from=reused_from.get(rel),
                    )
                if error:
                    log.error("OCR error on %s: %s", rel, error)
                written_since_commit += 1

            if written_since_commit >= OCR_COMMIT_EVERY:
                conn = _commit_with_reopen(conn, "OCR batch")
                written_since_commit = 0

            done = min(start + len(batch_rows), len(rows))
            with _index_lock:
                _index_progress["done"] = done
            bar.update(len(batch_rows))
            pending_reads = next_reads

        conn.commit()
    finally:
        executor.shutdown(wait=True)
        bar.close()
        conn.close()
        with _index_lock:
            _index_progress.update(running=False, stage="idle")
        unload_ocr()

    log.info("OCR pre-warm complete.")

YOLO_BATCH = 32   # frames per YOLO GPU call — lower to 4 if still OOM

def backfill_detections():
    """
    Backfill YOLO detections + colors for frames missing them.
    Passes path strings directly to YOLO in batches — YOLO handles its own
    image loading internally so Python never holds full-res images in RAM.
    Uses imgsz=640 to keep VRAM usage low and stable.
    """
    from shared import class_to_category

    with get_db() as conn:
        rows = conn.execute(
            "SELECT file FROM frames WHERE detections = '[]' OR detections IS NULL"
        ).fetchall()
    if not rows:
        return

    paths, rels = [], []
    for r in rows:
        p = KEYFRAMES_DIR / r["file"]
        if p.exists():
            paths.append(p)
            rels.append(r["file"])

    log.info(f"Backfilling detections for {len(paths)} frames (batch={YOLO_BATCH})...")
    model = get_yolo()

    with _index_lock:
        _index_progress.update(stage="backfill", running=True, done=0, total=len(paths), started_at=time.time())

    conn = _configure_conn(get_db())
    done = 0
    bar  = tqdm(total=len(paths), desc="Backfill detections", unit="frame")
    try:
        for start in range(0, len(paths), YOLO_BATCH):
            batch_paths = paths[start:start + YOLO_BATCH]
            batch_rels  = rels[start:start + YOLO_BATCH]

            # Pass path strings — YOLO loads images internally, no Python RAM spike
            try:
                results = model(
                    [str(p) for p in batch_paths],
                    verbose=False,
                    stream=False,
                    imgsz=YOLO_IMGSZ,
                )
            except Exception as e:
                log.error(f"YOLO batch failed at {start}: {e} — skipping batch")
                done += len(batch_paths)
                bar.update(len(batch_paths))
                with _index_lock:
                    _index_progress["done"] = done
                continue

            for rel, path, r in zip(batch_rels, batch_paths, results):
                try:
                    w_img, h_img = r.orig_shape[1], r.orig_shape[0]
                    detections = []
                    for box, cls_id, conf in zip(
                        r.boxes.xyxy.tolist(),
                        r.boxes.cls.tolist(),
                        r.boxes.conf.tolist()
                    ):
                        cls = r.names[int(cls_id)]
                        x1, y1, x2, y2 = box
                        detections.append({
                            "class":    cls,
                            "category": class_to_category(cls),
                            "conf":     round(float(conf), 3),
                            "box":      [round(x1/w_img,4), round(y1/h_img,4),
                                         round(x2/w_img,4), round(y2/h_img,4)],
                        })
                    colors     = extract_dominant_colors(path)
                    objects    = sorted({d["class"] for d in detections})
                    categories = sorted({d["category"] for d in detections})
                    conn.execute("""
                        UPDATE frames
                        SET detections=?, objects=?, categories=?, colors=?
                        WHERE file=?
                    """, (
                        json.dumps(detections), json.dumps(objects),
                        json.dumps(categories), json.dumps(colors), rel
                    ))
                except Exception as e:
                    log.error(f"Backfill failed for {rel}: {e}")

                done += 1

            # Commit once per batch
            try:
                conn.commit()
            except sqlite3.DatabaseError as e:
                log.error(f"DB commit failed at {start}: {e} — reopening")
                try:
                    conn.close()
                except Exception:
                    pass
                conn = _configure_conn(get_db())

            bar.update(len(batch_paths))
            with _index_lock:
                _index_progress["done"] = done

        conn.commit()
    finally:
        bar.close()
        conn.close()

    log.info("Backfill complete.")
    unload_yolo()
    with _index_lock:
        _index_progress.update(stage="idle", running=False)



def build_index():
    """
    Incrementally index keyframes:
    - Skip already-indexed frames (by file + mtime)
    - Resume CLIP from FAISS if shutdown happened before SQLite metadata
    - CLIP/FAISS pass first, then release CLIP from VRAM
    - YOLO/OCR metadata pass second
    - OCR only on frames flagged as having text
    """
    if not KEYFRAMES_DIR.exists():
        KEYFRAMES_DIR.mkdir(parents=True)
        return

    col    = get_collection()
    images = sorted(p for p in KEYFRAMES_DIR.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    log.info(f"Found {len(images)} images, checking which need indexing...")

    # One bulk query instead of one SELECT per frame — critical at 100k+ scale
    existing = get_existing_mtimes()
    records = []
    metadata_todo = []
    for p in images:
        rel   = str(p.relative_to(KEYFRAMES_DIR))
        mtime = p.stat().st_mtime
        fid   = _frame_id(rel)
        stale = rel in existing and abs(existing[rel] - mtime) >= 0.01
        missing_meta = rel not in existing
        rec = {
            "path": p,
            "rel": rel,
            "mtime": mtime,
            "fid": fid,
            "stale": stale,
            "missing_meta": missing_meta,
        }
        records.append(rec)
        if missing_meta or stale:
            metadata_todo.append(rec)

    vector_candidates = records if VERIFY_FAISS else metadata_todo
    vector_ids = [r["fid"] for r in vector_candidates]
    vector_done = _vector_existing_ids(col, vector_ids) if vector_ids else set()
    clip_todo = [
        r for r in vector_candidates
        if r["stale"] or r["fid"] not in vector_done
    ]

    log.info(
        "Resume status: %s metadata todo, %s CLIP todo, %s CLIP already in FAISS%s.",
        len(metadata_todo),
        len(clip_todo),
        len(vector_done),
        " (verified all images)" if VERIFY_FAISS else "",
    )

    # Optional full validation. Disabled by default for 300k+ images because
    # OpenCV reads are already guarded per batch.
    if metadata_todo and VALIDATE_IMAGES:
        log.info("Validating images before indexing (this catches corrupt files early)...")
        valid, corrupt = [], []
        for r in tqdm(metadata_todo, desc="Validating", unit="img"):
            (valid if is_valid_image(r["path"]) else corrupt).append(r)
        if corrupt:
            log.warning(f"Skipping {len(corrupt)} corrupt/unreadable image(s):")
            for c in corrupt[:50]:
                log.warning(f"  CORRUPT: {c['path']}")
            if len(corrupt) > 50:
                log.warning(f"  ...and {len(corrupt) - 50} more (see corrupt_frames.log)")
            try:
                Path("corrupt_frames.log").write_text(
                    "\n".join(str(c["path"]) for c in corrupt), encoding="utf-8"
                )
            except Exception:
                pass
        valid_ids = {r["fid"] for r in valid}
        metadata_todo = valid
        clip_todo = [r for r in clip_todo if r["fid"] in valid_ids]

    if not clip_todo and not metadata_todo:
        with _index_lock:
            _index_progress.update(stage="idle", running=False, done=0, total=0)
        return

    if clip_todo:
        with _index_lock:
            _index_progress.update(
                stage="clip_index", running=True,
                done=0, total=len(clip_todo), started_at=time.time()
            )

        # Pass 1: CLIP embeddings + FAISS. Keep only CLIP on GPU here.
        bar = tqdm(range(0, len(clip_todo), BATCH_SIZE), desc="CLIP", unit="batch")
        for start in bar:
            batch = clip_todo[start:start + BATCH_SIZE]
            batch_paths = [r["path"] for r in batch]
            bar.set_postfix(frames=f"{start}-{min(start+BATCH_SIZE, len(clip_todo))}")
            batch_images = [_read_image(p) for p in batch_paths]
            batch_sources = [img if img is not None else p for img, p in zip(batch_images, batch_paths)]

            try:
                # CLIP batch embed
                embeddings = embed_image_batch(batch_sources)
            except Exception as e:
                log.warning(f"Batch embed failed ({e}) — falling back to one-by-one for this batch")
                embeddings = []
                good_recs = []
                for r, img in zip(batch, batch_images):
                    try:
                        emb = embed_image_batch([img if img is not None else r["path"]])[0]
                        embeddings.append(emb)
                        good_recs.append(r)
                    except Exception as e2:
                        log.error(f"  Skipping unreadable frame {r['path']}: {e2}")
                if not embeddings:
                    continue
                embeddings = np.stack(embeddings)
                batch = good_recs

            ids, embs, metas = [], [], []
            for i, r in enumerate(batch):
                ids.append(r["fid"])
                embs.append(embeddings[i].tolist())
                metas.append({"file": r["rel"], "mtime": r["mtime"]})

            try:
                col.upsert(ids=ids, embeddings=embs, metadatas=metas)
            except Exception as e:
                log.error(f"FAISS upsert failed: {e}")

            with _index_lock:
                _index_progress["done"] = min(start + BATCH_SIZE, len(clip_todo))

        flush_vector_store()
        unload_clip()
    else:
        log.info("CLIP pass skipped: all pending embeddings already exist in FAISS.")

    with _index_lock:
        _index_progress.update(
            stage="metadata_index",
            running=True,
            done=0,
            total=len(metadata_todo),
            started_at=time.time(),
        )

    # Pass 2: YOLO/color/OCR metadata. CLIP is no longer occupying VRAM.
    bar = tqdm(range(0, len(metadata_todo), BATCH_SIZE), desc="Metadata", unit="batch")
    for start in bar:
        batch = metadata_todo[start:start + BATCH_SIZE]
        batch_paths = [r["path"] for r in batch]
        batch_images = [_read_image(p) for p in batch_paths]
        batch_conn = _configure_conn(get_db())
        try:
            for i, r in enumerate(batch):
                path = r["path"]
                rel = r["rel"]
                mtime = r["mtime"]
                fid = r["fid"]

                try:
                    image = batch_images[i]
                    img_size = _image_size(path, image)
                    source = image if image is not None else path
                    detections, has_text_candidate = detect_objects(source)
                    colors = extract_dominant_colors(source)

                    # OCR is deliberately deferred to the batched pass after
                    # YOLO is unloaded; never run one-image OCR in metadata.
                    ocr_texts = []
                    ocr_joined = ""
                    ocr_ascii = ""

                    meta = {
                        "id":         fid,
                        "file":       rel,
                        "mtime":      mtime,
                        "width":      img_size[0],
                        "height":     img_size[1],
                        "objects":    sorted({d["class"] for d in detections}),
                        "categories": sorted({d["category"] for d in detections}),
                        "colors":     colors,
                        "has_text":   bool(has_text_candidate),
                        "detections": detections,
                        "ocr_status": "pending" if has_text_candidate else "skipped",
                        "ocr_texts":  ocr_texts,
                        "ocr_joined": ocr_joined,
                        "ocr_ascii":  ocr_ascii,
                    }
                    upsert_frame(meta, conn=batch_conn)

                except Exception as e:
                    log.error(f"Failed {rel}: {e}")

            batch_conn.commit()
        finally:
            batch_conn.close()

        with _index_lock:
            _index_progress["done"] = min(start + BATCH_SIZE, len(metadata_todo))

    unload_yolo()
    unload_ocr()

    log.info("Indexing complete.")
    with _index_lock:
        _index_progress.update(stage="idle", running=False)


# ══════════════════════════════════════════════════════════════════════════════
# Standalone entry point — run this file directly to index everything
# ══════════════════════════════════════════════════════════════════════════════

def main():
    from shared import init_db
    log.info("=== OCR/Image indexing: initializing database ===")
    init_db()
    log.info("=== OCR/Image indexing: scanning keyframes/ ===")
    build_index()
    backfill_detections()
    backfill_ocr()
    n = get_collection().count()
    log.info(f"=== OCR/Image indexing complete — {n} frames indexed ===")


if __name__ == "__main__":
    main()
