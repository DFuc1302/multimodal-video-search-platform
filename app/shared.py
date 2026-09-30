"""
shared.py — Common config, DB layer, and helpers
==================================================
Shared by ocr_index.py, asr_index.py, and main.py.
Contains config, SQLite, FAISS vector search, color/category logic, CLIP
embedding, and query expansion.
"""

import json
from folder_filter import matches_folder
import logging
import os
from redis_store import ENABLED as REDIS_ENABLED, store as redis_store
import gc
import hashlib
from importlib.metadata import version as package_version, PackageNotFoundError
import sqlite3
import threading
import time
import unicodedata
from pathlib import Path
from typing import Optional, Union

# PaddleOCR CPU on WSL can crash in Paddle's oneDNN/PIR executor with:
# "ConvertPirAttribute2RuntimeAttribute not support ...". Disable oneDNN
# before paddle/paddleocr is imported; PyTorch CUDA is unaffected.
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
os.environ.setdefault("FLAGS_enable_pir_api", "0")
os.environ.setdefault("FLAGS_enable_pir_in_executor", "0")

import numpy as np
from PIL import ImageFile
import cv2
# Allow PIL to load slightly truncated/corrupt JPEGs instead of erroring or
# spamming "Corrupt JPEG data: bad Huffman code" warnings on every decode.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════

PROJECT_ROOT   = Path(__file__).resolve().parents[1]
DATA_ROOT      = Path(
    os.getenv("DATA_ROOT", str(PROJECT_ROOT / "data"))
).expanduser()
# Original videos are runtime inputs, not merged index artifacts. Keep their
# location independent from DATA_ROOT so serving a merged dataset does not make
# ASR look for a non-existent merged_data/videos directory.
VIDEOS_DIR     = Path(
    os.getenv("VIDEOS_DIR", "/mnt/d/AIC/data/videos")
).expanduser()
KEYFRAMES_DIR  = Path(os.getenv("KEYFRAMES_DIR", str(DATA_ROOT / "keyframes")))
ASR_FRAMES_DIR = Path(os.getenv("ASR_FRAMES_DIR", str(DATA_ROOT / "asr_frames")))
ASR_CACHE_FILE = Path(os.getenv("ASR_CACHE_FILE", str(DATA_ROOT / "asr_cache.json")))
FAISS_DIR      = Path(os.getenv("FAISS_DIR", str(DATA_ROOT / "faiss_index")))
META_DB        = Path(os.getenv("META_DB", str(DATA_ROOT / "meta.db")))
IMAGE_EXTS     = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS     = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".m4v"}
BATCH_SIZE     = int(os.getenv("CLIP_BATCH_SIZE", "16"))
TOP_K_MAX      = 10000
FAISS_INDEX_TYPE = os.getenv("FAISS_INDEX_TYPE", "hnsw").lower()  # hnsw | flat
FAISS_HNSW_M = int(os.getenv("FAISS_HNSW_M", "32"))
FAISS_EF_SEARCH = int(os.getenv("FAISS_EF_SEARCH", "96"))
FAISS_SAVE_EVERY = int(os.getenv("FAISS_SAVE_EVERY", "50000"))
OCR_ALL_FRAMES = os.getenv("OCR_ALL_FRAMES", "0").lower() in {"1", "true", "yes"}
YOLO_MODEL_NAME = os.getenv(
    "YOLO_MODEL", str(PROJECT_ROOT / "models" / "yolov8m.pt")
)
YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "640"))
CLIP_MODEL_NAME = os.getenv("CLIP_MODEL", "openai/clip-vit-large-patch14")
WHISPER_MODEL_NAME = os.getenv("WHISPER_MODEL", "small")
OCR_USE_ANGLE_CLS = os.getenv("OCR_USE_ANGLE_CLS", "0").lower() in {"1", "true", "yes"}
OCR_GPU = os.getenv("OCR_GPU", "1").lower() in {"1", "true", "yes"}
OCR_ENGINE = os.getenv("OCR_ENGINE", "easyocr").lower()  # none | easyocr | paddle
OCR_LANGS = [x.strip() for x in os.getenv("OCR_LANGS", "en,vi").split(",") if x.strip()]
OCR_BATCH_SIZE = int(os.getenv("OCR_BATCH_SIZE", "16"))
# Quality-affecting settings are part of provenance, across all source shards.
OCR_ENHANCE = os.getenv("OCR_ENHANCE", "1").lower() in {"1", "true", "yes"}
OCR_REGION_LIMIT = max(0, min(12, int(os.getenv("OCR_REGION_LIMIT", "8"))))
try:
    _ocr_engine_version = package_version("easyocr" if OCR_ENGINE == "easyocr" else "paddleocr")
except PackageNotFoundError:
    _ocr_engine_version = "uninstalled"
_ocr_recipe = json.dumps([OCR_ENGINE, _ocr_engine_version, OCR_LANGS, OCR_ENHANCE, OCR_REGION_LIMIT,
                          int(os.getenv("OCR_MAX_SIDE", "0")), "nfc-regions-v3.1"], sort_keys=True)
OCR_PIPELINE_VERSION = (os.getenv("OCR_PIPELINE_VERSION", "ocr-regions-v3") + "-" +
                        hashlib.sha256(_ocr_recipe.encode()).hexdigest()[:12])
SERVE_READ_ONLY = os.getenv("SERVE_READ_ONLY", "0").lower() in {"1", "true", "yes"}
META_CACHE_MODE = os.getenv("META_CACHE_MODE", "full").lower()  # full | lazy
FAISS_SQLITE_MAPPING = os.getenv("FAISS_SQLITE_MAPPING", "0").lower() in {"1", "true", "yes"}
FAISS_MMAP = os.getenv("FAISS_MMAP", "0").lower() in {"1", "true", "yes"}
CLIP_TEXT_ONLY = os.getenv("CLIP_TEXT_ONLY", "0").lower() in {"1", "true", "yes"}
CLIP_TEXT_MODEL_PATH = os.getenv("CLIP_TEXT_MODEL_PATH", "").strip()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Lazy model loaders
# ══════════════════════════════════════════════════════════════════════════════

_yolo_model     = None
_clip_model     = None
_clip_processor = None
_clip_text_model = None
_clip_tokenizer = None
_clip_load_lock = threading.RLock()
_ocr_model      = None
_whisper_model  = None

ImageLike = Union[Path, np.ndarray]

def _torch_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def configure_torch_runtime():
    try:
        import torch
        torch.backends.cudnn.benchmark = True
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass

def clear_gpu_cache():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass

def unload_clip_image():
    global _clip_model, _clip_processor
    _clip_model = None
    _clip_processor = None

def unload_clip_text():
    global _clip_text_model, _clip_tokenizer
    _clip_text_model = None
    _clip_tokenizer = None

def unload_clip():
    unload_clip_image()
    unload_clip_text()
    clear_gpu_cache()

def unload_yolo():
    global _yolo_model
    _yolo_model = None
    clear_gpu_cache()

def unload_ocr():
    global _ocr_model
    _ocr_model = None
    clear_gpu_cache()

def unload_whisper():
    global _whisper_model
    _whisper_model = None
    clear_gpu_cache()

def unload_all_models():
    unload_clip()
    unload_yolo()
    unload_ocr()
    unload_whisper()

def load_image(source: ImageLike):
    """
    Read image once using OpenCV, or normalize an already decoded ndarray.

    Returns:
        bgr : original BGR image
        rgb : RGB image
    """
    if isinstance(source, np.ndarray):
        if source.ndim != 3 or source.shape[2] != 3:
            raise RuntimeError("Expected a 3-channel image array")
        bgr = source
    else:
        bgr = cv2.imread(str(source), cv2.IMREAD_COLOR)

    if bgr is None:
        raise RuntimeError(f"Cannot read image: {source}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    return bgr, rgb

def get_yolo():
    global _yolo_model
    if _yolo_model is None:
        configure_torch_runtime()
        device = _torch_device()
        print(f"Loading YOLO ({YOLO_MODEL_NAME}) on {device}...", flush=True)
        from ultralytics import YOLO
        _yolo_model = YOLO(YOLO_MODEL_NAME)
        _yolo_model.to(device)
        try:
            _yolo_model.fuse()
        except Exception:
            pass
        print("YOLO ready.", flush=True)
    return _yolo_model

def _get_clip_unlocked():
    global _clip_model, _clip_processor
    if _clip_model is None:
        import torch
        from transformers import CLIPModel, CLIPProcessor
        configure_torch_runtime()
        name = CLIP_MODEL_NAME
        device = _torch_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"Loading CLIP ({name}) on {device}...", flush=True)
        _clip_processor = CLIPProcessor.from_pretrained(name)
        _clip_model     = CLIPModel.from_pretrained(name, torch_dtype=dtype).to(device)
        _clip_model.eval()
        print("CLIP ready.", flush=True)
    return _clip_model, _clip_processor

def get_clip():
    with _clip_load_lock:
        return _get_clip_unlocked()

def _get_clip_text_unlocked():
    """Load only CLIP's text tower for an already-indexed serving process."""
    global _clip_text_model, _clip_tokenizer
    if _clip_text_model is None:
        import torch
        from transformers import AutoTokenizer, CLIPTextModelWithProjection

        configure_torch_runtime()
        name = CLIP_TEXT_MODEL_PATH or CLIP_MODEL_NAME
        local_only = bool(CLIP_TEXT_MODEL_PATH)
        device = _torch_device()
        dtype = torch.float16 if device == "cuda" else torch.float32
        print(f"Loading CLIP text encoder ({name}) on {device}...", flush=True)
        _clip_tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_only)
        _clip_text_model = CLIPTextModelWithProjection.from_pretrained(
            name, torch_dtype=dtype, local_files_only=local_only
        ).to(device)
        _clip_text_model.eval()
        print("CLIP text encoder ready.", flush=True)
    return _clip_text_model, _clip_tokenizer

def get_clip_text():
    with _clip_load_lock:
        return _get_clip_text_unlocked()

def get_ocr():
    global _ocr_model
    if OCR_ENGINE in {"none", "off", "0", "false", "no"}:
        return None
    if _ocr_model is None:
        if OCR_ENGINE == "easyocr":
            import torch
            import easyocr
            configure_torch_runtime()
            use_gpu = OCR_GPU and torch.cuda.is_available()
            print(f"Loading EasyOCR ({','.join(OCR_LANGS)}) on {'cuda' if use_gpu else 'cpu'}...", flush=True)
            _ocr_model = easyocr.Reader(OCR_LANGS, gpu=use_gpu, verbose=False)
            print("EasyOCR ready.", flush=True)
            return _ocr_model

        print("Loading PaddleOCR...", flush=True)
        from paddleocr import PaddleOCR
        use_gpu = OCR_GPU and _torch_device() == "cuda"
        candidates = [
            {
                # PaddleOCR 2.x
                "use_angle_cls": OCR_USE_ANGLE_CLS,
                "lang": "en",
                "show_log": False,
                "use_gpu": use_gpu,
                "det_limit_side_len": 960,
            },
            {
                # PaddleOCR 3.x
                "lang": "en",
                "device": "gpu" if use_gpu else "cpu",
                "use_textline_orientation": OCR_USE_ANGLE_CLS,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
            },
            {
                "lang": "en",
                "device": "gpu" if use_gpu else "cpu",
                "use_textline_orientation": False,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
            },
            {"lang": "en"},
        ]
        last_error = None
        for kwargs in candidates:
            try:
                _ocr_model = PaddleOCR(**kwargs)
                break
            except Exception as e:
                last_error = e
                if "Unknown argument" not in str(e) and "unexpected keyword" not in str(e):
                    raise
        if _ocr_model is None:
            raise RuntimeError(f"Could not initialize PaddleOCR: {last_error}")
        print("PaddleOCR ready.", flush=True)
    return _ocr_model

_ocr_lock = threading.Lock()

# Whisper installs temporary forward hooks and maintains a KV cache while
# decoding. Reusing one global model from two FastAPI worker threads at the
# same time can corrupt that per-call state and produce zero-length attention
# tensors such as shape [1, 0, 12, -1]. Serialize all Whisper inference.
_whisper_lock = threading.RLock()

def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        configure_torch_runtime()
        device = _torch_device()
        print(f"Loading Whisper {WHISPER_MODEL_NAME} on {device}...", flush=True)
        import whisper
        _whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
        print("Whisper ready.", flush=True)
    return _whisper_model

# ══════════════════════════════════════════════════════════════════════════════
# FAISS — persistent vector store
# ══════════════════════════════════════════════════════════════════════════════

FAISS_INDEX_FILE = FAISS_DIR / "frames.faiss"
FAISS_META_FILE = FAISS_DIR / "frames_meta.json"

_faiss_collection = None

def _stable_int64_id(fid: str) -> int:
    digest = hashlib.blake2b(fid.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") & 0x7FFFFFFFFFFFFFFF

class FaissCollection:
    """
    Chroma-like adapter used by main.py and ocr_index.py.

    Stores normalized CLIP embeddings in FAISS for very fast search at 700k+
    frames, while keeping small metadata in a JSON sidecar.
    """
    def __init__(self):
        self._lock = threading.RLock()
        self._index = None
        self._dim = None
        self._added_since_save = 0
        self._sqlite_mapping = FAISS_SQLITE_MAPPING
        self._fid_to_meta: dict[str, dict] = {}
        self._faiss_id_to_fid: dict[int, str] = {}
        self._file_to_faiss_id: dict[str, int] = {}
        self._load()

    def _new_index(self, dim: int):
        import faiss
        if FAISS_INDEX_TYPE == "flat":
            base = faiss.IndexFlatIP(dim)
        else:
            base = faiss.IndexHNSWFlat(dim, FAISS_HNSW_M, faiss.METRIC_INNER_PRODUCT)
            base.hnsw.efSearch = FAISS_EF_SEARCH
        return faiss.IndexIDMap2(base)

    def _load(self):
        if not FAISS_DIR.exists():
            if SERVE_READ_ONLY:
                raise RuntimeError(f"Serving bundle has no FAISS directory: {FAISS_DIR}")
            FAISS_DIR.mkdir(parents=True, exist_ok=True)
        if not self._sqlite_mapping and FAISS_META_FILE.exists():
            try:
                data = json.loads(FAISS_META_FILE.read_text(encoding="utf-8"))
                self._fid_to_meta = data.get("fid_to_meta", {})
                self._faiss_id_to_fid = {
                    int(v["faiss_id"]): fid
                    for fid, v in self._fid_to_meta.items()
                    if "faiss_id" in v
                }
                self._file_to_faiss_id = {
                    v["file"]: int(v["faiss_id"])
                    for v in self._fid_to_meta.values()
                    if "faiss_id" in v and v.get("file")
                }
                self._dim = data.get("dim")
            except Exception as e:
                log.warning(f"Could not read FAISS metadata: {e}")
                self._fid_to_meta = {}
                self._faiss_id_to_fid = {}

        if FAISS_INDEX_FILE.exists():
            try:
                import faiss
                if FAISS_MMAP:
                    flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
                    self._index = faiss.read_index(str(FAISS_INDEX_FILE), flags)
                else:
                    self._index = faiss.read_index(str(FAISS_INDEX_FILE))
                try:
                    if hasattr(self._index, "construct_rev_map"):
                        self._index.construct_rev_map()
                except Exception:
                    pass
                base = getattr(self._index, "index", self._index)
                if hasattr(base, "hnsw"):
                    base.hnsw.efSearch = FAISS_EF_SEARCH
                self._dim = self._index.d
                log.info(f"FAISS index loaded: {self._index.ntotal} vectors, dim={self._dim}.")
            except Exception as e:
                log.warning(f"Could not load FAISS index, rebuilding on next index run: {e}")
                self._index = None

    def _save_locked(self):
        if self._index is None:
            return
        import faiss
        FAISS_DIR.mkdir(parents=True, exist_ok=True)
        tmp_index = FAISS_INDEX_FILE.with_suffix(".faiss.tmp")
        tmp_meta = FAISS_META_FILE.with_suffix(".json.tmp")
        faiss.write_index(self._index, str(tmp_index))
        tmp_meta.write_text(json.dumps({
            "format_version": 2,
            "dim": self._dim,
            "index_type": FAISS_INDEX_TYPE,
            "clip_model": CLIP_MODEL_NAME,
            "metric": "cosine_ip",
            "fid_to_meta": self._fid_to_meta,
        }, ensure_ascii=False), encoding="utf-8")
        tmp_index.replace(FAISS_INDEX_FILE)
        tmp_meta.replace(FAISS_META_FILE)
        self._added_since_save = 0
        log.info(f"FAISS checkpoint saved: {self._index.ntotal} vectors.")

    def persist(self):
        with self._lock:
            self._save_locked()

    def count(self) -> int:
        with self._lock:
            return int(self._index.ntotal) if self._index is not None else 0

    def dimension(self) -> Optional[int]:
        with self._lock:
            return int(self._dim) if self._dim is not None else None

    @staticmethod
    def _chunks(values: list, size: int = 800):
        for start in range(0, len(values), size):
            yield values[start:start + size]

    def _lookup_by_faiss_ids(self, faiss_ids: list[int]) -> dict[int, dict]:
        wanted = [int(value) for value in faiss_ids if int(value) >= 0]
        if not wanted:
            return {}
        found: dict[int, dict] = {}
        with get_db() as conn:
            for chunk in self._chunks(wanted):
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT faiss_id, file, mtime FROM frames "
                    f"WHERE faiss_id IN ({placeholders})",
                    chunk,
                ).fetchall()
                for row in rows:
                    found[int(row["faiss_id"])] = {
                        "file": row["file"],
                        "mtime": row["mtime"],
                    }
        return found

    def _lookup_faiss_ids_by_files(self, files: list[str]) -> dict[str, int]:
        if not files:
            return {}
        found: dict[str, int] = {}
        with get_db() as conn:
            for chunk in self._chunks(files):
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    f"SELECT file, faiss_id FROM frames "
                    f"WHERE file IN ({placeholders}) AND faiss_id IS NOT NULL",
                    chunk,
                ).fetchall()
                found.update({row["file"]: int(row["faiss_id"]) for row in rows})
        return found

    def get(self, ids: list[str]) -> dict:
        if self._sqlite_mapping:
            found: set[str] = set()
            with get_db() as conn:
                for chunk in self._chunks(ids):
                    placeholders = ",".join("?" for _ in chunk)
                    rows = conn.execute(
                        f"SELECT id FROM frames WHERE id IN ({placeholders}) "
                        "AND faiss_id IS NOT NULL",
                        chunk,
                    ).fetchall()
                    found.update(row["id"] for row in rows)
            return {"ids": [fid for fid in ids if fid in found]}
        with self._lock:
            found = [fid for fid in ids if fid in self._fid_to_meta]
        return {"ids": found}

    def upsert(self, ids: list[str], embeddings: list, metadatas: list[dict]):
        if self._sqlite_mapping or SERVE_READ_ONLY:
            raise RuntimeError("FAISS is read-only in serving mode")
        import numpy as np
        with self._lock:
            arr = np.asarray(embeddings, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] == 0:
                return
            norms = np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
            arr = arr / norms

            if self._index is None:
                self._dim = int(arr.shape[1])
                self._index = self._new_index(self._dim)

            add_vecs, add_ids = [], []
            for fid, emb, meta in zip(ids, arr, metadatas):
                faiss_id = _stable_int64_id(fid)
                # FAISS HNSW has no cheap true update/delete path. If the id
                # already exists, keep the original vector and refresh metadata.
                if fid not in self._fid_to_meta:
                    add_vecs.append(emb)
                    add_ids.append(faiss_id)
                    self._faiss_id_to_fid[faiss_id] = fid
                self._fid_to_meta[fid] = {
                    "faiss_id": faiss_id,
                    "file": meta.get("file", ""),
                    "mtime": meta.get("mtime", 0),
                }
                if meta.get("file"):
                    self._file_to_faiss_id[meta["file"]] = faiss_id

            if add_vecs:
                self._index.add_with_ids(
                    np.asarray(add_vecs, dtype=np.float32),
                    np.asarray(add_ids, dtype=np.int64),
                )
                try:
                    if hasattr(self._index, "construct_rev_map"):
                        self._index.construct_rev_map()
                except Exception:
                    pass
                self._added_since_save += len(add_vecs)
                if self._added_since_save >= FAISS_SAVE_EVERY:
                    self._save_locked()

    def query(self, query_embeddings: list, n_results: int):
        import numpy as np
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return {"metadatas": [[]], "distances": [[]]}

            q = np.asarray(query_embeddings, dtype=np.float32)
            if q.ndim == 1:
                q = q.reshape(1, -1)
            q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-8)

            k = min(max(int(n_results), 1), int(self._index.ntotal))
            if k == int(self._index.ntotal):
                # HNSW may not visit every vector even when k equals ntotal.
                # Score existing vectors in batches for exhaustive pagination.
                import faiss
                base = faiss.downcast_index(self._index.index)
                stored_ids = faiss.vector_to_array(self._index.id_map)
                similarities = np.empty((len(q), k), dtype=np.float32)
                for start in range(0, k, 8192):
                    stop = min(start + 8192, k)
                    vectors = base.reconstruct_n(start, stop - start)
                    similarities[:, start:stop] = q @ vectors.T
                order = np.argsort(-similarities, axis=1, kind="stable")
                scores = np.take_along_axis(similarities, order, axis=1)
                ids = stored_ids[order]
            else:
                scores, ids = self._index.search(q, k)

            all_metas, all_distances = [], []
            for query_scores, query_ids in zip(scores, ids):
                metas, distances = [], []
                seen_files = set()
                sqlite_meta = (
                    self._lookup_by_faiss_ids([int(value) for value in query_ids])
                    if self._sqlite_mapping else {}
                )
                for score, faiss_id in zip(query_scores, query_ids):
                    if faiss_id < 0:
                        continue
                    if self._sqlite_mapping:
                        meta = sqlite_meta.get(int(faiss_id))
                    else:
                        fid = self._faiss_id_to_fid.get(int(faiss_id))
                        meta = self._fid_to_meta.get(fid) if fid else None
                    if not meta or not meta.get("file") or meta["file"] in seen_files:
                        continue
                    seen_files.add(meta["file"])
                    metas.append({"file": meta["file"], "mtime": meta.get("mtime", 0)})
                    distances.append(float(1.0 - score))
                all_metas.append(metas)
                all_distances.append(distances)
        return {"metadatas": all_metas, "distances": all_distances}

    def reconstruct_by_files(self, files: list[str]) -> dict:
        """
        Reconstruct already-normalized CLIP vectors for a batch of frame files
        straight out of the FAISS index. Used by Conversational KIS so later
        clarification turns can re-rank an already narrowed pool without image
        I/O or image embedding.

        Returns {file: np.ndarray(dim,)}. Missing/unreconstructable files are
        skipped rather than failing the dialogue turn.
        """
        import numpy as np
        out = {}
        with self._lock:
            if self._index is None:
                return out
            sqlite_ids = (
                self._lookup_faiss_ids_by_files(files) if self._sqlite_mapping else {}
            )
            for f in files:
                faiss_id = (
                    sqlite_ids.get(f)
                    if self._sqlite_mapping
                    else self._file_to_faiss_id.get(f)
                )
                if faiss_id is None:
                    continue
                try:
                    vec = self._index.reconstruct(int(faiss_id))
                    out[f] = np.asarray(vec, dtype=np.float32)
                    continue
                except Exception:
                    pass
                # Fallback for FAISS builds where IndexIDMap2.reconstruct(id)
                # is unavailable: map external id -> internal row position.
                try:
                    import faiss
                    id_arr = faiss.vector_to_array(self._index.id_map)
                    pos = np.where(id_arr == int(faiss_id))[0]
                    if len(pos):
                        vec = self._index.index.reconstruct(int(pos[0]))
                        out[f] = np.asarray(vec, dtype=np.float32)
                except Exception:
                    continue
        return out

def get_collection():
    if REDIS_ENABLED:
        from redis_store import RedisCollection
        return RedisCollection()
    global _faiss_collection
    if _faiss_collection is None:
        _faiss_collection = FaissCollection()
    return _faiss_collection

def flush_vector_store():
    if _faiss_collection is not None:
        _faiss_collection.persist()

# ══════════════════════════════════════════════════════════════════════════════
# SQLite — metadata store
# ══════════════════════════════════════════════════════════════════════════════

def get_db() -> sqlite3.Connection:
    if REDIS_ENABLED:
        raise RuntimeError("SQLite access is disabled in Redis mode")
    if SERVE_READ_ONLY:
        uri = f"file:{META_DB.resolve()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=30.0)
        conn.execute("PRAGMA query_only=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("PRAGMA mmap_size=1073741824")
        conn.execute("PRAGMA cache_size=-65536")
    else:
        conn = sqlite3.connect(str(META_DB), timeout=30.0)
        # WAL mode handles rapid offline indexing writes safely.
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
    conn.row_factory = sqlite3.Row
    return conn

FTS5_AVAILABLE = False   # set by init_db() at startup

def init_db():
    global FTS5_AVAILABLE
    if SERVE_READ_ONLY:
        with get_db() as conn:
            tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type IN ('table','view')"
                )
            }
        if "frames" not in tables:
            raise RuntimeError(f"Serving database has no frames table: {META_DB}")
        FTS5_AVAILABLE = "frames_fts" in tables
        if not FTS5_AVAILABLE:
            raise RuntimeError("Serving bundle is missing the required frames_fts index")
        log.info("SQLite serving database ready (read-only, FTS5 enabled).")
        return
    with get_db() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS frames (
                id          TEXT PRIMARY KEY,
                file        TEXT NOT NULL,
                mtime       REAL NOT NULL,
                width       INTEGER,
                height      INTEGER,
                objects     TEXT DEFAULT '[]',
                categories  TEXT DEFAULT '[]',
                colors      TEXT DEFAULT '[]',
                has_text    INTEGER DEFAULT 0,
                ocr_texts   TEXT DEFAULT '[]',
                ocr_joined  TEXT DEFAULT '',
                ocr_ascii   TEXT DEFAULT '',
                detections  TEXT DEFAULT '[]',
                faiss_id    INTEGER,
                ocr_status  TEXT DEFAULT 'legacy',
                ocr_version TEXT DEFAULT '',
                ocr_source_file TEXT DEFAULT '',
                ocr_content_hash TEXT DEFAULT '',
                ocr_regions TEXT DEFAULT '[]',
                ocr_attempts INTEGER DEFAULT 0,
                ocr_error TEXT DEFAULT '',
                ocr_updated_at REAL
            );
            CREATE INDEX IF NOT EXISTS idx_file ON frames(file);
            CREATE INDEX IF NOT EXISTS idx_has_text ON frames(has_text);
        """)
        # Migrate existing DBs that lack the detections column
        cols = [r[1] for r in conn.execute("PRAGMA table_info(frames)").fetchall()]
        if "detections" not in cols:
            conn.execute("ALTER TABLE frames ADD COLUMN detections TEXT DEFAULT '[]'")
        if "ocr_ascii" not in cols:
            conn.execute("ALTER TABLE frames ADD COLUMN ocr_ascii TEXT DEFAULT ''")
        runtime_columns = {
            "faiss_id": "INTEGER",
            "ocr_status": "TEXT DEFAULT 'legacy'",
            "ocr_version": "TEXT DEFAULT ''",
            "ocr_source_file": "TEXT DEFAULT ''",
            "ocr_content_hash": "TEXT DEFAULT ''",
            "ocr_regions": "TEXT DEFAULT '[]'",
            "ocr_attempts": "INTEGER DEFAULT 0",
            "ocr_error": "TEXT DEFAULT ''",
            "ocr_updated_at": "REAL",
        }
        cols = [r[1] for r in conn.execute("PRAGMA table_info(frames)").fetchall()]
        for name, declaration in runtime_columns.items():
            if name not in cols:
                conn.execute(f"ALTER TABLE frames ADD COLUMN {name} {declaration}")
        conn.execute(
            """
            UPDATE frames SET
                ocr_status = CASE
                    WHEN COALESCE(ocr_joined,'') != '' THEN 'done'
                    WHEN COALESCE(has_text,0)=1 THEN 'pending'
                    ELSE 'skipped'
                END,
                ocr_version = CASE
                    WHEN COALESCE(ocr_joined,'') != ''
                         OR COALESCE(has_text,0)=0 THEN 'legacy-v0'
                    ELSE COALESCE(ocr_version,'')
                END,
                ocr_source_file = CASE
                    WHEN COALESCE(ocr_joined,'') != ''
                         OR COALESCE(has_text,0)=0 THEN file
                    ELSE COALESCE(ocr_source_file,'')
                END
            WHERE COALESCE(ocr_status,'legacy')='legacy'
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_ocr_status ON frames(ocr_status)")
        conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_frames_faiss_id ON frames(faiss_id)")
        stale_ascii_rows = conn.execute(
            "SELECT file, ocr_joined FROM frames WHERE COALESCE(ocr_joined, '') != '' "
            "AND COALESCE(ocr_ascii, '') = ''"
        ).fetchall()
        if stale_ascii_rows:
            conn.executemany(
                "UPDATE frames SET ocr_ascii=? WHERE file=?",
                [(normalize_ascii_text(r["ocr_joined"]), r["file"]) for r in stale_ascii_rows]
            )
            log.info(f"OCR ASCII backfill: normalized {len(stale_ascii_rows)} existing row(s).")

        fts_cols = [r[1] for r in conn.execute("PRAGMA table_info(frames_fts)").fetchall()]
        if fts_cols and "ocr_ascii" not in fts_cols:
            conn.executescript("""
                DROP TRIGGER IF EXISTS frames_ai;
                DROP TRIGGER IF EXISTS frames_ad;
                DROP TRIGGER IF EXISTS frames_au;
                DROP TABLE IF EXISTS frames_fts;
            """)

        # FTS5 inverted index over ocr_joined, for fast OCR search.
        # Wrapped in try/except since some minimal SQLite builds lack FTS5 —
        # the app falls back to a plain Python scan if so (see search route).
        try:
            conn.executescript("""
                CREATE VIRTUAL TABLE IF NOT EXISTS frames_fts USING fts5(
                    file UNINDEXED,
                    ocr_joined,
                    ocr_ascii,
                    content='frames',
                    content_rowid='rowid'
                );
                CREATE TRIGGER IF NOT EXISTS frames_ai AFTER INSERT ON frames BEGIN
                    INSERT INTO frames_fts(rowid, file, ocr_joined, ocr_ascii)
                    VALUES (new.rowid, new.file, new.ocr_joined, new.ocr_ascii);
                END;
                CREATE TRIGGER IF NOT EXISTS frames_ad AFTER DELETE ON frames BEGIN
                    INSERT INTO frames_fts(frames_fts, rowid, file, ocr_joined, ocr_ascii)
                    VALUES ('delete', old.rowid, old.file, old.ocr_joined, old.ocr_ascii);
                END;
                DROP TRIGGER IF EXISTS frames_au;
                CREATE TRIGGER frames_au
                AFTER UPDATE OF file, ocr_joined, ocr_ascii ON frames BEGIN
                    INSERT INTO frames_fts(frames_fts, rowid, file, ocr_joined, ocr_ascii)
                    VALUES ('delete', old.rowid, old.file, old.ocr_joined, old.ocr_ascii);
                    INSERT INTO frames_fts(rowid, file, ocr_joined, ocr_ascii)
                    VALUES (new.rowid, new.file, new.ocr_joined, new.ocr_ascii);
                END;
            """)
            fts_count    = conn.execute("SELECT count(*) FROM frames_fts").fetchone()[0]
            frames_count = conn.execute("SELECT count(*) FROM frames").fetchone()[0]
            if fts_count < frames_count:
                conn.execute(
                    "INSERT INTO frames_fts(rowid, file, ocr_joined, ocr_ascii) "
                    "SELECT rowid, file, ocr_joined, ocr_ascii FROM frames WHERE rowid NOT IN (SELECT rowid FROM frames_fts)"
                )
                log.info(f"FTS backfill: indexed {frames_count - fts_count} existing row(s) for OCR search.")
            FTS5_AVAILABLE = True
            log.info("FTS5 OCR search index ready.")
        except sqlite3.OperationalError as e:
            FTS5_AVAILABLE = False
            log.warning(f"FTS5 not available in this SQLite build ({e}) — OCR search will use a slower fallback scan.")
    log.info("SQLite DB ready.")

def get_existing_mtimes() -> dict:
    """One bulk query for all (file -> mtime) pairs, instead of per-frame SELECTs."""
    with get_db() as conn:
        rows = conn.execute("SELECT file, mtime FROM frames").fetchall()
    return {r["file"]: r["mtime"] for r in rows}

def frame_exists(file_rel: str, mtime: float) -> bool:
    with get_db() as conn:
        row = conn.execute(
            "SELECT mtime FROM frames WHERE file=?", (file_rel,)
        ).fetchone()
    return row is not None and abs(row["mtime"] - mtime) < 0.01

# ══════════════════════════════════════════════════════════════════════════════
# In-memory metadata cache
# ══════════════════════════════════════════════════════════════════════════════
# Offline/indexing mode may load the full metadata table. Production serving
# uses META_CACHE_MODE=lazy and fetches only FAISS/FTS candidates, avoiding a
# multi-gigabyte Python-object cache and a full-table startup scan.

_meta_cache: dict = {}          # file -> dict  (no ocr_texts — fetched on demand)
_meta_cache_lock = threading.Lock()

def _metadata_list(row, field):
    try:
        value = json.loads(row[field] or '[]')
        if not isinstance(value, list):
            raise ValueError('Expected a JSON list')
        return value
    except (ValueError, TypeError) as exc:
        from server_data_health import record_issue
        record_issue('metadata.' + field, str(exc), path=row['file'])
        return []


def _row_to_slim(r) -> dict:
    """Parse a DB row into a slim dict — omits ocr_texts to save RAM."""
    return {
        "file":       r["file"],
        "mtime":      r["mtime"],
        "width":      r["width"],
        "height":     r["height"],
        "objects":    json.loads(r["objects"]   or "[]"),
        "categories": json.loads(r["categories"] or "[]"),
        "colors":     json.loads(r["colors"]    or "[]"),
        "has_text":   bool(r["has_text"]),
        "ocr_joined": r["ocr_joined"] or "",
        "ocr_ascii":  r["ocr_ascii"] or "",
        "detections": json.loads(r["detections"] or "[]"),
    }

def warm_meta_cache(force: bool = False):
    """Optionally load all metadata; serving uses lazy candidate lookups."""
    global _meta_cache
    if META_CACHE_MODE == "lazy" and not force:
        with _meta_cache_lock:
            _meta_cache = {}
        log.info("Metadata cache skipped (META_CACHE_MODE=lazy).")
        return
    with get_db() as conn:
        rows = conn.execute(
            "SELECT file, mtime, width, height, objects, categories, colors, "
            "has_text, ocr_joined, ocr_ascii, detections FROM frames"
        ).fetchall()
    with _meta_cache_lock:
        _meta_cache = {r["file"]: _row_to_slim(r) for r in rows}
    log.info(f"Metadata cache warm: {len(_meta_cache)} frames in RAM.")

def cache_upsert(data: dict):
    """Update in-memory cache after a write — keeps RAM and DB in sync."""
    slim = {
        "file":       data["file"],
        "mtime":      data["mtime"],
        "width":      data.get("width"),
        "height":     data.get("height"),
        "objects":    data.get("objects", []),
        "categories": data.get("categories", []),
        "colors":     data.get("colors", []),
        "has_text":   bool(data.get("has_text", False)),
        "ocr_joined": data.get("ocr_joined", ""),
        "ocr_ascii":  data.get("ocr_ascii", normalize_ascii_text(data.get("ocr_joined", ""))),
        "detections": data.get("detections", []),
    }
    with _meta_cache_lock:
        _meta_cache[data["file"]] = slim

def cache_update_ocr(file_rel: str, texts: list, joined: str, ascii_text: Optional[str] = None):
    """Patch just the OCR fields in the cache after backfill_ocr writes them."""
    ascii_text = normalize_ascii_text(joined) if ascii_text is None else ascii_text
    with _meta_cache_lock:
        if file_rel in _meta_cache:
            _meta_cache[file_rel]["ocr_joined"] = joined
            _meta_cache[file_rel]["ocr_ascii"]  = ascii_text
            _meta_cache[file_rel]["has_text"]   = len(texts) > 0

def upsert_frame(data: dict, conn: Optional[sqlite3.Connection] = None):
    own_conn = conn is None
    if own_conn:
        conn = get_db()
    try:
        ocr_texts = data.get("ocr_texts", [])
        has_text = bool(data.get("has_text", False))
        status = data.get("ocr_status") or (
            "done" if ocr_texts else ("pending" if has_text else "skipped")
        )
        terminal = status in {"done", "no_text", "reused", "skipped"}
        conn.execute("""
            INSERT OR REPLACE INTO frames
            (id, file, mtime, width, height, objects, categories, colors,
             has_text, ocr_texts, ocr_joined, ocr_ascii, detections, faiss_id,
             ocr_status, ocr_version, ocr_source_file, ocr_content_hash,
             ocr_regions, ocr_attempts, ocr_error, ocr_updated_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            data["id"], data["file"], data["mtime"],
            data.get("width"), data.get("height"),
            json.dumps(data.get("objects", [])),
            json.dumps(data.get("categories", [])),
            json.dumps(data.get("colors", [])),
            int(has_text),
            json.dumps(ocr_texts),
            data.get("ocr_joined", ""),
            data.get("ocr_ascii", normalize_ascii_text(data.get("ocr_joined", ""))),
            json.dumps(data.get("detections", [])),
            data.get("faiss_id", _stable_int64_id(data["id"])),
            status,
            data.get("ocr_version", OCR_PIPELINE_VERSION if terminal else ""),
            data.get("ocr_source_file", data["file"] if terminal else ""),
            data.get("ocr_content_hash", ""),
            json.dumps(data.get("ocr_regions", [])),
            int(data.get("ocr_attempts", 0)),
            data.get("ocr_error", ""),
            data.get("ocr_updated_at", time.time() if terminal else None),
        ))
        if own_conn:
            conn.commit()
    finally:
        if own_conn:
            conn.close()
    # Keep RAM cache in sync
    cache_upsert(data)

def get_frame_meta(file_rel: str) -> Optional[dict]:
    """Full metadata including ocr_texts — fetched from DB (used by lightbox only)."""
    if REDIS_ENABLED:
        rows = redis_store().rows([file_rel])
        return rows[0] if rows else None
    with get_db() as conn:
        row = conn.execute("SELECT * FROM frames WHERE file=?", (file_rel,)).fetchone()
    if not row:
        return None
    return {
        "file":       row["file"],
        "width":      row["width"],
        "height":     row["height"],
        "objects":    json.loads(row["objects"]   or "[]"),
        "categories": json.loads(row["categories"] or "[]"),
        "colors":     json.loads(row["colors"]    or "[]"),
        "has_text":   bool(row["has_text"]),
        "ocr_texts":  json.loads(row["ocr_texts"] or "[]"),
        "ocr_joined": row["ocr_joined"] or "",
        "ocr_ascii":  row["ocr_ascii"] or "",
        "detections": json.loads(row["detections"] or "[]"),
    }

def get_all_meta() -> list:
    """Return all slim metadata; avoid this on million-scale serving paths."""
    if REDIS_ENABLED:
        return redis_store().all_rows()
    if META_CACHE_MODE != "lazy":
        with _meta_cache_lock:
            return list(_meta_cache.values())
    with get_db() as conn:
        rows = conn.execute(
            "SELECT file, mtime, width, height, objects, categories, colors, "
            "has_text, ocr_joined, ocr_ascii, detections FROM frames"
        ).fetchall()
    return [_row_to_slim(row) for row in rows]

def get_all_frame_files() -> list[str]:
    if REDIS_ENABLED:
        return [row['file'] for row in redis_store().all_rows()]
    with get_db() as conn:
        return [row["file"] for row in conn.execute("SELECT file FROM frames")]

def get_meta_by_files(files: list) -> list:
    """Fetch only requested metadata, preserving caller order."""
    if REDIS_ENABLED:
        return redis_store().rows(files)
    ordered = list(dict.fromkeys(str(value) for value in files if value))
    if not ordered:
        return []
    with _meta_cache_lock:
        found = {name: _meta_cache[name] for name in ordered if name in _meta_cache}
    missing = [name for name in ordered if name not in found]
    if missing:
        with get_db() as conn:
            for start in range(0, len(missing), 800):
                chunk = missing[start:start + 800]
                placeholders = ",".join("?" for _ in chunk)
                rows = conn.execute(
                    "SELECT file, mtime, width, height, objects, categories, colors, "
                    "has_text, ocr_joined, ocr_ascii, detections FROM frames "
                    f"WHERE file IN ({placeholders})",
                    chunk,
                ).fetchall()
                found.update({row["file"]: _row_to_slim(row) for row in rows})
    return [found[name] for name in ordered if name in found]

def query_frame_files(
    *,
    objects: set[str] | None = None,
    categories: set[str] | None = None,
    colors: set[str] | None = None,
    video: str | None = None,
    exclude_video: bool = False,
    limit: int = TOP_K_MAX,
) -> list[str]:
    """Use the offline facet index for filter-only searches."""
    if REDIS_ENABLED:
        return redis_store().query_files(objects, categories, colors, video, exclude_video, limit)
    filters = [
        ("object", {value.casefold() for value in (objects or set())}),
        ("category", {value.casefold() for value in (categories or set())}),
        ("color", {value.casefold() for value in (colors or set())}),
    ]
    subqueries: list[str] = []
    params: list[object] = []
    for kind, values in filters:
        if not values:
            continue
        placeholders = ",".join("?" for _ in values)
        subqueries.append(
            "SELECT file FROM frame_facets "
            f"WHERE kind=? AND value IN ({placeholders})"
        )
        params.append(kind)
        params.extend(sorted(values))
    if video:
        subqueries.append(
            "SELECT file FROM frames WHERE "
            f"{'NOT ' if exclude_video else ''}matches_folder(file, ?)"
        )
        params.append(video)
    if not subqueries:
        return []
    intersect = " INTERSECT ".join(subqueries)
    params.append(max(1, int(limit)))
    try:
        with get_db() as conn:
            conn.create_function("matches_folder", 2, matches_folder, deterministic=True)
            rows = conn.execute(
                f"SELECT file FROM ({intersect}) ORDER BY file LIMIT ?",
                params,
            ).fetchall()
    except sqlite3.OperationalError as exc:
        if "frame_facets" in str(exc) and META_CACHE_MODE != "lazy":
            return [row["file"] for row in get_all_meta()][: int(params[-1])]
        raise RuntimeError(
            "Serving bundle has no frame_facets index; run prepare_serve_bundle.py"
        ) from exc
    return [row["file"] for row in rows]


# ══════════════════════════════════════════════════════════════════════════════
# Category + color config
# ══════════════════════════════════════════════════════════════════════════════

CATEGORY_MAP = {
    "People":      ["person"],
    "Vehicles":    ["bicycle","car","motorcycle","airplane","bus","train","truck","boat"],
    "Animals":     ["bird","cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe"],
    "Food":        ["banana","apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake"],
    "Furniture":   ["chair","couch","potted plant","bed","dining table","toilet"],
    "Electronics": ["tv","laptop","mouse","remote","keyboard","cell phone","microwave","oven","toaster","refrigerator"],
    "Sports":      ["frisbee","skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket"],
    "Accessories": ["backpack","umbrella","handbag","tie","suitcase"],
    "Other":       ["traffic light","fire hydrant","stop sign","parking meter","bench","wine glass","cup","fork","knife","spoon","bowl","bottle","book","clock","vase","scissors","teddy bear","hair drier","toothbrush","sink"],
}
ALL_CATEGORIES = list(CATEGORY_MAP.keys())

def class_to_category(cls: str) -> str:
    for cat, members in CATEGORY_MAP.items():
        if cls.lower() in members:
            return cat
    return "Other"

COLOR_BUCKETS = [
    ("Red",    (0,   10),  0.35, 0.25, 1.0),
    ("Red",    (350, 360), 0.35, 0.25, 1.0),
    ("Orange", (11,  35),  0.40, 0.30, 1.0),
    ("Yellow", (36,  65),  0.40, 0.30, 1.0),
    ("Green",  (66, 165),  0.25, 0.15, 1.0),
    ("Cyan",   (166,200),  0.30, 0.20, 1.0),
    ("Blue",   (201,260),  0.25, 0.15, 1.0),
    ("Purple", (261,310),  0.25, 0.15, 1.0),
    ("Pink",   (311,349),  0.25, 0.20, 1.0),
    ("White",  (0,  360),  0.00, 0.85, 1.0),
    ("Gray",   (0,  360),  0.00, 0.20, 0.80),
    ("Black",  (0,  360),  0.00, 0.00, 0.18),
]

COLOR_ALIASES = {
    "red":"Red","orange":"Orange","yellow":"Yellow","green":"Green","cyan":"Cyan",
    "blue":"Blue","purple":"Purple","pink":"Pink","white":"White","gray":"Gray",
    "grey":"Gray","black":"Black","brown":"Orange",
    "đỏ":"Red","cam":"Orange","vàng":"Yellow","xanh lá":"Green","xanh dương":"Blue",
    "xanh":"Blue","tím":"Purple","hồng":"Pink","trắng":"White","xám":"Gray","đen":"Black",
}

# ══════════════════════════════════════════════════════════════════════════════
# Conversational-KIS (KISC) attribute vocabulary
# ══════════════════════════════════════════════════════════════════════════════
# Slots the clarification dialogue can ask about. Each slot has:
#   - zero_shot: CLIP text prompts per option, scored against stored vectors.
#   - keywords: bilingual triggers for auto-filling from user wording.

ATTRIBUTE_SLOTS = {
    "location": {
        "question": "Bối cảnh trong nhà hay ngoài trời? (indoor / outdoor)",
        "zero_shot": {
            "indoor":  ["a photo taken indoors, inside a room or building",
                        "an indoor scene"],
            "outdoor": ["a photo taken outdoors, in the open air",
                        "an outdoor scene, street or nature"],
        },
        "keywords": {
            "indoor":  ["indoor", "indoors", "inside", "in a room", "trong nhà", "trong phòng", "nội thất"],
            "outdoor": ["outdoor", "outdoors", "outside", "open air", "ngoài trời", "đường phố", "ngoài đường"],
        },
    },
    "gender": {
        "question": "Người trong video là nam hay nữ? (man / woman)",
        "requires_object": "person",
        "zero_shot": {
            "man":   ["a photo of a man", "a male person"],
            "woman": ["a photo of a woman", "a female person"],
        },
        "keywords": {
            "man":   ["man", "male", "guy", "boy", "đàn ông", "nam", "chú", "anh"],
            "woman": ["woman", "female", "girl", "lady", "phụ nữ", "nữ", "cô", "chị"],
        },
    },
    "time_of_day": {
        "question": "Cảnh quay là ban ngày hay ban đêm? (day / night)",
        "zero_shot": {
            "day":   ["a photo taken during daytime, bright daylight"],
            "night": ["a photo taken at night, dark, artificial lighting"],
        },
        "keywords": {
            "day":   ["daytime", "day time", "ban ngày", "buổi sáng", "buổi chiều"],
            "night": ["night", "nighttime", "ban đêm", "tối"],
        },
    },
}

def classify_color(h, s, v) -> str:
    for name, (h0, h1), ms, mnv, mxv in COLOR_BUCKETS:
        if h0 <= h <= h1 and s >= ms and mnv <= v <= mxv:
            return name
    return "Other"

def extract_dominant_colors(source: ImageLike, n: int = 4) -> list:
    try:
        _, rgb = load_image(source)
        img = cv2.resize(rgb, (80, 80))
        pixels = img.reshape(-1,3).astype(np.float32)/255.0

        if len(pixels) == 0:
            return []

        # Can't pick more unique centers than there are pixels
        k = min(n, len(pixels))
        if k < 1:
            return []

        np.random.seed(0)
        centers = pixels[np.random.choice(len(pixels), k, replace=False)]

        for _ in range(10):
            dists  = np.linalg.norm(pixels[:, None] - centers[None], axis=2)
            labels = dists.argmin(axis=1)
            new_centers = []
            for i in range(k):
                mask = labels == i
                new_centers.append(pixels[mask].mean(0) if mask.any() else centers[i])
            new_c = np.stack(new_centers) if new_centers else centers
            if np.allclose(centers, new_c, atol=1e-4):
                break
            centers = new_c

        colors = []
        for rgb in centers:
            r, g, b = rgb
            cmax, cmin = max(rgb), min(rgb)
            delta = cmax - cmin
            v = cmax
            s = delta/cmax if cmax > 0 else 0
            if delta == 0: h = 0
            elif cmax == r: h = 60*(((g-b)/delta)%6)
            elif cmax == g: h = 60*(((b-r)/delta)+2)
            else:           h = 60*(((r-g)/delta)+4)
            colors.append(classify_color(h, s, v))
        return sorted(set(c for c in colors if c != "Other"))

    except Exception as e:
        name = source.name if isinstance(source, Path) else "image"
        log.warning(f"extract_dominant_colors failed for {name}: {e}")
        return []

# ══════════════════════════════════════════════════════════════════════════════
# Detection helpers
# ══════════════════════════════════════════════════════════════════════════════

def _looks_like_text_region(rgb: np.ndarray) -> bool:
    """
    Lightweight text-region heuristic. Set OCR_ALL_FRAMES=1 to preserve the
    old exhaustive OCR behavior when recall matters more than indexing speed.
    """
    if OCR_ALL_FRAMES:
        return True

    try:
        h, w = rgb.shape[:2]
        scale = min(1.0, 640 / max(h, w))
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        grad_x = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
        grad_x = cv2.convertScaleAbs(grad_x)
        _, binary = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3))
        closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        image_area = rgb.shape[0] * rgb.shape[1]
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            area = cw * ch
            if area < max(60, image_area * 0.00025):
                continue
            aspect = cw / max(ch, 1)
            if 1.2 <= aspect <= 25 and 6 <= ch <= max(80, rgb.shape[0] * 0.35):
                return True
    except Exception:
        return True

    return False


def detect_objects(source: ImageLike) -> tuple[list, bool]:
    """Returns (detections, has_text_regions)."""
    model   = get_yolo()
    _, rgb = load_image(source)
    h, w = rgb.shape[:2]
    results = model(rgb, verbose=False, imgsz=YOLO_IMGSZ)
    detections = []
    for r in results:
        for box, cls_id, conf in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
            cls = r.names[int(cls_id)]
            x1, y1, x2, y2 = box
            detections.append({
                "class":    cls,
                "category": class_to_category(cls),
                "conf":     round(float(conf), 3),
                "box":      [round(x1/w,4), round(y1/h,4), round(x2/w,4), round(y2/h,4)],
            })

    has_text = _looks_like_text_region(rgb)

    return detections, has_text

# ══════════════════════════════════════════════════════════════════════════════
# CLIP batch embedding
# ══════════════════════════════════════════════════════════════════════════════

def embed_image_batch(paths: list[ImageLike]) -> np.ndarray:
    """Embed a batch of images with CLIP on GPU. Returns (N, D) float32."""
    import torch
    model, processor = get_clip()
    imgs   = [load_image(p)[1] for p in paths]
    inputs = processor(images=imgs, return_tensors="pt", padding=True)
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    inputs = {
        k: (v.to(device=device, dtype=dtype) if k == "pixel_values" else v.to(device=device))
        for k, v in inputs.items()
    }
    with torch.inference_mode():
        out  = model.vision_model(**inputs)
        feat = model.visual_projection(out.pooler_output)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    return feat.cpu().numpy().astype(np.float32)

def embed_text_groups(query_groups: list[list[str]]) -> np.ndarray:
    """Embed query groups in one model pass; average each group to one vector."""
    import torch
    if not query_groups:
        return np.empty((0, 0), dtype=np.float32)
    if CLIP_TEXT_ONLY:
        model, processor = get_clip_text()
    else:
        model, processor = get_clip()
    groups = [group or [""] for group in query_groups]
    flat_queries = [query for group in groups for query in group]
    device = next(model.parameters()).device
    if CLIP_TEXT_ONLY:
        inputs = processor(flat_queries, return_tensors="pt", padding=True, truncation=True)
    else:
        inputs = processor(text=flat_queries, return_tensors="pt", padding=True, truncation=True)
    inputs = {key: value.to(device=device) for key, value in inputs.items()}
    with torch.inference_mode():
        if CLIP_TEXT_ONLY:
            feat = model(**inputs).text_embeds
        else:
            out = model.text_model(**inputs)
            feat = model.text_projection(out.pooler_output)
        feat = feat / feat.norm(dim=-1, keepdim=True)
    vectors = feat.cpu().numpy().astype(np.float32)

    averaged = []
    offset = 0
    for group in groups:
        avg = vectors[offset:offset + len(group)].mean(axis=0)
        averaged.append(avg / (np.linalg.norm(avg) + 1e-8))
        offset += len(group)
    return np.asarray(averaged, dtype=np.float32)


def embed_texts(queries: list[str]) -> np.ndarray:
    """Embed multiple query strings, return averaged normalised (D,) vector."""
    return embed_text_groups([queries])[0]

def embed_single_image(path: Path) -> np.ndarray:
    return embed_image_batch([path])[0]

# ══════════════════════════════════════════════════════════════════════════════
# Query expansion
# ══════════════════════════════════════════════════════════════════════════════

VI_EN_MAP = {
    "người":"person","xe máy":"motorcycle","xe hơi":"car","ô tô":"car",
    "xe buýt":"bus","xe tải":"truck","đường phố":"street","con đường":"road",
    "thành phố":"city","biển":"sea ocean beach","rừng":"forest","núi":"mountain",
    "cầu":"bridge","tòa nhà":"building","ngôi nhà":"house","trẻ em":"children",
    "phụ nữ":"woman","đàn ông":"man","chạy":"running","đi bộ":"walking",
    "ngồi":"sitting","đứng":"standing","ăn":"eating","ban đêm":"night",
    "ban ngày":"daytime","hoàng hôn":"sunset","mưa":"rain","nắng":"sunny",
    "đám đông":"crowd","chợ":"market","trường học":"school","bệnh viện":"hospital",
    "cảnh sát":"police","bóng đá":"football soccer","bơi lội":"swimming",
}

def expand_query(query: str) -> list[str]:
    queries  = [query]
    q_lower  = unicodedata.normalize("NFC", query.lower())
    en_trans = q_lower
    found_vi = False
    for vi, en in VI_EN_MAP.items():
        if vi in q_lower:
            en_trans = en_trans.replace(vi, en)
            found_vi = True
    if found_vi and en_trans != q_lower:
        queries.append(en_trans)
    if any(ord(c) > 127 for c in query):
        stripped = "".join(
            c for c in unicodedata.normalize("NFD", query)
            if unicodedata.category(c) != "Mn"
        )
        if stripped != query:
            queries.append(stripped)
    return list(dict.fromkeys(queries))

def normalize_text(t: str) -> str:
    return unicodedata.normalize("NFC", t).lower().strip()

def strip_accents(t: str) -> str:
    """Remove Vietnamese accents while preserving ASCII search terms."""
    text = unicodedata.normalize("NFD", t or "")
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    return text.replace("đ", "d").replace("Đ", "D")

def normalize_ascii_text(t: str) -> str:
    return strip_accents(normalize_text(t))

def join_ocr_texts(texts: list) -> tuple[str, str]:
    from ocr_quality import clean_ocr_text
    joined = normalize_text(clean_ocr_text(" ".join(str(t.get("text", "")) for t in texts)))
    return joined, normalize_ascii_text(joined)

# ══════════════════════════════════════════════════════════════════════════════
# Shared indexing-progress tracker (used by ocr_index.py during startup/backfill)
# ══════════════════════════════════════════════════════════════════════════════

_index_progress = {
    "running":    False,
    "stage":      "idle",      # idle | loading_models | indexing | backfill | done
    "done":       0,
    "total":      0,
    "started_at": None,
    "error":      None,
}
_index_lock = threading.Lock()

def is_valid_image(path: Path) -> bool:
    """Quick check that an image file can be fully decoded, not just opened."""
    try:
        return cv2.imread(str(path), cv2.IMREAD_COLOR) is not None
    except Exception:
        return False
