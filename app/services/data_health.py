"""Bounded, atomic diagnostics for partially processed serving datasets."""
import json
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from collections import defaultdict

ALLOW_PARTIAL_DATA = os.getenv('ALLOW_PARTIAL_DATA', '1').lower() in {'1', 'true', 'yes'}
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = Path(os.getenv('SERVER_LOG_PATH', str(REPOSITORY_ROOT / 'runtime_logs' / 'sever_log.json')))
_lock = threading.RLock()
_issues = {}
_scan = {'status': 'not_started'}


def _save():
    payload = {'updated_at': time.time(), 'scan': dict(_scan),
               'asr_processing': 'external', 'issues': list(_issues.values())}
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = LOG_PATH.with_name(LOG_PATH.name + f'.{os.getpid()}.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
        temporary.replace(LOG_PATH)
    except OSError as exc:
        logging.getLogger(__name__).error('Cannot write %s: %s', LOG_PATH, exc)


def record_issue(component, message, *, path='', count=1, samples=None):
    key = (component, str(path), str(message))
    with _lock:
        # Repeated requests must not grow the log or rewrite it on every hit.
        if key in _issues and _issues[key]['count'] == count:
            return
        if len(_issues) >= 500 and key not in _issues:
            key = ('diagnostics', '', 'Additional distinct issues suppressed')
            component, path, message = key
            count = _issues.get(key, {}).get('count', 0) + 1
            if key in _issues:
                _issues[key]['count'] = count
                if count & (count - 1):
                    return
        _issues[key] = dict(component=component, path=str(path), message=str(message),
                            count=count, samples=list(samples or [])[:30], observed_at=time.time())
        _save()


def snapshot():
    with _lock:
        return {'log_file': str(LOG_PATH), 'scan': dict(_scan),
                'issues': list(_issues.values()), 'asr_processing': 'external'}


def scan_dataset(root, keyframes=None, videos=None):
    """Background full inventory; no writes to source files or metadata DB."""
    from external_assets import ExternalAssetResolver, video_id_from_path
    root = Path(root)
    with _lock:
        _scan.update(status='running', data_root=str(root), started_at=time.time())
        # Replace this scan's old findings; runtime errors remain in the log.
        for key in list(_issues):
            if key[0].startswith('inventory.'):
                del _issues[key]
        _save()
    counts = defaultdict(int)
    examples = defaultdict(list)
    def missing(kind, value, amount=1):
        counts[kind] += amount
        if len(examples[kind]) < 30:
            examples[kind].append(str(value))
    try:
        resolver = ExternalAssetResolver.load(root)
        video_files = defaultdict(set)
        directory_entries = {}
        scanned_frames = 0
        with sqlite3.connect((root / 'meta.db').resolve().as_uri() + '?mode=ro', uri=True) as conn:
            columns = {r[1] for r in conn.execute('PRAGMA table_info(frames)')}
            status = 'ocr_status' if 'ocr_status' in columns else "'unknown'"
            for filename, state in conn.execute(f'SELECT file, {status} FROM frames'):
                video = video_id_from_path(filename)
                video_files[video.casefold()].add(Path(filename).name)
                if state not in {'done', 'no_text', 'reused'}:
                    missing('ocr.' + str(state), filename)
                image = (resolver.resolve_image(filename) if resolver else
                         Path(keyframes or root / 'keyframes') / filename)
                if image is not None and image.parent not in directory_entries:
                    try:
                        with os.scandir(image.parent) as entries:
                            directory_entries[image.parent] = {entry.name for entry in entries if entry.is_file()}
                    except OSError:
                        directory_entries[image.parent] = set()
                if image is None or image.name not in directory_entries[image.parent]:
                    missing('keyframe_missing', filename)
                scanned_frames += 1
                if scanned_frames % 10000 == 0:
                    with _lock:
                        _scan['frames_checked'] = scanned_frames
                        _save()
        records = list(resolver.records.values()) if resolver else [
            dict(keyframes_dir=str(keyframes or root / 'keyframes'),
                 videos_dir=str(videos or root / 'videos'), map_dir=str(root / 'map-keyframes'))]
        for record in records:
            base = Path(record.get('data_root', root))
            media = Path(record.get('keyframes_dir', base / 'keyframes'))
            if not media.is_dir():
                missing('keyframes_root_missing', media)
            else:
                for folder, dirs, files in os.walk(media):
                    jpgs = {name for name in files if Path(name).suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp'}}
                    if not jpgs:
                        continue
                    video = Path(folder).name
                    absent = jpgs - video_files.get(video.casefold(), set())
                    if absent:
                        missing('metadata_missing', f'{folder}: {len(absent)} frames, e.g. {sorted(absent)[:3]}', len(absent))
                    mapping = Path(record.get('map_dir', base / 'map-keyframes')) / (video + '.csv')
                    if not mapping.is_file():
                        missing('map_missing', mapping)
            video_root = Path(record.get('videos_dir', videos or base / 'videos'))
            if not video_root.is_dir():
                missing('videos_root_missing', video_root)
            else:
                source_ids = set()
                for folder, dirs, files in os.walk(video_root):
                    for name in files:
                        if Path(name).suffix.lower() in {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.ts'}:
                            source_ids.add(Path(name).stem.casefold())
                            if Path(name).stem.casefold() not in video_files:
                                missing('video_unprocessed', Path(folder) / name)
                # Only check this source's video IDs for federated external media.
                expected = (set(k for k, sid in resolver.video_sources.items()
                                if resolver.records.get(sid) is record) if resolver else set(video_files))
                for video in expected - source_ids:
                    missing('video_missing', video)
        if not (root / 'asr_cache.json').is_file():
            missing('asr_external_pending', root / 'asr_cache.json')
        for kind, count in counts.items():
            record_issue('inventory.' + kind, 'Missing or incomplete dataset component',
                         path=root, count=count, samples=examples[kind])
        with _lock:
            _scan.update(status='complete', finished_at=time.time(), videos=len(video_files),
                         frames_checked=scanned_frames)
            _save()
    except Exception as exc:
        record_issue('inventory.scan_error', str(exc), path=root)
        with _lock:
            _scan.update(status='failed', finished_at=time.time())
            _save()


class EmptyCollection:
    """Unavailable vectors never prevent OCR/filter-only queries."""
    def count(self):
        return 0

    def dimension(self):
        return 0

    def query(self, *args, **kwargs):
        return {'metadatas': [[]], 'distances': [[]], 'ids': [[]]}

    def get(self, *args, **kwargs):
        return {'ids': [], 'metadatas': []}

    def reconstruct_by_files(self, files):
        return {}
