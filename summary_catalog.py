"""Shared summary identities, topic mapping and immutable release selection."""
import json
import os
import re
from pathlib import Path
from functools import lru_cache

TOPICS = {'10': 'Bản tin', '11': 'Giải Xe Đạp', '12': 'Camera an ninh'}

def canonical_video(value):
    return Path(str(value).replace('\\', '/')).stem.upper().replace('-', '_')

def local_path(value):
    value = str(value)
    if os.name == 'nt':
        value = re.sub(r'^/mnt/([a-z])/', lambda m: m[1].upper()+':/', value)
    else:
        value = re.sub(r'^([A-Za-z]):[/\\]', lambda m: '/mnt/'+m[1].lower()+'/', value).replace('\\', '/')
    return Path(value)

def topic(video, previous=''):
    video = canonical_video(video)
    if re.fullmatch(r'[MSN]\d+_V\d+', video):
        return {'M': '10', 'S': '11', 'N': '12'}[video[0]]
    code = previous[:2]
    return '10' if code == '09' else code

def active_index(data_root=None):
    root = Path(data_root or os.getenv('DATA_ROOT') or Path(__file__).resolve().parents[1]/'final_search_index')
    pointer = root/'active_summary.json'
    if pointer.is_file():
        name = json.loads(pointer.read_text(encoding='utf-8'))['release']
        path = (root/name).resolve()
        if not path.is_relative_to((root/'summary_releases').resolve()):
            raise ValueError('Invalid summary release path')
        return path
    return Path(os.getenv('SUMMARY_INDEX_DIR') or root/'summary_index')

@lru_cache(maxsize=1)
def active_categories():
    path = active_index()/'manifest.json'
    return {r['video_id']: r['category_code'] for r in json.loads(path.read_text(encoding='utf-8'))} if path.is_file() else {}

def file_topic(filename):
    path = str(filename).replace('\\', '/')
    video = canonical_video(Path(path).parent.name)
    return active_categories().get(video) or topic(video, path.split('/')[0])
