"""Read timestamp maps for submissions without importing search/ML modules."""
import json
from pathlib import Path
import threading
from dres_frames import FrameError, video_id
from summary_catalog import canonical_video, local_path

class SubmissionMaps:
    def __init__(self, data_root, map_root=None):
        self.root = Path(data_root)
        self.maps = Path(map_root) if map_root else self.root/'map-keyframes'
        self.lock = threading.Lock()
        self.cache = {}

    def find(self, value):
        key = canonical_video(video_id(value))
        with self.lock:
            manifest = self.root/'external_storage.json'
            if manifest.is_file():
                data = json.loads(manifest.read_text(encoding='utf-8'))
                assignments = {canonical_video(k):str(v) for k,v in data.get('video_sources',{}).items()}
                source = next((s for s in data.get('sources',[]) if str(s['id'])==assignments.get(key)), None)
                if source:
                    root = local_path(source.get('map_dir',str(local_path(source['data_root'])/'map-keyframes')))
                    return self.lookup(root,key,False)
            return self.lookup(self.maps,key,True)

    def lookup(self, root, key, recursive):
        for variant in (key, key.replace('_','-')):
            path=root/(variant+'.csv')
            if path.is_file():
                return path
        cache_key=(root,recursive)
        if cache_key not in self.cache or key not in self.cache[cache_key]:
            found={}
            for p in (root.rglob('*.csv') if recursive else root.glob('*.csv')):
                found.setdefault(canonical_video(p.stem),[]).append(p)
            self.cache[cache_key]=found
        paths=self.cache[cache_key].get(key,[])
        if len(paths)>1:
            raise FrameError(f'Nhiều bảng ánh xạ cho {key}; cần xác định đúng nguồn.')
        return paths[0] if paths else None
