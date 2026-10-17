"""Resolve distributed keyframes/maps from a local centralized search bundle."""

from __future__ import annotations

import json
from pathlib import Path
from asset_paths import relative_asset_path


def video_id_from_path(filename: str) -> str:
    """Support both VIDEO/001.jpg and category/keyframes/VIDEO/001.jpg."""
    normalized = str(filename).replace("\\", "/")
    return Path(normalized).parent.name


class ExternalAssetResolver:
    def __init__(self, bundle_root: str | Path):
        self.bundle_root = Path(bundle_root)
        manifest = self.bundle_root / "external_storage.json"
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        sources = payload.get("sources", [])
        self.sources = {
            str(item["id"]): Path(item["data_root"])
            for item in sources
            if isinstance(item, dict) and item.get("id") and item.get("data_root")
        }
        self.records = {str(item["id"]): item for item in sources
                        if isinstance(item, dict) and item.get("id") and item.get("data_root")}
        self._map_cache = {}
        self.video_sources = {
            str(video).casefold(): str(source_id)
            for video, source_id in payload.get("video_sources", {}).items()
        }

    @classmethod
    def load(cls, bundle_root: str | Path) -> "ExternalAssetResolver | None":
        manifest = Path(bundle_root) / "external_storage.json"
        return cls(bundle_root) if manifest.is_file() else None

    def root_for_video(self, video: str) -> Path | None:
        source_id = self.video_sources.get(Path(video).stem.casefold())
        return self.sources.get(source_id) if source_id else None

    def resolve_image(self, filename: str) -> Path | None:
        normalized = relative_asset_path(filename)
        if normalized is None:
            return None
        source_id = self.video_sources.get(video_id_from_path(normalized).casefold())
        # Backwards compatibility for existing manifests keyed by top-level folder.
        source_id = source_id or self.video_sources.get(normalized.split("/", 1)[0].casefold())
        if source_id not in self.records:
            return None
        record = self.records[source_id]
        root = Path(record.get("keyframes_dir", Path(record["data_root"]) / "keyframes"))
        # Do not stat removable storage while building ASR timelines. The
        # centralized SQLite locator was validated when the bundle was built;
        # /image performs the actual existence check only for requested files.
        return root / normalized

    def resolve_map(self, video: str) -> Path | None:
        if relative_asset_path(video) is None or "/" in video.replace("\\", "/"):
            return None
        source_id = self.video_sources.get(Path(video).stem.casefold())
        if source_id not in self.records:
            return None
        record = self.records[source_id]
        root = Path(record.get("map_dir", Path(record["data_root"]) / "map-keyframes"))
        path = root / f"{Path(video).stem}.csv"
        if path.is_file():
            return path
        if source_id not in self._map_cache:
            self._map_cache[source_id] = {p.stem.casefold(): p for p in root.glob("*.csv")}
        return self._map_cache[source_id].get(Path(video).stem.casefold())
