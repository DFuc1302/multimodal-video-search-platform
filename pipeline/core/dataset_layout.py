"""Configuration and safety checks for datasets stored on multiple volumes."""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MARKER_FILENAME = ".dataset-volume.json"
SHARD_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class DatasetConfigError(RuntimeError):
    """The multi-storage configuration is invalid."""


class StorageUnavailableError(RuntimeError):
    """A configured local/removable storage is not safe to use."""


def _resolve_path(value: str | os.PathLike[str], base: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve(strict=False)


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _mounted_paths() -> set[Path]:
    mounted: set[Path] = set()
    try:
        for line in Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines():
            fields = line.split()
            if len(fields) >= 5:
                mounted.add(Path(fields[4].replace("\\040", " ")).resolve(strict=False))
    except OSError:
        pass
    return mounted


def is_mount_point(path: Path) -> bool:
    resolved = path.resolve(strict=False)
    return os.path.ismount(resolved) or resolved in _mounted_paths()


def read_volume_marker(marker_file: Path) -> str:
    try:
        payload = json.loads(marker_file.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise StorageUnavailableError(
            f"Thiếu marker nhận diện volume: {marker_file}"
        ) from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise StorageUnavailableError(
            f"Không đọc được marker volume {marker_file}: {exc}"
        ) from exc
    if isinstance(payload, str):
        value = payload
    elif isinstance(payload, dict):
        value = payload.get("volume_marker") or payload.get("id")
    else:
        value = None
    if not isinstance(value, str) or not value.strip():
        raise StorageUnavailableError(
            f"Marker volume không hợp lệ, cần trường volume_marker: {marker_file}"
        )
    return value.strip()


@dataclass(frozen=True)
class StorageSnapshot:
    device: int
    marker: str | None
    videos_device: int | None = None


@dataclass(frozen=True)
class DatasetShard:
    shard_id: str
    data_root: Path
    videos_dir: Path
    enabled: bool = True
    mount_root: Path | None = None
    volume_marker: str | None = None
    require_mount: bool = False
    min_free_gb: float = 0.0
    video_manifest: Path | None = None

    @property
    def keyframes_dir(self) -> Path:
        return self.data_root / "keyframes"

    @property
    def map_dir(self) -> Path:
        return self.data_root / "map-keyframes"

    @property
    def faiss_dir(self) -> Path:
        return self.data_root / "faiss_index"

    @property
    def meta_db(self) -> Path:
        return self.data_root / "meta.db"

    @property
    def asr_cache(self) -> Path:
        return self.data_root / "asr_cache.json"

    @property
    def marker_file(self) -> Path | None:
        return self.mount_root / MARKER_FILENAME if self.mount_root else None

    def environment(self) -> dict[str, str]:
        return {
            "DATA_ROOT": str(self.data_root),
            "VIDEOS_DIR": str(self.videos_dir),
            "KEYFRAMES_DIR": str(self.keyframes_dir),
            "MAPFILES_DIR": str(self.map_dir),
            "FAISS_DIR": str(self.faiss_dir),
            "META_DB": str(self.meta_db),
            "ASR_CACHE_FILE": str(self.asr_cache),
            "ASR_FRAMES_DIR": str(self.data_root / "asr_frames"),
        }

    def _validate_volume(self) -> str | None:
        if self.mount_root is None:
            if self.volume_marker:
                raise DatasetConfigError(
                    f"Shard {self.shard_id}: volume_marker cần mount_root"
                )
            return None
        if not self.mount_root.is_dir():
            raise StorageUnavailableError(
                f"Mount root của shard {self.shard_id} không tồn tại: {self.mount_root}"
            )
        if self.require_mount and not is_mount_point(self.mount_root):
            raise StorageUnavailableError(
                f"Volume của shard {self.shard_id} chưa được mount tại {self.mount_root}"
            )
        if not _is_relative_to(self.data_root, self.mount_root):
            raise DatasetConfigError(
                f"Shard {self.shard_id}: data_root phải nằm dưới mount_root"
            )
        if self.volume_marker:
            actual = read_volume_marker(self.mount_root / MARKER_FILENAME)
            if actual != self.volume_marker:
                raise StorageUnavailableError(
                    f"Sai USB/volume tại {self.mount_root}: cần marker "
                    f"{self.volume_marker!r}, nhận {actual!r}"
                )
            return actual
        return None

    def initialize(self, *, write_marker: bool = False) -> None:
        if write_marker:
            if not self.mount_root or not self.volume_marker:
                raise DatasetConfigError(
                    f"Shard {self.shard_id}: --write-marker cần mount_root và volume_marker"
                )
            if not self.mount_root.is_dir():
                raise StorageUnavailableError(
                    f"Mount root không tồn tại: {self.mount_root}"
                )
            if self.require_mount and not is_mount_point(self.mount_root):
                raise StorageUnavailableError(
                    f"Từ chối ghi marker vì {self.mount_root} chưa phải mount point"
                )
            marker_file = self.mount_root / MARKER_FILENAME
            if marker_file.exists():
                actual = read_volume_marker(marker_file)
                if actual != self.volume_marker:
                    raise StorageUnavailableError(
                        f"Marker hiện có là {actual!r}; không ghi đè bằng "
                        f"{self.volume_marker!r}"
                    )
            else:
                marker_file.write_text(
                    json.dumps({"volume_marker": self.volume_marker}, indent=2) + "\n",
                    encoding="utf-8",
                )
        self._validate_volume()
        self.data_root.mkdir(parents=True, exist_ok=True)
        self.videos_dir.mkdir(parents=True, exist_ok=True)
        self.keyframes_dir.mkdir(parents=True, exist_ok=True)
        self.map_dir.mkdir(parents=True, exist_ok=True)

    def preflight(
        self,
        *,
        write: bool = False,
        require_videos: bool = False,
    ) -> StorageSnapshot:
        marker = self._validate_volume()
        if not self.data_root.is_dir():
            raise StorageUnavailableError(
                f"Data root của shard {self.shard_id} chưa tồn tại: {self.data_root}; "
                "hãy chạy lệnh init trước"
            )
        if require_videos and not self.videos_dir.is_dir():
            raise StorageUnavailableError(
                f"Videos directory của shard {self.shard_id} không tồn tại: {self.videos_dir}"
            )
        try:
            device = self.data_root.stat().st_dev
            videos_device = self.videos_dir.stat().st_dev if require_videos else None
            free_bytes = shutil.disk_usage(self.data_root).free
        except OSError as exc:
            raise StorageUnavailableError(
                f"Không truy cập được shard {self.shard_id}: {exc}"
            ) from exc
        required = int(self.min_free_gb * 1024**3)
        if write and free_bytes < required:
            raise StorageUnavailableError(
                f"Shard {self.shard_id} chỉ còn {free_bytes / 1024**3:.2f} GiB; "
                f"cần tối thiểu {self.min_free_gb:.2f} GiB"
            )
        if write:
            try:
                with tempfile.NamedTemporaryFile(
                    prefix=".storage-write-probe-", dir=self.data_root, delete=True
                ) as probe:
                    probe.write(b"ok")
                    probe.flush()
                    os.fsync(probe.fileno())
            except OSError as exc:
                raise StorageUnavailableError(
                    f"Shard {self.shard_id} không ghi được: {exc}"
                ) from exc
        return StorageSnapshot(
            device=device,
            marker=marker,
            videos_device=videos_device,
        )

    def verify_unchanged(self, snapshot: StorageSnapshot) -> None:
        current = self.preflight(
            write=False,
            require_videos=snapshot.videos_device is not None,
        )
        if current != snapshot:
            raise StorageUnavailableError(
                f"Storage của shard {self.shard_id} đã thay đổi trong lúc xử lý"
            )


@dataclass(frozen=True)
class DatasetConfig:
    path: Path
    shards: tuple[DatasetShard, ...]

    def select(self, shard_ids: list[str] | None = None) -> list[DatasetShard]:
        available = {shard.shard_id: shard for shard in self.shards}
        if shard_ids:
            missing = sorted(set(shard_ids) - set(available))
            if missing:
                raise DatasetConfigError("Không có shard: " + ", ".join(missing))
            selected = [available[shard_id] for shard_id in shard_ids]
        else:
            selected = [shard for shard in self.shards if shard.enabled]
        if not selected:
            raise DatasetConfigError("Không có shard nào được chọn")
        return selected


def load_dataset_config(path: str | os.PathLike[str]) -> DatasetConfig:
    config_path = Path(path).expanduser().resolve(strict=False)
    try:
        payload: Any = json.loads(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise DatasetConfigError(f"Không tìm thấy config: {config_path}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetConfigError(f"Không đọc được config {config_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("shards"), list):
        raise DatasetConfigError("Config phải có mảng shards")

    base = config_path.parent
    shards: list[DatasetShard] = []
    seen_ids: set[str] = set()
    seen_roots: set[Path] = set()
    for index, raw in enumerate(payload["shards"]):
        if not isinstance(raw, dict):
            raise DatasetConfigError(f"shards[{index}] phải là object")
        shard_id = str(raw.get("id", "")).strip()
        if not SHARD_ID_RE.fullmatch(shard_id):
            raise DatasetConfigError(f"Shard id không hợp lệ: {shard_id!r}")
        if shard_id in seen_ids:
            raise DatasetConfigError(f"Shard id bị trùng: {shard_id}")
        if not raw.get("data_root"):
            raise DatasetConfigError(f"Shard {shard_id} thiếu data_root")
        data_root = _resolve_path(raw["data_root"], base)
        if data_root in seen_roots:
            raise DatasetConfigError(f"data_root bị dùng trùng: {data_root}")
        mount_root = (
            _resolve_path(raw["mount_root"], base) if raw.get("mount_root") else None
        )
        videos_dir = _resolve_path(raw.get("videos_dir", data_root / "videos"), base)
        manifest = (
            _resolve_path(raw["video_manifest"], base)
            if raw.get("video_manifest") else None
        )
        shard = DatasetShard(
            shard_id=shard_id,
            data_root=data_root,
            videos_dir=videos_dir,
            enabled=bool(raw.get("enabled", True)),
            mount_root=mount_root,
            volume_marker=(str(raw["volume_marker"]) if raw.get("volume_marker") else None),
            require_mount=bool(raw.get("require_mount", mount_root is not None)),
            min_free_gb=max(0.0, float(raw.get("min_free_gb", 0.0))),
            video_manifest=manifest,
        )
        if mount_root and not _is_relative_to(data_root, mount_root):
            raise DatasetConfigError(
                f"Shard {shard_id}: data_root {data_root} không nằm dưới {mount_root}"
            )
        shards.append(shard)
        seen_ids.add(shard_id)
        seen_roots.add(data_root)
    if not shards:
        raise DatasetConfigError("Config chưa có shard")
    return DatasetConfig(path=config_path, shards=tuple(shards))
