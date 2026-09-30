"""Validate portable relative media paths before consulting any storage root."""
from pathlib import PurePosixPath, PureWindowsPath


def relative_asset_path(value):
    if not isinstance(value, str) or not value or '\x00' in value:
        return None
    normalized = value.replace('\\', '/')
    path = PurePosixPath(normalized)
    if path.is_absolute() or PureWindowsPath(normalized).drive or ':' in normalized:
        return None
    if any(part in {'', '.', '..'} for part in normalized.split('/')):
        return None
    return path.as_posix()
