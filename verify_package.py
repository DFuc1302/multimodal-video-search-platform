"""Verify downloaded processed artifacts before the first server run."""
import hashlib
import json
import os
import sqlite3
from pathlib import Path

root = Path(os.getenv('TEAM_DATA', str(Path(__file__).resolve().parent.parent / 'DATA_FOR_TEAM')))
items = json.loads((root / 'inventory.json').read_text())
for item in items:
    path = root / item['path']
    if not path.is_file() or path.stat().st_size != item['bytes']:
        raise RuntimeError(f'Missing or different size: {path}')
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != item['sha256']:
        raise RuntimeError(f'Checksum mismatch: {path}')
print(f'OK: {len(items)} processed files verified')
media = Path(os.getenv('KEYFRAMES_DIR', str(root.parent / 'filtered_data_all_873_moved')))
available = set()
for directory, _, filenames in os.walk(media):
    for name in filenames:
        available.add((Path(directory) / name).relative_to(media).as_posix())
with sqlite3.connect(f'file:{(root / "search_index_ocr/meta.db").as_posix()}?mode=ro', uri=True) as connection:
    expected = {row[0] for row in connection.execute('SELECT file FROM frames')}
missing = expected - available
if missing:
    raise RuntimeError(f'Missing {len(missing)} keyframes; examples: {sorted(missing)[:5]}')
print(f'OK: {len(expected)} indexed keyframes found in media folder')
