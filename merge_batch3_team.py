"""Run in WSL alongside merge_processed_data.py; preserve all source data."""
import argparse
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sqlite3
import subprocess
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/mnt/f/AIC_2026'))
    p.add_argument('--new-index', type=Path, nargs='+', help='Explicit folders containing meta.db and faiss_index')
    p.add_argument('--output', type=Path)
    p.add_argument('--check-only', action='store_true')
    a = p.parse_args()
    if os.name == 'nt':
        p.error('Run this script in WSL Ubuntu.')
    root = a.root.resolve()
    batch = root / 'batch_3_downloads'
    old = root / 'DATA_FOR_TEAM/search_index_ocr'
    output = (a.output or root / 'merged_search_index_batch3').resolve()
    if output.exists():
        raise RuntimeError(f'Output already exists; choose another --output: {output}')
    search = batch / 'processed_data'
    new = [x.resolve() for x in a.new_index] if a.new_index else sorted({x.parent.resolve() for x in search.rglob('meta.db')})
    if not new:
        raise RuntimeError(f'No meta.db found under {search}. Use --new-index with the processed index folder(s).')
    sources = [old, *new]
    model = old / 'clip_text'
    if not (model / 'config.json').is_file():
        raise RuntimeError(f'Missing local CLIP model: {model}')
    records, owners, assets = [], {}, []
    for i, source in enumerate(sources):
        for rel in ('meta.db', 'faiss_index/frames.faiss', 'faiss_index/frames_meta.json'):
            if not (source / rel).is_file():
                raise RuntimeError(f'Missing: {source / rel}')
        if (source / 'external_storage.json').exists():
            raise RuntimeError(f'Source already uses external storage; needs its own asset mapping: {source}')
        candidates = [source / 'keyframes', root / 'filtered_data_all_873_moved'] if i == 0 else [source / 'keyframes', batch / 'keyframes']
        with sqlite3.connect((source / 'meta.db').as_uri() + '?mode=ro', uri=True) as db:
            files = [str(row[0]).replace('\\', '/') for row in db.execute('SELECT file FROM frames')]
        if not files or any(PurePosixPath(f).is_absolute() or '..' in PurePosixPath(f).parts or ':' in f for f in files):
            raise RuntimeError(f'Empty or unsafe frame paths: {source}')
        frames = next((c for c in candidates if c.is_dir() and all((c / f).is_file() for f in files)), None)
        if frames is None:
            raise RuntimeError(f'Cannot resolve every frame in {source}; checked {candidates}')
        videos = {PurePosixPath(f).parent.name.casefold() for f in files}
        conflicts = videos & owners.keys()
        if conflicts:
            raise RuntimeError(f'Overlapping videos in source indices; select non-overlapping inputs: {sorted(conflicts)[:20]}')
        map_candidates = [source / 'map-keyframes', frames / 'map-keyframes', batch / 'map-keyframes']
        maps = next((c for c in map_candidates if c.is_dir() and videos <= {x.stem.casefold() for x in c.glob('*.csv')}), None)
        if maps is None:
            raise RuntimeError(f'Missing timestamp CSVs for {source}')
        sid = f'source_{i}'
        owners.update({v: sid for v in videos})
        records.append(dict(id=sid, data_root=str(source), keyframes_dir=str(frames), map_dir=str(maps), videos_dir=str(root / 'videos'), require_mount=False))
        assets.append((frames, maps))
        print(f'{source}: {len(files)} frames, {len(videos)} videos', flush=True)
    print(f'Output: {output}', flush=True)
    if a.check_only:
        print('Path checks OK. No data merged. SQLite/FAISS integrity is checked during merge.')
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='aic-batch3-') as temp:
        scratch = Path(temp)
        views = []
        for i, source in enumerate(sources):
            view = scratch / f'source_{i}'
            view.mkdir()
            for name in ('meta.db', 'faiss_index', 'asr_cache.json'):
                if (source / name).exists():
                    (view / name).symlink_to(source / name, target_is_directory=(source / name).is_dir())
            (view / 'keyframes').symlink_to(assets[i][0], target_is_directory=True)
            (view / 'map-keyframes').symlink_to(assets[i][1], target_is_directory=True)
            views.append(str(view))
        manifest = scratch / 'external_storage.json'
        manifest.write_text(json.dumps(dict(format_version=3, asset_mode='external', sources=records, video_sources=owners)), encoding='utf-8')
        env = {**os.environ, 'BACKEND_MODE': 'sqlite', 'STORAGE_BACKEND': 'sqlite', 'AIC_USE_ACTIVE_DATASET': '0'}
        # Publish only once both the merged index and local model are ready.
        with tempfile.TemporaryDirectory(prefix='.batch3-build-', dir=output.parent) as stage:
            bundle = Path(stage) / 'bundle'
            subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('merge_processed_data.py')), '--sources', *views, '--output', str(bundle), '--asset-mode', 'external', '--external-storage-manifest', str(manifest), '--skip-clip-text-export'], env=env, check=True)
            shutil.copytree(model, bundle / 'clip_text', dirs_exist_ok=True)
            bundle.rename(output)
    print(f'READY: {output}\nKeep the original image/map folders in place.')


if __name__ == '__main__':
    main()
