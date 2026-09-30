"""Copy already processed artifacts for sharing; run on the source WSL machine."""
import hashlib
import json
from pathlib import Path
import shutil
import socket
import sqlite3


def main():
    source = Path('/home/dtp/project/aic_local_prep')
    output = Path(__file__).resolve().parent.parent / 'DATA_FOR_TEAM'
    if output.exists():
        raise RuntimeError(f'Destination already exists: {output}')
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1', 16379)) == 0:
            raise RuntimeError('Stop local Redis before copying its AOF files')
    index = source / 'search_index_ocr'
    target = output / 'search_index_ocr'
    target.mkdir(parents=True)
    with sqlite3.connect(f'file:{index / "meta.db"}?mode=ro', uri=True) as src:
        with sqlite3.connect(target / 'meta.db') as dst:
            src.backup(dst)
            assert dst.execute('PRAGMA quick_check').fetchone()[0] == 'ok'
    for name in ('serve_manifest.json', 'serve_stats.json', 'asr_cache.json',
                 'ocr_upgrade_report.json'):
        shutil.copy2(index / name, target / name)
    for name in ('faiss_index', 'clip_text'):
        shutil.copytree(index / name, target / name, symlinks=False)
    snapshots = Path('/home/dtp/.cache/huggingface/hub/models--intfloat--multilingual-e5-small/snapshots')
    model = next(snapshots.iterdir())
    shutil.copytree(model, output / 'models/multilingual-e5-small', symlinks=False)
    shutil.copytree(source / 'redis_runtime', output / 'redis_runtime', symlinks=False)
    shutil.copytree(source / 'redis_data_local/appendonlydir', output / 'redis_data/appendonlydir')
    inventory = []
    for path in sorted(output.rglob('*')):
        if path.is_file():
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
                    digest.update(chunk)
            inventory.append(dict(path=str(path.relative_to(output)), bytes=path.stat().st_size,
                                  sha256=digest.hexdigest()))
    (output / 'inventory.json').write_text(json.dumps(inventory, indent=2))
    print(json.dumps(dict(files=len(inventory), bytes=sum(x['bytes'] for x in inventory),
                          output=str(output)), indent=2))


if __name__ == '__main__':
    main()
