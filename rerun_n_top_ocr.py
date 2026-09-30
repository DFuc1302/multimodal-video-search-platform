"""Resumable supplemental white-text OCR over all N folders, including unindexed N."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import sqlite3
import time

VERSION = 'n-white-top-v1'


def white_readings(image, texts, top_fraction):
    """Check original colors, without requiring any black background."""
    import numpy as np
    h, w = image.shape[:2]
    kept = []
    for item in texts:
        box = np.asarray(item['box'], dtype=float)
        if box.shape != (4,) or not np.isfinite(box).all():
            continue
        x1, y1, x2, y2 = box
        crop = image[max(0, int(y1*h)):min(h, int(np.ceil(y2*h))),
                     max(0, int(x1*w)):min(w, int(np.ceil(x2*w)))]
        if not crop.size or item['conf'] < .15:
            continue
        low, high = crop.min(axis=2), crop.max(axis=2)
        white = (low >= 170) & (high.astype(float)-low <= 45)
        # White strokes may lie on colored scenery. Do not require neutral background.
        if white.mean() < .015:
            continue
        kept.append(dict(item, box=[float(x1), float(y1*top_fraction),
                                   float(x2), float(y2*top_fraction)]))
    return kept


def inventory(roots):
    found = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for folder in sorted(root.iterdir()):
            if not folder.is_dir() or not re.fullmatch(r'N\d{3}[-_]V\d+', folder.name):
                continue
            for path in sorted(folder.iterdir()):
                if path.suffix.lower() not in {'.jpg', '.jpeg', '.png'}:
                    continue
                key = folder.name + '/' + path.name
                if key in found and found[key] != path:
                    raise ValueError(f'Duplicate frame across roots: {key}')
                found[key] = path
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', action='append', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--top-fraction', type=float, default=.25)
    parser.add_argument('--batch-size', type=int, default=12)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if not 0 < args.top_fraction <= 1 or args.batch_size < 1:
        parser.error('Invalid crop or batch size')
    files = inventory(args.root)
    groups = sorted({key.split('-')[0] for key in files})
    print(json.dumps(dict(frames=len(files), groups=groups, output=str(args.output)), ensure_ascii=False), flush=True)
    if not files:
        raise RuntimeError('No N frames found')
    if args.check:
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = open(str(args.output)+'.lock', 'a+')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        print('OCR dang chay trong mot tien trinh khac; khong khoi dong trung. '
              'Ket qua va tien do cu van duoc giu nguyen.', flush=True)
        print('Theo doi luot chay nen: tail -f '
              '/mnt/c/PERSONAL_PROJECT/AIC_26/runtime_logs/n_white_top_ocr.log', flush=True)
        return
    lock.seek(0)
    lock.truncate()
    lock.write(str(os.getpid()))
    lock.flush()
    conn = sqlite3.connect(args.output)
    conn.execute('PRAGMA journal_mode=WAL')
    conn.execute('CREATE TABLE IF NOT EXISTS readings (file TEXT PRIMARY KEY, source TEXT, size INTEGER, mtime_ns INTEGER, version TEXT, texts TEXT, error TEXT, updated REAL)')
    version = f'{VERSION}:top={args.top_fraction}'
    completed = {r[0]:r[1:] for r in conn.execute('SELECT file,size,mtime_ns FROM readings WHERE version=? AND error=""', (version,))}
    todo = []
    for key, path in files.items():
        if key in completed:
            stat = path.stat()
            if completed[key] == (stat.st_size, stat.st_mtime_ns):
                continue
        todo.append((key, path))
    resumed = len(files)-len(todo)
    if args.limit:
        # Spread a sample across videos rather than measuring just one clip.
        todo = [todo[i*len(todo)//min(args.limit,len(todo))] for i in range(min(args.limit,len(todo)))]
    print(f'Selected={len(todo)}; already complete={resumed}', flush=True)
    if not todo:
        return
    os.environ['OCR_WHITE_ON_BLACK_ONLY'] = '0'
    os.environ['OCR_ENHANCE'] = '0'
    os.environ['OCR_MAX_SIDE'] = '1920'
    os.environ['OCR_ENGINE'] = 'easyocr'
    os.environ['OCR_GPU'] = '1'
    os.environ['OCR_LANGS'] = 'en,vi'
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    from ocr_index import _read_image, _easyocr_read_group
    from shared import get_ocr
    reader = get_ocr()
    start = time.monotonic()
    errors = matched = 0
    with ThreadPoolExecutor(max_workers=6) as pool:
        for offset in range(0, len(todo), args.batch_size):
            batch = []
            for key,path in todo[offset:offset+args.batch_size]:
                stat = path.stat()
                batch.append((key,path,stat.st_size,stat.st_mtime_ns))
            images = list(pool.map(_read_image, [r[1] for r in batch]))
            shape_groups = {}
            records = []
            for record, image in zip(batch, images):
                if image is None:
                    records.append((record, [], 'Cannot decode image'))
                    continue
                height = max(1, round(image.shape[0]*args.top_fraction))
                crop = image[:height].copy()
                h, w = crop.shape[:2]
                shape_groups.setdefault((h,w), []).append(dict(source=crop, w=w, h=h, record=record, fraction=height/image.shape[0]))
            for items in shape_groups.values():
                results = _easyocr_read_group(reader, items)
                for item, (texts,error) in zip(items,results):
                    records.append((item['record'], white_readings(item['source'],texts,item['fraction']) if not error else [], error or ''))
            for (key,path,size,mtime), texts,error in records:
                errors += bool(error)
                matched += bool(texts)
                conn.execute('INSERT OR REPLACE INTO readings VALUES (?,?,?,?,?,?,?,?)',
                             (key,str(path),size,mtime,version,json.dumps(texts,ensure_ascii=False),error,time.time()))
            conn.commit()
            done = min(offset+args.batch_size,len(todo))
            elapsed = time.monotonic()-start
            if offset == 0 or done % (args.batch_size*25) == 0 or done == len(todo):
                print(f'{done}/{len(todo)} matched={matched} errors={errors} fps={done/elapsed:.2f} ETA_hours={(len(todo)-done)*elapsed/done/3600:.2f}',flush=True)
    print(f'COMPLETE processed={len(todo)} matched={matched} errors={errors}; supplemental results: {args.output}',flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
