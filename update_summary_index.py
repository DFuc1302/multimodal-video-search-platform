"""Build and validate a new summary release without touching the running index."""
import argparse
from contextlib import closing
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sqlite3
import uuid
from summary_catalog import canonical_video, local_path, topic

PROJECT = Path(__file__).resolve().parents[1]
MODEL = 'intfloat/multilingual-e5-small'

def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')

def seconds(value):
    result = 0.
    for part in str(value).split(':'):
        result = result*60+float(part)
    if not math.isfinite(result) or result < 0:
        raise ValueError('Invalid timestamp')
    return result

def flatten(value):
    if isinstance(value, dict):
        return ' '.join(flatten(v) for v in value.values())
    if isinstance(value, list):
        return ' '.join(flatten(v) for v in value)
    return str(value or '')

def discover(roots):
    found = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for path in sorted(root.rglob('*.json')):
            if path.parent.name != 'video_summaries_json' and path.parent != root:
                continue
            video = canonical_video(path.stem)
            if not __import__('re').fullmatch(r'[LMSN]\d+_V\d+', video):
                continue
            previous = next((p.name for p in path.parents if __import__('re').match(r'^\d{2}_', p.name)), '')
            category = topic(video, previous)
            if category not in {f'{i:02}' for i in range(1,13)}:
                raise ValueError(f'Unknown category: {path}')
            row = dict(video_id=video, category_code=category, summary_json=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            if video in found:
                old = found[video]
                if Path(old['summary_json']).read_bytes() != path.read_bytes():
                    raise ValueError(f'Conflicting summaries: {old["summary_json"]} and {path}')
            # Later roots (classified summaries) supply reviewed categories.
            found[video] = row
    if not found:
        raise ValueError('No summary JSON found')
    return list(found.values())

def collect(rows, data_root):
    docs, frames, issues = [], [], []
    wanted = {r['video_id'] for r in rows}
    locators = defaultdict(dict)
    with closing(sqlite3.connect((data_root/'meta.db').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        for (file,) in db.execute('SELECT file FROM frames'):
            path = Path(file.replace('\\', '/'))
            video = canonical_video(path.parent.name)
            if video not in wanted:
                continue
            stem = path.stem.removeprefix('frame_')
            if stem.isdigit():
                n = int(stem)
                if n in locators[video] and locators[video][n] != file:
                    raise ValueError(f'Ambiguous frame identity: {video}/{n}')
                locators[video][n] = file
    external = json.loads((data_root/'external_storage.json').read_text(encoding='utf-8'))
    sources = {str(s['id']): s for s in external['sources']}
    assignments = {canonical_video(v): str(s) for v,s in external['video_sources'].items()}
    map_cache = {}
    for row_number, row in enumerate(rows, 1):
        if row_number == 1 or row_number % 50 == 0:
            print(f"Summary {row_number}/{len(rows)}; issues={len(issues)}", flush=True)
        video = row['video_id']
        try:
            data = json.loads(Path(row['summary_json']).read_text(encoding='utf-8-sig'))
            if not isinstance(data, dict) or not isinstance(data.get('phan_doan_thoi_gian', []), list):
                raise ValueError('Invalid summary schema')
            segments = []
            for i, segment in enumerate(data.get('phan_doan_thoi_gian', [])):
                try:
                    stamp = segment['moc_thoi_gian']
                    start = seconds(stamp['bat_dau'])
                    end = seconds(stamp.get('ket_thuc', stamp.get('ket_thúc', stamp.get('ket_thi_gian'))))
                    text = flatten({k:v for k,v in segment.items() if k != 'moc_thoi_gian'}).strip()
                    if end <= start or not any(c.isalnum() for c in text):
                        raise ValueError('Empty/invalid segment text or interval')
                    segments.append(dict(video=video, category=row['category_code'], start=start, end=end, text=text, kind='segment'))
                except (ValueError, KeyError, TypeError) as exc:
                    issues.append(dict(video=video, segment=i, error=str(exc)))
            if not segments:
                raise ValueError('No valid summary segments')
            docs.extend(segments)
            metadata = data.get('video_metadata', {})
            if metadata:
                docs.append(dict(video=video, category=row['category_code'], start=None, end=None, text=flatten(metadata), kind='video'))
        except (OSError, ValueError, TypeError) as exc:
            issues.append(dict(video=video, error=str(exc)))
            continue
        try:
            source = sources[assignments[video]]
            map_dir = local_path(source.get('map_dir', str(local_path(source['data_root'])/'map-keyframes')))
            if map_dir not in map_cache:
                mapping = {}
                for p in map_dir.glob('*.csv'):
                    key = canonical_video(p.stem)
                    if key in mapping:
                        raise ValueError(f'Ambiguous map: {key}')
                    mapping[key] = p
                map_cache[map_dir] = mapping
            map_path = map_cache[map_dir][video]
            mapped_frames = []
            image_folders = {}
            with map_path.open(encoding='utf-8-sig', newline='') as f:
                seen = set()
                for mapped in csv.DictReader(f):
                    n = int(mapped['n'])
                    file = locators[video].get(n)
                    if file:
                        if n in seen:
                            raise ValueError(f'Duplicate map ordinal {n}')
                        seen.add(n)
                        image_path = local_path(source.get('keyframes_dir', str(local_path(source['data_root'])/'keyframes')))/file
                        if image_path.parent not in image_folders:
                            image_folders[image_path.parent] = {entry.name for entry in os.scandir(image_path.parent) if entry.is_file()}
                        if image_path.name not in image_folders[image_path.parent]:
                            raise FileNotFoundError(image_path)
                        mapped_frames.append((video, file, seconds(mapped['pts_time']), int(mapped['frame_idx'])))
            if not mapped_frames:
                raise ValueError('No mapped keyframes')
            frames.extend(mapped_frames)
            row['map_csv'] = str(map_path)
        except (OSError, ValueError, KeyError) as exc:
            issues.append(dict(video=video, kind='mapping', error=str(exc)))
    return docs, frames, issues

def write_database(path, docs, frames):
    with closing(sqlite3.connect(path)) as db, db:
        db.execute('CREATE TABLE docs(id INTEGER PRIMARY KEY, video TEXT, category TEXT, start REAL, end REAL, text TEXT, kind TEXT)')
        db.executemany('INSERT INTO docs VALUES(?,?,?,?,?,?,?)', [(i,d['video'],d['category'],d['start'],d['end'],d['text'],d['kind']) for i,d in enumerate(docs)])
        db.execute("CREATE VIRTUAL TABLE keywords USING fts5(text, tokenize='unicode61 remove_diacritics 2')")
        db.executemany('INSERT INTO keywords(rowid,text) VALUES(?,?)', [(i,d['text']) for i,d in enumerate(docs)])
        db.execute('CREATE TABLE frames(video TEXT, file TEXT, timestamp REAL, frame_idx INTEGER)')
        db.executemany('INSERT INTO frames VALUES(?,?,?,?)', frames)
        db.execute('CREATE INDEX frame_video_time ON frames(video,timestamp)')
        missing = db.execute("SELECT video,start,end FROM docs d WHERE kind='segment' AND NOT EXISTS (SELECT 1 FROM frames f WHERE f.video=d.video AND f.timestamp>=d.start AND f.timestamp<d.end)").fetchall()
    return [dict(video=r[0], start=r[1], end=r[2]) for r in missing]

def validate_release(release):
    import faiss
    import numpy as np
    idx = faiss.read_index(str(release/'summaries.faiss'))
    with closing(sqlite3.connect(release/'summaries.db')) as db:
        if db.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
            raise ValueError('Summary database integrity check failed')
        count = db.execute('SELECT count(*) FROM docs').fetchone()[0]
        if count <= 0 or idx.ntotal != count:
            raise ValueError('Vector/document count mismatch')
        if db.execute('SELECT count(*) FROM keywords').fetchone()[0] != count:
            raise ValueError('Keyword/document count mismatch')
        if db.execute('SELECT min(id),max(id) FROM docs').fetchone() != (0,count-1):
            raise ValueError('Invalid vector/document identity mapping')
        if db.execute('SELECT count(*) FROM frames').fetchone()[0] <= 0:
            raise ValueError('No mapped keyframes')
    for start in range(0, idx.ntotal, 4096):
        if not np.isfinite(idx.reconstruct_n(start, min(4096, idx.ntotal-start))).all():
            raise ValueError('Non-finite summary vectors')
    distances, ids = idx.search(idx.reconstruct_n(0,1), 1)
    if ids[0][0] < 0 or not np.isfinite(distances).all():
        raise ValueError('Summary vector search smoke check failed')

def publish(data_root, release):
    validate_release(release)
    pointer = data_root/'active_summary.json'
    payload = dict(release=release.relative_to(data_root).as_posix(), backend='sqlite')
    if pointer.exists():
        payload['previous_release'] = json.loads(pointer.read_text(encoding='utf-8'))['release']
    temp = data_root/f'.active_summary-{uuid.uuid4().hex}.tmp'
    dump(temp,payload)
    os.replace(temp,pointer)

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-root', type=local_path, default=PROJECT/'final_search_index')
    p.add_argument('--old', type=local_path, default=PROJECT/'filtered_data_all_873_moved')
    p.add_argument('--classified', type=local_path, default=local_path('/mnt/d/AIC_26/data/classified_MS'))
    p.add_argument('--extra-summary-root', type=local_path, action='append', default=[])
    p.add_argument('--audit-only', action='store_true', help='Scan and report without embedding or publishing')
    p.add_argument('--allow-issues', action='store_true', help='Publish valid documents despite reported source/mapping issues')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    args.data_root = args.data_root.resolve()
    release = args.data_root/'summary_releases'/(datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8])
    release.mkdir(parents=True)
    rows = discover([args.old, *args.extra_summary_root, args.classified])
    print(f'Scanning {len(rows)} summaries and existing frame locators...', flush=True)
    docs, frames, issues = collect(rows,args.data_root)
    dump(release/'manifest.json', rows)
    missing = write_database(release/'summaries.db',docs,frames)
    dump(release/'unmapped_segments.json',missing)
    successful = {d['video'] for d in docs}
    report = dict(videos=len(successful), source_videos=len(rows), documents=len(docs), mapped_frames=len(frames),
                  videos_by_category=dict(Counter(r['category_code'] for r in rows if r['video_id'] in successful)),
                  segments_without_frames=len(missing), issues=issues, model=MODEL, published=False)
    dump(release/'report.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='issues'}, ensure_ascii=False), flush=True)
    print(f'Issues: {len(issues)}. Full report: {release / "report.json"}', flush=True)
    if args.audit_only:
        return
    if not docs or not frames:
        raise RuntimeError('No usable summary/keyframes; existing index unchanged')
    if issues and not args.allow_issues:
        raise RuntimeError('Resolve report issues or explicitly use --allow-issues. Existing index unchanged.')
    from sentence_transformers import SentenceTransformer
    import numpy as np
    import faiss
    model = SentenceTransformer(MODEL, device=args.device)
    vectors = model.encode(['passage: '+d['text'] for d in docs], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    idx = faiss.IndexFlatIP(vectors.shape[1])
    idx.add(np.asarray(vectors,dtype='float32'))
    faiss.write_index(idx,str(release/'summaries.faiss'))
    publish(args.data_root,release)
    report['published'] = True
    dump(release/'report.json', report)
    print('Validated and activated. Restart UI to load this release; previous index retained.', flush=True)

if __name__ == '__main__':
    main()
