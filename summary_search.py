"""Offline summary retrieval with exact CSV keyframe mapping."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import sqlite3
import threading

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[1] / 'aic_local_prep'
INDEX = ROOT / 'summary_index'
MODEL = 'intfloat/multilingual-e5-small'


def seconds(value):
    parts = str(value).split(':')
    result = 0.0
    for part in parts:
        result = result * 60 + float(part)
    if not np.isfinite(result) or result < 0:
        raise ValueError('Invalid timestamp')
    return result


def flatten(value):
    if isinstance(value, dict):
        return ' '.join(flatten(v) for v in value.values())
    if isinstance(value, list):
        return ' '.join(flatten(v) for v in value)
    return str(value or '')


def build():
    from sentence_transformers import SentenceTransformer
    INDEX.mkdir(exist_ok=True)
    rows = list(csv.DictReader((ROOT / 'video_manifest_873_fast.csv').open(encoding='utf-8-sig')))
    docs, frames, issues = [], [], []
    for row in rows:
        video = row['video_id']
        try:
            data = json.loads(Path(row['summary_json']).read_text(encoding='utf-8-sig'))
            folder = Path(row['keyframes_dir'])
            data_root = folder.parents[2]
            files = {int(p.stem): p for p in folder.iterdir() if p.suffix.lower() in {'.jpg', '.png', '.jpeg'} and p.stem.isdigit()}
            with (data_root / 'map-keyframes' / f'{video}.csv').open(encoding='utf-8-sig') as handle:
                for mapped in csv.DictReader(handle):
                    path = files.get(int(mapped['n']))
                    if path:
                        frames.append((video, str(path.relative_to(data_root)), seconds(mapped['pts_time']), int(mapped['frame_idx'])))
            metadata = data.get('video_metadata', {})
            if metadata:
                docs.append(dict(video=video, category=row['category_code'], start=None, end=None, text=flatten(metadata), kind='video'))
            for segment in data.get('phan_doan_thoi_gian', []):
                try:
                    stamp = segment['moc_thoi_gian']
                    start, end = seconds(stamp['bat_dau']), seconds(stamp.get('ket_thuc', stamp.get('ket_thúc', stamp.get('ket_thi_gian'))))
                    if end <= start:
                        raise ValueError('End must follow start')
                    content = {k: v for k, v in segment.items() if k != 'moc_thoi_gian'}
                    text = flatten(content).strip()
                    if text:
                        docs.append(dict(video=video, category=row['category_code'], start=start, end=end, text=text, kind='segment'))
                except (ValueError, KeyError, TypeError) as exc:
                    issues.append(dict(video=video, error=str(exc)))
        except Exception as exc:
            issues.append(dict(video=video, error=str(exc)))
    if not docs:
        raise RuntimeError('No summary documents found')
    print(f'Documents: {len(docs)}, mapped frames: {len(frames)}, issues: {len(issues)}', flush=True)
    model = SentenceTransformer(MODEL, device=os.getenv('SUMMARY_DEVICE', 'cpu'))
    vectors = model.encode(['passage: ' + d['text'] for d in docs], batch_size=32, normalize_embeddings=True, show_progress_bar=True)
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(np.asarray(vectors, dtype='float32'))
    db_path = INDEX / 'summaries.new.db'
    db_path.unlink(missing_ok=True)
    with sqlite3.connect(db_path) as db:
        db.execute('CREATE TABLE docs(id INTEGER PRIMARY KEY, video TEXT, category TEXT, start REAL, end REAL, text TEXT, kind TEXT)')
        db.executemany('INSERT INTO docs VALUES(?,?,?,?,?,?,?)', [(i, d['video'], d['category'], d['start'], d['end'], d['text'], d['kind']) for i, d in enumerate(docs)])
        db.execute("CREATE VIRTUAL TABLE keywords USING fts5(text, tokenize='unicode61 remove_diacritics 2')")
        db.executemany('INSERT INTO keywords(rowid,text) VALUES(?,?)', [(i,d['text']) for i,d in enumerate(docs)])
        db.execute('CREATE TABLE frames(video TEXT, file TEXT, timestamp REAL, frame_idx INTEGER)')
        db.executemany('INSERT INTO frames VALUES(?,?,?,?)', frames)
        db.execute('CREATE INDEX frame_video_time ON frames(video,timestamp)')
        missing_rows = db.execute("SELECT d.video,d.start,d.end,d.text FROM docs d WHERE kind='segment' AND NOT EXISTS(SELECT 1 FROM frames f WHERE f.video=d.video AND f.timestamp>=d.start AND f.timestamp<d.end)").fetchall()
        missing = len(missing_rows)
        (INDEX / 'unmapped_segments.json').write_text(json.dumps([dict(zip(('video','start','end','text'), row)) for row in missing_rows], ensure_ascii=False, indent=2))
    faiss.write_index(index, str(INDEX / 'summaries.faiss'))
    db_path.replace(INDEX / 'summaries.db')
    report = dict(videos=len(rows), documents=len(docs), mapped_frames=len(frames), segments_without_frames=missing, model=MODEL, issues=issues)
    (INDEX / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False), flush=True)


class SummarySearch:
    def __init__(self):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(MODEL, device='cpu', local_files_only=True)
        self.index = faiss.read_index(str(INDEX / 'summaries.faiss'))
        self.lock = threading.Lock()

    def search(self, query, top_k=40, category='', video=''):
        if not query.strip():
            return dict(results=[], total=0, segments=[])
        with self.lock:
            vector = self.model.encode(['query: ' + query], normalize_embeddings=True)
        _, ids = self.index.search(np.asarray(vector, dtype='float32'), self.index.ntotal)
        with sqlite3.connect(f'file:{INDEX / "summaries.db"}?mode=ro', uri=True) as db:
            db.row_factory = sqlite3.Row
            docs = {r['id']: dict(r) for r in db.execute('SELECT * FROM docs') if (not category or r['category']==category) and (not video or r['video']==video)}
            dense = [int(i) for i in ids[0] if int(i) in docs][:None if top_k is None else 200]
            tokens = re.findall(r'\w+', query, re.UNICODE)
            expression = ' OR '.join('"' + token + '"' for token in tokens)
            sparse = [r[0] for r in db.execute('SELECT rowid FROM keywords WHERE keywords MATCH ? ORDER BY bm25(keywords)', (expression,)) if r[0] in docs][:None if top_k is None else 200] if expression else []
            scores = {}
            for ranking in (dense, sparse):
                for rank, key in enumerate(ranking, 1):
                    scores[key] = scores.get(key, 0) + 1 / (60 + rank)
            ranked = sorted(scores, key=lambda key: (-scores[key], key))
            results, seen, segments = [], set(), []
            for key in ranked:
                doc = docs[key]
                segments.append({**doc, 'rrf_score': scores[key]})
                if doc['kind'] == 'segment':
                    candidates = db.execute('SELECT * FROM frames WHERE video=? AND timestamp>=? AND timestamp<? ORDER BY abs(timestamp-?),file LIMIT ?', (doc['video'],doc['start'],doc['end'],(doc['start']+doc['end'])/2, -1 if top_k is None else 3)).fetchall()
                else:
                    continue
                for frame in candidates:
                    if frame['file'] in seen:
                        continue
                    seen.add(frame['file'])
                    results.append({**dict(frame), 'score': scores[key], 'summary': doc['text'], 'category_code':doc['category'], 'segment_start':doc['start'], 'segment_end':doc['end'], 'source':'summary', 'objects':[], 'categories':[], 'colors':[]})
                    if top_k is not None and len(results) >= top_k:
                        return dict(results=results,total=len(results),segments=segments[:20],score_type='rrf')
            return dict(results=results,total=len(results),segments=segments[:20],score_type='rrf')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--build', action='store_true')
    parser.add_argument('--query')
    args = parser.parse_args()
    if args.build:
        build()
    elif args.query:
        print(json.dumps(SummarySearch().search(args.query, top_k=5),ensure_ascii=False,indent=2))
