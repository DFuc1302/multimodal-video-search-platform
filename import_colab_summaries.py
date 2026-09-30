"""Append validated Colab summaries to an existing index using the final bundle's exact maps."""
import argparse
from bisect import bisect_left
import csv
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3

from colab_summary_ms import covered_until
from external_assets import ExternalAssetResolver

TEXT_FIELDS = ('mo_ta_hinh_anh', 'am_thanh_loi_thoai', 'chu_hien_thi_ocr', 'tu_khoa_truy_van')


def flatten(value):
    if isinstance(value, list):
        return ' '.join(flatten(x) for x in value)
    if isinstance(value, dict):
        return ' '.join(flatten(x) for x in value.values())
    return str(value or '')


def segment_text(segment):
    return '\n'.join(flatten(segment.get(k)).strip() for k in TEXT_FIELDS if segment.get(k))


def collect(source, bundle, families=('M', 'S')):
    resolver = ExternalAssetResolver(bundle)
    docs, frames, unmapped, hashes = [], [], [], {}
    paths = sorted(p for p in source.glob('*.json') if p.stem[:1] in families)
    if not paths:
        raise ValueError('No summary JSONs found for selected families')
    with closing(sqlite3.connect((bundle/'meta.db').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        for path in paths:
            data = json.loads(path.read_text(encoding='utf-8-sig'))
            video = path.stem
            if data['video_metadata']['video_id'] != video:
                raise ValueError(f'ID mismatch: {path}')
            segments = data['phan_doan_thoi_gian']
            if not segments:
                raise ValueError(f'Empty summary: {video}')
            covered_until(segments, float(segments[-1]['moc_thoi_gian']['ket_thuc']))
            lookup = dict(db.execute('SELECT n,file FROM frame_locator WHERE video=?', (video.lower(),)))
            if not lookup:
                raise ValueError(f'No indexed keyframes: {video}')
            mapping = resolver.resolve_map(video)
            if mapping is None:
                raise ValueError(f'No timestamp map: {video}')
            timeline = []
            with mapping.open(encoding='utf-8-sig', newline='') as handle:
                for row in csv.DictReader(handle):
                    file = lookup.get(int(row['n']))
                    if file:
                        timeline.append((video, file, float(row['pts_time']), int(row['frame_idx'])))
            times = sorted(row[2] for row in timeline)
            if not times:
                raise ValueError(f'No matching frames/map: {video}')
            frames.extend(timeline)
            hashes[video] = hashlib.sha256(path.read_bytes()).hexdigest()
            for segment in segments:
                span = segment['moc_thoi_gian']
                start, end = float(span['bat_dau']), float(span['ket_thuc'])
                text = segment_text(segment)
                if not text:
                    raise ValueError(f'Empty text: {video} {start}')
                doc = dict(video=video, category={'M':'10','S':'11'}[video[0]], start=start, end=end, text=text, kind='segment')
                docs.append(doc)
                pos = bisect_left(times, start)
                if pos >= len(times) or times[pos] >= end:
                    unmapped.append(dict(video=video, start=start, end=end))
    return docs, frames, unmapped, hashes


def build(source, bundle, base, output, device, families):
    import faiss
    import numpy as np
    from sentence_transformers import SentenceTransformer
    if output.exists():
        raise FileExistsError(f'Choose a new output directory: {output}')
    docs, frames, unmapped, hashes = collect(source, bundle, families)
    index = faiss.read_index(str(base/'summaries.faiss'))
    with closing(sqlite3.connect((base/'summaries.db').resolve().as_uri()+'?mode=ro', uri=True)) as db:
        old = [dict(zip(('id','video'), row)) for row in db.execute('SELECT id,video FROM docs ORDER BY id')]
    if [d['id'] for d in old] != list(range(index.ntotal)):
        raise ValueError('Base index/DB IDs differ')
    if {d['video'].upper() for d in old} & set(hashes):
        raise ValueError('Selected video already exists in base; refusing to duplicate/overwrite')
    model_id = 'intfloat/multilingual-e5-small'
    report_path = base/'report.json'
    if report_path.exists() and json.loads(report_path.read_text()).get('model') != model_id:
        raise ValueError('Base uses a different embedding model')
    print(f'Keep {index.ntotal} old docs; add {len(docs)} segments, {len(hashes)} videos, '
          f'{len(frames)} frames; unmapped new segments: {len(unmapped)}', flush=True)
    model = SentenceTransformer(model_id, device=device, local_files_only=True)
    vectors = model.encode(['passage: '+d['text'] for d in docs], batch_size=32,
                          normalize_embeddings=True, show_progress_bar=True)
    if vectors.shape != (len(docs), index.d):
        raise ValueError('Embedding dimension mismatch')
    output.mkdir(parents=True)
    offset = index.ntotal
    index.add(np.asarray(vectors, dtype='float32'))
    with closing(sqlite3.connect((base/'summaries.db').resolve().as_uri()+'?mode=ro', uri=True)) as old_db, \
            closing(sqlite3.connect(output/'summaries.db')) as db, db:
        old_db.backup(db)
        db.executemany('INSERT INTO docs VALUES(?,?,?,?,?,?,?)',
            [(offset+i,d['video'],d['category'],d['start'],d['end'],d['text'],d['kind']) for i,d in enumerate(docs)])
        db.executemany('INSERT INTO keywords(rowid,text) VALUES(?,?)', [(offset+i,d['text']) for i,d in enumerate(docs)])
        db.executemany('INSERT INTO frames VALUES(?,?,?,?)', frames)
        assert db.execute('SELECT count(*) FROM docs').fetchone()[0] == index.ntotal
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
    faiss.write_index(index, str(output/'summaries.faiss'))
    report = dict(model=model_id, base_index=str(base), bundle=str(bundle),
        old_documents=offset, added_documents=len(docs), documents=index.ntotal,
        added_videos=len(hashes), added_frames=len(frames),
        unmapped_new_segments=unmapped, source_sha256=hashes)
    (output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in ('source_sha256','unmapped_new_segments')}), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--bundle', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', default='cuda')
    p.add_argument('--families', nargs='+', choices=['M','S'], default=['M','S'])
    a=p.parse_args()
    build(a.source, a.bundle, a.base, a.output, a.device, a.families)
