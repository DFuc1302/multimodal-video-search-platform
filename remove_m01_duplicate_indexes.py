"""Stage removal of M01 ' (1)' indexes; retain source media and originals."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import shutil
import sqlite3


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')


def duplicate(name):
    return bool(re.fullmatch(r'M01_V\d+ \(1\)', name, re.I))


def build(source, work, copied_db=False, metadata_only=False):
    stage = work / 'staged'
    if stage.exists() and not copied_db:
        raise RuntimeError('Staging already exists; inspect before rerunning')
    stage.mkdir(parents=True,exist_ok=copied_db)
    cache = read(source / 'asr_cache.json')
    drop_keys = [k for k in cache if duplicate(Path(k.split('::')[-1]).stem)]
    for k in drop_keys:
        if cache.get(k.replace(' (1)', '')) != cache[k]:
            raise RuntimeError(f'Transcript is not identical: {k}')
    with sqlite3.connect((source / 'meta.db').as_uri() + '?mode=ro', uri=True) as db:
        db.row_factory = sqlite3.Row
        videos = {r[0] for r in db.execute('SELECT DISTINCT video FROM frame_locator')}
        drops = sorted(v for v in videos if duplicate(v))
        for v in drops:
            original = v.replace(' (1)', '')
            copied = list(db.execute('SELECT n FROM frame_locator WHERE video=? ORDER BY n', (v,)))
            primary = list(db.execute('SELECT n FROM frame_locator WHERE video=? ORDER BY n', (original,)))
            if copied != primary or not copied:
                raise RuntimeError(f'Frame inventory mismatch: {v}')
        selected = [dict(r) for v in drops for r in db.execute(
            'SELECT frames.* FROM frames JOIN frame_locator USING(file) WHERE video=?', (v,))]
        if not selected or len(drops) != len(drop_keys):
            raise RuntimeError('ASR and frame duplicate inventories disagree')
        print(f'Removing {len(drops)} duplicate videos / {len(selected)} frames', flush=True)
        if copied_db:
            with sqlite3.connect(stage / 'meta.db') as target:
                if target.execute('SELECT count(*) FROM frames').fetchone()[0] != db.execute('SELECT count(*) FROM frames').fetchone()[0]:
                    raise RuntimeError('Precopied DB is not the source snapshot')
        else:
            with sqlite3.connect(stage / 'meta.db') as target:
                target.execute('PRAGMA synchronous=OFF')
                db.backup(target,pages=16384)
    write(work / 'removed.json', {'videos': drops, 'asr_keys': drop_keys,
                                  'frames': [{k:r[k] for k in ('id','file','faiss_id')} for r in selected]})
    for k in drop_keys:
        del cache[k]
    write(stage / 'asr_cache.json', cache)
    asr_count = sum(len(v) for v in cache.values())
    print('Filtering SQLite search metadata', flush=True)
    with sqlite3.connect(stage / 'meta.db') as db:
        db.execute('PRAGMA synchronous=OFF')
        db.executemany('DELETE FROM frame_facets WHERE file=?', [(r['file'],) for r in selected])
        db.executemany('DELETE FROM frame_locator WHERE file=?', [(r['file'],) for r in selected])
        db.executemany('DELETE FROM frames WHERE id=?', [(r['id'],) for r in selected])
        for video in drops:
            db.execute("INSERT INTO asr_segments_fts(asr_segments_fts,rowid,search_text) SELECT 'delete',id,search_text FROM asr_segments WHERE lower(video_id)=?", (video.lower(),))
            db.execute('DELETE FROM asr_segments WHERE lower(video_id)=?', (video.lower(),))
        count = db.execute('SELECT count(*) FROM frames').fetchone()[0]
        for k,v in {'frame_count':count,'asr_segment_count':asr_count,'asr_video_count':len(cache)}.items():
            db.execute('INSERT OR REPLACE INTO serve_meta VALUES (?,?)', (k,str(v)))
        db.execute("INSERT INTO frames_fts(frames_fts,rank) VALUES('integrity-check',1)")
        db.execute("INSERT INTO asr_segments_fts(asr_segments_fts,rank) VALUES('integrity-check',1)")
        assert db.execute('PRAGMA integrity_check').fetchone()[0] == 'ok'
        db.commit()
        db.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    stats = read(source / 'serve_stats.json')
    for field in ('objects','categories','colors'):
        removed = Counter(value.casefold() for r in selected for value in json.loads(r[field]))
        stats[field] = {k:v-removed[k.casefold()] for k,v in stats[field].items() if v-removed[k.casefold()] > 0}
    stats['frames'] = count
    stats['has_text_count'] -= sum(bool(r['has_text']) for r in selected)
    stats['ocr_cached_count'] -= sum(r['ocr_status'] in ('done','reused') for r in selected)
    for key,status in [('pending_ocr','pending'),('processing_ocr','processing'),('ocr_errors','error')]:
        stats[key] -= sum(r['ocr_status']==status for r in selected)
    versions = Counter(r['ocr_version'] for r in selected)
    stats['ocr_versions'] = {k:v-versions[k] for k,v in stats['ocr_versions'].items() if v-versions[k]>0}
    stats['generated_at'] = datetime.now(timezone.utc).isoformat()
    write(stage / 'serve_stats.json',stats)
    manifest=read(source / 'serve_manifest.json')
    manifest.update(frame_count=count,faiss_count=count,asr_segment_count=asr_count,
                    asr_video_count=len(cache),generated_at=stats['generated_at'])
    for key in ('has_text_count','ocr_cached_count','pending_ocr','processing_ocr','ocr_errors','ocr_versions'):
        if key in manifest: manifest[key]=stats[key]
    write(stage / 'serve_manifest.json',manifest)
    print('METADATA VERIFIED',flush=True)
    if not metadata_only:
        build_vectors(source,work)


def build_vectors(source,work):
    import faiss
    import numpy as np
    faiss.omp_set_num_threads(8)
    stage=work/'staged'
    selected=read(work/'removed.json')['frames']
    manifest=read(stage/'serve_manifest.json')
    count=manifest['frame_count']
    with sqlite3.connect(stage/'meta.db') as db:
        kept_ids=np.array([r[0] for r in db.execute('SELECT faiss_id FROM frames')],dtype='int64')
    print('Loading existing FAISS vectors',flush=True)
    index=faiss.read_index(str(source / 'faiss_index/frames.faiss'))
    ids=faiss.vector_to_array(index.id_map)
    assert set(ids.tolist()) == set(kept_ids.tolist()) | {r['faiss_id'] for r in selected}
    base=faiss.downcast_index(index.index)
    print(f'FAISS type: {type(base).__name__}',flush=True)
    rebuilt=faiss.clone_index(index)
    rebuilt.reset()
    mask=np.isin(ids,kept_ids)
    for start in range(0,len(ids),8192):
        keep=mask[start:start+8192]
        if keep.any():
            vectors=base.reconstruct_n(start,len(keep))
            rebuilt.add_with_ids(np.ascontiguousarray(vectors[keep]),ids[start:start+8192][keep])
        print(f'FAISS {min(start+8192,len(ids))}/{len(ids)}',flush=True)
    assert rebuilt.ntotal==count
    assert set(faiss.vector_to_array(rebuilt.id_map).tolist())==set(kept_ids.tolist())
    (stage / 'faiss_index').mkdir()
    faiss.write_index(rebuilt,str(stage / 'faiss_index/frames.faiss'))
    sidecar=read(source / 'faiss_index/frames_meta.json')
    removed_ids={r['id'] for r in selected}
    sidecar['fid_to_meta']={k:v for k,v in sidecar['fid_to_meta'].items() if k not in removed_ids}
    assert {v['faiss_id'] for v in sidecar['fid_to_meta'].values()}==set(kept_ids.tolist())
    write(stage / 'faiss_index/frames_meta.json',sidecar)
    write(work / 'ready.json',{'source':str(source),'frames':count,'asr_videos':manifest['asr_video_count'],'asr_segments':manifest['asr_segment_count']})
    print('STAGED AND VERIFIED',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--work',type=Path,required=True)
    parser.add_argument('--copied-db',action='store_true')
    parser.add_argument('--metadata-only',action='store_true')
    parser.add_argument('--vectors-only',action='store_true')
    args=parser.parse_args()
    if args.vectors_only:
        build_vectors(args.source.resolve(),args.work.resolve())
    else:
        build(args.source.resolve(),args.work.resolve(),args.copied_db,args.metadata_only)
