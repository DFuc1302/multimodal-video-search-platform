"""Audit every source map referenced by the active merged dataset, read-only."""
import argparse
from collections import Counter
import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone

from dres_frames import read_map


def local_path(value):
    if __import__('os').name == 'nt' and value.startswith('/mnt/') and len(value) > 7:
        return Path(value[5].upper() + ':/' + value[7:])
    return Path(value)


def audit(root):
    config = json.loads((root / 'external_storage.json').read_text(encoding='utf-8'))
    sources = {item['id']: item for item in config['sources']}
    maps = {key: {p.stem.casefold(): p for p in local_path(value.get('map_dir', value['data_root'] + '/map-keyframes')).glob('*.csv')}
            for key, value in sources.items()}
    records, indexed_n = [], {}
    for video, source in config['video_sources'].items():
        path = maps.get(source, {}).get(video.casefold())
        entry = {'video_id': path.stem if path else video, 'source': source, 'rows': 0,
                 'errors': [], 'fps_delta_over_one_frame': 0}
        if path is None:
            entry['errors'] = ['Missing CSV']
        else:
            try:
                rows, errors = read_map(path)
                indexed_n[video.casefold()] = {row['n'] for row in rows}
                frame_times = {}
                for row in rows:
                    frame_times.setdefault(row['frame_idx'], set()).add(row['pts_time'])
                conflicts = [frame for frame, times in frame_times.items() if len(times) > 1]
                delta = [abs(row['pts_time'] - row['frame_idx'] / row['fps']) for row in rows]
                entry.update(path=str(path), rows=len(rows), errors=errors,
                             max_fps_delta_ms=round(max(delta, default=0) * 1000, 3),
                             fps_delta_over_one_frame=sum(d > 1 / row['fps'] + .002 for row, d in zip(rows, delta)),
                             first_frame=min((r['frame_idx'] for r in rows), default=None),
                             last_frame=max((r['frame_idx'] for r in rows), default=None),
                             ambiguous_frame_ids=conflicts,
                             duplicate_frame_rows=len(rows) - len(frame_times))
            except (OSError, ValueError) as exc:
                entry['errors'] = [str(exc)]
        records.append(entry)
    db_result = {'checked': 0, 'unmapped': 0, 'unmapped_examples': []}
    with sqlite3.connect((root / 'meta.db').resolve().as_uri() + '?mode=ro', uri=True) as conn:
        for (file,) in conn.execute('SELECT file FROM frames'):
            path = Path(file)
            number = path.stem.rsplit('_', 1)[-1]
            ok = number.isdigit() and int(number) in indexed_n.get(path.parent.name.casefold(), set())
            db_result['checked'] += 1
            if not ok:
                db_result['unmapped'] += 1
                if len(db_result['unmapped_examples']) < 30:
                    db_result['unmapped_examples'].append(file)
    media_ids = set()
    for source in sources.values():
        folder = local_path(source.get('videos_dir', ''))
        if folder.is_dir():
            media_ids.update(p.stem.casefold() for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in {'.mp4', '.mov', '.mkv', '.avi', '.webm'})
    return {'created_at': datetime.now(timezone.utc).isoformat(), 'data_root': str(root),
            'videos': len(records), 'sources': dict(Counter(r['source'] for r in records)),
            'rows': sum(r['rows'] for r in records),
            'videos_with_errors': sum(bool(r['errors']) for r in records),
            'videos_with_fps_delta': sum(bool(r['fps_delta_over_one_frame']) for r in records),
            'ambiguous_frame_ids': sum(len(r.get('ambiguous_frame_ids', [])) for r in records),
            'index_mapping': db_result,
            'original_videos_found': sum(video.casefold() in media_ids for video in config['video_sources']),
            'scope': 'CSV integrity and frame/time consistency; not visual verification against original video.',
            'records': records}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.data_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'records'}, ensure_ascii=True))
