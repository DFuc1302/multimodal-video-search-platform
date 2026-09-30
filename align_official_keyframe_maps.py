"""Audit/align self-extracted maps to the organizer's timestamp-based IDX convention.

Never join organizer n to local n: they identify different sampled images.
Original decoded indices/FPS are retained for exact source-frame extraction.
Default is an audit. --apply backs up and atomically replaces changed CSVs.
"""
import argparse
import csv
import hashlib
import io
import json
from datetime import datetime, timezone
from pathlib import Path

from summary_catalog import canonical_video, local_path


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def align(rows, official):
    rates = {r['fps'] for r in official}
    if len(rates) != 1 or float(next(iter(rates))) <= 0:
        raise ValueError('Organizer FPS must be constant and positive')
    fps = next(iter(rates))
    # Intentionally match the float/truncation convention observed in BTC CSVs.
    if any(int(float(r['pts_time']) * float(fps)) != int(r['frame_idx']) for r in official):
        raise ValueError('Organizer rows do not follow the verified IDX convention')
    anchors = {}
    for row in official:
        t = float(row['pts_time'])
        idx = int(row['frame_idx'])
        if t in anchors and anchors[t] != idx:
            raise ValueError('Conflicting organizer timestamp anchors')
        anchors[t] = idx
    result, deltas, exact = [], [], 0
    for row in rows:
        t = float(row['pts_time'])
        if t < 0:
            raise ValueError('Negative local timestamp requires manual investigation')
        idx = anchors.get(t, int(t * float(fps)))
        exact += t in anchors
        updated = dict(row)
        updated.setdefault('source_frame_idx', row['frame_idx'])
        updated.setdefault('source_fps', row['fps'])
        updated.update(fps=fps, frame_idx=str(idx))
        deltas.append(idx - int(row['frame_idx']))
        result.append(updated)
    if len({r['n'] for r in result}) != len(result):
        raise ValueError('Duplicate local image n')
    ordered = sorted(result, key=lambda r: float(r['pts_time']))
    if any(int(b['frame_idx']) <= int(a['frame_idx']) for a, b in zip(ordered, ordered[1:])):
        raise ValueError('IDX collision/non-monotonic timestamp; manual investigation required')
    return result, {'rows': len(rows), 'idx_changed': sum(d != 0 for d in deltas),
                    'max_abs_idx_delta': max(map(abs, deltas), default=0),
                    'official_fps': fps, 'previous_fps': sorted({r['fps'] for r in rows}),
                    'fps_changed': any(float(r['fps']) != float(fps) for r in rows),
                    'exact_timestamp_anchors': exact}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--report-dir', type=Path, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    manifest = json.loads((args.data_root / 'external_storage.json').read_text(encoding='utf-8'))
    officials = {}
    for path in args.official.glob('*.csv'):
        key = canonical_video(path.stem)
        if key in officials:
            raise ValueError(f'Duplicate official video: {key}')
        officials[key] = path
    assignments = {canonical_video(k): v for k, v in manifest['video_sources'].items()}
    sources = {s['id']: s for s in manifest['sources']}
    records, plans = [], []
    for video, official_path in sorted(officials.items()):
        source = sources.get(assignments.get(video))
        if source is None or source['id'] == 's0_old':
            records.append({'video': video, 'status': 'skip_not_self_extracted'})
            continue
        active_dir = local_path(source['map_dir'])
        variants = (video + '.csv', video.replace('_', '-') + '.csv')
        active = next((active_dir / name for name in variants if (active_dir / name).is_file()), None)
        if active is None:
            raise ValueError(f'Missing active map: {video}')
        fields, rows = read(active)
        _, official = read(official_path)
        try:
            updated, stats = align(rows, official)
        except ValueError as exc:
            records.append({'video': video, 'active_map': str(active), 'status': 'needs_review', 'error': str(exc)})
            continue
        record = {'video': video, 'active_map': str(active), **stats, 'status': 'unchanged'}
        targets = [active]
        original_dir = local_path(source['data_root']) / 'map-keyframes'
        original = next((original_dir / name for name in variants if (original_dir / name).is_file()), None)
        if original is not None and original.resolve() != active.resolve():
            if read(original)[1] != rows:
                raise ValueError(f'Active/source map diverged: {video}')
            targets.append(original)
        needs_change = any(a['fps'] != b['fps'] or a['frame_idx'] != b['frame_idx'] for a, b in zip(rows, updated))
        if needs_change:
            record['status'] = 'updated' if args.apply else 'would_update'
            output = io.StringIO(newline='')
            writer = csv.DictWriter(output, fieldnames=fields + [f for f in ('source_frame_idx', 'source_fps') if f not in fields])
            writer.writeheader()
            writer.writerows(updated)
            for target in targets:
                plans.append((target, target.read_bytes(), output.getvalue().encode('utf-8')))
        records.append(record)
    args.report_dir.mkdir(parents=True, exist_ok=False)
    journal = []
    for i, (target, before, after) in enumerate(plans):
        if args.apply:
            if target.read_bytes() != before:
                raise RuntimeError(f'Map changed during audit: {target}')
            backup = args.report_dir / 'backups' / f'{i:04d}_{target.name}'
            backup.parent.mkdir(exist_ok=True)
            backup.write_bytes(before)
            temporary = target.with_suffix('.csv.align-tmp')
            with temporary.open('xb') as handle:
                handle.write(after)
            temporary.replace(target)
            assert target.read_bytes() == after
            journal.append({'path': str(target), 'backup': str(backup),
                            'before_sha256': hashlib.sha256(before).hexdigest(),
                            'after_sha256': hashlib.sha256(after).hexdigest()})
            (args.report_dir / 'journal.json').write_text(json.dumps(journal, indent=2), encoding='utf-8')
    summary = {'created_utc': datetime.now(timezone.utc).isoformat(), 'applied': args.apply,
               'official_videos': len(officials), 'videos_changed': sum(r['status'] in ('updated', 'would_update') for r in records),
               'idx_rows_changed': sum(r.get('idx_changed', 0) for r in records),
               'files_changed': len(plans), 'needs_review': sum(r['status'] == 'needs_review' for r in records), 'records': records,
               'method': 'Preserve n/pts_time. Match BTC int(float(pts_time)*float(official_fps)); preserve decoded source_frame_idx/source_fps. Non-anchor IDX is convention-derived, not image-verified.'}
    (args.report_dir / 'report.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in summary.items() if k != 'records'}, indent=2))


if __name__ == '__main__':
    main()
