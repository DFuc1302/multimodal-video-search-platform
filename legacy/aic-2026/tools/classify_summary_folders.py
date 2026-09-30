"""Classify M/S summaries using old category examples; review CSV before moving JSON."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import unicodedata

NEW_CATEGORIES = {'M': '10_Bản_tin', 'S': '11_Giải_Xe_Đạp', 'N': '12_Camera_an_ninh'}


def classify_rule(path, previous=''):
    match = re.match(r'^([MSN])\d+[-_]V\d+', path.stem, re.I)
    if match:
        category = NEW_CATEGORIES[match[1].upper()]
        return category, category, 'video_prefix', False
    if previous.startswith('09_'):
        return NEW_CATEGORIES['M'], NEW_CATEGORIES['M'], 'old_news_category', False
    text = ' '.join(text_of(path)).lower().replace('đ', 'd')
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    news = any(phrase in text for phrase in ('ban tin', 'thoi su', 'phong su', 'bien tap vien', 'nguoi dan chuong trinh tin tuc'))
    current = previous or '00_CAN_KIEM_TRA'
    return current, NEW_CATEGORIES['M'] if news else current, 'news_candidate' if news else 'keep_old_category', news


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_of(path):
    data = json.loads(path.read_text(encoding='utf-8-sig'))
    def flatten(value):
        if isinstance(value, dict):
            return ' '.join(flatten(v) for k, v in value.items() if k not in {'moc_thoi_gian', 'video_id'})
        if isinstance(value, list):
            return ' '.join(map(flatten, value))
        return str(value or '')
    blocks = [flatten(data.get('video_metadata', {}))]
    blocks += [flatten(s) for s in data.get('phan_doan_thoi_gian', [])]
    blocks = [s.strip() for s in blocks if s.strip()]
    if not blocks:
        raise ValueError(f'Summary rỗng: {path}')
    # Sample evenly across the whole video, not just its opening.
    indices = sorted({round(i*(len(blocks)-1)/min(11, len(blocks)-1)) for i in range(min(12, len(blocks)))}) if len(blocks)>1 else [0]
    return [blocks[i][:1800] for i in indices]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['plan', 'apply'])
    p.add_argument('--summaries', type=Path, default=Path('/mnt/d/AIC_26/data/summary_MS/video_summaries_json'))
    p.add_argument('--old', type=Path, default=Path('/mnt/c/PERSONAL_PROJECT/AIC_26/filtered_data_all_873_moved'))
    p.add_argument('--output', type=Path, default=Path('/mnt/d/AIC_26/data/classified_MS'))
    p.add_argument('--report', type=Path, default=Path('/mnt/d/AIC_26/data/summary_classification_12groups.csv'))
    p.add_argument('--extra-summary-root', type=Path, action='append', default=[], help='Additional folder containing summary JSONs, e.g. N summaries')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()
    categories = sorted(d.name for d in args.old.iterdir() if d.is_dir() and re.match(r'^\d{2}_', d.name))
    if not categories:
        raise RuntimeError('Không tìm thấy thư mục chủ đề dataset cũ.')
    allowed = set(categories) | set(NEW_CATEGORIES.values()) | {'00_CAN_KIEM_TRA'}
    if args.action == 'plan':
        if args.report.exists():
            raise FileExistsError(f'Giữ bảng đã duyệt: {args.report}. Chọn --report khác nếu muốn phân loại lại.')
        files = {f.resolve(): '' for root in [args.summaries, *args.extra_summary_root]
                 for f in root.rglob('*.json') if re.match(r'^[MSNL]\d+[-_]V\d+', f.stem, re.I)}
        for category in categories:
            for f in (args.old/category/'video_summaries_json').glob('*.json'):
                files[f.resolve()] = category
        # Also find previously classified new summaries so rules can be reapplied.
        for f in args.output.glob('*/video_summaries_json/*.json'):
            files[f.resolve()] = f.parent.parent.name
        if not files:
            raise RuntimeError('Không tìm thấy summary JSON.')
        rows = []
        for path, previous in sorted(files.items()):
            category, proposed, reason, review = classify_rule(path, previous)
            # Preserve unchanged old summaries in place to avoid unnecessary moves.
            destination = path if category == previous else args.output.resolve()/category/'video_summaries_json'/path.name
            rows.append(dict(video_id=path.stem, source=str(path.resolve()), sha256=digest(path),
                             suggested_category=proposed, category=category,
                             previous_category=previous, reason=reason, needs_review=int(review),
                             planned_destination=str(destination)))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        with args.report.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
        print(f'Đã tạo {args.report}. Duyệt/sửa cột category rồi chạy apply. Chưa di chuyển file.')
        return
    with args.report.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    moves, seen, targets = [], set(), set()
    roots = [args.summaries.resolve(), args.output.resolve(), *[r.resolve() for r in args.extra_summary_root]]
    for row in rows:
        if row['category'] not in allowed:
            raise ValueError(f'Chủ đề không hợp lệ: {row["category"]}')
        src = Path(row['source']).resolve()
        old_summary = src.parent.name == 'video_summaries_json' and src.parent.parent.parent == args.old.resolve()
        if (not any(src.is_relative_to(root) for root in roots) and not old_summary) or src.suffix.lower() != '.json':
            raise ValueError(f'Nguồn ngoài thư mục summary: {src}')
        dst = args.output.resolve()/row['category']/'video_summaries_json'/src.name
        if row['category'] == row.get('previous_category'):
            dst = src
        if src in seen:
            raise ValueError(f'Lặp nguồn: {src}')
        seen.add(src)
        if dst in targets:
            raise ValueError(f'Hai summary cùng đích; cần xử lý trùng trước: {dst}')
        targets.add(dst)
        if src == dst:
            if not src.is_file() or digest(src) != row['sha256']:
                raise ValueError(f'Nguồn đã đổi: {src}')
            continue
        if not src.exists() and dst.is_file() and digest(dst)==row['sha256']:
            continue
        if not src.is_file() or digest(src)!=row['sha256']:
            raise ValueError(f'Nguồn đã đổi hoặc thiếu: {src}')
        if dst.is_file() and digest(dst) == row['sha256']:
            moves.append((src, dst, row['sha256']))
            continue
        if dst.exists():
            raise FileExistsError(dst)
        moves.append((src, dst, row['sha256']))
    # Preflight all rows before performing any moves. Originals are deleted only after verification.
    for src, dst, checksum in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            with src.open('rb') as source, dst.open('xb') as target:
                shutil.copyfileobj(source, target)
        if digest(dst) != checksum:
            raise RuntimeError(f'Copy không khớp, giữ nguồn: {src}')
        src.unlink()
        print(f'{src.name} -> {dst.parent.parent.name}', flush=True)
    print(f'Đã chuyển {len(moves)} summary. Keyframe, video, map và index giữ nguyên.')


if __name__ == '__main__':
    main()
