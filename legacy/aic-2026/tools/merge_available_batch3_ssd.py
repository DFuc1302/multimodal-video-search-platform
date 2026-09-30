"""Merge unpacked Batch 3 + existing index + ASR onto SSD, without moving frames."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', type=Path, default=Path('/mnt/d/AIC_26/data/batch3_downloads'))
    parser.add_argument('--old-index', type=Path, default=Path('/mnt/d/AIC_26/data/merged_search_index'))
    parser.add_argument('--asr', type=Path, default=Path('/mnt/d/AIC_26/data/asr_MS'))
    parser.add_argument('--output', type=Path, default=Path('/mnt/c/PERSONAL_PROJECT/AIC_26/final_search_index'))
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if os.name == 'nt':
        parser.error('Chạy script trong WSL Ubuntu, không chạy Python Windows.')
    from import_colab_batch3 import build_manifest
    expected = {f'Videos_M{i:02d}_processed' for i in range(1, 11)} | {'Video_N031-N040_processed'}
    groups = {}
    for root in sorted(args.downloads.iterdir()):
        if not root.is_dir():
            continue
        name = re.sub(r'-\d{3}$', '', root.name)
        if name not in expected:
            continue
        if name in groups:
            raise RuntimeError(f'Trùng nhóm {name}; kiểm tra hai thư mục trước khi gộp.')
        for item in ('meta.db', 'faiss_index', 'keyframes', 'map-keyframes', 'serve_manifest.json', 'batch3_receipt.json'):
            if not (root / item).exists():
                raise FileNotFoundError(root / item)
        groups[name] = root.resolve()
    if not groups:
        raise RuntimeError('Không thấy nhóm Batch 3 đã giải nén.')
    missing = sorted(expected - groups.keys())
    visual = [args.old_index.resolve(), *groups.values()]
    if not (args.asr / 'asr_cache.json').is_file():
        raise FileNotFoundError(args.asr / 'asr_cache.json')
    # ASR first: external merging keeps the first transcript on collisions.
    sources = [args.asr.resolve(), *visual]
    print('Nguồn dữ liệu:', *map(str, sources), sep='\n  ', flush=True)
    print('Nhóm còn thiếu:', ', '.join(missing) or 'Không', flush=True)
    print('SSD output:', args.output, flush=True)
    print('Keyframe giữ nguyên. ASR mới được ưu tiên nếu trùng tên video với ASR cũ.', flush=True)
    if args.output.exists():
        raise FileExistsError(f'{args.output} đã tồn tại. Chọn --output mới; không ghi đè index.')
    manifest = build_manifest(visual)
    for source in manifest['sources']:
        for field in ('keyframes_dir', 'map_dir'):
            if not Path(source[field]).is_dir():
                raise FileNotFoundError(source[field])
    if args.plan_only:
        print('PLAN OK — chưa gộp hoặc sửa dữ liệu.')
        return
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # All work remains in a staging directory until maps and manifest are ready.
    with tempfile.TemporaryDirectory(prefix='.merge-ssd-', dir=args.output.parent) as temp:
        staging = Path(temp) / 'index'
        manifest_file = Path(temp) / 'external_storage.json'
        manifest_file.write_text(json.dumps(manifest), encoding='utf-8')
        env = {**os.environ, 'BACKEND_MODE': 'sqlite', 'STORAGE_BACKEND': 'sqlite', 'AIC_USE_ACTIVE_DATASET': '0'}
        subprocess.run([sys.executable, '-u', str(Path(__file__).with_name('merge_processed_data.py')),
                        '--sources', *map(str, sources), '--output', str(staging),
                        '--asset-mode', 'external', '--external-storage-manifest', str(manifest_file)],
                       env=env, check=True)
        for number, source in enumerate(manifest['sources']):
            relative = Path('map-keyframes') / f'source_{number}'
            print(f'Copy timestamp maps lên SSD: {source["map_dir"]}', flush=True)
            shutil.copytree(source['map_dir'], staging / relative)
            source['map_dir'] = str(args.output.resolve() / relative)
        (staging / 'external_storage.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        report = dict(visual_sources=list(map(str, visual)), asr_source=str(args.asr),
                      included_groups=sorted(groups), missing_groups=missing,
                      complete_batch3=not missing, keyframes_moved=False)
        (staging / 'merge_scope.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        staging.rename(args.output)
    print(f'HOÀN TẤT: {args.output}', flush=True)
    print('Chưa thay cấu hình UI. Giữ các thư mục keyframe nguồn.', flush=True)


if __name__ == '__main__':
    main()
