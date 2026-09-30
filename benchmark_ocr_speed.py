"""Read-only OCR throughput measurement on frames spread across video folders."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--keyframes', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=48)
    parser.add_argument('--remaining', type=int, default=482058)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    from ocr_index import (_read_image, _prepare_ocr_image, _easyocr_read_group,
                           _finish_ocr, OCR_PREFETCH_WORKERS)
    from shared import get_ocr, OCR_BATCH_SIZE, OCR_ENHANCE, OCR_PIPELINE_VERSION
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this throughput benchmark')
    directories = sorted(p for p in args.keyframes.iterdir() if p.is_dir())
    paths = []
    for i in range(min(args.samples, len(directories))):
        directory = directories[i * len(directories) // min(args.samples, len(directories))]
        with os.scandir(directory) as entries:
            path = next((Path(e.path) for e in entries if e.name.lower().endswith('.jpg')), None)
        if path:
            paths.append(path)
    if not paths:
        raise RuntimeError('No sample frames found')
    reader = get_ocr()
    # Warm up models before timing; no database is opened or changed.
    source = _prepare_ocr_image(_read_image(paths[0]))
    _easyocr_read_group(reader, [dict(source=source, w=source.shape[1], h=source.shape[0])])
    torch.cuda.synchronize()
    start = time.perf_counter()
    readings = []
    with ThreadPoolExecutor(max_workers=OCR_PREFETCH_WORKERS) as pool:
        for offset in range(0, len(paths), OCR_BATCH_SIZE):
            batch_paths = paths[offset:offset + OCR_BATCH_SIZE]
            images = list(pool.map(_read_image, batch_paths))
            groups = {}
            for path, image in zip(batch_paths, images):
                if image is None:
                    raise RuntimeError(f'Cannot read {path}')
                source = _prepare_ocr_image(image)
                h, w = source.shape[:2]
                groups.setdefault((h, w), []).append(dict(source=source, original=image, w=w, h=h, path=path))
            for items in groups.values():
                results = _easyocr_read_group(reader, items)
                for item, (texts, error) in zip(items, results):
                    if error:
                        raise RuntimeError(error)
                    texts = _finish_ocr(item['original'], texts)
                    readings.append(dict(file=str(item['path']), texts=texts))
            print(f'Measured {min(offset + OCR_BATCH_SIZE, len(paths))}/{len(paths)}', flush=True)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    fps = len(paths) / elapsed
    report = dict(samples=len(paths), seconds=elapsed, frames_per_second=fps,
                  projected_hours=args.remaining / fps / 3600,
                  required_fps=args.remaining / (30 * 3600), batch_size=OCR_BATCH_SIZE,
                  enhance=OCR_ENHANCE, pipeline_version=OCR_PIPELINE_VERSION,
                  gpu=torch.cuda.get_device_name(), readings=readings)
    if args.output:
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'readings'}, indent=2), flush=True)


if __name__ == '__main__':
    main()
