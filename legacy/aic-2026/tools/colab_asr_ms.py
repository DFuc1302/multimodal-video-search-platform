"""Standalone Colab ASR for M (batch 3) and S (batch 2); excludes N."""
import argparse
import json
import math
import re
import shutil
import tempfile
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def prefetched(videos):
    """Copy just one video ahead while the GPU transcribes the current video."""
    with tempfile.TemporaryDirectory(prefix='asr-') as temp, ThreadPoolExecutor(max_workers=1) as pool:
        def copy(video):
            local = Path(temp) / video.name
            shutil.copyfile(video, local)
            return local
        future = pool.submit(copy, videos[0]) if videos else None
        for index, video in enumerate(videos):
            local = future.result()
            if index + 1 < len(videos):
                future = pool.submit(copy, videos[index + 1])
            try:
                yield video, local
            finally:
                local.unlink(missing_ok=True)


def duration_seconds(path):
    import av
    with av.open(str(path)) as media:
        audio = next(iter(media.streams.audio), None)
        if audio is not None and audio.duration is not None and audio.time_base is not None:
            duration = float(audio.duration * audio.time_base)
        else:
            duration = float(media.duration or 0) / av.time_base
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'Không đọc được thời lượng: {path}')
    return duration


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def discover(roots):
    videos = []
    for family, root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        selected = [p for p in root.rglob('*') if p.is_file()
                    and p.suffix.lower() in {'.mp4', '.mkv', '.avi', '.mov', '.webm', '.m4v'}
                    and re.match(r'^' + family + r'\d+[-_]V\d+', p.stem, re.I)]
        print(f'{family}: {len(selected)} video — {root}', flush=True)
        if not selected:
            raise RuntimeError(f'Không tìm thấy video {family} dạng {family}01_V001.mp4. Kiểm tra tên và giải nén nếu cần.')
        videos.extend(selected)
    names = [p.name.casefold() for p in videos]
    if len(names) != len(set(names)):
        raise RuntimeError('Trùng tên video; cần kiểm tra trước khi tạo cache.')
    return sorted(videos)


def fixed_windows(segments):
    buckets = {}
    for segment in segments:
        words = segment.words or []
        if segment.text.strip() and not words:
            raise RuntimeError('ASR thiếu word timestamps; không ghi kết quả sai mốc thời gian.')
        for word in words:
            start, end = float(word.start), float(word.end)
            if not (math.isfinite(start) and math.isfinite(end) and 0 <= start <= end):
                raise ValueError('Word timestamp không hợp lệ')
            buckets.setdefault(int(((start + end) / 2) // 2), []).append(word.word)
    result = []
    for index, tokens in sorted(buckets.items()):
        text = ' '.join(' '.join(tokens).split())
        text = re.sub(r'\s+([,.;:!?%\)\]])', r'\1', text).strip()
        search = unicodedata.normalize('NFD', text.lower().replace('đ', 'd'))
        search = ' '.join(''.join(c for c in search if unicodedata.category(c) != 'Mn').split())
        if search:
            result.append(dict(cache_version=2, text=text, search_text=search,
                               start=float(index * 2), end=float(index * 2 + 2)))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--m-root', default='/content/drive/MyDrive/DATA_AIC/batch 3/videos')
    parser.add_argument('--s-root', default='/content/drive/MyDrive/DATA_AIC/batch 2')
    parser.add_argument('--output', default='/content/drive/MyDrive/DATA_AIC/asr_MS')
    parser.add_argument('--model', default='large-v3-turbo')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--language', default='vi')
    args = parser.parse_args()
    from tqdm.auto import tqdm
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    import ctranslate2
    if not ctranslate2.get_cuda_device_count():
        raise RuntimeError('Hãy chọn GPU trong Runtime > Change runtime type.')
    videos = discover([('M', Path(args.m_root)), ('S', Path(args.s_root))])
    output = Path(args.output)
    recipe = {'model': args.model, 'language': args.language, 'beam_size': 1,
              'word_timestamps': True, 'vad_filter': True, 'format': 2}
    def identity(p):
        stat = p.stat()
        return dict(path=str(p), size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    def checkpoint(p):
        return output / 'videos' / (p.name + '.json')
    def read_done(p):
        try:
            data = json.loads(checkpoint(p).read_text(encoding='utf-8'))
            if data['source'] == identity(p) and data['recipe'] == recipe and isinstance(data['segments'], list):
                return data
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None
    done = {p: read_done(p) for p in videos}
    sources = {p: identity(p) for p in videos}
    durations = {p: duration_seconds(p) for p in tqdm(videos, desc='Đọc thời lượng (không benchmark)', unit='video')}
    pending = [p for p in videos if done[p] is None]
    print(f'Đã xong {len(videos)-len(pending)}/{len(videos)}; còn {len(pending)}', flush=True)
    pipeline = None
    if pending:
        pipeline = BatchedInferencePipeline(WhisperModel(args.model, device='cuda', compute_type='float16'))
    failures = {}
    completed_seconds = sum(durations[p] for p in videos if done[p] is not None)
    with tqdm(total=sum(durations.values()), initial=completed_seconds, desc='TỔNG ASR', unit='s',
              bar_format='{desc}: {percentage:6.2f}%|{bar}| {n:.0f}/{total:.0f}s [{elapsed}<{remaining}]',
              mininterval=2) as timeline, \
         tqdm(total=len(videos), initial=len(videos)-len(pending), desc='Video đã lưu', unit='video') as overall:
        for video, local in prefetched(pending):
            overall.set_postfix_str(video.name)
            try:
                before = sources[video]
                print(f'ASR: {video.name} (video kế tiếp được copy song song)', flush=True)
                segments, info = pipeline.transcribe(str(local), batch_size=args.batch_size,
                    language=None if args.language == 'auto' else args.language,
                    beam_size=1, word_timestamps=True, vad_filter=True,
                    condition_on_previous_text=False)
                collected = []
                with tqdm(total=info.duration, desc=video.name, unit='s', leave=False) as progress:
                    for segment in segments:
                        collected.append(segment)
                        progress.update(max(0, min(info.duration, segment.end)-progress.n))
                        # Reserve the final 1% until the checkpoint is safely saved.
                        target = completed_seconds + durations[video] * min(0.99, progress.n / max(info.duration, 0.001))
                        timeline.update(target - timeline.n)
                    progress.update(max(0, info.duration-progress.n))
                result = fixed_windows(collected)
                if identity(video) != before:
                    raise RuntimeError('Video thay đổi trong lúc đọc; hãy chờ upload hoàn tất rồi chạy lại.')
                payload = dict(source=before, recipe=recipe, segments=result)
                write_json(checkpoint(video), payload)
                done[video] = payload
                completed_seconds += durations[video]
                timeline.n = completed_seconds
                timeline.refresh()
                overall.update(1)
            except Exception as exc:
                timeline.n = completed_seconds
                timeline.refresh()
                failures[video.name] = repr(exc)
                tqdm.write(f'LỖI {video.name}: {exc}')
                write_json(output / 'errors.json', failures)
                if 'out of memory' in str(exc).lower() or 'cuda' in str(exc).lower():
                    print('Dừng lỗi GPU. Nếu thiếu VRAM: giảm BATCH_SIZE xuống 8 rồi chạy lại.', flush=True)
                    break
    cache = {'video-name-v1::' + p.name: data['segments'] for p, data in done.items() if data is not None}
    complete = len(cache) == len(videos)
    target = output / ('asr_cache.json' if complete else 'asr_cache.partial.json')
    write_json(target, cache)
    write_json(output / 'status.json', dict(completed=len(cache), total=len(videos), complete=complete, errors=failures))
    print(f'Đã lưu {len(cache)}/{len(videos)} video: {target}', flush=True)
    if not complete:
        raise RuntimeError('Chưa xong toàn bộ. Xem errors.json; chạy lại để tiếp tục các video chưa xong.')
    print('HOÀN TẤT — chỉ cần tải asr_cache.json; không có video trong kết quả.', flush=True)


if __name__ == '__main__':
    main()
