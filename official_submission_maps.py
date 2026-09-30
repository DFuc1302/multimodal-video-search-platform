"""Read organizer maps independently of local extraction maps; never modify either."""
import io
import os
from pathlib import Path
import zipfile

from dres_frames import FrameError, read_map
from summary_catalog import canonical_video, local_path


class OfficialMaps:
    def __init__(self, source=None):
        self.source = local_path(source or os.getenv('DRES_OFFICIAL_MAPS') or
                                 'D:/AIC_26/data/aic26-b2-map-keyframes.zip')

    def read(self, video):
        key = canonical_video(video)
        try:
            if self.source.is_file() and self.source.suffix.lower() == '.zip':
                with zipfile.ZipFile(self.source) as archive:
                    matches = [n for n in archive.namelist() if n.lower().endswith('.csv')
                               and canonical_video(n) == key]
                    if len(matches) != 1:
                        raise FrameError(f'Map BTC của {video}: tìm thấy {len(matches)} file; cần đúng một file.')
                    name = matches[0]
                    rows, errors = read_map(io.StringIO(archive.read(name).decode('utf-8-sig'), newline=None))
                    source = f'{self.source}!{name}'
            elif self.source.is_dir():
                matches = [p for p in self.source.rglob('*.csv') if canonical_video(p.stem) == key]
                if len(matches) != 1:
                    raise FrameError(f'Map BTC của {video}: tìm thấy {len(matches)} file; cần đúng một file.')
                name = str(matches[0])
                rows, errors = read_map(matches[0])
                source = name
            else:
                raise FrameError(f'Không tìm thấy map BTC: {self.source}. Cấu hình DRES_OFFICIAL_MAPS.')
        except (OSError, zipfile.BadZipFile, UnicodeError) as exc:
            raise FrameError(f'Không đọc được map BTC: {exc}') from None
        fatal = [e for e in errors if 'n hoặc frame_idx bị trùng' not in e]
        if fatal:
            raise FrameError('Map BTC không hợp lệ: ' + '; '.join(fatal[:3]))
        return Path(name).stem, source, rows

    def align(self, mapping):
        video, source, official = self.read(mapping['video_id'])
        result, warnings = [], []
        for local in mapping['frames']:
            # Join by presentation timestamp, never by sampled-image ordinal or FPS multiplication.
            distance = min(abs(r['time_ms'] - local['time_ms']) for r in official)
            candidates = [r for r in official if abs(r['time_ms'] - local['time_ms']) == distance]
            if distance > 1000:
                raise FrameError(f"idx {local['frame_idx']} cách keyframe BTC gần nhất {distance} ms (> 1000 ms). Chọn frame khác.")
            if len({r['frame_idx'] for r in candidates}) != 1:
                raise FrameError(f"idx {local['frame_idx']} nằm giữa hai keyframe BTC. Chọn frame khác để tránh ánh xạ mơ hồ.")
            row = candidates[0]
            same_frame = [r for r in official if r['frame_idx'] == row['frame_idx']]
            same_n = [r for r in official if r['n'] == row['n']]
            if len({r['pts_time'] for r in same_frame}) > 1 or len({r['frame_idx'] for r in same_n}) > 1:
                raise FrameError(f"Map BTC không hợp lệ tại frame {row['frame_idx']}: ánh xạ trùng, mâu thuẫn. Chọn frame khác.")
            result.append({**local, 'official_frame_idx': row['frame_idx'], 'official_n': row['n'],
                           'official_time_ms': row['time_ms'], 'delta_ms': row['time_ms'] - local['time_ms']})
            if distance or row['frame_idx'] != local['frame_idx']:
                warnings.append(f"idx {local['frame_idx']} → frame BTC {row['frame_idx']}, lệch {row['time_ms'] - local['time_ms']:+d} ms. Kiểm tra đúng sự kiện trước khi nộp.")
        if any(b['official_frame_idx'] <= a['official_frame_idx'] for a, b in zip(result, result[1:])):
            raise FrameError('Các sự kiện trỏ tới frame BTC trùng nhau hoặc sai thứ tự. Chọn lại frame.')
        return {**mapping, 'video_id': video, 'official_source': source,
                'frames': result, 'official_warnings': warnings}
