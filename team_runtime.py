"""Own the two local server processes and shut them down together."""
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

import redis


def main():
    root = Path(os.environ['TEAM_DATA']).resolve()
    ports = [int(os.environ['REDIS_PORT']), int(os.environ['SERVER_PORT'])]
    for port in ports:
        with socket.socket() as sock:
            sock.bind(('127.0.0.1', port))
    for path in ('search_index_ocr/meta.db', 'search_index_ocr/serve_manifest.json',
                 'search_index_ocr/faiss_index/frames.faiss',
                 'search_index_ocr/clip_text/config.json',
                 'models/multilingual-e5-small/config.json'):
        if not (root / path).is_file():
            raise RuntimeError(f'Missing processed data: {root / path}')
    if not Path(os.environ['MAPFILES_DIR']).is_dir():
        raise RuntimeError('Missing filtered_data_all_873_moved/map-keyframes')
    binary = root / 'redis_runtime/usr/bin/redis-server'
    binary.chmod(binary.stat().st_mode | 0o111)
    modules = root / 'redis_runtime/usr/lib/redis/modules'
    command = [str(binary), '--bind', '127.0.0.1', '--protected-mode', 'yes',
               '--port', str(ports[0]), '--dir', os.getenv('TEAM_REDIS_DATA', str(root / 'redis_data')),
               '--appendonly', 'yes', '--appendfsync', 'everysec', '--save', '',
               '--maxmemory-policy', 'noeviction']
    for name in ('rejson.so', 'redisearch.so'):
        command += ['--loadmodule', str(modules / name)]
    command += ['--search-max-search-results', '1000000', '--search-timeout', '30000',
                '--search-on-timeout', 'fail']
    children = []

    def interrupt(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, interrupt)
    try:
        children.append(subprocess.Popen(command, start_new_session=True))
        client = redis.Redis.from_url(os.environ['REDIS_URL'], socket_timeout=5)
        deadline = time.monotonic() + 900
        print('Dang nap Redis local...', flush=True)
        while True:
            if children[0].poll() is not None:
                raise RuntimeError('Redis stopped during startup')
            try:
                client.ping()
                break
            except (redis.exceptions.BusyLoadingError, redis.exceptions.ConnectionError):
                if time.monotonic() > deadline:
                    raise RuntimeError('Redis startup timed out')
                time.sleep(1)
        from redis_store import store
        from migrate_to_redis import verify
        store().manifest()
        report = verify(client, os.environ['REDIS_PREFIX'])
        if any(v['indexing'] or v['expected'] != v['indexed'] for v in report.values()):
            raise RuntimeError(f'Incomplete Redis indexes: {report}')
        print('Dang nap CLIP text encoder va kiem tra search readiness...', flush=True)
        children.append(subprocess.Popen([sys.executable, '-m', 'uvicorn', 'main:app',
                         '--host', '127.0.0.1', '--port', str(ports[1]), '--no-access-log'],
                         start_new_session=True))
        while all(child.poll() is None for child in children):
            time.sleep(0.5)
        if any(child.returncode not in (None, 0) for child in children):
            raise RuntimeError('A local server process exited with an error')
    except KeyboardInterrupt:
        pass
    finally:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        print('Da tat server va Redis local.', flush=True)


if __name__ == '__main__':
    main()
