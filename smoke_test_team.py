"""Packaging check using separate ports and a temporary Redis snapshot copy."""
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import tempfile
import time
from urllib.parse import urlencode, quote
from urllib.request import urlopen

root = Path(__file__).resolve().parent
data = root.parent / 'DATA_FOR_TEAM'


def get(path):
    with urlopen('http://127.0.0.1:18081' + path, timeout=180) as response:
        return json.load(response)


with tempfile.TemporaryDirectory(prefix='aic-team-test-') as temporary:
    shutil.copytree(data / 'redis_data', Path(temporary) / 'redis')
    env = dict(os.environ, SERVER_PORT='18081', REDIS_PORT='16381',
               TEAM_REDIS_DATA=str(Path(temporary) / 'redis'))
    with (root / 'smoke_test.log').open('w') as log:
        process = subprocess.Popen(['bash', str(root / 'run_local_server_wsl.sh')],
                                   env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            deadline = time.monotonic() + 900
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Server failed; see smoke_test.log')
                try:
                    ready = get('/ready')
                    if ready['ready']:
                        break
                except Exception:
                    pass
                if time.monotonic() > deadline:
                    raise RuntimeError('Startup timed out')
                time.sleep(2)
            print('READY', ready, flush=True)
            status = get('/status')
            assert status['indexed'] == 177321 and status['backend'] == 'hybrid'
            for params in ({'video':'L26_V001', 'top_k':10},
                           {'ocr_query':'hei', 'top_k':10},
                           {'query':'cooking', 'top_k':10}):
                result = get('/search?' + urlencode(params))
                assert result['results'], result
                print('SEARCH', params, len(result['results']), flush=True)
            file = result['results'][0]['file']
            with urlopen('http://127.0.0.1:18081/image/' + quote(file), timeout=30) as response:
                assert response.status == 200
            result = get('/summary/search?' + urlencode({'query':'cooking','top_k':3}))
            assert result['results'], result
            print('SUMMARY', len(result['results']), flush=True)
            from urllib.request import Request
            request = Request('http://127.0.0.1:18081/asr/search',
                              data=json.dumps({'phrase':'xin chao','limit':3}).encode(),
                              headers={'Content-Type':'application/json'})
            with urlopen(request, timeout=180) as response:
                assert response.status == 200
            print('ASR OK', flush=True)
        finally:
            os.killpg(process.pid, signal.SIGINT)
            process.wait(timeout=90)
        for port in (18081, 16381):
            with socket.socket() as sock:
                assert sock.connect_ex(('127.0.0.1', port)) != 0, port
        print('PASS: ports closed after Ctrl+C', flush=True)
