"""DRES v2 adapter. Credentials stay in memory; submissions are never retried.

The local SQLite ledger survives restarts, including an indeterminate send.
Only a server-built, short-lived preview can be submitted.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

from fastapi import APIRouter, HTTPException, Request
from dres_frames import FrameError, FrameResolver, build_answer


class RemoteError(Exception):
    def __init__(self, message, code=502, uncertain=False):
        self.message, self.code, self.uncertain = message, code, uncertain
        super().__init__(message)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def base_url(value):
    value = str(value).strip().rstrip('/')
    parsed = urllib.parse.urlsplit(value)
    if (parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password
            or parsed.query or parsed.fragment or parsed.path):
        raise FrameError('Địa chỉ DRES phải là HTTPS, chỉ gồm tên miền và cổng nếu có.')
    return value


def remote(base, path, session=None, body=None):
    url = base + path
    if session:
        url += '?' + urllib.parse.urlencode({'session': session})
    request = urllib.request.Request(url, method='POST' if body is not None else 'GET',
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={'Content-Type': 'application/json', 'Accept': 'application/json'})
    try:
        with urllib.request.build_opener(NoRedirect()).open(request, timeout=20) as response:
            status = response.status
            result = json.loads(response.read(1_000_000))
            return status, result
    except urllib.error.HTTPError as exc:
        # Do not expose the request URL: it contains the remote session token.
        detail = {400: 'Dữ liệu bị DRES từ chối.', 401: 'Session hết hạn hoặc tài khoản không hợp lệ.',
                  403: 'Tài khoản không có quyền.', 404: 'Không tìm thấy phiên hoặc câu đang thi.',
                  412: 'DRES từ chối nộp bài: kiểm tra câu đang thi hoặc đáp án trùng.'}.get(exc.code, 'DRES trả về lỗi.')
        raise RemoteError(f'{detail} HTTP {exc.code}.', exc.code,
                          uncertain=exc.code not in {400, 401, 403, 404, 412}) from None
    except (OSError, ValueError, TimeoutError):
        raise RemoteError('Không xác định được phản hồi DRES. Kiểm tra kết nối và lịch sử trên DRES; không tự gửi lại.',
                          uncertain=True) from None


class SubmissionLedger:
    def __init__(self, path):
        self.path = Path(path)
        self.lock = threading.Lock()

    def connect(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute('''CREATE TABLE IF NOT EXISTS submissions (
            fingerprint TEXT PRIMARY KEY, created REAL, account TEXT, evaluation TEXT,
            task TEXT, mode TEXT, payload TEXT, state TEXT, response TEXT)''')
        conn.commit()
        return conn

    def reserve(self, account, evaluation, task, payload, mode):
        canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        fingerprint = hashlib.sha256(json.dumps([account, evaluation, task, canonical]).encode()).hexdigest()
        with self.lock, closing(self.connect()) as conn, conn:
            conn.execute('BEGIN IMMEDIATE')
            prior = conn.execute('SELECT state FROM submissions WHERE fingerprint=?', (fingerprint,)).fetchone()
            if prior and prior['state'] != 'rejected':
                raise FrameError('Đáp án đã gửi hoặc chưa rõ kết quả trong câu này. Không gửi trùng; xem lịch sử DRES.')
            conn.execute('INSERT OR REPLACE INTO submissions VALUES (?,?,?,?,?,?,?,?,?)',
                         (fingerprint, time.time(), account, evaluation, task, mode, canonical, 'sending', '{}'))
        return fingerprint

    def finish(self, fingerprint, state, response):
        with self.lock, closing(self.connect()) as conn, conn:
            conn.execute('UPDATE submissions SET state=?,response=? WHERE fingerprint=?',
                         (state, json.dumps(response, ensure_ascii=False), fingerprint))

    def history(self, account):
        with closing(self.connect()) as conn:
            rows = conn.execute('SELECT * FROM submissions WHERE account=? ORDER BY created DESC LIMIT 100', (account,)).fetchall()
        return [{**{key: row[key] for key in ('created', 'evaluation', 'task', 'mode', 'state')},
                 'payload': json.loads(row['payload']), 'response': json.loads(row['response'])} for row in rows]


class DresService:
    def __init__(self, find_map, ledger, transport=remote):
        self.resolver = FrameResolver(find_map)
        self.ledger = SubmissionLedger(ledger)
        self.transport = transport
        self.connections, self.previews = {}, {}
        self.lock = threading.RLock()

    def cleanup(self):
        now = time.time()
        self.connections = {k: v for k, v in self.connections.items() if v['expires'] > now}
        self.previews = {k: v for k, v in self.previews.items() if v['expires'] > now}

    def login(self, body):
        base = base_url(body.get('base_url', 'https://eventretrieval.one'))
        username, password = body.get('username', ''), body.get('password', '')
        if not isinstance(username, str) or not username.strip() or not isinstance(password, str) or not password:
            raise FrameError('Nhập tài khoản và mật khẩu DRES.')
        _, account = self.transport(base, '/api/v2/login', body={'username': username.strip(), 'password': password})
        if not isinstance(account, dict) or not account.get('sessionId'):
            raise RemoteError('DRES không trả về session hợp lệ.')
        token = secrets.token_urlsafe(32)
        name = str(account.get('username') or username.strip())
        with self.lock:
            self.cleanup()
            self.connections[token] = {'base': base, 'session': account['sessionId'], 'username': name,
                                       'account': base + '|' + name, 'expires': time.time() + 12 * 3600}
        return {'connection': token, 'username': name, 'base_url': base}

    def connection(self, token):
        with self.lock:
            self.cleanup()
            value = self.connections.get(token)
        if not value:
            raise HTTPException(401, 'Đăng nhập DRES trong tab này trước (đăng nhập lại sau khi server khởi động lại).')
        return value

    def evaluations(self, token):
        conn = self.connection(token)
        _, result = self.transport(conn['base'], '/api/v2/client/evaluation/list', session=conn['session'])
        if not isinstance(result, list):
            raise RemoteError('Danh sách phiên DRES không hợp lệ.')
        return [{key: item.get(key) for key in ('id', 'name', 'status', 'type')} for item in result if isinstance(item, dict)]

    def current(self, token, evaluation):
        if not isinstance(evaluation, str) or not evaluation:
            raise FrameError('Chọn phiên thi đang ACTIVE.')
        conn = self.connection(token)
        active = any(e.get('id') == evaluation and e.get('status') == 'ACTIVE' for e in self.evaluations(token))
        if not active:
            raise FrameError('Phiên đã chọn không ACTIVE hoặc không thuộc tài khoản này.')
        path = '/api/v2/client/evaluation/currentTask/' + urllib.parse.quote(evaluation, safe='')
        _, task = self.transport(conn['base'], path, session=conn['session'])
        if not isinstance(task, dict) or not task.get('name'):
            raise FrameError('DRES chưa có câu đang thi. Chưa thể nộp thật.')
        return {key: task.get(key) for key in ('name', 'taskGroup', 'taskType', 'duration')}

    def preview(self, body, token):
        result = build_answer(self.resolver, body.get('task_type'), body.get('video', ''),
                              body.get('frames'), body.get('answer', ''))
        live = body.get('live') is True
        task, evaluation = None, ''
        if live:
            evaluation = body.get('evaluation_id', '')
            task = self.current(token, evaluation)
            # Bind the answer to the task; DRES rejects stale task names even if
            # it advances between the final currentTask read and POST.
            result['payload']['answerSets'][0]['taskName'] = task['name']
        preview_id = secrets.token_urlsafe(24)
        with self.lock:
            self.cleanup()
            self.previews[preview_id] = {**result, 'live': live, 'token': token,
                                        'evaluation_id': evaluation, 'task': task, 'expires': time.time() + 120}
        return {**result, 'preview_id': preview_id, 'live': live, 'task': task, 'expires_in': 120}

    def submit(self, preview_id, token):
        with self.lock:
            self.cleanup()
            preview = self.previews.get(preview_id)
            if preview is None or preview['token'] != token:
                raise FrameError('Bản xem trước hết hạn hoặc không thuộc phiên này. Kiểm tra đáp án lại.')
            # Consume once; a second click cannot send a second request.
            del self.previews[preview_id]
        if not preview['live']:
            return {'state': 'simulated', 'message': 'Mô phỏng thành công — không gửi lên DRES.', 'payload': preview['payload']}
        conn = self.connection(token)
        current = self.current(token, preview['evaluation_id'])
        if current != preview['task']:
            raise FrameError('Câu thi đã thay đổi. Kiểm tra lại đáp án cho câu mới.')
        task_key = json.dumps(current, sort_keys=True, ensure_ascii=False)
        fingerprint = self.ledger.reserve(conn['account'], preview['evaluation_id'], task_key,
                                          preview['payload'], preview['task_type'])
        try:
            code, response = self.transport(conn['base'], '/api/v2/submit/' + urllib.parse.quote(preview['evaluation_id'], safe=''),
                                             session=conn['session'], body=preview['payload'])
        except RemoteError as exc:
            state = 'unknown' if exc.uncertain else 'rejected'
            response = {'message': exc.message, 'http_status': exc.code}
            self.ledger.finish(fingerprint, state, response)
            return {'state': state, **response}
        # Only pass known non-secret response fields into persistent history.
        if isinstance(response, dict):
            clean = {k: response[k] for k in ('status', 'submission', 'description') if k in response}
        else:
            clean = {}
        for key, value in list(clean.items()):
            if isinstance(value, str):
                clean[key] = value.replace(conn['session'], '[redacted]')
        clean['http_status'] = code
        state = 'received' if isinstance(response, dict) and response.get('status') is True else 'unknown'
        self.ledger.finish(fingerprint, state, clean)
        return {'state': state, 'response': clean,
                'message': 'DRES đã nhận; xem verdict bên dưới. Đã nhận không đồng nghĩa đáp án đúng.' if state == 'received'
                else 'Chưa xác định kết quả. Kiểm tra lịch sử DRES trước khi thao tác tiếp.'}


def create_router(find_map, ledger_path=None, service=None):
    service = service or DresService(find_map, ledger_path or os.getenv('DRES_HISTORY_PATH', str(Path(__file__).parent / 'runtime' / 'dres_history.sqlite')))
    router = APIRouter(prefix='/dres', tags=['DRES'])

    def invoke(call, *args):
        try:
            return call(*args)
        except FrameError as exc:
            raise HTTPException(400, str(exc)) from None
        except RemoteError as exc:
            raise HTTPException(502, exc.message) from None

    def token(request):
        return request.headers.get('X-DRES-Connection', '')

    @router.post('/login')
    def login(body: dict):
        return invoke(service.login, body)

    @router.post('/disconnect')
    def disconnect(request: Request):
        with service.lock:
            service.connections.pop(token(request), None)
        return {'disconnected': True}

    @router.get('/evaluations')
    def evaluations(request: Request):
        return {'evaluations': invoke(service.evaluations, token(request))}

    @router.get('/current')
    def current(request: Request, evaluation_id: str):
        return invoke(service.current, token(request), evaluation_id)

    @router.post('/preview')
    def preview(body: dict, request: Request):
        return invoke(service.preview, body, token(request))

    @router.post('/submit')
    def submit(body: dict, request: Request):
        return invoke(service.submit, str(body.get('preview_id', '')), token(request))

    @router.get('/history')
    def history(request: Request):
        conn = service.connection(token(request))
        return {'history': service.ledger.history(conn['account'])}

    return router
