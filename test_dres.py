"""DRES contract tests: no requests to a real competition server."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from dres_frames import FrameError, FrameResolver, build_answer
from dres_submission import DresService, RemoteError, SubmissionLedger, base_url, create_router


class FakeDres:
    def __init__(self):
        self.calls = []
        self.active = True
        self.task = {'name': 'Question 1', 'taskGroup': 'KIS', 'taskType': 'KIS', 'duration': 300}
        self.failure = None

    def __call__(self, base, path, session=None, body=None):
        self.calls.append((path, body))
        if path.endswith('/login'):
            return 200, {'username': 'test-team', 'sessionId': 'secret-remote-token'}
        if path.endswith('/list'):
            return 200, [{'id': 'eval-1', 'name': 'Practice', 'status': 'ACTIVE' if self.active else 'ENDED', 'type': 'SYNCHRONOUS'}]
        if '/currentTask/' in path:
            return 200, dict(self.task)
        if '/submit/' in path:
            if self.failure:
                raise self.failure
            return 202, {'status': True, 'submission': 'INDETERMINATE', 'description': 'accepted'}
        raise AssertionError(path)


class DresTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.csv = self.root / 'N001-V001.csv'
        self.csv.write_text('n,pts_time,fps,frame_idx\n1,0,25,0\n11,5.521,25,138\n12,9.021,25,200\n', encoding='utf8')
        self.find = lambda video: self.csv if video.casefold() == 'n001-v001' else None
        self.resolver = FrameResolver(self.find)
        self.remote = FakeDres()
        self.service = DresService(self.find, self.root / 'history.sqlite', self.remote)
        self.token = self.service.login({'username': 'test-team', 'password': 'not-real'})['connection']

    def body(self, **overrides):
        return {'task_type': 'textual_kis', 'video': 'n001-v001', 'frames': [138],
                'live': True, 'evaluation_id': 'eval-1', **overrides}

    def test_kis_uses_pts_not_fps_and_preserves_video_spelling(self):
        result = build_answer(self.resolver, 'textual_kis', 'n001-v001', [138])
        item = result['payload']['answerSets'][0]['answers'][0]
        self.assertEqual(item, {'mediaItemName': 'N001-V001', 'start': 5521, 'end': 5521})
        self.assertEqual(build_answer(self.resolver, 'video_kis', 'N001-V001', [0])['mapping']['frames'][0]['time_ms'], 0)

    def test_qa_and_trake_contracts(self):
        qa = build_answer(self.resolver, 'qa', 'N001-V001', [138], '  Thành phố Hồ Chí Minh  ')
        self.assertEqual(qa['payload']['answerSets'][0]['answers'][0]['text'], 'QA-Thành phố Hồ Chí Minh-N001-V001-5521')
        tr = build_answer(self.resolver, 'trake', 'N001-V001', [0, 138, 200])
        self.assertEqual(tr['payload']['answerSets'][0]['answers'][0]['text'], 'TR-N001-V001-0,138,200')

    def test_invalid_frames_and_paths_are_rejected(self):
        for frames in ([11], [-1], [1.5], [True], [138, 138], [], ['NaN'], ['Infinity']):
            with self.subTest(frames=frames), self.assertRaises(FrameError):
                self.resolver.resolve('N001-V001', frames)
        for video in ('../N001-V001', 'N001-V001.mp4', '/tmp/map', 'unknown'):
            with self.subTest(video=video), self.assertRaises(FrameError):
                self.resolver.resolve(video, [138])

    def test_task_constraints(self):
        for task, frames, answer in [('qa', [138], ''), ('qa', [138], 'a\nb'), ('textual_kis', [0, 138], ''),
                                      ('trake', [138, 0], ''), ('wrong', [138], '')]:
            with self.subTest(task=task), self.assertRaises(FrameError):
                build_answer(self.resolver, task, 'N001-V001', frames, answer)

    def test_conflicting_frame_is_blocked_other_frames_remain_usable(self):
        with self.csv.open('a') as handle:
            handle.write('13,0.033,25,0\n')
        with self.assertRaisesRegex(FrameError, 'nhiều pts_time'):
            self.resolver.resolve('N001-V001', [0])
        result = self.resolver.resolve('N001-V001', [138])
        self.assertTrue(result['map_warnings'])

    def test_bad_timestamp_or_nonmonotonic_map_blocks(self):
        for time_value in ('NaN', '-1', '0'):
            self.csv.write_text(f'n,pts_time,fps,frame_idx\n1,0,25,0\n2,{time_value},25,138\n')
            with self.assertRaises(FrameError):
                self.resolver.resolve('N001-V001', [138])

    def test_simulation_does_not_call_dres(self):
        before = len(self.remote.calls)
        preview = self.service.preview(self.body(live=False), '')
        result = self.service.submit(preview['preview_id'], '')
        self.assertEqual(result['state'], 'simulated')
        self.assertEqual(len(self.remote.calls), before)
        self.assertFalse((self.root / 'history.sqlite').exists())

    def test_live_submit_has_task_binding_and_not_claimed_correct(self):
        preview = self.service.preview(self.body(), self.token)
        self.assertEqual(preview['payload']['answerSets'][0]['taskName'], 'Question 1')
        result = self.service.submit(preview['preview_id'], self.token)
        self.assertEqual(result['state'], 'received')
        self.assertEqual(result['response']['submission'], 'INDETERMINATE')
        self.assertNotIn('secret-remote-token', (self.root / 'history.sqlite').read_bytes().decode(errors='ignore'))

    def test_duplicate_persists_across_service_restart(self):
        preview = self.service.preview(self.body(), self.token)
        self.service.submit(preview['preview_id'], self.token)
        service = DresService(self.find, self.root / 'history.sqlite', self.remote)
        token = service.login({'username': 'test-team', 'password': 'not-real'})['connection']
        preview = service.preview(self.body(task_type='video_kis'), token)
        with self.assertRaisesRegex(FrameError, 'Không gửi trùng'):
            service.submit(preview['preview_id'], token)
        self.assertEqual(sum('/submit/' in path for path, _ in self.remote.calls), 1)

    def test_new_question_can_submit_same_answer(self):
        preview = self.service.preview(self.body(), self.token)
        self.service.submit(preview['preview_id'], self.token)
        self.remote.task['name'] = 'Question 2'
        preview = self.service.preview(self.body(), self.token)
        self.assertEqual(self.service.submit(preview['preview_id'], self.token)['state'], 'received')

    def test_changed_question_or_ended_evaluation_does_not_send(self):
        preview = self.service.preview(self.body(), self.token)
        self.remote.task['name'] = 'Question 2'
        with self.assertRaisesRegex(FrameError, 'đã thay đổi'):
            self.service.submit(preview['preview_id'], self.token)
        preview = self.service.preview(self.body(), self.token)
        self.remote.active = False
        with self.assertRaises(FrameError):
            self.service.submit(preview['preview_id'], self.token)
        self.assertFalse(any('/submit/' in path for path, _ in self.remote.calls))

    def test_timeout_is_persisted_and_cannot_be_retried(self):
        self.remote.failure = RemoteError('timeout', uncertain=True)
        preview = self.service.preview(self.body(), self.token)
        self.assertEqual(self.service.submit(preview['preview_id'], self.token)['state'], 'unknown')
        preview = self.service.preview(self.body(), self.token)
        with self.assertRaises(FrameError):
            self.service.submit(preview['preview_id'], self.token)

    def test_rejected_request_can_be_corrected_after_auth_recovery(self):
        self.remote.failure = RemoteError('expired', 401)
        preview = self.service.preview(self.body(), self.token)
        self.assertEqual(self.service.submit(preview['preview_id'], self.token)['state'], 'rejected')
        self.remote.failure = None
        preview = self.service.preview(self.body(), self.token)
        self.assertEqual(self.service.submit(preview['preview_id'], self.token)['state'], 'received')

    def test_preview_expiry_and_token_binding(self):
        preview = self.service.preview(self.body(), self.token)
        with self.assertRaises(FrameError):
            self.service.submit(preview['preview_id'], 'other')
        with patch('dres_submission.time.time', return_value=__import__('time').time() + 121):
            with self.assertRaises(FrameError):
                self.service.submit(preview['preview_id'], self.token)

    def test_parallel_duplicate_sends_only_once(self):
        previews = [self.service.preview(self.body(), self.token) for _ in range(2)]
        def send(preview):
            try:
                return self.service.submit(preview['preview_id'], self.token)['state']
            except FrameError:
                return 'blocked'
        with ThreadPoolExecutor(2) as pool:
            self.assertCountEqual(list(pool.map(send, previews)), ['received', 'blocked'])

    def test_api_preview_and_validation(self):
        app = FastAPI()
        app.include_router(create_router(self.find, service=self.service))
        client = TestClient(app)
        response = client.post('/dres/preview', json=self.body(live=False))
        self.assertEqual(response.status_code, 200)
        self.assertEqual(client.post('/dres/submit', json={'preview_id': response.json()['preview_id']}).json()['state'], 'simulated')
        self.assertEqual(client.get('/dres/evaluations').status_code, 401)
        self.assertEqual(client.post('/dres/preview', json=self.body(frames=[11], live=False)).status_code, 400)
        self.assertEqual(client.get('/dres/history', headers={'X-DRES-Connection': self.token}).status_code, 200)

    def test_host_validation(self):
        self.assertEqual(base_url('https://eventretrieval.one/'), 'https://eventretrieval.one')
        for host in ('http://example.com', 'https://user:password@example.com', 'https://example.com?session=x', 'https://example.com/elsewhere'):
            with self.assertRaises(FrameError):
                base_url(host)


if __name__ == '__main__':
    unittest.main()
