from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

from dres_submission import DresService
from submission_maps import SubmissionMaps
from submission_server import create_app
from test_dres import FakeDres


class SubmissionServerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        maps = self.root / "map-keyframes"
        maps.mkdir()
        (maps / "N001-V001.csv").write_text("n,pts_time,fps,frame_idx\n11,5.521,25,138\n", encoding="utf-8")
        self.remote = FakeDres()
        finder = SubmissionMaps(self.root).find
        self.service = DresService(finder, self.root / "history.db", self.remote)

    def test_standalone_ui_auto_login_and_simulation(self):
        app = create_app(data_root=self.root, service=self.service, username="team", password="test-only")
        with TestClient(app) as client:
            page = client.get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn("Server tối giản chỉ dùng để kiểm tra và nộp bài", page.text)
            self.assertNotIn("type=\"password\"", page.text)
            login = client.post("/dres/auto-login")
            self.assertEqual(login.status_code, 200)
            token = login.json()["connection"]
            headers = {"X-DRES-Connection": token}
            preview = client.post("/dres/preview", headers=headers, json={"task_type": "textual_kis", "video": "N001-V001", "frames": [138], "live": False})
            self.assertEqual(preview.status_code, 200, preview.text)
            result = client.post("/dres/submit", headers=headers, json={"preview_id": preview.json()["preview_id"]})
            self.assertEqual(result.json()["state"], "simulated")

    def test_missing_credentials_is_reported_without_password_form(self):
        app = create_app(data_root=self.root, service=self.service, username="", password="")
        with TestClient(app) as client:
            self.assertEqual(client.post("/dres/auto-login").status_code, 503)


if __name__ == "__main__":
    unittest.main()
