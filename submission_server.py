"""Small, standalone DRES submission server (no search models or Redis)."""
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from dres_frames import FrameError
from dres_submission import DresService, RemoteError, create_router
from submission_maps import SubmissionMaps


ROOT = Path(__file__).resolve().parent


def create_app(*, data_root=None, service=None, username=None, password=None, base_url=None):
    data_root = Path(data_root or os.getenv("DATA_ROOT", ROOT.parent / "final_search_index"))
    maps = SubmissionMaps(data_root, os.getenv("MAPFILES_DIR"))
    service = service or DresService(
        maps.find,
        os.getenv("DRES_HISTORY_PATH", str(ROOT / "runtime" / "dres_history.sqlite")),
    )
    credentials = {
        "base_url": base_url or os.getenv("DRES_BASE_URL", "https://eventretrieval.one"),
        "username": username if username is not None else os.getenv("DRES_USERNAME", ""),
        "password": password if password is not None else os.getenv("DRES_PASSWORD", ""),
    }

    app = FastAPI(title="AIC 2026 - Nộp bài DRES")
    app.include_router(create_router(maps.find, service=service))
    app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

    @app.get("/")
    def index():
        return FileResponse(ROOT / "submission.html", headers={"Cache-Control": "no-store"})

    @app.get("/health")
    def health():
        return {"ready": True, "submission_only": True, "credentials_configured": bool(credentials["username"] and credentials["password"])}

    @app.post("/dres/auto-login")
    def auto_login():
        if not credentials["username"] or not credentials["password"]:
            raise HTTPException(503, "Server chưa được cấu hình tài khoản DRES.")
        try:
            return service.login(credentials)
        except FrameError as exc:
            raise HTTPException(400, str(exc)) from None
        except RemoteError as exc:
            raise HTTPException(502, exc.message) from None

    return app


app = create_app()
