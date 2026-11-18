"""
Shared Test Fixtures and Configurations (Milestone 3)
=====================================================
Provides reusable test fixtures:
  - Database initialization
  - FastAPI TestClient (with fallback for restricted socket environments)
  - Synthetic medical image generation
  - Authentication tokens for different roles via 2FA
"""

import io
import sys
from pathlib import Path
from typing import Generator

import pytest
from PIL import Image
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.database import init_database
from api.routes import _2fa_codes


class SyncASGITestClient:
    """
    Lightweight synchronous ASGI test client compatible with FastAPI/Starlette
    that drives the ASGI app directly in-memory without requiring OS loopback
    sockets (`socket.socketpair()`), ensuring 100% reliability even in
    network-isolated sandboxes while preserving the exact `TestClient` API.
    """

    def __init__(self, app, base_url: str = "http://testserver"):
        import httpx
        self._app = app
        self._transport = httpx.ASGITransport(app=app)
        self._client = httpx.AsyncClient(transport=self._transport, base_url=base_url)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        return False

    @staticmethod
    def _drive_coroutine(coro):
        """Drive an in-memory ASGI coroutine to completion without socketpair."""
        import time
        while True:
            try:
                yielded = coro.send(None)
                # Handle anyio/asyncio sleep or Future if yielded
                if hasattr(yielded, "__await__"):
                    time.sleep(0.0001)
            except StopIteration as stop:
                return stop.value

    def request(self, method: str, url: str, **kwargs):
        return self._drive_coroutine(self._client.request(method, url, **kwargs))

    def get(self, url: str, **kwargs):
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs):
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs):
        return self.request("PUT", url, **kwargs)

    def delete(self, url: str, **kwargs):
        return self.request("DELETE", url, **kwargs)


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """
    Ensure database tables exist before running the test suite and clean up
    any synthetic test records/files upon session completion so the production/demo
    database and storage directories remain spotless.
    """
    init_database()
    from config.settings import UPLOAD_DIR as UPLOADS_DIR, REPORTS_DIR
    from infrastructure.database import (
        session_scope,
        UserDB,
        MedicalImageDB,
        PredictionResultDB,
        AuditEntryDB,
        OneTimeCodeDB,
        ModelVersionDB,
    )

    uploads_orig = UPLOADS_DIR / "originals"
    pre_uploads = set(uploads_orig.glob("*")) if uploads_orig.exists() else set()
    pre_reports = set(REPORTS_DIR.glob("*")) if REPORTS_DIR.exists() else set()

    with session_scope() as session:
        max_user_id = max([u.id for u in session.query(UserDB.id).all()] or [0])
        max_img_id = max([i.id for i in session.query(MedicalImageDB.id).all()] or [0])
        max_res_id = max([r.id for r in session.query(PredictionResultDB.id).all()] or [0])
        max_aud_id = max([a.id for a in session.query(AuditEntryDB.id).all()] or [0])

    yield

    # Clean up test-generated DB rows and restore active v1.1.0 model state
    try:
        with session_scope() as session:
            session.query(OneTimeCodeDB).delete(synchronize_session=False)
            session.query(PredictionResultDB).filter(PredictionResultDB.id > max_res_id).delete(synchronize_session=False)
            session.query(MedicalImageDB).filter(MedicalImageDB.id > max_img_id).delete(synchronize_session=False)
            session.query(AuditEntryDB).filter(AuditEntryDB.id > max_aud_id).delete(synchronize_session=False)
            session.query(UserDB).filter(UserDB.id > max_user_id).delete(synchronize_session=False)
            session.query(ModelVersionDB).filter(ModelVersionDB.model_name != "scoliosis_classifier").delete(synchronize_session=False)
            for mv in session.query(ModelVersionDB).filter_by(model_name="scoliosis_classifier").all():
                mv.is_approved = (mv.version == "1.1.0")
                mv.is_active = (mv.version == "1.1.0")
    except Exception:
        pass

    # Clean up test-generated files in uploads/originals and reports
    if uploads_orig.exists():
        for f in set(uploads_orig.glob("*")) - pre_uploads:
            try:
                f.unlink()
            except Exception:
                pass
    if REPORTS_DIR.exists():
        for f in set(REPORTS_DIR.glob("*")) - pre_reports:
            try:
                f.unlink()
            except Exception:
                pass

    try:
        from ver_db import limpiar_artefactos_prueba
        limpiar_artefactos_prueba()
    except Exception:
        pass


@pytest.fixture(scope="session")
def app():
    """Application fixture reusing the single initialized FastAPI instance."""
    import main as main_module
    return main_module.app


@pytest.fixture(scope="session")
def client(app) -> Generator:
    """FastAPI TestClient fixture."""
    with SyncASGITestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def auth_tokens(client) -> dict:
    """
    Log in default users and obtain their active JWT tokens.
    Completes 2FA verification code generation and validation.
    """
    roles = {
        "admin": "password123",
        "radiologist": "password123",
        "technician": "password123",
        "ml_engineer": "password123",
    }
    tokens = {}

    for username, password in roles.items():
        res = client.post("/api/v1/auth/login", json={"username": username, "password": password})
        if res.status_code == 200:
            data = res.json()
            temp_token = data.get("access_token")
            if username in _2fa_codes:
                code = _2fa_codes[username]["code"]
                res_2fa = client.post("/api/v1/auth/verify-2fa", json={"username": username, "code": code})
                if res_2fa.status_code == 200:
                    v_data = res_2fa.json()
                    tokens[username] = v_data.get("access_token") or temp_token
            else:
                tokens[username] = temp_token

    return tokens


@pytest.fixture
def sample_xray_image() -> bytes:
    """Generate a valid synthetic spinal X-ray image in JPEG format (400x600 px)."""
    x_coords = np.arange(400, dtype=np.int16)
    dist = np.abs(x_coords - 200)
    row = np.clip(np.maximum(0, 180 - dist) + 40, 0, 255).astype(np.uint8)
    arr = np.broadcast_to(row, (600, 400)).copy()

    img = Image.fromarray(arr, mode="L").convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def sample_invalid_small_image() -> bytes:
    """Generate an image below minimum size threshold (16x16 < 32x32) to test Fail-Fast."""
    img = Image.new("L", (16, 16), color=128).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture
def sample_corrupt_file() -> bytes:
    """Generate random non-image bytes to test corruption handling."""
    return b"%CORRUPTED_IMAGE_HEADER_FAKE_DATA_1234567890%"
