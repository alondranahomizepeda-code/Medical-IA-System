"""
Shared Test Fixtures and Configurations (Milestone 3)
=====================================================
Provides reusable test fixtures:
  - Database initialization
  - FastAPI TestClient
  - Synthetic medical image generation
  - Authentication tokens for different roles via 2FA
"""

import io
import sys
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from PIL import Image
import numpy as np

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from infrastructure.database import init_database
from main import create_application
from api.routes import _2fa_codes


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Ensure database tables exist before running test suite and reset test passwords."""
    init_database()
    from infrastructure.repositories import UserRepository, AuditRepository
    from services.auth_service import AuthService
    auth_svc = AuthService(UserRepository(), AuditRepository())
    for u in ["radiologist", "technician", "admin", "ml_engineer"]:
        try:
            auth_svc.update_password(u, "password123")
        except Exception:
            pass
    yield


@pytest.fixture(scope="session")
def app():
    """Application fixture."""
    return create_application()


@pytest.fixture(scope="session")
def client(app) -> Generator[TestClient, None, None]:
    """FastAPI TestClient fixture."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def auth_tokens(client: TestClient) -> dict:
    """
    Log in default users and obtain their active JWT tokens.
    Simulates 2FA verification code generation and validation.
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
            # Retrieve generated 2FA code from memory store
            if username in _2fa_codes:
                code = _2fa_codes[username]["code"]
                res_2fa = client.post("/api/v1/auth/verify-2fa", json={"username": username, "code": code})
                if res_2fa.status_code == 200:
                    tokens[username] = temp_token
            else:
                tokens[username] = temp_token

    return tokens


@pytest.fixture
def sample_xray_image() -> bytes:
    """Generate a valid synthetic spinal X-ray image in JPEG format (800x1200 px)."""
    # Create realistic grayscale gradient simulating spine column
    arr = np.zeros((600, 400), dtype=np.uint8)
    for y in range(600):
        for x in range(400):
            dist_to_center = abs(x - 200)
            density = max(0, 180 - dist_to_center)
            arr[y, x] = np.clip(density + 40, 0, 255)

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
