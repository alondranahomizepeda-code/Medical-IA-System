"""
Application Settings & Configuration
=====================================
Centralizes all configurable parameters for the Medical Image Classification System.
Follows the Single Responsibility Principle (SRP) and 12-Factor App configuration
via environment variables.
"""

import os
from pathlib import Path


# ── Project Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("DATA_DIR", str(BASE_DIR / "data")))
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = Path(os.getenv("MODEL_DIR", str(DATA_DIR / "models")))
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = Path(os.getenv("LOGS_DIR", str(BASE_DIR / "logs")))

# Create directories if they don't exist
for directory in [DATA_DIR, UPLOAD_DIR, PROCESSED_DIR, MODEL_DIR, REPORTS_DIR, LOGS_DIR]:
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

# Ensure Keras, Matplotlib, and tempfile directories are writable inside workspace / container / sandbox
os.environ.setdefault("KERAS_HOME", str(DATA_DIR / ".keras"))
os.environ.setdefault("MPLCONFIGDIR", str(DATA_DIR / ".mplconfig"))
_tmp_dir = DATA_DIR / ".tmp"
try:
    Path(os.environ["KERAS_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    _tmp_dir.mkdir(parents=True, exist_ok=True)
    os.environ["TMP"] = str(_tmp_dir)
    os.environ["TEMP"] = str(_tmp_dir)
    os.environ["TMPDIR"] = str(_tmp_dir)
    import tempfile
    tempfile.tempdir = str(_tmp_dir)
    _default_keras_cfg = Path(os.path.expanduser("~")) / ".keras" / "keras.json"
    if _default_keras_cfg.exists():
        with open(_default_keras_cfg, "r"):
            pass
except PermissionError:
    os.environ["USERPROFILE"] = str(DATA_DIR)
    os.environ["HOME"] = str(DATA_DIR)
except Exception:
    pass

# ── Database (Repository Pattern - Infrastructure Layer) ──────
DATABASE_URL = os.getenv(
    "TEST_DATABASE_URL",
    os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR / 'medical_system.db'}"),
)

# ── Environment & Security (Authentication & User Service) ────
ENVIRONMENT = os.getenv("ENVIRONMENT", "development").lower()
SECRET_KEY = os.getenv("SECRET_KEY", "team3-medical-ai-system-secret-key-2026-secure")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
OTP_LOG_CODES = os.getenv("OTP_LOG_CODES", "true").lower() in ("1", "true", "yes")

# ── Image Validation (Fail Fast Principle) ────────────────────
ALLOWED_EXTENSIONS = {".dcm", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}
MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_IMAGE_DIMENSION = 32
MAX_IMAGE_DIMENSION = 8192

# ── Preprocessing (Preprocessing Service) ─────────────────────
TARGET_IMAGE_SIZE = (224, 224)  # Standard input for MobileNetV2
NORMALIZATION_RANGE = (0.0, 1.0)

# ── AI / Inference (Inference Service) ─────────────────────────
DEFAULT_MODEL_NAME = "scoliosis_classifier"
DEFAULT_MODEL_VERSION = os.getenv("DEFAULT_MODEL_VERSION", "1.1.0")
ALLOW_UNAPPROVED_MODEL = os.getenv("ALLOW_UNAPPROVED_MODEL", "false").lower() in ("1", "true", "yes")
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.60"))
BATCH_SIZE = 32
NUM_CLASSES = 3  # Normal, Scoliosis Grave (en C), Scoliosis Severa (en S)
CLASS_LABELS = {
    0: "NO SCOLIOSIS",
    1: "SERIOUS SCOLIOSIS (C-CURVE)",
    2: "SEVERE SCOLIOSIS (S-CURVE)",
}
TRANSFER_LEARNING_BASE_MODEL = "MobileNetV2"  # Pre-trained on ImageNet

# Scoliosis dataset path (provided in repository)
SCOLIOSIS_DATASET_DIR = Path(
    os.getenv(
        "SCOLIOSIS_DATASET_DIR",
        str(BASE_DIR.parent / "Escoliosis" / "Escoliosis" / "scoliosis2"),
    )
)

# ── Logging & Audit (Audit Service) ──────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}"
AUDIT_LOG_RETENTION_DAYS = 365  # Immutable audit trail

# ── API Gateway Configuration ─────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_TITLE = "Scoliosis AI Diagnostic System"
API_DESCRIPTION = (
    "AI-powered clinical decision support system for spinal scoliosis "
    "classification and vertebral curve analysis"
)
API_VERSION = "3.2.1"
