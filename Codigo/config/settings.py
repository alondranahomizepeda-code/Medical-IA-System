"""
Application Settings & Configuration
=====================================
Centralizes all configurable parameters for the Medical Image Classification System.
Follows the Single Responsibility Principle (SRP) — this module is solely
responsible for providing configuration values to the rest of the application.
"""

import os
from pathlib import Path


# ── Project Paths ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = DATA_DIR / "models"
REPORTS_DIR = DATA_DIR / "reports"
LOGS_DIR = BASE_DIR / "logs"

# Create directories if they don't exist
for directory in [DATA_DIR, UPLOAD_DIR, PROCESSED_DIR, MODEL_DIR, REPORTS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ── Database (Repository Pattern - Infrastructure Layer) ──────
DATABASE_URL = f"sqlite:///{BASE_DIR / 'data' / 'medical_system.db'}"

# ── Authentication & Security (Authentication & User Service) ─
SECRET_KEY = os.getenv("SECRET_KEY", "team3-medical-ai-system-secret-key-2024")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

# ── Image Validation (Fail Fast Principle) ────────────────────
ALLOWED_EXTENSIONS = {".dcm", ".jpg", ".jpeg", ".png", ".tiff", ".tif"}
MAX_FILE_SIZE_MB = 15
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MIN_IMAGE_DIMENSION = 32
MAX_IMAGE_DIMENSION = 8192

# ── Preprocessing (Preprocessing Service) ─────────────────────
TARGET_IMAGE_SIZE = (224, 224)  # Standard input for most CNN architectures
NORMALIZATION_RANGE = (0.0, 1.0)

# ── AI / Inference (Inference Service) ─────────────────────────
DEFAULT_MODEL_NAME = "scoliosis_classifier"
DEFAULT_MODEL_VERSION = "1.0.0"
CONFIDENCE_THRESHOLD = 0.5
BATCH_SIZE = 32
NUM_CLASSES = 3  # Normal, Scoliosis Grave (en C), Scoliosis Severa (en S)
CLASS_LABELS = {
    0: "NO ESCOLIOSIS",
    1: "ESCOLIOSIS GRAVE (EN C)",
    2: "ESCOLIOSIS SEVERA (EN S)",
}
TRANSFER_LEARNING_BASE_MODEL = "MobileNetV2"  # Pre-trained on ImageNet

# Scoliosis dataset path (provided by user in repository)
SCOLIOSIS_DATASET_DIR = BASE_DIR.parent / "Escoliosis" / "Escoliosis" / "scoliosis2"

# ── Logging & Audit (Audit Service) ──────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {name}:{function}:{line} | {message}"
AUDIT_LOG_RETENTION_DAYS = 365  # Immutable audit trail

# ── API Gateway Configuration ─────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
API_TITLE = "Scoliosis AI Diagnostic System"
API_DESCRIPTION = "AI-powered clinical decision support system for spinal scoliosis classification and vertebral curve analysis"
API_VERSION = "2.0.0"

