"""
Database Connection & ORM Models
==================================
Infrastructure layer for data persistence using SQLAlchemy.
Implements the database schema aligned with the domain entities.

The Repository pattern abstracts all SQL details from the business logic,
so the domain layer never imports SQLAlchemy directly (DIP).
"""

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    Boolean,
    DateTime,
    Text,
    Enum as SQLEnum,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime

from config.settings import DATABASE_URL
from domain.interfaces.contracts import UserRole, PredictionStatus, AuditAction

# ── SQLAlchemy Setup ──────────────────────────────────────────
engine = create_engine(DATABASE_URL, echo=False, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ═══════════════════════════════════════════════════════════════
# ORM Models (Database Tables)
# ═══════════════════════════════════════════════════════════════

class UserDB(Base):
    """Users table — Authentication & User Service (RF-01)."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(200), nullable=False)
    role = Column(String(50), nullable=False, default=UserRole.TECHNICIAN.value)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class MedicalImageDB(Base):
    """Medical images table — Image Management Service (RF-02)."""
    __tablename__ = "medical_images"

    id = Column(Integer, primary_key=True, autoincrement=True)
    filename = Column(String(255), nullable=False)
    original_path = Column(String(500), nullable=False)
    processed_path = Column(String(500), nullable=True)
    file_size_bytes = Column(Integer, nullable=False)
    format = Column(String(20), nullable=False)
    width = Column(Integer, nullable=True)
    height = Column(Integer, nullable=True)
    uploaded_by = Column(Integer, nullable=True)
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    is_anonymized = Column(Boolean, default=False)
    metadata_json = Column(Text, nullable=True)


class PredictionResultDB(Base):
    """Prediction results table — Clinical Results Service (RF-05, RF-08)."""
    __tablename__ = "prediction_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    image_id = Column(Integer, nullable=False, index=True)
    classification = Column(String(100), nullable=False)
    confidence_score = Column(Float, nullable=False)
    model_name = Column(String(100), nullable=False)
    model_version = Column(String(50), nullable=False)
    processing_time_ms = Column(Float, nullable=False)
    predicted_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String(50), default=PredictionStatus.PENDING_REVIEW.value)
    reviewed_by = Column(Integer, nullable=True)
    reviewer_notes = Column(Text, nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    affected_vertebrae = Column(String(200), nullable=True)
    curve_type = Column(String(100), nullable=True)
    curve_detail = Column(Text, nullable=True)


class AuditEntryDB(Base):
    """Audit log table — Audit Service (RNF-09, RD-04). Append-only."""
    __tablename__ = "audit_log"

    id = Column(Integer, primary_key=True, autoincrement=True)
    action = Column(String(100), nullable=False, index=True)
    user_id = Column(Integer, nullable=True)
    resource_type = Column(String(100), nullable=False)
    resource_id = Column(Integer, nullable=True)
    details = Column(Text, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    ip_address = Column(String(50), nullable=True)


class ModelVersionDB(Base):
    """Model registry table — Model Registry (RF-12, RD-10)."""
    __tablename__ = "model_versions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    model_name = Column(String(100), nullable=False, index=True)
    version = Column(String(50), nullable=False)
    file_path = Column(String(500), nullable=False)
    accuracy = Column(Float, nullable=True)
    precision_score = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    is_approved = Column(Boolean, default=False)
    is_active = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    created_by = Column(String(100), nullable=True)
    description = Column(Text, nullable=True)


def init_database():
    """Create all tables if they don't exist and ensure schema migrations."""
    Base.metadata.create_all(bind=engine)
    try:
        with engine.connect() as conn:
            cursor = conn.connection.cursor()
            cursor.execute("PRAGMA table_info(prediction_results)")
            columns = [row[1] for row in cursor.fetchall()]
            if "affected_vertebrae" not in columns:
                cursor.execute("ALTER TABLE prediction_results ADD COLUMN affected_vertebrae VARCHAR(200)")
            if "curve_type" not in columns:
                cursor.execute("ALTER TABLE prediction_results ADD COLUMN curve_type VARCHAR(100)")
            if "curve_detail" not in columns:
                cursor.execute("ALTER TABLE prediction_results ADD COLUMN curve_detail TEXT")
            conn.connection.commit()
    except Exception:
        pass


def get_db_session():
    """Provide a database session (used by repositories)."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
