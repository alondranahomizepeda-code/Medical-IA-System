"""
Repository Implementations (Repository Pattern)
=================================================
Concrete implementations of the repository interfaces defined in the domain layer.

These classes contain ALL SQL/ORM details, isolating the business logic from
the database technology. If the team migrates from SQLite to PostgreSQL,
only these files need to change — the services remain untouched (DIP).

Related Requirements: RF-01, RF-07, RF-08, RF-11, RNF-03, RNF-05, RNF-09
Related Principles:   DIP, SRP, Repository Pattern
"""

import hashlib
import hmac
import json
import secrets
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from config.settings import SECRET_KEY
from domain.interfaces.contracts import (
    IImageRepository,
    IResultRepository,
    IUserRepository,
    IAuditRepository,
    MedicalImage,
    PredictionResult,
    User,
    AuditEntry,
    PredictionStatus,
    AuditAction,
    UserRole,
)
from infrastructure.database import (
    session_scope,
    MedicalImageDB,
    PredictionResultDB,
    UserDB,
    AuditEntryDB,
    OneTimeCodeDB,
)
from utils.logger import get_logger
from utils.exceptions import RepositoryError, OneTimeCodeError

logger = get_logger("Repository")


# ═══════════════════════════════════════════════════════════════
# ImageRepository
# ═══════════════════════════════════════════════════════════════

class ImageRepository(IImageRepository):
    """Concrete repository for medical image records (RF-02, RF-08)."""

    def save(self, image: MedicalImage) -> MedicalImage:
        try:
            with session_scope() as session:
                db_image = MedicalImageDB(
                    filename=image.filename,
                    original_path=image.original_path,
                    processed_path=image.processed_path,
                    file_size_bytes=image.file_size_bytes,
                    format=image.format,
                    width=image.width,
                    height=image.height,
                    uploaded_by=image.uploaded_by,
                    is_anonymized=image.is_anonymized,
                    metadata_json=json.dumps(image.metadata) if image.metadata else None,
                )
                session.add(db_image)
                session.flush()
                image.id = db_image.id
            logger.info(f"Image saved with ID: {image.id}")
            return image
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise RepositoryError("save_image", str(e))

    def find_by_id(self, image_id: int) -> Optional[MedicalImage]:
        try:
            with session_scope() as session:
                db_image = session.query(MedicalImageDB).filter_by(id=image_id).first()
                if db_image:
                    return self._to_domain(db_image)
                return None
        except Exception as e:
            raise RepositoryError("find_image_by_id", str(e))

    def find_all(self, limit: int = 50, offset: int = 0) -> List[MedicalImage]:
        try:
            with session_scope() as session:
                results = (
                    session.query(MedicalImageDB)
                    .order_by(MedicalImageDB.uploaded_at.desc())
                    .offset(offset)
                    .limit(limit)
                    .all()
                )
                return [self._to_domain(r) for r in results]
        except Exception as e:
            raise RepositoryError("find_all_images", str(e))

    @staticmethod
    def _to_domain(db: MedicalImageDB) -> MedicalImage:
        return MedicalImage(
            id=db.id,
            filename=db.filename,
            original_path=db.original_path,
            processed_path=db.processed_path,
            file_size_bytes=db.file_size_bytes,
            format=db.format,
            width=db.width,
            height=db.height,
            uploaded_by=db.uploaded_by,
            uploaded_at=db.uploaded_at,
            is_anonymized=db.is_anonymized,
            metadata=json.loads(db.metadata_json) if db.metadata_json else {},
        )


# ═══════════════════════════════════════════════════════════════
# ResultRepository
# ═══════════════════════════════════════════════════════════════

class ResultRepository(IResultRepository):
    """Concrete repository for prediction results (RF-08, RF-11)."""

    def save(self, result: PredictionResult) -> PredictionResult:
        try:
            with session_scope() as session:
                db_result = PredictionResultDB(
                    image_id=result.image_id,
                    classification=result.classification,
                    confidence_score=result.confidence_score,
                    model_name=result.model_name,
                    model_version=result.model_version,
                    processing_time_ms=result.processing_time_ms,
                    status=result.status.value,
                    affected_vertebrae=result.affected_vertebrae,
                    curve_type=result.curve_type,
                    curve_detail=result.curve_detail,
                )
                session.add(db_result)
                session.flush()
                result.id = db_result.id
            logger.info(f"Prediction result saved with ID: {result.id}")
            return result
        except Exception as e:
            raise RepositoryError("save_result", str(e))

    def find_by_id(self, result_id: int) -> Optional[PredictionResult]:
        try:
            with session_scope() as session:
                db_result = session.query(PredictionResultDB).filter_by(id=result_id).first()
                return self._to_domain(db_result) if db_result else None
        except Exception as e:
            raise RepositoryError("find_result_by_id", str(e))

    def find_by_image_id(self, image_id: int) -> Optional[PredictionResult]:
        try:
            with session_scope() as session:
                db_result = (
                    session.query(PredictionResultDB)
                    .filter_by(image_id=image_id)
                    .order_by(PredictionResultDB.predicted_at.desc())
                    .first()
                )
                return self._to_domain(db_result) if db_result else None
        except Exception as e:
            raise RepositoryError("find_result_by_image_id", str(e))

    def find_all(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[PredictionStatus] = None,
    ) -> List[PredictionResult]:
        try:
            with session_scope() as session:
                query = session.query(PredictionResultDB)
                if status:
                    query = query.filter_by(status=status.value)
                results = (
                    query.order_by(PredictionResultDB.predicted_at.desc())
                    .offset(offset)
                    .limit(limit)
                    .all()
                )
                return [self._to_domain(r) for r in results]
        except Exception as e:
            raise RepositoryError("find_all_results", str(e))

    def update_review(
        self,
        result_id: int,
        status: PredictionStatus,
        reviewer_id: int,
        notes: str,
    ) -> Optional[PredictionResult]:
        try:
            with session_scope() as session:
                db_result = session.query(PredictionResultDB).filter_by(id=result_id).first()
                if db_result:
                    db_result.status = status.value
                    db_result.reviewed_by = reviewer_id
                    db_result.reviewer_notes = notes
                    db_result.reviewed_at = datetime.utcnow()
                    session.flush()
                    result = self._to_domain(db_result)
                    logger.info(f"Prediction {result_id} reviewed: {status.value}")
                    return result
                return None
        except Exception as e:
            raise RepositoryError("update_review", str(e))

    @staticmethod
    def _to_domain(db: PredictionResultDB) -> PredictionResult:
        return PredictionResult(
            id=db.id,
            image_id=db.image_id,
            classification=db.classification,
            confidence_score=db.confidence_score,
            model_name=db.model_name,
            model_version=db.model_version,
            processing_time_ms=db.processing_time_ms,
            predicted_at=db.predicted_at,
            status=PredictionStatus(db.status),
            reviewed_by=db.reviewed_by,
            reviewer_notes=db.reviewer_notes,
            reviewed_at=db.reviewed_at,
            affected_vertebrae=getattr(db, "affected_vertebrae", None),
            curve_type=getattr(db, "curve_type", None),
            curve_detail=getattr(db, "curve_detail", None),
        )


# ═══════════════════════════════════════════════════════════════
# UserRepository
# ═══════════════════════════════════════════════════════════════

class UserRepository(IUserRepository):
    """Concrete repository for user records (RF-01)."""

    def save(self, user: User) -> User:
        try:
            with session_scope() as session:
                db_user = UserDB(
                    username=user.username,
                    hashed_password=user.hashed_password,
                    full_name=user.full_name,
                    role=user.role.value,
                    is_active=user.is_active,
                )
                session.add(db_user)
                session.flush()
                user.id = db_user.id
            logger.info(f"User '{user.username}' saved with ID: {user.id}")
            return user
        except Exception as e:
            raise RepositoryError("save_user", str(e))

    def find_by_username(self, username: str) -> Optional[User]:
        try:
            with session_scope() as session:
                db_user = session.query(UserDB).filter_by(username=username).first()
                return self._to_domain(db_user) if db_user else None
        except Exception as e:
            raise RepositoryError("find_user_by_username", str(e))

    def find_by_id(self, user_id: int) -> Optional[User]:
        try:
            with session_scope() as session:
                db_user = session.query(UserDB).filter_by(id=user_id).first()
                return self._to_domain(db_user) if db_user else None
        except Exception as e:
            raise RepositoryError("find_user_by_id", str(e))

    def update_password(self, username: str, new_hashed_password: str) -> bool:
        try:
            with session_scope() as session:
                db_user = session.query(UserDB).filter_by(username=username).first()
                if db_user:
                    db_user.hashed_password = new_hashed_password
                    session.flush()
                    logger.info(f"Password updated for user '{username}'")
                    return True
                return False
        except Exception as e:
            raise RepositoryError("update_password", str(e))

    @staticmethod
    def _to_domain(db: UserDB) -> User:
        return User(
            id=db.id,
            username=db.username,
            hashed_password=db.hashed_password,
            full_name=db.full_name,
            role=UserRole(db.role),
            is_active=db.is_active,
            created_at=db.created_at,
        )


# ═══════════════════════════════════════════════════════════════
# AuditRepository (Append-Only — RNF-09)
# ═══════════════════════════════════════════════════════════════

class AuditRepository(IAuditRepository):
    """
    Concrete repository for audit log entries.
    
    This repository is APPEND-ONLY: it provides no update or delete methods.
    Once an audit entry is written, it cannot be modified, ensuring
    the immutability required by RNF-09 and RD-04.
    """

    def append(self, entry: AuditEntry) -> AuditEntry:
        try:
            with session_scope() as session:
                db_entry = AuditEntryDB(
                    action=entry.action.value,
                    user_id=entry.user_id,
                    resource_type=entry.resource_type,
                    resource_id=entry.resource_id,
                    details=entry.details,
                    ip_address=entry.ip_address,
                )
                session.add(db_entry)
                session.flush()
                entry.id = db_entry.id
            return entry
        except Exception as e:
            raise RepositoryError("append_audit", str(e))

    def find_all(
        self,
        limit: int = 100,
        offset: int = 0,
        action: Optional[AuditAction] = None,
    ) -> List[AuditEntry]:
        try:
            with session_scope() as session:
                query = session.query(AuditEntryDB)
                if action:
                    query = query.filter_by(action=action.value)
                results = (
                    query.order_by(AuditEntryDB.timestamp.desc())
                    .offset(offset)
                    .limit(limit)
                    .all()
                )
                return [self._to_domain(r) for r in results]
        except Exception as e:
            raise RepositoryError("find_all_audit", str(e))

    @staticmethod
    def _to_domain(db: AuditEntryDB) -> AuditEntry:
        return AuditEntry(
            id=db.id,
            action=AuditAction(db.action),
            user_id=db.user_id,
            resource_type=db.resource_type,
            resource_id=db.resource_id,
            details=db.details,
            timestamp=db.timestamp,
            ip_address=db.ip_address,
        )


# ═══════════════════════════════════════════════════════════════
# OneTimeCodeRepository (2FA & Password Recovery — RNF-03)
# ═══════════════════════════════════════════════════════════════

class OneTimeCodeRepository:
    """
    Persists 6-digit one-time verification codes (2FA and password reset)
    in the database as HMAC-SHA256 digests so that multiple replicas share state
    and raw codes are never stored in plaintext in the database.
    """

    MAX_ATTEMPTS = 5
    MAX_SENDS = 3

    @staticmethod
    def hash_code(username: str, purpose: str, code: str) -> str:
        msg = f"{username}:{purpose}:{code.strip()}".encode("utf-8")
        return hmac.new(SECRET_KEY.encode("utf-8"), msg, hashlib.sha256).hexdigest()

    @staticmethod
    def generate_6digit_code() -> str:
        return f"{secrets.randbelow(900000) + 100000:06d}"

    def issue_code(
        self,
        username: str,
        purpose: str = "2fa",
        ttl_minutes: int = 5,
        is_resend: bool = False,
    ) -> Tuple[str, int]:
        """
        Create or rotate a one-time code for (username, purpose).
        Returns (raw_code, sends_used).
        """
        raw_code = self.generate_6digit_code()
        digest = self.hash_code(username, purpose, raw_code)
        expires_at = datetime.utcnow() + timedelta(minutes=ttl_minutes)

        with session_scope() as session:
            existing = (
                session.query(OneTimeCodeDB)
                .filter_by(username=username, purpose=purpose)
                .first()
            )
            if is_resend:
                if not existing:
                    raise OneTimeCodeError("No active verification session to resend.")
                if existing.sends >= self.MAX_SENDS:
                    raise OneTimeCodeError("Maximum resend attempts (3) reached.")
                existing.code_hash = digest
                existing.expires_at = expires_at
                existing.attempts = 0
                existing.sends += 1
                sends = existing.sends
            else:
                if existing:
                    session.delete(existing)
                    session.flush()
                record = OneTimeCodeDB(
                    username=username,
                    purpose=purpose,
                    code_hash=digest,
                    expires_at=expires_at,
                    attempts=0,
                    sends=1,
                )
                session.add(record)
                sends = 1

        return raw_code, sends

    def verify_and_consume(
        self,
        username: str,
        code: str,
        purpose: str = "2fa",
    ) -> bool:
        """
        Verify a one-time code in constant time and consume it upon success.
        Locks and deletes the code after MAX_ATTEMPTS (5) failed tries.
        """
        expected_digest = self.hash_code(username, purpose, code)
        now = datetime.utcnow()
        err_msg: Optional[str] = None

        with session_scope() as session:
            record = (
                session.query(OneTimeCodeDB)
                .filter_by(username=username, purpose=purpose)
                .first()
            )
            if not record:
                err_msg = "No active verification code found or already used."
            elif now > record.expires_at:
                session.delete(record)
                err_msg = "Verification code has expired. Please request a new one."
            elif hmac.compare_digest(record.code_hash, expected_digest):
                session.delete(record)
                return True
            else:
                record.attempts += 1
                attempts_left = self.MAX_ATTEMPTS - record.attempts
                if attempts_left <= 0:
                    session.delete(record)
                    err_msg = "Maximum verification attempts (5) exceeded. Code invalidated."
                else:
                    err_msg = f"Invalid verification code ({attempts_left} attempts left)."

        if err_msg:
            raise OneTimeCodeError(err_msg)
        return True
