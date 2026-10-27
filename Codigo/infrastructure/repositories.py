"""
Repository Implementations (Repository Pattern)
=================================================
Concrete implementations of the repository interfaces defined in the domain layer.

These classes contain ALL SQL/ORM details, isolating the business logic from
the database technology. If the team migrates from SQLite to PostgreSQL,
only these files need to change — the services remain untouched (DIP).

Related Requirements: RF-07, RF-08, RF-11, RNF-05, RNF-09
Related Principles:   DIP, SRP, Repository Pattern
"""

import json
from typing import List, Optional
from datetime import datetime

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
    SessionLocal,
    MedicalImageDB,
    PredictionResultDB,
    UserDB,
    AuditEntryDB,
)
from utils.logger import get_logger
from utils.exceptions import RepositoryError

logger = get_logger("Repository")


# ═══════════════════════════════════════════════════════════════
# ImageRepository
# ═══════════════════════════════════════════════════════════════

class ImageRepository(IImageRepository):
    """Concrete repository for medical image records (RF-02, RF-08)."""

    def save(self, image: MedicalImage) -> MedicalImage:
        try:
            session = SessionLocal()
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
            session.commit()
            image.id = db_image.id
            session.close()
            logger.info(f"Image saved with ID: {image.id}")
            return image
        except Exception as e:
            logger.error(f"Failed to save image: {e}")
            raise RepositoryError("save_image", str(e))

    def find_by_id(self, image_id: int) -> Optional[MedicalImage]:
        try:
            session = SessionLocal()
            db_image = session.query(MedicalImageDB).filter_by(id=image_id).first()
            session.close()
            if db_image:
                return self._to_domain(db_image)
            return None
        except Exception as e:
            raise RepositoryError("find_image_by_id", str(e))

    def find_all(self, limit: int = 50, offset: int = 0) -> List[MedicalImage]:
        try:
            session = SessionLocal()
            results = (
                session.query(MedicalImageDB)
                .order_by(MedicalImageDB.uploaded_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            session.close()
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
            session = SessionLocal()
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
            session.commit()
            result.id = db_result.id
            session.close()
            logger.info(f"Prediction result saved with ID: {result.id}")
            return result
        except Exception as e:
            raise RepositoryError("save_result", str(e))

    def find_by_id(self, result_id: int) -> Optional[PredictionResult]:
        try:
            session = SessionLocal()
            db_result = session.query(PredictionResultDB).filter_by(id=result_id).first()
            session.close()
            return self._to_domain(db_result) if db_result else None
        except Exception as e:
            raise RepositoryError("find_result_by_id", str(e))

    def find_by_image_id(self, image_id: int) -> Optional[PredictionResult]:
        try:
            session = SessionLocal()
            db_result = (
                session.query(PredictionResultDB)
                .filter_by(image_id=image_id)
                .order_by(PredictionResultDB.predicted_at.desc())
                .first()
            )
            session.close()
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
            session = SessionLocal()
            query = session.query(PredictionResultDB)
            if status:
                query = query.filter_by(status=status.value)
            results = (
                query.order_by(PredictionResultDB.predicted_at.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            session.close()
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
            session = SessionLocal()
            db_result = session.query(PredictionResultDB).filter_by(id=result_id).first()
            if db_result:
                db_result.status = status.value
                db_result.reviewed_by = reviewer_id
                db_result.reviewer_notes = notes
                db_result.reviewed_at = datetime.utcnow()
                session.commit()
                result = self._to_domain(db_result)
                session.close()
                logger.info(f"Prediction {result_id} reviewed: {status.value}")
                return result
            session.close()
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
            session = SessionLocal()
            db_user = UserDB(
                username=user.username,
                hashed_password=user.hashed_password,
                full_name=user.full_name,
                role=user.role.value,
                is_active=user.is_active,
            )
            session.add(db_user)
            session.commit()
            user.id = db_user.id
            session.close()
            logger.info(f"User '{user.username}' saved with ID: {user.id}")
            return user
        except Exception as e:
            raise RepositoryError("save_user", str(e))

    def find_by_username(self, username: str) -> Optional[User]:
        try:
            session = SessionLocal()
            db_user = session.query(UserDB).filter_by(username=username).first()
            session.close()
            return self._to_domain(db_user) if db_user else None
        except Exception as e:
            raise RepositoryError("find_user_by_username", str(e))

    def find_by_id(self, user_id: int) -> Optional[User]:
        try:
            session = SessionLocal()
            db_user = session.query(UserDB).filter_by(id=user_id).first()
            session.close()
            return self._to_domain(db_user) if db_user else None
        except Exception as e:
            raise RepositoryError("find_user_by_id", str(e))

    def update_password(self, username: str, new_hashed_password: str) -> bool:
        try:
            session = SessionLocal()
            db_user = session.query(UserDB).filter_by(username=username).first()
            if db_user:
                db_user.hashed_password = new_hashed_password
                session.commit()
                session.close()
                logger.info(f"Password updated for user '{username}'")
                return True
            session.close()
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
            session = SessionLocal()
            db_entry = AuditEntryDB(
                action=entry.action.value,
                user_id=entry.user_id,
                resource_type=entry.resource_type,
                resource_id=entry.resource_id,
                details=entry.details,
                ip_address=entry.ip_address,
            )
            session.add(db_entry)
            session.commit()
            entry.id = db_entry.id
            session.close()
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
            session = SessionLocal()
            query = session.query(AuditEntryDB)
            if action:
                query = query.filter_by(action=action.value)
            results = (
                query.order_by(AuditEntryDB.timestamp.desc())
                .offset(offset)
                .limit(limit)
                .all()
            )
            session.close()
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
