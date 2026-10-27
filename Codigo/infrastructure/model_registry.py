"""
Model Registry
================
Manages the versioning and lifecycle of AI models (RF-12, RD-10).

Responsibilities:
  - Register new model versions after training and validation.
  - Store and retrieve model metadata (accuracy, precision, recall, F1).
  - Activate an approved version for the Inference Service.
  - Maintain version history for prediction traceability.

The Model Registry does NOT perform training — it only manages the
artifacts resulting from the ML process (Milestone 1 Correction #5).
"""

from typing import Dict, List, Optional
from datetime import datetime

from domain.interfaces.contracts import ModelVersion
from infrastructure.database import SessionLocal, ModelVersionDB
from utils.logger import get_logger
from utils.exceptions import ModelNotFoundError, RepositoryError

logger = get_logger("ModelRegistry")


class ModelRegistry:
    """
    Central registry for managing AI model versions.

    The ML Engineer / Data Scientist registers new versions here after
    completing Fine-Tuning and evaluation. Only approved and activated
    versions are available to the Inference Service.
    """

    def register_version(
        self,
        model_name: str,
        version: str,
        file_path: str,
        metrics: Optional[Dict[str, float]] = None,
        created_by: Optional[str] = None,
        description: str = "",
    ) -> ModelVersion:
        """
        Register a new model version in the registry.

        The version starts as NOT approved and NOT active.
        It must be explicitly approved and activated before use.
        """
        try:
            session = SessionLocal()
            db_version = ModelVersionDB(
                model_name=model_name,
                version=version,
                file_path=file_path,
                accuracy=metrics.get("accuracy") if metrics else None,
                precision_score=metrics.get("precision") if metrics else None,
                recall=metrics.get("recall") if metrics else None,
                f1_score=metrics.get("f1_score") if metrics else None,
                is_approved=False,
                is_active=False,
                created_by=created_by,
                description=description,
            )
            session.add(db_version)
            session.commit()

            model_version = ModelVersion(
                id=db_version.id,
                model_name=model_name,
                version=version,
                file_path=file_path,
                accuracy=db_version.accuracy,
                precision_score=db_version.precision_score,
                recall=db_version.recall,
                f1_score=db_version.f1_score,
                created_by=created_by,
                description=description,
            )
            session.close()

            logger.info(
                f"Model version registered: {model_name} v{version} "
                f"(accuracy: {metrics.get('accuracy', 'N/A') if metrics else 'N/A'})"
            )
            return model_version

        except Exception as e:
            raise RepositoryError("register_model_version", str(e))

    def approve_version(self, model_name: str, version: str) -> ModelVersion:
        """Mark a model version as approved for potential activation."""
        try:
            session = SessionLocal()
            db_version = (
                session.query(ModelVersionDB)
                .filter_by(model_name=model_name, version=version)
                .first()
            )
            if not db_version:
                session.close()
                raise ModelNotFoundError(f"{model_name} v{version}")

            db_version.is_approved = True
            session.commit()

            logger.info(f"Model version approved: {model_name} v{version}")
            result = self._to_domain(db_version)
            session.close()
            return result

        except ModelNotFoundError:
            raise
        except Exception as e:
            raise RepositoryError("approve_model_version", str(e))

    def activate_version(self, model_name: str, version: str) -> ModelVersion:
        """
        Set a version as the active model for inference.
        
        Only ONE version can be active at a time per model name.
        The version must be approved before activation (RF-12).
        """
        try:
            session = SessionLocal()

            # Verify the version exists and is approved
            target = (
                session.query(ModelVersionDB)
                .filter_by(model_name=model_name, version=version)
                .first()
            )
            if not target:
                session.close()
                raise ModelNotFoundError(f"{model_name} v{version}")

            if not target.is_approved:
                session.close()
                raise RepositoryError(
                    "activate_version",
                    f"Version {version} is not approved. Approve it first.",
                )

            # Deactivate all other versions of this model
            session.query(ModelVersionDB).filter_by(
                model_name=model_name, is_active=True
            ).update({"is_active": False})

            # Activate the target version
            target.is_active = True
            session.commit()

            logger.info(f"Model version ACTIVATED: {model_name} v{version}")
            result = self._to_domain(target)
            session.close()
            return result

        except (ModelNotFoundError, RepositoryError):
            raise
        except Exception as e:
            raise RepositoryError("activate_model_version", str(e))

    def get_active_version(self, model_name: str) -> Optional[ModelVersion]:
        """Retrieve the currently active model version for a given model name."""
        try:
            session = SessionLocal()
            db_version = (
                session.query(ModelVersionDB)
                .filter_by(model_name=model_name, is_active=True, is_approved=True)
                .first()
            )
            result = self._to_domain(db_version) if db_version else None
            session.close()
            return result
        except Exception as e:
            raise RepositoryError("get_active_version", str(e))

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all registered versions for a given model name."""
        try:
            session = SessionLocal()
            versions = (
                session.query(ModelVersionDB)
                .filter_by(model_name=model_name)
                .order_by(ModelVersionDB.created_at.desc())
                .all()
            )
            result = [self._to_domain(v) for v in versions]
            session.close()
            return result
        except Exception as e:
            raise RepositoryError("list_model_versions", str(e))

    @staticmethod
    def _to_domain(db: ModelVersionDB) -> ModelVersion:
        return ModelVersion(
            id=db.id,
            model_name=db.model_name,
            version=db.version,
            file_path=db.file_path,
            accuracy=db.accuracy,
            precision_score=db.precision_score,
            recall=db.recall,
            f1_score=db.f1_score,
            is_approved=db.is_approved,
            is_active=db.is_active,
            created_at=db.created_at,
            created_by=db.created_by,
            description=db.description,
        )
