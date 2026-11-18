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

import json
from pathlib import Path
from typing import Dict, List, Optional

from config.settings import MODEL_DIR, DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION
from domain.interfaces.contracts import ModelVersion
from infrastructure.database import session_scope, ModelVersionDB
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

    @staticmethod
    def resolve_model_path(file_path: str) -> str:
        """
        Resolve a stored model path (relative or legacy Windows path)
        against MODEL_DIR so artefacts work across Windows and Linux containers.
        """
        if not file_path:
            return ""
        normalized = file_path.replace("\\", "/")
        candidate = Path(normalized)
        if candidate.is_absolute() and candidate.exists():
            return str(candidate)

        # Check relative to MODEL_DIR
        rel_candidate = MODEL_DIR / normalized
        if rel_candidate.exists():
            return str(rel_candidate)

        # Handle legacy paths containing 'models/<folder>/model.keras'
        parts = normalized.split("/")
        if len(parts) >= 2:
            tail = Path(parts[-2]) / parts[-1]
            tail_candidate = MODEL_DIR / tail
            if tail_candidate.exists():
                return str(tail_candidate)

        return str(candidate)

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
        Register or update a model version in the registry.

        The version starts as NOT approved and NOT active.
        It must be explicitly approved and activated before use.
        """
        try:
            rel_path = file_path.replace("\\", "/")
            try:
                p = Path(file_path).resolve()
                m_dir = MODEL_DIR.resolve()
                if p.is_relative_to(m_dir):
                    rel_path = str(p.relative_to(m_dir)).replace("\\", "/")
            except Exception:
                pass

            with session_scope() as session:
                existing = (
                    session.query(ModelVersionDB)
                    .filter_by(model_name=model_name, version=version)
                    .first()
                )
                if existing:
                    existing.file_path = rel_path
                    if metrics:
                        existing.accuracy = metrics.get("accuracy")
                        existing.precision_score = metrics.get("precision")
                        existing.recall = metrics.get("recall")
                        existing.f1_score = metrics.get("f1_score")
                    existing.is_approved = False
                    existing.is_active = False
                    if created_by:
                        existing.created_by = created_by
                    if description:
                        existing.description = description
                    session.flush()
                    model_version = self._to_domain(existing)
                else:
                    db_version = ModelVersionDB(
                        model_name=model_name,
                        version=version,
                        file_path=rel_path,
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
                    session.flush()
                    model_version = self._to_domain(db_version)

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
            with session_scope() as session:
                db_version = (
                    session.query(ModelVersionDB)
                    .filter_by(model_name=model_name, version=version)
                    .first()
                )
                if not db_version:
                    raise ModelNotFoundError(f"{model_name} v{version}")

                db_version.is_approved = True
                session.flush()
                result = self._to_domain(db_version)

            logger.info(f"Model version approved: {model_name} v{version}")
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
            with session_scope() as session:
                target = (
                    session.query(ModelVersionDB)
                    .filter_by(model_name=model_name, version=version)
                    .first()
                )
                if not target:
                    raise ModelNotFoundError(f"{model_name} v{version}")

                if not target.is_approved:
                    raise RepositoryError(
                        "activate_version",
                        f"Version {version} is not approved. Approve it first.",
                    )

                session.query(ModelVersionDB).filter_by(
                    model_name=model_name, is_active=True
                ).update({"is_active": False})

                target.is_active = True
                session.flush()
                result = self._to_domain(target)

            logger.info(f"Model version ACTIVATED: {model_name} v{version}")
            return result

        except (ModelNotFoundError, RepositoryError):
            raise
        except Exception as e:
            raise RepositoryError("activate_model_version", str(e))

    def get_active_version(self, model_name: str) -> Optional[ModelVersion]:
        """Retrieve the currently active and approved model version."""
        try:
            with session_scope() as session:
                db_version = (
                    session.query(ModelVersionDB)
                    .filter_by(model_name=model_name, is_active=True, is_approved=True)
                    .first()
                )
                if not db_version:
                    return None
                domain_ver = self._to_domain(db_version)
                domain_ver.file_path = self.resolve_model_path(domain_ver.file_path)
                return domain_ver
        except Exception as e:
            raise RepositoryError("get_active_version", str(e))

    def list_versions(self, model_name: str) -> List[ModelVersion]:
        """List all registered versions for a given model name."""
        try:
            with session_scope() as session:
                versions = (
                    session.query(ModelVersionDB)
                    .filter_by(model_name=model_name)
                    .order_by(ModelVersionDB.created_at.desc())
                    .all()
                )
                return [self._to_domain(v) for v in versions]
        except Exception as e:
            raise RepositoryError("list_model_versions", str(e))

    def bootstrap_packaged_models(self, model_name: str = DEFAULT_MODEL_NAME) -> Optional[ModelVersion]:
        """
        Ensure packaged model artefacts on disk (e.g. v1.1.0, v1.0.0) are registered
        in the database on startup, and activate the preferred approved version.
        """
        active = self.get_active_version(model_name)
        if active and Path(active.file_path).exists():
            return active

        # Inspect candidate versions on disk, prioritizing DEFAULT_MODEL_VERSION (1.1.0)
        candidate_versions = [DEFAULT_MODEL_VERSION, "1.1.0", "1.0.0"]
        seen = set()
        for ver in candidate_versions:
            if ver in seen:
                continue
            seen.add(ver)
            ver_dir = MODEL_DIR / f"{model_name}_v{ver}"
            keras_file = ver_dir / "model.keras"
            meta_file = ver_dir / "metadata.json"
            if not keras_file.exists():
                continue

            meta = {}
            if meta_file.exists():
                try:
                    meta = json.loads(meta_file.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}

            metrics = meta.get("metrics", {})
            approved_in_meta = meta.get("approved", True)

            self.register_version(
                model_name=model_name,
                version=ver,
                file_path=f"{model_name}_v{ver}/model.keras",
                metrics=metrics,
                created_by="System Bootstrap",
                description=meta.get("technique", "Transfer Learning (MobileNetV2)"),
            )
            if approved_in_meta:
                self.approve_version(model_name, ver)
                return self.activate_version(model_name, ver)

        return self.get_active_version(model_name)

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
