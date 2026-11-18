"""
Inference Service (RF-05, RF-06, RF-12)
=========================================
Contains the AI classification engine. Loads an approved model from the
Model Registry and executes predictions using the Strategy and Decorator patterns.

This service represents the highest computational demand in the system and
is designed to be independently scalable (RNF-02) and decoupled from concrete
CNN architectures via Dependency Inversion (DIP) and Strategy Pattern (Question 3).
"""

from pathlib import Path
from typing import Callable, Dict, Optional
import numpy as np

from domain.interfaces.contracts import (
    IModelInferencer,
    IMLStrategy,
    IAuditRepository,
    AuditEntry,
    AuditAction,
)
from services.model_factory import ModelFactory
from services.inference_strategy import ScoliosisStrategy
from infrastructure.model_registry import ModelRegistry
from config.settings import DEFAULT_MODEL_NAME, ALLOW_UNAPPROVED_MODEL
from utils.logger import get_logger
from utils.exceptions import InferenceError, ModelNotLoadedError

logger = get_logger("InferenceService")


class InferenceService:
    """
    AI Inference Service for Scoliosis.

    Responsibilities:
      - Load ONLY an approved model version from the Model Registry (Question 10).
      - Execute classification via Strategy + Decorator chain.
      - Return predictions with confidence scores, review flags, and vertebral analysis.

    The service depends on abstractions (IModelInferencer, IMLStrategy) injected
    through `strategy_factory`, never hard-coding a concrete CNN inside business logic (DIP, OCP).
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        audit_repository: IAuditRepository,
        strategy_factory: Callable[[], IMLStrategy] = ScoliosisStrategy,
        auto_bootstrap: bool = True,
    ):
        self._registry = model_registry
        self._audit_repo = audit_repository
        self._strategy_factory = strategy_factory
        self._auto_bootstrap = auto_bootstrap
        self._inferencer: Optional[IModelInferencer] = None
        self._current_model_name: Optional[str] = None
        self._current_model_version: Optional[str] = None
        self._is_approved: bool = False

    @property
    def is_ready(self) -> bool:
        """Return True only when an approved model is loaded and ready to serve predictions."""
        return self._inferencer is not None and (self._is_approved or ALLOW_UNAPPROVED_MODEL)

    def load_model(self, model_name: str = DEFAULT_MODEL_NAME) -> Dict:
        """
        Load the active and approved model version from the Model Registry.

        Uses Factory Method to create a fully-decorated inferencer:
        LoggingDecorator → MetricsDecorator → BaseModelInferencer(Strategy).
        """
        logger.info(f"Loading active model for: {model_name}")

        active_version = self._registry.get_active_version(model_name)
        if not active_version and self._auto_bootstrap:
            active_version = self._registry.bootstrap_packaged_models(model_name)

        if active_version and active_version.is_approved and Path(active_version.file_path).exists():
            strategy = self._strategy_factory()
            self._inferencer = ModelFactory.create_model(
                strategy=strategy,
                model_name=active_version.model_name,
                model_version=active_version.version,
                model_path=active_version.file_path,
            )
            self._current_model_name = active_version.model_name
            self._current_model_version = active_version.version
            self._is_approved = True

            logger.info(
                f"Approved model loaded: {active_version.model_name} "
                f"v{active_version.version}"
            )
            return {
                "status": "loaded",
                "model_name": active_version.model_name,
                "version": active_version.version,
                "accuracy": active_version.accuracy,
                "f1_score": active_version.f1_score,
                "is_approved": True,
            }

        self._inferencer = None
        self._current_model_name = model_name
        self._current_model_version = None
        self._is_approved = False

        logger.warning(
            f"No approved active model version found for '{model_name}'. "
            f"Readiness probe and classification will return 503 until an approved model is activated."
        )
        return {
            "status": "not_ready",
            "model_name": model_name,
            "version": None,
            "is_approved": False,
            "note": "No approved model available in Model Registry.",
        }

    def classify(
        self,
        preprocessed_image: np.ndarray,
        image_id: int,
        user_id: Optional[int] = None,
        filename: str = "",
        original_image: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Execute AI classification on a preprocessed image.

        Raises ModelNotLoadedError (mapped to HTTP 503) if no approved model is active.
        """
        if self._inferencer is None:
            self.load_model(self._current_model_name or DEFAULT_MODEL_NAME)

        if self._inferencer is None or not self.is_ready:
            raise ModelNotLoadedError(
                f"No approved model version is active for '{self._current_model_name or DEFAULT_MODEL_NAME}'. "
                f"A model must pass the quality gate and be approved in ModelRegistry before clinical use."
            )

        logger.info(f"Classifying image ID={image_id}...")

        try:
            result = self._inferencer.predict(
                preprocessed_image,
                filename=filename,
                original_image=original_image,
            )

            result["image_id"] = image_id
            result["model_name"] = self._current_model_name
            result["model_version"] = self._current_model_version
            result["model_approved"] = self._is_approved

            # Audit trail: classification completed
            self._audit_repo.append(AuditEntry(
                action=AuditAction.CLASSIFICATION_COMPLETED,
                user_id=user_id,
                resource_type="prediction",
                resource_id=image_id,
                details=(
                    f"Classification: {result.get('classification')} | "
                    f"Vertebrae: {result.get('affected_vertebrae')} | "
                    f"Confidence: {result.get('confidence_score')} | "
                    f"RequiresReview: {result.get('requires_review')} | "
                    f"Model: {self._current_model_name} v{self._current_model_version}"
                ),
            ))

            logger.info(
                f"Classification complete for image ID={image_id}: "
                f"{result.get('classification')} ({result.get('confidence_score')}) | "
                f"Vertebrae: {result.get('affected_vertebrae')}"
            )
            return result

        except ModelNotLoadedError:
            raise
        except Exception as e:
            logger.error(f"Classification failed for image ID={image_id}: {e}")
            raise InferenceError(f"Classification failed: {str(e)}")

    def get_model_info(self) -> Dict:
        """Return information about the currently loaded model."""
        return {
            "model_name": self._current_model_name,
            "model_version": self._current_model_version,
            "is_loaded": self._inferencer is not None,
            "is_approved": self._is_approved,
            "is_ready": self.is_ready,
        }
