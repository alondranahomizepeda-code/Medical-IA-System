"""
Inference Service (RF-05, RF-06, RF-12)
=========================================
Contains the AI classification engine. Loads an approved model from the
Model Registry and executes predictions using the Strategy and Decorator patterns.

This service represents the highest computational demand in the system and
is designed to be independently scalable (RNF-02).
"""

from typing import Dict, Optional
import numpy as np

from domain.interfaces.contracts import (
    IModelInferencer,
    IAuditRepository,
    AuditEntry,
    AuditAction,
)
from services.model_factory import ModelFactory
from services.inference_strategy import ScoliosisStrategy
from infrastructure.model_registry import ModelRegistry
from config.settings import DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION
from utils.logger import get_logger
from utils.exceptions import InferenceError, ModelNotFoundError

logger = get_logger("InferenceService")


class InferenceService:
    """
    AI Inference Service for Scoliosis.

    Responsibilities:
      - Load an approved model version from the Model Registry.
      - Execute classification via Strategy + Decorator chain.
      - Return predictions with confidence scores and affected vertebrae analysis.

    The service depends on IModelInferencer (abstraction), not on
    any specific model implementation (DIP).
    """

    def __init__(
        self,
        model_registry: ModelRegistry,
        audit_repository: IAuditRepository,
    ):
        self._registry = model_registry
        self._audit_repo = audit_repository
        self._inferencer: Optional[IModelInferencer] = None
        self._current_model_name: Optional[str] = None
        self._current_model_version: Optional[str] = None

    def load_model(self, model_name: str = DEFAULT_MODEL_NAME) -> Dict:
        """
        Load the active model version from the Model Registry.

        Uses Factory Method to create a fully-decorated inferencer:
        LoggingDecorator → MetricsDecorator → BaseClassifier(Strategy).
        """
        logger.info(f"Loading active model for: {model_name}")

        # Check the registry for an active version
        active_version = self._registry.get_active_version(model_name)

        strategy = ScoliosisStrategy()

        if active_version:
            # Load the approved and active model
            self._inferencer = ModelFactory.create_model(
                strategy=strategy,
                model_name=active_version.model_name,
                model_version=active_version.version,
                model_path=active_version.file_path,
            )
            self._current_model_name = active_version.model_name
            self._current_model_version = active_version.version

            logger.info(
                f"Model loaded: {active_version.model_name} "
                f"v{active_version.version}"
            )
            return {
                "status": "loaded",
                "model_name": active_version.model_name,
                "version": active_version.version,
                "accuracy": active_version.accuracy,
            }
        else:
            # No active model — create with base strategy (fallback)
            logger.warning(
                f"No active model version found for '{model_name}'. "
                f"Using base strategy (no fine-tuned model)."
            )
            self._inferencer = ModelFactory.create_model(
                strategy=strategy,
                model_name=model_name,
                model_version="base",
            )
            self._current_model_name = model_name
            self._current_model_version = "base"

            return {
                "status": "fallback",
                "model_name": model_name,
                "version": "base",
                "note": "No fine-tuned model available. Register and activate one.",
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

        The inference passes through the Decorator chain:
        LoggingDecorator → MetricsDecorator → BaseClassifier(Strategy).

        Args:
            preprocessed_image: NumPy array ready for model input.
            image_id:          Database ID of the source image.
            user_id:           ID of the user requesting classification.
            filename:          Filename of the original image.
            original_image:    Original image array before resizing.

        Returns:
            Classification result with confidence score, affected vertebrae, and metadata.
        """
        if self._inferencer is None:
            self.load_model()

        if self._inferencer is None:
            raise InferenceError("No model inferencer available")

        logger.info(f"Classifying image ID={image_id}...")

        try:
            # Execute through the decorated inferencer
            result = self._inferencer.predict(
                preprocessed_image,
                filename=filename,
                original_image=original_image,
            )

            # Enrich result with context
            result["image_id"] = image_id
            result["model_name"] = self._current_model_name
            result["model_version"] = self._current_model_version

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
                    f"Model: {self._current_model_name} v{self._current_model_version}"
                ),
            ))

            logger.info(
                f"Classification complete for image ID={image_id}: "
                f"{result.get('classification')} ({result.get('confidence_score')}) | "
                f"Vertebrae: {result.get('affected_vertebrae')}"
            )
            return result

        except Exception as e:
            logger.error(f"Classification failed for image ID={image_id}: {e}")
            raise InferenceError(f"Classification failed: {str(e)}")


    def get_model_info(self) -> Dict:
        """Return information about the currently loaded model."""
        return {
            "model_name": self._current_model_name,
            "model_version": self._current_model_version,
            "is_loaded": self._inferencer is not None,
        }
