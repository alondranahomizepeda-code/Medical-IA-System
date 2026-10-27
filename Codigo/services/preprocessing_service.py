"""
Preprocessing Service (RF-04)
===============================
Prepares a working copy of the medical image for AI inference.

Operations: normalization, resizing, and quality assessment.
The original image is NOT modified (RD-13).
"""

from typing import Dict, Optional
import numpy as np

from pipeline.core import AIPipeline, PipelineContext
from pipeline.stages import NormalizationStage, QualityCheckStage
from domain.interfaces.contracts import IAuditRepository, AuditEntry, AuditAction
from utils.logger import get_logger
from utils.exceptions import PipelineStageError

logger = get_logger("PreprocessingService")


class PreprocessingService:
    """
    Prepares images for the Inference Service.

    Uses the Pipeline pattern with composition of specialized stages:
      1. QualityCheckStage — assess image quality (RD-09).
      2. NormalizationStage — resize and normalize for model input (RF-04).

    This is a separate service from Image Management, following SoC.
    """

    def __init__(self, audit_repository: IAuditRepository):
        self._audit_repo = audit_repository

        # Build the preprocessing pipeline
        self._pipeline = AIPipeline()
        self._pipeline.add_stage(QualityCheckStage())
        self._pipeline.add_stage(NormalizationStage())

    def preprocess(
        self,
        image_array: np.ndarray,
        image_id: int,
        filename: str = "",
        user_id: Optional[int] = None,
    ) -> Dict:
        """
        Preprocess an image for AI inference.

        Args:
            image_array: Raw image as NumPy array (from validation stage).
            image_id:    Database ID of the image.
            filename:    Original filename (for logging).
            user_id:     ID of the user who requested the operation.

        Returns:
            Dictionary with preprocessed image and metadata.
        """
        logger.info(f"Preprocessing image ID={image_id}: {filename} (User: {user_id})")

        context = PipelineContext(
            original_filename=filename,
            image_array=image_array,
        )
        context = self._pipeline.execute(context)

        if context.errors:
            error_msg = "; ".join(context.errors)
            logger.error(f"Preprocessing failed: {error_msg}")
            raise PipelineStageError("Preprocessing", error_msg)

        # Audit trail
        self._audit_repo.append(AuditEntry(
            action=AuditAction.IMAGE_PREPROCESSED,
            user_id=user_id,
            resource_type="medical_image",
            resource_id=image_id,
            details=f"Image preprocessed: {context.stages_completed}",
        ))

        logger.info(
            f"Preprocessing complete for image ID={image_id} — "
            f"Shape: {context.preprocessed_image.shape if context.preprocessed_image is not None else 'N/A'}"
        )

        return {
            "preprocessed_image": context.preprocessed_image,
            "quality_report": context.metadata.get("quality", {}),
            "preprocessing_details": context.metadata.get("preprocessing", {}),
            "stages_completed": context.stages_completed,
        }
