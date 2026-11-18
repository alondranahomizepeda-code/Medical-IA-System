"""
Image Management Service (RF-02, RF-03)
=========================================
Receives images, executes the Fail Fast validation, and stores valid
images for subsequent processing.
"""

import uuid
from typing import Dict, Optional

from domain.interfaces.contracts import (
    IImageRepository,
    IStorageAdapter,
    IAuditRepository,
    MedicalImage,
    AuditEntry,
    AuditAction,
)
from pipeline.core import AIPipeline, PipelineContext
from pipeline.stages import ValidationStage, AnonymizationStage, sanitize_filename
from utils.logger import get_logger
from utils.exceptions import ImageValidationError

logger = get_logger("ImageManagementService")


class ImageManagementService:
    """
    Handles image upload, validation (Fail Fast), and initial storage.

    This service is the entry point for medical images into the system.
    It orchestrates the first stages of the pipeline and delegates
    storage to the IStorageAdapter.
    """

    def __init__(
        self,
        image_repository: IImageRepository,
        storage_adapter: IStorageAdapter,
        audit_repository: IAuditRepository,
    ):
        self._image_repo = image_repository
        self._storage = storage_adapter
        self._audit_repo = audit_repository

        # Build the validation pipeline (Fail Fast stages only)
        self._validation_pipeline = AIPipeline()
        self._validation_pipeline.add_stage(ValidationStage())
        self._validation_pipeline.add_stage(AnonymizationStage())

    def upload_image(
        self,
        file_data: bytes,
        filename: str,
        uploaded_by: Optional[int] = None,
    ) -> Dict:
        """
        Upload and validate a medical image.

        Process:
          1. Sanitize filename and run Fail Fast validation pipeline.
          2. Store original image (unmodified — RD-13).
          3. Save metadata to database.
          4. Record audit entry.

        Returns:
            Dictionary with image metadata and validation results.

        Raises:
            ImageValidationError: If the image fails Fail Fast validation.
        """
        safe_name = sanitize_filename(filename)
        logger.info(f"Processing upload: {safe_name} ({len(file_data)} bytes)")

        # Step 1: Run validation pipeline (Fail Fast)
        context = PipelineContext(
            original_filename=safe_name,
            file_data=file_data,
        )
        context = self._validation_pipeline.execute(context)

        # Check for pipeline errors (Fail Fast triggered)
        if context.errors:
            error_msg = "; ".join(context.errors)
            logger.warning(f"Upload rejected (Fail Fast): {error_msg}")
            if isinstance(context.last_exception, ImageValidationError):
                raise context.last_exception
            raise ImageValidationError(error_msg)

        # Step 2: Store original image (preserved unmodified — RD-13)
        unique_filename = f"{uuid.uuid4().hex}_{safe_name}"
        stored_path = self._storage.store(file_data, unique_filename, "originals")

        # Step 3: Save metadata to database (with compensating disk cleanup if DB fails — Question 4)
        image = MedicalImage(
            filename=safe_name,
            original_path=stored_path,
            file_size_bytes=len(file_data),
            format=context.metadata.get("format", ""),
            width=context.metadata.get("original_width"),
            height=context.metadata.get("original_height"),
            uploaded_by=uploaded_by,
            is_anonymized=context.is_anonymized,
            metadata=context.metadata,
        )
        try:
            saved_image = self._image_repo.save(image)
        except Exception:
            try:
                self._storage.delete(stored_path)
            except Exception:
                pass
            raise

        # Step 4: Audit trail
        self._audit_repo.append(AuditEntry(
            action=AuditAction.IMAGE_UPLOADED,
            user_id=uploaded_by,
            resource_type="medical_image",
            resource_id=saved_image.id,
            details=f"Image '{safe_name}' uploaded and validated successfully",
        ))

        logger.info(f"Image uploaded successfully: ID={saved_image.id}")

        return {
            "image_id": saved_image.id,
            "filename": safe_name,
            "validation": "passed",
            "stages_completed": context.stages_completed,
            "metadata": context.metadata,
        }

    def get_image(self, image_id: int) -> Optional[MedicalImage]:
        """Retrieve image metadata by ID."""
        return self._image_repo.find_by_id(image_id)

    def list_images(self, limit: int = 50, offset: int = 0):
        """List all uploaded images."""
        return self._image_repo.find_all(limit=limit, offset=offset)
