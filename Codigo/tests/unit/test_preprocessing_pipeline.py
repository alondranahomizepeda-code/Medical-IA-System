"""
Unit Tests: Preprocessing Pipeline (Pipeline Pattern & Fail-Fast)
=================================================================
Tests medical image validation, fail-fast boundary rejection,
normalization, resizing, and quality checks using the Pipeline Pattern.
"""

import io
import pytest
import numpy as np
from PIL import Image

from pipeline.core import AIPipeline, PipelineContext
from pipeline.stages import (
    ValidationStage,
    AnonymizationStage,
    NormalizationStage,
    QualityCheckStage,
)
from services.preprocessing_service import PreprocessingService
from infrastructure.repositories import AuditRepository
from utils.exceptions import UnsupportedFormatError, ImageQualityError, CorruptedImageError


class TestPreprocessingPipelineUnit:
    """Unit test suite for clinical image preprocessing pipeline."""

    def test_pipeline_valid_image_execution(self, sample_xray_image: bytes):
        """Test end-to-end pipeline processing for a valid spinal X-ray."""
        pipeline = AIPipeline()
        pipeline.add_stage(ValidationStage())
        pipeline.add_stage(AnonymizationStage())
        pipeline.add_stage(QualityCheckStage())
        pipeline.add_stage(NormalizationStage())

        context = PipelineContext(
            original_filename="spine_ap.jpg",
            file_data=sample_xray_image,
        )

        result_ctx = pipeline.execute(context)

        assert len(result_ctx.errors) == 0
        assert result_ctx.validation_passed is True
        assert result_ctx.preprocessed_image is not None
        # Must be normalized with batch dimension (1, 224, 224, 3)
        assert result_ctx.preprocessed_image.shape == (1, 224, 224, 3)
        # Array must be normalized between 0.0 and 1.0
        assert result_ctx.preprocessed_image.min() >= 0.0
        assert result_ctx.preprocessed_image.max() <= 1.0

    def test_fail_fast_image_below_min_dimension(self, sample_invalid_small_image: bytes):
        """Test Fail-Fast rejection when image is smaller than 32x32 px."""
        stage = ValidationStage()
        context = PipelineContext(
            original_filename="tiny.jpg",
            file_data=sample_invalid_small_image,
        )

        with pytest.raises(ImageQualityError) as exc_info:
            stage.execute(context)

        assert "too small" in str(exc_info.value).lower()

    def test_fail_fast_corrupted_image(self, sample_corrupt_file: bytes):
        """Test Fail-Fast rejection when input bytes cannot be parsed as valid image."""
        stage = ValidationStage()
        context = PipelineContext(
            original_filename="corrupted.jpg",
            file_data=sample_corrupt_file,
        )

        with pytest.raises(CorruptedImageError):
            stage.execute(context)

    def test_format_validation_rejected(self):
        """Test format stage rejects non-medical file extensions."""
        stage = ValidationStage()
        context = PipelineContext(
            original_filename="malicious_script.exe",
            file_data=b"fake exe content",
        )

        with pytest.raises(UnsupportedFormatError):
            stage.execute(context)

    def test_preprocessing_service_preprocess_method(self, sample_xray_image: bytes):
        """Test PreprocessingService wrapper executes pipeline on raw numpy array."""
        audit_repo = AuditRepository()
        svc = PreprocessingService(audit_repository=audit_repo)

        # Decode image to numpy array
        img = Image.open(io.BytesIO(sample_xray_image)).convert("RGB")
        raw_arr = np.array(img)

        result = svc.preprocess(image_array=raw_arr, image_id=1, filename="test.jpg", user_id=1)
        assert "preprocessed_image" in result
        assert result["preprocessed_image"].shape == (1, 224, 224, 3)
