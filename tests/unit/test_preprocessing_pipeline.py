"""
Unit Tests: Preprocessing Pipeline (Pipeline Pattern, Fail-Fast & Question 6)
=============================================================================
Tests medical image validation, fail-fast boundary rejection, memory-efficient
int16/float32 operations vs int64/float64 (Question 6), 16-bit DICOM decoding
and PHI anonymization, filename sanitization, and normalization.
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
    decode_image,
    sanitize_filename,
)
from services.preprocessing_service import PreprocessingService
from infrastructure.repositories import AuditRepository
from utils.exceptions import (
    UnsupportedFormatError,
    FileSizeExceededError,
    ImageQualityError,
    CorruptedImageError,
)


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
        assert result_ctx.preprocessed_image.shape == (1, 224, 224, 3)
        assert result_ctx.preprocessed_image.dtype == np.float32
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

    def test_fail_fast_oversized_file_rejected(self):
        """Test FileSizeExceededError when payload exceeds 15 MB."""
        stage = ValidationStage()
        oversized = b"0" * (15 * 1024 * 1024 + 100)
        context = PipelineContext(original_filename="huge.jpg", file_data=oversized)

        with pytest.raises(FileSizeExceededError):
            stage.execute(context)

    def test_color_photograph_rejected(self):
        """Verify colorful non-radiograph images are rejected by ValidationStage."""
        arr = np.zeros((128, 128, 3), dtype=np.uint8)
        arr[:, :, 0] = 240  # Red channel
        arr[:, :, 1] = 30   # Green channel
        arr[:, :, 2] = 20   # Blue channel
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")

        stage = ValidationStage()
        ctx = PipelineContext(original_filename="color_photo.png", file_data=buf.getvalue())
        with pytest.raises(ImageQualityError, match="color photograph"):
            stage.execute(ctx)

    def test_white_background_document_rejected(self):
        """Verify white-background documents or logos are rejected as non-radiographs."""
        arr = np.full((128, 128, 3), 250, dtype=np.uint8)
        arr[50:70, 50:70, :] = 20
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")

        stage = ValidationStage()
        ctx = PipelineContext(original_filename="document_scan.png", file_data=buf.getvalue())
        with pytest.raises(ImageQualityError, match="white background"):
            stage.execute(ctx)

    def test_pipeline_stops_at_first_failure(self, sample_corrupt_file: bytes):
        """Verify that when ValidationStage fails, subsequent stages do not execute."""
        pipeline = AIPipeline()
        pipeline.add_stage(ValidationStage())
        pipeline.add_stage(AnonymizationStage())
        pipeline.add_stage(NormalizationStage())

        ctx = PipelineContext(original_filename="broken.jpg", file_data=sample_corrupt_file)
        res = pipeline.execute(ctx)
        assert len(res.errors) == 1
        assert res.stages_completed == []
        assert res.is_anonymized is False

    def test_int16_float32_numerical_equivalence_and_overflow_safety_question_6(self):
        """
        Question 6 verification:
        1. Shows why uint8 subtraction wraps around (e.g. uint8(10) - uint8(200) = 66 instead of -190),
           whereas int16 safely represents [-255, +255] in 2 bytes instead of int64's 8 bytes (75% less RAM).
        2. Shows that float32 luminance/contrast statistics match float64 within < 1e-4 tolerance.
        """
        rng = np.random.default_rng(42)
        rgb = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)

        # Reference int64 / float64 calculation
        r64, g64, b64 = rgb[:, :, 0].astype(np.int64), rgb[:, :, 1].astype(np.int64), rgb[:, :, 2].astype(np.int64)
        var_ref = float(np.mean(np.abs(r64 - g64), dtype=np.float64) + np.mean(np.abs(r64 - b64), dtype=np.float64))

        # Optimized int16 / float32 calculation
        r16, g16, b16 = rgb[:, :, 0].astype(np.int16), rgb[:, :, 1].astype(np.int16), rgb[:, :, 2].astype(np.int16)
        var_opt = float(np.mean(np.abs(r16 - g16), dtype=np.float32) + np.mean(np.abs(r16 - b16), dtype=np.float32))

        assert r16.nbytes == r64.nbytes // 4
        assert abs(var_ref - var_opt) < 1e-4

    def test_dicom_16bit_decode_and_phi_anonymization(self):
        """Verify 16-bit DICOM decoding (MONOCHROME1/2) and PHI tag masking in AnonymizationStage."""
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = generate_uid()
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset("test.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.PatientName = "Juan Perez"
        ds.PatientID = "MX-998877"
        ds.Modality = "DX"
        ds.PhotometricInterpretation = "MONOCHROME1"
        ds.Rows = 64
        ds.Columns = 64
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0

        # Create 16-bit radiograph-like array (MONOCHROME1: high values = dark background)
        px = np.full((64, 64), 3500, dtype=np.uint16)
        px[:, 24:40] = 800  # Inverted bright spine center
        ds.PixelData = px.tobytes()

        buf = io.BytesIO()
        pydicom.dcmwrite(buf, ds)
        dcm_bytes = buf.getvalue()

        rgb_arr, dcm_meta = decode_image(dcm_bytes, "study.dcm")
        assert rgb_arr.shape == (64, 64, 3)
        assert rgb_arr.dtype == np.uint8
        assert dcm_meta["PatientName"] == "Juan Perez"

        ctx = PipelineContext(original_filename="study.dcm", file_data=dcm_bytes, image_array=rgb_arr)
        ctx.metadata["dicom_metadata"] = dcm_meta
        ctx = AnonymizationStage().execute(ctx)
        assert ctx.is_anonymized is True
        assert ctx.metadata["dicom_metadata"]["PatientName"] == "[ANONYMIZED]"
        assert ctx.metadata["dicom_metadata"]["PatientID"] == "[ANONYMIZED]"

    @pytest.mark.parametrize(
        "raw_name,expected_safe",
        [
            ("../../main.py.jpg", "main.py.jpg"),
            (r"..\..\evil.png", "evil.png"),
            ("radiografía paciente#1.jpg", "radiograf_a_paciente_1.jpg"),
            ("", "unnamed"),
        ],
    )
    def test_upload_filename_sanitization(self, raw_name: str, expected_safe: str):
        """Verify sanitize_filename blocks path traversal and strips unsafe characters."""
        assert sanitize_filename(raw_name) == expected_safe

    def test_preprocessing_service_preprocess_method(self, sample_xray_image: bytes):
        """Test PreprocessingService wrapper executes pipeline on raw numpy array."""
        audit_repo = AuditRepository()
        svc = PreprocessingService(audit_repository=audit_repo)

        img = Image.open(io.BytesIO(sample_xray_image)).convert("RGB")
        raw_arr = np.array(img)

        result = svc.preprocess(image_array=raw_arr, image_id=1, filename="test.jpg", user_id=1)
        assert "preprocessed_image" in result
        assert result["preprocessed_image"].shape == (1, 224, 224, 3)
