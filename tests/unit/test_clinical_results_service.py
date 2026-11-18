"""
Unit Tests: ClinicalResultsService & Timezone Localization
==========================================================
Tests clinical prediction persistence, specialist review (confirm/correct),
report generation (PDF in ES and EN & text fallback), and Mexican timezone conversion.
"""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from services.clinical_results_service import (
    ClinicalResultsService,
    format_mexico_datetime,
)
from infrastructure.repositories import ResultRepository, ImageRepository, AuditRepository
from domain.interfaces.contracts import PredictionStatus, MedicalImage


@pytest.fixture
def clinical_service():
    result_repo = ResultRepository()
    image_repo = ImageRepository()
    audit_repo = AuditRepository()
    return ClinicalResultsService(result_repo, image_repo, audit_repo)


@pytest.fixture
def seeded_prediction(clinical_service: ClinicalResultsService):
    """Ensure at least one image and prediction exist in the database for unit tests."""
    img = clinical_service._image_repo.save(
        MedicalImage(
            filename="unit_test_spine.jpg",
            original_path="originals/unit_test_spine.jpg",
            file_size_bytes=20480,
            format=".jpg",
            width=600,
            height=800,
            uploaded_by=1,
            is_anonymized=True,
        )
    )
    pred = clinical_service.save_prediction(
        image_id=img.id,
        classification="ESCOLIOSIS GRAVE (EN C)",
        confidence_score=0.88,
        model_name="scoliosis_classifier",
        model_version="1.1.0",
        processing_time_ms=25.4,
        affected_vertebrae="T7 - T11",
        curve_type="C-Shaped Curve (Simple)",
        curve_detail="Single C-shaped curve toward the right. Apex at T9.",
    )
    return pred


class TestClinicalResultsServiceUnit:
    """Unit test suite for ClinicalResultsService and Timezone."""

    def test_format_mexico_datetime_spanish(self):
        """Test formatting UTC datetime to Mexico Central Time in Spanish (12-hour format)."""
        utc_dt = datetime(2026, 10, 26, 20, 30, tzinfo=timezone.utc)
        formatted = format_mexico_datetime(utc_dt, is_es=True)
        assert "26/10/2026 02:30 p. m. (CST)" in formatted

    def test_format_mexico_datetime_english(self):
        """Test formatting UTC datetime to Mexico Central Time in English (12-hour format)."""
        utc_dt = datetime(2026, 10, 26, 15, 15, tzinfo=timezone.utc)
        formatted = format_mexico_datetime(utc_dt, is_es=False)
        assert "2026-10-26 09:15 AM (CST)" in formatted

    def test_format_mexico_datetime_naive_utc(self):
        """Verify naive datetime is treated as UTC and converted to Mexico City time."""
        naive_dt = datetime(2026, 10, 26, 18, 0)
        formatted = format_mexico_datetime(naive_dt, is_es=True)
        assert "12:00 p. m. (CST)" in formatted

    def test_format_mexico_datetime_none_handling(self):
        """Test safe handling of None datetimes."""
        assert format_mexico_datetime(None) == "N/A"

    def test_save_prediction_starts_as_pending_review(self, seeded_prediction):
        """Every newly saved prediction must start in PENDING_REVIEW status."""
        assert seeded_prediction.id is not None
        assert seeded_prediction.status == PredictionStatus.PENDING_REVIEW

    def test_clinical_review_confirmation(self, clinical_service: ClinicalResultsService, seeded_prediction):
        """Test specialist confirming an AI diagnosis."""
        updated = clinical_service.review_prediction(
            result_id=seeded_prediction.id,
            reviewer_id=1,
            status=PredictionStatus.CONFIRMED,
            notes="Diagnostico verificado y concordante con proyeccion AP.",
        )
        assert updated is not None
        assert updated.status == PredictionStatus.CONFIRMED
        assert "Diagnostico verificado" in updated.reviewer_notes

    def test_clinical_review_correction(self, clinical_service: ClinicalResultsService, seeded_prediction):
        """Test specialist correcting an AI diagnosis."""
        updated = clinical_service.review_prediction(
            result_id=seeded_prediction.id,
            reviewer_id=1,
            status=PredictionStatus.CORRECTED,
            notes="Se corrige a escoliosis en S tras medicion manual de Cobb.",
        )
        assert updated is not None
        assert updated.status == PredictionStatus.CORRECTED

    def test_get_result_and_get_result_by_image(self, clinical_service: ClinicalResultsService, seeded_prediction):
        """Verify retrieving prediction by ID and by image_id."""
        by_id = clinical_service.get_result(seeded_prediction.id)
        by_img = clinical_service.get_result_by_image(seeded_prediction.image_id)
        assert by_id is not None and by_id.id == seeded_prediction.id
        assert by_img is not None and by_img.image_id == seeded_prediction.image_id

    def test_generate_pdf_report_spanish_and_english(
        self, clinical_service: ClinicalResultsService, seeded_prediction
    ):
        """Test PDF generation produces existing files for both Spanish and English."""
        r_id = seeded_prediction.id

        pdf_es = clinical_service.generate_report(r_id, reviewer_name="Dr. Ramirez", language="es")
        assert Path(pdf_es).exists()
        assert Path(pdf_es).stat().st_size > 1000

        pdf_en = clinical_service.generate_report(r_id, reviewer_name="Dr. Ramirez", language="en")
        assert Path(pdf_en).exists()
        assert Path(pdf_en).stat().st_size > 1000

    def test_generate_text_report_fallback(self, clinical_service: ClinicalResultsService, seeded_prediction):
        """Test text report fallback generation in both languages."""
        img = clinical_service._image_repo.find_by_id(seeded_prediction.image_id)
        txt_es = clinical_service._generate_text_report(seeded_prediction, img, "Dr. Test", "es")
        txt_en = clinical_service._generate_text_report(seeded_prediction, img, "Dr. Test", "en")
        assert Path(txt_es).exists()
        assert Path(txt_en).exists()

    def test_generate_report_unknown_result_raises(self, clinical_service: ClinicalResultsService):
        """Generating a report for a non-existent prediction ID must raise ValueError."""
        with pytest.raises(ValueError, match="not found"):
            clinical_service.generate_report(999999, reviewer_name="Dr. Nobody", language="es")
