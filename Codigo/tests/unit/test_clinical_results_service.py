"""
Unit Tests: ClinicalResultsService & Timezone Localization
==========================================================
Tests clinical prediction persistence, specialist review (confirm/correct),
report generation (PDF & text fallback), and Mexican timezone conversion.
"""

from datetime import datetime, timezone
from pathlib import Path
import pytest

from services.clinical_results_service import (
    ClinicalResultsService,
    format_mexico_datetime,
    MEXICO_TZ,
)
from infrastructure.repositories import ResultRepository, ImageRepository, AuditRepository
from domain.interfaces.contracts import PredictionStatus


@pytest.fixture
def clinical_service():
    result_repo = ResultRepository()
    image_repo = ImageRepository()
    audit_repo = AuditRepository()
    return ClinicalResultsService(result_repo, image_repo, audit_repo)


class TestClinicalResultsServiceUnit:
    """Unit test suite for ClinicalResultsService and Timezone."""

    def test_format_mexico_datetime_spanish(self):
        """Test formatting UTC datetime to Mexico Central Time in Spanish (12-hour format)."""
        # 2026-10-26 20:30 UTC = 2026-10-26 14:30 Mexico (CST)
        utc_dt = datetime(2026, 10, 26, 20, 30, tzinfo=timezone.utc)
        formatted = format_mexico_datetime(utc_dt, is_es=True)

        assert "26/10/2026 02:30 p. m. (CST)" in formatted

    def test_format_mexico_datetime_english(self):
        """Test formatting UTC datetime to Mexico Central Time in English (12-hour format)."""
        utc_dt = datetime(2026, 10, 26, 15, 15, tzinfo=timezone.utc)
        # 15:15 UTC = 09:15 AM Mexico
        formatted = format_mexico_datetime(utc_dt, is_es=False)

        assert "2026-10-26 09:15 AM (CST)" in formatted

    def test_format_mexico_datetime_none_handling(self):
        """Test safe handling of None datetimes."""
        assert format_mexico_datetime(None) == "N/A"

    def test_clinical_review_confirmation(self, clinical_service: ClinicalResultsService):
        """Test specialist confirming an AI diagnosis."""
        history = clinical_service.get_history(limit=1)
        if not history:
            pytest.skip("No prediction results in test database")

        r_id = history[0].id
        updated = clinical_service.review_prediction(
            result_id=r_id,
            reviewer_id=1,
            status=PredictionStatus.CONFIRMED,
            notes="Diagnostico verificado y concordante con proyeccion AP.",
        )

        assert updated.status == PredictionStatus.CONFIRMED
        assert "Diagnostico verificado" in updated.reviewer_notes

    def test_generate_pdf_report_spanish_and_english(self, clinical_service: ClinicalResultsService):
        """Test PDF generation produces existing files for both Spanish and English."""
        history = clinical_service.get_history(limit=1)
        if not history:
            pytest.skip("No prediction results in test database")

        r_id = history[0].id

        pdf_es = clinical_service.generate_report(r_id, reviewer_name="Dr. Ramirez", language="es")
        assert Path(pdf_es).exists()
        assert Path(pdf_es).stat().st_size > 1000

        pdf_en = clinical_service.generate_report(r_id, reviewer_name="Dr. Ramirez", language="en")
        assert Path(pdf_en).exists()
        assert Path(pdf_en).stat().st_size > 1000
