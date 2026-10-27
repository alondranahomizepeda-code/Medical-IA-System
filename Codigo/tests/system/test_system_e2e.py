"""
System (End-to-End) Testing
===========================
Executes the complete clinical diagnostic lifecycle across all system actors:
  - Actor 1 (Imaging Technician): Authenticates, uploads X-ray, triggers preprocessing.
  - AI Inference Engine: Performs classification with Transfer Learning & Decorators.
  - Actor 2 (Radiologist): Reviews diagnosis, adds clinical notes, generates PDF report.
  - Actor 3 (Hospital Admin): Audits full chronological sequence with user attribution.
"""

import pytest
from fastapi.testclient import TestClient
from api.routes import _2fa_codes


class TestSystemEndToEnd:
    """System E2E test suite simulating full hospital workflow."""

    def test_full_clinical_diagnostic_workflow(self, client: TestClient, sample_xray_image: bytes):
        """End-to-end test simulating multi-role clinical journey."""

        # ── Step 1: Technician Authentication (2FA) ──
        tech_login = client.post(
            "/api/v1/auth/login",
            json={"username": "technician", "password": "password123"}
        )
        assert tech_login.status_code == 200
        tech_token = tech_login.json()["access_token"]
        tech_2fa_code = _2fa_codes["technician"]["code"]

        tech_verify = client.post(
            "/api/v1/auth/verify-2fa",
            json={"username": "technician", "code": tech_2fa_code}
        )
        assert tech_verify.status_code == 200
        tech_headers = {"Authorization": f"Bearer {tech_token}"}

        # ── Step 2: Technician Uploads Spinal Radiograph ──
        upload_resp = client.post(
            "/api/v1/images/upload",
            files={"file": ("patient_spine_AP.jpg", sample_xray_image, "image/jpeg")},
            headers=tech_headers,
        )
        assert upload_resp.status_code == 200
        upload_data = upload_resp.json()
        image_id = upload_data["image_id"]
        assert image_id > 0

        # ── Step 3: AI Classification (Pipelines Preprocessing -> Inference -> Save) ──
        class_resp = client.post(f"/api/v1/classify/{image_id}", headers=tech_headers)
        assert class_resp.status_code == 200
        class_data = class_resp.json()
        result_id = class_data["prediction_id"]
        assert class_data["classification"] in [
            "NO ESCOLIOSIS",
            "ESCOLIOSIS GRAVE (EN C)",
            "ESCOLIOSIS SEVERA (EN S)",
        ]
        assert class_data["confidence_score"] > 0.0

        # ── Step 5: Radiologist Authentication (2FA) ──
        rad_login = client.post(
            "/api/v1/auth/login",
            json={"username": "radiologist", "password": "password123"}
        )
        assert rad_login.status_code == 200
        rad_token = rad_login.json()["access_token"]
        rad_2fa_code = _2fa_codes["radiologist"]["code"]

        rad_verify = client.post(
            "/api/v1/auth/verify-2fa",
            json={"username": "radiologist", "code": rad_2fa_code}
        )
        assert rad_verify.status_code == 200
        rad_headers = {"Authorization": f"Bearer {rad_token}"}

        # ── Step 6: Radiologist Clinical Review ──
        review_resp = client.post(
            f"/api/v1/results/{result_id}/review",
            json={
                "status": "confirmed",
                "notes": "Angulo de Cobb y alineacion vertebral confirmados en consenso.",
            },
            headers=rad_headers,
        )
        assert review_resp.status_code == 200
        assert review_resp.json()["status"] == "confirmed"

        # ── Step 7: Radiologist Emits Formal PDF Report ──
        report_resp = client.post(
            f"/api/v1/results/{result_id}/report",
            json={"language": "es", "reviewer_name": "Dr. Radiologist Specialist"},
            headers=rad_headers,
        )
        assert report_resp.status_code == 200
        report_data = report_resp.json()
        assert "download_url" in report_data
        pdf_download_url = report_data["download_url"]

        # Validate PDF binary can be downloaded
        download_resp = client.get(pdf_download_url)
        assert download_resp.status_code == 200
        assert download_resp.headers["content-type"] == "application/pdf"
        assert len(download_resp.content) > 1000

        # ── Step 8: Administrator Audits Hospital Activities ──
        admin_login = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "password123"}
        )
        admin_token = admin_login.json()["access_token"]
        admin_2fa_code = _2fa_codes["admin"]["code"]
        admin_verify = client.post(
            "/api/v1/auth/verify-2fa",
            json={"username": "admin", "code": admin_2fa_code}
        )
        assert admin_verify.status_code == 200
        admin_headers = {"Authorization": f"Bearer {admin_token}"}

        audit_resp = client.get("/api/v1/audit?limit=50", headers=admin_headers)
        assert audit_resp.status_code == 200
        audit_data = audit_resp.json()
        assert audit_data["count"] > 0

        # Verify that reported actions are present
        actions = [entry["action"] for entry in audit_data["entries"]]
        assert "report_generated" in actions
        assert "classification_completed" in actions
