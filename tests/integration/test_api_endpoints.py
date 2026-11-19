"""
Integration Tests: API Endpoints (API Gateway Layer — Questions 5 & 10)
=======================================================================
Tests the full integration between FastAPI routes, 2FA authentication security,
RBAC authorization enforcement on every endpoint (Question 5), Fail-Fast HTTP 413/422
rejection, DICOM upload & classification, path-traversal protection, and audit trail logging.
"""

import io
import pytest
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient
from api.routes import _2fa_codes


class TestApiIntegration:
    """Integration test suite for the REST API Gateway (20 tests)."""

    def test_health_and_readiness_probes(self, client: TestClient):
        """1. Test /health, /ready, /api/v1/health, and /api/v1/ready probes."""
        res_h = client.get("/api/v1/health")
        assert res_h.status_code == 200
        assert res_h.json()["status"] == "healthy"

        res_root_h = client.get("/health")
        assert res_root_h.status_code == 200

        res_r = client.get("/api/v1/ready")
        assert res_r.status_code in (200, 503)

    def test_auth_login_and_2fa_flow(self, client: TestClient):
        """2. Test authentication flow including two-factor verification."""
        login_res = client.post(
            "/api/v1/auth/login",
            json={"username": "radiologist", "password": "password123"},
        )
        assert login_res.status_code == 200
        login_data = login_res.json()
        assert login_data.get("status") == "2fa_required"

        # Before 2FA is verified, using the token on a protected endpoint must fail (401)
        unverified_headers = {"Authorization": f"Bearer {login_data['access_token']}"}
        blocked_res = client.get("/api/v1/auth/me", headers=unverified_headers)
        assert blocked_res.status_code == 401

        assert "radiologist" in _2fa_codes
        code = _2fa_codes["radiologist"]["code"]

        v2fa_res = client.post(
            "/api/v1/auth/verify-2fa",
            json={"username": "radiologist", "code": code},
        )
        assert v2fa_res.status_code == 200
        assert v2fa_res.json()["status"] == "verified"

    def test_wrong_2fa_code_rejected(self, client: TestClient):
        """3. Test that submitting an invalid 2FA code returns 401 Unauthorized."""
        client.post("/api/v1/auth/login", json={"username": "ml_engineer", "password": "password123"})
        real_code = _2fa_codes["ml_engineer"]["code"]

        bad_res = client.post(
            "/api/v1/auth/verify-2fa",
            json={"username": "ml_engineer", "code": "000000"},
        )
        assert bad_res.status_code == 401

        # Complete login with valid code to leave session clean
        ok_res = client.post(
            "/api/v1/auth/verify-2fa",
            json={"username": "ml_engineer", "code": real_code},
        )
        assert ok_res.status_code == 200

    def test_invalid_login_credentials_rejected(self, client: TestClient):
        """4. Test login with invalid password returns 401."""
        res = client.post(
            "/api/v1/auth/login",
            json={"username": "radiologist", "password": "WrongPassword999!"},
        )
        assert res.status_code == 401

    def test_resend_2fa_endpoint(self, client: TestClient):
        """5. Test /api/v1/auth/resend-2fa rotates the code and tracks remaining attempts."""
        client.post("/api/v1/auth/login", json={"username": "technician", "password": "password123"})
        resend_res = client.post("/api/v1/auth/resend-2fa", json={"username": "technician"})
        assert resend_res.status_code == 200
        assert resend_res.json()["status"] == "resent"

        new_code = _2fa_codes["technician"]["code"]
        v_res = client.post("/api/v1/auth/verify-2fa", json={"username": "technician", "code": new_code})
        assert v_res.status_code == 200

    def test_admin_self_registration_blocked(self, client: TestClient):
        """6. Verify anonymous callers cannot self-register as 'admin' (HTTP 403)."""
        res = client.post(
            "/api/v1/auth/register",
            json={
                "username": "rogue_admin_user",
                "password": "Password123!",
                "full_name": "Rogue User",
                "role": "admin",
            },
        )
        assert res.status_code == 403

    def test_auth_me_profile_endpoint(self, client: TestClient, auth_tokens: dict):
        """7. Verify /api/v1/auth/me returns the authenticated user's profile and role."""
        token = auth_tokens["radiologist"]
        res = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
        assert res.status_code == 200
        assert res.json()["username"] == "radiologist"
        assert res.json()["role"] == "radiologist"

    def test_image_upload_and_fail_fast_integration(
        self, client: TestClient, auth_tokens: dict, sample_xray_image: bytes, sample_invalid_small_image: bytes
    ):
        """8. Test uploading valid image and verify Fail-Fast rejection on undersized image."""
        token = auth_tokens.get("technician")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        valid_file = {"file": ("spine_xray.jpg", sample_xray_image, "image/jpeg")}
        res_valid = client.post("/api/v1/images/upload", files=valid_file, headers=headers)
        assert res_valid.status_code == 200
        assert "image_id" in res_valid.json()

        small_file = {"file": ("tiny.jpg", sample_invalid_small_image, "image/jpeg")}
        res_small = client.post("/api/v1/images/upload", files=small_file, headers=headers)
        assert res_small.status_code == 422
        assert "small" in res_small.json()["detail"].lower()

    def test_oversized_upload_returns_413(self, client: TestClient, auth_tokens: dict):
        """9. Verify uploading a file larger than 15 MB returns HTTP 413 Payload Too Large."""
        token = auth_tokens["technician"]
        huge_payload = b"X" * (15 * 1024 * 1024 + 512)
        res = client.post(
            "/api/v1/images/upload",
            files={"file": ("oversized_study.jpg", huge_payload, "image/jpeg")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 413

    def test_unsupported_extension_upload_returns_422(self, client: TestClient, auth_tokens: dict):
        """10. Verify uploading an unsupported file format (.exe/.pdf) returns HTTP 422."""
        token = auth_tokens["technician"]
        res = client.post(
            "/api/v1/images/upload",
            files={"file": ("malware.exe", b"MZ9000", "application/octet-stream")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 422

    def test_color_photo_upload_returns_422(self, client: TestClient, auth_tokens: dict):
        """11. Verify uploading a color photograph instead of a grayscale X-ray returns HTTP 422."""
        token = auth_tokens["technician"]
        arr = np.zeros((100, 100, 3), dtype=np.uint8)
        arr[:, :, 0] = 250
        arr[:, :, 2] = 10
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="JPEG")

        res = client.post(
            "/api/v1/images/upload",
            files={"file": ("vacation_photo.jpg", buf.getvalue(), "image/jpeg")},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert res.status_code == 422

    def test_complete_image_processing_and_inference_cycle(
        self, client: TestClient, auth_tokens: dict, sample_xray_image: bytes
    ):
        """12. Test complete cycle: Upload -> Preprocess -> Classify -> Review -> PDF Report."""
        token = auth_tokens.get("radiologist") or auth_tokens.get("admin")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        upload_res = client.post(
            "/api/v1/images/upload",
            files={"file": ("clinical_xray.jpg", sample_xray_image, "image/jpeg")},
            headers=headers,
        )
        assert upload_res.status_code == 200
        image_id = upload_res.json()["image_id"]

        class_res = client.post(f"/api/v1/classify/{image_id}", headers=headers)
        assert class_res.status_code == 200
        class_data = class_res.json()
        assert "classification" in class_data
        assert "confidence_score" in class_data
        assert "requires_review" in class_data
        assert "prediction_id" in class_data
        result_id = class_data["prediction_id"]

        review_res = client.post(
            f"/api/v1/results/{result_id}/review",
            json={"status": "confirmed", "notes": "Integrity validated by test suite."},
            headers=headers,
        )
        assert review_res.status_code == 200
        assert review_res.json()["status"] == "confirmed"

        rpt_res = client.post(
            f"/api/v1/results/{result_id}/report",
            json={"language": "es", "reviewer_name": "Dr. Test Integration"},
            headers=headers,
        )
        assert rpt_res.status_code == 200
        rpt_data = rpt_res.json()
        assert "download_url" in rpt_data
        download_url = rpt_data["download_url"]

        down_res = client.get(download_url, headers=headers)
        assert down_res.status_code == 200
        assert down_res.headers["content-type"] == "application/pdf"

    def test_technician_blocked_from_confirming_diagnosis_via_api_question_5(
        self, client: TestClient, auth_tokens: dict, sample_xray_image: bytes
    ):
        """
        13. Question 5 integration verification:
        Even if an imaging technician has logged in with valid credentials and 2FA
        (holding a valid signed JWT) and bypasses the web UI to call
        POST /api/v1/results/{id}/review directly via the API, the backend RBAC guard
        MUST reject the request with HTTP 403 Forbidden.
        """
        tech_headers = {"Authorization": f"Bearer {auth_tokens['technician']}"}

        # Technician CAN upload and trigger classification
        up_res = client.post(
            "/api/v1/images/upload",
            files={"file": ("tech_xray.jpg", sample_xray_image, "image/jpeg")},
            headers=tech_headers,
        )
        assert up_res.status_code == 200
        image_id = up_res.json()["image_id"]

        cls_res = client.post(f"/api/v1/classify/{image_id}", headers=tech_headers)
        assert cls_res.status_code == 200
        result_id = cls_res.json()["prediction_id"]

        # Technician CANNOT confirm or correct a diagnosis via the API (must return 403 Forbidden)
        forbidden_review = client.post(
            f"/api/v1/results/{result_id}/review",
            json={"status": "confirmed", "notes": "Technician trying to bypass UI"},
            headers=tech_headers,
        )
        assert forbidden_review.status_code == 403
        assert "technician" in forbidden_review.json()["detail"].lower()

    def test_technician_blocked_from_results_history_and_pdf_report_question_5(
        self, client: TestClient, auth_tokens: dict
    ):
        """14. Verify Technician also receives 403 Forbidden on GET /results and POST /results/{id}/report."""
        tech_headers = {"Authorization": f"Bearer {auth_tokens['technician']}"}

        res_list = client.get("/api/v1/results", headers=tech_headers)
        assert res_list.status_code == 403

        res_rpt = client.post(
            "/api/v1/results/1/report",
            json={"language": "es"},
            headers=tech_headers,
        )
        assert res_rpt.status_code == 403

    def test_dicom_upload_and_classification_end_to_end(self, client: TestClient, auth_tokens: dict):
        """15. Verify a 16-bit DICOM (.dcm) radiograph can be both uploaded, anonymized, and classified."""
        import pydicom
        from pydicom.dataset import FileDataset, FileMetaDataset
        from pydicom.uid import ExplicitVRLittleEndian, generate_uid

        meta = FileMetaDataset()
        meta.MediaStorageSOPClassUID = generate_uid()
        meta.MediaStorageSOPInstanceUID = generate_uid()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian

        ds = FileDataset("spine.dcm", {}, file_meta=meta, preamble=b"\0" * 128)
        ds.is_little_endian = True
        ds.is_implicit_VR = False
        ds.PatientName = "Maria Lopez"
        ds.PatientID = "PAC-102030"
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.Rows = 128
        ds.Columns = 128
        ds.SamplesPerPixel = 1
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0

        px = np.full((128, 128), 300, dtype=np.uint16)
        px[:, 50:78] = 2800
        ds.PixelData = px.tobytes()

        buf = io.BytesIO()
        pydicom.dcmwrite(buf, ds)

        rad_headers = {"Authorization": f"Bearer {auth_tokens['radiologist']}"}
        up_res = client.post(
            "/api/v1/images/upload",
            files={"file": ("patient_spine.dcm", buf.getvalue(), "application/dicom")},
            headers=rad_headers,
        )
        assert up_res.status_code == 200
        img_id = up_res.json()["image_id"]

        # Verify PHI was masked before persistence
        img_meta = client.get(f"/api/v1/images/{img_id}", headers=rad_headers).json()
        assert img_meta["is_anonymized"] is True
        assert img_meta["metadata"]["dicom_metadata"]["PatientName"] == "[ANONYMIZED]"

        # Verify DICOM can be classified
        cls_res = client.post(f"/api/v1/classify/{img_id}", headers=rad_headers)
        assert cls_res.status_code == 200
        assert "classification" in cls_res.json()

    def test_report_download_path_traversal_blocked(self, client: TestClient, auth_tokens: dict):
        """16. Verify GET /api/v1/reports/{filename} blocks path traversal attempts."""
        rad_headers = {"Authorization": f"Bearer {auth_tokens['radiologist']}"}
        res = client.get("/api/v1/reports/..%5Cmain.py", headers=rad_headers)
        assert res.status_code in (400, 404)

    def test_audit_trail_rbac_blocked_for_technician_and_radiologist(
        self, client: TestClient, auth_tokens: dict
    ):
        """17. Verify GET /api/v1/audit is restricted to Admin and ML Engineer (403 for Technician/Radiologist)."""
        tech_res = client.get("/api/v1/audit", headers={"Authorization": f"Bearer {auth_tokens['technician']}"})
        rad_res = client.get("/api/v1/audit", headers={"Authorization": f"Bearer {auth_tokens['radiologist']}"})
        assert tech_res.status_code == 403
        assert rad_res.status_code == 403

    def test_audit_trail_endpoint_with_mexico_timezone(self, client: TestClient, auth_tokens: dict):
        """18. Test GET /api/v1/audit returns audit records with Mexico timezone conversion for Admin."""
        token = auth_tokens.get("admin")
        headers = {"Authorization": f"Bearer {token}"} if token else {}

        res = client.get("/api/v1/audit?limit=20", headers=headers)
        assert res.status_code == 200
        data = res.json()
        assert "entries" in data
        assert isinstance(data["entries"], list)

        if len(data["entries"]) > 0:
            first_entry = data["entries"][0]
            assert "action" in first_entry
            assert "mexico_time" in first_entry
            assert "m." in first_entry["mexico_time"]

    def test_list_and_get_images_endpoints(self, client: TestClient, auth_tokens: dict):
        """19. Verify GET /api/v1/images and GET /api/v1/model/info endpoints."""
        headers = {"Authorization": f"Bearer {auth_tokens['radiologist']}"}
        imgs_res = client.get("/api/v1/images?limit=10", headers=headers)
        assert imgs_res.status_code == 200
        assert "images" in imgs_res.json()

        info_res = client.get("/api/v1/model/info", headers=headers)
        assert info_res.status_code == 200
        assert "is_loaded" in info_res.json()

    def test_password_recovery_flow_via_api(self, client: TestClient):
        """20. Verify /auth/forgot-password and /auth/reset-password flow."""
        from api.routes import _recovery_codes

        f_res = client.post("/api/v1/auth/forgot-password", json={"username": "technician"})
        assert f_res.status_code == 200
        code = _recovery_codes["technician"]["code"]

        r_res = client.post(
            "/api/v1/auth/reset-password",
            json={"username": "technician", "code": code, "new_password": "password123"},
        )
        assert r_res.status_code == 200
        assert r_res.json()["status"] == "success"

    def test_zero_byte_empty_upload_rejected_cleanly(self, client: TestClient, auth_tokens: dict):
        """21. Verify a 0-byte empty file upload is rejected cleanly with HTTP 422."""
        headers = {"Authorization": f"Bearer {auth_tokens['technician']}"}
        res = client.post(
            "/api/v1/images/upload",
            files={"file": ("empty_study.jpg", b"", "image/jpeg")},
            headers=headers,
        )
        assert res.status_code == 422

    def test_invalid_query_filters_and_pagination_bounds(self, client: TestClient, auth_tokens: dict):
        """22. Verify malformed query filters return 400 and negative pagination bounds do not crash."""
        rad_headers = {"Authorization": f"Bearer {auth_tokens['radiologist']}"}
        admin_headers = {"Authorization": f"Bearer {auth_tokens['admin']}"}

        bad_status = client.get("/api/v1/results?status=nonexistent_status", headers=rad_headers)
        assert bad_status.status_code == 400

        bad_action = client.get("/api/v1/audit?action=nonexistent_action", headers=admin_headers)
        assert bad_action.status_code == 400

        neg_bounds = client.get("/api/v1/images?limit=-25&offset=-10", headers=rad_headers)
        assert neg_bounds.status_code == 200

    def test_invalid_review_status_rejected_with_400(
        self, client: TestClient, auth_tokens: dict, sample_xray_image: bytes
    ):
        """23. Verify submitting 'pending_review' or an unknown status to /review returns HTTP 400."""
        rad_headers = {"Authorization": f"Bearer {auth_tokens['radiologist']}"}
        up = client.post(
            "/api/v1/images/upload",
            files={"file": ("rev_test.jpg", sample_xray_image, "image/jpeg")},
            headers=rad_headers,
        )
        image_id = up.json()["image_id"]
        cls = client.post(f"/api/v1/classify/{image_id}", headers=rad_headers)
        pred_id = cls.json()["prediction_id"]

        bad_rev = client.post(
            f"/api/v1/results/{pred_id}/review",
            json={"status": "pending_review", "notes": "Should not allow resetting to pending_review"},
            headers=rad_headers,
        )
        assert bad_rev.status_code == 400

    def test_pdf_report_resilient_to_unicode_and_special_chars(
        self, client: TestClient, auth_tokens: dict, sample_xray_image: bytes
    ):
        """24. Verify PDF generation never crashes when clinical notes or reviewer name contain Unicode/emojis."""
        rad_headers = {"Authorization": f"Bearer {auth_tokens['radiologist']}"}
        up = client.post(
            "/api/v1/images/upload",
            files={"file": ("unicode_xray.jpg", sample_xray_image, "image/jpeg")},
            headers=rad_headers,
        )
        image_id = up.json()["image_id"]
        cls = client.post(f"/api/v1/classify/{image_id}", headers=rad_headers)
        pred_id = cls.json()["prediction_id"]

        rev = client.post(
            f"/api/v1/results/{pred_id}/review",
            json={
                "status": "corrected",
                "notes": "Curva T7—T11 “compensatoria” • Ápice verificado ✓ ⚠️ 18.5° 😊",
            },
            headers=rad_headers,
        )
        assert rev.status_code == 200

        rpt = client.post(
            f"/api/v1/results/{pred_id}/report",
            json={"language": "es", "reviewer_name": "Dra. María José — Especialista ✓"},
            headers=rad_headers,
        )
        assert rpt.status_code == 200
        assert "download_url" in rpt.json()

    def test_compensating_storage_cleanup_on_db_write_failure_question_4(
        self, sample_xray_image: bytes
    ):
        """25. Question 4 verification: if DB write fails mid-upload, the stored file on disk is cleaned up."""
        from services.image_management_service import ImageManagementService
        from infrastructure.storage import LocalStorageAdapter
        from infrastructure.repositories import AuditRepository
        from utils.exceptions import RepositoryError

        class FailingImageRepo:
            def save(self, image):
                raise RepositoryError("save_image", "Simulated mid-transaction DB failure")

        storage = LocalStorageAdapter()
        deleted_paths = []
        orig_delete = storage.delete

        def spy_delete(path: str):
            deleted_paths.append(path)
            return orig_delete(path)

        storage.delete = spy_delete
        svc = ImageManagementService(
            image_repository=FailingImageRepo(),
            storage_adapter=storage,
            audit_repository=AuditRepository(),
        )
        with pytest.raises(RepositoryError):
            svc.upload_image(sample_xray_image, "rollback_test.jpg", uploaded_by=1)

        assert len(deleted_paths) == 1

    def test_keyboard_mash_flood_and_long_unbroken_inputs(
        self, client: TestClient, auth_tokens: dict, sample_xray_image: bytes
    ):
        """26. Verify system survives 500+ char keyboard-mashed filenames, notes, and login inputs."""
        rad_headers = {"Authorization": f"Bearer {auth_tokens['radiologist']}"}

        # 1. Keyboard-mashed login with null bytes and 1,000 chars -> clean 401, no 500 crash
        mashed_login = client.post(
            "/api/v1/auth/login",
            json={
                "username": ("asdfghjklqwertyuiop\x00" * 50),
                "password": ("zxcvbnmqwerty12345!\x00" * 50),
            },
        )
        assert mashed_login.status_code == 401

        # 2. Keyboard-mashed 400-character filename upload -> truncated safely below Windows MAX_PATH
        mashed_filename = ("qwertyuiopasdfghjklzxcvbnm1234567890" * 12) + ".jpg"
        up_res = client.post(
            "/api/v1/images/upload",
            files={"file": (mashed_filename, sample_xray_image, "image/jpeg")},
            headers=rad_headers,
        )
        assert up_res.status_code == 200
        assert len(up_res.json()["filename"]) <= 74
        image_id = up_res.json()["image_id"]

        # 3. Classify and submit 1,200-char unbroken keyboard-mashed clinical notes + PDF generation
        cls_res = client.post(f"/api/v1/classify/{image_id}", headers=rad_headers)
        assert cls_res.status_code == 200
        pred_id = cls_res.json()["prediction_id"]

        unbroken_mash = "asdfghjklqwertyuiopzxcvbnm1234567890" * 35
        rev_res = client.post(
            f"/api/v1/results/{pred_id}/review",
            json={"status": "confirmed", "notes": unbroken_mash},
            headers=rad_headers,
        )
        assert rev_res.status_code == 200

        pdf_res = client.post(
            f"/api/v1/results/{pred_id}/report",
            json={"language": "es", "reviewer_name": unbroken_mash},
            headers=rad_headers,
        )
        assert pdf_res.status_code == 200

    def test_concurrent_inference_thread_safety(self, app, sample_xray_image: bytes):
        """27. Verify concurrent multi-threaded inference executes safely without race conditions."""
        from concurrent.futures import ThreadPoolExecutor

        prep_svc = app.state.preprocessing_service
        inf_svc = app.state.inference_service
        img = Image.open(io.BytesIO(sample_xray_image)).convert("RGB")
        img_arr = np.asarray(img, dtype=np.uint8)
        pre = prep_svc.preprocess(img_arr, image_id=1, filename="concurrent_test.jpg", user_id=1)
        tensor = pre["preprocessed_image"]

        def _worker(idx: int):
            res = inf_svc.classify(
                preprocessed_image=tensor,
                image_id=idx + 1,
                user_id=1,
                filename=f"concurrent_{idx}.jpg",
                original_image=img_arr,
            )
            return res["classification"], sum(res["all_probabilities"].values())

        with ThreadPoolExecutor(max_workers=4) as pool:
            outputs = list(pool.map(_worker, range(8)))

        assert len(outputs) == 8
        first_label = outputs[0][0]
        for label, prob_sum in outputs:
            assert label == first_label
            assert abs(prob_sum - 1.0) < 1e-2
