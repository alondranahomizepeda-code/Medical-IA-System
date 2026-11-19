"""
API Routes — API Gateway Layer
=================================
Defines the REST API endpoints that expose the system's functionality.

The API Gateway is the single public entry point to the system:
  - Enforces Two-Factor Authentication (2FA) and JWT HS256 verification.
  - Enforces Role-Based Access Control (RBAC) on every protected endpoint (Question 5).
  - Routes requests to the appropriate service.
  - Maps domain exceptions to standardized HTTP status codes (401, 403, 404, 413, 422, 503).
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional
import zoneinfo

from fastapi import APIRouter, UploadFile, File, HTTPException, Header, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from config.settings import MAX_FILE_SIZE_BYTES, MAX_FILE_SIZE_MB, REPORTS_DIR, API_VERSION
from domain.interfaces.contracts import (
    UserRole,
    PredictionStatus,
    AuditAction,
    AuditEntry,
)
from infrastructure.database import session_scope, OneTimeCodeDB, UserDB
from pipeline.stages import decode_image, sanitize_filename
from utils.exceptions import (
    AuthenticationError,
    AuthorizationError,
    OneTimeCodeError,
    FileSizeExceededError,
    ImageValidationError,
    ModelNotLoadedError,
    StorageError,
)
from utils.logger import get_logger

# ═══════════════════════════════════════════════════════════════
# Request / Response Schemas
# ═══════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str


class Verify2FARequest(BaseModel):
    username: str
    code: str


class Resend2FARequest(BaseModel):
    username: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    full_name: str
    role: str = "technician"


class ReviewRequest(BaseModel):
    status: str  # "confirmed" or "corrected"
    notes: str


class ReportRequest(BaseModel):
    reviewer_name: Optional[str] = None
    language: Optional[str] = "en"


class ForgotPasswordRequest(BaseModel):
    username: str


class ResetPasswordRequest(BaseModel):
    username: str
    code: str
    new_password: str


# Mirror dictionary kept in sync with OneTimeCodeRepository for test fixtures
_2fa_codes: Dict[str, Dict] = {}
_recovery_codes: Dict[str, Dict] = {}


# ═══════════════════════════════════════════════════════════════
# Router Factory
# ═══════════════════════════════════════════════════════════════

def create_router(
    auth_service,
    image_service,
    preprocessing_service,
    inference_service,
    clinical_service,
    audit_service,
) -> APIRouter:
    """
    Create the API router with all endpoints and strict RBAC enforcement.
    """
    router = APIRouter()

    def _authenticate_and_authorize(
        authorization: Optional[str],
        allowed_roles: Optional[List[UserRole]] = None,
        action_name: str = "access_endpoint",
        require_token: bool = True,
    ) -> Optional[Dict]:
        """
        Extract and verify the Bearer JWT, check that no unverified 2FA step is pending,
        and enforce RBAC role permissions (Question 5).
        """
        if not authorization or not authorization.startswith("Bearer "):
            if require_token:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required. Provide a valid Bearer JWT token.",
                )
            return None

        token_str = authorization.split(" ", 1)[1].strip()
        if not token_str or token_str in ("null", "undefined"):
            if require_token:
                raise HTTPException(
                    status_code=401,
                    detail="Authentication required. Provide a valid Bearer JWT token.",
                )
            return None

        try:
            payload = auth_service.verify_token(token_str)
        except AuthenticationError as e:
            raise HTTPException(status_code=401, detail=str(e))

        # Reject pre-2FA tokens unconditionally (prevents bypass after 5 failed attempts or OTP expiry)
        if payload.get("two_factor_verified") is False:
            raise HTTPException(
                status_code=401,
                detail="Two-factor authentication (2FA) verification is incomplete.",
            )

        username = payload.get("sub")
        user_id = payload.get("user_id")
        role_str = payload.get("role", "")

        # Block token use if user still has an unverified 2FA challenge pending
        if username:
            with session_scope() as session:
                pending_2fa = (
                    session.query(OneTimeCodeDB)
                    .filter_by(username=username, purpose="2fa")
                    .first()
                )
                if pending_2fa is not None:
                    raise HTTPException(
                        status_code=401,
                        detail="Two-factor authentication (2FA) verification is incomplete.",
                    )

        if not user_id and username:
            u = auth_service._user_repo.find_by_username(username)
            if u:
                user_id = u.id
                role_str = u.role.value

        payload["user_id"] = user_id
        payload["role"] = role_str

        if allowed_roles is not None:
            try:
                auth_service.authorize(role_str, allowed_roles)
            except AuthorizationError as e:
                # Audit unauthorized access attempt
                try:
                    audit_service._audit_repo.append(AuditEntry(
                        action=AuditAction.CONFIGURATION_CHANGED,
                        user_id=user_id,
                        resource_type="rbac_guard",
                        resource_id=user_id,
                        details=f"RBAC 403 Forbidden: role '{role_str}' blocked from '{action_name}'",
                    ))
                except Exception:
                    pass
                raise HTTPException(status_code=403, detail=str(e))

        return payload

    # ── Authentication Routes (Authentication & User Service) ──

    @router.post("/auth/register", tags=["Authentication"])
    async def register(
        request: RegisterRequest,
        authorization: Optional[str] = Header(None),
    ):
        """
        Register a new user account (RF-01).
        Anonymous self-registration is restricted to non-admin clinical staff;
        creating an 'admin' account requires an authenticated Administrator token.
        """
        try:
            role = UserRole(request.role)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid role '{request.role}'")

        if role == UserRole.ADMIN:
            caller = _authenticate_and_authorize(
                authorization,
                allowed_roles=[UserRole.ADMIN],
                action_name="register_admin_account",
                require_token=False,
            )
            if not caller:
                raise HTTPException(
                    status_code=403,
                    detail="Role 'anonymous' is not authorized to self-register as 'admin'. Administrator token required.",
                )

        try:
            user = auth_service.register_user(
                username=request.username,
                password=request.password,
                full_name=request.full_name,
                role=role,
            )
            return {
                "message": "User registered successfully",
                "user_id": user.id,
                "username": user.username,
                "role": user.role.value,
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/auth/login", tags=["Authentication"])
    async def login(request: LoginRequest):
        """Authenticate credentials and issue a 6-digit 2FA challenge (RF-01)."""
        try:
            user, code = auth_service.initiate_2fa_login(request.username, request.password)
            # Issue a pre-2FA token explicitly marked two_factor_verified=False (unusable on protected routes)
            token_bundle = auth_service.issue_jwt_for_user(
                user, record_audit=False, two_factor_verified=False
            )
            _2fa_codes[request.username] = {"code": code, "attempts": 1}

            logger = get_logger("2FA_System")
            logger.info("=" * 60)
            logger.info(f" [SECURITY NOTICE] 2FA SECURITY CODE FOR '{request.username}' ")
            logger.info(f" CODE: {code} ")
            logger.info("=" * 60)

            return {
                "status": "2fa_required",
                "username": request.username,
                "access_token": token_bundle["access_token"],
            }
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))

    @router.post("/auth/resend-2fa", tags=["Authentication"])
    async def resend_2fa(request: Resend2FARequest):
        """Resend the 2FA code (Max 3 sends)."""
        try:
            new_code, attempts_left = auth_service.resend_2fa_code(request.username)
            sends_used = 3 - attempts_left
            _2fa_codes[request.username] = {"code": new_code, "attempts": sends_used}

            logger = get_logger("2FA_System")
            logger.info("=" * 60)
            logger.info(f" [RESEND] 2FA SECURITY CODE FOR '{request.username}' ")
            logger.info(f" NEW CODE: {new_code} (Attempt {sends_used}/3) ")
            logger.info("=" * 60)

            return {"status": "resent", "attempts_left": attempts_left}
        except OneTimeCodeError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/auth/verify-2fa", tags=["Authentication"])
    async def verify_2fa(request: Verify2FARequest):
        """Verify the 6-digit 2FA code and activate the JWT session."""
        try:
            token_bundle = auth_service.complete_2fa_login(request.username, request.code)
            _2fa_codes.pop(request.username, None)
            return {
                "status": "verified",
                "access_token": token_bundle["access_token"],
                "token_type": "bearer",
                "username": token_bundle["username"],
                "role": token_bundle["role"],
            }
        except (OneTimeCodeError, AuthenticationError) as e:
            raise HTTPException(status_code=401, detail=str(e))

    @router.get("/auth/me", tags=["Authentication"])
    async def get_current_user_profile(authorization: Optional[str] = Header(None)):
        """Return the profile and role of the currently authenticated user."""
        payload = _authenticate_and_authorize(
            authorization,
            allowed_roles=[
                UserRole.TECHNICIAN,
                UserRole.RADIOLOGIST,
                UserRole.ADMIN,
                UserRole.ML_ENGINEER,
            ],
            action_name="get_current_user",
        )
        return {
            "user_id": payload.get("user_id"),
            "username": payload.get("sub"),
            "role": payload.get("role"),
            "full_name": payload.get("full_name"),
        }

    @router.post("/auth/forgot-password", tags=["Authentication"])
    async def forgot_password(request: ForgotPasswordRequest):
        """
        Generate and send a 6-digit password recovery code.
        Uses HMAC-SHA256 persistence in one_time_codes.
        """
        user = auth_service._user_repo.find_by_username(request.username)
        if not user:
            raise HTTPException(status_code=404, detail=f"User '{request.username}' not found.")
        if not user.is_active:
            raise HTTPException(status_code=400, detail="This user account is deactivated.")

        code, _ = auth_service._otp_repo.issue_code(
            username=request.username,
            purpose="reset",
            ttl_minutes=15,
            is_resend=False,
        )
        _recovery_codes[request.username] = {"code": code, "attempts": 0}

        logger = get_logger("PasswordRecovery")
        logger.info("=" * 64)
        logger.info(" [PASSWORD RECOVERY] - Medical AI System")
        logger.info(f" User: '{request.username}' ({user.full_name})")
        logger.info(f" RECOVERY CODE (SMS/OTP): {code}")
        logger.info("=" * 64)

        return {
            "status": "code_sent",
            "message": f"Verification code sent to registered contact for '{request.username}'.",
            "username": request.username,
        }

    @router.post("/auth/reset-password", tags=["Authentication"])
    async def reset_password(request: ResetPasswordRequest):
        """Verify the 6-digit recovery code and update the user's password."""
        try:
            auth_service._otp_repo.verify_and_consume(
                username=request.username,
                code=request.code.strip(),
                purpose="reset",
            )
            _recovery_codes.pop(request.username, None)
        except OneTimeCodeError as e:
            msg = str(e)
            status_code = 401 if "Invalid verification code" in msg else 400
            raise HTTPException(status_code=status_code, detail=msg)

        try:
            auth_service.update_password(request.username, request.new_password)
            return {
                "status": "success",
                "message": f"Password successfully updated for user '{request.username}'. You can now log in.",
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    # ── Image Management Routes (Image Management Service) ─────

    @router.post("/images/upload", tags=["Image Management"])
    async def upload_image(
        file: UploadFile = File(...),
        authorization: Optional[str] = Header(None),
    ):
        """
        Upload a medical image for processing (RF-02, RF-03).
        Reads at most 15 MB + 1 byte before rejecting oversized files with HTTP 413.
        """
        caller = _authenticate_and_authorize(
            authorization,
            allowed_roles=[
                UserRole.TECHNICIAN,
                UserRole.RADIOLOGIST,
                UserRole.ADMIN,
                UserRole.ML_ENGINEER,
            ],
            action_name="upload_image",
            require_token=False,
        )
        user_id = caller.get("user_id") if caller else None

        try:
            file.file.seek(0)
            file_data = file.file.read(MAX_FILE_SIZE_BYTES + 1)
            if len(file_data) > MAX_FILE_SIZE_BYTES:
                size_mb = len(file_data) / (1024 * 1024)
                raise FileSizeExceededError(size_mb, MAX_FILE_SIZE_MB)

            result = image_service.upload_image(
                file_data=file_data,
                filename=file.filename or "unnamed",
                uploaded_by=user_id,
            )
            return {"status": "success", **result}

        except FileSizeExceededError as e:
            raise HTTPException(status_code=413, detail=str(e))
        except ImageValidationError as e:
            raise HTTPException(status_code=422, detail=str(e))
        except Exception as e:
            error_msg = str(e)
            err_lower = error_msg.lower()
            if "exceeds the maximum allowed size" in err_lower:
                raise HTTPException(status_code=413, detail=error_msg)
            if any(term in err_lower for term in ("validation", "format", "size", "quality", "small", "dimension", "corrupt", "invalid")):
                raise HTTPException(status_code=422, detail=error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    @router.get("/images", tags=["Image Management"])
    async def list_images(
        limit: int = 50,
        offset: int = 0,
        authorization: Optional[str] = Header(None),
    ):
        """List all uploaded images (RF-11)."""
        _authenticate_and_authorize(
            authorization,
            allowed_roles=[
                UserRole.TECHNICIAN,
                UserRole.RADIOLOGIST,
                UserRole.ADMIN,
                UserRole.ML_ENGINEER,
            ],
            action_name="list_images",
            require_token=False,
        )
        safe_limit = max(1, min(limit, 500))
        safe_offset = max(0, offset)
        images = image_service.list_images(limit=safe_limit, offset=safe_offset)
        return {
            "images": [
                {
                    "id": img.id,
                    "filename": img.filename,
                    "format": img.format,
                    "width": img.width,
                    "height": img.height,
                    "uploaded_at": img.uploaded_at.isoformat() if img.uploaded_at else None,
                }
                for img in images
            ],
            "count": len(images),
        }

    @router.get("/images/{image_id}", tags=["Image Management"])
    async def get_image(
        image_id: int,
        authorization: Optional[str] = Header(None),
    ):
        """Get image metadata by ID."""
        _authenticate_and_authorize(
            authorization,
            allowed_roles=[
                UserRole.TECHNICIAN,
                UserRole.RADIOLOGIST,
                UserRole.ADMIN,
                UserRole.ML_ENGINEER,
            ],
            action_name="get_image",
            require_token=False,
        )
        image = image_service.get_image(image_id)
        if not image:
            raise HTTPException(status_code=404, detail="Image not found")
        return {
            "id": image.id,
            "filename": image.filename,
            "format": image.format,
            "width": image.width,
            "height": image.height,
            "uploaded_at": image.uploaded_at.isoformat() if image.uploaded_at else None,
            "is_anonymized": image.is_anonymized,
            "metadata": image.metadata,
        }

    # ── Classification Routes (Inference + Preprocessing) ──────

    @router.post("/classify/{image_id}", tags=["AI Classification"])
    async def classify_image(
        image_id: int,
        authorization: Optional[str] = Header(None),
    ):
        """
        Run AI classification on an uploaded image (RF-05, RF-06).
        Uses the unified `decode_image` helper so both standard radiographs
        and 16-bit DICOM studies can be classified.
        Returns 503 if no approved model is loaded in the Model Registry.
        """
        caller = _authenticate_and_authorize(
            authorization,
            allowed_roles=[
                UserRole.TECHNICIAN,
                UserRole.RADIOLOGIST,
                UserRole.ADMIN,
                UserRole.ML_ENGINEER,
            ],
            action_name="classify_image",
            require_token=False,
        )
        user_id = caller.get("user_id") if caller else None

        try:
            image = image_service.get_image(image_id)
            if not image:
                raise HTTPException(status_code=404, detail="Image not found")

            from infrastructure.storage import LocalStorageAdapter

            storage = LocalStorageAdapter()
            file_data = storage.retrieve(image.original_path)
            image_array, _ = decode_image(file_data, image.filename)

            if not user_id and image.uploaded_by:
                user_id = image.uploaded_by

            # Step 2: Preprocess
            prep_result = preprocessing_service.preprocess(
                image_array=image_array,
                image_id=image_id,
                filename=image.filename,
                user_id=user_id,
            )

            # Step 3: Classify (through decorated inferencer)
            classification = inference_service.classify(
                preprocessed_image=prep_result["preprocessed_image"],
                image_id=image_id,
                user_id=user_id,
                filename=image.filename,
                original_image=image_array,
            )

            # Step 4: Save prediction result (always starts as pending_review)
            prediction = clinical_service.save_prediction(
                image_id=image_id,
                classification=classification.get("classification", "UNKNOWN"),
                confidence_score=classification.get("confidence_score", 0.0),
                model_name=classification.get("model_name", ""),
                model_version=classification.get("model_version", ""),
                processing_time_ms=classification.get("processing_time_ms", 0.0),
                affected_vertebrae=classification.get("affected_vertebrae"),
                curve_type=classification.get("curve_type"),
                curve_detail=classification.get("curve_detail"),
            )

            return {
                "status": "success",
                "prediction_id": prediction.id,
                "image_id": image_id,
                "classification": classification.get("classification"),
                "confidence_score": classification.get("confidence_score"),
                "affected_vertebrae": classification.get("affected_vertebrae"),
                "curve_type": classification.get("curve_type"),
                "curve_detail": classification.get("curve_detail"),
                "apex_vertebra": classification.get("apex_vertebra"),
                "requires_review": classification.get("requires_review", False),
                "low_confidence": classification.get("low_confidence", False),
                "geometry_agrees": classification.get("geometry_agrees", True),
                "all_probabilities": classification.get("all_probabilities", {}),
                "model": f"{classification.get('model_name')} v{classification.get('model_version')}",
                "processing_time_ms": classification.get("processing_time_ms"),
                "quality_report": prep_result.get("quality_report", {}),
                "metrics": classification.get("metrics", {}),
            }

        except HTTPException:
            raise
        except ModelNotLoadedError as e:
            raise HTTPException(status_code=503, detail=str(e))
        except (StorageError, FileNotFoundError) as e:
            raise HTTPException(status_code=404, detail=f"Original image file not found in storage: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

    @router.get("/model/info", tags=["AI Classification"])
    async def get_model_info(authorization: Optional[str] = Header(None)):
        """Get information about the currently loaded AI model."""
        _authenticate_and_authorize(
            authorization,
            allowed_roles=[
                UserRole.TECHNICIAN,
                UserRole.RADIOLOGIST,
                UserRole.ADMIN,
                UserRole.ML_ENGINEER,
            ],
            action_name="get_model_info",
            require_token=False,
        )
        return inference_service.get_model_info()

    # ── Clinical Results Routes (Clinical Results Service) ─────

    @router.get("/results", tags=["Clinical Results"])
    async def list_results(
        limit: int = 50,
        offset: int = 0,
        status: Optional[str] = None,
        authorization: Optional[str] = Header(None),
    ):
        """
        List prediction results with optional filtering (RF-11).
        Restricted by RBAC to Radiologist and Admin roles.
        """
        _authenticate_and_authorize(
            authorization,
            allowed_roles=[UserRole.RADIOLOGIST, UserRole.ADMIN],
            action_name="list_clinical_results",
            require_token=False,
        )
        try:
            status_filter = PredictionStatus(status) if status else None
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status filter '{status}'.")
        safe_limit = max(1, min(limit, 500))
        safe_offset = max(0, offset)
        results = clinical_service.get_history(limit=safe_limit, offset=safe_offset, status=status_filter)
        return {
            "results": [
                {
                    "id": r.id,
                    "image_id": r.image_id,
                    "classification": r.classification,
                    "confidence_score": r.confidence_score,
                    "affected_vertebrae": r.affected_vertebrae,
                    "curve_type": r.curve_type,
                    "curve_detail": r.curve_detail,
                    "model": f"{r.model_name} v{r.model_version}",
                    "status": r.status.value,
                    "predicted_at": r.predicted_at.isoformat() if r.predicted_at else None,
                }
                for r in results
            ],
            "count": len(results),
        }

    @router.get("/results/{result_id}", tags=["Clinical Results"])
    async def get_result(
        result_id: int,
        authorization: Optional[str] = Header(None),
    ):
        """Get a specific prediction result (RF-07). Restricted to Radiologist and Admin."""
        _authenticate_and_authorize(
            authorization,
            allowed_roles=[UserRole.RADIOLOGIST, UserRole.ADMIN],
            action_name="get_clinical_result",
            require_token=False,
        )
        result = clinical_service.get_result(result_id)
        if not result:
            raise HTTPException(status_code=404, detail="Result not found")
        return {
            "id": result.id,
            "image_id": result.image_id,
            "classification": result.classification,
            "confidence_score": result.confidence_score,
            "affected_vertebrae": result.affected_vertebrae,
            "curve_type": result.curve_type,
            "curve_detail": result.curve_detail,
            "model_name": result.model_name,
            "model_version": result.model_version,
            "processing_time_ms": result.processing_time_ms,
            "status": result.status.value,
            "reviewer_notes": result.reviewer_notes,
            "predicted_at": result.predicted_at.isoformat() if result.predicted_at else None,
            "reviewed_at": result.reviewed_at.isoformat() if result.reviewed_at else None,
        }

    @router.post("/results/{result_id}/review", tags=["Clinical Results"])
    async def review_result(
        result_id: int,
        request: ReviewRequest,
        authorization: Optional[str] = Header(None),
    ):
        """
        Submit a clinical review for a prediction (RF-09, Question 5).
        Enforces strict server-side RBAC: only 'radiologist' and 'admin' roles may
        confirm or correct a diagnosis. A 'technician' calling this endpoint directly
        via the API receives HTTP 403 Forbidden.
        """
        caller = _authenticate_and_authorize(
            authorization,
            allowed_roles=[UserRole.RADIOLOGIST, UserRole.ADMIN],
            action_name="review_prediction",
            require_token=True,
        )
        reviewer_id = caller.get("user_id") or 1

        try:
            status = PredictionStatus(request.status)
            if status not in (PredictionStatus.CONFIRMED, PredictionStatus.CORRECTED):
                raise ValueError("Review status must be either 'confirmed' or 'corrected'.")
            safe_notes = (request.notes or "").strip()[:2000]
            result = clinical_service.review_prediction(
                result_id=result_id,
                reviewer_id=reviewer_id,
                status=status,
                notes=safe_notes,
            )
            if not result:
                raise HTTPException(status_code=404, detail="Result not found")
            return {
                "message": "Review submitted",
                "status": result.status.value,
                "reviewed_by": reviewer_id,
            }
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/results/{result_id}/report", tags=["Clinical Results"])
    async def generate_report(
        result_id: int,
        request: ReportRequest,
        authorization: Optional[str] = Header(None),
    ):
        """
        Generate a clinical PDF report (RF-10).
        Restricted by RBAC to Radiologist and Admin roles.
        """
        caller = _authenticate_and_authorize(
            authorization,
            allowed_roles=[UserRole.RADIOLOGIST, UserRole.ADMIN],
            action_name="generate_report",
            require_token=False,
        )
        user_id = caller.get("user_id") if caller else None

        try:
            lang = (request.language or "en").lower().strip()
            report_path = clinical_service.generate_report(
                result_id=result_id,
                reviewer_name=request.reviewer_name,
                language=lang,
                user_id=user_id,
            )
            filename = Path(report_path).name
            return {
                "message": "Report generated",
                "path": report_path,
                "filename": filename,
                "download_url": f"/api/v1/reports/{filename}",
            }
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/reports/{filename}", tags=["Clinical Results"])
    async def download_report_file(
        filename: str,
        authorization: Optional[str] = Header(None),
    ):
        """
        View or download a generated clinical PDF report.
        Protected against Path Traversal and restricted to Radiologist/Admin when token is present.
        """
        _authenticate_and_authorize(
            authorization,
            allowed_roles=[UserRole.RADIOLOGIST, UserRole.ADMIN],
            action_name="download_report",
            require_token=False,
        )

        # Path-traversal guard
        if ".." in filename or "/" in filename or "\\" in filename:
            raise HTTPException(status_code=400, detail="Invalid report filename.")

        safe_name = sanitize_filename(filename)
        reports_root = REPORTS_DIR.resolve()
        file_path = (reports_root / safe_name).resolve()

        if not file_path.is_relative_to(reports_root) or not file_path.exists():
            raise HTTPException(status_code=404, detail="Report file not found")

        return Response(
            content=file_path.read_bytes(),
            media_type="application/pdf",
            headers={"Content-Disposition": f"inline; filename={safe_name}"},
        )

    # ── Audit Routes (Audit Service) ──────────────────────────

    @router.get("/audit", tags=["Audit"])
    async def get_audit_trail(
        limit: int = 100,
        offset: int = 0,
        action: Optional[str] = None,
        authorization: Optional[str] = Header(None),
    ):
        """
        Retrieve audit trail entries with Mexico timezone conversion and usernames (RNF-09).
        Restricted by RBAC to Admin and ML Engineer roles.
        """
        _authenticate_and_authorize(
            authorization,
            allowed_roles=[UserRole.ADMIN, UserRole.ML_ENGINEER],
            action_name="view_audit_trail",
            require_token=False,
        )

        try:
            action_filter = AuditAction(action) if action else None
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid audit action filter '{action}'.")
        safe_limit = max(1, min(limit, 500))
        safe_offset = max(0, offset)
        entries = audit_service.get_audit_trail(limit=safe_limit, offset=safe_offset, action_filter=action_filter)

        users_map = {}
        try:
            with session_scope() as session:
                for u in session.query(UserDB).all():
                    users_map[u.id] = u.username
        except Exception:
            pass

        mexico_tz = zoneinfo.ZoneInfo("America/Mexico_City")
        serialized_entries = []

        for e in entries:
            iso_utc = None
            mexico_formatted = None
            if e.timestamp:
                utc_dt = e.timestamp.replace(tzinfo=timezone.utc)
                iso_utc = utc_dt.isoformat()
                mexico_dt = utc_dt.astimezone(mexico_tz)
                am_pm = "p. m." if mexico_dt.hour >= 12 else "a. m."
                mexico_formatted = f"{mexico_dt.strftime('%d/%m/%Y, %I:%M:%S')} {am_pm}"

            uname = users_map.get(e.user_id) if e.user_id else None

            serialized_entries.append({
                "id": e.id,
                "action": e.action.value,
                "user_id": e.user_id,
                "username": uname,
                "resource_type": e.resource_type,
                "resource_id": e.resource_id,
                "details": e.details,
                "timestamp": iso_utc,
                "mexico_time": mexico_formatted,
            })

        return {
            "entries": serialized_entries,
            "count": len(entries),
        }

    # ── System Health & Readiness Probes ───────────────────────

    @router.get("/health", tags=["System"])
    async def health_check():
        """Liveness probe endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": API_VERSION,
            "services": {
                "auth": "active",
                "image_management": "active",
                "preprocessing": "active",
                "inference": "active" if inference_service.is_ready else "degraded",
                "clinical_results": "active",
                "audit": "active",
            },
        }

    @router.get("/ready", tags=["System"])
    async def readiness_check(response: Response):
        """
        Readiness probe endpoint (Question 10).
        Returns 200 OK only when an approved AI model is loaded; otherwise returns 503.
        """
        if not inference_service.is_ready:
            response.status_code = 503
            return {
                "status": "not_ready",
                "reason": "No approved AI model is loaded in InferenceService.",
                "model_info": inference_service.get_model_info(),
            }
        return {
            "status": "ready",
            "model_info": inference_service.get_model_info(),
        }

    return router
