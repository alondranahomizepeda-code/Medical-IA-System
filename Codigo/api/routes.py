"""
API Routes — API Gateway Layer
=================================
Defines the REST API endpoints that expose the system's functionality.

The API Gateway is the single public entry point to the system:
  - Validates authentication tokens.
  - Routes requests to the appropriate service.
  - Returns standardized JSON responses with error handling.

Each route group corresponds to a service in the architecture.
"""

import io
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Header
from pydantic import BaseModel

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


# ═══════════════════════════════════════════════════════════════
# Dependency: Get services from app state
# ═══════════════════════════════════════════════════════════════

def get_app_services():
    """Placeholder — services are injected at startup via app.state."""
    pass


# In-memory dictionary for 2FA security codes: { username: { "code": str, "attempts": int } }
_2fa_codes = {}


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
    Create the API router with all endpoints.
    
    Services are injected following DIP — the routes depend on
    service interfaces, not concrete implementations.
    """
    router = APIRouter()

    # ── Authentication Routes (Authentication & User Service) ──

    @router.post("/auth/register", tags=["Authentication"])
    async def register(request: RegisterRequest):
        """Register a new user account (RF-01)."""
        try:
            from domain.interfaces.contracts import UserRole
            role = UserRole(request.role)
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
        """Authenticate and trigger 2FA (RF-01)."""
        try:
            from utils.logger import get_logger
            import random
            
            # 1. Verify standard credentials
            result = auth_service.authenticate(request.username, request.password)
            
            # 2. Generate 2FA Code (Simulating SMS)
            code = str(random.randint(100000, 999999))
            _2fa_codes[request.username] = {"code": code, "attempts": 1}
            
            # 3. Print distinctly to the CMD
            logger = get_logger("2FA_System")
            logger.info(" ")
            logger.info("=" * 60)
            logger.info(f" [SECURITY NOTICE] 2FA SECURITY CODE FOR '{request.username}' ")
            logger.info(f" CODE: {code} ")
            logger.info("=" * 60)
            logger.info(" ")
            
            return {
                "status": "2fa_required", 
                "username": request.username, 
                "access_token": result["access_token"]
            }
        except Exception as e:
            raise HTTPException(status_code=401, detail=str(e))

    @router.post("/auth/resend-2fa", tags=["Authentication"])
    async def resend_2fa(request: Resend2FARequest):
        """Resend the 2FA code (Max 3 attempts)."""
        user_data = _2fa_codes.get(request.username)
        if not user_data:
            raise HTTPException(status_code=400, detail="No active 2FA session.")
        if user_data["attempts"] >= 3:
            raise HTTPException(status_code=400, detail="Maximum 2FA resend attempts reached.")
        
        import random
        from utils.logger import get_logger
        
        new_code = str(random.randint(100000, 999999))
        _2fa_codes[request.username] = {"code": new_code, "attempts": user_data["attempts"] + 1}
        
        logger = get_logger("2FA_System")
        logger.info(" ")
        logger.info("=" * 60)
        logger.info(f" [RESEND] 2FA SECURITY CODE FOR '{request.username}' ")
        logger.info(f" NEW CODE: {new_code} (Attempt {_2fa_codes[request.username]['attempts']}/3) ")
        logger.info("=" * 60)
        logger.info(" ")
        
        return {"status": "resent", "attempts_left": 3 - _2fa_codes[request.username]['attempts']}

    @router.post("/auth/verify-2fa", tags=["Authentication"])
    async def verify_2fa(request: Verify2FARequest):
        """Verify the 2FA code provided by the user."""
        if request.username in _2fa_codes and _2fa_codes[request.username]["code"] == request.code:
            del _2fa_codes[request.username]
            return {"status": "verified"}
        raise HTTPException(status_code=401, detail="Invalid 2FA code. Please check your CMD.")

    # In-memory dictionary for password recovery codes: { username: { "code": str, "expires_at": datetime, "attempts": int } }
    _recovery_codes = {}

    @router.post("/auth/forgot-password", tags=["Authentication"])
    async def forgot_password(request: ForgotPasswordRequest):
        """Generate and send a 6-digit password recovery code via simulated SMS / CMD."""
        user = auth_service._user_repo.find_by_username(request.username)
        if not user:
            raise HTTPException(status_code=404, detail=f"Usuario '{request.username}' no encontrado.")
        if not user.is_active:
            raise HTTPException(status_code=400, detail="Esta cuenta de usuario está desactivada.")

        import random
        from datetime import datetime, timedelta
        from utils.logger import get_logger

        code = str(random.randint(100000, 999999))
        expires_at = datetime.utcnow() + timedelta(minutes=15)
        _recovery_codes[request.username] = {
            "code": code,
            "expires_at": expires_at,
            "attempts": 0,
        }

        logger = get_logger("PasswordRecovery")
        logger.info(" ")
        logger.info("=" * 64)
        logger.info(" [RECUPERACION DE CONTRASENA] - Medical AI System")
        logger.info(f" Usuario: '{request.username}' ({user.full_name})")
        logger.info(f" CODIGO DE RECUPERACION (SMS/OTP): {code}")
        logger.info(f" Vigencia: 15 minutos (Expira: {expires_at.strftime('%H:%M:%S')} UTC)")
        logger.info("=" * 64)
        logger.info(" ")

        return {
            "status": "code_sent",
            "message": f"Verification code sent to registered contact for '{request.username}'.",
            "username": request.username,
        }

    @router.post("/auth/reset-password", tags=["Authentication"])
    async def reset_password(request: ResetPasswordRequest):
        """Verify the 6-digit recovery code and update the user's password."""
        recovery_data = _recovery_codes.get(request.username)
        if not recovery_data:
            raise HTTPException(
                status_code=400, 
                detail="No hay una solicitud de recuperación activa para este usuario o el código ha expirado."
            )

        from datetime import datetime
        if datetime.utcnow() > recovery_data["expires_at"]:
            del _recovery_codes[request.username]
            raise HTTPException(status_code=400, detail="El código de recuperación ha expirado. Solicita uno nuevo.")

        if recovery_data["code"] != request.code.strip():
            recovery_data["attempts"] += 1
            if recovery_data["attempts"] >= 3:
                del _recovery_codes[request.username]
                raise HTTPException(status_code=400, detail="Demasiados intentos fallidos. Solicita un nuevo código.")
            raise HTTPException(status_code=401, detail="Código de recuperación incorrecto. Verifica tu consola/SMS.")

        try:
            auth_service.update_password(request.username, request.new_password)
            del _recovery_codes[request.username]
            return {
                "status": "success",
                "message": f"Contraseña actualizada exitosamente para el usuario '{request.username}'. Ya puedes iniciar sesión.",
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
        
        The image goes through Fail Fast validation immediately.
        Supported formats: DICOM, JPEG, PNG, TIFF. Max size: 15 MB.
        """
        try:
            # Read file data
            file_data = await file.read()

            # Get user ID from token if available
            user_id = None
            if authorization and authorization.startswith("Bearer "):
                try:
                    token_str = authorization.split(" ")[1].strip()
                    if token_str and token_str not in ("null", "undefined"):
                        token_data = auth_service.verify_token(token_str)
                        user_id = token_data.get("user_id")
                        if not user_id and token_data.get("sub"):
                            u = auth_service._user_repo.find_by_username(token_data.get("sub"))
                            if u:
                                user_id = u.id
                except Exception:
                    pass

            result = image_service.upload_image(
                file_data=file_data,
                filename=file.filename or "unnamed",
                uploaded_by=user_id,
            )
            return {"status": "success", **result}

        except Exception as e:
            error_msg = str(e)
            # Fail Fast errors return 422 (Unprocessable Entity)
            err_lower = error_msg.lower()
            if any(term in err_lower for term in ("validation", "format", "size", "quality", "small", "dimension", "corrupt")):
                raise HTTPException(status_code=422, detail=error_msg)
            raise HTTPException(status_code=500, detail=error_msg)

    @router.get("/images", tags=["Image Management"])
    async def list_images(limit: int = 50, offset: int = 0):
        """List all uploaded images (RF-11)."""
        images = image_service.list_images(limit=limit, offset=offset)
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
    async def get_image(image_id: int):
        """Get image metadata by ID."""
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
        
        Full workflow: Preprocessing → Inference (Factory + Strategy + Decorator).
        Returns classification label and confidence score.
        """
        try:
            # Get image from database
            image = image_service.get_image(image_id)
            if not image:
                raise HTTPException(status_code=404, detail="Image not found")

            # Load image data for preprocessing
            from infrastructure.storage import LocalStorageAdapter
            from PIL import Image
            import numpy as np

            storage = LocalStorageAdapter()
            file_data = storage.retrieve(image.original_path)
            img = Image.open(io.BytesIO(file_data)).convert("RGB")
            image_array = np.array(img)

            # Step 1: Resolve user ID from token or image record
            user_id = None
            if authorization and authorization.startswith("Bearer "):
                try:
                    token_str = authorization.split(" ")[1].strip()
                    if token_str and token_str not in ("null", "undefined"):
                        token_data = auth_service.verify_token(token_str)
                        user_id = token_data.get("user_id")
                        if not user_id and token_data.get("sub"):
                            u = auth_service._user_repo.find_by_username(token_data.get("sub"))
                            if u:
                                user_id = u.id
                except Exception:
                    pass

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

            # Step 3: Save prediction result
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
                "all_probabilities": classification.get("all_probabilities", {}),
                "model": f"{classification.get('model_name')} v{classification.get('model_version')}",
                "processing_time_ms": classification.get("processing_time_ms"),
                "quality_report": prep_result.get("quality_report", {}),
                "metrics": classification.get("metrics", {}),
            }

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

    @router.get("/model/info", tags=["AI Classification"])
    async def get_model_info():
        """Get information about the currently loaded AI model."""
        return inference_service.get_model_info()

    # ── Clinical Results Routes (Clinical Results Service) ─────

    @router.get("/results", tags=["Clinical Results"])
    async def list_results(limit: int = 50, offset: int = 0, status: Optional[str] = None):
        """List prediction results with optional filtering (RF-11)."""
        from domain.interfaces.contracts import PredictionStatus
        status_filter = PredictionStatus(status) if status else None
        results = clinical_service.get_history(limit=limit, offset=offset, status=status_filter)
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
    async def get_result(result_id: int):
        """Get a specific prediction result (RF-07)."""
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
    async def review_result(result_id: int, request: ReviewRequest):
        """Submit a clinical review for a prediction (RF-09)."""
        try:
            from domain.interfaces.contracts import PredictionStatus
            status = PredictionStatus(request.status)
            result = clinical_service.review_prediction(
                result_id=result_id,
                reviewer_id=1,  # In production: extract from JWT
                status=status,
                notes=request.notes,
            )
            if not result:
                raise HTTPException(status_code=404, detail="Result not found")
            return {"message": "Review submitted", "status": result.status.value}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    @router.post("/results/{result_id}/report", tags=["Clinical Results"])
    async def generate_report(
        result_id: int,
        request: ReportRequest,
        authorization: Optional[str] = Header(None),
    ):
        """Generate a clinical PDF report (RF-10)."""
        try:
            from pathlib import Path

            user_id = None
            if authorization and authorization.startswith("Bearer "):
                try:
                    token_str = authorization.split(" ")[1].strip()
                    if token_str and token_str not in ("null", "undefined"):
                        token_data = auth_service.verify_token(token_str)
                        user_id = token_data.get("user_id")
                        if not user_id and token_data.get("sub"):
                            u = auth_service._user_repo.find_by_username(token_data.get("sub"))
                            if u:
                                user_id = u.id
                except Exception:
                    pass

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
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @router.get("/reports/{filename}", tags=["Clinical Results"])
    async def download_report_file(filename: str):
        """View or download a generated clinical PDF report."""
        from fastapi.responses import FileResponse
        from config.settings import REPORTS_DIR
        file_path = REPORTS_DIR / filename
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Report file not found")
        return FileResponse(
            path=str(file_path),
            media_type="application/pdf",
            filename=filename,
            headers={"Content-Disposition": f"inline; filename={filename}"}
        )

    # ── Audit Routes (Audit Service) ──────────────────────────

    @router.get("/audit", tags=["Audit"])
    async def get_audit_trail(limit: int = 100, offset: int = 0, action: Optional[str] = None):
        """Retrieve audit trail entries with Mexico timezone conversion and usernames (RNF-09)."""
        from domain.interfaces.contracts import AuditAction
        from infrastructure.database import engine
        from datetime import timezone
        import zoneinfo

        action_filter = AuditAction(action) if action else None
        entries = audit_service.get_audit_trail(limit=limit, offset=offset, action_filter=action_filter)

        users_map = {}
        try:
            with engine.connect() as conn:
                cursor = conn.connection.cursor()
                cursor.execute("SELECT id, username FROM users")
                for row in cursor.fetchall():
                    users_map[row[0]] = row[1]
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
                # Formato en español México con 12 horas: ej. 26/10/2026, 01:52:14 p. m.
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

    # ── System Health ──────────────────────────────────────────

    @router.get("/health", tags=["System"])
    async def health_check():
        """System health check endpoint."""
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "2.0.0",
            "services": {
                "auth": "active",
                "image_management": "active",
                "preprocessing": "active",
                "inference": "active",
                "clinical_results": "active",
                "audit": "active",
            },
        }

    return router
