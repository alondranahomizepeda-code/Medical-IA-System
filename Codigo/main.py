"""
Medical Image Classification System — Main Application
========================================================
Entry point for the application. Initializes all services with their
dependencies following Dependency Injection and starts the API server.

Architecture: Service-Based Modular Architecture
  - API Gateway (FastAPI)
  - Authentication & User Service
  - Image Management Service
  - Preprocessing Service
  - Inference Service
  - Clinical Results Service
  - Audit Service

Team 3: Aguilar, Hernández, Salinas, Zepeda
Software Engineering II — Milestone 2
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.openapi.docs import get_swagger_ui_html
import uvicorn

from config.settings import API_HOST, API_PORT, API_TITLE, API_DESCRIPTION, API_VERSION
from utils.logger import setup_logger, get_logger

# Initialize logging FIRST
setup_logger()
logger = get_logger("Main")


def create_application() -> FastAPI:
    """
    Application Factory: creates and configures the FastAPI application.

    This function wires all dependencies following Dependency Injection:
      1. Creates infrastructure components (repositories, storage, registry).
      2. Injects them into services.
      3. Injects services into the API routes.

    No service knows about concrete implementations — they all depend
    on abstractions (DIP).
    """
    logger.info("=" * 60)
    logger.info("  Medical Image Classification System")
    logger.info("  Team 3 — Software Engineering II")
    logger.info("=" * 60)

    # ── Step 1: Initialize Database ───────────────────────────
    logger.info("Initializing database...")
    from infrastructure.database import init_database
    init_database()

    # ── Step 2: Create Infrastructure (Concrete Implementations) ──
    logger.info("Creating infrastructure components...")
    from infrastructure.repositories import (
        ImageRepository,
        ResultRepository,
        UserRepository,
        AuditRepository,
    )
    from infrastructure.storage import LocalStorageAdapter
    from infrastructure.model_registry import ModelRegistry

    image_repo = ImageRepository()
    result_repo = ResultRepository()
    user_repo = UserRepository()
    audit_repo = AuditRepository()
    storage = LocalStorageAdapter()
    model_registry = ModelRegistry()

    # ── Step 3: Create Services (Injecting Dependencies) ──────
    logger.info("Creating services with dependency injection...")
    from services.auth_service import AuthService
    from services.image_management_service import ImageManagementService
    from services.preprocessing_service import PreprocessingService
    from services.inference_service import InferenceService
    from services.clinical_results_service import ClinicalResultsService
    from services.audit_service import AuditService

    auth_service = AuthService(
        user_repository=user_repo,       # IUserRepository
        audit_repository=audit_repo,     # IAuditRepository
    )

    image_service = ImageManagementService(
        image_repository=image_repo,     # IImageRepository
        storage_adapter=storage,         # IStorageAdapter
        audit_repository=audit_repo,     # IAuditRepository
    )

    preprocessing_service = PreprocessingService(
        audit_repository=audit_repo,     # IAuditRepository
    )

    inference_service = InferenceService(
        model_registry=model_registry,   # ModelRegistry
        audit_repository=audit_repo,     # IAuditRepository
    )

    clinical_service = ClinicalResultsService(
        result_repository=result_repo,   # IResultRepository
        image_repository=image_repo,     # IImageRepository
        audit_repository=audit_repo,     # IAuditRepository
    )

    audit_service = AuditService(
        audit_repository=audit_repo,     # IAuditRepository
    )

    # ── Step 4: Create default users ─────────────────────────
    logger.info("Setting up default users...")
    from domain.interfaces.contracts import UserRole
    default_users = [
        ("radiologist", "password123", "Dr. Radiologist", UserRole.RADIOLOGIST),
        ("technician", "password123", "Imaging Technician", UserRole.TECHNICIAN),
        ("admin", "password123", "System Administrator", UserRole.ADMIN),
        ("ml_engineer", "password123", "ML Engineer", UserRole.ML_ENGINEER),
    ]
    for username, password, full_name, role in default_users:
        try:
            auth_service.register_user(username, password, full_name, role)
            logger.info(f"  Created user: {username} ({role.value})")
        except Exception:
            logger.debug(f"  User '{username}' already exists")

    # ── Step 5: Load AI Model ─────────────────────────────────
    logger.info("Loading AI model...")
    model_info = inference_service.load_model()
    logger.info(f"Model status: {model_info.get('status')}")

    # ── Step 6: Create FastAPI Application ────────────────────
    app = FastAPI(
        title=API_TITLE,
        description=API_DESCRIPTION,
        version=API_VERSION,
        docs_url=None,          # Custom Swagger UI route below
        redoc_url="/redoc",     # ReDoc
    )

    @app.get("/docs", include_in_schema=False)
    async def custom_swagger_ui_html():
        html_response = get_swagger_ui_html(
            openapi_url=app.openapi_url,
            title=app.title + " - UI Interactiva"
        )
        custom_css = """
        <style>
            /* Tema Médico Profesional e Intuitivo */
            body { background-color: #f0f4f8; font-family: 'Inter', 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
            
            /* Encabezado Principal */
            .swagger-ui .info { margin: 40px auto; padding: 30px; background: white; border-radius: 12px; box-shadow: 0 10px 25px rgba(0,0,0,0.05); border-top: 6px solid #1E3A8A; }
            .swagger-ui .info .title { color: #1E3A8A !important; font-weight: 800; font-size: 32px; }
            
            /* Ocultar etiquetas numéricas (2.0.0 y OAS 3.1) */
            .swagger-ui .info .title small { display: none !important; }
            .swagger-ui .info .title a { display: none !important; }
            .swagger-ui .info .base-url { display: none !important; }
            .swagger-ui .version-stamp { display: none !important; }
            
            /* Ocultar barra verde por defecto de Swagger */
            .swagger-ui .topbar { display: none !important; }
            
            /* Ocultar sección técnica de Schemas */
            .swagger-ui section.models { display: none !important; }
            
            /* Colores de los Bloques (POST en Azul, GET en Verde) */
            .swagger-ui .opblock { border-radius: 8px; margin-bottom: 15px; border-width: 1px; box-shadow: 0 2px 5px rgba(0,0,0,0.02); }
            
            .swagger-ui .opblock.opblock-post { border-color: #1E3A8A !important; background: rgba(30, 58, 138, 0.03) !important; }
            .swagger-ui .opblock.opblock-post .opblock-summary { border-color: #1E3A8A !important; }
            .swagger-ui .opblock.opblock-post .opblock-summary-method { background: #1E3A8A !important; border-radius: 4px; }
            
            .swagger-ui .opblock.opblock-get { border-color: #10B981 !important; background: rgba(16, 185, 129, 0.03) !important; }
            .swagger-ui .opblock.opblock-get .opblock-summary { border-color: #10B981 !important; }
            .swagger-ui .opblock.opblock-get .opblock-summary-method { background: #10B981 !important; border-radius: 4px; }
            
            /* Botón gigante de Execute (Ejecutar) */
            .swagger-ui .btn.execute { background-color: #1E3A8A !important; color: white !important; border-color: #1E3A8A !important; padding: 10px 25px; border-radius: 6px; font-weight: bold; transition: 0.3s; }
            .swagger-ui .btn.execute:hover { background-color: #152C6B !important; transform: translateY(-2px); box-shadow: 0 4px 10px rgba(30,58,138,0.3); }
            
            /* Botón de Autorización (Candado) */
            .swagger-ui .btn.authorize { background-color: white !important; color: #1E3A8A !important; border-color: #1E3A8A !important; border-radius: 6px; font-weight: bold; }
            .swagger-ui .btn.authorize svg { fill: #1E3A8A !important; }
            .swagger-ui .btn.authorize:hover { background-color: #1E3A8A !important; color: white !important; }
            .swagger-ui .btn.authorize:hover svg { fill: white !important; }
        </style>
        """
        custom_html = html_response.body.decode().replace("</head>", custom_css + "</head>")
        return HTMLResponse(custom_html)

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Step 7: Register API Routes ───────────────────────────
    from api.routes import create_router
    router = create_router(
        auth_service=auth_service,
        image_service=image_service,
        preprocessing_service=preprocessing_service,
        inference_service=inference_service,
        clinical_service=clinical_service,
        audit_service=audit_service,
    )
    app.include_router(router, prefix="/api/v1")

    # ── Step 8: Mount Frontend Web Application ─────────────────
    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists():
        app.mount("/static", StaticFiles(directory=str(frontend_dir)), name="static")

        @app.get("/", include_in_schema=False)
        async def serve_index():
            return FileResponse(frontend_dir / "index.html")

    logger.info(f"API Gateway ready at http://localhost:{API_PORT}/docs")
    logger.info(f"Web Dashboard ready at http://localhost:{API_PORT}/")
    logger.info("=" * 60)

    return app


# Create the application instance
app = create_application()

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True,
        log_level="info",
    )
