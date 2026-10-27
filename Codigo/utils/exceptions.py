"""
Custom Exception Hierarchy
===========================
Defines domain-specific exceptions for the Medical Image Classification System.
Supports Rubric Criterion 8: Error Handling and Exception Management.

Each exception maps to a specific failure scenario, enabling the Fail Fast
principle and providing clear, user-friendly error messages (RF-03).
"""


class MedicalSystemError(Exception):
    """Base exception for all system errors."""

    def __init__(self, message: str, error_code: str = "SYSTEM_ERROR"):
        self.message = message
        self.error_code = error_code
        super().__init__(self.message)


# ── Image Validation Errors (Fail Fast - RF-03) ──────────────

class ImageValidationError(MedicalSystemError):
    """Raised when an image fails the Fail Fast validation stage."""

    def __init__(self, message: str):
        super().__init__(message, error_code="IMAGE_VALIDATION_ERROR")


class UnsupportedFormatError(ImageValidationError):
    """Raised when the image format is not in the allowed list (DICOM, JPEG, PNG, TIFF)."""

    def __init__(self, extension: str):
        allowed = ", ".join([".dcm", ".jpg", ".jpeg", ".png", ".tiff"])
        super().__init__(
            f"Unsupported image format '{extension}'. Allowed formats: {allowed}"
        )


class FileSizeExceededError(ImageValidationError):
    """Raised when the uploaded file exceeds the maximum allowed size (15 MB)."""

    def __init__(self, size_mb: float, max_mb: float = 15.0):
        super().__init__(
            f"File size ({size_mb:.1f} MB) exceeds the maximum allowed size ({max_mb} MB)"
        )


class CorruptedImageError(ImageValidationError):
    """Raised when the image file is corrupted or cannot be read."""

    def __init__(self, filename: str):
        super().__init__(f"Image file '{filename}' is corrupted or cannot be decoded")


class ImageQualityError(ImageValidationError):
    """Raised when the image does not meet minimum quality standards (RD-09)."""

    def __init__(self, reason: str):
        super().__init__(f"Image quality check failed: {reason}")


# ── Authentication & Authorization Errors (RF-01) ────────────

class AuthenticationError(MedicalSystemError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Invalid credentials"):
        super().__init__(message, error_code="AUTH_ERROR")


class AuthorizationError(MedicalSystemError):
    """Raised when a user lacks the required permissions (RBAC)."""

    def __init__(self, role: str, action: str):
        super().__init__(
            f"Role '{role}' is not authorized to perform action '{action}'",
            error_code="AUTHORIZATION_ERROR",
        )


# ── Inference & Model Errors (RF-05, RF-12) ──────────────────

class InferenceError(MedicalSystemError):
    """Raised when the AI inference process fails."""

    def __init__(self, message: str):
        super().__init__(message, error_code="INFERENCE_ERROR")


class ModelNotFoundError(InferenceError):
    """Raised when no approved model version is found in the Model Registry."""

    def __init__(self, model_name: str):
        super().__init__(
            f"No approved model version found for '{model_name}' in the Model Registry"
        )


class ModelLoadError(InferenceError):
    """Raised when a model cannot be loaded from the registry."""

    def __init__(self, model_name: str, version: str):
        super().__init__(
            f"Failed to load model '{model_name}' version '{version}'"
        )


# ── Pipeline Errors ──────────────────────────────────────────

class PipelineStageError(MedicalSystemError):
    """Raised when a specific pipeline stage fails."""

    def __init__(self, stage_name: str, reason: str):
        super().__init__(
            f"Pipeline stage '{stage_name}' failed: {reason}",
            error_code="PIPELINE_ERROR",
        )


# ── Persistence Errors (Repository Pattern) ──────────────────

class RepositoryError(MedicalSystemError):
    """Raised when a database operation fails."""

    def __init__(self, operation: str, reason: str):
        super().__init__(
            f"Repository operation '{operation}' failed: {reason}",
            error_code="REPOSITORY_ERROR",
        )
