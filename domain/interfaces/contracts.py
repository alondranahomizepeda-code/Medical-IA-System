"""
Domain Interfaces (Abstractions)
=================================
Defines the contracts (interfaces) that decouple high-level business logic
from low-level implementation details.

Implements:
  - Dependency Inversion Principle (DIP): High-level modules depend on
    abstractions, not concrete implementations.
  - Interface Segregation Principle (ISP): Each interface is focused on
    a specific responsibility — no client depends on methods it doesn't use.
  - Liskov Substitution Principle (LSP): Any concrete implementation can
    replace its abstraction without altering the expected behavior.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


# ═══════════════════════════════════════════════════════════════
# Domain Enums
# ═══════════════════════════════════════════════════════════════

class UserRole(str, Enum):
    """User roles for RBAC (RF-01)."""
    RADIOLOGIST = "radiologist"
    TECHNICIAN = "technician"
    ADMIN = "admin"
    ML_ENGINEER = "ml_engineer"


class PredictionStatus(str, Enum):
    """Status of a clinical prediction (RF-09)."""
    PENDING_REVIEW = "pending_review"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"


class AuditAction(str, Enum):
    """Auditable actions in the system (RNF-09)."""
    IMAGE_UPLOADED = "image_uploaded"
    IMAGE_VALIDATED = "image_validated"
    IMAGE_PREPROCESSED = "image_preprocessed"
    CLASSIFICATION_COMPLETED = "classification_completed"
    PREDICTION_REVIEWED = "prediction_reviewed"
    REPORT_GENERATED = "report_generated"
    MODEL_VERSION_ACTIVATED = "model_version_activated"
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    PASSWORD_RESET_REQUESTED = "password_reset_requested"
    PASSWORD_RESET_COMPLETED = "password_reset_completed"
    CONFIGURATION_CHANGED = "configuration_changed"


# ═══════════════════════════════════════════════════════════════
# Domain Entities (Data Classes)
# ═══════════════════════════════════════════════════════════════

@dataclass
class User:
    """Represents a system user with role-based access (RF-01)."""
    id: Optional[int] = None
    username: str = ""
    hashed_password: str = ""
    full_name: str = ""
    role: UserRole = UserRole.TECHNICIAN
    is_active: bool = True
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class MedicalImage:
    """Represents a medical image in the system (RF-02)."""
    id: Optional[int] = None
    filename: str = ""
    original_path: str = ""
    processed_path: Optional[str] = None
    file_size_bytes: int = 0
    format: str = ""
    width: Optional[int] = None
    height: Optional[int] = None
    uploaded_by: Optional[int] = None
    uploaded_at: datetime = field(default_factory=datetime.utcnow)
    is_anonymized: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PredictionResult:
    """Represents an AI classification result (RF-05, RF-06, RF-08)."""
    id: Optional[int] = None
    image_id: int = 0
    classification: str = ""
    confidence_score: float = 0.0
    model_name: str = ""
    model_version: str = ""
    processing_time_ms: float = 0.0
    predicted_at: datetime = field(default_factory=datetime.utcnow)
    status: PredictionStatus = PredictionStatus.PENDING_REVIEW
    reviewed_by: Optional[int] = None
    reviewer_notes: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    affected_vertebrae: Optional[str] = None
    curve_type: Optional[str] = None
    curve_detail: Optional[str] = None


@dataclass
class AuditEntry:
    """Represents an immutable audit log entry (RNF-09, RD-04)."""
    id: Optional[int] = None
    action: AuditAction = AuditAction.IMAGE_UPLOADED
    user_id: Optional[int] = None
    resource_type: str = ""
    resource_id: Optional[int] = None
    details: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    ip_address: Optional[str] = None


@dataclass
class ModelVersion:
    """Represents a registered model version in the Model Registry (RF-12, RD-10)."""
    id: Optional[int] = None
    model_name: str = ""
    version: str = ""
    file_path: str = ""
    accuracy: Optional[float] = None
    precision_score: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    is_approved: bool = False
    is_active: bool = False
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_by: Optional[str] = None
    description: str = ""


# ═══════════════════════════════════════════════════════════════
# Interfaces — Inference (Strategy Pattern, Decorator Pattern)
# ═══════════════════════════════════════════════════════════════

class IModelInferencer(ABC):
    """
    Core interface for the AI inference operation (DIP, Strategy).
    
    Both the base classifier AND decorators implement this interface,
    enabling transparent wrapping via the Decorator pattern.
    The Inference Service depends on this abstraction, not on any
    concrete model implementation.
    """

    @abstractmethod
    def predict(self, preprocessed_image, *args, **kwargs) -> Dict[str, Any]:
        """
        Execute inference on a preprocessed image.

        Args:
            preprocessed_image: A NumPy array ready for model input.

        Returns:
            Dictionary with 'classification', 'confidence_score', and metadata.
        """
        ...


class IMLStrategy(ABC):
    """
    Strategy interface for interchangeable classification algorithms.
    
    Allows the Inference Service to swap classification strategies
    without modifying its own code (OCP, LSP).
    """

    @abstractmethod
    def classify(self, preprocessed_image, *args, **kwargs) -> Dict[str, Any]:
        """Execute classification using the specific strategy."""
        ...

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load a trained model from the given path."""
        ...

    @abstractmethod
    def get_model_info(self) -> Dict[str, str]:
        """Return metadata about the loaded model."""
        ...


# ═══════════════════════════════════════════════════════════════
# Interfaces — Repositories (Repository Pattern, DIP)
# ═══════════════════════════════════════════════════════════════

class IImageRepository(ABC):
    """Abstracts persistence for medical image records."""

    @abstractmethod
    def save(self, image: MedicalImage) -> MedicalImage:
        ...

    @abstractmethod
    def find_by_id(self, image_id: int) -> Optional[MedicalImage]:
        ...

    @abstractmethod
    def find_all(self, limit: int = 50, offset: int = 0) -> List[MedicalImage]:
        ...


class IResultRepository(ABC):
    """Abstracts persistence for prediction results."""

    @abstractmethod
    def save(self, result: PredictionResult) -> PredictionResult:
        ...

    @abstractmethod
    def find_by_id(self, result_id: int) -> Optional[PredictionResult]:
        ...

    @abstractmethod
    def find_by_image_id(self, image_id: int) -> Optional[PredictionResult]:
        ...

    @abstractmethod
    def find_all(
        self,
        limit: int = 50,
        offset: int = 0,
        status: Optional[PredictionStatus] = None,
    ) -> List[PredictionResult]:
        ...

    @abstractmethod
    def update_review(
        self,
        result_id: int,
        status: PredictionStatus,
        reviewer_id: int,
        notes: str,
    ) -> Optional[PredictionResult]:
        ...


class IUserRepository(ABC):
    """Abstracts persistence for user records."""

    @abstractmethod
    def save(self, user: User) -> User:
        ...

    @abstractmethod
    def find_by_username(self, username: str) -> Optional[User]:
        ...

    @abstractmethod
    def find_by_id(self, user_id: int) -> Optional[User]:
        ...

    @abstractmethod
    def update_password(self, username: str, new_hashed_password: str) -> bool:
        ...


class IAuditRepository(ABC):
    """Abstracts persistence for audit log entries (append-only)."""

    @abstractmethod
    def append(self, entry: AuditEntry) -> AuditEntry:
        ...

    @abstractmethod
    def find_all(
        self,
        limit: int = 100,
        offset: int = 0,
        action: Optional[AuditAction] = None,
    ) -> List[AuditEntry]:
        ...


# ═══════════════════════════════════════════════════════════════
# Interfaces — Services
# ═══════════════════════════════════════════════════════════════

class IReportGenerator(ABC):
    """Interface for clinical report generation (RF-10, ISP)."""

    @abstractmethod
    def generate(
        self,
        prediction: PredictionResult,
        image: MedicalImage,
        reviewer_name: Optional[str] = None,
    ) -> str:
        """Generate a clinical report and return the file path."""
        ...


class IStorageAdapter(ABC):
    """
    Abstracts file storage operations (DIP).
    
    The domain depends on this interface, not on a specific
    filesystem or cloud storage implementation.
    """

    @abstractmethod
    def store(self, file_data: bytes, filename: str, subdirectory: str = "") -> str:
        """Store file data and return the storage path."""
        ...

    @abstractmethod
    def retrieve(self, file_path: str) -> bytes:
        """Retrieve file data from storage."""
        ...

    @abstractmethod
    def delete(self, file_path: str) -> bool:
        """Delete a file from storage."""
        ...
