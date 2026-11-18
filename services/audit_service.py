"""
Audit Service (RNF-09, RD-04)
================================
Records all relevant system operations for clinical and quality traceability.
The audit log is append-only (immutable) — no update or delete operations.
"""

from typing import List, Optional

from domain.interfaces.contracts import (
    IAuditRepository,
    AuditEntry,
    AuditAction,
)
from utils.logger import get_logger, get_audit_logger

logger = get_logger("AuditService")
audit_logger = get_audit_logger("AuditService")


class AuditService:
    """
    Centralized audit logging service.

    All auditable actions (image upload, classification, review, report
    generation, model changes, login/logout) pass through this service,
    which writes them to an append-only repository.
    """

    def __init__(self, audit_repository: IAuditRepository):
        self._audit_repo = audit_repository

    def log_event(
        self,
        action: AuditAction,
        user_id: Optional[int] = None,
        resource_type: str = "",
        resource_id: Optional[int] = None,
        details: str = "",
        ip_address: Optional[str] = None,
    ) -> AuditEntry:
        """Record an auditable event (immutable, append-only)."""
        entry = AuditEntry(
            action=action,
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            details=details,
            ip_address=ip_address,
        )

        saved = self._audit_repo.append(entry)

        # Also write to the audit log file for redundancy
        audit_logger.info(
            f"AUDIT | {action.value} | "
            f"user={user_id} | "
            f"resource={resource_type}:{resource_id} | "
            f"{details}"
        )

        return saved

    def get_audit_trail(
        self,
        limit: int = 100,
        offset: int = 0,
        action_filter: Optional[AuditAction] = None,
    ) -> List[AuditEntry]:
        """Retrieve audit entries with optional action filtering."""
        return self._audit_repo.find_all(
            limit=limit,
            offset=offset,
            action=action_filter,
        )
