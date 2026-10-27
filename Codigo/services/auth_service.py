"""
Authentication & User Service (RF-01)
=======================================
Manages authentication, user accounts, roles, and permissions via RBAC.

This service handles:
  - User registration and credential validation.
  - JWT token generation and verification.
  - Role-based access control (RBAC).
"""

from datetime import datetime, timedelta
from typing import Optional

from jose import jwt, JWTError
from passlib.context import CryptContext

from config.settings import SECRET_KEY, ALGORITHM, ACCESS_TOKEN_EXPIRE_MINUTES
from domain.interfaces.contracts import (
    IUserRepository,
    IAuditRepository,
    User,
    UserRole,
    AuditEntry,
    AuditAction,
)
from utils.logger import get_logger
from utils.exceptions import AuthenticationError, AuthorizationError

logger = get_logger("AuthService")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """
    Authentication & User Service.

    Depends on abstractions (IUserRepository, IAuditRepository) —
    not on concrete database implementations (DIP).
    """

    def __init__(
        self,
        user_repository: IUserRepository,
        audit_repository: IAuditRepository,
    ):
        self._user_repo = user_repository
        self._audit_repo = audit_repository

    def register_user(
        self,
        username: str,
        password: str,
        full_name: str,
        role: UserRole = UserRole.TECHNICIAN,
    ) -> User:
        """Register a new user with hashed password."""
        # Check if username already exists
        existing = self._user_repo.find_by_username(username)
        if existing:
            raise AuthenticationError(f"Username '{username}' already exists")

        pwd_bytes = password.encode('utf-8')
        safe_password = pwd_bytes[:72].decode('utf-8', 'ignore') if len(pwd_bytes) > 72 else password

        user = User(
            username=username,
            hashed_password=pwd_context.hash(safe_password),
            full_name=full_name,
            role=role,
        )

        saved_user = self._user_repo.save(user)
        logger.info(f"User registered: {username} (role: {role.value})")

        return saved_user

    def authenticate(self, username: str, password: str) -> dict:
        """
        Authenticate a user and return a JWT token.

        Raises AuthenticationError if credentials are invalid.
        """
        user = self._user_repo.find_by_username(username)

        pwd_bytes = password.encode('utf-8')
        safe_password = pwd_bytes[:72].decode('utf-8', 'ignore') if len(pwd_bytes) > 72 else password

        if not user or not pwd_context.verify(safe_password, user.hashed_password):
            logger.warning(f"Failed login attempt for: {username}")
            raise AuthenticationError("Invalid username or password")

        if not user.is_active:
            raise AuthenticationError("User account is deactivated")

        # Generate JWT token
        token_data = {
            "sub": user.username,
            "user_id": user.id,
            "role": user.role.value,
            "full_name": user.full_name,
            "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        }
        access_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

        # Audit: record login event
        self._audit_repo.append(AuditEntry(
            action=AuditAction.USER_LOGIN,
            user_id=user.id,
            resource_type="user",
            resource_id=user.id,
            details=f"User '{username}' logged in successfully",
        ))

        logger.info(f"User authenticated: {username}")
        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": user.id,
            "username": user.username,
            "role": user.role.value,
            "full_name": user.full_name,
        }

    def verify_token(self, token: str) -> dict:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            raise AuthenticationError(f"Invalid or expired token: {str(e)}")

    def authorize(self, user_role: str, required_roles: list) -> bool:
        """
        Check if a user role has permission for an action.

        Raises AuthorizationError if the role is not in the required list.
        """
        if user_role not in [r.value for r in required_roles]:
            raise AuthorizationError(user_role, f"requires one of {[r.value for r in required_roles]}")
        return True

    def update_password(self, username: str, new_password: str) -> bool:
        """
        Update a user's password using bcrypt hashing and record an audit log entry.
        """
        user = self._user_repo.find_by_username(username)
        if not user:
            raise AuthenticationError(f"User '{username}' not found")

        if len(new_password) < 6:
            raise AuthenticationError("New password must be at least 6 characters long")

        pwd_bytes = new_password.encode('utf-8')
        safe_password = pwd_bytes[:72].decode('utf-8', 'ignore') if len(pwd_bytes) > 72 else new_password
        new_hashed = pwd_context.hash(safe_password)

        updated = self._user_repo.update_password(username, new_hashed)
        if updated:
            self._audit_repo.append(AuditEntry(
                action=AuditAction.PASSWORD_RESET_COMPLETED,
                user_id=user.id,
                resource_type="user",
                resource_id=user.id,
                details=f"Password was successfully reset for user '{username}'",
            ))
            logger.info(f"Password reset successfully for user: {username}")
        return updated
