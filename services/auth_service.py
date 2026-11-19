"""
Authentication & User Service (RF-01)
=======================================
Manages authentication, user accounts, roles, 2FA, and permissions via RBAC.

This service handles:
  - User registration and credential validation with bcrypt.
  - Two-Factor Authentication (2FA) via HMAC-SHA256 one-time codes.
  - JWT token generation and cryptographic verification (HS256).
  - Role-based access control (RBAC) enforcement.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
from infrastructure.repositories import OneTimeCodeRepository
from utils.logger import get_logger
from utils.exceptions import AuthenticationError, AuthorizationError, OneTimeCodeError

logger = get_logger("AuthService")

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class AuthService:
    """
    Authentication & User Service.

    Depends on abstractions (IUserRepository, IAuditRepository) and
    OneTimeCodeRepository — not on concrete database sessions (DIP).
    """

    def __init__(
        self,
        user_repository: IUserRepository,
        audit_repository: IAuditRepository,
        one_time_code_repository: Optional[OneTimeCodeRepository] = None,
    ):
        self._user_repo = user_repository
        self._audit_repo = audit_repository
        self._otp_repo = one_time_code_repository or OneTimeCodeRepository()

    def register_user(
        self,
        username: str,
        password: str,
        full_name: str,
        role: UserRole = UserRole.TECHNICIAN,
    ) -> User:
        """Register a new user with hashed password and input validation."""
        clean_username = (username or "").replace("\x00", "").strip()
        clean_full_name = (full_name or "").replace("\x00", "").strip()
        clean_password = (password or "").replace("\x00", "")

        if len(clean_username) < 3 or len(clean_username) > 64:
            raise AuthenticationError("Username must be between 3 and 64 characters long")
        if not clean_password or len(clean_password) < 6:
            raise AuthenticationError("Password must be at least 6 characters long")
        if not clean_full_name or len(clean_full_name) > 120:
            raise AuthenticationError("Full name must be between 1 and 120 characters long")

        existing = self._user_repo.find_by_username(clean_username)
        if existing:
            raise AuthenticationError(f"Username '{clean_username}' already exists")

        pwd_bytes = clean_password.encode("utf-8")
        safe_password = pwd_bytes[:72].decode("utf-8", "ignore") if len(pwd_bytes) > 72 else clean_password

        user = User(
            username=clean_username,
            hashed_password=pwd_context.hash(safe_password),
            full_name=clean_full_name,
            role=role,
        )

        saved_user = self._user_repo.save(user)
        logger.info(f"User registered: {clean_username} (role: {role.value})")

        return saved_user

    def verify_credentials(self, username: str, password: str) -> User:
        """Validate username and password without issuing a JWT yet."""
        clean_username = (username or "").replace("\x00", "").strip()[:64]
        clean_password = (password or "").replace("\x00", "")
        user = self._user_repo.find_by_username(clean_username)

        pwd_bytes = clean_password.encode("utf-8")
        safe_password = pwd_bytes[:72].decode("utf-8", "ignore") if len(pwd_bytes) > 72 else clean_password

        if not user or not pwd_context.verify(safe_password, user.hashed_password):
            logger.warning(f"Failed login attempt for: {username}")
            raise AuthenticationError("Invalid username or password")

        if not user.is_active:
            raise AuthenticationError("User account is deactivated")

        return user

    def issue_jwt_for_user(
        self,
        user: User,
        record_audit: bool = True,
        two_factor_verified: bool = True,
    ) -> dict:
        """Issue a signed JWT HS256 token for a verified user."""
        token_data = {
            "sub": user.username,
            "user_id": user.id,
            "role": user.role.value,
            "full_name": user.full_name,
            "two_factor_verified": two_factor_verified,
            "exp": datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
        }
        access_token = jwt.encode(token_data, SECRET_KEY, algorithm=ALGORITHM)

        if record_audit:
            self._audit_repo.append(AuditEntry(
                action=AuditAction.USER_LOGIN,
                user_id=user.id,
                resource_type="user",
                resource_id=user.id,
                details=f"User '{user.username}' logged in successfully",
            ))

        return {
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": user.id,
            "username": user.username,
            "role": user.role.value,
            "full_name": user.full_name,
        }

    def authenticate(self, username: str, password: str) -> dict:
        """
        Authenticate a user and return a JWT token payload.
        Raises AuthenticationError if credentials are invalid.
        """
        user = self.verify_credentials(username, password)
        logger.info(f"User authenticated: {username}")
        return self.issue_jwt_for_user(user, record_audit=True)

    def initiate_2fa_login(self, username: str, password: str) -> Tuple[User, str]:
        """
        Step 1 of 2FA login: verify password and issue a 6-digit HMAC-SHA256 code
        persisted in the one_time_codes table.
        """
        user = self.verify_credentials(username, password)
        raw_code, _ = self._otp_repo.issue_code(username=user.username, purpose="2fa", ttl_minutes=5, is_resend=False)
        return user, raw_code

    def resend_2fa_code(self, username: str) -> Tuple[str, int]:
        """Resend 2FA code (up to 3 total sends)."""
        raw_code, sends = self._otp_repo.issue_code(username=username, purpose="2fa", ttl_minutes=5, is_resend=True)
        return raw_code, (OneTimeCodeRepository.MAX_SENDS - sends)

    def complete_2fa_login(self, username: str, code: str) -> dict:
        """
        Step 2 of 2FA login: verify and consume the 6-digit code, then issue the JWT.
        """
        self._otp_repo.verify_and_consume(username=username, code=code, purpose="2fa")
        user = self._user_repo.find_by_username(username)
        if not user or not user.is_active:
            raise AuthenticationError("User not found or inactive")
        return self.issue_jwt_for_user(user, record_audit=True)

    def verify_token(self, token: str) -> dict:
        """Verify and decode a JWT token."""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError as e:
            raise AuthenticationError(f"Invalid or expired token: {str(e)}")

    def authorize(self, user_role: str, required_roles: List[UserRole]) -> bool:
        """
        Check if a user role has permission for an action.

        Raises AuthorizationError if the role is not in the required list.
        """
        allowed_values = [r.value if hasattr(r, "value") else str(r) for r in required_roles]
        role_val = user_role.value if hasattr(user_role, "value") else str(user_role)
        if role_val not in allowed_values:
            raise AuthorizationError(role_val, f"requires one of {allowed_values}")
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

        pwd_bytes = new_password.encode("utf-8")
        safe_password = pwd_bytes[:72].decode("utf-8", "ignore") if len(pwd_bytes) > 72 else new_password
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
