"""
Unit Tests: AuthService
=======================
Tests authentication, password hashing, user registration,
JWT token generation, role verification, and password recovery.
"""

import pytest
from passlib.context import CryptContext
from domain.interfaces.contracts import UserRole
from services.auth_service import AuthService, pwd_context
from infrastructure.repositories import UserRepository, AuditRepository
from utils.exceptions import AuthenticationError, AuthorizationError


@pytest.fixture
def auth_service():
    user_repo = UserRepository()
    audit_repo = AuditRepository()
    return AuthService(user_repo, audit_repo)


class TestAuthServiceUnit:
    """Unit test suite for AuthService business logic."""

    def test_password_hashing_and_verification(self):
        """Verify that passwords are securely hashed with bcrypt and verify correctly."""
        raw_pwd = "SecureMedicalPassword2026!"
        hashed = pwd_context.hash(raw_pwd)

        assert hashed != raw_pwd
        assert hashed.startswith("$2b$") or hashed.startswith("$2a$")
        assert pwd_context.verify(raw_pwd, hashed) is True
        assert pwd_context.verify("WrongPassword!", hashed) is False

    def test_user_registration_success(self, auth_service: AuthService):
        """Test registering a new clinical user."""
        test_user = "test_doc_unit"
        existing = auth_service._user_repo.find_by_username(test_user)
        if not existing:
            user = auth_service.register_user(
                username=test_user,
                password="password123",
                full_name="Dr. Unit Test",
                role=UserRole.RADIOLOGIST,
            )
            assert user.id is not None
            assert user.username == test_user
            assert user.role == UserRole.RADIOLOGIST

    def test_user_registration_duplicate_fails(self, auth_service: AuthService):
        """Test that registering duplicate usernames raises an AuthenticationError."""
        test_user = "test_doc_unit"
        with pytest.raises(AuthenticationError):
            auth_service.register_user(
                username=test_user,
                password="password123",
                full_name="Dr. Duplicate",
                role=UserRole.RADIOLOGIST,
            )

    def test_authenticate_valid_credentials(self, auth_service: AuthService):
        """Test that authenticate returns token and user payload for valid credentials."""
        result = auth_service.authenticate("test_doc_unit", "password123")
        assert "access_token" in result
        assert result["token_type"] == "bearer"
        assert result["username"] == "test_doc_unit"
        assert result["role"] == UserRole.RADIOLOGIST.value

    def test_authenticate_invalid_password_raises(self, auth_service: AuthService):
        """Test that wrong credentials raise AuthenticationError."""
        with pytest.raises(AuthenticationError):
            auth_service.authenticate("test_doc_unit", "WrongPassword!")

    def test_verify_token_payload_and_role_validation(self, auth_service: AuthService):
        """Test decoding and verifying claims inside JWT token."""
        auth_res = auth_service.authenticate("test_doc_unit", "password123")
        token = auth_res["access_token"]

        payload = auth_service.verify_token(token)
        assert payload is not None
        assert payload.get("sub") == "test_doc_unit"
        assert payload.get("role") == UserRole.RADIOLOGIST.value

    def test_role_authorization_check(self, auth_service: AuthService):
        """Test RBAC role checking logic."""
        # Radiologist should be authorized for RADIOLOGIST and ADMIN roles
        assert auth_service.authorize("radiologist", [UserRole.RADIOLOGIST, UserRole.ADMIN]) is True

        # Technician should NOT be authorized for RADIOLOGIST-only actions
        with pytest.raises(AuthorizationError):
            auth_service.authorize("technician", [UserRole.RADIOLOGIST])

    def test_password_update(self, auth_service: AuthService):
        """Test password reset updates hash and allows login with new password."""
        updated = auth_service.update_password("test_doc_unit", "newSecurePassword2026!")
        assert updated is True

        # Must authenticate with new password
        res = auth_service.authenticate("test_doc_unit", "newSecurePassword2026!")
        assert "access_token" in res

        # Revert back for other tests
        auth_service.update_password("test_doc_unit", "password123")
