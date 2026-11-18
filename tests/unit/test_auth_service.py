"""
Unit Tests: AuthService (RF-01, RNF-03, Question 5)
===================================================
Tests authentication, bcrypt hashing, user registration, 2FA single-use codes,
lockout after 5 failed attempts, resend limit (3), code expiration, forged JWT
rejection, and strict RBAC authorization rules (Question 5).
"""

from datetime import datetime, timedelta
import pytest
from jose import jwt

from config.settings import SECRET_KEY, ALGORITHM
from domain.interfaces.contracts import UserRole
from infrastructure.database import session_scope, OneTimeCodeDB
from infrastructure.repositories import UserRepository, AuditRepository, OneTimeCodeRepository
from services.auth_service import AuthService, pwd_context
from utils.exceptions import AuthenticationError, AuthorizationError, OneTimeCodeError


@pytest.fixture
def auth_service():
    user_repo = UserRepository()
    audit_repo = AuditRepository()
    otp_repo = OneTimeCodeRepository()
    return AuthService(user_repo, audit_repo, otp_repo)


class TestAuthServiceUnit:
    """Unit test suite for AuthService business logic and security controls."""

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

    def test_tampered_or_forged_jwt_rejected(self, auth_service: AuthService):
        """Verify that a JWT signed with a wrong secret key is rejected."""
        forged = jwt.encode(
            {"sub": "technician", "role": "admin", "exp": datetime.utcnow() + timedelta(minutes=30)},
            "wrong-attacker-secret-key",
            algorithm=ALGORITHM,
        )
        with pytest.raises(AuthenticationError):
            auth_service.verify_token(forged)

    def test_expired_jwt_rejected(self, auth_service: AuthService):
        """Verify that an expired JWT token is rejected."""
        expired = jwt.encode(
            {"sub": "radiologist", "role": "radiologist", "exp": datetime.utcnow() - timedelta(minutes=5)},
            SECRET_KEY,
            algorithm=ALGORITHM,
        )
        with pytest.raises(AuthenticationError):
            auth_service.verify_token(expired)

    def test_role_authorization_check_question_5(self, auth_service: AuthService):
        """
        Question 5 unit verification:
        Radiologist and Admin are authorized to review predictions, whereas Technician
        and ML Engineer raise AuthorizationError (mapped to HTTP 403 Forbidden).
        """
        assert auth_service.authorize("radiologist", [UserRole.RADIOLOGIST, UserRole.ADMIN]) is True
        assert auth_service.authorize("admin", [UserRole.RADIOLOGIST, UserRole.ADMIN]) is True

        with pytest.raises(AuthorizationError):
            auth_service.authorize("technician", [UserRole.RADIOLOGIST, UserRole.ADMIN])

        with pytest.raises(AuthorizationError):
            auth_service.authorize("ml_engineer", [UserRole.RADIOLOGIST, UserRole.ADMIN])

    def test_2fa_code_single_use_and_hmac_storage(self, auth_service: AuthService):
        """Verify 2FA code is stored as HMAC-SHA256 (not plaintext) and cannot be reused."""
        _, raw_code = auth_service.initiate_2fa_login("test_doc_unit", "password123")

        with session_scope() as session:
            rec = session.query(OneTimeCodeDB).filter_by(username="test_doc_unit", purpose="2fa").first()
            assert rec is not None
            assert rec.code_hash != raw_code
            assert len(rec.code_hash) == 64

        res = auth_service.complete_2fa_login("test_doc_unit", raw_code)
        assert "access_token" in res

        # Reusing the same 2FA code a second time must fail
        with pytest.raises(OneTimeCodeError):
            auth_service.complete_2fa_login("test_doc_unit", raw_code)

    def test_2fa_lockout_after_5_failed_attempts(self, auth_service: AuthService):
        """Verify 2FA code is invalidated after 5 consecutive wrong attempts."""
        _, raw_code = auth_service.initiate_2fa_login("test_doc_unit", "password123")

        for _ in range(5):
            with pytest.raises(OneTimeCodeError):
                auth_service.complete_2fa_login("test_doc_unit", "000000")

        # Even the originally valid code must now be rejected because the record was purged
        with pytest.raises(OneTimeCodeError):
            auth_service.complete_2fa_login("test_doc_unit", raw_code)

    def test_2fa_expired_code_rejected(self, auth_service: AuthService):
        """Verify an expired 2FA code in one_time_codes is rejected and deleted."""
        _, raw_code = auth_service.initiate_2fa_login("test_doc_unit", "password123")
        with session_scope() as session:
            rec = session.query(OneTimeCodeDB).filter_by(username="test_doc_unit", purpose="2fa").first()
            rec.expires_at = datetime.utcnow() - timedelta(minutes=1)

        with pytest.raises(OneTimeCodeError, match="expired"):
            auth_service.complete_2fa_login("test_doc_unit", raw_code)

    def test_2fa_resend_limit_enforced(self, auth_service: AuthService):
        """Verify a user can resend a 2FA code at most twice after initial send (3 total sends)."""
        auth_service.initiate_2fa_login("test_doc_unit", "password123")
        _, left1 = auth_service.resend_2fa_code("test_doc_unit")
        assert left1 == 1
        code3, left2 = auth_service.resend_2fa_code("test_doc_unit")
        assert left2 == 0

        # 4th send must raise OneTimeCodeError
        with pytest.raises(OneTimeCodeError, match="Maximum resend"):
            auth_service.resend_2fa_code("test_doc_unit")

        # Clean up by completing login with the 3rd code
        auth_service.complete_2fa_login("test_doc_unit", code3)

    def test_password_update(self, auth_service: AuthService):
        """Test password reset updates hash and allows login with new password."""
        updated = auth_service.update_password("test_doc_unit", "newSecurePassword2026!")
        assert updated is True

        res = auth_service.authenticate("test_doc_unit", "newSecurePassword2026!")
        assert "access_token" in res

        auth_service.update_password("test_doc_unit", "password123")

    def test_password_too_short_rejected(self, auth_service: AuthService):
        """Verify passwords shorter than 6 characters are rejected."""
        with pytest.raises(AuthenticationError, match="at least 6 characters"):
            auth_service.update_password("test_doc_unit", "123")
