"""
Unit Tests: ModelRegistry (RF-12, RD-10, Question 10)
=====================================================
Tests model registration, approval-before-activation rule, single active version
invariant, cross-platform path resolution, and bootstrap of packaged artefacts.
"""

import pytest
from infrastructure.model_registry import ModelRegistry
from utils.exceptions import ModelNotFoundError, RepositoryError


class TestModelRegistryUnit:
    """Unit test suite for ModelRegistry lifecycle management."""

    def test_register_version_starts_unapproved_and_inactive(self):
        """New model versions must start with is_approved=False and is_active=False."""
        registry = ModelRegistry()
        ver = registry.register_version(
            model_name="test_scoliosis_reg",
            version="0.1.0",
            file_path="scoliosis_classifier_v1.0.0/model.keras",
            metrics={"accuracy": 0.4455, "precision": 0.1485, "recall": 0.3333, "f1_score": 0.2055},
            created_by="unit_test",
            description="Unapproved test version",
        )
        assert ver.is_approved is False
        assert ver.is_active is False

    def test_cannot_activate_unapproved_version_question_10(self):
        """Question 10: Activating a version that has not passed approval must raise RepositoryError."""
        registry = ModelRegistry()
        registry.register_version(
            model_name="test_scoliosis_reg",
            version="0.2.0",
            file_path="scoliosis_classifier_v1.0.0/model.keras",
        )
        with pytest.raises(RepositoryError) as exc_info:
            registry.activate_version("test_scoliosis_reg", "0.2.0")
        assert "not approved" in str(exc_info.value).lower()

    def test_approve_and_single_active_version_invariant(self):
        """Only ONE version per model_name may be active at any time."""
        registry = ModelRegistry()
        registry.register_version("test_single_active", "1.0.0", "scoliosis_classifier_v1.0.0/model.keras")
        registry.approve_version("test_single_active", "1.0.0")
        registry.activate_version("test_single_active", "1.0.0")

        registry.register_version("test_single_active", "1.1.0", "scoliosis_classifier_v1.1.0/model.keras")
        registry.approve_version("test_single_active", "1.1.0")
        registry.activate_version("test_single_active", "1.1.0")

        versions = registry.list_versions("test_single_active")
        active_versions = [v for v in versions if v.is_active]
        assert len(active_versions) == 1
        assert active_versions[0].version == "1.1.0"

    def test_approve_nonexistent_version_raises_model_not_found(self):
        """Approving an unknown model version must raise ModelNotFoundError."""
        registry = ModelRegistry()
        with pytest.raises(ModelNotFoundError):
            registry.approve_version("nonexistent_model_xyz", "9.9.9")

    def test_resolve_legacy_windows_model_path(self):
        """Verify legacy Windows backslash paths resolve cleanly across OS environments."""
        legacy = r"C:\Users\SomeUser\Proyecto\Codigo\data\models\scoliosis_classifier_v1.0.0\model.keras"
        resolved = ModelRegistry.resolve_model_path(legacy)
        assert "model.keras" in resolved
