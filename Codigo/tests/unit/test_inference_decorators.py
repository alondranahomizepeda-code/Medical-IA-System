"""
Unit Tests: Inference Decorators (Decorator Pattern)
====================================================
Tests the GoF Decorator Pattern wrapping IModelInferencer:
  - LoggingInferencerDecorator (logs diagnostic events and audit records)
  - MetricsInferencerDecorator (tracks throughput and latency statistics)
  - Multi-layer decorator stacking
"""

import pytest
import numpy as np
from typing import Dict, Any
from domain.interfaces.contracts import IModelInferencer
from services.inference_decorators import (
    LoggingInferencerDecorator,
    MetricsInferencerDecorator,
)


class MockModelInferencer(IModelInferencer):
    """Mock inferencer implementing domain contract without loading heavy neural network."""

    def __init__(self, fixed_class: str = "NO ESCOLIOSIS", fixed_confidence: float = 0.95):
        self.fixed_class = fixed_class
        self.fixed_confidence = fixed_confidence

    def predict(self, preprocessed_image, *args, **kwargs) -> Dict[str, Any]:
        return {
            "classification": self.fixed_class,
            "confidence_score": self.fixed_confidence,
            "class_probabilities": {"NO ESCOLIOSIS": 0.95, "ESCOLIOSIS GRAVE (EN C)": 0.03, "ESCOLIOSIS SEVERA (EN S)": 0.02},
            "model_name": "mock_scoliosis_classifier",
            "model_version": "1.0.0",
        }


class TestInferenceDecoratorsUnit:
    """Unit tests for inference decorators."""

    def test_metrics_decorator_measures_latency_and_counts(self):
        """Verify MetricsInferencerDecorator injects metrics and tracks inference counts."""
        base_inferencer = MockModelInferencer()
        metrics_inferencer = MetricsInferencerDecorator(base_inferencer)

        dummy_input = np.zeros((224, 224, 3), dtype=np.float32)
        result1 = metrics_inferencer.predict(dummy_input)

        assert result1["classification"] == "NO ESCOLIOSIS"
        assert "metrics" in result1
        assert result1["metrics"]["inference_count"] == 1
        assert result1["metrics"]["total_processing_time_ms"] >= 0.0

        # Run 2 more
        metrics_inferencer.predict(dummy_input)
        metrics_inferencer.predict(dummy_input)

        summary = metrics_inferencer.get_metrics_summary()
        assert summary["total_inferences"] == 3
        assert summary["average_latency_ms"] >= 0.0
        assert summary["classification_distribution"]["NO ESCOLIOSIS"] == 3

    def test_logging_decorator_execution(self):
        """Verify LoggingInferencerDecorator adds logged_at timestamp without modifying predictions."""
        base_inferencer = MockModelInferencer(fixed_class="ESCOLIOSIS GRAVE (EN C)", fixed_confidence=0.88)
        logged_inferencer = LoggingInferencerDecorator(base_inferencer)

        dummy_input = np.zeros((224, 224, 3), dtype=np.float32)
        result = logged_inferencer.predict(dummy_input)

        assert result["classification"] == "ESCOLIOSIS GRAVE (EN C)"
        assert result["confidence_score"] == 0.88
        assert "logged_at" in result

    def test_stacked_decorators_pipeline(self):
        """Verify stacking Logging -> Metrics -> BaseInferencer."""
        base = MockModelInferencer(fixed_class="ESCOLIOSIS SEVERA (EN S)", fixed_confidence=0.91)
        stacked = LoggingInferencerDecorator(MetricsInferencerDecorator(base))

        dummy_input = np.zeros((224, 224, 3), dtype=np.float32)
        result = stacked.predict(dummy_input)

        assert result["classification"] == "ESCOLIOSIS SEVERA (EN S)"
        assert "metrics" in result
        assert "logged_at" in result
