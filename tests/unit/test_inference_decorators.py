"""
Unit Tests: Inference Decorators (Decorator Pattern)
====================================================
Tests the GoF Decorator Pattern wrapping IModelInferencer:
  - LoggingInferencerDecorator (logs diagnostic events and audit records)
  - MetricsInferencerDecorator (tracks throughput and latency statistics in bounded deque)
  - Multi-layer decorator stacking
  - Bounded ring-buffer memory protection
  - Exception propagation
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

    def __init__(self, fixed_class: str = "NO SCOLIOSIS", fixed_confidence: float = 0.95):
        self.fixed_class = fixed_class
        self.fixed_confidence = fixed_confidence

    def predict(self, preprocessed_image, *args, **kwargs) -> Dict[str, Any]:
        return {
            "classification": self.fixed_class,
            "confidence_score": self.fixed_confidence,
            "class_probabilities": {
                "NO SCOLIOSIS": 0.95,
                "SERIOUS SCOLIOSIS (C-CURVE)": 0.03,
                "SEVERE SCOLIOSIS (S-CURVE)": 0.02,
            },
            "model_name": "mock_scoliosis_classifier",
            "model_version": "1.1.0",
        }


class FailingModelInferencer(IModelInferencer):
    """Mock inferencer that raises an error to verify decorator error logging and propagation."""

    def predict(self, preprocessed_image, *args, **kwargs) -> Dict[str, Any]:
        raise RuntimeError("Simulated tensor shape failure")


class TestInferenceDecoratorsUnit:
    """Unit tests for inference decorators."""

    def test_metrics_decorator_measures_latency_and_counts(self):
        """Verify MetricsInferencerDecorator injects metrics and tracks inference counts."""
        base_inferencer = MockModelInferencer()
        metrics_inferencer = MetricsInferencerDecorator(base_inferencer)

        dummy_input = np.zeros((224, 224, 3), dtype=np.float32)
        result1 = metrics_inferencer.predict(dummy_input)

        assert result1["classification"] == "NO SCOLIOSIS"
        assert "metrics" in result1
        assert result1["metrics"]["inference_count"] == 1
        assert result1["metrics"]["total_processing_time_ms"] >= 0.0

        metrics_inferencer.predict(dummy_input)
        metrics_inferencer.predict(dummy_input)

        summary = metrics_inferencer.get_metrics_summary()
        assert summary["total_inferences"] == 3
        assert summary["average_latency_ms"] >= 0.0
        assert summary["classification_distribution"]["NO SCOLIOSIS"] == 3

    def test_logging_decorator_execution(self):
        """Verify LoggingInferencerDecorator adds logged_at timestamp without modifying predictions."""
        base_inferencer = MockModelInferencer(fixed_class="SERIOUS SCOLIOSIS (C-CURVE)", fixed_confidence=0.88)
        logged_inferencer = LoggingInferencerDecorator(base_inferencer)

        dummy_input = np.zeros((224, 224, 3), dtype=np.float32)
        result = logged_inferencer.predict(dummy_input)

        assert result["classification"] == "SERIOUS SCOLIOSIS (C-CURVE)"
        assert result["confidence_score"] == 0.88
        assert "logged_at" in result

    def test_stacked_decorators_pipeline(self):
        """Verify stacking Logging -> Metrics -> BaseInferencer honours LSP."""
        base = MockModelInferencer(fixed_class="SEVERE SCOLIOSIS (S-CURVE)", fixed_confidence=0.91)
        stacked = LoggingInferencerDecorator(MetricsInferencerDecorator(base))

        dummy_input = np.zeros((224, 224, 3), dtype=np.float32)
        result = stacked.predict(dummy_input)

        assert result["classification"] == "SEVERE SCOLIOSIS (S-CURVE)"
        assert "metrics" in result
        assert "logged_at" in result

    def test_metrics_decorator_bounded_history_ring_buffer(self):
        """Verify _latency_history uses a bounded deque so history never grows past maxlen."""
        base = MockModelInferencer()
        metrics_inf = MetricsInferencerDecorator(base, maxlen=10)
        dummy_input = np.zeros((224, 224, 3), dtype=np.float32)

        for _ in range(25):
            metrics_inf.predict(dummy_input)

        summary = metrics_inf.get_metrics_summary()
        assert summary["total_inferences"] == 25
        assert summary["history_window_size"] == 10

    def test_logging_decorator_propagates_exceptions(self):
        """Verify LoggingInferencerDecorator logs and re-raises underlying inference errors."""
        logged = LoggingInferencerDecorator(FailingModelInferencer())
        dummy_input = np.zeros((224, 224, 3), dtype=np.float32)

        with pytest.raises(RuntimeError, match="Simulated tensor shape failure"):
            logged.predict(dummy_input)
