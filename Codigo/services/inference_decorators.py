"""
Decorator Pattern (Structural GoF)
====================================
Adds cross-cutting responsibilities around the inference engine WITHOUT
modifying the classifier's base implementation.

Problem Solved:
  The system needs logging and metrics collection during inference, but mixing
  those concerns with the mathematical model logic would violate SRP.

Application:
  - LoggingInferencerDecorator: records execution details of each inference.
  - MetricsInferencerDecorator: collects performance metrics (timing, throughput).
  
  Both implement IModelInferencer and wrap another IModelInferencer, enabling
  transparent composition: LoggingDecorator → MetricsDecorator → BaseClassifier.

IMPORTANT DESIGN DECISION (Milestone 1 Correction #4):
  Decorator is used EXCLUSIVELY within the Inference Service for observability.
  Image preprocessing operations (validation, anonymization, normalization,
  resizing) are NOT decorators — they are pipeline stages implemented via
  Composition over Inheritance.

Related Requirements: RF-05, RNF-01, RNF-05
Related Principles:   SRP, OCP, Composition over Inheritance
"""

import time
from typing import Dict, List
from datetime import datetime

from domain.interfaces.contracts import IModelInferencer
from utils.logger import get_logger, get_audit_logger

logger = get_logger("Decorator")
audit_logger = get_audit_logger("InferenceAudit")


class LoggingInferencerDecorator(IModelInferencer):
    """
    Wraps an IModelInferencer to log execution details.

    Records: timestamp, classification result, confidence score, and
    any errors — without modifying the underlying inference logic.
    """

    def __init__(self, wrapped: IModelInferencer):
        self._wrapped = wrapped

    def predict(self, preprocessed_image, *args, **kwargs) -> Dict:
        """Execute prediction with full logging around the call."""
        timestamp = datetime.utcnow().isoformat()
        logger.info(f"[{timestamp}] Inference request received")

        try:
            result = self._wrapped.predict(preprocessed_image, *args, **kwargs)

            logger.info(
                f"[{timestamp}] Inference completed — "
                f"Classification: {result.get('classification', 'N/A')} | "
                f"Confidence: {result.get('confidence_score', 0.0):.4f}"
            )

            # Audit trail entry (immutable, append-only)
            audit_logger.info(
                f"INFERENCE_COMPLETED | "
                f"model={result.get('model_name', 'unknown')} | "
                f"version={result.get('model_version', 'unknown')} | "
                f"result={result.get('classification', 'N/A')} | "
                f"confidence={result.get('confidence_score', 0.0):.4f}"
            )

            result["logged_at"] = timestamp
            return result

        except Exception as e:
            logger.error(f"[{timestamp}] Inference FAILED: {str(e)}")
            audit_logger.error(f"INFERENCE_FAILED | error={str(e)}")
            raise


class MetricsInferencerDecorator(IModelInferencer):
    """
    Wraps an IModelInferencer to collect performance metrics.

    Tracks: processing time (ms), total inferences, average latency,
    and classification distribution — for monitoring and optimization.
    """

    def __init__(self, wrapped: IModelInferencer):
        self._wrapped = wrapped
        self._total_inferences: int = 0
        self._total_time_ms: float = 0.0
        self._classification_counts: Dict[str, int] = {}
        self._latency_history: List[float] = []

    def predict(self, preprocessed_image, *args, **kwargs) -> Dict:
        """Execute prediction with performance metrics collection."""
        start_time = time.perf_counter()

        result = self._wrapped.predict(preprocessed_image, *args, **kwargs)

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Update metrics
        self._total_inferences += 1
        self._total_time_ms += elapsed_ms
        self._latency_history.append(elapsed_ms)

        classification = result.get("classification", "UNKNOWN")
        self._classification_counts[classification] = (
            self._classification_counts.get(classification, 0) + 1
        )

        # Attach metrics to result
        result["metrics"] = {
            "total_processing_time_ms": round(elapsed_ms, 2),
            "inference_count": self._total_inferences,
            "average_latency_ms": round(
                self._total_time_ms / self._total_inferences, 2
            ),
        }

        logger.debug(
            f"Metrics — Latency: {elapsed_ms:.2f}ms | "
            f"Total inferences: {self._total_inferences} | "
            f"Avg: {self._total_time_ms / self._total_inferences:.2f}ms"
        )

        return result

    def get_metrics_summary(self) -> Dict:
        """Return a summary of collected performance metrics."""
        avg_latency = (
            self._total_time_ms / self._total_inferences
            if self._total_inferences > 0
            else 0.0
        )
        return {
            "total_inferences": self._total_inferences,
            "total_processing_time_ms": round(self._total_time_ms, 2),
            "average_latency_ms": round(avg_latency, 2),
            "min_latency_ms": round(min(self._latency_history), 2)
            if self._latency_history
            else 0.0,
            "max_latency_ms": round(max(self._latency_history), 2)
            if self._latency_history
            else 0.0,
            "classification_distribution": self._classification_counts.copy(),
        }
