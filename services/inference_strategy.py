"""
Strategy Pattern (Behavioral GoF) — Scoliosis Spine Classification
===================================================================
Encapsulates interchangeable classification algorithms behind a common interface.

ScoliosisStrategy implements IMLStrategy for spinal radiographs:
  - 3-class CNN classification (MobileNetV2 Transfer Learning):
      0: NO ESCOLIOSIS
      1: ESCOLIOSIS GRAVE (EN C)
      2: ESCOLIOSIS SEVERA (EN S)
  - Compiled @tf.function forward pass for low-latency CPU execution.
  - Geometric vertebral curve tracking (T1-L5) to estimate affected levels
    and raise the mandatory clinical review flag (requires_review) whenever
    confidence < CONFIDENCE_THRESHOLD (0.60) or geometry disagrees with the CNN.
  - Strictly pixel-driven inference: filenames and external annotation files
    are never consulted at inference time.

Related Requirements: RF-05, RF-06, RF-12, RNF-01, RNF-05
Related Principles:   OCP, LSP, DIP
"""

import threading
import time
from typing import Dict, Optional
import numpy as np

from domain.interfaces.contracts import IMLStrategy
from config.settings import CLASS_LABELS, CONFIDENCE_THRESHOLD, TARGET_IMAGE_SIZE
from ai.transfer_learning import estimate_vertebrae_from_image
from utils.logger import get_logger
from utils.exceptions import ModelNotLoadedError

logger = get_logger("Strategy")


class ScoliosisStrategy(IMLStrategy):
    """
    Concrete strategy for spinal radiograph classification and vertebral analysis
    using Transfer Learning (MobileNetV2 base) with a compiled tf.function pass.
    """

    def __init__(self):
        self._model = None
        self._forward = None
        self._lock = threading.RLock()
        self._model_info = {
            "name": "ScoliosisClassifier",
            "base_architecture": "MobileNetV2",
            "technique": "Transfer Learning (Fine-Tuning)",
            "num_classes": str(len(CLASS_LABELS)),
            "input_shape": f"{TARGET_IMAGE_SIZE[0]}x{TARGET_IMAGE_SIZE[1]}x3",
            "classes": list(CLASS_LABELS.values()),
        }

    def load_model(self, model_path: str) -> None:
        """Load a trained TensorFlow/Keras model from disk and compile its forward pass."""
        try:
            import tensorflow as tf
            self._model = tf.keras.models.load_model(model_path, compile=False)

            @tf.function(reduce_retracing=True)
            def _compiled_forward(x):
                return self._model(x, training=False)

            self._forward = _compiled_forward
            # Warm-up trace with a dummy tensor
            dummy = tf.zeros((1, TARGET_IMAGE_SIZE[0], TARGET_IMAGE_SIZE[1], 3), dtype=tf.float32)
            _ = self._forward(dummy)
            logger.info(f"Scoliosis model loaded and compiled with tf.function from: {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model from {model_path}: {e}")
            raise

    def classify(
        self,
        preprocessed_image: np.ndarray,
        filename: str = "",
        original_image: Optional[np.ndarray] = None,
    ) -> Dict:
        """
        Classify a spinal X-ray strictly from its pixels using the CNN softmax
        and compute the geometric vertebral review flag.

        Note: `filename` is accepted only for interface compatibility / logging
        and has zero influence on the diagnosis or probabilities.
        """
        start_time = time.perf_counter()

        if self._model is None and self._forward is None:
            raise ModelNotLoadedError(
                "No approved Keras model is loaded in ScoliosisStrategy."
            )

        tensor = np.asarray(preprocessed_image, dtype=np.float32)
        if not np.all(np.isfinite(tensor)):
            tensor = np.nan_to_num(tensor, nan=0.0, posinf=1.0, neginf=0.0).astype(np.float32)
        if tensor.ndim == 3:
            tensor = np.expand_dims(tensor, axis=0)

        # Step 1: Compiled CNN forward pass -> true softmax probabilities (thread-safe under flood)
        with self._lock:
            if self._forward is not None:
                raw_out = self._forward(tensor)
                probabilities = np.asarray(raw_out.numpy() if hasattr(raw_out, "numpy") else raw_out, dtype=np.float32)[0]
            else:
                probabilities = np.asarray(self._model.predict(tensor, verbose=0)[0], dtype=np.float32)

        # Ensure valid probability distribution
        prob_sum = float(np.sum(probabilities))
        if prob_sum <= 0 or not np.all(np.isfinite(probabilities)):
            probabilities = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=np.float32)
        elif abs(prob_sum - 1.0) > 1e-3:
            probabilities = probabilities / prob_sum

        predicted_class = int(np.argmax(probabilities))
        confidence_score = float(probabilities[predicted_class])
        classification = CLASS_LABELS.get(predicted_class, "NO ESCOLIOSIS")

        # Step 2: Geometric vertebral curve estimation from pixels (never overrides CNN diagnosis)
        geometry_source = original_image if original_image is not None else tensor[0]
        geometry = estimate_vertebrae_from_image(geometry_source)

        geom_curve_type = geometry.get("curve_type", "")
        if "Normal" in geom_curve_type:
            geometry_class = 0
        elif "S-Shaped" in geom_curve_type or "en S" in geom_curve_type:
            geometry_class = 2
        else:
            geometry_class = 1

        geometry_agrees = (geom_curve_type != "Indeterminado") and (geometry_class == predicted_class)
        low_confidence = confidence_score < CONFIDENCE_THRESHOLD
        requires_review = bool(low_confidence or not geometry_agrees)

        # Align displayed anatomical summary with the clinical state while preserving geometric note
        if predicted_class == 0 and geometry_agrees:
            affected_vertebrae = "None"
            curve_type = "Normal Physiological Alignment"
            apex_vertebra = "N/A"
        else:
            affected_vertebrae = geometry.get("affected_vertebrae", "T7 - T11")
            if predicted_class == 0 and affected_vertebrae == "None":
                affected_vertebrae = "None"
            curve_type = geometry.get("curve_type", "C-Shaped Curve (Simple)")
            apex_vertebra = geometry.get("apex_vertebra", "N/A")

        curve_detail = f"Geometric estimate: {geometry.get('curve_detail', '')}"
        if requires_review:
            reasons = []
            if low_confidence:
                reasons.append(f"confidence {confidence_score:.1%} < {CONFIDENCE_THRESHOLD:.0%}")
            if not geometry_agrees:
                reasons.append("CNN and geometric estimate disagree")
            curve_detail += f" [Mandatory specialist review: {', '.join(reasons)}.]"

        processing_time_ms = (time.perf_counter() - start_time) * 1000.0

        logger.info(
            f"Classification: {classification} | "
            f"Confidence: {confidence_score:.2%} | "
            f"RequiresReview: {requires_review} | "
            f"Time: {processing_time_ms:.1f}ms"
        )

        return {
            "classification": classification,
            "confidence_score": round(confidence_score, 4),
            "predicted_class": predicted_class,
            "affected_vertebrae": affected_vertebrae,
            "curve_type": curve_type,
            "curve_detail": curve_detail.strip(),
            "apex_vertebra": apex_vertebra,
            "requires_review": requires_review,
            "low_confidence": low_confidence,
            "geometry_agrees": geometry_agrees,
            "processing_time_ms": round(processing_time_ms, 2),
            "all_probabilities": {
                CLASS_LABELS[i]: round(float(probabilities[i]), 4)
                for i in range(len(CLASS_LABELS))
            },
        }

    def get_model_info(self) -> Dict[str, str]:
        """Return metadata about the current model."""
        info = self._model_info.copy()
        info["status"] = "loaded" if self._model is not None else "not_loaded"
        return info


class AlternativeBackboneStrategy(ScoliosisStrategy):
    """
    Interchangeable IMLStrategy implementation (e.g., ResNet50 / EfficientNetB0)
    demonstrating Open/Closed (OCP), Liskov Substitution (LSP), and Dependency
    Inversion (DIP) principles (Question 3): InferenceService can switch to this
    strategy via constructor injection without modifying a single line of service code.
    """

    def __init__(self, backbone_name: str = "ResNet50"):
        super().__init__()
        self._model_info["base_architecture"] = backbone_name


# Backwards compatibility alias
ChestXRayStrategy = ScoliosisStrategy
