"""
Unit Tests: AI Validation Metrics & Quality Gate (Question 1 & Question 10)
===========================================================================
Tests:
  - Majority-class baseline comparison on the 101-image test distribution (33 Normal, 45 C, 23 S)
  - Balanced accuracy vs standard accuracy when a model collapses to predicting C-curve (Question 1)
  - Per-class sensitivity (recall), specificity, and confusion matrix computation
  - Automatic Quality Gate enforcement (`passes_quality_gate`)
"""

import numpy as np
from ai.metrics import (
    compute_majority_baseline,
    compute_classification_metrics,
    compute_calibration_summary,
    passes_quality_gate,
)


class TestMetricsUnit:
    """Unit tests for clinical validation metrics and quality gate."""

    def test_collapsed_c_curve_model_matches_majority_baseline_question_1(self):
        """
        Question 1 verification:
        On 101 test images (33 Normal, 45 C-curve, 23 S-curve), a model that predicts
        all 101 images as C-curve (class 1) obtains:
          - Accuracy = 45 / 101 = 44.55%
          - Balanced Accuracy = (0/33 + 45/45 + 0/23) / 3 = 33.33%
          - Macro-F1 = (0 + 2*45/(45+101) + 0) / 3 = 20.55%
        And does NOT outperform a trivial majority-class classifier.
        """
        y_true = np.array([0] * 33 + [1] * 45 + [2] * 23, dtype=np.int32)
        y_pred_collapsed = np.ones(101, dtype=np.int32)

        metrics = compute_classification_metrics(y_true, y_pred_collapsed)
        maj = compute_majority_baseline(y_true)

        assert metrics["total_samples"] == 101
        assert abs(metrics["accuracy"] - 0.4455) < 1e-3
        assert abs(metrics["balanced_accuracy"] - 0.3333) < 1e-3
        assert abs(metrics["macro_f1"] - 0.2055) < 1e-3
        assert metrics["predicts_single_class"] is True

        # Does NOT beat majority baseline
        assert metrics["accuracy"] == maj["accuracy"]
        assert metrics["quality_gate"]["approved"] is False
        assert len(metrics["quality_gate"]["reasons"]) >= 3

    def test_confusion_matrix_and_per_class_specificity(self):
        """Verify 3x3 confusion matrix, sensitivity, and specificity per class."""
        y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int32)
        y_pred = np.array([0, 0, 1, 1, 2, 1], dtype=np.int32)
        y_prob = np.array([
            [0.90, 0.05, 0.05],
            [0.85, 0.10, 0.05],
            [0.05, 0.88, 0.07],
            [0.04, 0.91, 0.05],
            [0.02, 0.08, 0.90],
            [0.10, 0.55, 0.35],
        ], dtype=np.float32)

        metrics = compute_classification_metrics(y_true, y_pred, y_prob)
        assert metrics["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 1, 1]]
        normal_stats = metrics["per_class_metrics"]["NO ESCOLIOSIS"]
        assert normal_stats["recall_sensitivity"] == 1.0
        assert normal_stats["specificity"] == 1.0

    def test_quality_gate_approves_strong_multiclass_model(self):
        """Verify passes_quality_gate approves a model meeting macro-F1 >= 0.70 and recall >= 0.60."""
        y_true = np.array([0] * 33 + [1] * 45 + [2] * 23, dtype=np.int32)
        # Construct predictions with >80% recall on every class
        y_pred = y_true.copy()
        y_pred[0] = 1
        y_pred[33] = 0
        y_pred[78] = 1

        metrics = compute_classification_metrics(y_true, y_pred)
        gate = passes_quality_gate(metrics)
        assert gate["approved"] is True
        assert gate["reasons"] == []

    def test_calibration_summary_ece_computation(self):
        """Verify Expected Calibration Error (ECE) and confidence bins."""
        y_true = np.array([0, 1, 2, 1], dtype=np.int32)
        y_pred = np.array([0, 1, 2, 0], dtype=np.int32)
        confs = np.array([0.92, 0.85, 0.78, 0.52], dtype=np.float32)

        cal = compute_calibration_summary(y_true, y_pred, confs, n_bins=5)
        assert 0.0 <= cal["ece"] <= 1.0
        assert len(cal["bins"]) == 5
