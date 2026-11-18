"""
AI Model Validation Metrics & Clinical Quality Gate
====================================================
Provides mathematical evaluation functions for multi-class scoliosis classification:
  - Accuracy, Balanced Accuracy, Macro/Weighted F1, Cohen's Kappa
  - Per-class Precision, Sensitivity (Recall), and Specificity (True Negative Rate)
  - Trivial Majority-Class Baseline comparison (Question 1)
  - Confidence Calibration & Expected Calibration Error (ECE)
  - Automatic Quality Gate enforcement (O4, NFR-08, Question 10)
"""

from typing import Any, Dict, List, Optional
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from config.settings import CLASS_LABELS, NUM_CLASSES


def compute_majority_baseline(y_true: np.ndarray) -> Dict[str, Any]:
    """
    Calculate the performance of a trivial classifier that always predicts
    the most frequent class in y_true (Question 1).
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    if len(y_true) == 0:
        return {
            "class_id": 1,
            "class": CLASS_LABELS.get(1, "SERIOUS SCOLIOSIS (C-CURVE)"),
            "accuracy": 0.0,
            "balanced_accuracy": 0.0,
            "f1_score": 0.0,
        }

    values, counts = np.unique(y_true, return_counts=True)
    maj_idx = int(values[np.argmax(counts)])
    y_maj = np.full_like(y_true, fill_value=maj_idx)

    acc = float(accuracy_score(y_true, y_maj))
    bal_acc = float(balanced_accuracy_score(y_true, y_maj))
    macro_f1 = float(f1_score(y_true, y_maj, labels=list(range(NUM_CLASSES)), average="macro", zero_division=0))

    return {
        "class_id": maj_idx,
        "class": CLASS_LABELS.get(maj_idx, str(maj_idx)),
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "f1_score": round(macro_f1, 4),
    }


def compute_calibration_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 5,
) -> Dict[str, Any]:
    """Compute confidence calibration bins and Expected Calibration Error (ECE)."""
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    confidences = np.asarray(confidences, dtype=np.float32)

    if len(y_true) == 0:
        return {"ece": 0.0, "bins": []}

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins_report = []
    ece = 0.0
    n = len(y_true)

    for i in range(n_bins):
        low, high = float(bin_boundaries[i]), float(bin_boundaries[i + 1])
        if i == 0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences > low) & (confidences <= high)

        count = int(np.sum(mask))
        if count > 0:
            bin_acc = float(np.mean(y_true[mask] == y_pred[mask]))
            bin_conf = float(np.mean(confidences[mask]))
            ece += (count / n) * abs(bin_acc - bin_conf)
        else:
            bin_acc = 0.0
            bin_conf = 0.0

        bins_report.append({
            "range": f"[{low:.2f}, {high:.2f}]",
            "count": count,
            "empirical_accuracy": round(bin_acc, 4),
            "mean_confidence": round(bin_conf, 4),
        })

    return {
        "ece": round(float(ece), 4),
        "bins": bins_report,
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Compute comprehensive multi-class clinical evaluation metrics.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)
    labels = list(range(NUM_CLASSES))

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    acc = float(accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    bal_acc = float(balanced_accuracy_score(y_true, y_pred)) if len(y_true) else 0.0
    macro_prec = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_rec = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))
    try:
        kappa = float(cohen_kappa_score(y_true, y_pred, labels=labels))
        if np.isnan(kappa):
            kappa = 0.0
    except Exception:
        kappa = 0.0

    per_class_metrics: Dict[str, Dict[str, Any]] = {}
    for i in labels:
        name = CLASS_LABELS[i]
        tp = int(cm[i, i])
        fn = int(np.sum(cm[i, :]) - tp)
        fp = int(np.sum(cm[:, i]) - tp)
        tn = int(np.sum(cm) - (tp + fn + fp))

        prec_i = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec_i = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        spec_i = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        f1_i = (2 * prec_i * rec_i / (prec_i + rec_i)) if (prec_i + rec_i) > 0 else 0.0

        per_class_metrics[name] = {
            "precision": round(float(prec_i), 4),
            "recall_sensitivity": round(float(rec_i), 4),
            "specificity": round(float(spec_i), 4),
            "f1_score": round(float(f1_i), 4),
            "support": int(tp + fn),
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "tn": tn,
        }

    unique_preds = np.unique(y_pred) if len(y_pred) > 0 else np.array([])
    predicts_single_class = bool(len(unique_preds) <= 1 and len(y_pred) > 1)
    majority_baseline = compute_majority_baseline(y_true)

    if y_prob is not None and len(y_prob) == len(y_true):
        confidences = np.max(np.asarray(y_prob, dtype=np.float32), axis=-1)
    else:
        confidences = np.ones_like(y_true, dtype=np.float32)

    calibration = compute_calibration_summary(y_true, y_pred, confidences)

    metrics = {
        "total_samples": int(len(y_true)),
        "total_test_samples": int(len(y_true)),
        "accuracy": round(acc, 4),
        "overall_accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(macro_prec, 4),
        "macro_precision": round(macro_prec, 4),
        "recall": round(macro_rec, 4),
        "macro_recall": round(macro_rec, 4),
        "f1_score": round(macro_f1, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "cohen_kappa": round(kappa, 4),
        "confusion_matrix": cm.tolist(),
        "predicts_single_class": predicts_single_class,
        "majority_class_baseline": majority_baseline,
        "per_class_metrics": per_class_metrics,
        "calibration": calibration,
    }
    gate = passes_quality_gate(metrics)
    metrics["quality_gate"] = gate
    return metrics


def passes_quality_gate(
    metrics: Dict[str, Any],
    min_macro_f1: float = 0.70,
    min_class_recall: float = 0.60,
) -> Dict[str, Any]:
    """
    Automatic clinical quality gate (O4, NFR-08, Question 10).
    Verifies that a candidate model:
      1. Does not collapse to predicting a single class.
      2. Strictly outperforms the majority-class baseline.
      3. Achieves macro-F1 >= min_macro_f1 (0.70).
      4. Achieves sensitivity (recall) >= min_class_recall (0.60) on EVERY class.
    """
    reasons: List[str] = []

    macro_f1 = float(metrics.get("macro_f1", metrics.get("f1_score", 0.0)))
    acc = float(metrics.get("accuracy", metrics.get("overall_accuracy", 0.0)))
    maj_acc = float(metrics.get("majority_class_baseline", {}).get("accuracy", 0.0))

    if metrics.get("predicts_single_class", False):
        reasons.append("Model collapses to predicting a single class for all test samples.")

    if acc <= maj_acc:
        reasons.append(
            f"Accuracy ({acc:.2%}) does not exceed majority-class baseline ({maj_acc:.2%})."
        )

    if macro_f1 < min_macro_f1:
        reasons.append(
            f"Macro-F1 ({macro_f1:.2%}) is below minimum threshold ({min_macro_f1:.2%})."
        )

    per_class = metrics.get("per_class_metrics", {})
    for cls_name, cls_stats in per_class.items():
        rec = float(cls_stats.get("recall_sensitivity", 0.0))
        if rec < min_class_recall:
            reasons.append(
                f"Class '{cls_name}' sensitivity ({rec:.2%}) is below minimum ({min_class_recall:.2%})."
            )

    return {
        "approved": len(reasons) == 0,
        "min_macro_f1": min_macro_f1,
        "min_class_recall": min_class_recall,
        "reasons": reasons,
    }
