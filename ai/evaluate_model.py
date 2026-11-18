"""
AI Model Validation & Error Analysis Script
============================================
Evaluates the trained Transfer Learning Scoliosis Classifier on the held-out test split.
Measures strictly the deployed CNN softmax path (no label look-ups or data leakage).
Calculates Accuracy, Balanced Accuracy, Macro/Weighted F1, Cohen's Kappa, Specificity,
Confusion Matrix, Majority-Class Baseline comparison, Confidence Calibration, and
Quality Gate verification for Milestone 3 (Criterion 3, Questions 1, 2, 9 & 10).
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    SCOLIOSIS_DATASET_DIR,
    MODEL_DIR,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_VERSION,
    CLASS_LABELS,
)
from ai.transfer_learning import TransferLearningTrainer, analyze_spine_curve
from ai.metrics import compute_classification_metrics
from utils.logger import setup_logger, get_logger

setup_logger()
logger = get_logger("ModelEvaluation")


def run_evaluation(
    version: Optional[str] = None,
    dataset_dir: Optional[Path] = None,
    output_path: Optional[Path] = None,
):
    target_version = version or DEFAULT_MODEL_VERSION
    ds_root = dataset_dir or SCOLIOSIS_DATASET_DIR

    logger.info("=" * 70)
    logger.info(f"  AI Model Validation & Error Analysis — {DEFAULT_MODEL_NAME} v{target_version}")
    logger.info("=" * 70)

    model_dir = MODEL_DIR / f"{DEFAULT_MODEL_NAME}_v{target_version}"
    model_path = model_dir / "model.keras"
    if not model_path.exists():
        logger.error(f"Model file not found at {model_path}")
        return None

    import tensorflow as tf
    logger.info(f"Loading trained model from {model_path}...")
    model = tf.keras.models.load_model(str(model_path), compile=False)

    trainer = TransferLearningTrainer()
    logger.info(f"Loading test split from {ds_root}...")
    test_imgs, test_lbls, test_filenames = trainer._load_split_with_names(ds_root, "test")

    logger.info(f"Test samples loaded: {len(test_imgs)}")
    if len(test_imgs) == 0:
        logger.error("No test samples found.")
        return None

    # 1. Pure CNN Softmax Evaluation (exact same path as InferenceService)
    raw_probs = np.asarray(model.predict(test_imgs, batch_size=32, verbose=0), dtype=np.float32)
    pred_classes = np.argmax(raw_probs, axis=-1).astype(np.int32)
    pred_confidences = np.max(raw_probs, axis=-1).astype(np.float32)

    metrics = compute_classification_metrics(
        y_true=test_lbls,
        y_pred=pred_classes,
        y_prob=raw_probs,
    )

    # Collect failure cases with true CNN probabilities
    failure_cases = []
    error_breakdown = {}
    for idx in range(len(test_lbls)):
        true_cls = int(test_lbls[idx])
        pred_cls = int(pred_classes[idx])
        conf = float(pred_confidences[idx])

        if true_cls != pred_cls:
            pair_key = f"{CLASS_LABELS[true_cls]} -> {CLASS_LABELS[pred_cls]}"
            error_breakdown[pair_key] = error_breakdown.get(pair_key, 0) + 1
            img_name = test_filenames[idx] if idx < len(test_filenames) else f"sample_{idx:03d}.jpg"
            failure_cases.append({
                "file_name": img_name,
                "true_class_id": true_cls,
                "true_class_name": CLASS_LABELS[true_cls],
                "pred_class_id": pred_cls,
                "pred_class_name": CLASS_LABELS[pred_cls],
                "confidence": round(conf, 4),
                "probabilities": {
                    CLASS_LABELS[c]: round(float(raw_probs[idx][c]), 4) for c in range(3)
                },
            })

    # Separate reference: annotation-geometry rule (documented separately, NOT used as CNN result)
    lbl_dir = ds_root / "labels" / "test"
    label_files = sorted(lbl_dir.glob("*.txt"), key=lambda p: p.name)
    geom_preds = []
    for idx, lbl_path in enumerate(label_files):
        if idx >= len(test_lbls):
            break
        lines = lbl_path.read_text().strip().splitlines()
        v_boxes = []
        for line in lines:
            parts = line.strip().split()
            if parts and int(parts[0]) == 0:
                v_boxes.append(tuple(map(float, parts[1:5])))
        geom_name, _, _ = analyze_spine_curve(v_boxes, spine_class=None)
        if "NO ESCOLIOSIS" in geom_name:
            geom_preds.append(0)
        elif "EN C" in geom_name:
            geom_preds.append(1)
        else:
            geom_preds.append(2)

    geom_acc = float(np.mean(np.array(geom_preds) == test_lbls)) if geom_preds else 0.0

    results = {
        "model_name": DEFAULT_MODEL_NAME,
        "model_version": target_version,
        "split": "test",
        "evaluated_component": "CNN softmax output (same path as the Inference Service)",
        **metrics,
        "error_breakdown": error_breakdown,
        "total_errors": len(failure_cases),
        "error_rate": round(float(len(failure_cases) / len(test_lbls)), 4),
        "failure_cases": failure_cases,
        "annotation_geometry_reference": {
            "note": "Reference accuracy of the geometric rule on expert bounding boxes; NOT the deployed CNN model.",
            "accuracy": round(geom_acc, 4),
        },
    }

    out_file = output_path or (model_dir / "evaluation_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    logger.info(f"\n[+] Validation Complete. Saved to: {out_file}")
    logger.info(f"Overall Accuracy:   {results['overall_accuracy'] * 100:.2f}%")
    logger.info(f"Balanced Accuracy:  {results['balanced_accuracy'] * 100:.2f}%")
    logger.info(f"Macro Precision:    {results['macro_precision'] * 100:.2f}%")
    logger.info(f"Macro Recall:       {results['macro_recall'] * 100:.2f}%")
    logger.info(f"Macro F1-Score:     {results['macro_f1'] * 100:.2f}%")
    logger.info(f"Quality Gate Pass:  {results['quality_gate']['approved']}")
    logger.info(f"Confusion Matrix (3x3):\n{np.array(results['confusion_matrix'])}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained Scoliosis Classifier model")
    parser.add_argument("--version", type=str, default=DEFAULT_MODEL_VERSION, help="Model version to evaluate")
    parser.add_argument("--dataset", type=str, default=None, help="Optional dataset path")
    args = parser.parse_args()
    ds = Path(args.dataset) if args.dataset else None
    run_evaluation(version=args.version, dataset_dir=ds)
