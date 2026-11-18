"""
Confusion Matrix & Model Metrics Visualizer
============================================
Visualizes the AI evaluation results from evaluation_results.json.
Displays a 3x3 Confusion Matrix heatmap and per-class clinical metrics.
Saves the plot as 'Codigo/data/confusion_matrix.png' and inside the model version folder.
"""

import argparse
import sys
import json
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import MODEL_DIR, DEFAULT_MODEL_NAME, DEFAULT_MODEL_VERSION, DATA_DIR


def plot_metrics(version: Optional[str] = None, show_gui: bool = False):
    target_version = version or DEFAULT_MODEL_VERSION
    version_dir = MODEL_DIR / f"{DEFAULT_MODEL_NAME}_v{target_version}"
    eval_json = version_dir / "evaluation_results.json"
    output_png = DATA_DIR / "confusion_matrix.png"
    version_png = version_dir / "confusion_matrix.png"

    if not eval_json.exists():
        print(f"Error: Evaluation file not found at {eval_json}")
        print(f"Please run 'python ai/evaluate_model.py --version {target_version}' first.")
        return

    with open(eval_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"])
    labels = ["NO SCOLIOSIS", "SEVERE (C-CURVE)", "SEVERE (S-CURVE)"]
    per_class = data["per_class_metrics"]
    acc = data.get("overall_accuracy", data.get("accuracy", 0.0))
    bal_acc = data.get("balanced_accuracy", acc)
    macro_f1 = data.get("macro_f1", data.get("f1_score", 0.0))
    total_samples = data.get("total_test_samples", data.get("total_samples", int(cm.sum())))

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"AI Model Evaluation: {data['model_name']} v{data['model_version']} (CNN Softmax)\n"
        f"Test Samples: {total_samples} | Accuracy: {acc*100:.2f}% | "
        f"Balanced Accuracy: {bal_acc*100:.2f}% | Macro F1-Score: {macro_f1*100:.2f}%",
        fontsize=12.5,
        fontweight="bold",
    )

    # 1. Heatmap Confusion Matrix
    ax_cm = axes[0]
    im = ax_cm.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax_cm, fraction=0.046, pad=0.04)

    ax_cm.set_xticks(np.arange(len(labels)))
    ax_cm.set_yticks(np.arange(len(labels)))
    ax_cm.set_xticklabels(labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax_cm.set_yticklabels(labels, fontsize=9, fontweight="bold")
    ax_cm.set_xlabel("Predicted Class", fontsize=11, fontweight="bold", labelpad=8)
    ax_cm.set_ylabel("Ground Truth Class", fontsize=11, fontweight="bold", labelpad=8)
    ax_cm.set_title("Multi-Class Confusion Matrix (3x3)", fontsize=12, pad=10)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 1.0
    total_cm = max(int(cm.sum()), 1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax_cm.text(
                j,
                i,
                f"{val}\n({val/total_cm*100:.1f}%)",
                ha="center",
                va="center",
                color="white" if val > thresh else "black",
                fontweight="bold",
                fontsize=11,
            )

    # 2. Bar chart of metrics per class
    ax_bar = axes[1]
    metrics = ["precision", "recall_sensitivity", "specificity", "f1_score"]
    metric_names = ["Precision", "Sensitivity", "Specificity", "F1-Score"]
    x = np.arange(len(labels))
    width = 0.2

    for idx, (m_key, m_label) in enumerate(zip(metrics, metric_names)):
        vals = [
            per_class[k][m_key] * 100
            for k in ["NO ESCOLIOSIS", "ESCOLIOSIS GRAVE (EN C)", "ESCOLIOSIS SEVERA (EN S)"]
        ]
        ax_bar.bar(x + (idx - 1.5) * width, vals, width, label=m_label)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax_bar.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax_bar.set_title("Clinical Metrics by Pathological Category", fontsize=12, pad=10)
    ax_bar.set_ylim(0, 115)
    ax_bar.legend(loc="upper right", fontsize=9)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.savefig(version_png, dpi=300)
    plt.close(fig)
    print(f"[+] Confusion matrix plot saved successfully to: {output_png} and {version_png}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=DEFAULT_MODEL_VERSION)
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()
    plot_metrics(version=args.version, show_gui=not args.no_gui)
