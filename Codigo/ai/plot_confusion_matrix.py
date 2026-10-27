"""
Confusion Matrix & Model Metrics Visualizer
============================================
Visualizes the AI evaluation results from evaluation_results.json.
Displays a 3x3 Confusion Matrix heatmap and per-class clinical metrics.
Saves the plot as 'Codigo/data/confusion_matrix.png' and optionally displays an interactive window.
"""

import sys
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# File paths
BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_JSON = BASE_DIR / "data" / "models" / "scoliosis_classifier_v1.0.0" / "evaluation_results.json"
OUTPUT_PNG = BASE_DIR / "data" / "confusion_matrix.png"


def plot_metrics(show_gui: bool = False):
    if not EVAL_JSON.exists():
        print(f"Error: Evaluation file not found at {EVAL_JSON}")
        print("Please run 'python ai/evaluate_model.py' first.")
        return

    with open(EVAL_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    cm = np.array(data["confusion_matrix"])
    labels = ["NO SCOLIOSIS", "SEVERE (C-CURVE)", "SEVERE (S-CURVE)"]
    per_class = data["per_class_metrics"]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(
        f"AI Model Evaluation: {data['model_name']} v{data['model_version']}\n"
        f"Test Samples: {data['total_test_samples']} | Accuracy: {data['overall_accuracy']*100:.2f}% | Macro F1-Score: {data['macro_f1']*100:.2f}%",
        fontsize=13,
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

    # Annotate numbers in confusion matrix
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            ax_cm.text(
                j,
                i,
                f"{val}\n({val/cm.sum()*100:.1f}%)",
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
        vals = [per_class[k][m_key] * 100 for k in ["NO ESCOLIOSIS", "ESCOLIOSIS GRAVE (EN C)", "ESCOLIOSIS SEVERA (EN S)"]]
        ax_bar.bar(x + (idx - 1.5) * width, vals, width, label=m_label)

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(labels, rotation=15, ha="right", fontsize=9, fontweight="bold")
    ax_bar.set_ylabel("Percentage (%)", fontsize=11, fontweight="bold")
    ax_bar.set_title("Clinical Metrics by Pathological Category", fontsize=12, pad=10)
    ax_bar.set_ylim(0, 115)
    ax_bar.legend(loc="upper right", fontsize=9)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=300)
    print(f"[+] Confusion matrix plot saved successfully to: {OUTPUT_PNG}")

    if show_gui:
        try:
            plt.show()
        except Exception:
            pass


if __name__ == "__main__":
    show_window = "--no-gui" not in sys.argv
    plot_metrics(show_gui=show_window)
