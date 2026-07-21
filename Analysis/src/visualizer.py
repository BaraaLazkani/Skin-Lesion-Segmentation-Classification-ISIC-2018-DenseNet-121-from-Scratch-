from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

METRIC_COLS = ["accuracy", "precision", "recall", "specificity", "f1", "dice", "jaccard", "auc_roc"]
METRIC_LABELS = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "Dice", "Jaccard", "AUC-ROC"]

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)


def _save(fig, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")



def plot_training_curves(history: pd.DataFrame, output_dir: str):
    out = Path(output_dir)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(history["epoch"], history["train_loss"], label="Train Loss", linewidth=2)
    ax.plot(history["epoch"], history["val_loss"], label="Val Loss", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training & Validation Loss")
    ax.legend()
    _save(fig, out / "training_loss_curves.png")

    metric_curves = {
        "macro_f1": "Macro F1",
        "macro_accuracy": "Macro Accuracy",
        "macro_auc_roc": "Macro AUC-ROC",
        "macro_recall": "Macro Recall",
    }
    available = {k: v for k, v in metric_curves.items() if k in history.columns}
    if available:
        fig, ax = plt.subplots(figsize=(9, 5))
        for col, label in available.items():
            ax.plot(history["epoch"], history[col], label=label, linewidth=2)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Score")
        ax.set_title("Validation Metrics Over Training")
        ax.set_ylim(0, 1)
        ax.legend()
        _save(fig, out / "training_metric_curves.png")



def plot_per_class_metrics_bar(best_metrics: dict, class_names: List[str], output_dir: str):
    out = Path(output_dir)
    pc = best_metrics["per_class"]

    df = pd.DataFrame(
        {name: [pc[name][m] for m in METRIC_COLS] for name in class_names},
        index=METRIC_LABELS,
    ).T  # rows=classes, cols=metrics

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    colors = sns.color_palette("muted", len(class_names))

    for i, (metric_col, metric_label) in enumerate(zip(METRIC_COLS, METRIC_LABELS)):
        ax = axes[i]
        vals = [pc[name][metric_col] for name in class_names]
        bars = ax.bar(class_names, vals, color=colors)
        ax.set_title(metric_label)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=30, ha="right")
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Per-Class Metrics at Best Epoch", fontsize=14, fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "per_class_bars.png")



def plot_confusion_matrix(cm: np.ndarray, class_names: List[str], output_dir: str):
    out = Path(output_dir)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (counts)")
    _save(fig, out / "confusion_matrix_counts.png")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix (normalized)")
    _save(fig, out / "confusion_matrix_normalized.png")



def plot_per_class_metric_comparison(best_metrics: dict, class_names: List[str], output_dir: str):
    out = Path(output_dir)
    pc = best_metrics["per_class"]

    fig, axes = plt.subplots(4, 2, figsize=(14, 22))
    axes = axes.flatten()

    for i, (metric_col, metric_label) in enumerate(zip(METRIC_COLS, METRIC_LABELS)):
        ax = axes[i]
        vals = [pc[name][metric_col] for name in class_names]
        best_idx = int(np.argmax(vals))
        worst_idx = int(np.argmin(vals))

        palette = ["#aec6cf"] * len(class_names)
        palette[best_idx] = "#2ecc71"
        palette[worst_idx] = "#e74c3c"

        bars = ax.bar(class_names, vals, color=palette)
        ax.set_title(metric_label, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_xticklabels(class_names, rotation=30, ha="right", fontsize=9)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    fig.suptitle("Per-Class Performance by Metric\n(green = best, red = worst)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save(fig, out / "per_metric_comparison.png")
