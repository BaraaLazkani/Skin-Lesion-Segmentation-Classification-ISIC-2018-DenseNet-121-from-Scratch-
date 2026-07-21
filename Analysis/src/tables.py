import os
import json
from pathlib import Path
from typing import List

import pandas as pd


METRIC_COLS = ["accuracy", "precision", "recall", "specificity", "f1", "dice", "jaccard", "auc_roc"]
METRIC_DISPLAY = ["Accuracy", "Precision", "Recall", "Specificity", "F1", "Dice", "Jaccard", "AUC-ROC"]


def build_per_class_table(best_metrics: dict, class_names: List[str], output_dir: str):
    pc = best_metrics["per_class"]
    rows = []
    for name in class_names:
        row = {"Class": name}
        for m in METRIC_COLS:
            row[m.capitalize() if m != "auc_roc" else "AUC-ROC"] = pc[name][m]
        rows.append(row)

    df = pd.DataFrame(rows).set_index("Class")
    df.columns = METRIC_DISPLAY

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df.to_csv(out / "per_class_metrics.csv")

    md_lines = ["## Per-Class Metrics (at Best Epoch)\n"]
    md_lines.append("| Class | " + " | ".join(METRIC_DISPLAY) + " |")
    md_lines.append("|" + "|".join(["---"] * (len(METRIC_DISPLAY) + 1)) + "|")
    for name in class_names:
        vals = [f"{pc[name][m]:.4f}" for m in METRIC_COLS]
        md_lines.append(f"| {name} | " + " | ".join(vals) + " |")

    with open(out / "per_class_metrics.md", "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print(f"Saved per_class_metrics.csv and .md to {out}")
    return df


def macro_summary_table(best_metrics: dict, output_dir: str):
    macro = best_metrics["macro"]
    row = {m.capitalize() if m != "auc_roc" else "AUC-ROC": macro[m] for m in METRIC_COLS}
    df = pd.DataFrame([row], index=["Macro"])

    out = Path(output_dir)
    df.to_csv(out / "macro_summary.csv")
    print(f"Saved macro_summary.csv to {out}")
    return df


def best_and_worst_classes(best_metrics: dict, class_names: List[str], output_dir: str, metric: str = "f1"):
    pc = best_metrics["per_class"]
    scores = {name: pc[name][metric] for name in class_names}
    best_cls = max(scores, key=scores.get)
    worst_cls = min(scores, key=scores.get)

    lines = [
        "=" * 50,
        "CLASS PERFORMANCE SUMMARY",
        "=" * 50,
        f"Metric used for ranking: {metric.upper()}",
        "",
        "Ranked classes (best to worst):",
    ]
    for rank, (cls, score) in enumerate(sorted(scores.items(), key=lambda x: -x[1]), 1):
        marker = " <- BEST" if cls == best_cls else (" <- WORST" if cls == worst_cls else "")
        lines.append(f"  {rank}. {cls:<8} {metric.upper()}={score:.4f}{marker}")

    lines += [
        "",
        f"Best class:  {best_cls} ({metric.upper()}={scores[best_cls]:.4f})",
        f"Worst class: {worst_cls} ({metric.upper()}={scores[worst_cls]:.4f})",
        "",
        "All macro averages:",
    ]
    for m in METRIC_COLS:
        lines.append(f"  {m:<15} {best_metrics['macro'][m]:.4f}")
    lines.append("=" * 50)

    out = Path(output_dir)
    with open(out / "summary.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("\n".join(lines))
    return best_cls, worst_cls


def epoch_history_table(metrics_csv_path: str, output_dir: str):
    df = pd.read_csv(metrics_csv_path)
    out = Path(output_dir)
    df.to_csv(out / "epoch_history.csv", index=False)
    print(f"Saved epoch_history.csv ({len(df)} epochs) to {out}")
    return df
