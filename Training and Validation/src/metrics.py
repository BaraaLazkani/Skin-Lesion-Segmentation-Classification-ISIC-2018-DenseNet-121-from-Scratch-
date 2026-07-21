from typing import List, Dict
import numpy as np


def _confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true.astype(int), y_pred.astype(int)):
        cm[t, p] += 1
    return cm


def _roc_auc_binary(y_true_bin: np.ndarray, y_score: np.ndarray) -> float:
    P = int(y_true_bin.sum())
    N = len(y_true_bin) - P
    if P == 0 or N == 0:
        return float("nan")

    order = np.argsort(-y_score)
    y_sorted = y_true_bin[order]

    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)

    tpr = np.concatenate([[0.0], tps / P, [1.0]])
    fpr = np.concatenate([[0.0], fps / N, [1.0]])

    return float(np.trapz(tpr, fpr))


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
) -> Dict:
    num_classes = len(class_names)
    cm = _confusion_matrix(y_true, y_pred, num_classes)

    precision_arr = np.zeros(num_classes)
    recall_arr    = np.zeros(num_classes)
    f1_arr        = np.zeros(num_classes)
    specificity_arr = np.zeros(num_classes)
    dice_arr      = np.zeros(num_classes)
    jaccard_arr   = np.zeros(num_classes)
    acc_arr       = np.zeros(num_classes)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        tn = cm.sum() - tp - fp - fn

        precision_arr[i]   = tp / max(tp + fp, 1)
        recall_arr[i]      = tp / max(tp + fn, 1)
        f1_arr[i]          = 2 * tp / max(2 * tp + fp + fn, 1)
        specificity_arr[i] = tn / max(tn + fp, 1)
        dice_arr[i]        = 2 * tp / max(2 * tp + fp + fn, 1)
        jaccard_arr[i]     = tp / max(tp + fp + fn, 1)
        acc_arr[i]         = (tp + tn) / max(tp + tn + fp + fn, 1)

    overall_accuracy = float((y_true == y_pred).sum()) / len(y_true)

    y_true_onehot = np.zeros((len(y_true), num_classes), dtype=np.int32)
    for idx, label in enumerate(y_true.astype(int)):
        y_true_onehot[idx, label] = 1

    auc_arr = np.full(num_classes, np.nan)
    for i in range(num_classes):
        if y_true_onehot[:, i].sum() == 0:
            continue
        auc_arr[i] = _roc_auc_binary(y_true_onehot[:, i], y_proba[:, i])

    valid_aucs = auc_arr[~np.isnan(auc_arr)]
    macro_auc = float(np.mean(valid_aucs)) if len(valid_aucs) > 0 else 0.0
    auc_arr = np.where(np.isnan(auc_arr), 0.0, auc_arr)

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "accuracy":    round(float(acc_arr[i]),         4),
            "precision":   round(float(precision_arr[i]),   4),
            "recall":      round(float(recall_arr[i]),       4),
            "specificity": round(float(specificity_arr[i]), 4),
            "f1":          round(float(f1_arr[i]),           4),
            "dice":        round(float(dice_arr[i]),         4),
            "jaccard":     round(float(jaccard_arr[i]),      4),
            "auc_roc":     round(float(auc_arr[i]),          4),
        }

    macro = {
        "accuracy":    round(float(overall_accuracy),          4),
        "precision":   round(float(precision_arr.mean()),      4),
        "recall":      round(float(recall_arr.mean()),         4),
        "specificity": round(float(specificity_arr.mean()),    4),
        "f1":          round(float(f1_arr.mean()),             4),
        "dice":        round(float(dice_arr.mean()),           4),
        "jaccard":     round(float(jaccard_arr.mean()),        4),
        "auc_roc":     round(macro_auc,                        4),
    }

    return {
        "per_class": per_class,
        "macro": macro,
        "confusion_matrix": cm.tolist(),
    }


def format_metrics_table(metrics: Dict, class_names: List[str]) -> str:
    header = (
        f"{'Class':<10} {'Acc':>6} {'Prec':>6} {'Rec':>6} "
        f"{'Spec':>6} {'F1':>6} {'Dice':>6} {'Jaccard':>8} {'AUC':>6}"
    )
    sep = "-" * len(header)
    lines = [sep, header, sep]

    for name in class_names:
        m = metrics["per_class"][name]
        lines.append(
            f"{name:<10} {m['accuracy']:>6.4f} {m['precision']:>6.4f} {m['recall']:>6.4f} "
            f"{m['specificity']:>6.4f} {m['f1']:>6.4f} {m['dice']:>6.4f} "
            f"{m['jaccard']:>8.4f} {m['auc_roc']:>6.4f}"
        )

    lines.append(sep)
    m = metrics["macro"]
    lines.append(
        f"{'MACRO':<10} {m['accuracy']:>6.4f} {m['precision']:>6.4f} {m['recall']:>6.4f} "
        f"{m['specificity']:>6.4f} {m['f1']:>6.4f} {m['dice']:>6.4f} "
        f"{m['jaccard']:>8.4f} {m['auc_roc']:>6.4f}"
    )
    lines.append(sep)
    return "\n".join(lines)
