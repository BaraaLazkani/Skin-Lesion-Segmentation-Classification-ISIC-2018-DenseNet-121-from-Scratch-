import os
import csv
import json
import logging
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from losses import _softmax
from metrics import compute_all_metrics, format_metrics_table
from model import DenseNet121, freeze_backbone, unfreeze_backbone
from utils import AdamOptimizer, CosineAnnealingScheduler


def _parse_monitor(monitor_key: str) -> str:
    for prefix in ("val_macro_", "macro_", "val_"):
        if monitor_key.startswith(prefix):
            return monitor_key[len(prefix):]
    return monitor_key


class Trainer:
    def __init__(
        self,
        cfg,
        model: DenseNet121,
        train_loader,
        val_loader,
        loss_fn,
        optimizer: AdamOptimizer,
        scheduler,
        logger: logging.Logger,
        class_names: List[str],
        device=None,
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.logger = logger
        self.class_names = class_names

        self.log_dir = Path(cfg.logging.log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.metrics_csv_path = self.log_dir / "metrics.csv"
        self._csv_fieldnames = None

        self.best_metric_value = -float("inf")
        self.best_epoch = 0
        self.epochs_without_improvement = 0

        monitor_cfg = cfg.training.early_stopping.monitor
        self.monitor_key = monitor_cfg
        self._monitor_metric = _parse_monitor(monitor_cfg)
        valid_metrics = ("accuracy", "precision", "recall", "specificity",
                         "f1", "dice", "jaccard", "auc_roc")
        if self._monitor_metric not in valid_metrics:
            raise ValueError(
                f"Unknown monitor metric '{self._monitor_metric}' (from '{monitor_cfg}'). "
                f"Valid: {valid_metrics}"
            )

        self._phase = 1

    def _switch_to_phase2(self, epoch: int):
        trainable = unfreeze_backbone(self.model)
        total = self.model.num_parameters()
        self.logger.info("=" * 60)
        self.logger.info(f"Phase 2 — full fine-tuning from epoch {epoch}")
        self.logger.info(f"  Trainable params: {trainable:,} / {total:,}")
        self.logger.info(f"  LR: {self.cfg.training.learning_rate_finetune}")
        self.logger.info("=" * 60)

        self.optimizer = AdamOptimizer(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate_finetune,
            weight_decay=self.cfg.training.weight_decay,
        )
        remaining = self.cfg.training.epochs - epoch + 1
        self.scheduler = CosineAnnealingScheduler(
            self.optimizer, T_max=max(remaining, 1), eta_min=1e-7
        )
        self._phase = 2

    def train_one_epoch(self, epoch: int) -> float:
        self.model.train()
        total_loss = 0.0
        log_every = self.cfg.logging.log_every_n_batches

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch:03d} [train|ph{self._phase}]",
            leave=False,
            dynamic_ncols=True,
        )
        for batch_idx, (images, labels) in enumerate(pbar):
            logits = self.model.forward(images)
            loss_val, _ = self.loss_fn.forward(logits, labels)

            dlogits = self.loss_fn.backward(logits, labels)
            self.model.backward(dlogits)

            if self._phase == 1 and self.model._backbone_frozen:
                grads = self.model.gradients(classifier_only=True)
                params = self.model.parameters(classifier_only=True)
                self.optimizer.params = params
                if len(self.optimizer._m) != len(params):
                    self.optimizer._m = [np.zeros_like(p, dtype=np.float64) for p in params]
                    self.optimizer._v = [np.zeros_like(p, dtype=np.float64) for p in params]
            else:
                grads = self.model.gradients()

            self.optimizer.step(grads)

            total_loss += loss_val
            current_lr = self.optimizer.lr
            pbar.set_postfix(loss=f"{loss_val:.4f}", lr=f"{current_lr:.6f}")

            if (batch_idx + 1) % log_every == 0:
                self.logger.debug(
                    f"Epoch {epoch} | Batch {batch_idx+1}/{len(self.train_loader)} | "
                    f"loss={loss_val:.4f} | lr={current_lr:.6f}"
                )

        return total_loss / len(self.train_loader)

    def validate(self, epoch: int) -> dict:
        self.model.eval()
        total_loss = 0.0
        all_true, all_pred, all_proba = [], [], []

        pbar = tqdm(
            self.val_loader,
            desc=f"Epoch {epoch:03d} [val]  ",
            leave=False,
            dynamic_ncols=True,
        )
        for images, labels in pbar:
            logits = self.model.forward(images)
            loss_val, _ = self.loss_fn.forward(logits, labels)
            total_loss += loss_val

            proba = _softmax(logits)
            preds = proba.argmax(axis=1)

            all_true.append(labels)
            all_pred.append(preds)
            all_proba.append(proba)

        self.model.train()

        y_true  = np.concatenate(all_true)
        y_pred  = np.concatenate(all_pred)
        y_proba = np.concatenate(all_proba)

        avg_loss = total_loss / len(self.val_loader)
        metrics = compute_all_metrics(y_true, y_pred, y_proba, self.class_names)
        metrics["val_loss"] = round(avg_loss, 6)

        table = format_metrics_table(metrics, self.class_names)
        self.logger.info(f"\nEpoch {epoch} — Validation metrics:\n{table}")

        return metrics

    def _write_metrics_csv(self, epoch: int, train_loss: float, val_metrics: dict):
        macro = val_metrics["macro"]
        pc    = val_metrics["per_class"]
        row = {
            "epoch":      epoch,
            "phase":      self._phase,
            "train_loss": round(train_loss, 6),
            "val_loss":   val_metrics["val_loss"],
        }
        for k, v in macro.items():
            row[f"macro_{k}"] = v
        for cls in self.class_names:
            for k, v in pc[cls].items():
                row[f"{cls}_{k}"] = v

        if self._csv_fieldnames is None:
            self._csv_fieldnames = list(row.keys())
            with open(self.metrics_csv_path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=self._csv_fieldnames).writeheader()

        with open(self.metrics_csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self._csv_fieldnames).writerow(row)

    def _atomic_save(self, obj, path: Path, mode: str = "npy"):
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix
        with tempfile.NamedTemporaryFile(dir=path.parent, suffix=suffix, delete=False) as tmp:
            tmp_path = tmp.name
        try:
            if mode == "npy":
                np.save(tmp_path, obj)
            elif mode == "json":
                with open(tmp_path, "w") as f:
                    json.dump(obj, f, indent=2)
            os.replace(tmp_path, path)
        except Exception:
            os.unlink(tmp_path)
            raise

    def _save_best_artifacts(self, epoch: int, metrics: dict):
        cm = np.array(metrics["confusion_matrix"])
        self._atomic_save(cm, self.log_dir / "best_confusion_matrix.npy", mode="npy")
        payload = {k: v for k, v in metrics.items() if k != "confusion_matrix"}
        payload["best_epoch"] = epoch
        self._atomic_save(payload, self.log_dir / "best_metrics.json", mode="json")

    def fit(self):
        es_cfg = self.cfg.training.early_stopping
        freeze_epochs = getattr(self.cfg.training, "freeze_backbone_epochs", 0)

        if freeze_epochs > 0:
            trainable, total = freeze_backbone(self.model)
            self.logger.info("=" * 60)
            self.logger.info("Phase 1 — backbone frozen, training classifier head only")
            self.logger.info(f"  Trainable params: {trainable:,} / {total:,}")
            self.logger.info(f"  LR: {self.cfg.training.learning_rate} for {freeze_epochs} epochs")
            self.logger.info("=" * 60)
        else:
            self._phase = 2

        self.logger.info(f"  Classes:  {self.class_names}")
        self.logger.info(f"  Monitor:  {self.monitor_key} ('{self._monitor_metric}')")
        self.logger.info(f"  Patience: {es_cfg.patience}")

        last_epoch = 1
        for epoch in range(1, self.cfg.training.epochs + 1):
            last_epoch = epoch

            if self._phase == 1 and epoch > freeze_epochs:
                self._switch_to_phase2(epoch)
                self.best_metric_value = -float("inf")
                self.epochs_without_improvement = 0

            train_loss  = self.train_one_epoch(epoch)
            val_metrics = self.validate(epoch)

            self._write_metrics_csv(epoch, train_loss, val_metrics)

            monitored_value = val_metrics["macro"][self._monitor_metric]
            self.logger.info(
                f"Epoch {epoch:03d} [ph{self._phase}] | train_loss={train_loss:.4f} | "
                f"val_loss={val_metrics['val_loss']:.4f} | {self.monitor_key}={monitored_value:.4f}"
            )

            if monitored_value > self.best_metric_value + es_cfg.min_delta:
                self.best_metric_value = monitored_value
                self.best_epoch = epoch
                self.epochs_without_improvement = 0
                self._save_best_artifacts(epoch, val_metrics)
                self.logger.info(f"  -> New best {self.monitor_key}={monitored_value:.4f}.")
            else:
                if self._phase == 2:
                    self.epochs_without_improvement += 1
                self.logger.info(
                    f"  -> No improvement for {self.epochs_without_improvement}/{es_cfg.patience} epochs."
                )

            if self._phase == 2 and self.scheduler is not None:
                self.scheduler.step()

            if self._phase == 2 and self.epochs_without_improvement >= es_cfg.patience:
                self.logger.info(f"Early stopping triggered at epoch {epoch}.")
                break

        self.logger.info(
            f"Saving final-epoch artifacts (epoch {last_epoch}, best was epoch {self.best_epoch})."
        )
        final_metrics = self.validate(last_epoch)
        cm = np.array(final_metrics["confusion_matrix"])
        payload = {k: v for k, v in final_metrics.items() if k != "confusion_matrix"}
        payload["final_epoch"] = last_epoch
        payload["best_epoch"]  = self.best_epoch
        self._atomic_save(cm, self.log_dir / "final_confusion_matrix.npy", mode="npy")
        self._atomic_save(payload, self.log_dir / "final_metrics.json", mode="json")

        self.logger.info("=" * 60)
        self.logger.info(
            f"Training complete. Best {self.monitor_key}={self.best_metric_value:.4f} "
            f"at epoch {self.best_epoch}"
        )
        self.logger.info("=" * 60)
