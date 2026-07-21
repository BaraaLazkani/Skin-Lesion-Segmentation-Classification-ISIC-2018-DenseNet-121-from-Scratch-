import os
import random
import logging
import yaml
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def load_config(path: str) -> SimpleNamespace:
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _dict_to_namespace(raw)


def _dict_to_namespace(d):
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_namespace(i) for i in d]
    return d


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)


def setup_logger(log_dir: str, name: str = "train") -> logging.Logger:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.handlers:
        logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    fh = logging.FileHandler(os.path.join(log_dir, f"{name}.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def get_device(cfg, logger=None) -> str:
    device = "cpu"
    msg = "Running on CPU (pure NumPy implementation)."
    if logger:
        logger.info(msg)
    else:
        print(msg)
    return device



class AdamOptimizer:
    def __init__(self, params, lr: float = 0.001, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 0.0):
        self.params = list(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self._m = [np.zeros_like(p, dtype=np.float64) for p in self.params]
        self._v = [np.zeros_like(p, dtype=np.float64) for p in self.params]
        self._t = 0
        self.param_groups = [{"lr": lr}]

    def step(self, grads):
        self._t += 1
        lr_t = (self.lr
                * np.sqrt(1.0 - self.beta2 ** self._t)
                / (1.0 - self.beta1 ** self._t))

        for i, (param, grad) in enumerate(zip(self.params, grads)):
            if grad is None:
                continue
            g = grad.astype(np.float64)
            if self.weight_decay != 0.0:
                g = g + self.weight_decay * param

            self._m[i] = self.beta1 * self._m[i] + (1.0 - self.beta1) * g
            self._v[i] = self.beta2 * self._v[i] + (1.0 - self.beta2) * g * g
            param -= (lr_t * self._m[i] / (np.sqrt(self._v[i]) + self.eps)).astype(param.dtype)

        self.param_groups[0]["lr"] = self.lr

    def zero_grad(self):
        pass  # gradients are recomputed each step via model.backward()

    def get_lr(self) -> float:
        return self.lr



class CosineAnnealingScheduler:
    def __init__(self, optimizer: AdamOptimizer, T_max: int, eta_min: float = 1e-7):
        self.optimizer = optimizer
        self.T_max = max(T_max, 1)
        self.eta_min = eta_min
        self._base_lr = optimizer.lr
        self._step_count = 0

    def step(self):
        self._step_count += 1
        t = self._step_count
        T = self.T_max
        new_lr = self.eta_min + 0.5 * (self._base_lr - self.eta_min) * (
            1.0 + np.cos(np.pi * t / T)
        )
        self.optimizer.lr = float(new_lr)
        self.optimizer.param_groups[0]["lr"] = self.optimizer.lr

    def get_last_lr(self):
        return [self.optimizer.lr]


class ReduceOnPlateauScheduler:
    def __init__(self, optimizer: AdamOptimizer, mode: str = "max",
                 factor: float = 0.5, patience: int = 5, min_lr: float = 1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self._best = -float("inf") if mode == "max" else float("inf")
        self._wait = 0

    def step(self, metric: float = None):
        if metric is None:
            return
        improved = (
            (self.mode == "max" and metric > self._best) or
            (self.mode == "min" and metric < self._best)
        )
        if improved:
            self._best = metric
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self.patience:
                new_lr = max(self.optimizer.lr * self.factor, self.min_lr)
                self.optimizer.lr = new_lr
                self.optimizer.param_groups[0]["lr"] = new_lr
                self._wait = 0

    def get_last_lr(self):
        return [self.optimizer.lr]


def build_scheduler(cfg, optimizer: AdamOptimizer):
    t = cfg.training
    if t.scheduler == "cosine":
        return CosineAnnealingScheduler(
            optimizer,
            T_max=t.epochs - getattr(t, "warmup_epochs", 0),
            eta_min=1e-7,
        )
    if t.scheduler == "reduce_on_plateau":
        return ReduceOnPlateauScheduler(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6
        )
    raise ValueError(f"Unknown scheduler: {t.scheduler}")
