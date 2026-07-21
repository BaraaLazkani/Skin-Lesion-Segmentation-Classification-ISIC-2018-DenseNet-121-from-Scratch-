import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class FocalLoss:
    def __init__(self, gamma: float = 2.0, alpha=None, reduction: str = "mean"):
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, logits: np.ndarray, targets: np.ndarray):
        N = logits.shape[0]
        proba = _softmax(logits)
        proba = np.clip(proba, 1e-7, 1.0 - 1e-7)

        pt = proba[np.arange(N), targets]
        ce = -np.log(pt)
        focal_weight = (1.0 - pt) ** self.gamma

        loss = focal_weight * ce

        if self.alpha is not None:
            at = self.alpha[targets]
            loss = at * loss

        if self.reduction == "mean":
            return float(loss.mean()), proba
        if self.reduction == "sum":
            return float(loss.sum()), proba
        return loss, proba

    def backward(self, logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
        N = logits.shape[0]
        proba = _softmax(logits)
        proba = np.clip(proba, 1e-7, 1.0 - 1e-7)

        pt = proba[np.arange(N), targets]
        fw = (1.0 - pt) ** self.gamma

        one_hot = np.zeros_like(proba)
        one_hot[np.arange(N), targets] = 1.0

        dlogits = fw[:, np.newaxis] * (proba - one_hot)

        gamma_correction = (
            self.gamma * (1.0 - pt) ** (self.gamma - 1.0) * pt
        )
        dlogits[np.arange(N), targets] -= gamma_correction * np.log(pt + 1e-7)

        if self.alpha is not None:
            at = self.alpha[targets]
            dlogits *= at[:, np.newaxis]

        if self.reduction == "mean":
            dlogits /= N
        return dlogits


def build_loss(cfg, class_counts: list, device=None) -> FocalLoss:
    num_classes = len(class_counts)
    total = sum(class_counts)

    use_alpha = not cfg.training.use_weighted_sampler
    alpha = None
    if use_alpha and cfg.training.loss in ("focal", "weighted_ce"):
        weights = [total / (num_classes * max(c, 1)) for c in class_counts]
        alpha = np.array(weights, dtype=np.float32)

    if cfg.training.loss == "focal":
        return FocalLoss(gamma=cfg.training.focal_gamma, alpha=alpha)

    if cfg.training.loss == "weighted_ce":
        return FocalLoss(gamma=0.0, alpha=alpha)

    raise ValueError(f"Unknown loss: {cfg.training.loss}")
