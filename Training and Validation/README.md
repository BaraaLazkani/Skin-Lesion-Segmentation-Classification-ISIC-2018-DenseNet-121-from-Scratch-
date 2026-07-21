# Phase 2 — Training and Validation (from scratch)

This is the original contribution of the project: a full **DenseNet-121**, trained on the ISIC 2018 Task 3 seven-class skin lesion classification benchmark, implemented **entirely in NumPy** — no PyTorch, TensorFlow, Keras, or JAX anywhere in the model, optimizer, loss, or data pipeline.

That means everything below was written by hand:

- A full DenseNet-121 forward **and** backward pass
- Convolution via `im2col` / `col2im`
- Batch Normalization, Dropout, Max/Average Pooling, Linear layers — each with its own analytic backward pass
- An Adam optimizer with bias correction
- Focal Loss with an analytic gradient
- Cosine-annealing LR scheduling and early stopping
- A weighted random sampler for class imbalance
- A minimal `DataLoader` (batching, shuffling, weighted sampling) built on plain NumPy/Pandas/Pillow, no `torch.utils.data`

## Layout

```
Training and Validation/
├── train.py                 # entry point
├── configs/config.yaml       # every hyperparameter, one file
├── src/
│   ├── dataset.py            # CSV loading, augmentation, DataLoader
│   ├── model.py               # DenseNet-121, built layer by layer
│   ├── losses.py              # Focal Loss (forward + backward)
│   ├── metrics.py              # per-class + macro metrics
│   ├── trainer.py               # two-phase training loop
│   └── utils.py                 # Adam, schedulers, config/seed/logging helpers
└── logs/                       # written by train.py
    ├── metrics.csv               # per-epoch train/val metrics
    ├── train.log                  # full DEBUG-level log
    ├── best_metrics.json           # full metrics snapshot at the best epoch
    ├── best_confusion_matrix.npy    # confusion matrix at the best epoch
    ├── final_metrics.json            # metrics snapshot at the last epoch run
    └── final_confusion_matrix.npy     # confusion matrix at the last epoch run
```

## Data pipeline — `src/dataset.py`

- **`load_and_clean_csv`** reads a ground-truth CSV, drops any row whose image file is missing on disk, and derives an integer `label` column from the one-hot class columns via `argmax`.
- **`ImageTransform`** is a callable that resizes to `image_size` (224, bilinear), optionally rotates (±30°, PIL-side, before the array conversion), converts to a `[0,1]` float array, then optionally applies horizontal flip (p=0.5), vertical flip (p=0.5), and colour jitter (brightness/contrast/saturation, factor 0.2) — all hand-written NumPy, no `torchvision`. It finishes with ImageNet mean/std normalisation and an `HWC → CHW` transpose.
- **`DataLoader`** batches a `SkinLesionDataset`. When `use_weighted_sampler` is enabled it draws indices **with replacement** each epoch, weighted by `1 / count(class)` — over-sampling the rare classes (DF, VASC) and under-sampling the dominant one (NV) without ever discarding data.

## Model — `src/model.py`

Built bottom-up from primitives:

- **`_im2col` / `_col2im`** — the core trick that turns convolution into a single matrix multiply, using `as_strided` views (no explicit copy until the reshape) for the forward `im2col`, and pure NumPy scatter-add for `col2im` in the backward pass.
- **`Conv2D`, `BatchNorm2D`, `ReLU`, `Dropout`, `MaxPool2D`, `AvgPool2D`, `Linear`** — each a `Layer` with `forward`, `backward`, `parameters()`, and `gradients()`. `BatchNorm2D` tracks running mean/var for eval mode and derives its backward pass analytically from the batch-norm chain rule. `MaxPool2D`/`AvgPool2D` reuse the same `im2col` machinery as convolution.
- **`DenseLayer`** — the bottleneck (`BN → ReLU → 1×1 conv → BN → ReLU → 3×3 conv`) that DenseNet concatenates onto its input, so growth is additive in channels rather than replacing the input.
- **`DenseBlock`** — stacks `num_layers` `DenseLayer`s, each seeing every previous layer's output via concatenation.
- **`TransitionLayer`** — `BN → ReLU → 1×1 conv (compression=0.5) → 2×2 avg-pool` between dense blocks, halving channel count and spatial size.
- **`DenseNet121`** — assembles the standard `(6, 12, 24, 16)` block configuration with growth rate 32 and 64 initial features, followed by global average pooling, dropout, and a linear classifier head. Supports `freeze_backbone()` / `unfreeze_backbone()` for the two-phase schedule below, and `save()`/`load()` via `np.savez` with named parameters (so weights can be inspected or reloaded independent of any framework).

## Loss — `src/losses.py`

`FocalLoss` implements both the forward pass and a fully analytic backward pass (no autograd):

```
loss = -alpha_t * (1 - p_t)^gamma * log(p_t)
```

`build_loss` optionally derives per-class `alpha` weights (`total / (num_classes * count)`) — but only when the weighted *sampler* is off, since combining both would double-correct for imbalance.

## Metrics — `src/metrics.py`

`compute_all_metrics` computes, per class and as a macro average: accuracy, precision, recall, specificity, F1, Dice, Jaccard, and AUC-ROC (via a hand-rolled trapezoidal ROC curve in `_roc_auc_binary` — no `sklearn`).

## Trainer — `src/trainer.py`

Two-phase training loop:

- **Phase 1** (`freeze_backbone_epochs` epochs): only the classifier head trains, at a higher LR (`1e-3`). This lets the randomly-initialised head adapt before gradients start flowing into the frozen backbone.
- **Phase 2** (remaining epochs): the full network unfreezes and fine-tunes at a much lower LR (`3e-5`) under cosine annealing. Early stopping (patience-based, monitoring `val_macro_f1`) and the best-metric bookkeeping both reset when phase 2 begins.

Every epoch's metrics are appended to `logs/metrics.csv`; whenever the monitored metric improves, `logs/best_metrics.json` and `logs/best_confusion_matrix.npy` are overwritten atomically (write-to-temp-then-`os.replace`, so a crash mid-write never corrupts the previous best).

## Utilities — `src/utils.py`

- **`AdamOptimizer`** — bias-corrected Adam, hand-written update rule, supports per-call `params`/`grads` lists so the optimizer can be pointed at just the classifier head in phase 1.
- **`CosineAnnealingScheduler`** / **`ReduceOnPlateauScheduler`** — LR schedules; only cosine is used by the shipped config.
- **`load_config`** / **`set_seed`** / **`setup_logger`** / **`get_device`** — YAML config → `SimpleNamespace`, RNG seeding, dual file+console logging, and a device string (always `"cpu"` here — this is a NumPy model, there is no CUDA path).

## Configuration — `configs/config.yaml`

| Parameter | Value | Why |
|---|---|---|
| Architecture | DenseNet-121 | Built from scratch |
| Input size | 224×224 | Resized from the 256×256 Phase 1 output |
| Dropout | 0.6 | Heavy regularisation — ~7M params on <10k images |
| Batch size | 32 | |
| Phase-1 epochs | 8 | Frozen backbone, head only |
| Phase-1 LR | 1e-3 | Safe at this LR — only ~1,024 head params update |
| Phase-2 LR | 3e-5 | Very low — full fine-tune needs a gentle touch |
| Scheduler | Cosine annealing | Phase 2 only |
| Loss | Focal, γ=2.0 | Targets class imbalance |
| Weight decay | 1e-3 | L2 |
| Weighted sampler | Enabled | Over-samples DF/VASC, under-samples NV |
| Early stopping | patience=12, monitors `val_macro_f1` | |
| Seed | 42 | |

## Running it

```bash
cd "Training and Validation"
python train.py --config configs/config.yaml
```

Outputs land in `logs/`, which is exactly what [`../Analysis`](../Analysis/README.md) reads to produce tables and plots — no re-inference needed.

## Result achieved on this run

Best epoch: **34**. Macro F1: **0.9052**. Macro AUC-ROC: **0.9729**. Full breakdown, plots, and discussion live in [`../Analysis/README.md`](../Analysis/README.md).
