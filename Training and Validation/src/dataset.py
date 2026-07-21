import os
import math
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


CLASS_COLUMNS = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]


def load_and_clean_csv(
    csv_path: str, images_dir: str, image_suffix: str, logger=None
) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    original_len = len(df)

    def _exists(image_id):
        return os.path.isfile(os.path.join(images_dir, f"{image_id}{image_suffix}"))

    mask = df["image"].apply(_exists)
    df = df[mask].reset_index(drop=True)
    dropped = original_len - len(df)

    msg = (
        f"CSV: {os.path.basename(csv_path)} | "
        f"original rows: {original_len} | "
        f"dropped (image missing): {dropped} | "
        f"remaining: {len(df)}"
    )
    if logger:
        logger.info(msg)
    else:
        print(msg)

    missing = [c for c in CLASS_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected class columns: {missing}")
    df["label"] = df[CLASS_COLUMNS].values.argmax(axis=1).astype("int64")
    return df


class ImageTransform:
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, size: int, augment: bool = False, augmentation_cfg=None):
        self.size = size
        self.augment = augment
        self.aug = augmentation_cfg

    def _hflip(self, img: np.ndarray, p: float) -> np.ndarray:
        return img[:, ::-1, :].copy() if np.random.rand() < p else img

    def _vflip(self, img: np.ndarray, p: float) -> np.ndarray:
        return img[::-1, :, :].copy() if np.random.rand() < p else img

    def _rotate(self, pil_img: Image.Image, max_deg: float) -> Image.Image:
        angle = float(np.random.uniform(-max_deg, max_deg))
        return pil_img.rotate(angle, resample=Image.BILINEAR, expand=False)

    def _color_jitter(self, img: np.ndarray, factor: float) -> np.ndarray:
        img = img.astype(np.float32)
        b = float(np.random.uniform(1.0 - factor, 1.0 + factor))
        img = np.clip(img * b, 0.0, 1.0)
        c = float(np.random.uniform(1.0 - factor, 1.0 + factor))
        mean = img.mean(axis=(0, 1), keepdims=True)
        img = np.clip((img - mean) * c + mean, 0.0, 1.0)
        s = float(np.random.uniform(1.0 - factor, 1.0 + factor))
        gray = img.mean(axis=2, keepdims=True)
        img = np.clip(gray + s * (img - gray), 0.0, 1.0)
        return img

    def __call__(self, pil_img: Image.Image) -> np.ndarray:
        pil_img = pil_img.resize((self.size, self.size), Image.BILINEAR)

        if self.augment and self.aug is not None:
            pil_img = self._rotate(pil_img, self.aug.rotation_deg)

        img = np.array(pil_img, dtype=np.float32) / 255.0

        if self.augment and self.aug is not None:
            img = self._hflip(img, self.aug.horizontal_flip)
            img = self._vflip(img, self.aug.vertical_flip)
            img = self._color_jitter(img, self.aug.color_jitter)

        img = (img - self.MEAN) / self.STD
        return img.transpose(2, 0, 1).astype(np.float32)


class SkinLesionDataset:
    def __init__(self, df: pd.DataFrame, images_dir: str,
                 transform: ImageTransform, image_suffix: str):
        self.df = df.reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform
        self.image_suffix = image_suffix

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.images_dir, f"{row['image']}{self.image_suffix}")
        image = Image.open(img_path).convert("RGB")
        return self.transform(image), int(row["label"])


class DataLoader:
    def __init__(self, dataset: SkinLesionDataset, batch_size: int,
                 shuffle: bool = False, sample_weights: np.ndarray = None,
                 drop_last: bool = False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.sample_weights = sample_weights
        self.drop_last = drop_last

    def __len__(self) -> int:
        n = len(self.dataset)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)

    def __iter__(self):
        n = len(self.dataset)

        if self.sample_weights is not None:
            probs = self.sample_weights / self.sample_weights.sum()
            indices = np.random.choice(n, size=n, replace=True, p=probs)
        elif self.shuffle:
            indices = np.random.permutation(n)
        else:
            indices = np.arange(n)

        num_batches = n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)
        for b in range(num_batches):
            batch_idx = indices[b * self.batch_size: (b + 1) * self.batch_size]
            images, labels = [], []
            for idx in batch_idx:
                img, label = self.dataset[int(idx)]
                images.append(img)
                labels.append(label)
            yield np.stack(images, axis=0), np.array(labels, dtype=np.int64)


def build_dataloaders(cfg, logger=None):
    train_suffix = cfg.data.train_image_suffix
    val_suffix   = cfg.data.val_image_suffix

    train_df = load_and_clean_csv(
        cfg.data.train_csv, cfg.data.train_images_dir, train_suffix, logger
    )
    val_df = load_and_clean_csv(
        cfg.data.val_csv, cfg.data.val_images_dir, val_suffix, logger
    )

    aug_cfg = cfg.training.augmentation
    train_transform = ImageTransform(cfg.data.image_size, augment=True, augmentation_cfg=aug_cfg)
    val_transform   = ImageTransform(cfg.data.image_size, augment=False)

    train_dataset = SkinLesionDataset(
        train_df, cfg.data.train_images_dir, train_transform, train_suffix
    )
    val_dataset = SkinLesionDataset(
        val_df, cfg.data.val_images_dir, val_transform, val_suffix
    )

    num_classes = len(CLASS_COLUMNS)
    class_counts = [int((train_df["label"] == i).sum()) for i in range(num_classes)]

    sample_weights = None
    if cfg.training.use_weighted_sampler:
        weights_per_class = [1.0 / max(c, 1) for c in class_counts]
        sample_weights = np.array(
            [weights_per_class[int(label)] for label in train_df["label"]], dtype=np.float64
        )
        if logger:
            counts_str = {cfg.data.classes[i]: class_counts[i] for i in range(num_classes)}
            logger.info(f"Weighted sampling enabled. Class counts: {counts_str}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=(sample_weights is None),
        sample_weights=sample_weights,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=False,
    )

    if logger:
        logger.info(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    return train_loader, val_loader, class_counts
