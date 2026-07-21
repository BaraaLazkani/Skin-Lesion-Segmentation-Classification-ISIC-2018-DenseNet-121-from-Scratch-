import sys
import argparse
import os
from pathlib import Path

HERE = Path(__file__).parent.resolve()
os.chdir(HERE)
sys.path.insert(0, str(HERE / "src"))

from utils import load_config, set_seed, setup_logger, get_device, AdamOptimizer
from dataset import build_dataloaders
from model import build_model
from losses import build_loss
from trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train DenseNet-121 on ISIC 2018")
    parser.add_argument("--config", default="configs/config.yaml", help="Path to YAML config file")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(HERE / args.config)

    set_seed(cfg.seed)
    logger = setup_logger(cfg.logging.log_dir)
    device = get_device(cfg, logger)

    logger.info(f"Config: {args.config} | Seed: {cfg.seed} | Device: {device}")

    train_loader, val_loader, class_counts = build_dataloaders(cfg, logger)
    class_names = cfg.data.classes

    model = build_model(cfg, num_classes=len(class_names))
    logger.info(f"Model: DenseNet-121 (pretrained=False, dropout={cfg.model.dropout})")

    loss_fn = build_loss(cfg, class_counts, device)
    logger.info(f"Loss: {cfg.training.loss} (gamma={cfg.training.focal_gamma})")

    freeze_epochs = getattr(cfg.training, "freeze_backbone_epochs", 0)
    if freeze_epochs > 0:
        head_params = [p for n, p in model.named_parameters() if "classifier" in n]
        optimizer = AdamOptimizer(
            head_params,
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )
    else:
        optimizer = AdamOptimizer(
            model.parameters(),
            lr=cfg.training.learning_rate,
            weight_decay=cfg.training.weight_decay,
        )

    scheduler = None

    logger.info(
        f"Phase 1: freeze_epochs={freeze_epochs}, lr={cfg.training.learning_rate} | "
        f"Phase 2: lr={cfg.training.learning_rate_finetune}, scheduler=cosine"
    )

    trainer = Trainer(
        cfg=cfg,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        logger=logger,
        class_names=class_names,
        device=device,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
