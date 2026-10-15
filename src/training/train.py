"""Training pipeline for Weed vs Crop Faster R-CNN object detector."""

import argparse
import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import load_config
from src.data.dataset import WeedCropDetectionDataset, collate_fn
from src.evaluation.metrics import evaluate_model_map
from src.models.detector import build_detector, freeze_backbone, save_checkpoint, unfreeze_backbone
from src.utils.logging import setup_logger
from src.utils.seed import set_seed


def train(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    """Execute training pipeline according to specification."""
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    # Directories
    raw_dir = Path(cfg.paths.raw_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    models_dir = Path(cfg.paths.models_dir)
    outputs_dir = Path(cfg.paths.outputs_dir)
    reports_dir = outputs_dir / "reports"

    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    log_file = reports_dir / "train.log"
    logger = setup_logger("train", log_file=log_file)
    logger.info(f"Loaded configuration from {config_path}")

    # Device
    device_str = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    logger.info(f"Training on device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # Datasets and Loaders
    images_dir = raw_dir / "images"
    variant = cfg.data.get("variant", "CropOrWeed2")
    bboxes_dir = raw_dir / "bboxes" / variant
    splits_path = processed_dir / "splits.csv"

    subset_fraction = float(cfg.data.get("subset_fraction", 1.0))
    max_samples = cfg.data.get("max_samples", None)

    train_dataset = WeedCropDetectionDataset(
        images_dir=images_dir,
        bboxes_dir=bboxes_dir,
        splits_path=splits_path,
        split="train",
        subset_fraction=subset_fraction,
        max_samples=max_samples,
        is_train=True,
    )
    val_dataset = WeedCropDetectionDataset(
        images_dir=images_dir,
        bboxes_dir=bboxes_dir,
        splits_path=splits_path,
        split="val",
        subset_fraction=subset_fraction,
        max_samples=max(1, max_samples // 2) if max_samples else None,
        is_train=False,
    )

    batch_size = int(cfg.train.get("batch_size", 4))
    num_workers = int(cfg.train.get("num_workers", 0))

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    # Model
    model = build_detector(
        num_classes=int(cfg.model.get("num_classes", 3)),
        min_size=int(cfg.model.get("min_size", 640)),
        max_size=int(cfg.model.get("max_size", 1024)),
        pretrained=bool(cfg.model.get("pretrained", True)),
    )
    model.to(device)

    # Optimizer & Scheduler
    lr = float(cfg.train.get("lr", 0.005))
    momentum = float(cfg.train.get("momentum", 0.9))
    weight_decay = float(cfg.train.get("weight_decay", 0.0005))

    optimizer_name = cfg.train.get("optimizer", "sgd").lower()
    if optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    epochs = int(cfg.train.get("epochs", 20))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    freeze_epochs = int(cfg.train.get("freeze_backbone_epochs", 0))
    if freeze_epochs > 0:
        freeze_backbone(model)

    patience = int(cfg.train.get("early_stopping_patience", 5))
    best_map = -1.0
    patience_counter = 0

    train_log_path = reports_dir / "train_log.csv"
    with open(train_log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_mAP", "val_mAP_50", "val_crop_AP", "val_weed_AP", "lr", "epoch_time_sec"])

    logger.info(f"Beginning training for {epochs} epochs (Early stopping patience: {patience})...")
    start_total_time = time.time()
    epoch = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Handle freezing schedule
        if freeze_epochs > 0 and epoch == freeze_epochs + 1:
            unfreeze_backbone(model)
            # Re-initialize optimizer param groups if needed
            logger.info("Unfrozen backbone for fine-tuning.")

        model.train()
        running_loss = 0.0
        total_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch:2d}/{epochs}", unit="batch", leave=True)
        for images, targets in pbar:
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

            running_loss += losses.item()
            total_batches += 1
            pbar.set_postfix({
                "loss": f"{losses.item():.4f}",
                "avg_loss": f"{running_loss / total_batches:.4f}"
            })

        scheduler.step()
        epoch_loss = running_loss / max(1, total_batches)

        # Validation evaluation
        print(f" -> Running validation evaluation for Epoch {epoch}...", flush=True)
        val_metrics = evaluate_model_map(model, val_loader, device)
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch}/{epochs} [{epoch_time:.1f}s] - Train Loss: {epoch_loss:.4f} | "
            f"Val mAP: {val_metrics['mAP']:.4f} | Val mAP@50: {val_metrics['mAP_50']:.4f} | "
            f"Crop AP: {val_metrics['crop_AP']:.4f} | Weed AP: {val_metrics['weed_AP']:.4f}"
        )

        # Append to log CSV
        with open(train_log_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch,
                round(epoch_loss, 4),
                val_metrics["mAP"],
                val_metrics["mAP_50"],
                val_metrics["crop_AP"],
                val_metrics["weed_AP"],
                round(current_lr, 6),
                round(epoch_time, 2),
            ])

        # Save last checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics={"train_loss": epoch_loss, **val_metrics},
            config_dict=dict(cfg),
            checkpoint_path=models_dir / "last.pt",
        )

        # Check for best model
        current_map = val_metrics["mAP"]
        if current_map > best_map:
            best_map = current_map
            patience_counter = 0
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                metrics={"train_loss": epoch_loss, **val_metrics},
                config_dict=dict(cfg),
                checkpoint_path=models_dir / "best.pt",
            )
            logger.info(f"--> New best model saved to {models_dir / 'best.pt'} with mAP={best_map:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs (no improvement for {patience} epochs).")
                break

    total_duration = time.time() - start_total_time
    logger.info(f"Training completed in {total_duration / 60:.2f} minutes. Best Val mAP: {best_map:.4f}")

    return {"best_mAP": best_map, "total_epochs": epoch, "duration_minutes": total_duration / 60}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Weed vs Crop Faster R-CNN detector.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
