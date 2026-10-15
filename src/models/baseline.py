"""Baseline patch classifier for weed vs crop discrimination.

Crops plant patches from annotated bounding boxes and trains a lightweight 3-layer CNN
to establish an upper-bound classification baseline assuming perfect plant localization.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.config import load_config
from src.data.parse_annotations import parse_bbox_csv
from src.data.splits import load_splits
from src.utils.seed import set_seed

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("baseline")


class PatchDataset(Dataset):
    """Dataset extracting cropped plant patches from images given ground-truth boxes."""

    def __init__(
        self,
        patch_records: List[Dict[str, Any]],
        image_size: int = 96,
        is_train: bool = False,
    ) -> None:
        self.records = patch_records
        self.image_size = image_size
        self.is_train = is_train

        if is_train:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.records[idx]
        image_path = item["image_path"]
        box = item["box"]  # [x1, y1, x2, y2]
        # Binary target: 0 for Crop (mapped label 1), 1 for Weed (mapped label 2)
        target = 0 if item["label"] == 1 else 1

        try:
            with Image.open(image_path) as full_img:
                img_rgb = full_img.convert("RGB")
                w, h = img_rgb.size
                x1, y1, x2, y2 = box

                # Add 5% padding around box
                bw, bh = x2 - x1, y2 - y1
                pad_x = bw * 0.05
                pad_y = bh * 0.05
                cx1 = max(0, int(x1 - pad_x))
                cy1 = max(0, int(y1 - pad_y))
                cx2 = min(w, int(x2 + pad_x))
                cy2 = min(h, int(y2 + pad_y))

                if cx2 <= cx1 or cy2 <= cy1:
                    dummy = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
                    return dummy, target

                crop = img_rgb.crop((cx1, cy1, cx2, cy2))
                tensor = self.transform(crop)
                return tensor, target
        except Exception as e:
            logger.warning(f"Error loading patch from {image_path}: {e}")
            dummy = torch.zeros((3, self.image_size, self.image_size), dtype=torch.float32)
            return dummy, target


class PatchCNN(nn.Module):
    """Lightweight 3-layer convolutional network for patch classification."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            # Block 1: 96x96 -> 48x48
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 2: 48x48 -> 24x24
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Block 3: 24x24 -> 12x12
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.features(x)
        return self.classifier(feat)


def extract_patch_records(
    splits_df: pd.DataFrame,
    images_dir: Path,
    bboxes_dir: Path,
    split_name: str,
    max_patches: int = 3000,
) -> List[Dict[str, Any]]:
    """Extract patch metadata records for a specified split."""
    sub_df = splits_df[splits_df["split"] == split_name]
    records = []

    for _, row in sub_df.iterrows():
        img_id = row["image_id"]
        img_path = images_dir / f"{img_id}.jpg"
        if not img_path.exists():
            continue

        csv_path = bboxes_dir / f"{img_id}.csv"
        boxes, labels, _ = parse_bbox_csv(csv_path)

        for box, label in zip(boxes, labels):
            records.append({
                "image_path": img_path,
                "box": box,
                "label": label,
            })
            if len(records) >= max_patches:
                break
        if len(records) >= max_patches:
            break

    logger.info(f"Extracted {len(records)} patches for split '{split_name}'.")
    return records


def train_baseline(config_path: str = "configs/default.yaml", epochs: int = 5, batch_size: int = 32) -> Dict[str, Any]:
    """Train and evaluate the baseline patch classifier."""
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training baseline patch classifier on device: {device}")

    raw_dir = Path(cfg.paths.raw_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    outputs_dir = Path(cfg.paths.outputs_dir)
    plots_dir = outputs_dir / "plots"
    reports_dir = outputs_dir / "reports"
    plots_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    images_dir = raw_dir / "images"
    variant = cfg.data.get("variant", "CropOrWeed2")
    bboxes_dir = raw_dir / "bboxes" / variant
    splits_path = processed_dir / "splits.csv"

    splits_df = load_splits(splits_path)

    # Extract patches from train and test splits
    train_records = extract_patch_records(splits_df, images_dir, bboxes_dir, "train", max_patches=4000)
    test_records = extract_patch_records(splits_df, images_dir, bboxes_dir, "test", max_patches=1000)

    train_ds = PatchDataset(train_records, image_size=96, is_train=True)
    test_ds = PatchDataset(test_records, image_size=96, is_train=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    model = PatchCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

    logger.info(f"Starting patch classifier training for {epochs} epochs...")
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

        epoch_loss = running_loss / max(1, total)
        epoch_acc = correct / max(1, total)
        logger.info(f"Epoch {epoch}/{epochs} - Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc * 100:.2f}%")

    # Evaluation on Test Split
    logger.info("Evaluating baseline patch classifier on test patches...")
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())

    all_preds_arr = np.array(all_preds)
    all_targets_arr = np.array(all_targets)

    # Metrics
    test_acc = float(accuracy_score(all_targets_arr, all_preds_arr))
    # Class 1 is Weed in this binary classifier
    weed_prec = float(precision_score(all_targets_arr, all_preds_arr, pos_label=1, zero_division=0))
    weed_rec = float(recall_score(all_targets_arr, all_preds_arr, pos_label=1, zero_division=0))
    weed_f1 = float(f1_score(all_targets_arr, all_preds_arr, pos_label=1, zero_division=0))

    # Trivial baseline: majority class predictor
    majority_class = int(np.bincount(all_targets_arr).argmax())
    majority_acc = float((all_targets_arr == majority_class).mean())

    cm = confusion_matrix(all_targets_arr, all_preds_arr)

    metrics = {
        "model": "PatchCNN_3Layer_Baseline",
        "description": "Upper-bound classification baseline assuming perfect bounding box crops",
        "test_patch_count": len(all_targets_arr),
        "test_accuracy": round(test_acc, 4),
        "majority_class_accuracy": round(majority_acc, 4),
        "weed_precision": round(weed_prec, 4),
        "weed_recall": round(weed_rec, 4),
        "weed_f1": round(weed_f1, 4),
        "confusion_matrix": cm.tolist(),
        "class_labels": ["Crop (0)", "Weed (1)"],
    }
    logger.info(f"Baseline Test Results: Accuracy={test_acc*100:.2f}% (Majority={majority_acc*100:.2f}%), "
                f"Weed Precision={weed_prec:.4f}, Recall={weed_rec:.4f}, F1={weed_f1:.4f}")

    # Save metrics JSON
    out_metrics_path = reports_dir / "baseline_metrics.json"
    with open(out_metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved baseline metrics to {out_metrics_path}")

    # Plot Confusion Matrix
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Crop", "Weed"], yticklabels=["Crop", "Weed"])
    plt.title("Baseline Patch Classifier Confusion Matrix", fontsize=12, fontweight="bold")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(plots_dir / "baseline_confusion_matrix.png", dpi=200)
    plt.close()
    logger.info(f"Saved baseline confusion matrix to {plots_dir / 'baseline_confusion_matrix.png'}")

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline patch classifier.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()
    train_baseline(args.config, epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()
