"""PyTorch Dataset and transforms for CropAndWeed object detection."""

import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset

from src.data.parse_annotations import parse_bbox_csv
from src.data.splits import load_splits

logger = logging.getLogger(__name__)


class WeedCropDetectionDataset(Dataset):
    """PyTorch Dataset for object detection on field photographs."""

    def __init__(
        self,
        images_dir: Path | str,
        bboxes_dir: Path | str,
        splits_path: Path | str,
        split: str = "train",
        subset_fraction: float = 1.0,
        max_samples: Optional[int] = None,
        is_train: bool = False,
    ) -> None:
        """Initialize WeedCropDetectionDataset.

        Args:
            images_dir: Directory containing image files.
            bboxes_dir: Directory containing bbox CSVs for the active variant.
            splits_path: Path to splits.csv.
            split: 'train', 'val', or 'test'.
            subset_fraction: Fraction of split images to use (0.0 to 1.0).
            max_samples: Optional hard ceiling on sample count (for smoke testing).
            is_train: Whether to apply training data augmentation.
        """
        self.images_dir = Path(images_dir)
        self.bboxes_dir = Path(bboxes_dir)
        self.split = split
        self.is_train = is_train

        splits_df = load_splits(splits_path)
        sub_df = splits_df[splits_df["split"] == split].reset_index(drop=True)

        if subset_fraction < 1.0:
            sample_size = max(1, int(len(sub_df) * subset_fraction))
            sub_df = sub_df.iloc[:sample_size].reset_index(drop=True)

        if max_samples is not None and max_samples > 0:
            sub_df = sub_df.iloc[:max_samples].reset_index(drop=True)

        self.samples: List[Dict[str, Any]] = []
        for _, row in sub_df.iterrows():
            img_id = row["image_id"]
            img_path = self.images_dir / f"{img_id}.jpg"
            if not img_path.exists():
                img_path = self.images_dir / f"{img_id}.png"
            if img_path.exists():
                self.samples.append({
                    "image_id": img_id,
                    "image_path": img_path,
                    "csv_path": self.bboxes_dir / f"{img_id}.csv",
                    "session": row["session"],
                })

        logger.info(f"Loaded {len(self.samples)} samples for '{split}' split (subset_fraction={subset_fraction}).")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        sample = self.samples[idx]
        image_path = sample["image_path"]
        csv_path = sample["csv_path"]

        with Image.open(image_path) as raw_img:
            img = raw_img.convert("RGB")
        w, h = img.size

        boxes_list, labels_list, _ = parse_bbox_csv(csv_path, img_width=w, img_height=h)

        # Apply augmentation for training
        if self.is_train:
            # 1. Random horizontal flip (p=0.5)
            if random.random() < 0.5:
                img = F.hflip(img)
                flipped_boxes = []
                for b in boxes_list:
                    new_x1 = max(0.0, float(w - b[2]))
                    new_x2 = min(float(w), float(w - b[0]))
                    if new_x2 - new_x1 >= 2.0:
                        flipped_boxes.append([new_x1, b[1], new_x2, b[3]])
                boxes_list = flipped_boxes

            # 2. Color jitter
            if random.random() < 0.5:
                brightness = random.uniform(0.85, 1.15)
                contrast = random.uniform(0.85, 1.15)
                img = F.adjust_brightness(img, brightness)
                img = F.adjust_contrast(img, contrast)

        # Convert image to float tensor in [0, 1]
        img_tensor = F.to_tensor(img)

        # Prepare target tensors
        if len(boxes_list) > 0:
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx], dtype=torch.int64),
            "area": area,
            "iscrowd": iscrowd,
        }

        return img_tensor, target


def collate_fn(batch: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]]) -> Tuple[Tuple[torch.Tensor, ...], Tuple[Dict[str, torch.Tensor], ...]]:
    """Collate function for object detection batching."""
    return tuple(zip(*batch))
