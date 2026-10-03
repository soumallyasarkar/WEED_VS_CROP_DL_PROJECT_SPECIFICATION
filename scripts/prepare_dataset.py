#!/usr/bin/env python3
"""Dataset validation, exploratory data analysis (EDA), and session-grouped splitting.

Outputs:
  - data/processed/splits.csv
  - data/processed/stats.json
  - outputs/plots/samples.png
  - outputs/plots/class_distribution.png
  - outputs/plots/box_size_hist.png
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
import logging
import math
import random
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

from src.config import load_config
from src.data.parse_annotations import parse_bbox_csv, parse_image_session, parse_params_csv
from src.data.splits import create_session_grouped_splits, verify_splits_integrity
from src.utils.seed import set_seed
from src.utils.viz import draw_bounding_boxes

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s]: %(message)s")
logger = logging.getLogger("prepare_dataset")


def run_eda_and_preparation(config_path: str = "configs/default.yaml") -> Dict[str, Any]:
    cfg = load_config(config_path)
    set_seed(cfg.get("seed", 42))

    raw_dir = Path(cfg.paths.raw_dir)
    processed_dir = Path(cfg.paths.processed_dir)
    outputs_dir = Path(cfg.paths.outputs_dir)
    plots_dir = outputs_dir / "plots"

    processed_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    images_dir = raw_dir / "images"
    variant = cfg.data.get("variant", "CropOrWeed2")
    bboxes_dir = raw_dir / "bboxes" / variant
    params_dir = raw_dir / "params"

    if not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not bboxes_dir.is_dir():
        raise FileNotFoundError(f"Bounding boxes directory not found: {bboxes_dir}")

    # Gather available image files
    all_image_paths = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")))
    logger.info(f"Found {len(all_image_paths)} images in {images_dir}")

    stats: Dict[str, Any] = {
        "total_images_in_dir": len(all_image_paths),
        "valid_images": 0,
        "corrupt_images": 0,
        "images_with_annotations": 0,
        "images_without_annotations": 0,
        "total_crop_instances": 0,
        "total_weed_instances": 0,
        "total_dropped_boxes": 0,
        "image_resolutions": Counter(),
        "sessions": set(),
        "box_size_buckets": {"tiny (<32x32)": 0, "small (32-96)": 0, "medium (96-256)": 0, "large (>256)": 0},
    }

    image_records = []
    box_areas = []
    box_widths = []
    box_heights = []

    corrupt_files = []
    session_instance_counts = defaultdict(lambda: {"crop": 0, "weed": 0, "images": 0})
    param_counts = {"moisture": Counter(), "soil": Counter(), "lighting": Counter(), "separability": Counter()}

    logger.info("Validating images and parsing annotations...")
    for img_path in all_image_paths:
        stem = img_path.stem
        # 1. Verify PIL image readability
        try:
            with Image.open(img_path) as img:
                img.verify()
            with Image.open(img_path) as img:
                w, h = img.size
            stats["valid_images"] += 1
            stats["image_resolutions"][f"{w}x{h}"] += 1
        except Exception as e:
            logger.warning(f"Corrupt or unreadable image: {img_path.name} ({e})")
            corrupt_files.append(img_path.name)
            stats["corrupt_images"] += 1
            continue

        prefix, session = parse_image_session(img_path.name)
        stats["sessions"].add(session)
        session_instance_counts[session]["images"] += 1

        # 2. Parse bounding boxes
        csv_path = bboxes_dir / f"{stem}.csv"
        boxes, labels, dropped = parse_bbox_csv(csv_path, img_width=w, img_height=h)
        stats["total_dropped_boxes"] += dropped

        crop_count = labels.count(1)
        weed_count = labels.count(2)
        stats["total_crop_instances"] += crop_count
        stats["total_weed_instances"] += weed_count

        session_instance_counts[session]["crop"] += crop_count
        session_instance_counts[session]["weed"] += weed_count

        if len(boxes) > 0:
            stats["images_with_annotations"] += 1
        else:
            stats["images_without_annotations"] += 1

        # Box dimensions
        for box in boxes:
            bw = box[2] - box[0]
            bh = box[3] - box[1]
            box_widths.append(bw)
            box_heights.append(bh)
            area = bw * bh
            box_areas.append(area)

            max_dim = max(bw, bh)
            if max_dim < 32:
                stats["box_size_buckets"]["tiny (<32x32)"] += 1
            elif max_dim < 96:
                stats["box_size_buckets"]["small (32-96)"] += 1
            elif max_dim < 256:
                stats["box_size_buckets"]["medium (96-256)"] += 1
            else:
                stats["box_size_buckets"]["large (>256)"] += 1

        # 3. Parse params metadata if available
        params_file = params_dir / f"{stem}.csv"
        params_meta = parse_params_csv(params_file)
        for k, v in params_meta.items():
            if k in param_counts:
                param_counts[k][v] += 1

        image_records.append(
            {
                "image_id": stem,
                "filename": img_path.name,
                "session": session,
                "width": w,
                "height": h,
                "crop_count": crop_count,
                "weed_count": weed_count,
                "total_plants": len(boxes),
                **params_meta,
            }
        )

    stats["sessions"] = sorted(list(stats["sessions"]))
    stats["num_sessions"] = len(stats["sessions"])
    stats["corrupt_files"] = corrupt_files
    stats["image_resolutions"] = dict(stats["image_resolutions"])
    stats["params_distributions"] = {k: dict(v) for k, v in param_counts.items()}

    logger.info(
        f"Validation summary: {stats['valid_images']} valid images, {stats['corrupt_images']} corrupt, "
        f"{stats['num_sessions']} recording sessions."
    )
    logger.info(
        f"Instance counts: Crop={stats['total_crop_instances']}, Weed={stats['total_weed_instances']} "
        f"(Ratio Crop/Weed: {stats['total_crop_instances'] / max(1, stats['total_weed_instances']):.2f})"
    )

    # 4. Generate Session-Grouped Splits
    logger.info("Creating session-grouped train/val/test splits...")
    valid_filenames = [r["filename"] for r in image_records]
    splits_csv_path = processed_dir / "splits.csv"
    splits_df = create_session_grouped_splits(
        valid_filenames,
        train_ratio=cfg.data.split.get("train", 0.70),
        val_ratio=cfg.data.split.get("val", 0.15),
        test_ratio=cfg.data.split.get("test", 0.15),
        seed=cfg.get("seed", 42),
        output_path=splits_csv_path,
    )

    # Calculate split breakdown
    split_records_df = pd.DataFrame(image_records).merge(splits_df, on=["image_id", "session"])
    split_summary = {}
    for sp in ["train", "val", "test"]:
        sub = split_records_df[split_records_df["split"] == sp]
        split_summary[sp] = {
            "images": int(len(sub)),
            "sessions": int(sub["session"].nunique()),
            "crop_instances": int(sub["crop_count"].sum()),
            "weed_instances": int(sub["weed_count"].sum()),
            "total_instances": int(sub["crop_count"].sum() + sub["weed_count"].sum()),
        }
    stats["split_summary"] = split_summary
    logger.info(f"Split breakdown: {json.dumps(split_summary, indent=2)}")

    # Save stats.json
    stats_path = processed_dir / "stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info(f"Saved stats to {stats_path}")

    # 5. Visualizations
    sns.set_theme(style="whitegrid")

    # Plot 1: Class distribution
    plt.figure(figsize=(8, 5))
    classes = ["Crop (class 1)", "Weed (class 2)"]
    counts = [stats["total_crop_instances"], stats["total_weed_instances"]]
    colors = ["#2ecc71", "#e74c3c"]
    bars = plt.bar(classes, counts, color=colors, edgecolor="black", width=0.5)
    for bar in bars:
        h = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, h + max(counts) * 0.01, f"{int(h):,}", ha="center", fontweight="bold")
    plt.title("Crop vs Weed Instance Distribution in Dataset", fontsize=13, fontweight="bold")
    plt.ylabel("Number of Annotated Instances")
    plt.tight_layout()
    plt.savefig(plots_dir / "class_distribution.png", dpi=200)
    plt.close()
    logger.info(f"Saved class distribution plot to {plots_dir / 'class_distribution.png'}")

    # Plot 2: Box size histogram
    plt.figure(figsize=(9, 5))
    bucket_names = list(stats["box_size_buckets"].keys())
    bucket_vals = list(stats["box_size_buckets"].values())
    plt.bar(bucket_names, bucket_vals, color="#3498db", edgecolor="black")
    for i, v in enumerate(bucket_vals):
        plt.text(i, v + max(bucket_vals) * 0.01, f"{v:,}", ha="center", fontweight="bold")
    plt.title("Bounding Box Size Distribution (max dimension)", fontsize=13, fontweight="bold")
    plt.xlabel("Size Category (pixels)")
    plt.ylabel("Box Count")
    plt.tight_layout()
    plt.savefig(plots_dir / "box_size_hist.png", dpi=200)
    plt.close()
    logger.info(f"Saved box size histogram to {plots_dir / 'box_size_hist.png'}")

    # Plot 3: 12 Random Sample Images with Annotations
    logger.info("Generating sample grid visualization...")
    annotated_samples = [r for r in image_records if r["total_plants"] > 0]
    sample_pool = random.sample(annotated_samples, min(12, len(annotated_samples)))

    fig, axes = plt.subplots(3, 4, figsize=(16, 12))
    axes = axes.flatten()

    for idx, sample in enumerate(sample_pool):
        img_path = images_dir / sample["filename"]
        with Image.open(img_path) as raw_img:
            img = raw_img.convert("RGB")
        csv_path = bboxes_dir / f"{sample['image_id']}.csv"
        boxes, labels, _ = parse_bbox_csv(csv_path, img_width=sample["width"], img_height=sample["height"])

        annotated_img = draw_bounding_boxes(img, boxes, labels, width=4)
        axes[idx].imshow(annotated_img)
        axes[idx].set_title(
            f"{sample['image_id']}\nCrops: {sample['crop_count']} | Weeds: {sample['weed_count']}",
            fontsize=9,
        )
        axes[idx].axis("off")

    plt.suptitle("Sample Field Images with Ground Truth Annotations (Green: Crop, Red: Weed)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plots_dir / "samples.png", dpi=200)
    plt.close()
    logger.info(f"Saved sample visualization to {plots_dir / 'samples.png'}")

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare dataset and generate EDA outputs.")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to YAML config")
    args = parser.parse_args()
    run_eda_and_preparation(args.config)


if __name__ == "__main__":
    main()
