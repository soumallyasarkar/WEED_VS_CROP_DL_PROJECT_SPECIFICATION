#!/usr/bin/env python3
"""Generate sample annotated detection predictions for qualitative evaluation."""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.splits import load_splits
from src.inference.predict import run_predict_cli


def main() -> None:
    parser = argparse.ArgumentParser(description="Make sample predictions on test set.")
    parser.add_argument("--checkpoint", type=str, default="models/best.pt", help="Path to checkpoint")
    parser.add_argument("--count", type=int, default=10, help="Number of test samples to annotate")
    parser.add_argument("--out", type=str, default="outputs/predictions", help="Output directory")
    args = parser.parse_args()

    splits_path = Path("data/processed/splits.csv")
    images_dir = Path("data/raw/cropandweed/images")

    if not splits_path.exists():
        print("data/processed/splits.csv not found. Run scripts/prepare_dataset.py first.")
        sys.exit(1)

    df = load_splits(splits_path)
    test_images = df[df["split"] == "test"]["image_id"].tolist()[:args.count]

    print(f"Running predictions on {len(test_images)} test images...")
    for img_id in test_images:
        img_path = images_dir / f"{img_id}.jpg"
        if img_path.exists():
            run_predict_cli(
                image_path=str(img_path),
                checkpoint_path=args.checkpoint,
                score_threshold=0.4,
                out_dir=args.out,
            )


if __name__ == "__main__":
    main()
