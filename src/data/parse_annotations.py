"""Parsing and validation for CropAndWeed annotations."""

import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from PIL import Image

logger = logging.getLogger(__name__)

# Variant label to torchvision class mapping:
# In CropOrWeed2 CSV: 0 is Crop, 1 is Weed.
# In TorchVision: 0 is reserved for background, 1 is Crop, 2 is Weed.
RAW_TO_TORCHVISION_LABEL = {
    0: 1,  # Crop
    1: 2,  # Weed
}


def parse_bbox_csv(
    csv_path: Path | str,
    img_width: Optional[int] = None,
    img_height: Optional[int] = None,
    min_box_size: float = 2.0,
) -> Tuple[List[List[float]], List[int], int]:
    """Parse bounding boxes from a CSV file and convert to torchvision convention.

    Format in raw CSV: left, top, right, bottom, label_id, stem_x, stem_y (no header).

    Args:
        csv_path: Path to the annotation CSV file.
        img_width: Optional image width for clipping.
        img_height: Optional image height for clipping.
        min_box_size: Minimum width and height for a box to be considered valid.

    Returns:
        Tuple of (boxes, labels, dropped_count) where:
            boxes: list of [x1, y1, x2, y2] coordinates.
            labels: list of mapped class labels (1=crop, 2=weed).
            dropped_count: number of invalid/out-of-bounds boxes dropped.
    """
    path = Path(csv_path)
    if not path.is_file():
        return [], [], 0

    boxes: List[List[float]] = []
    labels: List[int] = []
    dropped = 0

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for line_num, row in enumerate(reader, start=1):
            if not row or len(row) < 5:
                dropped += 1
                continue

            try:
                x1 = float(row[0])
                y1 = float(row[1])
                x2 = float(row[2])
                y2 = float(row[3])
                raw_label = int(float(row[4]))
            except (ValueError, IndexError):
                dropped += 1
                continue

            # Check if label is in mapped classes
            if raw_label not in RAW_TO_TORCHVISION_LABEL:
                dropped += 1
                continue

            mapped_label = RAW_TO_TORCHVISION_LABEL[raw_label]

            # Ensure x1 <= x2 and y1 <= y2
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1

            # Clip to image boundaries if dimensions provided
            if img_width is not None and img_height is not None:
                x1 = max(0.0, min(float(img_width), x1))
                y1 = max(0.0, min(float(img_height), y1))
                x2 = max(0.0, min(float(img_width), x2))
                y2 = max(0.0, min(float(img_height), y2))

            # Validate box area / dimensions
            w = x2 - x1
            h = y2 - y1
            if w < min_box_size or h < min_box_size:
                dropped += 1
                continue

            boxes.append([x1, y1, x2, y2])
            labels.append(mapped_label)

    return boxes, labels, dropped


def parse_params_csv(params_path: Path | str) -> Dict[str, int]:
    """Parse image metadata from params CSV file.

    Format: moisture,soil,lighting,separability (with header).
    """
    path = Path(params_path)
    if not path.is_file():
        return {}

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            return {
                "moisture": int(row.get("moisture", 0)),
                "soil": int(row.get("soil", 0)),
                "lighting": int(row.get("lighting", 0)),
                "separability": int(row.get("separability", 0)),
            }
    return {}


def parse_image_session(image_filename: str) -> Tuple[str, str]:
    """Extract prefix and session from image filename.

    Example: 'ave-0020-0010.jpg' -> prefix='ave', session='ave-0020'
    """
    stem = Path(image_filename).stem
    parts = stem.split("-")
    if len(parts) >= 2:
        prefix = parts[0]
        session = f"{parts[0]}-{parts[1]}"
        return prefix, session
    return "unknown", "unknown-0000"
