"""Visualization utilities for annotations and detections."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Color palette: Green for Crop, Red for Weed
CLASS_COLORS = {
    1: (46, 204, 113),   # Crop: Emerald Green
    2: (231, 76, 60),    # Weed: Crimson Red
    0: (128, 128, 128),  # Background: Gray
}
CLASS_NAMES = {
    1: "crop",
    2: "weed",
}


def draw_bounding_boxes(
    image: Image.Image,
    boxes: Union[List[List[float]], np.ndarray],
    labels: Union[List[int], np.ndarray],
    scores: Optional[Union[List[float], np.ndarray]] = None,
    score_threshold: float = 0.0,
    width: int = 3,
) -> Image.Image:
    """Draw bounding boxes and class labels with scores on a PIL Image.

    Args:
        image: PIL Image object in RGB mode.
        boxes: List or array of boxes in [x1, y1, x2, y2] format.
        labels: List or array of integer class labels (1=crop, 2=weed).
        scores: Optional list or array of confidence scores in [0, 1].
        score_threshold: Minimum score threshold to render box.
        width: Line width for bounding box outline.

    Returns:
        New PIL Image with rendered bounding boxes.
    """
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    for i in range(len(boxes)):
        box = boxes[i]
        label = int(labels[i])
        score = float(scores[i]) if scores is not None and len(scores) > i else None

        if score is not None and score < score_threshold:
            continue

        color = CLASS_COLORS.get(label, (255, 255, 0))
        label_name = CLASS_NAMES.get(label, f"class_{label}")

        x1, y1, x2, y2 = [float(coord) for coord in box]
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)

        text = f"{label_name}"
        if score is not None:
            text += f" {score:.2f}"

        # Draw a small text background tag
        text_bbox = draw.textbbox((x1, max(0.0, y1 - 16)), text)
        draw.rectangle(text_bbox, fill=color)
        draw.text((x1 + 2, max(0.0, y1 - 16)), text, fill=(255, 255, 255))

    return canvas
