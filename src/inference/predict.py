"""Standalone inference pipeline for Weed vs Crop discrimination."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchvision.transforms.functional as F
from PIL import Image

from src.models.detector import load_checkpoint
from src.utils.viz import draw_bounding_boxes

logger = logging.getLogger("predict")

CLASS_NAMES = {
    1: "crop",
    2: "weed",
}


def load_model(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """Load trained detector model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint .pt file.
        device: Device to load onto ('cpu' or 'cuda').

    Returns:
        Tuple of (model, checkpoint_dictionary).
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file does not exist: {path}")

    # Fallback to cpu if cuda requested but unavailable
    if "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    model, ckpt = load_checkpoint(path, device=device)
    model.eval()
    return model, ckpt


def predict_image(
    model: torch.nn.Module,
    image: Image.Image,
    score_threshold: float = 0.5,
    device: str = "cpu",
) -> List[Dict[str, Any]]:
    """Run object detection on a PIL Image.

    Args:
        model: Trained detector model in eval mode.
        image: PIL Image object.
        score_threshold: Minimum confidence score to retain detection.
        device: Device model is residing on ('cpu' or 'cuda').

    Returns:
        List of detection dicts: [{'label': 'crop'|'weed', 'class_id': int, 'score': float, 'box': [x1, y1, x2, y2]}]
    """
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL Image, received {type(image)}")

    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    if w <= 0 or h <= 0:
        raise ValueError("Image has zero width or height.")

    # Shared preprocessing: PIL to tensor in [0.0, 1.0]
    try:
        target_device = next(model.parameters()).device
    except (StopIteration, AttributeError):
        target_device = torch.device(device)

    img_tensor = F.to_tensor(img_rgb).to(target_device)

    model.eval()
    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]
    boxes = pred["boxes"].detach().cpu().numpy()
    scores = pred["scores"].detach().cpu().numpy()
    labels = pred["labels"].detach().cpu().numpy()

    detections: List[Dict[str, Any]] = []
    for b, s, l in zip(boxes, scores, labels):
        if s < score_threshold:
            continue
        label_id = int(l)
        label_name = CLASS_NAMES.get(label_id, f"class_{label_id}")
        detections.append({
            "label": label_name,
            "class_id": label_id,
            "score": round(float(s), 4),
            "box": [round(float(c), 2) for c in b],
        })

    return detections


def summarize(detections: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute summary statistics for a set of detections."""
    n_crop = sum(1 for d in detections if d["label"] == "crop")
    n_weed = sum(1 for d in detections if d["label"] == "weed")
    total = n_crop + n_weed

    mean_conf = float(sum(d["score"] for d in detections) / total) if total > 0 else 0.0
    weed_share = float(n_weed / total) if total > 0 else 0.0

    return {
        "n_crop": n_crop,
        "n_weed": n_weed,
        "total_plants": total,
        "mean_confidence": round(mean_conf, 4),
        "weed_share": round(weed_share, 4),
    }


def draw_detections(
    image: Image.Image,
    detections: List[Dict[str, Any]],
    score_threshold: float = 0.0,
) -> Image.Image:
    """Draw bounding boxes and confidence tags on a PIL Image."""
    img_rgb = image.convert("RGB")
    boxes = [d["box"] for d in detections]
    labels = [d["class_id"] for d in detections]
    scores = [d["score"] for d in detections]

    return draw_bounding_boxes(
        image=img_rgb,
        boxes=boxes,
        labels=labels,
        scores=scores,
        score_threshold=score_threshold,
        width=3,
    )


def run_predict_cli(
    image_path: str,
    checkpoint_path: str = "models/best.pt",
    score_threshold: float = 0.5,
    out_dir: str = "outputs/predictions",
    device: str = "cpu",
) -> Dict[str, Any]:
    """Execute prediction from CLI."""
    img_p = Path(image_path)
    if not img_p.exists():
        logger.error(f"Image file not found: {img_p}")
        sys.exit(1)

    try:
        with Image.open(img_p) as raw:
            raw.verify()
        with Image.open(img_p) as raw:
            img = raw.convert("RGB")
    except Exception as e:
        logger.error(f"Failed to read image {img_p}: {e}")
        sys.exit(1)

    model, _ = load_model(checkpoint_path, device=device)
    detections = predict_image(model, img, score_threshold=score_threshold, device=device)
    summary_data = summarize(detections)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Annotated image
    annotated_img = draw_detections(img, detections)
    annotated_img_path = out_path / f"{img_p.stem}_annotated.jpg"
    annotated_img.save(annotated_img_path, quality=95)

    # JSON results
    result_dict = {
        "image": img_p.name,
        "width": img.width,
        "height": img.height,
        "score_threshold": score_threshold,
        "summary": summary_data,
        "detections": detections,
    }
    json_path = out_path / f"{img_p.stem}_detections.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"Saved annotated image to {annotated_img_path}")
    logger.info(f"Saved detection results to {json_path}")
    logger.info(f"Summary: {summary_data['n_crop']} crops, {summary_data['n_weed']} weeds (Weed share: {summary_data['weed_share']*100:.1f}%)")

    return result_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict crops and weeds in a field image.")
    parser.add_argument("--image", type=str, required=True, help="Path to input field image")
    parser.add_argument("--checkpoint", type=str, default="models/best.pt", help="Path to model checkpoint")
    parser.add_argument("--score-threshold", type=float, default=0.5, help="Confidence score threshold")
    parser.add_argument("--out", type=str, default="outputs/predictions", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Inference device")
    args = parser.parse_args()

    run_predict_cli(
        image_path=args.image,
        checkpoint_path=args.checkpoint,
        score_threshold=args.score_threshold,
        out_dir=args.out,
        device=args.device,
    )


if __name__ == "__main__":
    main()
