"""Faster R-CNN MobileNetV3-Large FPN detector for weed and crop discrimination."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import (
    FasterRCNN_MobileNet_V3_Large_FPN_Weights,
    fasterrcnn_mobilenet_v3_large_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

logger = logging.getLogger(__name__)

CLASS_MAPPING = {
    0: "background",
    1: "crop",
    2: "weed",
}


def build_detector(
    num_classes: int = 3,
    min_size: int = 640,
    max_size: int = 1024,
    pretrained: bool = True,
) -> nn.Module:
    """Build a Faster R-CNN detector with MobileNetV3-Large FPN backbone.

    Args:
        num_classes: Total classes including background (3: background, crop, weed).
        min_size: Minimum image size for internal detector resizing.
        max_size: Maximum image size for internal detector resizing.
        pretrained: Whether to initialize with COCO-pretrained weights.

    Returns:
        torchvision FasterRCNN model with customized FastRCNNPredictor head.
    """
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT if pretrained else None

    logger.info(
        f"Building Faster R-CNN MobileNetV3-Large FPN (num_classes={num_classes}, "
        f"min_size={min_size}, max_size={max_size}, pretrained={pretrained})"
    )

    model = fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights,
        min_size=min_size,
        max_size=max_size,
    )

    # Replace box predictor head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


def freeze_backbone(model: nn.Module) -> None:
    """Freeze all layers in the backbone for Stage 1 warm-up."""
    logger.info("Freezing detector backbone parameters.")
    for param in model.backbone.parameters():
        param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    """Unfreeze backbone parameters for fine-tuning in Stage 2."""
    logger.info("Unfreezing detector backbone parameters.")
    for param in model.backbone.parameters():
        param.requires_grad = True


def save_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    metrics: Dict[str, Any],
    config_dict: Dict[str, Any],
    checkpoint_path: Path | str,
) -> None:
    """Save model checkpoint with weights, metadata, and configuration snapshot."""
    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "epoch": epoch,
        "metrics": metrics,
        "class_mapping": CLASS_MAPPING,
        "config": config_dict,
        "torch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
    }
    torch.save(checkpoint, path)
    logger.info(f"Saved checkpoint to {path}")


def load_checkpoint(
    checkpoint_path: Path | str,
    device: str | torch.device = "cpu",
    model: Optional[nn.Module] = None,
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Load model checkpoint from disk.

    Args:
        checkpoint_path: Path to .pt checkpoint file.
        device: Device to map weights to ('cpu' or 'cuda').
        model: Optional pre-built model instance to load weights into.

    Returns:
        Tuple of (model, checkpoint_dict).
    """
    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint file not found: {path}")

    # Fallback to CPU if CUDA is requested on a system without an active CUDA runtime
    if isinstance(device, str) and "cuda" in device and not torch.cuda.is_available():
        logger.warning("CUDA requested for checkpoint loading but not available. Falling back to CPU.")
        device = "cpu"
    elif isinstance(device, torch.device) and device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA device requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    checkpoint = torch.load(path, map_location=device, weights_only=False)

    if model is None:
        cfg = checkpoint.get("config", {})
        model_cfg = cfg.get("model", {})
        min_size = model_cfg.get("min_size", 640)
        max_size = model_cfg.get("max_size", 1024)
        num_classes = model_cfg.get("num_classes", 3)
        model = build_detector(num_classes=num_classes, min_size=min_size, max_size=max_size, pretrained=False)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    logger.info(f"Loaded checkpoint from {path} (Epoch {checkpoint.get('epoch', 'unknown')})")
    return model, checkpoint
