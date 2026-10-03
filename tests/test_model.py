"""Unit tests for Faster R-CNN model building and forward passes."""

import pytest
import torch
from src.models.detector import build_detector, freeze_backbone, unfreeze_backbone


def test_build_detector_classes():
    """Verify detector architecture and number of output classes in FastRCNNPredictor."""
    model = build_detector(num_classes=3, min_size=320, max_size=480, pretrained=False)
    predictor = model.roi_heads.box_predictor

    assert predictor.cls_score.out_features == 3
    assert predictor.bbox_pred.out_features == 3 * 4  # 4 coordinates per class


def test_model_forward_pass_train_and_eval():
    """Verify forward pass in both training and evaluation modes."""
    model = build_detector(num_classes=3, min_size=320, max_size=480, pretrained=False)

    # 1. Train mode
    model.train()
    images = [torch.rand(3, 100, 100), torch.rand(3, 100, 100)]
    targets = [
        {"boxes": torch.tensor([[10.0, 10.0, 30.0, 30.0]]), "labels": torch.tensor([1])},
        {"boxes": torch.tensor([[20.0, 20.0, 50.0, 50.0]]), "labels": torch.tensor([2])},
    ]

    loss_dict = model(images, targets)
    assert isinstance(loss_dict, dict)
    assert "loss_classifier" in loss_dict
    assert "loss_box_reg" in loss_dict
    assert "loss_objectness" in loss_dict
    assert "loss_rpn_box_reg" in loss_dict

    total_loss = sum(v for v in loss_dict.values())
    assert total_loss.item() > 0.0

    # 2. Eval mode
    model.eval()
    with torch.no_grad():
        preds = model(images)
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert "boxes" in preds[0]
    assert "labels" in preds[0]
    assert "scores" in preds[0]


def test_freeze_unfreeze_backbone():
    """Verify backbone parameter freezing and unfreezing."""
    model = build_detector(num_classes=3, min_size=320, max_size=480, pretrained=False)

    freeze_backbone(model)
    for p in model.backbone.parameters():
        assert not p.requires_grad

    unfreeze_backbone(model)
    for p in model.backbone.parameters():
        assert p.requires_grad
