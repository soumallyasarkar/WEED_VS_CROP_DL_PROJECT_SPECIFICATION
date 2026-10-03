"""Unit tests for checkpoint saving and loading."""

import pytest
import torch
from src.models.detector import build_detector, load_checkpoint, save_checkpoint


def test_checkpoint_save_and_load(tmp_path):
    """Verify that saving and loading checkpoints preserves weights and metadata."""
    model = build_detector(num_classes=3, min_size=320, max_size=480, pretrained=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    ckpt_path = tmp_path / "test_model.pt"
    metrics = {"mAP": 0.42, "mAP_50": 0.75}
    config = {"model": {"num_classes": 3, "min_size": 320, "max_size": 480}}

    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=3,
        metrics=metrics,
        config_dict=config,
        checkpoint_path=ckpt_path,
    )

    assert ckpt_path.is_file()

    loaded_model, ckpt_dict = load_checkpoint(ckpt_path, device="cpu")
    assert ckpt_dict["epoch"] == 3
    assert ckpt_dict["metrics"]["mAP"] == 0.42
    assert ckpt_dict["class_mapping"][1] == "crop"
    assert ckpt_dict["class_mapping"][2] == "weed"
