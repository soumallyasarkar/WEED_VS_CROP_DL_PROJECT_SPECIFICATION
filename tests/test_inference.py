"""Unit tests for inference functions and error handling."""

import pytest
from PIL import Image
import torch
from src.inference.predict import predict_image, summarize
from src.models.detector import build_detector


def test_predict_image_and_summarize():
    """Verify predict_image outputs expected list structure and summary statistics."""
    model = build_detector(num_classes=3, min_size=320, max_size=480, pretrained=False)
    img = Image.new("RGB", (640, 480), color=(100, 150, 100))

    detections = predict_image(model, img, score_threshold=0.0, device="cpu")
    assert isinstance(detections, list)

    summary = summarize(detections)
    assert "n_crop" in summary
    assert "n_weed" in summary
    assert "total_plants" in summary
    assert "weed_share" in summary
    assert summary["total_plants"] == summary["n_crop"] + summary["n_weed"]


def test_invalid_image_handling():
    """Verify clean error handling for non-PIL or corrupt image inputs."""
    model = build_detector(num_classes=3, min_size=320, max_size=480, pretrained=False)

    with pytest.raises(TypeError):
        predict_image(model, "not_an_image", score_threshold=0.5)  # type: ignore

    with pytest.raises(ValueError):
        empty_img = Image.new("RGB", (0, 0))
        predict_image(model, empty_img, score_threshold=0.5)
