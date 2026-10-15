"""Unit tests for inference functions and error handling."""

import unittest
from PIL import Image
import torch
from src.inference.predict import predict_image, summarize
from src.models.detector import build_detector


class TestInference(unittest.TestCase):
    """Unit tests for inference functions and error handling."""

    def test_predict_image_and_summarize(self):
        """Verify predict_image outputs expected list structure and summary statistics."""
        model = build_detector(num_classes=3, min_size=320, max_size=480, pretrained=False)
        img = Image.new("RGB", (640, 480), color=(100, 150, 100))

        detections = predict_image(model, img, score_threshold=0.0, device="cpu")
        self.assertIsInstance(detections, list)

        summary = summarize(detections)
        self.assertIn("n_crop", summary)
        self.assertIn("n_weed", summary)
        self.assertIn("total_plants", summary)
        self.assertIn("weed_share", summary)
        self.assertEqual(summary["total_plants"], summary["n_crop"] + summary["n_weed"])

    def test_invalid_image_handling(self):
        """Verify clean error handling for non-PIL or corrupt image inputs."""
        model = build_detector(num_classes=3, min_size=320, max_size=480, pretrained=False)

        with self.assertRaises(TypeError):
            predict_image(model, "not_an_image", score_threshold=0.5)  # type: ignore

        with self.assertRaises(ValueError):
            empty_img = Image.new("RGB", (0, 0))
            predict_image(model, empty_img, score_threshold=0.5)


if __name__ == "__main__":
    unittest.main()
