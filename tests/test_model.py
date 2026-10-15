"""Unit tests for Faster R-CNN model building and forward passes."""

import unittest
import torch
from src.models.detector import build_detector, freeze_backbone, unfreeze_backbone


class TestModel(unittest.TestCase):
    """Unit tests for Faster R-CNN model building and forward passes."""

    def test_build_detector_classes(self):
        """Verify detector architecture and number of output classes in FastRCNNPredictor."""
        model = build_detector(num_classes=3, min_size=320, max_size=480, pretrained=False)
        predictor = model.roi_heads.box_predictor

        self.assertEqual(predictor.cls_score.out_features, 3)
        self.assertEqual(predictor.bbox_pred.out_features, 3 * 4)

    def test_model_forward_pass_train_and_eval(self):
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
        self.assertIsInstance(loss_dict, dict)
        self.assertIn("loss_classifier", loss_dict)
        self.assertIn("loss_box_reg", loss_dict)
        self.assertIn("loss_objectness", loss_dict)
        self.assertIn("loss_rpn_box_reg", loss_dict)

        total_loss = sum(v for v in loss_dict.values())
        self.assertGreater(total_loss.item(), 0.0)

        # 2. Eval mode
        model.eval()
        with torch.no_grad():
            preds = model(images)
        self.assertIsInstance(preds, list)
        self.assertEqual(len(preds), 2)
        self.assertIn("boxes", preds[0])
        self.assertIn("labels", preds[0])
        self.assertIn("scores", preds[0])

    def test_freeze_unfreeze_backbone(self):
        """Verify backbone parameter freezing and unfreezing."""
        model = build_detector(num_classes=3, min_size=320, max_size=480, pretrained=False)

        freeze_backbone(model)
        for p in model.backbone.parameters():
            self.assertFalse(p.requires_grad)

        unfreeze_backbone(model)
        for p in model.backbone.parameters():
            self.assertTrue(p.requires_grad)


if __name__ == "__main__":
    unittest.main()
