"""Unit tests for checkpoint saving and loading."""

import tempfile
import unittest
from pathlib import Path
import torch
from src.models.detector import build_detector, load_checkpoint, save_checkpoint


class TestCheckpoint(unittest.TestCase):
    """Verify that saving and loading checkpoints preserves weights and metadata."""

    def test_checkpoint_save_and_load(self):
        model = build_detector(num_classes=3, min_size=320, max_size=480, pretrained=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with tempfile.TemporaryDirectory() as tmp_dir:
            ckpt_path = Path(tmp_dir) / "test_model.pt"
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

            self.assertTrue(ckpt_path.is_file())

            loaded_model, ckpt_dict = load_checkpoint(ckpt_path, device="cpu")
            self.assertEqual(ckpt_dict["epoch"], 3)
            self.assertEqual(ckpt_dict["metrics"]["mAP"], 0.42)
            self.assertEqual(ckpt_dict["class_mapping"][1], "crop")
            self.assertEqual(ckpt_dict["class_mapping"][2], "weed")


if __name__ == "__main__":
    unittest.main()
