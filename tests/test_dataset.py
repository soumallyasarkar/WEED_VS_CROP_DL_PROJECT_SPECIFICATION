"""Unit tests for WeedCropDetectionDataset and transformations."""

from pathlib import Path
import unittest
import torch
from src.data.dataset import WeedCropDetectionDataset, collate_fn


class TestDataset(unittest.TestCase):
    """Unit tests for WeedCropDetectionDataset and transformations."""

    def test_dataset_loading_and_tensors(self):
        """Verify that dataset loads items correctly with expected types and shapes."""
        splits_file = Path("data/processed/splits.csv")
        images_dir = Path("data/raw/cropandweed/images")
        bboxes_dir = Path("data/raw/cropandweed/bboxes/CropOrWeed2")

        if not splits_file.exists() or not images_dir.exists():
            self.skipTest("Dataset or splits file not present.")

        dataset = WeedCropDetectionDataset(
            images_dir=images_dir,
            bboxes_dir=bboxes_dir,
            splits_path=splits_file,
            split="train",
            subset_fraction=0.01,
            max_samples=5,
            is_train=False,
        )

        self.assertGreater(len(dataset), 0)

        img_tensor, target = dataset[0]

        # Verify image tensor
        self.assertIsInstance(img_tensor, torch.Tensor)
        self.assertEqual(img_tensor.dtype, torch.float32)
        self.assertEqual(img_tensor.ndim, 3)
        self.assertEqual(img_tensor.shape[0], 3)  # RGB channels
        self.assertGreaterEqual(img_tensor.min().item(), 0.0)
        self.assertLessEqual(img_tensor.max().item(), 1.0)

        # Verify target dict
        self.assertIn("boxes", target)
        self.assertIn("labels", target)
        self.assertIn("image_id", target)
        self.assertIsInstance(target["boxes"], torch.Tensor)
        self.assertIsInstance(target["labels"], torch.Tensor)

        if len(target["boxes"]) > 0:
            self.assertEqual(target["boxes"].shape[1], 4)
            self.assertEqual(target["labels"].dtype, torch.int64)
            # Labels must be in {1, 2}
            for lbl in target["labels"]:
                self.assertIn(lbl.item(), [1, 2])

    def test_collate_fn(self):
        """Verify that collate_fn batches images and targets as tuples."""
        img1 = torch.zeros((3, 100, 100))
        tgt1 = {"boxes": torch.zeros((1, 4)), "labels": torch.tensor([1])}
        img2 = torch.zeros((3, 100, 100))
        tgt2 = {"boxes": torch.zeros((2, 4)), "labels": torch.tensor([1, 2])}

        batch = [(img1, tgt1), (img2, tgt2)]
        imgs, tgts = collate_fn(batch)

        self.assertIsInstance(imgs, tuple)
        self.assertIsInstance(tgts, tuple)
        self.assertEqual(len(imgs), 2)
        self.assertEqual(len(tgts), 2)


if __name__ == "__main__":
    unittest.main()
