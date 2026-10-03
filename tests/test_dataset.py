"""Unit tests for WeedCropDetectionDataset and transformations."""

from pathlib import Path
import pytest
import torch
from src.data.dataset import WeedCropDetectionDataset, collate_fn


def test_dataset_loading_and_tensors():
    """Verify that dataset loads items correctly with expected types and shapes."""
    splits_file = Path("data/processed/splits.csv")
    images_dir = Path("data/raw/cropandweed/images")
    bboxes_dir = Path("data/raw/cropandweed/bboxes/CropOrWeed2")

    if not splits_file.exists() or not images_dir.exists():
        pytest.skip("Dataset or splits file not present.")

    dataset = WeedCropDetectionDataset(
        images_dir=images_dir,
        bboxes_dir=bboxes_dir,
        splits_path=splits_file,
        split="train",
        subset_fraction=0.01,
        max_samples=5,
        is_train=False,
    )

    assert len(dataset) > 0

    img_tensor, target = dataset[0]

    # Verify image tensor
    assert isinstance(img_tensor, torch.Tensor)
    assert img_tensor.dtype == torch.float32
    assert img_tensor.ndim == 3
    assert img_tensor.shape[0] == 3  # RGB channels
    assert img_tensor.min() >= 0.0
    assert img_tensor.max() <= 1.0

    # Verify target dict
    assert "boxes" in target
    assert "labels" in target
    assert "image_id" in target
    assert isinstance(target["boxes"], torch.Tensor)
    assert isinstance(target["labels"], torch.Tensor)

    if len(target["boxes"]) > 0:
        assert target["boxes"].shape[1] == 4
        assert target["labels"].dtype == torch.int64
        # Labels must be in {1, 2}
        for lbl in target["labels"]:
            assert lbl.item() in [1, 2]


def test_collate_fn():
    """Verify that collate_fn batches images and targets as tuples."""
    img1 = torch.zeros((3, 100, 100))
    tgt1 = {"boxes": torch.zeros((1, 4)), "labels": torch.tensor([1])}
    img2 = torch.zeros((3, 100, 100))
    tgt2 = {"boxes": torch.zeros((2, 4)), "labels": torch.tensor([1, 2])}

    batch = [(img1, tgt1), (img2, tgt2)]
    imgs, tgts = collate_fn(batch)

    assert isinstance(imgs, tuple)
    assert isinstance(tgts, tuple)
    assert len(imgs) == 2
    assert len(tgts) == 2
