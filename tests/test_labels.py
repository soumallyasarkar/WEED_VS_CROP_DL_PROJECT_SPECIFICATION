"""Unit tests for label mapping and project conventions."""

import pytest
from src.data.parse_annotations import RAW_TO_TORCHVISION_LABEL, parse_bbox_csv
from src.models.detector import CLASS_MAPPING


def test_label_mapping_values():
    """Verify that CropOrWeed2 labels map to 1 (Crop) and 2 (Weed), with 0 reserved for background."""
    # Torchvision convention
    assert CLASS_MAPPING[0] == "background"
    assert CLASS_MAPPING[1] == "crop"
    assert CLASS_MAPPING[2] == "weed"

    # Variant raw mapping
    assert RAW_TO_TORCHVISION_LABEL[0] == 1  # Raw Crop -> 1
    assert RAW_TO_TORCHVISION_LABEL[1] == 2  # Raw Weed -> 2


def test_unmapped_species_excluded(tmp_path):
    """Verify that species IDs outside 0 and 1 are ignored."""
    test_csv = tmp_path / "test.csv"
    # Row with valid crop (0), valid weed (1), and unmapped species (99)
    test_csv.write_text(
        "10,10,50,50,0,30,30\n"
        "60,60,100,100,1,80,80\n"
        "120,120,150,150,99,135,135\n"
    )

    boxes, labels, dropped = parse_bbox_csv(test_csv)
    assert len(boxes) == 2
    assert len(labels) == 2
    assert labels == [1, 2]
    assert dropped == 1
