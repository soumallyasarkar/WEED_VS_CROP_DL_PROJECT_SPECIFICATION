"""Unit tests for label mapping and project conventions."""

import tempfile
import unittest
from pathlib import Path
from src.data.parse_annotations import RAW_TO_TORCHVISION_LABEL, parse_bbox_csv
from src.models.detector import CLASS_MAPPING


class TestLabels(unittest.TestCase):
    """Unit tests for label mapping and project conventions."""

    def test_label_mapping_values(self):
        """Verify that CropOrWeed2 labels map to 1 (Crop) and 2 (Weed), with 0 reserved for background."""
        # Torchvision convention
        self.assertEqual(CLASS_MAPPING[0], "background")
        self.assertEqual(CLASS_MAPPING[1], "crop")
        self.assertEqual(CLASS_MAPPING[2], "weed")

        # Variant raw mapping
        self.assertEqual(RAW_TO_TORCHVISION_LABEL[0], 1)  # Raw Crop -> 1
        self.assertEqual(RAW_TO_TORCHVISION_LABEL[1], 2)  # Raw Weed -> 2

    def test_unmapped_species_excluded(self):
        """Verify that species IDs outside 0 and 1 are ignored."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            test_csv = Path(tmp_dir) / "test.csv"
            # Row with valid crop (0), valid weed (1), and unmapped species (99)
            test_csv.write_text(
                "10,10,50,50,0,30,30\n"
                "60,60,100,100,1,80,80\n"
                "120,120,150,150,99,135,135\n"
            )

            boxes, labels, dropped = parse_bbox_csv(test_csv)
            self.assertEqual(len(boxes), 2)
            self.assertEqual(len(labels), 2)
            self.assertEqual(labels, [1, 2])
            self.assertEqual(dropped, 1)


if __name__ == "__main__":
    unittest.main()
